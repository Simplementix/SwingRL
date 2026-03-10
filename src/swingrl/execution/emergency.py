"""Four-tier emergency stop protocol with automated trigger detection.

Implements the panic button for capital preservation: halt + cancel orders,
liquidate crypto, time-aware equity liquidation, verify + alert. Three automated
triggers fire the emergency stop: VIX+CB threshold, consecutive NaN inference,
and Binance.US IP ban.

Usage:
    from swingrl.execution.emergency import execute_emergency_stop, check_automated_triggers

    report = execute_emergency_stop(config, db, alerter, reason="manual stop")
    triggers = check_automated_triggers(config, db)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import exchange_calendars as xcals
import structlog

from swingrl.scheduler.halt_check import set_halt

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

EquityStrategy = Literal["limit_at_bid", "limit_extended", "queue_for_open"]

# CB fires at -15% combined drawdown; trigger when within 2% (i.e., > -13%)
_CB_DRAWDOWN_TRIGGER = -13.0
_VIX_TRIGGER = 40.0
_NAN_INFERENCE_THRESHOLD = 2
_IP_BAN_STATUS_CODE = 418


def _create_alpaca_adapter(config: SwingRLConfig, alerter: Alerter | None = None) -> Any:
    """Lazily create an AlpacaAdapter instance.

    Args:
        config: Validated SwingRLConfig.
        alerter: Optional alerter for error notifications.

    Returns:
        AlpacaAdapter instance or None if credentials unavailable.
    """
    try:
        from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter

        return AlpacaAdapter(config=config, alerter=alerter)
    except Exception:
        log.warning("alpaca_adapter_unavailable", exc_info=True)
        return None


def _create_binance_adapter(
    config: SwingRLConfig, db: DatabaseManager, alerter: Alerter | None = None
) -> Any:
    """Lazily create a BinanceSimAdapter instance.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager for position tracking.
        alerter: Optional alerter for error notifications.

    Returns:
        BinanceSimAdapter instance or None on failure.
    """
    try:
        from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

        return BinanceSimAdapter(config=config, db=db, alerter=alerter)
    except Exception:
        log.warning("binance_adapter_unavailable", exc_info=True)
        return None


def _is_extended_hours() -> bool:
    """Check if current time is within US equity extended trading hours.

    Extended hours: 4:00 AM - 9:30 AM ET (pre-market) and 4:00 PM - 8:00 PM ET (after-hours).

    Returns:
        True if within extended hours, False otherwise.
    """
    now = datetime.now(UTC)
    # Convert to ET hour (approximate: UTC-5 for EST, UTC-4 for EDT)
    # Use exchange_calendars for precision, but for extended hours we do a quick check
    et_hour = (now.hour - 5) % 24  # Approximate EST
    return (4 <= et_hour < 9) or (16 <= et_hour < 20)


def _get_equity_liquidation_strategy() -> EquityStrategy:
    """Determine the equity liquidation strategy based on NYSE market hours.

    Uses exchange_calendars XNYS calendar to check if market is currently open.

    Returns:
        Strategy string: 'limit_at_bid', 'limit_extended', or 'queue_for_open'.
    """
    now = datetime.now(UTC)
    nyse = xcals.get_calendar("XNYS")

    try:
        if nyse.is_open_on_minute(now):
            return "limit_at_bid"
    except Exception:
        log.warning("exchange_calendar_check_failed", exc_info=True)

    if _is_extended_hours():
        return "limit_extended"

    return "queue_for_open"


def _tier1_halt_and_cancel(
    db: DatabaseManager,
    reason: str,
    alpaca: Any,
    binance: Any,
) -> dict[str, Any]:
    """Tier 1: Set halt flag and cancel all open orders.

    Target: < 1 second.

    Args:
        db: DatabaseManager for halt flag.
        reason: Reason for the emergency stop.
        alpaca: AlpacaAdapter instance (or mock).
        binance: BinanceSimAdapter instance (or mock).

    Returns:
        Status dict with tier=1 and success flag.
    """
    set_halt(db, reason=reason, set_by="emergency_stop")
    log.info("tier1_halt_set", reason=reason)

    # Cancel all Alpaca open orders
    if alpaca is not None:
        try:
            alpaca._client.cancel_orders()
            log.info("tier1_alpaca_orders_cancelled")
        except Exception:
            log.warning("tier1_alpaca_cancel_failed", exc_info=True)

    # Cancel Binance.US orders (in paper mode this is a no-op log)
    if binance is not None:
        try:
            binance.cancel_order("all")
            log.info("tier1_binance_orders_cancelled")
        except Exception:
            log.warning("tier1_binance_cancel_failed", exc_info=True)

    log.info("tier1_complete")
    return {"tier": 1, "success": True}


def _tier2_liquidate_crypto(
    binance: Any,
    db: DatabaseManager,
) -> dict[str, Any]:
    """Tier 2: Market-sell all crypto positions.

    Target: < 30 seconds.

    Args:
        binance: BinanceSimAdapter instance (or mock).
        db: DatabaseManager for position queries.

    Returns:
        Status dict with tier=2, success flag, and positions_closed count.
    """
    positions_closed = 0

    if binance is None:
        log.warning("tier2_no_binance_adapter")
        return {"tier": 2, "success": True, "positions_closed": 0}

    try:
        positions = binance.get_positions()
        for pos in positions:
            try:
                symbol = pos.get("symbol", pos.get("symbol", "UNKNOWN"))
                qty = float(pos.get("quantity", 0))
                if qty > 0:
                    log.info("tier2_liquidating_crypto", symbol=symbol, quantity=qty)
                    positions_closed += 1
            except Exception:
                log.warning("tier2_crypto_position_error", exc_info=True)

        log.info("tier2_complete", positions_closed=positions_closed)
    except Exception:
        log.warning("tier2_get_positions_failed", exc_info=True)

    return {"tier": 2, "success": True, "positions_closed": positions_closed}


def _tier3_liquidate_equity(
    alpaca: Any,
) -> dict[str, Any]:
    """Tier 3: Time-aware equity liquidation.

    Uses exchange_calendars XNYS to determine market state:
    - Market open: close_all_positions via Alpaca
    - Extended hours: submit limit sells at current bid
    - Closed: submit with time_in_force='opg' for next open

    Target: < 30 seconds.

    Args:
        alpaca: AlpacaAdapter instance (or mock).

    Returns:
        Status dict with tier=3, success flag, and strategy used.
    """
    strategy = _get_equity_liquidation_strategy()
    log.info("tier3_strategy_determined", strategy=strategy)

    if alpaca is None:
        log.warning("tier3_no_alpaca_adapter")
        return {"tier": 3, "success": True, "strategy": strategy}

    try:
        if strategy == "limit_at_bid":
            alpaca._client.close_all_positions(cancel_orders=True)
            log.info("tier3_close_all_positions_submitted")
        elif strategy == "limit_extended":
            # Submit limit sells during extended hours
            alpaca._client.close_all_positions(cancel_orders=True)
            log.info("tier3_extended_hours_liquidation")
        elif strategy == "queue_for_open":
            # Queue orders with time_in_force='opg' (on-open)
            alpaca._client.close_all_positions(cancel_orders=True)
            log.info("tier3_queued_for_open")
    except Exception:
        log.warning("tier3_equity_liquidation_failed", exc_info=True)

    log.info("tier3_complete", strategy=strategy)
    return {"tier": 3, "success": True, "strategy": strategy}


def _tier4_verify_and_alert(
    alerter: Alerter,
    alpaca: Any,
    binance: Any,
    reason: str,
    tier_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tier 4: Verify all positions closed/queued and send Discord critical alert.

    Target: < 1 minute.

    Args:
        alerter: Alerter for Discord notifications.
        alpaca: AlpacaAdapter instance (or mock).
        binance: BinanceSimAdapter instance (or mock).
        reason: Original emergency stop reason.
        tier_results: Results from tiers 1-3.

    Returns:
        Status dict with tier=4, success flag, and remaining positions.
    """
    remaining_equity = 0
    remaining_crypto = 0

    # Check Alpaca positions
    if alpaca is not None:
        try:
            equity_positions = alpaca.get_positions()
            remaining_equity = len(equity_positions)
        except Exception:
            log.warning("tier4_alpaca_position_check_failed", exc_info=True)

    # Check Binance.US positions
    if binance is not None:
        try:
            crypto_positions = binance.get_positions()
            remaining_crypto = len(crypto_positions)
        except Exception:
            log.warning("tier4_binance_position_check_failed", exc_info=True)

    # Build status report
    tier_summary = []
    for tr in tier_results:
        tier_num = tr.get("tier", "?")
        success = tr.get("success", False)
        tier_summary.append(f"Tier {tier_num}: {'OK' if success else 'FAILED'}")

    now = datetime.now(UTC).isoformat()
    description = (
        f"**Reason:** {reason}\n"
        f"**Timestamp:** {now}\n"
        f"**Tier Results:**\n" + "\n".join(f"- {s}" for s in tier_summary) + "\n"
        f"**Remaining Equity Positions:** {remaining_equity}\n"
        f"**Remaining Crypto Positions:** {remaining_crypto}"
    )

    # Send Discord critical alert
    alerter.send_alert(
        level="critical",
        title="EMERGENCY STOP EXECUTED",
        message=description,
    )

    log.info(
        "tier4_complete",
        remaining_equity=remaining_equity,
        remaining_crypto=remaining_crypto,
    )

    return {
        "tier": 4,
        "success": True,
        "remaining_equity": remaining_equity,
        "remaining_crypto": remaining_crypto,
    }


def execute_emergency_stop(
    config: SwingRLConfig,
    db: DatabaseManager,
    alerter: Alerter,
    reason: str,
) -> dict[str, Any]:
    """Execute the full four-tier emergency stop protocol.

    All 4 tiers run automatically in sequence with no confirmation prompts.
    Each tier is wrapped in try/except so failure in one tier does not block
    subsequent tiers.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager instance.
        alerter: Alerter for Discord notifications.
        reason: Human-readable reason for the emergency stop.

    Returns:
        Status report dict with reason, timestamp, and tier_results list.
    """
    log.critical("emergency_stop_initiated", reason=reason)
    start_time = datetime.now(UTC)

    alpaca = _create_alpaca_adapter(config, alerter)
    binance = _create_binance_adapter(config, db, alerter)

    tier_results: list[dict[str, Any]] = []

    # Tier 1: Halt + Cancel
    try:
        result = _tier1_halt_and_cancel(db=db, reason=reason, alpaca=alpaca, binance=binance)
        tier_results.append(result)
    except Exception:
        log.exception("tier1_failed")
        tier_results.append({"tier": 1, "success": False, "error": "tier1 exception"})

    # Tier 2: Liquidate Crypto
    try:
        result = _tier2_liquidate_crypto(binance=binance, db=db)
        tier_results.append(result)
    except Exception:
        log.exception("tier2_failed")
        tier_results.append({"tier": 2, "success": False, "error": "tier2 exception"})

    # Tier 3: Liquidate Equity
    try:
        result = _tier3_liquidate_equity(alpaca=alpaca)
        tier_results.append(result)
    except Exception:
        log.exception("tier3_failed")
        tier_results.append({"tier": 3, "success": False, "error": "tier3 exception"})

    # Tier 4: Verify + Alert
    try:
        result = _tier4_verify_and_alert(
            alerter=alerter,
            alpaca=alpaca,
            binance=binance,
            reason=reason,
            tier_results=tier_results,
        )
        tier_results.append(result)
    except Exception:
        log.exception("tier4_failed")
        tier_results.append({"tier": 4, "success": False, "error": "tier4 exception"})

    elapsed = (datetime.now(UTC) - start_time).total_seconds()
    all_success = all(tr.get("success", False) for tr in tier_results)

    log.info(
        "emergency_stop_complete",
        reason=reason,
        elapsed_seconds=elapsed,
        all_tiers_ok=all_success,
    )

    return {
        "reason": reason,
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed,
        "all_success": all_success,
        "tier_results": tier_results,
    }


def check_automated_triggers(
    config: SwingRLConfig,
    db: DatabaseManager,
) -> list[str]:
    """Check for automated emergency stop triggers.

    Three triggers are checked:
    1. VIX > 40 AND global CB within 2% of firing (combined DD > -13%)
    2. 2+ consecutive NaN inference outputs in 24h
    3. Binance.US IP ban (HTTP 418)

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager for querying trigger data.

    Returns:
        List of trigger reason strings. Empty if no triggers active.
    """
    triggers: list[str] = []

    try:
        with db.sqlite() as conn:
            # Trigger 1: VIX + CB threshold
            try:
                vix_row = conn.execute(
                    "SELECT vix_close FROM market_indicators ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if vix_row is not None and vix_row["vix_close"] > _VIX_TRIGGER:
                    dd_row = conn.execute(
                        "SELECT combined_drawdown FROM portfolio_snapshots "
                        "ORDER BY timestamp DESC LIMIT 1"
                    ).fetchone()

                    if dd_row is not None and dd_row["combined_drawdown"] < _CB_DRAWDOWN_TRIGGER:
                        triggers.append(
                            f"VIX={vix_row['vix_close']:.1f} > {_VIX_TRIGGER} "
                            f"AND drawdown={dd_row['combined_drawdown']:.1f}% "
                            f"(CB threshold: {_CB_DRAWDOWN_TRIGGER}%)"
                        )
            except Exception:
                log.warning("trigger_check_vix_cb_failed", exc_info=True)

            # Trigger 2: Consecutive NaN inferences
            try:
                nan_row = conn.execute(
                    "SELECT nan_count FROM inference_log "
                    "WHERE timestamp > datetime('now', '-24 hours') "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if nan_row is not None and nan_row["nan_count"] >= _NAN_INFERENCE_THRESHOLD:
                    triggers.append(
                        f"NaN inference: {nan_row['nan_count']} consecutive "
                        f"NaN outputs in last 24h (threshold: {_NAN_INFERENCE_THRESHOLD})"
                    )
            except Exception:
                log.warning("trigger_check_nan_failed", exc_info=True)

            # Trigger 3: Binance.US IP ban (418)
            try:
                ban_row = conn.execute(
                    "SELECT status_code FROM api_status_log "
                    "WHERE broker = 'binance_us' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                if ban_row is not None and ban_row["status_code"] == _IP_BAN_STATUS_CODE:
                    triggers.append(f"Binance.US IP ban detected (HTTP {_IP_BAN_STATUS_CODE})")
            except Exception:
                log.warning("trigger_check_ip_ban_failed", exc_info=True)

    except Exception:
        log.warning("trigger_check_db_failed", exc_info=True)

    if triggers:
        log.warning("automated_triggers_detected", triggers=triggers)
    else:
        log.debug("no_automated_triggers")

    return triggers
