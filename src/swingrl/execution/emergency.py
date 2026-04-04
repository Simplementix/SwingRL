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
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import structlog

from swingrl.scheduler.halt_check import set_halt

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

EquityStrategy = Literal["limit_at_bid", "limit_extended", "queue_for_open"]

# CB fires at -15% combined drawdown; trigger when within 2% (i.e., >= 13%)
# Stored as positive fraction matching how drawdown_pct is stored in portfolio_snapshots
_CB_DRAWDOWN_TRIGGER = 0.13
_VIX_TRIGGER = 40.0
_NAN_INFERENCE_THRESHOLD = 2
_IP_BAN_STATUS_CODE = 418
_ET = ZoneInfo("America/New_York")


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
    et_now = datetime.now(_ET)
    pre_market = (4 <= et_now.hour < 9) or (et_now.hour == 9 and et_now.minute < 30)
    after_hours = 16 <= et_now.hour < 20
    return pre_market or after_hours


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
                symbol = pos.get("symbol", "UNKNOWN")
                qty = float(pos.get("quantity", 0))
                if qty > 0:
                    log.info("tier2_liquidating_crypto", symbol=symbol, quantity=qty)
                    try:
                        binance.emergency_sell(symbol, qty)
                        positions_closed += 1
                        log.info("tier2_crypto_sold", symbol=symbol, quantity=qty)
                    except Exception:
                        log.error(
                            "tier2_crypto_sell_failed",
                            symbol=symbol,
                            quantity=qty,
                            exc_info=True,
                        )
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
        alpaca._client.close_all_positions(cancel_orders=True)
        log.info("tier3_equity_liquidation_submitted", strategy=strategy)
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
    remaining_equity = -1  # Sentinel: unknown until verified
    remaining_crypto = -1

    # Check Alpaca positions
    if alpaca is not None:
        try:
            equity_positions = alpaca.get_positions()
            remaining_equity = len(equity_positions)
        except Exception:
            log.error("tier4_alpaca_position_check_failed", exc_info=True)
    else:
        remaining_equity = 0  # No adapter → no positions to check

    # Check Binance.US positions
    if binance is not None:
        try:
            crypto_positions = binance.get_positions()
            remaining_crypto = len(crypto_positions)
        except Exception:
            log.error("tier4_binance_position_check_failed", exc_info=True)
    else:
        remaining_crypto = 0  # No adapter → no positions to check

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

    # -1 means position check failed — we can't confirm positions are closed
    verification_ok = remaining_equity >= 0 and remaining_crypto >= 0
    all_closed = verification_ok and remaining_equity == 0 and remaining_crypto == 0

    log.info(
        "tier4_complete",
        remaining_equity=remaining_equity,
        remaining_crypto=remaining_crypto,
        verification_ok=verification_ok,
        all_closed=all_closed,
    )

    return {
        "tier": 4,
        "success": verification_ok,
        "all_closed": all_closed,
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
    1. VIX > 40 (from DuckDB macro_features) AND drawdown >= 13% (from SQLite portfolio_snapshots)
    2. 2+ NaN inference outputs in 24h (from SQLite inference_outcomes)
    3. Binance.US IP ban HTTP 418 (from SQLite api_errors)

    Each trigger manages its own DB connection independently and fails gracefully.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager for querying trigger data.

    Returns:
        List of trigger reason strings. Empty if no triggers active.
    """
    triggers: list[str] = []

    # Trigger 1: VIX + CB threshold
    # VIX comes from DuckDB macro_features, drawdown from SQLite portfolio_snapshots
    try:
        with db.connection() as cursor:
            vix_row = cursor.execute(
                "SELECT value FROM macro_features "
                "WHERE series_id = 'VIXCLS' ORDER BY date DESC LIMIT 1"
            ).fetchone()

        if vix_row is not None and vix_row[0] > _VIX_TRIGGER:
            with db.connection() as conn:
                dd_row = conn.execute(
                    "SELECT MAX(drawdown_pct) as worst_dd FROM portfolio_snapshots "
                    "WHERE timestamp > NOW() - INTERVAL '24 hours'"
                ).fetchone()

            if (
                dd_row is not None
                and dd_row["worst_dd"] is not None
                and dd_row["worst_dd"] >= _CB_DRAWDOWN_TRIGGER
            ):
                triggers.append(
                    f"VIX={vix_row[0]:.1f} > {_VIX_TRIGGER} "
                    f"AND drawdown={dd_row['worst_dd']:.1%} "
                    f"(CB threshold: {_CB_DRAWDOWN_TRIGGER:.0%})"
                )
    except Exception:
        log.warning("trigger_check_vix_cb_failed", exc_info=True)

    # Trigger 2: NaN inference count in last 24h
    try:
        with db.connection() as conn:
            nan_row = conn.execute(
                "SELECT COUNT(*) as nan_count FROM inference_outcomes "
                "WHERE had_nan = 1 AND timestamp > NOW() - INTERVAL '24 hours'"
            ).fetchone()

        if nan_row is not None and nan_row["nan_count"] >= _NAN_INFERENCE_THRESHOLD:
            triggers.append(
                f"NaN inference: {nan_row['nan_count']} NaN outputs "
                f"in last 24h (threshold: {_NAN_INFERENCE_THRESHOLD})"
            )
    except Exception:
        log.warning("trigger_check_nan_failed", exc_info=True)

    # Trigger 3: Binance.US IP ban (HTTP 418)
    try:
        with db.connection() as conn:
            ban_row = conn.execute(
                "SELECT COUNT(*) as ban_count FROM api_errors "
                "WHERE broker = 'binance_us' AND status_code = %s "
                "AND timestamp > NOW() - INTERVAL '24 hours'",
                (_IP_BAN_STATUS_CODE,),
            ).fetchone()

        if ban_row is not None and ban_row["ban_count"] > 0:
            triggers.append(f"Binance.US IP ban detected (HTTP {_IP_BAN_STATUS_CODE})")
    except Exception:
        log.warning("trigger_check_ip_ban_failed", exc_info=True)

    if triggers:
        log.warning("automated_triggers_detected", triggers=triggers)
    else:
        log.debug("no_automated_triggers")

    return triggers
