"""Crypto stop-price polling daemon thread.

Polls open crypto positions every 60 seconds and checks if current price
has hit stop-loss or take-profit levels. Prices are fetched from the SAME book
the fills are recorded on (the configured symbol, e.g. ``BTCUSDT``) — never a
remapped ``BTCUSD`` book (review H5). On a stop-loss breach the poller records an
append-only ``circuit_breaker_events`` row and sends a Discord alert.

RISK — auto-sell is OUT of scope (revisit before live):
    A stop-loss breach records the event, halts new crypto trading via the
    circuit breaker, and alerts a human — but it does NOT liquidate the position.
    Paper-trading positions are therefore NOT auto-protected: the breached
    position is not sold automatically; a human must act on the alert. Automated
    liquidation is deferred to a later hardening pass and must be built before
    live trading.

Usage:
    from swingrl.scheduler.stop_polling import start_stop_polling_thread
    thread = start_stop_polling_thread(config, db, alerter)
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from swingrl.scheduler.halt_check import is_halted

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


def start_stop_polling_thread(
    config: SwingRLConfig,
    db: DatabaseManager,
    alerter: Alerter | None = None,
) -> threading.Thread:
    """Create and start the crypto stop-price polling daemon thread.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager providing PostgreSQL connection.
        alerter: Optional Discord alerter for stop-breach notifications (review H5).

    Returns:
        The started daemon thread.
    """
    thread = threading.Thread(
        target=_poll_stop_prices,
        args=(config, db, alerter),
        name="stop-price-polling",
        daemon=True,
    )
    thread.start()
    log.info("stop_polling_thread_started")
    return thread


def _poll_stop_prices(
    config: SwingRLConfig,
    db: DatabaseManager,
    alerter: Alerter | None = None,
) -> None:
    """Poll crypto positions for stop-loss and take-profit triggers.

    Runs in an infinite loop. Each iteration:
    1. Check halt flag -- skip if halted
    2. Query open crypto positions from positions table
    3. For each position with stop/TP levels, check current price
    4. Log and alert if triggered

    Never crashes the thread -- all exceptions are caught and logged.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager providing PostgreSQL connection.
        alerter: Optional Discord alerter for stop-breach notifications.
    """
    while True:
        try:
            if is_halted(db):
                log.info("stop_polling_halted", reason="halt_flag_active")
                time.sleep(60)
                continue

            # Query open crypto positions
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT symbol, side, quantity, stop_loss_price, take_profit_price "
                    "FROM positions "
                    "WHERE environment = 'crypto' AND quantity > 0"
                ).fetchall()

            if not rows:
                time.sleep(60)
                continue

            for row in rows:
                _check_stop_levels(row, config, db, alerter)

        except StopIteration:
            # Allow test-injected StopIteration to break the loop
            raise
        except Exception:
            log.exception("stop_polling_error")

        time.sleep(60)


def _check_stop_levels(
    row: dict,
    config: SwingRLConfig,
    db: DatabaseManager,
    alerter: Alerter | None = None,
) -> None:
    """Check a single position's stop-loss and take-profit levels.

    Best-effort: logs warnings on any failure rather than crashing. On a stop-loss
    breach, records a ``circuit_breaker_events`` row and alerts (review H5). The
    position is NOT auto-sold — see the module docstring risk statement.

    Args:
        row: Position row dict with symbol, stop_loss_price, take_profit_price.
        config: Validated SwingRLConfig.
        db: DatabaseManager.
        alerter: Optional Discord alerter for stop-breach notifications.
    """
    symbol = row["symbol"]
    stop_loss = row.get("stop_loss_price")
    take_profit = row.get("take_profit_price")

    if stop_loss is None and take_profit is None:
        return

    try:
        import httpx

        # Price the SAME book the fills are recorded on: the configured symbol
        # (e.g. BTCUSDT) is used verbatim — remapping to BTCUSD priced a different
        # book than the position was opened on (review H5).
        resp = httpx.get(
            f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}",
            timeout=10.0,
        )
        resp.raise_for_status()
        current_price = float(resp.json()["price"])

        if stop_loss is not None and current_price <= float(stop_loss):
            log.warning(
                "stop_loss_triggered",
                symbol=symbol,
                current_price=current_price,
                stop_loss=stop_loss,
            )
            _record_stop_breach(db, symbol, current_price, float(stop_loss), alerter)

        if take_profit is not None and current_price >= float(take_profit):
            log.info(
                "take_profit_triggered",
                symbol=symbol,
                current_price=current_price,
                take_profit=take_profit,
            )

    except Exception:
        log.warning("stop_level_check_failed", symbol=symbol, exc_info=True)


def _record_stop_breach(
    db: DatabaseManager,
    symbol: str,
    current_price: float,
    stop_loss: float,
    alerter: Alerter | None,
) -> None:
    """Record a stop-loss breach as an append-only circuit_breaker_events row.

    Mirrors the ``CircuitBreaker._trigger`` writer pattern (same table, ``resumed_at``
    left NULL) so the breach halts new crypto trading until the cooldown elapses.
    Deduped per symbol: while an unresolved breach row already exists for this
    symbol the poll (which runs every 60s) neither re-records nor re-alerts, so a
    persistent breach does not flood the ledger or Discord.

    Auto-sell stays out of scope — the position is not liquidated here.

    Args:
        db: DatabaseManager.
        symbol: The configured/fills symbol that breached (e.g. "BTCUSDT").
        current_price: The price that breached the stop.
        stop_loss: The stop-loss level that was breached.
        alerter: Optional Discord alerter.
    """
    reason = f"stop_loss_breach_{symbol}"
    with db.connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM circuit_breaker_events "
            "WHERE environment = %s AND reason = %s AND resumed_at IS NULL LIMIT 1",
            ("crypto", reason),
        ).fetchone()
        if existing is not None:
            log.info("stop_breach_already_recorded", symbol=symbol)
            return
        conn.execute(
            "INSERT INTO circuit_breaker_events "
            "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                str(uuid4()),
                "crypto",
                datetime.now(tz=UTC).isoformat(),
                current_price,
                stop_loss,
                reason,
            ),
        )

    log.critical(
        "crypto_stop_loss_breach_recorded",
        symbol=symbol,
        current_price=current_price,
        stop_loss=stop_loss,
    )

    if alerter is not None:
        alerter.send_alert(
            level="critical",
            title=f"Crypto Stop-Loss Breached — {symbol}",
            message=(
                f"State: HALTED (crypto)\n{symbol} price {current_price} <= stop "
                f"{stop_loss}. New crypto trading halted; the position is NOT "
                "auto-sold — manual action required."
            ),
            environment="crypto",
        )
