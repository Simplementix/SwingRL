"""Crypto stop-price polling daemon thread.

Polls open crypto positions every 60 seconds and checks if current price
has hit stop-loss or take-profit levels. Best-effort daemon -- Phase 10
hardening will refine the position tracking and price fetch logic.

Usage:
    from swingrl.scheduler.stop_polling import start_stop_polling_thread
    thread = start_stop_polling_thread(config, db)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import structlog

from swingrl.scheduler.halt_check import is_halted

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)


def start_stop_polling_thread(
    config: SwingRLConfig,
    db: DatabaseManager,
) -> threading.Thread:
    """Create and start the crypto stop-price polling daemon thread.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager providing SQLite connection.

    Returns:
        The started daemon thread.
    """
    thread = threading.Thread(
        target=_poll_stop_prices,
        args=(config, db),
        name="stop-price-polling",
        daemon=True,
    )
    thread.start()
    log.info("stop_polling_thread_started")
    return thread


def _poll_stop_prices(
    config: SwingRLConfig,
    db: DatabaseManager,
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
        db: DatabaseManager providing SQLite connection.
    """
    while True:
        try:
            if is_halted(db):
                log.info("stop_polling_halted", reason="halt_flag_active")
                time.sleep(60)
                continue

            # Query open crypto positions
            with db.sqlite() as conn:
                rows = conn.execute(
                    "SELECT symbol, side, quantity, stop_loss_price, take_profit_price "
                    "FROM positions "
                    "WHERE environment = 'crypto' AND quantity > 0"
                ).fetchall()

            if not rows:
                time.sleep(60)
                continue

            for row in rows:
                _check_stop_levels(row, config, db)

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
) -> None:
    """Check a single position's stop-loss and take-profit levels.

    Best-effort: logs warnings on any failure rather than crashing.

    Args:
        row: Position row dict with symbol, stop_loss_price, take_profit_price.
        config: Validated SwingRLConfig.
        db: DatabaseManager.
    """
    symbol = row["symbol"]
    stop_loss = row.get("stop_loss_price")
    take_profit = row.get("take_profit_price")

    if stop_loss is None and take_profit is None:
        return

    try:
        import httpx

        # Fetch current price from Binance.US REST API
        # Convert BTCUSDT -> BTCUSD for Binance.US endpoint
        api_symbol = symbol.replace("USDT", "USD") if "USDT" in symbol else symbol
        resp = httpx.get(
            f"https://api.binance.us/api/v3/ticker/price?symbol={api_symbol}",
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

        if take_profit is not None and current_price >= float(take_profit):
            log.info(
                "take_profit_triggered",
                symbol=symbol,
                current_price=current_price,
                take_profit=take_profit,
            )

    except Exception:
        log.warning("stop_level_check_failed", symbol=symbol, exc_info=True)
