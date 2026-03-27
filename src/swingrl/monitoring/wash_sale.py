"""Wash sale scanner for equity fills.

Detects equity buy fills that fall within the 30-day wash sale window
of a previously realized loss on the same symbol. Crypto fills are
ignored (wash sale rules do not apply to cryptocurrency).

Usage:
    from swingrl.monitoring.wash_sale import scan_wash_sales, record_realized_loss

    record_realized_loss("SPY", "2026-03-01", 150.0, db)
    warnings = scan_wash_sales(fills, db)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.types import FillResult

log = structlog.get_logger(__name__)


def record_realized_loss(
    symbol: str,
    sale_date: str,
    loss_amount: float,
    db: DatabaseManager,
) -> None:
    """Record a realized loss and create a 30-day wash sale window.

    Args:
        symbol: Ticker symbol of the security sold at a loss.
        sale_date: ISO date string of the loss-generating sale.
        loss_amount: Dollar amount of the realized loss (positive).
        db: DatabaseManager providing SQLite connection.
    """
    wash_window_end = (datetime.strptime(sale_date, "%Y-%m-%d") + timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )

    with db.sqlite() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO wash_sale_tracker "
            "(symbol, sale_date, loss_amount, wash_window_end, triggered) "
            "VALUES (?, ?, ?, ?, 0)",
            (symbol, sale_date, loss_amount, wash_window_end),
        )

    log.info(
        "wash_sale_window_recorded",
        symbol=symbol,
        sale_date=sale_date,
        loss_amount=loss_amount,
        wash_window_end=wash_window_end,
    )


def scan_wash_sales(
    fills: list[FillResult],
    db: DatabaseManager,
) -> list[dict[str, object]]:
    """Scan equity buy fills for wash sale violations.

    Checks each equity buy fill against the wash_sale_tracker table
    for active (unexpired, untriggered) wash sale windows. Crypto
    fills are ignored.

    Args:
        fills: List of FillResult objects from recent trades.
        db: DatabaseManager providing SQLite connection.

    Returns:
        List of warning dicts with symbol, window_end, loss_amount
        for each wash sale violation detected.
    """
    equity_buys = [f for f in fills if f.side == "buy" and f.environment == "equity"]

    if not equity_buys:
        return []

    warnings: list[dict[str, object]] = []

    with db.sqlite() as conn:
        for fill in equity_buys:
            row = conn.execute(
                "SELECT sale_date, wash_window_end, loss_amount "
                "FROM wash_sale_tracker "
                "WHERE symbol = ? AND date(wash_window_end) > date('now') AND triggered = 0 "
                "ORDER BY sale_date DESC LIMIT 1",
                (fill.symbol,),
            ).fetchone()

            if row is None:
                continue

            row_dict = dict(row)
            sale_date = row_dict["sale_date"]
            window_end = row_dict["wash_window_end"]
            loss_amount = row_dict["loss_amount"]

            # Mark as triggered
            conn.execute(
                "UPDATE wash_sale_tracker SET triggered = 1 WHERE symbol = ? AND sale_date = ?",
                (fill.symbol, sale_date),
            )

            log.warning(
                "wash_sale_detected",
                symbol=fill.symbol,
                sale_date=sale_date,
                window_end=window_end,
                loss_amount=loss_amount,
                buy_trade_id=fill.trade_id,
            )

            warnings.append(
                {
                    "symbol": fill.symbol,
                    "window_end": window_end,
                    "loss_amount": loss_amount,
                }
            )

    return warnings
