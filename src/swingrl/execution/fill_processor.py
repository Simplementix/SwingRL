"""Fill processor: Stage 5 DB recording and position updates.

Records every fill to the trades table and maintains the positions table
with weighted-average cost basis tracking.

Usage:
    from swingrl.execution.fill_processor import FillProcessor
    processor = FillProcessor(db=db_manager)
    processor.process(fill_result)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from swingrl.execution.types import FillResult

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)


class FillProcessor:
    """Records fills to trades table and maintains position state.

    Handles position creation (buy with no existing row), cost basis
    averaging (subsequent buys), position reduction (sells), and
    position deletion (sell-to-zero).
    """

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize fill processor.

        Args:
            db: DatabaseManager for SQLite trades/positions access.
        """
        self._db = db

    def process(self, fill: FillResult) -> None:
        """Record a fill to trades table and update positions.

        Args:
            fill: FillResult from an exchange adapter.
        """
        self._record_trade(fill)
        self._update_position(fill)

        log.info(
            "fill_processed",
            trade_id=fill.trade_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            environment=fill.environment,
        )

    def record_adjustment(
        self,
        symbol: str,
        environment: str,
        quantity_delta: float,
        price: float,
        reason: str,
    ) -> None:
        """Record a reconciliation adjustment as a trade.

        Args:
            symbol: Ticker symbol.
            environment: Trading environment ("equity" or "crypto").
            quantity_delta: Quantity change (positive or negative).
            price: Reference price for the adjustment.
            reason: Explanation for the adjustment.
        """
        trade_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        side = "buy" if quantity_delta > 0 else "sell"

        with self._db.sqlite() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(trade_id, timestamp, symbol, side, quantity, price, commission, "
                "slippage, environment, broker, order_type, trade_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade_id,
                    now,
                    symbol,
                    side,
                    abs(quantity_delta),
                    price,
                    0.0,
                    0.0,
                    environment,
                    "adjustment",
                    "market",
                    "adjustment",
                ),
            )

        log.info(
            "adjustment_recorded",
            trade_id=trade_id,
            symbol=symbol,
            quantity_delta=quantity_delta,
            reason=reason,
        )

    def _record_trade(self, fill: FillResult) -> None:
        """Insert a trade row into the trades table.

        Args:
            fill: FillResult with trade details.
        """
        now = datetime.now(UTC).isoformat()

        with self._db.sqlite() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(trade_id, timestamp, symbol, side, quantity, price, commission, "
                "slippage, environment, broker, order_type, trade_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fill.trade_id,
                    now,
                    fill.symbol,
                    fill.side,
                    fill.quantity,
                    fill.fill_price,
                    fill.commission,
                    fill.slippage,
                    fill.environment,
                    fill.broker,
                    "market",
                    "signal",
                ),
            )

    def _update_position(self, fill: FillResult) -> None:
        """Update the positions table based on the fill.

        Buy: create new position or adjust cost basis (weighted average).
        Sell: reduce quantity. Delete row if quantity reaches zero.

        Args:
            fill: FillResult with trade details.
        """
        now = datetime.now(UTC).isoformat()

        with self._db.sqlite() as conn:
            existing = conn.execute(
                "SELECT quantity, cost_basis FROM positions WHERE symbol = ? AND environment = ?",
                (fill.symbol, fill.environment),
            ).fetchone()

            if fill.side == "buy":
                if existing is None:
                    # New position
                    unrealized_pnl = 0.0
                    conn.execute(
                        "INSERT INTO positions "
                        "(symbol, environment, quantity, cost_basis, last_price, "
                        "unrealized_pnl, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            fill.symbol,
                            fill.environment,
                            fill.quantity,
                            fill.fill_price,
                            fill.fill_price,
                            unrealized_pnl,
                            now,
                        ),
                    )
                else:
                    # Add to existing -- weighted average cost basis
                    old_qty = existing["quantity"]
                    old_cost = existing["cost_basis"]
                    new_qty = old_qty + fill.quantity
                    new_cost = (old_qty * old_cost + fill.quantity * fill.fill_price) / new_qty
                    unrealized_pnl = (fill.fill_price - new_cost) * new_qty

                    conn.execute(
                        "INSERT OR REPLACE INTO positions "
                        "(symbol, environment, quantity, cost_basis, last_price, "
                        "unrealized_pnl, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            fill.symbol,
                            fill.environment,
                            new_qty,
                            new_cost,
                            fill.fill_price,
                            unrealized_pnl,
                            now,
                        ),
                    )
            else:
                # Sell: reduce quantity
                if existing is not None:
                    new_qty = existing["quantity"] - fill.quantity

                    if new_qty <= 0:
                        # Position fully closed
                        conn.execute(
                            "DELETE FROM positions WHERE symbol = ? AND environment = ?",
                            (fill.symbol, fill.environment),
                        )
                    else:
                        # Partial sell -- cost basis unchanged
                        old_cost = existing["cost_basis"]
                        unrealized_pnl = (fill.fill_price - old_cost) * new_qty

                        conn.execute(
                            "INSERT OR REPLACE INTO positions "
                            "(symbol, environment, quantity, cost_basis, last_price, "
                            "unrealized_pnl, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                fill.symbol,
                                fill.environment,
                                new_qty,
                                old_cost,
                                fill.fill_price,
                                unrealized_pnl,
                                now,
                            ),
                        )
