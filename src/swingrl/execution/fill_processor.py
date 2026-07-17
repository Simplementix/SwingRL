"""Fill processor: Stage 5 DB recording and position updates.

Records every fill to the trades table and maintains the positions table
with weighted-average cost basis tracking. Also writes the ``fill_quality``
sidecar row (§3.7.5, Task 10) for every recorded fill — never for reconciliation
adjustments, which go through the separate ``record_adjustment`` path.

Usage:
    from swingrl.execution.fill_processor import FillProcessor
    processor = FillProcessor(db=db_manager, config=config)
    processor.process(fill_result, cycle_id=cycle_id, decision_price=current_price)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from swingrl.execution.types import FillResult, SizedOrder
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)


class FillProcessor:
    """Records fills to trades table and maintains position state.

    Handles position creation (buy with no existing row), cost basis
    averaging (subsequent buys), position reduction (sells), and
    position deletion (sell-to-zero). Also writes one ``fill_quality`` sidecar
    row per recorded fill (fail-open, Task 10).
    """

    def __init__(self, db: DatabaseManager, config: SwingRLConfig | None = None) -> None:
        """Initialize fill processor.

        Args:
            db: DatabaseManager for database access.
            config: Optional validated SwingRLConfig, used to snapshot the modeled
                crypto trading cost into ``fill_quality.expected_cost_frac``
                (§3.7.5). Callers that construct FillProcessor without a config
                (e.g. the crypto sim adapter's emergency-sell path, reconciliation)
                fall back to a 0.0 modeled cost rather than failing.
        """
        self._db = db
        self._config = config

    def process(
        self,
        fill: FillResult,
        sized_order: SizedOrder | None = None,
        *,
        cycle_id: int | None = None,
        decision_price: float | None = None,
    ) -> None:
        """Record a fill to trades table and update positions.

        Only ``status="filled"`` results are recorded (review C2/M11): a ``pending`` or
        ``rejected`` result carries zero quantity/price and is dropped here so it never
        becomes a $0 trades row that zeroes the position price and triggers a re-buy.

        Args:
            fill: FillResult from an exchange adapter.
            sized_order: Optional SizedOrder with stop/TP prices to persist (crypto buys).
            cycle_id: The inference cycle this fill belongs to (Task 9's
                ``CycleRecorder.record_cycle`` return value), or ``None`` when capture
                failed upstream (fail-open) or the caller has no cycle context (e.g. an
                emergency sell). Stored on the ``trades`` row.
            decision_price: The sizing-time price read before order submission (the
                ``get_current_price()`` value pipeline.py's Step 9 used to size the
                order), or ``None`` when unavailable. Feeds the ``fill_quality``
                slippage calculation — never blocks fill recording when absent.

        Raises:
            DataError: If a fill marked ``filled`` carries a non-positive quantity or price
                (backstop against silently recording a $0 trade).
        """
        if fill.status != "filled":
            log.info(
                "fill_dropped_not_filled",
                trade_id=fill.trade_id,
                symbol=fill.symbol,
                status=fill.status,
                environment=fill.environment,
            )
            return

        self._record_trade(fill, cycle_id=cycle_id)
        self._update_position(fill, sized_order)
        self._record_fill_quality(fill, decision_price=decision_price)

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

        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(trade_id, timestamp, symbol, side, quantity, price, commission, "
                "slippage, environment, broker, order_type, trade_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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

    def _record_trade(self, fill: FillResult, cycle_id: int | None = None) -> None:
        """Insert a trade row into the trades table.

        Args:
            fill: FillResult with trade details.
            cycle_id: The inference cycle this fill belongs to (Task 10), or ``None``
                when no cycle context is available. The ``trades.cycle_id`` FK column
                is nullable — ``record_adjustment`` never sets it either.

        Raises:
            DataError: If quantity or price is non-positive — a $0/zero-quantity trade
                would zero the position's price and trigger spurious re-buys (review M11).
        """
        if fill.quantity <= 0 or fill.fill_price <= 0:
            log.error(
                "zero_value_trade_rejected",
                trade_id=fill.trade_id,
                symbol=fill.symbol,
                quantity=fill.quantity,
                price=fill.fill_price,
                environment=fill.environment,
            )
            raise DataError(
                f"refusing to record trade with non-positive quantity/price for "
                f"{fill.symbol}: quantity={fill.quantity}, price={fill.fill_price}"
            )

        now = datetime.now(UTC).isoformat()

        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(trade_id, timestamp, symbol, side, quantity, price, commission, "
                "slippage, environment, broker, order_type, trade_type, cycle_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
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
                    cycle_id,
                ),
            )

    def _record_fill_quality(self, fill: FillResult, decision_price: float | None) -> None:
        """Write one ``fill_quality`` sidecar row for a recorded fill (§3.7.5, Task 10).

        Fail-open (Task 9 precedent): any error here is logged and swallowed — a
        fill_quality capture failure must never disturb fill processing, which has
        already completed (trade + position rows written) by the time this runs.

        ``decision_price`` is the sizing-time ``get_current_price()`` value pipeline.py's
        Step 9 reads before order submission. It differs from ``fill.fill_price`` by
        construction, not just execution noise: equity submits by notional (the price
        only converts a dollar amount into a share quantity; Alpaca's source for it is
        the last IEX trade), and the crypto sim adapter fetches a SECOND mid-price at
        fill time — so decision != fill even on a fill with zero real slippage.

        Args:
            fill: The recorded FillResult (status == "filled").
            decision_price: The sizing-time price, or ``None`` when unavailable (e.g.
                an emergency sell that never went through the Step 9 sizing loop) —
                slippage_frac/expected_fill_price_usd are then NULL, but the row is
                still written so realized_cost_frac (commission-only) is captured.
        """
        try:
            slippage_frac: float | None = None
            expected_fill_price: float | None = None
            expected_cost_frac = self._expected_cost_frac(fill.environment)

            if decision_price is not None and decision_price > 0:
                if fill.side == "buy":
                    # positive = adverse: paying MORE than the decision-time price.
                    slippage_frac = (fill.fill_price - decision_price) / decision_price
                    expected_fill_price = decision_price * (1 + expected_cost_frac)
                else:
                    # positive = adverse: receiving LESS than the decision-time price.
                    slippage_frac = (decision_price - fill.fill_price) / decision_price
                    expected_fill_price = decision_price * (1 - expected_cost_frac)

            # realized_cost_frac = commission/(fill_price*quantity) + max(slippage_frac, 0.0)
            # -- favorable slippage (negative) never offsets the commission cost, it is
            # floored at 0.0; only adverse slippage adds to the realized cost.
            notional = fill.fill_price * fill.quantity
            commission_frac = fill.commission / notional if notional > 0 else 0.0
            realized_cost_frac = commission_frac + max(slippage_frac or 0.0, 0.0)

            time_to_fill_ms = self._time_to_fill_ms(fill)

            with self._db.connection() as conn:
                conn.execute(
                    "INSERT INTO fill_quality "
                    "(trade_id, decision_price_usd, expected_fill_price_usd, fill_price_usd, "
                    "slippage_frac, expected_cost_frac, realized_cost_frac, time_to_fill_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        fill.trade_id,
                        self._as_numeric(decision_price),
                        self._as_numeric(expected_fill_price),
                        self._as_numeric(fill.fill_price),
                        slippage_frac,
                        expected_cost_frac,
                        realized_cost_frac,
                        time_to_fill_ms,
                    ),
                )
        except Exception:
            log.warning("fill_quality_capture_failed", trade_id=fill.trade_id, exc_info=True)

    def _expected_cost_frac(self, environment: str) -> float:
        """Return the modeled round-trip cost fraction for the environment (P-A5).

        Equity is commission-free (Alpaca) -> always 0.0. Crypto's modeled cost is
        ``config.environment.crypto_transaction_cost_pct`` — the same figure the
        training reward function uses — since the live sim adapter's commission +
        slippage constants aren't themselves surfaced through config. Falls back to
        0.0 when no config was injected (legacy callers: the crypto sim adapter's
        emergency-sell path, reconciliation) rather than raising.

        Args:
            environment: "equity" or "crypto".

        Returns:
            The modeled cost fraction, or 0.0 for equity / when config is unavailable.
        """
        if environment != "crypto" or self._config is None:
            return 0.0
        return self._config.environment.crypto_transaction_cost_pct

    @staticmethod
    def _time_to_fill_ms(fill: FillResult) -> int | None:
        """Compute time-to-fill in milliseconds from Task B's fill lifecycle timestamps.

        Args:
            fill: FillResult whose ``submitted_at``/``filled_at`` may be None.

        Returns:
            Milliseconds between submission and fill, or ``None`` if either
            timestamp is missing or unparseable.
        """
        if fill.submitted_at is None or fill.filled_at is None:
            return None
        try:
            submitted = datetime.fromisoformat(fill.submitted_at)
            filled = datetime.fromisoformat(fill.filled_at)
        except ValueError:
            return None
        return int((filled - submitted).total_seconds() * 1000)

    @staticmethod
    def _as_numeric(value: float | None) -> str | None:
        """Format a price for a NUMERIC(18, 8) column as psycopg expects.

        psycopg adapts a Python ``str`` to NUMERIC without float rounding surprises;
        passing a raw ``float`` risks binary floating-point drift on round-trip.

        Args:
            value: The price, or None.

        Returns:
            ``str(round(value, 8))``, or None unchanged.
        """
        return None if value is None else str(round(value, 8))

    def _update_position(self, fill: FillResult, sized_order: SizedOrder | None = None) -> None:
        """Update the positions table based on the fill.

        Buy: create new position or adjust cost basis (weighted average).
            Persists stop_loss_price, take_profit_price, and side from sized_order when provided.
        Sell: reduce quantity. Delete row if quantity reaches zero.
            Carries forward existing stop/TP values on partial sells.

        Args:
            fill: FillResult with trade details.
            sized_order: Optional SizedOrder with stop/TP prices (buy fills only).
        """
        now = datetime.now(UTC).isoformat()

        # Extract stop/TP values from sized_order (buy path only)
        new_stop = sized_order.stop_loss_price if sized_order is not None else None
        new_tp = sized_order.take_profit_price if sized_order is not None else None
        new_side = sized_order.side if sized_order is not None else fill.side

        with self._db.connection() as conn:
            existing = conn.execute(
                "SELECT quantity, cost_basis, stop_loss_price, take_profit_price, side "
                "FROM positions WHERE symbol = %s AND environment = %s",
                (fill.symbol, fill.environment),
            ).fetchone()

            if fill.side == "buy":
                if existing is None:
                    # New position
                    unrealized_pnl = 0.0
                    conn.execute(
                        "INSERT INTO positions "
                        "(symbol, environment, quantity, cost_basis, last_price, "
                        "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            fill.symbol,
                            fill.environment,
                            fill.quantity,
                            fill.fill_price,
                            fill.fill_price,
                            unrealized_pnl,
                            now,
                            new_stop,
                            new_tp,
                            new_side,
                        ),
                    )
                else:
                    # Add to existing -- weighted average cost basis; update stop/TP to new values
                    old_qty = existing["quantity"]
                    old_cost = existing["cost_basis"]
                    new_qty = old_qty + fill.quantity
                    new_cost = (old_qty * old_cost + fill.quantity * fill.fill_price) / new_qty
                    unrealized_pnl = (fill.fill_price - new_cost) * new_qty

                    conn.execute(
                        "INSERT INTO positions "
                        "(symbol, environment, quantity, cost_basis, last_price, "
                        "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (symbol, environment) DO UPDATE SET "
                        "quantity = EXCLUDED.quantity, cost_basis = EXCLUDED.cost_basis, "
                        "last_price = EXCLUDED.last_price, unrealized_pnl = EXCLUDED.unrealized_pnl, "
                        "updated_at = EXCLUDED.updated_at, stop_loss_price = EXCLUDED.stop_loss_price, "
                        "take_profit_price = EXCLUDED.take_profit_price, side = EXCLUDED.side",
                        (
                            fill.symbol,
                            fill.environment,
                            new_qty,
                            new_cost,
                            fill.fill_price,
                            unrealized_pnl,
                            now,
                            new_stop,
                            new_tp,
                            new_side,
                        ),
                    )
            else:
                # Sell: reduce quantity
                if existing is not None:
                    new_qty = existing["quantity"] - fill.quantity

                    if new_qty <= 0:
                        # Position fully closed
                        conn.execute(
                            "DELETE FROM positions WHERE symbol = %s AND environment = %s",
                            (fill.symbol, fill.environment),
                        )
                    else:
                        # Partial sell -- cost basis unchanged; carry forward stop/TP from existing
                        old_cost = existing["cost_basis"]
                        carried_stop = existing["stop_loss_price"]
                        carried_tp = existing["take_profit_price"]
                        carried_side = existing["side"]
                        unrealized_pnl = (fill.fill_price - old_cost) * new_qty

                        conn.execute(
                            "INSERT INTO positions "
                            "(symbol, environment, quantity, cost_basis, last_price, "
                            "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT (symbol, environment) DO UPDATE SET "
                            "quantity = EXCLUDED.quantity, cost_basis = EXCLUDED.cost_basis, "
                            "last_price = EXCLUDED.last_price, unrealized_pnl = EXCLUDED.unrealized_pnl, "
                            "updated_at = EXCLUDED.updated_at, stop_loss_price = EXCLUDED.stop_loss_price, "
                            "take_profit_price = EXCLUDED.take_profit_price, side = EXCLUDED.side",
                            (
                                fill.symbol,
                                fill.environment,
                                new_qty,
                                old_cost,
                                fill.fill_price,
                                unrealized_pnl,
                                now,
                                carried_stop,
                                carried_tp,
                                carried_side,
                            ),
                        )
