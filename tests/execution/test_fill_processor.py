"""Tests for FillProcessor database recording and position management.

PAPER-10: FillProcessor records fills to trades table and maintains
positions with cost basis tracking and quantity management.

Task 10: cycle_id threading (trades.cycle_id) and the fill_quality sidecar writer
(§3.7.5). The Task 10 additions use the ``make_mock_db`` MagicMock helper (no live
PostgreSQL, mirrors Task 9's ``test_cycle_recorder.py`` precedent) so they run
foreground without DATABASE_URL — the pre-existing tests above depend on the
Postgres-backed ``mock_db`` fixture and skip without it.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.types import FillResult
from swingrl.utils.exceptions import DataError
from tests.conftest import make_mock_db

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager


@pytest.fixture
def processor(mock_db: DatabaseManager) -> FillProcessor:
    """FillProcessor wired to mock database."""
    return FillProcessor(db=mock_db)


@pytest.fixture
def buy_fill() -> FillResult:
    """Sample buy fill for testing."""
    return FillResult(
        trade_id="fill-001",
        symbol="SPY",
        side="buy",
        quantity=10.0,
        fill_price=150.0,
        commission=0.0,
        slippage=0.0,
        environment="equity",
        broker="alpaca",
    )


@pytest.fixture
def second_buy_fill() -> FillResult:
    """Second buy fill for cost basis averaging."""
    return FillResult(
        trade_id="fill-002",
        symbol="SPY",
        side="buy",
        quantity=5.0,
        fill_price=160.0,
        commission=0.0,
        slippage=0.0,
        environment="equity",
        broker="alpaca",
    )


@pytest.fixture
def sell_fill() -> FillResult:
    """Sell fill to reduce position."""
    return FillResult(
        trade_id="fill-003",
        symbol="SPY",
        side="sell",
        quantity=5.0,
        fill_price=155.0,
        commission=0.0,
        slippage=0.0,
        environment="equity",
        broker="alpaca",
    )


class TestTradeRecording:
    """Verify trades are recorded to SQLite trades table."""

    def test_buy_creates_trade_row(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """Buy fill inserts a row in the trades table."""
        processor.process(buy_fill)

        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = %s", (buy_fill.trade_id,)
            ).fetchone()

        assert row is not None
        assert row["symbol"] == "SPY"
        assert row["side"] == "buy"
        assert row["quantity"] == 10.0
        assert row["price"] == 150.0
        assert row["environment"] == "equity"
        assert row["broker"] == "alpaca"
        assert row["order_type"] == "market"
        assert row["trade_type"] == "signal"

    def test_adjustment_trade_type(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """record_adjustment creates trade with trade_type='adjustment'."""
        processor.record_adjustment(
            symbol="SPY",
            environment="equity",
            quantity_delta=2.0,
            price=155.0,
            reason="reconciliation",
        )

        with mock_db.connection() as conn:
            row = conn.execute("SELECT * FROM trades WHERE trade_type = 'adjustment'").fetchone()

        assert row is not None
        assert row["symbol"] == "SPY"
        assert row["quantity"] == 2.0


class TestHonestFillGuards:
    """Review C2/M11: non-filled results dropped; zero-value fills refused as a backstop."""

    def test_process_drops_non_filled_result(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """A pending fill is dropped — no trades row, no position row (primary guard)."""
        pending = FillResult(
            trade_id="pending-001",
            symbol="SPY",
            side="buy",
            quantity=0.0,
            fill_price=0.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="pending",
        )

        processor.process(pending)

        with mock_db.connection() as conn:
            trade = conn.execute(
                "SELECT * FROM trades WHERE trade_id = %s", ("pending-001",)
            ).fetchone()
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert trade is None
        assert pos is None

    def test_process_rejects_zero_quantity_filled_fill(
        self,
        processor: FillProcessor,
    ) -> None:
        """(d) A filled fill with quantity=0 raises DataError (never a $0 trades row)."""
        bad = FillResult(
            trade_id="zero-qty-001",
            symbol="SPY",
            side="buy",
            quantity=0.0,
            fill_price=150.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="filled",
        )

        with pytest.raises(DataError):
            processor.process(bad)

    def test_process_rejects_zero_price_filled_fill(
        self,
        processor: FillProcessor,
    ) -> None:
        """(d) A filled fill with price=0 raises DataError (zeroed position price bug)."""
        bad = FillResult(
            trade_id="zero-price-001",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=0.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="filled",
        )

        with pytest.raises(DataError):
            processor.process(bad)


class TestPositionManagement:
    """Verify position creation, update, and deletion."""

    def test_buy_creates_new_position(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """First buy creates a new position row."""
        processor.process(buy_fill)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos is not None
        assert pos["quantity"] == 10.0
        assert pos["cost_basis"] == 150.0

    def test_second_buy_adjusts_cost_basis(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        second_buy_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """Second buy computes weighted-average cost basis."""
        processor.process(buy_fill)
        processor.process(second_buy_fill)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos["quantity"] == 15.0
        # Weighted avg: (10*150 + 5*160) / 15 = 2300/15 = 153.333...
        assert pos["cost_basis"] == pytest.approx(153.333, rel=1e-3)

    def test_sell_reduces_quantity(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        sell_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """Sell reduces position quantity without changing cost basis."""
        processor.process(buy_fill)
        processor.process(sell_fill)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos["quantity"] == 5.0
        assert pos["cost_basis"] == 150.0  # Cost basis unchanged on sell

    def test_sell_to_zero_deletes_position(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """Selling entire position removes the row."""
        processor.process(buy_fill)

        full_sell = FillResult(
            trade_id="fill-full-sell",
            symbol="SPY",
            side="sell",
            quantity=10.0,
            fill_price=155.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        processor.process(full_sell)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos is None

    def test_position_tracks_last_price(
        self,
        processor: FillProcessor,
        buy_fill: FillResult,
        mock_db: DatabaseManager,
    ) -> None:
        """Position last_price updates on each fill."""
        processor.process(buy_fill)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos["last_price"] == 150.0


class TestStopTPPersistence:
    """PAPER-07: stop/TP prices persisted to positions table from SizedOrder."""

    def test_buy_with_sized_order_persists_stop_tp(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Buy fill with sized_order writes stop_loss_price, take_profit_price, and side."""
        from swingrl.execution.types import SizedOrder

        fill = FillResult(
            trade_id="fill-stop-001",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            fill_price=50000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order = SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            dollar_amount=500.0,
            stop_loss_price=45000.0,
            take_profit_price=55000.0,
            environment="crypto",
        )

        processor.process(fill, sized_order=sized_order)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()

        assert pos is not None
        assert pos["stop_loss_price"] == 45000.0
        assert pos["take_profit_price"] == 55000.0
        assert pos["side"] == "buy"

    def test_buy_without_sized_order_writes_null_stop_tp(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Buy fill without sized_order (backward compat) writes NULL for stop/TP columns."""
        fill = FillResult(
            trade_id="fill-compat-001",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=150.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        processor.process(fill)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("SPY", "equity"),
            ).fetchone()

        assert pos is not None
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None

    def test_second_buy_updates_stop_tp_to_new_values(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Second buy into same position updates stop/TP to new sized_order values."""
        from swingrl.execution.types import SizedOrder

        fill1 = FillResult(
            trade_id="fill-update-001",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            fill_price=50000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order1 = SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            dollar_amount=500.0,
            stop_loss_price=45000.0,
            take_profit_price=55000.0,
            environment="crypto",
        )

        fill2 = FillResult(
            trade_id="fill-update-002",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.005,
            fill_price=52000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order2 = SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.005,
            dollar_amount=260.0,
            stop_loss_price=46000.0,
            take_profit_price=58000.0,
            environment="crypto",
        )

        processor.process(fill1, sized_order=sized_order1)
        processor.process(fill2, sized_order=sized_order2)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()

        assert pos is not None
        assert pos["stop_loss_price"] == 46000.0
        assert pos["take_profit_price"] == 58000.0

    def test_partial_sell_carries_forward_stop_tp(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Partial sell carries forward existing stop_loss_price and take_profit_price."""
        from swingrl.execution.types import SizedOrder

        buy_fill = FillResult(
            trade_id="fill-carry-buy",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.02,
            fill_price=50000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order = SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.02,
            dollar_amount=1000.0,
            stop_loss_price=45000.0,
            take_profit_price=55000.0,
            environment="crypto",
        )
        processor.process(buy_fill, sized_order=sized_order)

        # Partial sell -- no sized_order on sell path
        partial_sell = FillResult(
            trade_id="fill-carry-sell",
            symbol="BTCUSDT",
            side="sell",
            quantity=0.01,
            fill_price=52000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        processor.process(partial_sell)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()

        assert pos is not None
        assert pos["quantity"] == pytest.approx(0.01)
        assert pos["stop_loss_price"] == 45000.0
        assert pos["take_profit_price"] == 55000.0

    def test_full_sell_deletes_position(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Full sell (position closed) still deletes the position row."""
        from swingrl.execution.types import SizedOrder

        buy_fill = FillResult(
            trade_id="fill-del-buy",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            fill_price=50000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order = SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            dollar_amount=500.0,
            stop_loss_price=45000.0,
            take_profit_price=55000.0,
            environment="crypto",
        )
        processor.process(buy_fill, sized_order=sized_order)

        full_sell = FillResult(
            trade_id="fill-del-sell",
            symbol="BTCUSDT",
            side="sell",
            quantity=0.01,
            fill_price=52000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        processor.process(full_sell)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("BTCUSDT", "crypto"),
            ).fetchone()

        assert pos is None

    def test_buy_side_column_populated(
        self,
        processor: FillProcessor,
        mock_db: DatabaseManager,
    ) -> None:
        """Side column is populated as 'buy' on new position creation."""
        from swingrl.execution.types import SizedOrder

        fill = FillResult(
            trade_id="fill-side-001",
            symbol="ETHUSDT",
            side="buy",
            quantity=0.1,
            fill_price=3000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        sized_order = SizedOrder(
            symbol="ETHUSDT",
            side="buy",
            quantity=0.1,
            dollar_amount=300.0,
            stop_loss_price=2700.0,
            take_profit_price=3300.0,
            environment="crypto",
        )

        processor.process(fill, sized_order=sized_order)

        with mock_db.connection() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("ETHUSDT", "crypto"),
            ).fetchone()

        assert pos is not None
        assert pos["side"] == "buy"


# ---------------------------------------------------------------------------
# Task 10: cycle_id threading + fill_quality writer (§3.7.5)
# ---------------------------------------------------------------------------


def _mock_db_raising_on(substring: str) -> tuple[MagicMock, MagicMock]:
    """DatabaseManager mock whose conn.execute raises only for SQL containing `substring`.

    Every other statement succeeds and returns a result whose fetchone() is None
    (no existing position — the new-position branch of ``_update_position``).
    """
    db = MagicMock(spec=["connection", "close", "init_schema", "reset"])
    conn = MagicMock()

    def _execute(sql: str, *args: Any, **kwargs: Any) -> MagicMock:
        if substring in sql:
            raise RuntimeError("db unavailable")
        result = MagicMock()
        result.fetchone.return_value = None
        return result

    conn.execute.side_effect = _execute

    @contextmanager
    def _ctx() -> Generator[MagicMock, None, None]:
        yield conn

    db.connection.side_effect = _ctx
    return db, conn


def _buy_fill(
    *,
    trade_id: str = "fill-t10-buy",
    symbol: str = "SPY",
    quantity: float = 10.0,
    fill_price: float = 105.0,
    commission: float = 1.5,
    environment: str = "equity",
    broker: str = "alpaca",
    submitted_at: str | None = None,
    filled_at: str | None = None,
) -> FillResult:
    """A filled buy FillResult with sensible Task 10 defaults."""
    return FillResult(
        trade_id=trade_id,
        symbol=symbol,
        side="buy",
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
        slippage=0.0,
        environment=environment,  # type: ignore[arg-type]
        broker=broker,  # type: ignore[arg-type]
        status="filled",
        submitted_at=submitted_at,
        filled_at=filled_at,
    )


def _sell_fill(
    *,
    trade_id: str = "fill-t10-sell",
    symbol: str = "SPY",
    quantity: float = 10.0,
    fill_price: float = 95.0,
    commission: float = 1.5,
    environment: str = "equity",
    broker: str = "alpaca",
) -> FillResult:
    """A filled sell FillResult with sensible Task 10 defaults."""
    return FillResult(
        trade_id=trade_id,
        symbol=symbol,
        side="sell",
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
        slippage=0.0,
        environment=environment,  # type: ignore[arg-type]
        broker=broker,  # type: ignore[arg-type]
        status="filled",
    )


class TestCycleIdThreading:
    """Step 1(a): the trades INSERT carries the cycle_id passed to process()."""

    def test_cycle_id_lands_in_trades_insert(self) -> None:
        """process(..., cycle_id=42) writes cycle_id=42 to the trades row."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(), cycle_id=42, decision_price=100.0)

        trade_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO trades" in c.args[0]
        )
        assert "cycle_id" in trade_call.args[0]
        assert trade_call.args[1][-1] == 42

    def test_cycle_id_defaults_to_none(self) -> None:
        """process() without cycle_id writes NULL (backward compatible with pre-Task-10 callers)."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill())

        trade_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO trades" in c.args[0]
        )
        assert trade_call.args[1][-1] is None

    def test_adjustment_cycle_id_stays_null(self) -> None:
        """record_adjustment is unchanged — no cycle_id param, column stays NULL by default."""
        db, conn = make_mock_db()
        processor = FillProcessor(db=db)

        processor.record_adjustment(
            symbol="SPY",
            environment="equity",
            quantity_delta=2.0,
            price=155.0,
            reason="reconciliation",
        )

        sql, params = conn.execute.call_args.args[0], conn.execute.call_args.args[1]
        assert "cycle_id" not in sql
        assert len(params) == 12  # unchanged column count — no cycle_id slot


class TestFillQualitySlippage:
    """Step 1(b): side-aware slippage_frac (positive = adverse, per V003 comment)."""

    def test_buy_adverse_slippage_is_positive(self) -> None:
        """Buy fills above decision price -> positive (adverse) slippage_frac."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(fill_price=105.0), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        slippage_frac = fq_call.args[1][4]
        assert slippage_frac == pytest.approx(0.05)

    def test_buy_favorable_slippage_is_negative(self) -> None:
        """Buy fills below decision price -> negative (favorable) slippage_frac."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(fill_price=95.0), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][4] == pytest.approx(-0.05)

    def test_sell_adverse_slippage_is_positive(self) -> None:
        """Sell fills below decision price -> positive (adverse) slippage_frac."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_sell_fill(fill_price=95.0), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][4] == pytest.approx(0.05)

    def test_sell_favorable_slippage_is_negative(self) -> None:
        """Sell fills above decision price -> negative (favorable) slippage_frac."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_sell_fill(fill_price=105.0), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][4] == pytest.approx(-0.05)

    def test_none_decision_price_yields_null_slippage(self) -> None:
        """decision_price=None (e.g. an emergency sell) -> slippage_frac NULL, row still written."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill())

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][1] is None  # decision_price_usd
        assert fq_call.args[1][4] is None  # slippage_frac


class TestFillQualityRealizedCost:
    """Step 1(c): realized_cost_frac = commission/(fill_price*quantity) + max(slippage_frac, 0.0)."""

    def test_adverse_slippage_adds_to_commission_frac(self) -> None:
        """Adverse slippage (positive) is added on top of the commission fraction."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        # commission=1.5, fill_price=105, quantity=10 -> commission_frac = 1.5/1050
        processor.process(
            _buy_fill(fill_price=105.0, quantity=10.0, commission=1.5), decision_price=100.0
        )

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        commission_frac = 1.5 / (105.0 * 10.0)
        expected = commission_frac + 0.05  # slippage_frac computed above
        assert fq_call.args[1][6] == pytest.approx(expected)

    def test_favorable_slippage_is_floored_at_zero(self) -> None:
        """Favorable slippage never subtracts from realized_cost_frac (floored at 0.0)."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        # fill below decision -> favorable (negative) slippage for a buy
        processor.process(
            _buy_fill(fill_price=95.0, quantity=10.0, commission=1.5), decision_price=100.0
        )

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        commission_frac = 1.5 / (95.0 * 10.0)
        assert fq_call.args[1][6] == pytest.approx(commission_frac)  # +max(-0.05, 0.0) == +0

    def test_realized_cost_frac_computed_even_without_decision_price(self) -> None:
        """No decision_price -> slippage term is 0, but commission-based cost is still recorded."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(fill_price=105.0, quantity=10.0, commission=1.5))

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        commission_frac = 1.5 / (105.0 * 10.0)
        assert fq_call.args[1][6] == pytest.approx(commission_frac)


class TestFillQualityAdjustmentExclusion:
    """Step 1(d): record_adjustment never writes a fill_quality row."""

    def test_adjustment_writes_no_fill_quality_row(self) -> None:
        """A reconciliation adjustment produces zero fill_quality INSERTs."""
        db, conn = make_mock_db()
        processor = FillProcessor(db=db)

        processor.record_adjustment(
            symbol="SPY",
            environment="equity",
            quantity_delta=2.0,
            price=155.0,
            reason="reconciliation",
        )

        fq_calls = [c for c in conn.execute.call_args_list if "fill_quality" in c.args[0]]
        assert fq_calls == []


class TestFillQualityExpectedCost:
    """Step 3: expected_cost_frac / expected_fill_price_usd — config-driven, side-aware."""

    def test_equity_expected_cost_frac_is_zero(self, exec_config: SwingRLConfig) -> None:
        """Equity is commission-free -> expected_cost_frac = 0.0."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        processor.process(_buy_fill(environment="equity", broker="alpaca"), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][5] == pytest.approx(0.0)
        # expected_fill_price_usd == decision_price (no cost applied) -> "100.0"
        assert fq_call.args[1][2] == str(round(100.0, 8))

    def test_crypto_expected_cost_frac_from_config(self, exec_config: SwingRLConfig) -> None:
        """Crypto's modeled cost is config.environment.crypto_transaction_cost_pct."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        fill = _buy_fill(
            symbol="BTCUSDT", environment="crypto", broker="binance_us", fill_price=50100.0
        )
        processor.process(fill, decision_price=50000.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        expected_cost_frac = exec_config.environment.crypto_transaction_cost_pct
        assert fq_call.args[1][5] == pytest.approx(expected_cost_frac)
        expected_fill_price = 50000.0 * (1 + expected_cost_frac)
        assert fq_call.args[1][2] == str(round(expected_fill_price, 8))

    def test_crypto_expected_fill_price_side_aware_for_sell(
        self, exec_config: SwingRLConfig
    ) -> None:
        """Sell side subtracts the modeled cost from decision_price (side-aware)."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        fill = _sell_fill(
            symbol="BTCUSDT", environment="crypto", broker="binance_us", fill_price=49900.0
        )
        processor.process(fill, decision_price=50000.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        expected_cost_frac = exec_config.environment.crypto_transaction_cost_pct
        expected_fill_price = 50000.0 * (1 - expected_cost_frac)
        assert fq_call.args[1][2] == str(round(expected_fill_price, 8))

    def test_no_config_injected_falls_back_to_zero_cost(self) -> None:
        """A FillProcessor built without config (legacy callers) never crashes; cost is 0.0."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)  # no config kwarg — binance_sim.py/reconciliation.py path

        fill = _buy_fill(symbol="BTCUSDT", environment="crypto", broker="binance_us")
        processor.process(fill, decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][5] == pytest.approx(0.0)


class TestFillQualityNumericAdaptation:
    """Step 3: NUMERIC columns are written as str(round(value, 8)) for psycopg adaptation."""

    def test_price_columns_are_strings(self) -> None:
        """decision_price_usd, expected_fill_price_usd, fill_price_usd are str, not float."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(fill_price=105.123456789), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        params = fq_call.args[1]
        assert isinstance(params[1], str)  # decision_price_usd
        assert isinstance(params[3], str)  # fill_price_usd
        assert params[3] == str(round(105.123456789, 8))

    def test_null_decision_price_stays_none_not_string(self) -> None:
        """A None decision_price is never coerced into the string "None"."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        processor.process(_buy_fill())

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][1] is None


class TestFillQualityTimeToFill:
    """Step 3: time_to_fill_ms derives from FillResult.submitted_at/filled_at (Task B)."""

    def test_computed_when_both_timestamps_present(self) -> None:
        """A 250ms gap between submitted_at and filled_at yields time_to_fill_ms == 250."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        fill = _buy_fill(
            submitted_at="2026-07-16T20:00:00.000000+00:00",
            filled_at="2026-07-16T20:00:00.250000+00:00",
        )
        processor.process(fill, decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][7] == 250

    def test_none_when_submitted_at_missing(self) -> None:
        """submitted_at=None (e.g. a broker that doesn't report it) -> time_to_fill_ms NULL."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        fill = _buy_fill(submitted_at=None, filled_at="2026-07-16T20:00:00.250000+00:00")
        processor.process(fill, decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][7] is None

    def test_none_when_filled_at_missing(self) -> None:
        """filled_at=None -> time_to_fill_ms NULL."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db)

        fill = _buy_fill(submitted_at="2026-07-16T20:00:00.000000+00:00", filled_at=None)
        processor.process(fill, decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][7] is None


class TestFillQualityFailOpen:
    """Fill-quality capture errors are logged and never disturb fill processing (Task 9 precedent)."""

    def test_fill_quality_db_error_does_not_raise(self) -> None:
        """A fill_quality INSERT failure is swallowed — process() still returns normally."""
        db, _conn = _mock_db_raising_on("fill_quality")
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(), cycle_id=7, decision_price=100.0)  # must not raise

    def test_trade_and_position_rows_still_written_when_fill_quality_fails(self) -> None:
        """The trades INSERT and position write happen before the fail-open fill_quality write."""
        db, conn = _mock_db_raising_on("fill_quality")
        processor = FillProcessor(db=db)

        processor.process(_buy_fill(), cycle_id=7, decision_price=100.0)

        assert any("INSERT INTO trades" in c.args[0] for c in conn.execute.call_args_list)
        assert any("INSERT INTO positions" in c.args[0] for c in conn.execute.call_args_list)


class TestFillQualityLiveDB:
    """Real-Postgres round-trip: fill_quality lands with correct NUMERIC adaptation.

    The mock-based tests above only prove the string value handed to conn.execute —
    nothing else proves psycopg actually adapts that string into the NUMERIC(18, 8)
    columns and it round-trips back as the expected value (mirrors Task 9's
    TestRecordCycleLiveDB precedent). Uses the mock_db fixture (auto-skips without
    DATABASE_URL).
    """

    def test_fill_quality_row_round_trips(self, mock_db: DatabaseManager) -> None:
        """A buy fill's fill_quality row lands with slippage/cost/time_to_fill intact."""
        processor = FillProcessor(db=mock_db)
        fill = FillResult(
            trade_id="fill-t10-live-001",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=105.0,
            commission=1.5,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="filled",
            submitted_at="2026-07-16T20:00:00.000000+00:00",
            filled_at="2026-07-16T20:00:00.250000+00:00",
        )

        processor.process(fill, decision_price=100.0)

        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM fill_quality WHERE trade_id = %s", (fill.trade_id,)
            ).fetchone()

        assert row is not None
        assert float(row["decision_price_usd"]) == pytest.approx(100.0)
        assert float(row["fill_price_usd"]) == pytest.approx(105.0)
        assert float(row["expected_fill_price_usd"]) == pytest.approx(100.0)  # no config -> 0 cost
        assert row["slippage_frac"] == pytest.approx(0.05)
        assert row["expected_cost_frac"] == pytest.approx(0.0)
        commission_frac = 1.5 / (105.0 * 10.0)
        assert row["realized_cost_frac"] == pytest.approx(commission_frac + 0.05)
        assert row["time_to_fill_ms"] == 250

    def test_trades_cycle_id_round_trips_null(self, mock_db: DatabaseManager) -> None:
        """process() without cycle_id writes NULL to trades.cycle_id (nullable FK)."""
        processor = FillProcessor(db=mock_db)
        fill = FillResult(
            trade_id="fill-t10-live-002",
            symbol="SPY",
            side="buy",
            quantity=5.0,
            fill_price=150.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="filled",
        )

        processor.process(fill)

        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT cycle_id FROM trades WHERE trade_id = %s", (fill.trade_id,)
            ).fetchone()

        assert row is not None
        assert row["cycle_id"] is None
