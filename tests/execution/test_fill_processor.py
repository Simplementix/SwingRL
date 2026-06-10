"""Tests for FillProcessor database recording and position management.

PAPER-10: FillProcessor records fills to trades table and maintains
positions with cost basis tracking and quantity management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.types import FillResult

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
