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

        with mock_db.sqlite() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (buy_fill.trade_id,)
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

        with mock_db.sqlite() as conn:
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

        with mock_db.sqlite() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ? AND environment = ?",
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

        with mock_db.sqlite() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ? AND environment = ?",
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

        with mock_db.sqlite() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ? AND environment = ?",
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

        with mock_db.sqlite() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ? AND environment = ?",
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

        with mock_db.sqlite() as conn:
            pos = conn.execute(
                "SELECT * FROM positions WHERE symbol = ? AND environment = ?",
                ("SPY", "equity"),
            ).fetchone()

        assert pos["last_price"] == 150.0
