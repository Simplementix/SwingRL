"""Tests for PositionTracker -- portfolio state reader from SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.risk.position_tracker import PositionTracker


class TestPositionTrackerPortfolioValue:
    """PAPER-03: Portfolio value reads from SQLite positions + snapshots."""

    def test_returns_initial_capital_when_no_data(self, position_tracker: PositionTracker) -> None:
        """PAPER-03: Returns initial capital from config when no snapshot exists."""
        value = position_tracker.get_portfolio_value("equity")
        assert value == pytest.approx(400.0)

    def test_returns_initial_capital_crypto(self, position_tracker: PositionTracker) -> None:
        """PAPER-03: Returns crypto initial capital when no snapshot exists."""
        value = position_tracker.get_portfolio_value("crypto")
        assert value == pytest.approx(47.0)

    def test_returns_snapshot_value_when_available(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Returns latest snapshot total_value when snapshot exists."""
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, high_water_mark, "
                "daily_pnl, drawdown_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-03-09T10:00:00Z", "equity", 450.0, 100.0, 450.0, 5.0, 0.0),
            )
        value = position_tracker.get_portfolio_value("equity")
        assert value == pytest.approx(450.0)


class TestPositionTrackerPositions:
    """PAPER-03: Position reads from SQLite positions table."""

    def test_returns_empty_list_when_no_positions(self, position_tracker: PositionTracker) -> None:
        """PAPER-03: Returns empty list when no positions exist."""
        positions = position_tracker.get_positions("equity")
        assert positions == []

    def test_returns_positions_for_environment(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Returns positions filtered by environment."""
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("SPY", "equity", 2.0, 470.0, 475.0, 10.0, "2026-03-09T10:00:00Z"),
            )
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTCUSDT", "crypto", 0.001, 42000.0, 43000.0, 1.0, "2026-03-09T10:00:00Z"),
            )
        equity_pos = position_tracker.get_positions("equity")
        assert len(equity_pos) == 1
        assert equity_pos[0]["symbol"] == "SPY"


class TestPositionTrackerHighWaterMark:
    """PAPER-03: HWM reads from portfolio_snapshots."""

    def test_returns_initial_capital_when_no_snapshot(
        self, position_tracker: PositionTracker
    ) -> None:
        """PAPER-03: HWM defaults to initial capital when no snapshots exist."""
        hwm = position_tracker.get_high_water_mark("equity")
        assert hwm == pytest.approx(400.0)

    def test_returns_hwm_from_snapshot(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Returns HWM from latest snapshot for environment."""
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, high_water_mark, "
                "daily_pnl, drawdown_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-03-09T10:00:00Z", "equity", 420.0, 100.0, 425.0, 5.0, 0.0),
            )
        hwm = position_tracker.get_high_water_mark("equity")
        assert hwm == pytest.approx(425.0)


class TestPositionTrackerDailyPnl:
    """PAPER-03: Daily P&L from latest snapshot."""

    def test_returns_zero_when_no_snapshot(self, position_tracker: PositionTracker) -> None:
        """PAPER-03: Daily PnL is 0.0 when no snapshots exist."""
        pnl = position_tracker.get_daily_pnl("equity")
        assert pnl == pytest.approx(0.0)

    def test_returns_daily_pnl_from_todays_snapshot(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Returns daily_pnl from today's snapshot."""
        today_ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, high_water_mark, "
                "daily_pnl, drawdown_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (today_ts, "equity", 410.0, 100.0, 420.0, -5.0, 0.024),
            )
        pnl = position_tracker.get_daily_pnl("equity")
        assert pnl == pytest.approx(-5.0)


class TestPositionTrackerExposure:
    """PAPER-03: Exposure ratio = invested / total value."""

    def test_returns_zero_when_no_positions(self, position_tracker: PositionTracker) -> None:
        """PAPER-03: Exposure is 0.0 when no positions held."""
        exposure = position_tracker.get_exposure("equity")
        assert exposure == pytest.approx(0.0)

    def test_returns_correct_exposure(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Exposure = sum(abs(qty * last_price)) / portfolio_value."""
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("SPY", "equity", 0.5, 470.0, 480.0, 5.0, "2026-03-09T10:00:00Z"),
            )
        exposure = position_tracker.get_exposure("equity")
        # 0.5 * 480 = 240, portfolio_value = 400 (initial), exposure = 240/400 = 0.6
        assert exposure == pytest.approx(0.6)


class TestPositionTrackerRecordSnapshot:
    """PAPER-03: Snapshot recording with auto HWM and drawdown computation."""

    def test_record_snapshot_first_time(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: First snapshot uses portfolio_value as HWM."""
        position_tracker.record_snapshot(
            env="equity", portfolio_value=410.0, cash=100.0, daily_pnl=10.0
        )
        with mock_db.sqlite() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE environment = 'equity' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row["total_value"] == pytest.approx(410.0)
        assert row["high_water_mark"] == pytest.approx(410.0)
        assert row["drawdown_pct"] == pytest.approx(0.0)

    def test_record_snapshot_updates_hwm(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-03: Subsequent snapshot preserves HWM if portfolio dropped."""
        position_tracker.record_snapshot(
            env="equity", portfolio_value=420.0, cash=100.0, daily_pnl=20.0
        )
        position_tracker.record_snapshot(
            env="equity", portfolio_value=400.0, cash=80.0, daily_pnl=-20.0
        )
        with mock_db.sqlite() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE environment = 'equity' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        assert row["high_water_mark"] == pytest.approx(420.0)
        assert row["drawdown_pct"] == pytest.approx(1 - 400.0 / 420.0)


class TestPositionTrackerPortfolioStateArray:
    """PAPER-05: Portfolio state arrays match ObservationAssembler format."""

    def test_equity_state_shape(self, position_tracker: PositionTracker) -> None:
        """PAPER-05: Equity portfolio state is (35,) array."""
        state = position_tracker.get_portfolio_state_array("equity")
        assert state.shape == (35,)

    def test_crypto_state_shape(self, position_tracker: PositionTracker) -> None:
        """PAPER-05: Crypto portfolio state is (11,) array."""
        state = position_tracker.get_portfolio_state_array("crypto")
        assert state.shape == (11,)

    def test_equity_state_default_all_cash(self, position_tracker: PositionTracker) -> None:
        """PAPER-05: Default state is 100% cash with zero weights."""
        state = position_tracker.get_portfolio_state_array("equity")
        # [cash_ratio, exposure, daily_return,
        #  per-asset interleaved: weight, weight_dev, unrealized_pnl_pct, days_since_trade]
        assert state[0] == pytest.approx(1.0)  # cash_ratio
        assert state[1] == pytest.approx(0.0)  # exposure
        assert state[2] == pytest.approx(0.0)  # daily_return
        # All per-asset fields should be 0
        assert np.all(state[3:] == 0.0)

    def test_equity_state_with_positions(
        self, position_tracker: PositionTracker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-05: State reflects positions when held with interleaved layout."""
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("SPY", "equity", 0.5, 470.0, 480.0, 5.0, "2026-03-09T10:00:00Z"),
            )
        state = position_tracker.get_portfolio_state_array("equity")
        assert state.shape == (35,)
        # cash_ratio should be < 1.0 (some capital in SPY)
        assert state[0] < 1.0
        # SPY is at config index 0 (config order: SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK)
        # In interleaved layout: base_idx = 3 + 0*4 = 3
        spy_weight_idx = 3 + 0 * 4  # = 3
        assert state[spy_weight_idx] > 0.0  # SPY weight
        # weight_deviation = weight - 1/8
        assert state[spy_weight_idx + 1] == pytest.approx(state[spy_weight_idx] - 1.0 / 8.0)
        # unrealized_pnl_pct = (480 - 470) / 470
        assert state[spy_weight_idx + 2] == pytest.approx((480.0 - 470.0) / 470.0)
