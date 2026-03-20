"""Tests for shadow model auto-promotion logic.

PROD-04: Auto-promotion fires when all 4 criteria are met after minimum evaluation period.
Failed shadow models move to archive with Discord alert.

Gate criteria:
1. Shadow annualized Sharpe > Active annualized Sharpe
2. Shadow MDD <= mdd_tolerance_ratio * Active MDD
3. Shadow profit factor > 1.5
4. No circuit breaker triggers during shadow period
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


@dataclass
class _FakeShadow:
    """Minimal shadow config for promoter tests."""

    equity_eval_days: int = 10
    crypto_eval_cycles: int = 30
    auto_promote: bool = True
    mdd_tolerance_ratio: float = 1.2


@dataclass
class _FakeConfig:
    """Minimal config for promoter tests."""

    shadow: _FakeShadow = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.shadow is None:
            self.shadow = _FakeShadow()


def _setup_db(db_path: Path) -> None:
    """Create shadow_trades, trades, portfolio_snapshots, and circuit_breaker_events tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL DEFAULT 0.0,
            slippage REAL DEFAULT 0.0,
            environment TEXT NOT NULL,
            broker TEXT,
            order_type TEXT,
            trade_type TEXT,
            model_version TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL DEFAULT 0.0,
            slippage REAL DEFAULT 0.0,
            environment TEXT NOT NULL,
            broker TEXT,
            order_type TEXT,
            trade_type TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp TEXT NOT NULL,
            environment TEXT NOT NULL,
            total_value REAL NOT NULL,
            equity_value REAL,
            crypto_value REAL,
            cash_balance REAL,
            high_water_mark REAL,
            daily_pnl REAL,
            drawdown_pct REAL,
            PRIMARY KEY (timestamp, environment)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS circuit_breaker_events (
            event_id TEXT PRIMARY KEY,
            environment TEXT NOT NULL,
            triggered_at TEXT NOT NULL,
            resumed_at TEXT,
            trigger_value REAL,
            threshold REAL,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def _make_mock_db(db_path: Path) -> MagicMock:
    """Create a mock DatabaseManager that uses real SQLite for queries."""
    mock_db = MagicMock()

    class _SqliteCM:
        def __enter__(self) -> sqlite3.Connection:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row
            return self._conn

        def __exit__(self, *args: Any) -> None:
            self._conn.commit()
            self._conn.close()

    mock_db.sqlite = _SqliteCM
    return mock_db


def _insert_shadow_trades_paired(
    db_path: Path,
    buy_sell_pairs: list[tuple[float, float]],
    env: str = "equity",
    symbol: str = "SPY",
    start_date: str = "2026-03-01",
) -> None:
    """Insert paired buy/sell shadow trades with explicit prices.

    Each pair is (buy_price, sell_price). Creates 2 trades per pair.
    """
    conn = sqlite3.connect(str(db_path))
    idx = 0
    for i, (buy_price, sell_price) in enumerate(buy_sell_pairs):
        day_buy = i * 2 + 1
        day_sell = i * 2 + 2
        conn.execute(
            "INSERT INTO shadow_trades "
            "(trade_id, timestamp, symbol, side, quantity, price, environment, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"shadow-{idx}",
                f"2026-03-{day_buy:02d}T00:00:00Z",
                symbol,
                "buy",
                10,
                buy_price,
                env,
                "v2",
            ),
        )
        idx += 1
        conn.execute(
            "INSERT INTO shadow_trades "
            "(trade_id, timestamp, symbol, side, quantity, price, environment, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"shadow-{idx}",
                f"2026-03-{day_sell:02d}T00:00:00Z",
                symbol,
                "sell",
                10,
                sell_price,
                env,
                "v2",
            ),
        )
        idx += 1
    conn.commit()
    conn.close()


def _insert_portfolio_snapshots(
    db_path: Path,
    values: list[float],
    env: str = "equity",
) -> None:
    """Insert portfolio_snapshots with total_value series for the active model."""
    conn = sqlite3.connect(str(db_path))
    for i, val in enumerate(values):
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp, environment, total_value, cash_balance, high_water_mark, "
            "daily_pnl, drawdown_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"2026-03-{i + 1:02d}T00:00:00Z",
                env,
                val,
                val * 0.5,
                max(values[: i + 1]),
                0.0,
                0.0,
            ),
        )
    conn.commit()
    conn.close()


def _insert_cb_event(
    db_path: Path,
    env: str = "equity",
    triggered_at: str = "2026-03-05T12:00:00Z",
) -> None:
    """Insert a circuit breaker event."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO circuit_breaker_events "
        "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("cb-1", env, triggered_at, -0.08, -0.10, "daily loss"),
    )
    conn.commit()
    conn.close()


# ---- Test data ----
# Shadow: 10 winning pairs (buy low, sell high) → high Sharpe, high PF
_WINNING_PAIRS = [
    (100, 105),
    (105, 110),
    (110, 116),
    (116, 122),
    (122, 128),
    (128, 134),
    (134, 141),
    (141, 148),
    (148, 155),
    (155, 163),
]

# Shadow: mixed pairs (some wins, some losses) → low Sharpe, low PF
_MIXED_PAIRS = [
    (100, 102),
    (102, 99),
    (99, 101),
    (101, 98),
    (98, 100),
    (100, 97),
    (97, 99),
    (99, 96),
    (96, 98),
    (98, 100),
]

# Shadow: big drawdown pairs
_HIGH_MDD_PAIRS = [
    (100, 102),
    (102, 80),
    (80, 60),
    (60, 70),
    (70, 75),
    (75, 80),
    (80, 85),
    (85, 90),
    (90, 95),
    (95, 100),
]

# Active: steady upward portfolio (good Sharpe, low MDD)
_ACTIVE_GOOD_VALUES = [
    400,
    402,
    404,
    406,
    408,
    410,
    412,
    414,
    416,
    418,
    420,
    422,
    424,
    426,
    428,
    430,
    432,
    434,
    436,
    438,
]

# Active: flat/noisy portfolio (low Sharpe)
_ACTIVE_FLAT_VALUES = [
    400,
    402,
    399,
    401,
    398,
    400,
    397,
    399,
    396,
    398,
    400,
    403,
    400,
    402,
    399,
    401,
    398,
    400,
    397,
    399,
]


class TestInsufficientData:
    """PROD-04: Returns False when fewer than minimum cycles completed."""

    def test_returns_false_when_insufficient_equity_trades(self, tmp_path: Path) -> None:
        """PROD-04: evaluate_shadow_promotion returns False with < 10 equity trades."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Only 3 pairs = 6 trades, less than 10
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS[:3], env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_returns_false_when_insufficient_crypto_cycles(self, tmp_path: Path) -> None:
        """PROD-04: evaluate_shadow_promotion returns False with < 30 crypto trades."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # 10 pairs = 20 trades, less than 30
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="crypto")

        config = _FakeConfig(shadow=_FakeShadow(crypto_eval_cycles=30))
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "crypto", lifecycle, alerter)
        assert result is False


class TestPromotionCriteria:
    """PROD-04: Promotion evaluates 4 criteria after minimum eval period."""

    def test_promotes_when_all_criteria_met(self, tmp_path: Path) -> None:
        """PROD-04: Returns True when all 4 gates pass."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow: 10 winning pairs (20 trades >= 10 min) → high Sharpe, high PF
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")
        # Active: flat/noisy portfolio → low Sharpe
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is True
        lifecycle.promote.assert_called_once_with("equity")

    def test_returns_false_when_shadow_sharpe_lower(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when Shadow Sharpe <= Active Sharpe."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow: mixed trades → low Sharpe
        _insert_shadow_trades_paired(db_path, _MIXED_PAIRS, env="equity")
        # Active: steady upward portfolio → high Sharpe
        _insert_portfolio_snapshots(db_path, _ACTIVE_GOOD_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_returns_false_when_mdd_exceeds_tolerance(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when Shadow MDD > tolerance * Active MDD."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow: big drawdown pairs → high MDD
        _insert_shadow_trades_paired(db_path, _HIGH_MDD_PAIRS, env="equity")
        # Active: steady upward → low MDD
        _insert_portfolio_snapshots(db_path, _ACTIVE_GOOD_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_returns_false_when_cb_triggered(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when circuit breaker triggered during shadow period."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")
        # CB event during shadow period
        _insert_cb_event(db_path, env="equity", triggered_at="2026-03-05T12:00:00Z")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_returns_false_when_profit_factor_too_low(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when shadow profit factor <= 1.5."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Mixed pairs have low profit factor (close to 1.0)
        _insert_shadow_trades_paired(db_path, _MIXED_PAIRS, env="equity")
        # Active: flat to let Sharpe pass
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_cb_before_shadow_period_does_not_block(self, tmp_path: Path) -> None:
        """PROD-04: CB events before shadow start date do not block promotion."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow trades start 2026-03-01
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")
        # CB event BEFORE shadow period
        _insert_cb_event(db_path, env="equity", triggered_at="2026-02-15T12:00:00Z")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        # Should pass since CB is before shadow period
        assert result is True


class TestPromotionLifecycle:
    """PROD-04: On promotion, active is archived and shadow becomes active."""

    def test_promote_calls_lifecycle(self, tmp_path: Path) -> None:
        """PROD-04: lifecycle.promote() is called on successful evaluation."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        lifecycle.promote.assert_called_once_with("equity")

    def test_discord_alert_on_promotion(self, tmp_path: Path) -> None:
        """PROD-04: Discord alert sent with comparison stats after promotion."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_FLAT_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        alerter.send_alert.assert_called_once()
        alert_args = alerter.send_alert.call_args
        assert "promot" in alert_args[0][1].lower() or "promot" in alert_args[0][2].lower()


class TestFailedShadowArchival:
    """PROD-04: Failed shadow models are archived with Discord alert."""

    def test_archives_failed_shadow_model(self, tmp_path: Path) -> None:
        """PROD-04: On failure after eval period, shadow model moves to archive."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _MIXED_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_GOOD_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False
        lifecycle.archive_shadow.assert_called_once_with("equity")

    def test_discord_alert_on_failure(self, tmp_path: Path) -> None:
        """PROD-04: Discord alert sent with comparison stats on shadow failure."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _MIXED_PAIRS, env="equity")
        _insert_portfolio_snapshots(db_path, _ACTIVE_GOOD_VALUES, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        alerter.send_alert.assert_called_once()
        alert_args = alerter.send_alert.call_args
        # Should mention failure/archive
        assert "fail" in alert_args[0][2].lower() or "archive" in alert_args[0][2].lower()


class TestAnnualizedSharpeUsed:
    """PROD-04: Sharpe is annualized using agents/metrics.py framework."""

    def test_sharpe_is_annualized(self, tmp_path: Path) -> None:
        """PROD-04: _safe_annualized_sharpe uses annualized_sharpe from agents.metrics."""
        import numpy as np

        from swingrl.shadow.promoter import _safe_annualized_sharpe

        # 20 returns of +1% each: mean=0.01, std~0, but with slight noise
        rng = np.random.default_rng(42)
        returns = 0.01 + rng.normal(0, 0.002, 20)

        # Un-annualized would be ~5.0; annualized (252 trading days) should be much larger
        result = _safe_annualized_sharpe(returns, 252.0)
        assert result > 10.0, f"Expected annualized Sharpe >> 1, got {result}"

    def test_returns_zero_for_insufficient_data(self, tmp_path: Path) -> None:
        """PROD-04: Returns 0.0 when < 2 data points."""
        import numpy as np

        from swingrl.shadow.promoter import _safe_annualized_sharpe

        assert _safe_annualized_sharpe(np.array([0.01]), 252.0) == 0.0
        assert _safe_annualized_sharpe(np.array([]), 252.0) == 0.0


class TestProfitFactorGate:
    """PROD-04: Profit factor computed from paired trades."""

    def test_winning_trades_high_pf(self, tmp_path: Path) -> None:
        """PROD-04: All-winning paired trades produce high profit factor."""
        from swingrl.shadow.promoter import _compute_profit_factor

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _WINNING_PAIRS, env="equity")

        mock_db = _make_mock_db(db_path)
        pf = _compute_profit_factor(mock_db, "shadow_trades", "equity")
        assert pf == float("inf"), f"All wins should be inf PF, got {pf}"

    def test_mixed_trades_low_pf(self, tmp_path: Path) -> None:
        """PROD-04: Mixed winning/losing trades produce low profit factor."""
        from swingrl.shadow.promoter import _compute_profit_factor

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades_paired(db_path, _MIXED_PAIRS, env="equity")

        mock_db = _make_mock_db(db_path)
        pf = _compute_profit_factor(mock_db, "shadow_trades", "equity")
        assert pf < 1.5, f"Mixed trades should have PF < 1.5, got {pf}"
