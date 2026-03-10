"""Tests for shadow model auto-promotion logic.

PROD-04: Auto-promotion fires when all 3 criteria are met after minimum evaluation period.
Failed shadow models move to archive with Discord alert.
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
    """Create shadow_trades, trades, and circuit_breaker_events tables."""
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


def _insert_shadow_trades(
    db_path: Path,
    count: int,
    env: str = "equity",
    base_price: float = 100.0,
    price_delta: float = 1.0,
) -> None:
    """Insert shadow trades with ascending prices (profitable)."""
    conn = sqlite3.connect(str(db_path))
    for i in range(count):
        side = "buy" if i % 2 == 0 else "sell"
        price = base_price + (i * price_delta)
        conn.execute(
            "INSERT INTO shadow_trades "
            "(trade_id, timestamp, symbol, side, quantity, price, environment, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"shadow-{i}", f"2026-03-{i + 1:02d}T00:00:00Z", "SPY", side, 10, price, env, "v2"),
        )
    conn.commit()
    conn.close()


def _insert_active_trades(
    db_path: Path,
    count: int,
    env: str = "equity",
    base_price: float = 100.0,
    price_delta: float = 0.5,
) -> None:
    """Insert active trades with smaller price increases (lower performance)."""
    conn = sqlite3.connect(str(db_path))
    for i in range(count):
        side = "buy" if i % 2 == 0 else "sell"
        price = base_price + (i * price_delta)
        conn.execute(
            "INSERT INTO trades "
            "(trade_id, timestamp, symbol, side, quantity, price, environment) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"active-{i}", f"2026-03-{i + 1:02d}T00:00:00Z", "SPY", side, 10, price, env),
        )
    conn.commit()
    conn.close()


def _insert_cb_event(db_path: Path, env: str = "equity") -> None:
    """Insert a circuit breaker event during the shadow evaluation period."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO circuit_breaker_events "
        "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("cb-1", env, "2026-03-05T12:00:00Z", -0.08, -0.10, "daily loss"),
    )
    conn.commit()
    conn.close()


class TestInsufficientData:
    """PROD-04: Returns False when fewer than minimum cycles completed."""

    def test_returns_false_when_insufficient_equity_trades(self, tmp_path: Path) -> None:
        """PROD-04: evaluate_shadow_promotion returns False with < 10 equity trades."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades(db_path, count=5, env="equity")  # < 10

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
        _insert_shadow_trades(db_path, count=15, env="crypto")  # < 30

        config = _FakeConfig(shadow=_FakeShadow(crypto_eval_cycles=30))
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "crypto", lifecycle, alerter)
        assert result is False


class TestPromotionCriteria:
    """PROD-04: Promotion evaluates 3 criteria after minimum eval period."""

    def test_promotes_when_all_criteria_met(self, tmp_path: Path) -> None:
        """PROD-04: Returns True when Sharpe >= active, MDD <= 120% active, no CB."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow with better performance
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=2.0)
        # Active with worse performance
        _insert_active_trades(db_path, count=20, env="equity", price_delta=0.5)

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is True
        lifecycle.promote.assert_called_once_with("equity")

    def test_returns_false_when_shadow_sharpe_lower(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when Shadow Sharpe < Active Sharpe."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow with lower performance
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=0.1)
        # Active with higher performance
        _insert_active_trades(db_path, count=20, env="equity", price_delta=3.0)

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False

    def test_returns_false_when_mdd_exceeds_tolerance(self, tmp_path: Path) -> None:
        """PROD-04: Returns False when Shadow MDD > 120% Active MDD."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        # Shadow trades with a big drawdown (price drops significantly)
        conn = sqlite3.connect(str(db_path))
        prices = [
            100,
            102,
            80,
            60,
            70,
            75,
            78,
            80,
            85,
            90,
            92,
            95,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
        ]
        for i, price in enumerate(prices):
            side = "buy" if i % 2 == 0 else "sell"
            conn.execute(
                "INSERT INTO shadow_trades "
                "(trade_id, timestamp, symbol, side, quantity, price, environment, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"shadow-{i}",
                    f"2026-03-{i + 1:02d}T00:00:00Z",
                    "SPY",
                    side,
                    10,
                    price,
                    "equity",
                    "v2",
                ),
            )
        # Active trades with very small drawdown
        for i, price in enumerate(
            [
                100,
                101,
                100.5,
                101,
                101.5,
                102,
                102.5,
                103,
                103.5,
                104,
                104.5,
                105,
                105.5,
                106,
                106.5,
                107,
                107.5,
                108,
                108.5,
                109,
            ]
        ):
            side = "buy" if i % 2 == 0 else "sell"
            conn.execute(
                "INSERT INTO trades "
                "(trade_id, timestamp, symbol, side, quantity, price, environment) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"active-{i}", f"2026-03-{i + 1:02d}T00:00:00Z", "SPY", side, 10, price, "equity"),
            )
        conn.commit()
        conn.close()

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
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=2.0)
        _insert_active_trades(db_path, count=20, env="equity", price_delta=0.5)
        _insert_cb_event(db_path, env="equity")

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        result = evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        assert result is False


class TestPromotionLifecycle:
    """PROD-04: On promotion, active is archived and shadow becomes active."""

    def test_promote_calls_lifecycle(self, tmp_path: Path) -> None:
        """PROD-04: lifecycle.promote() is called on successful evaluation."""
        from swingrl.shadow.promoter import evaluate_shadow_promotion

        db_path = tmp_path / "trading_ops.db"
        _setup_db(db_path)
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=2.0)
        _insert_active_trades(db_path, count=20, env="equity", price_delta=0.5)

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
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=2.0)
        _insert_active_trades(db_path, count=20, env="equity", price_delta=0.5)

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
        # Shadow with lower performance (will fail)
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=0.1)
        _insert_active_trades(db_path, count=20, env="equity", price_delta=3.0)

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
        _insert_shadow_trades(db_path, count=20, env="equity", price_delta=0.1)
        _insert_active_trades(db_path, count=20, env="equity", price_delta=3.0)

        config = _FakeConfig()
        mock_db = _make_mock_db(db_path)
        lifecycle = MagicMock()
        alerter = MagicMock()

        evaluate_shadow_promotion(config, mock_db, "equity", lifecycle, alerter)
        alerter.send_alert.assert_called_once()
        alert_args = alerter.send_alert.call_args
        # Should mention failure/archive
        assert "fail" in alert_args[0][2].lower() or "archive" in alert_args[0][2].lower()
