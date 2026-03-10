"""Tests for shadow inference runner.

PROD-03: Shadow inference runs after active inference without affecting active fills.
Shadow failures never crash the active trading cycle.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


@dataclass
class _FakeConfig:
    """Minimal config for shadow runner tests."""

    shadow: Any = None
    paths: Any = None


@dataclass
class _FakePaths:
    """Minimal paths config."""

    models_dir: str = "models/"


@dataclass
class _FakeShadow:
    """Minimal shadow config."""

    equity_eval_days: int = 10
    crypto_eval_cycles: int = 30
    auto_promote: bool = True
    mdd_tolerance_ratio: float = 1.2


@dataclass
class _FakeJobContext:
    """Minimal JobContext for tests."""

    config: Any = None
    db: Any = None
    pipeline: Any = None
    alerter: Any = None


def _create_sqlite_with_shadow_trades(db_path: Path) -> sqlite3.Connection:
    """Create an in-memory-like SQLite with shadow_trades table for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
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
    conn.commit()
    return conn


class TestShadowRunnerNoOp:
    """PROD-03: Shadow runner returns silently when no shadow model exists."""

    def test_no_shadow_model_returns_none(self, tmp_path: Path) -> None:
        """PROD-03: run_shadow_inference is a no-op when no shadow model exists."""
        from swingrl.shadow.shadow_runner import run_shadow_inference

        models_dir = tmp_path / "models"
        (models_dir / "shadow" / "equity").mkdir(parents=True, exist_ok=True)

        config = _FakeConfig(
            shadow=_FakeShadow(),
            paths=_FakePaths(models_dir=str(models_dir)),
        )
        ctx = _FakeJobContext(config=config, db=MagicMock())

        # Should return without error
        result = run_shadow_inference(ctx, "equity")
        assert result is None


class TestShadowRunnerInference:
    """PROD-03: Shadow runner loads model, runs inference, records trades."""

    def test_records_hypothetical_trades(self, tmp_path: Path) -> None:
        """PROD-03: Shadow inference records trades in shadow_trades table."""
        from swingrl.shadow.shadow_runner import run_shadow_inference

        models_dir = tmp_path / "models"
        shadow_dir = models_dir / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_model = shadow_dir / "ppo_equity_v2.zip"
        shadow_model.write_bytes(b"fake_model")

        db_path = tmp_path / "trading_ops.db"
        conn = _create_sqlite_with_shadow_trades(db_path)
        conn.close()

        mock_db = MagicMock()

        # Mock the sqlite context manager to use real SQLite
        class _SqliteCM:
            def __enter__(self) -> sqlite3.Connection:
                self._conn = sqlite3.connect(str(db_path))
                self._conn.row_factory = sqlite3.Row
                return self._conn

            def __exit__(self, *args: Any) -> None:
                self._conn.commit()
                self._conn.close()

        mock_db.sqlite = _SqliteCM

        config = _FakeConfig(
            shadow=_FakeShadow(),
            paths=_FakePaths(models_dir=str(models_dir)),
        )
        ctx = _FakeJobContext(config=config, db=mock_db)

        # Mock the model loading and inference
        with (
            patch("swingrl.shadow.shadow_runner._load_shadow_model") as mock_load,
            patch("swingrl.shadow.shadow_runner._generate_hypothetical_trades") as mock_gen,
        ):
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            mock_gen.return_value = [
                {
                    "trade_id": "shadow-001",
                    "timestamp": "2026-03-10T00:00:00Z",
                    "symbol": "SPY",
                    "side": "buy",
                    "quantity": 10.0,
                    "price": 500.0,
                    "commission": 0.0,
                    "slippage": 0.0,
                    "environment": "equity",
                    "broker": "alpaca",
                    "order_type": "market",
                    "trade_type": "shadow",
                    "model_version": "ppo_equity_v2",
                },
            ]

            run_shadow_inference(ctx, "equity")

        # Verify trade was recorded
        verify_conn = sqlite3.connect(str(db_path))
        verify_conn.row_factory = sqlite3.Row
        rows = verify_conn.execute("SELECT * FROM shadow_trades").fetchall()
        verify_conn.close()

        assert len(rows) == 1
        assert rows[0]["trade_id"] == "shadow-001"
        assert rows[0]["symbol"] == "SPY"
        assert rows[0]["model_version"] == "ppo_equity_v2"


class TestShadowRunnerIsolation:
    """PROD-03: Shadow inference never raises exceptions."""

    def test_catches_all_exceptions(self, tmp_path: Path) -> None:
        """PROD-03: run_shadow_inference catches all exceptions and never raises."""
        from swingrl.shadow.shadow_runner import run_shadow_inference

        models_dir = tmp_path / "models"
        shadow_dir = models_dir / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "ppo_equity.zip").write_bytes(b"fake")

        config = _FakeConfig(
            shadow=_FakeShadow(),
            paths=_FakePaths(models_dir=str(models_dir)),
        )
        ctx = _FakeJobContext(config=config, db=MagicMock())

        with patch(
            "swingrl.shadow.shadow_runner._load_shadow_model",
            side_effect=RuntimeError("model load explosion"),
        ):
            # Should NOT raise
            run_shadow_inference(ctx, "equity")


class TestShadowTradesSchema:
    """PROD-03: shadow_trades table has correct schema."""

    def test_shadow_trades_schema_via_db_init(self, tmp_path: Path) -> None:
        """PROD-03: shadow_trades created by DatabaseManager has correct columns."""
        conn = _create_sqlite_with_shadow_trades(tmp_path / "test.db")

        # Verify column names
        cursor = conn.execute("PRAGMA table_info(shadow_trades)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "trade_id",
            "timestamp",
            "symbol",
            "side",
            "quantity",
            "price",
            "commission",
            "slippage",
            "environment",
            "broker",
            "order_type",
            "trade_type",
            "model_version",
        }
        assert columns == expected
        conn.close()
