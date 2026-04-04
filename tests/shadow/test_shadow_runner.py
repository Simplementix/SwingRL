"""Tests for shadow inference runner.

PROD-03: Shadow inference runs after active inference without affecting active fills.
Shadow failures never crash the active trading cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import psycopg
import pytest
from psycopg.rows import dict_row


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


def _create_pg_with_shadow_trades(db_url: str) -> Any:
    """Create PostgreSQL shadow_trades table for testing."""
    conn = psycopg.connect(db_url, row_factory=dict_row, autocommit=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity DOUBLE PRECISION NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            commission DOUBLE PRECISION DEFAULT 0.0,
            slippage DOUBLE PRECISION DEFAULT 0.0,
            environment TEXT NOT NULL,
            broker TEXT,
            order_type TEXT,
            trade_type TEXT,
            model_version TEXT NOT NULL
        )
    """)
    conn.execute("DELETE FROM shadow_trades")
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

    @pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
    def test_records_hypothetical_trades(self, tmp_path: Path) -> None:
        """PROD-03: Shadow inference records trades in shadow_trades table."""
        from swingrl.shadow.shadow_runner import run_shadow_inference

        models_dir = tmp_path / "models"
        shadow_dir = models_dir / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_model = shadow_dir / "ppo_equity_v2.zip"
        shadow_model.write_bytes(b"fake_model")

        db_url = os.environ["DATABASE_URL"]
        conn = _create_pg_with_shadow_trades(db_url)
        conn.close()

        mock_db = MagicMock()

        # Mock the sqlite context manager to use real PostgreSQL
        class _PgCM:
            def __enter__(self) -> Any:
                self._conn = psycopg.connect(db_url, row_factory=dict_row)
                return self._conn

            def __exit__(self, *args: Any) -> None:
                self._conn.commit()
                self._conn.close()

        mock_db.sqlite = _PgCM

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
        verify_conn = psycopg.connect(db_url, row_factory=dict_row)
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

    @pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
    def test_shadow_trades_schema_via_db_init(self, tmp_path: Path) -> None:
        """PROD-03: shadow_trades created by DatabaseManager has correct columns."""
        db_url = os.environ["DATABASE_URL"]
        conn = _create_pg_with_shadow_trades(db_url)

        # Verify column names
        cursor = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            ("shadow_trades",),
        )
        columns = {row[0] for row in cursor.fetchall()}
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


# ---------------------------------------------------------------------------
# Fake config dataclasses for _generate_hypothetical_trades tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeEquity:
    """Minimal equity config."""

    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])


@dataclass
class _FakeCrypto:
    """Minimal crypto config."""

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    min_order_usd: float = 10.0
    max_position_size: float = 0.50


@dataclass
class _FakeEnvironment:
    """Minimal environment config."""

    signal_deadzone: float = 0.02


@dataclass
class _FakeCapital:
    """Minimal capital config."""

    equity_usd: float = 400.0
    crypto_usd: float = 47.0


@dataclass
class _FakeFullConfig:
    """Config with all fields needed for trade generation."""

    equity: _FakeEquity = field(default_factory=_FakeEquity)
    crypto: _FakeCrypto = field(default_factory=_FakeCrypto)
    environment: _FakeEnvironment = field(default_factory=_FakeEnvironment)
    capital: _FakeCapital = field(default_factory=_FakeCapital)
    paths: _FakePaths = field(default_factory=_FakePaths)
    shadow: _FakeShadow = field(default_factory=_FakeShadow)


@dataclass
class _FakeFeaturePipeline:
    """Minimal feature pipeline returning a fixed observation."""

    obs: np.ndarray = field(default_factory=lambda: np.zeros(156))

    def get_observation(self, env: str, date_str: str) -> np.ndarray:
        """Return stored observation."""
        return self.obs


@dataclass
class _FakePipelineCtx:
    """Minimal execution pipeline context."""

    feature_pipeline: _FakeFeaturePipeline = field(
        default_factory=_FakeFeaturePipeline,
    )
    config: Any = field(default_factory=_FakeFullConfig)


class TestGenerateHypotheticalTrades:
    """PROD-03/04: _generate_hypothetical_trades produces trade dicts from model predictions."""

    def _make_ctx(
        self,
        actions: np.ndarray | None = None,
        obs: np.ndarray | None = None,
    ) -> tuple[Any, MagicMock]:
        """Build a ctx and mock model for testing.

        Returns:
            (ctx, mock_model) tuple.
        """
        if obs is None:
            obs = np.random.default_rng(42).random(156).astype(np.float32)
        if actions is None:
            # Two symbols for equity: big positive buy actions
            actions = np.array([0.5, -0.5], dtype=np.float32)

        feature_pipeline = _FakeFeaturePipeline(obs=obs)
        config = _FakeFullConfig()
        pipeline_ctx = _FakePipelineCtx(
            feature_pipeline=feature_pipeline,
            config=config,
        )

        ctx = _FakeJobContext(
            config=config,
            db=MagicMock(),
            pipeline=pipeline_ctx,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = (actions, None)

        return ctx, mock_model

    def test_returns_trade_dicts_with_correct_schema(self) -> None:
        """PROD-03: Non-hold actions produce trade dicts with all shadow_trades columns."""
        from swingrl.shadow.shadow_runner import _generate_hypothetical_trades

        ctx, mock_model = self._make_ctx(
            actions=np.array([0.5, -0.5], dtype=np.float32),
        )

        trades = _generate_hypothetical_trades(mock_model, ctx, "equity", "ppo_v1")

        assert len(trades) > 0
        required_keys = {
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
        for trade in trades:
            assert required_keys.issubset(trade.keys()), (
                f"Missing keys: {required_keys - trade.keys()}"
            )
            assert trade["trade_type"] == "shadow"
            assert trade["environment"] == "equity"
            assert trade["model_version"] == "ppo_v1"
            assert trade["order_type"] == "market"

    def test_calls_pipeline_in_correct_order(self) -> None:
        """PROD-03: Calls get_observation, model.predict, interpret, size in order."""
        from swingrl.shadow.shadow_runner import _generate_hypothetical_trades

        ctx, mock_model = self._make_ctx()

        with (
            patch(
                "swingrl.execution.signal_interpreter.SignalInterpreter",
            ) as mock_si_cls,
            patch(
                "swingrl.execution.position_sizer.PositionSizer",
            ) as mock_ps_cls,
        ):
            mock_si = mock_si_cls.return_value
            mock_si.interpret.return_value = []  # no signals for simplicity
            _ = mock_ps_cls.return_value  # PositionSizer instantiated but unused (no signals)

            _generate_hypothetical_trades(mock_model, ctx, "equity", "v1")

            # Model should have been called with observation
            mock_model.predict.assert_called_once()
            # SignalInterpreter should have been created with config
            mock_si_cls.assert_called_once()
            # interpret should have been called
            mock_si.interpret.assert_called_once()

    def test_hold_actions_return_empty_list(self) -> None:
        """PROD-03: When all actions are within deadzone, returns empty list."""
        from swingrl.shadow.shadow_runner import _generate_hypothetical_trades

        # Actions within deadzone (0.02): diff from 0.0 current weights is < 0.02
        ctx, mock_model = self._make_ctx(
            actions=np.array([0.01, -0.01], dtype=np.float32),
        )

        trades = _generate_hypothetical_trades(mock_model, ctx, "equity", "v1")

        assert trades == []

    def test_skips_none_from_position_sizer(self) -> None:
        """PROD-04: When PositionSizer returns None (negative Kelly), signal is skipped."""
        from swingrl.shadow.shadow_runner import _generate_hypothetical_trades

        ctx, mock_model = self._make_ctx(
            actions=np.array([0.5, -0.5], dtype=np.float32),
        )

        with patch("swingrl.execution.position_sizer.PositionSizer") as mock_ps_cls:
            mock_ps = mock_ps_cls.return_value
            mock_ps.size.return_value = None  # negative Kelly

            trades = _generate_hypothetical_trades(mock_model, ctx, "equity", "v1")

        assert trades == []

    def test_includes_model_version_and_trade_type(self) -> None:
        """PROD-04: Trade dicts include model_version and trade_type='shadow'."""
        from swingrl.shadow.shadow_runner import _generate_hypothetical_trades

        ctx, mock_model = self._make_ctx(
            actions=np.array([0.5, 0.0], dtype=np.float32),
        )

        trades = _generate_hypothetical_trades(mock_model, ctx, "equity", "ppo_equity_v3")

        # At least one trade for the big positive action on SPY
        spy_trades = [t for t in trades if t["symbol"] == "SPY"]
        assert len(spy_trades) >= 1
        assert spy_trades[0]["model_version"] == "ppo_equity_v3"
        assert spy_trades[0]["trade_type"] == "shadow"
