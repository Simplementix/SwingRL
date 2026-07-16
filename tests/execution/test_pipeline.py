"""Tests for ExecutionPipeline orchestrator.

Verifies weight-based rebalancing execution pipeline with mocked dependencies:
model loading, per-algo VecNormalize, ensemble blending, process_actions, and
weight-based delta order generation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from swingrl.execution.pipeline import ExecutionPipeline
from swingrl.execution.types import FillResult


@pytest.fixture
def mock_feature_pipeline() -> MagicMock:
    """Mock FeaturePipeline returning a dummy observation."""
    fp = MagicMock()
    fp.get_observation.return_value = np.zeros(156, dtype=np.float32)
    return fp


@pytest.fixture
def pipeline(
    exec_config: Any,
    mock_db: Any,
    mock_feature_pipeline: MagicMock,
    mock_alerter: MagicMock,
    tmp_path: Path,
) -> ExecutionPipeline:
    """Create ExecutionPipeline with mocked dependencies."""
    return ExecutionPipeline(
        config=exec_config,
        db=mock_db,
        feature_pipeline=mock_feature_pipeline,
        alerter=mock_alerter,
        models_dir=tmp_path / "models",
    )


def _per_algo_obs_dict(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Helper: build per-algo normalized obs dict (same obs for all algos)."""
    return {"ppo": obs, "a2c": obs, "sac": obs}


class TestExecuteCycle:
    """Test the execute_cycle orchestrator method."""

    def test_cb_halted_returns_empty(self, pipeline: ExecutionPipeline) -> None:
        """PAPER-09: Circuit breaker halt short-circuits to empty list."""
        # Trigger CB for equity
        with pipeline._db.connection() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_events "
                "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ("cb1", "equity", "2099-01-01T00:00:00+00:00", 0.15, 0.10, "test"),
            )

        result = pipeline.execute_cycle("equity")
        assert result == []

    def test_dry_run_skips_submission(self, pipeline: ExecutionPipeline) -> None:
        """PAPER-09: Dry-run logs actions but does not submit orders."""
        obs = np.zeros(156, dtype=np.float32)
        # Model outputs 9 dims (8 assets + 1 cash) for process_actions softmax
        with patch.object(pipeline, "_load_models") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = (
                np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
                None,
            )
            mock_load.return_value = {
                "ppo": (mock_model, None),
                "a2c": (mock_model, None),
                "sac": (mock_model, None),
            }

            with patch.object(pipeline, "_get_ensemble_weights") as mock_weights:
                mock_weights.return_value = {"ppo": 0.4, "a2c": 0.3, "sac": 0.3}

                with patch.object(pipeline, "_normalize_observation") as mock_norm:
                    mock_norm.return_value = _per_algo_obs_dict(obs)

                    mock_adapter = MagicMock()
                    mock_adapter.get_current_price.return_value = 450.0

                    with patch.object(pipeline, "_get_adapter") as mock_get_adapter:
                        mock_get_adapter.return_value = mock_adapter

                        result = pipeline.execute_cycle("equity", dry_run=True)

                        # Dry-run should not produce fills (no broker submission)
                        assert isinstance(result, list)
                        assert result == []
                        # Adapter submit_order should NOT be called
                        mock_adapter.submit_order.assert_not_called()

    def test_full_cycle_with_mocked_stages(self, pipeline: ExecutionPipeline) -> None:
        """PAPER-09: Full end-to-end cycle with weight-based rebalancing."""
        fill = FillResult(
            trade_id="test-fill-1",
            symbol="SPY",
            side="buy",
            quantity=1.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        obs = np.zeros(156, dtype=np.float32)
        with patch.object(pipeline, "_load_models") as mock_load:
            mock_model = MagicMock()
            # Strong buy on SPY (idx 0), rest near zero, cash dim last
            mock_model.predict.return_value = (
                np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
            )
            mock_load.return_value = {
                "ppo": (mock_model, None),
                "a2c": (mock_model, None),
                "sac": (mock_model, None),
            }

            with patch.object(pipeline, "_get_ensemble_weights") as mock_weights:
                mock_weights.return_value = {"ppo": 0.4, "a2c": 0.3, "sac": 0.3}

                with patch.object(pipeline, "_normalize_observation") as mock_norm:
                    mock_norm.return_value = _per_algo_obs_dict(obs)

                    mock_adapter = MagicMock()
                    mock_adapter.submit_order.return_value = fill
                    mock_adapter.get_current_price.return_value = 450.0

                    with patch.object(pipeline, "_get_adapter") as mock_get_adapter:
                        mock_get_adapter.return_value = mock_adapter

                        result = pipeline.execute_cycle("equity")
                        assert isinstance(result, list)

    def test_risk_veto_catches_and_continues(self, pipeline: ExecutionPipeline) -> None:
        """PAPER-09: RiskVetoError on one symbol does not abort entire cycle."""
        obs = np.zeros(156, dtype=np.float32)
        with patch.object(pipeline, "_load_models") as mock_load:
            mock_model = MagicMock()
            # Buy signals on first two assets
            mock_model.predict.return_value = (
                np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                None,
            )
            mock_load.return_value = {
                "ppo": (mock_model, None),
                "a2c": (mock_model, None),
                "sac": (mock_model, None),
            }

            with patch.object(pipeline, "_get_ensemble_weights") as mock_weights:
                mock_weights.return_value = {"ppo": 0.4, "a2c": 0.3, "sac": 0.3}

                with patch.object(pipeline, "_normalize_observation") as mock_norm:
                    mock_norm.return_value = _per_algo_obs_dict(obs)

                    mock_adapter = MagicMock()
                    mock_adapter.get_current_price.return_value = 450.0

                    with patch.object(pipeline, "_get_adapter") as mock_get_adapter:
                        mock_get_adapter.return_value = mock_adapter

                        # Pipeline should handle RiskVetoError gracefully
                        result = pipeline.execute_cycle("equity")
                        assert isinstance(result, list)

    def test_turbulence_crash_protection(self, pipeline: ExecutionPipeline) -> None:
        """PAPER-20: High turbulence triggers CB and returns empty."""
        with patch.object(pipeline, "_check_turbulence") as mock_turb:
            mock_turb.return_value = True  # turbulence exceeded

            result = pipeline.execute_cycle("equity")
            assert result == []


class TestTurbulenceHaltBaseline:
    """F1 regression (Task 6c): the nonzero baseline reaches RiskManager."""

    def test_check_turbulence_reaches_risk_manager(self, pipeline: ExecutionPipeline) -> None:
        """A nonzero baseline must reach RiskManager.check_turbulence.

        Old code queried a phantom ``turbulence`` column (always 0.0) so the
        ``_check_turbulence`` guard short-circuited and the halt never fired.
        The baseline now comes from FeaturePipeline.turbulence_halt_baseline.
        """
        fp = pipeline._feature_pipeline
        fp.compute_turbulence.return_value = 5.0
        fp.turbulence_halt_baseline.return_value = 2.0

        risk_manager = MagicMock()
        risk_manager.check_turbulence.return_value = True
        pipeline._risk_manager = risk_manager

        result = pipeline._check_turbulence("equity", "2026-07-07")

        risk_manager.check_turbulence.assert_called_once_with("equity", 5.0, 2.0)
        assert result is True

    def test_baseline_cached_per_env_date(self, pipeline: ExecutionPipeline) -> None:
        """The obs-path baseline is cached per (env, date) — one delegate call/cycle."""
        fp = pipeline._feature_pipeline
        fp.turbulence_halt_baseline.return_value = 3.0

        first = pipeline._get_turbulence_90th_pct("equity", "2026-07-07")
        second = pipeline._get_turbulence_90th_pct("equity", "2026-07-07")

        assert first == second == 3.0
        fp.turbulence_halt_baseline.assert_called_once_with("equity", "2026-07-07")


class TestF1bTurbulenceZeroing:
    """Task 7 (F1b): the turbulence obs slot is frozen at 0.0 for inference, real value kept."""

    @staticmethod
    def _equity_obs_with_turb(
        pipeline: ExecutionPipeline, real_turb: float
    ) -> tuple[np.ndarray, int]:
        """Build a full-size equity observation carrying a nonzero turbulence."""
        from swingrl.features.assembler import equity_obs_dim, turbulence_obs_index

        n = len(pipeline.config.equity.symbols)
        sentiment = pipeline.config.sentiment.enabled
        turb_idx = turbulence_obs_index("equity", n, sentiment)
        obs = np.zeros(equity_obs_dim(sentiment, n), dtype=np.float64)
        obs[turb_idx] = real_turb
        return obs, turb_idx

    def _run_cycle_capturing_predict(
        self, pipeline: ExecutionPipeline, obs: np.ndarray
    ) -> MagicMock:
        """Run a dry-run equity cycle with mocked models/adapter; return the mock model."""
        pipeline._feature_pipeline.get_observation.return_value = obs

        mock_model = MagicMock()
        # equity action space = 8 assets + 1 cash dim
        mock_model.predict.return_value = (np.zeros(9), None)
        with patch.object(pipeline, "_load_models") as mock_load:
            mock_load.return_value = {
                "ppo": (mock_model, None),
                "a2c": (mock_model, None),
                "sac": (mock_model, None),
            }
            mock_adapter = MagicMock()
            mock_adapter.get_current_price.return_value = 450.0
            with patch.object(pipeline, "_get_adapter") as mock_get_adapter:
                mock_get_adapter.return_value = mock_adapter
                pipeline.execute_cycle("equity", dry_run=True)
        return mock_model

    def test_flag_on_zeros_slot_fed_to_predict(self, pipeline: ExecutionPipeline) -> None:
        """F1b (b): with the flag on, model.predict receives 0.0 at the turbulence slot."""
        real_turb = 7.25
        obs, turb_idx = self._equity_obs_with_turb(pipeline, real_turb)

        mock_model = self._run_cycle_capturing_predict(pipeline, obs)

        fed_obs = mock_model.predict.call_args.args[0]
        assert fed_obs[turb_idx] == 0.0
        # The observation object handed downstream was zeroed in place.
        assert obs[turb_idx] == 0.0

    def test_flag_on_captures_real_value_before_zeroing(self, pipeline: ExecutionPipeline) -> None:
        """F1b (c): the real sensor value is read out before the slot is zeroed."""
        real_turb = 3.5
        obs, _turb_idx = self._equity_obs_with_turb(pipeline, real_turb)

        self._run_cycle_capturing_predict(pipeline, obs)

        assert pipeline._turbulence_at_decision == pytest.approx(real_turb)

    def test_flag_off_keeps_real_value_in_obs(self, pipeline: ExecutionPipeline) -> None:
        """F1b: with the flag off, the turbulence slot flows through unchanged (era-1 path)."""
        pipeline.config.environment.zero_turbulence_obs = False
        real_turb = 4.75
        obs, turb_idx = self._equity_obs_with_turb(pipeline, real_turb)

        mock_model = self._run_cycle_capturing_predict(pipeline, obs)

        fed_obs = mock_model.predict.call_args.args[0]
        assert fed_obs[turb_idx] == pytest.approx(real_turb)
        # Real value is still captured regardless of the flag.
        assert pipeline._turbulence_at_decision == pytest.approx(real_turb)


class TestMarketGateAndSnapshotLifecycle:
    """Task B: market-calendar gate + every-cycle snapshot + fresh-value risk eval.

    Reviews C2/M11 + the 2026-07-16 amendment (snapshot every cycle; pre-trade risk
    evaluation consumes the freshly computed mark-to-market value).
    """

    @staticmethod
    def _clear(pipeline: ExecutionPipeline) -> None:
        """Reset the shared PostgreSQL test tables for deterministic isolation."""
        with pipeline._db.connection() as conn:
            for table in (
                "circuit_breaker_events",
                "positions",
                "trades",
                "portfolio_snapshots",
            ):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed table names

    @staticmethod
    def _seed_equity_state(
        pipeline: ExecutionPipeline,
        *,
        qty: float,
        buy_price: float,
        snapshot_value: float,
        hwm: float,
    ) -> None:
        """Seed one held SPY position, its buy trade, and a prior-day snapshot."""
        now = datetime.now(tz=UTC)
        yesterday = (now - timedelta(days=1)).isoformat()
        with pipeline._db.connection() as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, price, "
                "commission, slippage, environment, broker, order_type, trade_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    "seed-buy-spy",
                    yesterday,
                    "SPY",
                    "buy",
                    qty,
                    buy_price,
                    0.0,
                    0.0,
                    "equity",
                    "alpaca",
                    "market",
                    "signal",
                ),
            )
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, last_price, "
                "unrealized_pnl, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("SPY", "equity", qty, buy_price, buy_price, 0.0, yesterday),
            )
            conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp, environment, total_value, "
                "cash_balance, high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (yesterday, "equity", snapshot_value, buy_price, hwm, 0.0, 0.0),
            )

    def _patch_common(self, pipeline: ExecutionPipeline) -> None:
        """Bypass turbulence + feature-health gates for a clean order-loop run."""
        pipeline._health_tracker = MagicMock()
        pipeline._health_tracker.assess.return_value = MagicMock(should_block=False)

    def test_market_closed_skips_cycle(self, pipeline: ExecutionPipeline) -> None:
        """(c) A closed market clock skips the equity cycle before any order is placed."""
        self._clear(pipeline)

        mock_adapter = MagicMock()
        mock_adapter.get_clock.return_value = MagicMock(is_open=False)

        with (
            patch.object(pipeline, "_check_turbulence", return_value=False),
            patch.object(pipeline, "_get_adapter", return_value=mock_adapter),
        ):
            result = pipeline.execute_cycle("equity")

        assert result == []
        mock_adapter.get_clock.assert_called()
        mock_adapter.submit_order.assert_not_called()

    def test_no_fill_cycle_persists_snapshot_with_moved_prices(
        self, pipeline: ExecutionPipeline
    ) -> None:
        """(h) A no-fill (pending) cycle still writes a snapshot marked to moved prices."""
        self._clear(pipeline)
        self._seed_equity_state(pipeline, qty=1.0, buy_price=100.0, snapshot_value=400.0, hwm=400.0)
        self._patch_common(pipeline)

        pending_fill = FillResult(
            trade_id="pending-h",
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
        mock_adapter = MagicMock()
        mock_adapter.get_clock.return_value = MagicMock(is_open=True)
        mock_adapter.get_current_price.return_value = 120.0  # SPY moved up from 100
        mock_adapter.submit_order.return_value = pending_fill

        # SPY target 0.30 vs current 0.25 -> small buy that clears the risk gate.
        target = np.zeros(8, dtype=np.float32)
        target[0] = 0.30

        obs = np.zeros(156, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(9), None)

        with (
            patch.object(pipeline, "_check_turbulence", return_value=False),
            patch.object(pipeline, "_get_adapter", return_value=mock_adapter),
            patch.object(
                pipeline,
                "_load_models",
                return_value={
                    "ppo": (mock_model, None),
                    "a2c": (mock_model, None),
                    "sac": (mock_model, None),
                },
            ),
            patch.object(
                pipeline,
                "_get_ensemble_weights",
                return_value={"ppo": 0.4, "a2c": 0.3, "sac": 0.3},
            ),
            patch.object(pipeline, "_normalize_observation", return_value=_per_algo_obs_dict(obs)),
            patch("swingrl.execution.pipeline.process_actions", return_value=target),
        ):
            result = pipeline.execute_cycle("equity")

        assert result == []  # pending fill is not a successful fill
        mock_adapter.submit_order.assert_called_once()

        with pipeline._db.connection() as conn:
            snap = conn.execute(
                "SELECT total_value FROM portfolio_snapshots "
                "WHERE environment = 'equity' ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            trade = conn.execute("SELECT * FROM trades WHERE trade_id = 'pending-h'").fetchone()

        # Snapshot marks SPY to the moved price (120), not the stored last_price (100):
        # value = 1*120 + cash(400 - 100) = 420, NOT 400.
        assert snap is not None
        assert float(snap["total_value"]) == pytest.approx(420.0)
        # The pending order never became a trades row.
        assert trade is None

    def test_held_position_crash_trips_drawdown_breaker(self, pipeline: ExecutionPipeline) -> None:
        """(i) A held-position crash on a no-fill cycle trips the drawdown breaker.

        With the stale stored snapshot (500) the drawdown is 0 and nothing trips; the
        fresh mark-to-market value (310) shows a 38% drawdown, so the breaker fires at
        the cycle's risk evaluation.
        """
        self._clear(pipeline)
        self._seed_equity_state(pipeline, qty=1.0, buy_price=100.0, snapshot_value=500.0, hwm=500.0)
        self._patch_common(pipeline)

        mock_adapter = MagicMock()
        mock_adapter.get_clock.return_value = MagicMock(is_open=True)
        mock_adapter.get_current_price.return_value = 10.0  # SPY crashes from 100

        # SPY target 0.25 vs current 0.20 -> small ($25) buy that clears position-size.
        target = np.zeros(8, dtype=np.float32)
        target[0] = 0.25

        obs = np.zeros(156, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros(9), None)

        with (
            patch.object(pipeline, "_check_turbulence", return_value=False),
            patch.object(pipeline, "_get_adapter", return_value=mock_adapter),
            patch.object(
                pipeline,
                "_load_models",
                return_value={
                    "ppo": (mock_model, None),
                    "a2c": (mock_model, None),
                    "sac": (mock_model, None),
                },
            ),
            patch.object(
                pipeline,
                "_get_ensemble_weights",
                return_value={"ppo": 0.4, "a2c": 0.3, "sac": 0.3},
            ),
            patch.object(pipeline, "_normalize_observation", return_value=_per_algo_obs_dict(obs)),
            patch("swingrl.execution.pipeline.process_actions", return_value=target),
        ):
            result = pipeline.execute_cycle("equity")

        assert result == []
        mock_adapter.submit_order.assert_not_called()  # breaker trips before submission

        with pipeline._db.connection() as conn:
            event = conn.execute(
                "SELECT reason FROM circuit_breaker_events "
                "WHERE environment = 'equity' ORDER BY triggered_at DESC LIMIT 1"
            ).fetchone()

        assert event is not None
        assert "drawdown" in str(event["reason"])


class TestNormalizeObservation:
    """Test per-algo VecNormalize observation normalization."""

    def test_returns_per_algo_dict(self, pipeline: ExecutionPipeline) -> None:
        """Fix #10: _normalize_observation returns dict keyed by algo name."""
        obs = np.ones(156, dtype=np.float32)

        # Simulate loaded models with VecNormalize
        mock_vec_norm_ppo = MagicMock()
        mock_vec_norm_ppo.normalize_obs.return_value = obs * 0.5

        mock_vec_norm_a2c = MagicMock()
        mock_vec_norm_a2c.normalize_obs.return_value = obs * 0.8

        pipeline._models["equity"] = {
            "ppo": (MagicMock(), mock_vec_norm_ppo),
            "a2c": (MagicMock(), mock_vec_norm_a2c),
            "sac": (MagicMock(), None),  # no VecNormalize
        }

        result = pipeline._normalize_observation("equity", obs)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"ppo", "a2c", "sac"}
        np.testing.assert_array_almost_equal(result["ppo"], obs * 0.5)
        np.testing.assert_array_almost_equal(result["a2c"], obs * 0.8)
        np.testing.assert_array_equal(result["sac"], obs)  # raw obs fallback

    def test_empty_dict_when_no_models(self, pipeline: ExecutionPipeline) -> None:
        """Fix #10: Returns empty dict when no models loaded for env."""
        obs = np.ones(156, dtype=np.float32)
        result = pipeline._normalize_observation("equity", obs)
        assert result == {}


class TestPipelineInit:
    """Test pipeline initialization and lazy loading."""

    def test_pipeline_creates_successfully(self, pipeline: ExecutionPipeline) -> None:
        """Pipeline initializes without errors."""
        assert pipeline is not None

    def test_models_not_loaded_on_init(self, pipeline: ExecutionPipeline) -> None:
        """Models are lazy-loaded, not loaded on construction."""
        assert pipeline._models == {}

    def test_load_models_path_no_double_active(
        self,
        exec_config: Any,
        mock_db: Any,
        mock_feature_pipeline: MagicMock,
        mock_alerter: MagicMock,
        tmp_path: Path,
    ) -> None:
        """PAPER-02: _load_models constructs path as models_dir/active/{env}/{algo}/model.zip.

        When pipeline receives a bare models_dir (not models_dir/active), it must
        internally append 'active/{env}/{algo}/model.zip' -- no double 'active' nesting.
        """
        bare_models_dir = tmp_path / "models"
        bare_models_dir.mkdir(parents=True, exist_ok=True)

        pipe = ExecutionPipeline(
            config=exec_config,
            db=mock_db,
            feature_pipeline=mock_feature_pipeline,
            alerter=mock_alerter,
            models_dir=bare_models_dir,
        )

        # Create a model file at the correct (non-double-nested) path
        expected_path = bare_models_dir / "active" / "equity" / "ppo" / "model.zip"
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(b"fake-model-zip")

        # Verify the double-nested path does NOT exist (proves no double nesting)
        double_nested = bare_models_dir / "active" / "active" / "equity" / "ppo" / "model.zip"
        assert not double_nested.exists()
        assert expected_path.exists()

        # _load_models should find model at bare_models_dir/active/equity/ppo/model.zip
        # PPO/A2C/SAC are imported locally inside _load_models from stable_baselines3
        with patch("stable_baselines3.PPO") as mock_ppo_cls:
            with patch("stable_baselines3.A2C"):
                with patch("stable_baselines3.SAC"):
                    mock_ppo_cls.load.return_value = MagicMock()
                    # Only PPO path exists; A2C and SAC model.zip files don't exist
                    models = pipe._load_models("equity")

        # PPO model was found and loaded from the non-double-nested path
        assert "ppo" in models
        mock_ppo_cls.load.assert_called_once_with(str(expected_path))
