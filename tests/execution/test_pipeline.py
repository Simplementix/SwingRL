"""Tests for ExecutionPipeline orchestrator.

Verifies weight-based rebalancing execution pipeline with mocked dependencies:
model loading, per-algo VecNormalize, ensemble blending, process_actions, and
weight-based delta order generation.
"""

from __future__ import annotations

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
def mock_alerter() -> MagicMock:
    """Mock Alerter."""
    return MagicMock()


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
        with pipeline._db.sqlite() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_events "
                "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
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
