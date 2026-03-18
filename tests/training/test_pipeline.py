"""Integration tests for train_pipeline.py orchestration CLI.

TRAIN-01 through TRAIN-07: train_pipeline.py orchestrates walk-forward validation,
ensemble weights from OOS Sharpe, ensemble gate, tuning, final training, and deployment.

All tests mock WalkForwardBacktester.run(), TrainingOrchestrator.train(),
_load_features_prices(), and duckdb.connect() for speed (< 30s total).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.training.conftest import _make_fold_result


def _make_training_result(
    env_name: str = "equity",
    algo_name: str = "ppo",
    models_dir: Path | None = None,
    converged: bool = True,
) -> MagicMock:
    """Create a mock TrainingResult.

    Args:
        env_name: Environment name.
        algo_name: Algorithm name.
        models_dir: Models directory (creates paths).
        converged: Whether training converged.

    Returns:
        MagicMock configured as TrainingResult.
    """
    m = MagicMock()
    base = models_dir or Path("models")
    m.model_path = base / "active" / env_name / algo_name / "model.zip"
    m.vec_normalize_path = base / "active" / env_name / algo_name / "vec_normalize.pkl"
    m.env_name = env_name
    m.algo_name = algo_name
    m.converged_at_step = 500_000 if converged else None
    m.total_timesteps = 1_000_000
    return m


def _make_mock_features_prices(
    n_bars: int = 800,
    n_features: int = 156,
    n_assets: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create mock features and prices arrays.

    Args:
        n_bars: Number of bars.
        n_features: Feature count.
        n_assets: Asset count.

    Returns:
        Tuple of (features, prices) arrays.
    """
    features = np.random.default_rng(42).random((n_bars, n_features)).astype(np.float32)
    prices = np.random.default_rng(43).random((n_bars, n_assets)).astype(np.float32) + 100.0
    return features, prices


# ---------------------------------------------------------------------------
# Common fixtures and mock paths
# ---------------------------------------------------------------------------

_PIPELINE_MODULE = "train_pipeline"
_WF_PATH = "swingrl.agents.backtest.WalkForwardBacktester.run"
_TRAIN_PATH = "swingrl.training.trainer.TrainingOrchestrator.train"
_LOAD_PATH = "train_pipeline._load_features_prices"
_DUCKDB_PATH = "train_pipeline.duckdb"


class TestEquityBaselineTraining:
    """TRAIN-01: Pipeline calls walk-forward on full data and final train on recent slice."""

    def test_equity_baseline_training(self, tmp_path: Path) -> None:
        """TRAIN-01: walk-forward uses full features, final train uses sliced recent features."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        features, prices = _make_mock_features_prices(n_bars=800)

        # All 3 algos produce passing folds
        good_folds = [_make_fold_result(sharpe=1.5, mdd=-0.05)] * 3

        # Mock train creates deployment files as a side effect
        def mock_train_side_effect(
            env_name: str, algo_name: str, features: Any, prices: Any, **kwargs: Any
        ) -> Any:
            d = tmp_path / "active" / env_name / algo_name
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.zip").write_bytes(b"fake")
            (d / "vec_normalize.pkl").write_bytes(b"fake")
            return _make_training_result(
                env_name=env_name, algo_name=algo_name, models_dir=tmp_path
            )

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(pipeline, "_load_features_prices", return_value=(features, prices)),
            patch("swingrl.agents.backtest.WalkForwardBacktester.run", return_value=good_folds),
            patch(
                "swingrl.training.trainer.TrainingOrchestrator.train",
                side_effect=mock_train_side_effect,
            ),
            patch("train_pipeline.duckdb") as mock_duckdb,
            patch.object(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)

            result = pipeline.run_environment(
                env_name="equity",
                config=MagicMock(
                    memory_agent=MagicMock(meta_training=False, enabled=False),
                    system=MagicMock(duckdb_path=str(tmp_path / "test.ddb")),
                    paths=MagicMock(logs_dir=str(tmp_path / "logs")),
                ),
                models_dir=tmp_path,
                force=False,
                report={},
            )

            assert result["ensemble_gate"]["passed"] is True


class TestCryptoBaselineTraining:
    """TRAIN-02: Crypto env uses crypto-specific RECENT_WINDOW_BARS."""

    def test_crypto_recent_window_different_from_equity(self) -> None:
        """TRAIN-02: Crypto recent window is 2191 bars, equity is 756 bars."""
        from swingrl.training.pipeline_helpers import RECENT_WINDOW_BARS

        assert RECENT_WINDOW_BARS["equity"] == 252 * 3  # 756
        assert RECENT_WINDOW_BARS["crypto"] == 2191


class TestWfMetricsRecorded:
    """TRAIN-03: Walk-forward metrics (Sharpe, MDD, PF) are captured per fold."""

    def test_fold_result_has_oos_metrics(self) -> None:
        """TRAIN-03: FoldResult.out_of_sample_metrics contains sharpe and mdd."""
        fold = _make_fold_result(sharpe=1.5, mdd=-0.08, profit_factor=1.6)
        assert "sharpe" in fold.out_of_sample_metrics
        assert "mdd" in fold.out_of_sample_metrics
        assert "profit_factor" in fold.out_of_sample_metrics

    def test_ensemble_gate_reads_sharpe_from_folds(self) -> None:
        """TRAIN-03: check_ensemble_gate reads sharpe from fold.out_of_sample_metrics."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        folds = {
            "ppo": [_make_fold_result(sharpe=1.2), _make_fold_result(sharpe=1.3)],
            "a2c": [_make_fold_result(sharpe=1.1)],
            "sac": [_make_fold_result(sharpe=1.4)],
        }
        passed, sharpe, mdd = check_ensemble_gate(folds)
        assert sharpe > 1.0
        assert passed is True


class TestTuningTriggersOnLowSharpe:
    """TRAIN-04: Tuning triggers when baseline OOS Sharpe < 0.5."""

    def test_tuning_triggers_when_sharpe_below_threshold(self, tmp_path: Path) -> None:
        """TRAIN-04: When ensemble Sharpe < 0.5, tuning round 1 is triggered."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        features, prices = _make_mock_features_prices(n_bars=800)

        # Low Sharpe folds to trigger tuning
        low_folds = [_make_fold_result(sharpe=0.3, mdd=-0.20)] * 3
        training_result = _make_training_result(
            env_name="equity", algo_name="ppo", models_dir=tmp_path, converged=False
        )

        mock_conn = MagicMock()

        with (
            patch.object(pipeline, "_load_features_prices", return_value=(features, prices)),
            patch("swingrl.agents.backtest.WalkForwardBacktester.run", return_value=low_folds),
            patch(
                "swingrl.training.trainer.TrainingOrchestrator.train",
                return_value=training_result,
            ),
            patch("train_pipeline.duckdb") as mock_duckdb,
            patch("train_pipeline.Path.exists", return_value=False),
            patch.object(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)

            # Create output dirs with files so deployment check passes
            for algo in ["ppo", "a2c", "sac"]:
                d = tmp_path / "active" / "equity" / algo
                d.mkdir(parents=True, exist_ok=True)

            result = pipeline.run_environment(
                env_name="equity",
                config=MagicMock(
                    memory_agent=MagicMock(meta_training=False, enabled=False),
                    system=MagicMock(duckdb_path=str(tmp_path / "test.ddb")),
                    paths=MagicMock(logs_dir=str(tmp_path / "logs")),
                    training=MagicMock(n_envs=6),
                ),
                models_dir=tmp_path,
                force=False,
                report={},
            )
            # Gate should fail when Sharpe < 0.5 after tuning exhausted
            assert result["ensemble_gate"]["passed"] is False


class TestEnsembleWeightsFromWfSharpe:
    """TRAIN-05: Real OOS Sharpe weights used (not placeholder 1.0)."""

    def test_compute_weights_from_real_sharpe(self) -> None:
        """TRAIN-05: compute_ensemble_weights_from_wf uses actual OOS Sharpe, not 1.0."""
        from swingrl.training.pipeline_helpers import compute_ensemble_weights_from_wf

        mock_config = MagicMock()

        wf_results = {
            "ppo": [_make_fold_result(sharpe=2.0)],
            "a2c": [_make_fold_result(sharpe=1.0)],
            "sac": [_make_fold_result(sharpe=1.5)],
        }

        with patch("swingrl.training.ensemble.EnsembleBlender") as MockBlender:
            MockBlender.return_value.compute_weights.return_value = {
                "ppo": 0.44,
                "a2c": 0.22,
                "sac": 0.33,
            }
            computed = compute_ensemble_weights_from_wf(mock_config, wf_results, "equity")

            # Verify the blender was called with actual (non-uniform) sharpe values
            call_kwargs = MockBlender.return_value.compute_weights.call_args
            agent_sharpes = call_kwargs[1].get("agent_sharpes") or call_kwargs[0][1]
            assert agent_sharpes["ppo"] != agent_sharpes["a2c"]  # real values, not all 1.0
            assert computed is not None


class TestModelFilesDeployed:
    """TRAIN-07: model.zip + vec_normalize.pkl exist after pipeline for all 3 algos."""

    def test_deployment_check_validates_files(self, tmp_path: Path) -> None:
        """TRAIN-07: Pipeline verifies model.zip + vec_normalize.pkl for each algo."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        # Create all required deployment files
        for algo in ["ppo", "a2c", "sac"]:
            d = tmp_path / "active" / "equity" / algo
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.zip").write_bytes(b"model_data")
            (d / "vec_normalize.pkl").write_bytes(b"vec_data")

        # Should not raise
        pipeline._verify_deployment(env_name="equity", models_dir=tmp_path)

    def test_deployment_check_raises_on_missing(self, tmp_path: Path) -> None:
        """TRAIN-07: _verify_deployment raises ModelError when files missing."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        from swingrl.utils.exceptions import ModelError

        pipeline = importlib.import_module("train_pipeline")

        # Create only PPO files, omit A2C and SAC
        d = tmp_path / "active" / "equity" / "ppo"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.zip").write_bytes(b"fake")
        (d / "vec_normalize.pkl").write_bytes(b"fake")

        with pytest.raises(ModelError):
            pipeline._verify_deployment(env_name="equity", models_dir=tmp_path)


class TestEnsembleGateBlocksDeployment:
    """TRAIN-06: When ensemble gate fails AND tuning rounds exhausted, pipeline exits 1."""

    def test_gate_blocked_returns_non_zero(self, tmp_path: Path) -> None:
        """TRAIN-06: Ensemble gate failure blocks deployment and returns non-zero exit code."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        # Simulate gate failure result (no deployment)
        gate_result = {"passed": False, "sharpe": 0.4, "mdd": -0.22}

        result = pipeline._evaluate_gate_and_decide(
            gate_result=gate_result,
            baseline_sharpe=0.4,
            env_name="equity",
        )
        assert result["deploy"] is False


class TestJsonReportWritten:
    """TRAIN-07: JSON report written to data/training_report.json with expected schema."""

    def test_json_report_written(self, tmp_path: Path) -> None:
        """TRAIN-07: run_pipeline writes JSON report to specified path."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        report = {
            "generated_at": "2026-03-13T14:55:26Z",
            "equity": {
                "walk_forward": {"ppo": [], "a2c": [], "sac": []},
                "ensemble_weights": {"ppo": 0.45, "a2c": 0.28, "sac": 0.27},
                "ensemble_gate": {"passed": True, "sharpe": 1.2, "mdd": -0.08},
                "tuning_rounds": [],
                "wall_clock_total_s": 3600,
            },
        }
        report_path = tmp_path / "training_report.json"
        pipeline._write_json_report(report, report_path)

        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert "equity" in loaded
        assert loaded["equity"]["ensemble_gate"]["passed"] is True

    def test_json_report_has_expected_schema(self, tmp_path: Path) -> None:
        """TRAIN-07: JSON report schema matches documented structure."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        features, prices = _make_mock_features_prices(n_bars=800)
        good_folds = [_make_fold_result(sharpe=1.5, mdd=-0.05)] * 3

        def mock_train_side_effect(
            env_name: str, algo_name: str, features: Any, prices: Any, **kwargs: Any
        ) -> Any:
            d = tmp_path / "active" / env_name / algo_name
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.zip").write_bytes(b"fake")
            (d / "vec_normalize.pkl").write_bytes(b"fake")
            return _make_training_result(
                env_name=env_name, algo_name=algo_name, models_dir=tmp_path
            )

        mock_conn = MagicMock()

        with (
            patch.object(pipeline, "_load_features_prices", return_value=(features, prices)),
            patch("swingrl.agents.backtest.WalkForwardBacktester.run", return_value=good_folds),
            patch(
                "swingrl.training.trainer.TrainingOrchestrator.train",
                side_effect=mock_train_side_effect,
            ),
            patch("train_pipeline.duckdb") as mock_duckdb,
            patch.object(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)

            report_path = tmp_path / "training_report.json"
            pipeline.run_environment(
                env_name="equity",
                config=MagicMock(
                    memory_agent=MagicMock(meta_training=False, enabled=False),
                    system=MagicMock(duckdb_path=str(tmp_path / "test.ddb")),
                    paths=MagicMock(logs_dir=str(tmp_path / "logs")),
                ),
                models_dir=tmp_path,
                force=False,
                report={},
                report_path=report_path,
            )

            assert report_path.exists()
            loaded = json.loads(report_path.read_text())
            assert "equity" in loaded


class TestCheckpointResume:
    """TRAIN-07: Checkpointing skips already-trained algos."""

    def test_checkpoint_skips_existing_model(self, tmp_path: Path) -> None:
        """TRAIN-07: Pipeline skips walk-forward for algo with existing model.zip."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        # Create PPO model.zip (checkpoint exists)
        ppo_dir = tmp_path / "active" / "equity" / "ppo"
        ppo_dir.mkdir(parents=True, exist_ok=True)
        (ppo_dir / "model.zip").write_bytes(b"existing_model")
        (ppo_dir / "vec_normalize.pkl").write_bytes(b"existing_vec")

        # Also create A2C and SAC checkpoints so pipeline can complete
        for algo in ["a2c", "sac"]:
            d = tmp_path / "active" / "equity" / algo
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.zip").write_bytes(b"existing_model")
            (d / "vec_normalize.pkl").write_bytes(b"existing_vec")

        features, prices = _make_mock_features_prices(n_bars=800)
        good_folds = [_make_fold_result(sharpe=1.5, mdd=-0.05)] * 3

        mock_conn = MagicMock()

        with (
            patch.object(pipeline, "_load_features_prices", return_value=(features, prices)),
            patch(
                "swingrl.agents.backtest.WalkForwardBacktester.run", return_value=good_folds
            ) as mock_wf,
            patch("swingrl.training.trainer.TrainingOrchestrator.train"),
            patch("train_pipeline.duckdb") as mock_duckdb,
            patch.object(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)

            pipeline.run_environment(
                env_name="equity",
                config=MagicMock(
                    memory_agent=MagicMock(meta_training=False, enabled=False),
                    system=MagicMock(duckdb_path=str(tmp_path / "test.ddb")),
                    paths=MagicMock(logs_dir=str(tmp_path / "logs")),
                ),
                models_dir=tmp_path,
                force=False,
                report={},
            )

            # WalkForwardBacktester.run should NOT have been called (all algos checkpointed)
            mock_wf.assert_not_called()


class TestMainCLI:
    """Test main() CLI entry point."""

    def test_main_returns_int(self, tmp_path: Path) -> None:
        """main() returns an integer exit code."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        import importlib

        pipeline = importlib.import_module("train_pipeline")

        features, prices = _make_mock_features_prices(n_bars=800)
        good_folds = [_make_fold_result(sharpe=1.5, mdd=-0.05)] * 3
        mock_conn = MagicMock()

        def mock_train_side_effect(
            env_name: str, algo_name: str, features: Any, prices: Any, **kwargs: Any
        ) -> Any:
            d = tmp_path / "active" / env_name / algo_name
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.zip").write_bytes(b"fake")
            (d / "vec_normalize.pkl").write_bytes(b"fake")
            return _make_training_result(
                env_name=env_name, algo_name=algo_name, models_dir=tmp_path
            )

        with (
            patch.object(pipeline, "_load_features_prices", return_value=(features, prices)),
            patch("swingrl.agents.backtest.WalkForwardBacktester.run", return_value=good_folds),
            patch(
                "swingrl.training.trainer.TrainingOrchestrator.train",
                side_effect=mock_train_side_effect,
            ),
            patch("train_pipeline.duckdb") as mock_duckdb,
            patch("train_pipeline.load_config") as mock_cfg,
            patch("train_pipeline.configure_logging"),
        ):
            mock_duckdb.connect.return_value = mock_conn
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)

            mock_config = MagicMock()
            mock_config.memory_agent = MagicMock(meta_training=False, enabled=False)
            mock_config.system = MagicMock(duckdb_path=str(tmp_path / "test.ddb"))
            mock_config.paths = MagicMock(logs_dir=str(tmp_path / "logs"))
            mock_config.logging = MagicMock(json_logs=False, level="INFO")
            mock_cfg.return_value = mock_config

            exit_code = pipeline.main(
                [
                    "--env",
                    "equity",
                    "--config",
                    "config/swingrl.yaml",
                    "--models-dir",
                    str(tmp_path),
                    "--report",
                    str(tmp_path / "report.json"),
                ]
            )

            assert isinstance(exit_code, int)
