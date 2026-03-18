"""Tests for multi-iteration training loop in train_pipeline.py.

TRAIN-06: Models deploy to models/active/{env}/{algo}/ with VecNormalize files.
TRAIN-07: Training success gate: ensemble Sharpe > 1.0 and MDD < 15%.

Tests cover: state persistence, model selection, deployment, comparison report,
resume, CLI parsing, and iteration memory configuration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"

_pipeline_module: Any = None


def _import_pipeline() -> Any:
    """Import train_pipeline module from scripts/ without polluting sys.path.

    Uses importlib.util.spec_from_file_location to avoid adding scripts/ to
    sys.path, which would cause import collisions with services/memory/main.py.

    Returns:
        The train_pipeline module object.
    """
    global _pipeline_module  # noqa: PLW0603
    if _pipeline_module is not None:
        return _pipeline_module

    module_path = _SCRIPTS_DIR / "train_pipeline.py"
    spec = importlib.util.spec_from_file_location("train_pipeline", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_pipeline"] = module
    # Temporarily add scripts/ so train_pipeline's own imports resolve,
    # then remove it to avoid contaminating subsequent test imports.
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPTS_DIR))
    _pipeline_module = module
    return module


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_env_result(
    env_name: str = "equity",
    sharpe: float = 1.2,
    mdd: float = -0.08,
    sortino: float = 1.5,
    calmar: float = 0.6,
) -> dict[str, Any]:
    """Build a synthetic run_environment() result dict."""
    return {
        "ensemble_gate": {"passed": True, "sharpe": sharpe, "mdd": mdd},
        "ensemble_weights": {"ppo": 0.40, "a2c": 0.30, "sac": 0.30},
        "walk_forward": {
            "ppo": [{"fold": 1, "oos_sharpe": sharpe, "sortino": sortino, "calmar": calmar}],
            "a2c": [{"fold": 1, "oos_sharpe": sharpe, "sortino": sortino, "calmar": calmar}],
            "sac": [{"fold": 1, "oos_sharpe": sharpe, "sortino": sortino, "calmar": calmar}],
        },
        "final_training": {
            "ppo": {"sortino": sortino, "calmar": calmar, "timesteps": 1_000_000},
            "a2c": {"sortino": sortino, "calmar": calmar, "timesteps": 1_000_000},
            "sac": {"sortino": sortino, "calmar": calmar, "timesteps": 500_000},
        },
        "tuning_rounds": [],
        "wall_clock_total_s": 120.0,
    }


def _build_iter_state_with_results(
    iteration_count: int = 3,
    sortino_per_iter: list[float] | None = None,
    calmar_per_iter: list[float] | None = None,
) -> dict[str, Any]:
    """Build a training_state dict with N completed iterations.

    Args:
        iteration_count: Number of completed iterations (0..N-1).
        sortino_per_iter: Sortino per iteration (defaults to 1.0, 1.5, 1.2).
        calmar_per_iter: Calmar per iteration (defaults to 0.5, 0.6, 0.4).

    Returns:
        Training state dict.
    """
    sortinos = sortino_per_iter or [1.0, 1.5, 1.2][:iteration_count]
    calmars = calmar_per_iter or [0.5, 0.6, 0.4][:iteration_count]

    state: dict[str, Any] = {
        "completed_iterations": list(range(iteration_count)),
        "current_iteration": iteration_count - 1 if iteration_count > 0 else 0,
        "iteration_results": {},
    }

    for i in range(iteration_count):
        s = sortinos[i] if i < len(sortinos) else 1.0
        c = calmars[i] if i < len(calmars) else 0.5
        state[f"iteration_{i}_result"] = {
            "equity": _make_env_result("equity", sortino=s, calmar=c),
            "crypto": _make_env_result("crypto", sortino=s, calmar=c),
        }

    return state


# ===========================================================================
# TestTrainingState
# ===========================================================================


class TestTrainingState:
    """TRAIN-06: Training state persisted atomically across iterations."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """TRAIN-06: save_training_state then load returns identical dict."""
        pipeline = _import_pipeline()
        state_path = tmp_path / "training_state.json"
        original = {
            "completed_iterations": [0, 1],
            "current_iteration": 2,
            "iteration_0_result": {"equity": {"ensemble_gate": {"passed": True}}},
        }
        pipeline.save_training_state(original, state_path)
        loaded = pipeline.load_training_state(state_path)
        assert loaded["completed_iterations"] == [0, 1]
        assert loaded["current_iteration"] == 2
        assert "iteration_0_result" in loaded

    def test_save_atomic_creates_no_tmp(self, tmp_path: Path) -> None:
        """TRAIN-06: After save, no .tmp file remains alongside the state file."""
        pipeline = _import_pipeline()
        state_path = tmp_path / "training_state.json"
        pipeline.save_training_state({"completed_iterations": []}, state_path)
        tmp_file = state_path.with_suffix(".tmp")
        assert not tmp_file.exists(), ".tmp file must be cleaned up after atomic rename"

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """TRAIN-06: load_training_state returns default dict when file does not exist."""
        pipeline = _import_pipeline()
        state_path = tmp_path / "nonexistent_state.json"
        result = pipeline.load_training_state(state_path)
        assert result["completed_iterations"] == []
        assert result["current_iteration"] == 0

    def test_save_is_atomic_write_tmp_then_replace(self, tmp_path: Path, monkeypatch: Any) -> None:
        """TRAIN-06: save_training_state uses os.replace (POSIX atomic rename)."""
        pipeline = _import_pipeline()
        calls: list[tuple[str, str]] = []

        real_replace = os.replace

        def spy_replace(src: str, dst: str) -> None:
            calls.append((str(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr("os.replace", spy_replace)
        state_path = tmp_path / "training_state.json"
        pipeline.save_training_state({"completed_iterations": [0]}, state_path)

        assert len(calls) == 1, "os.replace must be called exactly once"
        src_path, dst_path = calls[0]
        assert src_path.endswith(".tmp"), "source of os.replace must be a .tmp file"
        assert dst_path == str(state_path)


# ===========================================================================
# TestSelectBestModels
# ===========================================================================


class TestSelectBestModels:
    """TRAIN-06: Best iteration selected by Sortino (Calmar as tiebreak)."""

    def test_selects_highest_sortino(self, tmp_path: Path) -> None:
        """TRAIN-06: select_best_per_algo_env picks iteration with highest Sortino."""
        pipeline = _import_pipeline()
        # iter 0: sortino=1.0, iter 1: sortino=1.5 (best), iter 2: sortino=1.2
        state = _build_iter_state_with_results(
            iteration_count=3,
            sortino_per_iter=[1.0, 1.5, 1.2],
            calmar_per_iter=[0.5, 0.6, 0.4],
        )
        result = pipeline.select_best_per_algo_env(state)
        # iteration 1 has highest Sortino — all algos/envs should pick iter 1
        assert result["equity"]["ppo"] == 1
        assert result["equity"]["a2c"] == 1
        assert result["equity"]["sac"] == 1
        assert result["crypto"]["ppo"] == 1

    def test_calmar_tiebreak(self, tmp_path: Path) -> None:
        """TRAIN-06: Calmar tiebreaks when two iterations have equal Sortino."""
        pipeline = _import_pipeline()
        # iter 0: sortino=1.5 calmar=0.4, iter 1: sortino=1.5 calmar=0.8 (better Calmar)
        state = _build_iter_state_with_results(
            iteration_count=2,
            sortino_per_iter=[1.5, 1.5],
            calmar_per_iter=[0.4, 0.8],
        )
        result = pipeline.select_best_per_algo_env(state)
        assert result["equity"]["ppo"] == 1
        assert result["crypto"]["ppo"] == 1

    def test_handles_missing_metrics(self, tmp_path: Path) -> None:
        """TRAIN-06: select_best_per_algo_env handles iterations missing sortino gracefully."""
        pipeline = _import_pipeline()
        # Build state with missing final_training metrics
        state: dict[str, Any] = {
            "completed_iterations": [0],
            "current_iteration": 0,
            "iteration_0_result": {
                "equity": {
                    "ensemble_gate": {"passed": True, "sharpe": 1.2},
                    "final_training": {},  # no per-algo metrics
                },
                "crypto": {
                    "ensemble_gate": {"passed": True, "sharpe": 1.1},
                    "final_training": {},
                },
            },
        }
        # Should not raise
        result = pipeline.select_best_per_algo_env(state)
        # Only 1 iteration, all should pick 0
        for env in result:
            for algo in result[env]:
                assert result[env][algo] == 0

    def test_single_iteration_baseline(self, tmp_path: Path) -> None:
        """TRAIN-06: Single iteration (baseline only) returns iteration 0 as best."""
        pipeline = _import_pipeline()
        state = _build_iter_state_with_results(iteration_count=1, sortino_per_iter=[1.3])
        result = pipeline.select_best_per_algo_env(state)
        assert result["equity"]["ppo"] == 0


# ===========================================================================
# TestDeployBestModels
# ===========================================================================


class TestDeployBestModels:
    """TRAIN-06: Best models copied from iterations/ to active/."""

    def _create_iter_models(
        self,
        tmp_path: Path,
        iter_idx: int,
        envs: list[str] | None = None,
        algos: list[str] | None = None,
    ) -> None:
        """Create fake model files for a given iteration."""
        envs = envs or ["equity", "crypto"]
        algos = algos or ["ppo", "a2c", "sac"]
        for env in envs:
            for algo in algos:
                d = tmp_path / "iterations" / f"iter_{iter_idx}" / env / algo
                d.mkdir(parents=True, exist_ok=True)
                (d / "model.zip").write_bytes(f"model_iter{iter_idx}_{env}_{algo}".encode())
                (d / "vec_normalize.pkl").write_bytes(f"vec_iter{iter_idx}_{env}_{algo}".encode())

    def test_copies_model_and_vec_normalize(self, tmp_path: Path) -> None:
        """TRAIN-06: deploy_best_models copies model.zip + vec_normalize.pkl to active/."""
        pipeline = _import_pipeline()
        self._create_iter_models(tmp_path, 0)
        self._create_iter_models(tmp_path, 1)

        winners = {
            "equity": {"ppo": 1, "a2c": 0, "sac": 1},
            "crypto": {"ppo": 0, "a2c": 1, "sac": 0},
        }
        pipeline.deploy_best_models(winners, tmp_path)

        # ppo equity should come from iter 1
        ppo_equity = tmp_path / "active" / "equity" / "ppo" / "model.zip"
        assert ppo_equity.exists()
        assert b"model_iter1_equity_ppo" in ppo_equity.read_bytes()

        # a2c equity should come from iter 0
        a2c_equity = tmp_path / "active" / "equity" / "a2c" / "model.zip"
        assert a2c_equity.exists()
        assert b"model_iter0_equity_a2c" in a2c_equity.read_bytes()

    def test_copies_vec_normalize(self, tmp_path: Path) -> None:
        """TRAIN-06: deploy_best_models also copies vec_normalize.pkl."""
        pipeline = _import_pipeline()
        self._create_iter_models(tmp_path, 0)
        winners = {
            "equity": {"ppo": 0, "a2c": 0, "sac": 0},
            "crypto": {"ppo": 0, "a2c": 0, "sac": 0},
        }
        pipeline.deploy_best_models(winners, tmp_path)

        vec_path = tmp_path / "active" / "equity" / "ppo" / "vec_normalize.pkl"
        assert vec_path.exists()

    def test_creates_active_directories(self, tmp_path: Path) -> None:
        """TRAIN-06: deploy_best_models creates active/ dirs if they don't exist."""
        pipeline = _import_pipeline()
        self._create_iter_models(tmp_path, 0)
        winners = {
            "equity": {"ppo": 0, "a2c": 0, "sac": 0},
            "crypto": {"ppo": 0, "a2c": 0, "sac": 0},
        }
        pipeline.deploy_best_models(winners, tmp_path)
        for env in ["equity", "crypto"]:
            for algo in ["ppo", "a2c", "sac"]:
                assert (tmp_path / "active" / env / algo).is_dir()


# ===========================================================================
# TestComparisonReport
# ===========================================================================


class TestComparisonReport:
    """TRAIN-07: training_comparison.json generated with full per-iteration metrics."""

    def test_report_contains_all_iterations(self, tmp_path: Path) -> None:
        """TRAIN-07: 3 iterations -> JSON has iteration_0, iteration_1, iteration_2."""
        pipeline = _import_pipeline()
        state = _build_iter_state_with_results(3)
        report_path = tmp_path / "training_comparison.json"
        winners = {
            "equity": {"ppo": 1, "a2c": 1, "sac": 1},
            "crypto": {"ppo": 1, "a2c": 1, "sac": 1},
        }
        pipeline.write_comparison_report(state, report_path, winners)

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "iteration_0" in data
        assert "iteration_1" in data
        assert "iteration_2" in data

    def test_report_contains_winner_per_algo_env(self, tmp_path: Path) -> None:
        """TRAIN-07: JSON has 'winners' section with algo x env breakdown."""
        pipeline = _import_pipeline()
        state = _build_iter_state_with_results(2)
        report_path = tmp_path / "training_comparison.json"
        winners = {
            "equity": {"ppo": 1, "a2c": 0, "sac": 1},
            "crypto": {"ppo": 0, "a2c": 1, "sac": 0},
        }
        pipeline.write_comparison_report(state, report_path, winners)

        data = json.loads(report_path.read_text())
        assert "winners" in data
        assert data["winners"]["equity"]["ppo"] == 1
        assert data["winners"]["equity"]["a2c"] == 0

    def test_report_is_valid_json(self, tmp_path: Path) -> None:
        """TRAIN-07: write_comparison_report output is valid JSON."""
        pipeline = _import_pipeline()
        state = _build_iter_state_with_results(2)
        report_path = tmp_path / "training_comparison.json"
        winners: dict[str, dict[str, int]] = {}
        pipeline.write_comparison_report(state, report_path, winners)
        # json.loads raises if invalid
        json.loads(report_path.read_text())

    def test_report_records_memory_enabled_flag(self, tmp_path: Path) -> None:
        """TRAIN-07: Iteration 0 is marked memory_enabled=False in report."""
        pipeline = _import_pipeline()
        state = _build_iter_state_with_results(2)
        report_path = tmp_path / "training_comparison.json"
        winners: dict[str, dict[str, int]] = {}
        pipeline.write_comparison_report(state, report_path, winners)
        data = json.loads(report_path.read_text())
        assert data["iteration_0"]["memory_enabled"] is False
        assert data["iteration_1"]["memory_enabled"] is True


# ===========================================================================
# TestIterationLoop
# ===========================================================================


class TestIterationLoop:
    """TRAIN-06/07: Multi-iteration training loop with resume and memory configuration."""

    def _mock_config(self, tmp_path: Path) -> MagicMock:
        """Build a minimal mock SwingRLConfig."""
        cfg = MagicMock()
        cfg.system = MagicMock(duckdb_path=str(tmp_path / "test.ddb"))
        cfg.paths = MagicMock(logs_dir=str(tmp_path / "logs"))
        cfg.logging = MagicMock(json_logs=False, level="INFO")
        cfg.memory_agent = MagicMock(
            enabled=False,
            meta_training=False,
            base_url="http://localhost:8889",
            timeout_sec=30.0,
            api_key="",
        )
        cfg.model_copy = MagicMock(return_value=cfg)
        return cfg

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """TRAIN-06: run_all_iterations skips completed iterations from state."""
        pipeline = _import_pipeline()

        # Pre-populate state with iteration 0 completed
        state_path = tmp_path / "training_state.json"
        initial_state = {
            "completed_iterations": [0],
            "current_iteration": 1,
            "iteration_0_result": {
                "equity": _make_env_result("equity"),
                "crypto": _make_env_result("crypto"),
            },
        }
        pipeline.save_training_state(initial_state, state_path)

        cfg = self._mock_config(tmp_path)
        run_calls: list[int] = []

        def mock_run_environment(
            env_name: str, config: Any, models_dir: Path, **kwargs: Any
        ) -> Any:
            # Detect which iteration by models_dir path
            parts = models_dir.parts
            for part in parts:
                if part.startswith("iter_"):
                    run_calls.append(int(part.split("_")[1]))
            return _make_env_result(env_name)

        comparison_path = tmp_path / "training_comparison.json"

        with (
            patch.object(pipeline, "run_environment", side_effect=mock_run_environment),
            patch.object(pipeline, "check_memory_service_health", return_value=True),
        ):
            pipeline.run_all_iterations(
                base_config=cfg,
                iterations=1,
                state_path=state_path,
                models_dir=tmp_path,
                report_path=tmp_path / "report.json",
                comparison_path=comparison_path,
            )

        # Only iteration 1 should have been run (iter 0 already completed)
        assert 0 not in run_calls, "Iteration 0 must be skipped (already completed)"
        assert 1 in run_calls, "Iteration 1 must run"

    def test_baseline_disables_memory(self, tmp_path: Path) -> None:
        """TRAIN-06: Iteration 0 (baseline) sets memory_agent.enabled=False."""
        pipeline = _import_pipeline()

        cfg = self._mock_config(tmp_path)

        def mock_model_copy(deep: bool = False) -> Any:
            new_cfg = MagicMock()
            new_cfg.system = cfg.system
            new_cfg.paths = cfg.paths
            new_cfg.memory_agent = MagicMock(
                enabled=True, meta_training=True, base_url="http://localhost:8889"
            )
            return new_cfg

        cfg.model_copy = mock_model_copy

        mock_run_env = MagicMock(side_effect=lambda env_name, **kw: _make_env_result(env_name))

        with patch.object(pipeline, "run_environment", mock_run_env):
            pipeline.run_all_iterations(
                base_config=cfg,
                iterations=0,
                state_path=tmp_path / "state.json",
                models_dir=tmp_path,
                report_path=tmp_path / "report.json",
                comparison_path=tmp_path / "comparison.json",
            )

        # Verify the config that run_environment was actually called with
        assert mock_run_env.call_count >= 1
        # Check the first call's config argument (iteration 0 = baseline)
        first_call_kwargs = mock_run_env.call_args_list[0]
        iter0_cfg = first_call_kwargs.kwargs.get("config") or first_call_kwargs[1].get("config")
        if iter0_cfg is None:
            # Positional: run_environment(env_name, config, ...)
            iter0_cfg = first_call_kwargs[0][1] if len(first_call_kwargs[0]) > 1 else None
        assert iter0_cfg is not None, "run_environment must receive a config argument"
        assert iter0_cfg.memory_agent.enabled is False
        assert iter0_cfg.memory_agent.meta_training is False

    def test_memory_iterations_enable_memory(self, tmp_path: Path) -> None:
        """TRAIN-06: Iterations 1+ set memory_agent.enabled=True, meta_training=True."""
        pipeline = _import_pipeline()

        cfg = self._mock_config(tmp_path)
        captured_configs: list[Any] = []

        def mock_model_copy(deep: bool = False) -> Any:
            new_cfg = MagicMock()
            new_cfg.system = cfg.system
            new_cfg.paths = cfg.paths
            new_cfg.memory_agent = MagicMock(
                enabled=False, meta_training=False, base_url="http://localhost:8889"
            )
            captured_configs.append(new_cfg)
            return new_cfg

        cfg.model_copy = mock_model_copy

        def mock_run_environment(
            env_name: str, config: Any, models_dir: Path, **kwargs: Any
        ) -> Any:
            return _make_env_result(env_name)

        mock_memory_client = MagicMock()
        mock_memory_client.consolidate.return_value = True

        with (
            patch.object(pipeline, "run_environment", side_effect=mock_run_environment),
            patch.object(pipeline, "check_memory_service_health", return_value=True),
            patch("swingrl.memory.client.MemoryClient", return_value=mock_memory_client),
        ):
            pipeline.run_all_iterations(
                base_config=cfg,
                iterations=2,
                state_path=tmp_path / "state.json",
                models_dir=tmp_path,
                report_path=tmp_path / "report.json",
                comparison_path=tmp_path / "comparison.json",
            )

        # iter 1+ must have memory enabled
        assert len(captured_configs) >= 2
        for iter_cfg in captured_configs[1:]:
            assert iter_cfg.memory_agent.enabled is True
            assert iter_cfg.memory_agent.meta_training is True

    def test_consolidate_called_between_memory_iterations(self, tmp_path: Path) -> None:
        """TRAIN-06: consolidate() called after each memory-enabled iteration (i > 0)."""
        pipeline = _import_pipeline()

        cfg = self._mock_config(tmp_path)

        def mock_model_copy(deep: bool = False) -> Any:
            new_cfg = MagicMock()
            new_cfg.system = cfg.system
            new_cfg.paths = cfg.paths
            new_cfg.memory_agent = MagicMock(
                enabled=False, meta_training=False, base_url="http://localhost:8889", api_key=""
            )
            return new_cfg

        cfg.model_copy = mock_model_copy

        def mock_run_environment(
            env_name: str, config: Any, models_dir: Path, **kwargs: Any
        ) -> Any:
            return _make_env_result(env_name)

        mock_client = MagicMock()
        mock_client.consolidate.return_value = True

        with (
            patch.object(pipeline, "run_environment", side_effect=mock_run_environment),
            patch.object(pipeline, "check_memory_service_health", return_value=True),
            patch("train_pipeline.MemoryClient", return_value=mock_client),
        ):
            pipeline.run_all_iterations(
                base_config=cfg,
                iterations=2,
                state_path=tmp_path / "state.json",
                models_dir=tmp_path,
                report_path=tmp_path / "report.json",
                comparison_path=tmp_path / "comparison.json",
            )

        # consolidate should be called after each memory iteration (i=1,2)
        assert mock_client.consolidate.call_count >= 1


# ===========================================================================
# TestMemoryServiceHealth
# ===========================================================================


class TestMemoryServiceHealth:
    """TRAIN-06: Memory service health checking."""

    def test_check_memory_service_health_returns_true_on_200(self) -> None:
        """TRAIN-06: check_memory_service_health returns True on HTTP 200."""
        pipeline = _import_pipeline()

        class _MockResponse:
            status = 200

            def read(self) -> bytes:
                return b'{"status": "ok"}'

            def __enter__(self) -> _MockResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        with patch("urllib.request.urlopen", return_value=_MockResponse()):
            result = pipeline.check_memory_service_health("http://localhost:8889")
        assert result is True

    def test_check_memory_service_health_returns_false_on_error(self) -> None:
        """TRAIN-06: check_memory_service_health returns False on connection error."""
        pipeline = _import_pipeline()

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = pipeline.check_memory_service_health("http://localhost:8889")
        assert result is False


# ===========================================================================
# TestCLIParser
# ===========================================================================


class TestCLIParser:
    """TRAIN-07: CLI argument parsing for --iterations flag."""

    def test_iterations_flag_default_zero(self) -> None:
        """TRAIN-07: --iterations defaults to 0 (baseline only, no multi-iteration)."""
        pipeline = _import_pipeline()
        parser = pipeline.build_parser()
        args = parser.parse_args([])
        assert args.iterations == 0

    def test_iterations_flag_custom(self) -> None:
        """TRAIN-07: --iterations 5 parsed as 5."""
        pipeline = _import_pipeline()
        parser = pipeline.build_parser()
        args = parser.parse_args(["--iterations", "5"])
        assert args.iterations == 5

    def test_iterations_flag_with_other_args(self) -> None:
        """TRAIN-07: --iterations works alongside --env and --force."""
        pipeline = _import_pipeline()
        parser = pipeline.build_parser()
        args = parser.parse_args(["--env", "equity", "--iterations", "3", "--force"])
        assert args.iterations == 3
        assert args.env == "equity"
        assert args.force is True

    def test_iterations_state_path_arg(self) -> None:
        """TRAIN-07: --state-path arg parses to custom path."""
        pipeline = _import_pipeline()
        parser = pipeline.build_parser()
        args = parser.parse_args(["--state-path", "data/my_state.json"])
        assert "data/my_state.json" in args.state_path
