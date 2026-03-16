"""Full training pipeline orchestration CLI.

Orchestrates walk-forward validation, ensemble gate, tuning rounds, final
deployment training, and JSON report generation for all 6 algo/env combos.

Key invariants:
- Walk-forward uses FULL history. Final training uses RECENT data only.
- Ensemble gate and weight computation are PER-ENVIRONMENT.
- Per-fold gate results are informational; only ensemble gate blocks deployment.
- DuckDB connection opened/closed per-algo unit (not held for entire pipeline).

Usage:
    python scripts/train_pipeline.py --env all
    python scripts/train_pipeline.py --env equity
    python scripts/train_pipeline.py --env crypto --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.memory.client import MemoryClient
from swingrl.training.pipeline_helpers import (
    DEFAULT_TIMESTEPS,
    TUNING_GRID,
    check_ensemble_gate,
    compute_ensemble_weights_from_wf,
    decide_final_timesteps,
    slice_recent,
)
from swingrl.utils.exceptions import ModelError
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

# Algorithm order (equity first, then crypto — sequential per spec)
ALGO_NAMES: list[str] = ["ppo", "a2c", "sac"]

# Tuning round 1 triggers when baseline ensemble Sharpe is below this threshold
_TUNING_SHARPE_THRESHOLD: float = 0.5

# Default paths for multi-iteration state and comparison report
_DEFAULT_STATE_PATH: str = "data/training_state.json"
_DEFAULT_COMPARISON_PATH: str = "data/training_comparison.json"

# Memory service health check endpoint
_MEMORY_HEALTH_ENDPOINT: str = "/health"


# ---------------------------------------------------------------------------
# Multi-iteration training state helpers
# ---------------------------------------------------------------------------


def save_training_state(state: dict[str, Any], path: Path) -> None:
    """Atomically write training state to JSON file.

    Uses write-to-temp-then-rename pattern (POSIX atomic) to survive power outages.

    Args:
        state: Training state dict to serialize.
        path: Destination path for the JSON state file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(str(tmp), str(path))
    log.debug("training_state_saved", path=str(path))


def load_training_state(path: Path) -> dict[str, Any]:
    """Load training state from JSON, returning defaults if file is absent.

    Args:
        path: Path to training_state.json.

    Returns:
        State dict with 'completed_iterations' (list) and 'current_iteration' (int).
    """
    if not path.exists():
        return {"completed_iterations": [], "current_iteration": 0}
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("training_state_load_failed", path=str(path), error=str(exc))
        return {"completed_iterations": [], "current_iteration": 0}


# ---------------------------------------------------------------------------
# Best-model selection and deployment
# ---------------------------------------------------------------------------


def select_best_per_algo_env(
    state: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Select best iteration per algo x env by Sortino ratio (Calmar as tiebreak).

    Looks at final_training per-algo Sortino/Calmar. Falls back to ensemble Sharpe
    if per-algo metrics are missing.

    Args:
        state: Training state dict from load_training_state().

    Returns:
        Dict[env_name][algo_name] -> best_iteration_index.
    """
    envs = ["equity", "crypto"]
    algos = ["ppo", "a2c", "sac"]

    # best[env][algo] = {"iter": int, "sortino": float, "calmar": float}
    best: dict[str, dict[str, dict[str, Any]]] = {e: {} for e in envs}

    for key, iter_data in state.items():
        if not str(key).startswith("iteration_") or not str(key).endswith("_result"):
            continue
        # key format: "iteration_N_result"
        parts = str(key).split("_")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue

        for env_name in envs:
            env_result = iter_data.get(env_name, {}) if isinstance(iter_data, dict) else {}
            final_training = (
                env_result.get("final_training", {}) if isinstance(env_result, dict) else {}
            )
            ensemble_sharpe: float = 0.0
            if isinstance(env_result, dict) and "ensemble_gate" in env_result:
                ensemble_sharpe = float(env_result["ensemble_gate"].get("sharpe", 0.0))

            for algo_name in algos:
                ft = final_training.get(algo_name, {}) if isinstance(final_training, dict) else {}
                sortino = float(ft.get("sortino", ensemble_sharpe)) if isinstance(ft, dict) else 0.0
                calmar = float(ft.get("calmar", 0.0)) if isinstance(ft, dict) else 0.0

                current = best[env_name].get(algo_name)
                if current is None:
                    best[env_name][algo_name] = {"iter": idx, "sortino": sortino, "calmar": calmar}
                elif sortino > current["sortino"] or (
                    sortino == current["sortino"] and calmar > current["calmar"]
                ):
                    best[env_name][algo_name] = {"iter": idx, "sortino": sortino, "calmar": calmar}

    # If state has no iteration results, fall back to completed_iterations list
    for env_name in envs:
        for algo_name in algos:
            if algo_name not in best[env_name]:
                # Default to first completed iteration
                completed = state.get("completed_iterations", [])
                best[env_name][algo_name] = {
                    "iter": completed[0] if completed else 0,
                    "sortino": 0.0,
                    "calmar": 0.0,
                }

    return {
        env: {algo: v["iter"] for algo, v in algos_map.items()} for env, algos_map in best.items()
    }


def deploy_best_models(
    winners: dict[str, dict[str, int]],
    models_dir: Path,
) -> None:
    """Copy best iteration models from iterations/ to active/.

    Args:
        winners: Dict[env_name][algo_name] -> best_iteration_index.
        models_dir: Root models directory (contains iterations/ and active/).
    """
    for env_name, algo_winners in winners.items():
        for algo_name, iter_idx in algo_winners.items():
            src_dir = models_dir / "iterations" / f"iter_{iter_idx}" / env_name / algo_name
            dst_dir = models_dir / "active" / env_name / algo_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for filename in ["model.zip", "vec_normalize.pkl"]:
                src = src_dir / filename
                dst = dst_dir / filename
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                    log.info(
                        "best_model_deployed",
                        env=env_name,
                        algo=algo_name,
                        iter=iter_idx,
                        file=filename,
                    )
                else:
                    log.warning(
                        "best_model_file_missing",
                        env=env_name,
                        algo=algo_name,
                        iter=iter_idx,
                        file=filename,
                        src=str(src),
                    )


def write_comparison_report(
    state: dict[str, Any],
    report_path: Path,
    winners: dict[str, dict[str, int]],
) -> None:
    """Write data/training_comparison.json with per-iteration metrics.

    Args:
        state: Training state dict.
        report_path: Output path for the comparison JSON.
        winners: Best iteration per algo x env (from select_best_per_algo_env).
    """
    report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "winners": winners,
    }

    for key, iter_data in state.items():
        if not str(key).startswith("iteration_") or not str(key).endswith("_result"):
            continue
        parts = str(key).split("_")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue

        iter_key = f"iteration_{idx}"
        is_memory = idx > 0
        report[iter_key] = {
            "memory_enabled": is_memory,
            "results": iter_data,
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("comparison_report_written", path=str(report_path))


# ---------------------------------------------------------------------------
# Memory service health helpers
# ---------------------------------------------------------------------------


def check_memory_service_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check if the swingrl-memory service is healthy.

    Args:
        base_url: Base URL of the memory service (e.g. "http://localhost:8889").
        timeout: HTTP request timeout in seconds.

    Returns:
        True if /health returns HTTP 200, False otherwise.
    """
    url = base_url.rstrip("/") + _MEMORY_HEALTH_ENDPOINT
    try:
        req = urllib.request.Request(url)  # noqa: S310  # nosec B310
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310
        return resp.status == 200  # type: ignore[no-any-return]
    except Exception:
        return False


def wait_for_memory_service(base_url: str, poll_interval: float = 30.0) -> None:
    """Poll memory service health endpoint until healthy.

    Logs a warning on each failed poll. Runs indefinitely until healthy.

    Args:
        base_url: Base URL of the memory service.
        poll_interval: Seconds between health polls.
    """
    log.warning("memory_service_unhealthy_polling_start", base_url=base_url)
    while not check_memory_service_health(base_url):
        log.warning(
            "memory_service_unhealthy_retrying",
            base_url=base_url,
            retry_in_s=poll_interval,
        )
        time.sleep(poll_interval)
    log.info("memory_service_healthy", base_url=base_url)


# ---------------------------------------------------------------------------
# Multi-iteration orchestration
# ---------------------------------------------------------------------------


def run_all_iterations(
    base_config: SwingRLConfig,
    iterations: int,
    state_path: Path,
    models_dir: Path,
    report_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Run baseline + N memory-enhanced training iterations.

    Iteration 0 = baseline (memory disabled).
    Iterations 1..N = memory-enhanced (memory + meta_training enabled).

    State is persisted atomically to state_path between iterations so training
    can resume after power outages.

    Args:
        base_config: Base SwingRLConfig (deep-copied per iteration).
        iterations: Number of memory-enhanced iterations (0 = baseline only).
        state_path: Path for data/training_state.json (atomic writes).
        models_dir: Root models directory.
        report_path: Path for the final training_report.json.
        comparison_path: Path for data/training_comparison.json.

    Returns:
        Dict summarising all iterations and winners.
    """
    state = load_training_state(state_path)
    total = iterations + 1  # 0..N inclusive

    for i in range(total):
        if i in state.get("completed_iterations", []):
            log.info("iteration_skipped_checkpointed", iteration=i, total=total)
            continue

        iter_start = time.monotonic()

        # Deep-copy config and configure memory per iteration
        cfg = base_config.model_copy(deep=True)
        if i == 0:
            # Baseline: memory disabled
            cfg.memory_agent.enabled = False
            cfg.memory_agent.meta_training = False
            log.info("iteration_start_baseline", iteration=i, total=total)
        else:
            # Memory-enhanced
            cfg.memory_agent.enabled = True
            cfg.memory_agent.meta_training = True
            log.info("iteration_start_memory", iteration=i, total=total)
            # Wait for memory service before running memory iterations
            if not check_memory_service_health(cfg.memory_agent.base_url):
                wait_for_memory_service(cfg.memory_agent.base_url)

        state["current_iteration"] = i
        save_training_state(state, state_path)

        # Isolated model directory per iteration
        iter_models_dir = models_dir / "iterations" / f"iter_{i}"

        iter_result: dict[str, Any] = {}
        for env_name in ["equity", "crypto"]:
            try:
                env_result = run_environment(
                    env_name=env_name,
                    config=cfg,
                    models_dir=iter_models_dir,
                    force=True,  # Always force within iterations — each has own directory
                    report={},
                )
                iter_result[env_name] = env_result
            except Exception as exc:
                log.error(
                    "iteration_env_failed_retry",
                    iteration=i,
                    env=env_name,
                    error=str(exc),
                )
                # Retry once
                try:
                    env_result = run_environment(
                        env_name=env_name,
                        config=cfg,
                        models_dir=iter_models_dir,
                        force=True,
                        report={},
                    )
                    iter_result[env_name] = env_result
                except Exception as exc2:
                    log.error(
                        "iteration_env_failed_skip",
                        iteration=i,
                        env=env_name,
                        error=str(exc2),
                    )
                    iter_result[env_name] = {"error": str(exc2)}

        # Save iteration results
        state[f"iteration_{i}_result"] = iter_result
        completed = state.get("completed_iterations", [])
        completed.append(i)
        state["completed_iterations"] = completed
        save_training_state(state, state_path)

        iter_elapsed = time.monotonic() - iter_start
        log.info(
            "iteration_complete",
            iteration=i,
            total=total,
            elapsed_seconds=round(iter_elapsed, 1),
            elapsed_hours=round(iter_elapsed / 3600, 2),
        )

        # Consolidate memories after each memory-enabled iteration
        if i > 0:
            try:
                memory_client = MemoryClient(
                    base_url=cfg.memory_agent.base_url,
                    default_timeout=cfg.memory_agent.timeout_sec,
                    api_key=getattr(cfg.memory_agent, "api_key", ""),
                )
                memory_client.consolidate(timeout=120.0)
                log.info("memory_consolidated", after_iteration=i)
            except Exception as exc:
                log.warning("memory_consolidation_failed", iteration=i, error=str(exc))

    # Select best per algo x env by Sortino (Calmar tiebreak)
    winners = select_best_per_algo_env(state)
    log.info("best_models_selected", winners=winners)

    # Deploy winners to models/active/
    deploy_best_models(winners, models_dir)

    # Write comparison report
    write_comparison_report(state, comparison_path, winners)

    # Build final summary
    final_report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "total_iterations": total,
        "winners": winners,
        "comparison_report": str(comparison_path),
    }
    _write_json_report(final_report, report_path)

    return final_report


# ---------------------------------------------------------------------------
# Feature loading (imported from scripts/train.py for reuse)
# ---------------------------------------------------------------------------


# Re-export the loader from train.py so tests can patch train_pipeline._load_features_prices
def _load_features_prices(
    conn: Any,
    env_name: str,
    config: SwingRLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and prices from DuckDB.

    Delegates to the implementation in scripts/train.py.

    Args:
        conn: Active DuckDB connection.
        env_name: Environment name ("equity" or "crypto").
        config: Validated SwingRLConfig.

    Returns:
        Tuple of (features_array, prices_array) both float32.
    """
    import importlib
    import sys

    # Import train.py by path (it's in scripts/, not a package)
    train_mod_name = "train"
    if train_mod_name not in sys.modules:
        scripts_dir = Path(__file__).parent
        spec_path = scripts_dir / "train.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location(train_mod_name, spec_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[train_mod_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]

    train_mod = sys.modules[train_mod_name]
    return train_mod._load_features_prices(conn, env_name, config)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------


def _check_algo_checkpoint(env_name: str, algo_name: str, models_dir: Path) -> bool:
    """Check if a trained model already exists for this algo/env.

    Args:
        env_name: Environment name.
        algo_name: Algorithm name.
        models_dir: Models root directory.

    Returns:
        True if model.zip exists (checkpoint present).
    """
    model_path = models_dir / "active" / env_name / algo_name / "model.zip"
    return model_path.exists()


def _verify_deployment(env_name: str, models_dir: Path) -> None:
    """Verify model.zip + vec_normalize.pkl exist for all 3 algos.

    Args:
        env_name: Environment name.
        models_dir: Models root directory.

    Raises:
        ModelError: If any required file is missing.
    """
    for algo in ALGO_NAMES:
        model_path = models_dir / "active" / env_name / algo / "model.zip"
        vec_path = models_dir / "active" / env_name / algo / "vec_normalize.pkl"

        if not model_path.exists():
            msg = f"Deployment check failed: model.zip missing for {env_name}/{algo}"
            log.error("deployment_file_missing", path=str(model_path))
            raise ModelError(msg)

        if not vec_path.exists():
            msg = f"Deployment check failed: vec_normalize.pkl missing for {env_name}/{algo}"
            log.error("deployment_file_missing", path=str(vec_path))
            raise ModelError(msg)

    log.info("deployment_verified", env_name=env_name)


def _evaluate_gate_and_decide(
    gate_result: dict[str, Any],
    baseline_sharpe: float,
    env_name: str,
) -> dict[str, Any]:
    """Evaluate ensemble gate and decide whether to deploy or block.

    Args:
        gate_result: Dict with 'passed', 'sharpe', 'mdd' from check_ensemble_gate.
        baseline_sharpe: Baseline OOS Sharpe (used to decide if tuning is needed).
        env_name: Environment name (for logging).

    Returns:
        Dict with 'deploy' bool and diagnostic info.
    """
    passed = gate_result.get("passed", False)
    sharpe = gate_result.get("sharpe", 0.0)
    mdd = gate_result.get("mdd", -1.0)

    if passed:
        log.info(
            "ensemble_gate_passed",
            env_name=env_name,
            sharpe=round(float(sharpe), 4),
            mdd=round(float(mdd), 4),
        )
        return {"deploy": True, "sharpe": sharpe, "mdd": mdd, "tuning_triggered": False}

    log.warning(
        "ensemble_gate_failed",
        env_name=env_name,
        sharpe=round(float(sharpe), 4),
        mdd=round(float(mdd), 4),
    )
    return {
        "deploy": False,
        "sharpe": sharpe,
        "mdd": mdd,
        "tuning_triggered": baseline_sharpe < _TUNING_SHARPE_THRESHOLD,
    }


def _write_json_report(report: dict[str, Any], report_path: Path) -> None:
    """Write training report to JSON file.

    Args:
        report: Report dict to serialize.
        report_path: Output path for the JSON file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("json_report_written", path=str(report_path))


def _write_model_metadata(
    conn: Any,
    env_name: str,
    algo_name: str,
    model_path: Path,
    vec_normalize_path: Path,
    total_timesteps: int,
    converged_at: int | None,
    ensemble_weight: float | None,
) -> None:
    """Write model metadata to DuckDB model_metadata table.

    Args:
        conn: DuckDB connection.
        env_name: Environment name.
        algo_name: Algorithm name.
        model_path: Path to saved model.zip.
        vec_normalize_path: Path to vec_normalize.pkl.
        total_timesteps: Total training timesteps.
        converged_at: Step at which training converged (or None).
        ensemble_weight: Ensemble weight for this algo (or None).
    """
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    model_id = f"{env_name}-v1.1.0-{algo_name}-{date_str}"

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO model_metadata (
                model_id, environment, algorithm, version,
                training_start_date, training_end_date,
                total_timesteps, converged_at_step,
                validation_sharpe, ensemble_weight,
                model_path, vec_normalize_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_id,
                env_name,
                algo_name,
                "v1.1.0",
                date_str,
                date_str,
                total_timesteps,
                converged_at,
                None,
                ensemble_weight,
                str(model_path),
                str(vec_normalize_path),
            ],
        )
    except Exception as exc:
        # Log and continue — metadata write failure is non-fatal
        log.warning("model_metadata_write_failed", error=str(exc), algo=algo_name)


# ---------------------------------------------------------------------------
# Parallelization workers (top-level for picklability with ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _reconstruct_config(config_dict: dict[str, Any]) -> SwingRLConfig:
    """Reconstruct SwingRLConfig from a dict in a worker process.

    load_config() produces an unpicklable local class (_ConfigWithYaml).
    Serialize config to dict in the parent and reconstruct in each worker.
    """
    return SwingRLConfig(**config_dict)


def _run_wf_for_algo(
    env_name: str,
    algo_name: str,
    config_dict: dict[str, Any],
    features: np.ndarray,
    prices: np.ndarray,
    models_dir: Path,
    timesteps: int,
) -> tuple[str, list[Any]]:
    """Run walk-forward validation for one algo. Top-level for picklability.

    Args:
        env_name: Environment name.
        algo_name: Algorithm name.
        config_dict: SwingRLConfig serialized as dict (avoids pickle issues).
        features: Full feature array.
        prices: Full price array.
        models_dir: Models root directory.
        timesteps: Training timesteps per fold.

    Returns:
        Tuple of (algo_name, fold_results_list).
    """
    from swingrl.agents.backtest import WalkForwardBacktester

    config = _reconstruct_config(config_dict)

    log.info("pipeline_wf_started", env_name=env_name, algo=algo_name, pid=os.getpid())

    backtester = WalkForwardBacktester(config=config, db=None)
    folds = backtester.run(
        env_name=env_name,
        algo_name=algo_name,
        features=features,
        prices=prices,
        models_dir=models_dir,
        total_timesteps=timesteps,
    )

    log.info("pipeline_wf_complete", env_name=env_name, algo=algo_name, fold_count=len(folds))
    return (algo_name, folds)


def _train_final_algo(
    env_name: str,
    algo_name: str,
    config_dict: dict[str, Any],
    features: np.ndarray,
    prices: np.ndarray,
    models_dir: Path,
    logs_dir: Path,
    timesteps: int,
    hp_override: dict[str, Any] | None,
    use_meta: bool,
) -> tuple[str, dict[str, Any]]:
    """Train final deployment model for one algo. Top-level for picklability.

    Args:
        env_name: Environment name.
        algo_name: Algorithm name.
        config: SwingRLConfig.
        features: Recent feature array for final training.
        prices: Recent price array for final training.
        models_dir: Models root directory.
        logs_dir: Logs directory.
        timesteps: Training timesteps.
        hp_override: Optional hyperparameter overrides from tuning.
        use_meta: Whether to use MetaTrainingOrchestrator.

    Returns:
        Tuple of (algo_name, result_dict) with training metadata.
    """
    from swingrl.training.trainer import TrainingOrchestrator

    config = _reconstruct_config(config_dict)
    t_start = time.monotonic()

    orchestrator = TrainingOrchestrator(
        config=config,
        models_dir=models_dir,
        logs_dir=logs_dir,
    )

    if use_meta:
        from swingrl.memory.client import MemoryClient
        from swingrl.memory.training.meta_orchestrator import MetaTrainingOrchestrator

        client = MemoryClient(
            base_url=config.memory_agent.base_url,
            default_timeout=config.memory_agent.timeout_sec,
        )
        meta = MetaTrainingOrchestrator(config=config, memory_client=client)
        train_result = meta.run(
            env_name=env_name,
            algo_name=algo_name,
            trainer=orchestrator,
            features=features,
            prices=prices,
            total_timesteps=timesteps,
            hyperparams_override=hp_override,
        )
    else:
        train_result = orchestrator.train(
            env_name=env_name,
            algo_name=algo_name,
            features=features,
            prices=prices,
            total_timesteps=timesteps,
            hyperparams_override=hp_override,
        )

    wall_algo = time.monotonic() - t_start

    result_dict = {
        "timesteps": timesteps,
        "wall_clock_s": round(wall_algo, 1),
        "converged_at": train_result.converged_at_step,
        "model_path": str(train_result.model_path),
        "vec_normalize_path": str(train_result.vec_normalize_path),
    }

    log.info(
        "pipeline_algo_trained",
        env_name=env_name,
        algo=algo_name,
        timesteps=timesteps,
        wall_s=round(wall_algo, 1),
        pid=os.getpid(),
    )

    return (algo_name, result_dict)


# ---------------------------------------------------------------------------
# Per-environment training flow
# ---------------------------------------------------------------------------


def run_environment(
    env_name: str,
    config: SwingRLConfig,
    models_dir: Path,
    force: bool,
    report: dict[str, Any],
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline for one environment.

    Flow:
    1. Load features + prices from DuckDB
    2. Walk-forward validation for each algo (or skip if checkpointed)
    3. Compute ensemble weights from real OOS Sharpe
    4. Check ensemble gate
    5. Trigger tuning rounds if gate fails AND baseline Sharpe < 0.5
    6. Train final deployment models on RECENT data
    7. Verify deployment files exist
    8. Write model metadata to DuckDB
    9. Update report dict and optionally write to disk

    Args:
        env_name: Environment name ("equity" or "crypto").
        config: Validated SwingRLConfig.
        models_dir: Models root directory.
        force: Re-run even if checkpoints exist.
        report: Shared report dict to update (mutated in-place).
        report_path: Optional path to write JSON report after completion.

    Returns:
        Dict with 'ensemble_gate', 'ensemble_weights', 'walk_forward' keys.
    """
    from swingrl.agents.backtest import WalkForwardBacktester
    from swingrl.training.trainer import TrainingOrchestrator

    env_start = time.monotonic()
    db_path = Path(config.system.duckdb_path)

    log.info("pipeline_env_started", env_name=env_name)

    # Load full features + prices (used for walk-forward)
    conn = duckdb.connect(str(db_path))
    try:
        features_full, prices_full = _load_features_prices(conn, env_name, config)
    finally:
        conn.close()

    log.info(
        "features_loaded",
        env_name=env_name,
        bars=len(features_full),
        feature_dim=features_full.shape[1],
    )

    # Slice recent data (used for final deployment training)
    features_recent, prices_recent = slice_recent(features_full, prices_full, env_name)

    # -------------------------------------------------------------------
    # Walk-forward validation for each algo
    # -------------------------------------------------------------------
    all_wf_results: dict[str, list[Any]] = {}
    tuning_best_params: dict[str, dict[str, Any]] = {}

    # Determine which algos need WF (not checkpointed)
    algos_to_run: list[str] = []
    for algo_name in ALGO_NAMES:
        if not force and _check_algo_checkpoint(env_name, algo_name, models_dir):
            log.info("pipeline_checkpoint_skip", env_name=env_name, algo=algo_name)
            all_wf_results[algo_name] = []
        else:
            algos_to_run.append(algo_name)

    # Run walk-forward for all non-checkpointed algos in parallel
    if algos_to_run:
        max_workers = min(3, len(algos_to_run))
        log.info(
            "pipeline_wf_parallel_start",
            env_name=env_name,
            algos=algos_to_run,
            max_workers=max_workers,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_wf_for_algo,
                    env_name,
                    algo,
                    config.model_dump(),
                    features_full,
                    prices_full,
                    models_dir,
                    DEFAULT_TIMESTEPS[env_name],
                ): algo
                for algo in algos_to_run
            }
            for future in as_completed(futures):
                algo_name_result, folds = future.result()
                all_wf_results[algo_name_result] = folds

    # If all algos were checkpointed, skip ensemble computation
    has_wf_data = any(len(v) > 0 for v in all_wf_results.values())

    if not has_wf_data:
        log.info("pipeline_all_checkpointed", env_name=env_name)
        result = {
            "ensemble_gate": {"passed": True, "sharpe": 0.0, "mdd": 0.0},
            "ensemble_weights": {},
            "walk_forward": {k: [] for k in ALGO_NAMES},
            "tuning_rounds": [],
            "wall_clock_total_s": round(time.monotonic() - env_start, 1),
        }
        report[env_name] = result
        if report_path:
            _write_json_report(report, report_path)
        return result

    # -------------------------------------------------------------------
    # Ensemble gate and weights
    # -------------------------------------------------------------------
    passed, ensemble_sharpe, ensemble_mdd = check_ensemble_gate(all_wf_results)
    ensemble_weights = compute_ensemble_weights_from_wf(config, all_wf_results, env_name)

    gate_result = {
        "passed": passed,
        "sharpe": float(ensemble_sharpe),
        "mdd": float(ensemble_mdd),
    }
    _evaluate_gate_and_decide(gate_result, ensemble_sharpe, env_name)

    tuning_rounds: list[dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Tuning rounds (if gate fails AND baseline Sharpe < 0.5)
    # -------------------------------------------------------------------
    if not passed and ensemble_sharpe < _TUNING_SHARPE_THRESHOLD:
        log.info(
            "pipeline_tuning_round1_start",
            env_name=env_name,
            baseline_sharpe=round(ensemble_sharpe, 4),
        )

        # Round 1: PPO variants
        best_ppo_result = None
        best_ppo_sharpe = ensemble_sharpe
        best_ppo_params: dict[str, Any] = {}

        orchestrator_r1 = TrainingOrchestrator(
            config=config,
            models_dir=models_dir,
            logs_dir=Path(config.paths.logs_dir),
        )

        for variant_params in TUNING_GRID.get(1, {}).get("ppo", []):
            conn = duckdb.connect(str(db_path))
            try:
                result_v = orchestrator_r1.train(
                    env_name=env_name,
                    algo_name="ppo",
                    features=features_full,
                    prices=prices_full,
                    total_timesteps=DEFAULT_TIMESTEPS[env_name],
                    hyperparams_override=variant_params,
                )
                backtester_v = WalkForwardBacktester(config=config, db=None)
                folds_v = backtester_v.run(
                    env_name=env_name,
                    algo_name="ppo",
                    features=features_full,
                    prices=prices_full,
                    models_dir=models_dir,
                    total_timesteps=DEFAULT_TIMESTEPS[env_name],
                )
                _, ppo_sharpe, _ = check_ensemble_gate({"ppo": folds_v})
                if ppo_sharpe > best_ppo_sharpe:
                    best_ppo_sharpe = ppo_sharpe
                    best_ppo_result = result_v
                    best_ppo_params = variant_params
            finally:
                conn.close()

        if best_ppo_params:
            tuning_best_params["ppo"] = best_ppo_params

        tuning_rounds.append(
            {
                "round": 1,
                "algo": "ppo",
                "best_sharpe": round(best_ppo_sharpe, 4),
                "best_params": best_ppo_params,
            }
        )

        # Recompute gate after tuning round 1
        if best_ppo_result:
            all_wf_results["ppo"] = []  # reset to trigger recompute
            passed, ensemble_sharpe, ensemble_mdd = check_ensemble_gate(all_wf_results)
            gate_result = {
                "passed": passed,
                "sharpe": float(ensemble_sharpe),
                "mdd": float(ensemble_mdd),
            }

        # Round 2: A2C + SAC variants (if still failing)
        if not passed:
            log.info("pipeline_tuning_round2_start", env_name=env_name)

            for algo_r2, variants in TUNING_GRID.get(2, {}).items():
                best_sharpe_r2 = 0.0
                best_params_r2: dict[str, Any] = {}

                for variant_params in variants:
                    conn = duckdb.connect(str(db_path))
                    try:
                        orchestrator_r2 = TrainingOrchestrator(
                            config=config,
                            models_dir=models_dir,
                            logs_dir=Path(config.paths.logs_dir),
                        )
                        orchestrator_r2.train(
                            env_name=env_name,
                            algo_name=algo_r2,
                            features=features_full,
                            prices=prices_full,
                            total_timesteps=DEFAULT_TIMESTEPS[env_name],
                            hyperparams_override=variant_params,
                        )
                        backtester_r2 = WalkForwardBacktester(config=config, db=None)
                        folds_r2 = backtester_r2.run(
                            env_name=env_name,
                            algo_name=algo_r2,
                            features=features_full,
                            prices=prices_full,
                            models_dir=models_dir,
                        )
                        _, sharpe_r2, _ = check_ensemble_gate({algo_r2: folds_r2})
                        if sharpe_r2 > best_sharpe_r2:
                            best_sharpe_r2 = sharpe_r2
                            best_params_r2 = variant_params
                    finally:
                        conn.close()

                if best_params_r2:
                    tuning_best_params[algo_r2] = best_params_r2

                tuning_rounds.append(
                    {
                        "round": 2,
                        "algo": algo_r2,
                        "best_sharpe": round(best_sharpe_r2, 4),
                        "best_params": best_params_r2,
                    }
                )

            # Final gate check after both rounds
            passed, ensemble_sharpe, ensemble_mdd = check_ensemble_gate(all_wf_results)
            gate_result = {
                "passed": passed,
                "sharpe": float(ensemble_sharpe),
                "mdd": float(ensemble_mdd),
            }

    # -------------------------------------------------------------------
    # Gate block (both tuning rounds exhausted and still failing)
    # -------------------------------------------------------------------
    if not gate_result["passed"] and ensemble_sharpe < _TUNING_SHARPE_THRESHOLD:
        log.error(
            "pipeline_deployment_blocked",
            env_name=env_name,
            ensemble_sharpe=round(ensemble_sharpe, 4),
            ensemble_mdd=round(ensemble_mdd, 4),
        )
        # Write diagnostic report and return failure result
        result = {
            "ensemble_gate": gate_result,
            "ensemble_weights": ensemble_weights,
            "walk_forward": {
                k: [
                    {"fold": f.fold_number, "oos_sharpe": f.out_of_sample_metrics.get("sharpe")}
                    for f in folds
                ]
                for k, folds in all_wf_results.items()
            },
            "tuning_rounds": tuning_rounds,
            "wall_clock_total_s": round(time.monotonic() - env_start, 1),
        }
        report[env_name] = result
        if report_path:
            _write_json_report(report, report_path)
        return result

    # -------------------------------------------------------------------
    # Final deployment training (RECENT data only)
    # -------------------------------------------------------------------
    log.info(
        "pipeline_final_training_start",
        env_name=env_name,
        recent_bars=len(features_recent),
    )

    final_training: dict[str, dict[str, Any]] = {}

    use_meta = bool(
        getattr(config, "memory_agent", None)
        and config.memory_agent.enabled
        and config.memory_agent.meta_training
    )

    # Prepare per-algo timesteps and overrides
    algo_configs: dict[str, tuple[int, dict[str, Any] | None]] = {}
    for algo_name in ALGO_NAMES:
        ts = DEFAULT_TIMESTEPS[env_name]
        if all_wf_results.get(algo_name):
            last_fold = all_wf_results[algo_name][-1] if all_wf_results[algo_name] else None
            if last_fold is not None:
                from swingrl.training.trainer import TrainingResult

                dummy_result = TrainingResult(
                    model_path=models_dir / "active" / env_name / algo_name / "model.zip",
                    vec_normalize_path=models_dir
                    / "active"
                    / env_name
                    / algo_name
                    / "vec_normalize.pkl",
                    env_name=env_name,
                    algo_name=algo_name,
                    converged_at_step=None,
                    total_timesteps=ts,
                )
                ts = decide_final_timesteps(env_name, dummy_result)
        algo_configs[algo_name] = (ts, tuning_best_params.get(algo_name))

    # Train all 3 algos in parallel
    max_workers = min(3, len(ALGO_NAMES))
    log.info(
        "pipeline_final_training_parallel",
        env_name=env_name,
        algos=ALGO_NAMES,
        max_workers=max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_final_algo,
                env_name,
                algo_name,
                config.model_dump(),
                features_recent,
                prices_recent,
                models_dir,
                Path(config.paths.logs_dir),
                algo_configs[algo_name][0],
                algo_configs[algo_name][1],
                use_meta,
            ): algo_name
            for algo_name in ALGO_NAMES
        }
        for future in as_completed(futures):
            algo_name = futures[future]
            _, result_dict = future.result()
            final_training[algo_name] = result_dict

    # Deferred DuckDB writes — sequential after all parallel training completes
    conn = duckdb.connect(str(db_path))
    try:
        for algo_name in ALGO_NAMES:
            if algo_name in final_training:
                _write_model_metadata(
                    conn=conn,
                    env_name=env_name,
                    algo_name=algo_name,
                    model_path=Path(final_training[algo_name]["model_path"]),
                    vec_normalize_path=Path(final_training[algo_name]["vec_normalize_path"]),
                    total_timesteps=final_training[algo_name]["timesteps"],
                    converged_at=final_training[algo_name]["converged_at"],
                    ensemble_weight=ensemble_weights.get(algo_name),
                )
    finally:
        conn.close()

    # -------------------------------------------------------------------
    # Verify deployment files
    # -------------------------------------------------------------------
    _verify_deployment(env_name=env_name, models_dir=models_dir)

    # -------------------------------------------------------------------
    # Build result + update report
    # -------------------------------------------------------------------
    wall_total = time.monotonic() - env_start

    result = {
        "walk_forward": {
            k: [
                {
                    "fold": f.fold_number,
                    "oos_sharpe": f.out_of_sample_metrics.get("sharpe"),
                    "oos_mdd": f.out_of_sample_metrics.get("mdd"),
                    "oos_profit_factor": f.out_of_sample_metrics.get("profit_factor"),
                }
                for f in folds
            ]
            for k, folds in all_wf_results.items()
        },
        "ensemble_weights": ensemble_weights,
        "ensemble_gate": gate_result,
        "tuning_rounds": tuning_rounds,
        "final_training": final_training,
        "wall_clock_total_s": round(wall_total, 1),
    }

    report[env_name] = result
    if report_path:
        _write_json_report(report, report_path)

    log.info(
        "pipeline_env_complete",
        env_name=env_name,
        gate_passed=gate_result["passed"],
        ensemble_sharpe=round(float(gate_result["sharpe"]), 4),
        wall_s=round(wall_total, 1),
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training pipeline CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="SwingRL full training pipeline: walk-forward -> ensemble -> deploy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/train_pipeline.py --env all
  python scripts/train_pipeline.py --env equity --force
  python scripts/train_pipeline.py --env crypto --config config/swingrl.yaml
""",
    )
    parser.add_argument(
        "--env",
        choices=["equity", "crypto", "all"],
        default="all",
        help="Environment to train (default: all — sequential equity then crypto).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Root directory for model storage (default: models).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/training_report.json",
        help="Path to write JSON training report (default: data/training_report.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run even if checkpoints exist.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help=(
            "Number of memory-enhanced iterations after baseline (default: 0 = baseline only). "
            "When N > 0: runs iteration 0 (baseline, memory off) then N iterations with memory on."
        ),
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=_DEFAULT_STATE_PATH,
        help=f"Path to training state JSON for resume support (default: {_DEFAULT_STATE_PATH}).",
    )
    parser.add_argument(
        "--comparison-path",
        type=str,
        default=_DEFAULT_COMPARISON_PATH,
        help=f"Path to write training comparison JSON (default: {_DEFAULT_COMPARISON_PATH}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full training pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = deployment blocked or error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    models_dir = Path(args.models_dir)
    report_path = Path(args.report)
    force: bool = args.force
    iterations: int = args.iterations

    env_arg: str = args.env
    envs = ["equity", "crypto"] if env_arg == "all" else [env_arg]

    log.info(
        "training_pipeline_started",
        envs=envs,
        models_dir=str(models_dir),
        report=str(report_path),
        force=force,
        iterations=iterations,
    )

    start_time = time.monotonic()

    # ---------------------------------------------------------------------------
    # Multi-iteration mode (--iterations N > 0): hand off to run_all_iterations()
    # ---------------------------------------------------------------------------
    if iterations > 0:
        state_path = Path(args.state_path)
        comparison_path = Path(args.comparison_path)
        try:
            run_all_iterations(
                base_config=config,
                iterations=iterations,
                state_path=state_path,
                models_dir=models_dir,
                report_path=report_path,
                comparison_path=comparison_path,
            )
        except Exception:
            log.exception("training_pipeline_iterations_failed")
            return 1

        elapsed = time.monotonic() - start_time
        log.info(
            "training_pipeline_complete",
            envs=envs,
            elapsed_seconds=round(elapsed, 1),
            iterations=iterations,
            success=True,
        )
        return 0

    # ---------------------------------------------------------------------------
    # Single-run mode (--iterations 0, default): existing per-env loop
    # ---------------------------------------------------------------------------
    report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }

    overall_success = True

    for env_name in envs:
        try:
            env_result = run_environment(
                env_name=env_name,
                config=config,
                models_dir=models_dir,
                force=force,
                report=report,
                report_path=report_path,
            )

            gate = env_result.get("ensemble_gate", {})
            if not gate.get("passed", False):
                log.error(
                    "pipeline_deployment_blocked_summary",
                    env_name=env_name,
                    sharpe=gate.get("sharpe"),
                    mdd=gate.get("mdd"),
                )
                overall_success = False

        except ModelError as exc:
            log.error("pipeline_model_error", env_name=env_name, error=str(exc))
            overall_success = False
        except Exception:
            log.exception("pipeline_unexpected_error", env_name=env_name)
            overall_success = False

    elapsed = time.monotonic() - start_time
    _write_json_report(report, report_path)

    log.info(
        "training_pipeline_complete",
        envs=envs,
        elapsed_seconds=round(elapsed, 1),
        success=overall_success,
    )

    return 0 if overall_success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("training_pipeline_interrupted")
        sys.exit(130)
