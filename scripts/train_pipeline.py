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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig, load_config
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

    for algo_name in ALGO_NAMES:
        # Checkpoint check
        if not force and _check_algo_checkpoint(env_name, algo_name, models_dir):
            log.info(
                "pipeline_checkpoint_skip",
                env_name=env_name,
                algo=algo_name,
            )
            all_wf_results[algo_name] = []  # no WF results (skipped)
            continue

        log.info("pipeline_wf_started", env_name=env_name, algo=algo_name)

        conn = duckdb.connect(str(db_path))
        try:
            backtester = WalkForwardBacktester(config=config, db=None)
            folds = backtester.run(
                env_name=env_name,
                algo_name=algo_name,
                features=features_full,
                prices=prices_full,
                models_dir=models_dir,
                total_timesteps=DEFAULT_TIMESTEPS[env_name],
            )
            all_wf_results[algo_name] = folds
        finally:
            conn.close()

        log.info(
            "pipeline_wf_complete",
            env_name=env_name,
            algo=algo_name,
            fold_count=len(folds),
        )

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

    for algo_name in ALGO_NAMES:
        t_start = time.monotonic()

        # Determine final timesteps based on tuning convergence
        ts = DEFAULT_TIMESTEPS[env_name]
        if all_wf_results.get(algo_name):
            # Use a dummy TrainingResult-like for decide_final_timesteps
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

        hp_override = tuning_best_params.get(algo_name)
        conn = duckdb.connect(str(db_path))

        try:
            orchestrator = TrainingOrchestrator(
                config=config,
                models_dir=models_dir,
                logs_dir=Path(config.paths.logs_dir),
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
                    features=features_recent,
                    prices=prices_recent,
                    total_timesteps=ts,
                    hyperparams_override=hp_override,
                )
            else:
                train_result = orchestrator.train(
                    env_name=env_name,
                    algo_name=algo_name,
                    features=features_recent,
                    prices=prices_recent,
                    total_timesteps=ts,
                    hyperparams_override=hp_override,
                )

            wall_algo = time.monotonic() - t_start

            # Write model metadata
            _write_model_metadata(
                conn=conn,
                env_name=env_name,
                algo_name=algo_name,
                model_path=train_result.model_path,
                vec_normalize_path=train_result.vec_normalize_path,
                total_timesteps=ts,
                converged_at=train_result.converged_at_step,
                ensemble_weight=ensemble_weights.get(algo_name),
            )

            final_training[algo_name] = {
                "timesteps": ts,
                "wall_clock_s": round(wall_algo, 1),
                "converged_at": train_result.converged_at_step,
                "model_path": str(train_result.model_path),
            }

            log.info(
                "pipeline_algo_trained",
                env_name=env_name,
                algo=algo_name,
                timesteps=ts,
                wall_s=round(wall_algo, 1),
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

    env_arg: str = args.env
    envs = ["equity", "crypto"] if env_arg == "all" else [env_arg]

    log.info(
        "training_pipeline_started",
        envs=envs,
        models_dir=str(models_dir),
        report=str(report_path),
        force=force,
    )

    start_time = time.monotonic()

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
