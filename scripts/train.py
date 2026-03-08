"""CLI entry point for RL agent training.

Trains PPO/A2C/SAC agents on equity or crypto environments, computes
ensemble weights via Sharpe-weighted softmax, and writes model metadata
to DuckDB.

Usage:
    python scripts/train.py --env equity --algo ppo
    python scripts/train.py --env equity --algo all
    python scripts/train.py --env crypto --algo all --timesteps 500000
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import structlog

from swingrl.config.schema import load_config
from swingrl.training.ensemble import EnsembleBlender
from swingrl.training.trainer import ALGO_MAP, TrainingOrchestrator
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

ALGO_NAMES = list(ALGO_MAP.keys())


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Train SwingRL RL agents (PPO/A2C/SAC) on equity or crypto environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/train.py --env equity --algo ppo
  python scripts/train.py --env equity --algo all
  python scripts/train.py --env crypto --algo all --timesteps 500000
  python scripts/train.py --env equity --algo sac --config config/swingrl.yaml
""",
    )

    parser.add_argument(
        "--env",
        choices=["equity", "crypto"],
        required=True,
        help="Environment to train on.",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "a2c", "sac", "all"],
        default="all",
        help="Algorithm to train (default: all).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps per algorithm (default: 1000000).",
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

    return parser


def _load_features_prices(
    conn: duckdb.DuckDBPyConnection,
    env_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and prices from DuckDB for the given environment.

    Args:
        conn: Active DuckDB connection.
        env_name: Environment name ("equity" or "crypto").

    Returns:
        Tuple of (features_array, prices_array).

    Raises:
        RuntimeError: If no data found in DuckDB.
    """
    table_name = f"features_{env_name}"
    df = conn.execute(f"SELECT * FROM {table_name} ORDER BY rowid").fetchdf()  # noqa: S608  # nosec B608

    if df.empty:
        msg = f"No data found in {table_name} table"
        raise RuntimeError(msg)

    # Price columns are the close prices; features are everything else
    # Convention: first columns are metadata, rest are feature values
    # For now, extract as numpy arrays
    features = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    prices = features[:, 0:1]  # First numeric column as price proxy

    log.info(
        "data_loaded",
        env_name=env_name,
        rows=len(df),
        feature_cols=features.shape[1],
    )

    return features, prices


def _write_model_metadata(
    conn: duckdb.DuckDBPyConnection,
    env_name: str,
    algo_name: str,
    model_path: Path,
    vec_normalize_path: Path,
    total_timesteps: int,
    converged_at: int | None,
    validation_sharpe: float | None,
    ensemble_weight: float | None,
) -> None:
    """Write model metadata row to DuckDB.

    Args:
        conn: Active DuckDB connection.
        env_name: Environment name.
        algo_name: Algorithm name.
        model_path: Path to saved model.
        vec_normalize_path: Path to VecNormalize file.
        total_timesteps: Total training timesteps.
        converged_at: Step at which training converged (or None).
        validation_sharpe: Validation Sharpe ratio (or None).
        ensemble_weight: Ensemble weight (or None).
    """
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    model_id = f"{env_name}-v1.0.0-{algo_name}-{date_str}"
    version = "v1.0.0"

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
            version,
            date_str,
            date_str,
            total_timesteps,
            converged_at,
            validation_sharpe,
            ensemble_weight,
            str(model_path),
            str(vec_normalize_path),
        ],
    )

    log.info(
        "model_metadata_written",
        model_id=model_id,
        ensemble_weight=ensemble_weight,
    )


def main(argv: list[str] | None = None) -> int:
    """Run training pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    env_name: str = args.env
    algos = ALGO_NAMES if args.algo == "all" else [args.algo]
    models_dir = Path(args.models_dir)
    total_timesteps: int = args.timesteps

    log.info(
        "training_pipeline_started",
        env_name=env_name,
        algos=algos,
        total_timesteps=total_timesteps,
    )

    start_time = time.monotonic()

    # Connect to DuckDB for data loading and metadata writes
    db_path = Path(config.system.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    try:
        # Load features and prices
        features, prices = _load_features_prices(conn, env_name)

        orchestrator = TrainingOrchestrator(
            config=config,
            models_dir=models_dir,
            logs_dir=Path(config.paths.logs_dir),
        )

        # Train each algorithm
        results = {}
        for algo_name in algos:
            log.info("training_algo", algo=algo_name, env=env_name)
            try:
                result = orchestrator.train(
                    env_name=env_name,
                    algo_name=algo_name,
                    features=features,
                    prices=prices,
                    total_timesteps=total_timesteps,
                )
                results[algo_name] = result
                log.info(
                    "training_algo_complete",
                    algo=algo_name,
                    model_path=str(result.model_path),
                    converged_at=result.converged_at_step,
                )
            except Exception:
                log.exception("training_algo_failed", algo=algo_name, env=env_name)
                return 1

        # Compute ensemble weights if all algos trained
        if len(results) == len(ALGO_NAMES):
            # Use placeholder Sharpe (actual validation comes from backtest.py)
            # For initial weights, use equal weights
            placeholder_sharpes = dict.fromkeys(ALGO_NAMES, 1.0)
            blender = EnsembleBlender(config)
            ensemble_weights = blender.compute_weights(env_name, placeholder_sharpes)
        else:
            # Single algo trained
            ensemble_weights = dict.fromkeys(results, 1.0)

        log.info("ensemble_weights", weights=ensemble_weights)

        # Write model metadata to DuckDB
        for algo_name, result in results.items():
            _write_model_metadata(
                conn=conn,
                env_name=env_name,
                algo_name=algo_name,
                model_path=result.model_path,
                vec_normalize_path=result.vec_normalize_path,
                total_timesteps=total_timesteps,
                converged_at=result.converged_at_step,
                validation_sharpe=None,  # Set by backtest.py
                ensemble_weight=ensemble_weights.get(algo_name),
            )

    finally:
        conn.close()

    elapsed = time.monotonic() - start_time
    log.info(
        "training_pipeline_complete",
        env_name=env_name,
        algos=algos,
        elapsed_seconds=round(elapsed, 2),
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("training_interrupted")
        sys.exit(130)
