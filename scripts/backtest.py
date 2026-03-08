"""CLI entry point for walk-forward backtesting.

Runs walk-forward validation for each algorithm on an environment,
evaluates per-fold metrics, checks validation gates, and writes
results to DuckDB.

Usage:
    python scripts/backtest.py --env equity
    python scripts/backtest.py --env crypto --timesteps 500000
    python scripts/backtest.py --env equity --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import structlog

from swingrl.agents.backtest import WalkForwardBacktester
from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.training.trainer import ALGO_MAP
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

ALGO_NAMES = list(ALGO_MAP.keys())


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtesting for SwingRL agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/backtest.py --env equity
  python scripts/backtest.py --env crypto --timesteps 500000
  python scripts/backtest.py --env equity --models-dir models
""",
    )

    parser.add_argument(
        "--env",
        choices=["equity", "crypto"],
        required=True,
        help="Environment to backtest.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps per fold per algorithm (default: 1000000).",
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

    features = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    prices = features[:, 0:1]

    log.info(
        "data_loaded",
        env_name=env_name,
        rows=len(df),
        feature_cols=features.shape[1],
    )

    return features, prices


def _print_summary(
    env_name: str,
    algo_name: str,
    fold_results: list,
) -> bool:
    """Print summary table for one algorithm's backtest results.

    Args:
        env_name: Environment name.
        algo_name: Algorithm name.
        fold_results: List of FoldResult objects.

    Returns:
        True if all gates passed across folds.
    """
    all_passed = True
    print(f"\n{'=' * 60}")  # noqa: T201
    print(f"  {env_name.upper()} - {algo_name.upper()} Walk-Forward Results")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201
    print(f"  {'Fold':>4}  {'OOS Sharpe':>10}  {'MDD':>8}  {'PF':>8}  {'Gate':>6}")  # noqa: T201
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 6}")  # noqa: T201

    for fr in fold_results:
        oos = fr.out_of_sample_metrics
        sharpe = oos.get("sharpe", float("nan"))
        mdd = oos.get("mdd", float("nan"))
        pf = oos.get("profit_factor", float("nan"))
        passed = "PASS" if fr.gate_result.passed else "FAIL"
        if not fr.gate_result.passed:
            all_passed = False

        print(f"  {fr.fold_number:>4}  {sharpe:>10.4f}  {mdd:>8.4f}  {pf:>8.2f}  {passed:>6}")  # noqa: T201

    # Aggregate
    oos_sharpes = [fr.out_of_sample_metrics.get("sharpe", 0.0) for fr in fold_results]
    avg_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0
    print(f"\n  Average OOS Sharpe: {avg_sharpe:.4f}")  # noqa: T201
    print(f"  Folds: {len(fold_results)}")  # noqa: T201
    print(f"  All gates passed: {all_passed}")  # noqa: T201

    return all_passed


def main(argv: list[str] | None = None) -> int:
    """Run walk-forward backtesting pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = all gates pass, 1 = any gate fails).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    env_name: str = args.env
    models_dir = Path(args.models_dir)
    total_timesteps: int = args.timesteps

    log.info(
        "backtest_pipeline_started",
        env_name=env_name,
        total_timesteps=total_timesteps,
    )

    start_time = time.monotonic()

    # Connect to DuckDB
    db_path = Path(config.system.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    # Initialize DatabaseManager for result storage
    DatabaseManager.reset()
    db = DatabaseManager(config)
    db.init_schema()

    try:
        features, prices = _load_features_prices(conn, env_name)

        backtester = WalkForwardBacktester(config=config, db=db)

        all_passed = True
        for algo_name in ALGO_NAMES:
            log.info("backtesting_algo", algo=algo_name, env=env_name)
            try:
                fold_results = backtester.run(
                    env_name=env_name,
                    algo_name=algo_name,
                    features=features,
                    prices=prices,
                    models_dir=models_dir,
                    total_timesteps=total_timesteps,
                )

                algo_passed = _print_summary(env_name, algo_name, fold_results)
                if not algo_passed:
                    all_passed = False

                log.info(
                    "backtesting_algo_complete",
                    algo=algo_name,
                    folds=len(fold_results),
                    all_gates_passed=algo_passed,
                )

            except Exception:
                log.exception("backtesting_algo_failed", algo=algo_name, env=env_name)
                all_passed = False

    finally:
        conn.close()

    elapsed = time.monotonic() - start_time
    log.info(
        "backtest_pipeline_complete",
        env_name=env_name,
        elapsed_seconds=round(elapsed, 2),
        all_passed=all_passed,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("backtest_interrupted")
        sys.exit(130)
