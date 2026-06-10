"""CLI entry point for RL agent training.

Trains PPO/A2C/SAC agents on equity or crypto environments, computes
ensemble weights via Sharpe-weighted softmax, and writes model metadata
to PostgreSQL.

Usage:
    python scripts/train.py --env equity --algo ppo
    python scripts/train.py --env equity --algo all
    python scripts/train.py --env crypto --algo all --timesteps 500000
    python scripts/train.py --env equity --algo all --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import structlog
from psycopg.rows import dict_row

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.pg_helpers import fetchdf
from swingrl.features.assembler import ObservationAssembler
from swingrl.features.pipeline import _CRYPTO_FEATURE_COLS, _EQUITY_FEATURE_COLS
from swingrl.training.ensemble import EnsembleBlender
from swingrl.training.trainer import ALGO_MAP, TrainingOrchestrator
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

ALGO_NAMES = list(ALGO_MAP.keys())

# Macro series IDs to (6,) array position mapping
_MACRO_SERIES_IDS = ["VIXCLS", "T10Y2Y", "DFF", "CPIAUCSL", "UNRATE"]


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
  python scripts/train.py --env equity --dry-run
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate feature loading and model construction without training. "
            "Loads features, checks shapes, and exits with code 0 on success."
        ),
    )

    return parser


def _get_macro_array(conn: Any, date_str: str) -> np.ndarray:
    """Fetch macro features as (6,) array from macro_features table.

    Reads latest value per series_id on or before date_str. Falls back to
    zeros for missing series. The 6 positions map to: [VIXCLS, T10Y2Y,
    yield_curve_direction, DFF, CPIAUCSL, UNRATE].

    Args:
        conn: PostgreSQL connection.
        date_str: ISO date string (YYYY-MM-DD) used as upper bound for ASOF lookup.

    Returns:
        (6,) float64 array.
    """
    try:
        rows = fetchdf(
            conn.execute(
                """
            SELECT series_id, value FROM macro_features
            WHERE date <= CAST(%s AS DATE)
            ORDER BY date DESC
            """,
                [date_str],
            )
        )

        if rows.empty:
            return np.zeros(6)

        # Get latest value per series_id
        latest: dict[str, float] = {}
        for _, row in rows.iterrows():
            sid = str(row["series_id"])
            if sid not in latest:
                latest[sid] = float(row["value"])

        vix = latest.get("VIXCLS", 0.0)
        spread = latest.get("T10Y2Y", 0.0)
        direction = 1.0 if spread > 0 else 0.0
        fed = latest.get("DFF", 0.0)
        cpi = latest.get("CPIAUCSL", 0.0)
        unemp = latest.get("UNRATE", 0.0)
        return np.array([vix, spread, direction, fed, cpi, unemp])
    except Exception:
        log.warning("macro_fetch_failed", date=date_str)
        return np.zeros(6)


def _get_hmm_probs(conn: Any, environment: str, date_str: str) -> np.ndarray:
    """Fetch HMM regime probabilities as (2,) array from hmm_state_history.

    Returns [0.5, 0.5] when no state history exists yet.

    Args:
        conn: PostgreSQL connection.
        environment: "equity" or "crypto".
        date_str: ISO date/datetime string used as upper bound.

    Returns:
        (2,) float64 array [p_bull, p_bear].
    """
    try:
        row = fetchdf(
            conn.execute(
                """
            SELECT p_bull, p_bear FROM hmm_state_history
            WHERE environment = %s AND date <= CAST(%s AS DATE)
            ORDER BY date DESC
            LIMIT 1
            """,
                [environment, date_str],
            )
        )

        if row.empty:
            return np.array([0.5, 0.5])
        return np.array([float(row["p_bull"].iloc[0]), float(row["p_bear"].iloc[0])])
    except Exception:
        return np.array([0.5, 0.5])


def _load_features_prices(
    conn: Any,
    env_name: str,
    config: SwingRLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and prices from PostgreSQL for the given environment.

    Delegates to the canonical implementation in swingrl.training.data_loader.

    Args:
        conn: Active PostgreSQL connection.
        env_name: Environment name ("equity" or "crypto").
        config: Validated SwingRLConfig for symbol lists and sentiment flag.

    Returns:
        Tuple of (features_array, prices_array) both float32.

    Raises:
        DataError: If no data found in the feature table.
    """
    from swingrl.training.data_loader import load_features_prices  # noqa: PLC0415

    return load_features_prices(conn, env_name, config)


def _load_equity(
    conn: Any,
    config: SwingRLConfig,
    assembler: ObservationAssembler,
) -> tuple[np.ndarray, np.ndarray]:
    """Load equity features and prices from PostgreSQL.

    Groups features_equity rows by date, calls assembler.assemble_equity() per
    timestep with macro/HMM context, and aligns with OHLCV close prices via
    INNER JOIN on date.

    Args:
        conn: PostgreSQL connection.
        config: SwingRLConfig with equity symbol list.
        assembler: ObservationAssembler initialized from config.

    Returns:
        Tuple of ((N, 164) features, (N, n_symbols) prices) float32.
    """
    equity_symbols = sorted(config.equity.symbols)
    per_asset_size = 15  # EQUITY_PER_ASSET_BASE

    # Load all feature rows INNER JOINed with OHLCV dates to ensure alignment
    feat_df = fetchdf(
        conn.execute(
            """
        SELECT
            f.date,
            f.symbol,
            {feat_cols}
        FROM features_equity f
        INNER JOIN (
            SELECT DISTINCT date FROM ohlcv_daily
        ) o ON f.date = o.date
        ORDER BY f.date, f.symbol
        """.format(feat_cols=", ".join(f"f.{c}" for c in _EQUITY_FEATURE_COLS))  # nosec B608
        )
    )

    if feat_df.empty:
        msg = "No data found in features_equity table"
        raise RuntimeError(msg)

    # Load close prices — pivot to (dates x symbols)
    prices_df = fetchdf(
        conn.execute(
            """
        SELECT date, symbol, close
        FROM ohlcv_daily
        WHERE symbol IN ({sym_list})
        ORDER BY date, symbol
        """.format(sym_list=", ".join(f"'{s}'" for s in equity_symbols))  # nosec B608
        )
    )

    # Build set of dates present in features (already filtered by INNER JOIN)
    feat_dates = sorted(feat_df["date"].unique())

    # Bulk-load macro and HMM for all dates to avoid N+1 queries
    macro_map: dict[str, np.ndarray] = {}
    hmm_map: dict[str, np.ndarray] = {}
    for d in feat_dates:
        date_str = str(d)[:10]
        macro_map[date_str] = _get_macro_array(conn, date_str)
        hmm_map[date_str] = _get_hmm_probs(conn, "equity", date_str)

    # Pivot prices to DataFrame indexed by date for fast lookup
    prices_pivot = prices_df.pivot_table(index="date", columns="symbol", values="close")
    # Ensure all expected symbols are present; fill missing with zeros
    for sym in equity_symbols:
        if sym not in prices_pivot.columns:
            prices_pivot[sym] = 0.0
    prices_pivot = prices_pivot[equity_symbols]  # enforce alpha-sorted column order

    # Assemble one observation vector per date
    obs_rows: list[np.ndarray] = []
    price_rows: list[np.ndarray] = []

    for date_val in feat_dates:
        date_str = str(date_val)[:10]
        sym_group = feat_df[feat_df["date"] == date_val]

        # Build per_asset dict — fill missing symbols with zeros
        per_asset: dict[str, np.ndarray] = {}
        for sym in equity_symbols:
            sym_row = sym_group[sym_group["symbol"] == sym]
            if sym_row.empty:
                per_asset[sym] = np.zeros(per_asset_size)
            else:
                vals = sym_row[_EQUITY_FEATURE_COLS].values[0]
                per_asset[sym] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        macro = macro_map[date_str]
        hmm = hmm_map[date_str]

        obs = assembler.assemble_equity(
            per_asset_features=per_asset,
            macro=macro,
            hmm_probs=hmm,
            turbulence=0.0,
            portfolio_state=None,
        )
        obs_rows.append(obs)

        # Extract prices for this date
        if date_val in prices_pivot.index:
            price_row = prices_pivot.loc[date_val].values.astype(np.float32)
        else:
            price_row = np.ones(len(equity_symbols), dtype=np.float32)
        price_rows.append(price_row)

    features = np.array(obs_rows, dtype=np.float32)
    prices = np.array(price_rows, dtype=np.float32)

    log.info(
        "data_loaded",
        env_name="equity",
        rows=len(feat_dates),
        obs_dim=features.shape[1],
        n_symbols=len(equity_symbols),
    )

    return features, prices


def _load_crypto(
    conn: Any,
    config: SwingRLConfig,
    assembler: ObservationAssembler,
) -> tuple[np.ndarray, np.ndarray]:
    """Load crypto features and prices from PostgreSQL.

    Groups features_crypto rows by datetime, calls assembler.assemble_crypto() per
    timestep with macro/HMM context, and aligns with ohlcv_4h close prices via
    INNER JOIN on datetime.

    Args:
        conn: PostgreSQL connection.
        config: SwingRLConfig with crypto symbol list.
        assembler: ObservationAssembler initialized from config.

    Returns:
        Tuple of ((N, 47) features, (N, n_symbols) prices) float32.
    """
    crypto_symbols = sorted(config.crypto.symbols)
    per_asset_size = 13  # CRYPTO_PER_ASSET

    # Load all feature rows INNER JOINed with ohlcv_4h datetimes
    feat_df = fetchdf(
        conn.execute(
            """
        SELECT
            f.datetime,
            f.symbol,
            {feat_cols}
        FROM features_crypto f
        INNER JOIN (
            SELECT DISTINCT datetime FROM ohlcv_4h
        ) o ON f.datetime = o.datetime
        ORDER BY f.datetime, f.symbol
        """.format(feat_cols=", ".join(f"f.{c}" for c in _CRYPTO_FEATURE_COLS))  # nosec B608
        )
    )

    if feat_df.empty:
        msg = "No data found in features_crypto table"
        raise RuntimeError(msg)

    # Load close prices — pivot to (datetimes x symbols)
    prices_df = fetchdf(
        conn.execute(
            """
        SELECT datetime, symbol, close
        FROM ohlcv_4h
        WHERE symbol IN ({sym_list})
        ORDER BY datetime, symbol
        """.format(sym_list=", ".join(f"'{s}'" for s in crypto_symbols))  # nosec B608
        )
    )

    feat_datetimes = sorted(feat_df["datetime"].unique())

    # Bulk-load macro and HMM for all timestamps
    macro_map: dict[str, np.ndarray] = {}
    hmm_map: dict[str, np.ndarray] = {}
    for dt in feat_datetimes:
        date_str = str(dt)[:10]
        macro_map[str(dt)] = _get_macro_array(conn, date_str)
        hmm_map[str(dt)] = _get_hmm_probs(conn, "crypto", date_str)

    # Pivot prices
    prices_pivot = prices_df.pivot_table(index="datetime", columns="symbol", values="close")
    for sym in crypto_symbols:
        if sym not in prices_pivot.columns:
            prices_pivot[sym] = 0.0
    prices_pivot = prices_pivot[crypto_symbols]

    obs_rows: list[np.ndarray] = []
    price_rows: list[np.ndarray] = []

    for dt_val in feat_datetimes:
        dt_str = str(dt_val)
        sym_group = feat_df[feat_df["datetime"] == dt_val]

        per_asset: dict[str, np.ndarray] = {}
        for sym in crypto_symbols:
            sym_row = sym_group[sym_group["symbol"] == sym]
            if sym_row.empty:
                per_asset[sym] = np.zeros(per_asset_size)
            else:
                vals = sym_row[_CRYPTO_FEATURE_COLS].values[0]
                per_asset[sym] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        macro = macro_map[dt_str]
        hmm = hmm_map[dt_str]

        obs = assembler.assemble_crypto(
            per_asset_features=per_asset,
            macro=macro,
            hmm_probs=hmm,
            turbulence=0.0,
            overnight_context=0.0,
            portfolio_state=None,
        )
        obs_rows.append(obs)

        if dt_val in prices_pivot.index:
            price_row = prices_pivot.loc[dt_val].values.astype(np.float32)
        else:
            price_row = np.ones(len(crypto_symbols), dtype=np.float32)
        price_rows.append(price_row)

    features = np.array(obs_rows, dtype=np.float32)
    prices = np.array(price_rows, dtype=np.float32)

    log.info(
        "data_loaded",
        env_name="crypto",
        rows=len(feat_datetimes),
        obs_dim=features.shape[1],
        n_symbols=len(crypto_symbols),
    )

    return features, prices


def _write_model_metadata(
    conn: Any,
    env_name: str,
    algo_name: str,
    model_path: Path,
    vec_normalize_path: Path,
    total_timesteps: int,
    converged_at: int | None,
    validation_sharpe: float | None,
    ensemble_weight: float | None,
) -> None:
    """Write model metadata row to PostgreSQL.

    Args:
        conn: Active PostgreSQL connection.
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
        INSERT INTO model_metadata (
            model_id, environment, algorithm, version,
            training_start_date, training_end_date,
            total_timesteps, converged_at_step,
            validation_sharpe, ensemble_weight,
            model_path, vec_normalize_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_id) DO UPDATE SET
            environment = EXCLUDED.environment,
            algorithm = EXCLUDED.algorithm,
            version = EXCLUDED.version,
            training_start_date = EXCLUDED.training_start_date,
            training_end_date = EXCLUDED.training_end_date,
            total_timesteps = EXCLUDED.total_timesteps,
            converged_at_step = EXCLUDED.converged_at_step,
            validation_sharpe = EXCLUDED.validation_sharpe,
            ensemble_weight = EXCLUDED.ensemble_weight,
            model_path = EXCLUDED.model_path,
            vec_normalize_path = EXCLUDED.vec_normalize_path
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
    dry_run: bool = args.dry_run

    log.info(
        "training_pipeline_started",
        env_name=env_name,
        algos=algos,
        total_timesteps=total_timesteps,
        dry_run=dry_run,
    )

    start_time = time.monotonic()

    # Connect to PostgreSQL for data loading and metadata writes
    database_url = config.system.database_url
    conn = psycopg.connect(database_url, row_factory=dict_row)

    try:
        # Load features and prices using ObservationAssembler
        features, prices = _load_features_prices(conn, env_name, config)

        log.info(
            "features_validated",
            env_name=env_name,
            features_shape=list(features.shape),
            prices_shape=list(prices.shape),
        )

        if dry_run:
            log.info(
                "dry_run_complete",
                env_name=env_name,
                obs_dim=features.shape[1],
                n_timesteps=features.shape[0],
            )
            return 0

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

        # Write model metadata to PostgreSQL
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
