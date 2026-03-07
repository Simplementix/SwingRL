"""CLI entry point for feature computation.

Computes technical indicators, normalizes, and stores features to DuckDB
for equity and/or crypto environments.

Usage:
    python scripts/compute_features.py --environment equity
    python scripts/compute_features.py --environment both --start 2023-01-01 --end 2023-12-31
    python scripts/compute_features.py --environment crypto --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import structlog

from swingrl.config.schema import load_config
from swingrl.features.pipeline import FeaturePipeline
from swingrl.features.schema import init_feature_schema
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Compute and store SwingRL features to DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/compute_features.py --environment equity
  python scripts/compute_features.py --environment both
  python scripts/compute_features.py --environment crypto --start 2024-01-01 --end 2024-06-30
  python scripts/compute_features.py --environment equity --symbols SPY QQQ
""",
    )

    parser.add_argument(
        "--environment",
        choices=["equity", "crypto", "both"],
        default="both",
        help="Which environment(s) to compute features for (default: both).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbol list (space-separated). Uses config defaults if not set.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    parser.add_argument(
        "--check-fundamentals",
        action="store_true",
        help="Test yfinance availability and exit.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run feature computation pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load config
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    if args.check_fundamentals:
        return _check_fundamentals()

    # Connect to DuckDB
    db_path = Path(config.system.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    try:
        # Initialize feature schema
        init_feature_schema(conn)

        pipeline = FeaturePipeline(config, conn)
        start_time = time.monotonic()

        if args.environment in ("equity", "both"):
            log.info("computing_equity_features", symbols=args.symbols)
            result = pipeline.compute_equity(symbols=args.symbols, start=args.start, end=args.end)
            log.info("equity_features_done", rows=len(result))

        if args.environment in ("crypto", "both"):
            log.info("computing_crypto_features", symbols=args.symbols)
            result = pipeline.compute_crypto(symbols=args.symbols, start=args.start, end=args.end)
            log.info("crypto_features_done", rows=len(result))

        elapsed = time.monotonic() - start_time
        log.info("feature_computation_complete", elapsed_seconds=round(elapsed, 2))

    finally:
        conn.close()

    return 0


def _check_fundamentals() -> int:
    """Test yfinance availability."""
    try:
        import yfinance

        ticker = yfinance.Ticker("SPY")
        info = ticker.info
        pe = info.get("trailingPE", "N/A")
        log.info("yfinance_check_passed", symbol="SPY", trailing_pe=pe)
        return 0
    except Exception as e:
        log.error("yfinance_check_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
