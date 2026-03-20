"""Single-command data ingestion orchestrator for SwingRL.

Wires AlpacaIngestor, BinanceIngestor, FREDIngestor, FeaturePipeline, and
the verification gate into a fail-fast sequential pipeline.

Usage:
    python -m swingrl.data.ingest_all --backfill
    python -m swingrl.data.ingest_all          # incremental (default)
    python -m swingrl.data.ingest_all --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.alpaca import AlpacaIngestor
from swingrl.data.binance import BinanceIngestor
from swingrl.data.db import DatabaseManager
from swingrl.data.fred import FREDIngestor
from swingrl.data.gap_fill import detect_and_fill_crypto_gaps
from swingrl.data.verification import (
    VerificationResult,
    print_summary,
    run_verification,
    write_report,
)
from swingrl.features.pipeline import FeaturePipeline
from swingrl.utils.exceptions import DataError
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

# JSON verification report path (relative to repo root / Docker CWD)
_VERIFICATION_PATH: Path = Path("data/verification.json")


# ---------------------------------------------------------------------------
# Row counting helper
# ---------------------------------------------------------------------------


def _count_rows(db: DatabaseManager, table: str) -> int:
    """Count total rows in the given DuckDB table.

    Args:
        db: Initialized DatabaseManager instance.
        table: Table name to count (e.g. "ohlcv_daily", "ohlcv_4h").

    Returns:
        Row count as integer.
    """
    with db.duckdb() as cursor:
        result = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608  # nosec B608
        return int(result[0]) if result else 0


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def run_equity(config: SwingRLConfig, backfill: bool) -> int:
    """Ingest equity OHLCV data for all configured symbols via AlpacaIngestor.

    Counts rows in ohlcv_daily before and after ingestion, returning the delta.
    Raises DataError if any symbol fails to ingest.

    Args:
        config: Validated SwingRLConfig with equity.symbols.
        backfill: If True, fetch full history (since=None). If False, incremental.

    Returns:
        Integer row delta (rows added this run).

    Raises:
        DataError: If AlpacaIngestor.run_all returns any failed symbols.
    """
    db = DatabaseManager(config)
    rows_before = _count_rows(db, "ohlcv_daily")

    ingestor = AlpacaIngestor(config)
    since: str | None = None if backfill else "incremental"
    failed = ingestor.run_all(config.equity.symbols, since=since)

    if failed:
        log.error("equity_ingestion_failed", failed_symbols=failed)
        raise DataError(f"Equity ingestion failed for symbols: {failed}")

    rows_after = _count_rows(db, "ohlcv_daily")
    delta = max(0, rows_after - rows_before)
    log.info("equity_ingestion_complete", rows_added=delta)
    return delta


def run_crypto(config: SwingRLConfig, backfill: bool) -> int:
    """Ingest crypto 4H OHLCV data for all configured symbols via BinanceIngestor.

    When backfill=True, calls backfill(symbol) per symbol for full history
    with archive stitching. When backfill=False, calls run_all for incremental.

    Args:
        config: Validated SwingRLConfig with crypto.symbols.
        backfill: If True, use backfill(). If False, use run_all() for incremental.

    Returns:
        Integer row delta (rows added this run).

    Raises:
        DataError: If run_all returns any failed symbols (incremental mode only).
    """
    db = DatabaseManager(config)
    rows_before = _count_rows(db, "ohlcv_4h")

    with BinanceIngestor(config) as ingestor:
        if backfill:
            for symbol in config.crypto.symbols:
                ingestor.backfill(symbol)
        else:
            failed = ingestor.run_all(config.crypto.symbols, since=None)
            if failed:
                log.error("crypto_ingestion_failed", failed_symbols=failed)
                raise DataError(f"Crypto ingestion failed for symbols: {failed}")

    rows_after = _count_rows(db, "ohlcv_4h")
    delta = max(0, rows_after - rows_before)
    log.info("crypto_ingestion_complete", rows_added=delta)
    return delta


def run_macro(config: SwingRLConfig, backfill: bool) -> int:
    """Ingest FRED macro data for all configured series via FREDIngestor.

    Args:
        config: Validated SwingRLConfig.
        backfill: If True, perform full historical backfill.

    Returns:
        Integer row delta (rows added this run).

    Raises:
        DataError: If FREDIngestor.run_all returns any failed series.
    """
    db = DatabaseManager(config)
    rows_before = _count_rows(db, "macro_features")

    ingestor = FREDIngestor(config)
    failed = ingestor.run_all(backfill=backfill)

    if failed:
        log.error("macro_ingestion_failed", failed_series=failed)
        raise DataError(f"Macro ingestion failed for series: {failed}")

    rows_after = _count_rows(db, "macro_features")
    delta = max(0, rows_after - rows_before)
    log.info("macro_ingestion_complete", rows_added=delta)
    return delta


def resolve_crypto_gaps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Forward-fill short gaps (<=2 consecutive NaN bars) in crypto OHLCV data.

    A gap is defined as one or more consecutive NaN rows. Gaps of 1-2 bars are
    forward-filled from the last valid value. Gaps >2 bars raise DataError since
    they indicate significant data quality issues.

    Args:
        df: DataFrame with crypto OHLCV data; may contain NaN rows.
        symbol: Crypto symbol for error logging.

    Returns:
        DataFrame with short gaps forward-filled.

    Raises:
        DataError: If any gap of >2 consecutive NaN bars remains after forward-fill.
    """
    filled = df.ffill(limit=2)
    remaining = int(filled.isna().any(axis=1).sum())
    if remaining > 0:
        log.error(
            "crypto_gap_unfillable",
            symbol=symbol,
            unfillable_rows=remaining,
        )
        raise DataError(
            f"Crypto gap in {symbol} exceeds 2 bars ({remaining} unfillable rows). "
            "Manual data repair required."
        )
    return filled


def run_features(config: SwingRLConfig) -> None:
    """Compute equity and crypto features via FeaturePipeline.

    Opens a DuckDB connection and runs both compute_equity() and
    compute_crypto() on the fresh data.

    Args:
        config: Validated SwingRLConfig.
    """
    db = DatabaseManager(config)
    with db.duckdb() as conn:
        pipeline = FeaturePipeline(config, conn)
        pipeline.compute_equity()
        pipeline.compute_crypto()
    log.info("feature_computation_complete")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(config: SwingRLConfig, backfill: bool) -> int:
    """Execute the full ingestion pipeline: equity -> crypto -> macro -> features -> verify.

    Fails fast on any ingestor error. Feature computation is skipped when no
    new rows are added. Verification always runs as the final gate.

    Args:
        config: Validated SwingRLConfig.
        backfill: If True, perform full historical backfill for all sources.

    Returns:
        0 if verification passes, 1 if verification fails.
    """
    log.info("pipeline_start", backfill=backfill)

    equity_delta = run_equity(config, backfill=backfill)
    crypto_delta = run_crypto(config, backfill=backfill)
    macro_delta = run_macro(config, backfill=backfill)

    # Detect and fill crypto gaps from alternate sources (Binance Global)
    gap_results = detect_and_fill_crypto_gaps(config)
    gaps_filled = sum(1 for g in gap_results if g.filled)
    gaps_remaining = sum(1 for g in gap_results if not g.filled)
    if gap_results:
        log.info(
            "crypto_gap_fill_summary",
            total=len(gap_results),
            filled=gaps_filled,
            remaining=gaps_remaining,
        )

    total_rows_added = equity_delta + crypto_delta + macro_delta

    if total_rows_added > 0 or gaps_filled > 0:
        log.info("running_feature_computation", total_rows_added=total_rows_added)
        run_features(config)
    else:
        log.info("skipping_feature_computation", reason="no_new_rows")

    # Verification always runs as the final gate
    result: VerificationResult = run_verification(config)
    write_report(result, _VERIFICATION_PATH)
    print_summary(result)

    if result.passed:
        log.info("pipeline_complete", status="PASSED")
        return 0
    else:
        log.error("pipeline_complete", status="FAILED")
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ingestion pipeline.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace with backfill (bool) and config (str) attributes.
    """
    parser = argparse.ArgumentParser(
        prog="swingrl.data.ingest_all",
        description="SwingRL data ingestion pipeline — equity, crypto, macro, features.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help="Perform full historical backfill (default: incremental update).",
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ingestion pipeline CLI.

    Parses args, configures logging, loads config, and runs the pipeline.
    Exits with the pipeline return code (0=pass, 1=fail).

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    args = _parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    sys.exit(run_pipeline(config, args.backfill))


if __name__ == "__main__":
    main()
