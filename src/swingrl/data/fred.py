"""FREDIngestor — FRED macro data ingestor for Tier 1 series.

Fetches 5 macro series: VIXCLS, T10Y2Y, DFF (daily), CPIAUCSL, UNRATE (vintage).
Vintage series use ALFRED get_series_all_releases() to prevent look-ahead bias
from revised economic numbers.

CLI usage: python -m swingrl.data.fred --backfill
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import structlog
from fredapi import Fred

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import BaseIngestor
from swingrl.data.parquet_store import ParquetStore
from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

# Series that are rarely revised — use get_series() (no vintage tracking)
DAILY_SERIES: list[str] = ["VIXCLS", "T10Y2Y", "DFF"]

# Series with significant revisions — use get_series_all_releases() (ALFRED)
VINTAGE_SERIES: list[str] = ["CPIAUCSL", "UNRATE"]

# All Tier 1 macro series
ALL_SERIES: list[str] = DAILY_SERIES + VINTAGE_SERIES

# Default backfill start date (10-year equity training window)
_DEFAULT_START = "2016-01-01"

# Retry configuration
_MAX_RETRIES = 3


class FREDIngestor(BaseIngestor):
    """Ingest FRED macro data for equity and crypto environments.

    Supports both daily series (VIX, yield curve, Fed Funds rate) and
    vintage series with ALFRED revision tracking (CPI, unemployment).

    Args:
        config: Validated SwingRLConfig instance.
    """

    _environment = "macro"
    _duckdb_table = "macro_features"

    def __init__(self, config: SwingRLConfig) -> None:
        super().__init__(config)
        api_key = os.environ.get("FRED_API_KEY", "")
        self._fred = Fred(api_key=api_key)
        self._store = ParquetStore()
        self._validator = DataValidator(source="fred")
        self._macro_dir = self._data_dir / "macro"

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch a FRED series.

        Args:
            symbol: FRED series ID (e.g. "VIXCLS", "CPIAUCSL").
            since: "auto" for incremental from existing Parquet,
                   None for full backfill from 2016-01-01,
                   or an ISO date string.

        Returns:
            DataFrame with observation_date index and value column.
            Vintage series also include a vintage_date column.
        """
        start_date = self._resolve_start_date(symbol, since)
        log.info("fred_fetch_start", series=symbol, start_date=start_date)

        if symbol in VINTAGE_SERIES:
            return self._fetch_vintage(symbol, start_date)
        return self._fetch_daily(symbol, start_date)

    def _resolve_start_date(self, symbol: str, since: str | None) -> str:
        """Determine the start date for fetching.

        Args:
            symbol: FRED series ID.
            since: "auto" for incremental, None for backfill, or ISO date.

        Returns:
            ISO date string for observation_start.
        """
        if since == "auto":
            parquet_path = self._macro_dir / f"{symbol}.parquet"
            existing = self._store.read(parquet_path)
            if not existing.empty:
                max_date = existing.index.max()
                next_day = pd.Timestamp(max_date) + timedelta(days=1)
                return next_day.strftime("%Y-%m-%d")
        elif since is not None:
            return since
        return _DEFAULT_START

    def _fetch_daily(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Fetch a daily (non-vintage) FRED series with retry.

        Args:
            symbol: FRED series ID.
            start_date: ISO date string for observation_start.

        Returns:
            DataFrame with observation_date index and value column.
        """
        raw = self._call_with_retry(
            lambda: self._fred.get_series(symbol, observation_start=start_date),
            symbol=symbol,
        )
        df = pd.DataFrame({"value": raw.values}, index=raw.index)
        df.index.name = "observation_date"
        return df

    def _fetch_vintage(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Fetch a vintage (ALFRED) FRED series with retry.

        Args:
            symbol: FRED series ID.
            start_date: ISO date string for observation_start.

        Returns:
            DataFrame with observation_date index, value and vintage_date columns.
        """
        raw = self._call_with_retry(
            lambda: self._fred.get_series_all_releases(symbol, realtime_start=start_date),
            symbol=symbol,
        )
        # get_series_all_releases returns DataFrame with date index,
        # realtime_start, and value columns
        df = pd.DataFrame(
            {
                "value": raw["value"].values,
                "vintage_date": raw["realtime_start"].values,
            },
            index=raw.index,
        )
        df.index.name = "observation_date"
        return df

    def _call_with_retry(
        self,
        func: object,
        symbol: str,
    ) -> pd.Series | pd.DataFrame:
        """Call a function with retry and exponential backoff.

        Args:
            func: Callable to invoke.
            symbol: Series ID for error messages.

        Returns:
            Result of the callable.

        Raises:
            DataError: After all retry attempts exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return func()  # type: ignore[operator, no-any-return]
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    log.warning(
                        "fred_retry",
                        series=symbol,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    time.sleep(wait)
        msg = f"Failed to fetch {symbol} after {_MAX_RETRIES} attempts: {last_error}"
        log.error("fred_fetch_failed", series=symbol, error=str(last_error))
        raise DataError(msg)

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate FRED data with fred-specific checks.

        FRED data has only value (+ optional vintage_date), not OHLCV.
        Applies: null value check (step 1), dedup (step 8), gap detection
        (step 9), staleness (step 10). Skips OHLCV-specific steps 2-7.

        Args:
            df: Raw FRED DataFrame.
            symbol: Series ID for logging context.

        Returns:
            Tuple of (clean_df, quarantine_df).
        """
        if df.empty:
            return df, pd.DataFrame()

        # Step 1: Null value check — quarantine rows with NaN value
        null_mask = df["value"].isna()
        if null_mask.any():
            quarantine_df = df[null_mask].copy()
            quarantine_df["reason"] = "Step 1: null value"
            clean_df = df[~null_mask].copy()
            log.warning(
                "fred_null_values",
                symbol=symbol,
                quarantined=int(null_mask.sum()),
            )
        else:
            clean_df = df.copy()
            quarantine_df = pd.DataFrame()

        # Batch-level checks via DataValidator (steps 8-10: dedup, gap, stale)
        if not clean_df.empty:
            clean_df = self._validator.validate_batch(clean_df, symbol)

        return clean_df, quarantine_df

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert DataFrame into Parquet storage under data/macro/.

        Args:
            df: Validated FRED DataFrame.
            symbol: Series ID for filename.

        Returns:
            Path to the written Parquet file.
        """
        path = self._macro_dir / f"{symbol}.parquet"
        self._store.upsert(path, df)
        return path

    def run_all(
        self,
        series_ids: list[str] | None = None,
        since: str | None = None,
        backfill: bool = False,
    ) -> list[str]:
        """Orchestrate fetch/validate/store for multiple series.

        Args:
            series_ids: Series to fetch (defaults to ALL_SERIES).
            since: Start date override; "auto" for incremental.
            backfill: If True, force start from 2016-01-01.

        Returns:
            List of series IDs that failed.
        """
        if series_ids is None:
            series_ids = list(ALL_SERIES)

        failed: list[str] = []
        for series_id in series_ids:
            try:
                effective_since = None if backfill else (since or "auto")
                self.run(series_id, effective_since)
            except Exception as exc:
                log.error(
                    "fred_series_failed",
                    series=series_id,
                    error=str(exc),
                )
                failed.append(series_id)

        log.info(
            "fred_run_all_complete",
            total=len(series_ids),
            succeeded=len(series_ids) - len(failed),
            failed=len(failed),
        )
        return failed


def _run_cli(
    config: SwingRLConfig,
    series: list[str] | None = None,
    backfill: bool = False,
) -> int:
    """CLI entry point logic (testable without argparse).

    Args:
        config: Validated SwingRLConfig.
        series: Override series list.
        backfill: Force full backfill from 2016-01-01.

    Returns:
        Exit code (0 success, 1 if any series failed).
    """
    ingestor = FREDIngestor(config)
    failed = ingestor.run_all(
        series_ids=series or list(ALL_SERIES),
        backfill=backfill,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRED macro data ingestor")
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help=f"Series to fetch (default: {', '.join(ALL_SERIES)})",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Force full backfill from 2016-01-01",
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.exit(_run_cli(cfg, series=args.series, backfill=args.backfill))
