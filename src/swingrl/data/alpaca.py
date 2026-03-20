"""AlpacaIngestor — equity OHLCV data ingestion via Alpaca.

Fetches daily bars for 8 equity ETFs (SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK)
with incremental and 10-year backfill modes. Uses SIP feed for historical
backfill (full depth back to 2016) and IEX feed for incremental/trading.

Usage:
    python -m swingrl.data.alpaca --backfill
    python -m swingrl.data.alpaca --symbols SPY QQQ --days 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import structlog
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import BaseIngestor
from swingrl.data.parquet_store import ParquetStore
from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import ConfigError, DataError
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

# Backfill start date: 10 years of equity data (2016-present)
_BACKFILL_START = datetime(2016, 1, 1, tzinfo=UTC)

# Retry configuration
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2


class AlpacaIngestor(BaseIngestor):
    """Concrete ingestor for Alpaca equity daily bars.

    Uses SIP feed for historical backfill (full depth back to 2016) and
    IEX feed for incremental fetches (paper/live trading context).
    Reads ALPACA_API_KEY and ALPACA_SECRET_KEY from environment variables.

    Args:
        config: Validated SwingRLConfig instance.
    """

    _environment = "equity"
    _duckdb_table = "ohlcv_daily"

    def __init__(self, config: SwingRLConfig) -> None:
        super().__init__(config)
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key:
            msg = "ALPACA_API_KEY environment variable is empty or not set"
            log.error("alpaca_missing_api_key")
            raise ConfigError(msg)
        if not secret_key:
            msg = "ALPACA_SECRET_KEY environment variable is empty or not set"
            log.error("alpaca_missing_secret_key")
            raise ConfigError(msg)
        self._client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self._store = ParquetStore()
        self._validator = DataValidator(source="equity")

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch daily OHLCV bars for an equity symbol from Alpaca.

        Uses SIP feed for backfill (since=None) to get full historical depth
        back to 2016. Uses IEX feed for incremental fetches (paper/live trading).

        Args:
            symbol: Ticker symbol (e.g. "SPY").
            since: "incremental" to resume from last bar in Parquet,
                   ISO date string for specific start, or None for full backfill.

        Returns:
            DataFrame with open/high/low/close/volume columns and UTC DatetimeIndex.

        Raises:
            DataError: After 3 failed retry attempts.
        """
        start = self._resolve_start(symbol, since)
        end = datetime.now(UTC)

        # SIP for historical backfill (full depth back to 2016), IEX for
        # incremental/trading. Free-tier SIP blocks "recent" data, so cap
        # end at start-of-today (yesterday's close) for SIP requests.
        if since == "incremental":
            feed = DataFeed.IEX
        else:
            feed = DataFeed.SIP
            end = end.replace(hour=0, minute=0, second=0, microsecond=0)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=feed,
            adjustment=Adjustment.ALL,
        )

        log.info(
            "alpaca_fetch_start",
            symbol=symbol,
            start=start.isoformat(),
            mode="incremental" if since == "incremental" else "backfill",
            feed=feed.value,
        )

        barset = self._fetch_with_retry(request, symbol)
        df: pd.DataFrame = barset.df

        if df.empty:
            return pd.DataFrame()

        # Alpaca returns MultiIndex (symbol, timestamp) — flatten to timestamp only
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        # Keep only OHLCV columns
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        result: pd.DataFrame = df[ohlcv_cols]

        log.info("alpaca_fetch_complete", symbol=symbol, rows=len(result))
        return result

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate OHLCV data using the equity DataValidator.

        Args:
            df: Raw OHLCV DataFrame.
            symbol: Ticker symbol for logging context.

        Returns:
            Tuple of (clean_df, quarantine_df).
        """
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol)
        clean = validator.validate_batch(clean, symbol)
        return clean, quarantine

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert DataFrame into equity Parquet file.

        Args:
            df: Validated OHLCV DataFrame.
            symbol: Ticker symbol for filename.

        Returns:
            Path to the written Parquet file.
        """
        path = self._data_dir / "equity" / f"{symbol}_daily.parquet"
        self._store.upsert(path, df)
        return path

    def run_all(self, symbols: list[str], since: str | None = None) -> list[str]:
        """Run fetch-validate-store pipeline for multiple symbols.

        Continues on failure for individual symbols. Returns list of
        symbols that failed.

        Args:
            symbols: List of ticker symbols to ingest.
            since: Mode for fetch — "incremental", ISO date, or None for backfill.

        Returns:
            List of symbols that failed during ingestion.
        """
        failed: list[str] = []
        for symbol in symbols:
            try:
                self.run(symbol, since)
            except Exception as exc:
                log.error("symbol_failed", symbol=symbol, error=str(exc))
                failed.append(symbol)
        return failed

    def _resolve_start(self, symbol: str, since: str | None) -> datetime:
        """Determine the start date based on the since parameter.

        Args:
            symbol: Ticker symbol (used for incremental Parquet lookup).
            since: "incremental", ISO date string, or None.

        Returns:
            Start datetime in UTC.
        """
        if since is None:
            return _BACKFILL_START

        if since == "incremental":
            path = self._data_dir / "equity" / f"{symbol}_daily.parquet"
            existing = self._store.read(path)
            if existing.empty:
                log.info(
                    "incremental_no_existing_data",
                    symbol=symbol,
                    fallback="backfill",
                )
                return _BACKFILL_START
            max_ts = existing.index.max()
            # Start from the next day after the last bar
            if max_ts.tzinfo is None:
                max_ts = max_ts.tz_localize("UTC")
            start_ts = max_ts + timedelta(days=1)
            start_dt = datetime(start_ts.year, start_ts.month, start_ts.day, tzinfo=UTC)
            log.info(
                "incremental_start_resolved",
                symbol=symbol,
                last_bar=max_ts.isoformat(),
                start=start_dt.isoformat(),
            )
            return start_dt

        # Assume ISO date string
        return datetime.fromisoformat(since).replace(tzinfo=UTC)

    def _fetch_with_retry(self, request: StockBarsRequest, symbol: str) -> BarSet:
        """Execute API call with exponential backoff retry.

        Args:
            request: StockBarsRequest to execute.
            symbol: Symbol for error context.

        Returns:
            BarSet response from Alpaca.

        Raises:
            DataError: After _MAX_RETRIES failed attempts.
        """
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return cast(BarSet, self._client.get_stock_bars(request))
            except Exception as exc:
                last_error = exc
                backoff = _BASE_BACKOFF_SECONDS ** (attempt + 1)
                log.warning(
                    "alpaca_api_retry",
                    symbol=symbol,
                    attempt=attempt + 1,
                    max_retries=_MAX_RETRIES,
                    backoff_seconds=backoff,
                    error=str(exc),
                )
                time.sleep(backoff)

        msg = f"Alpaca API failed for {symbol} after {_MAX_RETRIES} attempts: {last_error}"
        log.error("alpaca_api_exhausted", symbol=symbol, error=str(last_error))
        raise DataError(msg)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Alpaca ingestor.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with symbols, days, and backfill flags.
    """
    parser = argparse.ArgumentParser(
        description="Alpaca equity OHLCV data ingestor",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override equity symbols (e.g. --symbols SPY QQQ)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Fetch last N days of data",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Full 10-year backfill from 2016-01-01",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the Alpaca ingestor.

    Args:
        argv: CLI argument list (defaults to sys.argv[1:]).
    """
    args = _parse_args(argv)
    config = load_config()
    configure_logging(
        json_logs=config.logging.json_logs,
        log_level=config.logging.level,
    )

    symbols = args.symbols or config.equity.symbols

    if args.backfill:
        since = None
    elif args.days:
        since = (datetime.now(UTC) - timedelta(days=args.days)).isoformat()
    else:
        since = "incremental"

    ingestor = AlpacaIngestor(config)
    failed = ingestor.run_all(symbols, since=since)

    if failed:
        log.error("alpaca_run_failures", failed_symbols=failed)
        sys.exit(1)

    log.info("alpaca_run_complete", symbols=symbols)


if __name__ == "__main__":
    main()
