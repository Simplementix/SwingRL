"""BinanceIngestor — crypto 4H OHLCV data ingestion from Binance.US.

Fetches BTC/ETH 4-hour klines from the Binance.US REST API with rate limit
monitoring, pagination, retry logic, and historical backfill support from
Binance public archives (data.binance.vision).

Usage as CLI:
    python -m swingrl.data.binance --symbols BTCUSDT ETHUSDT
    python -m swingrl.data.binance --backfill --symbols BTCUSDT
    python -m swingrl.data.binance --days 30
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import structlog

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import BaseIngestor
from swingrl.data.parquet_store import ParquetStore
from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

# --- Module-level constants ---

BINANCE_US_BASE_URL: str = "https://api.binance.us"
KLINES_ENDPOINT: str = "/api/v3/klines"
WEIGHT_LIMIT: int = 1200
WEIGHT_THRESHOLD: float = 0.80
MAX_KLINES_PER_REQUEST: int = 1000
FOUR_HOURS_MS: int = 4 * 3600 * 1000

# Binance public archive base URL
ARCHIVE_BASE_URL: str = "https://data.binance.vision/data/spot/monthly/klines"

# Default API start date (Binance.US launch ~Sep 2019, but use Jan 2019 for margin)
DEFAULT_API_START: str = "2019-01-01"

# Stitch point: Binance.US launched Sep 2019; use this as archive/API boundary
STITCH_DATE: str = "2019-09-01"

# Stitch validation: maximum acceptable price deviation between archive and API data
STITCH_MAX_DEVIATION: float = 0.005  # 0.5%

# Retry configuration
MAX_RETRIES: int = 3

# Klines response column names (positions 0-11)
_KLINES_COLUMNS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "close_time",
    "volume_quote",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

# Microsecond timestamp threshold: values > this are in microseconds
_MICROSECOND_THRESHOLD: int = 2_000_000_000_000


class BinanceIngestor(BaseIngestor):
    """Ingest crypto 4H OHLCV bars from Binance.US.

    Implements the BaseIngestor protocol with rate-limited API fetching,
    pagination for large date ranges, and historical backfill from
    Binance public archives.

    Args:
        config: Validated SwingRLConfig instance.
    """

    _environment = "crypto"
    _duckdb_table = "ohlcv_4h"

    def __init__(self, config: SwingRLConfig) -> None:
        super().__init__(config)
        self._validator = DataValidator(source="crypto")
        self._store = ParquetStore()
        self._session = requests.Session()

    def close(self) -> None:
        """Close the HTTP session to prevent resource leaks."""
        if hasattr(self, "_session") and self._session:
            self._session.close()

    def __enter__(self) -> BinanceIngestor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
    ) -> list[list[str | int]]:
        """Fetch up to 1000 klines from Binance.US API.

        Monitors the X-MBX-USED-WEIGHT-1M header and sleeps when usage
        exceeds 80% of the 1200/minute limit.

        Args:
            symbol: Trading pair (e.g. BTCUSDT).
            interval: Kline interval (e.g. "4h").
            start_ms: Start time in milliseconds.
            end_ms: End time in milliseconds (optional).

        Returns:
            Raw klines as list of 12-element lists.

        Raises:
            DataError: After MAX_RETRIES failed attempts.
        """
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.get(
                    BINANCE_US_BASE_URL + KLINES_ENDPOINT,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()

                # Monitor rate limit
                used_weight = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
                if used_weight > WEIGHT_LIMIT * WEIGHT_THRESHOLD:
                    log.warning(
                        "rate_limit_throttle",
                        used_weight=used_weight,
                        threshold=int(WEIGHT_LIMIT * WEIGHT_THRESHOLD),
                    )
                    time.sleep(60)

                return resp.json()  # type: ignore[no-any-return]
            except requests.RequestException as e:
                last_error = e
                backoff = 2 ** (attempt + 1)
                log.warning(
                    "klines_request_failed",
                    symbol=symbol,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                    error=str(e),
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)

        msg = f"API request failed after {MAX_RETRIES} attempts for {symbol}"
        log.error("klines_exhausted_retries", symbol=symbol, error=str(last_error))
        raise DataError(msg)

    def _parse_klines(self, raw: list[list[str | int]]) -> pd.DataFrame:
        """Parse raw Binance klines response into a clean OHLCV DataFrame.

        Volume uses quote_asset_volume (position [7]) which is already in USD.

        Args:
            raw: List of 12-element kline arrays from the API.

        Returns:
            DataFrame with open/high/low/close/volume columns and UTC DatetimeIndex.
        """
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=_KLINES_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        df = df.set_index("timestamp")

        for col in ("open", "high", "low", "close", "volume_quote"):
            df[col] = df[col].astype(float)

        df["volume"] = df["volume_quote"]
        return df[["open", "high", "low", "close", "volume"]]

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch 4H OHLCV bars for a crypto symbol.

        If since is None, attempts incremental update from existing Parquet.
        If since is an ISO date string, fetches from that date.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            since: ISO date string for start date, or None for incremental.

        Returns:
            DataFrame with OHLCV columns and UTC DatetimeIndex.
        """
        if since is None:
            # Try incremental from existing parquet
            parquet_path = self._parquet_path(symbol)
            existing = self._store.read(parquet_path)
            if not existing.empty:
                max_ts = existing.index.max()
                if max_ts.tzinfo is None:
                    max_ts = max_ts.tz_localize("UTC")
                start_ms = int(max_ts.timestamp() * 1000) + FOUR_HOURS_MS
                log.info(
                    "incremental_fetch",
                    symbol=symbol,
                    from_ts=str(max_ts),
                )
            else:
                start_ms = int(pd.Timestamp(DEFAULT_API_START, tz="UTC").timestamp() * 1000)
        else:
            start_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        all_klines: list[list[str | int]] = []
        current_start = start_ms

        while current_start < now_ms:
            batch = self._fetch_klines(symbol, "4h", current_start, now_ms)
            if not batch:
                break
            all_klines.extend(batch)
            # Advance past the last bar received
            last_open_time = int(batch[-1][0])
            current_start = last_open_time + FOUR_HOURS_MS

        return self._parse_klines(all_klines)

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        *,
        skip_staleness: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate crypto OHLCV data using the 12-step DataValidator.

        Args:
            df: Raw OHLCV DataFrame.
            symbol: Trading pair for logging context.
            skip_staleness: If True, skip staleness check during backfill.

        Returns:
            Tuple of (clean_df, quarantine_df).
        """
        clean, quarantine = self._validator.validate_rows(df, symbol)
        if not clean.empty:
            clean = self._validator.validate_batch(clean, symbol, skip_staleness=skip_staleness)
        return clean, quarantine

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert DataFrame into crypto Parquet storage.

        Args:
            df: Validated OHLCV DataFrame.
            symbol: Trading pair used to determine file path.

        Returns:
            Path to the written Parquet file.
        """
        path = self._parquet_path(symbol)
        self._store.upsert(path, df)
        return path

    def _parquet_path(self, symbol: str) -> Path:
        """Build the Parquet file path for a crypto symbol.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").

        Returns:
            Path like data/crypto/BTCUSDT_4h.parquet.
        """
        return self._data_dir / "crypto" / f"{symbol}_4h.parquet"

    def _parse_archive_csv(self, data: bytes | str | io.BytesIO) -> pd.DataFrame:
        """Parse a Binance archive CSV with automatic timestamp unit detection.

        Detects millisecond vs microsecond timestamps based on magnitude.
        Volume normalization: uses quote_asset_volume if > 0, else volume_base * close.

        Args:
            data: CSV content as bytes, string path, or BytesIO buffer.

        Returns:
            Standard OHLCV DataFrame with UTC DatetimeIndex.
        """
        if isinstance(data, bytes):
            buf = io.BytesIO(data)
        elif isinstance(data, str):
            buf = data  # type: ignore[assignment]
        else:
            buf = data

        df = pd.read_csv(buf, header=None, names=_KLINES_COLUMNS)

        # Detect timestamp unit per row and convert
        timestamps = []
        for ts_raw in df["open_time"].astype(int):
            if ts_raw > _MICROSECOND_THRESHOLD:
                timestamps.append(pd.Timestamp(ts_raw, unit="us", tz="UTC"))
            else:
                timestamps.append(pd.Timestamp(ts_raw, unit="ms", tz="UTC"))

        df["timestamp"] = timestamps
        df = df.set_index("timestamp")

        for col in ("open", "high", "low", "close", "volume_base", "volume_quote"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Volume normalization: quote_asset_volume if > 0, else base * close
        df["volume"] = df["volume_quote"].where(
            df["volume_quote"] > 0,
            df["volume_base"] * df["close"],
        )

        return df[["open", "high", "low", "close", "volume"]]

    def _validate_stitch(
        self,
        archive_df: pd.DataFrame,
        api_df: pd.DataFrame,
        overlap_days: int = 30,
    ) -> bool:
        """Validate price consistency at the archive/API stitch point.

        Compares close prices in the overlap window. Logs discrepancies > 0.5%.

        Args:
            archive_df: Data from Binance public archives.
            api_df: Data from Binance.US API.
            overlap_days: Number of days in the overlap window.

        Returns:
            True if stitch is acceptable (all deviations < 0.5%), False otherwise.
        """
        if archive_df.empty or api_df.empty:
            log.warning("stitch_validation_skipped", reason="empty dataframe")
            return True

        # Find common timestamps
        common_idx = archive_df.index.intersection(api_df.index)
        if common_idx.empty:
            log.warning("stitch_validation_no_overlap", reason="no common timestamps")
            return True

        archive_close = archive_df.loc[common_idx, "close"]
        api_close = api_df.loc[common_idx, "close"]

        deviation = ((archive_close - api_close) / api_close).abs()
        bad_mask = deviation > STITCH_MAX_DEVIATION
        acceptable = not bad_mask.any()

        if not acceptable:
            bad_count = bad_mask.sum()
            max_dev = deviation.max()
            log.warning(
                "stitch_price_deviation",
                bad_bars=int(bad_count),
                max_deviation=float(max_dev),
                threshold=STITCH_MAX_DEVIATION,
                first_bad_ts=str(deviation[bad_mask].index[0]),
            )
        else:
            log.info(
                "stitch_validation_passed",
                overlap_bars=len(common_idx),
                max_deviation=float(deviation.max()) if len(deviation) > 0 else 0.0,
            )

        return acceptable

    def backfill(
        self,
        symbol: str,
        start_date: str = "2017-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Full historical backfill combining archive and API data.

        Downloads archive CSVs from data.binance.vision for the early period,
        fetches API data from the stitch point onward, validates the overlap,
        and combines into a single continuous DataFrame.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            start_date: Start date for archive data (ISO format).
            end_date: End date (ISO format), defaults to now.

        Returns:
            Combined OHLCV DataFrame covering the full date range.
        """
        if end_date is None:
            end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        stitch_ts = pd.Timestamp(STITCH_DATE, tz="UTC")
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC")

        log.info(
            "backfill_start",
            symbol=symbol,
            start=start_date,
            end=end_date,
            stitch=STITCH_DATE,
        )

        # Phase 1: Download archive data (start -> stitch + overlap)
        overlap_end = stitch_ts + pd.Timedelta(days=30)
        archive_df = self._download_archives(symbol, start_ts, overlap_end)

        # Phase 2: Fetch API data from (stitch - overlap) to end
        overlap_start = stitch_ts - pd.Timedelta(days=30)
        api_start_ms = int(overlap_start.timestamp() * 1000)
        now_ms = int(end_ts.timestamp() * 1000)

        all_klines: list[list[str | int]] = []
        current_start = api_start_ms
        while current_start < now_ms:
            batch = self._fetch_klines(symbol, "4h", current_start, now_ms)
            if not batch:
                break
            all_klines.extend(batch)
            last_open_time = int(batch[-1][0])
            current_start = last_open_time + FOUR_HOURS_MS

        api_df = self._parse_klines(all_klines)

        # Phase 3: Validate stitch
        self._validate_stitch(archive_df, api_df)

        # Phase 4: Combine — archive up to stitch, API from stitch onward
        if not archive_df.empty and not api_df.empty:
            archive_portion = archive_df[archive_df.index < stitch_ts]
            api_portion = api_df[api_df.index >= stitch_ts]
            combined = pd.concat([archive_portion, api_portion])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        elif not archive_df.empty:
            combined = archive_df
        else:
            combined = api_df

        log.info("backfill_complete", symbol=symbol, total_bars=len(combined))

        # Validate and store — skip staleness during backfill since API data
        # may lag by hours and the check is meant for incremental freshness.
        clean, quarantine = self.validate(combined, symbol, skip_staleness=True)
        if not quarantine.empty:
            self._store_quarantine(quarantine, symbol)
        if not clean.empty:
            self.store(clean, symbol)
            self._sync_to_db(clean, symbol)

        return combined

    def _download_archives(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Download and parse monthly archive CSVs from data.binance.vision.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            start: Start timestamp.
            end: End timestamp.

        Returns:
            Combined archive DataFrame.
        """
        all_dfs: list[pd.DataFrame] = []
        # Drop timezone before converting to Period to avoid UserWarning
        current = start.tz_localize(None).to_period("M")
        end_period = end.tz_localize(None).to_period("M")

        while current <= end_period:
            year = current.year
            month = str(current.month).zfill(2)
            url = f"{ARCHIVE_BASE_URL}/{symbol}/4h/{symbol}-4h-{year}-{month}.zip"
            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    csv_data = zf.read(csv_name)
                    df = self._parse_archive_csv(csv_data)
                    all_dfs.append(df)
                    log.info(
                        "archive_downloaded",
                        symbol=symbol,
                        period=f"{year}-{month}",
                        rows=len(df),
                    )
            except requests.RequestException as e:
                log.warning(
                    "archive_download_failed",
                    symbol=symbol,
                    period=f"{year}-{month}",
                    error=str(e),
                )
            current = current + 1

        if not all_dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        combined = pd.concat(all_dfs)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def run_all(self, symbols: list[str], since: str | None = None) -> list[str]:
        """Run ingestion for multiple symbols.

        Args:
            symbols: List of trading pairs.
            since: ISO date string or None for incremental.

        Returns:
            List of symbols that failed.
        """
        failed: list[str] = []
        for symbol in symbols:
            try:
                self.run(symbol, since)
            except Exception as e:
                log.error("symbol_failed", symbol=symbol, error=str(e))
                failed.append(symbol)
        return failed


def main() -> int:
    """CLI entry point for Binance crypto data ingestion."""
    parser = argparse.ArgumentParser(description="Binance.US crypto data ingestor")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols from config (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to fetch from today",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Full historical backfill (2017-present)",
    )
    args = parser.parse_args()

    config = load_config("config/swingrl.yaml")
    symbols = args.symbols or config.crypto.symbols

    with BinanceIngestor(config) as ingestor:
        if args.backfill:
            failed: list[str] = []
            for symbol in symbols:
                try:
                    ingestor.backfill(symbol)
                except Exception as e:
                    log.error("backfill_failed", symbol=symbol, error=str(e))
                    failed.append(symbol)
            return 1 if failed else 0

        since: str | None = None
        if args.days is not None:
            since = (datetime.now(UTC) - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")

        failed_symbols = ingestor.run_all(symbols, since)
        return 1 if failed_symbols else 0


if __name__ == "__main__":
    sys.exit(main())
