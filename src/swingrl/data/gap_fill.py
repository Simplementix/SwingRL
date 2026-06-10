"""Gap detection and filling for crypto OHLCV data from alternate sources.

Detects timestamp gaps in PostgreSQL ohlcv_4h data and attempts to fill them
using Binance Global (api.binance.com) as a fallback source. Remaining
unfillable gaps are logged and stored as metadata for the training layer
to use as episode boundaries.

Usage:
    from swingrl.data.gap_fill import detect_and_fill_crypto_gaps

    remaining = detect_and_fill_crypto_gaps(config)
    # remaining: list of GapRecord for gaps that could not be filled
"""

from __future__ import annotations

import dataclasses
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import requests
import structlog

from swingrl.data.db import DatabaseManager
from swingrl.data.pg_helpers import executemany_from_df
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

# --- Constants ---

# Binance Global API (public, no auth required for klines)
BINANCE_GLOBAL_BASE_URL: str = "https://api.binance.com"
KLINES_ENDPOINT: str = "/api/v3/klines"
MAX_KLINES_PER_REQUEST: int = 1000
FOUR_HOURS_MS: int = 4 * 3600 * 1000

# Gap thresholds
CRYPTO_GAP_THRESHOLD: timedelta = timedelta(hours=24)
EQUITY_GAP_THRESHOLD: timedelta = timedelta(days=5)

# Rate limiting
MAX_RETRIES: int = 3
RATE_LIMIT_SLEEP: float = 1.0  # seconds between requests


@dataclasses.dataclass(frozen=True)
class GapRecord:
    """Represents a detected gap in OHLCV data."""

    symbol: str
    environment: str
    gap_start: datetime
    gap_end: datetime
    gap_duration: timedelta
    filled: bool
    source: str  # "binance_global", "unfillable", etc.

    @property
    def gap_hours(self) -> float:
        """Gap duration in hours."""
        return self.gap_duration.total_seconds() / 3600


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def detect_crypto_gaps(
    config: SwingRLConfig,
    threshold: timedelta = CRYPTO_GAP_THRESHOLD,
) -> list[GapRecord]:
    """Detect gaps exceeding threshold in ohlcv_4h data.

    Args:
        config: Validated SwingRLConfig.
        threshold: Minimum gap duration to report.

    Returns:
        List of GapRecord for each detected gap.
    """
    db = DatabaseManager(config)
    gaps: list[GapRecord] = []

    with db.connection() as conn:
        for symbol in config.crypto.symbols:
            rows = conn.execute(
                "SELECT datetime FROM ohlcv_4h WHERE symbol = %s ORDER BY datetime",
                [symbol],
            ).fetchall()

            if len(rows) < 2:
                continue

            for i in range(1, len(rows)):
                prev_ts = _ensure_datetime(rows[i - 1]["datetime"])
                curr_ts = _ensure_datetime(rows[i]["datetime"])
                gap = curr_ts - prev_ts

                if gap > threshold:
                    gaps.append(
                        GapRecord(
                            symbol=symbol,
                            environment="crypto",
                            gap_start=prev_ts,
                            gap_end=curr_ts,
                            gap_duration=gap,
                            filled=False,
                            source="detected",
                        )
                    )

    log.info("crypto_gaps_detected", count=len(gaps))
    return gaps


def detect_equity_gaps(
    config: SwingRLConfig,
    threshold: timedelta = EQUITY_GAP_THRESHOLD,
) -> list[GapRecord]:
    """Detect gaps exceeding threshold in ohlcv_daily data.

    Args:
        config: Validated SwingRLConfig.
        threshold: Minimum gap duration to report.

    Returns:
        List of GapRecord for each detected gap.
    """
    db = DatabaseManager(config)
    gaps: list[GapRecord] = []

    with db.connection() as conn:
        for symbol in config.equity.symbols:
            rows = conn.execute(
                "SELECT date FROM ohlcv_daily WHERE symbol = %s ORDER BY date",
                [symbol],
            ).fetchall()

            if len(rows) < 2:
                continue

            for i in range(1, len(rows)):
                prev_dt = _ensure_date(rows[i - 1]["date"])
                curr_dt = _ensure_date(rows[i]["date"])
                gap = curr_dt - prev_dt

                if gap > threshold:
                    gaps.append(
                        GapRecord(
                            symbol=symbol,
                            environment="equity",
                            gap_start=datetime.combine(prev_dt, datetime.min.time(), tzinfo=UTC),
                            gap_end=datetime.combine(curr_dt, datetime.min.time(), tzinfo=UTC),
                            gap_duration=gap,
                            filled=False,
                            source="detected",
                        )
                    )

    log.info("equity_gaps_detected", count=len(gaps))
    return gaps


# ---------------------------------------------------------------------------
# Gap filling from Binance Global
# ---------------------------------------------------------------------------


def _fetch_binance_global_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list[str | int]]:
    """Fetch klines from Binance Global public API.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.

    Returns:
        Raw klines as list of 12-element lists.

    Raises:
        DataError: After MAX_RETRIES failed attempts.
    """
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": "4h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_KLINES_PER_REQUEST,
    }

    last_error: Exception | None = None
    with requests.Session() as session:
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.get(
                    BINANCE_GLOBAL_BASE_URL + KLINES_ENDPOINT,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                time.sleep(RATE_LIMIT_SLEEP)
                return resp.json()  # type: ignore[no-any-return]
            except requests.RequestException as e:
                last_error = e
                backoff = 2 ** (attempt + 1)
                log.warning(
                    "binance_global_request_failed",
                    symbol=symbol,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                    error=str(e),
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)

    msg = f"Binance Global API failed after {MAX_RETRIES} attempts for {symbol}"
    log.error("binance_global_exhausted", symbol=symbol, error=str(last_error))
    raise DataError(msg)


def _parse_klines_to_df(raw: list[list[str | int]]) -> pd.DataFrame:
    """Parse raw Binance klines into OHLCV DataFrame.

    Args:
        raw: List of 12-element kline arrays.

    Returns:
        DataFrame with open/high/low/close/volume and UTC DatetimeIndex.
    """
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume_base",
        "close_time",
        "volume_quote",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")

    for col in ("open", "high", "low", "close", "volume_quote"):
        df[col] = df[col].astype(float)

    df["volume"] = df["volume_quote"]
    return df[["open", "high", "low", "close", "volume"]]


def fill_crypto_gap(gap: GapRecord) -> pd.DataFrame:
    """Attempt to fill a single crypto gap from Binance Global.

    Fetches 4H klines from Binance Global for the gap period.
    Paginates if the gap spans more than 1000 candles.

    Args:
        gap: GapRecord describing the gap to fill.

    Returns:
        DataFrame with OHLCV data for the gap period (may be empty if API fails).
    """
    # Fetch from gap_start + 4h (next expected bar) to gap_end
    start_ms = int(gap.gap_start.timestamp() * 1000) + FOUR_HOURS_MS
    end_ms = int(gap.gap_end.timestamp() * 1000)

    all_klines: list[list[str | int]] = []
    current_start = start_ms

    while current_start < end_ms:
        try:
            batch = _fetch_binance_global_klines(gap.symbol, current_start, end_ms)
        except DataError:
            log.warning(
                "gap_fill_api_failed",
                symbol=gap.symbol,
                gap_start=str(gap.gap_start),
            )
            break

        if not batch:
            break

        all_klines.extend(batch)
        last_open_time = int(batch[-1][0])
        current_start = last_open_time + FOUR_HOURS_MS

    return _parse_klines_to_df(all_klines)


def _insert_gap_fill_to_db(
    config: SwingRLConfig,
    df: pd.DataFrame,
    symbol: str,
) -> int:
    """Insert gap-fill data into PostgreSQL ohlcv_4h table.

    Uses INSERT ... ON CONFLICT DO NOTHING for idempotency.

    Args:
        config: Validated SwingRLConfig.
        df: Gap-fill OHLCV DataFrame with UTC DatetimeIndex.
        symbol: Trading pair.

    Returns:
        Number of rows sent to PostgreSQL.
    """
    if df.empty:
        return 0

    db = DatabaseManager(config)
    sync_df = pd.DataFrame(
        {
            "symbol": symbol,
            "datetime": df.index,
            "open": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "close": df["close"].values,
            "volume": df["volume"].values,
            "source": "binance_global",
        }
    )

    columns = ["symbol", "datetime", "open", "high", "low", "close", "volume", "source"]
    with db.connection() as conn:
        inserted = executemany_from_df(conn, "ohlcv_4h", sync_df, columns, on_conflict="DO NOTHING")

    log.info("gap_fill_inserted", symbol=symbol, rows=inserted)
    return inserted


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def detect_and_fill_crypto_gaps(config: SwingRLConfig) -> list[GapRecord]:
    """Detect crypto gaps, attempt to fill from Binance Global, return remaining.

    Args:
        config: Validated SwingRLConfig.

    Returns:
        List of GapRecord — filled gaps have filled=True, unfillable have filled=False.
    """
    gaps = detect_crypto_gaps(config)
    if not gaps:
        log.info("no_crypto_gaps_to_fill")
        return []

    results: list[GapRecord] = []
    total_filled = 0

    for gap in gaps:
        log.info(
            "attempting_gap_fill",
            symbol=gap.symbol,
            gap_start=str(gap.gap_start),
            gap_end=str(gap.gap_end),
            gap_hours=gap.gap_hours,
        )

        fill_df = fill_crypto_gap(gap)
        if fill_df.empty:
            log.warning(
                "gap_fill_empty",
                symbol=gap.symbol,
                gap_start=str(gap.gap_start),
            )
            results.append(dataclasses.replace(gap, source="unfillable"))
            continue

        inserted = _insert_gap_fill_to_db(config, fill_df, gap.symbol)
        total_filled += inserted

        if inserted > 0:
            results.append(dataclasses.replace(gap, filled=True, source="binance_global"))
            log.info(
                "gap_filled",
                symbol=gap.symbol,
                gap_start=str(gap.gap_start),
                rows_inserted=inserted,
            )
        else:
            results.append(dataclasses.replace(gap, source="unfillable"))

    # Re-detect to find any remaining gaps after fill
    remaining = detect_crypto_gaps(config)
    if remaining:
        log.warning(
            "crypto_gaps_remaining_after_fill",
            count=len(remaining),
            gaps=[f"{g.symbol}: {g.gap_hours:.0f}h at {g.gap_start}" for g in remaining],
        )
    else:
        log.info("all_crypto_gaps_filled", total_rows=total_filled)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_datetime(value: object) -> datetime:
    """Coerce a value to a timezone-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    msg = f"Cannot convert {type(value)} to datetime"
    raise TypeError(msg)


def _ensure_date(value: object) -> datetime:
    """Coerce a date-like value to a date object."""
    import datetime as dt_mod  # noqa: PLC0415

    if isinstance(value, dt_mod.date) and not isinstance(value, datetime):
        return value  # type: ignore[return-value]
    if isinstance(value, datetime):
        return value.date()  # type: ignore[return-value]
    if isinstance(value, str):
        return dt_mod.date.fromisoformat(value)  # type: ignore[return-value]
    msg = f"Cannot convert {type(value)} to date"
    raise TypeError(msg)
