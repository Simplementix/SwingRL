"""DataValidator — 12-step data validation checklist.

Row-level checks (steps 1-7) quarantine individual bad rows.
Batch-level checks (steps 8-12) operate on the full dataset.

Accepts a source type ("equity", "crypto", "fred") to switch gap detection logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import structlog

from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Tolerance for float comparison in OHLC bounds check (step 5)
_TOLERANCE = 0.0001  # 0.01%

# Price spike threshold (step 6): >50% move in one bar
_SPIKE_THRESHOLD = 0.50

# Staleness thresholds
_EQUITY_STALE_DAYS = 4  # ~2 trading days plus weekends buffer
_CRYPTO_STALE_HOURS = 8

# Expected bar frequency by source
_CRYPTO_FREQ = timedelta(hours=4)


class DataValidator:
    """Validate OHLCV data with a 12-step checklist.

    Args:
        source: Data source type — determines gap detection logic and
            source-specific checks (e.g., zero-volume is only flagged for equity).
        db: Optional DatabaseManager for cross-source validation (Step 12).
        config: Optional SwingRLConfig for cross-source validation (Step 12).
    """

    def __init__(
        self,
        source: Literal["equity", "crypto", "fred"],
        db: DatabaseManager | None = None,
        config: SwingRLConfig | None = None,
    ) -> None:
        self._source = source
        self._db = db
        self._config = config
        if source == "equity":
            self._nyse = xcals.get_calendar("XNYS")

    def validate_rows(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply row-level validation steps 1-7.

        Returns (clean_df, quarantine_df). The quarantine_df includes a 'reason'
        column describing which check(s) failed. Never raises — bad rows are
        quarantined, not rejected.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            symbol: Symbol for logging context.

        Returns:
            Tuple of (clean rows, quarantined rows with reason column).
        """
        if df.empty:
            return df, pd.DataFrame()

        reasons: dict[int, list[str]] = {}

        def _flag(mask: pd.Series, reason: str) -> None:
            """Record reason for each True index position in mask."""
            for pos in mask[mask].index:
                idx = df.index.get_loc(pos)
                if isinstance(idx, (int, np.integer)):
                    reasons.setdefault(int(idx), []).append(reason)
                else:
                    # Handle slice or array (shouldn't happen with unique index)
                    reasons.setdefault(0, []).append(reason)

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        price_cols = ["open", "high", "low", "close"]

        # Step 1: Null check on OHLCV columns
        null_mask = df[ohlcv_cols].isna().any(axis=1)
        _flag(null_mask, "Step 1: null OHLCV value")

        # Step 2: Price > 0 for open/high/low/close
        price_le_zero = (df[price_cols] <= 0).any(axis=1) & ~df[price_cols].isna().any(axis=1)
        _flag(price_le_zero, "Step 2: price <= 0")

        # Step 3: Volume >= 0
        neg_vol = df["volume"].lt(0) & df["volume"].notna()
        _flag(neg_vol, "Step 3: negative volume")

        # Step 4: OHLC ordering — high >= low
        ohlc_order = (df["high"] < df["low"]) & df["high"].notna() & df["low"].notna()
        _flag(ohlc_order, "Step 4: high < low")

        # Step 5: O/C between H/L (with tolerance)
        if not df.empty:
            tol = df["high"].abs() * _TOLERANCE
            h_plus_tol = df["high"] + tol
            l_minus_tol = df["low"] - tol
            open_oob = ((df["open"] > h_plus_tol) | (df["open"] < l_minus_tol)) & df[
                ["open", "high", "low"]
            ].notna().all(axis=1)
            close_oob = ((df["close"] > h_plus_tol) | (df["close"] < l_minus_tol)) & df[
                ["close", "high", "low"]
            ].notna().all(axis=1)
            _flag(open_oob, "Step 5: open outside high/low bounds")
            _flag(close_oob, "Step 5: close outside high/low bounds")

        # Step 6: Price spike > 50% bar-to-bar
        if len(df) > 1:
            prev_close = df["close"].shift(1)
            pct_change = ((df["close"] - prev_close) / prev_close).abs()
            spike_mask = (pct_change > _SPIKE_THRESHOLD) & prev_close.notna()
            _flag(spike_mask, "Step 6: price spike > 50%")

        # Step 7: Zero volume on non-holiday (equity only)
        if self._source == "equity":
            zero_vol = df["volume"].eq(0) & df["volume"].notna()
            # Check that the date is a trading day
            if zero_vol.any():
                for ts in zero_vol[zero_vol].index:
                    try:
                        normalized = ts.normalize().tz_localize(None)
                        if self._nyse.is_session(normalized):
                            idx = df.index.get_loc(ts)
                            if isinstance(idx, (int, np.integer)):
                                reasons.setdefault(int(idx), []).append(
                                    "Step 7: zero volume on trading day"
                                )
                    except Exception:
                        # If calendar check fails, still flag it
                        idx = df.index.get_loc(ts)
                        if isinstance(idx, (int, np.integer)):
                            reasons.setdefault(int(idx), []).append(
                                "Step 7: zero volume (calendar check failed)"
                            )

        # Build quarantine and clean DataFrames
        bad_positions = set(reasons.keys())
        if bad_positions:
            quarantine_idx = [df.index[i] for i in sorted(bad_positions)]
            quarantine_df = df.loc[quarantine_idx].copy()
            quarantine_df["reason"] = [
                "; ".join(reasons[int(df.index.get_loc(ts))])  # type: ignore[arg-type]
                if isinstance(df.index.get_loc(ts), (int, np.integer))
                else "; ".join(reasons.get(0, ["unknown"]))
                for ts in quarantine_idx
            ]
            clean_df = df.drop(index=quarantine_idx)
            log.warning(
                "rows_failed_validation",
                symbol=symbol,
                quarantined=len(quarantine_df),
                reasons={
                    r: sum(1 for v in reasons.values() if r in "; ".join(v))
                    for r in {r for rs in reasons.values() for r in rs}
                },
            )
        else:
            clean_df = df
            quarantine_df = pd.DataFrame()

        return clean_df, quarantine_df

    def validate_batch(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply batch-level validation steps 8-12.

        Returns the (potentially deduplicated) DataFrame. Raises DataError
        only for stale data (step 10). Gap detection and row count warnings
        are logged but do not remove data.

        Args:
            df: OHLCV DataFrame (already row-validated).
            symbol: Symbol for logging context.

        Returns:
            DataFrame after deduplication (step 8).

        Raises:
            DataError: If data is stale beyond threshold (step 10).
        """
        if df.empty:
            return df

        # Step 8: Duplicate timestamps — deduplicate, keep last
        if df.index.duplicated().any():
            dup_count = df.index.duplicated(keep="last").sum()
            log.warning(
                "duplicate_timestamps",
                symbol=symbol,
                count=int(dup_count),
            )
            df = df[~df.index.duplicated(keep="last")]

        # Step 9: Gap detection
        self._detect_gaps(df, symbol)

        # Step 10: Stale data check
        self._check_staleness(df, symbol)

        # Step 11: Row count threshold
        if len(df) < 2:
            log.warning(
                "low_row_count",
                symbol=symbol,
                rows=len(df),
            )

        # Step 12: Cross-source consistency check
        if self._source == "equity" and self._db is not None and self._config is not None:
            try:
                from swingrl.data.cross_source import CrossSourceValidator

                csv = CrossSourceValidator(db=self._db, config=self._config)
                results = csv.validate_prices(symbols=[symbol], lookback_days=7)
                for r in results:
                    if r.status == "warning":
                        log.warning(
                            "cross_source_discrepancy",
                            symbol=r.symbol,
                            date=str(r.date),
                            diff=round(r.diff, 4),
                        )
                    elif r.status == "error":
                        log.warning(
                            "cross_source_error",
                            symbol=r.symbol,
                        )
            except Exception:  # noqa: BLE001
                log.warning("cross_source_check_failed", symbol=symbol)
        else:
            log.debug("cross_source_check_skipped", step=12, source=self._source)

        return df

    def _detect_gaps(self, df: pd.DataFrame, symbol: str) -> None:
        """Step 9: Detect gaps based on source type.

        Args:
            df: Validated DataFrame.
            symbol: Symbol for logging context.
        """
        if len(df) < 2:
            return

        if self._source == "equity":
            self._detect_equity_gaps(df, symbol)
        elif self._source == "crypto":
            self._detect_crypto_gaps(df, symbol)
        elif self._source == "fred":
            log.debug("fred_gap_detection", symbol=symbol, note="per-series cadence")

    def _detect_equity_gaps(self, df: pd.DataFrame, symbol: str) -> None:
        """Detect missing NYSE trading days in equity data.

        Args:
            df: Equity OHLCV DataFrame.
            symbol: Symbol for logging context.
        """
        start = df.index.min().normalize().tz_localize(None)
        end = df.index.max().normalize().tz_localize(None)
        expected = self._nyse.sessions_in_range(start, end)
        actual = pd.DatetimeIndex(pd.DatetimeIndex(df.index).normalize().tz_localize(None)).unique()
        missing = expected.difference(actual)
        if len(missing) > 0:
            log.warning(
                "equity_gaps_detected",
                symbol=symbol,
                missing_days=len(missing),
                first_gap=str(missing[0].date()),
            )

    def _detect_crypto_gaps(self, df: pd.DataFrame, symbol: str) -> None:
        """Detect missing 4H bars in continuous crypto series.

        Args:
            df: Crypto OHLCV DataFrame.
            symbol: Symbol for logging context.
        """
        time_diffs = df.index.to_series().diff()
        expected_diff = _CRYPTO_FREQ
        # Allow small tolerance for exact timestamp matching
        gaps = time_diffs[time_diffs > expected_diff * 1.5].dropna()
        if len(gaps) > 0:
            log.warning(
                "crypto_gaps_detected",
                symbol=symbol,
                gap_count=len(gaps),
                first_gap=str(gaps.index[0]),
            )

    def _check_staleness(self, df: pd.DataFrame, symbol: str) -> None:
        """Step 10: Check if data is stale beyond threshold.

        Args:
            df: Validated DataFrame.
            symbol: Symbol for logging context.

        Raises:
            DataError: If max timestamp is older than the staleness threshold.
        """
        now = datetime.now(UTC)
        max_ts = df.index.max()
        # Convert to offset-aware if needed
        if max_ts.tzinfo is None:
            max_ts = max_ts.tz_localize("UTC")

        if self._source == "equity":
            threshold = timedelta(days=_EQUITY_STALE_DAYS)
        elif self._source == "crypto":
            threshold = timedelta(hours=_CRYPTO_STALE_HOURS)
        else:
            # FRED: more lenient — monthly series can be 35 days old
            threshold = timedelta(days=35)

        age = now - max_ts
        if age > threshold:
            msg = (
                f"Stale data for {symbol}: most recent bar is "
                f"{age.days}d {age.seconds // 3600}h old "
                f"(threshold: {threshold})"
            )
            log.error("stale_data", symbol=symbol, age_hours=age.total_seconds() / 3600)
            raise DataError(msg)
