"""Macro feature alignment via PostgreSQL LATERAL JOIN.

Aligns 5 FRED macro series (VIX, yield curve, Fed Funds, CPI, unemployment)
to OHLCV bars using release_date for look-ahead bias prevention. Derives
6 macro features shared across equity and crypto environments.

Usage:
    from swingrl.features.macro import MacroFeatureAligner

    aligner = MacroFeatureAligner(db)
    macro_df = aligner.align_equity("SPY", "2023-01-01", "2023-12-31")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
import structlog

from swingrl.data.pg_helpers import fetchdf

log = structlog.get_logger(__name__)

# VIX z-score rolling window (business days)
_VIX_ZSCORE_WINDOW = 252

# LATERAL JOIN query for equity daily bars — PostgreSQL equivalent of ASOF JOIN.
# Each LATERAL subquery finds the most recent macro value on or before the bar date.
# Requires index: macro_features(series_id, release_date DESC)
_EQUITY_LATERAL_QUERY = """
    SELECT
        p.symbol,
        p.date,
        p.close,
        vix.value AS vix_value,
        t10y2y.value AS yield_curve_spread,
        dff.value AS fed_funds_rate,
        cpi.value AS cpi_value,
        unemp.value AS unemployment_rate
    FROM ohlcv_daily p
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'VIXCLS' AND release_date <= p.date
        ORDER BY release_date DESC LIMIT 1
    ) AS vix ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'T10Y2Y' AND release_date <= p.date
        ORDER BY release_date DESC LIMIT 1
    ) AS t10y2y ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'DFF' AND release_date <= p.date
        ORDER BY release_date DESC LIMIT 1
    ) AS dff ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'CPIAUCSL' AND release_date <= p.date
        ORDER BY release_date DESC LIMIT 1
    ) AS cpi ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'UNRATE' AND release_date <= p.date
        ORDER BY release_date DESC LIMIT 1
    ) AS unemp ON true
    WHERE p.symbol = %s
      AND p.date >= %s::DATE
      AND p.date <= %s::DATE
    ORDER BY p.date
"""

# LATERAL JOIN query for crypto 4H bars — joins against ohlcv_4h.
# Macro data forward-fills from last available equity close.
_CRYPTO_LATERAL_QUERY = """
    SELECT
        p.symbol,
        p.datetime,
        p.close,
        vix.value AS vix_value,
        t10y2y.value AS yield_curve_spread,
        dff.value AS fed_funds_rate,
        cpi.value AS cpi_value,
        unemp.value AS unemployment_rate
    FROM ohlcv_4h p
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'VIXCLS' AND release_date <= p.datetime::DATE
        ORDER BY release_date DESC LIMIT 1
    ) AS vix ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'T10Y2Y' AND release_date <= p.datetime::DATE
        ORDER BY release_date DESC LIMIT 1
    ) AS t10y2y ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'DFF' AND release_date <= p.datetime::DATE
        ORDER BY release_date DESC LIMIT 1
    ) AS dff ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'CPIAUCSL' AND release_date <= p.datetime::DATE
        ORDER BY release_date DESC LIMIT 1
    ) AS cpi ON true
    LEFT JOIN LATERAL (
        SELECT value FROM macro_features
        WHERE series_id = 'UNRATE' AND release_date <= p.datetime::DATE
        ORDER BY release_date DESC LIMIT 1
    ) AS unemp ON true
    WHERE p.symbol = %s
      AND p.datetime >= %s::TIMESTAMPTZ
      AND p.datetime <= %s::TIMESTAMPTZ
    ORDER BY p.datetime
"""


class MacroFeatureAligner:
    """Aligns macro features to OHLCV bars via PostgreSQL LATERAL JOIN.

    Produces 6 derived macro features:
    1. vix_zscore: VIX 1-year rolling z-score
    2. yield_curve_spread: raw T10Y2Y value
    3. yield_curve_direction: binary (1 if spread > 0, else 0)
    4. fed_funds_90d_change: current DFF - DFF 90 business days ago
    5. cpi_yoy: year-over-year CPI change
    6. unemployment_3m_direction: binary (1 if improving = lower, else 0)
    """

    def __init__(self, db: Any) -> None:
        """Initialize with DatabaseManager or raw psycopg connection.

        Args:
            db: DatabaseManager instance providing PostgreSQL connections,
                or a raw psycopg Connection for direct use.
        """
        self._db = db
        # Detect whether db is a raw psycopg Connection vs a DatabaseManager
        self._is_raw_conn = isinstance(db, psycopg.Connection)

    def _execute_query(self, query: str, params: list[object]) -> pd.DataFrame:
        """Execute a query and return a DataFrame via fetchdf.

        Handles both DatabaseManager (context manager) and raw psycopg
        connections transparently.

        Args:
            query: SQL query string with %s placeholders.
            params: Query parameters.

        Returns:
            DataFrame with query results.
        """
        if self._is_raw_conn:
            cur = self._db.execute(query, params)
            return fetchdf(cur)
        with self._db.connection() as conn:
            cur = conn.execute(query, params)
            return fetchdf(cur)

    def _fetch_macro_aligned_equity(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch raw macro values aligned to equity daily bars via LATERAL JOIN.

        Args:
            symbol: Equity symbol (e.g., "SPY").
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with date index and raw macro columns.
        """
        result = self._execute_query(_EQUITY_LATERAL_QUERY, [symbol, start, end])

        if "date" in result.columns:
            result = result.set_index("date")

        log.info(
            "macro_aligned_equity",
            symbol=symbol,
            rows=len(result),
            start=start,
            end=end,
        )
        return result

    def _fetch_macro_aligned_crypto(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch raw macro values aligned to crypto 4H bars via LATERAL JOIN.

        Args:
            symbol: Crypto symbol (e.g., "BTCUSDT").
            start: Start datetime string.
            end: End datetime string.

        Returns:
            DataFrame with datetime index and raw macro columns.
        """
        result = self._execute_query(_CRYPTO_LATERAL_QUERY, [symbol, start, end])

        if "datetime" in result.columns:
            result = result.set_index("datetime")

        log.info(
            "macro_aligned_crypto",
            symbol=symbol,
            rows=len(result),
            start=start,
            end=end,
        )
        return result

    def compute_derived_macro(self, raw_macro: pd.DataFrame) -> pd.DataFrame:
        """Compute 6 derived macro features from raw LATERAL-joined values.

        Args:
            raw_macro: DataFrame with columns: vix_value, yield_curve_spread,
                fed_funds_rate, cpi_value, unemployment_rate.

        Returns:
            DataFrame with 6 derived macro feature columns.
        """
        result = pd.DataFrame(index=raw_macro.index)

        # 1. VIX z-score (252-day rolling)
        vix = raw_macro["vix_value"]
        vix_mean = vix.rolling(_VIX_ZSCORE_WINDOW, min_periods=1).mean()
        vix_std = vix.rolling(_VIX_ZSCORE_WINDOW, min_periods=1).std()
        vix_std = vix_std.clip(lower=1e-8)
        result["vix_zscore"] = (vix - vix_mean) / vix_std

        # 2. Yield curve spread (raw T10Y2Y)
        result["yield_curve_spread"] = raw_macro["yield_curve_spread"]

        # 3. Yield curve direction (binary: 1 if spread > 0, else 0)
        result["yield_curve_direction"] = np.where(raw_macro["yield_curve_spread"] > 0, 1.0, 0.0)

        # 4. Fed Funds 90-day change
        ff = raw_macro["fed_funds_rate"]
        result["fed_funds_90d_change"] = ff - ff.shift(90)

        # 5. CPI YoY (current / 12 months ago - 1)
        cpi = raw_macro["cpi_value"]
        # Approximate 12 months as 252 business days for daily, or use shift
        # For daily bars: ~252 trading days per year
        # For 4H bars: ~1512 bars per year (252 * 6)
        # Use a simpler approach: shift by enough bars to approximate 12 months
        cpi_lag = cpi.shift(252) if len(cpi) > 300 else cpi.shift(min(60, len(cpi) - 1))
        result["cpi_yoy"] = np.where(cpi_lag > 0, (cpi / cpi_lag) - 1, np.nan)

        # 6. Unemployment 3-month direction (1 if improving = current < 3 months ago)
        unemp = raw_macro["unemployment_rate"]
        # ~63 business days = 3 months for daily data
        unemp_lag = unemp.shift(63) if len(unemp) > 100 else unemp.shift(min(20, len(unemp) - 1))
        result["unemployment_3m_direction"] = np.where(unemp < unemp_lag, 1.0, 0.0)

        return result

    def align_equity(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Align macro features to equity daily bars.

        Fetches raw macro data via LATERAL JOIN, computes 6 derived features,
        and forward-fills any remaining NaN.

        Args:
            symbol: Equity symbol (e.g., "SPY").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            DataFrame with 6 macro feature columns aligned to daily bars.
        """
        raw = self._fetch_macro_aligned_equity(symbol, start, end)
        derived = self.compute_derived_macro(raw)
        derived = derived.ffill()

        log.info(
            "macro_features_aligned",
            environment="equity",
            symbol=symbol,
            rows=len(derived),
            nan_count=int(derived.isna().sum().sum()),
        )
        return derived

    def align_crypto(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Align macro features to crypto 4H bars.

        Crypto bars run 24/7 but macro data updates on business days only.
        Forward-fill handles weekend/holiday gaps.

        Args:
            symbol: Crypto symbol (e.g., "BTCUSDT").
            start: Start datetime (YYYY-MM-DD).
            end: End datetime (YYYY-MM-DD).

        Returns:
            DataFrame with 6 macro feature columns aligned to 4H bars.
        """
        raw = self._fetch_macro_aligned_crypto(symbol, start, end)
        derived = self._compute_derived_macro_crypto(raw)
        derived = derived.ffill()

        log.info(
            "macro_features_aligned",
            environment="crypto",
            symbol=symbol,
            rows=len(derived),
            nan_count=int(derived.isna().sum().sum()),
        )
        return derived

    def _compute_derived_macro_crypto(self, raw_macro: pd.DataFrame) -> pd.DataFrame:
        """Compute derived macro features for crypto (adjusted lag periods).

        Crypto uses 4H bars, so lag periods are scaled: 1 day = 6 bars.

        Args:
            raw_macro: DataFrame with raw macro columns.

        Returns:
            DataFrame with 6 derived macro feature columns.
        """
        result = pd.DataFrame(index=raw_macro.index)

        # 1. VIX z-score (scaled for 4H: 252 days * 6 bars = 1512 bars)
        vix = raw_macro["vix_value"]
        crypto_vix_window = min(1512, len(vix))
        vix_mean = vix.rolling(crypto_vix_window, min_periods=1).mean()
        vix_std = vix.rolling(crypto_vix_window, min_periods=1).std()
        vix_std = vix_std.clip(lower=1e-8)
        result["vix_zscore"] = (vix - vix_mean) / vix_std

        # 2. Yield curve spread
        result["yield_curve_spread"] = raw_macro["yield_curve_spread"]

        # 3. Yield curve direction
        result["yield_curve_direction"] = np.where(raw_macro["yield_curve_spread"] > 0, 1.0, 0.0)

        # 4. Fed Funds 90-day change (90 days * 6 bars = 540 bars)
        ff = raw_macro["fed_funds_rate"]
        ff_lag = min(540, len(ff) - 1) if len(ff) > 1 else 1
        result["fed_funds_90d_change"] = ff - ff.shift(ff_lag)

        # 5. CPI YoY (12 months * ~30 days * 6 bars = ~2160 bars, approximate)
        cpi = raw_macro["cpi_value"]
        cpi_lag = min(1512, len(cpi) - 1) if len(cpi) > 1 else 1
        cpi_shifted = cpi.shift(cpi_lag)
        result["cpi_yoy"] = np.where(cpi_shifted > 0, (cpi / cpi_shifted) - 1, np.nan)

        # 6. Unemployment direction (3 months * ~22 days * 6 = ~396 bars)
        unemp = raw_macro["unemployment_rate"]
        unemp_lag = min(396, len(unemp) - 1) if len(unemp) > 1 else 1
        result["unemployment_3m_direction"] = np.where(unemp < unemp.shift(unemp_lag), 1.0, 0.0)

        return result
