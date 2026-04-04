"""Fundamental data fetcher for equity ETFs.

Fetches P/E ratio, earnings growth, debt-to-equity, and dividend yield from
yfinance (primary) with Alpha Vantage fallback. Validates, computes sector-relative
z-scores, and stores to DuckDB.

Usage:
    from swingrl.config.schema import load_config
    from swingrl.features.fundamentals import FundamentalFetcher

    config = load_config()
    fetcher = FundamentalFetcher(config)
    df = fetcher.fetch_all(config.equity.symbols)
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog
import yfinance

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Field mapping: yfinance info key -> our column name
_YF_FIELD_MAP: dict[str, str] = {
    "trailingPE": "pe_ratio",
    "earningsQuarterlyGrowth": "earnings_growth",
    "debtToEquity": "debt_to_equity",
    "dividendYield": "dividend_yield",
}

# Alpha Vantage field mapping
_AV_FIELD_MAP: dict[str, str] = {
    "TrailingPE": "pe_ratio",
    "QuarterlyEarningsGrowthYOY": "earnings_growth",
    "DebtToEquityRatio": "debt_to_equity",
    "DividendYield": "dividend_yield",
}


def _try_import_av() -> Any:
    """Lazily import Alpha Vantage FundamentalData."""
    try:
        from alpha_vantage.fundamentaldata import (  # type: ignore[import-untyped,import-not-found]
            FundamentalData,
        )

        return FundamentalData
    except ImportError:
        return None


# Lazy import at module level for mocking in tests
try:
    from alpha_vantage.fundamentaldata import FundamentalData
except ImportError:
    FundamentalData = None  # type: ignore[assignment, misc]


class FundamentalFetcher:
    """Fetches and validates fundamental data for equity ETFs.

    Primary source: yfinance Ticker.info
    Fallback source: Alpha Vantage FundamentalData (if API key available)
    """

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize with config.

        Args:
            config: Validated SwingRLConfig instance.
        """
        self._config = config
        self._av_api_key: str | None = os.environ.get("ALPHA_VANTAGE_API_KEY")

    def fetch_symbol(self, symbol: str) -> dict[str, float | None]:
        """Fetch fundamental metrics for a single symbol.

        Tries yfinance first, falls back to Alpha Vantage if available.

        Args:
            symbol: Ticker symbol (e.g., "SPY").

        Returns:
            Dict with keys: pe_ratio, earnings_growth, debt_to_equity, dividend_yield.
            Missing values are float('nan').
        """
        try:
            result = self._fetch_from_yfinance(symbol)
            return result
        except Exception:
            log.warning("yfinance_fetch_failed", symbol=symbol)
            if self._av_api_key and FundamentalData is not None:
                return self._fetch_from_alpha_vantage(symbol)
            return self._empty_fundamentals()

    def _fetch_from_yfinance(self, symbol: str) -> dict[str, float | None]:
        """Fetch from yfinance Ticker.info.

        Args:
            symbol: Ticker symbol.

        Returns:
            Fundamental metrics dict.
        """
        ticker = yfinance.Ticker(symbol)
        info = ticker.info

        result: dict[str, float | None] = {}
        for yf_key, our_key in _YF_FIELD_MAP.items():
            value = info.get(yf_key)
            result[our_key] = float(value) if value is not None else float("nan")

        log.info(
            "fundamentals_fetched",
            symbol=symbol,
            source="yfinance",
            fields_available=sum(1 for v in result.values() if v is not None and not math.isnan(v)),
        )
        return result

    # Alpha Vantage free tier: 5 requests/minute. Track last call time for rate limiting.
    _av_last_call: float = 0.0

    def _fetch_from_alpha_vantage(self, symbol: str) -> dict[str, float | None]:
        """Fetch from Alpha Vantage as fallback (rate-limited to 5 req/min).

        Args:
            symbol: Ticker symbol.

        Returns:
            Fundamental metrics dict.
        """
        # Enforce 12s gap between AV calls (5 req/min = 1 per 12s)
        elapsed = time.monotonic() - self._av_last_call
        if elapsed < 12.0:
            time.sleep(12.0 - elapsed)
        self._av_last_call = time.monotonic()

        try:
            av = FundamentalData(key=self._av_api_key, output_format="pandas")
            overview_df, _ = av.get_company_overview(symbol)

            result: dict[str, float | None] = {}
            for av_key, our_key in _AV_FIELD_MAP.items():
                if av_key in overview_df.columns:
                    raw_value = overview_df[av_key].iloc[0]
                    try:
                        result[our_key] = float(raw_value)
                    except (ValueError, TypeError):
                        result[our_key] = float("nan")
                else:
                    result[our_key] = float("nan")

            log.info(
                "fundamentals_fetched",
                symbol=symbol,
                source="alpha_vantage",
                fields_available=sum(
                    1 for v in result.values() if v is not None and not math.isnan(v)
                ),
            )
            return result
        except Exception:
            log.error("alpha_vantage_fetch_failed", symbol=symbol)
            return self._empty_fundamentals()

    @staticmethod
    def _empty_fundamentals() -> dict[str, float | None]:
        """Return dict with all NaN values."""
        return {
            "pe_ratio": float("nan"),
            "earnings_growth": float("nan"),
            "debt_to_equity": float("nan"),
            "dividend_yield": float("nan"),
        }

    @staticmethod
    def validate_fundamentals(data: dict[str, float | None]) -> dict[str, float | None]:
        """Validate fundamental data constraints.

        - P/E must be positive or None/NaN (negative -> NaN)
        - D/E must be non-negative or None/NaN (negative -> NaN)

        Args:
            data: Fundamental metrics dict.

        Returns:
            Validated dict with invalid values set to NaN.
        """
        result = dict(data)

        pe = result.get("pe_ratio")
        if pe is not None and not math.isnan(pe) and pe < 0:
            result["pe_ratio"] = float("nan")

        de = result.get("debt_to_equity")
        if de is not None and not math.isnan(de) and de < 0:
            result["debt_to_equity"] = float("nan")

        return result

    def fetch_all(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch and validate fundamentals for all symbols.

        Partial failure resilient -- individual symbol failures are logged
        and the symbol is included with NaN values.

        Args:
            symbols: List of ticker symbols.

        Returns:
            DataFrame with columns: symbol, pe_ratio, earnings_growth,
            debt_to_equity, dividend_yield, sector.
        """
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                raw = self.fetch_symbol(symbol)
                validated = self.validate_fundamentals(raw)
                sector = self._get_sector(symbol)
                row: dict[str, Any] = {"symbol": symbol, **validated, "sector": sector}
                rows.append(row)
            except Exception:
                log.error("fetch_all_symbol_failed", symbol=symbol)
                row = {
                    "symbol": symbol,
                    **self._empty_fundamentals(),
                    "sector": "Unknown",
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol from yfinance.

        Args:
            symbol: Ticker symbol.

        Returns:
            Sector string or "Unknown".
        """
        try:
            ticker = yfinance.Ticker(symbol)
            return str(ticker.info.get("sector", "Unknown"))
        except Exception:
            return "Unknown"

    @staticmethod
    def sector_relative_zscore(fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Compute sector-relative z-scores for P/E ratio.

        Only P/E gets z-scored (per spec). Other metrics remain as raw values.

        Args:
            fundamentals: DataFrame with columns: symbol, pe_ratio, earnings_growth,
                debt_to_equity, dividend_yield, sector.

        Returns:
            DataFrame with pe_zscore column added (pe_ratio removed).
        """
        result = fundamentals.copy()

        def _zscore_group(group: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
            pe = group["pe_ratio"]
            mean = pe.mean()
            std = pe.std()
            if pd.isna(std) or pd.isna(mean):
                # All values NaN — propagate NaN (not 0) to distinguish from identical values
                return pd.Series(float("nan"), index=group.index)
            if std < 1e-10:
                # All values identical (non-NaN) — z-score is 0 by definition
                return pd.Series(0.0, index=group.index)
            return (pe - mean) / std

        result["pe_zscore"] = result.groupby("sector", group_keys=False).apply(
            _zscore_group,
            include_groups=False,  # type: ignore[call-overload]
        )
        return result

    def store_fundamentals(self, db: DatabaseManager, fundamentals: pd.DataFrame) -> int:
        """Store fundamentals to DuckDB via replacement scan.

        Args:
            db: DatabaseManager instance.
            fundamentals: DataFrame with fundamental data.

        Returns:
            Number of rows written.
        """
        now = datetime.now(tz=UTC)
        today = now.date()

        store_df = fundamentals[
            ["symbol", "pe_ratio", "earnings_growth", "debt_to_equity", "dividend_yield", "sector"]
        ].copy()
        store_df["date"] = today
        store_df["fetched_at"] = now

        from swingrl.data.pg_helpers import executemany_from_df  # noqa: PLC0415

        columns = [
            "symbol",
            "date",
            "pe_ratio",
            "earnings_growth",
            "debt_to_equity",
            "dividend_yield",
            "sector",
            "fetched_at",
        ]
        on_conflict = (
            "(symbol, date) DO UPDATE SET "
            "pe_ratio=EXCLUDED.pe_ratio, earnings_growth=EXCLUDED.earnings_growth, "
            "debt_to_equity=EXCLUDED.debt_to_equity, dividend_yield=EXCLUDED.dividend_yield, "
            "sector=EXCLUDED.sector, fetched_at=EXCLUDED.fetched_at"
        )
        with db.connection() as conn:
            executemany_from_df(conn, "fundamentals", store_df, columns, on_conflict=on_conflict)

        rows_written = len(store_df)
        log.info("fundamentals_stored", rows=rows_written)
        return rows_written
