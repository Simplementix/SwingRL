"""Cross-source price validation: Alpaca (DuckDB) vs yfinance.

Compares closing prices stored in DuckDB ohlcv_daily with yfinance reference
data. Discrepancies beyond $0.05 tolerance are logged as warnings. Used by
DataValidator Step 12 for equity cross-source consistency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import structlog
import yfinance

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Tolerance for price comparison between sources
_TOLERANCE_USD = 0.05


@dataclass
class CrossSourceResult:
    """Result of comparing a single price point across sources."""

    symbol: str
    date: date
    alpaca_close: float
    yfinance_close: float
    diff: float
    status: str  # "ok" | "warning" | "error"


class CrossSourceValidator:
    """Compare Alpaca closing prices in DuckDB with yfinance as reference."""

    def __init__(self, db: DatabaseManager, config: SwingRLConfig) -> None:
        self._db = db
        self._config = config

    def validate_prices(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 7,
        as_of_date: date | None = None,
    ) -> list[CrossSourceResult]:
        """Compare DuckDB ohlcv_daily prices with yfinance for recent dates.

        Args:
            symbols: List of equity symbols to validate. Defaults to config equity symbols.
            lookback_days: Number of days to look back for comparison.
            as_of_date: Reference date for the lookback window. Defaults to today.

        Returns:
            List of CrossSourceResult with per-date comparison results.
        """
        if symbols is None:
            symbols = list(self._config.equity.symbols)

        # Query DuckDB for recent closing prices
        end_date = as_of_date if as_of_date is not None else date.today()
        start_date = end_date - timedelta(days=lookback_days)

        alpaca_data: dict[str, dict[date, float]] = {}
        with self._db.duckdb() as cursor:
            rows = cursor.execute(
                "SELECT symbol, date, close FROM ohlcv_daily "
                "WHERE symbol IN (SELECT UNNEST(?::TEXT[])) "
                "AND date >= ? AND date <= ? "
                "ORDER BY symbol, date",
                [symbols, str(start_date), str(end_date)],
            ).fetchall()
            for row in rows:
                sym = row[0]
                dt = row[1] if isinstance(row[1], date) else date.fromisoformat(str(row[1]))
                close = float(row[2])
                alpaca_data.setdefault(sym, {})[dt] = close

        # Download yfinance data
        yf_start = start_date.isoformat()
        yf_end = (end_date + timedelta(days=1)).isoformat()
        yf_df = yfinance.download(
            tickers=symbols,
            start=yf_start,
            end=yf_end,
            auto_adjust=False,
            progress=False,
        )

        # Parse yfinance data into dict
        yf_data = _parse_yfinance_data(yf_df, symbols)

        # Compare prices
        results: list[CrossSourceResult] = []
        for sym in symbols:
            if sym not in yf_data or not yf_data[sym]:
                results.append(
                    CrossSourceResult(
                        symbol=sym,
                        date=end_date,
                        alpaca_close=0.0,
                        yfinance_close=0.0,
                        diff=0.0,
                        status="error",
                    )
                )
                log.warning("cross_source_missing_symbol", symbol=sym, source="yfinance")
                continue

            alpaca_sym = alpaca_data.get(sym, {})
            yf_sym = yf_data[sym]

            for dt, alpaca_close in sorted(alpaca_sym.items()):
                yf_close = yf_sym.get(dt)
                if yf_close is None:
                    continue

                diff = abs(alpaca_close - yf_close)
                status = "warning" if diff > _TOLERANCE_USD else "ok"

                if status == "warning":
                    log.warning(
                        "cross_source_price_discrepancy",
                        symbol=sym,
                        date=str(dt),
                        alpaca_close=alpaca_close,
                        yfinance_close=yf_close,
                        diff=round(diff, 4),
                    )
                else:
                    log.info(
                        "cross_source_price_match",
                        symbol=sym,
                        date=str(dt),
                        diff=round(diff, 4),
                    )

                results.append(
                    CrossSourceResult(
                        symbol=sym,
                        date=dt,
                        alpaca_close=alpaca_close,
                        yfinance_close=yf_close,
                        diff=diff,
                        status=status,
                    )
                )

        return results


def _parse_yfinance_data(
    yf_df: pd.DataFrame,
    symbols: list[str],
) -> dict[str, dict[date, float]]:
    """Parse yfinance download result into {symbol: {date: adj_close}} dict."""
    yf_data: dict[str, dict[date, float]] = {}
    if yf_df.empty:
        return yf_data

    has_multiindex = isinstance(yf_df.columns, pd.MultiIndex)
    if has_multiindex:
        level_0 = yf_df.columns.get_level_values(0)
        if "Adj Close" in level_0:
            adj_close_df = yf_df["Adj Close"]
            if isinstance(adj_close_df, pd.Series):
                sym = adj_close_df.name if isinstance(adj_close_df.name, str) else symbols[0]
                for ts, val in adj_close_df.items():
                    if not _is_nan(val):
                        dt: date = ts.date() if hasattr(ts, "date") else ts  # type: ignore[assignment]
                        yf_data.setdefault(sym, {})[dt] = float(val)
            else:
                for sym in symbols:
                    if sym in adj_close_df.columns:
                        for ts, val in adj_close_df[sym].items():
                            if not _is_nan(val):
                                dt = ts.date() if hasattr(ts, "date") else ts
                                yf_data.setdefault(sym, {})[dt] = float(val)
    elif "Adj Close" in yf_df.columns:
        sym = symbols[0]
        for ts, adj_close in yf_df["Adj Close"].items():
            if not _is_nan(adj_close):
                dt = ts.date() if hasattr(ts, "date") else ts  # type: ignore[assignment]
                yf_data.setdefault(sym, {})[dt] = float(adj_close)

    return yf_data


def _is_nan(value: object) -> bool:
    """Check if a value is NaN."""
    try:
        return float(value) != float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
