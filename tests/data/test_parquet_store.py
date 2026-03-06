"""Tests for ParquetStore — Parquet read/upsert/write helpers.

Covers create, merge, dedup, and read-missing behaviors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swingrl.data.parquet_store import ParquetStore


def _make_ohlcv(start: str, rows: int = 5, base_price: float = 470.0) -> pd.DataFrame:
    """Return a small OHLCV DataFrame with business-day index."""
    dates = pd.date_range(start, periods=rows, freq="B", tz="UTC")
    close = base_price + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 80e6),
        },
        index=dates,
    )


class TestParquetStore:
    """Tests for ParquetStore upsert and read operations."""

    def test_parquet_upsert_creates_new(self, tmp_path: Path) -> None:
        """DATA-05: Upsert to non-existent file creates it."""
        store = ParquetStore()
        path = tmp_path / "equity" / "SPY_daily.parquet"
        df = _make_ohlcv("2024-01-01")
        store.upsert(path, df)
        assert path.exists()
        result = store.read(path)
        assert len(result) == len(df)

    def test_parquet_upsert_merges(self, tmp_path: Path) -> None:
        """DATA-05: Upsert with overlapping timestamps keeps latest values."""
        store = ParquetStore()
        path = tmp_path / "SPY_daily.parquet"
        # Write initial data
        df1 = _make_ohlcv("2024-01-01", rows=5)
        store.upsert(path, df1)
        # Write overlapping data with different close prices
        df2 = _make_ohlcv("2024-01-03", rows=5, base_price=480.0)
        store.upsert(path, df2)
        result = store.read(path)
        # Should have merged: 2 unique from df1 + 5 from df2 = 7 (3 overlap → use df2 values)
        # df1: Jan 1, 2, 3, 4, 5 (B days)
        # df2: Jan 3, 4, 5, 8, 9 (B days)
        # Merged: Jan 1, 2 (from df1) + Jan 3, 4, 5, 8, 9 (from df2) = 7
        assert len(result) == 7
        # Overlapping dates should have df2's (latest) values
        overlap_date = df2.index[0]
        assert result.loc[overlap_date, "close"] == df2.iloc[0]["close"]

    def test_parquet_upsert_preserves_existing(self, tmp_path: Path) -> None:
        """DATA-05: Upsert with new timestamps appends without modifying existing."""
        store = ParquetStore()
        path = tmp_path / "SPY_daily.parquet"
        df1 = _make_ohlcv("2024-01-01", rows=3)
        store.upsert(path, df1)
        # New data with no overlap
        df2 = _make_ohlcv("2024-01-15", rows=3)
        store.upsert(path, df2)
        result = store.read(path)
        assert len(result) == 6
        # Original data should be unchanged
        for ts in df1.index:
            assert result.loc[ts, "close"] == df1.loc[ts, "close"]

    def test_parquet_read_missing(self, tmp_path: Path) -> None:
        """DATA-05: Reading non-existent file returns empty DataFrame."""
        store = ParquetStore()
        path = tmp_path / "nonexistent.parquet"
        result = store.read(path)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
