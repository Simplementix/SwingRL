"""Reusable defect sample DataFrames for validation testing.

Each function returns a small DataFrame (3-5 rows) with exactly one defective
row alongside clean rows. Used by test_validation.py to verify specific
validation steps catch specific defect types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _base_equity_df(rows: int = 5) -> pd.DataFrame:
    """Return a base clean equity OHLCV DataFrame with recent dates."""
    end = pd.Timestamp.now(tz="UTC").normalize()
    dates = pd.bdate_range(end=end, periods=rows, tz="UTC")
    close = 470.0 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 80_000_000.0),
        },
        index=dates,
    )


def _base_crypto_df(rows: int = 5) -> pd.DataFrame:
    """Return a base clean crypto OHLCV DataFrame with recent dates."""
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    start = now - pd.Timedelta(hours=(rows - 1) * 4)
    dates = pd.date_range(start, now, freq="4h", tz="UTC")
    close = 42_000.0 + np.arange(rows, dtype=float) * 10
    return pd.DataFrame(
        {
            "open": close - 5.0,
            "high": close + 50.0,
            "low": close - 50.0,
            "close": close,
            "volume": np.full(rows, 1000.0),
        },
        index=dates,
    )


def make_null_price_df() -> pd.DataFrame:
    """DataFrame with one row having NaN close price (triggers Step 1)."""
    df = _base_equity_df()
    df.iloc[2, df.columns.get_loc("close")] = np.nan
    return df


def make_negative_volume_df() -> pd.DataFrame:
    """DataFrame with one row having negative volume (triggers Step 3)."""
    df = _base_equity_df()
    df.iloc[1, df.columns.get_loc("volume")] = -1.0
    return df


def make_ohlc_violation_df() -> pd.DataFrame:
    """DataFrame with one row having high < low (triggers Step 4)."""
    df = _base_equity_df()
    idx = 3
    df.iloc[idx, df.columns.get_loc("high")] = df.iloc[idx]["low"] - 1.0
    return df


def make_ohlc_bounds_violation_df() -> pd.DataFrame:
    """DataFrame with one row having open > high (triggers Step 5)."""
    df = _base_equity_df()
    idx = 2
    df.iloc[idx, df.columns.get_loc("open")] = df.iloc[idx]["high"] + 10.0
    return df


def make_price_spike_df() -> pd.DataFrame:
    """DataFrame with one row having >50% close-to-close jump (triggers Step 6)."""
    df = _base_equity_df()
    idx = 3
    spike_close = df.iloc[idx - 1]["close"] * 2.0
    df.iloc[idx, df.columns.get_loc("close")] = spike_close
    df.iloc[idx, df.columns.get_loc("high")] = spike_close + 1.0
    df.iloc[idx, df.columns.get_loc("open")] = spike_close - 0.5
    return df


def make_zero_volume_df() -> pd.DataFrame:
    """DataFrame with one row having zero volume on a trading day (triggers Step 7)."""
    df = _base_equity_df()
    df.iloc[1, df.columns.get_loc("volume")] = 0.0
    return df


def make_duplicate_timestamps_df() -> pd.DataFrame:
    """DataFrame with duplicate timestamps (triggers Step 8)."""
    df = _base_equity_df()
    dup = df.iloc[[-1]].copy()
    dup["close"] = dup["close"] + 1.0
    return pd.concat([df, dup])


def make_stale_df() -> pd.DataFrame:
    """DataFrame with very old timestamps (triggers Step 10)."""
    dates = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": [470.0] * 5,
            "high": [472.0] * 5,
            "low": [469.0] * 5,
            "close": [471.0] * 5,
            "volume": [80e6] * 5,
        },
        index=dates,
    )


def make_gapped_equity_df() -> pd.DataFrame:
    """DataFrame with a missing NYSE trading day (triggers Step 9).

    Returns 5 recent business days with the second day removed.
    """
    df = _base_equity_df(rows=5)
    return pd.concat([df.iloc[:1], df.iloc[2:]])
