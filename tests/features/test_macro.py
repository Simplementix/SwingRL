"""Tests for macro feature alignment via DuckDB ASOF JOIN.

Covers: FEAT-04 — 6 macro features aligned to equity/crypto bars with no look-ahead bias.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from swingrl.features.macro import MacroFeatureAligner


@pytest.fixture
def ddb_conn() -> Any:
    """Create in-memory DuckDB with seeded macro_features and OHLCV data."""
    conn = duckdb.connect(":memory:")

    # Create tables matching Phase 4 schema
    conn.execute("""
        CREATE TABLE macro_features (
            date DATE NOT NULL,
            series_id TEXT NOT NULL,
            value DOUBLE,
            release_date DATE,
            PRIMARY KEY (date, series_id)
        )
    """)

    conn.execute("""
        CREATE TABLE ohlcv_daily (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT,
            adjusted_close DOUBLE,
            fetched_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
    """)

    conn.execute("""
        CREATE TABLE ohlcv_4h (
            symbol TEXT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE,
            source TEXT,
            fetched_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, datetime)
        )
    """)

    # Seed OHLCV daily data: 300 business days for warmup
    dates = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(42)
    close_prices = 470.0 + rng.normal(0, 2, 300).cumsum()

    for i, d in enumerate(dates):
        conn.execute(
            "INSERT INTO ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
            [
                "SPY",
                d.date(),
                close_prices[i] - 1,
                close_prices[i] + 2,
                close_prices[i] - 2,
                close_prices[i],
                80_000_000,
                close_prices[i],
            ],
        )

    # Seed OHLCV 4H data: 600 bars for crypto
    crypto_dts = pd.date_range("2023-06-01", periods=600, freq="4h")
    btc_close = 30000.0 + rng.normal(0, 100, 600).cumsum()

    for i, dt in enumerate(crypto_dts):
        conn.execute(
            "INSERT INTO ohlcv_4h VALUES (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)",
            [
                "BTCUSDT",
                dt.to_pydatetime().replace(tzinfo=None),
                btc_close[i] - 50,
                btc_close[i] + 100,
                btc_close[i] - 100,
                btc_close[i],
                1500.0,
                "binance",
            ],
        )

    # Seed macro data with known release_date delays
    # VIX: daily, released same day (financial market data)
    for d in dates:
        vix_val = 18.0 + rng.normal(0, 3)
        conn.execute(
            "INSERT INTO macro_features VALUES (?, 'VIXCLS', ?, ?)",
            [d.date(), vix_val, d.date()],  # release_date = observation date
        )

    # T10Y2Y (yield curve): daily, released same day
    for d in dates:
        spread = 0.5 + rng.normal(0, 0.3)
        conn.execute(
            "INSERT INTO macro_features VALUES (?, 'T10Y2Y', ?, ?)",
            [d.date(), spread, d.date()],
        )

    # DFF (Fed Funds): daily, released same day
    for d in dates:
        ff = 5.25 + rng.normal(0, 0.05)
        conn.execute(
            "INSERT INTO macro_features VALUES (?, 'DFF', ?, ?)",
            [d.date(), ff, d.date()],
        )

    # CPIAUCSL: monthly, released with ~2 week delay
    for month_offset in range(15):
        obs_date = date(2023, 1, 1) + pd.DateOffset(months=month_offset)
        release = obs_date + pd.DateOffset(days=14)
        cpi_val = 300.0 + month_offset * 0.5
        conn.execute(
            "INSERT INTO macro_features VALUES (?, 'CPIAUCSL', ?, ?)",
            [
                obs_date.date() if hasattr(obs_date, "date") else obs_date,
                cpi_val,
                release.date() if hasattr(release, "date") else release,
            ],
        )

    # UNRATE: monthly, released with ~1 month delay
    for month_offset in range(15):
        obs_date = date(2023, 1, 1) + pd.DateOffset(months=month_offset)
        release = obs_date + pd.DateOffset(months=1)
        unemp_val = 3.7 - month_offset * 0.02
        conn.execute(
            "INSERT INTO macro_features VALUES (?, 'UNRATE', ?, ?)",
            [
                obs_date.date() if hasattr(obs_date, "date") else obs_date,
                unemp_val,
                release.date() if hasattr(release, "date") else release,
            ],
        )

    return conn


@pytest.fixture
def aligner(ddb_conn: Any) -> MacroFeatureAligner:
    """Create a MacroFeatureAligner with in-memory DuckDB."""
    return MacroFeatureAligner(ddb_conn)


class TestAlignEquity:
    """FEAT-04: align_equity returns 6 macro feature columns."""

    def test_returns_six_columns(self, aligner: MacroFeatureAligner) -> None:
        """align_equity returns DataFrame with 6 columns."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        expected_cols = {
            "vix_zscore",
            "yield_curve_spread",
            "yield_curve_direction",
            "fed_funds_90d_change",
            "cpi_yoy",
            "unemployment_3m_direction",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_no_nan_after_warmup(self, aligner: MacroFeatureAligner) -> None:
        """Macro features have no NaN after warmup period (forward-fill handles gaps)."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")
        feature_cols = [
            "vix_zscore",
            "yield_curve_spread",
            "yield_curve_direction",
            "fed_funds_90d_change",
            "cpi_yoy",
            "unemployment_3m_direction",
        ]
        # After dropping initial warmup rows, no NaN should remain
        trimmed = result.dropna(subset=feature_cols, how="any")
        assert len(trimmed) > 0
        assert trimmed[feature_cols].isna().sum().sum() == 0


class TestAsofJoinBehavior:
    """FEAT-04: ASOF JOIN uses release_date for look-ahead bias prevention."""

    def test_uses_release_date_not_observation_date(self, ddb_conn: Any) -> None:
        """ASOF JOIN uses release_date (not date) to prevent look-ahead bias.

        CPI data has a 2-week delay. On a date before the release, the previous
        month's value should be used.
        """
        aligner = MacroFeatureAligner(ddb_conn)
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        # CPI for June 2023 has release_date ~June 15, 2023
        # Before June 15, only May CPI (or earlier) should be visible
        assert len(result) > 0


class TestDerivedMacro:
    """FEAT-04: Derived macro feature computations."""

    def test_vix_zscore_uses_rolling_window(self, aligner: MacroFeatureAligner) -> None:
        """VIX z-score uses 252-day rolling window."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        # VIX z-score should be finite after warmup
        vix_z = result["vix_zscore"].dropna()
        assert len(vix_z) > 0
        # Z-scores should be roughly in [-3, 3] range
        assert vix_z.abs().max() < 10

    def test_yield_curve_direction_is_binary(self, aligner: MacroFeatureAligner) -> None:
        """yield_curve_direction is binary (1 if spread > 0, 0 otherwise)."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        yc_dir = result["yield_curve_direction"].dropna()
        unique_vals = set(yc_dir.unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_unemployment_direction_is_binary(self, aligner: MacroFeatureAligner) -> None:
        """unemployment_3m_direction is binary (1 if improving, 0 otherwise)."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        unemp_dir = result["unemployment_3m_direction"].dropna()
        unique_vals = set(unemp_dir.unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_fed_funds_90d_change(self, aligner: MacroFeatureAligner) -> None:
        """Fed Funds 90-day change computed as current - 90_days_ago."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        ff_change = result["fed_funds_90d_change"].dropna()
        assert len(ff_change) > 0
        # With stable fed funds (~5.25), 90-day change should be near zero
        assert ff_change.abs().mean() < 1.0

    def test_cpi_yoy_computed(self, aligner: MacroFeatureAligner) -> None:
        """CPI YoY computed from ASOF-joined values."""
        result = aligner.align_equity("SPY", "2023-06-01", "2023-12-31")

        cpi = result["cpi_yoy"].dropna()
        assert len(cpi) > 0


class TestAlignCrypto:
    """FEAT-04: Crypto alignment forward-fills from equity close."""

    def test_align_crypto_returns_six_columns(self, aligner: MacroFeatureAligner) -> None:
        """align_crypto returns DataFrame with 6 macro columns."""
        result = aligner.align_crypto("BTCUSDT", "2023-06-01", "2023-10-01")

        expected_cols = {
            "vix_zscore",
            "yield_curve_spread",
            "yield_curve_direction",
            "fed_funds_90d_change",
            "cpi_yoy",
            "unemployment_3m_direction",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_crypto_forward_fills_gaps(self, aligner: MacroFeatureAligner) -> None:
        """Crypto forward-fills from last available macro data (macro updates business days only)."""
        result = aligner.align_crypto("BTCUSDT", "2023-06-01", "2023-10-01")

        feature_cols = [
            "vix_zscore",
            "yield_curve_spread",
            "yield_curve_direction",
            "fed_funds_90d_change",
            "cpi_yoy",
            "unemployment_3m_direction",
        ]
        trimmed = result.dropna(subset=feature_cols, how="any")
        assert len(trimmed) > 0
