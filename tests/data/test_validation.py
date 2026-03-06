"""Tests for DataValidator — 12-step validation checklist.

Covers row-level quarantine (steps 1-7) and batch-level checks (steps 8-12).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import DataError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_equity_df(rows: int = 5) -> pd.DataFrame:
    """Return a small clean equity OHLCV DataFrame with recent dates."""
    # End near today so staleness check passes; generate backwards
    end = pd.Timestamp.now(tz="UTC").normalize()
    dates = pd.bdate_range(end=end, periods=rows, tz="UTC")
    rng = np.random.default_rng(42)
    close = 470.0 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(50_000_000, 100_000_000, rows).astype(float),
        },
        index=dates,
    )


def _make_clean_crypto_df(rows: int = 5) -> pd.DataFrame:
    """Return a small clean crypto OHLCV DataFrame with recent dates."""
    # End near now so staleness check passes even after removing rows
    now = pd.Timestamp.now(tz="UTC").floor("4h")
    end = now
    start = end - pd.Timedelta(hours=(rows - 1) * 4)
    dates = pd.date_range(start, end, freq="4h", tz="UTC")
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


# ---------------------------------------------------------------------------
# Row-level tests (steps 1-7)
# ---------------------------------------------------------------------------


class TestValidateRows:
    """DATA-05: Row-level validation steps 1-7."""

    def test_null_price_quarantined(self) -> None:
        """DATA-05: Row with NaN close is quarantined, other rows proceed."""
        df = _make_clean_equity_df()
        df.iloc[2, df.columns.get_loc("close")] = np.nan
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) == 1
        assert len(clean) == len(df) - 1
        assert "reason" in quarantine.columns

    def test_negative_volume_quarantined(self) -> None:
        """DATA-05: Row with negative volume is quarantined."""
        df = _make_clean_equity_df()
        df.iloc[1, df.columns.get_loc("volume")] = -1.0
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("volume" in r.lower() for r in quarantine["reason"].values)

    def test_ohlc_ordering_quarantined(self) -> None:
        """DATA-05: Row with high < low is quarantined."""
        df = _make_clean_equity_df()
        # Swap high and low for one row
        idx = 3
        df.iloc[idx, df.columns.get_loc("high")] = df.iloc[idx]["low"] - 1.0
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1

    def test_ohlc_bounds_quarantined(self) -> None:
        """DATA-05: Row with open > high is quarantined (with 0.01% tolerance)."""
        df = _make_clean_equity_df()
        idx = 2
        # Set open well above high (beyond tolerance)
        df.iloc[idx, df.columns.get_loc("open")] = df.iloc[idx]["high"] + 10.0
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1

    def test_price_spike_quarantined(self) -> None:
        """DATA-05: Row with >50% close-to-close jump is quarantined."""
        df = _make_clean_equity_df()
        # Make a massive jump at row 3
        df.iloc[3, df.columns.get_loc("close")] = df.iloc[2]["close"] * 2.0
        df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["close"] + 1.0
        df.iloc[3, df.columns.get_loc("open")] = df.iloc[3]["close"] - 0.5
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1

    def test_zero_volume_equity_quarantined(self) -> None:
        """DATA-05: Equity row with volume=0 on a trading day is quarantined."""
        df = _make_clean_equity_df()
        df.iloc[1, df.columns.get_loc("volume")] = 0.0
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1

    def test_zero_volume_crypto_not_quarantined(self) -> None:
        """DATA-05: Crypto row with volume=0 is NOT quarantined by step 7."""
        df = _make_clean_crypto_df()
        df.iloc[1, df.columns.get_loc("volume")] = 0.0
        validator = DataValidator(source="crypto")
        clean, quarantine = validator.validate_rows(df, symbol="BTCUSDT")
        # Step 7 is equity-only; volume=0 should pass for crypto
        # (step 3 only checks volume >= 0, not volume > 0)
        assert len(quarantine) == 0

    def test_valid_rows_pass(self) -> None:
        """DATA-05: Clean OHLCV data passes all row checks with empty quarantine."""
        df = _make_clean_equity_df(rows=10)
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) == 0
        assert len(clean) == len(df)


# ---------------------------------------------------------------------------
# Batch-level tests (steps 8-12)
# ---------------------------------------------------------------------------


class TestValidateBatch:
    """DATA-05: Batch-level validation steps 8-12."""

    def test_duplicate_dedup(self) -> None:
        """DATA-05: Batch with duplicate timestamps is deduplicated, warning logged."""
        df = _make_clean_equity_df()
        # Duplicate the last row
        dup = df.iloc[[-1]].copy()
        dup["close"] = dup["close"] + 1.0  # different value, same timestamp
        df_with_dup = pd.concat([df, dup])
        validator = DataValidator(source="equity")
        result = validator.validate_batch(df_with_dup, symbol="SPY")
        # Should have no duplicate timestamps after dedup
        assert not result.index.duplicated().any()
        # Should keep the last (most recent) value
        assert result.iloc[-1]["close"] == dup.iloc[0]["close"]

    def test_equity_gap_logged(self) -> None:
        """DATA-05: Missing NYSE trading day is logged as warning, not quarantined."""
        # Use recent dates so staleness check doesn't fire
        # Build 3 dates with one NYSE gap by getting recent business days
        df = _make_clean_equity_df(rows=5)
        # Remove a middle row to create a gap
        gap_df = pd.concat([df.iloc[:1], df.iloc[2:]])
        validator = DataValidator(source="equity")
        # validate_batch should return the data unchanged (gaps are warnings, not quarantine)
        result = validator.validate_batch(gap_df, symbol="SPY")
        assert len(result) == len(gap_df)

    def test_crypto_gap_logged(self) -> None:
        """DATA-05: Missing 4H bar in continuous series is logged as warning."""
        df = _make_clean_crypto_df(rows=5)
        # Remove a middle row to create a gap
        gap_df = pd.concat([df.iloc[:1], df.iloc[2:]])
        validator = DataValidator(source="crypto")
        result = validator.validate_batch(gap_df, symbol="BTCUSDT")
        assert len(result) == len(gap_df)

    def test_stale_data_error(self) -> None:
        """DATA-05: max(timestamp) > 2 trading days old raises DataError for equity."""
        # Create data that is very old
        dates = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [470.0] * 5,
                "high": [472.0] * 5,
                "low": [469.0] * 5,
                "close": [471.0] * 5,
                "volume": [80e6] * 5,
            },
            index=dates,
        )
        validator = DataValidator(source="equity")
        with pytest.raises(DataError):
            validator.validate_batch(df, symbol="SPY")
