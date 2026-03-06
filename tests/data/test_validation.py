"""Tests for DataValidator — 12-step validation checklist.

Covers row-level quarantine (steps 1-7) and batch-level checks (steps 8-12).
Uses defect_samples helpers for structured defect injection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.fixtures.defect_samples import (
    _base_crypto_df,
    _base_equity_df,
    make_duplicate_timestamps_df,
    make_gapped_equity_df,
    make_negative_volume_df,
    make_null_price_df,
    make_ohlc_bounds_violation_df,
    make_ohlc_violation_df,
    make_price_spike_df,
    make_stale_df,
    make_zero_volume_df,
)
from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import DataError

# ---------------------------------------------------------------------------
# Row-level tests (steps 1-7)
# ---------------------------------------------------------------------------


class TestValidateRows:
    """DATA-05: Row-level validation steps 1-7."""

    def test_null_price_quarantined(self) -> None:
        """DATA-05: Row with NaN close is quarantined, other rows proceed."""
        df = make_null_price_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) == 1
        assert len(clean) == len(df) - 1
        assert "reason" in quarantine.columns
        assert "Step 1" in quarantine["reason"].iloc[0]

    def test_negative_volume_quarantined(self) -> None:
        """DATA-05: Row with negative volume is quarantined."""
        df = make_negative_volume_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("volume" in r.lower() for r in quarantine["reason"].values)

    def test_ohlc_ordering_quarantined(self) -> None:
        """DATA-05: Row with high < low is quarantined."""
        df = make_ohlc_violation_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("Step 4" in r for r in quarantine["reason"].values)

    def test_ohlc_bounds_quarantined(self) -> None:
        """DATA-05: Row with open > high is quarantined (with 0.01% tolerance)."""
        df = make_ohlc_bounds_violation_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("Step 5" in r for r in quarantine["reason"].values)

    def test_price_spike_quarantined(self) -> None:
        """DATA-05: Row with >50% close-to-close jump is quarantined."""
        df = make_price_spike_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("Step 6" in r for r in quarantine["reason"].values)

    def test_zero_volume_equity_quarantined(self) -> None:
        """DATA-05: Equity row with volume=0 on a trading day is quarantined."""
        df = make_zero_volume_df()
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        assert any("Step 7" in r for r in quarantine["reason"].values)

    def test_zero_volume_crypto_not_quarantined(self) -> None:
        """DATA-05: Crypto row with volume=0 is NOT quarantined by step 7."""
        df = _base_crypto_df()
        df.iloc[1, df.columns.get_loc("volume")] = 0.0
        validator = DataValidator(source="crypto")
        clean, quarantine = validator.validate_rows(df, symbol="BTCUSDT")
        # Step 7 is equity-only; volume=0 should pass for crypto
        # (step 3 only checks volume >= 0, not volume > 0)
        assert len(quarantine) == 0

    def test_valid_rows_pass(self) -> None:
        """DATA-05: Clean OHLCV data passes all row checks with empty quarantine."""
        df = _base_equity_df(rows=10)
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) == 0
        assert len(clean) == len(df)

    def test_multiple_defects_single_row(self) -> None:
        """DATA-05: Row with multiple defects gets all reasons combined."""
        df = _base_equity_df()
        # Make one row fail both null and negative volume
        idx = 2
        df.iloc[idx, df.columns.get_loc("close")] = float("nan")
        df.iloc[idx, df.columns.get_loc("volume")] = -5.0
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol="SPY")
        assert len(quarantine) >= 1
        # The quarantined row should mention multiple steps
        reasons = quarantine["reason"].iloc[0]
        assert "Step 1" in reasons
        assert "Step 3" in reasons


# ---------------------------------------------------------------------------
# Batch-level tests (steps 8-12)
# ---------------------------------------------------------------------------


class TestValidateBatch:
    """DATA-05: Batch-level validation steps 8-12."""

    def test_duplicate_dedup(self) -> None:
        """DATA-05: Batch with duplicate timestamps is deduplicated, warning logged."""
        df_with_dup = make_duplicate_timestamps_df()
        validator = DataValidator(source="equity")
        result = validator.validate_batch(df_with_dup, symbol="SPY")
        # Should have no duplicate timestamps after dedup
        assert not result.index.duplicated().any()
        # Should keep the last (most recent) value — the dup row has close+1.0
        original_close = _base_equity_df().iloc[-1]["close"]
        assert result.iloc[-1]["close"] == original_close + 1.0

    def test_equity_gap_logged(self) -> None:
        """DATA-05: Missing NYSE trading day is logged as warning, not quarantined."""
        gap_df = make_gapped_equity_df()
        validator = DataValidator(source="equity")
        # validate_batch returns data unchanged (gaps are warnings, not quarantine)
        result = validator.validate_batch(gap_df, symbol="SPY")
        assert len(result) == len(gap_df)

    def test_crypto_gap_logged(self) -> None:
        """DATA-05: Missing 4H bar in continuous series is logged as warning."""
        df = _base_crypto_df(rows=5)
        # Remove a middle row to create a gap
        gap_df = pd.concat([df.iloc[:1], df.iloc[2:]])
        validator = DataValidator(source="crypto")
        result = validator.validate_batch(gap_df, symbol="BTCUSDT")
        assert len(result) == len(gap_df)

    def test_stale_data_error(self) -> None:
        """DATA-05: max(timestamp) > 2 trading days old raises DataError for equity."""
        df = make_stale_df()
        validator = DataValidator(source="equity")
        with pytest.raises(DataError, match="Stale data"):
            validator.validate_batch(df, symbol="SPY")

    def test_fresh_data_no_error(self) -> None:
        """DATA-05: Fresh data does not raise DataError."""
        df = _base_equity_df(rows=5)
        validator = DataValidator(source="equity")
        result = validator.validate_batch(df, symbol="SPY")
        assert len(result) == len(df)

    def test_empty_df_passes_batch(self) -> None:
        """DATA-05: Empty DataFrame passes batch validation without error."""
        validator = DataValidator(source="equity")
        result = validator.validate_batch(pd.DataFrame(), symbol="SPY")
        assert result.empty
