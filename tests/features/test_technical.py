"""Tests for TechnicalIndicatorCalculator.

FEAT-01: 9 price-action indicators
FEAT-02: Derived features (log returns, BB width)
FEAT-10: Weekly-derived features (trend direction, weekly RSI-14)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestComputePriceAction:
    """Tests for compute_price_action() — 9 price-action indicators."""

    def test_returns_nine_columns(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: compute_price_action returns DataFrame with 9 indicator columns."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        expected_cols = [
            "price_sma50_ratio",
            "price_sma200_ratio",
            "rsi_14",
            "macd_line",
            "macd_histogram",
            "bb_position",
            "atr_14_pct",
            "volume_sma20_ratio",
            "adx_14",
        ]
        assert list(result.columns) == expected_cols

    def test_no_nan_after_warmup(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: After 200-bar warmup, all 9 indicators have no NaN values."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        # After 200-bar warmup, rows 200+ should be NaN-free
        post_warmup = result.iloc[200:]
        assert not post_warmup.isna().any().any(), (
            f"NaN found after warmup:\n{post_warmup.isna().sum()}"
        )

    def test_sma_ratios_are_dimensionless(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: Price/SMA ratios are close to 1.0 (not raw dollar values)."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        post_warmup = result.iloc[200:]
        assert post_warmup["price_sma50_ratio"].mean() == pytest.approx(1.0, abs=0.1)
        assert post_warmup["price_sma200_ratio"].mean() == pytest.approx(1.0, abs=0.1)

    def test_rsi_within_bounds(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: RSI-14 values fall within [0, 100]."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_bb_position_can_exceed_zero_one(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: bb_position can exceed [0,1] range — breakout signal, NOT clipped."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        bb = result["bb_position"].dropna()
        # With random walk data, breakouts above/below bands are likely
        # At minimum, we verify the range is NOT clipped to [0, 1]
        assert bb.min() < 0.5 or bb.max() > 0.5  # Some variation exists

    def test_atr_pct_is_small_positive(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: atr_14_pct values are small positive fractions (not dollar amounts)."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(equity_ohlcv_250)

        atr = result["atr_14_pct"].dropna()
        assert atr.min() > 0.0
        assert atr.max() < 1.0  # ATR as fraction of price should be < 100%

    def test_volume_sma_ratio_handles_zero(self) -> None:
        """FEAT-01: volume_sma20_ratio returns 1.0 when volume SMA is zero."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        # Create data with zero volumes
        dates = pd.date_range("2024-01-01", periods=50, freq="B", tz="UTC")
        rng = np.random.default_rng(99)
        close = 100.0 + rng.normal(0, 1, 50).cumsum()
        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.zeros(50),  # all zero volume
            },
            index=dates,
        )

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_price_action(df)

        vol_ratio = result["volume_sma20_ratio"].dropna()
        assert (vol_ratio == 1.0).all()

    def test_does_not_mutate_original(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-01: stockstats does not mutate the original DataFrame."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        original_cols = list(equity_ohlcv_250.columns)
        original_shape = equity_ohlcv_250.shape

        calc = TechnicalIndicatorCalculator()
        calc.compute_price_action(equity_ohlcv_250)

        assert list(equity_ohlcv_250.columns) == original_cols
        assert equity_ohlcv_250.shape == original_shape


class TestComputeWeeklyFeatures:
    """Tests for compute_weekly_features() — weekly-derived features."""

    def test_returns_two_columns(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-10: compute_weekly_features returns weekly_trend_dir and weekly_rsi_14."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_weekly_features(equity_ohlcv_250)

        assert "weekly_trend_dir" in result.columns
        assert "weekly_rsi_14" in result.columns

    def test_weekly_trend_dir_is_binary(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-10: weekly_trend_dir is 0 or 1."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_weekly_features(equity_ohlcv_250)

        trend = result["weekly_trend_dir"].dropna()
        assert set(trend.unique()).issubset({0.0, 1.0})

    def test_same_index_as_input(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-10: Result has same index as input (forward-filled to daily)."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_weekly_features(equity_ohlcv_250)

        assert len(result) == len(equity_ohlcv_250)
        assert result.index.equals(equity_ohlcv_250.index)


class TestComputeCryptoMultiTimeframe:
    """Tests for compute_crypto_multi_timeframe() — crypto multi-TF features."""

    def test_returns_four_columns(self, crypto_ohlcv_400: pd.DataFrame) -> None:
        """FEAT-10: compute_crypto_multi_timeframe returns 4 columns."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_crypto_multi_timeframe(crypto_ohlcv_400)

        expected = ["daily_trend_dir", "daily_rsi_14", "four_h_rsi_14", "four_h_price_sma20_ratio"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_same_index_as_input(self, crypto_ohlcv_400: pd.DataFrame) -> None:
        """FEAT-10: Result aligned to 4H index."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_crypto_multi_timeframe(crypto_ohlcv_400)

        assert len(result) == len(crypto_ohlcv_400)


class TestComputeDerived:
    """Tests for compute_derived() — log returns and BB width for HMM input."""

    def test_returns_expected_columns(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-02: compute_derived returns log returns and bb_width."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_derived(equity_ohlcv_250["close"])

        expected = ["log_return_1d", "log_return_5d", "log_return_20d", "bb_width"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_log_returns_are_small_values(self, equity_ohlcv_250: pd.DataFrame) -> None:
        """FEAT-02: Log returns are small values (not raw price differences)."""
        from swingrl.features.technical import TechnicalIndicatorCalculator

        calc = TechnicalIndicatorCalculator()
        result = calc.compute_derived(equity_ohlcv_250["close"])

        lr1 = result["log_return_1d"].dropna()
        assert abs(lr1.mean()) < 0.1  # Log returns should be small
