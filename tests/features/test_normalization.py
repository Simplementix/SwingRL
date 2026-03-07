"""Tests for rolling z-score normalization.

FEAT-06: Rolling z-score with per-environment windows (252 equity, 360 crypto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from swingrl.features.normalization import RollingZScoreNormalizer


class TestRollingZScoreNormalizer:
    """Tests for RollingZScoreNormalizer."""

    def test_normalize_returns_same_shape(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: normalize() returns DataFrame with same shape as input."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        assert result.shape == features.shape

    def test_no_nan_after_warmup_equity(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: After 252-bar warmup, no NaN values in equity output."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        # Window=252, so rows 252+ should have no NaN
        post_warmup = result.iloc[252:]
        assert not post_warmup.isna().any().any(), (
            f"NaN found after warmup: {post_warmup.isna().sum()}"
        )

    def test_nan_during_warmup(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: Before warmup period, output is NaN (expected)."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        # First row should be NaN (not enough data for rolling window)
        assert result.iloc[0].isna().all()

    def test_zscore_values_roughly_in_range(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: Z-scored values roughly within [-3, 3] for realistic data (>90%)."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        post_warmup = result.iloc[252:]
        values = post_warmup.values.flatten()
        valid = values[~np.isnan(values)]
        in_range = np.sum((valid >= -3) & (valid <= 3))
        pct = in_range / len(valid)
        assert pct > 0.90, f"Only {pct:.1%} of values in [-3, 3]"

    def test_flat_price_no_inf(self, feature_config: object) -> None:
        """FEAT-06: Flat-price scenario (std=0) produces valid output, not inf."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        dates = pd.date_range("2023-01-02", periods=300, freq="B", tz="UTC")
        flat = pd.DataFrame({"flat_col": [100.0] * 300}, index=dates)
        result = normalizer.normalize(flat, environment="equity")
        post_warmup = result.iloc[252:]
        assert not np.isinf(post_warmup.values).any(), "inf found with flat prices"
        # Flat prices -> mean = value, std = epsilon -> should be ~0
        valid = post_warmup["flat_col"].dropna()
        assert (valid.abs() < 1.0).all(), "Flat price z-scores should be near zero"

    def test_crypto_window_360(
        self, feature_config: object, crypto_ohlcv_400: pd.DataFrame
    ) -> None:
        """FEAT-06: Window=360 works correctly for crypto environment."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = crypto_ohlcv_400[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="crypto")
        # After 360-bar warmup, no NaN
        post_warmup = result.iloc[360:]
        assert not post_warmup.isna().any().any(), (
            f"NaN found after crypto warmup: {post_warmup.isna().sum()}"
        )

    def test_output_index_matches_input(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: Output index matches input index exactly."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        pd.testing.assert_index_equal(result.index, features.index)

    def test_column_names_preserved(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: Column names preserved after normalization."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        result = normalizer.normalize(features, environment="equity")
        assert list(result.columns) == list(features.columns)


class TestValidateBounds:
    """Tests for validate_bounds method."""

    def test_validate_bounds_returns_stats(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: validate_bounds returns expected stats dict."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        normalized = normalizer.normalize(features, environment="equity")
        stats = normalizer.validate_bounds(normalized, warmup=252)
        assert "pct_in_range" in stats
        assert "min_val" in stats
        assert "max_val" in stats
        assert "outlier_columns" in stats

    def test_validate_bounds_pct_in_range(
        self, feature_config: object, equity_ohlcv_250: pd.DataFrame
    ) -> None:
        """FEAT-06: validate_bounds reports >90% in [-3, 3] for realistic data."""
        normalizer = RollingZScoreNormalizer(feature_config)  # type: ignore[arg-type]
        features = equity_ohlcv_250[["close", "volume"]].copy()
        normalized = normalizer.normalize(features, environment="equity")
        stats = normalizer.validate_bounds(normalized, warmup=252)
        assert stats["pct_in_range"] > 0.90
