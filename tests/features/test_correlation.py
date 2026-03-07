"""Tests for correlation pruning with domain-driven rules.

FEAT-09: Pairwise Pearson r > 0.85 threshold with domain-driven drop priority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingrl.features.correlation import CorrelationPruner


@pytest.fixture
def correlated_features() -> pd.DataFrame:
    """DataFrame with known correlated and uncorrelated columns."""
    rng = np.random.default_rng(42)
    n = 500
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "rsi_14": base + rng.normal(0, 0.1, n),  # high corr with base
            "macd_line": base + rng.normal(0, 0.1, n),  # high corr with rsi_14
            "macd_histogram": rng.normal(0, 1, n),  # uncorrelated
            "adx_14": rng.normal(0, 1, n),  # uncorrelated
            "volume_sma20_ratio": rng.normal(0, 1, n),  # uncorrelated
        }
    )


@pytest.fixture
def sma_correlated_features() -> pd.DataFrame:
    """DataFrame with SMA ratios correlated at different levels."""
    rng = np.random.default_rng(43)
    n = 500
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            # Correlated at ~0.87 (above 0.85 but below 0.90)
            "price_sma50_ratio": base + rng.normal(0, 0.35, n),
            "price_sma200_ratio": base + rng.normal(0, 0.35, n),
            "rsi_14": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def highly_correlated_sma_features() -> pd.DataFrame:
    """DataFrame with SMA ratios correlated above 0.90 threshold."""
    rng = np.random.default_rng(44)
    n = 500
    base = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            # Correlated at ~0.98 (above 0.90)
            "price_sma50_ratio": base + rng.normal(0, 0.05, n),
            "price_sma200_ratio": base + rng.normal(0, 0.05, n),
            "rsi_14": rng.normal(0, 1, n),
        }
    )


class TestCorrelationPrunerAnalyze:
    """Tests for correlation matrix analysis."""

    def test_analyze_returns_correlation_matrix(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: analyze() returns full correlation matrix."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        corr = pruner.analyze(correlated_features)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (5, 5)
        # Diagonal should be 1.0
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10


class TestFindCorrelatedPairs:
    """Tests for finding correlated pairs."""

    def test_find_correlated_pairs_above_threshold(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: find_correlated_pairs identifies pairs above 0.85."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        corr = pruner.analyze(correlated_features)
        pairs = pruner.find_correlated_pairs(corr)
        # rsi_14 and macd_line should be highly correlated
        pair_cols = [(a, b) for a, b, _ in pairs]
        found = any(("rsi_14" in (a, b) and "macd_line" in (a, b)) for a, b in pair_cols)
        assert found, f"Expected rsi_14/macd_line pair, got {pair_cols}"

    def test_find_correlated_pairs_sorted_desc(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: Pairs sorted by abs(r) descending."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        corr = pruner.analyze(correlated_features)
        pairs = pruner.find_correlated_pairs(corr)
        if len(pairs) > 1:
            r_values = [abs(r) for _, _, r in pairs]
            assert r_values == sorted(r_values, reverse=True)


class TestDomainRules:
    """Tests for domain-driven drop rules."""

    def test_sma_exception_below_090(
        self, feature_config: object, sma_correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: Both SMAs kept when r > 0.85 but < 0.90."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        pruned_df, dropped = pruner.prune(sma_correlated_features)
        # Both SMA columns should survive
        assert "price_sma50_ratio" in pruned_df.columns
        assert "price_sma200_ratio" in pruned_df.columns

    def test_sma_exception_above_090(
        self, feature_config: object, highly_correlated_sma_features: pd.DataFrame
    ) -> None:
        """FEAT-09: One SMA dropped when r > 0.90."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        pruned_df, dropped = pruner.prune(highly_correlated_sma_features)
        # At least one SMA should be dropped
        sma_cols_remaining = [c for c in pruned_df.columns if c.startswith("price_sma")]
        assert len(sma_cols_remaining) <= 2  # at most both if corr isn't quite above 0.90
        # If correlation truly > 0.90, one should be dropped
        corr = pruner.analyze(highly_correlated_sma_features)
        r = abs(corr.loc["price_sma50_ratio", "price_sma200_ratio"])
        if r > 0.90:
            assert len(sma_cols_remaining) == 1

    def test_rsi_preferred_over_oscillators(self, feature_config: object) -> None:
        """FEAT-09: RSI-14 preferred (higher priority) over other oscillators."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        rng = np.random.default_rng(45)
        n = 500
        base = rng.normal(50, 10, n)
        df = pd.DataFrame(
            {
                "rsi_14": base + rng.normal(0, 0.5, n),
                "stoch_k": base + rng.normal(0, 0.5, n),  # highly correlated with RSI
            }
        )
        _, dropped = pruner.prune(df)
        # RSI should be kept, stoch_k dropped
        assert "rsi_14" not in dropped
        assert "stoch_k" in dropped

    def test_macd_histogram_preferred_over_line(self, feature_config: object) -> None:
        """FEAT-09: MACD histogram preferred over raw MACD line when correlated."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        rng = np.random.default_rng(46)
        n = 500
        base = rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "macd_line": base + rng.normal(0, 0.1, n),
                "macd_histogram": base + rng.normal(0, 0.1, n),
            }
        )
        _, dropped = pruner.prune(df)
        assert "macd_histogram" not in dropped
        assert "macd_line" in dropped

    def test_vix_zscore_preferred_over_raw(self, feature_config: object) -> None:
        """FEAT-09: VIX z-score preferred over raw VIX when correlated."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        rng = np.random.default_rng(47)
        n = 500
        base = rng.normal(20, 5, n)
        df = pd.DataFrame(
            {
                "vix_raw": base + rng.normal(0, 0.3, n),
                "vix_zscore": base + rng.normal(0, 0.3, n),
            }
        )
        _, dropped = pruner.prune(df)
        assert "vix_zscore" not in dropped
        assert "vix_raw" in dropped


class TestPrune:
    """Tests for the full prune pipeline."""

    def test_prune_returns_dataframe_and_dropped(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: prune() returns (DataFrame, list[str])."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        pruned_df, dropped = pruner.prune(correlated_features)
        assert isinstance(pruned_df, pd.DataFrame)
        assert isinstance(dropped, list)
        # Dropped columns should not be in pruned_df
        for col in dropped:
            assert col not in pruned_df.columns

    def test_prune_removes_correlated_columns(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: Pruned DataFrame has fewer columns than input when correlations exist."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        pruned_df, dropped = pruner.prune(correlated_features)
        assert len(dropped) > 0
        assert len(pruned_df.columns) < len(correlated_features.columns)

    def test_threshold_configurable(self, feature_config: object) -> None:
        """FEAT-09: Threshold is configurable from FeaturesConfig."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        assert pruner._threshold == 0.85


class TestReport:
    """Tests for pruning report generation."""

    def test_report_returns_string(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: report() returns human-readable summary."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        corr = pruner.analyze(correlated_features)
        _, dropped = pruner.prune(correlated_features)
        report = pruner.report(corr, dropped)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_mentions_dropped_columns(
        self, feature_config: object, correlated_features: pd.DataFrame
    ) -> None:
        """FEAT-09: Report mentions which columns were dropped."""
        pruner = CorrelationPruner(feature_config)  # type: ignore[arg-type]
        corr = pruner.analyze(correlated_features)
        _, dropped = pruner.prune(correlated_features)
        report = pruner.report(corr, dropped)
        for col in dropped:
            assert col in report
