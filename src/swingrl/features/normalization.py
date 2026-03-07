"""Rolling z-score normalization for feature pipeline.

Normalizes technical features using per-environment rolling windows
(252 bars for equity, 360 bars for crypto) before observation vector assembly.

Usage:
    from swingrl.features.normalization import RollingZScoreNormalizer

    normalizer = RollingZScoreNormalizer(config)
    normalized = normalizer.normalize(features_df, environment="equity")
    stats = normalizer.validate_bounds(normalized, warmup=252)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class RollingZScoreNormalizer:
    """Apply rolling z-score normalization with per-environment windows.

    Z-scoring makes features scale-invariant: (x - rolling_mean) / rolling_std.
    Uses epsilon floor on std to prevent division by zero with flat prices.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize with config-driven window sizes and epsilon.

        Args:
            config: SwingRLConfig with features section containing window sizes.
        """
        self._equity_window: int = config.features.equity_zscore_window
        self._crypto_window: int = config.features.crypto_zscore_window
        self._epsilon: float = config.features.zscore_epsilon

    def normalize(
        self,
        features: pd.DataFrame,
        environment: Literal["equity", "crypto"],
    ) -> pd.DataFrame:
        """Apply rolling z-score normalization to all feature columns.

        Args:
            features: DataFrame of numeric features to normalize.
            environment: Environment type determining window size.

        Returns:
            DataFrame with same shape, index, and columns. NaN for warmup period.
        """
        window = self._equity_window if environment == "equity" else self._crypto_window

        rolling_mean = features.rolling(window).mean()
        rolling_std = features.rolling(window).std().clip(lower=self._epsilon)
        result = (features - rolling_mean) / rolling_std

        log.info(
            "zscore_normalized",
            environment=environment,
            window=window,
            rows=len(result),
            columns=len(result.columns),
            nan_rows=int(result.isna().any(axis=1).sum()),
        )
        return result

    def validate_bounds(self, normalized: pd.DataFrame, warmup: int) -> dict[str, Any]:
        """Check what fraction of post-warmup values fall within [-3, 3].

        Args:
            normalized: Z-scored DataFrame from normalize().
            warmup: Number of warmup rows to skip (window size).

        Returns:
            Dict with pct_in_range, min_val, max_val, outlier_columns.
        """
        post_warmup = normalized.iloc[warmup:]
        values = post_warmup.values.flatten()
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            log.warning("validate_bounds_empty", warmup=warmup, total_rows=len(normalized))
            return {
                "pct_in_range": 0.0,
                "min_val": float("nan"),
                "max_val": float("nan"),
                "outlier_columns": [],
            }

        in_range = np.sum((valid >= -3) & (valid <= 3))
        pct = float(in_range / len(valid))
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))

        # Identify columns with outliers
        outlier_columns: list[str] = []
        for col in post_warmup.columns:
            col_vals = np.asarray(post_warmup[col].dropna().values, dtype=float)
            if len(col_vals) > 0:
                col_in_range = np.sum((col_vals >= -3) & (col_vals <= 3)) / len(col_vals)
                if col_in_range < 0.90:
                    outlier_columns.append(col)

        if pct < 0.90:
            log.warning(
                "zscore_bounds_check_failed",
                pct_in_range=pct,
                min_val=min_val,
                max_val=max_val,
                outlier_columns=outlier_columns,
            )
        else:
            log.info(
                "zscore_bounds_check_passed",
                pct_in_range=pct,
                min_val=min_val,
                max_val=max_val,
            )

        return {
            "pct_in_range": pct,
            "min_val": min_val,
            "max_val": max_val,
            "outlier_columns": outlier_columns,
        }
