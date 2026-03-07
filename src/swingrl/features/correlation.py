"""Correlation pruning with domain-driven rules for feature pipeline.

Identifies redundant features using pairwise Pearson correlation and applies
domain-driven drop priority rules to reduce dimensionality before training.

This is a one-time pre-training analysis tool, not a runtime component.

Usage:
    from swingrl.features.correlation import CorrelationPruner

    pruner = CorrelationPruner(config)
    pruned_df, dropped = pruner.prune(features_df)
    report_text = pruner.report(pruner.analyze(features_df), dropped)
"""

from __future__ import annotations

import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

# Domain-driven keep priority: higher value = more important to keep.
# When two features are correlated, drop the one with lower priority.
KEEP_PRIORITY: dict[str, int] = {
    "rsi_14": 10,
    "macd_histogram": 9,
    "price_sma50_ratio": 8,
    "price_sma200_ratio": 8,
    "vix_zscore": 7,
}

# Both SMAs are kept unless correlation exceeds this higher threshold.
SMA_EXCEPTION_THRESHOLD: float = 0.90

# Columns that are SMA ratio features (for exception logic).
_SMA_RATIO_COLS = frozenset({"price_sma50_ratio", "price_sma200_ratio"})


class CorrelationPruner:
    """Prune redundant features using correlation analysis and domain rules.

    Identifies feature pairs with Pearson r above threshold and drops
    the less important one based on domain-driven priority rules.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize with configurable correlation threshold.

        Args:
            config: SwingRLConfig with features.correlation_threshold.
        """
        self._threshold: float = config.features.correlation_threshold

    def analyze(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise Pearson correlation matrix.

        Args:
            features: DataFrame of numeric feature columns.

        Returns:
            Full correlation matrix as DataFrame.
        """
        corr_matrix = features.corr()
        log.info("correlation_matrix_computed", columns=len(corr_matrix.columns))
        return corr_matrix

    def find_correlated_pairs(self, corr_matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
        """Find all feature pairs with abs(r) above threshold.

        Args:
            corr_matrix: Correlation matrix from analyze().

        Returns:
            List of (col_a, col_b, r_value) sorted by abs(r) descending.
            Self-correlations and duplicate pairs excluded.
        """
        pairs: list[tuple[str, str, float]] = []
        cols = corr_matrix.columns.tolist()
        seen: set[frozenset[str]] = set()

        for i, col_a in enumerate(cols):
            for j, col_b in enumerate(cols):
                if i >= j:
                    continue
                r = float(corr_matrix.iloc[i, j])  # type: ignore[arg-type]
                if abs(r) > self._threshold:
                    pair_key = frozenset({col_a, col_b})
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append((col_a, col_b, r))

        # Sort by abs(r) descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        log.info("correlated_pairs_found", count=len(pairs), threshold=self._threshold)
        return pairs

    def select_drops(self, correlated_pairs: list[tuple[str, str, float]]) -> list[str]:
        """Apply domain rules to select which columns to drop.

        Args:
            correlated_pairs: Output from find_correlated_pairs().

        Returns:
            Deduplicated list of column names to drop.
        """
        drops: set[str] = set()

        for col_a, col_b, r in correlated_pairs:
            # Skip if either column is already marked for dropping
            if col_a in drops or col_b in drops:
                continue

            # SMA exception: keep both if abs(r) < SMA_EXCEPTION_THRESHOLD
            both_sma = col_a in _SMA_RATIO_COLS and col_b in _SMA_RATIO_COLS
            if both_sma and abs(r) < SMA_EXCEPTION_THRESHOLD:
                log.info(
                    "sma_exception_applied",
                    col_a=col_a,
                    col_b=col_b,
                    r=r,
                )
                continue

            # Domain priority: drop the column with lower priority
            priority_a = KEEP_PRIORITY.get(col_a, 0)
            priority_b = KEEP_PRIORITY.get(col_b, 0)

            if priority_a != priority_b:
                to_drop = col_a if priority_a < priority_b else col_b
            else:
                # Equal priority: drop column with lower variance (less informative)
                # We need actual data for this, but since we only have pairs here,
                # use alphabetical as tiebreaker (deterministic)
                to_drop = max(col_a, col_b)

            drops.add(to_drop)
            log.info(
                "drop_selected",
                dropped=to_drop,
                kept=col_a if to_drop == col_b else col_b,
                r=r,
                reason="domain_priority" if priority_a != priority_b else "tiebreaker",
            )

        return sorted(drops)

    def prune(self, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Run full pruning pipeline: analyze, find pairs, select drops.

        Args:
            features: DataFrame of numeric feature columns.

        Returns:
            Tuple of (pruned DataFrame, list of dropped column names).
        """
        corr_matrix = self.analyze(features)
        pairs = self.find_correlated_pairs(corr_matrix)
        dropped = self.select_drops(pairs)
        pruned_df = features.drop(columns=dropped)

        log.info(
            "pruning_complete",
            input_cols=len(features.columns),
            output_cols=len(pruned_df.columns),
            dropped_count=len(dropped),
            dropped=dropped,
        )
        return pruned_df, dropped

    def report(self, corr_matrix: pd.DataFrame, dropped: list[str]) -> str:
        """Generate human-readable summary of pruning decisions.

        Args:
            corr_matrix: Full correlation matrix from analyze().
            dropped: List of dropped column names from prune().

        Returns:
            Multi-line report string documenting pruning decisions.
        """
        lines: list[str] = []
        lines.append("=== Correlation Pruning Report ===")
        lines.append(f"Threshold: {self._threshold}")
        lines.append(f"Total features: {len(corr_matrix.columns)}")
        lines.append(f"Dropped: {len(dropped)}")
        lines.append(f"Remaining: {len(corr_matrix.columns) - len(dropped)}")
        lines.append("")

        # Flagged pairs
        pairs = self.find_correlated_pairs(corr_matrix)
        if pairs:
            lines.append("Flagged pairs (abs(r) > threshold):")
            for col_a, col_b, r in pairs:
                lines.append(f"  {col_a} <-> {col_b}: r={r:.4f}")
        else:
            lines.append("No pairs above threshold.")

        lines.append("")

        # Dropped columns
        if dropped:
            lines.append("Dropped columns:")
            for col in dropped:
                priority = KEEP_PRIORITY.get(col, 0)
                lines.append(f"  - {col} (priority={priority})")
        else:
            lines.append("No columns dropped.")

        lines.append("")
        lines.append("Domain rules applied:")
        lines.append("  - RSI-14 preferred over other oscillators (priority=10)")
        lines.append("  - MACD histogram preferred over raw line (priority=9)")
        lines.append(f"  - SMA ratios: both kept unless r > {SMA_EXCEPTION_THRESHOLD}")
        lines.append("  - VIX z-score preferred over raw VIX (priority=7)")

        report_text = "\n".join(lines)
        log.info("pruning_report_generated", dropped=dropped)
        return report_text
