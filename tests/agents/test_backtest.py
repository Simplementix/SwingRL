"""Tests for walk-forward backtester fold generation and integration.

VAL-01: Walk-forward generates non-overlapping folds with growing training windows.
VAL-02: Purge gap enforced between train/test boundaries (no data leakage).
"""

from __future__ import annotations

import pytest

from swingrl.agents.backtest import FoldResult, generate_folds
from swingrl.utils.exceptions import DataError


class TestGenerateFolds:
    """Tests for generate_folds function."""

    def test_equity_fold_count(self) -> None:
        """Equity parameters produce at least 13 folds."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        assert len(folds) >= 13

    def test_minimum_three_folds(self) -> None:
        """Must produce at least 3 folds."""
        folds = generate_folds(
            total_bars=600,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        assert len(folds) >= 3

    def test_no_leakage_embargo_gap(self) -> None:
        """Train end + embargo <= test start for every fold."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for train_range, test_range in folds:
            assert train_range.stop + 10 <= test_range.start, (
                f"Leakage: train_end={train_range.stop}, embargo=10, test_start={test_range.start}"
            )

    def test_growing_window(self) -> None:
        """Train window starts at 0 and grows with each fold."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for train_range, _ in folds:
            assert train_range.start == 0, "Train window must always start at index 0"

        # Each successive train range should be longer
        train_sizes = [train_range.stop - train_range.start for train_range, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Train window not growing: fold {i - 1} has {train_sizes[i - 1]} bars, "
                f"fold {i} has {train_sizes[i]}"
            )

    def test_insufficient_data_raises(self) -> None:
        """DataError when not enough data for min_folds."""
        with pytest.raises(DataError, match="fewer than 3"):
            generate_folds(
                total_bars=300,
                test_bars=63,
                min_train_bars=252,
                embargo_bars=10,
            )

    def test_test_range_size(self) -> None:
        """Each test range has exactly test_bars elements."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for _, test_range in folds:
            assert len(test_range) == 63

    def test_no_test_overlap_across_folds(self) -> None:
        """No test range overlaps with any other test range."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        all_test_indices: set[int] = set()
        for _, test_range in folds:
            test_set = set(test_range)
            overlap = all_test_indices & test_set
            assert len(overlap) == 0, f"Test overlap found: {overlap}"
            all_test_indices.update(test_set)

    def test_custom_min_folds(self) -> None:
        """Raising min_folds increases the minimum fold requirement."""
        with pytest.raises(DataError, match="fewer than 5"):
            generate_folds(
                total_bars=600,
                test_bars=63,
                min_train_bars=252,
                embargo_bars=10,
                min_folds=5,
            )

    def test_crypto_parameters(self) -> None:
        """Crypto parameters (larger windows) produce valid folds."""
        folds = generate_folds(
            total_bars=10000,
            test_bars=540,
            min_train_bars=2190,
            embargo_bars=130,
        )
        assert len(folds) >= 3
        for train_range, test_range in folds:
            assert train_range.start == 0
            assert train_range.stop + 130 <= test_range.start
            assert len(test_range) == 540


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self) -> None:
        """FoldResult holds all required fields."""
        from swingrl.agents.validation import GateResult

        result = FoldResult(
            fold_number=1,
            train_range=(0, 252),
            test_range=(262, 325),
            in_sample_metrics={"sharpe": 1.2},
            out_of_sample_metrics={"sharpe": 0.9},
            trades=[{"pnl": 100.0}],
            gate_result=GateResult(passed=True, failures=[], details={}),
            overfitting={"gap": 0.25, "classification": "marginal"},
        )
        assert result.fold_number == 1
        assert result.train_range == (0, 252)
