"""Shared fixtures and helpers for training tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_fold_result(
    sharpe: float = 1.2,
    mdd: float = -0.08,
    profit_factor: float = 1.5,
    sortino: float | None = None,
    calmar: float | None = None,
    fold_number: int = 1,
) -> MagicMock:
    """Create a mock FoldResult with given OOS metrics.

    Args:
        sharpe: OOS Sharpe ratio.
        mdd: OOS max drawdown (negative float).
        profit_factor: OOS profit factor.
        sortino: OOS Sortino ratio (defaults to sharpe * 1.2 if None).
        calmar: OOS Calmar ratio (defaults to sharpe * 0.5 if None).
        fold_number: Fold number for the mock.

    Returns:
        MagicMock with .out_of_sample_metrics dict and .fold_number.
    """
    fold = MagicMock()
    fold.fold_number = fold_number
    fold.out_of_sample_metrics = {
        "sharpe": sharpe,
        "mdd": mdd,
        "profit_factor": profit_factor,
        "sortino": sortino if sortino is not None else sharpe * 1.2,
        "calmar": calmar if calmar is not None else sharpe * 0.5,
    }
    return fold
