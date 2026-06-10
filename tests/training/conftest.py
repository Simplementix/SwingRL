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
    is_sharpe: float | None = None,
    is_sortino: float | None = None,
    is_mdd: float | None = None,
    gate_passed: bool = True,
    overfitting_gap: float = 0.15,
    overfitting_class: str = "healthy",
    converged_at_step: int | None = 50000,
    total_timesteps: int | None = 100000,
) -> MagicMock:
    """Create a mock FoldResult with given OOS + IS metrics.

    Args:
        sharpe: OOS Sharpe ratio.
        mdd: OOS max drawdown (negative float).
        profit_factor: OOS profit factor.
        sortino: OOS Sortino ratio (defaults to sharpe * 1.2 if None).
        calmar: OOS Calmar ratio (defaults to sharpe * 0.5 if None).
        fold_number: Fold number for the mock.
        is_sharpe: IS Sharpe ratio (defaults to sharpe * 1.3 if None).
        is_sortino: IS Sortino ratio (defaults to sortino * 1.3 if None).
        is_mdd: IS MDD (defaults to mdd * 0.8 if None).
        gate_passed: Whether the fold's validation gate passed.
        overfitting_gap: Overfitting gap value.
        overfitting_class: Overfitting classification.
        converged_at_step: Step where training converged (None = not converged).
        total_timesteps: Total timesteps configured.

    Returns:
        MagicMock with full FoldResult interface.
    """
    _sortino = sortino if sortino is not None else sharpe * 1.2
    _calmar = calmar if calmar is not None else sharpe * 0.5

    fold = MagicMock()
    fold.fold_number = fold_number
    fold.train_range = (0, 252 * (fold_number + 1))
    fold.test_range = (252 * (fold_number + 1) + 10, 252 * (fold_number + 1) + 73)
    fold.out_of_sample_metrics = {
        "sharpe": sharpe,
        "mdd": mdd,
        "profit_factor": profit_factor,
        "sortino": _sortino,
        "calmar": _calmar,
        "win_rate": 0.55,
        "total_trades": 42,
        "avg_drawdown": mdd * 0.5,
        "max_dd_duration": 15,
        "final_portfolio_value": 105000.0,
        "total_return": 0.05,
    }
    fold.in_sample_metrics = {
        "sharpe": is_sharpe if is_sharpe is not None else sharpe * 1.3,
        "sortino": is_sortino if is_sortino is not None else _sortino * 1.3,
        "mdd": is_mdd if is_mdd is not None else mdd * 0.8,
        "total_return": 0.08,
    }
    fold.trades = [
        {"pnl": 50.0, "asset_idx": 0, "shares": 10, "entry_price": 100, "exit_price": 105},
        {"pnl": -20.0, "asset_idx": 1, "shares": 5, "entry_price": 200, "exit_price": 196},
        {"pnl": 30.0, "asset_idx": 0, "shares": 8, "entry_price": 102, "exit_price": 105.75},
    ]
    fold.gate_result = MagicMock()
    fold.gate_result.passed = gate_passed
    fold.gate_result.failures = [] if gate_passed else ["sharpe"]
    fold.overfitting = {
        "gap": overfitting_gap,
        "classification": overfitting_class,
    }
    fold.converged_at_step = converged_at_step
    fold.total_timesteps = total_timesteps
    return fold
