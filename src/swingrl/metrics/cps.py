"""Capital Preservation Score (CPS) — single-iteration goal metric for memory agents.

Three formulas are computed in parallel and persisted side-by-side so we can A/B
across iterations without committing to one shape too early:

- **v1 multiplicative** (primary): hardest to game; any factor collapsing kills the score.
  ``CPS_v1 = median_return × (1 - max_mdd)² × tanh(mean_winner_sharpe / 2) × (winners/total)``

- **v2 additive with hard penalties**: easier to interpret and attribute.
  ``CPS_v2 = 0.5·median_return + 0.3·(mean_winner_sharpe/4)
           - 1.0·max(0, max_mdd - 0.15)
           - 2.0·max(0, |max_single_loss| - 0.10)
           - 0.10·chronic_failure_count``

- **v3 sortino-anchored with regression penalty**: explicitly penalizes the
  "gave back returns to inflate other metrics" anti-pattern.
  ``CPS_v3 = median_sortino × (1 - max_mdd) - 2.0·max(0, prev_return - this_return)``

  Returns ``None`` when ``prev_iter_median_return`` is ``None`` (iter 0 has no baseline).

Per the project's capital-preservation principle (CLAUDE.md), all formulas treat
drawdown and worst-case losses as primary penalties. Pass rate is *not* a target —
the multiplicative interaction in v1 means a high winners/total ratio cannot
compensate for a falling median return.

**Control fold handling**: CPS computation includes ALL folds regardless of
``is_control_fold`` to preserve cross-iteration comparability (iter 0-2 had no
control folds). For the treatment-vs-control split view (iter 3+ only), call
the formulas twice with pre-filtered fold lists.

All functions are pure: deterministic, no DB access, no side effects.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import structlog

log = structlog.get_logger(__name__)


class FoldMetrics(TypedDict):
    """Per-fold metrics consumed by CPS formulas.

    Maps directly to columns in the ``backtest_results`` Postgres table so the
    backfill script can pass DB rows in unchanged.

    Fields:
        fold_number: 0-based fold index.
        sharpe: Annualized OOS Sharpe ratio.
        mdd: Max drawdown as a positive fraction (0.0-1.0).
        total_return: OOS total return as a fraction (e.g. 0.08 = +8%).
        profit_factor: Sum of winning trade PnL / sum of losing trade PnL.
        win_rate: Fraction of trades profitable (0.0-1.0).
        total_trades: Number of round-trip trades in the OOS window.
        sortino: Annualized OOS Sortino ratio.
        max_single_loss: Largest single-trade loss as a signed fraction
            (e.g. -0.08 = lost 8% of equity on the worst trade). May be None
            for legacy rows; treated as 0.0 in the CPS computation.
        overfitting_class: One of "healthy", "marginal", "reject". A fold is a
            "winner" iff its overfitting_class == "healthy".
        is_control_fold: True if the fold was held as a scientific control (no
            LLM reward adjustments). NOT used by the formulas themselves —
            included so callers can pre-filter for treatment-only / control-only
            views before invoking these functions.
    """

    fold_number: int
    sharpe: float
    mdd: float
    total_return: float
    profit_factor: float
    win_rate: float
    total_trades: int
    sortino: float
    max_single_loss: float | None
    overfitting_class: str
    is_control_fold: bool


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    """Median of a list of floats; returns 0.0 for empty list (defensive)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def _mean(values: list[float]) -> float:
    """Mean of a list of floats; returns 0.0 for empty list (defensive)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_winner(fold: FoldMetrics) -> bool:
    """A fold is a 'winner' iff its overfitting_class is 'healthy'.

    The legacy gate (Sharpe>0.7, MDD<0.15, PF>1.5, OG<0.20) is implicit in
    the overfitting_class assignment performed upstream by validation.py.
    """
    return fold["overfitting_class"] == "healthy"


def _components(per_fold: list[FoldMetrics]) -> dict[str, float]:
    """Extract the shared statistical inputs used by all three formulas."""
    if not per_fold:
        return {
            "median_return": 0.0,
            "max_mdd": 0.0,
            "max_abs_single_loss": 0.0,
            "median_sortino": 0.0,
            "mean_winner_sharpe": 0.0,
            "winners": 0.0,
            "total": 0.0,
        }
    returns = [f["total_return"] for f in per_fold]
    mdds = [f["mdd"] for f in per_fold]
    sortinos = [f["sortino"] for f in per_fold]
    winner_sharpes = [f["sharpe"] for f in per_fold if _is_winner(f)]
    single_losses: list[float] = []
    for f in per_fold:
        msl = f.get("max_single_loss")
        if msl is not None:
            single_losses.append(abs(msl))
    return {
        "median_return": _median(returns),
        "max_mdd": max(mdds) if mdds else 0.0,
        "max_abs_single_loss": max(single_losses) if single_losses else 0.0,
        "median_sortino": _median(sortinos),
        "mean_winner_sharpe": _mean(winner_sharpes),
        "winners": float(len(winner_sharpes)),
        "total": float(len(per_fold)),
    }


# ---------------------------------------------------------------------------
# Formula v1 — Multiplicative
# ---------------------------------------------------------------------------


def compute_cps_v1_multiplicative(per_fold: list[FoldMetrics]) -> float:
    """CPS v1: ``median_return × (1 - max_mdd)² × tanh(mean_winner_sharpe/2) × (winners/total)``.

    Multiplicative form: any factor going to zero kills the score, so a
    high pass rate cannot compensate for a falling median return. Preserves
    sign — a negative median return produces a negative CPS.

    Args:
        per_fold: List of FoldMetrics for one iteration's worth of folds.

    Returns:
        CPS v1 score. ``0.0`` for empty input.
    """
    if not per_fold:
        return 0.0

    c = _components(per_fold)
    if c["total"] == 0.0:
        return 0.0

    drawdown_factor = (1.0 - c["max_mdd"]) ** 2
    sharpe_factor = math.tanh(c["mean_winner_sharpe"] / 2.0) if c["winners"] > 0 else 0.0
    winner_ratio = c["winners"] / c["total"]

    score = c["median_return"] * drawdown_factor * sharpe_factor * winner_ratio
    return float(score)


# ---------------------------------------------------------------------------
# Formula v2 — Additive with hard penalties
# ---------------------------------------------------------------------------


def compute_cps_v2_additive(
    per_fold: list[FoldMetrics],
    chronic_failure_count: int = 0,
) -> float:
    """CPS v2: additive with hard penalties for catastrophic outcomes.

    ``CPS_v2 = 0.5·median_return + 0.3·(mean_winner_sharpe/4)
             - 1.0·max(0, max_mdd - 0.15)
             - 2.0·max(0, max_abs_single_loss - 0.10)
             - 0.10·chronic_failure_count``

    Easier to interpret and attribute than v1 — each term contributes
    independently. Tradeoff: a fold trading "0% return + 0% MDD" can score
    above an "8% return + 14% MDD" fold, so it can reward timid folds.

    Args:
        per_fold: List of FoldMetrics for one iteration.
        chronic_failure_count: Number of chronic-failure folds (passed in by
            caller because chronic determination requires cross-iteration
            history that this pure function cannot see).

    Returns:
        CPS v2 score. ``0.0`` for empty input.
    """
    if not per_fold:
        return 0.0

    c = _components(per_fold)

    return_term = 0.5 * c["median_return"]
    sharpe_term = 0.3 * (c["mean_winner_sharpe"] / 4.0)
    mdd_penalty = 1.0 * max(0.0, c["max_mdd"] - 0.15)
    single_loss_penalty = 2.0 * max(0.0, c["max_abs_single_loss"] - 0.10)
    chronic_penalty = 0.10 * float(chronic_failure_count)

    score = return_term + sharpe_term - mdd_penalty - single_loss_penalty - chronic_penalty
    return float(score)


# ---------------------------------------------------------------------------
# Formula v3 — Sortino-anchored with regression penalty
# ---------------------------------------------------------------------------


def compute_cps_v3_sortino(
    per_fold: list[FoldMetrics],
    prev_iter_median_return: float | None,
) -> float | None:
    """CPS v3: ``median_sortino × (1 - max_mdd) - 2.0·max(0, prev_return - this_return)``.

    Explicitly penalizes the "gave back returns to inflate other metrics"
    anti-pattern that we observed in iter 5 A2C (returns dropped 6.0% → 4.9%
    while pass rate climbed 61% → 65%).

    Args:
        per_fold: List of FoldMetrics for one iteration.
        prev_iter_median_return: Median total_return from the previous
            iteration. Pass ``None`` for iter 0 (no baseline available) and
            this function will return ``None``.

    Returns:
        CPS v3 score, or ``None`` when ``prev_iter_median_return`` is ``None``.
    """
    if prev_iter_median_return is None:
        return None
    if not per_fold:
        return 0.0

    c = _components(per_fold)
    drawdown_factor = 1.0 - c["max_mdd"]
    regression_penalty = 2.0 * max(0.0, prev_iter_median_return - c["median_return"])

    score = c["median_sortino"] * drawdown_factor - regression_penalty
    return float(score)


# ---------------------------------------------------------------------------
# Dispatch — compute_all_cps
# ---------------------------------------------------------------------------


def compute_all_cps(
    per_fold: list[FoldMetrics],
    prev_iter_median_return: float | None = None,
    chronic_failure_count: int = 0,
) -> dict[str, Any]:
    """Compute all three CPS formulas plus shared components for one iteration.

    Args:
        per_fold: List of FoldMetrics for one iteration's worth of folds.
        prev_iter_median_return: Median total_return from previous iteration
            (for v3 regression penalty). ``None`` for iter 0.
        chronic_failure_count: Number of chronic-failure folds in this
            iteration's window (for v2 chronic penalty).

    Returns:
        Dict containing:
            cps_v1_multiplicative: float
            cps_v2_additive: float
            cps_v3_sortino: float | None
            components: dict of shared statistical inputs
            worst_fold_number: int (fold with highest MDD)
            worst_fold_mdd: float
            worst_fold_max_single_loss: float | None
            median_return: float
            mean_winner_sharpe: float
            winners_count: int
            return_regression_delta: float (max(0, prev - this))
    """
    components = _components(per_fold)

    # Identify worst fold (highest MDD)
    worst_fold_number: int | None = None
    worst_fold_mdd: float = 0.0
    worst_fold_max_single_loss: float | None = None
    if per_fold:
        worst = max(per_fold, key=lambda f: f["mdd"])
        worst_fold_number = worst["fold_number"]
        worst_fold_mdd = worst["mdd"]
        worst_fold_max_single_loss = worst.get("max_single_loss")

    return_regression_delta = 0.0
    if prev_iter_median_return is not None:
        return_regression_delta = max(0.0, prev_iter_median_return - components["median_return"])

    return {
        "cps_v1_multiplicative": compute_cps_v1_multiplicative(per_fold),
        "cps_v2_additive": compute_cps_v2_additive(per_fold, chronic_failure_count),
        "cps_v3_sortino": compute_cps_v3_sortino(per_fold, prev_iter_median_return),
        "components": components,
        "worst_fold_number": worst_fold_number,
        "worst_fold_mdd": worst_fold_mdd,
        "worst_fold_max_single_loss": worst_fold_max_single_loss,
        "median_return": components["median_return"],
        "mean_winner_sharpe": components["mean_winner_sharpe"],
        "winners_count": int(components["winners"]),
        "return_regression_delta": return_regression_delta,
    }
