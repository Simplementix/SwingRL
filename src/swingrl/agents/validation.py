"""Validation gates and overfitting detection for trained RL agents.

Provides threshold-based validation gates (Sharpe, MDD, Profit Factor,
overfitting gap) and an overfitting diagnostic classifier. Used by the
training pipeline to accept/reject models before promotion.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclasses.dataclass
class GateResult:
    """Result of validation gate checks.

    Args:
        passed: True if all gates passed.
        failures: List of gate names that failed.
        details: Dict mapping gate name to threshold/value info.
    """

    passed: bool
    failures: list[str]
    details: dict[str, Any]


def diagnose_overfitting(
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
) -> dict[str, Any]:
    """Classify overfitting risk from in-sample vs out-of-sample Sharpe gap.

    Gap is computed as 1 - (OOS / IS). Thresholds:
    - < 0.20: healthy
    - 0.20 - 0.50: marginal
    - > 0.50: reject

    Args:
        in_sample_sharpe: Sharpe ratio on training data.
        out_of_sample_sharpe: Sharpe ratio on validation/test data.

    Returns:
        Dict with 'gap' (float) and 'classification' (str).
    """
    if in_sample_sharpe <= 0:
        log.warning(
            "overfitting_diagnosis_reject",
            reason="in_sample_sharpe_non_positive",
            in_sample_sharpe=in_sample_sharpe,
        )
        return {"gap": float("inf"), "classification": "reject"}

    gap = 1.0 - (out_of_sample_sharpe / in_sample_sharpe)

    if gap < 0.20:
        classification = "healthy"
    elif gap <= 0.50:
        classification = "marginal"
    else:
        classification = "reject"

    log.info(
        "overfitting_diagnosed",
        gap=round(gap, 4),
        classification=classification,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
    )

    return {"gap": gap, "classification": classification}


def check_validation_gates(
    sharpe: float,
    mdd: float,
    profit_factor: float,
    overfit_gap: float,
) -> GateResult:
    """Check whether a model passes all four validation gates.

    Gate thresholds:
    - sharpe > 0.7
    - mdd < 0.15
    - profit_factor > 1.5
    - overfit_gap < 0.20

    Args:
        sharpe: Annualized Sharpe ratio.
        mdd: Maximum drawdown as a positive fraction.
        profit_factor: Sum of wins / sum of losses.
        overfit_gap: 1 - (OOS Sharpe / IS Sharpe).

    Returns:
        GateResult with pass/fail status, failure list, and threshold details.
    """
    failures: list[str] = []
    details: dict[str, Any] = {}

    # Gate 1: Sharpe > 0.7
    sharpe_passed = sharpe > 0.7
    details["sharpe"] = {"value": sharpe, "threshold": 0.7, "passed": sharpe_passed}
    if not sharpe_passed:
        failures.append("sharpe")

    # Gate 2: MDD < 0.15
    mdd_passed = mdd < 0.15
    details["mdd"] = {"value": mdd, "threshold": 0.15, "passed": mdd_passed}
    if not mdd_passed:
        failures.append("mdd")

    # Gate 3: Profit Factor > 1.5
    pf_passed = profit_factor > 1.5
    details["profit_factor"] = {
        "value": profit_factor,
        "threshold": 1.5,
        "passed": pf_passed,
    }
    if not pf_passed:
        failures.append("profit_factor")

    # Gate 4: Overfit Gap < 0.20
    og_passed = overfit_gap < 0.20
    details["overfit_gap"] = {
        "value": overfit_gap,
        "threshold": 0.20,
        "passed": og_passed,
    }
    if not og_passed:
        failures.append("overfit_gap")

    passed = len(failures) == 0

    log.info(
        "validation_gates_checked",
        passed=passed,
        failures=failures,
        sharpe=sharpe,
        mdd=mdd,
        profit_factor=profit_factor,
        overfit_gap=overfit_gap,
    )

    return GateResult(passed=passed, failures=failures, details=details)
