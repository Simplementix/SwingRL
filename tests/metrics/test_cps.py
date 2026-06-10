"""Tests for swingrl.metrics.cps — Capital Preservation Score formulas.

CPS-01: Three formulas (v1 multiplicative, v2 additive, v3 sortino-anchored) compute
        a single iteration's capital-preservation score from per-fold metrics.
CPS-02: All formulas must rank a catastrophic iteration below a normal one.
CPS-03: v1 must flag the empirical iter 4 → iter 5 A2C regression as a CPS drop.
CPS-04: v3 returns None for iter 0 (no baseline) and a regression-penalized score otherwise.
CPS-05: compute_all_cps dispatches to all three formulas and returns a stable schema.

TDD RED phase: these tests define expected behavior; implementation in cps.py follows.
"""

from __future__ import annotations

import pytest

from swingrl.metrics.cps import (
    FoldMetrics,
    compute_all_cps,
    compute_cps_v1_multiplicative,
    compute_cps_v2_additive,
    compute_cps_v3_sortino,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fold(
    fold_number: int = 0,
    sharpe: float = 2.0,
    mdd: float = 0.05,
    total_return: float = 0.08,
    profit_factor: float = 3.0,
    win_rate: float = 0.65,
    total_trades: int = 400,
    sortino: float = 2.5,
    max_single_loss: float = -0.04,
    overfitting_class: str = "healthy",
    is_control_fold: bool = False,
) -> FoldMetrics:
    """Construct a FoldMetrics dict with sensible defaults for testing."""
    return {
        "fold_number": fold_number,
        "sharpe": sharpe,
        "mdd": mdd,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "sortino": sortino,
        "max_single_loss": max_single_loss,
        "overfitting_class": overfitting_class,
        "is_control_fold": is_control_fold,
    }


def make_normal_iteration() -> list[FoldMetrics]:
    """Construct a synthetic 'normal' iteration: 6 healthy folds, 2 marginal, 2 reject."""
    folds: list[FoldMetrics] = []
    # 6 winners
    for i in range(6):
        folds.append(
            make_fold(
                fold_number=i,
                sharpe=3.0 + i * 0.2,
                mdd=0.03 + i * 0.005,
                total_return=0.08 + i * 0.01,
                sortino=4.0,
                overfitting_class="healthy",
            )
        )
    # 2 marginal
    for i in range(6, 8):
        folds.append(
            make_fold(
                fold_number=i,
                sharpe=1.0,
                mdd=0.07,
                total_return=0.04,
                sortino=1.2,
                overfitting_class="marginal",
            )
        )
    # 2 reject
    for i in range(8, 10):
        folds.append(
            make_fold(
                fold_number=i,
                sharpe=-0.5,
                mdd=0.18,
                total_return=-0.02,
                sortino=-0.4,
                overfitting_class="reject",
            )
        )
    return folds


def make_catastrophic_iteration() -> list[FoldMetrics]:
    """Construct a synthetic catastrophic iteration: one fold has 30%+ MDD."""
    folds = make_normal_iteration()
    # Inject a fold with catastrophic loss
    folds.append(
        make_fold(
            fold_number=10,
            sharpe=-2.5,
            mdd=0.35,
            total_return=-0.20,
            sortino=-3.0,
            max_single_loss=-0.15,
            overfitting_class="reject",
        )
    )
    return folds


# ---------------------------------------------------------------------------
# CPS v1 — Multiplicative
# ---------------------------------------------------------------------------


class TestCpsV1Multiplicative:
    """CPS-01: v1 = median_return × (1 - max_mdd)² × tanh(mean_winner_sharpe/2) × (winners/total)."""

    def test_normal_iteration_yields_positive_score(self) -> None:
        """A healthy iteration with 6/10 winners produces a positive CPS."""
        folds = make_normal_iteration()
        score = compute_cps_v1_multiplicative(folds)
        assert score > 0
        assert score < 0.1  # bounded — not unrealistically high

    def test_zero_return_yields_zero(self) -> None:
        """If all folds have zero total_return (median = 0), CPS v1 must be 0."""
        folds = [make_fold(fold_number=i, total_return=0.0) for i in range(5)]
        score = compute_cps_v1_multiplicative(folds)
        assert score == 0.0

    def test_high_drawdown_collapses_score(self) -> None:
        """A 50% max drawdown should collapse v1 to ≤25% of the no-drawdown version."""
        no_dd = [make_fold(fold_number=i, mdd=0.0, total_return=0.05) for i in range(5)]
        with_dd = [make_fold(fold_number=i, mdd=0.50, total_return=0.05) for i in range(5)]
        score_no_dd = compute_cps_v1_multiplicative(no_dd)
        score_with_dd = compute_cps_v1_multiplicative(with_dd)
        # (1-0.5)² = 0.25 → with_dd is at most 25% of no_dd
        assert score_with_dd <= score_no_dd * 0.26
        assert score_with_dd > 0  # still positive (not zero)

    def test_zero_winners_yields_zero(self) -> None:
        """If no folds are 'healthy', the winners/total factor is 0 and CPS v1 = 0."""
        folds = [
            make_fold(fold_number=i, overfitting_class="reject", sharpe=-1.0) for i in range(5)
        ]
        score = compute_cps_v1_multiplicative(folds)
        assert score == 0.0

    def test_negative_median_return_yields_negative_score(self) -> None:
        """A negative median return should produce a negative CPS (v1 preserves sign)."""
        # 6 winners with positive return, but 5 losers dragging median negative
        folds = [
            make_fold(fold_number=i, total_return=-0.05, overfitting_class="reject", sharpe=-0.5)
            for i in range(7)
        ]
        # Add 4 winners but they don't shift the median
        for i in range(7, 11):
            folds.append(make_fold(fold_number=i, total_return=0.10, sharpe=3.0))
        score = compute_cps_v1_multiplicative(folds)
        assert score < 0

    def test_iter5_a2c_regression_visible_empirically(self) -> None:
        """CPS-03: v1 must show iter 5 A2C < iter 4 A2C on actual empirical aggregates.

        From pg16 audit (2026-04-06):
        - Iter 4 A2C equity: 14/23 winners, avg return 6.02%, avg passing sharpe 3.32, avg mdd 7.25%
        - Iter 5 A2C equity: 15/23 winners, avg return 4.90%, avg passing sharpe 2.58, avg mdd 7.41%

        v1 should drop from iter 4 to iter 5 despite the +1 winner — capital
        preservation worsened (returns gave back, conviction dropped).
        """
        # Build synthetic folds matching the empirical aggregates
        # Iter 4: 14 winners × sharpe 3.32, 9 losers × sharpe 0.5; median return 0.0602; max_mdd 0.145
        iter4 = [
            make_fold(
                fold_number=i, sharpe=3.32, mdd=0.04, total_return=0.07, overfitting_class="healthy"
            )
            for i in range(14)
        ]
        for i in range(14, 23):
            iter4.append(
                make_fold(
                    fold_number=i,
                    sharpe=0.5,
                    mdd=0.145,
                    total_return=0.05,
                    overfitting_class="reject",
                )
            )

        # Iter 5: 15 winners × sharpe 2.58, 8 losers × sharpe 0.6; median return 0.049; max_mdd 0.146
        iter5 = [
            make_fold(
                fold_number=i,
                sharpe=2.58,
                mdd=0.04,
                total_return=0.055,
                overfitting_class="healthy",
            )
            for i in range(15)
        ]
        for i in range(15, 23):
            iter5.append(
                make_fold(
                    fold_number=i,
                    sharpe=0.6,
                    mdd=0.146,
                    total_return=0.04,
                    overfitting_class="reject",
                )
            )

        score_iter4 = compute_cps_v1_multiplicative(iter4)
        score_iter5 = compute_cps_v1_multiplicative(iter5)
        # Iter 5 must be lower despite +1 winner
        assert score_iter5 < score_iter4, (
            f"v1 failed to flag iter 5 A2C regression: iter4={score_iter4:.5f} iter5={score_iter5:.5f}"
        )

    def test_empty_fold_list_returns_zero(self) -> None:
        """Defensive: empty input → 0.0, not NaN or exception."""
        assert compute_cps_v1_multiplicative([]) == 0.0


# ---------------------------------------------------------------------------
# CPS v2 — Additive with hard penalties
# ---------------------------------------------------------------------------


class TestCpsV2Additive:
    """CPS-01: v2 = 0.5·median_return + 0.3·(mean_winner_sharpe/4) - penalties."""

    def test_normal_iteration_positive(self) -> None:
        """A normal iteration with 6/10 winners produces a positive v2 score."""
        folds = make_normal_iteration()
        score = compute_cps_v2_additive(folds, chronic_failure_count=0)
        assert score > 0

    def test_penalizes_catastrophic_single_loss(self) -> None:
        """A fold with max_single_loss < -0.10 triggers the 2.0× penalty."""
        clean = [make_fold(max_single_loss=-0.05) for _ in range(5)]
        catastrophic = [make_fold(max_single_loss=-0.05) for _ in range(4)]
        catastrophic.append(make_fold(max_single_loss=-0.20))
        score_clean = compute_cps_v2_additive(clean, chronic_failure_count=0)
        score_cat = compute_cps_v2_additive(catastrophic, chronic_failure_count=0)
        # Penalty: 2.0 × (0.20 - 0.10) = 0.20
        assert score_cat == pytest.approx(score_clean - 0.20, abs=1e-6)

    def test_penalizes_max_drawdown_above_15pct(self) -> None:
        """max_mdd > 0.15 triggers the 1.0× penalty."""
        clean = [make_fold(mdd=0.10) for _ in range(5)]
        with_dd = [make_fold(mdd=0.10) for _ in range(4)]
        with_dd.append(make_fold(mdd=0.25))  # 0.10 over the 0.15 threshold
        score_clean = compute_cps_v2_additive(clean, chronic_failure_count=0)
        score_dd = compute_cps_v2_additive(with_dd, chronic_failure_count=0)
        assert score_dd == pytest.approx(score_clean - 0.10, abs=1e-6)

    def test_penalizes_chronic_failure_count(self) -> None:
        """Each chronic failure subtracts 0.10 from the score."""
        folds = make_normal_iteration()
        score_no_chronic = compute_cps_v2_additive(folds, chronic_failure_count=0)
        score_5_chronic = compute_cps_v2_additive(folds, chronic_failure_count=5)
        assert score_5_chronic == pytest.approx(score_no_chronic - 0.50, abs=1e-6)


# ---------------------------------------------------------------------------
# CPS v3 — Sortino-anchored with regression penalty
# ---------------------------------------------------------------------------


class TestCpsV3Sortino:
    """CPS-04: v3 = median_sortino × (1 - max_mdd) - 2 × max(0, prev_return - this_return)."""

    def test_handles_iter0_no_baseline_returns_none(self) -> None:
        """When prev_iter_median_return is None, v3 returns None (undefined for iter 0)."""
        folds = make_normal_iteration()
        result = compute_cps_v3_sortino(folds, prev_iter_median_return=None)
        assert result is None

    def test_normal_with_baseline_yields_positive(self) -> None:
        """A healthy iteration with no return regression yields a positive score."""
        folds = make_normal_iteration()
        result = compute_cps_v3_sortino(folds, prev_iter_median_return=0.05)
        assert result is not None
        assert result > 0

    def test_penalizes_return_regression(self) -> None:
        """Dropping median return relative to prev iteration triggers 2× penalty.

        Synthetic: prev_return = 0.08, this_return = 0.06 → regression = 0.02 → penalty = 0.04
        """
        # Build folds where median total_return = 0.06
        folds = [make_fold(fold_number=i, total_return=0.06, sortino=2.0) for i in range(5)]
        with_baseline = compute_cps_v3_sortino(folds, prev_iter_median_return=0.08)
        no_baseline_change = compute_cps_v3_sortino(folds, prev_iter_median_return=0.06)
        assert with_baseline is not None and no_baseline_change is not None
        # The 0.02pt return regression should subtract 0.04
        assert with_baseline == pytest.approx(no_baseline_change - 0.04, abs=1e-6)

    def test_no_penalty_when_returns_improved(self) -> None:
        """If this_return >= prev_return, no penalty applied."""
        folds = [make_fold(fold_number=i, total_return=0.10, sortino=2.0) for i in range(5)]
        improved = compute_cps_v3_sortino(folds, prev_iter_median_return=0.05)
        flat = compute_cps_v3_sortino(folds, prev_iter_median_return=0.10)
        assert improved == flat  # max(0, 0.05 - 0.10) = 0


# ---------------------------------------------------------------------------
# compute_all_cps — dispatch + schema
# ---------------------------------------------------------------------------


class TestComputeAllCps:
    """CPS-05: compute_all_cps returns a stable dict schema with all three scores + components."""

    def test_returns_required_keys(self) -> None:
        """The result dict must contain all expected keys."""
        folds = make_normal_iteration()
        result = compute_all_cps(folds, prev_iter_median_return=0.05)
        required_keys = {
            "cps_v1_multiplicative",
            "cps_v2_additive",
            "cps_v3_sortino",
            "components",
            "worst_fold_number",
            "worst_fold_mdd",
            "worst_fold_max_single_loss",
            "median_return",
            "mean_winner_sharpe",
            "winners_count",
            "return_regression_delta",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_iter0_has_null_v3(self) -> None:
        """When prev_iter_median_return=None, cps_v3_sortino must be None."""
        folds = make_normal_iteration()
        result = compute_all_cps(folds, prev_iter_median_return=None)
        assert result["cps_v3_sortino"] is None

    def test_components_breakdown_present(self) -> None:
        """The components dict must include the inputs used by the formulas."""
        folds = make_normal_iteration()
        result = compute_all_cps(folds, prev_iter_median_return=0.05)
        comp = result["components"]
        for key in ("median_return", "max_mdd", "mean_winner_sharpe", "winners", "total"):
            assert key in comp

    def test_worst_fold_identified(self) -> None:
        """worst_fold_number = fold with the highest MDD."""
        folds = make_catastrophic_iteration()
        result = compute_all_cps(folds, prev_iter_median_return=0.05)
        # Fold 10 has mdd=0.35 — the highest
        assert result["worst_fold_number"] == 10
        assert result["worst_fold_mdd"] == pytest.approx(0.35, abs=1e-6)

    def test_chronic_failure_count_passthrough(self) -> None:
        """chronic_failure_count is taken as a parameter and threaded into v2."""
        folds = make_normal_iteration()
        result_zero = compute_all_cps(folds, prev_iter_median_return=0.05, chronic_failure_count=0)
        result_five = compute_all_cps(folds, prev_iter_median_return=0.05, chronic_failure_count=5)
        # v2 should drop by 0.5 (5 × 0.10)
        assert result_five["cps_v2_additive"] == pytest.approx(
            result_zero["cps_v2_additive"] - 0.50, abs=1e-6
        )


# ---------------------------------------------------------------------------
# Cross-formula consistency
# ---------------------------------------------------------------------------


class TestCpsFormulasAgreeOnExtremeCases:
    """CPS-02: All three formulas must rank a catastrophic iteration BELOW a normal one."""

    def test_all_formulas_rank_catastrophic_below_normal(self) -> None:
        """Sanity: catastrophic iteration loses on every formula."""
        normal = make_normal_iteration()
        catastrophic = make_catastrophic_iteration()
        prev_baseline = 0.05  # arbitrary positive baseline

        n_v1 = compute_cps_v1_multiplicative(normal)
        n_v2 = compute_cps_v2_additive(normal, chronic_failure_count=0)
        n_v3 = compute_cps_v3_sortino(normal, prev_iter_median_return=prev_baseline)

        c_v1 = compute_cps_v1_multiplicative(catastrophic)
        c_v2 = compute_cps_v2_additive(catastrophic, chronic_failure_count=0)
        c_v3 = compute_cps_v3_sortino(catastrophic, prev_iter_median_return=prev_baseline)

        assert c_v1 < n_v1, f"v1 failed: catastrophic={c_v1} >= normal={n_v1}"
        assert c_v2 < n_v2, f"v2 failed: catastrophic={c_v2} >= normal={n_v2}"
        assert n_v3 is not None and c_v3 is not None
        assert c_v3 < n_v3, f"v3 failed: catastrophic={c_v3} >= normal={n_v3}"
