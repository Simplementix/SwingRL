"""Tests for swingrl.reporting.iteration_report.

REPORT-01: load_iteration_history reads iteration_results in chronological order.
REPORT-02: load_fold_history dedupes iter 1 by latest created_at (post-fix wins).
REPORT-03: compute_iter_deltas flags regression on cps drop, return drop, mdd jump.
REPORT-04: compute_iter_deltas correctly classifies the empirical iter 4 → iter 5
            A2C scenario as a regression (return -1.1pp, sharpe -22%).
REPORT-05: compute_iter_deltas correctly classifies iter 5 PPO as lateral
            (cps_v1_delta within ±0.003).
REPORT-06: detect_chronic_failures returns folds that failed in 4+ of last 6 iters.
REPORT-07: detect_protected_winners returns folds with sharpe>4 in 4+ of last 6 iters.
REPORT-08: format_iteration_summary renders a markdown table for Discord/CLI use.

Pure-function tests use synthetic DataFrames. Live DB tests skip without
DATABASE_URL set.
"""

from __future__ import annotations

import os

import pandas as pd
import psycopg
import pytest

from swingrl.reporting.iteration_report import (
    _BACKTEST_INITIAL_CAPITAL,
    REGRESSION_RETURN_THRESHOLD,
    REGRESSION_WORST_MDD_THRESHOLD,
    _fold_row_to_metrics,
    compute_iter_deltas,
    compute_iteration_cps,
    detect_chronic_failures,
    detect_protected_winners,
    format_iteration_summary,
    load_fold_history,
    load_iteration_history,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic DataFrames mirroring the Postgres schema
# ---------------------------------------------------------------------------


def make_iteration_history_df() -> pd.DataFrame:
    """Build a synthetic iteration_results DataFrame for iter 0-5 equity."""
    return pd.DataFrame(
        [
            # Each row = one (iteration_number, environment) iteration_results row.
            {
                "iteration_number": 0,
                "environment": "equity",
                "ensemble_sharpe": 1.69,
                "ensemble_mdd": 0.075,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 11,
                "median_return": 0.0829,
                "mean_winner_sharpe": 4.44,
                "worst_fold_mdd": 0.151,
                "cps_v1_multiplicative": 0.0294,
                "cps_v2_additive": -0.10,
                "cps_v3_sortino": None,  # iter 0 has no baseline
                "chronic_failure_count": 0,
                "memory_enabled": True,
            },
            {
                "iteration_number": 1,
                "environment": "equity",
                "ensemble_sharpe": 1.90,
                "ensemble_mdd": 0.074,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 14,
                "median_return": 0.0766,
                "mean_winner_sharpe": 3.83,
                "worst_fold_mdd": 0.150,
                "cps_v1_multiplicative": 0.0310,
                "cps_v2_additive": -0.05,
                "cps_v3_sortino": 0.040,
                "chronic_failure_count": 2,
                "memory_enabled": True,
            },
            {
                "iteration_number": 2,
                "environment": "equity",
                "ensemble_sharpe": 2.01,
                "ensemble_mdd": 0.072,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 12,
                "median_return": 0.0958,
                "mean_winner_sharpe": 5.09,
                "worst_fold_mdd": 0.149,
                "cps_v1_multiplicative": 0.0345,
                "cps_v2_additive": -0.02,
                "cps_v3_sortino": 0.060,
                "chronic_failure_count": 4,
                "memory_enabled": True,
            },
            {
                "iteration_number": 3,
                "environment": "equity",
                "ensemble_sharpe": 1.89,
                "ensemble_mdd": 0.073,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 13,
                "median_return": 0.0822,
                "mean_winner_sharpe": 3.83,
                "worst_fold_mdd": 0.150,
                "cps_v1_multiplicative": 0.0319,
                "cps_v2_additive": -0.03,
                "cps_v3_sortino": 0.055,
                "chronic_failure_count": 5,
                "memory_enabled": True,
            },
            {
                "iteration_number": 4,
                "environment": "equity",
                "ensemble_sharpe": 2.07,
                "ensemble_mdd": 0.073,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 14,
                "median_return": 0.0854,
                "mean_winner_sharpe": 4.00,
                "worst_fold_mdd": 0.149,
                "cps_v1_multiplicative": 0.0334,
                "cps_v2_additive": -0.01,
                "cps_v3_sortino": 0.063,
                "chronic_failure_count": 5,
                "memory_enabled": True,
            },
            {
                # The empirical iter 5 A2C-style regression: more winners but
                # returns gave back and mean_winner_sharpe collapsed.
                "iteration_number": 5,
                "environment": "equity",
                "ensemble_sharpe": 1.75,
                "ensemble_mdd": 0.074,
                "gate_passed": True,
                "total_folds": 23,
                "winners_count": 15,
                "median_return": 0.0490,
                "mean_winner_sharpe": 2.58,
                "worst_fold_mdd": 0.150,
                "cps_v1_multiplicative": 0.0210,
                "cps_v2_additive": -0.05,
                "cps_v3_sortino": 0.040,
                "chronic_failure_count": 5,
                "memory_enabled": True,
            },
        ]
    )


def make_fold_history_df() -> pd.DataFrame:
    """Build a synthetic per-fold backtest_results DataFrame for chronic-failure tests.

    Mirrors the empirical pattern: folds 2, 4, 7, 13, 15 are chronic failures
    (overfitting_class='reject' in 4+ of 6 iters); fold 1 is a protected
    winner (sharpe > 4.0 in 4+ of 6 iters).
    """
    rows: list[dict] = []
    for it in range(6):
        for fold in range(20):
            if fold in (2, 4, 7, 13, 15):
                # Chronic failure
                rows.append(
                    {
                        "iteration_number": it,
                        "environment": "equity",
                        "algorithm": "ppo",
                        "fold_number": fold,
                        "sharpe": -0.5,
                        "mdd": 0.20,
                        "overfitting_class": "reject",
                    }
                )
            elif fold == 1:
                # Protected winner
                rows.append(
                    {
                        "iteration_number": it,
                        "environment": "equity",
                        "algorithm": "ppo",
                        "fold_number": fold,
                        "sharpe": 5.5,
                        "mdd": 0.02,
                        "overfitting_class": "healthy",
                    }
                )
            else:
                # Normal fold
                rows.append(
                    {
                        "iteration_number": it,
                        "environment": "equity",
                        "algorithm": "ppo",
                        "fold_number": fold,
                        "sharpe": 2.5,
                        "mdd": 0.05,
                        "overfitting_class": "healthy",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_iter_deltas
# ---------------------------------------------------------------------------


class TestComputeIterDeltas:
    """REPORT-03/04/05: per-iteration deltas and regression flagging."""

    def test_returns_one_row_per_iteration(self) -> None:
        """Output has the same row count as input (one delta row per iter)."""
        history = make_iteration_history_df()
        deltas = compute_iter_deltas(history)
        assert len(deltas) == len(history)

    def test_iter0_has_null_deltas(self) -> None:
        """Iteration 0 has no prior; all delta columns are NaN/None."""
        history = make_iteration_history_df()
        deltas = compute_iter_deltas(history)
        iter0 = deltas[deltas["iteration_number"] == 0].iloc[0]
        assert pd.isna(iter0["cps_v1_delta"])
        assert pd.isna(iter0["return_delta"])
        assert pd.isna(iter0["worst_mdd_delta"])
        assert not bool(iter0["regression_flag"])  # no delta = no regression

    def test_iter5_a2c_regression_flagged(self) -> None:
        """REPORT-04: iter 5 A2C-style scenario must set regression_flag=True.

        Iter 5 vs iter 4: median_return 0.0854 → 0.0490 (-0.0364, > 0.02 threshold).
        Multiple regression dimensions tripped.
        """
        history = make_iteration_history_df()
        deltas = compute_iter_deltas(history)
        iter5 = deltas[deltas["iteration_number"] == 5].iloc[0]
        assert bool(iter5["regression_flag"])
        # Specific dimensional checks
        assert iter5["return_delta"] == pytest.approx(0.0490 - 0.0854, abs=1e-4)
        assert iter5["cps_v1_delta"] == pytest.approx(0.0210 - 0.0334, abs=1e-4)

    def test_regression_flag_on_return_drop_alone(self) -> None:
        """A return drop > REGRESSION_RETURN_THRESHOLD trips the flag even if CPS up."""
        history = pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "median_return": 0.10,
                    "cps_v1_multiplicative": 0.03,
                    "worst_fold_mdd": 0.10,
                },
                {
                    "iteration_number": 1,
                    "environment": "equity",
                    "median_return": 0.10 - REGRESSION_RETURN_THRESHOLD - 0.001,  # just over
                    "cps_v1_multiplicative": 0.04,  # higher cps
                    "worst_fold_mdd": 0.10,
                },
            ]
        )
        deltas = compute_iter_deltas(history)
        iter1 = deltas[deltas["iteration_number"] == 1].iloc[0]
        assert bool(iter1["regression_flag"])

    def test_regression_flag_on_worst_mdd_jump_alone(self) -> None:
        """A worst-fold MDD increase > REGRESSION_WORST_MDD_THRESHOLD trips flag alone."""
        history = pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "median_return": 0.05,
                    "cps_v1_multiplicative": 0.02,
                    "worst_fold_mdd": 0.10,
                },
                {
                    "iteration_number": 1,
                    "environment": "equity",
                    "median_return": 0.06,  # better return
                    "cps_v1_multiplicative": 0.03,  # better cps
                    "worst_fold_mdd": 0.10 + REGRESSION_WORST_MDD_THRESHOLD + 0.001,
                },
            ]
        )
        deltas = compute_iter_deltas(history)
        iter1 = deltas[deltas["iteration_number"] == 1].iloc[0]
        assert bool(iter1["regression_flag"])

    def test_no_regression_when_all_metrics_improve(self) -> None:
        """When CPS up, return up, MDD down → regression_flag=False."""
        history = pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "median_return": 0.05,
                    "cps_v1_multiplicative": 0.02,
                    "worst_fold_mdd": 0.10,
                },
                {
                    "iteration_number": 1,
                    "environment": "equity",
                    "median_return": 0.07,
                    "cps_v1_multiplicative": 0.04,
                    "worst_fold_mdd": 0.08,
                },
            ]
        )
        deltas = compute_iter_deltas(history)
        iter1 = deltas[deltas["iteration_number"] == 1].iloc[0]
        assert not bool(iter1["regression_flag"])

    def test_handles_null_cps_columns(self) -> None:
        """Pre-backfill rows have NULL cps_v1; deltas should be NaN, no exception."""
        history = pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "median_return": 0.05,
                    "cps_v1_multiplicative": None,
                    "worst_fold_mdd": 0.10,
                },
                {
                    "iteration_number": 1,
                    "environment": "equity",
                    "median_return": 0.06,
                    "cps_v1_multiplicative": None,
                    "worst_fold_mdd": 0.10,
                },
            ]
        )
        # Should not raise
        deltas = compute_iter_deltas(history)
        iter1 = deltas[deltas["iteration_number"] == 1].iloc[0]
        assert pd.isna(iter1["cps_v1_delta"])

    def test_per_environment_deltas_independent(self) -> None:
        """Iterations from different environments do not contaminate each other's deltas."""
        history = pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "median_return": 0.05,
                    "cps_v1_multiplicative": 0.02,
                    "worst_fold_mdd": 0.10,
                },
                {
                    "iteration_number": 0,
                    "environment": "crypto",
                    "median_return": 0.20,
                    "cps_v1_multiplicative": 0.08,
                    "worst_fold_mdd": 0.20,
                },
                {
                    "iteration_number": 1,
                    "environment": "equity",
                    "median_return": 0.06,
                    "cps_v1_multiplicative": 0.025,
                    "worst_fold_mdd": 0.10,
                },
            ]
        )
        deltas = compute_iter_deltas(history)
        equity_iter1 = deltas[
            (deltas["iteration_number"] == 1) & (deltas["environment"] == "equity")
        ].iloc[0]
        # Equity iter1 delta should compare against equity iter0, NOT crypto iter0
        assert equity_iter1["return_delta"] == pytest.approx(0.06 - 0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# detect_chronic_failures
# ---------------------------------------------------------------------------


class TestDetectChronicFailures:
    """REPORT-06: identifies folds that failed in 4+ of last 6 iters."""

    def test_returns_dict_keyed_by_environment(self) -> None:
        fold_history = make_fold_history_df()
        chronics = detect_chronic_failures(fold_history)
        assert "equity" in chronics

    def test_identifies_expected_chronic_folds(self) -> None:
        """Synthetic fixture has folds 2, 4, 7, 13, 15 as chronic failures."""
        fold_history = make_fold_history_df()
        chronics = detect_chronic_failures(fold_history)
        assert sorted(chronics["equity"]) == [2, 4, 7, 13, 15]

    def test_excludes_non_chronic_folds(self) -> None:
        """Folds that pass in 4+ iters are NOT marked chronic."""
        fold_history = make_fold_history_df()
        chronics = detect_chronic_failures(fold_history)
        # Fold 1 passes 6/6 → not chronic
        assert 1 not in chronics["equity"]
        # Fold 5 passes 6/6 (normal) → not chronic
        assert 5 not in chronics["equity"]

    def test_threshold_window_configurable(self) -> None:
        """min_fails parameter controls strictness."""
        fold_history = make_fold_history_df()
        # min_fails=6 (must fail in ALL 6) — same set in our fixture
        strict = detect_chronic_failures(fold_history, min_fails=6)
        assert sorted(strict["equity"]) == [2, 4, 7, 13, 15]
        # min_fails=7 — impossible (only 6 iters); empty
        impossible = detect_chronic_failures(fold_history, min_fails=7)
        assert impossible.get("equity", []) == []


# ---------------------------------------------------------------------------
# detect_protected_winners
# ---------------------------------------------------------------------------


class TestDetectProtectedWinners:
    """REPORT-07: identifies folds with sharpe > threshold in 4+ of last 6 iters."""

    def test_identifies_fold_1_as_protected(self) -> None:
        """Synthetic fixture: fold 1 has sharpe=5.5 in all 6 iters."""
        fold_history = make_fold_history_df()
        winners = detect_protected_winners(fold_history)
        assert 1 in winners["equity"]

    def test_excludes_normal_folds(self) -> None:
        """Folds with sharpe ~2.5 are not winners."""
        fold_history = make_fold_history_df()
        winners = detect_protected_winners(fold_history)
        assert 5 not in winners["equity"]

    def test_threshold_configurable(self) -> None:
        """sharpe_threshold parameter changes the bar."""
        fold_history = make_fold_history_df()
        # Lowering threshold to 2.0 should catch the 'normal' folds too
        permissive = detect_protected_winners(fold_history, sharpe_threshold=2.0)
        assert 5 in permissive["equity"]
        # Raising to 6.0 should drop fold 1 (sharpe=5.5)
        strict = detect_protected_winners(fold_history, sharpe_threshold=6.0)
        assert 1 not in strict.get("equity", [])


# ---------------------------------------------------------------------------
# format_iteration_summary
# ---------------------------------------------------------------------------


class TestFormatIterationSummary:
    """REPORT-08: markdown summary for Discord/CLI."""

    def test_contains_iteration_numbers(self) -> None:
        """Each iteration number 0-5 must appear in the rendered table.

        Allow for an optional regression marker prefix (⚠) before the digit.
        """
        history = make_iteration_history_df()
        out = format_iteration_summary(history)
        for it in range(6):
            # Expected forms: "| 0 |", "| ⚠ 5 |", "| 3 |", etc.
            # Just check the digit appears as a column value (followed by " |").
            assert f" {it} |" in out, f"Iteration {it} not found in summary output"

    def test_marks_iter5_regression(self) -> None:
        """Iter 5 row should be visually distinct (e.g., REGRESSION marker)."""
        history = make_iteration_history_df()
        out = format_iteration_summary(history)
        assert "REGRESSION" in out.upper() or "⚠" in out or "regression" in out.lower()

    def test_returns_string(self) -> None:
        history = make_iteration_history_df()
        out = format_iteration_summary(history)
        assert isinstance(out, str)
        assert len(out) > 0


# ---------------------------------------------------------------------------
# _fold_row_to_metrics — unit conversion boundary
# ---------------------------------------------------------------------------


class TestFoldRowToMetrics:
    """REPORT-10: pg16 → FoldMetrics conversion handles dollar→fraction unit fix."""

    def test_max_single_loss_converted_dollars_to_fraction(self) -> None:
        """A -$3713 max_single_loss must convert to -0.03713 fraction (vs $100K)."""
        row = pd.Series(
            {
                "fold_number": 7,
                "sharpe": -1.786,
                "mdd": 0.3823,
                "total_return": -0.2413,
                "profit_factor": 0.4,
                "win_rate": 0.55,
                "total_trades": 253,
                "sortino": -2.5,
                "max_single_loss": -3713.82,
                "overfitting_class": "reject",
                "is_control_fold": False,
            }
        )
        metrics = _fold_row_to_metrics(row)
        assert metrics["max_single_loss"] == pytest.approx(
            -3713.82 / _BACKTEST_INITIAL_CAPITAL, abs=1e-9
        )
        assert metrics["max_single_loss"] == pytest.approx(-0.0371382, abs=1e-7)

    def test_null_max_single_loss_passes_through(self) -> None:
        """A NULL max_single_loss column → None in FoldMetrics."""
        row = pd.Series(
            {
                "fold_number": 0,
                "sharpe": 1.5,
                "mdd": 0.04,
                "total_return": 0.06,
                "profit_factor": 2.0,
                "win_rate": 0.6,
                "total_trades": 200,
                "sortino": 1.8,
                "max_single_loss": None,
                "overfitting_class": "healthy",
                "is_control_fold": False,
            }
        )
        metrics = _fold_row_to_metrics(row)
        assert metrics["max_single_loss"] is None

    def test_total_return_and_mdd_unchanged(self) -> None:
        """total_return and mdd are already fractions; no conversion applied."""
        row = pd.Series(
            {
                "fold_number": 1,
                "sharpe": 6.4,
                "mdd": 0.013,
                "total_return": 0.0949,
                "profit_factor": 5.0,
                "win_rate": 0.85,
                "total_trades": 220,
                "sortino": 8.0,
                "max_single_loss": -120.0,
                "overfitting_class": "healthy",
                "is_control_fold": True,
            }
        )
        metrics = _fold_row_to_metrics(row)
        assert metrics["mdd"] == pytest.approx(0.013, abs=1e-9)
        assert metrics["total_return"] == pytest.approx(0.0949, abs=1e-9)
        assert metrics["is_control_fold"] is True


# ---------------------------------------------------------------------------
# compute_iteration_cps — per-algo aggregation
# ---------------------------------------------------------------------------


def make_per_algo_fold_history() -> pd.DataFrame:
    """Build a fold_history with 3 algos × 5 folds for one (env, iter), enough
    to exercise per-algo CPS aggregation."""
    rows: list[dict] = []
    # PPO: 5 healthy folds, mean sharpe ~3.0
    for fold in range(5):
        rows.append(
            {
                "iteration_number": 1,
                "environment": "equity",
                "algorithm": "ppo",
                "fold_number": fold,
                "sharpe": 3.0,
                "sortino": 4.0,
                "calmar": 2.0,
                "mdd": 0.04,
                "total_return": 0.08,
                "profit_factor": 4.0,
                "win_rate": 0.7,
                "total_trades": 400,
                "overfitting_class": "healthy",
                "max_single_loss": -0.04,
                "is_control_fold": False,
            }
        )
    # A2C: 5 healthy folds but lower sharpe (~2.0)
    for fold in range(5):
        rows.append(
            {
                "iteration_number": 1,
                "environment": "equity",
                "algorithm": "a2c",
                "fold_number": fold,
                "sharpe": 2.0,
                "sortino": 2.5,
                "calmar": 1.5,
                "mdd": 0.05,
                "total_return": 0.06,
                "profit_factor": 3.0,
                "win_rate": 0.65,
                "total_trades": 350,
                "overfitting_class": "healthy",
                "max_single_loss": -0.05,
                "is_control_fold": False,
            }
        )
    # SAC: mix of 3 healthy + 2 reject — only ~3 winners
    for fold in range(5):
        cls = "healthy" if fold < 3 else "reject"
        rows.append(
            {
                "iteration_number": 1,
                "environment": "equity",
                "algorithm": "sac",
                "fold_number": fold,
                "sharpe": 2.5 if cls == "healthy" else -0.5,
                "sortino": 3.0 if cls == "healthy" else -0.5,
                "calmar": 1.8,
                "mdd": 0.06 if cls == "healthy" else 0.18,
                "total_return": 0.07 if cls == "healthy" else -0.02,
                "profit_factor": 3.5 if cls == "healthy" else 0.8,
                "win_rate": 0.6,
                "total_trades": 380,
                "overfitting_class": cls,
                "max_single_loss": -0.06,
                "is_control_fold": False,
            }
        )
    return pd.DataFrame(rows)


class TestComputeIterationCps:
    """REPORT-09: per-iteration CPS via per-algo aggregation."""

    def test_returns_required_keys(self) -> None:
        """The result dict has all keys needed by the iteration_results UPSERT."""
        fold_history = make_per_algo_fold_history()
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        required = {
            "cps_v1_multiplicative",
            "cps_v2_additive",
            "cps_v3_sortino",
            "cps_v1_treatment_only",
            "cps_v1_control_only",
            "cps_components",
            "worst_fold_number",
            "worst_fold_mdd",
            "median_return",
            "mean_winner_sharpe",
            "winners_count",
            "chronic_failure_count",
            "return_regression_delta",
        }
        assert required.issubset(result.keys())

    def test_per_algo_breakdown_in_components(self) -> None:
        """The components dict includes per_algo CPS values for each algorithm present."""
        fold_history = make_per_algo_fold_history()
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        per_algo = result["cps_components"]["per_algo"]
        assert set(per_algo.keys()) == {"ppo", "a2c", "sac"}
        for algo in ("ppo", "a2c", "sac"):
            assert "cps_v1_multiplicative" in per_algo[algo]
            assert "cps_v2_additive" in per_algo[algo]

    def test_aggregate_is_mean_across_algos(self) -> None:
        """Primary cps_v1 must equal the mean of per-algo values."""
        fold_history = make_per_algo_fold_history()
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        per_algo = result["cps_components"]["per_algo"]
        v1_values = [per_algo[a]["cps_v1_multiplicative"] for a in ("ppo", "a2c", "sac")]
        expected_mean = sum(v1_values) / 3.0
        assert result["cps_v1_multiplicative"] == pytest.approx(expected_mean, abs=1e-9)

    def test_treatment_control_null_when_no_control_folds(self) -> None:
        """Iter 0-2 had no control folds; treatment_only / control_only must be None."""
        fold_history = make_per_algo_fold_history()
        # All is_control_fold=False in fixture
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        assert result["cps_v1_treatment_only"] is None
        assert result["cps_v1_control_only"] is None

    def test_treatment_control_populated_when_mixed(self) -> None:
        """When the iteration has both control and treatment folds, both subsets compute."""
        fold_history = make_per_algo_fold_history()
        # Mark folds 3,4 as control across all algos
        fold_history.loc[fold_history["fold_number"].isin([3, 4]), "is_control_fold"] = True
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        assert result["cps_v1_treatment_only"] is not None
        assert result["cps_v1_control_only"] is not None

    def test_iter0_no_baseline_v3_none(self) -> None:
        """When prev_iter_median_return is None, cps_v3_sortino is None."""
        fold_history = make_per_algo_fold_history()
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=None,
            chronic_failure_count=0,
        )
        assert result["cps_v3_sortino"] is None

    def test_chronic_count_threaded_through(self) -> None:
        """chronic_failure_count is passed to v2 and is reflected in components."""
        fold_history = make_per_algo_fold_history()
        result_zero = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        result_five = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=5,
        )
        assert result_zero["chronic_failure_count"] == 0
        assert result_five["chronic_failure_count"] == 5
        # v2 should drop by 0.50 per algo, then mean across 3 algos = 0.50 drop in mean
        delta = result_zero["cps_v2_additive"] - result_five["cps_v2_additive"]
        assert delta == pytest.approx(0.50, abs=1e-6)

    def test_worst_fold_identified_across_algos(self) -> None:
        """worst_fold_mdd uses the highest MDD across all (algo, fold) pairs."""
        fold_history = make_per_algo_fold_history()
        result = compute_iteration_cps(
            fold_history,
            env="equity",
            iteration=1,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        # SAC reject folds have mdd=0.18 — highest in fixture
        assert result["worst_fold_mdd"] == pytest.approx(0.18, abs=1e-6)

    def test_empty_iteration_returns_nones(self) -> None:
        """An iteration with no folds returns a dict of Nones / 0s, no exception."""
        empty = pd.DataFrame(
            columns=[
                "iteration_number",
                "environment",
                "algorithm",
                "fold_number",
                "sharpe",
                "sortino",
                "calmar",
                "mdd",
                "total_return",
                "profit_factor",
                "win_rate",
                "total_trades",
                "overfitting_class",
                "max_single_loss",
                "is_control_fold",
            ]
        )
        result = compute_iteration_cps(
            empty,
            env="equity",
            iteration=99,
            prev_iter_median_return=0.05,
            chronic_failure_count=0,
        )
        assert result["cps_v1_multiplicative"] is None
        assert result["winners_count"] == 0


# ---------------------------------------------------------------------------
# Live DB tests (skipped if DATABASE_URL unset)
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_conn() -> psycopg.Connection:
    """Open a Postgres connection or skip if DATABASE_URL unset."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set; skipping live Postgres test")
    return psycopg.connect(db_url)


class TestLoadIterationHistoryLive:
    """REPORT-01: live DB read against pg16 (skipped without DATABASE_URL)."""

    def test_returns_dataframe_with_expected_columns(self, pg_conn: psycopg.Connection) -> None:
        df = load_iteration_history(pg_conn, env="equity", n=10)
        assert isinstance(df, pd.DataFrame)
        for col in ("iteration_number", "environment", "ensemble_sharpe"):
            assert col in df.columns

    def test_filters_by_environment(self, pg_conn: psycopg.Connection) -> None:
        df = load_iteration_history(pg_conn, env="equity", n=10)
        assert (df["environment"] == "equity").all()

    def test_returns_chronological_order(self, pg_conn: psycopg.Connection) -> None:
        df = load_iteration_history(pg_conn, env="equity", n=10)
        if len(df) > 1:
            iters = df["iteration_number"].tolist()
            assert iters == sorted(iters)


class TestLoadFoldHistoryLive:
    """REPORT-02: live dedup test against the iter 1 restart-with-fixes case."""

    def test_iter1_dedups_to_23_folds_per_algo(self, pg_conn: psycopg.Connection) -> None:
        """Iter 1 equity A2C had 29 raw rows; dedup should yield 23 distinct folds."""
        df = load_fold_history(pg_conn, env="equity")
        iter1_a2c = df[(df["iteration_number"] == 1) & (df["algorithm"] == "a2c")]
        # Each fold_number should appear exactly once
        assert iter1_a2c["fold_number"].nunique() == len(iter1_a2c)
        # We expect 23 distinct folds for equity
        assert len(iter1_a2c) == 23

    def test_iter1_picks_post_fix_row(self, pg_conn: psycopg.Connection) -> None:
        """Iter 1 A2C fold 0 must pick the later (post-fix) row, sharpe ≈ 0.794."""
        df = load_fold_history(pg_conn, env="equity")
        row = df[
            (df["iteration_number"] == 1) & (df["algorithm"] == "a2c") & (df["fold_number"] == 0)
        ]
        assert len(row) == 1
        # Post-fix sharpe was 0.794 (vs pre-fix 0.919)
        assert row.iloc[0]["sharpe"] == pytest.approx(0.794, abs=0.01)


class TestComputeAndPersistIterationCpsLive:
    """REPORT-11: orchestrator end-to-end against pg16 (skipped without DATABASE_URL)."""

    def test_persists_and_returns_summary(self, pg_conn: psycopg.Connection) -> None:
        """Running the orchestrator on iter 4 equity should return a summary
        with non-null CPS values and persist them to iteration_results."""
        from swingrl.reporting.iteration_report import compute_and_persist_iteration_cps

        result = compute_and_persist_iteration_cps(pg_conn, env="equity", iteration=4)
        pg_conn.commit()
        assert result["env"] == "equity"
        assert result["iteration"] == 4
        assert result["cps_v1_multiplicative"] is not None
        assert result["winners_count"] > 0
        # Iter 4 has chronic_count=5 (folds 2,4,7,13,15) and the worst fold is 7
        assert result["chronic_failure_count"] == 5
        assert result["worst_fold_number"] == 7

    def test_idempotent_re_run(self, pg_conn: psycopg.Connection) -> None:
        """Running the orchestrator twice produces the same persisted values."""
        from swingrl.reporting.iteration_report import compute_and_persist_iteration_cps

        first = compute_and_persist_iteration_cps(pg_conn, env="equity", iteration=4)
        pg_conn.commit()
        second = compute_and_persist_iteration_cps(pg_conn, env="equity", iteration=4)
        pg_conn.commit()
        assert first["cps_v1_multiplicative"] == pytest.approx(
            second["cps_v1_multiplicative"], abs=1e-12
        )
        assert first["cps_v2_additive"] == pytest.approx(second["cps_v2_additive"], abs=1e-12)
        assert first["dedup_rows_dropped"] == second["dedup_rows_dropped"]

    def test_iter1_dedup_count_reported_in_summary(self, pg_conn: psycopg.Connection) -> None:
        """Iter 1 equity should report dedup_rows_dropped=9 in the summary."""
        from swingrl.reporting.iteration_report import compute_and_persist_iteration_cps

        result = compute_and_persist_iteration_cps(pg_conn, env="equity", iteration=1)
        pg_conn.commit()
        assert result["dedup_rows_dropped"] == 9

    def test_iter0_summary_has_no_deltas(self, pg_conn: psycopg.Connection) -> None:
        """Iter 0 has no prior iteration; all delta fields must be None."""
        from swingrl.reporting.iteration_report import compute_and_persist_iteration_cps

        result = compute_and_persist_iteration_cps(pg_conn, env="equity", iteration=0)
        pg_conn.commit()
        assert result["cps_v1_delta_vs_prev"] is None
        assert result["return_delta_vs_prev"] is None
        assert result["worst_mdd_delta_vs_prev"] is None
        assert result["regression_flag"] is False
