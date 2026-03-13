"""Tests for training pipeline helper functions.

TRAIN-04, TRAIN-05: Pipeline helpers for data slicing, ensemble gating,
weight computation, timestep scheduling, and tuning grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal FoldResult stub for tests that avoids importing the full backtest
# ---------------------------------------------------------------------------


@dataclass
class _GateResult:
    passed: bool = True
    reason: str = ""


@dataclass
class _FoldResult:
    fold_number: int = 0
    train_range: tuple[int, int] = (0, 252)
    test_range: tuple[int, int] = (252, 315)
    in_sample_metrics: dict[str, float] = field(default_factory=dict)
    out_of_sample_metrics: dict[str, float] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    gate_result: _GateResult = field(default_factory=_GateResult)
    overfitting: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# slice_recent
# ---------------------------------------------------------------------------


class TestSliceRecent:
    """Tests for slice_recent()."""

    def test_equity_returns_last_756_bars(self) -> None:
        """TRAIN-04: slice_recent equity returns last 756 bars (3yr x 252)."""
        from swingrl.training.pipeline_helpers import slice_recent

        features = np.zeros((1000, 10))
        prices = np.zeros((1000, 8))
        f_slice, p_slice = slice_recent(features, prices, "equity")
        assert f_slice.shape[0] == 756
        assert p_slice.shape[0] == 756

    def test_crypto_returns_last_2191_bars(self) -> None:
        """TRAIN-04: slice_recent crypto returns last 2191 bars (1yr x 365.25 x 6)."""
        from swingrl.training.pipeline_helpers import slice_recent

        features = np.zeros((5000, 10))
        prices = np.zeros((5000, 2))
        f_slice, p_slice = slice_recent(features, prices, "crypto")
        assert f_slice.shape[0] == 2191
        assert p_slice.shape[0] == 2191

    def test_slice_is_last_n_bars(self) -> None:
        """TRAIN-04: slice_recent returns the LAST n bars, not the first."""
        from swingrl.training.pipeline_helpers import slice_recent

        features = np.arange(1000).reshape(1000, 1).astype(float)
        prices = np.arange(1000).reshape(1000, 1).astype(float)
        f_slice, _ = slice_recent(features, prices, "equity")
        # Last 756 bars: indices 244 to 999
        assert f_slice[0, 0] == pytest.approx(244.0)
        assert f_slice[-1, 0] == pytest.approx(999.0)

    def test_data_shorter_than_window_returns_all(self) -> None:
        """TRAIN-04: slice_recent handles data shorter than window by returning all data."""
        from swingrl.training.pipeline_helpers import slice_recent

        features = np.zeros((100, 5))
        prices = np.zeros((100, 2))
        f_slice, p_slice = slice_recent(features, prices, "equity")
        assert f_slice.shape[0] == 100
        assert p_slice.shape[0] == 100


# ---------------------------------------------------------------------------
# check_ensemble_gate
# ---------------------------------------------------------------------------


class TestCheckEnsembleGate:
    """Tests for check_ensemble_gate()."""

    def test_passes_when_sharpe_above_1_and_mdd_below_015(self) -> None:
        """TRAIN-04: Gate passes when mean OOS Sharpe > 1.0 and mean OOS MDD < 0.15."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        fold = _FoldResult(out_of_sample_metrics={"sharpe": 1.5, "mdd": -0.10})
        all_fold_results: dict[str, list[Any]] = {"ppo": [fold]}
        passed, sharpe, mdd = check_ensemble_gate(all_fold_results)
        assert passed is True
        assert sharpe > 1.0

    def test_fails_when_sharpe_below_threshold(self) -> None:
        """TRAIN-04: Gate fails when mean OOS Sharpe <= 1.0."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        fold = _FoldResult(out_of_sample_metrics={"sharpe": 0.8, "mdd": -0.05})
        all_fold_results: dict[str, list[Any]] = {"ppo": [fold]}
        passed, sharpe, mdd = check_ensemble_gate(all_fold_results)
        assert passed is False

    def test_fails_when_mdd_above_threshold(self) -> None:
        """TRAIN-04: Gate fails when mean OOS MDD >= 0.15."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        fold = _FoldResult(out_of_sample_metrics={"sharpe": 1.5, "mdd": -0.20})
        all_fold_results: dict[str, list[Any]] = {"ppo": [fold]}
        passed, sharpe, mdd = check_ensemble_gate(all_fold_results)
        assert passed is False

    def test_handles_empty_fold_results(self) -> None:
        """TRAIN-04: check_ensemble_gate handles empty fold results gracefully."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        passed, sharpe, mdd = check_ensemble_gate({})
        assert passed is False
        assert isinstance(sharpe, float)
        assert isinstance(mdd, float)

    def test_averages_across_multiple_algos_and_folds(self) -> None:
        """TRAIN-04: Gate averages OOS metrics across all algos and folds."""
        from swingrl.training.pipeline_helpers import check_ensemble_gate

        fold1 = _FoldResult(out_of_sample_metrics={"sharpe": 1.2, "mdd": -0.08})
        fold2 = _FoldResult(out_of_sample_metrics={"sharpe": 1.8, "mdd": -0.10})
        all_fold_results: dict[str, list[Any]] = {
            "ppo": [fold1],
            "a2c": [fold2],
        }
        passed, sharpe, mdd = check_ensemble_gate(all_fold_results)
        assert passed is True
        assert sharpe == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# compute_ensemble_weights_from_wf
# ---------------------------------------------------------------------------


class TestComputeEnsembleWeightsFromWF:
    """Tests for compute_ensemble_weights_from_wf()."""

    def test_produces_real_weights_not_placeholder(self) -> None:
        """TRAIN-04: compute_ensemble_weights_from_wf produces real weights from walk-forward Sharpe."""
        from swingrl.training.pipeline_helpers import compute_ensemble_weights_from_wf

        fold_ppo = _FoldResult(out_of_sample_metrics={"sharpe": 1.5, "mdd": -0.08})
        fold_a2c = _FoldResult(out_of_sample_metrics={"sharpe": 0.8, "mdd": -0.05})
        wf_results: dict[str, list[Any]] = {
            "ppo": [fold_ppo],
            "a2c": [fold_a2c],
        }
        mock_config = MagicMock()
        weights = compute_ensemble_weights_from_wf(mock_config, wf_results, "equity")

        assert isinstance(weights, dict)
        assert "ppo" in weights
        assert "a2c" in weights
        # ppo should have higher weight than a2c (higher Sharpe)
        assert weights["ppo"] > weights["a2c"]
        # weights must sum to 1.0
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_weights_differ_for_different_sharpes(self) -> None:
        """TRAIN-04: Weights are not all equal when Sharpe ratios differ."""
        from swingrl.training.pipeline_helpers import compute_ensemble_weights_from_wf

        fold_ppo = _FoldResult(out_of_sample_metrics={"sharpe": 2.0, "mdd": -0.05})
        fold_a2c = _FoldResult(out_of_sample_metrics={"sharpe": 0.5, "mdd": -0.05})
        fold_sac = _FoldResult(out_of_sample_metrics={"sharpe": 1.0, "mdd": -0.05})
        wf_results: dict[str, list[Any]] = {
            "ppo": [fold_ppo],
            "a2c": [fold_a2c],
            "sac": [fold_sac],
        }
        mock_config = MagicMock()
        weights = compute_ensemble_weights_from_wf(mock_config, wf_results, "equity")

        # All weights should differ (softmax ensures this for different inputs)
        assert weights["ppo"] != weights["a2c"]
        assert weights["ppo"] != weights["sac"]


# ---------------------------------------------------------------------------
# decide_final_timesteps
# ---------------------------------------------------------------------------


class TestDecideFinalTimesteps:
    """Tests for decide_final_timesteps()."""

    def test_escalates_when_converged_at_step_is_none(self) -> None:
        """TRAIN-05: decide_final_timesteps escalates to 2M/1M when converged_at_step is None."""
        from swingrl.training.pipeline_helpers import (
            ESCALATED_TIMESTEPS,
            decide_final_timesteps,
        )
        from swingrl.training.trainer import TrainingResult

        result = TrainingResult(
            model_path=MagicMock(),
            vec_normalize_path=MagicMock(),
            env_name="equity",
            algo_name="ppo",
            converged_at_step=None,
            total_timesteps=1_000_000,
        )
        ts = decide_final_timesteps("equity", result)
        assert ts == ESCALATED_TIMESTEPS["equity"]

    def test_returns_default_when_converged_early(self) -> None:
        """TRAIN-05: decide_final_timesteps returns DEFAULT when converged_at_step is set."""
        from swingrl.training.pipeline_helpers import DEFAULT_TIMESTEPS, decide_final_timesteps
        from swingrl.training.trainer import TrainingResult

        result = TrainingResult(
            model_path=MagicMock(),
            vec_normalize_path=MagicMock(),
            env_name="equity",
            algo_name="ppo",
            converged_at_step=500_000,
            total_timesteps=1_000_000,
        )
        ts = decide_final_timesteps("equity", result)
        assert ts == DEFAULT_TIMESTEPS["equity"]

    def test_crypto_escalated_is_1m(self) -> None:
        """TRAIN-05: Crypto escalated timesteps is 1M."""
        from swingrl.training.pipeline_helpers import (
            ESCALATED_TIMESTEPS,
            decide_final_timesteps,
        )
        from swingrl.training.trainer import TrainingResult

        result = TrainingResult(
            model_path=MagicMock(),
            vec_normalize_path=MagicMock(),
            env_name="crypto",
            algo_name="ppo",
            converged_at_step=None,
            total_timesteps=500_000,
        )
        ts = decide_final_timesteps("crypto", result)
        assert ts == ESCALATED_TIMESTEPS["crypto"]
        assert ts == 1_000_000


# ---------------------------------------------------------------------------
# TUNING_GRID
# ---------------------------------------------------------------------------


class TestTuningGrid:
    """Tests for TUNING_GRID structure."""

    def test_round_1_has_ppo_entries_only(self) -> None:
        """TRAIN-05: TUNING_GRID round 1 has PPO-only entries."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        assert 1 in TUNING_GRID
        algo_keys = set(TUNING_GRID[1].keys())
        assert algo_keys == {"ppo"}

    def test_round_2_has_a2c_and_sac_entries(self) -> None:
        """TRAIN-05: TUNING_GRID round 2 has A2C and SAC entries."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        assert 2 in TUNING_GRID
        algo_keys = set(TUNING_GRID[2].keys())
        assert "a2c" in algo_keys
        assert "sac" in algo_keys

    def test_tuning_grid_never_changes_net_arch(self) -> None:
        """TRAIN-05: TUNING_GRID never sets net_arch (always uses [64,64] from trainer)."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        for _round_num, round_algos in TUNING_GRID.items():
            for _algo, variants in round_algos.items():
                for variant in variants:
                    assert "net_arch" not in variant, (
                        f"TUNING_GRID must not set net_arch; found in round {_round_num} {_algo}"
                    )

    def test_round_1_ppo_has_multiple_variants(self) -> None:
        """TRAIN-05: TUNING_GRID round 1 PPO has at least 4 variants."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        assert len(TUNING_GRID[1]["ppo"]) >= 4

    def test_round_2_a2c_has_at_least_2_variants(self) -> None:
        """TRAIN-05: TUNING_GRID round 2 A2C has at least 2 variants."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        assert len(TUNING_GRID[2]["a2c"]) >= 2

    def test_round_2_sac_has_at_least_2_variants(self) -> None:
        """TRAIN-05: TUNING_GRID round 2 SAC has at least 2 variants."""
        from swingrl.training.pipeline_helpers import TUNING_GRID

        assert len(TUNING_GRID[2]["sac"]) >= 2


# ---------------------------------------------------------------------------
# TrainingOrchestrator.train() hyperparams_override
# ---------------------------------------------------------------------------


class TestHyperparamsOverride:
    """Tests for hyperparams_override parameter on TrainingOrchestrator.train()."""

    def test_train_signature_accepts_hyperparams_override(self) -> None:
        """TRAIN-04: TrainingOrchestrator.train() accepts optional hyperparams_override dict."""
        import inspect

        from swingrl.training.trainer import TrainingOrchestrator

        sig = inspect.signature(TrainingOrchestrator.train)
        assert "hyperparams_override" in sig.parameters

    def test_hyperparams_override_default_is_none(self) -> None:
        """TRAIN-04: hyperparams_override defaults to None."""
        import inspect

        from swingrl.training.trainer import TrainingOrchestrator

        sig = inspect.signature(TrainingOrchestrator.train)
        param = sig.parameters["hyperparams_override"]
        assert param.default is None
