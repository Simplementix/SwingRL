"""Training pipeline helper functions.

Pure-function utilities for the train_pipeline.py orchestrator (Plan 02).
Provides data slicing, ensemble gate checking, weight computation from
walk-forward Sharpe, tuning grid definitions, and timestep scheduling.

All functions are side-effect free and importable without GPU or broker
initialization.

Usage:
    from swingrl.training.pipeline_helpers import (
        slice_recent,
        check_ensemble_gate,
        compute_ensemble_weights_from_wf,
        TUNING_GRID,
        decide_final_timesteps,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from swingrl.agents.backtest import FoldResult
    from swingrl.config.schema import SwingRLConfig
    from swingrl.training.trainer import TrainingResult

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Window constants
# ---------------------------------------------------------------------------

# Recent data window sizes used for final full-data training slices
RECENT_WINDOW_BARS: dict[str, int] = {
    "equity": 252 * 3,  # 756 bars — 3 trading years
    "crypto": 2191,  # 2191 4H bars — ~1 calendar year (365.25 * 6)
}

# Default timesteps for each environment (converged-early path)
DEFAULT_TIMESTEPS: dict[str, int] = {
    "equity": 1_000_000,
    "crypto": 500_000,
}

# Escalated timesteps when training did not converge (None converged_at_step)
ESCALATED_TIMESTEPS: dict[str, int] = {
    "equity": 2_000_000,
    "crypto": 1_000_000,
}

# Gate thresholds
_GATE_MIN_SHARPE: float = 1.0
_GATE_MAX_MDD: float = 0.15

# ---------------------------------------------------------------------------
# Tuning grid
# ---------------------------------------------------------------------------

# Round 1: PPO-only — 4 variants (lr x ent_coef x clip_range combos)
# Round 2: A2C (2 lr variants) + SAC (2 lr variants)
# net_arch is never set here — always inherits [64, 64] from TrainingOrchestrator
TUNING_GRID: dict[int, dict[str, list[dict[str, Any]]]] = {
    1: {
        "ppo": [
            {"learning_rate": 1e-4, "ent_coef": 0.005, "clip_range": 0.15},
            {"learning_rate": 1e-4, "ent_coef": 0.01, "clip_range": 0.2},
            {"learning_rate": 3e-4, "ent_coef": 0.005, "clip_range": 0.2},
            {"learning_rate": 3e-4, "ent_coef": 0.02, "clip_range": 0.25},
        ],
    },
    2: {
        "a2c": [
            {"learning_rate": 5e-4},
            {"learning_rate": 1e-3},
        ],
        "sac": [
            {"learning_rate": 1e-4},
            {"learning_rate": 3e-4},
        ],
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def slice_recent(
    features: np.ndarray,
    prices: np.ndarray,
    env_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice features and prices to the most recent N bars for the given env.

    If the data is shorter than the window, all data is returned unchanged.

    Args:
        features: Full feature array (bars x feature_count).
        prices: Full price array (bars x asset_count).
        env_name: Environment type ("equity" or "crypto") — determines window.

    Returns:
        Tuple of (features_slice, prices_slice) — last N bars.
    """
    window = RECENT_WINDOW_BARS[env_name]
    total_bars = len(features)

    if total_bars <= window:
        log.info(
            "slice_recent_data_shorter_than_window",
            env_name=env_name,
            total_bars=total_bars,
            window=window,
        )
        return features, prices

    log.info(
        "slice_recent",
        env_name=env_name,
        total_bars=total_bars,
        window=window,
        start_idx=total_bars - window,
    )
    return features[-window:], prices[-window:]


def check_ensemble_gate(
    all_fold_results: dict[str, list[FoldResult]],
    ensemble_weights: dict[str, float] | None = None,
) -> tuple[bool, float, float]:
    """Check if the ensemble meets minimum quality gates for promotion.

    Computes weighted mean OOS Sharpe and MDD across all algos and folds.
    When ensemble_weights are provided, each algo's folds are weighted by its
    ensemble weight (prevents low-weight algos like SAC at 0.19% from having
    disproportionate 33% influence on the gate decision).
    Gate: sharpe > 1.0 AND abs(mdd) < 0.15.

    Args:
        all_fold_results: Dict mapping algo name to list of FoldResult.
            FoldResult.out_of_sample_metrics must contain "sharpe" and "mdd".
        ensemble_weights: Optional dict mapping algo name to ensemble weight
            (e.g., {"ppo": 0.6, "a2c": 0.27, "sac": 0.13}). When provided,
            gate metrics are weight-proportional. When None, equal weighting
            (backward compatible).

    Returns:
        Tuple of (passed, ensemble_sharpe, ensemble_mdd).
        - passed: True if both thresholds met.
        - ensemble_sharpe: (Weighted) mean OOS Sharpe across all folds and algos.
        - ensemble_mdd: (Weighted) mean OOS MDD across all folds and algos.
    """
    if not all_fold_results:
        log.warning("ensemble_gate_empty_results")
        return False, 0.0, 0.0

    if ensemble_weights:
        # Weight-proportional gate: each algo's contribution scales by ensemble weight
        weighted_sharpe_sum = 0.0
        weighted_mdd_sum = 0.0
        total_weight = 0.0

        for algo_name, folds in all_fold_results.items():
            if not folds:
                continue
            w = ensemble_weights.get(algo_name, 0.0)
            algo_sharpes = [float(f.out_of_sample_metrics.get("sharpe", 0.0)) for f in folds]
            algo_mdds = [float(f.out_of_sample_metrics.get("mdd", -1.0)) for f in folds]
            weighted_sharpe_sum += w * float(np.mean(algo_sharpes))
            weighted_mdd_sum += w * float(np.mean(algo_mdds))
            total_weight += w

        if total_weight <= 0:
            return False, 0.0, 0.0

        ensemble_sharpe = weighted_sharpe_sum / total_weight
        ensemble_mdd = weighted_mdd_sum / total_weight
    else:
        # Backward compatible: equal weighting across all folds
        all_sharpes: list[float] = []
        all_mdds: list[float] = []

        for _algo_name, folds in all_fold_results.items():
            for fold in folds:
                oos = fold.out_of_sample_metrics
                sharpe = oos.get("sharpe", 0.0)
                mdd = oos.get("mdd", -1.0)
                all_sharpes.append(float(sharpe))
                all_mdds.append(float(mdd))

        if not all_sharpes:
            return False, 0.0, 0.0

        ensemble_sharpe = float(np.mean(all_sharpes))
        ensemble_mdd = float(np.mean(all_mdds))

    # MDD is stored as a negative float (e.g., -0.10 for 10% drawdown)
    # Gate checks the absolute value
    passed = ensemble_sharpe > _GATE_MIN_SHARPE and abs(ensemble_mdd) < _GATE_MAX_MDD

    log.info(
        "ensemble_gate_checked",
        passed=passed,
        ensemble_sharpe=round(ensemble_sharpe, 4),
        ensemble_mdd=round(ensemble_mdd, 4),
        fold_count=sum(len(f) for f in all_fold_results.values()),
        weighted=ensemble_weights is not None,
    )

    return passed, ensemble_sharpe, ensemble_mdd


def compute_ensemble_weights_from_wf(
    config: SwingRLConfig,
    wf_results: dict[str, list[FoldResult]],
    env_name: str,
) -> dict[str, float]:
    """Compute ensemble weights from walk-forward mean OOS Sharpe ratios.

    Aggregates mean OOS Sharpe per algo across all folds, then passes to
    EnsembleBlender.compute_weights() (Sharpe softmax). Not a placeholder
    — uses real walk-forward data.

    Args:
        config: Validated SwingRLConfig instance (passed to EnsembleBlender).
        wf_results: Dict mapping algo name to list of FoldResult.
        env_name: Environment type ("equity" or "crypto").

    Returns:
        Dict mapping algo name to ensemble weight (summing to 1.0).
    """
    from swingrl.training.ensemble import EnsembleBlender

    # Compute mean OOS Sharpe per algo across all folds
    agent_sharpes: dict[str, float] = {}
    for algo_name, folds in wf_results.items():
        if not folds:
            agent_sharpes[algo_name] = 0.0
            continue
        sharpes = [f.out_of_sample_metrics.get("sharpe", 0.0) for f in folds]
        agent_sharpes[algo_name] = float(np.mean(sharpes))

    log.info(
        "compute_ensemble_weights_from_wf",
        env_name=env_name,
        agent_sharpes={k: round(v, 4) for k, v in agent_sharpes.items()},
    )

    blender = EnsembleBlender(config)
    return blender.compute_weights(env_name=env_name, agent_sharpes=agent_sharpes)


def decide_final_timesteps(
    env_name: str,
    result: TrainingResult,
) -> int:
    """Determine final training timesteps based on convergence during tuning.

    If training did not converge (converged_at_step is None), escalates to
    a larger timestep budget. If it converged early, uses the standard budget.

    Args:
        env_name: Environment type ("equity" or "crypto").
        result: TrainingResult from a tuning round.

    Returns:
        Final timestep count for the production training run.
    """
    if result.converged_at_step is None:
        ts = ESCALATED_TIMESTEPS[env_name]
        log.info(
            "timesteps_escalated",
            env_name=env_name,
            timesteps=ts,
            reason="no_convergence_in_tuning",
        )
    else:
        ts = DEFAULT_TIMESTEPS[env_name]
        log.info(
            "timesteps_default",
            env_name=env_name,
            timesteps=ts,
            converged_at=result.converged_at_step,
        )
    return ts
