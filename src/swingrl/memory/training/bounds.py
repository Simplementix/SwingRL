"""Hyperparameter and reward weight bounds enforcement for memory agent outputs.

All LLM-suggested hyperparameters and reward weights must pass through this
module before reaching the trainer. This is a non-negotiable safety layer.

Constants match the spec in memory_meta_trainer.md Section 1.
The ollama_smart_model is qwen2.5:3b (sole local model; consolidation uses NVIDIA kimi-k2.5 cloud).

Usage:
    from swingrl.memory.training.bounds import clamp_run_config, clamp_reward_weights
    safe_config = clamp_run_config(llm_suggested_config)
    safe_weights = clamp_reward_weights(llm_suggested_weights)
"""

from __future__ import annotations

import math
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Bound definitions
# ---------------------------------------------------------------------------

HYPERPARAM_BOUNDS: dict[str, tuple[Any, Any]] = {
    "learning_rate": (1e-5, 1e-3),
    "entropy_coeff": (0.0, 0.05),
    "clip_range": (0.1, 0.4),
    "n_epochs": (3, 20),
    "batch_size": (32, 512),  # also forced to power of 2
    "gamma": (0.90, 0.9999),
}

REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}

MIN_EPOCHS_BEFORE_STOP: int = 10  # hard floor — never lower
MAX_EPOCHS: int = 200
MIN_WINDOW_YEARS: float = 1.0
CRISIS_PERIOD_PCT: tuple[float, float] = (0.05, 0.50)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_power_of_two(n: int) -> int:
    """Return the power of 2 nearest to n (rounds to nearest, not floor/ceil).

    Args:
        n: Input integer. Must be >= 1.

    Returns:
        Nearest power of 2.
    """
    return int(2 ** round(math.log2(max(1, n))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clamp_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Clamp all hyperparameters to HYPERPARAM_BOUNDS. Does not mutate input.

    Enforces:
    - learning_rate within [1e-5, 1e-3]
    - batch_size clamped to [32, 512] and then forced to nearest power of 2
    - clip_range, n_epochs, gamma, entropy_coeff within their ranges
    - Unknown keys are passed through unchanged

    Args:
        config: Dict of hyperparameter name → value from LLM suggestion.

    Returns:
        New dict with all known hyperparameters clamped.
    """
    out: dict[str, Any] = dict(config)

    for key, (lo, hi) in HYPERPARAM_BOUNDS.items():
        if key not in out:
            continue
        raw = out[key]
        clamped = max(lo, min(hi, raw))
        if clamped != raw:
            log.warning(
                "hyperparam_clamped",
                key=key,
                raw=raw,
                clamped=clamped,
                lo=lo,
                hi=hi,
            )
        out[key] = clamped

    # batch_size must also be a power of 2 (after clamping to range)
    if "batch_size" in out:
        p2 = _nearest_power_of_two(int(out["batch_size"]))
        if p2 != out["batch_size"]:
            log.warning(
                "batch_size_power_of_two",
                before=out["batch_size"],
                after=p2,
            )
        out["batch_size"] = p2

    return out


def clamp_reward_weights(weights: dict[str, Any]) -> dict[str, float]:
    """Clamp reward weights to REWARD_BOUNDS and renormalize to sum=1.0.

    Missing keys get the midpoint of their bounds as a safe default.
    If the total after clamping is 0 (all weights at zero), safe defaults
    are returned (uniform midpoints normalized to 1.0).

    Args:
        weights: Dict of reward component name → weight from LLM suggestion.

    Returns:
        New dict with clamped and renormalized reward weights (sum == 1.0).
    """
    out: dict[str, float] = {}

    for key, (lo, hi) in REWARD_BOUNDS.items():
        raw = weights.get(key, (lo + hi) / 2)
        clamped = max(lo, min(hi, float(raw)))
        if clamped != raw:
            log.warning(
                "reward_weight_clamped",
                key=key,
                raw=raw,
                clamped=clamped,
                lo=lo,
                hi=hi,
            )
        out[key] = clamped

    total = sum(out.values())

    if total == 0.0:
        # All weights are zero — return safe midpoint defaults normalized to 1.0
        log.warning("reward_weights_all_zero_returning_defaults")
        safe = {k: (lo + hi) / 2 for k, (lo, hi) in REWARD_BOUNDS.items()}
        safe_total = sum(safe.values())
        return {k: v / safe_total for k, v in safe.items()}

    return {k: v / total for k, v in out.items()}
