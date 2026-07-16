"""Sharpe-weighted softmax ensemble blending for RL agents.

Combines PPO/A2C/SAC agent actions using softmax weights derived from
per-agent validation Sharpe ratios.

Usage:
    from swingrl.training.ensemble import EnsembleBlender, sharpe_softmax_weights
    blender = EnsembleBlender(config)
    weights = blender.compute_weights("equity", {"ppo": 1.0, "a2c": 0.5, "sac": 0.8})
    action = blender.blend_actions(actions, weights)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from swingrl.utils.exceptions import ModelError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

# Equal-share fallback weight for a single algo in a 3-algo ensemble (ppo/a2c/sac).
# Used when a live model has no ``model_metadata`` row so blending never KeyErrors
# and never silently drifts (review M5).
DEFAULT_ENSEMBLE_WEIGHT: float = 1.0 / 3


def sharpe_softmax_weights(sharpe_ratios: dict[str, float]) -> dict[str, float]:
    """Compute softmax-normalized ensemble weights from Sharpe ratios.

    Uses numerical stability trick (subtract max before exp) to prevent
    overflow. Agents with higher Sharpe ratios receive proportionally
    higher weights.

    Args:
        sharpe_ratios: Dict mapping agent name to Sharpe ratio.

    Returns:
        Dict mapping agent name to weight (summing to 1.0).
    """
    names = list(sharpe_ratios.keys())
    values = np.array([sharpe_ratios[n] for n in names])

    # Numerical stability: subtract max
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)

    weights = exp_vals / total

    result = {name: float(w) for name, w in zip(names, weights, strict=True)}

    log.info(
        "ensemble_weights_computed",
        weights={k: round(v, 4) for k, v in result.items()},
        sharpe_ratios={k: round(v, 4) for k, v in sharpe_ratios.items()},
    )

    return result


class EnsembleBlender:
    """Sharpe-weighted ensemble blender for RL agents.

    Computes ensemble weights based on per-agent Sharpe ratios using
    softmax normalization, then blends actions via weighted sum.

    Args:
        config: Validated SwingRLConfig instance.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        self._config = config

    def compute_weights(
        self,
        env_name: str,
        agent_sharpes: dict[str, float],
    ) -> dict[str, float]:
        """Compute ensemble weights from per-agent Sharpe ratios.

        Args:
            env_name: Environment type ("equity" or "crypto").
            agent_sharpes: Dict mapping agent name to validation Sharpe.

        Returns:
            Dict mapping agent name to ensemble weight (summing to 1.0).
        """
        log.info(
            "ensemble_weights_request",
            env_name=env_name,
            agent_count=len(agent_sharpes),
        )

        return sharpe_softmax_weights(agent_sharpes)

    def blend_actions(
        self,
        actions: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Compute the weighted sum of per-agent actions over the actually-loaded algos.

        Weights are renormalized over ``actions`` (the algos that actually loaded),
        so a partial ensemble does not silently drift (review M5). A loaded algo
        with no weight entry (missing ``model_metadata`` row) falls back to an equal
        share instead of raising ``KeyError``.

        Args:
            actions: Dict mapping agent name to action array (loaded algos only).
            weights: Dict mapping agent name to ensemble weight (may be partial/empty).

        Returns:
            Blended action array (weighted sum, weights summing to 1.0).

        Raises:
            ModelError: If ``actions`` is empty (no models loaded to blend).
        """
        if not actions:
            raise ModelError("No actions to blend")

        # Resolve a weight per loaded algo, defaulting to equal share when missing.
        used: dict[str, float] = {}
        for name in actions:
            weight = weights.get(name)
            if weight is None:
                log.warning(
                    "ensemble_weight_missing",
                    algo=name,
                    default=DEFAULT_ENSEMBLE_WEIGHT,
                )
                weight = DEFAULT_ENSEMBLE_WEIGHT
            used[name] = weight

        total = sum(used.values())
        if total <= 0:
            # Degenerate weights — fall back to a uniform blend over the loaded algos.
            log.warning("ensemble_weights_nonpositive", total=total, algos=sorted(used))
            equal = 1.0 / len(actions)
            used = dict.fromkeys(actions, equal)
            total = 1.0

        result: np.ndarray | None = None
        for name, action in actions.items():
            weighted = (used[name] / total) * action
            if result is None:
                result = weighted.copy()
            else:
                result += weighted

        assert result is not None  # nosec B101 — actions is non-empty (guarded above)
        return result
