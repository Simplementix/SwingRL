"""SB3 training callbacks for convergence-based early stopping.

ConvergenceCallback monitors mean reward improvement from EvalCallback
and stops training when improvement stagnates for a configurable patience
window. Used by TrainingOrchestrator during PPO/A2C/SAC training.
"""

from __future__ import annotations

import structlog
from stable_baselines3.common.callbacks import BaseCallback

log = structlog.get_logger(__name__)


class ConvergenceCallback(BaseCallback):
    """Early stopping callback based on reward convergence.

    Stops training when mean reward improvement falls below a threshold
    for a specified number of consecutive evaluations. Must be used as
    a child callback of EvalCallback (set via parent attribute).

    Args:
        min_improvement_pct: Minimum relative improvement to count as progress.
            Uses absolute comparison fallback when best reward is near zero.
        patience: Number of consecutive stagnant evaluations before stopping.
        verbose: Verbosity level for BaseCallback.
    """

    def __init__(
        self,
        min_improvement_pct: float = 0.01,
        patience: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._min_improvement_pct = min_improvement_pct
        self._patience = patience
        self._best_reward: float | None = None
        self._stagnation_count: int = 0

    def _on_step(self) -> bool:
        """Check reward improvement and decide whether to continue.

        Returns:
            True to continue training, False to stop.
        """
        if self.parent is None:
            return True

        current_reward: float = self.parent.last_mean_reward  # type: ignore[attr-defined]

        # First evaluation: record baseline and continue
        if self._best_reward is None:
            self._best_reward = current_reward
            return True

        # Check improvement
        if self._has_improved(current_reward):
            self._best_reward = current_reward
            self._stagnation_count = 0
            return True

        self._stagnation_count += 1

        if self._stagnation_count >= self._patience:
            log.info(
                "convergence_detected",
                best_reward=self._best_reward,
                current_reward=current_reward,
                stagnation_count=self._stagnation_count,
                patience=self._patience,
            )
            return False

        return True

    def _has_improved(self, current_reward: float) -> bool:
        """Determine if current reward represents meaningful improvement.

        Uses relative improvement when best reward magnitude is large enough.
        Falls back to absolute comparison when best reward is near zero to
        avoid division-by-zero.

        Args:
            current_reward: The latest mean reward from EvalCallback.

        Returns:
            True if improvement exceeds the minimum threshold.
        """
        if self._best_reward is None:  # pragma: no cover
            return False

        abs_best = abs(self._best_reward)

        # Absolute fallback for near-zero best reward
        if abs_best < 1e-8:
            return abs(current_reward - self._best_reward) > self._min_improvement_pct

        # For negative rewards: improvement means moving toward zero (increasing)
        # For positive rewards: improvement means increasing
        # In both cases, higher is better
        improvement = (current_reward - self._best_reward) / abs_best

        return improvement > self._min_improvement_pct
