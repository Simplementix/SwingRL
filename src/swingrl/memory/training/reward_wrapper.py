"""Memory-guided reward shaping wrapper for VecEnv training.

MemoryVecRewardWrapper wraps a VecEnv and reshapes raw rewards using
a weighted combination of profit, Sharpe, drawdown, and turnover components.
Weights can be updated live by the MemoryEpochCallback when the LLM
suggests a re-weighting based on training health metrics.

Usage:
    from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper
    wrapped_env = MemoryVecRewardWrapper(vec_env, initial_weights={"profit": 0.5, ...})
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import structlog
from stable_baselines3.common.vec_env import VecEnvWrapper

log = structlog.get_logger(__name__)

# Component keys expected in the info dict from environments
REWARD_COMPONENT_KEYS = ("profit", "sharpe", "drawdown", "turnover")

# Default weights before any LLM advice
DEFAULT_WEIGHTS: dict[str, float] = {
    "profit": 0.50,
    "sharpe": 0.25,
    "drawdown": 0.15,
    "turnover": 0.10,
}

# Rolling window for Sharpe / MDD / win-rate metrics
_ROLLING_WINDOW = 500


class MemoryVecRewardWrapper(VecEnvWrapper):
    """VecEnvWrapper that shapes rewards via weighted profit/sharpe/drawdown/turnover components.

    The environment step() info dict must contain 'reward_components' with keys matching
    REWARD_COMPONENT_KEYS for shaping to activate. When the info dict lacks these keys,
    the original reward is passed through unchanged.

    Args:
        venv: VecEnv to wrap.
        initial_weights: Initial reward weights (default: DEFAULT_WEIGHTS).
    """

    def __init__(
        self,
        venv: Any,
        initial_weights: dict[str, float] | None = None,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize reward wrapper with optional weight override.

        Args:
            venv: VecEnv to wrap.
            initial_weights: Initial reward weights. Missing keys use DEFAULT_WEIGHTS.
            periods_per_year: Trading periods per year for Sharpe annualization
                (252 for equity daily, 2191 for crypto 4H).
        """
        super().__init__(venv)
        self._periods_per_year = periods_per_year

        # Merge provided weights with defaults
        weights = dict(DEFAULT_WEIGHTS)
        if initial_weights:
            weights.update(initial_weights)

        self._weights = self._normalize_weights(weights)

        # Rolling history for metrics (per-env index 0 only, single-env training)
        self._reward_history: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._positive_steps: deque[bool] = deque(maxlen=_ROLLING_WINDOW)

        log.info(
            "reward_wrapper_init",
            weights=self._weights,
        )

    @property
    def weights(self) -> dict[str, float]:
        """Current reward shaping weights (normalized to sum=1.0)."""
        return dict(self._weights)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step the environment and apply reward shaping.

        Returns:
            Tuple of (observations, shaped_rewards, dones, infos).
        """
        result = self.venv.step_wait()
        obs: np.ndarray = np.asarray(result[0])
        rewards: np.ndarray = np.asarray(result[1])
        dones: np.ndarray = np.asarray(result[2])
        infos: list[dict[str, Any]] = list(result[3])
        shaped = self._shape_rewards(rewards, infos)

        # Track rolling history
        for r in shaped:
            self._reward_history.append(float(r))
            self._positive_steps.append(float(r) > 0.0)

        return obs, shaped, dones, infos

    def reset(self) -> np.ndarray:
        """Reset environment and clear rolling history.

        Returns:
            Initial observations.
        """
        obs: np.ndarray = np.asarray(self.venv.reset())
        self._reward_history.clear()
        self._positive_steps.clear()
        return obs

    def _shape_rewards(
        self,
        rewards: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> np.ndarray:
        """Apply weighted shaping to raw rewards.

        If info dict contains 'reward_components' with matching keys, compute
        a weighted sum. Otherwise, pass raw reward through unchanged.

        Args:
            rewards: Raw reward array from environment.
            infos: Info dicts from environment step.

        Returns:
            Shaped reward array (same shape as rewards).
        """
        shaped = rewards.copy()

        for i, info in enumerate(infos):
            components = info.get("reward_components")
            if not isinstance(components, dict):
                continue

            # Check that at least one expected key is present
            if not any(k in components for k in REWARD_COMPONENT_KEYS):
                continue

            # Compute weighted reward from components
            weighted_reward = 0.0
            for key in REWARD_COMPONENT_KEYS:
                val = components.get(key, 0.0)
                weight = self._weights.get(key, 0.0)
                weighted_reward += weight * float(val)

            shaped[i] = weighted_reward

        return shaped

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update reward shaping weights (called by MemoryEpochCallback on LLM advice).

        Normalizes the new weights to sum to 1.0 before storing.

        Args:
            new_weights: New weights dict with keys from REWARD_COMPONENT_KEYS.
        """
        old_weights = dict(self._weights)
        self._weights = self._normalize_weights(new_weights)
        log.info(
            "reward_weights_updated",
            old_weights=old_weights,
            new_weights=self._weights,
        )

    def rolling_sharpe(self) -> float:
        """Compute Sharpe ratio over the rolling window.

        Returns:
            Annualized Sharpe ratio. 0.0 if fewer than 2 observations.
        """
        if len(self._reward_history) < 2:
            return 0.0
        arr = np.array(self._reward_history)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-10:
            return 0.0
        # Annualize using configured periods (252 equity daily, 2191 crypto 4H)
        return float(mean / std * np.sqrt(self._periods_per_year))

    def rolling_mdd(self) -> float:
        """Compute maximum drawdown over the rolling window.

        Returns:
            Maximum drawdown as a negative float (e.g., -0.05 for 5% DD). 0.0 if empty.
        """
        if not self._reward_history:
            return 0.0
        arr = np.array(self._reward_history)
        cumsum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = cumsum - running_max
        return float(np.min(drawdown))

    def rolling_mean_reward(self) -> float:
        """Compute mean reward over the rolling window.

        Returns:
            Mean per-step reward. 0.0 if no steps recorded.
        """
        if not self._reward_history:
            return 0.0
        return float(sum(self._reward_history) / len(self._reward_history))

    def rolling_win_rate(self) -> float:
        """Compute fraction of steps with positive reward over rolling window.

        Returns:
            Win rate in [0.0, 1.0]. 0.0 if no steps recorded.
        """
        if not self._positive_steps:
            return 0.0
        return float(sum(self._positive_steps)) / len(self._positive_steps)

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Raw weight dict.

        Returns:
            Normalized weight dict. Returns uniform DEFAULT_WEIGHTS if total is 0.
        """
        clamped = {k: max(0.0, v) for k, v in weights.items()}
        total = sum(clamped.values())
        if total <= 0:
            log.warning("reward_weights_zero_using_defaults")
            total_default = sum(DEFAULT_WEIGHTS.values())
            return {k: v / total_default for k, v in DEFAULT_WEIGHTS.items()}
        return {k: v / total for k, v in clamped.items()}
