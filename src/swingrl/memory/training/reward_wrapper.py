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
from typing import Any, Literal

import numpy as np
import structlog
from stable_baselines3.common.vec_env import VecEnvWrapper

from swingrl.config.schema import load_config

log = structlog.get_logger(__name__)

# Component keys expected in the info dict from environments
REWARD_COMPONENT_KEYS = ("profit", "sharpe", "drawdown", "turnover")

# Info dict key carrying the env's built-in risk penalty (position concentration +
# drawdown). A3 (spec §2.11): this is a safety term, never reweightable -- it is
# subtracted outside the weighted component sum rather than folded into it.
RISK_PENALTY_INFO_KEY = "risk_penalty"

# Default weights before any LLM advice
DEFAULT_WEIGHTS: dict[str, float] = {
    "profit": 0.50,
    "sharpe": 0.25,
    "drawdown": 0.15,
    "turnover": 0.10,
}

# Rolling window for Sharpe / MDD / win-rate metrics
_ROLLING_WINDOW = 500

# Fallback percent-of-fold window targets (spec §2.6, N1/N2) — used only if config
# load fails entirely. Must match TrainingWindowsConfig's pydantic defaults
# (config/schema.py) so a config-load failure degrades to the shipped intent.
_FALLBACK_SHORT_PCT_OF_FOLD: float = 0.01
_FALLBACK_TREND_PCT_OF_FOLD: float = 0.15


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
        self._trades_per_step: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._baseline_trade_rate: float = 0.0
        self._baseline_locked: bool = False

        # Percent-of-fold windows (spec §2.6): short = acute detector, trend = decision
        # basis. Percent targets are config-sourced (dual-unit reporting in
        # window_metrics()); actual sizes (in steps) start at 0/empty until the
        # callback's _on_training_start calls configure_windows() with the run's REAL
        # total_timesteps -- so escalated runs (ESCALATED_TIMESTEPS) resize correctly.
        self._short_pct_of_fold, self._trend_pct_of_fold = self._load_window_pcts()
        self._short_steps: int = 0
        self._trend_steps: int = 0
        self._short_window: deque[dict[str, Any]] = deque(maxlen=0)
        self._trend_window: deque[dict[str, Any]] = deque(maxlen=0)

        log.info(
            "reward_wrapper_init",
            weights=self._weights,
            short_pct_of_fold=self._short_pct_of_fold,
            trend_pct_of_fold=self._trend_pct_of_fold,
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

        # Track rolling trade activity
        trades = 0.0
        for info in infos:
            trades += float(info.get("trades_this_step", 0))
        self._trades_per_step.append(trades)
        if not self._baseline_locked and len(self._trades_per_step) == _ROLLING_WINDOW:
            self._baseline_trade_rate = float(
                sum(self._trades_per_step) / len(self._trades_per_step)
            )
            self._baseline_locked = True

        # Percent-of-fold window storage (spec §2.6): one entry per step_wait() call,
        # carrying per-sub-env shaped rewards and portfolio values (needed for the
        # per-sub-env drawdown decomposition) plus this call's summed trade count.
        # No-ops until configure_windows() sets a nonzero maxlen (deque(maxlen=0)
        # silently discards every append).
        entry = {
            "rewards": [float(r) for r in shaped],
            "portfolio_values": [float(info.get("portfolio_value", 0.0)) for info in infos],
            "trades": trades,
        }
        self._short_window.append(entry)
        self._trend_window.append(entry)

        return obs, shaped, dones, infos

    def reset(self) -> np.ndarray:
        """Reset environment and clear rolling history.

        Returns:
            Initial observations.
        """
        obs: np.ndarray = np.asarray(self.venv.reset())
        self._reward_history.clear()
        self._positive_steps.clear()
        self._trades_per_step.clear()
        self._baseline_trade_rate = 0.0
        self._baseline_locked = False
        self._short_window.clear()
        self._trend_window.clear()
        return obs

    def _shape_rewards(
        self,
        rewards: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> np.ndarray:
        """Apply weighted shaping to raw rewards.

        If info dict contains 'reward_components' with matching keys, compute
        a weighted sum and then subtract the env's risk penalty (info['risk_penalty'],
        default 0.0) outside that sum -- the safety term is never reweightable
        (A3, spec §2.11). Otherwise, pass raw reward through unchanged.

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

            # A3: the risk penalty is a safety term, never reweightable — it is
            # subtracted outside the weighted component sum (spec §2.11, amendment A3).
            weighted_reward -= float(info.get(RISK_PENALTY_INFO_KEY, 0.0))
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
        """DEPRECATED (spec §2.6, Task 5): use window_metrics("trend")["mdd_frac_worst"].

        Retained as a compatibility alias for existing diagnosis call sites
        (MemoryEpochCallback._should_store, _collect_metrics, _ingest_adjustment_trigger,
        _resolve_pending_adjustment) -- Task 6 rewires those callers directly onto
        window_metrics() and removes this alias. No longer the cumsum-of-shaped-rewards
        design (unbounded scale); now the trend window's worst-sub-env equity-fraction
        drawdown, negated to preserve the historical negative-means-drawdown sign
        convention even though the underlying metric is a non-negative magnitude.

        Returns:
            Negative equity-fraction drawdown (e.g. -0.10 for a 10% worst-sub-env
            drawdown in the trend window). 0.0 if the window has no data yet.
        """
        return -float(self.window_metrics("trend")["mdd_frac_worst"])

    def configure_windows(self, short_steps: int, trend_steps: int) -> None:
        """Size the short/trend percent-of-fold windows (spec §2.6).

        Called once by MemoryEpochCallback._on_training_start with the run's REAL
        total_timesteps (model._total_timesteps) -- so escalated runs (spec:
        ESCALATED_TIMESTEPS) resize correctly rather than inheriting a size baked in
        at construction time. Both step counts are retained (alongside the
        config-sourced percent targets from __init__) so window_metrics() can report
        dual units. Always starts both windows fresh (empty) at the new size.

        Args:
            short_steps: Short (acute-detector) window size, in total-timesteps units
                (same units as model.num_timesteps / model._total_timesteps).
            trend_steps: Trend (decision-basis) window size, in total-timesteps units.
        """
        self._short_steps = int(short_steps)
        self._trend_steps = int(trend_steps)

        # total-timesteps units advance by num_envs per step_wait() call, so the
        # window's call-count (deque maxlen) is the timestep size divided by num_envs.
        n_envs = max(self.num_envs, 1)
        short_calls = max(1, round(self._short_steps / n_envs))
        trend_calls = max(1, round(self._trend_steps / n_envs))
        self._short_window = deque(maxlen=short_calls)
        self._trend_window = deque(maxlen=trend_calls)

        log.info(
            "reward_wrapper_windows_configured",
            short_steps=self._short_steps,
            trend_steps=self._trend_steps,
            short_calls=short_calls,
            trend_calls=trend_calls,
            num_envs=self.num_envs,
        )

    def window_metrics(self, window: Literal["short", "trend"]) -> dict[str, Any]:
        """Dual-unit metrics for the short (acute) or trend (decision) window (spec §2.6).

        MDD is computed per sub-env from that sub-env's own portfolio-value curve
        (peak-to-trough fraction within the window), then both the worst sub-env and
        the mean across sub-envs are recorded (locked decision, 2026-07-12): triggers
        and coach evidence must read mdd_frac_worst (safety-first, conservative);
        mdd_frac_mean rides along for analysis/threshold recalibration only -- never
        the alarm basis. The two bases are never mixed.

        Args:
            window: Which window to summarize -- "short" or "trend".

        Returns:
            Dict with pct_of_fold, steps (dual-unit observability, D-T2.7),
            sharpe_annualized, mdd_frac_worst, mdd_frac_mean, win_rate, and trade_rate.
            All metrics default to 0.0 when the window has fewer than 2 recorded steps.

        Raises:
            ValueError: window is not "short" or "trend".
        """
        if window == "short":
            pct_of_fold = self._short_pct_of_fold
            steps = self._short_steps
            buf = self._short_window
        elif window == "trend":
            pct_of_fold = self._trend_pct_of_fold
            steps = self._trend_steps
            buf = self._trend_window
        else:
            raise ValueError(f"window must be 'short' or 'trend', got {window!r}")

        all_rewards: list[float] = []
        all_trades: list[float] = []
        n_envs = self.num_envs
        per_env_curves: list[list[float]] = [[] for _ in range(n_envs)]

        for entry in buf:
            all_rewards.extend(entry["rewards"])
            all_trades.append(entry["trades"])
            for i, pv in enumerate(entry["portfolio_values"]):
                if i < n_envs:
                    per_env_curves[i].append(pv)

        mdd_frac_worst, mdd_frac_mean = self._portfolio_mdd_fracs(per_env_curves)

        return {
            "pct_of_fold": pct_of_fold,
            "steps": steps,
            "sharpe_annualized": self._annualized_sharpe(all_rewards),
            "mdd_frac_worst": mdd_frac_worst,
            "mdd_frac_mean": mdd_frac_mean,
            "win_rate": (
                float(sum(1 for r in all_rewards if r > 0.0)) / len(all_rewards)
                if all_rewards
                else 0.0
            ),
            "trade_rate": float(sum(all_trades) / len(all_trades)) if all_trades else 0.0,
        }

    def _annualized_sharpe(self, rewards: list[float]) -> float:
        """Mean/std annualized Sharpe over an arbitrary reward sample.

        Args:
            rewards: Flat list of per-sub-env shaped rewards pooled across a window.

        Returns:
            Annualized Sharpe ratio. 0.0 if fewer than 2 observations or std < 1e-10.
        """
        if len(rewards) < 2:
            return 0.0
        arr = np.array(rewards)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(mean / std * np.sqrt(self._periods_per_year))

    @staticmethod
    def _portfolio_mdd_fracs(curves: list[list[float]]) -> tuple[float, float]:
        """Peak-to-trough drawdown fraction per sub-env curve -> (worst, mean).

        Args:
            curves: One portfolio-value time series per sub-env, computed within the
                window only (no external/pre-window running peak carried in).

        Returns:
            Tuple of (mdd_frac_worst, mdd_frac_mean) -- both non-negative fractions.
            (0.0, 0.0) if no sub-env curve has at least 2 points yet.
        """
        per_env_dd: list[float] = []
        for curve in curves:
            if len(curve) < 2:
                continue
            arr = np.array(curve)
            running_max = np.maximum.accumulate(arr)
            safe_max = np.where(running_max > 0.0, running_max, 1.0)
            drawdown = (arr - running_max) / safe_max
            per_env_dd.append(float(-np.min(drawdown)))  # magnitude, non-negative

        if not per_env_dd:
            return 0.0, 0.0
        return max(per_env_dd), float(sum(per_env_dd) / len(per_env_dd))

    @staticmethod
    def _load_window_pcts() -> tuple[float, float]:
        """Load percent-of-fold window targets from config, with hardcoded fallback.

        Returns:
            Tuple of (short_pct_of_fold, trend_pct_of_fold).
        """
        try:
            cfg = load_config()
            return (
                cfg.training.windows.short_pct_of_fold,
                cfg.training.windows.trend_pct_of_fold,
            )
        except Exception as exc:
            log.warning("training_windows_config_load_failed", error=str(exc), fallback="hardcoded")
            return _FALLBACK_SHORT_PCT_OF_FOLD, _FALLBACK_TREND_PCT_OF_FOLD

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

    def rolling_trade_rate(self) -> float:
        """Mean trades per step over the rolling window. 0.0 if empty."""
        if not self._trades_per_step:
            return 0.0
        return float(sum(self._trades_per_step) / len(self._trades_per_step))

    def baseline_trade_rate(self) -> float:
        """The fold's first-full-window trade rate (0.0 until the window fills).

        Locked once: later windows never overwrite it, so a mid-fold activity
        collapse is measured against how the fold STARTED trading.

        Note: a fold whose first full window contains zero trades locks baseline
        at 0.0, permanently disabling the collapse detector for that fold
        (``diagnose_rolling`` treats baseline 0.0 as "window not yet full").
        This is conservative and intentional — fail-safe over false alarms.
        """
        return self._baseline_trade_rate

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
