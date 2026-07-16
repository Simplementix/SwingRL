"""Tests for MemoryVecRewardWrapper.

TRAIN-06: MemoryVecRewardWrapper shapes rewards using weighted profit/sharpe/drawdown/turnover.

Task 5 (spec §2.6, P-B1 verification): the real per-fold timestep budget passed to
model.learn() is DEFAULT_TIMESTEPS in src/swingrl/training/pipeline_helpers.py:45-48 --
equity 1,000,000 / crypto 500,000 (escalated on non-convergence to 2,000,000 /
1,000,000 respectively, ESCALATED_TIMESTEPS at pipeline_helpers.py:51-54). Percent-of-fold
window sizes (training.windows.short_pct_of_fold / trend_pct_of_fold) are computed against
whatever total_timesteps the run's model actually receives -- which is why
MemoryVecRewardWrapper.configure_windows() takes literal step counts (not percentages) and
is called once per fold, from model._total_timesteps, so an escalated run resizes correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper


def _make_mock_venv(n_envs: int = 1) -> MagicMock:
    """Create a mock VecEnv with standard step_wait() and reset() behavior.

    Args:
        n_envs: Number of parallel environments.

    Returns:
        MagicMock configured as a VecEnv.
    """
    mock = MagicMock()
    mock.num_envs = n_envs
    mock.observation_space = MagicMock()
    mock.action_space = MagicMock()

    obs = np.zeros((n_envs, 4), dtype=np.float32)
    rewards = np.ones(n_envs, dtype=np.float32)
    dones = np.zeros(n_envs, dtype=bool)
    infos: list[dict] = [{}] * n_envs

    mock.step_wait.return_value = (obs, rewards, dones, infos)
    mock.reset.return_value = obs
    return mock


def _make_wrapper() -> MemoryVecRewardWrapper:
    """Build a MemoryVecRewardWrapper (default weights) over a mock VecEnv.

    Used by tests that call `_shape_rewards` directly rather than via `step_wait()`.

    Returns:
        A wrapper instance with DEFAULT_WEIGHTS applied.
    """
    return MemoryVecRewardWrapper(_make_mock_venv())


class TestMemoryVecRewardWrapperInit:
    """TRAIN-06: Wrapper initializes with correct default weights."""

    def test_default_weights_sum_to_one(self) -> None:
        """Default weights sum to 1.0."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        total = sum(wrapper.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_normalized(self) -> None:
        """Custom weights are normalized to sum to 1.0."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        raw_weights = {"profit": 2.0, "sharpe": 1.0, "drawdown": 1.0, "turnover": 0.0}
        wrapper = MemoryVecRewardWrapper(mock_venv, initial_weights=raw_weights)
        total = sum(wrapper.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_weights_property_returns_copy(self) -> None:
        """Mutating the returned weights dict does not affect internal state."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        w1 = wrapper.weights
        w1["profit"] = 999.0
        w2 = wrapper.weights
        assert w2["profit"] != 999.0


class TestRewardShaping:
    """TRAIN-06: Rewards are shaped when info contains reward_components."""

    def test_passthrough_when_no_components(self) -> None:
        """Raw rewards pass through when info lacks reward_components."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        raw_reward = np.array([2.5], dtype=np.float32)
        mock_venv.step_wait.return_value = (np.zeros((1, 4)), raw_reward, np.array([False]), [{}])

        wrapper = MemoryVecRewardWrapper(mock_venv)
        _, shaped, _, _ = wrapper.step_wait()
        assert shaped[0] == pytest.approx(2.5)

    def test_passthrough_when_no_expected_keys(self) -> None:
        """Passthrough when reward_components lacks expected keys."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4)),
            np.array([3.0]),
            np.array([False]),
            [{"reward_components": {"unknown_key": 1.0}}],
        )

        wrapper = MemoryVecRewardWrapper(mock_venv)
        _, shaped, _, _ = wrapper.step_wait()
        assert shaped[0] == pytest.approx(3.0)

    def test_shapes_with_valid_components(self) -> None:
        """Rewards are shaped when all components present."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        components = {"profit": 1.0, "sharpe": 0.5, "drawdown": -0.2, "turnover": 0.1}
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4)),
            np.array([0.0]),
            np.array([False]),
            [{"reward_components": components}],
        )

        # Use known weights for deterministic check
        weights = {"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}
        wrapper = MemoryVecRewardWrapper(mock_venv, initial_weights=weights)
        _, shaped, _, _ = wrapper.step_wait()

        expected = 0.5 * 1.0 + 0.25 * 0.5 + 0.15 * (-0.2) + 0.10 * 0.1
        assert shaped[0] == pytest.approx(expected, abs=1e-5)

    def test_partial_components_uses_zero_for_missing(self) -> None:
        """Missing component keys default to 0.0 in weighted sum."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        # Only profit present
        components = {"profit": 1.0}
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4)),
            np.array([0.0]),
            np.array([False]),
            [{"reward_components": components}],
        )

        weights = {"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}
        wrapper = MemoryVecRewardWrapper(mock_venv, initial_weights=weights)
        _, shaped, _, _ = wrapper.step_wait()

        # Only profit contributes
        assert shaped[0] == pytest.approx(0.5, abs=1e-5)


class TestRiskPenaltySurvivesShaping:
    """A3: the risk penalty is a safety term that must survive reward shaping."""

    def test_shaping_subtracts_risk_penalty(self) -> None:
        """A3: shaped reward = weighted component sum MINUS the unweighted risk penalty."""
        wrapper = _make_wrapper()
        infos = [
            {
                "reward_components": {
                    "profit": 1.0,
                    "sharpe": 1.0,
                    "drawdown": 0.0,
                    "turnover": 0.0,
                },
                "risk_penalty": 0.5,
            }
        ]
        shaped = wrapper._shape_rewards(np.array([0.7]), infos)
        weighted = 0.50 * 1.0 + 0.25 * 1.0  # DEFAULT_WEIGHTS: profit .50, sharpe .25
        assert shaped[0] == pytest.approx(weighted - 0.5)

    def test_shaping_without_penalty_key_is_unchanged_behavior(self) -> None:
        """Missing risk_penalty key (old envs, unit fixtures) -> penalty treated as 0.0."""
        wrapper = _make_wrapper()
        infos = [
            {
                "reward_components": {
                    "profit": 1.0,
                    "sharpe": 0.0,
                    "drawdown": 0.0,
                    "turnover": 0.0,
                }
            }
        ]
        shaped = wrapper._shape_rewards(np.array([0.3]), infos)
        assert shaped[0] == pytest.approx(0.50)


class TestUpdateWeights:
    """TRAIN-06: update_weights changes reward shaping behavior."""

    def test_update_weights_normalizes(self) -> None:
        """update_weights normalizes new weights to sum to 1.0."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        wrapper.update_weights({"profit": 3.0, "sharpe": 1.0, "drawdown": 0.0, "turnover": 0.0})
        total = sum(wrapper.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_update_weights_changes_shaping(self) -> None:
        """Reward shape changes after update_weights."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        components = {"profit": 1.0, "sharpe": 1.0, "drawdown": 0.0, "turnover": 0.0}
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4)),
            np.array([0.0]),
            np.array([False]),
            [{"reward_components": components}],
        )

        # Start with profit-heavy weights
        wrapper = MemoryVecRewardWrapper(
            mock_venv,
            initial_weights={"profit": 0.9, "sharpe": 0.1, "drawdown": 0.0, "turnover": 0.0},
        )
        _, shaped_before, _, _ = wrapper.step_wait()

        # Switch to sharpe-heavy weights
        wrapper.update_weights({"profit": 0.1, "sharpe": 0.9, "drawdown": 0.0, "turnover": 0.0})
        _, shaped_after, _, _ = wrapper.step_wait()

        # With profit=sharpe=1.0, result should be same but confirm weights changed
        assert wrapper.weights["sharpe"] > wrapper.weights["profit"]


class TestRollingMetrics:
    """TRAIN-06: Rolling Sharpe, MDD, and win-rate calculations."""

    def test_rolling_sharpe_empty(self) -> None:
        """rolling_sharpe returns 0.0 with no history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.rolling_sharpe() == pytest.approx(0.0)

    def test_rolling_mdd_empty(self) -> None:
        """rolling_mdd returns 0.0 with no history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.rolling_mdd() == pytest.approx(0.0)

    def test_rolling_win_rate_empty(self) -> None:
        """rolling_win_rate returns 0.0 with no history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.rolling_win_rate() == pytest.approx(0.0)

    def test_rolling_mean_reward_empty(self) -> None:
        """rolling_mean_reward returns 0.0 with no history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.rolling_mean_reward() == pytest.approx(0.0)

    def test_rolling_mean_reward_with_history(self) -> None:
        """rolling_mean_reward returns mean of reward history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper._reward_history.extend([1.0, 2.0, 3.0, 4.0])
        assert wrapper.rolling_mean_reward() == pytest.approx(2.5)

    def test_rolling_sharpe_positive_rewards(self) -> None:
        """rolling_sharpe is positive when all rewards are positive."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4)),
            np.array([1.0]),
            np.array([False]),
            [{}],
        )

        wrapper = MemoryVecRewardWrapper(mock_venv)
        # Accumulate 10 positive steps
        for _ in range(10):
            wrapper.step_wait()

        sharpe = wrapper.rolling_sharpe()
        # All same rewards → std=0 → 0.0
        assert sharpe == pytest.approx(0.0)

    def test_rolling_sharpe_with_variance(self) -> None:
        """rolling_sharpe is computed from mean/std over rolling window."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Manually inject varied rewards into history
        rewards = [0.01, -0.005, 0.02, -0.01, 0.015]
        for r in rewards:
            wrapper._reward_history.append(r)
            wrapper._positive_steps.append(r > 0)

        sharpe = wrapper.rolling_sharpe()
        arr = np.array(rewards)
        expected = float(arr.mean() / arr.std(ddof=1) * np.sqrt(252))
        assert sharpe == pytest.approx(expected, rel=1e-4)

    def test_rolling_mdd_with_drawdown(self) -> None:
        """rolling_mdd (deprecated alias, Task 5/§2.6) returns negative equity-fraction
        drawdown -- now sourced from window_metrics("trend")["mdd_frac_worst"], NOT the
        old cumsum-of-shaped-rewards design (which this test used to drive directly via
        `_reward_history`). See TestPercentOfFoldWindows for the new contract's own tests;
        this test only proves the alias still returns a negative float on a real drawdown."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper.configure_windows(short_steps=10, trend_steps=10)

        # Portfolio value sequence: rise then fall (peak 110 -> trough 99)
        for pv in (100.0, 110.0, 99.0):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"portfolio_value": pv}],
            )
            wrapper.step_wait()

        mdd = wrapper.rolling_mdd()
        assert mdd < 0.0

    def test_rolling_win_rate_half(self) -> None:
        """rolling_win_rate is 0.5 when half rewards are positive."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # 4 positive, 4 negative
        for r in [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]:
            wrapper._positive_steps.append(r > 0)

        assert wrapper.rolling_win_rate() == pytest.approx(0.5)

    def test_reset_clears_history(self) -> None:
        """reset() clears rolling history."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Add some history
        wrapper._reward_history.append(1.0)
        wrapper._positive_steps.append(True)
        assert len(wrapper._reward_history) == 1

        wrapper.reset()
        assert len(wrapper._reward_history) == 0
        assert wrapper.rolling_win_rate() == pytest.approx(0.0)


class TestRollingTradeRate:
    """TRAIN-TRADE-01/02: rolling trade rate + first-window self-baseline."""

    def test_trade_rate_counts_trades_per_step(self) -> None:
        """TRAIN-TRADE-01: rate = mean trades_this_step over the window."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Alternate: step with 1 trade, step with 0 trades → mean 0.5 over 10 steps
        for i in range(10):
            trades = 1 if i % 2 == 0 else 0
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": trades}],
            )
            wrapper.step_wait()

        assert wrapper.rolling_trade_rate() == pytest.approx(0.5)

    def test_empty_history_rate_zero(self) -> None:
        """TRAIN-TRADE-01: no steps → 0.0 (matches other rolling metrics)."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.rolling_trade_rate() == pytest.approx(0.0)

    def test_missing_info_key_counts_zero(self) -> None:
        """TRAIN-TRADE-01: infos without trades_this_step (old envs) count 0, no crash."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        # Default mock returns infos=[{}] — no trades_this_step key
        wrapper = MemoryVecRewardWrapper(mock_venv)

        for _ in range(5):
            wrapper.step_wait()

        # Should be 0.0 since no trades_this_step in any info
        assert wrapper.rolling_trade_rate() == pytest.approx(0.0)

    def test_baseline_locks_at_first_full_window(self) -> None:
        """TRAIN-TRADE-02: baseline_trade_rate() is 0.0 until the rolling window first
        fills, then locks to that window's rate permanently."""
        from swingrl.memory.training.reward_wrapper import (
            _ROLLING_WINDOW,
            MemoryVecRewardWrapper,
        )

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Before window fills: baseline stays 0.0
        for _ in range(_ROLLING_WINDOW - 1):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": 1}],
            )
            wrapper.step_wait()

        assert wrapper.baseline_trade_rate() == pytest.approx(0.0)

        # One more step fills the window → baseline locks at 1.0 (1 trade/step)
        mock_venv.step_wait.return_value = (
            np.zeros((1, 4), dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
            [{"trades_this_step": 1}],
        )
        wrapper.step_wait()

        assert wrapper.baseline_trade_rate() == pytest.approx(1.0)
        assert len(wrapper._trades_per_step) == _ROLLING_WINDOW

        # Drive another full window at 0 trades — baseline stays locked at 1.0
        for _ in range(_ROLLING_WINDOW):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": 0}],
            )
            wrapper.step_wait()

        assert wrapper.baseline_trade_rate() == pytest.approx(1.0)
        assert wrapper.rolling_trade_rate() == pytest.approx(0.0)

    def test_reset_clears_trade_history_and_baseline(self) -> None:
        """TRAIN-TRADE-02: reset() clears trade history and the locked baseline."""
        from swingrl.memory.training.reward_wrapper import (
            _ROLLING_WINDOW,
            MemoryVecRewardWrapper,
        )

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Fill the window so baseline locks
        for _ in range(_ROLLING_WINDOW):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": 1}],
            )
            wrapper.step_wait()

        assert wrapper.baseline_trade_rate() == pytest.approx(1.0)

        # reset() must clear trade history and baseline
        wrapper.reset()

        assert wrapper.rolling_trade_rate() == pytest.approx(0.0)
        assert wrapper.baseline_trade_rate() == pytest.approx(0.0)
        assert len(wrapper._trades_per_step) == 0

    def test_zero_trade_first_window_locks_baseline_at_zero(self) -> None:
        """TRAIN-TRADE-02: a zero-trade first full window locks baseline 0.0
        permanently — the collapse detector stays disabled even after trading starts."""
        from swingrl.memory.training.reward_wrapper import (
            _ROLLING_WINDOW,
            MemoryVecRewardWrapper,
        )

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Drive exactly _ROLLING_WINDOW steps at 0 trades → first full window = 0.0
        for _ in range(_ROLLING_WINDOW):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": 0}],
            )
            wrapper.step_wait()

        # Baseline must be locked at 0.0 after the zero-trade window fills
        assert wrapper.baseline_trade_rate() == pytest.approx(0.0)

        # Drive another _ROLLING_WINDOW steps at 2 trades/step — trading resumes
        for _ in range(_ROLLING_WINDOW):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"trades_this_step": 2}],
            )
            wrapper.step_wait()

        # rolling_trade_rate reflects the new activity
        assert wrapper.rolling_trade_rate() == pytest.approx(2.0)
        # baseline must still be 0.0 (locked from the zero-trade first window)
        assert wrapper.baseline_trade_rate() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestPercentOfFoldWindows (Task 5, spec §2.6)
# ---------------------------------------------------------------------------


class TestPercentOfFoldWindows:
    """Task 5 (spec §2.6): percent-of-fold windows, equity-fraction MDD, dual-unit output.

    Replaces the fixed 500-step deque design for MDD specifically: window MDD is now
    computed per sub-env from portfolio-value curves (peak-to-trough fraction), not
    from cumsum-of-shaped-rewards. rolling_sharpe()/rolling_win_rate()/rolling_trade_rate()/
    baseline_trade_rate() are untouched by this task (still _ROLLING_WINDOW-based) --
    only rolling_mdd() becomes a deprecated alias onto the new contract.
    """

    def test_configure_windows_sizes_deques_and_reports_dual_units(self) -> None:
        """configure_windows(short_steps, trend_steps) sizes both windows; window_metrics
        reports BOTH units (pct_of_fold from config/swingrl.yaml training.windows N1/N2
        defaults, and steps from the literal configure_windows() call) for each window."""
        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper.configure_windows(short_steps=10_000, trend_steps=150_000)

        short = wrapper.window_metrics("short")
        trend = wrapper.window_metrics("trend")

        assert short["steps"] == 10_000
        assert trend["steps"] == 150_000
        assert short["pct_of_fold"] == pytest.approx(0.01)  # N1 default
        assert trend["pct_of_fold"] == pytest.approx(0.15)  # N2 default

        expected_keys = {
            "pct_of_fold",
            "steps",
            "sharpe_annualized",
            "mdd_frac_worst",
            "mdd_frac_mean",
            "win_rate",
            "trade_rate",
        }
        assert set(short) == expected_keys
        assert set(trend) == expected_keys

    def test_trend_steps_property_is_o1_accessor_matching_window_metrics(self) -> None:
        """trend_steps property (perf fix) mirrors window_metrics("trend")["steps"]
        without paying the deque-iteration + numpy recompute cost -- 0 before
        configure_windows() has run, and the configured value after."""
        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        assert wrapper.trend_steps == 0

        wrapper.configure_windows(short_steps=10_000, trend_steps=150_000)
        assert wrapper.trend_steps == 150_000
        assert wrapper.trend_steps == wrapper.window_metrics("trend")["steps"]

    def test_window_metrics_invalid_window_raises(self) -> None:
        """window_metrics() only accepts "short" or "trend"."""
        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        with pytest.raises(ValueError):
            wrapper.window_metrics("bogus")  # type: ignore[arg-type]

    def test_mdd_frac_worst_from_portfolio_value_sequence(self) -> None:
        """Synthetic single-sub-env portfolio curve 100 -> 110 -> 99: peak-to-trough
        fraction = (99 - 110) / 110 = -0.1 -> mdd_frac_worst == approx(0.1) (magnitude)."""
        mock_venv = _make_mock_venv(n_envs=1)
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper.configure_windows(short_steps=10, trend_steps=10)

        for pv in (100.0, 110.0, 99.0):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"portfolio_value": pv}],
            )
            wrapper.step_wait()

        metrics = wrapper.window_metrics("trend")
        assert metrics["mdd_frac_worst"] == pytest.approx(0.1, abs=1e-6)
        # single sub-env: worst == mean
        assert metrics["mdd_frac_mean"] == pytest.approx(0.1, abs=1e-6)

    def test_mdd_frac_worst_vs_mean_across_sub_envs(self) -> None:
        """Locked design decision (2026-07-12): mdd_frac_worst = worst sub-env (safety-first,
        drives triggers); mdd_frac_mean = mean across sub-envs (analysis-only). With one
        sub-env drawing down 10% and another flat, worst=0.10 and mean=0.05."""
        mock_venv = _make_mock_venv(n_envs=2)
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper.configure_windows(short_steps=10, trend_steps=10)

        # env0: 100 -> 110 -> 99 (10% drawdown); env1: flat at 100 (0% drawdown)
        for pv0, pv1 in ((100.0, 100.0), (110.0, 100.0), (99.0, 100.0)):
            mock_venv.step_wait.return_value = (
                np.zeros((2, 4), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=bool),
                [{"portfolio_value": pv0}, {"portfolio_value": pv1}],
            )
            wrapper.step_wait()

        metrics = wrapper.window_metrics("trend")
        assert metrics["mdd_frac_worst"] == pytest.approx(0.1, abs=1e-6)
        assert metrics["mdd_frac_mean"] == pytest.approx(0.05, abs=1e-6)

    def test_mdd_frac_zero_when_window_empty(self) -> None:
        """No steps recorded yet (or configure_windows never called) -> 0.0, not a crash."""
        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        metrics = wrapper.window_metrics("trend")
        assert metrics["mdd_frac_worst"] == pytest.approx(0.0)
        assert metrics["mdd_frac_mean"] == pytest.approx(0.0)

    def test_rolling_mdd_alias_negates_trend_worst(self) -> None:
        """Deprecated alias: rolling_mdd() == -window_metrics("trend")["mdd_frac_worst"]
        (Task 6 rewires the remaining callers off this alias onto window_metrics directly)."""
        mock_venv = _make_mock_venv(n_envs=1)
        wrapper = MemoryVecRewardWrapper(mock_venv)
        wrapper.configure_windows(short_steps=10, trend_steps=10)

        for pv in (100.0, 110.0, 99.0):
            mock_venv.step_wait.return_value = (
                np.zeros((1, 4), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=bool),
                [{"portfolio_value": pv}],
            )
            wrapper.step_wait()

        assert wrapper.rolling_mdd() == pytest.approx(-0.1, abs=1e-6)
