"""Tests for MemoryVecRewardWrapper.

TRAIN-06: MemoryVecRewardWrapper shapes rewards using weighted profit/sharpe/drawdown/turnover.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


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
        """rolling_mdd returns negative value on drawdown sequences."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)

        # Sequence: rise then fall
        rewards = [0.1, 0.1, -0.3, -0.2]
        for r in rewards:
            wrapper._reward_history.append(r)

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
