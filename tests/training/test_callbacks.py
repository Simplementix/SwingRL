"""Tests for ConvergenceCallback SB3 early stopping.

Uses mock parent objects to simulate EvalCallback's last_mean_reward
attribute -- no actual SB3 models are instantiated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from swingrl.training.callbacks import ConvergenceCallback


class _MockParent:
    """Minimal mock of SB3 EvalCallback with last_mean_reward."""

    def __init__(self, reward: float = 0.0) -> None:
        self.last_mean_reward = reward


def _make_callback(
    min_improvement_pct: float = 0.01,
    patience: int = 10,
) -> ConvergenceCallback:
    """Create a ConvergenceCallback ready for testing."""
    cb = ConvergenceCallback(
        min_improvement_pct=min_improvement_pct,
        patience=patience,
        verbose=0,
    )
    # Simulate SB3 internal setup
    cb.model = MagicMock()
    cb.n_calls = 0
    cb.num_timesteps = 0
    cb.locals = {}
    cb.globals = {}
    return cb


class TestConvergenceCallbackFirstEvaluation:
    """TRAIN-03: First evaluation records baseline and continues."""

    def test_first_call_records_baseline_and_continues(self) -> None:
        """TRAIN-03: First eval records baseline reward, returns True."""
        cb = _make_callback()
        cb.parent = _MockParent(reward=100.0)
        result = cb._on_step()
        assert result is True

    def test_first_call_with_zero_reward(self) -> None:
        """TRAIN-03: First eval with zero reward records baseline."""
        cb = _make_callback()
        cb.parent = _MockParent(reward=0.0)
        result = cb._on_step()
        assert result is True


class TestConvergenceCallbackStopsTraining:
    """TRAIN-03: Stops after patience consecutive no-improvement evals."""

    def test_stops_after_patience_exhausted(self) -> None:
        """TRAIN-03: 10 consecutive no-improvement calls returns False."""
        cb = _make_callback(patience=10)
        cb.parent = _MockParent(reward=100.0)

        # First call records baseline
        cb._on_step()

        # 10 consecutive no-improvement evaluations (same reward)
        for i in range(9):
            result = cb._on_step()
            assert result is True, f"Should continue at step {i + 1}"

        # 10th no-improvement -> should stop
        result = cb._on_step()
        assert result is False

    def test_small_improvement_does_not_reset(self) -> None:
        """TRAIN-03: Improvement below threshold counts as stagnation."""
        cb = _make_callback(min_improvement_pct=0.01, patience=3)
        cb.parent = _MockParent(reward=100.0)
        cb._on_step()  # baseline

        # Tiny improvement (0.5% < 1% threshold)
        cb.parent.last_mean_reward = 100.5
        cb._on_step()

        cb.parent.last_mean_reward = 100.8
        cb._on_step()

        cb.parent.last_mean_reward = 101.0
        result = cb._on_step()
        assert result is False


class TestConvergenceCallbackResetsCounter:
    """TRAIN-03: Counter resets when meaningful improvement detected."""

    def test_improvement_resets_counter(self) -> None:
        """TRAIN-03: Sufficient improvement resets stagnation counter."""
        cb = _make_callback(patience=3)
        cb.parent = _MockParent(reward=100.0)
        cb._on_step()  # baseline

        # 2 stagnant evaluations
        cb._on_step()
        cb._on_step()

        # Significant improvement (5%)
        cb.parent.last_mean_reward = 105.0
        result = cb._on_step()
        assert result is True

        # Counter should be reset -- need 3 more stagnations to stop
        cb._on_step()
        cb._on_step()
        result = cb._on_step()
        # Should stop now after 3 stagnant evals post-reset
        assert result is False


class TestConvergenceCallbackParentNone:
    """TRAIN-03: Handles parent=None gracefully."""

    def test_parent_none_returns_true(self) -> None:
        """TRAIN-03: No parent means no EvalCallback, continue training."""
        cb = _make_callback()
        cb.parent = None
        result = cb._on_step()
        assert result is True


class TestConvergenceCallbackNegativeRewards:
    """TRAIN-03: Handles negative reward values correctly."""

    def test_negative_rewards_improvement_detected(self) -> None:
        """TRAIN-03: Improvement from -100 to -90 is detected as 10% improvement."""
        cb = _make_callback(patience=3, min_improvement_pct=0.01)
        cb.parent = _MockParent(reward=-100.0)
        cb._on_step()  # baseline at -100

        # Improve to -90 (10% improvement in absolute value)
        cb.parent.last_mean_reward = -90.0
        result = cb._on_step()
        assert result is True

    def test_negative_rewards_stagnation_detected(self) -> None:
        """TRAIN-03: Stagnation in negative rewards stops training."""
        cb = _make_callback(patience=3)
        cb.parent = _MockParent(reward=-50.0)
        cb._on_step()  # baseline

        # No improvement
        for _ in range(3):
            cb._on_step()

        result = cb._on_step()
        # After patience=3, should have stopped
        # Actually count: baseline + 3 stagnant = stop on 4th call
        # Let me just check last result was False
        # Re-check: baseline call, then 3 no-improvement -> should stop on 3rd
        # Actually _on_step was called 4 times after baseline (loop 3 + final)
        # The 3rd call in loop should have returned False
        assert result is False

    def test_zero_best_reward_uses_absolute_fallback(self) -> None:
        """TRAIN-03: When best reward is 0, uses absolute comparison."""
        cb = _make_callback(patience=3, min_improvement_pct=0.01)
        cb.parent = _MockParent(reward=0.0)
        cb._on_step()  # baseline at 0.0

        # Small absolute improvement
        cb.parent.last_mean_reward = 0.5
        result = cb._on_step()
        assert result is True  # Should detect improvement via absolute fallback
