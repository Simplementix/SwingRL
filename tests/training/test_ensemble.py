"""Tests for Sharpe-weighted softmax ensemble blending.

TRAIN-06: Ensemble weights via Sharpe-weighted softmax with adaptive validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from swingrl.training.ensemble import EnsembleBlender, sharpe_softmax_weights


class TestSharpeSoftmaxWeights:
    """Tests for sharpe_softmax_weights function."""

    def test_weights_sum_to_one(self) -> None:
        """Weights always sum to 1.0."""
        weights = sharpe_softmax_weights({"ppo": 1.0, "a2c": 0.5, "sac": 0.8})
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_higher_sharpe_higher_weight(self) -> None:
        """Agent with higher Sharpe gets higher weight."""
        weights = sharpe_softmax_weights({"ppo": 1.0, "a2c": 0.5, "sac": 0.8})
        assert weights["ppo"] > weights["sac"] > weights["a2c"]

    def test_equal_sharpes_equal_weights(self) -> None:
        """Equal Sharpes produce equal weights (1/3 each)."""
        weights = sharpe_softmax_weights({"ppo": 1.0, "a2c": 1.0, "sac": 1.0})
        for v in weights.values():
            assert abs(v - 1 / 3) < 1e-10

    def test_negative_sharpes_handled(self) -> None:
        """Negative Sharpe ratios don't cause errors."""
        weights = sharpe_softmax_weights({"ppo": -0.5, "a2c": -1.0, "sac": 0.2})
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        # SAC (highest) should have the most weight
        assert weights["sac"] > weights["ppo"] > weights["a2c"]

    def test_single_agent(self) -> None:
        """Single agent gets weight 1.0."""
        weights = sharpe_softmax_weights({"ppo": 1.5})
        assert abs(weights["ppo"] - 1.0) < 1e-10

    def test_large_difference(self) -> None:
        """Very large Sharpe differences still produce valid weights."""
        weights = sharpe_softmax_weights({"ppo": 10.0, "a2c": 0.1, "sac": 0.1})
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert weights["ppo"] > 0.9  # PPO dominates


class TestEnsembleBlender:
    """Tests for EnsembleBlender class."""

    @pytest.fixture()
    def blender(self, loaded_config) -> EnsembleBlender:  # type: ignore[no-untyped-def]
        """Create EnsembleBlender with test config."""
        return EnsembleBlender(loaded_config)

    def test_blend_actions_weighted_sum(self, blender: EnsembleBlender) -> None:
        """blend_actions produces correct weighted sum."""
        actions = {
            "ppo": np.array([1.0, 0.0]),
            "a2c": np.array([0.0, 1.0]),
            "sac": np.array([0.5, 0.5]),
        }
        weights = {"ppo": 0.5, "a2c": 0.3, "sac": 0.2}
        result = blender.blend_actions(actions, weights)
        expected = (
            0.5 * np.array([1.0, 0.0]) + 0.3 * np.array([0.0, 1.0]) + 0.2 * np.array([0.5, 0.5])
        )
        np.testing.assert_allclose(result, expected)

    def test_compute_weights_equity(self, blender: EnsembleBlender) -> None:
        """compute_weights for equity produces valid softmax weights."""
        weights = blender.compute_weights(
            env_name="equity",
            agent_sharpes={"ppo": 1.0, "a2c": 0.5, "sac": 0.8},
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert weights["ppo"] > weights["sac"] > weights["a2c"]

    def test_compute_weights_crypto(self, blender: EnsembleBlender) -> None:
        """compute_weights for crypto produces valid softmax weights."""
        weights = blender.compute_weights(
            env_name="crypto",
            agent_sharpes={"ppo": 0.8, "a2c": 1.2, "sac": 0.6},
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-10
        assert weights["a2c"] > weights["ppo"] > weights["sac"]

    def test_adaptive_window_shrink_high_turbulence(self, blender: EnsembleBlender) -> None:
        """During high turbulence, validation window shrinks by 50%."""
        # With turbulence above threshold, window should shrink
        # This should still produce valid weights
        weights = blender.compute_weights(
            env_name="equity",
            agent_sharpes={"ppo": 1.0, "a2c": 0.5, "sac": 0.8},
            turbulence=999.0,  # Very high turbulence
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_adaptive_window_normal_turbulence(self, blender: EnsembleBlender) -> None:
        """During normal turbulence, validation window stays full."""
        weights_normal = blender.compute_weights(
            env_name="equity",
            agent_sharpes={"ppo": 1.0, "a2c": 0.5, "sac": 0.8},
            turbulence=0.1,  # Very low turbulence
        )
        weights_none = blender.compute_weights(
            env_name="equity",
            agent_sharpes={"ppo": 1.0, "a2c": 0.5, "sac": 0.8},
        )
        # Should produce identical results when turbulence is low
        for algo in ["ppo", "a2c", "sac"]:
            assert abs(weights_normal[algo] - weights_none[algo]) < 1e-10
