"""Tests for memory training bounds clamping functions.

TRAIN-01, TRAIN-02: Bounds enforcement for LLM hyperparameter suggestions.
"""

from __future__ import annotations

import pytest


class TestClampRunConfig:
    """Tests for clamp_run_config() function."""

    def test_clamp_learning_rate_below_min(self) -> None:
        """TRAIN-01: learning_rate below 1e-5 is clamped to 1e-5."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"learning_rate": 1e-7})
        assert result["learning_rate"] == pytest.approx(1e-5)

    def test_clamp_learning_rate_above_max(self) -> None:
        """TRAIN-01: learning_rate above 1e-3 is clamped to 1e-3."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"learning_rate": 0.1})
        assert result["learning_rate"] == pytest.approx(1e-3)

    def test_learning_rate_in_range_unchanged(self) -> None:
        """TRAIN-01: learning_rate within [1e-5, 1e-3] is unchanged."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"learning_rate": 3e-4})
        assert result["learning_rate"] == pytest.approx(3e-4)

    def test_batch_size_forced_to_power_of_two(self) -> None:
        """TRAIN-01: batch_size is forced to nearest power of 2."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"batch_size": 100})
        # nearest power of 2 to 100 is 128
        assert result["batch_size"] == 128

    def test_batch_size_already_power_of_two_unchanged(self) -> None:
        """TRAIN-01: batch_size that is already a power of 2 stays the same."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"batch_size": 64})
        assert result["batch_size"] == 64

    def test_does_not_mutate_input(self) -> None:
        """TRAIN-01: clamp_run_config does not modify the input dict."""
        from swingrl.memory.training.bounds import clamp_run_config

        original = {"learning_rate": 1e-7, "batch_size": 100}
        clamp_run_config(original)
        assert original["learning_rate"] == 1e-7
        assert original["batch_size"] == 100

    def test_missing_keys_ignored(self) -> None:
        """TRAIN-01: Keys not in HYPERPARAM_BOUNDS are passed through unchanged."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"some_custom_key": 42})
        assert result["some_custom_key"] == 42

    def test_empty_dict_returns_empty(self) -> None:
        """TRAIN-01: Empty input returns empty dict."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({})
        assert result == {}

    def test_clip_range_clamped(self) -> None:
        """TRAIN-01: clip_range outside [0.1, 0.4] is clamped."""
        from swingrl.memory.training.bounds import clamp_run_config

        result_low = clamp_run_config({"clip_range": 0.0})
        result_high = clamp_run_config({"clip_range": 0.9})
        assert result_low["clip_range"] == pytest.approx(0.1)
        assert result_high["clip_range"] == pytest.approx(0.4)

    def test_n_epochs_clamped(self) -> None:
        """TRAIN-01: n_epochs outside [3, 20] is clamped."""
        from swingrl.memory.training.bounds import clamp_run_config

        result = clamp_run_config({"n_epochs": 1})
        assert result["n_epochs"] == 3

        result2 = clamp_run_config({"n_epochs": 50})
        assert result2["n_epochs"] == 20


class TestClampRewardWeights:
    """Tests for clamp_reward_weights() function."""

    def test_clamped_weights_normalized_to_one(self) -> None:
        """TRAIN-02: Clamped weights sum to 1.0."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        result = clamp_reward_weights(
            {"profit": 0.4, "sharpe": 0.3, "drawdown": 0.2, "turnover": 0.1}
        )
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_individual_weight_clamped_to_bound(self) -> None:
        """TRAIN-02: Individual weights outside REWARD_BOUNDS are clamped before normalizing."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        # profit max is 0.70, sharpe min is 0.10, drawdown min is 0.05
        # All inputs violate bounds
        result = clamp_reward_weights(
            {"profit": 0.9, "sharpe": 0.05, "drawdown": 0.03, "turnover": 0.02}
        )
        # After clamping+normalizing, result must still sum to 1.0
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        # All keys present
        assert set(result.keys()) == {"profit", "sharpe", "drawdown", "turnover"}

    def test_safe_defaults_when_total_zero(self) -> None:
        """TRAIN-02: Returns safe defaults when all weights clamp to 0."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        # turnover min is 0.0 — all other min values > 0, but we test a scenario
        # where weights would sum to zero by passing all-zero with only turnover key
        # that has lower bound of 0.0
        result = clamp_reward_weights(
            {"profit": 0.0, "sharpe": 0.0, "drawdown": 0.0, "turnover": 0.0}
        )
        # should return safe defaults, not div-by-zero
        assert isinstance(result, dict)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_does_not_mutate_input(self) -> None:
        """TRAIN-02: clamp_reward_weights does not modify the input dict."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        original = {"profit": 0.9, "sharpe": 0.05, "drawdown": 0.03, "turnover": 0.02}
        clamp_reward_weights(original)
        assert original["profit"] == 0.9

    def test_valid_weights_renormalized(self) -> None:
        """TRAIN-02: Valid weights still get renormalized."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        result = clamp_reward_weights(
            {"profit": 0.5, "sharpe": 0.5, "drawdown": 0.5, "turnover": 0.1}
        )
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_missing_keys_get_midpoint_default(self) -> None:
        """TRAIN-02: Missing reward keys get midpoint of their bounds."""
        from swingrl.memory.training.bounds import clamp_reward_weights

        # Only pass profit, rest should get defaults from midpoint
        result = clamp_reward_weights({"profit": 0.4})
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        # all four keys should be present
        assert set(result.keys()) == {"profit", "sharpe", "drawdown", "turnover"}


class TestMemoryClientIngest:
    """Tests for MemoryClient fail-open behavior."""

    def test_ingest_returns_false_on_connection_error(self) -> None:
        """TRAIN-01: MemoryClient.ingest() returns False on connection error (fail-open)."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://localhost:99999")
        result = client.ingest({"text": "test", "source": "test:historical"})
        assert result is False

    def test_ingest_training_accepts_any_source(self) -> None:
        """TRAIN-01: ingest_training() accepts any source tag (fail-open on connection error)."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://localhost:99999")
        # Walk-forward source tags should not raise (assertion removed)
        result = client.ingest_training(text="some text", source="walk_forward:equity:ppo")
        assert result is False

    def test_ingest_training_accepts_legacy_historical_source(self) -> None:
        """TRAIN-01: ingest_training() still accepts legacy :historical source tags."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://localhost:99999")
        result = client.ingest_training(text="some text", source="equity:historical")
        assert result is False


class TestMemoryAgentConfigDefaults:
    """Tests for MemoryAgentConfig default values."""

    def test_memory_agent_config_defaults_disabled(self) -> None:
        """TRAIN-01: MemoryAgentConfig defaults have enabled=False and meta_training=False."""
        from swingrl.config.schema import MemoryAgentConfig

        cfg = MemoryAgentConfig()
        assert cfg.enabled is False
        assert cfg.meta_training is False

    def test_memory_agent_live_endpoints_all_false(self) -> None:
        """TRAIN-01: MemoryLiveEndpointsConfig defaults have all endpoints disabled."""
        from swingrl.config.schema import MemoryLiveEndpointsConfig

        cfg = MemoryLiveEndpointsConfig()
        assert cfg.obs_enrichment is False
        assert cfg.blend_weights is False
        assert cfg.position_advice is False
        assert cfg.trade_veto is False
        assert cfg.cycle_gate is False
        assert cfg.risk_thresholds is False

    def test_swingrlconfig_loads_with_memory_agent(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-01: SwingRLConfig loads without error from existing config with memory_agent section."""
        from swingrl.config.schema import load_config

        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.memory_agent is not None
        assert cfg.memory_agent.enabled is False

    def test_memory_agent_cloud_smart_model(self) -> None:
        """TRAIN-01: MemoryAgentConfig default cloud_smart_model is nemotron-120b."""
        from swingrl.config.schema import MemoryAgentConfig

        cfg = MemoryAgentConfig()
        assert cfg.cloud_smart_model == "nvidia/nemotron-3-super-120b-a12b:free"
