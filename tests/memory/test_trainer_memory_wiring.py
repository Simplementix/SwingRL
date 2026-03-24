"""Integration tests for memory component wiring in TrainingOrchestrator.

TRAIN-07: MemoryVecRewardWrapper and MemoryEpochCallback are wired correctly
into trainer.train() when memory_client is provided. Fail-open: training
proceeds without memory when client is None or components fail to instantiate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecNormalize

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.memory.training.epoch_callback import MemoryEpochCallback
from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer_config(tmp_path: Path) -> SwingRLConfig:
    """Minimal config with 2 equity symbols for fast wiring tests."""
    yaml = tmp_path / "swingrl.yaml"
    yaml.write_text(
        """\
trading_mode: paper
equity:
  symbols: [SPY, QQQ]
  max_position_size: 0.50
  max_drawdown_pct: 0.10
  daily_loss_limit_pct: 0.02
crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50
  max_drawdown_pct: 0.12
  daily_loss_limit_pct: 0.03
  min_order_usd: 10.0
capital:
  equity_usd: 400.0
  crypto_usd: 47.0
paths:
  data_dir: data/
  db_dir: db/
  models_dir: models/
  logs_dir: logs/
logging:
  level: INFO
  json_logs: false
environment:
  initial_amount: 100000.0
  equity_episode_bars: 50
  crypto_episode_bars: 50
  equity_transaction_cost_pct: 0.0006
  crypto_transaction_cost_pct: 0.0022
  signal_deadzone: 0.02
  position_penalty_coeff: 10.0
  drawdown_penalty_coeff: 5.0
system:
  duckdb_path: data/db/market_data.ddb
  sqlite_path: data/db/trading_ops.db
alerting:
  alert_cooldown_minutes: 30
  consecutive_failures_before_alert: 3
"""
    )
    return load_config(yaml)


@pytest.fixture
def tiny_equity_features() -> np.ndarray:
    """Equity features array with correct obs_dim for 2-symbol config."""
    from swingrl.features.assembler import equity_obs_dim

    rng = np.random.default_rng(100)
    obs_dim = equity_obs_dim(sentiment_enabled=False, n_equity_symbols=2)
    return rng.standard_normal((60, obs_dim)).astype(np.float32)


@pytest.fixture
def tiny_equity_prices() -> np.ndarray:
    """Equity prices array: (60, 2) matching 2-symbol config."""
    rng = np.random.default_rng(101)
    base = np.array([470.0, 400.0], dtype=np.float32)
    returns = 1.0 + rng.normal(0.0002, 0.01, (60, 2))
    return (base * np.cumprod(returns, axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_memory_client() -> MagicMock:
    """Return a mock MemoryClient with ingest_training returning True."""
    client = MagicMock()
    client.base_url = "http://localhost:8889"
    client.api_key = "test-key"  # pragma: allowlist secret
    client.ingest_training.return_value = True
    return client


def _make_mock_venv(n_envs: int = 1) -> MagicMock:
    """Return a minimal mock VecEnv."""
    mock = MagicMock()
    mock.num_envs = n_envs
    mock.observation_space = MagicMock()
    mock.action_space = MagicMock()
    obs = np.zeros((n_envs, 4), dtype=np.float32)
    mock.step_wait.return_value = (
        obs,
        np.ones(n_envs),
        np.zeros(n_envs, dtype=bool),
        [{}] * n_envs,
    )
    mock.reset.return_value = obs
    return mock


# ---------------------------------------------------------------------------
# MemoryEpochCallback unit tests
# ---------------------------------------------------------------------------


class TestEpochCallbackInit:
    """TRAIN-07: MemoryEpochCallback initializes with correct state."""

    def test_initial_epoch_zero(self) -> None:
        """TRAIN-07: Epoch counter starts at 0."""
        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=_make_mock_memory_client(),
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        assert cb._epoch == 0  # noqa: SLF001

    def test_advice_failed_once_starts_false(self) -> None:
        """TRAIN-07: _advice_failed_once starts False (first failure logs at info)."""
        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=_make_mock_memory_client(),
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        assert cb._advice_failed_once is False  # noqa: SLF001

    def test_on_step_returns_true_when_not_stopped(self) -> None:
        """TRAIN-07: _on_step() returns True when stop_training is not set."""
        from unittest.mock import MagicMock

        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=_make_mock_memory_client(),
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        # Simulate model being attached (as SB3 does during learn())
        cb.model = MagicMock()
        cb.model.stop_training = False
        assert cb._on_step() is True  # noqa: SLF001

    def test_on_step_returns_false_when_stop_training(self) -> None:
        """TRAIN-07: _on_step() returns False when stop_training is set."""
        from unittest.mock import MagicMock

        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=_make_mock_memory_client(),
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        cb.model = MagicMock()
        cb.model.stop_training = True
        assert cb._on_step() is False  # noqa: SLF001


class TestEpochAdviceFirstFailureLogging:
    """TRAIN-07: First epoch advice failure logs at info; subsequent at debug."""

    def test_first_failure_sets_flag(self) -> None:
        """TRAIN-07: First connection failure transitions _advice_failed_once to True."""
        client = _make_mock_memory_client()
        client.epoch_advice.side_effect = ConnectionError("unreachable")
        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=client,
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        cb._epoch = 20  # noqa: SLF001 — set to PPO cadence epoch (20)

        cb._query_epoch_advice()  # noqa: SLF001

        assert cb._advice_failed_once is True  # noqa: SLF001

    def test_non_cadence_epoch_skips_query(self) -> None:
        """TRAIN-07: Epochs not divisible by EPOCH_STORE_CADENCE skip the advice query."""
        from swingrl.memory.training.epoch_callback import EPOCH_STORE_CADENCE

        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        cb = MemoryEpochCallback(
            memory_client=_make_mock_memory_client(),
            wrapper=wrapper,
            run_id="test_run",
            algo="PPO",
            env="equity",
        )
        cb._epoch = EPOCH_STORE_CADENCE + 1  # noqa: SLF001 — not a cadence epoch

        cb._query_epoch_advice()  # noqa: SLF001
        # Flag stays False because query was skipped entirely
        assert cb._advice_failed_once is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# VecEnvWrapper unwrapping logic
# ---------------------------------------------------------------------------


class TestVecEnvUnwrappingLogic:
    """TRAIN-07: VecEnvWrapper unwrapping loop stops at VecNormalize."""

    def test_unwrap_stops_at_vec_normalize(self) -> None:
        """TRAIN-07: Loop resolves MemoryVecRewardWrapper to inner VecNormalize."""
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

        base = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        vec_norm = VecNormalize(base, norm_obs=True, norm_reward=True)
        wrapped = MemoryVecRewardWrapper(vec_norm)

        # Replicate the unwrapping logic from trainer._save_model
        vec_to_save = wrapped
        while isinstance(vec_to_save, VecEnvWrapper) and not isinstance(vec_to_save, VecNormalize):
            vec_to_save = vec_to_save.venv  # type: ignore[assignment]

        assert isinstance(vec_to_save, VecNormalize)
        assert not isinstance(vec_to_save, MemoryVecRewardWrapper)
        assert hasattr(vec_to_save, "obs_rms")
        assert hasattr(vec_to_save, "ret_rms")
        base.close()

    def test_plain_vec_normalize_passthrough(self) -> None:
        """TRAIN-07: Without wrapper, unwrapping loop returns VecNormalize unchanged."""
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

        base = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        vec_norm = VecNormalize(base, norm_obs=True, norm_reward=True)

        vec_to_save = vec_norm
        while isinstance(vec_to_save, VecEnvWrapper) and not isinstance(vec_to_save, VecNormalize):
            vec_to_save = vec_to_save.venv  # type: ignore[assignment]

        assert isinstance(vec_to_save, VecNormalize)
        base.close()


# ---------------------------------------------------------------------------
# Trainer fail-open wiring (no-client path)
# ---------------------------------------------------------------------------


class TestTrainerMemoryFailOpen:
    """TRAIN-07: trainer.train() is fail-open when memory components are absent."""

    def test_no_memory_client_does_not_import_wrapper(
        self,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-07: With memory_client=None, reward wrapper is never instantiated."""
        from swingrl.training.trainer import TrainingOrchestrator

        orch = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )

        with patch(
            "swingrl.memory.training.reward_wrapper.MemoryVecRewardWrapper",
            wraps=MemoryVecRewardWrapper,
        ) as mock_wrapper:
            orch.train(
                env_name="equity",
                algo_name="ppo",
                features=tiny_equity_features,
                prices=tiny_equity_prices,
                total_timesteps=500,
                memory_client=None,
            )
            mock_wrapper.assert_not_called()

    def test_wrapper_failure_does_not_block_training(
        self,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-07: If MemoryVecRewardWrapper raises on init, training still completes."""
        from swingrl.training.trainer import TrainingOrchestrator

        client = _make_mock_memory_client()
        orch = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )

        with patch(
            "swingrl.memory.training.reward_wrapper.MemoryVecRewardWrapper",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = orch.train(
                env_name="equity",
                algo_name="ppo",
                features=tiny_equity_features,
                prices=tiny_equity_prices,
                total_timesteps=500,
                memory_client=client,
            )
        assert result.model_path.exists()
