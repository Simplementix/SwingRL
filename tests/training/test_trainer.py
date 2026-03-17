"""Tests for TrainingOrchestrator, HYPERPARAMS, and SEED_MAP.

Uses tiny feature arrays and minimal timesteps to test the training
pipeline without evaluating model quality. All tests must complete
within 60 seconds total.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.training.trainer import ALGO_MAP, HYPERPARAMS, SEED_MAP, TrainingOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer_config(tmp_path: Path) -> SwingRLConfig:
    """Config with 2 equity symbols and 2 crypto symbols for fast tests."""
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
    """Tiny equity features array: (60, 66) for fast training.

    Dimensions match equity_obs_dim(sentiment_enabled=False, n_equity_symbols=2):
    (15 * 2) + 6 + 2 + 1 + 27 = 66
    """
    from swingrl.features.assembler import equity_obs_dim

    rng = np.random.default_rng(100)
    obs_dim = equity_obs_dim(sentiment_enabled=False, n_equity_symbols=2)
    return rng.standard_normal((60, obs_dim)).astype(np.float32)


@pytest.fixture
def tiny_equity_prices() -> np.ndarray:
    """Tiny equity prices array: (60, 2) matching 2-symbol config."""
    rng = np.random.default_rng(101)
    base = np.array([470.0, 400.0], dtype=np.float32)
    returns = 1.0 + rng.normal(0.0002, 0.01, (60, 2))
    return (base * np.cumprod(returns, axis=0)).astype(np.float32)


@pytest.fixture
def tiny_crypto_features() -> np.ndarray:
    """Tiny crypto features array: (60, 45) for fast training."""
    rng = np.random.default_rng(102)
    return rng.standard_normal((60, 45)).astype(np.float32)


@pytest.fixture
def tiny_crypto_prices() -> np.ndarray:
    """Tiny crypto prices array: (60, 2) for fast training."""
    rng = np.random.default_rng(103)
    base = np.array([42_000.0, 2_500.0], dtype=np.float32)
    returns = 1.0 + rng.normal(0.0001, 0.02, (60, 2))
    return (base * np.cumprod(returns, axis=0)).astype(np.float32)


# ---------------------------------------------------------------------------
# HYPERPARAMS and SEED_MAP constant tests
# ---------------------------------------------------------------------------


class TestHyperparams:
    """TRAIN-04: Locked hyperparameter validation."""

    def test_ppo_hyperparams_correct(self) -> None:
        """TRAIN-04: PPO hyperparams match locked values from CONTEXT.md."""
        ppo = HYPERPARAMS["ppo"]
        assert ppo["learning_rate"] == 0.0003
        assert ppo["n_steps"] == 2048
        assert ppo["batch_size"] == 64
        assert ppo["n_epochs"] == 10
        assert ppo["gamma"] == 0.99
        assert ppo["gae_lambda"] == 0.95
        assert ppo["clip_range"] == 0.2
        assert ppo["ent_coef"] == 0.01
        assert ppo["vf_coef"] == 0.5

    def test_a2c_hyperparams_correct(self) -> None:
        """TRAIN-04: A2C hyperparams match locked values from CONTEXT.md."""
        a2c = HYPERPARAMS["a2c"]
        assert a2c["learning_rate"] == 0.0007
        assert a2c["n_steps"] == 5
        assert a2c["gamma"] == 0.99
        assert a2c["gae_lambda"] == 1.0
        assert a2c["ent_coef"] == 0.01
        assert a2c["vf_coef"] == 0.5

    def test_sac_hyperparams_correct(self) -> None:
        """TRAIN-04: SAC hyperparams match locked values with ent_coef=auto."""
        sac = HYPERPARAMS["sac"]
        assert sac["learning_rate"] == 0.0003
        assert sac["batch_size"] == 256
        assert sac["tau"] == 0.005
        assert sac["gamma"] == 0.99
        assert sac["ent_coef"] == "auto"
        assert sac["learning_starts"] == 10_000
        # buffer_size is injected at runtime from config.training.sac_buffer_size
        assert "buffer_size" not in sac

    def test_all_algos_present(self) -> None:
        """TRAIN-04: All three algorithms have entries."""
        assert set(HYPERPARAMS.keys()) == {"ppo", "a2c", "sac"}


class TestSeedMap:
    """TRAIN-04: Seed reproducibility constants."""

    def test_seeds_correct(self) -> None:
        """TRAIN-04: Seeds match locked values (42, 43, 44)."""
        assert SEED_MAP == {"ppo": 42, "a2c": 43, "sac": 44}


class TestAlgoMap:
    """TRAIN-04: Algorithm class mapping."""

    def test_algo_map_has_all_entries(self) -> None:
        """TRAIN-04: ALGO_MAP maps ppo, a2c, sac to SB3 classes."""
        assert set(ALGO_MAP.keys()) == {"ppo", "a2c", "sac"}


# ---------------------------------------------------------------------------
# TrainingOrchestrator pipeline tests
# ---------------------------------------------------------------------------


class TestTrainingOrchestratorEquity:
    """TRAIN-05: End-to-end training pipeline test for equity PPO."""

    def test_train_ppo_equity_produces_saved_model(
        self,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-05: PPO training on tiny equity data produces model.zip and vec_normalize.pkl."""
        orchestrator = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        result = orchestrator.train(
            env_name="equity",
            algo_name="ppo",
            features=tiny_equity_features,
            prices=tiny_equity_prices,
            total_timesteps=500,
        )

        assert result.model_path.exists()
        assert result.model_path.name == "model.zip"
        assert result.vec_normalize_path.exists()
        assert result.vec_normalize_path.name == "vec_normalize.pkl"
        assert result.env_name == "equity"
        assert result.algo_name == "ppo"

    def test_model_saved_to_correct_directory(
        self,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-05: Model saved to models/active/equity/ppo/ directory structure."""
        orchestrator = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        result = orchestrator.train(
            env_name="equity",
            algo_name="ppo",
            features=tiny_equity_features,
            prices=tiny_equity_prices,
            total_timesteps=500,
        )

        expected_dir = tmp_path / "models" / "active" / "equity" / "ppo"
        assert result.model_path.parent == expected_dir


class TestTrainingOrchestratorAllAlgos:
    """TRAIN-05: Each algo instantiates and trains without error."""

    @pytest.mark.parametrize("algo_name", ["ppo", "a2c", "sac"])
    def test_each_algo_trains_on_equity(
        self,
        algo_name: str,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-05: {algo_name} trains successfully on tiny equity data."""
        orchestrator = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        result = orchestrator.train(
            env_name="equity",
            algo_name=algo_name,
            features=tiny_equity_features,
            prices=tiny_equity_prices,
            total_timesteps=500,
        )

        assert result.model_path.exists()
        assert result.vec_normalize_path.exists()

    def test_crypto_env_trains(
        self,
        trainer_config: SwingRLConfig,
        tiny_crypto_features: np.ndarray,
        tiny_crypto_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-05: PPO trains successfully on tiny crypto data."""
        orchestrator = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        result = orchestrator.train(
            env_name="crypto",
            algo_name="ppo",
            features=tiny_crypto_features,
            prices=tiny_crypto_prices,
            total_timesteps=500,
        )

        assert result.model_path.exists()
        assert result.env_name == "crypto"


class TestPostTrainingSmokeTests:
    """TRAIN-05: Post-training smoke tests pass on freshly trained model."""

    def test_smoke_tests_pass(
        self,
        trainer_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """TRAIN-05: 6 smoke tests pass on freshly trained PPO model."""
        orchestrator = TrainingOrchestrator(
            config=trainer_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        # Should not raise -- smoke tests run internally during train()
        result = orchestrator.train(
            env_name="equity",
            algo_name="ppo",
            features=tiny_equity_features,
            prices=tiny_equity_prices,
            total_timesteps=500,
        )

        assert result.total_timesteps == 500
