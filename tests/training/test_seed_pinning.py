"""Tests for per-fold seed pinning (D-T2.5, M9 — Plan B Task 7).

D-T2.5's same-fold season-over-season comparison needs the fold's
advice/no-advice lever to be the only variable. This requires the model,
the training VecEnv, AND the eval VecEnv (used internally by EvalCallback's
early-stop) to all be pinned to deterministic, per-fold seeds.

Uses tiny feature arrays and minimal timesteps -- mirrors the fixture
pattern in test_trainer.py. n_envs=1 / vecenv_backend=dummy keeps the
wiring/determinism tests fast and avoids SubprocVecEnv fork overhead.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.training.trainer import SEED_MAP, TrainingOrchestrator, fold_seed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seed_pinning_config(tmp_path: Path) -> SwingRLConfig:
    """Config with 2 equity symbols, n_envs=1/dummy backend for fast+deterministic runs."""
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
  database_url: ""
alerting:
  alert_cooldown_minutes: 30
  consecutive_failures_before_alert: 3
training:
  n_envs: 1
  vecenv_backend: dummy
"""
    )
    return load_config(yaml)


@pytest.fixture
def tiny_equity_features() -> np.ndarray:
    """Tiny equity features array: (60, obs_dim) for fast training.

    Dimensions match equity_obs_dim(sentiment_enabled=False, n_equity_symbols=2):
    (15 * 2) + 6 + 2 + 1 + 35 = 74
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


# ---------------------------------------------------------------------------
# (a) fold_seed() contract -- pure function, no training required
# ---------------------------------------------------------------------------


class TestFoldSeed:
    """N9: fold_seed(algo, fold_number) = SEED_MAP[algo] * 1000 + fold_number."""

    def test_fold_seed_formula(self) -> None:
        """N9: fold_seed('ppo', 7) == 42007."""
        assert fold_seed("ppo", 7) == 42007

    def test_fold_seed_a2c(self) -> None:
        """N9: fold_seed('a2c', 3) == 43003."""
        assert fold_seed("a2c", 3) == 43003

    def test_fold_seed_sac(self) -> None:
        """N9: fold_seed('sac', 5) == 44005."""
        assert fold_seed("sac", 5) == 44005

    def test_fold_seed_zero(self) -> None:
        """N9: fold 0 uses the bare per-algo base seed."""
        assert fold_seed("ppo", 0) == 42000

    def test_algo_ranges_disjoint(self) -> None:
        """N9: seeds never collide across algos for any fold in a wide range."""
        ppo_seeds = {fold_seed("ppo", f) for f in range(0, 1000)}
        a2c_seeds = {fold_seed("a2c", f) for f in range(0, 1000)}
        sac_seeds = {fold_seed("sac", f) for f in range(0, 1000)}

        assert ppo_seeds.isdisjoint(a2c_seeds)
        assert ppo_seeds.isdisjoint(sac_seeds)
        assert a2c_seeds.isdisjoint(sac_seeds)

    def test_different_folds_produce_different_seeds(self) -> None:
        """N9: different fold_number -> different seed, for the same algo."""
        assert fold_seed("ppo", 0) != fold_seed("ppo", 1)
        assert fold_seed("a2c", 4) != fold_seed("a2c", 5)
        assert fold_seed("sac", 10) != fold_seed("sac", 11)

    def test_unknown_algo_raises_key_error(self) -> None:
        """N9: unknown algo name raises KeyError (mirrors SEED_MAP[algo] lookup)."""
        with pytest.raises(KeyError):
            fold_seed("unknown_algo", 1)


# ---------------------------------------------------------------------------
# (b) Seed wiring -- model, train env, eval env all receive the right seeds
# ---------------------------------------------------------------------------


class TestSeedWiring:
    """N9: Trainer.train(fold_number=...) threads fold_seed to model + both envs."""

    def test_fold_number_pins_model_train_env_and_eval_env(
        self,
        seed_pinning_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """N9: model seed, train-env seed, and eval-env seed all use fold_seed
        (train + model share fold_seed; eval uses fold_seed + 1, a distinct stream).
        """
        orchestrator = TrainingOrchestrator(
            config=seed_pinning_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )

        expected_seed = fold_seed("ppo", 3)
        expected_eval_seed = expected_seed + 1

        seed_calls: list[int | None] = []
        original_seed = DummyVecEnv.seed

        def _recording_seed(self: DummyVecEnv, seed: int | None = None) -> list[int | None]:
            seed_calls.append(seed)
            return original_seed(self, seed)

        with patch.object(DummyVecEnv, "seed", _recording_seed):
            result = orchestrator.train(
                env_name="equity",
                algo_name="ppo",
                features=tiny_equity_features,
                prices=tiny_equity_prices,
                total_timesteps=2_000,
                fold_number=3,
            )

        # Model's own seed is recorded on TrainingResult.
        assert result.seed == expected_seed

        # Both the train env and eval env base DummyVecEnv received .seed() calls;
        # the train env is seeded with fold_seed (both explicitly by us AND
        # implicitly by SB3's internal set_random_seed -- same value either way),
        # the eval env is seeded with the distinct fold_seed + 1 stream.
        assert expected_seed in seed_calls
        assert expected_eval_seed in seed_calls

    def test_fold_number_none_keeps_todays_constants(
        self,
        seed_pinning_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """N9: fold_number=None (ad-hoc runs) keeps the per-algo SEED_MAP constant."""
        orchestrator = TrainingOrchestrator(
            config=seed_pinning_config,
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )

        result = orchestrator.train(
            env_name="equity",
            algo_name="ppo",
            features=tiny_equity_features,
            prices=tiny_equity_prices,
            total_timesteps=2_000,
            fold_number=None,
        )

        assert result.seed == SEED_MAP["ppo"]


# ---------------------------------------------------------------------------
# (c) Determinism -- P-B3 verification. Marked slow: two full PPO runs.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFoldSeedDeterminism:
    """P-B3: same fold_number -> bitwise-reproducible training outcome."""

    def test_same_fold_number_reproduces_training(
        self,
        seed_pinning_config: SwingRLConfig,
        tiny_equity_features: np.ndarray,
        tiny_equity_prices: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """P-B3: two 2,000-step PPO runs with the same fold_number produce
        identical converged_at_step (convergence-callback stop point) AND
        bitwise-identical predict() outputs on a fixed observation batch.
        """
        from stable_baselines3 import PPO

        results = []
        for run_idx in range(2):
            orchestrator = TrainingOrchestrator(
                config=seed_pinning_config,
                models_dir=tmp_path / f"models_run{run_idx}",
                logs_dir=tmp_path / f"logs_run{run_idx}",
            )
            result = orchestrator.train(
                env_name="equity",
                algo_name="ppo",
                features=tiny_equity_features,
                prices=tiny_equity_prices,
                total_timesteps=2_000,
                fold_number=3,
            )
            results.append(result)

        result_a, result_b = results

        # Same seed threaded through both runs.
        assert result_a.seed == result_b.seed == fold_seed("ppo", 3)

        # Identical convergence-callback stop point (None if not converged --
        # still a meaningful equality: both runs must agree on whether/when
        # early-stop fired).
        assert result_a.converged_at_step == result_b.converged_at_step

        # Bitwise-identical predict() outputs on a fixed observation batch,
        # independent of any env randomness.
        model_a = PPO.load(str(result_a.model_path))
        model_b = PPO.load(str(result_b.model_path))

        # StockTradingEnv obs includes portfolio state on top of raw features --
        # use the actual saved VecNormalize's observation_space shape to build a
        # matching fixed obs batch instead of guessing the exact dimension.
        # Pickle load is safe here: the file is produced by this same test run
        # (trainer._save_model, tmp_path-local), not an untrusted source --
        # required for SB3 VecNormalize serialization (see trainer.py:511).
        import pickle

        with result_a.vec_normalize_path.open("rb") as f:
            vec_norm_a = pickle.load(f)  # noqa: S301  # nosec B301
        fixed_obs = np.zeros((1, *vec_norm_a.observation_space.shape), dtype=np.float32)

        action_a, _ = model_a.predict(fixed_obs, deterministic=True)
        action_b, _ = model_b.predict(fixed_obs, deterministic=True)

        np.testing.assert_array_equal(action_a, action_b)
