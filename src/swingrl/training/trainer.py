"""TrainingOrchestrator for PPO/A2C/SAC agent training.

Wraps Stable Baselines3 learn() with VecNormalize, callbacks, model saving,
and post-training smoke tests. Produces 6 trained models (3 algos x 2 envs)
saved to models/active/{env}/{algo}/.

Usage:
    from swingrl.training.trainer import TrainingOrchestrator
    orchestrator = TrainingOrchestrator(config=config)
    result = orchestrator.train("equity", "ppo", features, prices)
"""

from __future__ import annotations

import pickle  # nosec B403 -- required for SB3 VecNormalize serialization
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swingrl.config.schema import SwingRLConfig
from swingrl.envs.crypto import CryptoTradingEnv
from swingrl.envs.equity import StockTradingEnv
from swingrl.training.callbacks import ConvergenceCallback
from swingrl.utils.exceptions import ModelError

log = structlog.get_logger(__name__)

# Locked hyperparameters from CONTEXT.md -- never modify without design review
HYPERPARAMS: dict[str, dict[str, Any]] = {
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    "a2c": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    "sac": {
        "learning_rate": 0.0003,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
        "learning_starts": 10_000,
        "buffer_size": 1_000_000,
    },
}

SEED_MAP: dict[str, int] = {"ppo": 42, "a2c": 43, "sac": 44}

ALGO_MAP: dict[str, type[PPO] | type[A2C] | type[SAC]] = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
}

ENV_CLASS_MAP: dict[str, type[StockTradingEnv] | type[CryptoTradingEnv]] = {
    "equity": StockTradingEnv,
    "crypto": CryptoTradingEnv,
}


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_path: Path
    vec_normalize_path: Path
    env_name: str
    algo_name: str
    converged_at_step: int | None
    total_timesteps: int


class TrainingOrchestrator:
    """Orchestrates SB3 training with VecNormalize, callbacks, and smoke tests.

    Handles the full training lifecycle: environment wrapping, model
    instantiation with locked hyperparameters, callback setup, training,
    model + VecNormalize saving, and post-training smoke tests.

    Args:
        config: Validated SwingRLConfig instance.
        models_dir: Root directory for model storage (default: models/).
        logs_dir: Root directory for logs and tensorboard (default: logs/).
    """

    def __init__(
        self,
        config: SwingRLConfig,
        models_dir: Path = Path("models"),
        logs_dir: Path = Path("logs"),
    ) -> None:
        self._config = config
        self._models_dir = models_dir
        self._logs_dir = logs_dir

    def train(
        self,
        env_name: str,
        algo_name: str,
        features: np.ndarray,
        prices: np.ndarray,
        total_timesteps: int = 1_000_000,
    ) -> TrainingResult:
        """Train an SB3 algorithm on the specified environment.

        Creates DummyVecEnv + VecNormalize, instantiates the algorithm
        with locked hyperparameters, trains, saves model + VecNormalize,
        and runs 6 post-training smoke tests.

        Args:
            env_name: Environment type ("equity" or "crypto").
            algo_name: Algorithm name ("ppo", "a2c", or "sac").
            features: Feature array for the environment.
            prices: Price array for the environment.
            total_timesteps: Total training timesteps.

        Returns:
            TrainingResult with paths and metadata.

        Raises:
            ModelError: If smoke tests fail after training.
            KeyError: If algo_name or env_name is invalid.
        """
        log.info(
            "training_started",
            env_name=env_name,
            algo_name=algo_name,
            total_timesteps=total_timesteps,
        )

        # Create wrapped environment
        vec_env = self._create_env(env_name, features, prices)

        # Create eval environment (separate instance for EvalCallback)
        eval_vec_env = self._create_env(env_name, features, prices)

        # Instantiate algorithm with locked hyperparams
        algo_cls = ALGO_MAP[algo_name]
        params = HYPERPARAMS[algo_name].copy()
        seed = SEED_MAP[algo_name]

        tb_log = str(self._logs_dir / "tensorboard")

        model = algo_cls(
            policy="MlpPolicy",
            env=vec_env,
            seed=seed,
            tensorboard_log=tb_log,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
            **params,
        )

        # Setup callbacks
        convergence_cb = ConvergenceCallback(
            min_improvement_pct=0.01,
            patience=10,
            verbose=0,
        )

        eval_dir = self._logs_dir / "eval" / env_name / algo_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_cb = EvalCallback(
            eval_vec_env,
            callback_after_eval=convergence_cb,
            eval_freq=max(total_timesteps // 10, 1),
            n_eval_episodes=5,
            log_path=str(eval_dir),
            best_model_save_path=None,
            deterministic=True,
            verbose=0,
        )

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_cb,
        )

        # Save model and VecNormalize
        model_path, vec_path = self._save_model(model, vec_env, env_name, algo_name)

        # Run smoke tests
        self._run_smoke_tests(model, vec_env, env_name, algo_name)

        # Check if training converged early
        converged_at = None
        if convergence_cb._stagnation_count >= convergence_cb._patience:
            converged_at = model.num_timesteps

        log.info(
            "training_complete",
            env_name=env_name,
            algo_name=algo_name,
            model_path=str(model_path),
            converged_at=converged_at,
        )

        return TrainingResult(
            model_path=model_path,
            vec_normalize_path=vec_path,
            env_name=env_name,
            algo_name=algo_name,
            converged_at_step=converged_at,
            total_timesteps=total_timesteps,
        )

    def _create_env(
        self,
        env_name: str,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> VecNormalize:
        """Create DummyVecEnv wrapped in VecNormalize.

        Args:
            env_name: Environment type ("equity" or "crypto").
            features: Feature array for the environment.
            prices: Price array for the environment.

        Returns:
            VecNormalize-wrapped vectorized environment.
        """
        env_cls = ENV_CLASS_MAP[env_name]

        def _make_env() -> StockTradingEnv | CryptoTradingEnv:
            return env_cls(
                features=features,
                prices=prices,
                config=self._config,
            )

        dummy_env = DummyVecEnv([_make_env])
        vec_env: VecNormalize = VecNormalize(
            dummy_env,
            norm_obs=True,
            norm_reward=True,
        )
        return vec_env

    def _save_model(
        self,
        model: PPO | A2C | SAC,
        vec_env: VecNormalize,
        env_name: str,
        algo_name: str,
    ) -> tuple[Path, Path]:
        """Save trained model and VecNormalize statistics.

        Args:
            model: Trained SB3 model.
            vec_env: VecNormalize wrapper with computed statistics.
            env_name: Environment type.
            algo_name: Algorithm name.

        Returns:
            Tuple of (model_path, vec_normalize_path).
        """
        save_dir = self._models_dir / "active" / env_name / algo_name
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.zip"
        vec_path = save_dir / "vec_normalize.pkl"

        model.save(str(model_path))

        # Save VecNormalize stats (pickle required for SB3 VecNormalize objects)
        with vec_path.open("wb") as f:
            pickle.dump(vec_env, f)  # noqa: S301

        log.info(
            "model_saved",
            model_path=str(model_path),
            vec_normalize_path=str(vec_path),
        )

        return model_path, vec_path

    def _run_smoke_tests(
        self,
        model: PPO | A2C | SAC,
        vec_env: VecNormalize,
        env_name: str,
        algo_name: str,
    ) -> None:
        """Run 6 post-training smoke tests on a trained model.

        Checks:
        1. Model deserializes (save/load roundtrip)
        2. Output shape matches action space
        3. Non-degenerate actions (not all identical)
        4. Inference < 100ms per prediction
        5. VecNormalize loads successfully
        6. No NaN in model outputs

        Args:
            model: Trained SB3 model.
            vec_env: VecNormalize wrapper.
            env_name: Environment type.
            algo_name: Algorithm name.

        Raises:
            ModelError: If any smoke test fails.
        """
        algo_cls = ALGO_MAP[algo_name]
        save_dir = self._models_dir / "active" / env_name / algo_name
        model_path = save_dir / "model.zip"
        vec_path = save_dir / "vec_normalize.pkl"

        # 1. Model deserializes
        try:
            loaded_model = algo_cls.load(str(model_path))
        except Exception as exc:
            msg = f"Smoke test 1 FAILED: Model deserialization error for {algo_name}/{env_name}"
            log.error("smoke_test_failed", test=1, error=str(exc))
            raise ModelError(msg) from exc

        # 2. Output shape valid
        obs = vec_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        expected_shape = vec_env.action_space.shape
        if action.shape[1:] != expected_shape:
            msg = (
                f"Smoke test 2 FAILED: Action shape {action.shape[1:]} "
                f"!= expected {expected_shape} for {algo_name}/{env_name}"
            )
            log.error("smoke_test_failed", test=2, shape=action.shape, expected=expected_shape)
            raise ModelError(msg)

        # 3. Non-degenerate actions (10 different observations)
        actions = []
        for _ in range(10):
            a, _ = loaded_model.predict(obs, deterministic=False)
            actions.append(a.copy())
            step_result = vec_env.step(a)
            obs = step_result[0]  # type: ignore[assignment]
            done = step_result[2]
            if done[0]:
                obs = vec_env.reset()  # type: ignore[assignment]

        actions_arr = np.array(actions)
        if np.all(actions_arr == actions_arr[0]):
            msg = f"Smoke test 3 FAILED: All actions identical for {algo_name}/{env_name}"
            log.error("smoke_test_failed", test=3)
            raise ModelError(msg)

        # 4. Inference < 100ms
        obs = vec_env.reset()
        start = time.perf_counter()
        loaded_model.predict(obs, deterministic=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 100:
            msg = (
                f"Smoke test 4 FAILED: Inference took {elapsed_ms:.1f}ms "
                f"(> 100ms) for {algo_name}/{env_name}"
            )
            log.error("smoke_test_failed", test=4, elapsed_ms=elapsed_ms)
            raise ModelError(msg)

        # 5. VecNormalize loads successfully (pickle required for SB3 objects)
        try:
            with vec_path.open("rb") as f:
                pickle.load(f)  # noqa: S301  # nosec B301
        except Exception as exc:
            msg = f"Smoke test 5 FAILED: VecNormalize load error for {algo_name}/{env_name}"
            log.error("smoke_test_failed", test=5, error=str(exc))
            raise ModelError(msg) from exc

        # 6. No NaN in model outputs
        obs = vec_env.reset()
        final_action, _ = loaded_model.predict(obs, deterministic=True)
        if np.any(np.isnan(final_action)):
            msg = f"Smoke test 6 FAILED: NaN in model output for {algo_name}/{env_name}"
            log.error("smoke_test_failed", test=6)
            raise ModelError(msg)

        log.info(
            "smoke_tests_passed",
            env_name=env_name,
            algo_name=algo_name,
            inference_ms=round(elapsed_ms, 2),
        )
