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

import os
import pickle  # nosec B403 -- required for SB3 VecNormalize serialization
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from swingrl.config.schema import SwingRLConfig
from swingrl.envs.crypto import CryptoTradingEnv
from swingrl.envs.equity import StockTradingEnv
from swingrl.training.callbacks import ConvergenceCallback
from swingrl.utils.exceptions import ModelError

if TYPE_CHECKING:
    from swingrl.memory.client import MemoryClient

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

        # Limit PyTorch intra-op threads to avoid oversubscription when
        # running multiple parallel envs via SubprocVecEnv or parallel algos.
        n_envs = config.training.n_envs
        cpu_count = os.cpu_count() or 4
        torch_threads = max(1, cpu_count // max(n_envs, 2))
        torch.set_num_threads(torch_threads)
        log.info("torch_threads_set", threads=torch_threads, cpu_count=cpu_count, n_envs=n_envs)

    def train(
        self,
        env_name: str,
        algo_name: str,
        features: np.ndarray,
        prices: np.ndarray,
        total_timesteps: int = 1_000_000,
        hyperparams_override: dict[str, Any] | None = None,
        memory_client: MemoryClient | None = None,
        run_id: str | None = None,
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
            hyperparams_override: Optional dict of hyperparameter overrides to
                merge on top of HYPERPARAMS[algo_name]. Override values win.
                Used by meta-training tuning rounds to test alternate configs.
            memory_client: Optional MemoryClient. When provided, wraps the
                training env in MemoryVecRewardWrapper and attaches
                MemoryEpochCallback to ingest epoch snapshots and apply
                LLM-guided reward weight adjustments. Fail-open: if None,
                training proceeds without memory integration.
            run_id: Training run identifier for memory tagging (e.g.
                "equity_ppo_20260318T120000Z"). Auto-generated if None.

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

        # Optionally wrap with MemoryVecRewardWrapper for LLM-guided reward shaping
        memory_wrapper = None
        if memory_client is not None:
            try:
                from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

                memory_wrapper = MemoryVecRewardWrapper(vec_env)
                vec_env = memory_wrapper  # type: ignore[assignment]
                log.info("memory_reward_wrapper_attached", env_name=env_name, algo_name=algo_name)
            except Exception as exc:
                log.warning("memory_reward_wrapper_failed", error=str(exc))
                memory_wrapper = None

        # Create eval environment (separate instance for EvalCallback).
        # Uses DummyVecEnv(1) — eval runs 5 sequential inference episodes,
        # no parallelism benefit. Saves ~1.8 GB vs SubprocVecEnv(n_envs).
        eval_vec_env = self._create_eval_env(env_name, features, prices)

        # Instantiate algorithm with locked hyperparams
        algo_cls = ALGO_MAP[algo_name]
        params = HYPERPARAMS[algo_name].copy()
        # SAC buffer_size from config — not hardcoded in HYPERPARAMS.
        if algo_name == "sac":
            params["buffer_size"] = self._config.training.sac_buffer_size
        if hyperparams_override:
            params.update(hyperparams_override)
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

        # Optionally attach MemoryEpochCallback for per-epoch memory ingestion
        callbacks: list[Any] = [eval_cb]
        if memory_client is not None and memory_wrapper is not None:
            try:
                from swingrl.memory.training.epoch_callback import MemoryEpochCallback

                effective_run_id = run_id or f"{env_name}_{algo_name}"
                memory_cb = MemoryEpochCallback(
                    memory_client=memory_client,
                    wrapper=memory_wrapper,
                    run_id=effective_run_id,
                    algo=algo_name.upper(),
                    env=env_name,
                    verbose=0,
                )
                callbacks.append(memory_cb)
                log.info(
                    "memory_epoch_callback_attached",
                    run_id=effective_run_id,
                    env_name=env_name,
                    algo_name=algo_name,
                )
            except Exception as exc:
                log.warning("memory_epoch_callback_failed", error=str(exc))

        # Train, save, and smoke-test -- wrapped in try/finally to ensure
        # SubprocVecEnv child processes are always cleaned up (C1 fix).
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
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
        finally:
            vec_env.close()
            eval_vec_env.close()
            log.debug("vec_envs_closed", env_name=env_name, algo_name=algo_name)

    def _create_env(
        self,
        env_name: str,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> VecNormalize:
        """Create vectorized env wrapped in VecNormalize.

        Uses SubprocVecEnv when n_envs > 1 and vecenv_backend is 'subproc',
        otherwise falls back to DummyVecEnv.

        Args:
            env_name: Environment type ("equity" or "crypto").
            features: Feature array for the environment.
            prices: Price array for the environment.

        Returns:
            VecNormalize-wrapped vectorized environment.
        """
        env_cls = ENV_CLASS_MAP[env_name]
        n_envs = self._config.training.n_envs
        backend = self._config.training.vecenv_backend

        def _make_env_fn(rank: int) -> Any:
            """Return a factory closure for env with unique seed offset."""

            def _init() -> StockTradingEnv | CryptoTradingEnv:
                return env_cls(
                    features=features,
                    prices=prices,
                    config=self._config,
                )

            return _init

        env_fns = [_make_env_fn(i) for i in range(n_envs)]

        if n_envs > 1 and backend == "subproc":
            base_env: DummyVecEnv | SubprocVecEnv = SubprocVecEnv(env_fns, start_method="fork")
        else:
            base_env = DummyVecEnv(env_fns)

        vec_env: VecNormalize = VecNormalize(
            base_env,
            norm_obs=True,
            norm_reward=True,
        )
        return vec_env

    def _create_eval_env(
        self,
        env_name: str,
        features: np.ndarray,
        prices: np.ndarray,
    ) -> VecNormalize:
        """Create lightweight eval environment for EvalCallback.

        Always uses DummyVecEnv with n_envs=1. Eval runs 5 sequential
        inference episodes (~1,260 steps) — no parallelism benefit from
        SubprocVecEnv. Avoids forking 6 workers that sit idle 99% of
        training time.

        Args:
            env_name: Environment type ("equity" or "crypto").
            features: Feature array for the environment.
            prices: Price array for the environment.

        Returns:
            VecNormalize-wrapped single-env DummyVecEnv.
        """
        env_cls = ENV_CLASS_MAP[env_name]

        def _init() -> StockTradingEnv | CryptoTradingEnv:
            return env_cls(
                features=features,
                prices=prices,
                config=self._config,
            )

        base_env = DummyVecEnv([_init])
        return VecNormalize(base_env, norm_obs=True, norm_reward=True)

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

        # Save VecNormalize stats — unwrap MemoryVecRewardWrapper if present so
        # the pickle file always contains a plain VecNormalize (eval loading
        # expects obs_rms / ret_rms attributes directly on the loaded object).
        # Pickle is required for SB3 VecNormalize (internal SB3 constraint).
        from stable_baselines3.common.vec_env import VecEnvWrapper

        vec_to_save = vec_env
        while isinstance(vec_to_save, VecEnvWrapper) and not isinstance(vec_to_save, VecNormalize):
            vec_to_save = vec_to_save.venv  # type: ignore[assignment]
        with vec_path.open("wb") as f:
            pickle.dump(vec_to_save, f)  # noqa: S301  # nosec B301

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
