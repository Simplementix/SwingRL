"""Model lifecycle state machine and 6-point smoke test.

Manages model transitions: Training -> Shadow -> Active -> Archive -> Deletion.
Provides smoke_test_model() for validating deployed models before promotion.
"""

from __future__ import annotations

import enum
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from swingrl.utils.exceptions import ModelError

log = structlog.get_logger(__name__)

_SMOKE_TEST_ITERATIONS = 10
_INFERENCE_TIME_LIMIT_MS = 100.0


class ModelState(enum.Enum):
    """Model lifecycle states."""

    TRAINING = "training"
    SHADOW = "shadow"
    ACTIVE = "active"
    ARCHIVE = "archive"
    DELETED = "deleted"


class ModelLifecycle:
    """Manages model file transitions between lifecycle directories.

    Directory structure:
        models_dir/
            shadow/{env_name}/   - Candidate models under evaluation
            active/{env_name}/   - Currently serving model
            archive/{env_name}/  - Previously active models

    Args:
        models_dir: Base directory for model storage.
    """

    def __init__(self, models_dir: Path) -> None:
        """Initialize lifecycle manager and create subdirectories."""
        self._models_dir = models_dir
        for subdir in ("active", "shadow", "archive"):
            (models_dir / subdir).mkdir(parents=True, exist_ok=True)

    def deploy_to_shadow(self, model_path: Path, env_name: str) -> Path:
        """Copy a trained model to the shadow directory.

        Args:
            model_path: Path to the model file to deploy.
            env_name: Environment name (equity or crypto).

        Returns:
            Destination path in shadow directory.

        Raises:
            ModelError: If source model file does not exist.
        """
        if not model_path.exists():
            log.error("model_not_found", path=str(model_path))
            raise ModelError(f"Model file does not exist: {model_path}")

        dest_dir = self._models_dir / "shadow" / env_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / model_path.name
        shutil.copy2(model_path, dest)
        log.info("model_deployed_to_shadow", model=model_path.name, env=env_name)
        return dest

    def promote(self, env_name: str) -> Path:
        """Promote shadow model to active, archiving any current active model.

        Args:
            env_name: Environment name (equity or crypto).

        Returns:
            Path to the newly active model.

        Raises:
            ModelError: If no shadow model exists for the environment.
        """
        shadow_dir = self._models_dir / "shadow" / env_name
        shadow_models = list(shadow_dir.glob("*.zip")) if shadow_dir.exists() else []

        if not shadow_models:
            log.error("no_shadow_model", env=env_name)
            raise ModelError(f"No shadow model found for {env_name}")

        shadow_model = shadow_models[0]

        # Archive current active model if one exists
        active_dir = self._models_dir / "active" / env_name
        if active_dir.exists():
            active_models = list(active_dir.glob("*.zip"))
            if active_models:
                self._archive_model(active_models[0], env_name)

        # Move shadow to active
        active_dir.mkdir(parents=True, exist_ok=True)
        dest = active_dir / shadow_model.name
        shutil.move(str(shadow_model), str(dest))
        log.info("model_promoted", model=shadow_model.name, env=env_name)
        return dest

    def archive(self, env_name: str) -> Path:
        """Move the active model to archive.

        Args:
            env_name: Environment name (equity or crypto).

        Returns:
            Path to the archived model.

        Raises:
            ModelError: If no active model exists for the environment.
        """
        active_dir = self._models_dir / "active" / env_name
        active_models = list(active_dir.glob("*.zip")) if active_dir.exists() else []

        if not active_models:
            log.error("no_active_model", env=env_name)
            raise ModelError(f"No active model found for {env_name}")

        return self._archive_model(active_models[0], env_name)

    def archive_shadow(self, env_name: str) -> Path:
        """Move the shadow model to archive (failed evaluation).

        Args:
            env_name: Environment name (equity or crypto).

        Returns:
            Path to the archived shadow model.

        Raises:
            ModelError: If no shadow model exists for the environment.
        """
        shadow_dir = self._models_dir / "shadow" / env_name
        shadow_models = list(shadow_dir.glob("*.zip")) if shadow_dir.exists() else []

        if not shadow_models:
            log.error("no_shadow_model_to_archive", env=env_name)
            raise ModelError(f"No shadow model found for {env_name}")

        return self._archive_model(shadow_models[0], env_name)

    def rollback(self, env_name: str) -> Path:
        """Restore the most recent archived model to active.

        Args:
            env_name: Environment name (equity or crypto).

        Returns:
            Path to the restored active model.

        Raises:
            ModelError: If no archived model exists for the environment.
        """
        archive_dir = self._models_dir / "archive" / env_name
        archived = sorted(archive_dir.glob("*.zip")) if archive_dir.exists() else []

        if not archived:
            log.error("no_archive_model", env=env_name)
            raise ModelError(f"No archive model found for {env_name}")

        latest = archived[-1]
        active_dir = self._models_dir / "active" / env_name
        active_dir.mkdir(parents=True, exist_ok=True)

        # Strip timestamp suffix from archived name for active name
        dest = active_dir / latest.name
        shutil.move(str(latest), str(dest))
        log.info("model_rolled_back", model=latest.name, env=env_name)
        return dest

    def delete_archived(self, model_path: Path) -> None:
        """Remove a specific archived model file.

        Args:
            model_path: Path to the archived model to delete.

        Raises:
            ModelError: If the model file does not exist.
        """
        if not model_path.exists():
            log.error("archive_file_not_found", path=str(model_path))
            raise ModelError(f"Archive model does not exist: {model_path}")

        model_path.unlink()
        log.info("model_deleted", path=str(model_path))

    def get_state(self, env_name: str) -> dict[str, Any]:
        """Return current lifecycle state for an environment.

        Args:
            env_name: Environment name (equity or crypto).

        Returns:
            Dict with active_model, shadow_model, archive_count keys.
        """
        active_dir = self._models_dir / "active" / env_name
        shadow_dir = self._models_dir / "shadow" / env_name
        archive_dir = self._models_dir / "archive" / env_name

        active_models = list(active_dir.glob("*.zip")) if active_dir.exists() else []
        shadow_models = list(shadow_dir.glob("*.zip")) if shadow_dir.exists() else []
        archive_models = list(archive_dir.glob("*.zip")) if archive_dir.exists() else []

        return {
            "active_model": active_models[0].name if active_models else None,
            "shadow_model": shadow_models[0].name if shadow_models else None,
            "archive_count": len(archive_models),
        }

    def _archive_model(self, model_path: Path, env_name: str) -> Path:
        """Move a model file to the archive directory with timestamp suffix.

        Args:
            model_path: Path to the model file.
            env_name: Environment name.

        Returns:
            Path to the archived model.
        """
        archive_dir = self._models_dir / "archive" / env_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        stem = model_path.stem
        dest = archive_dir / f"{stem}_{timestamp}.zip"
        shutil.move(str(model_path), str(dest))
        log.info("model_archived", model=model_path.name, archive=dest.name, env=env_name)
        return dest


def _load_sb3_model(model_path: Path) -> Any:
    """Load a Stable Baselines3 model from file.

    Detects algorithm from filename prefix (ppo_, a2c_, sac_).

    Args:
        model_path: Path to the .zip model file.

    Returns:
        Loaded SB3 model instance.

    Raises:
        ModelError: If algorithm cannot be detected or model fails to load.
    """
    from stable_baselines3 import A2C, PPO, SAC  # noqa: PLC0415

    name_lower = model_path.stem.lower()
    algo_map: dict[str, type] = {"ppo": PPO, "a2c": A2C, "sac": SAC}

    for prefix, algo_cls in algo_map.items():
        if prefix in name_lower:
            log.info("loading_model", algorithm=prefix, path=str(model_path))
            return algo_cls.load(model_path)  # type: ignore[attr-defined]

    log.error("unknown_algorithm", filename=model_path.name)
    raise ModelError(f"Cannot detect algorithm from filename: {model_path.name}")


def _load_vec_normalize(stats_path: Path) -> Any:
    """Load VecNormalize statistics from a file.

    Args:
        stats_path: Path to the .pkl statistics file.

    Returns:
        Loaded VecNormalize instance with training disabled.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: PLC0415

    log.info("loading_vec_normalize", path=str(stats_path))
    dummy_env = DummyVecEnv([lambda: None])  # type: ignore[arg-type,return-value,list-item]
    vec_normalize = VecNormalize.load(str(stats_path), venv=dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def smoke_test_model(model_path: Path, env_name: str, obs_dim: int) -> dict[str, bool]:
    """Run 6-point smoke test on a model.

    Checks:
        1. deserialization - Model loads without error
        2. output_shape - predict() returns an array
        3. non_degenerate - 10 predictions with different obs produce varied actions
        4. inference_speed - Single predict() < 100ms
        5. vec_normalize - VecNormalize .pkl loads if present
        6. no_nan - No NaN values in any prediction

    Args:
        model_path: Path to the .zip model file.
        env_name: Environment name (equity or crypto).
        obs_dim: Observation dimension for generating test inputs.

    Returns:
        Dict mapping check name to pass/fail boolean.
    """
    results: dict[str, bool] = {
        "deserialization": False,
        "output_shape": False,
        "non_degenerate": False,
        "inference_speed": False,
        "vec_normalize": False,
        "no_nan": False,
    }

    # 1. Deserialization
    try:
        model = _load_sb3_model(model_path)
    except Exception:
        log.error("smoke_test_deserialization_failed", model=str(model_path))
        return results

    results["deserialization"] = True

    # 2. Output shape
    try:
        test_obs = np.zeros(obs_dim, dtype=np.float32)
        action, _ = model.predict(test_obs, deterministic=True)
        results["output_shape"] = isinstance(action, np.ndarray) and action.size > 0
    except Exception:
        log.error("smoke_test_output_shape_failed", model=str(model_path))
        return results

    # 3. Non-degenerate + 6. No NaN (collected over 10 iterations)
    rng = np.random.default_rng(seed=42)
    actions_list: list[np.ndarray] = []
    has_nan = False

    for _ in range(_SMOKE_TEST_ITERATIONS):
        test_obs = rng.standard_normal(obs_dim).astype(np.float32)
        action, _ = model.predict(test_obs, deterministic=True)
        actions_list.append(action)
        if np.any(np.isnan(action)):
            has_nan = True

    results["no_nan"] = not has_nan

    # Check non-degeneracy: not all actions identical
    if len(actions_list) >= 2:
        all_same = all(np.array_equal(actions_list[0], a) for a in actions_list[1:])
        results["non_degenerate"] = not all_same
    else:
        results["non_degenerate"] = False

    # 4. Inference speed
    try:
        test_obs = np.zeros(obs_dim, dtype=np.float32)
        start = time.perf_counter()
        model.predict(test_obs, deterministic=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        results["inference_speed"] = elapsed_ms < _INFERENCE_TIME_LIMIT_MS
        log.info("smoke_test_inference_time", elapsed_ms=round(elapsed_ms, 2))
    except Exception:
        log.error("smoke_test_inference_failed", model=str(model_path))

    # 5. VecNormalize
    stats_path = model_path.with_suffix(".pkl")
    if stats_path.exists():
        try:
            _load_vec_normalize(stats_path)
            results["vec_normalize"] = True
        except Exception:
            log.error("smoke_test_vec_normalize_failed", stats=str(stats_path))
    else:
        # No .pkl means VecNormalize not used -- pass by default
        results["vec_normalize"] = True

    log.info("smoke_test_complete", model=str(model_path), results=results)
    return results
