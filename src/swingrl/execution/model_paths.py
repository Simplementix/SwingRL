"""Single source of the on-disk active-model layout.

The trader loads, and the trainer / lifecycle write, models under one per-algo
layout: ``models/active/{env}/{algo}/model.zip`` with a sibling
``vec_normalize.pkl``. This module is the ONE place that encodes that layout so
the loader, the lifecycle promote/archive/rollback, and deployment verification
never drift apart (review H2 — promotion used to write a flat layout the loader
never read).

Path construction only: nothing here creates, moves, or deletes files, so
importing or calling it can never write under ``models/active/`` (A30 write ban).
"""

from __future__ import annotations

from pathlib import Path

# Canonical artifact filenames inside a per-algo active directory.
MODEL_FILENAME = "model.zip"
VEC_NORMALIZE_FILENAME = "vec_normalize.pkl"


def active_model_dir(models_dir: Path, env: str, algo: str) -> Path:
    """Return the per-algo active directory ``models_dir/active/{env}/{algo}``.

    Args:
        models_dir: Root models directory.
        env: Environment name ("equity" or "crypto").
        algo: Algorithm name ("ppo", "a2c", or "sac").

    Returns:
        The per-algo active directory path (not created).
    """
    return models_dir / "active" / env / algo


def active_model_paths(models_dir: Path, env: str, algo: str) -> tuple[Path, Path]:
    """Return the ``(model.zip, vec_normalize.pkl)`` paths for an active model.

    Args:
        models_dir: Root models directory.
        env: Environment name ("equity" or "crypto").
        algo: Algorithm name ("ppo", "a2c", or "sac").

    Returns:
        Tuple of (model_path, vec_normalize_path) at the canonical per-algo layout.
    """
    base = active_model_dir(models_dir, env, algo)
    return base / MODEL_FILENAME, base / VEC_NORMALIZE_FILENAME
