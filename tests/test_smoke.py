# Phase 1 smoke test — import verification + structure validation
# Note: pandas_ta replaced by stockstats (see deferred-items.md for resolution)
import importlib
import sys
from pathlib import Path

import pytest


def test_python_version() -> None:
    """ENV-01: Python 3.11 required."""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version_info}"


def test_torch_importable() -> None:
    """ENV-01: torch must import without error."""
    import torch  # noqa: F401

    assert torch.__version__


def test_torch_mps_available_on_mac() -> None:
    """ENV-01: MPS acceleration on M1 Mac (skip in Docker/CI)."""
    import platform

    import torch

    if platform.system() != "Darwin":
        pytest.skip("MPS only available on macOS")
    # On macOS, MPS should be available (fails test if not)
    assert torch.backends.mps.is_available(), (
        "MPS not available — install standard macOS torch, not CPU-only wheel"
    )


def test_core_imports() -> None:
    """All project dependencies must import cleanly.

    Note: pandas_ta (original) is not available for Python 3.11 on PyPI.
    Using stockstats (FinRL's native TA library) as replacement for Phase 1-5.
    pandas_ta re-evaluation is scheduled for Phase 6.
    """
    packages = [
        "stable_baselines3",
        "gymnasium",
        "pandas",
        "numpy",
        "stockstats",
        "hmmlearn",
        "pydantic",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_swingrl_package_importable() -> None:
    """src/swingrl package must be importable."""
    import swingrl  # noqa: F401


def test_py_typed_marker_exists() -> None:
    """py.typed marker must exist for mypy compliance."""
    import swingrl

    pkg_dir = Path(swingrl.__file__).parent  # type: ignore[arg-type]
    assert (pkg_dir / "py.typed").exists(), "py.typed marker missing from src/swingrl/"


def test_directory_structure() -> None:
    """ENV-03: Canonical directory structure must exist at repo root."""
    repo_root = Path(__file__).parent.parent
    required_dirs = [
        "src/swingrl",
        "config",
        "data",
        "db",
        "models",
        "tests",
        "scripts",
        "status",
    ]
    for d in required_dirs:
        assert (repo_root / d).is_dir(), f"Required directory missing: {d}"


def test_swingrl_subpackages_exist() -> None:
    """src/swingrl/ must have all subpackages with __init__.py."""
    import swingrl

    pkg_dir = Path(swingrl.__file__).parent  # type: ignore[arg-type]
    required_subpkgs = [
        "data",
        "envs",
        "agents",
        "training",
        "execution",
        "monitoring",
        "config",
        "utils",
    ]
    for subpkg in required_subpkgs:
        subpkg_dir = pkg_dir / subpkg
        assert subpkg_dir.is_dir(), f"Subpackage directory missing: src/swingrl/{subpkg}/"
        assert (subpkg_dir / "__init__.py").exists(), (
            f"__init__.py missing: src/swingrl/{subpkg}/__init__.py"
        )
