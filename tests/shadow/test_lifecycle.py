"""Tests for model lifecycle state machine transitions.

Validates Training -> Shadow -> Active -> Archive -> Deletion lifecycle. The
active and archive layouts are per-algo (``active/{env}/{algo}/model.zip`` plus
``vec_normalize.pkl``) so promotion writes the exact layout the loader reads
(review H2). Shadow stays flat (a single candidate per env, algo-prefixed name)
so the live shadow-inference runner keeps working.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from swingrl.execution.model_paths import active_model_paths

from swingrl.shadow.lifecycle import ModelLifecycle, ModelState
from swingrl.utils.exceptions import ModelError


class TestModelState:
    """PROD-02: ModelState enum has all required states."""

    def test_training_state_exists(self) -> None:
        """PROD-02: TRAINING state exists in enum."""
        assert ModelState.TRAINING.value == "training"

    def test_shadow_state_exists(self) -> None:
        """PROD-02: SHADOW state exists in enum."""
        assert ModelState.SHADOW.value == "shadow"

    def test_active_state_exists(self) -> None:
        """PROD-02: ACTIVE state exists in enum."""
        assert ModelState.ACTIVE.value == "active"

    def test_archive_state_exists(self) -> None:
        """PROD-02: ARCHIVE state exists in enum."""
        assert ModelState.ARCHIVE.value == "archive"

    def test_deleted_state_exists(self) -> None:
        """PROD-02: DELETED state exists in enum."""
        assert ModelState.DELETED.value == "deleted"


class TestModelLifecycleInit:
    """PROD-02: ModelLifecycle creates required directory structure."""

    def test_creates_subdirectories(self, tmp_path: Path) -> None:
        """PROD-02: Init creates active/shadow/archive subdirs."""
        _lifecycle = ModelLifecycle(models_dir=tmp_path)
        assert (tmp_path / "active").is_dir()
        assert (tmp_path / "shadow").is_dir()
        assert (tmp_path / "archive").is_dir()


class TestDeployToShadow:
    """PROD-02: deploy_to_shadow copies model to shadow directory."""

    def test_deploy_copies_model_to_shadow(self, tmp_path: Path) -> None:
        """PROD-02: Model file is copied to models/shadow/{env_name}/."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        model_file = tmp_path / "trained_model.zip"
        model_file.write_bytes(b"fake_model_data")

        dest = lifecycle.deploy_to_shadow(model_file, "equity")

        assert dest.exists()
        assert dest.parent == tmp_path / "shadow" / "equity"
        assert dest.read_bytes() == b"fake_model_data"

    def test_deploy_raises_if_model_missing(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if source model does not exist."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        missing = tmp_path / "nonexistent.zip"

        with pytest.raises(ModelError, match="does not exist"):
            lifecycle.deploy_to_shadow(missing, "equity")


class TestPromote:
    """PROD-02 / review H2: promote moves the flat shadow candidate into per-algo active."""

    def test_promote_moves_shadow_to_per_algo_active(self, tmp_path: Path) -> None:
        """H2: Shadow candidate becomes the active model at active/{env}/{algo}/ incl. vec."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "ppo_equity_v1.zip").write_bytes(b"shadow_model")
        (shadow_dir / "ppo_equity_v1.pkl").write_bytes(b"shadow_vec")

        results = lifecycle.promote("equity")

        model_path, vec_path = active_model_paths(tmp_path, "equity", "ppo")
        assert model_path.exists()
        assert model_path.read_bytes() == b"shadow_model"
        assert vec_path.exists()
        assert vec_path.read_bytes() == b"shadow_vec"
        assert model_path in results
        assert not (shadow_dir / "ppo_equity_v1.zip").exists()

    def test_promote_archives_existing_active(self, tmp_path: Path) -> None:
        """PROD-02: Current active per-algo model is archived before promotion."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        # Existing active ppo model (per-algo layout)
        model_path, vec_path = active_model_paths(tmp_path, "equity", "ppo")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"old_active")
        vec_path.write_bytes(b"old_vec")

        # Shadow candidate for the same algo
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "ppo_equity_v2.zip").write_bytes(b"new_shadow")

        lifecycle.promote("equity")

        archived = list((tmp_path / "archive" / "equity" / "ppo").glob("**/model.zip"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == b"old_active"
        assert model_path.read_bytes() == b"new_shadow"

    def test_promote_with_no_active_model_works(self, tmp_path: Path) -> None:
        """PROD-02: Promotion works when no active model exists (no archive step)."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "sac_equity.zip").write_bytes(b"shadow_data")

        results = lifecycle.promote("equity")

        model_path, _ = active_model_paths(tmp_path, "equity", "sac")
        assert model_path.exists()
        assert model_path in results
        archive_dir = tmp_path / "archive" / "equity"
        assert not archive_dir.exists() or not list(archive_dir.glob("**/*.zip"))

    def test_promote_raises_if_no_shadow_model(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if no shadow model to promote."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        with pytest.raises(ModelError, match="shadow"):
            lifecycle.promote("equity")

    def test_promote_raises_on_unknown_algo(self, tmp_path: Path) -> None:
        """H2: A shadow candidate with no algo prefix cannot be routed to a per-algo slot."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "mystery_model.zip").write_bytes(b"data")

        with pytest.raises(ModelError, match="algorithm"):
            lifecycle.promote("equity")


class TestArchive:
    """PROD-02: archive moves each active per-algo model to the archive directory."""

    def test_archive_moves_active_to_archive(self, tmp_path: Path) -> None:
        """PROD-02: Active per-algo model (both files) is moved to archive."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        model_path, vec_path = active_model_paths(tmp_path, "equity", "ppo")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"active_data")
        vec_path.write_bytes(b"active_vec")

        results = lifecycle.archive("equity")

        assert not model_path.exists()
        archived = list((tmp_path / "archive" / "equity" / "ppo").glob("**/model.zip"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == b"active_data"
        assert len(results) == 1

    def test_archive_raises_if_no_active_model(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if no active model to archive."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        with pytest.raises(ModelError, match="active"):
            lifecycle.archive("equity")


class TestRollback:
    """PROD-02: rollback restores most recent archived per-algo model."""

    def test_rollback_restores_latest_archive(self, tmp_path: Path) -> None:
        """PROD-02: Most recent per-algo archive (both files) becomes active."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        archive_dir = tmp_path / "archive" / "equity" / "ppo" / "20260101_120000"
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "model.zip").write_bytes(b"archived_model")
        (archive_dir / "vec_normalize.pkl").write_bytes(b"archived_vec")

        results = lifecycle.rollback("equity")

        model_path, vec_path = active_model_paths(tmp_path, "equity", "ppo")
        assert model_path.exists()
        assert model_path.read_bytes() == b"archived_model"
        assert vec_path.read_bytes() == b"archived_vec"
        assert model_path in results

    def test_rollback_raises_if_no_archive(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if no archived model exists."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        with pytest.raises(ModelError, match="archive"):
            lifecycle.rollback("equity")


class TestDeleteArchived:
    """PROD-02: delete_archived removes specific archived model."""

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        """PROD-02: Specified archive file is deleted."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        archive_dir = tmp_path / "archive" / "equity"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / "model.zip"
        archived.write_bytes(b"data")

        lifecycle.delete_archived(archived)

        assert not archived.exists()

    def test_delete_raises_if_file_missing(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if archive file does not exist."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        missing = tmp_path / "archive" / "equity" / "nonexistent.zip"

        with pytest.raises(ModelError, match="does not exist"):
            lifecycle.delete_archived(missing)


class TestGetState:
    """PROD-02: get_state returns current lifecycle state per environment."""

    def test_get_state_returns_model_info(self, tmp_path: Path) -> None:
        """PROD-02: State dict contains active, shadow, archive info (per-algo aware)."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        model_path, _ = active_model_paths(tmp_path, "equity", "ppo")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"active")

        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "ppo_candidate.zip").write_bytes(b"shadow")

        for ts in ("20260101_000000", "20260102_000000"):
            algo_archive = tmp_path / "archive" / "equity" / "ppo" / ts
            algo_archive.mkdir(parents=True, exist_ok=True)
            (algo_archive / "model.zip").write_bytes(b"old")

        state = lifecycle.get_state("equity")

        assert state["active_model"] is not None
        assert state["shadow_model"] is not None
        assert state["archive_count"] == 2

    def test_get_state_empty_env(self, tmp_path: Path) -> None:
        """PROD-02: State dict handles environment with no models."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        state = lifecycle.get_state("crypto")

        assert state["active_model"] is None
        assert state["shadow_model"] is None
        assert state["archive_count"] == 0
