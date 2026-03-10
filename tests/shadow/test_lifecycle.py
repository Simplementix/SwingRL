"""Tests for model lifecycle state machine transitions.

Validates Training -> Shadow -> Active -> Archive -> Deletion
lifecycle with proper error handling for invalid transitions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
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
    """PROD-02: promote moves shadow model to active, archiving current active."""

    def test_promote_moves_shadow_to_active(self, tmp_path: Path) -> None:
        """PROD-02: Shadow model becomes active model."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_model = shadow_dir / "model.zip"
        shadow_model.write_bytes(b"shadow_model")

        result = lifecycle.promote("equity")

        assert result.parent == tmp_path / "active" / "equity"
        assert result.read_bytes() == b"shadow_model"
        assert not shadow_model.exists()

    def test_promote_archives_existing_active(self, tmp_path: Path) -> None:
        """PROD-02: Current active model is archived before promotion."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        # Set up existing active model
        active_dir = tmp_path / "active" / "equity"
        active_dir.mkdir(parents=True, exist_ok=True)
        active_model = active_dir / "old_model.zip"
        active_model.write_bytes(b"old_active")

        # Set up shadow model
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_model = shadow_dir / "new_model.zip"
        shadow_model.write_bytes(b"new_shadow")

        lifecycle.promote("equity")

        # Old active should be in archive
        archive_files = list((tmp_path / "archive" / "equity").glob("*.zip"))
        assert len(archive_files) == 1
        assert archive_files[0].read_bytes() == b"old_active"

    def test_promote_with_no_active_model_works(self, tmp_path: Path) -> None:
        """PROD-02: Promotion works when no active model exists (no archive step)."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_model = shadow_dir / "model.zip"
        shadow_model.write_bytes(b"shadow_data")

        result = lifecycle.promote("equity")

        assert result.exists()
        assert (
            not (tmp_path / "archive" / "equity").exists()
            or len(list((tmp_path / "archive" / "equity").glob("*"))) == 0
        )

    def test_promote_raises_if_no_shadow_model(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if no shadow model to promote."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        with pytest.raises(ModelError, match="shadow"):
            lifecycle.promote("equity")


class TestArchive:
    """PROD-02: archive moves active model to archive directory."""

    def test_archive_moves_active_to_archive(self, tmp_path: Path) -> None:
        """PROD-02: Active model is moved to archive."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        active_dir = tmp_path / "active" / "equity"
        active_dir.mkdir(parents=True, exist_ok=True)
        active_model = active_dir / "model.zip"
        active_model.write_bytes(b"active_data")

        result = lifecycle.archive("equity")

        assert result.parent == tmp_path / "archive" / "equity"
        assert result.read_bytes() == b"active_data"
        assert not active_model.exists()

    def test_archive_raises_if_no_active_model(self, tmp_path: Path) -> None:
        """PROD-02: Raises ModelError if no active model to archive."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        with pytest.raises(ModelError, match="active"):
            lifecycle.archive("equity")


class TestRollback:
    """PROD-02: rollback restores most recent archived model."""

    def test_rollback_restores_latest_archive(self, tmp_path: Path) -> None:
        """PROD-02: Most recent archive model becomes active."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)
        archive_dir = tmp_path / "archive" / "equity"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / "model_20260101_120000.zip"
        archived.write_bytes(b"archived_model")

        result = lifecycle.rollback("equity")

        assert result.parent == tmp_path / "active" / "equity"
        assert result.read_bytes() == b"archived_model"

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
        """PROD-02: State dict contains active, shadow, archive info."""
        lifecycle = ModelLifecycle(models_dir=tmp_path)

        # Set up models
        active_dir = tmp_path / "active" / "equity"
        active_dir.mkdir(parents=True, exist_ok=True)
        (active_dir / "model.zip").write_bytes(b"active")

        shadow_dir = tmp_path / "shadow" / "equity"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        (shadow_dir / "candidate.zip").write_bytes(b"shadow")

        archive_dir = tmp_path / "archive" / "equity"
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "old1.zip").write_bytes(b"old1")
        (archive_dir / "old2.zip").write_bytes(b"old2")

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
