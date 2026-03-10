"""Tests for the security checklist verification script.

PROD-06: Validates non-root container, env_file permissions, key rotation schedule.
"""

from __future__ import annotations

import stat
from pathlib import Path


class TestNonRootUserDetection:
    """Verify Dockerfile USER directive check."""

    def test_detects_non_root_user(self, tmp_path: Path) -> None:
        """PROD-06: Security check detects non-root USER trader in Dockerfile."""
        from scripts.security_checklist import check_non_root_user

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11-slim\nUSER trader\n")
        result = check_non_root_user(dockerfile)
        assert result.passed is True
        assert "trader" in result.detail

    def test_fails_on_root_user(self, tmp_path: Path) -> None:
        """PROD-06: Security check fails when Dockerfile has no USER directive."""
        from scripts.security_checklist import check_non_root_user

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11-slim\nRUN echo hello\n")
        result = check_non_root_user(dockerfile)
        assert result.passed is False

    def test_fails_on_user_root(self, tmp_path: Path) -> None:
        """PROD-06: Security check fails when USER is root."""
        from scripts.security_checklist import check_non_root_user

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.11-slim\nUSER root\n")
        result = check_non_root_user(dockerfile)
        assert result.passed is False


class TestEnvFileReference:
    """Verify docker-compose.yml references env_file."""

    def test_detects_env_file_in_compose(self, tmp_path: Path) -> None:
        """PROD-06: Security check verifies env_file referenced in docker-compose.yml."""
        from scripts.security_checklist import check_env_file_reference

        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services:\n  swingrl:\n    env_file: .env\n")
        result = check_env_file_reference(compose)
        assert result.passed is True

    def test_fails_without_env_file(self, tmp_path: Path) -> None:
        """PROD-06: Security check fails when no env_file in compose."""
        from scripts.security_checklist import check_env_file_reference

        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services:\n  swingrl:\n    image: test\n")
        result = check_env_file_reference(compose)
        assert result.passed is False


class TestEnvFilePermissions:
    """Verify .env file permission checks."""

    def test_warns_on_wrong_permissions(self, tmp_path: Path) -> None:
        """PROD-06: Security check warns if .env permissions are not 600."""
        from scripts.security_checklist import check_env_permissions

        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value\n")
        env_file.chmod(0o644)
        result = check_env_permissions(env_file)
        assert result.passed is False
        assert "600" in result.detail or "644" in result.detail

    def test_passes_on_correct_permissions(self, tmp_path: Path) -> None:
        """PROD-06: Security check passes when .env is chmod 600."""
        from scripts.security_checklist import check_env_permissions

        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value\n")
        env_file.chmod(0o600)
        result = check_env_permissions(env_file)
        assert result.passed is True


class TestKeyRotationSchedule:
    """Verify key rotation schedule documentation."""

    def test_documents_staggered_rotation(self) -> None:
        """PROD-06: Security check documents staggered key rotation schedule."""
        from scripts.security_checklist import get_key_rotation_schedule

        schedule = get_key_rotation_schedule()
        assert "alpaca" in schedule.lower()
        assert "binance" in schedule.lower()
        assert "90" in schedule or "month" in schedule.lower()


class TestResourceLimits:
    """Verify docker-compose.yml resource limits."""

    def test_detects_resource_limits(self, tmp_path: Path) -> None:
        """PROD-06: Security check detects mem_limit and cpus in compose."""
        from scripts.security_checklist import check_resource_limits

        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services:\n  swingrl:\n    mem_limit: 2.5g\n    cpus: 1.0\n")
        result = check_resource_limits(compose)
        assert result.passed is True

    def test_fails_without_resource_limits(self, tmp_path: Path) -> None:
        """PROD-06: Security check fails when no resource limits."""
        from scripts.security_checklist import check_resource_limits

        compose = tmp_path / "docker-compose.yml"
        compose.write_text("services:\n  swingrl:\n    image: test\n")
        result = check_resource_limits(compose)
        assert result.passed is False


class TestFixPermissions:
    """Verify --fix flag auto-sets .env to 600."""

    def test_fix_sets_600(self, tmp_path: Path) -> None:
        """PROD-06: --fix flag auto-sets .env permissions to 600."""
        from scripts.security_checklist import fix_env_permissions

        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=value\n")
        env_file.chmod(0o644)
        fix_env_permissions(env_file)
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
