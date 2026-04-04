"""Tests for SQLite backup (now delegates to PostgreSQL pg_dump backup)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swingrl.backup.sqlite_backup import backup_sqlite, rotate_old_backups

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


class TestBackupSqlite:
    """Tests for backup_sqlite function (now delegates to pg_dump)."""

    def test_delegates_to_pg_dump(self, tmp_path: Path) -> None:
        """PROD-01: backup_sqlite delegates to backup_duckdb (pg_dump)."""
        config = MagicMock()
        config.system.database_url = os.environ.get("DATABASE_URL", "")
        config.backup.backup_dir = str(tmp_path / "backups")
        alerter = MagicMock()

        with patch("swingrl.backup.duckdb_backup.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = backup_sqlite(config, alerter)

        assert result is True
        mock_run.assert_called_once()

    def test_alerts_on_success(self, tmp_path: Path) -> None:
        """PROD-01: Backup calls alerter.send_alert on success."""
        config = MagicMock()
        config.system.database_url = os.environ.get("DATABASE_URL", "")
        config.backup.backup_dir = str(tmp_path / "backups")
        alerter = MagicMock()

        with patch("swingrl.backup.duckdb_backup.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            backup_sqlite(config, alerter)

        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "info"


class TestRotateOldBackups:
    """Tests for rotate_old_backups function."""

    def test_removes_old_backups(self, tmp_path: Path) -> None:
        """PROD-01: Rotation removes backups older than retention_days."""
        backup_dir = tmp_path / "sqlite"
        backup_dir.mkdir()

        old_file = backup_dir / "swingrl_20260101_000000.sql.gz"
        old_file.write_text("fake")
        old_mtime = time.time() - (20 * 86400)
        os.utime(str(old_file), (old_mtime, old_mtime))

        removed = rotate_old_backups(backup_dir, retention_days=14)

        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_backups(self, tmp_path: Path) -> None:
        """PROD-01: Rotation keeps backups within retention window."""
        backup_dir = tmp_path / "sqlite"
        backup_dir.mkdir()

        recent_file = backup_dir / "swingrl_20260310_000000.sql.gz"
        recent_file.write_text("fake")

        removed = rotate_old_backups(backup_dir, retention_days=14)

        assert removed == 0
        assert recent_file.exists()
