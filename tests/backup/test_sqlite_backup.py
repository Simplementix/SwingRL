"""Tests for SQLite backup with integrity verification and rotation."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swingrl.backup.sqlite_backup import backup_sqlite, rotate_old_backups


@pytest.fixture()
def backup_env(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal SQLite DB, config files, and backup dir for testing."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_path = db_dir / "trading_ops.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test_table (value) VALUES ('hello')")
    conn.commit()
    conn.close()

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_yaml = config_dir / "swingrl.yaml"
    config_yaml.write_text("trading_mode: paper\n")

    env_file = tmp_path / ".env"
    env_file.write_text("SWINGRL_TRADING_MODE=paper\n")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    return {
        "db_path": db_path,
        "config_yaml": config_yaml,
        "env_file": env_file,
        "backup_dir": backup_dir,
        "tmp_path": tmp_path,
    }


def _make_config(backup_env: dict[str, Path]) -> MagicMock:
    """Build a mock SwingRLConfig matching BackupConfig fields."""
    config = MagicMock()
    config.system.sqlite_path = str(backup_env["db_path"])
    config.backup.backup_dir = str(backup_env["backup_dir"])
    config.backup.sqlite_retention_days = 14
    return config


class TestBackupSqlite:
    """Tests for backup_sqlite function."""

    def test_creates_verified_copy(self, backup_env: dict[str, Path]) -> None:
        """PROD-01: SQLite backup creates a copy that passes PRAGMA integrity_check."""
        config = _make_config(backup_env)
        alerter = MagicMock()

        result = backup_sqlite(config, alerter)

        assert result is True
        sqlite_dir = backup_env["backup_dir"] / "sqlite"
        db_files = list(sqlite_dir.glob("trading_ops_*.db"))
        assert len(db_files) == 1

        # Verify integrity of backup
        conn = sqlite3.connect(str(db_files[0]))
        check = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        assert check[0] == "ok"

    def test_copies_config_files(self, backup_env: dict[str, Path]) -> None:
        """PROD-01: SQLite backup copies config files alongside the DB."""
        config = _make_config(backup_env)
        alerter = MagicMock()

        backup_sqlite(config, alerter)

        sqlite_dir = backup_env["backup_dir"] / "sqlite"
        assert (sqlite_dir / "swingrl.yaml").exists()
        assert (sqlite_dir / ".env").exists()

    def test_alerts_on_success(self, backup_env: dict[str, Path]) -> None:
        """PROD-01: Backup calls alerter.send_alert on success."""
        config = _make_config(backup_env)
        alerter = MagicMock()

        backup_sqlite(config, alerter)

        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "info"  # level

    def test_alerts_critical_on_failure(self, tmp_path: Path) -> None:
        """PROD-01: Backup calls alerter.send_alert with critical level on failure."""
        config = MagicMock()
        config.system.sqlite_path = str(tmp_path / "nonexistent.db")
        config.backup.backup_dir = str(tmp_path / "backups")
        config.backup.sqlite_retention_days = 14
        alerter = MagicMock()

        result = backup_sqlite(config, alerter)

        assert result is False
        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "critical"


class TestRotateOldBackups:
    """Tests for rotate_old_backups function."""

    def test_removes_old_backups(self, tmp_path: Path) -> None:
        """PROD-01: Rotation removes backups older than retention_days."""
        backup_dir = tmp_path / "sqlite"
        backup_dir.mkdir()

        # Create an old file (set mtime to 20 days ago)
        old_file = backup_dir / "trading_ops_20260101_000000.db"
        old_file.write_text("fake")
        old_mtime = time.time() - (20 * 86400)
        import os

        os.utime(str(old_file), (old_mtime, old_mtime))

        removed = rotate_old_backups(backup_dir, retention_days=14)

        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_backups(self, tmp_path: Path) -> None:
        """PROD-01: Rotation keeps backups within retention window."""
        backup_dir = tmp_path / "sqlite"
        backup_dir.mkdir()

        recent_file = backup_dir / "trading_ops_20260310_000000.db"
        recent_file.write_text("fake")

        removed = rotate_old_backups(backup_dir, retention_days=14)

        assert removed == 0
        assert recent_file.exists()
