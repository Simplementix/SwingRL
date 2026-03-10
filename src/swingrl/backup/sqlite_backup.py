"""SQLite online backup with integrity verification and retention rotation.

Uses sqlite3.backup() API for safe online backup of trading_ops.db.
Copies config/swingrl.yaml and .env alongside the backup for disaster recovery.
Rotates backups older than configured retention period.

Usage:
    from swingrl.backup.sqlite_backup import backup_sqlite
    success = backup_sqlite(config, alerter)
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


def backup_sqlite(config: SwingRLConfig, alerter: Alerter) -> bool:
    """Create a timestamped SQLite backup with integrity verification.

    Uses sqlite3.backup() API (not shutil.copy) for safe online backup.
    Copies config/swingrl.yaml and .env alongside the DB backup.
    Runs PRAGMA integrity_check on the backup copy.
    Rotates old backups based on sqlite_retention_days.

    Args:
        config: Validated SwingRLConfig with system.sqlite_path and backup settings.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True on success, False on failure.
    """
    try:
        source_path = Path(config.system.sqlite_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source SQLite DB not found: {source_path}")

        backup_dir = Path(config.backup.backup_dir) / "sqlite"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"trading_ops_{timestamp}.db"

        # Use sqlite3.backup() for online backup
        source_conn = sqlite3.connect(str(source_path))
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            source_conn.backup(backup_conn)
        finally:
            backup_conn.close()
            source_conn.close()

        # Verify integrity of backup
        verify_conn = sqlite3.connect(str(backup_path))
        try:
            check_result = verify_conn.execute("PRAGMA integrity_check").fetchone()
            if check_result is None or check_result[0] != "ok":
                raise RuntimeError(f"Backup integrity check failed: {check_result}")
        finally:
            verify_conn.close()

        # Copy config files alongside backup
        _copy_config_files(backup_dir, source_path.parent.parent)

        # Rotate old backups
        removed = rotate_old_backups(backup_dir, config.backup.sqlite_retention_days)

        log.info(
            "sqlite_backup_complete",
            backup_path=str(backup_path),
            rotated_count=removed,
        )

        alerter.send_alert(
            "info",
            "SQLite Backup Complete",
            f"Backup created: {backup_path.name}, rotated {removed} old backups",
        )
        return True

    except Exception as exc:
        log.error("sqlite_backup_failed", error=str(exc), exc_info=True)
        alerter.send_alert(
            "critical",
            "SQLite Backup Failed",
            f"Backup failed: {exc}",
        )
        return False


def _copy_config_files(backup_dir: Path, repo_root: Path) -> None:
    """Copy config/swingrl.yaml and .env to backup directory.

    Args:
        backup_dir: Target directory for config copies.
        repo_root: Repository root to search for config files.
    """
    config_yaml = repo_root / "config" / "swingrl.yaml"
    if config_yaml.exists():
        shutil.copy2(str(config_yaml), str(backup_dir / "swingrl.yaml"))

    env_file = repo_root / ".env"
    if env_file.exists():
        shutil.copy2(str(env_file), str(backup_dir / ".env"))


def rotate_old_backups(backup_dir: Path, retention_days: int) -> int:
    """Remove .db files older than retention_days from backup directory.

    Args:
        backup_dir: Directory containing backup .db files.
        retention_days: Maximum age in days before removal.

    Returns:
        Count of removed files.
    """
    cutoff = time.time() - (retention_days * 86400)
    removed = 0

    for db_file in backup_dir.glob("*.db"):
        if db_file.stat().st_mtime < cutoff:
            db_file.unlink()
            log.info("backup_rotated", file=str(db_file))
            removed += 1

    return removed
