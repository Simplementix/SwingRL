"""PostgreSQL backup (formerly SQLite online backup).

Now delegates to the same pg_dump-based backup as duckdb_backup.py since
both DuckDB and SQLite have been consolidated into PostgreSQL.

Usage:
    from swingrl.backup.sqlite_backup import backup_sqlite
    success = backup_sqlite(config, alerter)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from swingrl.backup.duckdb_backup import backup_duckdb

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


def backup_sqlite(config: SwingRLConfig, alerter: Alerter) -> bool:
    """Create a PostgreSQL backup (delegates to backup_duckdb).

    The SQLite database has been migrated to PostgreSQL.  This function
    preserves the old call signature and delegates to the shared pg_dump
    backup implementation.

    Args:
        config: Validated SwingRLConfig with system.database_url.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True on success, False on failure.
    """
    return backup_duckdb(config, alerter)


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
    """Remove backup files older than retention_days from backup directory.

    Args:
        backup_dir: Directory containing backup files.
        retention_days: Maximum age in days before removal.

    Returns:
        Count of removed files.
    """
    cutoff = time.time() - (retention_days * 86400)
    removed = 0

    for backup_file in list(backup_dir.glob("*.sql.gz")) + list(backup_dir.glob("*.db")):
        if backup_file.stat().st_mtime < cutoff:
            backup_file.unlink()
            log.info("backup_rotated", file=str(backup_file))
            removed += 1

    return removed
