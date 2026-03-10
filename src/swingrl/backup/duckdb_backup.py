"""DuckDB file backup with CHECKPOINT and table/row verification.

Flushes DuckDB WAL via CHECKPOINT, copies the file, and verifies the backup
contains expected tables with row counts > 0.

Usage:
    from swingrl.backup.duckdb_backup import backup_duckdb
    success = backup_duckdb(config, alerter)
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

# Tables that must exist and have rows in the backup
_REQUIRED_TABLES = ("ohlcv_daily", "ohlcv_4h")


def backup_duckdb(config: SwingRLConfig, alerter: Alerter) -> bool:
    """Create a timestamped DuckDB backup with table/row verification.

    Connects to DuckDB, runs CHECKPOINT to flush WAL, closes connection,
    then copies the file. Opens the backup to verify required tables exist
    and have row counts > 0. Never rotates backups (duckdb_rotate=False).

    Args:
        config: Validated SwingRLConfig with system.duckdb_path and backup settings.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True on success, False on failure.
    """
    try:
        source_path = Path(config.system.duckdb_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source DuckDB not found: {source_path}")

        # Flush WAL before copy
        conn = duckdb.connect(str(source_path))
        try:
            conn.execute("CHECKPOINT")
        finally:
            conn.close()

        backup_dir = Path(config.backup.backup_dir) / "duckdb"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"market_data_{timestamp}.ddb"

        shutil.copy2(str(source_path), str(backup_path))

        # Verify backup
        _verify_backup(backup_path)

        log.info("duckdb_backup_complete", backup_path=str(backup_path))

        alerter.send_alert(
            "info",
            "DuckDB Backup Complete",
            f"Backup created: {backup_path.name}",
        )
        return True

    except Exception as exc:
        log.error("duckdb_backup_failed", error=str(exc), exc_info=True)
        alerter.send_alert(
            "critical",
            "DuckDB Backup Failed",
            f"Backup failed: {exc}",
        )
        return False


def _verify_backup(backup_path: Path) -> None:
    """Verify the DuckDB backup has required tables with rows > 0.

    Args:
        backup_path: Path to the backup DuckDB file.

    Raises:
        RuntimeError: If required tables are missing or have zero rows.
    """
    conn = duckdb.connect(str(backup_path), read_only=True)
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]

        for table in _REQUIRED_TABLES:
            if table not in tables:
                raise RuntimeError(f"Required table missing from backup: {table}")

            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608  # nosec B608
            count = row[0] if row is not None else 0
            if count == 0:
                raise RuntimeError(f"Table {table} has zero rows in backup")

            log.info("duckdb_backup_verified", table=table, row_count=count)
    finally:
        conn.close()
