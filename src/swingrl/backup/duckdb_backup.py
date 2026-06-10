"""PostgreSQL backup via pg_dump with table/row verification.

Creates timestamped SQL dumps and verifies that required tables exist with
row counts > 0.

Usage:
    from swingrl.backup.duckdb_backup import backup_duckdb
    success = backup_duckdb(config, alerter)
"""

from __future__ import annotations

import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

# Tables that must exist and have rows in the backup
_REQUIRED_TABLES = ("ohlcv_daily", "ohlcv_4h")


def backup_duckdb(config: SwingRLConfig, alerter: Alerter) -> bool:
    """Create a timestamped PostgreSQL backup with table/row verification.

    Runs pg_dump to produce a compressed SQL dump, then verifies the
    live database has required tables with row counts > 0.

    Args:
        config: Validated SwingRLConfig with system.database_url and backup settings.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True on success, False on failure.
    """
    try:
        database_url = config.system.database_url

        backup_dir = Path(config.backup.backup_dir) / "postgres"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"swingrl_{timestamp}.sql.gz"

        # pg_dump with gzip compression
        result = subprocess.run(  # nosec B603,B607
            [
                "pg_dump",
                "--dbname",
                database_url,
                "--format=custom",
                "--compress=6",
                "--file",
                str(backup_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {result.stderr}")

        # Verify live database has expected tables
        _verify_backup(database_url)

        log.info("postgres_backup_complete", backup_path=str(backup_path))

        alerter.send_alert(
            "info",
            "PostgreSQL Backup Complete",
            f"Backup created: {backup_path.name}",
        )
        return True

    except Exception as exc:
        log.error("postgres_backup_failed", error=str(exc), exc_info=True)
        alerter.send_alert(
            "critical",
            "PostgreSQL Backup Failed",
            f"Backup failed: {exc}",
        )
        return False


def _verify_backup(database_url: str) -> None:
    """Verify the PostgreSQL database has required tables with rows > 0.

    Args:
        database_url: PostgreSQL connection string.

    Raises:
        RuntimeError: If required tables are missing or have zero rows.
    """
    with psycopg.connect(database_url) as conn:
        cur = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = [row[0] for row in cur.fetchall()]

        for table in _REQUIRED_TABLES:
            if table not in tables:
                raise RuntimeError(f"Required table missing: {table}")

            row = conn.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608  # nosec B608
            ).fetchone()
            count = row[0] if row is not None else 0
            if count == 0:
                raise RuntimeError(f"Table {table} has zero rows")

            log.info("postgres_backup_verified", table=table, row_count=count)
