# scripts/migrations/add_options_capture_tables.py
"""One-time additive migration: create options_snapshots + options_chains (spec §8.2).

This is the "V011_options_capture" migration. It is ADDITIVE ONLY (new tables),
safe to run against live pg16 while the trader runs (A30). Idempotent.

Usage:
    DATABASE_URL=... uv run python scripts/migrations/add_options_capture_tables.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import psycopg
import structlog

from swingrl.data.options.schema import ensure_monthly_partition, ensure_options_schema

log = structlog.get_logger(__name__)


def get_database_url() -> str:
    """Read DATABASE_URL from the environment (raises KeyError if unset)."""
    return os.environ["DATABASE_URL"]


def apply_migration(conn: psycopg.Connection) -> None:
    """Create both tables + the current-month partition (additive, idempotent)."""
    ensure_options_schema(conn)
    ensure_monthly_partition(conn, datetime.now(UTC).date())


def main() -> int:
    """Apply the migration against DATABASE_URL and commit."""
    with psycopg.connect(get_database_url()) as conn:
        apply_migration(conn)
        conn.commit()
    log.info("options_capture_migration_applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
