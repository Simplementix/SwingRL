#!/usr/bin/env python3
"""Migration: add attribution columns to reward_adjustments.

Phase 19.1 training-refocus (Group C5). Adds 6 nullable columns to the
existing reward_adjustments table so every LLM advice event can later be
judged: did CPS improve after it?

New columns:
    fold_number         INTEGER
    iteration_number    INTEGER
    advice_id           TEXT
    fold_cps_v1_before  DOUBLE PRECISION
    fold_cps_v1_after   DOUBLE PRECISION
    advice_was_effective BOOLEAN

Additive only — no data rewrites, no column drops, no defaults that would
force a table rewrite.

The DDL is wrapped in ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` so the
script is idempotent: safe to run multiple times against the same database.

Locking: Postgres ``ADD COLUMN`` without a default is metadata-only and
takes a microsecond ACCESS EXCLUSIVE lock. Safe to run while training is
in progress (the running container will not write the new columns until
its code is restarted, but old INSERTs continue to work because all new
columns are nullable).

Usage:
    export DATABASE_URL="$(grep DATABASE_URL .env | cut -d= -f2-)"
    uv run python scripts/migrations/add_attribution_columns.py

    # Or via SSH against homelab (DATABASE_URL is set in the container env):
    ssh homelab "docker exec swingrl python3 /app/scripts/migrations/add_attribution_columns.py"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import structlog

# Allow running from repo root or from inside the container
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

log = structlog.get_logger(__name__)

# Each entry is (column_name, column_definition).
# Defined as a list of tuples (not a single multi-statement DDL) so we can
# log per-column status and use ``ADD COLUMN IF NOT EXISTS`` for idempotency.
_NEW_COLUMNS: list[tuple[str, str]] = [
    ("fold_number", "INTEGER"),
    ("iteration_number", "INTEGER"),
    ("advice_id", "TEXT"),
    ("fold_cps_v1_before", "DOUBLE PRECISION"),
    ("fold_cps_v1_after", "DOUBLE PRECISION"),
    ("advice_was_effective", "BOOLEAN"),
]


def get_database_url() -> str:
    """Read DATABASE_URL from environment, with a sensible error message."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Source it from .env or the container env.")
    return url


def apply_migration(conn: psycopg.Connection) -> dict[str, int]:
    """Apply the migration. Returns counts: added, already_present.

    Args:
        conn: Open psycopg connection. Caller manages commit/rollback.

    Returns:
        Dict with 'added' and 'already_present' counts.
    """
    counts = {"added": 0, "already_present": 0}
    with conn.cursor() as cur:
        # Check existing columns up front so we can report what was
        # already in place vs what we just added.
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reward_adjustments' AND table_schema = 'public'"  # noqa: S608
        )
        existing = {row[0] for row in cur.fetchall()}

        for col_name, col_def in _NEW_COLUMNS:
            if col_name in existing:
                log.info("column_already_present", column=col_name)
                counts["already_present"] += 1
                continue
            # IF NOT EXISTS makes this defensive; the pre-check above is
            # the source of truth for the report.
            cur.execute(
                f"ALTER TABLE reward_adjustments ADD COLUMN IF NOT EXISTS {col_name} {col_def}"  # noqa: S608
            )
            log.info("column_added", column=col_name, definition=col_def)
            counts["added"] += 1
    return counts


def migrate(db_url: str) -> dict[str, int]:
    """Connect, apply migration, commit, and return counts.

    Convenience entry point for callers that have a URL rather than a
    connection object (e.g. tests, one-off scripts).

    Args:
        db_url: Postgres connection URL.

    Returns:
        Dict with 'added' and 'already_present' counts.
    """
    with psycopg.connect(db_url) as conn:
        counts = apply_migration(conn)
        conn.commit()
    return counts


def main() -> int:
    """Run the migration. Returns process exit code."""
    db_url = get_database_url()
    log.info("migration_started", target="reward_adjustments")
    counts = migrate(db_url)
    log.info(
        "migration_complete",
        added=counts["added"],
        already_present=counts["already_present"],
        total_target=len(_NEW_COLUMNS),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
