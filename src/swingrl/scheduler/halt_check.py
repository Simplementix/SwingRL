"""Emergency halt flag CRUD for pre-cycle halt checks.

The emergency_flags table in PostgreSQL stores a single 'halt' flag that all
scheduled jobs check before executing. When active, all trading cycles skip.

Usage:
    from swingrl.scheduler.halt_check import is_halted, set_halt, clear_halt

    if is_halted(db):
        log.warning("cycle_skipped", reason="halt_flag_active")
        return []
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)


def init_emergency_flags(db: DatabaseManager) -> None:
    """Ensure the emergency_flags table exists (managed by postgres_schema.py).

    Args:
        db: DatabaseManager providing PostgreSQL connection.
    """
    log.debug("init_emergency_flags_noop", reason="schema_managed_by_postgres_schema")


def is_halted(db: DatabaseManager) -> bool:
    """Check if the emergency halt flag is active.

    If the emergency_flags table does not exist, creates it and returns False.

    Args:
        db: DatabaseManager providing PostgreSQL connection.

    Returns:
        True if the halt flag is active, False otherwise.
    """
    with db.connection() as conn:
        try:
            row = conn.execute(
                "SELECT active FROM emergency_flags WHERE flag_name = 'halt'"
            ).fetchone()
        except Exception:
            # Table doesn't exist yet -- create it and return not halted
            init_emergency_flags(db)
            return False

    if row is None:
        return False
    return bool(row["active"])


def set_halt(db: DatabaseManager, reason: str, set_by: str = "emergency_stop") -> None:
    """Set the emergency halt flag to active.

    Inserts or replaces the halt row with active=1, current UTC timestamp,
    the provided reason, and who set it.

    Args:
        db: DatabaseManager providing PostgreSQL connection.
        reason: Human-readable reason for the halt.
        set_by: Identifier of who/what triggered the halt.
    """
    init_emergency_flags(db)
    now = datetime.now(UTC).isoformat()
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO emergency_flags (flag_name, active, set_at, set_by, reason) "
            "VALUES ('halt', 1, %s, %s, %s) "
            "ON CONFLICT (flag_name) DO UPDATE SET "
            "active=EXCLUDED.active, set_at=EXCLUDED.set_at, "
            "set_by=EXCLUDED.set_by, reason=EXCLUDED.reason",
            [now, set_by, reason],
        )
    log.warning("halt_flag_set", reason=reason, set_by=set_by, set_at=now)


def clear_halt(db: DatabaseManager) -> None:
    """Clear the emergency halt flag (set active=0).

    Args:
        db: DatabaseManager providing PostgreSQL connection.
    """
    init_emergency_flags(db)
    with db.connection() as conn:
        conn.execute("UPDATE emergency_flags SET active = 0 WHERE flag_name = 'halt'")
    log.info("halt_flag_cleared")
