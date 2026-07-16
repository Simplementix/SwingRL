"""Versioned SQL migration runner (spec §4.1 A7b — schema_migrations ledger).

Migrations are files named V{NNN}__{slug}.sql in src/swingrl/data/migrations/.
Each file is applied in one transaction and recorded in schema_migrations.
Legacy tables continue to come from postgres_schema.init_postgres_schema();
all NEW (Stage 2.R) tables arrive only through this runner.

``DatabaseManager.connection()`` opens each connection with ``autocommit=False``
and commits on clean exit / rolls back on exception (see src/swingrl/data/db.py),
so a single ``with db.connection() as conn:`` block already is one transaction —
no extra explicit transaction wrapping is needed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from swingrl.utils.exceptions import ConfigError, DataError

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Bumped by every task that ships a new V{NNN} file. This is a *floor*: a running
# trader refuses to start only when the database is BEHIND this version (missing
# migrations — genuinely broken). A database AHEAD of this floor (a newer additive
# migration applied by a trainer-side deploy) only logs a warning — the trader must
# survive its next restart against a newer schema (A30, 2026-07-12).
EXPECTED_SCHEMA_VERSION = 2  # was 1; becomes 3 in Task 8, 4 in Task 12

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILE_RE = re.compile(r"^V(\d{3})__[a-z0-9_]+\.sql$")

_LEDGER_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    " version SMALLINT PRIMARY KEY,"
    " description TEXT NOT NULL,"
    " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
)


def _discover(migrations_dir: Path) -> list[tuple[int, str, Path]]:
    """Return sorted (version, description, path); raise DataError on bad names/gaps."""
    found: list[tuple[int, str, Path]] = []
    for path in sorted(migrations_dir.glob("V*.sql")):
        m = _FILE_RE.match(path.name)
        if not m:
            raise DataError(f"Bad migration filename: {path.name}")
        found.append((int(m.group(1)), path.stem.split("__", 1)[1], path))
    versions = [v for v, _, _ in found]
    if versions != sorted(set(versions)):
        raise DataError(f"Duplicate migration versions: {versions}")
    return found


def apply_migrations(db: DatabaseManager, migrations_dir: Path | None = None) -> int:
    """Apply all unapplied migrations in version order.

    Args:
        db: DatabaseManager providing pooled PostgreSQL connections.
        migrations_dir: Directory of V{NNN}__slug.sql files. Defaults to the
            package's own ``migrations/`` directory.

    Returns:
        Count of migrations newly applied in this call.

    Raises:
        DataError: A migration file has a bad name or duplicate version.
    """
    mdir = migrations_dir or MIGRATIONS_DIR
    applied = 0
    with db.connection() as conn:
        conn.execute(_LEDGER_DDL)
        done = {
            r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
    for version, description, path in _discover(mdir):
        if version in done:
            continue
        sql = path.read_text()
        with db.connection() as conn:  # one transaction per migration
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
                (version, description),
            )
        log.info("migration_applied", version=version, description=description)
        applied += 1
    return applied


def current_schema_version(db: DatabaseManager) -> int:
    """Return the highest applied migration version; 0 if the ledger is empty/absent."""
    with db.connection() as conn:
        conn.execute(_LEDGER_DDL)
        row = conn.execute("SELECT max(version) AS v FROM schema_migrations").fetchone()
    if row and row["v"] is not None:
        return int(row["v"])
    return 0


def assert_schema_current(db: DatabaseManager) -> None:
    """Refuse to run against a schema behind the expected floor version.

    Floor semantics (A30, 2026-07-12): raises only when the database is BEHIND
    ``EXPECTED_SCHEMA_VERSION`` (missing migrations — genuinely broken). When the
    database is AHEAD (a newer additive migration applied by a trainer-side
    deploy), this logs a warning and returns normally — a running trader must
    survive its next restart against a newer schema.

    Args:
        db: DatabaseManager providing pooled PostgreSQL connections.

    Raises:
        ConfigError: The database schema version is behind EXPECTED_SCHEMA_VERSION.
    """
    actual = current_schema_version(db)
    if actual < EXPECTED_SCHEMA_VERSION:
        log.error("schema_version_behind", expected=EXPECTED_SCHEMA_VERSION, actual=actual)
        raise ConfigError(
            f"Database schema version {actual} is behind expected {EXPECTED_SCHEMA_VERSION}; "
            "run scripts/apply_migrations.py before starting."
        )
    if actual > EXPECTED_SCHEMA_VERSION:
        log.warning("schema_version_ahead", expected=EXPECTED_SCHEMA_VERSION, actual=actual)
        return
    log.info("schema_version_ok", version=actual)
