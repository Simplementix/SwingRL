"""C5-MIG-01: reward_adjustments attribution columns exist after migration.

Two test classes:
  - TestAttributionColumnsSchemaInit: verifies init_postgres_schema() creates
    all 6 new attribution columns (live-DB-gated via DATABASE_URL).
  - TestAttributionMigrationIdempotency: verifies the migration script is
    safe to run twice (live-DB-gated via DATABASE_URL).

Static tests (no DB required) are in the companion
``TestAttributionColumnsMigrationList`` class below.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the migration helper the same way add_cps_columns is imported in
# tests/data/test_iteration_results_extension.py — i.e., via sys.path insert
# so the ``scripts/`` tree is importable without any package install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent


def _load_migration():  # type: ignore[return]
    """Lazily import add_attribution_columns from scripts/migrations/."""
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.migrations import add_attribution_columns  # noqa: E402

    return add_attribution_columns


_NEW_COLS = {
    "fold_number",
    "iteration_number",
    "advice_id",
    "fold_cps_v1_before",
    "fold_cps_v1_after",
    "advice_was_effective",
}

# ---------------------------------------------------------------------------
# Static test — no DB required
# ---------------------------------------------------------------------------


class TestAttributionColumnsMigrationList:
    """C5-MIG-01 (static): migration script declares exactly the 6 attribution columns."""

    def test_new_columns_in_migration_script(self) -> None:
        """C5-MIG-01: _NEW_COLUMNS in add_attribution_columns matches expected set."""
        mod = _load_migration()
        col_names = {name for name, _ in mod._NEW_COLUMNS}
        assert col_names == _NEW_COLS, f"Unexpected columns in migration: {col_names ^ _NEW_COLS}"

    def test_new_columns_count(self) -> None:
        """C5-MIG-01: exactly 6 attribution columns are declared."""
        mod = _load_migration()
        assert len(mod._NEW_COLUMNS) == 6


# ---------------------------------------------------------------------------
# Live-DB tests — skip when DATABASE_URL is absent
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_conn():  # type: ignore[return]
    """Connect to pg16; skip if DATABASE_URL is not set."""
    import psycopg

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set; skipping live Postgres test")
    return psycopg.connect(db_url)


class TestAttributionColumnsSchemaInit:
    """C5-MIG-01: schema init creates the 6 attribution columns in reward_adjustments."""

    def test_attribution_columns_present_after_schema_init(self, pg_conn) -> None:
        """C5-MIG-01: init_postgres_schema() creates the 6 attribution columns."""

        from swingrl.data.postgres_schema import init_postgres_schema

        init_postgres_schema(pg_conn)
        pg_conn.commit()

        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'reward_adjustments' AND table_schema = 'public'"
            )
            present = {row[0] for row in cur.fetchall()}

        assert _NEW_COLS <= present, (
            f"Missing attribution columns after schema init: {_NEW_COLS - present}"
        )


class TestAttributionMigrationIdempotency:
    """C5-MIG-01: running the migration twice is a no-op the second time."""

    def test_migration_script_idempotent(self, pg_conn) -> None:
        """C5-MIG-01: migrate(url) twice — second call has added=0, already_present=6."""
        mod = _load_migration()
        db_url = os.environ.get("DATABASE_URL", "")

        # First call: adds whatever is missing (0 or 6 depending on state).
        first = mod.migrate(db_url)
        # Second call: must be a complete no-op.
        second = mod.migrate(db_url)

        assert first["added"] + first["already_present"] == 6, f"First run total != 6: {first}"
        assert second["added"] == 0, f"Second run still added columns: {second}"
        assert second["already_present"] == 6, f"Second run already_present != 6: {second}"
