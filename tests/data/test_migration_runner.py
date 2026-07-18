"""Migration runner tests.

D-T3.1/A7b: versioned ledger; which DDL is this database running becomes queryable.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.utils.exceptions import ConfigError, DataError

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# db_config_yaml fixture lives in tests/data/conftest.py — pytest auto-discovers
# it for this module.


@pytest.fixture
def db(tmp_path: Path, db_config_yaml: str) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager built from a tmp config whose system.database_url is DATABASE_URL."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    yield mgr
    with mgr.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS _mig_test_widgets")
        conn.execute(
            "DO $$ BEGIN "
            "IF to_regclass('public.schema_migrations') IS NOT NULL THEN "
            "DELETE FROM schema_migrations WHERE version IN (901, 902); "
            "END IF; "
            "END $$;"
        )
    DatabaseManager.reset()


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Fake migration files using reserved V9xx versions.

    V9xx never collides with the real V001/V002 (Task 1/2) once CI pre-applies
    those to this same scratch database (see conftest.py's
    ``db_with_legacy_schema`` for the real ones).
    """
    d = tmp_path / "migrations"
    d.mkdir()
    (d / "V901__widgets.sql").write_text(
        "CREATE TABLE IF NOT EXISTS _mig_test_widgets (id BIGINT PRIMARY KEY);"
    )
    (d / "V902__widgets_name.sql").write_text("ALTER TABLE _mig_test_widgets ADD COLUMN name TEXT;")
    return d


def test_apply_migrations_applies_in_order_and_records(db, migrations_dir: Path) -> None:
    """A7b: runner applies V-files in order and records each in schema_migrations."""
    from swingrl.data.migration_runner import apply_migrations

    applied = apply_migrations(db, migrations_dir=migrations_dir)
    assert applied == 2
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT version, description FROM schema_migrations "
            "WHERE version >= 901 ORDER BY version"
        ).fetchall()
    assert [r["version"] for r in rows] == [901, 902]


def test_apply_migrations_is_idempotent(db, migrations_dir: Path) -> None:
    """Re-running applies nothing new."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db, migrations_dir=migrations_dir)
    assert apply_migrations(db, migrations_dir=migrations_dir) == 0


def test_assert_schema_current_raises_on_stale(db, migrations_dir: Path, monkeypatch) -> None:
    """Merged ≠ deployed guard: stale schema (DB behind) refuses to run."""
    import swingrl.data.migration_runner as mr

    mr.apply_migrations(db, migrations_dir=migrations_dir)  # DB ends up at version 902
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 999)  # floor genuinely ahead of actual (902)
    with pytest.raises(ConfigError):
        mr.assert_schema_current(db)


def test_assert_schema_current_warns_on_ahead(
    db, migrations_dir: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Floor semantics (A30): DB ahead of EXPECTED_SCHEMA_VERSION warns, does not raise.

    A running trader must survive a newer additive migration applied by a
    trainer-side deploy — only a DB *behind* the floor is genuinely broken.
    """
    import logging

    import swingrl.data.migration_runner as mr

    mr.apply_migrations(db, migrations_dir=migrations_dir)  # DB ends up at version 902
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 1)  # floor is behind actual
    with caplog.at_level(logging.WARNING):
        mr.assert_schema_current(db)  # must not raise


def _columns(mgr: DatabaseManager, table: str) -> set[str]:
    """Column names of a public-schema table (empty set if the table is absent)."""
    with mgr.connection() as conn:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table,),
        ).fetchall()
    return {r["column_name"] for r in rows}


def test_legacy_init_schema_then_migrations_no_collision(db_with_legacy_schema) -> None:
    """CI repro: legacy ``init_schema()`` then ``apply_migrations()`` must not collide.

    The legacy memory schema and V006's §4.5 table both wanted the name
    ``pattern_presentations``. Running the CI schema-init sequence — legacy
    ``init_schema()`` FIRST, then ``apply_migrations()`` — raised
    ``psycopg.errors.DuplicateTable`` at V006 on the pre-fix schema. User ruling
    2026-07-17: the legacy table is renamed to ``pattern_presentations_legacy``; the
    §4.5 table keeps the spec name. Both must coexist with discriminating columns.
    """
    from swingrl.data.migration_runner import apply_migrations

    mgr = db_with_legacy_schema
    mgr.init_schema()  # (re)creates the legacy memory tables (the legacy pattern table)
    apply_migrations(mgr)  # V006 creates the §4.5 table alongside — must NOT DuplicateTable

    assert "consolidation_id" in _columns(mgr, "pattern_presentations_legacy")
    assert {"pattern_id", "llm_call_id"} <= _columns(mgr, "pattern_presentations")


def test_v006_renames_preexisting_legacy_pattern_presentations(db_with_legacy_schema) -> None:
    """V006 DO-block: an OLD-shape ``pattern_presentations`` present before the fix is
    renamed to ``pattern_presentations_legacy`` in place (data preserved), and the §4.5
    table is created alongside — the real in-place upgrade path for existing databases.
    """
    from swingrl.data.migration_runner import apply_migrations

    mgr = db_with_legacy_schema
    # Mimic a genuine pre-fix database: OLD-shape table present, no *_legacy yet.
    with mgr.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS pattern_presentations_legacy")
        conn.execute("DROP TABLE IF EXISTS pattern_presentations")
        conn.execute(
            "CREATE TABLE pattern_presentations ("
            " id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            " consolidation_id INTEGER,"
            " presented_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            " iteration INTEGER, env_name TEXT, request_type TEXT, advice_response TEXT)"
        )
        conn.execute(
            "INSERT INTO pattern_presentations (consolidation_id, iteration) VALUES (7, 3)"
        )

    apply_migrations(mgr)  # DO-block renames the old table, then creates the §4.5 one

    assert "consolidation_id" in _columns(mgr, "pattern_presentations_legacy")
    assert {"pattern_id", "llm_call_id"} <= _columns(mgr, "pattern_presentations")
    with mgr.connection() as conn:
        row = conn.execute(
            "SELECT consolidation_id, iteration FROM pattern_presentations_legacy"
        ).fetchone()
    assert row["consolidation_id"] == 7  # RENAME preserved the legacy rows in place
    assert row["iteration"] == 3


def test_discover_raises_on_version_gap(tmp_path: Path) -> None:
    """Fix 2 / carried minor #1: non-contiguous versions in the discovered set raise DataError.

    Gaps are relative to the discovered set, not to 1 — starting at 901 is fine
    (see migrations_dir above); a missing middle number (here: no V902) is not.
    """
    import swingrl.data.migration_runner as mr

    d = tmp_path / "migrations"
    d.mkdir()
    (d / "V901__widgets.sql").write_text(
        "CREATE TABLE IF NOT EXISTS _mig_test_widgets (id BIGINT PRIMARY KEY);"
    )
    (d / "V903__widgets_extra.sql").write_text(
        "ALTER TABLE _mig_test_widgets ADD COLUMN extra TEXT;"
    )
    with pytest.raises(DataError):
        mr._discover(d)
