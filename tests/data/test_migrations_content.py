"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0.

V002: identity-spine subset (training_runs, models, ensemble_weight_history) —
spine UNIQUE constraint makes duplicate runs impossible; retries are new attempt rows.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from swingrl.data.db import DatabaseManager

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


# db_with_legacy_schema fixture lives in tests/data/conftest.py (shared with
# test_bootstrap_era0_models.py) — pytest auto-discovers it for this module.


def test_v001_era0_bootstrap(db_with_legacy_schema: DatabaseManager) -> None:
    """D-T3.4/A7: era 0 + gate v0 rows exist; legacy result rows stamped era 0."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        era = conn.execute("SELECT * FROM eras WHERE era_id = 0").fetchone()
        assert era is not None and era["first_iteration"] == 0
        gates = conn.execute(
            "SELECT gate_type FROM gate_versions WHERE version_number = 0 ORDER BY gate_type"
        ).fetchall()
        assert [g["gate_type"] for g in gates] == ["ensemble", "per_fold"]
        stamped = conn.execute(
            "SELECT count(*) AS n FROM backtest_results WHERE era_id = 0"
        ).fetchone()
        total = conn.execute("SELECT count(*) AS n FROM backtest_results").fetchone()
        assert stamped["n"] == total["n"]


def test_v002_spine_unique(db_with_legacy_schema) -> None:
    """D-T3.1: duplicates impossible; retries are new attempt rows.

    Each insert attempt uses its own ``connection()`` block rather than sharing
    one transaction: ``DatabaseManager.connection()`` only rolls back when the
    exception propagates out of the ``with`` block (see db.py — commit on clean
    exit, rollback on exception exit). Catching the UniqueViolation with
    ``pytest.raises`` *inside* a single shared block would leave that block's
    transaction aborted at the Postgres level, so the following insert would
    fail with ``InFailedSqlTransaction`` rather than succeeding — verified
    empirically against the live scratch DB before writing this test this way.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO training_runs (iteration_number, environment, algorithm, fold_number,"
        " run_type, seed, attempt, status, era_id, code_version, data_fingerprint)"
        " VALUES (5, 'equity', 'ppo', 0, 'reference', 42, %s, 'completed', 0, 'abc123', 'fp1')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (1,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (1,))
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (2,))  # new attempt OK
