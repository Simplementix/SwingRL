"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0.

V002: identity-spine subset (training_runs, models, ensemble_weight_history) —
spine UNIQUE constraint makes duplicate runs impossible; retries are new attempt rows.

V005 (Task 8): §4.3 training-record tables — epoch_snapshots, fold_results,
season_results, backtest_trades.
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


def _insert_training_run(db: DatabaseManager, *, attempt: int = 1) -> int:
    """Insert a minimal V002 training_runs row (era 0) and return its run_pk.

    Shared helper for the V005 content tests below — every §4.3 training-record
    table hangs off a training_runs row via run_pk.
    """
    ins = (
        "INSERT INTO training_runs (iteration_number, environment, algorithm, fold_number,"
        " run_type, seed, attempt, status, era_id, code_version, data_fingerprint)"
        " VALUES (5, 'equity', 'ppo', 0, 'season', 42, %s, 'completed', 0, 'abc123', 'fp1')"
        " RETURNING run_pk"
    )
    with db.connection() as conn:
        return int(conn.execute(ins, (attempt,)).fetchone()["run_pk"])


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


def test_v005_fold_results_run_pk_unique(db_with_legacy_schema) -> None:
    """Task 8 / §4.3: fold_results.run_pk UNIQUE — one box score per run.

    The backtest_results 9-duplicate class (read-time dedup) becomes structurally
    unrepeatable, mirroring V002's training_runs UNIQUE-constraint proof above.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    ins = (
        "INSERT INTO fold_results (run_pk, era_id, gate_version_id, seed,"
        " fold_start_ts, fold_end_ts) VALUES (%s, 0, 0, 42, now(), now())"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (run_pk,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (run_pk,))


def test_v005_season_results_requires_coach_config(db_with_legacy_schema) -> None:
    """Task 8 / §4.3: season_results.coach_config is NOT NULL JSONB.

    coach_config is the staircase stamp (D-T2.6) — without it a season row can
    never be attributed to a coach configuration.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO season_results (iteration_number, environment, scope,"
        " era_id, gate_version_per_fold, gate_version_ensemble)"
        " VALUES (5, 'equity', 'ppo', 0, 0, 1)"
    )
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins)


def test_v005_season_results_scope_check(db_with_legacy_schema) -> None:
    """Task 8 / §4.3: season_results.scope CHECK (ppo, a2c, sac, ensemble)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO season_results (iteration_number, environment, scope,"
        " era_id, gate_version_per_fold, gate_version_ensemble, coach_config)"
        " VALUES (5, 'equity', 'not_a_scope', 0, 0, 1, '{\"schema_version\": 1}')"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins)


def test_v005_season_results_unique_iteration_env_scope_version(
    db_with_legacy_schema,
) -> None:
    """Task 8 / §4.3: UNIQUE(iteration_number, environment, scope, result_version).

    Recomputes/re-runs are new result_version rows, never UPDATEs (A10).
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO season_results (iteration_number, environment, scope,"
        " result_version, era_id, gate_version_per_fold, gate_version_ensemble,"
        " coach_config)"
        " VALUES (5, 'equity', 'ppo', %s, 0, 0, 1, '{\"schema_version\": 1}')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (1,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (1,))
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (2,))  # new result_version OK


def test_v005_backtest_trades_unique_run_bar_symbol(db_with_legacy_schema) -> None:
    """Task 8 / §4.3: backtest_trades UNIQUE(run_pk, bar_ts, symbol) — the physical ceiling."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    ins = (
        "INSERT INTO backtest_trades (run_pk, bar_ts, symbol, side)"
        " VALUES (%s, '2026-01-05T00:00:00Z', 'SPY', 'buy')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (run_pk,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (run_pk,))


def test_v005_epoch_snapshots_requires_learner_metrics(db_with_legacy_schema) -> None:
    """Task 8 / §4.3: epoch_snapshots.learner_metrics is NOT NULL JSONB.

    Per-algo key contract (PPO kl/clip; SAC actor/critic loss, ent_coef) — kills
    the PPO-only-keys bug (Epoch logger bug precedent).
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    ins = "INSERT INTO epoch_snapshots (run_pk) VALUES (%s)"
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (run_pk,))


def test_v005_schema_version_is_5(db_with_legacy_schema) -> None:
    """Task 8: SELECT max(version) FROM schema_migrations == 5 after apply_migrations."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute("SELECT max(version) AS v FROM schema_migrations").fetchone()
    assert row["v"] == 5
