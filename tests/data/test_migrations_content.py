"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0.

V002: identity-spine subset (training_runs, models, ensemble_weight_history) —
spine UNIQUE constraint makes duplicate runs impossible; retries are new attempt rows.
Also carries insert-level negative tests for V002's ``models``/``ensemble_weight_history``
load-bearing constraints (Task 8 carry-forward obligation) — V002 shipped with these
constraints proven only at DDL level (schema inspection), never exercised by an actual
rejected INSERT.

V003: trade-time tables (§4.7 + A27) — inference_cycles/cycle_algo_proposals/
trades.cycle_id/fill_quality/calendar_events/event_outcomes.
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


def _insert_training_run_and_model(
    db: DatabaseManager,
    model_id: str,
    *,
    algorithm: str = "ppo",
    status: str = "active",
) -> int:
    """Insert a minimal valid training_runs row + models row; return run_pk.

    Shared FK-satisfying setup for V002/V003 tests that need a real
    ``models.model_id`` to reference. ``algorithm`` varies the training_runs
    UNIQUE key (iteration_number, environment, algorithm, fold_number, run_type,
    attempt) so a single test can call this more than once without colliding.
    """
    with db.connection() as conn:
        run_pk = conn.execute(
            "INSERT INTO training_runs (iteration_number, environment, algorithm,"
            " fold_number, run_type, seed, attempt, status, era_id, code_version,"
            " data_fingerprint)"
            " VALUES (0, 'equity', %s, -1, 'final_train', 42, 1, 'completed', 0,"
            " 'abc123', 'fp1') RETURNING run_pk",
            (algorithm,),
        ).fetchone()["run_pk"]
        conn.execute(
            "INSERT INTO models (model_id, run_pk, artifact_path, vecnormalize_path, status)"
            " VALUES (%s, %s, 'models/p.zip', 'models/v.pkl', %s)",
            (model_id, run_pk, status),
        )
    return run_pk


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


def test_v002_models_model_id_pk_rejects_duplicate(db_with_legacy_schema) -> None:
    """models.model_id PRIMARY KEY: proven only at DDL level until now (carry-forward)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    run_pk = _insert_training_run_and_model(db_with_legacy_schema, "dup-model", algorithm="ppo")
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO models (model_id, run_pk, artifact_path, vecnormalize_path, status)"
                " VALUES ('dup-model', %s, 'models/p2.zip', 'models/v2.pkl', 'shadow')",
                (run_pk,),
            )


def test_v002_models_run_pk_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """models.run_pk FK -> training_runs: proven only at DDL level until now (carry-forward)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO models (model_id, run_pk, artifact_path, vecnormalize_path, status)"
                " VALUES ('orphan-model', 999999, 'models/p.zip', 'models/v.pkl', 'active')"
            )


def test_v002_models_status_check_rejects_invalid(db_with_legacy_schema) -> None:
    """models.status CHECK ('active','shadow','archived'): proven only at DDL level (carry-forward)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        run_pk = conn.execute(
            "INSERT INTO training_runs (iteration_number, environment, algorithm, fold_number,"
            " run_type, seed, attempt, status, era_id, code_version, data_fingerprint)"
            " VALUES (0, 'equity', 'a2c', -1, 'final_train', 43, 1, 'completed', 0,"
            " 'abc123', 'fp1') RETURNING run_pk"
        ).fetchone()["run_pk"]
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO models (model_id, run_pk, artifact_path, vecnormalize_path, status)"
                " VALUES ('bad-status-model', %s, 'models/p.zip', 'models/v.pkl', 'bogus')",
                (run_pk,),
            )


def test_v002_ensemble_weight_history_model_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """ensemble_weight_history.model_id FK -> models: DDL-only until now (carry-forward)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO ensemble_weight_history (model_id, weight_frac, set_by)"
                " VALUES ('nonexistent-model', 0.5, 'training')"
            )


def test_v002_ensemble_weight_history_set_by_check_rejects_invalid(
    db_with_legacy_schema,
) -> None:
    """ensemble_weight_history.set_by CHECK: DDL-only until now (carry-forward)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _insert_training_run_and_model(db_with_legacy_schema, "ewh-check-model", algorithm="sac")
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO ensemble_weight_history (model_id, weight_frac, set_by)"
                " VALUES ('ewh-check-model', 0.5, 'bogus_setter')"
            )


def test_v003_schema_version_is_3(db_with_legacy_schema) -> None:
    """V003 is the newest applied migration after this task."""
    from swingrl.data.migration_runner import apply_migrations, current_schema_version

    apply_migrations(db_with_legacy_schema)
    assert current_schema_version(db_with_legacy_schema) == 3


def test_v003_inference_cycles_has_turbulence_column(db_with_legacy_schema) -> None:
    """A27: turbulence is the decision-time sensor value, read out before F1b zeroing."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name = 'inference_cycles'"
        ).fetchall()
    assert "turbulence" in {c["column_name"] for c in cols}


def test_v003_trades_has_cycle_id_referencing_inference_cycles(db_with_legacy_schema) -> None:
    """D-T3.13: trades.cycle_id completes the proposal -> blend -> fill chain."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'trades'"
        ).fetchall()
    assert "cycle_id" in {c["column_name"] for c in cols}

    with db_with_legacy_schema.connection() as conn:
        cycle_id = conn.execute(
            "INSERT INTO inference_cycles (environment, mode, cycle_ts)"
            " VALUES ('equity', 'paper', now()) RETURNING cycle_id"
        ).fetchone()["cycle_id"]
        conn.execute(
            "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, price, environment,"
            " cycle_id)"
            " VALUES ('trade-cycle-1', now(), 'SPY', 'buy', 1.0, 500.0, 'equity', %s)",
            (cycle_id,),
        )
        row = conn.execute(
            "SELECT cycle_id FROM trades WHERE trade_id = 'trade-cycle-1'"
        ).fetchone()
    assert row["cycle_id"] == cycle_id


def test_v003_cycle_algo_proposals_unique_cycle_model_rejects_duplicate(
    db_with_legacy_schema,
) -> None:
    """D-T3.13: UNIQUE(cycle_id, model_id) — one proposal per algo per cycle."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _insert_training_run_and_model(db_with_legacy_schema, "proposal-model", algorithm="ppo")
    with db_with_legacy_schema.connection() as conn:
        cycle_id = conn.execute(
            "INSERT INTO inference_cycles (environment, mode, cycle_ts)"
            " VALUES ('equity', 'paper', now()) RETURNING cycle_id"
        ).fetchone()["cycle_id"]
    ins = (
        "INSERT INTO cycle_algo_proposals"
        " (cycle_id, model_id, algorithm, proposed_actions, weight_in_blend_frac)"
        " VALUES (%s, 'proposal-model', 'ppo', '{\"schema_version\": 1, \"raw\": {}}', 0.5)"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (cycle_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (cycle_id,))


def test_v003_calendar_events_nulls_not_distinct_rejects_duplicate_macro_row(
    db_with_legacy_schema,
) -> None:
    """A27 editorial rider: UNIQUE NULLS NOT DISTINCT — macro rows have NULL symbol,
    and a plain UNIQUE never treats two NULLs as equal (would admit duplicate macro rows)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO calendar_events"
        " (event_type, symbol, scheduled_at, window_start, window_end, importance, source)"
        " VALUES ('fomc', NULL, '2026-07-16T18:00:00Z', '2026-07-16T17:00:00Z',"
        " '2026-07-16T19:00:00Z', 'high', 'fred')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins)
