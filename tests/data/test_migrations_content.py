"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0.

V002: identity-spine subset (training_runs, models, ensemble_weight_history) —
spine UNIQUE constraint makes duplicate runs impossible; retries are new attempt rows.
Also carries insert-level negative tests for V002's ``models``/``ensemble_weight_history``
load-bearing constraints (Task 8 carry-forward obligation) — V002 shipped with these
constraints proven only at DDL level (schema inspection), never exercised by an actual
rejected INSERT.

V003: trade-time tables (§4.7 + A27) — inference_cycles/cycle_algo_proposals/
trades.cycle_id/fill_quality/calendar_events/event_outcomes.

V004: coach records (§4.4) — llm_calls (A15 identity CHECK matrix),
intent_records (per-lever CHECK + horizon_spec JSONB NOT NULL), intent_applications,
intent_verdicts (A16 UNIQUE(intent_id, grader_version)); ensemble_weight_history.intent_id
FK backfilled; A14 volume cap via two partial UNIQUE indexes (≤1 MT_commentary intent
per inference cycle).

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


def test_v004_schema_version_is_4(db_with_legacy_schema) -> None:
    """V004 is the newest applied migration after this task."""
    from swingrl.data.migration_runner import apply_migrations, current_schema_version

    apply_migrations(db_with_legacy_schema)
    assert current_schema_version(db_with_legacy_schema) == 4


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


# ---------------------------------------------------------------------------
# V004: coach records (§4.4) — llm_calls / intent_records / intent_applications /
# intent_verdicts + ewh.intent_id FK + A14 volume-cap partial UNIQUE indexes.
# ---------------------------------------------------------------------------


def _insert_inference_cycle(db: DatabaseManager, *, environment: str = "equity") -> int:
    """Insert a minimal inference_cycles row and return its cycle_id (A15 FK target)."""
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO inference_cycles (environment, mode, cycle_ts)"
                " VALUES (%s, 'paper', now()) RETURNING cycle_id",
                (environment,),
            ).fetchone()["cycle_id"]
        )


def _insert_commentary_chain(
    db: DatabaseManager, *, environment: str = "equity", algorithm: str = "ppo"
) -> tuple[int, int, int]:
    """Insert cycle -> trade_commentary llm_call -> MT_commentary intent.

    Returns (cycle_id, llm_call_id, intent_id). Mirrors the memory service's
    atomic write: one shadow MT_commentary intent per inference cycle.
    """
    cycle_id = _insert_inference_cycle(db, environment=environment)
    with db.connection() as conn:
        llm_call_id = int(
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, cycle_id, provider, model,"
                " prompt_version) VALUES ('meta_trader', 'trade_commentary', %s, 'cerebras',"
                " 'qwen-3', 'mt-commentary-v0') RETURNING llm_call_id",
                (cycle_id,),
            ).fetchone()["llm_call_id"]
        )
        intent_id = int(
            conn.execute(
                "INSERT INTO intent_records"
                " (llm_call_id, coach, lever, mode, environment, algorithm, iteration_number,"
                "  evidence, proposal, bet_metric, bet_direction, bet_baseline_value, horizon_spec)"
                " VALUES (%s, 'meta_trader', 'MT_commentary', 'shadow', %s, %s, 0,"
                "  '{}'::jsonb, '{}'::jsonb, 'cycle_pnl_frac', 'up', 0.0,"
                '  \'{"type": "wall_clock_hours", "hours": 24}\'::jsonb)'
                " RETURNING intent_id",
                (llm_call_id, environment, algorithm),
            ).fetchone()["intent_id"]
        )
    return cycle_id, llm_call_id, intent_id


def test_v004_llm_calls_epoch_advice_null_run_pk_rejected(db_with_legacy_schema) -> None:
    """A15 identity CHECK: epoch_advice requires run_pk NOT NULL (F3 fix)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, provider, model, prompt_version)"
                " VALUES ('meta_trainer', 'epoch_advice', 'cerebras', 'qwen-3', 'epoch-v1')"
            )


def test_v004_llm_calls_trade_commentary_with_cycle_id_accepted(db_with_legacy_schema) -> None:
    """A15 identity CHECK: trade_commentary with cycle_id set is accepted."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    cycle_id = _insert_inference_cycle(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        llm_call_id = conn.execute(
            "INSERT INTO llm_calls (coach, call_type, cycle_id, provider, model, prompt_version)"
            " VALUES ('meta_trader', 'trade_commentary', %s, 'cerebras', 'qwen-3',"
            " 'mt-commentary-v0') RETURNING llm_call_id",
            (cycle_id,),
        ).fetchone()["llm_call_id"]
    assert llm_call_id is not None


def test_v004_llm_calls_trade_commentary_null_cycle_id_rejected(db_with_legacy_schema) -> None:
    """A15 identity CHECK: trade_commentary requires cycle_id NOT NULL (no timestamp inference)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, provider, model, prompt_version)"
                " VALUES ('meta_trader', 'trade_commentary', 'cerebras', 'qwen-3', 'mt-commentary-v0')"
            )


def test_v004_intent_verdicts_unique_intent_grader_version(db_with_legacy_schema) -> None:
    """A16: UNIQUE(intent_id, grader_version) — regrades are new rows, not UPDATEs."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _cycle_id, _llm_call_id, intent_id = _insert_commentary_chain(db_with_legacy_schema)
    ins = (
        "INSERT INTO intent_verdicts (intent_id, grader_version, actual_value, direction_match)"
        " VALUES (%s, 1, 0.01, true)"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (intent_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (intent_id,))
    # A different grader_version IS allowed (a genuine regrade)
    with db_with_legacy_schema.connection() as conn:
        conn.execute(
            "INSERT INTO intent_verdicts (intent_id, grader_version, actual_value, direction_match)"
            " VALUES (%s, 2, 0.02, false)",
            (intent_id,),
        )


def test_v004_second_mt_commentary_intent_same_llm_call_rejected(db_with_legacy_schema) -> None:
    """A14 volume cap: partial UNIQUE(llm_call_id) WHERE lever='MT_commentary' — one intent/call."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _cycle_id, llm_call_id, _intent_id = _insert_commentary_chain(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO intent_records"
                " (llm_call_id, coach, lever, mode, environment, algorithm, iteration_number,"
                "  evidence, proposal, bet_metric, bet_direction, bet_baseline_value, horizon_spec)"
                " VALUES (%s, 'meta_trader', 'MT_commentary', 'shadow', 'equity', 'ppo', 0,"
                "  '{}'::jsonb, '{}'::jsonb, 'cycle_pnl_frac', 'up', 0.0,"
                '  \'{"type": "wall_clock_hours", "hours": 24}\'::jsonb)',
                (llm_call_id,),
            )


def test_v004_second_trade_commentary_call_same_cycle_rejected(db_with_legacy_schema) -> None:
    """A14 volume cap: partial UNIQUE(cycle_id) WHERE call_type='trade_commentary' — one call/cycle.

    Combined with the intent-level partial index, this bounds MT commentary at
    <=1 intent record per inference cycle (D-T3.19).
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    cycle_id = _insert_inference_cycle(db_with_legacy_schema)
    ins = (
        "INSERT INTO llm_calls (coach, call_type, cycle_id, provider, model, prompt_version)"
        " VALUES ('meta_trader', 'trade_commentary', %s, 'cerebras', 'qwen-3', 'mt-commentary-v0')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (cycle_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (cycle_id,))


def test_v004_ensemble_weight_history_intent_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """V004 backfills ensemble_weight_history.intent_id FK -> intent_records."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _insert_training_run_and_model(db_with_legacy_schema, "ewh-intent-fk-model", algorithm="ppo")
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO ensemble_weight_history (model_id, weight_frac, set_by, intent_id)"
                " VALUES ('ewh-intent-fk-model', 0.5, 'meta_trader', 999999)"
            )


def test_v004_intent_applications_unique_intent_id(db_with_legacy_schema) -> None:
    """A13: intent_applications sidecar — one application row per intent (UNIQUE intent_id)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _cycle_id, _llm_call_id, intent_id = _insert_commentary_chain(db_with_legacy_schema)
    ins = "INSERT INTO intent_applications (intent_id, applied) VALUES (%s, '{}'::jsonb)"
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (intent_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (intent_id,))


# ---------------------------------------------------------------------------
# V005 (Task 8): §4.3 training-record tables — epoch_snapshots / fold_results /
# season_results / backtest_trades.
# ---------------------------------------------------------------------------


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
