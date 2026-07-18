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
    """V004 lands in the ledger (was newest; V005 now raises the ceiling to 5).

    The newest-version invariant moved to ``test_v005_schema_version_is_5`` when
    Track B Task 8 shipped V005 — this asserts V004 was applied, not that it is the
    maximum, so it stays green as later migrations extend the ledger.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM schema_migrations WHERE version = 4"
        ).fetchone()
    assert row["n"] == 1


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
    """V005 lands in the ledger (was newest; V006 now raises the ceiling to 6).

    The newest-version invariant moved to ``test_v006_schema_version_is_6`` when
    Track B Task 9 shipped V006 — this asserts V005 was applied, not that it is the
    maximum, so it stays green as later migrations extend the ledger.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM schema_migrations WHERE version = 5"
        ).fetchone()
    assert row["n"] == 1


# ---------------------------------------------------------------------------
# V006 (Task 9): §4.5 patterns family — patterns / pattern_sources /
# pattern_links / pattern_presentations (the lineage DAG). Mandatory pattern_id +
# llm_call_id FKs on presentations kill the NULL-iteration failure class.
# ---------------------------------------------------------------------------


def _insert_pattern(
    db: DatabaseManager,
    *,
    era_id: int = 0,
    stage: int = 1,
    environment: str = "equity",
    category: str = "trade_shy",
    status: str = "active",
) -> int:
    """Insert a minimal valid patterns row (era 0) and return its pattern_id.

    ``claim`` (JSONB NOT NULL), ``era_id`` (FK NOT NULL) and ``status`` (NOT NULL)
    are the only columns the DDL forces a writer to supply; ``qa_passed`` and the
    two script-maintained counters carry safe defaults (false / 0).
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO patterns (created_iteration, environment, stage, era_id,"
                " category, claim, status)"
                " VALUES (5, %s, %s, %s, %s, '{\"schema_version\": 1}'::jsonb, %s)"
                " RETURNING pattern_id",
                (environment, stage, era_id, category, status),
            ).fetchone()["pattern_id"]
        )


def _insert_consolidator_call(db: DatabaseManager) -> int:
    """Insert a valid consolidate_stage1 llm_call and return its llm_call_id.

    The consolidator is the natural producer of pattern presentations; A15's
    identity matrix requires iteration_number + environment for that call_type.
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, iteration_number, environment,"
                " provider, model, prompt_version)"
                " VALUES ('consolidator', 'consolidate_stage1', 5, 'equity', 'cerebras',"
                " 'qwen-3', 'consolidate-v0') RETURNING llm_call_id"
            ).fetchone()["llm_call_id"]
        )


def test_v006_pattern_presentations_null_llm_call_rejected(db_with_legacy_schema) -> None:
    """§4.5: pattern_presentations.llm_call_id NOT NULL — the NULL-iteration class dies."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    pattern_id = _insert_pattern(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_presentations (pattern_id, llm_call_id) VALUES (%s, NULL)",
                (pattern_id,),
            )


def test_v006_pattern_presentations_llm_call_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """§4.5: pattern_presentations.llm_call_id FK -> llm_calls (identity inherited)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    pattern_id = _insert_pattern(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_presentations (pattern_id, llm_call_id) VALUES (%s, 999999)",
                (pattern_id,),
            )


def test_v006_pattern_presentations_valid_accepted(db_with_legacy_schema) -> None:
    """§4.5: a presentation with both mandatory FKs set is accepted (happy path)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    pattern_id = _insert_pattern(db_with_legacy_schema)
    llm_call_id = _insert_consolidator_call(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row_id = conn.execute(
            "INSERT INTO pattern_presentations (pattern_id, llm_call_id) VALUES (%s, %s)"
            " RETURNING id",
            (pattern_id, llm_call_id),
        ).fetchone()["id"]
    assert row_id is not None


def test_v006_pattern_links_self_link_rejected(db_with_legacy_schema) -> None:
    """§4.5: pattern_links CHECK(parent <> child) — a pattern can't be its own parent."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    pattern_id = _insert_pattern(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_links (parent_pattern_id, child_pattern_id, link_type)"
                " VALUES (%s, %s, 'merged_into')",
                (pattern_id, pattern_id),
            )


def test_v006_pattern_links_link_type_check(db_with_legacy_schema) -> None:
    """§4.5: pattern_links.link_type CHECK (merged_into, split_into, refined_into)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    parent = _insert_pattern(db_with_legacy_schema, environment="equity")
    child = _insert_pattern(db_with_legacy_schema, environment="crypto")
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_links (parent_pattern_id, child_pattern_id, link_type)"
                " VALUES (%s, %s, 'bogus_link')",
                (parent, child),
            )


def test_v006_pattern_links_pk_rejects_duplicate(db_with_legacy_schema) -> None:
    """§4.5: pattern_links PK(parent, child) — one edge per pair; a valid link is accepted first."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    parent = _insert_pattern(db_with_legacy_schema, environment="equity")
    child = _insert_pattern(db_with_legacy_schema, environment="crypto")
    ins = (
        "INSERT INTO pattern_links (parent_pattern_id, child_pattern_id, link_type)"
        " VALUES (%s, %s, 'refined_into')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (parent, child))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (parent, child))


def test_v006_pattern_sources_source_table_check(db_with_legacy_schema) -> None:
    """§4.5: pattern_sources.source_table CHECK — provenance points only at structured records.

    A structured-record table (fold_results) is accepted; retired raw ``memories``
    — outside the allowlist — bounces off the CHECK. ``source_id`` is polymorphic
    (no FK), so the accepted row need not reference a real fold_results id.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    pattern_id = _insert_pattern(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        conn.execute(
            "INSERT INTO pattern_sources (pattern_id, source_table, source_id)"
            " VALUES (%s, 'fold_results', 1)",
            (pattern_id,),
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_sources (pattern_id, source_table, source_id)"
                " VALUES (%s, 'memories', 2)",
                (pattern_id,),
            )


def test_v006_pattern_sources_pattern_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """§4.5: pattern_sources.pattern_id FK -> patterns."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO pattern_sources (pattern_id, source_table, source_id)"
                " VALUES (999999, 'fold_results', 1)"
            )


def test_v006_patterns_requires_claim(db_with_legacy_schema) -> None:
    """§4.5: patterns.claim is NOT NULL JSONB — no narrative-only records (§4.2 rule 3)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO patterns (created_iteration, environment, stage, era_id,"
                " category, status) VALUES (5, 'equity', 1, 0, 'trade_shy', 'active')"
            )


def test_v006_patterns_status_check(db_with_legacy_schema) -> None:
    """§4.5: patterns.status CHECK (active, conflicted, superseded, retired)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO patterns (created_iteration, environment, stage, era_id,"
                " category, claim, status)"
                " VALUES (5, 'equity', 1, 0, 'trade_shy', '{\"schema_version\": 1}', 'bogus')"
            )


def test_v006_patterns_stage_check(db_with_legacy_schema) -> None:
    """§4.5: patterns.stage CHECK (1, 2) — stage-1 per-env vs stage-2 cross-env."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO patterns (created_iteration, environment, stage, era_id,"
                " category, claim, status)"
                " VALUES (5, 'equity', 3, 0, 'trade_shy', '{\"schema_version\": 1}', 'active')"
            )


def test_v006_schema_version_is_6(db_with_legacy_schema) -> None:
    """V006 lands in the ledger (was newest; V007 now raises the ceiling to 7).

    The newest-version invariant moved to ``test_v007_schema_version_is_7`` when
    Track B Task 10 shipped V007 — this asserts V006 was applied, not that it is the
    maximum, so it stays green as later migrations extend the ledger.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM schema_migrations WHERE version = 6"
        ).fetchone()
    assert row["n"] == 1


# ---------------------------------------------------------------------------
# V007 (Task 10): §4.6 harness records — harness_experiments (pre-registration:
# lever/stage/environment/algorithm/fold + scripted pull_spec), harness_experiment_runs
# (Stage 1 mechanics: one row per arm/seed-pair run — "majority of seed-pairs agree"
# computable straight from keys), harness_replays (Stage 2 film-room: a scripted quiz
# graded against the coach's actual llm_calls response; mandatory llm_call_id FK mirrors
# V006's pattern_presentations never-orphaned-record discipline).
# ---------------------------------------------------------------------------


def _insert_harness_experiment(
    db: DatabaseManager,
    *,
    lever: str = "L1_reward_weights",
    stage: int = 1,
    environment: str = "equity",
    algorithm: str = "ppo",
    fold_number: int = 0,
    fold_role: str = "neutral",
) -> int:
    """Insert a minimal valid harness_experiments row (pre-registration) and return its id.

    ``pull_spec`` (JSONB NOT NULL) and ``min_run_length_steps`` are the only columns the
    DDL forces beyond the identity/classification fields — both are pre-registration data,
    written before any run starts.
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO harness_experiments (lever, stage, environment, algorithm,"
                " fold_number, fold_role, pull_spec, min_run_length_steps)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, 100000) RETURNING experiment_id",
                (
                    lever,
                    stage,
                    environment,
                    algorithm,
                    fold_number,
                    fold_role,
                    '{"direction": "up", "magnitude_frac": 0.1,'
                    ' "expected": {"metric": "oos_sharpe_annualized", "direction": "up"}}',
                ),
            ).fetchone()["experiment_id"]
        )


def _insert_harness_replay_call(db: DatabaseManager) -> int:
    """Insert a valid harness_replay llm_call and return its llm_call_id.

    A15's identity CHECK matrix marks ``harness_replay`` ``true`` unconditionally — no
    run_pk/cycle_id/iteration_number/environment/algorithm required (linkage is via
    ``harness_replays`` instead, per §4.6).
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, provider, model, prompt_version)"
                " VALUES ('meta_trainer', 'harness_replay', 'cerebras', 'qwen-3',"
                " 'harness-replay-v0') RETURNING llm_call_id"
            ).fetchone()["llm_call_id"]
        )


def test_v007_harness_experiments_requires_pull_spec(db_with_legacy_schema) -> None:
    """§4.6: harness_experiments.pull_spec is NOT NULL JSONB — pre-registration is mandatory.

    Scripted pull direction/magnitude + expected:{metric,direction} must exist before any
    run starts; a row with no pull_spec cannot be a pre-registered experiment.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiments (lever, stage, environment, algorithm,"
                " fold_number, fold_role, min_run_length_steps)"
                " VALUES ('L1_reward_weights', 1, 'equity', 'ppo', 0, 'neutral', 100000)"
            )


def test_v007_harness_experiments_lever_check(db_with_legacy_schema) -> None:
    """§4.6: harness_experiments.lever CHECK — same enum as intent_records.lever (§4.4)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiments (lever, stage, environment, algorithm,"
                " fold_number, fold_role, pull_spec, min_run_length_steps)"
                " VALUES ('bogus_lever', 1, 'equity', 'ppo', 0, 'neutral',"
                " '{\"schema_version\": 1}', 100000)"
            )


def test_v007_harness_experiments_stage_check(db_with_legacy_schema) -> None:
    """§4.6: harness_experiments.stage CHECK (1, 2) — Stage 1 mechanics vs Stage 2 judgment."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiments (lever, stage, environment, algorithm,"
                " fold_number, fold_role, pull_spec, min_run_length_steps)"
                " VALUES ('L1_reward_weights', 3, 'equity', 'ppo', 0, 'neutral',"
                " '{\"schema_version\": 1}', 100000)"
            )


def test_v007_harness_experiment_runs_arm_check(db_with_legacy_schema) -> None:
    """§4.6: harness_experiment_runs.arm CHECK (pull, control)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
                " VALUES (%s, %s, 'bogus_arm', 1)",
                (experiment_id, run_pk),
            )


def test_v007_harness_experiment_runs_run_pk_fk_rejects_nonexistent(
    db_with_legacy_schema,
) -> None:
    """§4.6: harness_experiment_runs.run_pk FK -> training_runs (consumes V002 spine)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
                " VALUES (%s, 999999, 'pull', 1)",
                (experiment_id,),
            )


def test_v007_harness_experiment_runs_experiment_id_fk_rejects_nonexistent(
    db_with_legacy_schema,
) -> None:
    """§4.6: harness_experiment_runs.experiment_id FK -> harness_experiments."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
                " VALUES (999999, %s, 'pull', 1)",
                (run_pk,),
            )


def test_v007_harness_experiment_runs_pk_rejects_duplicate_run(db_with_legacy_schema) -> None:
    """§4.6: harness_experiment_runs PK on run_pk alone — a run belongs to one experiment.

    run_pk (not a composite with experiment_id) is the PK: a training run is created for
    exactly one experiment arm, so this also enforces "a run cannot belong to two
    experiments." A valid row is accepted first (happy path), then the duplicate bounces.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema)
    run_pk = _insert_training_run(db_with_legacy_schema)
    ins = (
        "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
        " VALUES (%s, %s, 'pull', 1)"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (experiment_id, run_pk))
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(ins, (experiment_id, run_pk))


def test_v007_harness_experiment_runs_valid_accepted(db_with_legacy_schema) -> None:
    """§4.6: pull and control arms for the same experiment are both accepted (happy path)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema)
    pull_run_pk = _insert_training_run(db_with_legacy_schema, attempt=1)
    control_run_pk = _insert_training_run(db_with_legacy_schema, attempt=2)
    with db_with_legacy_schema.connection() as conn:
        conn.execute(
            "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
            " VALUES (%s, %s, 'pull', 1)",
            (experiment_id, pull_run_pk),
        )
        conn.execute(
            "INSERT INTO harness_experiment_runs (experiment_id, run_pk, arm, seed_pair)"
            " VALUES (%s, %s, 'control', 1)",
            (experiment_id, control_run_pk),
        )
        rows = conn.execute(
            "SELECT arm FROM harness_experiment_runs WHERE experiment_id = %s ORDER BY arm",
            (experiment_id,),
        ).fetchall()
    assert [r["arm"] for r in rows] == ["control", "pull"]


def test_v007_harness_replays_llm_call_null_rejected(db_with_legacy_schema) -> None:
    """§4.6: harness_replays.llm_call_id NOT NULL — mirrors V006's never-orphaned discipline."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema, stage=2)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_replays (experiment_id, llm_call_id, situation,"
                " expected_response)"
                " VALUES (%s, NULL, '{\"schema_version\": 1}', '{\"schema_version\": 1}')",
                (experiment_id,),
            )


def test_v007_harness_replays_llm_call_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """§4.6: harness_replays.llm_call_id FK -> llm_calls (consumes V004 llm_calls)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema, stage=2)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_replays (experiment_id, llm_call_id, situation,"
                " expected_response)"
                " VALUES (%s, 999999, '{\"schema_version\": 1}', '{\"schema_version\": 1}')",
                (experiment_id,),
            )


def test_v007_harness_replays_experiment_id_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """§4.6: harness_replays.experiment_id FK -> harness_experiments."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    llm_call_id = _insert_harness_replay_call(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO harness_replays (experiment_id, llm_call_id, situation,"
                " expected_response)"
                " VALUES (999999, %s, '{\"schema_version\": 1}', '{\"schema_version\": 1}')",
                (llm_call_id,),
            )


def test_v007_harness_replays_valid_accepted(db_with_legacy_schema) -> None:
    """§4.6: a replay quiz with both mandatory FKs set is accepted (happy path)."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    experiment_id = _insert_harness_experiment(db_with_legacy_schema, stage=2)
    llm_call_id = _insert_harness_replay_call(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row_id = conn.execute(
            "INSERT INTO harness_replays (experiment_id, llm_call_id, situation,"
            " expected_response)"
            " VALUES (%s, %s, '{\"schema_version\": 1}', '{\"schema_version\": 1}')"
            " RETURNING id",
            (experiment_id, llm_call_id),
        ).fetchone()["id"]
    assert row_id is not None


def test_v007_schema_version_is_7(db_with_legacy_schema) -> None:
    """V007 lands in the ledger (was newest; V008 now raises the ceiling to 8).

    The newest-version invariant moved to ``test_v008_schema_version_is_8`` when
    Track B Task 11 shipped V008 — this asserts V007 was applied, not that it is the
    maximum, so it stays green as later migrations extend the ledger.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM schema_migrations WHERE version = 7"
        ).fetchone()
    assert row["n"] == 1


# ---------------------------------------------------------------------------
# V008 (Task 11): §4.8 weakness_profiles (append-only per-(env,algo,failure_mode)
# versioning) + weakness_evidence (polymorphic, same shape as pattern_sources),
# operator_actions (N14 — append-only human interventions outside the pre-built
# slots), and the six derived views. Constraint tests here; view smoke tests in
# tests/data/test_views.py.
# ---------------------------------------------------------------------------


def _insert_weakness_profile(
    db: DatabaseManager,
    *,
    environment: str = "equity",
    algorithm: str = "ppo",
    failure_mode: str = "trade_shy",
    version: int = 1,
    status: str = "active",
) -> int:
    """Insert a minimal valid weakness_profiles row and return its weakness_id.

    ``signature`` (JSONB NOT NULL) is the only payload the DDL forces beyond the
    identity/classification fields; ``version``/``status`` carry defaults.
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO weakness_profiles (environment, algorithm, failure_mode,"
                " signature, version, status)"
                " VALUES (%s, %s, %s, '{\"schema_version\": 1}'::jsonb, %s, %s)"
                " RETURNING weakness_id",
                (environment, algorithm, failure_mode, version, status),
            ).fetchone()["weakness_id"]
        )


def test_v008_weakness_profiles_unique_env_algo_mode_version(db_with_legacy_schema) -> None:
    """§4.8: UNIQUE(environment, algorithm, failure_mode, version) — append-only versioning.

    Revisions are new rows (a new version); a duplicate of the same four bounces off
    the schema, and a fresh version of the same (env, algo, failure_mode) is accepted.
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    _insert_weakness_profile(db_with_legacy_schema, version=1)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_weakness_profile(db_with_legacy_schema, version=1)
    _insert_weakness_profile(db_with_legacy_schema, version=2)  # new version OK


def test_v008_weakness_profiles_requires_signature(db_with_legacy_schema) -> None:
    """§4.8: weakness_profiles.signature is NOT NULL JSONB — no scouting report without one."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO weakness_profiles (environment, algorithm, failure_mode, status)"
                " VALUES ('equity', 'ppo', 'trade_shy', 'active')"
            )


def test_v008_weakness_profiles_status_check(db_with_legacy_schema) -> None:
    """§4.8: weakness_profiles.status CHECK (active, retired) — trained-out weaknesses retire."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_weakness_profile(db_with_legacy_schema, status="bogus")


def test_v008_weakness_evidence_source_table_check(db_with_legacy_schema) -> None:
    """§4.8: weakness_evidence.source_table CHECK — points only at structured records + patterns.

    A confirmed pattern graduating into the career file (``patterns``) is accepted;
    retired raw ``memories`` bounces off the CHECK. ``source_id`` is polymorphic (no FK).
    """
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    weakness_id = _insert_weakness_profile(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        conn.execute(
            "INSERT INTO weakness_evidence (weakness_id, source_table, source_id)"
            " VALUES (%s, 'patterns', 1)",
            (weakness_id,),
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO weakness_evidence (weakness_id, source_table, source_id)"
                " VALUES (%s, 'memories', 2)",
                (weakness_id,),
            )


def test_v008_weakness_evidence_weakness_fk_rejects_nonexistent(db_with_legacy_schema) -> None:
    """§4.8: weakness_evidence.weakness_id FK -> weakness_profiles."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO weakness_evidence (weakness_id, source_table, source_id)"
                " VALUES (999999, 'fold_results', 1)"
            )


def test_v008_operator_actions_requires_actor(db_with_legacy_schema) -> None:
    """N14: operator_actions.actor is NOT NULL — an intervention names who did it."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO operator_actions (action_type, reason)"
                " VALUES ('demote', 'ladder demotion after 3 losing seasons')"
            )


def test_v008_operator_actions_requires_action_type(db_with_legacy_schema) -> None:
    """N14: operator_actions.action_type is NOT NULL."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO operator_actions (actor, reason)"
                " VALUES ('varun', 'ladder demotion after 3 losing seasons')"
            )


def test_v008_operator_actions_requires_reason(db_with_legacy_schema) -> None:
    """N14: operator_actions.reason is NOT NULL — every intervention carries its why."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with pytest.raises(psycopg.errors.NotNullViolation):
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO operator_actions (actor, action_type) VALUES ('varun', 'demote')"
            )


def test_v008_operator_actions_valid_accepted(db_with_legacy_schema) -> None:
    """N14: a fully specified intervention row (all NOT NULLs + optional payload) is accepted."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row_id = conn.execute(
            "INSERT INTO operator_actions (actor, action_type, target_table, target_id,"
            " reason, payload)"
            " VALUES ('varun', 'demote', 'season_results', 42, 'ladder demotion',"
            ' \'{"schema_version": 1, "from_level": 3, "to_level": 2}\'::jsonb)'
            " RETURNING id"
        ).fetchone()["id"]
    assert row_id is not None


def test_v008_schema_version_is_8(db_with_legacy_schema) -> None:
    """Task 11: SELECT max(version) FROM schema_migrations == 8 after apply_migrations."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        row = conn.execute("SELECT max(version) AS v FROM schema_migrations").fetchone()
    assert row["v"] == 8
