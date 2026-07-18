"""V008 view smoke tests (Task 11): the six derived surfaces (never store the derivable).

Focus assertions per the task brief:

- ``v_consolidation_corpus`` returns the canonical season/reference records and
  **zero** harness-tagged records — S4 criterion 5's foundation, the structural
  quarantine boundary (§4.6 / A19).
- ``v_l2_settings_history`` returns exactly one row per (environment, algorithm,
  iteration) with NULL ``source_intent_id`` for a reference season (§4.4 / D-T2.11).

The remaining four views (``v_lever_track_record``, ``v_consolidator_quality``,
``v_pattern_effectiveness``, ``v_live_transfer``) get well-formedness smoke checks
plus one data-bearing aggregation assertion for the per-lever track record.

The ``db_with_legacy_schema`` fixture lives in tests/data/conftest.py (shared with
the migration-content module) — pytest auto-discovers it for this module.
"""

from __future__ import annotations

import os

import pytest

from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

_ALL_VIEWS = (
    "v_consolidation_corpus",
    "v_l2_settings_history",
    "v_lever_track_record",
    "v_consolidator_quality",
    "v_pattern_effectiveness",
    "v_live_transfer",
)


def _insert_run(
    db: DatabaseManager,
    *,
    run_type: str,
    iteration: int = 5,
    environment: str = "equity",
    algorithm: str = "ppo",
    fold_number: int = 0,
    attempt: int = 1,
    status: str = "completed",
    seed: int = 42,
) -> int:
    """Insert a training_runs row with an explicit run_type/status and return its run_pk."""
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO training_runs (iteration_number, environment, algorithm,"
                " fold_number, run_type, seed, attempt, status, era_id, code_version,"
                " data_fingerprint)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 'abc123', 'fp1')"
                " RETURNING run_pk",
                (iteration, environment, algorithm, fold_number, run_type, seed, attempt, status),
            ).fetchone()["run_pk"]
        )


def _insert_epoch_and_fold(db: DatabaseManager, run_pk: int) -> None:
    """Attach one epoch_snapshots + one fold_results row to ``run_pk`` (corpus fodder)."""
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO epoch_snapshots (run_pk, learner_metrics)"
            " VALUES (%s, '{\"schema_version\": 1}'::jsonb)",
            (run_pk,),
        )
        conn.execute(
            "INSERT INTO fold_results (run_pk, era_id, gate_version_id, seed,"
            " fold_start_ts, fold_end_ts) VALUES (%s, 0, 0, 42, now(), now())",
            (run_pk,),
        )


def _insert_reference_season(
    db: DatabaseManager, *, iteration: int = 5, environment: str = "equity"
) -> None:
    """Insert a coach-free reference season: season_results for all four scopes.

    ``coach_config`` carries ``reference_season: true``; algo scopes carry
    ``hyperparams_used`` (reference seasons still record HPs — the L2 view invariant).
    """
    for scope in ("ppo", "a2c", "sac", "ensemble"):
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO season_results (iteration_number, environment, scope, era_id,"
                " gate_version_per_fold, gate_version_ensemble, coach_config,"
                " hyperparams_used, cps_v1)"
                " VALUES (%s, %s, %s, 0, 0, 1,"
                ' \'{"schema_version": 1, "reference_season": true}\'::jsonb,'
                ' \'{"schema_version": 1, "learning_rate": 0.0003}\'::jsonb, 0.15)',
                (iteration, environment, scope),
            )


def _insert_graded_commentary_intent(
    db: DatabaseManager, *, environment: str = "equity", algorithm: str = "ppo"
) -> int:
    """Insert cycle -> trade_commentary llm_call -> MT_commentary intent -> verdict.

    Returns the intent_id. Gives ``v_lever_track_record`` one graded (coach, lever,
    scope) bet to aggregate.
    """
    with db.connection() as conn:
        cycle_id = int(
            conn.execute(
                "INSERT INTO inference_cycles (environment, mode, cycle_ts)"
                " VALUES (%s, 'paper', now()) RETURNING cycle_id",
                (environment,),
            ).fetchone()["cycle_id"]
        )
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
                "  evidence, proposal, bet_metric, bet_direction, bet_baseline_value,"
                "  horizon_spec)"
                " VALUES (%s, 'meta_trader', 'MT_commentary', 'shadow', %s, %s, 0,"
                "  '{}'::jsonb, '{}'::jsonb, 'cycle_pnl_frac', 'up', 0.0,"
                '  \'{"type": "wall_clock_hours", "hours": 24}\'::jsonb)'
                " RETURNING intent_id",
                (llm_call_id, environment, algorithm),
            ).fetchone()["intent_id"]
        )
        conn.execute(
            "INSERT INTO intent_verdicts (intent_id, grader_version, actual_value,"
            " direction_match, menu_consistent) VALUES (%s, 1, 0.01, true, true)",
            (intent_id,),
        )
    return intent_id


def _insert_pattern(
    db: DatabaseManager,
    *,
    created_iteration: int,
    environment: str | None,
    confirmation_count: int,
    contradiction_count: int,
) -> int:
    """Insert one active stage-1 pattern (era 0) and return its pattern_id."""
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO patterns (created_iteration, environment, stage, era_id, claim,"
                " status, confirmation_count, contradiction_count)"
                " VALUES (%s, %s, 1, 0, '{\"schema_version\": 1}'::jsonb, 'active', %s, %s)"
                " RETURNING pattern_id",
                (created_iteration, environment, confirmation_count, contradiction_count),
            ).fetchone()["pattern_id"]
        )


def _insert_consolidator_call(
    db: DatabaseManager,
    *,
    iteration: int,
    environment: str,
    prompt_version: str,
    model: str,
    success: bool,
    age_hours: float,
) -> int:
    """Insert one consolidate_stage1 call with an explicit created_at age; return llm_call_id."""
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, iteration_number, environment,"
                " provider, model, prompt_version, success, created_at)"
                " VALUES ('consolidator', 'consolidate_stage1', %s, %s, 'cerebras', %s, %s, %s,"
                " now() - (%s * interval '1 hour')) RETURNING llm_call_id",
                (iteration, environment, model, prompt_version, success, age_hours),
            ).fetchone()["llm_call_id"]
        )


def _insert_llm_call(
    db: DatabaseManager,
    *,
    call_type: str,
    run_pk: int | None = None,
) -> int:
    """Insert one llm_call of the given call_type (meta_trainer). Return llm_call_id.

    ``run_pk`` is set for run-scoped types (e.g. epoch_advice). harness_replay carries
    no identity requirement, so run_pk stays NULL — the whole point of CASE B is that the
    quarantine keys off the intent's PARENT call_type, not its run_pk.
    """
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO llm_calls (coach, call_type, run_pk, provider, model,"
                " prompt_version, success) VALUES ('meta_trainer', %s, %s, 'cerebras',"
                " 'qwen-3', 'mt-l1-v0', true) RETURNING llm_call_id",
                (call_type, run_pk),
            ).fetchone()["llm_call_id"]
        )


def _insert_l1_intent(
    db: DatabaseManager,
    *,
    llm_call_id: int,
    run_pk: int,
    environment: str = "equity",
    algorithm: str = "ppo",
) -> int:
    """Insert one run-scoped L1_reward_weights (mid-fold) intent; return intent_id."""
    with db.connection() as conn:
        return int(
            conn.execute(
                "INSERT INTO intent_records"
                " (llm_call_id, coach, lever, mode, run_pk, iteration_number, environment,"
                "  algorithm, evidence, proposal, bet_metric, bet_direction, bet_baseline_value,"
                "  horizon_spec)"
                " VALUES (%s, 'meta_trainer', 'L1_reward_weights', 'shadow', %s, 5, %s, %s,"
                "  '{}'::jsonb, '{}'::jsonb, 'oos_sharpe', 'up', 0.0,"
                '  \'{"type": "folds", "n": 1}\'::jsonb)'
                " RETURNING intent_id",
                (llm_call_id, run_pk, environment, algorithm),
            ).fetchone()["intent_id"]
        )


def test_v008_all_views_exist_and_are_queryable(db_with_legacy_schema: DatabaseManager) -> None:
    """All six V008 views compile, resolve their columns, and execute (count(*))."""
    apply_migrations(db_with_legacy_schema)
    for view in _ALL_VIEWS:
        with db_with_legacy_schema.connection() as conn:
            row = conn.execute(f"SELECT count(*) AS n FROM {view}").fetchone()  # noqa: S608
        assert row["n"] >= 0


def test_v_consolidation_corpus_includes_canonical_season_excludes_harness(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """§4.6 / A19: the corpus returns canonical season records and ZERO harness records.

    S4 criterion 5's foundation. A season run (canonical) and a harness_stage1 run
    each get epoch + fold rows; only the season run's records are visible in the corpus.
    """
    apply_migrations(db_with_legacy_schema)
    season_run = _insert_run(db_with_legacy_schema, run_type="season", fold_number=0)
    harness_run = _insert_run(db_with_legacy_schema, run_type="harness_stage1", fold_number=1)
    _insert_epoch_and_fold(db_with_legacy_schema, season_run)
    _insert_epoch_and_fold(db_with_legacy_schema, harness_run)

    with db_with_legacy_schema.connection() as conn:
        season_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus WHERE run_pk = %s",
            (season_run,),
        ).fetchone()["n"]
        harness_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus WHERE run_pk = %s",
            (harness_run,),
        ).fetchone()["n"]
        season_tables = {
            r["source_table"]
            for r in conn.execute(
                "SELECT DISTINCT source_table FROM v_consolidation_corpus WHERE run_pk = %s",
                (season_run,),
            ).fetchall()
        }

    assert harness_rows == 0, "harness-tagged runs must never reach the consolidation corpus"
    assert season_rows >= 2, "canonical season run's epoch + fold records must be in the corpus"
    assert {"epoch_snapshots", "fold_results"} <= season_tables


def test_v_consolidation_corpus_excludes_noncanonical_attempt(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """A6: only the highest completed attempt is canonical; a superseded attempt is excluded."""
    apply_migrations(db_with_legacy_schema)
    old_attempt = _insert_run(db_with_legacy_schema, run_type="season", fold_number=0, attempt=1)
    new_attempt = _insert_run(db_with_legacy_schema, run_type="season", fold_number=0, attempt=2)
    _insert_epoch_and_fold(db_with_legacy_schema, old_attempt)
    _insert_epoch_and_fold(db_with_legacy_schema, new_attempt)

    with db_with_legacy_schema.connection() as conn:
        old_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus WHERE run_pk = %s",
            (old_attempt,),
        ).fetchone()["n"]
        new_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus WHERE run_pk = %s",
            (new_attempt,),
        ).fetchone()["n"]

    assert old_rows == 0, "the superseded (lower) attempt is non-canonical and excluded (A6)"
    assert new_rows >= 2, "the highest completed attempt is canonical and included"


def test_v_l2_settings_history_reference_season_null_source_intent(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """§4.4 / D-T2.11: one row per (env, algo, iteration); NULL source_intent for a reference season.

    A coach-free reference season writes season_results for ppo/a2c/sac (+ ensemble).
    The L2 history exposes exactly the three algo scopes, each with NULL
    source_intent_id (no live L2 pull drove them).
    """
    apply_migrations(db_with_legacy_schema)
    _insert_reference_season(db_with_legacy_schema, iteration=5, environment="equity")

    with db_with_legacy_schema.connection() as conn:
        rows = conn.execute(
            "SELECT environment, algorithm, iteration_number, source_intent_id"
            " FROM v_l2_settings_history"
            " WHERE iteration_number = 5 AND environment = 'equity'"
            " ORDER BY algorithm"
        ).fetchall()

    assert len(rows) == 3, "one row per algo scope (ensemble excluded — no hyperparams_used)"
    assert {r["algorithm"] for r in rows} == {"ppo", "a2c", "sac"}
    assert all(r["source_intent_id"] is None for r in rows)


def test_v_lever_track_record_aggregates_graded_intent(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """§4.4 / S8: intents ⋈ verdicts aggregate per (coach, lever, scope) — a graded bet shows up."""
    apply_migrations(db_with_legacy_schema)
    _insert_graded_commentary_intent(db_with_legacy_schema, environment="equity", algorithm="ppo")

    with db_with_legacy_schema.connection() as conn:
        row = conn.execute(
            "SELECT coach, lever, environment, algorithm, total_verdicts,"
            " direction_match_count"
            " FROM v_lever_track_record"
            " WHERE lever = 'MT_commentary' AND environment = 'equity' AND algorithm = 'ppo'"
        ).fetchone()

    assert row is not None
    assert row["coach"] == "meta_trader"
    assert row["total_verdicts"] >= 1
    assert row["direction_match_count"] >= 1


def test_v_consolidator_quality_dedups_and_filters_success(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """A18 hardening: a pattern is credited to exactly ONE producing call.

    The producing call is the newest SUCCESSFUL consolidator call for the pattern's
    (iteration, environment). A same-config retry must NOT double the tallies; an older
    successful call and a newer FAILED call are not credited. Reproduces both reviewer
    symptoms (double-count via same prompt_version/model; mis-attribution across configs).
    """
    apply_migrations(db_with_legacy_schema)
    _insert_pattern(
        db_with_legacy_schema,
        created_iteration=7,
        environment="crypto",
        confirmation_count=3,
        contradiction_count=1,
    )
    # older SUCCESSFUL call, DIFFERENT config -> mis-attribution candidate (old view credits it too)
    _insert_consolidator_call(
        db_with_legacy_schema,
        iteration=7,
        environment="crypto",
        prompt_version="cons-old",
        model="m-old",
        success=True,
        age_hours=3.0,
    )
    # same-config retry PAIR -> double-count candidate; the second is the newest success
    _insert_consolidator_call(
        db_with_legacy_schema,
        iteration=7,
        environment="crypto",
        prompt_version="cons-new",
        model="m-new",
        success=True,
        age_hours=2.5,
    )
    _insert_consolidator_call(
        db_with_legacy_schema,
        iteration=7,
        environment="crypto",
        prompt_version="cons-new",
        model="m-new",
        success=True,
        age_hours=2.0,
    )
    # newest overall but FAILED -> the success filter must drop it
    _insert_consolidator_call(
        db_with_legacy_schema,
        iteration=7,
        environment="crypto",
        prompt_version="cons-fail",
        model="m-fail",
        success=False,
        age_hours=1.0,
    )

    with db_with_legacy_schema.connection() as conn:
        rows = conn.execute(
            "SELECT prompt_version, model, pattern_count, total_confirmations,"
            " total_contradictions FROM v_consolidator_quality"
            " WHERE model IN ('m-old', 'm-new', 'm-fail') ORDER BY model"
        ).fetchall()

    assert len(rows) == 1, (
        "the pattern must be credited to exactly one producing call, not spread across the"
        " older/retry/failed calls (old view returned 3 rows)"
    )
    row = rows[0]
    assert row["model"] == "m-new"
    assert row["prompt_version"] == "cons-new"
    assert row["pattern_count"] == 1
    assert row["total_confirmations"] == 3, (
        "the same-config retry must NOT double the confirmation count (old view summed to 6)"
    )
    assert row["total_contradictions"] == 1


def test_v_consolidation_corpus_excludes_harness_parented_intent(
    db_with_legacy_schema: DatabaseManager,
) -> None:
    """A19 CASE B: an intent whose PARENT llm_call is a harness_replay is quarantined.

    The intent is tagged to a canonical season ``run_pk`` (so the run-scoped branch would
    otherwise admit it), but its parent call is a Stage-2 replay -> it must NOT reach the
    corpus. A normal season intent (parent = epoch_advice) tagged to the same run stays in.
    """
    apply_migrations(db_with_legacy_schema)
    season_run = _insert_run(db_with_legacy_schema, run_type="season", fold_number=0)

    normal_call = _insert_llm_call(
        db_with_legacy_schema, call_type="epoch_advice", run_pk=season_run
    )
    harness_call = _insert_llm_call(db_with_legacy_schema, call_type="harness_replay")
    normal_intent = _insert_l1_intent(
        db_with_legacy_schema, llm_call_id=normal_call, run_pk=season_run
    )
    harness_intent = _insert_l1_intent(
        db_with_legacy_schema, llm_call_id=harness_call, run_pk=season_run
    )

    with db_with_legacy_schema.connection() as conn:
        normal_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus"
            " WHERE source_table = 'intent_records' AND source_id = %s",
            (str(normal_intent),),
        ).fetchone()["n"]
        harness_rows = conn.execute(
            "SELECT count(*) AS n FROM v_consolidation_corpus"
            " WHERE source_table = 'intent_records' AND source_id = %s",
            (str(harness_intent),),
        ).fetchone()["n"]

    assert harness_rows == 0, "an intent parented by a harness_replay must not reach the corpus"
    assert normal_rows == 1, "a normal season intent (epoch_advice parent) stays in the corpus"
