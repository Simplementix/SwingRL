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
