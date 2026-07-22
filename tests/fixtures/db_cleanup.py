"""Autouse fixture that wipes the test database after every test.

Registered by importing ``wipe_db_after_test`` into ``tests/conftest.py``.
Function-scoped and autouse, so every test starts from a clean database —
killing the inter-test state pollution that fails ~50 tests on homelab CI.

Three safety layers (see ``tests/db_guard.py``):
  1. Postgres binds a connection to one database, so a ``*_test`` connection
     physically cannot reach production ``swingrl``.
  2. The suite-level guard (conftest ``pytest_configure``) refuses to start
     unless the resolved DB is a test database.
  3. ``ensure_wipe_target_is_test_db`` re-checks the resolved DB name ends in
     ``_test`` immediately before every TRUNCATE and refuses otherwise.
"""

from __future__ import annotations

from collections.abc import Generator

import psycopg
import pytest
from psycopg import sql

from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations
from tests.db_guard import classify_db_url, resolve_target_db_url


def ensure_wipe_target_is_test_db(db_url: str) -> bool:
    """Pre-wipe re-check (safety layer #3).

    Returns ``True`` when the resolved DB is a test database and should be wiped,
    ``False`` when no DB is configured (skip). Raises ``RuntimeError`` when a DB
    is present but is NOT a recognised test database — cleanup must never run
    against production.
    """
    verdict, db_name = classify_db_url(db_url)
    if verdict == "blank":
        return False
    if verdict != "safe":
        raise RuntimeError(
            f"Refusing to TRUNCATE: resolved database {db_name!r} is not a test "
            f"database (verdict={verdict!r}). The suite guard should have already "
            f"aborted — this is a defence-in-depth backstop."
        )
    return True


# Migration-managed registry tables — append-only, owned by
# src/swingrl/data/migration_runner.py, NOT test data. Excluded from the
# per-test TRUNCATE:
#   - schema_migrations: the migration ledger itself. Wiping it desyncs the
#     ledger from the tables it describes -- a later apply_migrations() call
#     sees an empty ledger, tries to re-run a V-file whose CREATE TABLE already
#     exists (relation "gate_versions" already exists), and errors instead of
#     no-opping. This is exactly the CI failure this exclusion fixes: CI
#     pre-applies V001/V002 to fresh swingrl_test (stage 2.7) before any test
#     runs, so the first TRUNCATE after any test would otherwise blow away that
#     pre-applied ledger for the rest of the run.
#   - eras, gate_versions: the V001 registry seed rows (era 0, gate version 1).
#     Truncating "eras" also breaks every later insert into back-stamped tables
#     whose era_id DEFAULT 0 FK needs era row 0 to still exist.
# Extend this list when a future migration (Task 8 / V003+) adds another
# registry table -- data tables created BY migrations (training_runs, models,
# ensemble_weight_history, and the back-stamped legacy tables) are NOT
# registries and STAY in the wipe.
_MIGRATION_REGISTRY_TABLES = frozenset({"schema_migrations", "eras", "gate_versions"})


def _truncate_all_public_tables(db_url: str) -> None:
    """TRUNCATE every non-registry table in the public schema of the test database.

    Uses a dedicated short-lived autocommit connection (not the pooled
    DatabaseManager connection) to avoid pool-affinity surprises, with a bounded
    ``lock_timeout`` so a stray lock from a misbehaving test fails fast rather
    than hanging CI. Catalog-enumerates tables so ad-hoc tables created by raw
    tests are covered; a single ``TRUNCATE … CASCADE`` handles FK ordering.
    Migration-managed registry tables (``_MIGRATION_REGISTRY_TABLES``) are
    excluded -- see that constant's docstring for why.
    """
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("SET lock_timeout = '10s'")
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        tables = [row[0] for row in rows if row[0] not in _MIGRATION_REGISTRY_TABLES]
        if not tables:
            return
        statement = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
            sql.SQL(", ").join(sql.Identifier(name) for name in tables)
        )
        conn.execute(statement)


def drop_migration_artifacts(mgr: DatabaseManager) -> None:
    """Drop all V001-V010 artifacts and clear their schema_migrations ledger rows.

    THE single shared de-migration teardown for every DB-gated fixture on the
    shared CI scratch database. It replaced two hand-rolled copies (one in
    tests/data/conftest.py, one in tests/memory/test_trade_router.py) that had
    drifted: the memory copy was frozen at V004 and broke the moment V008's views
    (which depend on ``intent_verdicts``) coexisted with it —
    ``DependentObjectsStillExist``. Keeping the drop order in exactly one place is
    the durable fix; extend THIS function (and nothing else) when a new migration
    ships.

    FK-safe order throughout: V008 views drop first (they depend on V002-V007 tables),
    then the three V008 tables (weakness_evidence -> weakness_profiles; operator_actions
    standalone); then V007 harness tables (harness_replays/harness_experiment_runs
    -- both point at llm_calls/training_runs dropped further down, and harness_experiments
    -- referenced by both, so all three drop first of all, before any of their FK targets);
    then V006 patterns-family tables (pattern_presentations/
    pattern_links/pattern_sources/patterns — nothing references them, and they point
    at llm_calls/eras dropped further down); then V005
    training-record leaf tables (backtest_trades/
    season_results/fold_results/epoch_snapshots — nothing references them, so they
    drop next) before their referenced training_runs/eras/gate_versions parents;
    then V004 coach-record artifacts (intent_verdicts/
    intent_applications -> intent_records, referenced by the ensemble_weight_history
    FK -> llm_calls; the ewh FK + two A14 partial UNIQUE indexes are dropped first so
    the DROP TABLE of intent_records succeeds), then V003 (event_outcomes/fill_quality
    -> calendar_events/trades.cycle_id -> cycle_algo_proposals -> inference_cycles),
    then V002 (ensemble_weight_history -> models -> training_runs, since
    training_runs.era_id references eras, dropped below), then V001 (the two new
    registry tables and the four added columns).

    The very first statement guards on ``to_regclass`` (table existence) before the
    ``ALTER TABLE ... DROP CONSTRAINT IF EXISTS`` — ``IF EXISTS`` there only makes the
    *constraint* drop conditional, not the table; calling this when
    ``ensemble_weight_history`` was never created (V002 not applied) raises
    ``UndefinedTable`` without the guard.
    """
    with mgr.connection() as conn:
        # V008 derived views (drop FIRST — they depend on V002-V007 tables below).
        conn.execute("DROP VIEW IF EXISTS v_consolidation_corpus")
        conn.execute("DROP VIEW IF EXISTS v_l2_settings_history")
        conn.execute("DROP VIEW IF EXISTS v_lever_track_record")
        conn.execute("DROP VIEW IF EXISTS v_consolidator_quality")
        conn.execute("DROP VIEW IF EXISTS v_pattern_effectiveness")
        conn.execute("DROP VIEW IF EXISTS v_live_transfer")
        # V008 tables (FK-safe: weakness_evidence -> weakness_profiles; operator_actions
        # is standalone). Nothing else references these, so they drop before everything.
        conn.execute("DROP TABLE IF EXISTS weakness_evidence")
        conn.execute("DROP TABLE IF EXISTS weakness_profiles")
        conn.execute("DROP TABLE IF EXISTS operator_actions")
        # V007 harness tables (leaf tables first): harness_replays references
        # llm_calls + harness_experiments (both dropped below); harness_experiment_runs
        # references training_runs + harness_experiments (both dropped below) — so the
        # two leaves drop before harness_experiments, which drops before any of those
        # FK targets.
        conn.execute("DROP TABLE IF EXISTS harness_replays")
        conn.execute("DROP TABLE IF EXISTS harness_experiment_runs")
        conn.execute("DROP TABLE IF EXISTS harness_experiments")
        # V006 patterns family (leaf tables first): pattern_presentations references
        # llm_calls (dropped below) and the three pattern_* children reference patterns
        # and eras (also dropped below), so all four must go before any of those FK
        # targets — dropped first since nothing references them.
        conn.execute("DROP TABLE IF EXISTS pattern_presentations")
        conn.execute("DROP TABLE IF EXISTS pattern_links")
        conn.execute("DROP TABLE IF EXISTS pattern_sources")
        conn.execute("DROP TABLE IF EXISTS patterns")
        conn.execute("DROP TABLE IF EXISTS backtest_trades")
        conn.execute("DROP TABLE IF EXISTS season_results")
        conn.execute("DROP TABLE IF EXISTS fold_results")
        conn.execute("DROP TABLE IF EXISTS epoch_snapshots")
        conn.execute(
            "DO $$ BEGIN "
            "IF to_regclass('public.ensemble_weight_history') IS NOT NULL THEN "
            "ALTER TABLE ensemble_weight_history DROP CONSTRAINT IF EXISTS fk_ewh_intent; "
            "END IF; "
            "END $$;"
        )
        conn.execute("DROP INDEX IF EXISTS uq_mt_commentary_per_cycle")
        conn.execute("DROP INDEX IF EXISTS uq_llm_commentary_cycle")
        conn.execute("DROP TABLE IF EXISTS intent_verdicts")
        conn.execute("DROP TABLE IF EXISTS intent_applications")
        conn.execute("DROP TABLE IF EXISTS intent_records")
        conn.execute("DROP TABLE IF EXISTS llm_calls")
        conn.execute("DROP TABLE IF EXISTS event_outcomes")
        conn.execute("DROP TABLE IF EXISTS calendar_events")
        conn.execute("DROP TABLE IF EXISTS fill_quality")
        # V009 pending_orders references inference_cycles(cycle_id), so it must be dropped
        # before inference_cycles (below) or the FK dependency blocks that DROP.
        conn.execute("DROP TABLE IF EXISTS pending_orders")
        # V010 benchmark_baselines is standalone (no FK in either direction), so it can drop
        # anywhere; kept beside the V009 exec-alignment drop above for provenance.
        conn.execute("DROP TABLE IF EXISTS benchmark_baselines")
        conn.execute("ALTER TABLE trades DROP COLUMN IF EXISTS cycle_id")
        conn.execute("DROP TABLE IF EXISTS cycle_algo_proposals")
        conn.execute("DROP TABLE IF EXISTS inference_cycles")
        conn.execute("DROP TABLE IF EXISTS ensemble_weight_history")
        conn.execute("DROP TABLE IF EXISTS models")
        conn.execute("DROP TABLE IF EXISTS training_runs")
        conn.execute(
            "ALTER TABLE backtest_results "
            "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_id"
        )
        conn.execute(
            "ALTER TABLE iteration_results "
            "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_ensemble_id"
        )
        conn.execute("DROP TABLE IF EXISTS eras")
        conn.execute("DROP TABLE IF EXISTS gate_versions")
        conn.execute(
            "DO $$ BEGIN "
            "IF to_regclass('public.schema_migrations') IS NOT NULL THEN "
            "DELETE FROM schema_migrations WHERE version IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10); "
            "END IF; "
            "END $$;"
        )


def reapply_migrated_schema(mgr: DatabaseManager) -> None:
    """De-migrate then re-apply every migration — the shared-CI-DB teardown invariant.

    Drops all V001-V010 artifacts (via ``drop_migration_artifacts``) and immediately
    re-runs ``apply_migrations`` so the shared scratch database is left FULLY MIGRATED
    (ledger at max version) after the fixture that used it finishes. This is the CI
    stage-2.7 invariant every DB-gated fixture assumes for the rest of the run; without
    the re-apply, the drops would strand the shared DB in legacy state and cause
    order-dependent ``UndefinedColumn``/``UndefinedTable`` failures in any later test.
    """
    drop_migration_artifacts(mgr)
    apply_migrations(mgr)


@pytest.fixture(autouse=True)
def wipe_db_after_test() -> Generator[None, None, None]:
    """Wipe all test-database tables after each test (no-op when no test DB)."""
    yield
    db_url = resolve_target_db_url()
    if not ensure_wipe_target_is_test_db(db_url):
        return
    # Close the pool so idle connections release their locks before TRUNCATE.
    # NOTE: reset() closes only idle/returned connections — a connection still
    # checked out when teardown runs is NOT force-closed. That can only happen if
    # a test leaks a checked-out connection past its own body (a bug); the
    # lock_timeout in _truncate_all_public_tables bounds such a case to a loud
    # ~10s failure that names the offending test, rather than a silent hang.
    # A proper drain/rollback fixture is deferred to Stage 3.0 (see the spec).
    DatabaseManager.reset()
    _truncate_all_public_tables(db_url)
