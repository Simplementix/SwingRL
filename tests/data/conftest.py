"""Shared PostgreSQL-backed fixtures for tests/data/'s migration-content and bootstrap tests.

Extracted from tests/data/test_migrations_content.py and
tests/data/test_bootstrap_era0_models.py (previously ~60-line duplicates of each
other, plus a third duplicate of just the YAML builder in
tests/data/test_migration_runner.py). Module-local to tests/data/ — pytest
discovers these fixtures automatically for every test module in this directory,
no import required.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations


@pytest.fixture
def db_config_yaml(tmp_path: Path) -> str:
    """Config YAML with system section pointing to DATABASE_URL."""
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://test:test@localhost:5432/swingrl_test",  # pragma: allowlist secret
    )
    return textwrap.dedent(f"""\
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ]
          max_position_size: 0.25
          max_drawdown_pct: 0.10
          daily_loss_limit_pct: 0.02
        crypto:
          symbols: [BTCUSDT, ETHUSDT]
          max_position_size: 0.50
          max_drawdown_pct: 0.12
          daily_loss_limit_pct: 0.03
          min_order_usd: 10.0
        capital:
          equity_usd: 400.0
          crypto_usd: 47.0
        paths:
          data_dir: data/
          db_dir: db/
          models_dir: models/
          logs_dir: logs/
        logging:
          level: INFO
          json_logs: false
        system:
          database_url: "{db_url}"
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


def _drop_migration_artifacts(mgr: DatabaseManager) -> None:
    """Drop all V001-V004 artifacts and clear their schema_migrations ledger rows.

    FK-safe order throughout: V004 coach-record artifacts (intent_verdicts/
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
            "DELETE FROM schema_migrations WHERE version IN (1, 2, 3, 4); "
            "END IF; "
            "END $$;"
        )


@pytest.fixture
def db_with_legacy_schema(
    tmp_path: Path, db_config_yaml: str
) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager on DATABASE_URL, forced to a deterministic pre-V001 (legacy) schema.

    Shared by tests/data/test_migrations_content.py and
    tests/data/test_bootstrap_era0_models.py. Legacy tables (backtest_results,
    iteration_results, ...) exist, possibly empty — the V001 back-stamp must hold for
    0 rows in CI and 574 in production. V001/V002/V003/V004's plain ``CREATE TABLE`` /
    ``ALTER TABLE ADD COLUMN`` statements are not idempotent by themselves, so both
    SETUP and TEARDOWN call ``_drop_migration_artifacts()`` (see above) before doing
    anything else, to guarantee a clean legacy starting point regardless of what any
    other test using this fixture left behind.

    Calling the drop at SETUP (not only teardown) is what makes the entry state
    deterministic: most tests using this fixture call ``apply_migrations()``
    themselves mid-test to reach migrated content, but
    ``test_bootstrap_era0_models_requires_schema_v2`` deliberately never does — it
    asserts that ``bootstrap_era0_models()`` raises ``ConfigError`` while the schema is
    genuinely behind V002. Without a setup-time drop, that assertion's validity
    depended entirely on the teardown behavior of whichever test happened to run
    immediately before it in the same session — order-dependent by construction, and
    already fragile before this fix (on an unmigrated fixture, that test's own teardown
    raised ``UndefinedTable`` trying to ``ALTER TABLE ensemble_weight_history`` when
    the table didn't exist — the guard above fixes that too).

    Calling the same drop again at TEARDOWN, followed by ``apply_migrations()``,
    restores the suite-wide invariant CI stage 2.7 establishes once per run (schema
    migrated before any test executes) and that every other DB-gated fixture (e.g.
    tests/execution/conftest.py's ``mock_db``) assumes holds for the rest of the
    suite — without the re-apply, the drops left the shared DB in legacy state for the
    remainder of the run, causing order-dependent ``UndefinedColumn`` failures (e.g.
    ``trades.cycle_id``) in any later DB-gated test that needs V001-V004 artifacts.
    This also keeps the persistent scratch database re-runnable across test sessions:
    every session both starts and ends fully migrated, never legacy.
    """
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    mgr.init_schema()
    _drop_migration_artifacts(mgr)
    yield mgr
    _drop_migration_artifacts(mgr)
    apply_migrations(mgr)
    DatabaseManager.reset()
