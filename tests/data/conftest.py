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


@pytest.fixture
def db_with_legacy_schema(
    tmp_path: Path, db_config_yaml: str
) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager on DATABASE_URL with init_schema() already run.

    Shared by tests/data/test_migrations_content.py and
    tests/data/test_bootstrap_era0_models.py. Legacy tables (backtest_results,
    iteration_results, ...) exist, possibly empty — the V001 back-stamp must
    hold for 0 rows in CI and 574 in production. V001/V002/V003's plain
    ``CREATE TABLE`` / ``ALTER TABLE ADD COLUMN`` statements are not idempotent
    by themselves, so teardown drops the V003 artifacts (event_outcomes/
    calendar_events/fill_quality/trades.cycle_id/cycle_algo_proposals/
    inference_cycles, FK-safe order), the V002 artifacts (training_runs/models/
    ensemble_weight_history, FK-safe order), and the V001 artifacts (the two new
    registry tables and the four added columns), then clears the version-1,
    version-2, and version-3 ledger rows — keeping the persistent scratch
    database re-runnable across test sessions.

    V004 (Task 12) extends this teardown: the coach-record artifacts
    (intent_verdicts/intent_applications/intent_records/llm_calls, FK-safe order),
    the ensemble_weight_history.intent_id FK constraint, and the two A14 partial
    UNIQUE indexes are all dropped, and the ledger cleanup covers version 4.
    """
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    mgr.init_schema()
    yield mgr
    with mgr.connection() as conn:
        # V004 teardown first (FK-safe order): intent_verdicts/intent_applications ->
        # intent_records (referenced by the ewh FK) -> llm_calls. Drop the ewh FK
        # + partial UNIQUE indexes before intent_records so the DROP TABLE succeeds.
        conn.execute("ALTER TABLE ensemble_weight_history DROP CONSTRAINT IF EXISTS fk_ewh_intent")
        conn.execute("DROP INDEX IF EXISTS uq_mt_commentary_per_cycle")
        conn.execute("DROP INDEX IF EXISTS uq_llm_commentary_cycle")
        conn.execute("DROP TABLE IF EXISTS intent_verdicts")
        conn.execute("DROP TABLE IF EXISTS intent_applications")
        conn.execute("DROP TABLE IF EXISTS intent_records")
        conn.execute("DROP TABLE IF EXISTS llm_calls")
        # V003 teardown (FK-safe order): event_outcomes/fill_quality ->
        # calendar_events/trades.cycle_id -> cycle_algo_proposals -> inference_cycles.
        conn.execute("DROP TABLE IF EXISTS event_outcomes")
        conn.execute("DROP TABLE IF EXISTS calendar_events")
        conn.execute("DROP TABLE IF EXISTS fill_quality")
        conn.execute("ALTER TABLE trades DROP COLUMN IF EXISTS cycle_id")
        conn.execute("DROP TABLE IF EXISTS cycle_algo_proposals")
        conn.execute("DROP TABLE IF EXISTS inference_cycles")
        # V002 teardown: ensemble_weight_history -> models ->
        # training_runs, since training_runs.era_id references eras (dropped below).
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
    DatabaseManager.reset()
