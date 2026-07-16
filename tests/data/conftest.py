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
    hold for 0 rows in CI and 574 in production. V001/V002's plain
    ``CREATE TABLE`` / ``ALTER TABLE ADD COLUMN`` statements are not idempotent
    by themselves, so teardown drops the V002 artifacts (training_runs/models/
    ensemble_weight_history, FK-safe order) and the V001 artifacts (the two new
    registry tables and the four added columns), then clears the version-1 and
    version-2 ledger rows — keeping the persistent scratch database re-runnable
    across test sessions.

    LOUD NOTE for Task 8 (V003): when V003 ships, this teardown's
    dropped-artifacts list AND its ``WHERE version IN (1, 2)`` ledger cleanup
    must BOTH be extended to cover V003's artifacts/version — otherwise the
    scratch database stops being re-runnable the moment V003 lands.
    """
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    mgr.init_schema()
    yield mgr
    with mgr.connection() as conn:
        # V002 teardown first (FK-safe order): ensemble_weight_history -> models ->
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
            "DELETE FROM schema_migrations WHERE version IN (1, 2); "
            "END IF; "
            "END $$;"
        )
    DatabaseManager.reset()
