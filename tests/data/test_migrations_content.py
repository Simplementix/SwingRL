"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0.

V002: identity-spine subset (training_runs, models, ensemble_weight_history) —
spine UNIQUE constraint makes duplicate runs impossible; retries are new attempt rows.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Generator
from pathlib import Path

import psycopg
import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


@pytest.fixture
def db_with_legacy_schema(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager on DATABASE_URL with init_schema() already run.

    Legacy tables (backtest_results, iteration_results, ...) exist, possibly
    empty — the V001 back-stamp must hold for 0 rows in CI and 574 in
    production. V001/V002's plain ``CREATE TABLE`` / ``ALTER TABLE ADD COLUMN``
    statements are not idempotent by themselves, so teardown drops the V002
    artifacts (training_runs/models/ensemble_weight_history, FK-safe order)
    and the V001 artifacts (the two new registry tables and the four added
    columns), then clears the version-1 and version-2 ledger rows, mirroring
    ``tests/data/test_migration_runner.py``'s ``db`` fixture cleanup — this
    keeps the persistent scratch database re-runnable across test sessions.
    """
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://test:test@localhost:5432/swingrl_test",  # pragma: allowlist secret
    )
    config_yaml = textwrap.dedent(f"""\
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
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(config_yaml)
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
