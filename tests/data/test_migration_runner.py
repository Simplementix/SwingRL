"""Migration runner tests.

D-T3.1/A7b: versioned ledger; which DDL is this database running becomes queryable.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.utils.exceptions import ConfigError

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


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
def db(tmp_path: Path, db_config_yaml: str) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager built from a tmp config whose system.database_url is DATABASE_URL."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    yield mgr
    with mgr.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS _mig_test_widgets")
        conn.execute(
            "DO $$ BEGIN "
            "IF to_regclass('public.schema_migrations') IS NOT NULL THEN "
            "DELETE FROM schema_migrations; "
            "END IF; "
            "END $$;"
        )
    DatabaseManager.reset()


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    d = tmp_path / "migrations"
    d.mkdir()
    (d / "V001__widgets.sql").write_text(
        "CREATE TABLE IF NOT EXISTS _mig_test_widgets (id BIGINT PRIMARY KEY);"
    )
    (d / "V002__widgets_name.sql").write_text("ALTER TABLE _mig_test_widgets ADD COLUMN name TEXT;")
    return d


def test_apply_migrations_applies_in_order_and_records(db, migrations_dir: Path) -> None:
    """A7b: runner applies V-files in order and records each in schema_migrations."""
    from swingrl.data.migration_runner import apply_migrations

    applied = apply_migrations(db, migrations_dir=migrations_dir)
    assert applied == 2
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT version, description FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [r["version"] for r in rows] == [1, 2]


def test_apply_migrations_is_idempotent(db, migrations_dir: Path) -> None:
    """Re-running applies nothing new."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db, migrations_dir=migrations_dir)
    assert apply_migrations(db, migrations_dir=migrations_dir) == 0


def test_assert_schema_current_raises_on_stale(db, migrations_dir: Path, monkeypatch) -> None:
    """Merged ≠ deployed guard: stale schema (DB behind) refuses to run."""
    import swingrl.data.migration_runner as mr

    mr.apply_migrations(db, migrations_dir=migrations_dir)
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 99)
    with pytest.raises(ConfigError):
        mr.assert_schema_current(db)


def test_assert_schema_current_warns_on_ahead(
    db, migrations_dir: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Floor semantics (A30): DB ahead of EXPECTED_SCHEMA_VERSION warns, does not raise.

    A running trader must survive a newer additive migration applied by a
    trainer-side deploy — only a DB *behind* the floor is genuinely broken.
    """
    import logging

    import swingrl.data.migration_runner as mr

    mr.apply_migrations(db, migrations_dir=migrations_dir)  # DB ends up at version 2
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 1)  # floor is behind actual
    with caplog.at_level(logging.WARNING):
        mr.assert_schema_current(db)  # must not raise
