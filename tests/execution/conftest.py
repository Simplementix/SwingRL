"""Shared fixtures for execution layer tests."""

from __future__ import annotations

import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager

if TYPE_CHECKING:
    from swingrl.execution.risk.position_tracker import PositionTracker


@pytest.fixture
def exec_config_yaml() -> str:
    """Config YAML with 8 equity symbols and 2 crypto symbols for execution tests."""
    return textwrap.dedent("""\
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]
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
        environment:
          initial_amount: 100000.0
          equity_episode_bars: 252
          crypto_episode_bars: 540
          equity_transaction_cost_pct: 0.0006
          crypto_transaction_cost_pct: 0.0022
          signal_deadzone: 0.02
          position_penalty_coeff: 10.0
          drawdown_penalty_coeff: 5.0
        system:
          database_url: ""
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def exec_config(tmp_path: Path, exec_config_yaml: str) -> SwingRLConfig:
    """Validated SwingRLConfig for execution tests."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(exec_config_yaml)
    return load_config(config_file)


@pytest.fixture
def mock_db(exec_config: SwingRLConfig) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager backed by PostgreSQL test database.

    Resets singleton and initializes schema for clean test isolation.
    Requires DATABASE_URL env var pointing to a test PostgreSQL instance.
    """
    import os  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available for testing")

    DatabaseManager.reset()
    exec_config.system.database_url = db_url
    db = DatabaseManager(exec_config)
    db.init_schema()
    yield db
    # Truncate all tables for test isolation
    with db.connection() as conn:
        conn.execute(
            "DO $$ DECLARE r RECORD; BEGIN "
            "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
            "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
            "END LOOP; END $$"
        )
    DatabaseManager.reset()


@pytest.fixture
def position_tracker(mock_db: DatabaseManager, exec_config: SwingRLConfig) -> PositionTracker:
    """PositionTracker wired to mock_db."""
    from swingrl.execution.risk.position_tracker import (
        PositionTracker as _PositionTracker,
    )

    return _PositionTracker(db=mock_db, config=exec_config)
