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
          duckdb_path: data/db/market_data.ddb
          sqlite_path: data/db/trading_ops.db
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
def mock_db(tmp_path: Path, exec_config: SwingRLConfig) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager with SQLite pointing to tmp_path.

    Resets singleton and initializes schema for clean test isolation.
    """
    DatabaseManager.reset()
    # Patch system paths to tmp_path
    sqlite_path = tmp_path / "trading_ops.db"
    duckdb_path = tmp_path / "market_data.ddb"
    exec_config.system.sqlite_path = str(sqlite_path)
    exec_config.system.duckdb_path = str(duckdb_path)
    db = DatabaseManager(exec_config)
    db.init_schema()
    yield db
    DatabaseManager.reset()


@pytest.fixture
def position_tracker(mock_db: DatabaseManager, exec_config: SwingRLConfig) -> PositionTracker:
    """PositionTracker wired to mock_db."""
    from swingrl.execution.risk.position_tracker import (
        PositionTracker as _PositionTracker,
    )

    return _PositionTracker(db=mock_db, config=exec_config)
