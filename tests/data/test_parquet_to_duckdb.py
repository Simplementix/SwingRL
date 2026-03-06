"""Parquet-to-DuckDB bulk migration tests.

Tests verify that existing Parquet files from Phase 3 can be loaded into
DuckDB tables via sync_parquet_to_duckdb() without duplicates.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import sync_parquet_to_duckdb
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_config(tmp_path: Path) -> SwingRLConfig:
    """Config with tmp_path DB paths for test isolation."""
    duckdb_path = str(tmp_path / "market_data.ddb")
    sqlite_path = str(tmp_path / "trading_ops.db")
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
          data_dir: {tmp_path}/data
          db_dir: {tmp_path}/db
          models_dir: models/
          logs_dir: logs/
        logging:
          level: INFO
          json_logs: false
        system:
          duckdb_path: "{duckdb_path}"
          sqlite_path: "{sqlite_path}"
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(config_yaml)
    return load_config(config_file)


@pytest.fixture
def db_manager(db_config: SwingRLConfig) -> DatabaseManager:
    """Create DatabaseManager with schema initialized, reset after test."""
    mgr = DatabaseManager(db_config)
    mgr.init_schema()
    yield mgr  # type: ignore[misc]
    DatabaseManager.reset()


# ---------------------------------------------------------------------------
# Tests: Bulk Parquet-to-DuckDB migration
# ---------------------------------------------------------------------------


class TestParquetToDuckDBMigration:
    """sync_parquet_to_duckdb() loads Parquet files into DuckDB."""

    def test_equity_parquet_loads_to_ohlcv_daily(
        self, db_config: SwingRLConfig, db_manager: DatabaseManager, tmp_path: Path
    ) -> None:
        """DATA-12: Equity Parquet file loads into ohlcv_daily."""
        import numpy as np

        dates = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
        rng = np.random.default_rng(42)
        close = 470.0 + rng.normal(0, 2, 10).cumsum()
        df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 1, 10),
                "high": close + rng.uniform(0, 2, 10),
                "low": close - rng.uniform(0, 2, 10),
                "close": close,
                "volume": rng.integers(50_000_000, 100_000_000, 10).astype(float),
            },
            index=dates,
        )

        parquet_path = tmp_path / "SPY_daily.parquet"
        df.to_parquet(parquet_path)

        count = sync_parquet_to_duckdb(
            parquet_path=parquet_path,
            table="ohlcv_daily",
            symbol="SPY",
            db=db_manager,
        )

        assert count == 10

        with db_manager.duckdb() as cursor:
            rows = cursor.execute(
                "SELECT COUNT(*) FROM ohlcv_daily WHERE symbol = 'SPY'"
            ).fetchone()[0]
        assert rows == 10

    def test_crypto_parquet_loads_to_ohlcv_4h(
        self, db_config: SwingRLConfig, db_manager: DatabaseManager, tmp_path: Path
    ) -> None:
        """DATA-12: Crypto Parquet file loads into ohlcv_4h."""
        import numpy as np

        dates = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        rng = np.random.default_rng(43)
        close = 42_000.0 + rng.normal(0, 200, 10).cumsum()
        df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 50, 10),
                "high": close + rng.uniform(0, 150, 10),
                "low": close - rng.uniform(0, 150, 10),
                "close": close,
                "volume": rng.uniform(500, 2000, 10),
            },
            index=dates,
        )

        parquet_path = tmp_path / "BTCUSDT_4h.parquet"
        df.to_parquet(parquet_path)

        count = sync_parquet_to_duckdb(
            parquet_path=parquet_path,
            table="ohlcv_4h",
            symbol="BTCUSDT",
            db=db_manager,
        )

        assert count == 10

        with db_manager.duckdb() as cursor:
            rows = cursor.execute(
                "SELECT COUNT(*) FROM ohlcv_4h WHERE symbol = 'BTCUSDT'"
            ).fetchone()[0]
        assert rows == 10

    def test_migration_idempotent(
        self, db_config: SwingRLConfig, db_manager: DatabaseManager, tmp_path: Path
    ) -> None:
        """DATA-12: Running migration twice does not create duplicates."""
        import numpy as np

        dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
        rng = np.random.default_rng(42)
        close = 470.0 + rng.normal(0, 2, 5).cumsum()
        df = pd.DataFrame(
            {
                "open": close - rng.uniform(0, 1, 5),
                "high": close + rng.uniform(0, 2, 5),
                "low": close - rng.uniform(0, 2, 5),
                "close": close,
                "volume": rng.integers(50_000_000, 100_000_000, 5).astype(float),
            },
            index=dates,
        )

        parquet_path = tmp_path / "SPY_daily.parquet"
        df.to_parquet(parquet_path)

        sync_parquet_to_duckdb(
            parquet_path=parquet_path, table="ohlcv_daily", symbol="SPY", db=db_manager
        )
        sync_parquet_to_duckdb(
            parquet_path=parquet_path, table="ohlcv_daily", symbol="SPY", db=db_manager
        )

        with db_manager.duckdb() as cursor:
            count = cursor.execute(
                "SELECT COUNT(*) FROM ohlcv_daily WHERE symbol = 'SPY'"
            ).fetchone()[0]
        assert count == 5
