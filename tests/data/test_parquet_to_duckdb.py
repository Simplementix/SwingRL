"""Parquet-to-PostgreSQL bulk migration tests.

Tests verify that existing Parquet files from Phase 3 can be loaded into
PostgreSQL tables via sync_parquet_to_duckdb() without duplicates.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import sync_parquet_to_duckdb
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Skip entire module if no PostgreSQL available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_config(tmp_path: Path) -> SwingRLConfig:
    """Config with DATABASE_URL for test isolation."""
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://test:test@localhost:5432/swingrl_test"
    )  # pragma: allowlist secret
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
          database_url: "{db_url}"
          duckdb_path: data/db/market_data.ddb
          sqlite_path: data/db/trading_ops.db
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
    DatabaseManager.reset()
    mgr = DatabaseManager(db_config)
    mgr.init_schema()
    yield mgr  # type: ignore[misc]
    # Truncate all tables for test isolation
    with mgr.connection() as conn:
        conn.execute(
            "DO $$ DECLARE r RECORD; BEGIN "
            "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
            "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
            "END LOOP; END $$"
        )
    DatabaseManager.reset()


# ---------------------------------------------------------------------------
# Tests: Bulk Parquet-to-PostgreSQL migration
# ---------------------------------------------------------------------------


class TestParquetToPostgreSQLMigration:
    """sync_parquet_to_duckdb() loads Parquet files into PostgreSQL."""

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

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ohlcv_daily WHERE symbol = 'SPY'"
            ).fetchone()["cnt"]
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

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ohlcv_4h WHERE symbol = 'BTCUSDT'"
            ).fetchone()["cnt"]
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

        with db_manager.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ohlcv_daily WHERE symbol = 'SPY'"
            ).fetchone()["cnt"]
        assert count == 5
