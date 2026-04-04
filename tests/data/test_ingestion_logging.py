"""Ingestion logging tests for BaseIngestor PostgreSQL integration.

Tests verify that every ingestor run() creates a data_ingestion_log row,
quarantine writes to PostgreSQL data_quarantine table, and DB sync works.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.base import BaseIngestor
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Skip entire module if no PostgreSQL available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# ---------------------------------------------------------------------------
# Test-only concrete ingestor
# ---------------------------------------------------------------------------


class _EquityTestIngestor(BaseIngestor):
    """Minimal concrete ingestor for testing. Simulates equity environment."""

    _environment = "equity"
    _duckdb_table = "ohlcv_daily"

    def __init__(self, config: SwingRLConfig, *, fetch_data: pd.DataFrame | None = None) -> None:
        super().__init__(config)
        self._fetch_data = fetch_data if fetch_data is not None else pd.DataFrame()
        self._quarantine_data = pd.DataFrame()

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Return pre-configured fetch data."""
        return self._fetch_data

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return clean data and pre-configured quarantine."""
        return df, self._quarantine_data

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """No-op store returning dummy path."""
        return Path("/dev/null")


class _CryptoTestIngestor(BaseIngestor):
    """Minimal concrete ingestor for testing. Simulates crypto environment."""

    _environment = "crypto"
    _duckdb_table = "ohlcv_4h"

    def __init__(self, config: SwingRLConfig, *, fetch_data: pd.DataFrame | None = None) -> None:
        super().__init__(config)
        self._fetch_data = fetch_data if fetch_data is not None else pd.DataFrame()
        self._quarantine_data = pd.DataFrame()

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Return pre-configured fetch data."""
        return self._fetch_data

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return clean data and pre-configured quarantine."""
        return df, self._quarantine_data

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """No-op store returning dummy path."""
        return Path("/dev/null")


class _FailingTestIngestor(BaseIngestor):
    """Ingestor that raises on fetch for testing failure logging."""

    _environment = "equity"
    _duckdb_table = "ohlcv_daily"

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Always raises."""
        msg = "Simulated fetch failure"
        raise RuntimeError(msg)

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Never called."""
        return df, pd.DataFrame()

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Never called."""
        return Path("/dev/null")


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
    """Create DatabaseManager with schema initialized, reset before and after test."""
    DatabaseManager.reset()  # Clear stale singleton from prior test files
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


@pytest.fixture
def equity_ohlcv_small() -> pd.DataFrame:
    """Small equity OHLCV DataFrame for sync tests."""
    import numpy as np

    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = 470.0 + rng.normal(0, 2, 5).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, 5),
            "high": close + rng.uniform(0, 2, 5),
            "low": close - rng.uniform(0, 2, 5),
            "close": close,
            "volume": rng.integers(50_000_000, 100_000_000, 5).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def crypto_ohlcv_small() -> pd.DataFrame:
    """Small crypto 4H OHLCV DataFrame for sync tests."""
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    rng = np.random.default_rng(43)
    close = 42_000.0 + rng.normal(0, 200, 5).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, 5),
            "high": close + rng.uniform(0, 150, 5),
            "low": close - rng.uniform(0, 150, 5),
            "close": close,
            "volume": rng.uniform(500, 2000, 5),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests: Ingestion logging
# ---------------------------------------------------------------------------


class TestIngestionLogOnSuccess:
    """run() creates data_ingestion_log row on successful ingestion."""

    def test_success_creates_log_row(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: run() creates log row with status=success and correct counts."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT run_id, environment, symbol, status, rows_inserted, "
                "errors_count, duration_ms FROM data_ingestion_log"
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert len(row["run_id"]) == 36  # UUID format
        assert row["environment"] == "equity"
        assert row["symbol"] == "SPY"
        assert row["status"] == "success"
        assert row["rows_inserted"] == len(equity_ohlcv_small)
        assert row["errors_count"] == 0
        assert row["duration_ms"] > 0

    def test_no_data_creates_log_row(
        self, db_config: SwingRLConfig, db_manager: DatabaseManager
    ) -> None:
        """DATA-12: run() on empty fetch creates log row with status=no_data."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=pd.DataFrame())
        ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT status, rows_inserted FROM data_ingestion_log").fetchall()

        assert len(rows) == 1
        assert rows[0]["status"] == "no_data"
        assert rows[0]["rows_inserted"] == 0


class TestIngestionLogOnFailure:
    """run() creates data_ingestion_log row on failure."""

    def test_failure_creates_log_row(
        self, db_config: SwingRLConfig, db_manager: DatabaseManager
    ) -> None:
        """DATA-12: run() on exception creates log row with status=failed."""
        ingestor = _FailingTestIngestor(db_config)
        with pytest.raises(RuntimeError):
            ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT status, rows_inserted, duration_ms FROM data_ingestion_log"
            ).fetchall()

        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["rows_inserted"] == 0
        assert rows[0]["duration_ms"] >= 0


class TestIngestionLogWithQuarantine:
    """run() with quarantined rows logs errors_count."""

    def test_quarantine_count_in_log(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: run() with quarantined rows logs errors_count."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        # Set up quarantine: 2 bad rows
        bad_rows = equity_ohlcv_small.iloc[:2].copy()
        bad_rows["reason"] = "test quarantine"
        ingestor._quarantine_data = bad_rows

        ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT errors_count FROM data_ingestion_log").fetchall()

        assert len(rows) == 1
        assert rows[0]["errors_count"] == 2


class TestIngestionLogBinanceWeight:
    """binance_weight_used captured when set by BinanceIngestor."""

    def test_binance_weight_in_log(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        crypto_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: binance_weight_used captured in log when available."""
        ingestor = _CryptoTestIngestor(db_config, fetch_data=crypto_ohlcv_small)
        ingestor._binance_weight_used = 450
        ingestor.run("BTCUSDT")

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT binance_weight_used FROM data_ingestion_log").fetchall()

        assert len(rows) == 1
        assert rows[0]["binance_weight_used"] == 450


# ---------------------------------------------------------------------------
# Tests: DB sync
# ---------------------------------------------------------------------------


class TestSyncToDB:
    """_sync_to_duckdb() inserts rows into correct PostgreSQL table."""

    def test_equity_sync_to_ohlcv_daily(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: _sync_to_duckdb() inserts into ohlcv_daily for equity."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, date, open, high, low, close, volume "
                "FROM ohlcv_daily WHERE symbol = 'SPY' ORDER BY date"
            ).fetchall()

        assert len(rows) == len(equity_ohlcv_small)

    def test_crypto_sync_to_ohlcv_4h(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        crypto_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: _sync_to_duckdb() inserts into ohlcv_4h for crypto."""
        ingestor = _CryptoTestIngestor(db_config, fetch_data=crypto_ohlcv_small)
        ingestor.run("BTCUSDT")

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, datetime, open, high, low, close, volume "
                "FROM ohlcv_4h WHERE symbol = 'BTCUSDT' ORDER BY datetime"
            ).fetchall()

        assert len(rows) == len(crypto_ohlcv_small)

    def test_sync_idempotent(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: _sync_to_duckdb() is idempotent - no duplicates on rerun."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        ingestor.run("SPY")
        ingestor.run("SPY")  # Second run with same data

        with db_manager.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM ohlcv_daily WHERE symbol = 'SPY'"
            ).fetchone()["cnt"]

        assert count == len(equity_ohlcv_small)


# ---------------------------------------------------------------------------
# Tests: Quarantine to DB
# ---------------------------------------------------------------------------


class TestQuarantineToDB:
    """_store_quarantine() writes to PostgreSQL data_quarantine table."""

    def test_quarantine_written_to_db(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
    ) -> None:
        """DATA-12: _store_quarantine() writes to data_quarantine with JSON."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        bad_rows = equity_ohlcv_small.iloc[:2].copy()
        bad_rows["reason"] = "test quarantine"
        ingestor._quarantine_data = bad_rows

        ingestor.run("SPY")

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT source, symbol, raw_data_json, failure_reason, severity "
                "FROM data_quarantine ORDER BY quarantined_at"
            ).fetchall()

        assert len(rows) == 2
        assert rows[0]["source"] == "equity"
        assert rows[0]["symbol"] == "SPY"
        assert rows[0]["raw_data_json"] is not None
        assert rows[0]["failure_reason"] == "test quarantine"
        assert rows[0]["severity"] == "warning"

    def test_quarantine_still_writes_parquet(
        self,
        db_config: SwingRLConfig,
        db_manager: DatabaseManager,
        equity_ohlcv_small: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """DATA-12: _store_quarantine() still writes Parquet for backward compat."""
        ingestor = _EquityTestIngestor(db_config, fetch_data=equity_ohlcv_small)
        bad_rows = equity_ohlcv_small.iloc[:1].copy()
        bad_rows["reason"] = "test quarantine"
        ingestor._quarantine_data = bad_rows

        ingestor.run("SPY")

        # Check that a quarantine parquet file was written
        q_dir = Path(db_config.paths.data_dir) / "quarantine"
        parquet_files = list(q_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1
