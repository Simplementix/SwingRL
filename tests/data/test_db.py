"""DatabaseManager tests for dual-database storage layer.

Tests cover singleton behavior, context managers, schema initialization,
cross-DB joins, and aggregation views.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import duckdb
import pytest

from swingrl.config.schema import load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_config_yaml(tmp_path: Path) -> str:
    """Config YAML with system paths pointing to tmp_path."""
    duckdb_path = str(tmp_path / "market_data.ddb")
    sqlite_path = str(tmp_path / "trading_ops.db")
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
          duckdb_path: "{duckdb_path}"
          sqlite_path: "{sqlite_path}"
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def db_config(tmp_path: Path, db_config_yaml: str) -> load_config:
    """Load config with tmp_path DB paths."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    return load_config(config_file)


@pytest.fixture
def db_manager(db_config: object) -> object:
    """Create a DatabaseManager and ensure cleanup after test."""
    from swingrl.data.db import DatabaseManager

    mgr = DatabaseManager(db_config)
    yield mgr
    DatabaseManager.reset()


# ---------------------------------------------------------------------------
# Task 1: Config schema additions
# ---------------------------------------------------------------------------


class TestConfigSystemSection:
    """Config loads with new system and alerting sections."""

    def test_config_has_system_section(self, db_config: object) -> None:
        """DATA-06: Config includes system section with DB paths."""
        assert hasattr(db_config, "system")
        assert hasattr(db_config.system, "duckdb_path")
        assert hasattr(db_config.system, "sqlite_path")

    def test_config_system_defaults(self, tmp_path: Path) -> None:
        """DATA-06: System section has correct defaults when not specified."""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("trading_mode: paper\n")
        config = load_config(config_file)
        assert config.system.duckdb_path == "data/db/market_data.ddb"
        assert config.system.sqlite_path == "data/db/trading_ops.db"

    def test_config_has_alerting_section(self, db_config: object) -> None:
        """DATA-06: Config includes alerting section."""
        assert hasattr(db_config, "alerting")
        assert db_config.alerting.alert_cooldown_minutes == 30
        assert db_config.alerting.consecutive_failures_before_alert == 3

    def test_config_alerting_defaults(self, tmp_path: Path) -> None:
        """DATA-06: Alerting section has correct defaults when not specified."""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("trading_mode: paper\n")
        config = load_config(config_file)
        assert config.alerting.alert_cooldown_minutes == 30
        assert config.alerting.consecutive_failures_before_alert == 3


# ---------------------------------------------------------------------------
# Task 1: DatabaseManager singleton and context managers
# ---------------------------------------------------------------------------


class TestDatabaseManagerSingleton:
    """DatabaseManager singleton pattern tests."""

    def test_singleton_returns_same_instance(self, db_config: object) -> None:
        """DATA-07: Two instantiations return the same object."""
        from swingrl.data.db import DatabaseManager

        mgr1 = DatabaseManager(db_config)
        mgr2 = DatabaseManager(db_config)
        assert mgr1 is mgr2
        DatabaseManager.reset()

    def test_reset_clears_singleton(self, db_config: object) -> None:
        """DATA-07: reset() clears singleton for test isolation."""
        from swingrl.data.db import DatabaseManager

        mgr1 = DatabaseManager(db_config)
        DatabaseManager.reset()
        mgr2 = DatabaseManager(db_config)
        assert mgr1 is not mgr2
        DatabaseManager.reset()


class TestDuckDBContextManager:
    """DuckDB context manager tests."""

    def test_duckdb_context_yields_cursor(self, db_manager: object) -> None:
        """DATA-07: duckdb() context manager yields a usable DuckDB cursor."""
        with db_manager.duckdb() as cursor:
            result = cursor.execute("SELECT 42 AS answer").fetchone()
            assert result[0] == 42

    def test_duckdb_cursor_closed_on_exit(self, db_manager: object) -> None:
        """DATA-07: duckdb() closes cursor on exit."""
        with db_manager.duckdb() as cursor:
            cursor.execute("SELECT 1")
        # After exiting, cursor should be closed
        # DuckDB cursors don't have a .closed attribute, but
        # attempting to use it after close should raise InvalidInputException
        with pytest.raises(duckdb.ConnectionException):
            cursor.execute("SELECT 1")


class TestSQLiteContextManager:
    """SQLite context manager tests."""

    def test_sqlite_context_yields_connection(self, db_manager: object) -> None:
        """DATA-07: sqlite() context manager yields a usable SQLite connection."""
        with db_manager.sqlite() as conn:
            result = conn.execute("SELECT 42 AS answer").fetchone()
            assert result["answer"] == 42

    def test_sqlite_wal_mode(self, db_manager: object) -> None:
        """DATA-07: sqlite() connection uses WAL journal mode."""
        with db_manager.sqlite() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"

    def test_sqlite_row_factory(self, db_manager: object) -> None:
        """DATA-07: sqlite() connection has row_factory=sqlite3.Row."""
        with db_manager.sqlite() as conn:
            assert conn.row_factory is sqlite3.Row

    def test_sqlite_autocommit_on_success(self, db_manager: object) -> None:
        """DATA-07: sqlite() auto-commits on successful exit."""
        with db_manager.sqlite() as conn:
            conn.execute("CREATE TABLE test_commit (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test_commit VALUES (1)")
        # Verify data persisted with a new connection
        with db_manager.sqlite() as conn:
            row = conn.execute("SELECT id FROM test_commit").fetchone()
            assert row["id"] == 1

    def test_sqlite_rollback_on_exception(self, db_manager: object) -> None:
        """DATA-07: sqlite() rolls back on exception."""
        with db_manager.sqlite() as conn:
            conn.execute("CREATE TABLE test_rollback (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test_rollback VALUES (1)")
        # Now attempt an insert that raises, verifying rollback
        with pytest.raises(RuntimeError):
            with db_manager.sqlite() as conn:
                conn.execute("INSERT INTO test_rollback VALUES (2)")
                raise RuntimeError("Intentional error")
        # Only the first insert should persist
        with db_manager.sqlite() as conn:
            rows = conn.execute("SELECT id FROM test_rollback").fetchall()
            assert len(rows) == 1
            assert rows[0]["id"] == 1
