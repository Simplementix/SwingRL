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


# ---------------------------------------------------------------------------
# Task 2: Schema initialization
# ---------------------------------------------------------------------------

DUCKDB_TABLES = [
    "ohlcv_daily",
    "ohlcv_4h",
    "macro_features",
    "data_quarantine",
    "data_ingestion_log",
]

FEATURE_TABLES = [
    "features_equity",
    "features_crypto",
    "fundamentals",
    "hmm_state_history",
]

SQLITE_TABLES = [
    "trades",
    "positions",
    "risk_decisions",
    "portfolio_snapshots",
    "system_events",
    "corporate_actions",
    "wash_sale_tracker",
    "circuit_breaker_events",
    "options_positions",
    "alert_log",
]


class TestFeatureTableInit:
    """init_schema() creates Phase 5 feature tables in DuckDB."""

    def test_init_schema_creates_feature_tables(self, db_manager: object) -> None:
        """DATA-09: init_schema() creates features_equity, features_crypto, fundamentals, hmm_state_history."""
        db_manager.init_schema()
        with db_manager.duckdb() as cursor:
            result = cursor.execute("SHOW TABLES").fetchall()
            table_names = {row[0] for row in result}
        for table in FEATURE_TABLES:
            assert table in table_names, f"Missing feature table: {table}"


class TestInitSchema:
    """init_schema() creates all tables, indexes, and views."""

    def test_duckdb_tables_created(self, db_manager: object) -> None:
        """DATA-06: init_schema() creates all 5 DuckDB tables."""
        db_manager.init_schema()
        with db_manager.duckdb() as cursor:
            result = cursor.execute("SHOW TABLES").fetchall()
            table_names = {row[0] for row in result}
        for table in DUCKDB_TABLES:
            assert table in table_names, f"Missing DuckDB table: {table}"

    def test_sqlite_tables_created(self, db_manager: object) -> None:
        """DATA-06: init_schema() creates all 10 SQLite tables."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {row["name"] for row in rows}
        for table in SQLITE_TABLES:
            assert table in table_names, f"Missing SQLite table: {table}"

    def test_sqlite_indexes_created(self, db_manager: object) -> None:
        """DATA-06: init_schema() creates expected SQLite indexes."""
        db_manager.init_schema()
        expected_indexes = {
            "idx_cb_env_resumed",
            "idx_trades_symbol_env",
            "idx_positions_symbol_env",
        }
        with db_manager.sqlite() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            index_names = {row["name"] for row in rows}
        for idx in expected_indexes:
            assert idx in index_names, f"Missing SQLite index: {idx}"

    def test_idempotent(self, db_manager: object) -> None:
        """DATA-06: Running init_schema() twice does not error."""
        db_manager.init_schema()
        db_manager.init_schema()  # Should not raise


# ---------------------------------------------------------------------------
# Task 2: Cross-DB join via sqlite_scanner
# ---------------------------------------------------------------------------


class TestCrossDBJoin:
    """Cross-DB join between DuckDB and SQLite via sqlite_scanner."""

    def test_attach_sqlite_enables_join(self, db_manager: object) -> None:
        """DATA-08: INSERT into DuckDB and SQLite, JOIN returns correct row."""
        db_manager.init_schema()

        # Insert into DuckDB ohlcv_daily
        with db_manager.duckdb() as cursor:
            cursor.execute(
                "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
                "VALUES ('SPY', '2024-01-02', 470.0, 475.0, 468.0, 473.0, 80000000)"
            )

        # Insert into SQLite trades
        with db_manager.sqlite() as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, "
                "price, environment) VALUES ('t1', '2024-01-02T10:00:00', 'SPY', "
                "'buy', 10, 472.0, 'equity')"
            )

        # Cross-DB join
        with db_manager.duckdb() as cursor:
            db_manager.attach_sqlite(cursor)
            result = cursor.execute("""
                SELECT d.symbol, d.close, t.price, t.side
                FROM ohlcv_daily d
                JOIN ops.trades t ON d.symbol = t.symbol
                WHERE d.symbol = 'SPY'
            """).fetchone()

        assert result is not None
        assert result[0] == "SPY"
        assert result[1] == 473.0
        assert result[2] == 472.0
        assert result[3] == "buy"


# ---------------------------------------------------------------------------
# Task 2: Aggregation views
# ---------------------------------------------------------------------------


class TestAggregationViews:
    """Weekly and monthly aggregation views on ohlcv_daily."""

    def test_weekly_view_aggregation(self, db_manager: object) -> None:
        """DATA-09: ohlcv_weekly view returns correct OHLCV aggregation."""
        db_manager.init_schema()

        # Insert 10 daily rows for SPY spanning 2 weeks
        # Week 1: Mon 2024-01-08 to Fri 2024-01-12
        # Week 2: Mon 2024-01-15 to Fri 2024-01-19
        daily_data = [
            # Week 1
            ("SPY", "2024-01-08", 470.0, 475.0, 468.0, 473.0, 80_000_000, 473.0),
            ("SPY", "2024-01-09", 473.0, 478.0, 471.0, 476.0, 75_000_000, 476.0),
            ("SPY", "2024-01-10", 476.0, 480.0, 474.0, 477.0, 70_000_000, 477.0),
            ("SPY", "2024-01-11", 477.0, 479.0, 472.0, 474.0, 85_000_000, 474.0),
            ("SPY", "2024-01-12", 474.0, 476.0, 470.0, 475.0, 90_000_000, 475.0),
            # Week 2
            ("SPY", "2024-01-15", 475.0, 482.0, 473.0, 480.0, 60_000_000, 480.0),
            ("SPY", "2024-01-16", 480.0, 485.0, 478.0, 483.0, 65_000_000, 483.0),
            ("SPY", "2024-01-17", 483.0, 486.0, 479.0, 481.0, 72_000_000, 481.0),
            ("SPY", "2024-01-18", 481.0, 484.0, 476.0, 478.0, 68_000_000, 478.0),
            ("SPY", "2024-01-19", 478.0, 483.0, 475.0, 482.0, 77_000_000, 482.0),
        ]

        with db_manager.duckdb() as cursor:
            cursor.executemany(
                "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, "
                "volume, adjusted_close) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                daily_data,
            )

        with db_manager.duckdb() as cursor:
            rows = cursor.execute(
                "SELECT * FROM ohlcv_weekly WHERE symbol = 'SPY' ORDER BY week"
            ).fetchall()

        assert len(rows) == 2

        # Week 1: first open=470, max high=480, min low=468, last close=475, sum vol=400M
        w1 = rows[0]
        assert w1[0] == "SPY"  # symbol
        assert w1[2] == 470.0  # first open
        assert w1[3] == 480.0  # max high
        assert w1[4] == 468.0  # min low
        assert w1[5] == 475.0  # last close
        assert w1[6] == 400_000_000  # sum volume
        assert w1[7] == 475.0  # last adjusted_close

        # Week 2: first open=475, max high=486, min low=473, last close=482, sum vol=342M
        w2 = rows[1]
        assert w2[2] == 475.0
        assert w2[3] == 486.0
        assert w2[4] == 473.0
        assert w2[5] == 482.0
        assert w2[6] == 342_000_000
        assert w2[7] == 482.0

    def test_monthly_view_aggregation(self, db_manager: object) -> None:
        """DATA-09: ohlcv_monthly view returns correct OHLCV aggregation."""
        db_manager.init_schema()

        # Insert rows in January 2024
        daily_data = [
            ("SPY", "2024-01-02", 470.0, 475.0, 468.0, 473.0, 80_000_000, 473.0),
            ("SPY", "2024-01-03", 473.0, 478.0, 471.0, 476.0, 75_000_000, 476.0),
            ("SPY", "2024-01-04", 476.0, 480.0, 465.0, 477.0, 70_000_000, 477.0),
        ]

        with db_manager.duckdb() as cursor:
            cursor.executemany(
                "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, "
                "volume, adjusted_close) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                daily_data,
            )

        with db_manager.duckdb() as cursor:
            rows = cursor.execute("SELECT * FROM ohlcv_monthly WHERE symbol = 'SPY'").fetchall()

        assert len(rows) == 1
        m1 = rows[0]
        assert m1[0] == "SPY"  # symbol
        assert m1[2] == 470.0  # first open
        assert m1[3] == 480.0  # max high
        assert m1[4] == 465.0  # min low
        assert m1[5] == 477.0  # last close
        assert m1[6] == 225_000_000  # sum volume
        assert m1[7] == 477.0  # last adjusted_close


# ---------------------------------------------------------------------------
# Phase 12 Task 1: Schema migrations for inference_outcomes, api_errors,
# positions columns
# ---------------------------------------------------------------------------


class TestPhase12SchemaMigrations:
    """init_schema() creates new tables and columns for Phase 12."""

    def test_inference_outcomes_table_created(self, db_manager: object) -> None:
        """INT-03: inference_outcomes table exists after init_schema()."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='inference_outcomes'"
            ).fetchall()
        assert len(rows) == 1

    def test_inference_outcomes_columns(self, db_manager: object) -> None:
        """INT-03: inference_outcomes has id, timestamp, environment, had_nan."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            info = conn.execute("PRAGMA table_info(inference_outcomes)").fetchall()
            col_names = {row["name"] for row in info}
        assert {"id", "timestamp", "environment", "had_nan"} <= col_names

    def test_api_errors_table_created(self, db_manager: object) -> None:
        """INT-04: api_errors table exists after init_schema()."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='api_errors'"
            ).fetchall()
        assert len(rows) == 1

    def test_api_errors_columns(self, db_manager: object) -> None:
        """INT-04: api_errors has id, timestamp, broker, status_code, endpoint, error_message."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            info = conn.execute("PRAGMA table_info(api_errors)").fetchall()
            col_names = {row["name"] for row in info}
        assert {
            "id",
            "timestamp",
            "broker",
            "status_code",
            "endpoint",
            "error_message",
        } <= col_names

    def test_positions_has_stop_columns(self, db_manager: object) -> None:
        """INT-03: positions table has stop_loss_price, take_profit_price, side after init_schema()."""
        db_manager.init_schema()
        with db_manager.sqlite() as conn:
            info = conn.execute("PRAGMA table_info(positions)").fetchall()
            col_names = {row["name"] for row in info}
        assert "stop_loss_price" in col_names
        assert "take_profit_price" in col_names
        assert "side" in col_names

    def test_positions_alter_idempotent(self, db_manager: object) -> None:
        """INT-03: Running init_schema() twice does not error on ALTER TABLE."""
        db_manager.init_schema()
        db_manager.init_schema()  # Second call should not raise
