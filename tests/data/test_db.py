"""DatabaseManager tests for PostgreSQL storage layer.

Tests cover singleton behavior, connection context manager, schema initialization,
aggregation views, and schema migrations.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from swingrl.config.schema import load_config

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
def db_config_yaml(tmp_path: Path) -> str:
    """Config YAML with system section pointing to DATABASE_URL."""
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://test:test@localhost:5432/swingrl_test"
    )  # pragma: allowlist secret
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
def db_config(tmp_path: Path, db_config_yaml: str) -> load_config:
    """Load config with DATABASE_URL."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(db_config_yaml)
    return load_config(config_file)


@pytest.fixture
def db_manager(db_config: object) -> object:
    """Create a DatabaseManager and ensure cleanup after test."""
    from swingrl.data.db import DatabaseManager

    DatabaseManager.reset()
    mgr = DatabaseManager(db_config)
    yield mgr
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
# Task 1: Config schema additions
# ---------------------------------------------------------------------------


class TestConfigSystemSection:
    """Config loads with new system and alerting sections."""

    def test_config_has_system_section(self, db_config: object) -> None:
        """DATA-06: Config includes system section with database_url."""
        assert hasattr(db_config, "system")
        assert hasattr(db_config.system, "database_url")

    def test_config_system_defaults(self, tmp_path: Path) -> None:
        """DATA-06: System section has correct defaults when not specified."""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("trading_mode: paper\n")
        config = load_config(config_file)
        assert hasattr(config.system, "database_url")

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

        DatabaseManager.reset()
        mgr1 = DatabaseManager(db_config)
        mgr2 = DatabaseManager(db_config)
        assert mgr1 is mgr2
        DatabaseManager.reset()

    def test_reset_clears_singleton(self, db_config: object) -> None:
        """DATA-07: reset() clears singleton for test isolation."""
        from swingrl.data.db import DatabaseManager

        DatabaseManager.reset()
        mgr1 = DatabaseManager(db_config)
        DatabaseManager.reset()
        mgr2 = DatabaseManager(db_config)
        assert mgr1 is not mgr2
        DatabaseManager.reset()


class TestPostgresContextManager:
    """PostgreSQL context manager tests."""

    def test_connection_context_yields_usable_conn(self, db_manager: object) -> None:
        """DATA-07: connection() context manager yields a usable PostgreSQL connection."""
        with db_manager.connection() as conn:
            result = conn.execute("SELECT 42 AS answer").fetchone()
            assert result["answer"] == 42

    def test_connection_autocommit_on_success(self, db_manager: object) -> None:
        """DATA-07: connection() auto-commits on successful exit."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test_commit (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test_commit VALUES (1)")
        # Verify data persisted with a new connection
        with db_manager.connection() as conn:
            row = conn.execute("SELECT id FROM test_commit").fetchone()
            assert row["id"] == 1
        # Clean up
        with db_manager.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS test_commit")

    def test_connection_rollback_on_exception(self, db_manager: object) -> None:
        """DATA-07: connection() rolls back on exception."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test_rollback (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test_rollback VALUES (1)")
        # Now attempt an insert that raises, verifying rollback
        with pytest.raises(RuntimeError):
            with db_manager.connection() as conn:
                conn.execute("INSERT INTO test_rollback VALUES (2)")
                raise RuntimeError("Intentional error")
        # Only the first insert should persist
        with db_manager.connection() as conn:
            rows = conn.execute("SELECT id FROM test_rollback").fetchall()
            assert len(rows) == 1
            assert rows[0]["id"] == 1
        # Clean up
        with db_manager.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS test_rollback")


# ---------------------------------------------------------------------------
# Task 2: Schema initialization
# ---------------------------------------------------------------------------

ALL_TABLES = [
    "ohlcv_daily",
    "ohlcv_4h",
    "macro_features",
    "data_quarantine",
    "data_ingestion_log",
    "features_equity",
    "features_crypto",
    "fundamentals",
    "hmm_state_history",
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


class TestInitSchema:
    """init_schema() creates all tables, indexes, and views in PostgreSQL."""

    def test_all_tables_created(self, db_manager: object) -> None:
        """DATA-06: init_schema() creates all expected PostgreSQL tables."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            result = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
            table_names = {row["tablename"] for row in result}
        for table in ALL_TABLES:
            assert table in table_names, f"Missing table: {table}"

    def test_idempotent(self, db_manager: object) -> None:
        """DATA-06: Running init_schema() twice does not error."""
        db_manager.init_schema()
        db_manager.init_schema()  # Should not raise


# ---------------------------------------------------------------------------
# Task 2: Aggregation views
# ---------------------------------------------------------------------------


class TestAggregationViews:
    """Weekly and monthly aggregation views on ohlcv_daily."""

    def test_weekly_view_aggregation(self, db_manager: object) -> None:
        """DATA-09: ohlcv_weekly view returns correct OHLCV aggregation."""
        db_manager.init_schema()

        # Insert 10 daily rows for SPY spanning 2 weeks
        daily_data = [
            ("SPY", "2024-01-08", 470.0, 475.0, 468.0, 473.0, 80_000_000, 473.0),
            ("SPY", "2024-01-09", 473.0, 478.0, 471.0, 476.0, 75_000_000, 476.0),
            ("SPY", "2024-01-10", 476.0, 480.0, 474.0, 477.0, 70_000_000, 477.0),
            ("SPY", "2024-01-11", 477.0, 479.0, 472.0, 474.0, 85_000_000, 474.0),
            ("SPY", "2024-01-12", 474.0, 476.0, 470.0, 475.0, 90_000_000, 475.0),
            ("SPY", "2024-01-15", 475.0, 482.0, 473.0, 480.0, 60_000_000, 480.0),
            ("SPY", "2024-01-16", 480.0, 485.0, 478.0, 483.0, 65_000_000, 483.0),
            ("SPY", "2024-01-17", 483.0, 486.0, 479.0, 481.0, 72_000_000, 481.0),
            ("SPY", "2024-01-18", 481.0, 484.0, 476.0, 478.0, 68_000_000, 478.0),
            ("SPY", "2024-01-19", 478.0, 483.0, 475.0, 482.0, 77_000_000, 482.0),
        ]

        with db_manager.connection() as conn:
            for row in daily_data:
                conn.execute(
                    "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, "
                    "volume, adjusted_close) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    row,
                )

        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ohlcv_weekly WHERE symbol = 'SPY' ORDER BY week"
            ).fetchall()

        assert len(rows) == 2

        # Week 1: first open=470, max high=480, min low=468, last close=475, sum vol=400M
        w1 = rows[0]
        assert w1["symbol"] == "SPY"
        assert w1["open"] == 470.0
        assert w1["high"] == 480.0
        assert w1["low"] == 468.0
        assert w1["close"] == 475.0
        assert w1["volume"] == 400_000_000
        assert w1["adjusted_close"] == 475.0

        # Week 2: first open=475, max high=486, min low=473, last close=482, sum vol=342M
        w2 = rows[1]
        assert w2["open"] == 475.0
        assert w2["high"] == 486.0
        assert w2["low"] == 473.0
        assert w2["close"] == 482.0
        assert w2["volume"] == 342_000_000
        assert w2["adjusted_close"] == 482.0

    def test_monthly_view_aggregation(self, db_manager: object) -> None:
        """DATA-09: ohlcv_monthly view returns correct OHLCV aggregation."""
        db_manager.init_schema()

        daily_data = [
            ("SPY", "2024-01-02", 470.0, 475.0, 468.0, 473.0, 80_000_000, 473.0),
            ("SPY", "2024-01-03", 473.0, 478.0, 471.0, 476.0, 75_000_000, 476.0),
            ("SPY", "2024-01-04", 476.0, 480.0, 465.0, 477.0, 70_000_000, 477.0),
        ]

        with db_manager.connection() as conn:
            for row in daily_data:
                conn.execute(
                    "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, "
                    "volume, adjusted_close) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    row,
                )

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT * FROM ohlcv_monthly WHERE symbol = 'SPY'").fetchall()

        assert len(rows) == 1
        m1 = rows[0]
        assert m1["symbol"] == "SPY"
        assert m1["open"] == 470.0
        assert m1["high"] == 480.0
        assert m1["low"] == 465.0
        assert m1["close"] == 477.0
        assert m1["volume"] == 225_000_000
        assert m1["adjusted_close"] == 477.0


# ---------------------------------------------------------------------------
# Phase 12 Task 1: Schema migrations for inference_outcomes, api_errors,
# positions columns
# ---------------------------------------------------------------------------


class TestPhase12SchemaMigrations:
    """init_schema() creates new tables and columns for Phase 12."""

    def test_inference_outcomes_table_created(self, db_manager: object) -> None:
        """INT-03: inference_outcomes table exists after init_schema()."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename = 'inference_outcomes'"
            ).fetchall()
        assert len(rows) == 1

    def test_inference_outcomes_columns(self, db_manager: object) -> None:
        """INT-03: inference_outcomes has id, timestamp, environment, had_nan."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            info = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                ("inference_outcomes",),
            ).fetchall()
            col_names = {row["column_name"] for row in info}
        assert {"id", "timestamp", "environment", "had_nan"} <= col_names

    def test_api_errors_table_created(self, db_manager: object) -> None:
        """INT-04: api_errors table exists after init_schema()."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename = 'api_errors'"
            ).fetchall()
        assert len(rows) == 1

    def test_api_errors_columns(self, db_manager: object) -> None:
        """INT-04: api_errors has id, timestamp, broker, status_code, endpoint, error_message."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            info = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                ("api_errors",),
            ).fetchall()
            col_names = {row["column_name"] for row in info}
        assert {
            "id",
            "timestamp",
            "broker",
            "status_code",
            "endpoint",
            "error_message",
        } <= col_names

    def test_positions_has_stop_columns(self, db_manager: object) -> None:
        """INT-03: positions table has stop_loss_price, take_profit_price, side."""
        db_manager.init_schema()
        with db_manager.connection() as conn:
            info = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                ("positions",),
            ).fetchall()
            col_names = {row["column_name"] for row in info}
        assert "stop_loss_price" in col_names
        assert "take_profit_price" in col_names
        assert "side" in col_names

    def test_positions_alter_idempotent(self, db_manager: object) -> None:
        """INT-03: Running init_schema() twice does not error on ALTER TABLE."""
        db_manager.init_schema()
        db_manager.init_schema()  # Second call should not raise
