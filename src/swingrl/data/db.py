"""Dual-database storage layer: DuckDB (market data) + SQLite (trading ops).

DatabaseManager is a singleton providing thread-safe context managers for both
databases. DuckDB handles OLAP workloads (OHLCV, features, quarantine); SQLite
handles OLTP workloads (trades, positions, risk decisions, alerts).

Usage:
    from swingrl.data.db import DatabaseManager
    from swingrl.config.schema import load_config

    config = load_config()
    db = DatabaseManager(config)
    db.init_schema()

    with db.duckdb() as cursor:
        cursor.execute("SELECT * FROM ohlcv_daily WHERE symbol = ?", ["SPY"])

    with db.sqlite() as conn:
        conn.execute("INSERT INTO trades ...")
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class DatabaseManager:
    """Singleton manager for DuckDB and SQLite database connections.

    Thread-safe via threading.Lock for singleton creation and DDL writes.
    DuckDB uses a persistent connection with per-call cursors.
    SQLite creates a new connection per context manager call (WAL mode).
    """

    _instance: DatabaseManager | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, config: SwingRLConfig | None = None) -> DatabaseManager:
        """Thread-safe singleton creation."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize database paths from config.

        Args:
            config: Validated SwingRLConfig with system.duckdb_path and
                    system.sqlite_path.
        """
        if self._initialized:
            return

        self._duckdb_path = Path(config.system.duckdb_path)
        self._sqlite_path = Path(config.system.sqlite_path)
        self._duckdb_conn: Any = None
        self._write_lock = threading.Lock()
        self._initialized = True

        log.info(
            "database_manager_initialized",
            duckdb_path=str(self._duckdb_path),
            sqlite_path=str(self._sqlite_path),
        )

    @classmethod
    def reset(cls) -> None:
        """Clear singleton instance and close connections. For test isolation."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    def _get_duckdb_conn(self) -> Any:
        """Lazy-initialize persistent DuckDB connection."""
        if self._duckdb_conn is None:
            self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            self._duckdb_conn = duckdb.connect(str(self._duckdb_path))
            log.info("duckdb_connected", path=str(self._duckdb_path))
        return self._duckdb_conn

    @contextlib.contextmanager
    def duckdb(self) -> Generator[Any, None, None]:
        """Yield a DuckDB cursor for thread-safe query execution.

        The cursor is closed on exit. The underlying connection persists.
        """
        conn = self._get_duckdb_conn()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    @contextlib.contextmanager
    def sqlite(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection in WAL mode with row_factory=sqlite3.Row.

        Auto-commits on success, rolls back on exception.
        A new connection is created per call (SQLite is lightweight).
        """
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def attach_sqlite(self, cursor: Any) -> None:  # noqa: ANN401
        """Attach SQLite database to DuckDB via sqlite_scanner for cross-DB joins.

        Args:
            cursor: Active DuckDB cursor to attach SQLite database to.
        """
        cursor.execute("INSTALL sqlite")
        cursor.execute("LOAD sqlite")
        cursor.execute(f"ATTACH '{self._sqlite_path}' AS ops (TYPE sqlite, READ_ONLY)")
        log.info("sqlite_attached", path=str(self._sqlite_path))

    def init_schema(self) -> None:
        """Create all tables, indexes, and views in both databases.

        Idempotent: uses CREATE TABLE IF NOT EXISTS / CREATE VIEW IF NOT EXISTS.
        Thread-safe via write lock.
        """
        with self._write_lock:
            self._init_duckdb_schema()
            self._init_sqlite_schema()
            log.info("schema_initialized")

    def _init_duckdb_schema(self) -> None:
        """Create DuckDB tables and aggregation views."""
        with self.duckdb() as cursor:
            # --- Tables ---
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_daily (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    adjusted_close DOUBLE,
                    fetched_at TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (symbol, date)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_4h (
                    symbol TEXT NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    source TEXT,
                    fetched_at TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (symbol, datetime)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_features (
                    date DATE NOT NULL,
                    series_id TEXT NOT NULL,
                    value DOUBLE,
                    release_date DATE,
                    PRIMARY KEY (date, series_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quarantine (
                    quarantined_at TIMESTAMP DEFAULT current_timestamp,
                    source TEXT,
                    symbol TEXT,
                    raw_data_json TEXT,
                    failure_reason TEXT,
                    severity TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_ingestion_log (
                    run_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT current_timestamp,
                    environment TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows_inserted INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    duration_ms INTEGER,
                    binance_weight_used INTEGER
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    version TEXT NOT NULL,
                    training_start_date TEXT,
                    training_end_date TEXT,
                    total_timesteps INTEGER,
                    converged_at_step INTEGER,
                    validation_sharpe DOUBLE,
                    ensemble_weight DOUBLE,
                    model_path TEXT NOT NULL,
                    vec_normalize_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    result_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    fold_number INTEGER NOT NULL,
                    fold_type TEXT NOT NULL,
                    train_start_idx INTEGER,
                    train_end_idx INTEGER,
                    test_start_idx INTEGER,
                    test_end_idx INTEGER,
                    sharpe DOUBLE,
                    sortino DOUBLE,
                    calmar DOUBLE,
                    mdd DOUBLE,
                    profit_factor DOUBLE,
                    win_rate DOUBLE,
                    total_trades INTEGER,
                    avg_drawdown DOUBLE,
                    max_dd_duration INTEGER,
                    final_portfolio_value DOUBLE,
                    total_return DOUBLE,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
            """)

            # --- Aggregation views ---
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS ohlcv_weekly AS
                SELECT
                    symbol,
                    date_trunc('week', date) AS week,
                    FIRST(open ORDER BY date) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close ORDER BY date) AS close,
                    SUM(volume) AS volume,
                    LAST(adjusted_close ORDER BY date) AS adjusted_close
                FROM ohlcv_daily
                GROUP BY symbol, date_trunc('week', date)
            """)

            cursor.execute("""
                CREATE VIEW IF NOT EXISTS ohlcv_monthly AS
                SELECT
                    symbol,
                    date_trunc('month', date) AS month,
                    FIRST(open ORDER BY date) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close ORDER BY date) AS close,
                    SUM(volume) AS volume,
                    LAST(adjusted_close ORDER BY date) AS adjusted_close
                FROM ohlcv_daily
                GROUP BY symbol, date_trunc('month', date)
            """)

    def _init_sqlite_schema(self) -> None:
        """Create SQLite tables and indexes."""
        with self.sqlite() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0.0,
                    slippage REAL DEFAULT 0.0,
                    environment TEXT NOT NULL,
                    broker TEXT,
                    order_type TEXT,
                    trade_type TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    last_price REAL,
                    unrealized_pnl REAL,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, environment)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    symbol TEXT,
                    proposed_action TEXT,
                    final_action TEXT,
                    risk_rule_triggered TEXT,
                    reason TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    equity_value REAL,
                    crypto_value REAL,
                    cash_balance REAL,
                    high_water_mark REAL,
                    daily_pnl REAL,
                    drawdown_pct REAL,
                    PRIMARY KEY (timestamp, environment)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    module TEXT,
                    event_type TEXT,
                    message TEXT,
                    metadata_json TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    action_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    effective_date TEXT NOT NULL,
                    ratio REAL,
                    amount REAL,
                    processed INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS wash_sale_tracker (
                    symbol TEXT NOT NULL,
                    sale_date TEXT NOT NULL,
                    loss_amount REAL NOT NULL,
                    wash_window_end TEXT NOT NULL,
                    triggered INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, sale_date)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                    event_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    resumed_at TEXT,
                    trigger_value REAL,
                    threshold REAL,
                    reason TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_positions (
                    spread_id TEXT PRIMARY KEY,
                    underlying TEXT NOT NULL,
                    strategy TEXT,
                    expiration TEXT,
                    short_strike REAL,
                    long_strike REAL,
                    premium_received REAL,
                    current_value REAL,
                    delta REAL,
                    theta REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_log (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT,
                    message_hash TEXT,
                    sent INTEGER DEFAULT 0
                )
            """)

            # --- Indexes ---
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cb_env_resumed
                ON circuit_breaker_events (environment, resumed_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_env
                ON trades (symbol, environment)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_symbol_env
                ON positions (symbol, environment)
            """)

    def close(self) -> None:
        """Gracefully close all database connections."""
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:  # noqa: BLE001
                log.warning("duckdb_close_error", error="failed to close connection gracefully")
            self._duckdb_conn = None
            log.info("duckdb_closed")
