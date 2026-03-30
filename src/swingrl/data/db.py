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

    with db.duckdb() as conn:
        conn.execute("SELECT * FROM ohlcv_daily WHERE symbol = ?", ["SPY"])

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

import duckdb as duckdb_module
import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

# Columns added to backtest_results for iteration tracking (Phase 20.1).
# Used by _migrate_backtest_results() to idempotently add missing columns.
_BACKTEST_RESULTS_MIGRATIONS: list[tuple[str, str]] = [
    ("iteration_number", "INTEGER DEFAULT 0"),
    ("run_type", "TEXT DEFAULT 'baseline'"),
    ("is_sharpe", "DOUBLE"),
    ("is_sortino", "DOUBLE"),
    ("is_mdd", "DOUBLE"),
    ("is_total_return", "DOUBLE"),
    ("overfitting_gap", "DOUBLE"),
    ("overfitting_class", "TEXT"),
    ("hmm_p_bull", "DOUBLE"),
    ("hmm_p_bear", "DOUBLE"),
    ("vix_mean", "DOUBLE"),
    ("yield_spread_mean", "DOUBLE"),
    ("converged_at_step", "INTEGER"),
    ("total_timesteps_configured", "INTEGER"),
    ("max_single_loss", "DOUBLE"),
    ("best_single_trade", "DOUBLE"),
    ("train_start_date", "TEXT"),
    ("train_end_date", "TEXT"),
    ("test_start_date", "TEXT"),
    ("test_end_date", "TEXT"),
    ("is_control_fold", "BOOLEAN DEFAULT FALSE"),
]


def _migrate_backtest_results(cursor: Any) -> None:
    """Idempotently add new columns to backtest_results for existing databases.

    Queries information_schema.columns and adds any missing columns from
    the _BACKTEST_RESULTS_MIGRATIONS list. Safe to call on fresh databases
    (all columns already exist) or old databases (columns get added).

    Args:
        cursor: Active DuckDB cursor.
    """
    existing_cols = {
        row[0]
        for row in cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'backtest_results'"
        ).fetchall()
    }
    for col_name, col_def in _BACKTEST_RESULTS_MIGRATIONS:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE backtest_results ADD COLUMN {col_name} {col_def}")
            log.info("backtest_results_column_added", column=col_name)


class DatabaseManager:
    """Singleton manager for DuckDB and SQLite database connections.

    Thread-safe via threading.Lock for singleton creation and DDL writes.
    DuckDB opens a fresh connection per context manager call (short-lived).
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

    @contextlib.contextmanager
    def duckdb(self, read_only: bool = False) -> Generator[Any, None, None]:
        """Yield a short-lived DuckDB connection.

        Opens a fresh connection per call, closes on exit.

        Args:
            read_only: Open connection in read-only mode (for concurrent readers).
        """
        self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb_module.connect(str(self._duckdb_path), read_only=read_only)
        try:
            yield conn
        finally:
            conn.close()

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
        """Create DuckDB tables, feature tables, and aggregation views."""
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
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    iteration_number INTEGER DEFAULT 0,
                    run_type TEXT DEFAULT 'baseline',
                    is_sharpe DOUBLE,
                    is_sortino DOUBLE,
                    is_mdd DOUBLE,
                    is_total_return DOUBLE,
                    overfitting_gap DOUBLE,
                    overfitting_class TEXT,
                    hmm_p_bull DOUBLE,
                    hmm_p_bear DOUBLE,
                    vix_mean DOUBLE,
                    yield_spread_mean DOUBLE,
                    converged_at_step INTEGER,
                    total_timesteps_configured INTEGER,
                    max_single_loss DOUBLE,
                    best_single_trade DOUBLE,
                    train_start_date TEXT,
                    train_end_date TEXT,
                    test_start_date TEXT,
                    test_end_date TEXT,
                    is_control_fold BOOLEAN DEFAULT FALSE
                )
            """)

            # Migration: add columns for existing databases with old schema
            _migrate_backtest_results(cursor)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iteration_results (
                    result_id TEXT PRIMARY KEY,
                    iteration_number INTEGER NOT NULL,
                    environment TEXT NOT NULL,
                    ensemble_sharpe DOUBLE,
                    ensemble_mdd DOUBLE,
                    gate_passed BOOLEAN,
                    ppo_weight DOUBLE,
                    a2c_weight DOUBLE,
                    sac_weight DOUBLE,
                    ppo_mean_sharpe DOUBLE,
                    a2c_mean_sharpe DOUBLE,
                    sac_mean_sharpe DOUBLE,
                    ppo_mean_mdd DOUBLE,
                    a2c_mean_mdd DOUBLE,
                    sac_mean_mdd DOUBLE,
                    total_folds INTEGER,
                    ppo_hyperparams TEXT,
                    a2c_hyperparams TEXT,
                    sac_hyperparams TEXT,
                    hp_source TEXT DEFAULT 'baseline',
                    run_type TEXT DEFAULT 'baseline',
                    wall_clock_s DOUBLE,
                    memory_enabled BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    UNIQUE(iteration_number, environment, run_type)
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

            # Phase 5 feature tables (idempotent -- CREATE TABLE IF NOT EXISTS)
            from swingrl.features.schema import init_feature_schema  # noqa: PLC0415

            init_feature_schema(cursor)

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
                CREATE TABLE IF NOT EXISTS shadow_trades (
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
                    trade_type TEXT,
                    model_version TEXT NOT NULL
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

            # Phase 12: Add stop/TP columns to positions (idempotent)
            for col_sql in [
                "ALTER TABLE positions ADD COLUMN stop_loss_price REAL",
                "ALTER TABLE positions ADD COLUMN take_profit_price REAL",
                "ALTER TABLE positions ADD COLUMN side TEXT",
            ]:
                try:
                    conn.execute(col_sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" in str(exc).lower():
                        pass  # Column already exists
                    else:
                        raise

            # Phase 12: inference_outcomes for NaN tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    had_nan INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Phase 12: api_errors for broker error tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    endpoint TEXT,
                    error_message TEXT
                )
            """)

            # --- Migrations for stale DB files ---
            # Tables with `environment` in the PRIMARY KEY cannot use ALTER TABLE
            # ADD COLUMN. If the column is missing, the table was created from an
            # older schema — recreate it (paper trading data is ephemeral).
            _tables_requiring_environment = {
                "positions": """
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
                """,
                "portfolio_snapshots": """
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
                """,
                "trades": """
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
                """,
            }
            for table_name, create_ddl in _tables_requiring_environment.items():
                try:
                    cols = {
                        row[1]
                        for row in conn.execute(
                            f"PRAGMA table_info({table_name})"  # noqa: S608
                        ).fetchall()
                    }
                except sqlite3.OperationalError:
                    continue  # Table doesn't exist yet — CREATE above handles it
                if cols and "environment" not in cols:
                    log.warning(
                        "sqlite_migration_recreating_table",
                        table=table_name,
                        reason="missing_environment_column",
                    )
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")  # noqa: S608
                    conn.execute(create_ddl)

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
        """Reset singleton state for test isolation.

        No persistent connections to close — DuckDB connections are short-lived.
        """
        self._initialized = False
        log.info("database_manager_closed")
