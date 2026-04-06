"""PostgreSQL storage layer via psycopg connection pool.

DatabaseManager is a singleton providing a thread-safe connection pool to a
single PostgreSQL database that consolidates all OLAP (market data, features)
and OLTP (trades, positions, risk decisions) tables.

Usage:
    from swingrl.data.db import DatabaseManager
    from swingrl.config.schema import load_config

    config = load_config()
    db = DatabaseManager(config)
    db.init_schema()

    with db.connection() as conn:
        conn.execute("SELECT * FROM ohlcv_daily WHERE symbol = %s", ["SPY"])
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import psycopg
import structlog
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class DatabaseManager:
    """Singleton manager for PostgreSQL connection pool.

    Thread-safe via threading.Lock for singleton creation.
    Connections are borrowed from the pool and returned on context exit.
    Uses ``dict_row`` factory so callers can access columns by name
    (``row["column"]``) — backward compatible with the old sqlite3
    ``sqlite3.Row`` pattern and DuckDB tuple indexing.
    """

    _instance: DatabaseManager | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    _pool: ConnectionPool[psycopg.Connection[dict[str, Any]]] | None

    def __new__(cls, config: SwingRLConfig | None = None) -> DatabaseManager:
        """Thread-safe singleton creation."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize the PostgreSQL connection pool.

        The database URL is resolved in this order:
        1. ``DATABASE_URL`` environment variable (preferred in Docker)
        2. ``config.system.database_url`` from YAML config

        Args:
            config: Validated SwingRLConfig with system.database_url.
        """
        if self._initialized:
            return

        self._database_url = os.environ.get("DATABASE_URL") or config.system.database_url
        # Pool is only created when a real DATABASE_URL is available.
        # Without one (e.g. in tests), _pool stays None and connection()
        # raises immediately instead of waiting for a TCP timeout.
        if self._database_url:
            self._pool = ConnectionPool(  # type: ignore[assignment]  # dict_row via kwargs
                self._database_url,
                min_size=0,
                max_size=20,
                timeout=30.0,
                open=False,
                kwargs={"row_factory": dict_row, "autocommit": False},
            )
        else:
            self._pool = None
        self._initialized = True

        # Mask password in log output
        safe_url = self._database_url
        if "@" in safe_url:
            pre_at = safe_url.split("@")[0]
            if ":" in pre_at:
                safe_url = pre_at.rsplit(":", 1)[0] + ":***@" + safe_url.split("@", 1)[1]

        log.info("database_manager_initialized", database_url=safe_url)

    @classmethod
    def reset(cls) -> None:
        """Clear singleton instance and close pool. For test isolation."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    @contextlib.contextmanager
    def connection(self) -> Generator[psycopg.Connection[dict[str, Any]], None, None]:
        """Yield a connection from the pool.

        Auto-commits on clean exit, rolls back on exception.
        The connection is returned to the pool when the context exits.
        Lazily opens the pool on first call.
        """
        if self._pool is None:
            raise RuntimeError("DATABASE_URL not set — cannot connect to PostgreSQL")
        if self._pool.closed:
            self._pool.open()
        with self._pool.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def init_schema(self) -> None:
        """Create all tables, indexes, and views in PostgreSQL.

        Idempotent: uses CREATE TABLE IF NOT EXISTS / CREATE OR REPLACE VIEW.
        """
        from swingrl.data.postgres_schema import init_postgres_schema  # noqa: PLC0415

        with self.connection() as conn:
            init_postgres_schema(conn)
        log.info("schema_initialized")

    def close(self) -> None:
        """Close the connection pool and reset singleton state."""
        if hasattr(self, "_pool") and self._pool is not None:
            self._pool.close()
        self._initialized = False
        log.info("database_manager_closed")
