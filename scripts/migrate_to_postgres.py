#!/usr/bin/env python3
"""One-time migration script: DuckDB + SQLite → PostgreSQL.

Reads existing DuckDB (market_data.ddb) and SQLite (trading_ops.db) files,
bulk-inserts all rows into the PostgreSQL swingrl database on pg16.

Usage:
    DATABASE_URL="postgresql://swingrl:pw@pg16:5432/swingrl" uv run python scripts/migrate_to_postgres.py

Total data volume: ~130K rows across 29 tables. Completes in <10 seconds.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import duckdb
import psycopg
import structlog
from psycopg.rows import dict_row

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swingrl.data.postgres_schema import init_postgres_schema  # noqa: E402

log = structlog.get_logger(__name__)

# DuckDB tables to migrate (order doesn't matter — no FKs)
_DUCKDB_TABLES = [
    "ohlcv_daily",
    "ohlcv_4h",
    "macro_features",
    "data_quarantine",
    "data_ingestion_log",
    "model_metadata",
    "backtest_results",
    "iteration_results",
    "features_equity",
    "features_crypto",
    "fundamentals",
    "hmm_state_history",
    "training_epochs",
    "meta_decisions",
    "reward_adjustments",
]

# SQLite tables to migrate
_SQLITE_TABLES = [
    "trades",
    "positions",
    "risk_decisions",
    "portfolio_snapshots",
    "system_events",
    "corporate_actions",
    "wash_sale_tracker",
    "circuit_breaker_events",
    "options_positions",
    "shadow_trades",
    "alert_log",
    "inference_outcomes",
    "api_errors",
]

# Memory service tables (from db/memory.db)
_MEMORY_TABLES = [
    "memories",
    "consolidations",
    "consolidation_quality",
    "consolidation_sources",
    "pattern_presentations",
    "pattern_outcomes",
    "llm_audit_log",
]

BATCH_SIZE = 1000


def _get_columns(cursor: duckdb.DuckDBPyConnection | sqlite3.Cursor, table: str) -> list[str]:
    """Get column names for a table."""
    if hasattr(cursor, "description") and cursor.description:
        return [desc[0] for desc in cursor.description]
    # Fallback: execute a dummy query
    result = cursor.execute(f"SELECT * FROM {table} LIMIT 0")  # noqa: S608  # nosec B608
    return [desc[0] for desc in result.description]


def _migrate_duckdb_table(
    ddb_conn: duckdb.DuckDBPyConnection,
    pg_conn: psycopg.Connection,
    table: str,
) -> int:
    """Migrate a single DuckDB table to PostgreSQL."""
    try:
        rows = ddb_conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608  # nosec B608
    except duckdb.CatalogException:
        log.warning("table_not_found_in_duckdb", table=table)
        return 0

    if not rows:
        log.info("table_empty", table=table, source="duckdb")
        return 0

    columns = [desc[0] for desc in ddb_conn.description]

    # Skip 'id' column for IDENTITY tables (auto-generated in Postgres)
    identity_tables = {"data_quarantine", "training_epochs", "meta_decisions", "reward_adjustments"}
    if table in identity_tables and "id" in columns:
        id_idx = columns.index("id")
        columns = [c for c in columns if c != "id"]
        rows = [tuple(v for i, v in enumerate(row) if i != id_idx) for row in rows]

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"  # noqa: S608  # nosec B608

    with pg_conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            cur.executemany(insert_sql, batch)

    pg_conn.commit()
    log.info("table_migrated", table=table, source="duckdb", rows=len(rows))
    return len(rows)


def _migrate_sqlite_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: psycopg.Connection,
    table: str,
) -> int:
    """Migrate a single SQLite table to PostgreSQL."""
    try:
        cursor = sqlite_conn.execute(f"SELECT * FROM {table}")  # noqa: S608  # nosec B608
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        log.warning("table_not_found_in_sqlite", table=table)
        return 0

    if not rows:
        log.info("table_empty", table=table, source="sqlite")
        return 0

    columns = [desc[0] for desc in cursor.description]

    # Skip 'id' for IDENTITY tables
    identity_tables = {"inference_outcomes", "api_errors"}
    if table in identity_tables and "id" in columns:
        id_idx = columns.index("id")
        columns = [c for c in columns if c != "id"]
        rows = [tuple(v for i, v in enumerate(row) if i != id_idx) for row in rows]

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"  # noqa: S608  # nosec B608

    with pg_conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            cur.executemany(insert_sql, batch)

    pg_conn.commit()
    log.info("table_migrated", table=table, source="sqlite", rows=len(rows))
    return len(rows)


def main() -> None:
    """Run the full migration."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    duckdb_path = Path(os.environ.get("DUCKDB_PATH", "db/market_data.ddb"))
    sqlite_path = Path(os.environ.get("SQLITE_PATH", "db/trading_ops.db"))
    memory_path = Path(os.environ.get("MEMORY_DB_PATH", "db/memory.db"))

    print(f"Source DuckDB: {duckdb_path}")
    print(f"Source SQLite: {sqlite_path}")
    print(f"Source Memory: {memory_path}")
    print(f"Target: {database_url.split('@')[1] if '@' in database_url else database_url}")
    print()

    # Connect to PostgreSQL and create schema
    pg_conn = psycopg.connect(database_url, row_factory=dict_row)
    init_postgres_schema(pg_conn)
    pg_conn.commit()
    print("Schema initialized.")

    total_rows = 0

    # Migrate DuckDB tables
    if duckdb_path.exists():
        print(f"\nMigrating DuckDB ({duckdb_path})...")
        ddb_conn = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            for table in _DUCKDB_TABLES:
                count = _migrate_duckdb_table(ddb_conn, pg_conn, table)
                total_rows += count
        finally:
            ddb_conn.close()
    else:
        print(f"DuckDB not found at {duckdb_path}, skipping.")

    # Migrate SQLite tables
    if sqlite_path.exists():
        print(f"\nMigrating SQLite ({sqlite_path})...")
        sqlite_conn = sqlite3.connect(str(sqlite_path))
        try:
            for table in _SQLITE_TABLES:
                count = _migrate_sqlite_table(sqlite_conn, pg_conn, table)
                total_rows += count
        finally:
            sqlite_conn.close()
    else:
        print(f"SQLite not found at {sqlite_path}, skipping.")

    # Migrate memory service SQLite tables
    if memory_path.exists():
        print(f"\nMigrating Memory SQLite ({memory_path})...")
        memory_conn = sqlite3.connect(str(memory_path))
        try:
            for table in _MEMORY_TABLES:
                count = _migrate_sqlite_table(memory_conn, pg_conn, table)
                total_rows += count
        finally:
            memory_conn.close()
    else:
        print(f"Memory DB not found at {memory_path}, skipping.")

    # Verify
    print("\n--- Verification ---")
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        tables = [row["tablename"] for row in cur.fetchall()]

    print(f"Tables in PostgreSQL: {len(tables)}")
    for table in tables:
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")  # noqa: S608  # nosec B608
            row = cur.fetchone()
            count = row["cnt"] if row else 0
            if count > 0:
                print(f"  {table}: {count:,} rows")

    pg_conn.close()
    print(f"\nMigration complete. Total rows migrated: {total_rows:,}")


if __name__ == "__main__":
    main()
