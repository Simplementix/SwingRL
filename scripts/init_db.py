"""Initialize SwingRL databases (DuckDB + SQLite).

Creates both databases with all tables, indexes, and views defined by
DatabaseManager.init_schema(). Idempotent — safe to run multiple times.

Usage:
    uv run python scripts/init_db.py
    uv run python scripts/init_db.py --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys

import structlog

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Initialize databases and verify integrity.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description="Initialize SwingRL DuckDB and SQLite databases.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML file (default: config/swingrl.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

        log.info("init_db_start", config_path=args.config)

        db = DatabaseManager(config)
        db.init_schema()

        # Verify DuckDB tables
        with db.duckdb() as cursor:
            tables = cursor.execute("SHOW TABLES").fetchall()
            duckdb_tables = [row[0] for row in tables]
        log.info("duckdb_tables_created", tables=duckdb_tables, count=len(duckdb_tables))

        # Verify SQLite tables and run integrity check
        with db.sqlite() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            sqlite_tables = [row["name"] for row in rows]

            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity[0] != "ok":
                log.error("sqlite_integrity_failed", result=integrity[0])
                return 1

        log.info("sqlite_tables_created", tables=sqlite_tables, count=len(sqlite_tables))
        log.info("sqlite_integrity_check", result="ok")

        print(f"DuckDB: {len(duckdb_tables)} tables created at {config.system.duckdb_path}")
        print(f"SQLite: {len(sqlite_tables)} tables created at {config.system.sqlite_path}")
        print("Integrity check: PASSED")

        db.close()
        return 0

    except Exception as e:
        log.error("init_db_failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
