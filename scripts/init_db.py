"""Initialize SwingRL PostgreSQL database.

Creates all tables, indexes, and views defined by DatabaseManager.init_schema().
Idempotent — safe to run multiple times.

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
    parser = argparse.ArgumentParser(description="Initialize SwingRL PostgreSQL database.")
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

        # Verify PostgreSQL tables
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
            ).fetchall()
            pg_tables = [row[0] for row in rows]

            # Basic connectivity check
            result = conn.execute("SELECT 1").fetchone()
            if result is None or result[0] != 1:
                log.error("pg_connectivity_failed")
                return 1

        log.info("pg_tables_created", tables=pg_tables, count=len(pg_tables))
        log.info("pg_connectivity_check", result="ok")

        print(f"PostgreSQL: {len(pg_tables)} tables created at {config.system.database_url}")
        print("Connectivity check: PASSED")

        db.close()
        return 0

    except Exception as e:
        log.error("init_db_failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
