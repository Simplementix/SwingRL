"""Docker HEALTHCHECK probe for SwingRL production container.

Verifies:
  1. Main Python process is alive (trivially true if this script executes)
  2. SQLite trading_ops.db is accessible and passes integrity check
  3. DuckDB market_data.ddb is accessible and responds to queries

Exits 0 on success, 1 on any failure. Designed to complete in < 5 seconds.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def check_sqlite(db_path: Path) -> None:
    """Verify SQLite database is accessible and passes integrity check."""
    if not db_path.exists():
        # DB may not exist yet on first startup — skip check
        return
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")
    finally:
        conn.close()


def check_duckdb(db_path: Path) -> None:
    """Verify DuckDB database is accessible and responds to queries."""
    if not db_path.exists():
        # DB may not exist yet on first startup — skip check
        return
    import duckdb  # noqa: PLC0415 — lazy import, duckdb may not be available

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        result = conn.execute("SELECT 1").fetchone()
        if result is None or result[0] != 1:
            raise RuntimeError(f"DuckDB connectivity check failed: {result}")
    finally:
        conn.close()


def main() -> int:
    """Run all health checks. Returns 0 on success, 1 on failure."""
    # Paths relative to /app working directory inside the container.
    # Bind-mounted volumes: db/ contains both databases.
    db_dir = Path("db")
    sqlite_path = db_dir / "trading_ops.db"
    duckdb_path = db_dir / "market_data.ddb"

    try:
        check_sqlite(sqlite_path)
        check_duckdb(duckdb_path)
    except Exception as exc:  # noqa: BLE001 — healthcheck must catch all
        print(f"UNHEALTHY: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print("HEALTHY")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
