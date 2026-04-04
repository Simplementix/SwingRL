"""Docker HEALTHCHECK probe for SwingRL production container.

Verifies:
  1. Main Python process is alive (trivially true if this script executes)
  2. PostgreSQL database is accessible and responds to queries

Exits 0 on success, 1 on any failure. Designed to complete in < 5 seconds.
"""

from __future__ import annotations

import os
import sys


def check_postgres() -> None:
    """Verify PostgreSQL database is accessible and responds to queries."""
    import psycopg  # noqa: PLC0415 — lazy import

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
    )
    conn = psycopg.connect(database_url, connect_timeout=5)
    try:
        result = conn.execute("SELECT 1").fetchone()
        if result is None or result[0] != 1:
            raise RuntimeError(f"PostgreSQL connectivity check failed: {result}")
    finally:
        conn.close()


def main() -> int:
    """Run all health checks. Returns 0 on success, 1 on failure."""
    try:
        check_postgres()
    except Exception as exc:  # noqa: BLE001 — healthcheck must catch all
        print(f"UNHEALTHY: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print("HEALTHY")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
