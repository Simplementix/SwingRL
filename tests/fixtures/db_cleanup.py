"""Autouse fixture that wipes the test database after every test.

Registered by importing ``wipe_db_after_test`` into ``tests/conftest.py``.
Function-scoped and autouse, so every test starts from a clean database —
killing the inter-test state pollution that fails ~50 tests on homelab CI.

Three safety layers (see ``tests/db_guard.py``):
  1. Postgres binds a connection to one database, so a ``*_test`` connection
     physically cannot reach production ``swingrl``.
  2. The suite-level guard (conftest ``pytest_configure``) refuses to start
     unless the resolved DB is a test database.
  3. ``ensure_wipe_target_is_test_db`` re-checks the resolved DB name ends in
     ``_test`` immediately before every TRUNCATE and refuses otherwise.
"""

from __future__ import annotations

from collections.abc import Generator

import psycopg
import pytest
from psycopg import sql

from swingrl.data.db import DatabaseManager
from tests.db_guard import classify_db_url, resolve_target_db_url


def ensure_wipe_target_is_test_db(db_url: str) -> bool:
    """Pre-wipe re-check (safety layer #3).

    Returns ``True`` when the resolved DB is a test database and should be wiped,
    ``False`` when no DB is configured (skip). Raises ``RuntimeError`` when a DB
    is present but is NOT a recognised test database — cleanup must never run
    against production.
    """
    verdict, db_name = classify_db_url(db_url)
    if verdict == "blank":
        return False
    if verdict != "safe":
        raise RuntimeError(
            f"Refusing to TRUNCATE: resolved database {db_name!r} is not a test "
            f"database (verdict={verdict!r}). The suite guard should have already "
            f"aborted — this is a defence-in-depth backstop."
        )
    return True


def _truncate_all_public_tables(db_url: str) -> None:
    """TRUNCATE every table in the public schema of the connected test database.

    Uses a dedicated short-lived autocommit connection (not the pooled
    DatabaseManager connection) to avoid pool-affinity surprises, with a bounded
    ``lock_timeout`` so a stray lock from a misbehaving test fails fast rather
    than hanging CI. Catalog-enumerates tables so ad-hoc tables created by raw
    tests are covered; a single ``TRUNCATE … CASCADE`` handles FK ordering.
    """
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("SET lock_timeout = '10s'")
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        tables = [row[0] for row in rows]
        if not tables:
            return
        statement = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
            sql.SQL(", ").join(sql.Identifier(name) for name in tables)
        )
        conn.execute(statement)


@pytest.fixture(autouse=True)
def wipe_db_after_test() -> Generator[None, None, None]:
    """Wipe all test-database tables after each test (no-op when no test DB)."""
    yield
    db_url = resolve_target_db_url()
    if not ensure_wipe_target_is_test_db(db_url):
        return
    # Close the pool so idle connections release their locks before TRUNCATE.
    # NOTE: reset() closes only idle/returned connections — a connection still
    # checked out when teardown runs is NOT force-closed. That can only happen if
    # a test leaks a checked-out connection past its own body (a bug); the
    # lock_timeout in _truncate_all_public_tables bounds such a case to a loud
    # ~10s failure that names the offending test, rather than a silent hang.
    # A proper drain/rollback fixture is deferred to Stage 3.0 (see the spec).
    DatabaseManager.reset()
    _truncate_all_public_tables(db_url)
