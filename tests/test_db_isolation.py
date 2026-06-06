"""STAGE1: The autouse wipe isolates DB state between tests.

Two tests insert into the same probe table and each assert exactly one row.
Without the wipe the second test would see two rows (the first test's leftover);
with the autouse wipe each test starts clean. DB-gated: skips without a test DB.
"""

from __future__ import annotations

import os

import psycopg
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="requires a *_test PostgreSQL database (set DATABASE_URL via ci-homelab.sh)",
)

_PROBE_TABLE = "stage1_isolation_probe"


def _insert_probe_and_count() -> int:
    """Create the probe table if needed, insert one row, return its row count."""
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_PROBE_TABLE} (n integer)")  # nosec B608
        conn.execute(f"INSERT INTO {_PROBE_TABLE} (n) VALUES (1)")  # nosec B608
        row = conn.execute(f"SELECT count(*) FROM {_PROBE_TABLE}").fetchone()  # nosec B608
    assert row is not None
    return int(row[0])


def test_isolation_first_insert() -> None:
    """REQ-STAGE1: first test inserts the probe row and sees exactly one."""
    assert _insert_probe_and_count() == 1


def test_isolation_second_insert() -> None:
    """REQ-STAGE1: after the wipe, the second test sees a clean table — one again."""
    assert _insert_probe_and_count() == 1
