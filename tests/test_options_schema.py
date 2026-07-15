# tests/test_options_schema.py
from __future__ import annotations

import os
from datetime import date

import psycopg
import pytest
from swingrl.data.options.schema import (
    ensure_monthly_partition,
    ensure_options_schema,
    monthly_partition_bounds,
)

_DB_URL = os.environ.get("DATABASE_URL")
_needs_db = pytest.mark.skipif(not _DB_URL, reason="DATABASE_URL not set")


def test_monthly_partition_bounds_normal_month() -> None:
    """OPT-SCHEMA-1: partition name + [lo, hi) bounds (spec §8.2)."""
    name, lo, hi = monthly_partition_bounds(date(2026, 7, 14))
    assert name == "options_chains_2026_07"
    assert lo == date(2026, 7, 1)
    assert hi == date(2026, 8, 1)


def test_monthly_partition_bounds_december_rolls_year() -> None:
    """OPT-SCHEMA-2: December -> next Jan (spec §8.2)."""
    name, lo, hi = monthly_partition_bounds(date(2026, 12, 3))
    assert name == "options_chains_2026_12"
    assert lo == date(2026, 12, 1)
    assert hi == date(2027, 1, 1)


@_needs_db
def test_ensure_schema_is_idempotent() -> None:
    """OPT-SCHEMA-3: ensure_options_schema runs twice cleanly (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        ensure_options_schema(conn)  # second call must not error
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.options_snapshots')")
            assert cur.fetchone()[0] is not None
            cur.execute("SELECT to_regclass('public.options_chains')")
            assert cur.fetchone()[0] is not None
        conn.rollback()


@_needs_db
def test_ensure_monthly_partition_creates_child() -> None:
    """OPT-SCHEMA-4: monthly partition auto-created + idempotent (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        name = ensure_monthly_partition(conn, date(2026, 7, 14))
        assert name == "options_chains_2026_07"
        ensure_monthly_partition(conn, date(2026, 7, 20))  # same month, no error
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
            assert cur.fetchone()[0] is not None
        conn.rollback()
