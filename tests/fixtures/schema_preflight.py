"""Session-start schema-integrity preflight (safety layer #4 — SHAPE-based).

The three existing guard layers are NAME-based (suite guard, URL classifier,
pre-wipe re-check) and cannot see a poisoned scratch DB: a test that
DROP+hand-recreated a production-named table without its PRIMARY KEY leaves a
shape that ``CREATE TABLE IF NOT EXISTS`` will never repair, and version-number
checks (``assert_schema_current``) pass regardless. That exact blind spot cost
three suite-hours on 2026-07-22 (hmm_state_history PK loss, found only by a
human diffing pg_constraint against production).

Rule: every canonical table (postgres_schema._ALL_TABLE_DDL) whose DDL declares
a PRIMARY KEY must, IF it exists in the target DB, carry a ``contype = 'p'``
row in pg_constraint. Absent tables are fine — they are created canonically on
demand. Cost: one catalog query per session.
"""

from __future__ import annotations

import re

import psycopg

from swingrl.data import postgres_schema

_CREATE_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", re.IGNORECASE)


def expected_pk_tables() -> frozenset[str]:
    """Canonical table names whose DDL declares a PRIMARY KEY."""
    names: set[str] = set()
    for ddl in postgres_schema._ALL_TABLE_DDL:
        match = _CREATE_TABLE_RE.search(ddl)
        if match and "PRIMARY KEY" in ddl.upper():
            names.add(match.group(1))
    return frozenset(names)


def schema_integrity_errors(db_url: str) -> list[str]:
    """Messages naming existing canonical tables that are missing their PRIMARY KEY."""
    expected = sorted(expected_pk_tables())
    with psycopg.connect(db_url, autocommit=True) as conn:
        rows = conn.execute(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname = ANY(%s)
              AND NOT EXISTS (
                  SELECT 1 FROM pg_constraint con
                  WHERE con.conrelid = c.oid AND con.contype = 'p'
              )
            ORDER BY c.relname
            """,
            (expected,),
        ).fetchall()
    return [
        f"table {row[0]!r} exists WITHOUT its canonical PRIMARY KEY — the scratch "
        f"DB is poisoned (some test DROP+hand-recreated it). Recreate the scratch "
        f"database (or DROP the table and re-run init_schema) before testing."
        for row in rows
    ]
