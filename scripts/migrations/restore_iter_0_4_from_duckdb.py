#!/usr/bin/env python3
"""Restore iter 0-N backtest_results + iteration_results from duckdb backup.

One-shot recovery script for the 2026-04-07 incident: a test fixture
(``tests/agents/test_backtest.py:_create_backtest_schema``) was run against
production pg16 via ``docker exec swingrl pytest`` (bypassing the
``scripts/ci-homelab.sh`` isolation that creates a temporary
``swingrl_test`` database). The fixture's ``DROP TABLE IF EXISTS``
statements wiped both ``backtest_results`` and ``iteration_results`` plus
reverted the Phase 0.2 CPS column migration on the latter.

This script reads the pre-postgres-migration duckdb backup at
``/app/data/db_backup_pre_postgres/market_data.ddb`` and bulk-inserts the
iter 0..max-iter rows into pg16. Uses the column intersection between
duckdb source and pg16 target so it tolerates schema drift in either
direction (extra pg16 columns get NULL; legacy duckdb columns get skipped).

Idempotency: ``INSERT ... ON CONFLICT (result_id) DO NOTHING``. Safe to
re-run any number of times — re-runs will be no-ops once data is in place.

Iter 5 is intentionally NOT recoverable from this script (and not from
training_iter5.log* either, which has been mostly rotated). The duckdb
backup is from 2026-04-04, before iter 5 ran. Per user decision, iter 5 is
"lost" and the next training run starts at iter 6.

Usage::

    # Default: restore iter 0..4 from the standard backup path
    ssh homelab "docker exec swingrl python3 /tmp/restore_iter_0_4_from_duckdb.py"

    # Custom range / source / dry-run
    docker exec swingrl python3 /tmp/restore_iter_0_4_from_duckdb.py \\
        --source /app/data/db_backup_pre_postgres/market_data.ddb \\
        --max-iter 4 \\
        --dry-run

Ref: ``.planning/PHASE_19.1_HANDOFF.md`` recovery section.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import duckdb
import psycopg
import structlog

# Allow running from repo root or from inside the container
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

log = structlog.get_logger(__name__)

_DEFAULT_SOURCE = "/app/data/db_backup_pre_postgres/market_data.ddb"
_DEFAULT_MAX_ITER = 4
_BATCH_SIZE = 500
_TABLES: tuple[str, ...] = ("backtest_results", "iteration_results")


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Source it from .env or run inside the container.")
    return url


def _duckdb_columns(ddb: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = ddb.execute(f"PRAGMA table_info({table})").fetchall()  # noqa: S608  # nosec B608
    return [r[1] for r in rows]


def _pg_columns(pg: psycopg.Connection, table: str) -> list[str]:
    with pg.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def _column_intersection(
    src_cols: list[str], tgt_cols: list[str]
) -> tuple[list[str], list[str], list[str]]:
    src_set = set(src_cols)
    tgt_set = set(tgt_cols)
    common = [c for c in src_cols if c in tgt_set]  # preserve src order
    src_only = sorted(src_set - tgt_set)
    tgt_only = sorted(tgt_set - src_set)
    return common, src_only, tgt_only


def _restore_table(
    ddb: duckdb.DuckDBPyConnection,
    pg: psycopg.Connection,
    table: str,
    max_iter: int,
    dry_run: bool,
) -> int:
    """Restore one table; return number of rows considered (or inserted)."""
    src_cols = _duckdb_columns(ddb, table)
    tgt_cols = _pg_columns(pg, table)
    common, src_only, tgt_only = _column_intersection(src_cols, tgt_cols)

    log.info(
        "schema_intersection",
        table=table,
        source_cols=len(src_cols),
        target_cols=len(tgt_cols),
        common_cols=len(common),
        source_only=src_only,
        target_only=tgt_only,
    )

    col_list = ", ".join(common)
    select_sql = (
        f"SELECT {col_list} FROM {table} "  # noqa: S608  # nosec B608
        f"WHERE iteration_number BETWEEN 0 AND {int(max_iter)} "
        f"ORDER BY iteration_number, environment"
    )
    rows = ddb.execute(select_sql).fetchall()
    log.info("source_rows_read", table=table, rows=len(rows), max_iter=max_iter)

    if dry_run:
        log.info("dry_run_skip_insert", table=table, would_insert=len(rows))
        return len(rows)

    if not rows:
        log.warning("no_rows_to_restore", table=table)
        return 0

    placeholders = ", ".join(["%s"] * len(common))
    insert_sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "  # noqa: S608  # nosec B608
        f"ON CONFLICT (result_id) DO NOTHING"
    )

    inserted_total = 0
    with pg.cursor() as cur:
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            cur.executemany(insert_sql, batch)
            inserted_total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            log.info("batch_inserted", table=table, batch_start=i, batch_size=len(batch))

    # Verify final count of restored iter range in the target
    with pg.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM {table} "  # noqa: S608  # nosec B608
            f"WHERE iteration_number BETWEEN 0 AND {int(max_iter)}"
        )
        final_count = cur.fetchone()[0]

    log.info(
        "table_restore_complete",
        table=table,
        source_rows=len(rows),
        executemany_rowcount=inserted_total,
        target_count_in_range=final_count,
    )
    return final_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(_DEFAULT_SOURCE),
        help=f"Path to duckdb backup (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=_DEFAULT_MAX_ITER,
        help=f"Maximum iteration_number to restore (inclusive). Default: {_DEFAULT_MAX_ITER}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + row counts without writing to pg16.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        log.error("source_not_found", path=str(args.source))
        return 1

    db_url = _get_database_url()
    log.info(
        "restore_started",
        source=str(args.source),
        max_iter=args.max_iter,
        dry_run=args.dry_run,
        target=db_url.split("@")[-1] if "@" in db_url else "unknown",
    )

    ddb = duckdb.connect(str(args.source), read_only=True)
    try:
        with psycopg.connect(db_url, autocommit=True) as pg:
            for table in _TABLES:
                _restore_table(ddb, pg, table, args.max_iter, args.dry_run)
    finally:
        ddb.close()

    log.info("restore_complete", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
