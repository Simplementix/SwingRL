"""PostgreSQL helper utilities replacing DuckDB-specific patterns.

Provides three functions used throughout the codebase during the DuckDB → PostgreSQL
migration:

1. ``fetchdf`` — replaces DuckDB's ``.fetchdf()`` method
2. ``executemany_from_df`` — replaces DuckDB's replacement scan (``FROM pandas_df``)
3. ``bulk_insert_parquet`` — replaces DuckDB's ``read_parquet()`` function

Usage:
    from swingrl.data.pg_helpers import fetchdf, executemany_from_df

    with db.connection() as conn:
        cur = conn.execute("SELECT * FROM ohlcv_daily WHERE symbol = %s", ["SPY"])
        df = fetchdf(cur)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog

if TYPE_CHECKING:
    import psycopg

log = structlog.get_logger(__name__)


def fetchdf(cursor: psycopg.Cursor) -> pd.DataFrame:
    """Fetch all rows from a psycopg cursor into a pandas DataFrame.

    Replaces DuckDB's ``cursor.execute(...).fetchdf()`` pattern.
    Returns an empty DataFrame (with correct column names) when the result
    set is empty.

    Args:
        cursor: An executed psycopg cursor with pending results.

    Returns:
        DataFrame with column names derived from ``cursor.description``.
    """
    columns = [desc.name for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def executemany_from_df(
    conn: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    columns: list[str],
    *,
    on_conflict: str = "DO NOTHING",
) -> int:
    """Bulk-insert a DataFrame into a PostgreSQL table.

    Replaces DuckDB's replacement scan pattern::

        INSERT OR IGNORE INTO t (cols) SELECT cols FROM sync_df

    Builds a parameterised INSERT with ON CONFLICT handling and executes
    via ``cursor.executemany()`` for datasets up to ~50 K rows.  For
    larger batches, consider switching to ``cursor.copy()`` with the COPY
    protocol.

    Args:
        conn: Open psycopg connection.
        table: Target table name (unquoted).
        df: Source DataFrame.  Only *columns* are extracted.
        columns: Column names to insert (must exist in both *df* and *table*).
        on_conflict: Conflict clause appended after VALUES, e.g.
            ``"DO NOTHING"`` or
            ``"(symbol, date) DO UPDATE SET col=EXCLUDED.col"``.

    Returns:
        Number of rows sent to the server (not necessarily inserted if
        ``ON CONFLICT DO NOTHING`` skips duplicates).
    """
    if df.empty:
        return 0

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = (
        f"INSERT INTO {table} ({col_list}) "  # noqa: S608  # nosec B608
        f"VALUES ({placeholders}) "
        f"ON CONFLICT {on_conflict}"
    )

    # Convert DataFrame rows to list of tuples, replacing NaN with None
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[columns].itertuples(index=False, name=None)
    ]

    with conn.cursor() as cur:
        cur.executemany(sql, records)

    log.debug(
        "executemany_from_df",
        table=table,
        rows=len(records),
        on_conflict=on_conflict,
    )
    return len(records)


def bulk_insert_parquet(
    conn: psycopg.Connection,
    parquet_path: Path,
    table: str,
    columns: list[str],
    *,
    on_conflict: str = "DO NOTHING",
) -> int:
    """Load a Parquet file into a PostgreSQL table.

    Replaces DuckDB's ``read_parquet()`` function::

        INSERT OR IGNORE INTO t (cols) SELECT cols FROM read_parquet('path')

    Reads the Parquet file via pandas and delegates to
    :func:`executemany_from_df`.

    Args:
        conn: Open psycopg connection.
        parquet_path: Path to the ``.parquet`` file.
        table: Target table name.
        columns: Column names to insert.
        on_conflict: Conflict clause (see :func:`executemany_from_df`).

    Returns:
        Number of rows sent.
    """
    df = pd.read_parquet(parquet_path)
    count = executemany_from_df(conn, table, df, columns, on_conflict=on_conflict)
    log.info(
        "bulk_insert_parquet",
        path=str(parquet_path),
        table=table,
        rows=count,
    )
    return count
