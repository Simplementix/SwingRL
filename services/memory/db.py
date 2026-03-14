"""SQLite database layer for the swingrl-memory service.

Manages the memory.db file with three tables:
- memories: raw ingested text with source tags
- consolidations: LLM-synthesized patterns from memory batches
- consolidation_quality: quality audit log for consolidation attempts

All config comes from environment variables — no swingrl imports.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    archived   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""

_CREATE_CONSOLIDATIONS = """
CREATE TABLE IF NOT EXISTS consolidations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_text     TEXT NOT NULL,
    source_count     INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    conflicting_with INTEGER REFERENCES consolidations(id)
);
"""

_CREATE_CONSOLIDATION_QUALITY = """
CREATE TABLE IF NOT EXISTS consolidation_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_count   INTEGER NOT NULL DEFAULT 1,
    accepted        INTEGER NOT NULL DEFAULT 0,
    rejected_reason TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _get_db_path() -> Path:
    """Return the SQLite database path from MEMORY_DB_PATH env var."""
    raw = os.environ.get("MEMORY_DB_PATH", "/app/db/memory.db")
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with WAL mode and row_factory.

    Callers are responsible for closing the connection.
    """
    conn = sqlite3.connect(str(_get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    conn = get_connection()
    try:
        conn.executescript(_CREATE_MEMORIES)
        conn.executescript(_CREATE_CONSOLIDATIONS)
        conn.executescript(_CREATE_CONSOLIDATION_QUALITY)
        conn.commit()
        log.info("memory_db_initialized", path=str(_get_db_path()))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def insert_memory(text: str, source: str) -> int:
    """Insert a raw memory row and return its row ID.

    Args:
        text: Memory text content (may be XML-wrapped by caller).
        source: Source tag, e.g. 'training_run:historical'.

    Returns:
        Integer row ID of the inserted row.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO memories (text, source) VALUES (?, ?)",
            (text, source),
        )
        conn.commit()
        row_id = cur.lastrowid
        log.info("memory_inserted", source=source, row_id=row_id)
        return int(row_id)  # type: ignore[arg-type]
    finally:
        conn.close()


def get_memories(
    source: str | None = None,
    limit: int = 100,
    since: str | None = None,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve raw memories with optional filters.

    Args:
        source: Filter by source tag (exact match). None returns all sources.
        limit: Maximum rows to return.
        since: ISO datetime string; returns rows created after this value.
        archived: If True, include archived memories.

    Returns:
        List of row dicts with keys: id, text, source, created_at, archived.
    """
    conn = get_connection()
    try:
        clauses: list[str] = []
        params: list[Any] = []

        if not archived:
            clauses.append("archived = 0")
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if since is not None:
            clauses.append("created_at > ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        sql = f"SELECT id, text, source, created_at, archived FROM memories {where} ORDER BY created_at DESC LIMIT ?"  # noqa: E501  # nosec B608 — where clause built from fixed strings only
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def archive_memories(row_ids: list[int]) -> None:
    """Mark memories as archived after successful consolidation.

    Args:
        row_ids: List of memory row IDs to archive.
    """
    if not row_ids:
        return
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(row_ids))
        conn.execute(
            f"UPDATE memories SET archived = 1 WHERE id IN ({placeholders})",  # nosec B608 — placeholders are all '?' literals
            row_ids,
        )
        conn.commit()
        log.info("memories_archived", count=len(row_ids))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Consolidation helpers
# ---------------------------------------------------------------------------


def insert_consolidation(
    pattern_text: str,
    source_count: int,
    conflicting_with: int | None = None,
) -> int:
    """Insert a consolidated pattern and return its row ID.

    Args:
        pattern_text: LLM-generated pattern text.
        source_count: Number of raw memories consumed.
        conflicting_with: ID of a conflicting consolidation row, if any.

    Returns:
        Integer row ID of the inserted row.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO consolidations (pattern_text, source_count, conflicting_with) VALUES (?, ?, ?)",
            (pattern_text, source_count, conflicting_with),
        )
        conn.commit()
        row_id = cur.lastrowid
        log.info("consolidation_inserted", source_count=source_count, row_id=row_id)
        return int(row_id)  # type: ignore[arg-type]
    finally:
        conn.close()


def get_consolidations(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent consolidation patterns.

    Args:
        limit: Maximum rows to return.

    Returns:
        List of row dicts with keys: id, pattern_text, source_count, created_at, conflicting_with.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, pattern_text, source_count, created_at, conflicting_with "
            "FROM consolidations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Consolidation quality
# ---------------------------------------------------------------------------


def log_consolidation_quality(
    attempt_count: int,
    accepted: bool,
    rejected_reason: str | None = None,
) -> None:
    """Log a consolidation quality event to the audit table.

    Args:
        attempt_count: Number of LLM call attempts made.
        accepted: Whether the consolidation was accepted.
        rejected_reason: Human-readable reason for rejection, if any.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO consolidation_quality (attempt_count, accepted, rejected_reason) VALUES (?, ?, ?)",
            (attempt_count, int(accepted), rejected_reason),
        )
        conn.commit()
        log.info(
            "consolidation_quality_logged",
            accepted=accepted,
            attempts=attempt_count,
            reason=rejected_reason,
        )
    finally:
        conn.close()
