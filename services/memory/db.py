"""SQLite database layer for the swingrl-memory service.

Manages the memory.db file with tables:
- memories: raw ingested text with source tags
- consolidations: LLM-synthesized patterns from memory batches (enriched schema)
- consolidation_quality: quality audit log for consolidation attempts
- consolidation_sources: join table linking consolidations to source memories
- pattern_presentations: tracks when patterns are presented to the query agent
- pattern_outcomes: records iteration outcomes for pattern effectiveness analysis

All config comes from environment variables — no swingrl imports.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import anyio
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Two-tier async thread pool (Fix #8)
# ---------------------------------------------------------------------------
# CapacityLimiter must be created inside an async context, so we lazily init.
# Live pool (2 threads): insert_memory, get_memories_for_query, record_outcome, health
# Background pool (3 threads): consolidation, pattern queries, debug queries

_live_limiter: anyio.CapacityLimiter | None = None
_background_limiter: anyio.CapacityLimiter | None = None


def init_capacity_limiters() -> None:
    """Eagerly initialize capacity limiters. Must be called inside an async context.

    Call this from the FastAPI startup event (after the event loop is running)
    to avoid TOCTOU races from lazy init in concurrent handlers.
    """
    global _live_limiter, _background_limiter
    _live_limiter = anyio.CapacityLimiter(2)
    _background_limiter = anyio.CapacityLimiter(3)
    log.info("capacity_limiters_initialized", live=2, background=3)


def _get_live_limiter() -> anyio.CapacityLimiter:
    """Return the live-request capacity limiter (2 threads).

    Falls back to lazy creation if init_capacity_limiters() was not called.
    """
    global _live_limiter
    if _live_limiter is None:
        _live_limiter = anyio.CapacityLimiter(2)
    return _live_limiter


def _get_background_limiter() -> anyio.CapacityLimiter:
    """Return the background-task capacity limiter (3 threads).

    Falls back to lazy creation if init_capacity_limiters() was not called.
    """
    global _background_limiter
    if _background_limiter is None:
        _background_limiter = anyio.CapacityLimiter(3)
    return _background_limiter


async def _run_live(func: Any, *args: Any) -> Any:
    """Run a sync function in the live thread pool (2 threads, <5ms)."""
    return await anyio.to_thread.run_sync(lambda: func(*args), limiter=_get_live_limiter())


async def _run_background(func: Any, *args: Any) -> Any:
    """Run a sync function in the background thread pool (3 threads)."""
    return await anyio.to_thread.run_sync(lambda: func(*args), limiter=_get_background_limiter())


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
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_text        TEXT NOT NULL,
    source_count        INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    conflicting_with    INTEGER REFERENCES consolidations(id),
    category            TEXT,
    affected_algos      TEXT,
    affected_envs       TEXT,
    actionable_implication TEXT,
    confidence          REAL,
    evidence            TEXT,
    stage               INTEGER DEFAULT 1,
    env_name            TEXT,
    confirmation_count  INTEGER DEFAULT 0,
    last_confirmed_at   TEXT,
    superseded_by       INTEGER REFERENCES consolidations(id),
    status              TEXT DEFAULT 'active',
    conflict_group_id   TEXT
);
"""
# NOTE: Consolidation indexes are created AFTER _migrate_consolidations() in init_db()
# to handle existing DBs where columns like 'status' and 'category' were added by migration.

_CREATE_CONSOLIDATION_QUALITY = """
CREATE TABLE IF NOT EXISTS consolidation_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_count   INTEGER NOT NULL DEFAULT 1,
    accepted        INTEGER NOT NULL DEFAULT 0,
    rejected_reason TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_CONSOLIDATION_SOURCES = """
CREATE TABLE IF NOT EXISTS consolidation_sources (
    consolidation_id INTEGER NOT NULL REFERENCES consolidations(id),
    memory_id        INTEGER NOT NULL REFERENCES memories(id),
    PRIMARY KEY (consolidation_id, memory_id)
);
"""

_CREATE_PATTERN_PRESENTATIONS = """
CREATE TABLE IF NOT EXISTS pattern_presentations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    consolidation_id  INTEGER NOT NULL REFERENCES consolidations(id),
    presented_at      TEXT NOT NULL DEFAULT (datetime('now')),
    iteration         INTEGER,
    env_name          TEXT,
    request_type      TEXT,
    advice_response   TEXT
);
"""

_CREATE_PATTERN_OUTCOMES = """
CREATE TABLE IF NOT EXISTS pattern_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration           INTEGER NOT NULL,
    env_name            TEXT NOT NULL,
    gate_passed         INTEGER,
    sharpe              REAL,
    mdd                 REAL,
    sortino             REAL,
    pnl                 REAL,
    recorded_at         TEXT NOT NULL DEFAULT (datetime('now')),
    patterns_presented  TEXT
);
"""

# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

_CONSOLIDATION_NEW_COLUMNS = [
    ("category", "TEXT"),
    ("affected_algos", "TEXT"),
    ("affected_envs", "TEXT"),
    ("actionable_implication", "TEXT"),
    ("confidence", "REAL"),
    ("evidence", "TEXT"),
    ("stage", "INTEGER DEFAULT 1"),
    ("env_name", "TEXT"),
    ("confirmation_count", "INTEGER DEFAULT 0"),
    ("last_confirmed_at", "TEXT"),
    ("superseded_by", "INTEGER REFERENCES consolidations(id)"),
    ("status", "TEXT DEFAULT 'active'"),
    ("conflict_group_id", "TEXT"),
]


def _migrate_consolidations(conn: sqlite3.Connection) -> None:
    """Add new columns to consolidations table if they don't exist."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(consolidations)").fetchall()}
    for col_name, col_type in _CONSOLIDATION_NEW_COLUMNS:
        if col_name not in existing:
            conn.execute(
                f"ALTER TABLE consolidations ADD COLUMN {col_name} {col_type}"  # nosec B608 — col names are fixed strings
            )
            log.info("consolidation_column_added", column=col_name)


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
    # Disable thread check: FastAPI async handlers may resume on different threads;
    # each function opens its own connection
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
        conn.executescript(_CREATE_CONSOLIDATION_SOURCES)
        conn.executescript(_CREATE_PATTERN_PRESENTATIONS)
        conn.executescript(_CREATE_PATTERN_OUTCOMES)
        # Migrate existing consolidations table with new columns
        _migrate_consolidations(conn)
        # Create indexes that may not exist on older DBs
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consolidations_status ON consolidations(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consolidations_env_stage ON consolidations(env_name, stage)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_consolidations_category_status "
            "ON consolidations(category, status)"
        )
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
        source: Source tag, e.g. 'walk_forward:equity:ppo'.

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
        if row_id is None:  # pragma: no cover
            row_id = 0
        log.info("memory_inserted", source=source, row_id=row_id)
        return int(row_id)
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


def get_memories_by_source_prefix(
    prefix: str,
    limit: int = 100,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve memories whose source tag starts with prefix.

    Args:
        prefix: Source tag prefix, e.g. 'walk_forward:equity'.
        limit: Maximum rows to return.
        archived: If True, include archived memories.

    Returns:
        List of row dicts.
    """
    conn = get_connection()
    try:
        clauses = ["source LIKE ?"]
        params: list[Any] = [f"{prefix}%"]
        if not archived:
            clauses.append("archived = 0")
        where = "WHERE " + " AND ".join(clauses)
        params.append(limit)
        sql = f"SELECT id, text, source, created_at, archived FROM memories {where} ORDER BY created_at DESC LIMIT ?"  # noqa: E501  # nosec B608
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
    category: str | None = None,
    affected_algos: list[str] | None = None,
    affected_envs: list[str] | None = None,
    actionable_implication: str | None = None,
    confidence: float | None = None,
    evidence: str | None = None,
    stage: int = 1,
    env_name: str | None = None,
    status: str = "active",
    conflict_group_id: str | None = None,
) -> int:
    """Insert a consolidated pattern and return its row ID.

    Args:
        pattern_text: LLM-generated pattern text.
        source_count: Number of raw memories consumed.
        conflicting_with: ID of a conflicting consolidation row, if any.
        category: Pattern category from enum.
        affected_algos: List of algo names (serialized to JSON).
        affected_envs: List of env names (serialized to JSON).
        actionable_implication: What to do differently.
        confidence: Confidence score 0.0-1.0.
        evidence: Supporting evidence text.
        stage: 1=per-env, 2=cross-env.
        env_name: Which environment this pattern is for.
        status: Pattern lifecycle status.
        conflict_group_id: UUID grouping contradicting patterns.

    Returns:
        Integer row ID of the inserted row.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO consolidations "
            "(pattern_text, source_count, conflicting_with, category, affected_algos, "
            "affected_envs, actionable_implication, confidence, evidence, stage, env_name, "
            "status, conflict_group_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pattern_text,
                source_count,
                conflicting_with,
                category,
                json.dumps(affected_algos) if affected_algos else None,
                json.dumps(affected_envs) if affected_envs else None,
                actionable_implication,
                confidence,
                evidence,
                stage,
                env_name,
                status,
                conflict_group_id,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        if row_id is None:  # pragma: no cover
            row_id = 0
        log.info(
            "consolidation_inserted",
            source_count=source_count,
            row_id=row_id,
            category=category,
            stage=stage,
        )
        return int(row_id)
    finally:
        conn.close()


def insert_consolidation_source(consolidation_id: int, memory_id: int) -> None:
    """Link a consolidation to one of its source memories.

    Args:
        consolidation_id: Consolidation row ID.
        memory_id: Memory row ID.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO consolidation_sources (consolidation_id, memory_id) VALUES (?, ?)",
            (consolidation_id, memory_id),
        )
        conn.commit()
    finally:
        conn.close()


def insert_consolidation_sources(consolidation_id: int, memory_ids: list[int]) -> None:
    """Link a consolidation to multiple source memories in a single connection.

    Args:
        consolidation_id: Consolidation row ID.
        memory_ids: List of memory row IDs to link.
    """
    if not memory_ids:
        return
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO consolidation_sources (consolidation_id, memory_id) VALUES (?, ?)",
            [(consolidation_id, mid) for mid in memory_ids],
        )
        conn.commit()
    finally:
        conn.close()


def get_consolidations(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent consolidation patterns.

    Args:
        limit: Maximum rows to return.

    Returns:
        List of row dicts with all consolidation columns.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM consolidations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_deserialize_consolidation(r) for r in rows]
    finally:
        conn.close()


def get_active_consolidations(
    env_name: str | None = None,
    stage: int | None = None,
    min_confidence: float | None = None,
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve active consolidation patterns with optional filters.

    Args:
        env_name: Filter by environment name.
        stage: Filter by consolidation stage (1=per-env, 2=cross-env).
        min_confidence: Minimum confidence threshold.
        categories: Filter to only these category values.
        limit_per_category: Keep top-N per category ranked by composite score.

    Returns:
        List of row dicts with deserialized JSON fields.
    """
    conn = get_connection()
    try:
        clauses = ["status = 'active'"]
        params: list[Any] = []

        if env_name is not None:
            clauses.append("(env_name = ? OR env_name IS NULL)")
            params.append(env_name)
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if min_confidence is not None:
            clauses.append("(confidence >= ? OR confidence IS NULL)")
            params.append(min_confidence)
        if categories:
            placeholders = ",".join("?" * len(categories))
            clauses.append(f"category IN ({placeholders})")
            params.extend(categories)

        where = "WHERE " + " AND ".join(clauses)

        if limit_per_category is not None:
            # Composite score ranking with ROW_NUMBER window function (SQLite 3.25+)
            # score = confidence*0.5 + (confirmation_count/max_active)*0.3 + recency*0.2
            sql = f"""
                WITH scored AS (
                    SELECT *,
                        COALESCE(confidence, 0) * 0.5
                        + CASE WHEN NULLIF(
                            (SELECT MAX(confirmation_count) FROM consolidations WHERE status = 'active'), 0
                          ) IS NULL THEN 0
                          ELSE (COALESCE(confirmation_count, 0) * 1.0
                            / (SELECT MAX(confirmation_count) FROM consolidations WHERE status = 'active')) * 0.3
                        END
                        + MAX(0.0, MIN(1.0,
                            1.0 - (julianday('now') - julianday(created_at)) / 90.0
                          )) * 0.2
                        AS composite_score
                    FROM consolidations
                    {where}
                ),
                ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY category ORDER BY composite_score DESC
                        ) AS _rn
                    FROM scored
                )
                SELECT * FROM ranked WHERE _rn <= ?
                ORDER BY composite_score DESC
            """  # nosec B608 — where clause built from fixed strings + parameterized values
            params.append(limit_per_category)
        else:
            sql = f"SELECT * FROM consolidations {where} ORDER BY confidence DESC, created_at DESC"  # nosec B608

        rows = conn.execute(sql, params).fetchall()
        return [_deserialize_consolidation(r) for r in rows]
    finally:
        conn.close()


def update_consolidation_status(
    row_id: int,
    status: str,
    superseded_by: int | None = None,
) -> None:
    """Update a consolidation's lifecycle status.

    Args:
        row_id: Consolidation row ID.
        status: New status ('active', 'superseded', 'retired').
        superseded_by: ID of the pattern that supersedes this one.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE consolidations SET status = ?, superseded_by = ? WHERE id = ?",
            (status, superseded_by, row_id),
        )
        conn.commit()
        log.info("consolidation_status_updated", row_id=row_id, status=status)
    finally:
        conn.close()


def increment_confirmation(row_id: int) -> None:
    """Increment confirmation count and update last_confirmed_at.

    Args:
        row_id: Consolidation row ID.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE consolidations SET confirmation_count = confirmation_count + 1, "
            "last_confirmed_at = datetime('now') WHERE id = ?",
            (row_id,),
        )
        conn.commit()
    finally:
        conn.close()


def insert_pattern_presentation(
    consolidation_id: int,
    iteration: int | None,
    env_name: str | None,
    request_type: str | None,
    advice_response: str | None,
) -> int:
    """Record that a pattern was presented to the query agent.

    Args:
        consolidation_id: Which consolidation was presented.
        iteration: Training iteration number.
        env_name: Environment name.
        request_type: 'run_config' or 'epoch_advice'.
        advice_response: Summary of the advice given.

    Returns:
        Row ID of the presentation record.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO pattern_presentations "
            "(consolidation_id, iteration, env_name, request_type, advice_response) "
            "VALUES (?, ?, ?, ?, ?)",
            (consolidation_id, iteration, env_name, request_type, advice_response),
        )
        conn.commit()
        row_id = cur.lastrowid if cur.lastrowid is not None else 0
        return int(row_id)
    finally:
        conn.close()


def insert_pattern_outcome(
    iteration: int,
    env_name: str,
    gate_passed: bool | None,
    sharpe: float | None,
    mdd: float | None,
    sortino: float | None,
    pnl: float | None,
    patterns_presented: list[int] | None = None,
) -> int:
    """Record an iteration outcome for pattern effectiveness analysis.

    Args:
        iteration: Training iteration number.
        env_name: Environment name.
        gate_passed: Whether the ensemble gate passed.
        sharpe: Ensemble Sharpe ratio.
        mdd: Maximum drawdown.
        sortino: Sortino ratio.
        pnl: Total PnL.
        patterns_presented: List of consolidation IDs presented this iteration.

    Returns:
        Row ID of the outcome record.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO pattern_outcomes "
            "(iteration, env_name, gate_passed, sharpe, mdd, sortino, pnl, patterns_presented) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                iteration,
                env_name,
                int(gate_passed) if gate_passed is not None else None,
                sharpe,
                mdd,
                sortino,
                pnl,
                json.dumps(patterns_presented) if patterns_presented else None,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid if cur.lastrowid is not None else 0
        return int(row_id)
    finally:
        conn.close()


def get_pattern_effectiveness() -> list[dict[str, Any]]:
    """Join pattern_presentations with pattern_outcomes for human review.

    Returns:
        List of dicts with presentation + outcome data joined by iteration/env.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT pp.consolidation_id, pp.iteration, pp.env_name, pp.request_type, "
            "pp.advice_response, po.gate_passed, po.sharpe, po.mdd, po.sortino, po.pnl "
            "FROM pattern_presentations pp "
            "LEFT JOIN pattern_outcomes po "
            "ON pp.iteration = po.iteration AND pp.env_name = po.env_name "
            "ORDER BY pp.presented_at DESC"
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


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _deserialize_consolidation(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a consolidation row to dict with deserialized JSON fields."""
    d = dict(row)
    for field in ("affected_algos", "affected_envs"):
        val = d.get(field)
        if val and isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d[field] = []
    return d


# ---------------------------------------------------------------------------
# Async wrappers (Fix #8 — non-blocking FastAPI handlers)
# ---------------------------------------------------------------------------
# Sync functions are preserved for backward compatibility.
# Live pool: latency-sensitive request handlers
# Background pool: consolidation, bulk queries, debug


async def insert_memory_async(text: str, source: str) -> int:
    """Async wrapper for insert_memory (live pool)."""
    return await _run_live(insert_memory, text, source)


async def get_memories_async(
    source: str | None = None,
    limit: int = 100,
    since: str | None = None,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """Async wrapper for get_memories (live pool)."""
    return await _run_live(get_memories, source, limit, since, archived)


async def get_memories_by_source_prefix_async(
    prefix: str,
    limit: int = 100,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """Async wrapper for get_memories_by_source_prefix (background pool)."""
    return await _run_background(get_memories_by_source_prefix, prefix, limit, archived)


async def archive_memories_async(row_ids: list[int]) -> None:
    """Async wrapper for archive_memories (background pool)."""
    return await _run_background(archive_memories, row_ids)


async def insert_consolidation_async(
    pattern_text: str,
    source_count: int,
    conflicting_with: int | None = None,
    category: str | None = None,
    affected_algos: list[str] | None = None,
    affected_envs: list[str] | None = None,
    actionable_implication: str | None = None,
    confidence: float | None = None,
    evidence: str | None = None,
    stage: int = 1,
    env_name: str | None = None,
    status: str = "active",
    conflict_group_id: str | None = None,
) -> int:
    """Async wrapper for insert_consolidation (background pool)."""
    return await _run_background(
        insert_consolidation,
        pattern_text,
        source_count,
        conflicting_with,
        category,
        affected_algos,
        affected_envs,
        actionable_implication,
        confidence,
        evidence,
        stage,
        env_name,
        status,
        conflict_group_id,
    )


async def insert_consolidation_source_async(consolidation_id: int, memory_id: int) -> None:
    """Async wrapper for insert_consolidation_source (background pool)."""
    return await _run_background(insert_consolidation_source, consolidation_id, memory_id)


async def insert_consolidation_sources_async(consolidation_id: int, memory_ids: list[int]) -> None:
    """Async wrapper for insert_consolidation_sources (background pool)."""
    return await _run_background(insert_consolidation_sources, consolidation_id, memory_ids)


async def get_consolidations_async(limit: int = 50) -> list[dict[str, Any]]:
    """Async wrapper for get_consolidations (background pool)."""
    return await _run_background(get_consolidations, limit)


async def get_active_consolidations_async(
    env_name: str | None = None,
    stage: int | None = None,
    min_confidence: float | None = None,
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
) -> list[dict[str, Any]]:
    """Async wrapper for get_active_consolidations (background pool)."""
    return await _run_background(
        get_active_consolidations,
        env_name,
        stage,
        min_confidence,
        categories,
        limit_per_category,
    )


async def update_consolidation_status_async(
    row_id: int,
    status: str,
    superseded_by: int | None = None,
) -> None:
    """Async wrapper for update_consolidation_status (background pool)."""
    return await _run_background(update_consolidation_status, row_id, status, superseded_by)


async def increment_confirmation_async(row_id: int) -> None:
    """Async wrapper for increment_confirmation (background pool)."""
    return await _run_background(increment_confirmation, row_id)


async def insert_pattern_presentation_async(
    consolidation_id: int,
    iteration: int | None,
    env_name: str | None,
    request_type: str | None,
    advice_response: str | None,
) -> int:
    """Async wrapper for insert_pattern_presentation (live pool)."""
    return await _run_live(
        insert_pattern_presentation,
        consolidation_id,
        iteration,
        env_name,
        request_type,
        advice_response,
    )


async def insert_pattern_outcome_async(
    iteration: int,
    env_name: str,
    gate_passed: bool | None,
    sharpe: float | None,
    mdd: float | None,
    sortino: float | None,
    pnl: float | None,
    patterns_presented: list[int] | None = None,
) -> int:
    """Async wrapper for insert_pattern_outcome (live pool)."""
    return await _run_live(
        insert_pattern_outcome,
        iteration,
        env_name,
        gate_passed,
        sharpe,
        mdd,
        sortino,
        pnl,
        patterns_presented,
    )


async def get_pattern_effectiveness_async() -> list[dict[str, Any]]:
    """Async wrapper for get_pattern_effectiveness (background pool)."""
    return await _run_background(get_pattern_effectiveness)


async def log_consolidation_quality_async(
    attempt_count: int,
    accepted: bool,
    rejected_reason: str | None = None,
) -> None:
    """Async wrapper for log_consolidation_quality (background pool)."""
    return await _run_background(
        log_consolidation_quality,
        attempt_count,
        accepted,
        rejected_reason,
    )
