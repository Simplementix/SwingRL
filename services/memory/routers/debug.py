"""Debug endpoints for operator inspection of memory store contents.

All debug endpoints require X-API-Key header. They are read-only.
These endpoints are for homelab operators to inspect memory state,
not for production use.
"""

from __future__ import annotations

from typing import Any

import structlog
from auth import verify_api_key
from fastapi import APIRouter, Depends, Query

from db import get_consolidations_async, get_memories_async

log = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/memories")
async def list_memories(
    source: str | None = Query(default=None, description="Filter by source tag"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum rows to return"),
    since: str | None = Query(
        default=None,
        description="ISO datetime string; return rows created after this value",
    ),
    _key: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return raw memories with optional filters.

    Requires X-API-Key header.
    """
    rows = await get_memories_async(source=source, limit=limit, since=since, archived=False)
    log.info("debug_memories_listed", count=len(rows), source=source)
    return rows


@router.get("/consolidations")
async def list_consolidations(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum rows to return"),
    _key: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return recent consolidation patterns.

    Requires X-API-Key header.
    """
    rows = await get_consolidations_async(limit=limit)
    log.info("debug_consolidations_listed", count=len(rows))
    return rows
