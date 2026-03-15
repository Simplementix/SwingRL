"""Core endpoints: /ingest, /consolidate, /health.

/health is unauthenticated (used by Docker healthcheck and Docker depends_on).
/ingest and /consolidate require X-API-Key header.
"""

from __future__ import annotations

import os

import httpx
import structlog
from auth import verify_api_key
from fastapi import APIRouter, Depends
from memory_agents.consolidate import ConsolidateAgent
from memory_agents.ingest import IngestAgent
from pydantic import BaseModel

from db import get_connection

log = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    text: str
    source: str


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    id: int
    status: str = "ok"


class ConsolidateResponse(BaseModel):
    """Response body for POST /consolidate."""

    status: str = "ok"
    consolidated: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    ollama: bool
    db: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    _key: str = Depends(verify_api_key),
) -> IngestResponse:
    """Store a raw memory text with source tag.

    Requires X-API-Key header.
    """
    agent = IngestAgent()
    row_id = agent.store(text=body.text, source=body.source)
    return IngestResponse(id=row_id)


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate(
    _key: str = Depends(verify_api_key),
) -> ConsolidateResponse:
    """Trigger an immediate LLM consolidation of unarchived memories.

    Requires X-API-Key header.
    Returns consolidated=0 if no memories available.
    """
    agent = ConsolidateAgent()
    count = await agent.run()
    return ConsolidateResponse(consolidated=count)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check — no authentication required.

    Returns overall status, Ollama reachability, and SQLite accessibility.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://swingrl-ollama:11434")

    # Check SQLite
    db_ok = False
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception as exc:
        log.warning("health_db_check_failed", error=str(exc))

    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception as exc:
        log.warning("health_ollama_check_failed", error=str(exc))

    overall = "healthy" if (db_ok and ollama_ok) else "degraded"
    return HealthResponse(status=overall, ollama=ollama_ok, db=db_ok)
