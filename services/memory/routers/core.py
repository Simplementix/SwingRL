"""Core endpoints: /ingest, /consolidate, /health.

/health is unauthenticated (used by Docker healthcheck and Docker depends_on).
/ingest and /consolidate require X-API-Key header.
"""

from __future__ import annotations

import structlog
from auth import verify_api_key
from fastapi import APIRouter, Depends
from memory_agents.consolidate import ConsolidateAgent
from memory_agents.ingest import IngestAgent
from pydantic import BaseModel, Field

from db import _run_live, get_connection

log = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    text: str = Field(max_length=50_000)
    source: str


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    id: int
    status: str = "ok"


class ConsolidateRequest(BaseModel):
    """Request body for POST /consolidate."""

    env_name: str | None = None  # None = consolidate both envs + cross-env


class ConsolidateResponse(BaseModel):
    """Response body for POST /consolidate."""

    status: str = "ok"
    consolidated: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
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
    row_id = await agent.store_async(text=body.text, source=body.source)
    return IngestResponse(id=row_id)


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate(
    body: ConsolidateRequest | None = None,
    _key: str = Depends(verify_api_key),
) -> ConsolidateResponse:
    """Trigger an immediate LLM consolidation of unarchived memories.

    Requires X-API-Key header.
    Returns consolidated=0 if no memories available.
    Optionally accepts env_name to consolidate only one environment.
    """
    agent = ConsolidateAgent()
    env_name = body.env_name if body else None
    count = await agent.run(env_name=env_name)
    return ConsolidateResponse(consolidated=count)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check — no authentication required.

    Returns overall status and PostgreSQL connectivity.
    """

    # Check PostgreSQL (via live thread pool to avoid blocking event loop)
    def _check_db() -> bool:
        conn = get_connection()
        try:
            conn.execute("SELECT 1")
            return True
        finally:
            conn.close()

    db_ok = False
    try:
        db_ok = await _run_live(_check_db)
    except Exception as exc:
        log.warning("health_db_check_failed", error=str(exc))

    overall = "healthy" if db_ok else "degraded"
    return HealthResponse(status=overall, db=db_ok)
