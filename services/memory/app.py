"""swingrl-memory FastAPI application entry point.

Startup sequence (lifespan):
1. Initialize PostgreSQL database (create tables if not exist)
2. Initialize capacity limiters
3. Validate consolidation config (cloud LLM provider)
4. Validate query config
5. Log ready

Consolidation is event-driven only (no background scheduler):
- From train_pipeline.py at iteration_complete
- From POST /consolidate for manual trigger

All config via environment variables:
- MEMORY_API_KEY: Required for all endpoints except /health
- DATABASE_URL: PostgreSQL connection URL
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from memory_agents.consolidate import validate_consolidation_config
from memory_agents.query import validate_query_config
from routers import core, debug, training

from db import init_capacity_limiters, init_db

# Configure structlog for JSON output (matches Docker log collection)
_LOG_LEVEL_MAP = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_LOG_LEVEL = _LOG_LEVEL_MAP.get(os.environ.get("LOG_LEVEL", "INFO").upper(), 20)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVEL),
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize service on startup; clean up on shutdown."""
    # 1. Init PostgreSQL tables
    init_db()
    log.info("db_initialized")

    # 2. Eagerly init capacity limiters (avoids TOCTOU race on first request)
    init_capacity_limiters()

    # 3. Validate consolidation config (warn on missing keys/URLs)
    validate_consolidation_config()

    # 4. Validate query agent cloud config
    validate_query_config()

    log.info("memory_service_ready")

    yield

    log.info("memory_service_stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="swingrl-memory",
    description="Memory store and LLM consolidation service for SwingRL training",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(core.router)
app.include_router(training.router, prefix="/training")
app.include_router(debug.router, prefix="/debug")
