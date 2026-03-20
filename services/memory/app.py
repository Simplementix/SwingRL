"""swingrl-memory FastAPI application entry point.

Startup sequence (lifespan):
1. Initialize SQLite database (create tables if not exist)
2. Wait for Ollama /api/tags to return 200 (block, timeout 300s)
3. Register routers and serve

Consolidation is event-driven only (no background scheduler):
- From train_pipeline.py at iteration_complete
- From POST /consolidate for manual trigger

All config via environment variables:
- MEMORY_API_KEY: Required for all endpoints except /health
- MEMORY_DB_PATH: Path to SQLite file (default: /app/db/memory.db)
- OLLAMA_URL: Ollama service URL (default: http://swingrl-ollama:11434)
- OLLAMA_TIMEOUT: LLM call timeout in seconds (default: 30)
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import httpx
import structlog
from fastapi import FastAPI
from memory_agents.consolidate import validate_consolidation_config
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
# Ollama health polling
# ---------------------------------------------------------------------------


async def _wait_for_ollama(
    url: str | None = None,
    timeout_sec: int = 300,
) -> None:
    """Block until Ollama /api/tags returns 200. Raises RuntimeError on timeout.

    Args:
        url: Ollama base URL. Defaults to OLLAMA_URL env var.
        timeout_sec: Total seconds to wait before raising.

    Raises:
        RuntimeError: If Ollama does not become healthy within timeout_sec.
    """
    ollama_url = url or os.environ.get("OLLAMA_URL", "http://swingrl-ollama:11434")
    deadline = time.monotonic() + timeout_sec
    log.info("waiting_for_ollama", url=ollama_url, timeout_sec=timeout_sec)

    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                if resp.status_code == 200:
                    log.info("ollama_ready", url=ollama_url)
                    return
        except Exception:  # nosec B110 — intentional: connection refused is expected during startup
            pass
        await anyio.sleep(5)

    raise RuntimeError(f"Ollama not healthy after {timeout_sec}s at {ollama_url}")


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize service on startup; clean up on shutdown."""
    # 1. Init SQLite tables
    init_db()
    log.info("db_initialized")

    # 2. Eagerly init capacity limiters (avoids TOCTOU race on first request)
    init_capacity_limiters()

    # 3. Validate consolidation config (warn on missing keys/URLs)
    validate_consolidation_config()

    # 4. Wait for Ollama — never start degraded
    await _wait_for_ollama()

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
