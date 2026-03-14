"""API key authentication dependency for the swingrl-memory service.

All endpoints except GET /health must depend on verify_api_key.
Key is read from MEMORY_API_KEY environment variable at request time.
"""

from __future__ import annotations

import os

import structlog
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

log = structlog.get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(_api_key_header)) -> str:
    """Validate X-API-Key header against MEMORY_API_KEY env var.

    Args:
        key: Value of the X-API-Key header (injected by FastAPI Security).

    Returns:
        The validated key string.

    Raises:
        HTTPException: 401 if key is missing or does not match.
    """
    expected = os.environ.get("MEMORY_API_KEY", "")
    if not expected or key != expected:
        log.warning("auth_rejected", has_key=bool(key))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )
    return key
