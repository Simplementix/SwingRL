"""Training advisor endpoints: /training/run_config, /training/epoch_advice.

Both endpoints require X-API-Key header.
Called by MetaTrainingOrchestrator in src/swingrl/memory/training/meta_orchestrator.py
via MemoryClient in src/swingrl/memory/client.py.
"""

from __future__ import annotations

from typing import Any

import structlog
from auth import verify_api_key
from fastapi import APIRouter, Depends
from memory_agents.query import QueryAgent
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunConfigRequest(BaseModel):
    """Request body for POST /training/run_config."""

    query: str


class RunConfigResponse(BaseModel):
    """Response body for POST /training/run_config.

    All numeric fields are optional — missing means 'use your defaults'.
    The service always clamps to safe ranges before returning.
    """

    learning_rate: float | None = None
    entropy_coeff: float | None = None
    clip_range: float | None = None
    n_epochs: int | None = None
    batch_size: int | None = None
    gamma: float | None = None
    reward_weights: dict[str, float] = {}
    rationale: str = "cold_start"


class EpochAdviceRequest(BaseModel):
    """Request body for POST /training/epoch_advice."""

    query: str


class EpochAdviceResponse(BaseModel):
    """Response body for POST /training/epoch_advice."""

    reward_weights: dict[str, float] = {}
    stop_training: bool = False
    rationale: str = "cold_start"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run_config", response_model=RunConfigResponse)
async def get_run_config(
    body: RunConfigRequest,
    _key: str = Depends(verify_api_key),
) -> RunConfigResponse:
    """Return LLM-advised hyperparameters for a new training run.

    Returns clamped safe defaults on cold start or LLM failure.
    Requires X-API-Key header.
    """
    agent = QueryAgent()
    result: dict[str, Any] = await agent.advise_run_config(body.query)
    return RunConfigResponse(
        learning_rate=result.get("learning_rate"),
        entropy_coeff=result.get("entropy_coeff"),
        clip_range=result.get("clip_range"),
        n_epochs=result.get("n_epochs"),
        batch_size=result.get("batch_size"),
        gamma=result.get("gamma"),
        reward_weights=result.get("reward_weights", {}),
        rationale=result.get("rationale", "cold_start"),
    )


@router.post("/epoch_advice", response_model=EpochAdviceResponse)
async def get_epoch_advice(
    body: EpochAdviceRequest,
    _key: str = Depends(verify_api_key),
) -> EpochAdviceResponse:
    """Return LLM-advised reward weight adjustments for the current epoch.

    Returns clamped safe defaults on cold start or LLM failure.
    Requires X-API-Key header.
    """
    agent = QueryAgent()
    result: dict[str, Any] = await agent.advise_epoch(body.query)
    return EpochAdviceResponse(
        reward_weights=result.get("reward_weights", {}),
        stop_training=bool(result.get("stop_training", False)),
        rationale=result.get("rationale", "cold_start"),
    )
