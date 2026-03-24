"""Training advisor endpoints: /training/run_config, /training/epoch_advice,
/training/record_outcome, /training/pattern_effectiveness.

All endpoints require X-API-Key header.
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

from db import (
    get_pattern_effectiveness_async,
    insert_pattern_outcome_async,
)

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


class RecordOutcomeRequest(BaseModel):
    """Request body for POST /training/record_outcome."""

    iteration: int
    env_name: str
    gate_passed: bool | None = None
    sharpe: float | None = None
    mdd: float | None = None
    sortino: float | None = None
    pnl: float | None = None
    patterns_presented: list[int] | None = None


class RecordOutcomeResponse(BaseModel):
    """Response body for POST /training/record_outcome."""

    id: int
    status: str = "ok"


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
    # LLM may return rationale as list[str] — normalize to single string
    raw_rationale = result.get("rationale", "cold_start")
    if isinstance(raw_rationale, list):
        raw_rationale = " ".join(str(r) for r in raw_rationale)

    return RunConfigResponse(
        learning_rate=result.get("learning_rate"),
        entropy_coeff=result.get("entropy_coeff"),
        clip_range=result.get("clip_range"),
        n_epochs=result.get("n_epochs"),
        batch_size=result.get("batch_size"),
        gamma=result.get("gamma"),
        reward_weights=result.get("reward_weights", {}),
        rationale=raw_rationale,
    )


@router.post("/epoch_advice", response_model=EpochAdviceResponse)
async def get_epoch_advice(
    body: EpochAdviceRequest,
    _key: str = Depends(verify_api_key),
) -> EpochAdviceResponse:
    """Return reward weight adjustments for the current epoch.

    Epoch advice is bypassed — local Ollama qwen3:1.7b takes ~30s on CPU
    (i5-13500 competing with 3 parallel training processes). Returns safe
    defaults instantly. Epoch snapshots still ingested to memory.

    The swingrl-ollama container and routing code are ready — re-enable
    by uncommenting the agent call below when hardware allows faster inference.

    Requires X-API-Key header.
    """
    # TODO: Re-enable when Ollama inference is < 5s per call.
    # agent = QueryAgent()
    # result: dict[str, Any] = await agent.advise_epoch(body.query)
    # return EpochAdviceResponse(
    #     reward_weights=result.get("reward_weights", {}),
    #     stop_training=bool(result.get("stop_training", False)),
    #     rationale=result.get("rationale", "cold_start"),
    # )
    return EpochAdviceResponse(
        reward_weights={"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
        stop_training=False,
        rationale="epoch_advice_bypassed_cpu_too_slow",
    )


@router.post("/record_outcome", response_model=RecordOutcomeResponse)
async def record_outcome(
    body: RecordOutcomeRequest,
    _key: str = Depends(verify_api_key),
) -> RecordOutcomeResponse:
    """Record training iteration outcome for pattern effectiveness analysis.

    Called by train_pipeline.py after each iteration completes.
    Requires X-API-Key header.
    """
    row_id = await insert_pattern_outcome_async(
        iteration=body.iteration,
        env_name=body.env_name,
        gate_passed=body.gate_passed,
        sharpe=body.sharpe,
        mdd=body.mdd,
        sortino=body.sortino,
        pnl=body.pnl,
        patterns_presented=body.patterns_presented,
    )
    log.info(
        "outcome_recorded",
        iteration=body.iteration,
        env_name=body.env_name,
        row_id=row_id,
    )
    return RecordOutcomeResponse(id=row_id)


@router.get("/pattern_effectiveness")
async def pattern_effectiveness(
    _key: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return pattern effectiveness data joining presentations with outcomes.

    For human review of which patterns actually improved training.
    Requires X-API-Key header.
    """
    return await get_pattern_effectiveness_async()
