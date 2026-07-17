"""Meta-Trader trade-time endpoints: POST /trade/commentary (Task 12).

Rotation-gated MT commentary skeleton. The swingrl scheduler POSTs one cycle's
context here after each inference cycle (only when ``meta_trader.enabled``). This
service renders the ``mt-commentary-v0`` prompt, calls the configured provider
(shadow mode — never changes a live order), and writes one ``llm_calls`` row plus
one shadow ``MT_commentary`` ``intent_records`` row, capped at <=1 intent per cycle
(A14). Graders/verdicts land in Plan B; day-one intents accumulate ungraded.

All endpoints require the X-API-Key header.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
import structlog
from auth import verify_api_key
from fastapi import APIRouter, Depends
from memory_agents.query import QueryAgent
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from db import insert_llm_call_async, insert_trade_commentary_async

log = structlog.get_logger(__name__)
router = APIRouter()

_PROMPT_VERSION = "mt-commentary-v0"


def _horizon_spec_for_env(environment: str) -> dict[str, Any]:
    """System-written horizon spec per env (D-T2.8, A14): never coach-chosen."""
    if environment == "crypto":
        return {"type": "next_n_cycles", "n": 6}
    return {"type": "wall_clock_hours", "hours": 24}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TradeCommentaryRequest(BaseModel):
    """Request body for POST /trade/commentary."""

    cycle_id: int
    environment: str
    algorithm: str
    deployed_iteration: int
    regime: dict[str, Any] = Field(default_factory=dict)
    proposals_summary: str = ""


class TradeCommentaryResponse(BaseModel):
    """Response body for POST /trade/commentary."""

    llm_call_id: int | None = None
    intent_id: int | None = None
    status: str = "ok"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/commentary", response_model=TradeCommentaryResponse)
async def trade_commentary(
    body: TradeCommentaryRequest,
    _key: str = Depends(verify_api_key),
) -> Any:
    """Record shadow Meta-Trader commentary for one inference cycle.

    Requires X-API-Key. Writes llm_calls + one MT_commentary intent atomically on a
    successful structured response; writes llm_calls only (counted, no intent) when
    the provider chain fails; returns 409 when the A14 per-cycle cap already holds.
    """
    query = (
        f"cycle_id={body.cycle_id} environment={body.environment} "
        f"algorithm={body.algorithm} deployed_iteration={body.deployed_iteration}\n"
        f"regime={json.dumps(body.regime)}\n"
        f"per_algo_proposals={body.proposals_summary}"
    )

    agent = QueryAgent()
    result = await agent.advise_trade_commentary(query)

    horizon_spec = _horizon_spec_for_env(body.environment)
    bet = (result or {}).get("falsifiable_bet") or {}
    has_bet = (
        result is not None
        and isinstance(bet, dict)
        and bet.get("metric")
        and bet.get("direction") in ("up", "down")
        and isinstance(bet.get("baseline_value"), (int, float))
    )

    # Fail-open but counted: a failed/unstructured call still records the transcript,
    # but writes no intent (no bet to grade).
    if not has_bet:
        llm_call_id = await insert_llm_call_async(
            call_type="trade_commentary",
            cycle_id=body.cycle_id,
            provider=(result or {}).get("provider", agent._last_provider),
            model=(result or {}).get("model", agent._last_model),
            prompt_version=_PROMPT_VERSION,
            prompt_text=query,
            response_text=json.dumps(result) if result else None,
            response_parsed=json.dumps(result) if result else None,
            latency_ms=(result or {}).get("latency_ms"),
            success=result is not None,
            error_text=None if result is not None else "provider_exhausted",
            environment=body.environment,
            algorithm=body.algorithm,
        )
        log.info("trade_commentary_no_intent", cycle_id=body.cycle_id, llm_call_id=llm_call_id)
        return TradeCommentaryResponse(llm_call_id=llm_call_id, intent_id=None, status="no_intent")

    evidence = {
        "regime": body.regime,
        "proposals_summary": body.proposals_summary,
        "diagnosis": result.get("diagnosis"),
        "matched_weakness": result.get("matched_weakness"),
    }
    proposal = {
        "proposal": result.get("proposal"),
        "matched_weakness": result.get("matched_weakness"),
    }

    try:
        llm_call_id, intent_id = await insert_trade_commentary_async(
            cycle_id=body.cycle_id,
            environment=body.environment,
            algorithm=body.algorithm,
            deployed_iteration=body.deployed_iteration,
            provider=result.get("provider", agent._last_provider),
            model=result.get("model", agent._last_model),
            prompt_version=_PROMPT_VERSION,
            prompt_text=query,
            response_text=json.dumps(result),
            response_parsed=json.dumps(result),
            latency_ms=result.get("latency_ms"),
            success=True,
            error_text=None,
            tokens_in=None,
            tokens_out=None,
            evidence=json.dumps(evidence),
            proposal=json.dumps(proposal),
            bet_metric=str(bet["metric"]),
            bet_direction=str(bet["direction"]),
            bet_baseline_value=float(bet["baseline_value"]),
            horizon_spec=json.dumps(horizon_spec),
        )
    except psycopg.errors.UniqueViolation:
        # A14 volume cap: one MT_commentary intent per inference cycle already exists.
        log.warning("trade_commentary_cap_hit", cycle_id=body.cycle_id)
        return JSONResponse(
            status_code=409,
            content={"llm_call_id": None, "intent_id": None, "status": "cap_reached"},
        )

    return TradeCommentaryResponse(llm_call_id=llm_call_id, intent_id=intent_id, status="ok")
