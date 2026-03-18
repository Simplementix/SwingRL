"""QueryAgent: advises RL training configuration using consolidated memory patterns.

Two public methods:
- advise_run_config: Returns hyperparameter suggestions + reward weights for a new run.
- advise_epoch: Returns reward weight adjustments + stop_training signal during a run.

Both methods:
- Pull active consolidations (filtered by confidence + status) and raw memories
- Prefer Stage 2 cross-env patterns over Stage 1 per-env patterns
- Call Ollama qwen3:14b with structured JSON output schema
- Return safe clamped defaults on any failure (cold-start guard)
- XML-wrap all memory text injected into prompts
- Track which patterns were presented via pattern_presentations table

Clamp bounds are defined inline (matching src/swingrl/memory/training/bounds.py constants)
to avoid importing from src/swingrl/ (cross-container boundary, Pitfall 4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

from db import (
    get_active_consolidations,
    get_memories,
    insert_pattern_presentation,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror of bounds.py — kept in sync manually)
# ---------------------------------------------------------------------------

_HYPERPARAM_BOUNDS: dict[str, tuple[Any, Any]] = {
    "learning_rate": (1e-5, 1e-3),
    "entropy_coeff": (0.0, 0.05),
    "clip_range": (0.1, 0.4),
    "n_epochs": (3, 20),
    "batch_size": (32, 512),
    "gamma": (0.90, 0.9999),
}

_REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}

_SAFE_DEFAULTS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "entropy_coeff": 0.01,
    "clip_range": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "gamma": 0.99,
    "reward_weights": {"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
    "rationale": "cold_start_defaults",
}

_SAFE_EPOCH_DEFAULTS: dict[str, Any] = {
    "reward_weights": {"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
    "stop_training": False,
    "rationale": "cold_start_defaults",
}

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://swingrl-ollama:11434")
_OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
_QUERY_MODEL = "qwen3:14b"


def _load_min_confidence() -> float:
    """Load min_confidence_for_advice from config, default 0.4."""
    config_path = Path("/app/config/swingrl.yaml")
    if config_path.exists():
        # yaml.safe_load OK here: memory service can't import swingrl.config.schema
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        return float(cons.get("min_confidence_for_advice", 0.4))
    return 0.4


_MIN_CONFIDENCE = _load_min_confidence()

# System prompt for training config / epoch advice endpoints.
_SYSTEM_PROMPT = """You are the training advisor agent for SwingRL, an RL-based swing trading system.

SwingRL context:
- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)
- Algorithms: PPO (on-policy), A2C (on-policy), SAC (off-policy, entropy-maximizing)
- Capital preservation is the PRIMARY constraint — Sortino ratio and MDD are the main metrics
- Market regimes: bull (0), bear (1), crisis (2) — detected by HMM from FRED indicators

Hyperparameter bounds (you MUST stay within these):
- learning_rate: [1e-5, 1e-3]
- entropy_coeff: [0.0, 0.05]
- clip_range: [0.1, 0.4]
- n_epochs: [3, 20]
- batch_size: [32, 512] (must be power of 2)
- gamma: [0.90, 0.9999]

Reward weight bounds (you MUST stay within these, weights should sum to ~1.0):
- profit: [0.10, 0.70]
- sharpe: [0.10, 0.60]
- drawdown: [0.05, 0.50]
- turnover: [0.00, 0.20]

You will receive recent memory patterns from past training runs and queries about current training state.
Provide specific, numerical advice grounded in the memory patterns.
If memory patterns are insufficient, stay close to safe defaults and explain why.
"""

# JSON schema for run_config response
_RUN_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "clip_range": {"type": "number"},
        "n_epochs": {"type": "integer"},
        "batch_size": {"type": "integer"},
        "gamma": {"type": "number"},
        "reward_weights": {
            "type": "object",
            "properties": {
                "profit": {"type": "number"},
                "sharpe": {"type": "number"},
                "drawdown": {"type": "number"},
                "turnover": {"type": "number"},
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["rationale"],
}

# JSON schema for epoch_advice response
_EPOCH_ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "reward_weights": {
            "type": "object",
            "properties": {
                "profit": {"type": "number"},
                "sharpe": {"type": "number"},
                "drawdown": {"type": "number"},
                "turnover": {"type": "number"},
            },
        },
        "stop_training": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["reward_weights", "stop_training", "rationale"],
}


# ---------------------------------------------------------------------------
# Clamping helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _clamp_run_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Clamp hyperparameters from LLM response to safe ranges.

    Args:
        raw: Dict from LLM response.

    Returns:
        New dict with clamped values.
    """
    import math

    out: dict[str, Any] = {}
    for key, (lo, hi) in _HYPERPARAM_BOUNDS.items():
        if key in raw:
            clamped = _clamp(float(raw[key]), float(lo), float(hi))
            if key == "batch_size":
                # Force to nearest power of 2
                clamped = int(2 ** round(math.log2(max(1, clamped))))
            elif key == "n_epochs":
                clamped = int(clamped)
            out[key] = clamped

    if "reward_weights" in raw:
        out["reward_weights"] = _clamp_reward_weights(raw["reward_weights"])

    out["rationale"] = raw.get("rationale", "llm_advised")
    return out


def _clamp_reward_weights(weights: dict[str, Any]) -> dict[str, float]:
    """Clamp reward weights to bounds and renormalize to sum=1.0.

    Args:
        weights: Dict from LLM response.

    Returns:
        Clamped and normalized reward weights.
    """
    out: dict[str, float] = {}
    for key, (lo, hi) in _REWARD_BOUNDS.items():
        raw_val = float(weights.get(key, (lo + hi) / 2))
        out[key] = _clamp(raw_val, lo, hi)

    total = sum(out.values())
    if total == 0:
        mid = {k: (lo + hi) / 2 for k, (lo, hi) in _REWARD_BOUNDS.items()}
        total = sum(mid.values())
        return {k: v / total for k, v in mid.items()}

    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QueryAgent:
    """Advises RL training configuration using memory patterns + Ollama LLM."""

    async def advise_run_config(self, query: str) -> dict[str, Any]:
        """Return LLM-advised run configuration hyperparameters.

        Builds a context prompt from active consolidations (filtered by confidence
        and status) and raw memories, queries Ollama, clamps the response.
        Returns safe defaults on any failure.

        Args:
            query: Context string from MetaTrainingOrchestrator (algo, env, iteration info).

        Returns:
            Dict with keys: learning_rate, entropy_coeff, clip_range, n_epochs,
            batch_size, gamma, reward_weights (dict), rationale.
        """
        context, presented_ids = self._build_context()
        user_content = (
            f"Training context: {query}\n\n"
            f"Relevant memory patterns:\n{context}\n\n"
            "Please recommend hyperparameters for this training run."
        )

        result = await self._call_ollama(user_content, _RUN_CONFIG_SCHEMA)
        if result is None:
            log.warning("run_config_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_DEFAULTS)

        # Track which patterns were presented
        self._track_presentations(presented_ids, query, "run_config", result)

        # Merge with defaults for any missing fields
        merged = dict(_SAFE_DEFAULTS)
        merged.update(result)
        clamped = _clamp_run_config(merged)

        # Ensure reward_weights always present
        if "reward_weights" not in clamped:
            clamped["reward_weights"] = dict(_SAFE_DEFAULTS["reward_weights"])

        log.info("run_config_advised", rationale=clamped.get("rationale", "")[:80])
        return clamped

    async def advise_epoch(self, query: str) -> dict[str, Any]:
        """Return LLM-advised reward weight adjustments mid-training.

        Args:
            query: Context string with current epoch metrics.

        Returns:
            Dict with keys: reward_weights (dict), stop_training (bool), rationale (str).
        """
        context, presented_ids = self._build_context()
        user_content = (
            f"Current epoch state: {query}\n\n"
            f"Memory patterns:\n{context}\n\n"
            "Should I adjust reward weights or stop training early?"
        )

        result = await self._call_ollama(user_content, _EPOCH_ADVICE_SCHEMA)
        if result is None:
            log.warning("epoch_advice_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_EPOCH_DEFAULTS)

        # Track which patterns were presented
        self._track_presentations(presented_ids, query, "epoch_advice", result)

        clamped_weights = _clamp_reward_weights(
            result.get("reward_weights", _SAFE_EPOCH_DEFAULTS["reward_weights"])
        )

        return {
            "reward_weights": clamped_weights,
            "stop_training": bool(result.get("stop_training", False)),
            "rationale": str(result.get("rationale", "llm_advised")),
        }

    def _build_context(self) -> tuple[str, list[int]]:
        """Build a context string from active consolidations and raw memories.

        Filters consolidations by min_confidence and status='active'.
        Prefers Stage 2 (cross-env) over Stage 1 (per-env) when both available.

        Returns:
            Tuple of (formatted context string, list of consolidation IDs presented).
        """
        # Get Stage 2 patterns first (cross-env), then Stage 1
        stage2 = get_active_consolidations(stage=2, min_confidence=_MIN_CONFIDENCE)
        stage1 = get_active_consolidations(stage=1, min_confidence=_MIN_CONFIDENCE)

        # Combine: Stage 2 first (higher priority)
        consolidations = stage2 + stage1
        memories = get_memories(archived=False, limit=20)

        parts: list[str] = []
        presented_ids: list[int] = []

        if consolidations:
            parts.append("=== Consolidated Patterns ===")
            for c in consolidations:
                cat = c.get("category", "unknown")
                conf = c.get("confidence")
                conf_str = f" (confidence={conf:.2f})" if conf is not None else ""
                evidence = c.get("evidence", "")
                evidence_str = f" evidence: {evidence}" if evidence else ""
                parts.append(
                    f'<pattern category="{cat}"{conf_str}>'
                    f"{c.get('pattern_text', '')}"
                    f"{evidence_str}"
                    f"</pattern>"
                )
                if c.get("id"):
                    presented_ids.append(int(c["id"]))

        if memories:
            parts.append("=== Recent Raw Memories ===")
            for m in memories:
                # Text is already XML-wrapped by IngestAgent
                parts.append(m["text"])

        if not parts:
            return "<no_memories>No memories available yet (cold start)</no_memories>", []

        return "\n".join(parts), presented_ids

    def _track_presentations(
        self,
        consolidation_ids: list[int],
        query: str,
        request_type: str,
        result: dict[str, Any],
    ) -> None:
        """Log which patterns were presented to pattern_presentations table.

        Args:
            consolidation_ids: IDs of consolidations included in context.
            query: The query string (extract iteration/env from it).
            request_type: 'run_config' or 'epoch_advice'.
            result: The LLM response (for advice_response summary).
        """
        # Parse iteration and env_name from query if available
        iteration = None
        env_name = None
        for part in query.split():
            if part.startswith("iteration=") or part.startswith("iter="):
                try:
                    iteration = int(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
            if part.startswith("env="):
                env_name = part.split("=")[1]

        advice_summary = result.get("rationale", "")[:200]

        for cid in consolidation_ids:
            try:
                insert_pattern_presentation(
                    consolidation_id=cid,
                    iteration=iteration,
                    env_name=env_name,
                    request_type=request_type,
                    advice_response=advice_summary,
                )
            except Exception as exc:
                log.debug("presentation_tracking_failed", consolidation_id=cid, error=str(exc))

    async def _call_ollama(
        self, user_content: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Call Ollama /api/chat with structured JSON output schema.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output.

        Returns:
            Parsed dict from LLM response, or None on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{_OLLAMA_URL}/api/chat",
                    json={
                        "model": _QUERY_MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "format": schema,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                return json.loads(body["message"]["content"])
        except Exception as exc:
            log.error("query_ollama_call_failed", error=str(exc))
            return None
