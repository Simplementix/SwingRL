"""ConsolidateAgent: synthesizes raw memories into structured patterns via LLM.

Supports two backends:
1. **Cloud API** (default): OpenAI-compatible endpoint (NVIDIA NIM, OpenRouter, etc.)
   Configured via memory_agent.consolidation in config/swingrl.yaml.
   API keys flow through env vars (e.g. NVIDIA_API_KEY) for security.
2. **Local Ollama** (fallback): Used when no cloud API key is available.

Runs as a background job every 30 minutes (scheduled by APScheduler in main.py).
Can also be triggered explicitly via POST /consolidate.

Consolidation flow:
1. Fetch recent unarchived memories from SQLite
2. Build a domain-specific prompt with SwingRL context
3. Call LLM with JSON schema format for structured output
4. Validate the response; retry once if malformed; discard on second failure
5. Check for contradictions against existing consolidations
6. Archive source memories and insert the new pattern
7. Log quality result to consolidation_quality table
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
    archive_memories,
    get_consolidations,
    get_memories,
    insert_consolidation,
    log_consolidation_quality,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _load_consolidation_config() -> tuple[str, str, str, float]:
    """Load consolidation config from mounted swingrl.yaml, fallback to env vars."""
    config_path = Path("/app/config/swingrl.yaml")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        provider_key = cons.get("provider", "nvidia")
        providers = cons.get("providers", {})
        provider = providers.get(provider_key, {})
        base_url = provider.get("base_url", "https://integrate.api.nvidia.com/v1")
        # API key: check env var first (secrets), then config
        api_key = os.environ.get(f"{provider_key.upper()}_API_KEY", provider.get("api_key", ""))
        model = cons.get("model") or provider.get("default_model", "moonshotai/kimi-k2.5")
        timeout = float(cons.get("timeout_sec", 120))
        return base_url, api_key, model, timeout, provider_key
    # Fallback to env vars (backward compat)
    return (
        os.environ.get("CONSOLIDATION_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        os.environ.get("CONSOLIDATION_API_KEY", ""),
        os.environ.get("CONSOLIDATION_MODEL", "moonshotai/kimi-k2.5"),
        float(os.environ.get("CONSOLIDATION_TIMEOUT", "120")),
        "env",
    )


_CLOUD_BASE_URL, _CLOUD_API_KEY, _CLOUD_MODEL, _CLOUD_TIMEOUT, _PROVIDER = (
    _load_consolidation_config()
)

# Local Ollama settings (fallback when no cloud API key is available)
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://swingrl-ollama:11434")
_OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
_OLLAMA_MODEL = os.environ.get("OLLAMA_CONSOLIDATION_MODEL", "qwen3:14b")

_MEMORY_BATCH_SIZE = 50

# Domain-specific system prompt — hardcoded and version-controlled.
# This gives the LLM context about SwingRL's architecture, constraints, and goals.
_SYSTEM_PROMPT = """You are the memory consolidation agent for SwingRL, an RL-based swing trading system.

SwingRL context:
- Two trading environments: equity daily (SPY, QQQ, IWM, EFA, TLT, GLD, USO, VNQ via Alpaca) and crypto 4H (BTC/ETH via Binance.US)
- Three RL algorithms: PPO (on-policy, clipped objective), A2C (on-policy, advantage actor-critic), SAC (off-policy, entropy-maximizing)
- Capital preservation is the PRIMARY constraint — never lose more than you can recover from
- Performance metrics: Sharpe ratio (risk-adjusted), MDD (maximum drawdown, stored as negative float), Sortino (upside-focused), Calmar (return/max-drawdown)
- Market regimes detected by HMM: bull (regime 0), bear (regime 1), crisis (regime 2)
- Reward components: profit (0.1-0.7), sharpe (0.1-0.6), drawdown (0.05-0.5), turnover (0.0-0.2)

Your job: Given a batch of raw training memories, synthesize the cross-run patterns that explain WHY certain configurations work better in certain regimes or environments.

Focus on:
- Which hyperparameter ranges produced the best Sharpe/Sortino for each algo x env combination
- How reward weight adjustments affected drawdown and win rate
- Macro correlations: which FRED indicators (VIX, DFF, T10Y2Y, CPI, UNRATE) predicted regime transitions
- Cross-asset patterns: when equity and crypto performance diverged or converged

Output a single consolidated pattern — the most important insight from this memory batch.
If there is no clear pattern, output a placeholder with low confidence.

Respond with ONLY a JSON object (no markdown, no explanation) with these exact fields:
- pattern_text: A specific, actionable pattern with numbers
- affected_algos: Array of algorithm names (ppo, a2c, sac)
- affected_envs: Array of environment names (equity, crypto)
- actionable_implication: What the training orchestrator should do differently
- confidence: Number from 0.0 to 1.0
"""

# JSON schema for structured output (used by Ollama format parameter)
_CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_text": {
            "type": "string",
            "description": "A specific, actionable pattern with numbers. E.g. 'PPO equity Sharpe improved 0.3 when learning_rate was reduced to 3e-5 during bear regime (VIX > 25)'",
        },
        "affected_algos": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of affected algorithms: ppo, a2c, sac",
        },
        "affected_envs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of affected environments: equity, crypto",
        },
        "actionable_implication": {
            "type": "string",
            "description": "What the training orchestrator should do differently based on this pattern",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this pattern from 0.0 to 1.0",
        },
    },
    "required": [
        "pattern_text",
        "affected_algos",
        "affected_envs",
        "actionable_implication",
        "confidence",
    ],
}

_REQUIRED_FIELDS = set(_CONSOLIDATION_SCHEMA["required"])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ConsolidateAgent:
    """Synthesizes raw memories into consolidated patterns via LLM."""

    async def run(self) -> int:
        """Fetch unarchived memories, consolidate via LLM, archive source rows.

        Returns:
            Number of new consolidation patterns created (0 or 1 per batch).
        """
        memories = get_memories(archived=False, limit=_MEMORY_BATCH_SIZE)
        if not memories:
            log.info("consolidation_skipped_no_memories")
            return 0

        log.info("consolidation_starting", memory_count=len(memories))
        memory_texts = "\n\n".join(f"- {m['text']}" for m in memories)
        memory_ids = [m["id"] for m in memories]

        result = await self._call_llm_with_retry(memory_texts)

        if result is None:
            log.warning("consolidation_discarded_all_attempts_failed")
            return 0

        conflict_id = self._detect_conflict(result["pattern_text"])

        row_id = insert_consolidation(
            pattern_text=result["pattern_text"],
            source_count=len(memories),
            conflicting_with=conflict_id,
        )
        archive_memories(memory_ids)

        if conflict_id is not None:
            log.warning(
                "consolidation_conflict_detected",
                new_id=row_id,
                conflicts_with=conflict_id,
            )

        return 1

    async def _call_llm_with_retry(self, memory_texts: str) -> dict[str, Any] | None:
        """Attempt LLM consolidation with one retry on malformed output.

        Args:
            memory_texts: Concatenated memory texts to consolidate.

        Returns:
            Parsed consolidation dict, or None if both attempts failed.
        """
        for attempt in range(1, 3):
            if _CLOUD_API_KEY:
                result = await self._call_cloud_api(memory_texts)
            else:
                result = await self._call_ollama(memory_texts)
            if result is not None:
                log_consolidation_quality(attempt_count=attempt, accepted=True)
                return result
            log.warning("consolidation_attempt_failed", attempt=attempt)

        log_consolidation_quality(
            attempt_count=2,
            accepted=False,
            rejected_reason="Missing required fields after 2 attempts",
        )
        return None

    async def _call_cloud_api(self, memory_texts: str) -> dict[str, Any] | None:
        """Call OpenAI-compatible cloud API for consolidation.

        Works with NVIDIA NIM, OpenRouter, Together AI, or any provider
        that implements /v1/chat/completions.

        Args:
            memory_texts: Formatted memory text batch.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        user_content = (
            f"Please consolidate the following training memories into a single pattern:\n\n"
            f"{memory_texts}\n\n"
            "Respond with ONLY a JSON object matching the required schema. No markdown."
        )

        try:
            async with httpx.AsyncClient(timeout=_CLOUD_TIMEOUT) as client:
                resp = await client.post(
                    f"{_CLOUD_BASE_URL.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {_CLOUD_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _CLOUD_MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                raw_content = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
        except Exception as exc:
            log.error("cloud_api_call_failed", provider=_PROVIDER, error=str(exc))
            return None

        return self._validate_consolidation(parsed)

    async def _call_ollama(self, memory_texts: str) -> dict[str, Any] | None:
        """Single Ollama /api/chat call for consolidation (local fallback).

        Args:
            memory_texts: Formatted memory text batch.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        user_content = (
            f"Please consolidate the following training memories into a single pattern:\n\n"
            f"{memory_texts}\n\n"
            "Respond with a JSON object matching the required schema."
        )

        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{_OLLAMA_URL}/api/chat",
                    json={
                        "model": _OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "format": _CONSOLIDATION_SCHEMA,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                raw_content = body["message"]["content"]
                parsed = json.loads(raw_content)
        except Exception as exc:
            log.error("ollama_call_failed", error=str(exc))
            return None

        return self._validate_consolidation(parsed)

    def _validate_consolidation(self, parsed: Any) -> dict[str, Any] | None:
        """Check that all required fields are present and non-empty.

        Args:
            parsed: Parsed JSON dict from LLM response.

        Returns:
            The dict if valid, None otherwise.
        """
        if not isinstance(parsed, dict):
            log.warning("consolidation_invalid_type", got=type(parsed).__name__)
            return None

        missing = _REQUIRED_FIELDS - set(parsed.keys())
        if missing:
            log.warning("consolidation_missing_fields", missing=sorted(missing))
            return None

        if not parsed.get("pattern_text"):
            log.warning("consolidation_empty_pattern_text")
            return None

        return parsed

    def _detect_conflict(self, new_pattern: str) -> int | None:
        """Check whether the new pattern contradicts existing consolidations.

        Uses simple keyword overlap heuristic — sufficient without embeddings.

        Args:
            new_pattern: The new pattern text to check.

        Returns:
            Row ID of conflicting consolidation, or None if no conflict.
        """
        existing = get_consolidations(limit=20)
        new_tokens = set(new_pattern.lower().split())

        # Look for contradicting signals: "improved" vs "degraded", "increased" vs "decreased"
        contradiction_pairs = [
            (
                {"improved", "better", "higher", "increased"},
                {"degraded", "worse", "lower", "decreased"},
            ),
            ({"bull", "growth", "momentum"}, {"bear", "crisis", "crash"}),
        ]

        for row in existing:
            existing_tokens = set(row["pattern_text"].lower().split())
            for pos_words, neg_words in contradiction_pairs:
                new_has_pos = bool(new_tokens & pos_words)
                new_has_neg = bool(new_tokens & neg_words)
                existing_has_pos = bool(existing_tokens & pos_words)
                existing_has_neg = bool(existing_tokens & neg_words)

                if (new_has_pos and existing_has_neg) or (new_has_neg and existing_has_pos):
                    log.info(
                        "consolidation_conflict_candidate",
                        new_pattern_snippet=new_pattern[:80],
                        existing_id=row["id"],
                    )
                    return int(row["id"])

        return None
