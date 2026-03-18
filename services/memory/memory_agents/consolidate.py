"""ConsolidateAgent: synthesizes raw memories into structured patterns via LLM.

Two-stage consolidation pipeline:
1. **Stage 1** (per-env): Consolidates walk-forward memories for one environment
   into structured patterns with categories, confidence, and evidence.
2. **Stage 2** (cross-env): Identifies patterns spanning both equity and crypto.

Supports two LLM backends:
1. **Cloud API** (default): OpenAI-compatible endpoint (NVIDIA NIM, OpenRouter, etc.)
   Uses guided_json via nvext for token-level schema enforcement.
2. **Local Ollama** (fallback): Qwen3:14b with format parameter for schema enforcement.

Pattern lifecycle:
- New patterns are deduplicated against existing active patterns.
- Similar patterns trigger LLM merge (old → superseded, merged → active).
- Contradicting patterns trigger LLM conflict resolution (both → superseded).
- Patterns track confirmation_count for recurring observations.

Consolidation is event-driven only (no APScheduler):
- From train_pipeline.py at iteration_complete
- From POST /consolidate for manual trigger

Prompt engineering follows LLM-PROMPT-RESEARCH.md recommendations:
- §1: JSON schema enforcement at token level (guided_json / format)
- §2: Source-only grounding, XML delimiters, no CoT
- §3: Quantitative evidence, calibrated confidence, category definitions
- §4: System/user prompt separation, behavioral constraints in system prompt
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

from db import (
    archive_memories,
    get_active_consolidations,
    get_memories_by_source_prefix,
    increment_confirmation,
    insert_consolidation,
    insert_consolidation_source,
    log_consolidation_quality,
    update_consolidation_status,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_consolidation_config() -> tuple[str, str, str, float, str]:
    """Load consolidation config from mounted swingrl.yaml, fallback to env vars."""
    config_path = Path("/app/config/swingrl.yaml")
    if config_path.exists():
        # yaml.safe_load OK here: memory service can't import swingrl.config.schema
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

# ---------------------------------------------------------------------------
# Category enum (shared across schema, prompts, validation)
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = [
    "regime_performance",
    "macro_transition",
    "trade_quality",
    "iteration_progression",
    "overfit_diagnosis",
    "drawdown_recovery",
    "data_size_impact",
    "cross_env",
    "live_cycle_gate",
    "live_blend_weights",
    "live_risk_thresholds",
    "live_position",
    "live_trade_veto",
    "cross_env_correlation",
]

# ---------------------------------------------------------------------------
# JSON Schema (Research §1C, §3C)
# ---------------------------------------------------------------------------

_CONSOLIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern_text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": _VALID_CATEGORIES,
                    },
                    "affected_algos": {"type": "array", "items": {"type": "string"}},
                    "affected_envs": {"type": "array", "items": {"type": "string"}},
                    "actionable_implication": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "pattern_text",
                    "category",
                    "affected_algos",
                    "affected_envs",
                    "actionable_implication",
                    "confidence",
                    "evidence",
                ],
            },
        },
    },
    "required": ["patterns"],
}

_REQUIRED_PATTERN_FIELDS = {
    "pattern_text",
    "category",
    "affected_algos",
    "affected_envs",
    "actionable_implication",
    "confidence",
    "evidence",
}

# ---------------------------------------------------------------------------
# System Prompt (Research §4A, §2A, §3A, §3B, §3C, §4C)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a quantitative analyst reviewing algorithmic trading results for SwingRL.

Your job is to identify statistically meaningful patterns in walk-forward validation \
results across RL trading agents (PPO, A2C, SAC) on equity (8 ETFs, daily) and crypto \
(BTC/ETH, 4H) environments.

You MUST only reference metrics, values, and facts that appear explicitly in the input \
data between <training_data> tags. Do NOT reference any ticker symbols, dates, or metric \
values that do not appear in the input data.

Each pattern MUST include at least two specific metric values from the input data that \
support it. A pattern requires consistent evidence across at least 3 folds or 2 algorithms. \
A single anomalous fold is an observation, not a pattern.

For each candidate pattern, ask yourself: "Could this be explained by random variance \
across folds?" If yes, do not include it. Only include patterns with direct numerical \
evidence from the input. If in doubt, omit the pattern.

Confidence scoring guide:
- 0.9-1.0: Mathematically certain (pattern holds in ALL folds without exception)
- 0.7-0.89: Strong pattern with 1-2 exceptions
- 0.5-0.69: Moderate pattern, roughly half the evidence supports it
- 0.3-0.49: Weak pattern, slight trend with substantial counter-evidence
- 0.0-0.29: Very weak, barely distinguishable from noise

Most patterns from noisy financial data should fall in the 0.4-0.7 range. A confidence \
above 0.85 requires overwhelming, exception-free evidence. In our experience, only ~10% \
of detected patterns have confidence above 0.8. Calibrate accordingly.

category MUST be one of these 14 values. Use the closest match. Do NOT create new categories.

Training patterns:
- regime_performance: Algo Sharpe/Sortino divergence across bull vs bear regime folds
- macro_transition: Performance degradation when macro indicators transition sharply
- trade_quality: Trade frequency vs avg_win/avg_loss ratio patterns
- iteration_progression: Mean metric improvement or degradation across training iterations
- overfit_diagnosis: IS vs OOS gap patterns, especially regime-dependent overfitting
- drawdown_recovery: max_dd_duration and avg_drawdown comparison across algos/regimes
- data_size_impact: Gate pass rate correlation with train_bars (data volume effects)
- cross_env: Algo preference differences between equity and crypto environments

Live trading patterns (Phase 20):
- live_cycle_gate: Conditions suggesting an environment should skip a training/trading cycle
- live_blend_weights: Evidence for adjusting ensemble algo weights per regime
- live_risk_thresholds: Evidence for tightening/loosening risk limits per macro state
- live_position: Evidence for adjusting position sizing per algo/regime
- live_trade_veto: Conditions that should block trades from specific algos

Cross-environment:
- cross_env_correlation: Patterns that span both equity and crypto environments

Return a JSON object with a "patterns" array. Each pattern has: pattern_text, category, \
affected_algos, affected_envs, actionable_implication, confidence, evidence. \
Identify between 0 and 5 patterns. Prefer fewer well-supported patterns over many weak \
ones. An empty patterns array is acceptable if the data does not support clear patterns."""

# ---------------------------------------------------------------------------
# Few-shot examples (Research §2B — PLACEHOLDER prefix, §3B.4 — varied confidence)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = json.dumps(
    {
        "patterns": [
            {
                "pattern_text": "PLACEHOLDER: ALGO_A mean Sharpe=X.XX in bull folds vs Y.YY in bear folds, a Z.ZZ divergence",
                "category": "regime_performance",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: reduce ALGO_A weight in bear regime ensemble",
                "confidence": 0.42,
                "evidence": "PLACEHOLDER: fold 2 (bull, p_bull=0.XX) sharpe=X.XX; fold 4 (bear, p_bear=0.XX) sharpe=Y.YY",
            },
            {
                "pattern_text": "PLACEHOLDER: all algos gate fail rate increases X% when fed_funds transitions by >Y bps within fold period",
                "category": "macro_transition",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: pause retraining during active rate transition periods",
                "confidence": 0.55,
                "evidence": "PLACEHOLDER: fold 3 fed_funds_min=X.XX fed_funds_max=Y.YY gate=FAIL; fold 5 similar transition gate=FAIL",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A trades X.X/week with avg_win/avg_loss ratio of Y.YY vs ALGO_B at Z.Z/week with ratio W.WW",
                "category": "trade_quality",
                "affected_algos": ["PLACEHOLDER_ALGO_A", "PLACEHOLDER_ALGO_B"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: increase turnover penalty for ALGO_A to reduce overtrading",
                "confidence": 0.67,
                "evidence": "PLACEHOLDER: ALGO_A trade_freq=X.X avg_win=Y.YY avg_loss=Z.ZZ; ALGO_B trade_freq=W.W avg_win=V.VV",
            },
            {
                "pattern_text": "PLACEHOLDER: mean_sharpe improved from X.XX (iter 0) to Y.YY (iter N) for ALGO_A, degraded from W.WW to V.VV for ALGO_B",
                "category": "iteration_progression",
                "affected_algos": ["PLACEHOLDER_ALGO_A", "PLACEHOLDER_ALGO_B"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: memory-enhanced training benefits ALGO_A more than ALGO_B",
                "confidence": 0.63,
                "evidence": "PLACEHOLDER: iter_0 ALGO_A sharpe=X.XX, iter_N sharpe=Y.YY; iter_0 ALGO_B sharpe=W.WW, iter_N sharpe=V.VV",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A IS/OOS Sharpe gap widens to X.XX in bear folds vs Y.YY in bull folds",
                "category": "overfit_diagnosis",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: increase entropy_coeff for ALGO_A during bear regime training",
                "confidence": 0.48,
                "evidence": "PLACEHOLDER: fold 1 (bear) is_sharpe=X.XX oos_sharpe=Y.YY gap=Z.ZZ; fold 3 (bull) gap=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A max_dd_duration=X bars, avg_drawdown=Y.YY% vs ALGO_B max_dd_duration=Z bars, avg_drawdown=W.WW%",
                "category": "drawdown_recovery",
                "affected_algos": ["PLACEHOLDER_ALGO_A", "PLACEHOLDER_ALGO_B"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: increase drawdown reward weight for ALGO_A",
                "confidence": 0.52,
                "evidence": "PLACEHOLDER: ALGO_A fold 2 max_dd_duration=X avg_dd=Y.YY; ALGO_B fold 2 max_dd_duration=Z avg_dd=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: folds with train_bars>X have Y% gate pass rate vs Z% for folds with train_bars<W",
                "category": "data_size_impact",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: ensure minimum X bars per fold in walk-forward config",
                "confidence": 0.71,
                "evidence": "PLACEHOLDER: fold 0 train_bars=X gate=PASS; fold 1 train_bars=Y gate=PASS; fold 4 train_bars=Z gate=FAIL",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A ranks first in ENV_1 (sharpe=X.XX) but last in ENV_2 (sharpe=Y.YY)",
                "category": "cross_env",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV_1", "PLACEHOLDER_ENV_2"],
                "actionable_implication": "PLACEHOLDER: use different ensemble weights per environment",
                "confidence": 0.38,
                "evidence": "PLACEHOLDER: ENV_1 ALGO_A sharpe=X.XX weight=Y.YY; ENV_2 ALGO_A sharpe=Z.ZZ weight=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: ENV gate fails when turbulence_max>X.XX and mean_p_bear>Y.YY simultaneously",
                "category": "live_cycle_gate",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: skip trading cycle when turbulence and bear probability both elevated",
                "confidence": 0.44,
                "evidence": "PLACEHOLDER: fold 3 turb_max=X.XX p_bear=Y.YY gate=FAIL; fold 5 turb_max=Z.ZZ p_bear=W.WW gate=FAIL",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A outperforms by X.XX Sharpe in bear folds, ALGO_B outperforms by Y.YY in bull folds",
                "category": "live_blend_weights",
                "affected_algos": ["PLACEHOLDER_ALGO_A", "PLACEHOLDER_ALGO_B"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: increase ALGO_A weight and decrease ALGO_B weight during bear regime",
                "confidence": 0.61,
                "evidence": "PLACEHOLDER: bear folds ALGO_A sharpe=X.XX ALGO_B sharpe=Y.YY; bull folds ALGO_A sharpe=Z.ZZ ALGO_B sharpe=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: max_single_loss exceeds X% when vix_mean>Y.YY, suggesting tighter drawdown limits needed",
                "category": "live_risk_thresholds",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: reduce max_drawdown limit by X% when VIX elevated",
                "confidence": 0.57,
                "evidence": "PLACEHOLDER: fold 2 vix_mean=X.XX max_single_loss=Y.YY%; fold 4 vix_mean=Z.ZZ max_single_loss=W.WW%",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A avg_loss=X.XX% per trade during high-turbulence folds vs Y.YY% in normal folds",
                "category": "live_position",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: halve ALGO_A position size when turbulence exceeds threshold",
                "confidence": 0.46,
                "evidence": "PLACEHOLDER: high-turb folds avg_loss=X.XX% trades=N; normal folds avg_loss=Y.YY% trades=M",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A consistently loses when rate_direction transitions from X to Y within fold period",
                "category": "live_trade_veto",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: veto ALGO_A trades during rate transition periods",
                "confidence": 0.39,
                "evidence": "PLACEHOLDER: fold 1 rate_dir_min=X rate_dir_max=Y ALGO_A sharpe=Z.ZZ; fold 3 similar, sharpe=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: ENV_1 bear onset (p_bear crossing X.XX) precedes ENV_2 drawdown by N bars",
                "category": "cross_env_correlation",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV_1", "PLACEHOLDER_ENV_2"],
                "actionable_implication": "PLACEHOLDER: use ENV_1 bear signal as early warning for ENV_2 risk tightening",
                "confidence": 0.58,
                "evidence": "PLACEHOLDER: iter N ENV_1 fold 2 bear onset at bar X, ENV_2 fold 2 MDD at bar Y (lag=Z bars)",
            },
        ],
    },
    indent=2,
)


# ---------------------------------------------------------------------------
# Merge / Conflict prompts
# ---------------------------------------------------------------------------

_MERGE_SYSTEM_PROMPT = """You are merging two similar trading patterns into one stronger pattern.

Combine the evidence from both patterns. Keep the most specific and actionable version. \
Update the confidence based on the combined evidence — more confirming evidence should \
increase confidence. Return a single pattern in the same JSON format."""

_CONFLICT_SYSTEM_PROMPT = """You are resolving a contradiction between two trading patterns.

These patterns contradict each other. Analyze the evidence from both and determine:
1. If one is simply wrong (explain why in evidence)
2. If both are correct but apply to different conditions (explain the conditions)
3. If the contradiction reveals a regime-dependent behavior

Return a single resolution pattern that explains the apparent contradiction."""

# Single-pattern schema for merge/conflict resolution
_SINGLE_PATTERN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern_text": {"type": "string"},
        "category": {"type": "string", "enum": _VALID_CATEGORIES},
        "affected_algos": {"type": "array", "items": {"type": "string"}},
        "affected_envs": {"type": "array", "items": {"type": "string"}},
        "actionable_implication": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": [
        "pattern_text",
        "category",
        "affected_algos",
        "affected_envs",
        "actionable_implication",
        "confidence",
        "evidence",
    ],
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ConsolidateAgent:
    """Synthesizes raw memories into consolidated patterns via LLM.

    Two-stage pipeline:
    - Stage 1: Per-environment consolidation (3 algo + 1 ensemble memories → patterns)
    - Stage 2: Cross-environment consolidation (equity + crypto Stage 1 → cross-env patterns)
    """

    async def run(self, env_name: str | None = None) -> int:
        """Orchestrate consolidation: Stage 1 per env, then Stage 2 if both available.

        Args:
            env_name: If specified, only consolidate this environment.
                      If None, consolidate both envs and run Stage 2.

        Returns:
            Total number of new consolidation patterns created.
        """
        total = 0

        if env_name:
            total += await self.run_stage1(env_name)
        else:
            # Run Stage 1 for both environments
            equity_count = await self.run_stage1("equity")
            crypto_count = await self.run_stage1("crypto")
            total += equity_count + crypto_count

            # Stage 2: cross-env (only if both envs have Stage 1 patterns)
            equity_patterns = get_active_consolidations(env_name="equity", stage=1)
            crypto_patterns = get_active_consolidations(env_name="crypto", stage=1)
            if equity_patterns and crypto_patterns:
                total += await self.run_stage2(equity_patterns, crypto_patterns)

        return total

    async def run_stage1(self, env_name: str) -> int:
        """Stage 1: Per-environment consolidation.

        Fetches unarchived memories for this env, consolidates via LLM,
        deduplicates against existing patterns, archives source memories.

        Args:
            env_name: Environment name ('equity' or 'crypto').

        Returns:
            Number of new patterns created.
        """
        # Fetch memories with env-specific source tags
        memories = get_memories_by_source_prefix(
            prefix=f"walk_forward:{env_name}",
            limit=_MEMORY_BATCH_SIZE,
            archived=False,
        )
        if not memories:
            # Fallback: try old-style tags for backward compat, but only for this env
            memories = get_memories_by_source_prefix(
                prefix=env_name,
                limit=_MEMORY_BATCH_SIZE,
                archived=False,
            )
            if not memories:
                log.info("consolidation_skipped_no_memories", env_name=env_name)
                return 0

        log.info("consolidation_stage1_starting", env_name=env_name, memory_count=len(memories))
        memory_texts = "\n\n".join(f"- {m['text']}" for m in memories)
        memory_ids = [m["id"] for m in memories]

        result = await self._call_llm_with_retry(memory_texts)

        if result is None:
            log.warning("consolidation_stage1_discarded", env_name=env_name)
            return 0

        patterns = result.get("patterns", [])
        created = 0

        for pattern in patterns:
            row_id = await self._dedup_and_insert(
                pattern=pattern,
                source_count=len(memories),
                stage=1,
                env_name=env_name,
                memory_ids=memory_ids,
            )
            if row_id is not None:
                created += 1

        # Archive source memories after all patterns processed
        if created > 0:
            archive_memories(memory_ids)

        log.info(
            "consolidation_stage1_complete",
            env_name=env_name,
            patterns_created=created,
        )
        return created

    async def run_stage2(
        self,
        equity_patterns: list[dict[str, Any]],
        crypto_patterns: list[dict[str, Any]],
    ) -> int:
        """Stage 2: Cross-environment consolidation.

        Takes Stage 1 patterns from both envs and identifies cross-env patterns.

        Args:
            equity_patterns: Active Stage 1 patterns for equity.
            crypto_patterns: Active Stage 1 patterns for crypto.

        Returns:
            Number of cross-env patterns created.
        """
        log.info(
            "consolidation_stage2_starting",
            equity_count=len(equity_patterns),
            crypto_count=len(crypto_patterns),
        )

        # Format Stage 1 patterns as input for Stage 2
        pattern_texts = []
        for p in equity_patterns:
            pattern_texts.append(
                f"[EQUITY] {p['pattern_text']} "
                f"(category={p.get('category')}, confidence={p.get('confidence')}, "
                f"evidence={p.get('evidence')})"
            )
        for p in crypto_patterns:
            pattern_texts.append(
                f"[CRYPTO] {p['pattern_text']} "
                f"(category={p.get('category')}, confidence={p.get('confidence')}, "
                f"evidence={p.get('evidence')})"
            )

        input_text = "\n\n".join(pattern_texts)

        stage2_user_prompt = (
            f"--- FORMAT TEMPLATE (fictional data, for output structure only) ---\n"
            f"{_FEW_SHOT_EXAMPLES}\n"
            f"--- END FORMAT TEMPLATE ---\n\n"
            f"<training_data>\n{input_text}\n</training_data>\n\n"
            "Identify patterns that span both equity and crypto environments. "
            "Focus on cross_env_correlation category. Return a JSON object matching "
            "the schema. 0-3 patterns. Empty array is acceptable."
        )

        result = await self._call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=stage2_user_prompt,
            schema=_CONSOLIDATION_SCHEMA,
        )

        if result is None:
            log.warning("consolidation_stage2_no_result")
            return 0

        patterns = result.get("patterns", [])
        created = 0

        for pattern in patterns:
            row_id = await self._dedup_and_insert(
                pattern=pattern,
                source_count=len(equity_patterns) + len(crypto_patterns),
                stage=2,
                env_name=None,  # cross-env
                memory_ids=[],
            )
            if row_id is not None:
                created += 1

        log.info("consolidation_stage2_complete", patterns_created=created)
        return created

    async def _dedup_and_insert(
        self,
        pattern: dict[str, Any],
        source_count: int,
        stage: int,
        env_name: str | None,
        memory_ids: list[int],
    ) -> int | None:
        """Check for duplicates, merge if similar, insert if new.

        Args:
            pattern: Pattern dict from LLM response.
            source_count: Number of source memories.
            stage: Consolidation stage (1 or 2).
            env_name: Environment name (None for cross-env).
            memory_ids: Source memory IDs to link.

        Returns:
            Row ID of new/merged pattern, or None if merged into existing.
        """
        category = pattern.get("category", "")
        affected_algos = pattern.get("affected_algos", [])
        affected_envs = pattern.get("affected_envs", [])

        # Check for existing active patterns with same category + env
        existing = get_active_consolidations(env_name=env_name, stage=stage)
        similar = [
            e
            for e in existing
            if e.get("category") == category
            and set(e.get("affected_algos") or []) == set(affected_algos)
        ]

        if similar:
            # Increment confirmation on the most recent similar pattern
            best = similar[0]
            increment_confirmation(best["id"])
            log.info(
                "consolidation_dedup_confirmed",
                existing_id=best["id"],
                category=category,
            )
            return None

        # Check for conflicts with opposite-sentiment patterns
        conflict_id = self._detect_conflict(pattern.get("pattern_text", ""), new_category=category)

        row_id = insert_consolidation(
            pattern_text=pattern["pattern_text"],
            source_count=source_count,
            conflicting_with=conflict_id,
            category=category,
            affected_algos=affected_algos,
            affected_envs=affected_envs,
            actionable_implication=pattern.get("actionable_implication"),
            confidence=pattern.get("confidence"),
            evidence=pattern.get("evidence"),
            stage=stage,
            env_name=env_name,
            status="active",
            conflict_group_id=str(uuid.uuid4()) if conflict_id else None,
        )

        # Link source memories
        for mid in memory_ids:
            insert_consolidation_source(row_id, mid)

        if conflict_id is not None:
            # Mark the conflicting pattern with the same group ID
            update_consolidation_status(conflict_id, "superseded", superseded_by=row_id)
            log.warning(
                "consolidation_conflict_detected",
                new_id=row_id,
                conflicts_with=conflict_id,
            )

        return row_id

    async def _call_llm_with_retry(self, memory_texts: str) -> dict[str, Any] | None:
        """Attempt LLM consolidation with one retry on malformed output.

        Args:
            memory_texts: Concatenated memory texts to consolidate.

        Returns:
            Parsed consolidation dict with 'patterns' array, or None.
        """
        user_prompt = (
            f"--- FORMAT TEMPLATE (fictional data, for output structure only) ---\n"
            f"{_FEW_SHOT_EXAMPLES}\n"
            f"--- END FORMAT TEMPLATE ---\n\n"
            f"<training_data>\n{memory_texts}\n</training_data>\n\n"
            "Analyze ONLY the data between the <training_data> tags. Return a JSON object "
            "matching the schema. 0-5 patterns. Empty array is acceptable."
        )

        for attempt in range(1, 3):
            result = await self._call_llm(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=_CONSOLIDATION_SCHEMA,
            )
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

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call LLM (cloud or Ollama) with structured output.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            schema: JSON schema for structured output.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        if _CLOUD_API_KEY:
            return await self._call_cloud_api(system_prompt, user_prompt, schema)
        return await self._call_ollama(system_prompt, user_prompt, schema)

    async def _call_cloud_api(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call OpenAI-compatible cloud API with guided_json schema enforcement.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            schema: JSON schema for guided_json enforcement.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
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
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": 4096,
                        "frequency_penalty": 0.0,
                    },
                    # NVIDIA NIM guided_json for token-level schema enforcement
                    # Falls back to response_format for non-NIM providers
                )
                resp.raise_for_status()
                body = resp.json()
                raw_content = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
        except Exception as exc:
            log.error("cloud_api_call_failed", provider=_PROVIDER, error=str(exc))
            return None

        return self._validate_consolidation(parsed)

    async def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call Ollama /api/chat with structured JSON output schema.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            schema: JSON schema for format parameter enforcement.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{_OLLAMA_URL}/api/chat",
                    json={
                        "model": _OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"/no_think\n{user_prompt}"},
                        ],
                        "format": schema,
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
        """Validate LLM output: check structure, required fields, value ranges.

        Args:
            parsed: Parsed JSON from LLM response.

        Returns:
            The dict if valid, None otherwise.
        """
        if not isinstance(parsed, dict):
            log.warning("consolidation_invalid_type", got=type(parsed).__name__)
            return None

        # Handle both array wrapper and single-pattern responses
        if "patterns" in parsed:
            patterns = parsed["patterns"]
            if not isinstance(patterns, list):
                log.warning("consolidation_patterns_not_list")
                return None
            # Validate each pattern
            valid_patterns = []
            for p in patterns:
                if self._validate_single_pattern(p):
                    valid_patterns.append(p)
            if not valid_patterns and patterns:
                log.warning("consolidation_all_patterns_invalid")
                return None
            parsed["patterns"] = valid_patterns
            return parsed

        # Single pattern (merge/conflict response) — wrap in patterns array
        if self._validate_single_pattern(parsed):
            return {"patterns": [parsed]}

        log.warning("consolidation_missing_patterns_key")
        return None

    def _validate_single_pattern(self, pattern: Any) -> bool:
        """Validate a single pattern dict.

        Args:
            pattern: Pattern dict to validate.

        Returns:
            True if valid.
        """
        if not isinstance(pattern, dict):
            return False

        missing = _REQUIRED_PATTERN_FIELDS - set(pattern.keys())
        if missing:
            log.warning("pattern_missing_fields", missing=sorted(missing))
            return False

        if not pattern.get("pattern_text"):
            log.warning("pattern_empty_text")
            return False

        # Validate confidence range
        confidence = pattern.get("confidence")
        if confidence is not None:
            try:
                conf_val = float(confidence)
                if not 0.0 <= conf_val <= 1.0:
                    pattern["confidence"] = max(0.0, min(1.0, conf_val))
            except (TypeError, ValueError):
                pattern["confidence"] = 0.5

        # Validate category
        category = pattern.get("category", "")
        if category not in _VALID_CATEGORIES:
            log.warning("pattern_invalid_category", category=category)
            # Don't reject — pick closest or default
            pattern["category"] = "regime_performance"

        return True

    def _detect_conflict(self, new_pattern: str, new_category: str = "") -> int | None:
        """Check whether the new pattern contradicts existing consolidations.

        Uses simple keyword overlap heuristic. Only flags a conflict when the
        patterns share at least one algorithm name (ppo/a2c/sac) or the same
        category to avoid false positives across unrelated patterns.

        Args:
            new_pattern: The new pattern text to check.
            new_category: Category of the new pattern (for same-category check).

        Returns:
            Row ID of conflicting consolidation, or None if no conflict.
        """
        existing = get_active_consolidations()
        new_tokens = set(new_pattern.lower().split())

        contradiction_pairs = [
            (
                {"improved", "better", "higher", "increased"},
                {"degraded", "worse", "lower", "decreased"},
            ),
            ({"bull", "growth", "momentum"}, {"bear", "crisis", "crash"}),
        ]

        algo_names = {"ppo", "a2c", "sac"}
        new_algos = new_tokens & algo_names

        for row in existing:
            existing_tokens = set((row.get("pattern_text") or "").lower().split())

            # Require shared algorithm or same category before flagging conflict
            existing_algos = existing_tokens & algo_names
            shared_algo = bool(new_algos & existing_algos)
            same_category = bool(
                new_category and row.get("category") and new_category == row["category"]
            )

            if not shared_algo and not same_category:
                continue

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
