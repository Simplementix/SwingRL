"""ConsolidateAgent: synthesizes raw memories into structured patterns via LLM.

Two-stage consolidation pipeline:
1. **Stage 1** (per-env): Consolidates walk-forward memories for one environment
   into structured patterns with categories, confidence, and evidence.
2. **Stage 2** (cross-env): Identifies patterns spanning both equity and crypto.

Supports two cloud LLM providers:
1. **Primary** (OpenRouter): nemotron-120b via OpenRouter free tier.
   Uses json_object response_format for structured output.
2. **Backup** (NVIDIA NIM): kimi-k2.5 via NVIDIA Integrate API.
   Primary failures automatically fall back to backup.

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

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

from db import (
    archive_memories_async,
    get_active_consolidations_async,
    get_memories_by_source_prefix_async,
    increment_confirmation_async,
    insert_consolidation_async,
    insert_consolidation_sources_async,
    log_consolidation_quality_async,
    update_consolidation_status_async,
)

log = structlog.get_logger(__name__)

# Config path: env var override for non-Docker environments, default for Docker.
_CONFIG_PATH = Path(os.environ.get("SWINGRL_CONFIG_PATH", "/app/config/swingrl.yaml"))


class _RateLimitError(Exception):
    """Internal signal for 429 rate-limit responses to trigger backoff retry."""

    def __init__(self, status_code: int, provider: str, body: str) -> None:
        self.status_code = status_code
        self.provider = provider
        self.body = body
        super().__init__(f"{provider} returned {status_code}: {body[:200]}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _build_provider_entry(
    name: str,
    provider_cfg: dict[str, Any],
    global_timeout: float = 600.0,
) -> dict[str, Any]:
    """Build a normalized provider entry from config.

    Resolves API key from env var ({NAME}_API_KEY) if not set in config.
    Reads per-provider timeout_sec and max_tokens with fallbacks.
    """
    env_key = f"{name.upper()}_API_KEY"
    return {
        "base_url": provider_cfg.get("base_url", ""),
        "api_key": os.environ.get(env_key, provider_cfg.get("api_key", "")),
        "model": provider_cfg.get("default_model", ""),
        "timeout": float(provider_cfg.get("timeout_sec", global_timeout)),
        "max_tokens": int(provider_cfg.get("max_tokens", 32768)),
        "provider": name,
    }


def _load_consolidation_config() -> dict[str, Any]:
    """Load consolidation config with primary + backup providers.

    Primary provider is determined by consolidation.provider config key.
    Backup is OpenRouter (fallback). All providers read per-provider
    timeout_sec and max_tokens from config.
    """
    config_path = _CONFIG_PATH
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        providers = cons.get("providers", {})
        global_timeout = float(cons.get("timeout_sec", 600))
        primary_name = cons.get("provider", "mistral")

        # Build primary from configured provider
        primary_cfg = providers.get(primary_name, {})
        primary = _build_provider_entry(primary_name, primary_cfg, global_timeout)

        # Build backup — use openrouter if primary isn't openrouter, else nvidia
        backup_name = "openrouter" if primary_name != "openrouter" else "nvidia"
        backup_cfg = providers.get(backup_name, {})
        backup = _build_provider_entry(backup_name, backup_cfg, global_timeout)

        return {
            "primary": primary,
            "backup": backup,
            "inter_phase_delay_sec": int(cons.get("inter_phase_delay_sec", 60)),
            "rate_limit_max_retries": int(cons.get("rate_limit_max_retries", 3)),
            "rate_limit_backoff_base_sec": int(cons.get("rate_limit_backoff_base_sec", 30)),
        }

    # Fallback to env vars (backward compat)
    return {
        "primary": {
            "base_url": "https://api.mistral.ai/v1",
            "api_key": os.environ.get("MISTRAL_API_KEY", ""),
            "model": "mistral-large-latest",
            "timeout": 600.0,
            "max_tokens": 128000,
            "provider": "mistral",
        },
        "backup": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "timeout": 1800.0,
            "max_tokens": 32768,
            "provider": "openrouter",
        },
        "inter_phase_delay_sec": 60,
        "rate_limit_max_retries": 3,
        "rate_limit_backoff_base_sec": 30,
    }


_DEFAULT_PROVIDER_CONFIG: dict[str, Any] = {
    "primary": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key": "",
        "model": "mistral-large-latest",
        "timeout": 600.0,
        "max_tokens": 128000,
        "provider": "mistral",
    },
    "backup": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "timeout": 1800.0,
        "max_tokens": 32768,
        "provider": "openrouter",
    },
    "inter_phase_delay_sec": 60,
    "rate_limit_max_retries": 3,
    "rate_limit_backoff_base_sec": 30,
}

try:
    _PROVIDER_CONFIG = _load_consolidation_config()
except Exception as _cfg_exc:
    log.warning(
        "consolidation_config_load_failed",
        error=str(_cfg_exc),
        fallback="defaults",
    )
    _PROVIDER_CONFIG = _DEFAULT_PROVIDER_CONFIG


def validate_consolidation_config() -> None:
    """Validate that consolidation config is usable at startup.

    Logs warnings for missing API keys or invalid base URLs rather than
    silently falling back to defaults. Call from app lifespan.
    """
    primary = _PROVIDER_CONFIG["primary"]
    backup = _PROVIDER_CONFIG["backup"]
    log.info(
        "consolidation_config_validated",
        primary_provider=primary["provider"],
        primary_model=primary["model"],
        primary_has_key=bool(primary["api_key"]),
        backup_provider=backup["provider"],
        backup_model=backup["model"],
        backup_has_key=bool(backup["api_key"]),
    )


_MEMORY_BATCH_SIZE = 200

# ---------------------------------------------------------------------------
# Epoch aggregation config (read from mounted swingrl.yaml)
# ---------------------------------------------------------------------------

# SB3 algos that produce approx_kl — architectural fact, not tunable.
_KL_CAPABLE_ALGOS: frozenset[str] = frozenset({"ppo"})

# Default aggregation parameters (overridden by yaml config)
_DEFAULT_AGGREGATION_CONFIG: dict[str, Any] = {
    "outlier_iqr_mild": 1.5,
    "outlier_iqr_extreme": 3.0,
    "max_outlier_events": 3,
    "skewness_min_n": 8,
    "confidence_n_low": 5,
    "confidence_n_high": 15,
    "epoch_cadence_ppo": 20,
    "epoch_cadence_a2c": 2000,
    "epoch_cadence_sac": 10000,
    "epoch_cadence_default": 100,
}


def _load_aggregation_config() -> dict[str, Any]:
    """Load epoch aggregation config from mounted swingrl.yaml.

    Reads memory_agent section for outlier detection, skewness, confidence,
    and epoch cadence parameters. Falls back to _DEFAULT_AGGREGATION_CONFIG
    for any missing keys.

    Same yaml.safe_load pattern as _load_consolidation_config() — memory
    service cannot import src/swingrl/config/schema.py.
    """
    config_path = _CONFIG_PATH
    result = dict(_DEFAULT_AGGREGATION_CONFIG)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text()) or {}
            mem = cfg.get("memory_agent", {})
            for key in _DEFAULT_AGGREGATION_CONFIG:
                val = mem.get(key)
                if val is not None:
                    expected_type = type(_DEFAULT_AGGREGATION_CONFIG[key])
                    result[key] = expected_type(val)
        except Exception:  # nosec B110  # Fail-open: use defaults on parse error
            log.warning("aggregation_config_load_failed", fallback="defaults")
    return result


_AGGREGATION_CONFIG = _load_aggregation_config()

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
    "reward_shaping",
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
# Phase-specific system prompts (Research §4A, §2A, §3A, §3B, §3C, §4C)
# ---------------------------------------------------------------------------

_PHASE_A_SYSTEM_PROMPT = """You are analyzing walk-forward validation results and trading patterns \
for SwingRL, an RL-based swing trading system.

Your job is to identify statistically meaningful patterns in per-algo fold OOS/IS metrics, gate \
results, overfit classification, regime context, macro features, and trading patterns per \
asset/regime/macro across RL trading agents (PPO, A2C, SAC) on equity (8 ETFs, daily) and crypto \
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

category MUST be one of these 11 values. Use the closest match. Do NOT create new categories.

Training patterns:
- regime_performance: Algo Sharpe/Sortino divergence across bull vs bear regime folds
- macro_transition: Performance degradation when macro indicators transition sharply
- trade_quality: Trade frequency vs avg_win/avg_loss ratio patterns
- overfit_diagnosis: IS vs OOS gap patterns, especially regime-dependent overfitting
- drawdown_recovery: max_dd_duration and avg_drawdown comparison across algos/regimes
- data_size_impact: Gate pass rate correlation with train_bars (data volume effects)

Live trading patterns (Phase 20):
- live_cycle_gate: Conditions suggesting an environment should skip a training/trading cycle
- live_blend_weights: Evidence for adjusting ensemble algo weights per regime
- live_risk_thresholds: Evidence for tightening/loosening risk limits per macro state
- live_position: Evidence for adjusting position sizing per algo/regime
- live_trade_veto: Conditions that should block trades from specific algos

Return a JSON object with a "patterns" array. Each pattern has: pattern_text, category, \
affected_algos, affected_envs, actionable_implication, confidence, evidence. \
Identify between 0 and 5 patterns. Prefer fewer well-supported patterns over many weak \
ones. An empty patterns array is acceptable if the data does not support clear patterns."""

_PHASE_B_SYSTEM_PROMPT = """You are analyzing TRAINING DYNAMICS data — per-fold training \
summaries and reward weight adjustment histories — for SwingRL, an RL-based swing trading system.

Your job is to identify statistically meaningful patterns in per-fold training statistics \
across RL trading agents (PPO, A2C, SAC) on equity (8 ETFs, daily) and crypto (BTC/ETH, 4H) \
environments.

DATA FORMAT — Each fold summary contains:
- STATS: Per-metric extended stats including IQM (interquartile mean, robust central \
tendency), 5-number summary (Q1/median/Q3/min/max), trend slope (OLS, positive=improving), \
and skewness (for large-N folds). IQM is MORE reliable than mean — prefer it for comparisons.
- TRAJECTORY: Temporal windows (early/mid/late) showing how metrics evolved during training. \
Use this to detect convergence, divergence, or overfitting (mid better than late).
- OUTLIERS: IQR-based detection (Tukey fences), reported as rate (count/N). Only PPO has \
approx_kl metrics — A2C and SAC do not produce KL divergence.
- REWARD WEIGHTS: Weight changes with pre/post sharpe deltas showing impact of adjustments.
- METADATA: N=sample_count, cadence=epochs_between_snapshots, confidence=low/moderate/high. \
Low-confidence folds (N<=5, typically PPO) have less reliable statistics.

The data shows per-fold TRAINING statistics, not OOS test results. Look for patterns in: \
(1) training convergence via trajectory and trend slopes, (2) reward shaping effectiveness \
via weight change deltas, (3) algo-specific instability via IQR outliers, (4) early vs late \
fold comparison for data window effects.

You MUST only reference metrics, values, and facts that appear explicitly in the input \
data between <training_data> tags. Do NOT reference any ticker symbols, dates, or metric \
values that do not appear in the input data.

Each pattern MUST include at least two specific metric values from the input data that \
support it. A pattern requires consistent evidence across at least 3 folds or 2 algorithms. \
A single anomalous fold is an observation, not a pattern. Weight low-confidence folds less \
heavily when assessing pattern consistency.

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

category MUST be one of these 4 values. Use the closest match. Do NOT create new categories.

- iteration_progression: Mean metric improvement or degradation across training iterations
- overfit_diagnosis: IS vs OOS gap patterns, especially regime-dependent overfitting
- drawdown_recovery: max_dd_duration and avg_drawdown comparison across algos/regimes
- reward_shaping: Reward weight adjustment effectiveness — which adjustments improved or degraded performance

Return a JSON object with a "patterns" array. Each pattern has: pattern_text, category, \
affected_algos, affected_envs, actionable_implication, confidence, evidence. \
Identify between 0 and 5 patterns. Prefer fewer well-supported patterns over many weak \
ones. An empty patterns array is acceptable if the data does not support clear patterns."""

_STAGE_2_SYSTEM_PROMPT = """You are analyzing CONSOLIDATED PATTERNS already extracted from \
equity and crypto training data for SwingRL, an RL-based swing trading system.

Your job is to identify cross-environment patterns by comparing Stage 1 pattern texts with \
their category, confidence, and evidence from both equity (8 ETFs, daily) and crypto \
(BTC/ETH, 4H) environments.

Look for patterns that appear in BOTH environments (convergent evidence). Look for patterns \
that DIFFER between environments (divergent behavior). Cross-env patterns are especially \
valuable for ensemble weight decisions.

You MUST only reference metrics, values, and facts that appear explicitly in the input \
data between <training_data> tags. Do NOT reference any ticker symbols, dates, or metric \
values that do not appear in the input data.

Each pattern MUST cite specific metric values from the Stage 1 patterns that \
support it. A cross-environment pattern requires supporting evidence from BOTH \
environments. Do not create a pattern based on data from only one environment. \
A single-environment observation is not a cross-environment pattern.

For each candidate pattern, ask yourself: "Is this truly a cross-environment insight, \
or just restating a single-environment pattern?" If the latter, do not include it. \
Only include patterns with direct numerical evidence from the input. If in doubt, omit \
the pattern.

Confidence scoring guide:
- 0.9-1.0: Mathematically certain (pattern holds across both environments without exception)
- 0.7-0.89: Strong pattern with minor inconsistencies in one environment
- 0.5-0.69: Moderate pattern, evidence supports it in both envs but with caveats
- 0.3-0.49: Weak pattern, slight cross-env trend with substantial counter-evidence
- 0.0-0.29: Very weak, barely distinguishable from noise

Most cross-environment patterns from noisy financial data should fall in the 0.4-0.7 \
range. A confidence above 0.85 requires overwhelming evidence in both environments. \
Calibrate accordingly.

category MUST be one of these 2 values. Use the closest match. Do NOT create new categories.

- cross_env: Algo preference differences between equity and crypto environments
- cross_env_correlation: Patterns that span both equity and crypto environments

Return a JSON object with a "patterns" array. Each pattern has: pattern_text, category, \
affected_algos, affected_envs, actionable_implication, confidence, evidence. \
Identify between 0 and 3 patterns. Prefer fewer well-supported patterns over many weak \
ones. An empty patterns array is acceptable if the data does not support clear patterns."""

# ---------------------------------------------------------------------------
# Phase-specific few-shot examples
# ---------------------------------------------------------------------------

_PHASE_A_FEW_SHOT_EXAMPLES = json.dumps(
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
                "pattern_text": "PLACEHOLDER: ALGO_A IS/OOS Sharpe gap widens to X.XX in bear folds vs Y.YY in bull folds",
                "category": "overfit_diagnosis",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: increase entropy_coeff for ALGO_A during bear regime training",
                "confidence": 0.48,
                "evidence": "PLACEHOLDER: fold 1 (bear) is_sharpe=X.XX oos_sharpe=Y.YY gap=Z.ZZ; fold 3 (bull) gap=W.WW",
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
        ],
    },
    indent=2,
)

_PHASE_B_FEW_SHOT_EXAMPLES = json.dumps(
    {
        "patterns": [
            {
                "pattern_text": "PLACEHOLDER: ALGO_A reward IQM improves from X.XX (early folds) to Y.YY (late folds) with positive trend slopes across N of M folds, showing data window benefits",
                "category": "iteration_progression",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: ALGO_A benefits from larger training windows, maintain expanding window strategy",
                "confidence": 0.44,
                "evidence": "PLACEHOLDER: fold 0 reward iqm=X.XX trend=+A.AA; fold 22 reward iqm=Y.YY trend=+B.BB; trajectory early->late improving in N/M folds",
            },
            {
                "pattern_text": "PLACEHOLDER: ALGO_A trajectory shows sharpe improving early->mid (X.XX->Y.YY) then declining mid->late (Y.YY->Z.ZZ) in N of M folds, suggesting overfitting",
                "category": "overfit_diagnosis",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: implement early stopping for ALGO_A or increase entropy regularization",
                "confidence": 0.55,
                "evidence": "PLACEHOLDER: fold N trajectory sharpe early=X.XX mid=Y.YY late=Z.ZZ (declining); pattern in M of P folds",
            },
            {
                "pattern_text": "PLACEHOLDER: fold N MDD IQR outlier (mdd=X.XX, below lower fence Y.YY) at epoch Z recovers after reward weight change increased drawdown penalty — sharpe delta +W.WW",
                "category": "drawdown_recovery",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: pre-set higher drawdown penalty for ALGO_A to avoid MDD breach and recovery cycle",
                "confidence": 0.62,
                "evidence": "PLACEHOLDER: fold N epoch Z mdd=X.XX (IQR outlier, fence=Y.YY); change@epoch=W sharpe delta=+V.VV; drawdown_weight 0.15->0.35",
            },
            {
                "pattern_text": "PLACEHOLDER: reward weight adjustments increasing drawdown penalty improved sharpe in NN% of changes (avg delta +Z.ZZ) for ALGO_A across N folds",
                "category": "reward_shaping",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: start ALGO_A training with higher drawdown penalty to skip adjustment cycle",
                "confidence": 0.48,
                "evidence": "PLACEHOLDER: N adjustments total, M positive (avg_sharpe_delta=+Z.ZZ), P negative (avg_sharpe_delta=-W.WW)",
            },
            {
                "pattern_text": "PLACEHOLDER: PPO KL IQR outliers (N/M rate) concentrated in bear-regime folds X-Y, while A2C/SAC (no KL metric) show elevated MDD outliers in same folds",
                "category": "iteration_progression",
                "affected_algos": ["PLACEHOLDER_ALGO_A", "PLACEHOLDER_ALGO_B"],
                "affected_envs": ["PLACEHOLDER_ENV"],
                "actionable_implication": "PLACEHOLDER: reduce PPO learning rate or clip range during bear regime folds to prevent KL instability",
                "confidence": 0.51,
                "evidence": "PLACEHOLDER: PPO fold X kl outlier=V.VV (fence=W.WW); fold Y kl outlier=U.UU; A2C/SAC fold X mdd outlier=S.SS (N=17, high confidence)",
            },
        ],
    },
    indent=2,
)

_STAGE_2_FEW_SHOT_EXAMPLES = json.dumps(
    {
        "patterns": [
            {
                "pattern_text": "PLACEHOLDER: ALGO_A ranks first in ENV_1 (sharpe=X.XX) but last in ENV_2 (sharpe=Y.YY), suggesting environment-specific ensemble weights needed",
                "category": "cross_env",
                "affected_algos": ["PLACEHOLDER_ALGO_A"],
                "affected_envs": ["PLACEHOLDER_ENV_1", "PLACEHOLDER_ENV_2"],
                "actionable_implication": "PLACEHOLDER: use different ensemble weights per environment, favor ALGO_A in ENV_1 and ALGO_B in ENV_2",
                "confidence": 0.38,
                "evidence": "PLACEHOLDER: ENV_1 ALGO_A sharpe=X.XX weight=Y.YY; ENV_2 ALGO_A sharpe=Z.ZZ weight=W.WW",
            },
            {
                "pattern_text": "PLACEHOLDER: ENV_1 bear onset (p_bear crossing X.XX) precedes ENV_2 drawdown by N bars, enabling cross-env early warning",
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
            {
                "pattern_text": "PLACEHOLDER: both equity and crypto show reward_shaping effectiveness — increasing drawdown penalty improved rolling_MDD in N% of adjustments across both environments",
                "category": "cross_env_correlation",
                "affected_algos": [
                    "PLACEHOLDER_ALGO_A",
                    "PLACEHOLDER_ALGO_B",
                    "PLACEHOLDER_ALGO_C",
                ],
                "affected_envs": ["PLACEHOLDER_ENV_1", "PLACEHOLDER_ENV_2"],
                "actionable_implication": "PLACEHOLDER: apply drawdown penalty increase as universal reward shaping strategy across both envs",
                "confidence": 0.52,
                "evidence": "PLACEHOLDER: equity N1 of M1 adjustments positive (avg_mdd_delta=X.XX); crypto N2 of M2 adjustments positive (avg_mdd_delta=Y.YY)",
            },
        ],
    },
    indent=2,
)


# ---------------------------------------------------------------------------
# Epoch memory aggregation (Tier 1 — local stats, no LLM)
# ---------------------------------------------------------------------------


def _quartiles(vals: list[float]) -> tuple[float, float, float]:
    """Compute Q1, median, Q3 using linear interpolation.

    Args:
        vals: Sorted list of floats (must have len >= 1).

    Returns:
        (Q1, median, Q3) tuple.
    """
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (vals[0], vals[0], vals[0])

    def _interp(sorted_vals: list[float], p: float) -> float:
        """Interpolated percentile (matches numpy default 'linear' method)."""
        k = (len(sorted_vals) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        d = k - f
        return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

    return (_interp(vals, 0.25), _interp(vals, 0.50), _interp(vals, 0.75))


def _iqr_outliers(
    vals: list[float],
    mild_k: float | None = None,
    extreme_k: float | None = None,
) -> dict[str, Any]:
    """Detect outliers using Tukey's IQR fences.

    Args:
        vals: Unsorted list of floats.
        mild_k: Multiplier for mild fences (default from config).
        extreme_k: Multiplier for extreme fences (default from config).

    Returns:
        Dict with q1, median, q3, iqr, lower/upper fences, and outlier lists.
    """
    if mild_k is None:
        mild_k = _AGGREGATION_CONFIG["outlier_iqr_mild"]
    if extreme_k is None:
        extreme_k = _AGGREGATION_CONFIG["outlier_iqr_extreme"]

    sorted_vals = sorted(vals)
    q1, median, q3 = _quartiles(sorted_vals)
    iqr = q3 - q1

    lower_fence = q1 - mild_k * iqr
    upper_fence = q3 + mild_k * iqr
    lower_extreme = q1 - extreme_k * iqr
    upper_extreme = q3 + extreme_k * iqr

    mild = [v for v in sorted_vals if v < lower_fence or v > upper_fence]
    extreme = [v for v in sorted_vals if v < lower_extreme or v > upper_extreme]

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "mild_outliers": mild,
        "extreme_outliers": extreme,
    }


def _iqm(vals: list[float]) -> float:
    """Interquartile mean — average of values between Q1 and Q3.

    Recommended by Google Research "Deep RL at Statistical Precipice"
    (NeurIPS 2021) for robust RL evaluation. Discards the most extreme
    25% on each tail, reducing sensitivity to outliers.

    Falls back to simple mean when N < 4 (insufficient for quartile exclusion).
    """
    n = len(vals)
    if n < 4:
        return sum(vals) / n if n else 0.0
    sorted_vals = sorted(vals)
    q1_idx = n // 4
    q3_idx = n - q1_idx
    middle = sorted_vals[q1_idx:q3_idx]
    return sum(middle) / len(middle) if middle else 0.0


def _trend_slope(vals: list[float]) -> float:
    """OLS slope of values vs index. Positive = metric increasing over time.

    Returns 0.0 for N < 3 (insufficient for meaningful trend).
    """
    n = len(vals)
    if n < 3:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(vals) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0


def _skewness(vals: list[float]) -> float | None:
    """Sample skewness (Fisher's definition). Returns None if N < min threshold.

    Uses skewness_min_n from config (default 8) as minimum sample size.
    """
    min_n = _AGGREGATION_CONFIG["skewness_min_n"]
    n = len(vals)
    if n < min_n:
        return None
    mean = sum(vals) / n
    m2 = sum((v - mean) ** 2 for v in vals) / n
    m3 = sum((v - mean) ** 3 for v in vals) / n
    if m2 == 0:
        return 0.0
    return m3 / (m2**1.5)


def _skewness_label(skew: float) -> str:
    """Classify skewness magnitude."""
    if skew < -0.5:
        return "left-skewed"
    if skew > 0.5:
        return "right-skewed"
    return "symmetric"


def _temporal_windows(
    epochs: list[dict[str, float]],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    """Split fold snapshots into temporal windows and compute per-window means.

    Window strategy:
    - N >= 6: 3 windows (early/mid/late)
    - 4 <= N < 6: 2 windows (early/late)
    - N < 4: empty list (too few for windowing)

    Args:
        epochs: List of epoch metric dicts (assumed chronologically ordered).
        metric_keys: Which metrics to compute window means for.

    Returns:
        List of window dicts with 'label' and per-metric mean values.
    """
    n = len(epochs)
    if n < 4:
        return []

    if n >= 6:
        third = n // 3
        splits = [
            ("early", epochs[:third]),
            ("mid", epochs[third : 2 * third]),
            ("late", epochs[2 * third :]),
        ]
    else:
        half = n // 2
        splits = [
            ("early", epochs[:half]),
            ("late", epochs[half:]),
        ]

    windows: list[dict[str, Any]] = []
    for label, window_epochs in splits:
        window: dict[str, Any] = {"label": label}
        for key in metric_keys:
            vals = [e[key] for e in window_epochs if key in e]
            window[key] = sum(vals) / len(vals) if vals else 0.0
        windows.append(window)
    return windows


def _trend_direction(
    slope: float,
    key: str,
) -> str:
    """Label trend direction. MDD is inverted (positive slope = recovering)."""
    if abs(slope) < 1e-6:
        return "flat"
    # For MDD, positive slope means getting less negative = recovering
    if key in ("rolling_mdd_500",):
        return "recovering" if slope > 0 else "worsening"
    return "improving" if slope > 0 else "declining"


def _format_outlier_events(
    epochs: list[dict[str, float]],
    key: str,
    outlier_vals: list[float],
    max_events: int,
    worst: bool = True,
) -> str:
    """Format up to max_events outlier epochs for a metric.

    Args:
        epochs: All fold epochs.
        key: Metric key to match outliers on.
        outlier_vals: List of values identified as outliers by IQR.
        max_events: Maximum events to report (from config).
        worst: If True, sort ascending (worst MDD first); if False, descending.

    Returns:
        Formatted string of outlier events, or "none (within IQR fences)".
    """
    if not outlier_vals:
        return "none (within IQR fences)"

    outlier_set = set(outlier_vals)
    candidates = [(e.get(key, 0.0), e) for e in epochs if e.get(key) in outlier_set]
    candidates.sort(key=lambda x: x[0], reverse=not worst)
    selected = candidates[:max_events]

    parts = []
    for _, event in selected:
        parts.append(
            f"epoch={event.get('epoch', '?'):.0f} "
            f"{key}={event.get(key, 0):.4f} "
            f"reward={event.get('mean_reward', 0):.4f} "
            f"sharpe={event.get('rolling_sharpe_500', 0):.4f}"
        )
    result = "; ".join(parts)
    if len(candidates) > max_events:
        result += f" (+{len(candidates) - max_events} more)"
    return result


def _aggregate_epoch_summaries(memories: list[dict[str, Any]]) -> list[str]:
    """Aggregate raw epoch memories into per-fold statistical summaries.

    Groups memories by run_id (algo_fold), computes robust statistics for each
    fold (5-number summary, IQM, trend slopes, temporal windows, IQR-based
    outlier detection, skewness), and returns human-readable summary strings
    suitable for a single LLM consolidation call.

    Each summary includes:
    - Fold metadata with N, cadence, and confidence label
    - Extended stats (mean/IQM/Q1/median/Q3/min/max/last + trend) per metric
    - Temporal trajectory (early/mid/late window means)
    - IQR-based outlier detection with up to N extreme events
    - Reward weight trajectory with pre/post sharpe deltas

    Args:
        memories: List of memory dicts with 'text' field containing epoch data.

    Returns:
        List of summary strings, one per fold (algo_fold combination).
    """
    import re
    from collections import defaultdict

    agg_cfg = _AGGREGATION_CONFIG
    max_events = agg_cfg["max_outlier_events"]

    # Parse metrics from memory text into per-fold groups
    folds: dict[str, list[dict[str, float]]] = defaultdict(list)
    fold_meta: dict[str, dict[str, str]] = {}
    # Track (epoch_number, weight_json) pairs for pre/post delta analysis
    fold_weights: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for mem in memories:
        text = mem.get("text", "")

        run_id_m = re.search(r"run_id=(\S+)", text)
        if not run_id_m:
            continue
        run_id = run_id_m.group(1)

        metrics: dict[str, float] = {}
        for key in (
            "epoch",
            "mean_reward",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "approx_kl",
            "rolling_sharpe_500",
            "rolling_mdd_500",
            "rolling_win_rate_500",
        ):
            m = re.search(rf"{key}=([0-9.e+-]+)", text)
            if m:
                try:
                    metrics[key] = float(m.group(1))
                except ValueError:
                    pass

        if metrics:
            folds[run_id].append(metrics)
        else:
            log.debug("epoch_memory_no_metrics_parsed", run_id=run_id)

        # Capture metadata once per fold
        if run_id not in fold_meta:
            algo_m = re.search(r"algo=(\S+)", text)
            env_m = re.search(r"env=(\S+)", text)
            fold_meta[run_id] = {
                "algo": algo_m.group(1) if algo_m else "unknown",
                "env": env_m.group(1) if env_m else "unknown",
            }

        # Track reward weight changes with epoch association
        weights_m = re.search(r"reward_weights=(\{[^}]+\})", text)
        if weights_m:
            epoch_num = metrics.get("epoch", 0.0)
            fold_weights[run_id].append((epoch_num, weights_m.group(1)))

    # Build summaries per fold
    summaries: list[str] = []
    for run_id in sorted(folds.keys()):
        epochs = folds[run_id]
        meta = fold_meta.get(run_id, {})
        n = len(epochs)
        if n == 0:
            continue

        algo = meta.get("algo", "unknown").lower()
        has_kl = algo in _KL_CAPABLE_ALGOS

        # Derive cadence from config
        cadence_key = f"epoch_cadence_{algo}"
        cadence = agg_cfg.get(cadence_key, agg_cfg["epoch_cadence_default"])

        # Confidence label based on sample size
        n_low = agg_cfg["confidence_n_low"]
        n_high = agg_cfg["confidence_n_high"]
        if n <= n_low:
            confidence = "low"
        elif n >= n_high:
            confidence = "high"
        else:
            confidence = "moderate"

        # Epoch range
        epoch_nums = [e.get("epoch", 0) for e in epochs if "epoch" in e]
        first_epoch = int(min(epoch_nums)) if epoch_nums else 0
        last_epoch = int(max(epoch_nums)) if epoch_nums else 0

        # --- STATS section: 5-number summary + IQM + trend ---
        # Metrics to report with extended stats
        core_metrics = [
            ("reward", "mean_reward"),
            ("rolling_sharpe", "rolling_sharpe_500"),
            ("rolling_mdd", "rolling_mdd_500"),
            ("policy_loss", "policy_loss"),
            ("value_loss", "value_loss"),
            ("win_rate", "rolling_win_rate_500"),
        ]
        if has_kl:
            core_metrics.append(("approx_kl", "approx_kl"))

        stat_lines: list[str] = []
        for display_name, key in core_metrics:
            vals = [e[key] for e in epochs if key in e]
            if not vals:
                continue
            sorted_vals = sorted(vals)
            mean = sum(vals) / len(vals)
            q1, median, q3 = _quartiles(sorted_vals)
            iqm_val = _iqm(vals)
            slope = _trend_slope(vals)
            direction = _trend_direction(slope, key)
            last = vals[-1]

            line = (
                f"    {display_name}: mean={mean:.4f} iqm={iqm_val:.4f} "
                f"Q1={q1:.4f} med={median:.4f} Q3={q3:.4f} "
                f"min={sorted_vals[0]:.4f} max={sorted_vals[-1]:.4f} "
                f"last={last:.4f} trend={slope:+.4f}({direction})"
            )

            # Append skewness for large-N folds
            skew = _skewness(vals)
            if skew is not None:
                line += f" skew={skew:.2f}({_skewness_label(skew)})"

            stat_lines.append(line)

        # --- TRAJECTORY section: temporal windows ---
        window_metric_keys = [
            "mean_reward",
            "rolling_sharpe_500",
            "rolling_mdd_500",
            "rolling_win_rate_500",
        ]
        windows = _temporal_windows(epochs, window_metric_keys)
        trajectory_lines: list[str] = []
        if windows:
            labels = [w["label"] for w in windows]
            arrow = "->".join(labels)
            trajectory_lines.append(f"  TRAJECTORY ({arrow}):")
            for display_name, key in [
                ("reward", "mean_reward"),
                ("sharpe", "rolling_sharpe_500"),
                ("mdd", "rolling_mdd_500"),
            ]:
                window_vals = [w.get(key, 0.0) for w in windows]
                vals_str = " -> ".join(f"{v:.4f}" for v in window_vals)
                # Determine direction from first to last window
                if len(window_vals) >= 2:
                    delta = window_vals[-1] - window_vals[0]
                    direction = _trend_direction(delta, key)
                else:
                    direction = "flat"
                trajectory_lines.append(f"    {display_name}: {vals_str} ({direction})")

        # --- OUTLIERS section: IQR-based detection ---
        mdd_vals = [e.get("rolling_mdd_500", 0) for e in epochs if "rolling_mdd_500" in e]
        mdd_iqr = _iqr_outliers(mdd_vals)
        mdd_mild_count = len(mdd_iqr["mild_outliers"])

        outlier_lines: list[str] = []
        outlier_lines.append(f"  OUTLIERS (IQR fences, N={n}):")

        # MDD outliers — worst events (most negative)
        mdd_worst_events = _format_outlier_events(
            epochs,
            "rolling_mdd_500",
            [v for v in mdd_iqr["mild_outliers"] if v < mdd_iqr["lower_fence"]],
            max_events,
            worst=True,
        )
        mdd_best_events = _format_outlier_events(
            epochs,
            "rolling_mdd_500",
            [v for v in mdd_iqr["mild_outliers"] if v > mdd_iqr["upper_fence"]],
            max_events,
            worst=False,
        )
        rate_str = f"{mdd_mild_count}/{n}" if n > 0 else "0/0"
        rate_pct = f"({mdd_mild_count / n * 100:.0f}%)" if n > 0 else "(0%)"
        outlier_lines.append(
            f"    mdd: {rate_str} {rate_pct} mild outliers "
            f"(fences=[{mdd_iqr['lower_fence']:.4f}, {mdd_iqr['upper_fence']:.4f}])"
        )
        outlier_lines.append(f"      worst: {mdd_worst_events}")
        outlier_lines.append(f"      best: {mdd_best_events}")

        # KL outliers (PPO only)
        if has_kl:
            kl_vals = [e.get("approx_kl", 0) for e in epochs if "approx_kl" in e]
            kl_iqr = _iqr_outliers(kl_vals)
            kl_mild_count = len(kl_iqr["mild_outliers"])
            kl_high_events = _format_outlier_events(
                epochs,
                "approx_kl",
                [v for v in kl_iqr["mild_outliers"] if v > kl_iqr["upper_fence"]],
                max_events,
                worst=False,
            )
            kl_rate = f"{kl_mild_count}/{n}" if n > 0 else "0/0"
            kl_pct = f"({kl_mild_count / n * 100:.0f}%)" if n > 0 else "(0%)"
            outlier_lines.append(
                f"    kl: {kl_rate} {kl_pct} mild outliers "
                f"(fences=[{kl_iqr['lower_fence']:.6f}, {kl_iqr['upper_fence']:.6f}])"
            )
            outlier_lines.append(f"      high: {kl_high_events}")

        # --- REWARD WEIGHTS section: trajectory with pre/post deltas ---
        weight_pairs = fold_weights.get(run_id, [])
        # Sort by epoch
        weight_pairs.sort(key=lambda x: x[0])
        weight_lines: list[str] = []
        if weight_pairs:
            first_w = weight_pairs[0][1]
            last_w = weight_pairs[-1][1]
            # Detect changes
            changes: list[tuple[float, str, str]] = []  # (epoch, old, new)
            for i in range(1, len(weight_pairs)):
                if weight_pairs[i][1] != weight_pairs[i - 1][1]:
                    changes.append((weight_pairs[i][0], weight_pairs[i - 1][1], weight_pairs[i][1]))

            weight_lines.append("  REWARD WEIGHTS:")
            weight_lines.append(f"    initial={first_w}")
            for change_epoch, old_w, new_w in changes:
                # Find nearest sharpe before and after the change
                pre_sharpe = None
                post_sharpe = None
                for e in epochs:
                    ep = e.get("epoch", -1)
                    if ep <= change_epoch and "rolling_sharpe_500" in e:
                        pre_sharpe = e["rolling_sharpe_500"]
                    if ep >= change_epoch and "rolling_sharpe_500" in e:
                        post_sharpe = e["rolling_sharpe_500"]
                        break
                delta_str = ""
                if pre_sharpe is not None and post_sharpe is not None:
                    delta = post_sharpe - pre_sharpe
                    delta_str = f" sharpe {pre_sharpe:.4f}->{post_sharpe:.4f} ({delta:+.4f})"
                weight_lines.append(
                    f"    change@epoch={change_epoch:.0f}: {old_w}->{new_w}{delta_str}"
                )
            weight_lines.append(f"    final={last_w} total_changes={len(changes)}")
        else:
            weight_lines.append("  REWARD WEIGHTS: none tracked")

        # --- Assemble summary ---
        header = (
            f"FOLD SUMMARY: {run_id} algo={meta.get('algo', '?')} "
            f"env={meta.get('env', '?')} "
            f"epochs={first_epoch}-{last_epoch} N={n} cadence={cadence} "
            f"confidence={confidence}"
        )
        parts = [header, "  STATS (all epochs):"]
        parts.extend(stat_lines)
        if trajectory_lines:
            parts.extend(trajectory_lines)
        parts.extend(outlier_lines)
        parts.extend(weight_lines)

        summaries.append("\n".join(parts))

    log.info(
        "epoch_summaries_aggregated",
        fold_count=len(summaries),
        total_memories=len(memories),
    )
    return summaries


def _summarize_reward_adjustments(memories: list[dict[str, Any]], env_name: str) -> list[str]:
    """Summarize reward adjustment memories per algo for consolidation.

    Groups reward adjustments by algo, computes trends and effectiveness,
    and returns per-algo summary strings.

    Args:
        memories: List of reward adjustment memory dicts.
        env_name: Environment name to filter by.

    Returns:
        List of per-algo summary strings.
    """
    import re
    from collections import defaultdict

    algo_adjustments: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for mem in memories:
        text = mem.get("text", "")
        if f"env={env_name}" not in text:
            continue

        algo_m = re.search(r"algo=(\S+)", text)
        algo = algo_m.group(1) if algo_m else "unknown"

        adj: dict[str, Any] = {}
        for key in ("epoch_triggered", "trigger_value"):
            m = re.search(rf"{key}=([0-9.e+-]+)", text)
            if m:
                try:
                    adj[key] = float(m.group(1))
                except ValueError:
                    pass

        trigger_m = re.search(r"trigger_metric=(\S+)", text)
        adj["trigger_metric"] = trigger_m.group(1) if trigger_m else "unknown"

        # Extract before/after weights
        before_m = re.search(r"weights_before=(\{[^}]+\})", text)
        after_m = re.search(r"weights_after=(\{[^}]+\})", text)
        adj["weights_before"] = before_m.group(1) if before_m else ""
        adj["weights_after"] = after_m.group(1) if after_m else ""

        # Check effectiveness (from outcome memories)
        effective_m = re.search(r"adjustment_effective=(True|False)", text)
        if effective_m:
            adj["effective"] = effective_m.group(1) == "True"

        sharpe_delta_m = re.search(r"post_adjustment_sharpe_delta=([0-9.e+-]+)", text)
        if sharpe_delta_m:
            try:
                adj["sharpe_delta"] = float(sharpe_delta_m.group(1))
            except ValueError:
                pass

        algo_adjustments[algo].append(adj)

    summaries: list[str] = []
    for algo in sorted(algo_adjustments.keys()):
        adjs = algo_adjustments[algo]
        n = len(adjs)
        if n == 0:
            continue

        # Trigger breakdown
        triggers: dict[str, int] = defaultdict(int)
        for a in adjs:
            triggers[a.get("trigger_metric", "unknown")] += 1
        trigger_str = ", ".join(f"{k}={v}" for k, v in sorted(triggers.items()))

        # Effectiveness
        effective_count = sum(1 for a in adjs if a.get("effective", False))
        deltas = [a["sharpe_delta"] for a in adjs if "sharpe_delta" in a]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

        # Weight trend (first and last)
        first_after = adjs[0].get("weights_after", "unknown")
        last_after = adjs[-1].get("weights_after", "unknown")

        summary = (
            f"REWARD ADJUSTMENTS: {algo} env={env_name} count={n}\n"
            f"  triggers: {trigger_str}\n"
            f"  effectiveness: {effective_count}/{n} positive "
            f"({effective_count / n * 100:.0f}%), avg_sharpe_delta={avg_delta:.4f}\n"
            f"  weight_trend: first_adjustment={first_after} "
            f"last_adjustment={last_after}"
        )
        summaries.append(summary)

    log.info(
        "reward_adjustments_summarized",
        env_name=env_name,
        algo_count=len(summaries),
        total_adjustments=sum(len(v) for v in algo_adjustments.values()),
    )
    return summaries


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ConsolidateAgent:
    """Synthesizes raw memories into consolidated patterns via LLM.

    Two-stage pipeline:
    - Stage 1: Per-environment consolidation (3 algo + 1 ensemble memories → patterns)
    - Stage 2: Cross-environment consolidation (equity + crypto Stage 1 → cross-env patterns)
    """

    _run_lock = asyncio.Lock()

    async def run(self, env_name: str | None = None) -> int:
        """Orchestrate consolidation: Stage 1 per env, then Stage 2 if both available.

        Serialized via _run_lock to prevent concurrent consolidation runs
        from duplicating patterns.

        Args:
            env_name: If specified, only consolidate this environment.
                      If None, consolidate both envs and run Stage 2.

        Returns:
            Total number of new consolidation patterns created.
        """
        if self._run_lock.locked():
            log.warning("consolidation_already_running_skipped")
            return 0

        async with self._run_lock:
            return await self._run_impl(env_name)

    async def _run_impl(self, env_name: str | None = None) -> int:
        """Internal consolidation implementation (runs under _run_lock)."""
        total = 0
        inter_phase_delay = _PROVIDER_CONFIG.get("inter_phase_delay_sec", 60)

        if env_name:
            total += await self.run_stage1(env_name)
        else:
            # Run Stage 1 for both environments with delays between phases
            equity_count = await self.run_stage1("equity")
            total += equity_count

            log.info("consolidation_inter_env_delay", delay_s=inter_phase_delay)
            await asyncio.sleep(inter_phase_delay)

            crypto_count = await self.run_stage1("crypto")
            total += crypto_count

            # Stage 2: cross-env (only if both envs have Stage 1 patterns)
            equity_patterns = await get_active_consolidations_async(env_name="equity", stage=1)
            crypto_patterns = await get_active_consolidations_async(env_name="crypto", stage=1)
            if equity_patterns and crypto_patterns:
                log.info("consolidation_pre_stage2_delay", delay_s=inter_phase_delay)
                await asyncio.sleep(inter_phase_delay)
                total += await self.run_stage2(equity_patterns, crypto_patterns)

        return total

    async def run_stage1(self, env_name: str) -> int:
        """Stage 1: Per-environment hybrid consolidation.

        Two phases per environment:
        - Phase A: WF + trading pattern memories (cross-algo fold analysis)
        - Phase B: Epoch + reward adjustment + run memories (training dynamics)

        Each phase gets its own LLM call so the model can focus on the
        signal type. Archives each phase's memories after its call.

        Args:
            env_name: Environment name ('equity' or 'crypto').

        Returns:
            Number of new patterns created across both phases.
        """
        total_created = 0

        # --- Phase A: WF + trading patterns (cross-algo analysis) ---
        wf_memories = await get_memories_by_source_prefix_async(
            prefix=f"walk_forward:{env_name}",
            limit=_MEMORY_BATCH_SIZE,
            archived=False,
        )
        trading_memories = await get_memories_by_source_prefix_async(
            prefix=f"trading_pattern:{env_name}",
            limit=_MEMORY_BATCH_SIZE,
            archived=False,
        )
        phase_a: list[dict[str, Any]] = (wf_memories or []) + (trading_memories or [])

        if not phase_a:
            # Fallback: old-style tags for backward compat
            phase_a = (
                await get_memories_by_source_prefix_async(
                    prefix=env_name,
                    limit=_MEMORY_BATCH_SIZE,
                    archived=False,
                )
                or []
            )

        if phase_a:
            log.info(
                "consolidation_stage1_wf_starting",
                env_name=env_name,
                memory_count=len(phase_a),
            )
            texts_a = "\n\n".join(f"- {m['text']}" for m in phase_a)
            ids_a = [m["id"] for m in phase_a]

            result = await self._call_llm_with_retry(
                texts_a,
                system_prompt=_PHASE_A_SYSTEM_PROMPT,
                few_shot_examples=_PHASE_A_FEW_SHOT_EXAMPLES,
            )
            if result is not None:
                for pattern in result.get("patterns", []):
                    row_id = await self._dedup_and_insert(
                        pattern=pattern,
                        source_count=len(phase_a),
                        stage=1,
                        env_name=env_name,
                        memory_ids=ids_a,
                    )
                    if row_id is not None:
                        total_created += 1
                await archive_memories_async(ids_a)
            else:
                log.warning(
                    "consolidation_memories_preserved",
                    env_name=env_name,
                    phase="A",
                    memory_count=len(ids_a),
                )

        # --- Phase B: Epoch + reward + run (training dynamics) ---
        # Instead of batching 200 memories per LLM call (which can't scale to
        # millions of epoch memories), we aggregate locally by fold/algo then
        # send a single summary to the LLM. This gives the LLM a holistic
        # cross-fold view in one call instead of fragmented per-batch views.

        # Delay between Phase A and Phase B to respect provider rate limits
        if phase_a:
            inter_phase_delay = _PROVIDER_CONFIG.get("inter_phase_delay_sec", 60)
            log.info(
                "consolidation_phase_a_to_b_delay",
                env_name=env_name,
                delay_s=inter_phase_delay,
            )
            await asyncio.sleep(inter_phase_delay)

        phase_b_total_created = 0
        phase_b_any = False

        # Collect ALL epoch/reward/run memory IDs and texts for this env
        # Uses OFFSET-based pagination — no archiving during fetch so memories
        # are preserved if the LLM call fails.
        all_phase_b: list[dict[str, Any]] = []
        for prefix in ("training_epoch", "reward_adjustment"):
            offset = 0
            _FETCH_CHUNK = 10_000
            while True:
                candidates = await get_memories_by_source_prefix_async(
                    prefix=prefix,
                    limit=_FETCH_CHUNK,
                    archived=False,
                    offset=offset,
                )
                if not candidates:
                    break
                filtered = [m for m in candidates if f"env={env_name}" in m.get("text", "")]
                all_phase_b.extend(filtered)
                if len(candidates) < _FETCH_CHUNK:
                    break  # No more pages
                offset += _FETCH_CHUNK

        if not all_phase_b:
            total_created += phase_b_total_created
            if not phase_a and not all_phase_b:
                log.info("consolidation_skipped_no_memories", env_name=env_name)
                return 0

            log.info(
                "consolidation_stage1_complete",
                env_name=env_name,
                patterns_created=total_created,
            )
            return total_created

        phase_b_any = True
        all_ids_b = [m["id"] for m in all_phase_b]

        log.info(
            "consolidation_stage1_epoch_aggregating",
            env_name=env_name,
            total_memories=len(all_phase_b),
        )

        # Tier 1: Local statistical aggregation (no LLM)
        # Split epoch memories from reward adjustment memories by source tag
        epoch_memories = [
            m for m in all_phase_b if m.get("source", "").startswith("training_epoch")
        ]
        reward_memories = [
            m for m in all_phase_b if m.get("source", "").startswith("reward_adjustment")
        ]
        # If source tags not available, fall back to text content matching
        if not epoch_memories and not reward_memories:
            epoch_memories = [m for m in all_phase_b if "EPOCH" in m.get("text", "").upper()]
            reward_memories = [
                m for m in all_phase_b if "REWARD_ADJUSTMENT" in m.get("text", "").upper()
            ]

        fold_summaries = _aggregate_epoch_summaries(epoch_memories)
        reward_summaries = _summarize_reward_adjustments(reward_memories, env_name)

        # Combine into single prompt with global preamble
        parts = []
        if fold_summaries:
            preamble = (
                f"AGGREGATION NOTE: {len(epoch_memories)} raw epoch memories reduced "
                f"to {len(fold_summaries)} fold summaries.\n"
                f"Small-N folds (PPO, ~4 snapshots) have lower statistical confidence "
                f"than large-N folds (A2C/SAC, ~17 snapshots).\n"
                f"Stats use IQM (interquartile mean) as primary central tendency — "
                f"more robust than mean for RL metrics.\n"
                f"Outliers detected via IQR fences (Tukey method), not percentiles."
            )
            parts.append(
                preamble + "\n\n=== FOLD TRAINING SUMMARIES ===\n" + "\n\n".join(fold_summaries)
            )
        if reward_summaries:
            parts.append("=== REWARD WEIGHT ADJUSTMENTS ===\n" + "\n\n".join(reward_summaries))
        summary_text = "\n\n".join(parts)

        log.info(
            "consolidation_stage1_epoch_aggregated",
            env_name=env_name,
            fold_summaries=len(fold_summaries),
            reward_summaries=len(reward_summaries),
            summary_tokens=len(summary_text) // 4,
        )

        # Tier 2: Single LLM call with aggregated summaries + reward adjustments
        result = await self._call_llm_with_retry(
            summary_text,
            system_prompt=_PHASE_B_SYSTEM_PROMPT,
            few_shot_examples=_PHASE_B_FEW_SHOT_EXAMPLES,
        )
        if result is not None:
            for pattern in result.get("patterns", []):
                row_id = await self._dedup_and_insert(
                    pattern=pattern,
                    source_count=len(all_phase_b),
                    stage=1,
                    env_name=env_name,
                    memory_ids=all_ids_b[:1000],  # Link first 1000 as sample
                )
                if row_id is not None:
                    phase_b_total_created += 1
            await archive_memories_async(all_ids_b)
        else:
            log.warning(
                "consolidation_memories_preserved",
                env_name=env_name,
                phase="B",
                memory_count=len(all_ids_b),
            )

        total_created += phase_b_total_created

        if not phase_a and not phase_b_any:
            log.info("consolidation_skipped_no_memories", env_name=env_name)
            return 0

        log.info(
            "consolidation_stage1_complete",
            env_name=env_name,
            patterns_created=total_created,
        )
        return total_created

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
                f"action={p.get('actionable_implication')}, "
                f"evidence={p.get('evidence')})"
            )
        for p in crypto_patterns:
            pattern_texts.append(
                f"[CRYPTO] {p['pattern_text']} "
                f"(category={p.get('category')}, confidence={p.get('confidence')}, "
                f"action={p.get('actionable_implication')}, "
                f"evidence={p.get('evidence')})"
            )

        input_text = "\n\n".join(pattern_texts)

        stage2_user_prompt = (
            f"--- FORMAT TEMPLATE (fictional data, for output structure only) ---\n"
            f"{_STAGE_2_FEW_SHOT_EXAMPLES}\n"
            f"--- END FORMAT TEMPLATE ---\n\n"
            f"<training_data>\n{input_text}\n</training_data>\n\n"
            "Identify patterns that span both equity and crypto environments. "
            "Focus on cross_env and cross_env_correlation categories. Return a JSON "
            "object matching the schema. 0-3 patterns. Empty array is acceptable."
        )

        result = await self._call_llm(
            system_prompt=_STAGE_2_SYSTEM_PROMPT,
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
        category = str(pattern.get("category", ""))
        affected_algos = pattern.get("affected_algos", [])
        affected_envs = pattern.get("affected_envs", [])
        # Coerce LLM outputs to DB-safe types — LLMs sometimes return dicts
        if isinstance(affected_algos, dict):
            affected_algos = list(affected_algos.values()) if affected_algos else []
        if isinstance(affected_envs, dict):
            affected_envs = list(affected_envs.values()) if affected_envs else []

        # Check for existing active patterns with same category + env
        existing = await get_active_consolidations_async(env_name=env_name, stage=stage)
        similar = [
            e
            for e in existing
            if e.get("category") == category
            and set(e.get("affected_algos") or []) == set(affected_algos)
        ]

        if similar:
            # Increment confirmation on the most recent similar pattern
            best = similar[0]
            await increment_confirmation_async(best["id"])
            log.info(
                "consolidation_dedup_confirmed",
                existing_id=best["id"],
                category=category,
            )
            return None

        # Check for conflicts with opposite-sentiment patterns
        conflict_id = await self._detect_conflict_async(
            pattern.get("pattern_text", ""),
            new_category=category,
        )

        # Coerce LLM fields to DB-safe scalar types
        evidence_raw = pattern.get("evidence")
        evidence_str = (
            json.dumps(evidence_raw) if isinstance(evidence_raw, (dict, list)) else evidence_raw
        )
        implication_raw = pattern.get("actionable_implication")
        implication_str = (
            json.dumps(implication_raw)
            if isinstance(implication_raw, (dict, list))
            else implication_raw
        )
        confidence_val = pattern.get("confidence")
        if isinstance(confidence_val, dict):
            confidence_val = confidence_val.get("score", confidence_val.get("value", 0.5))

        row_id = await insert_consolidation_async(
            pattern_text=str(pattern.get("pattern_text", "")),
            source_count=source_count,
            conflicting_with=conflict_id,
            category=category,
            affected_algos=affected_algos,
            affected_envs=affected_envs,
            actionable_implication=implication_str,
            confidence=float(confidence_val) if confidence_val is not None else None,
            evidence=evidence_str,
            stage=stage,
            env_name=env_name,
            status="active",
            conflict_group_id=str(uuid.uuid4()) if conflict_id else None,
        )

        # Link source memories (batch insert in single connection)
        await insert_consolidation_sources_async(row_id, memory_ids)

        if conflict_id is not None:
            # Mark the conflicting pattern with the same group ID
            await update_consolidation_status_async(
                conflict_id,
                "superseded",
                superseded_by=row_id,
            )
            log.warning(
                "consolidation_conflict_detected",
                new_id=row_id,
                conflicts_with=conflict_id,
            )

        return row_id

    async def _call_llm_with_retry(
        self,
        memory_texts: str,
        system_prompt: str,
        few_shot_examples: str,
    ) -> dict[str, Any] | None:
        """Attempt LLM consolidation with one retry on malformed output.

        Args:
            memory_texts: Concatenated memory texts to consolidate.
            system_prompt: Phase-specific system prompt (required).
            few_shot_examples: Phase-specific few-shot examples (required).

        Returns:
            Parsed consolidation dict with 'patterns' array, or None.
        """
        user_prompt = (
            f"--- FORMAT TEMPLATE (fictional data, for output structure only) ---\n"
            f"{few_shot_examples}\n"
            f"--- END FORMAT TEMPLATE ---\n\n"
            f"<training_data>\n{memory_texts}\n</training_data>\n\n"
            "Analyze ONLY the data between the <training_data> tags. Return a JSON object "
            "matching the schema. 0-5 patterns. Empty array is acceptable."
        )

        for attempt in range(1, 3):
            result = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_CONSOLIDATION_SCHEMA,
            )
            if result is not None:
                await log_consolidation_quality_async(attempt_count=attempt, accepted=True)
                return result
            log.warning("consolidation_attempt_failed", attempt=attempt)

        await log_consolidation_quality_async(
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
        """Call primary then backup cloud API (provider from config)."""
        for tier in ("primary", "backup"):
            cfg = _PROVIDER_CONFIG[tier]
            if not cfg["api_key"]:
                log.debug(
                    "consolidation_provider_skipped_no_key",
                    provider=cfg["provider"],
                    tier=tier,
                )
                continue

            if tier == "backup":
                log.info(
                    "consolidation_falling_back_to_backup",
                    backup_provider=cfg["provider"],
                    backup_model=cfg["model"],
                )

            # Total timeout = provider's configured timeout (hard wall-clock ceiling).
            # Prevents infinite hangs from keepalive bytes while respecting each
            # provider's expected response time (e.g., OpenRouter free = 1800s).
            call_total_timeout = cfg["timeout"]

            result = await self._call_cloud_api(
                system_prompt,
                user_prompt,
                schema,
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                timeout=cfg["timeout"],
                provider=cfg["provider"],
                max_tokens=cfg.get("max_tokens", 32768),
                total_timeout=call_total_timeout,
            )
            if result is not None:
                if tier == "backup":
                    log.info(
                        "consolidation_backup_succeeded",
                        provider=cfg["provider"],
                    )
                return result
            log.warning(
                "consolidation_provider_failed",
                provider=cfg["provider"],
                tier=tier,
            )
        return None

    @staticmethod
    def _build_request_body(
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        provider: str,
    ) -> dict[str, Any]:
        """Build provider-specific request body with structured output routing."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "frequency_penalty": 0.0,
        }
        if provider in ("cerebras", "gemini"):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "consolidation",
                    "strict": True,
                    "schema": schema,
                },
            }
        elif provider == "nvidia":
            body["response_format"] = {"type": "json_object"}
            body["guided_json"] = schema
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    async def _call_cloud_api(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        provider: str,
        max_tokens: int = 32768,
        total_timeout: float = 300.0,
    ) -> dict[str, Any] | None:
        """Call OpenAI-compatible cloud API with json_object response format.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            schema: JSON schema for structured output enforcement.
            base_url: API base URL.
            api_key: Bearer token for authorization.
            model: Model identifier string.
            timeout: Read timeout in seconds (per-chunk).
            provider: Provider name for logging.
            max_tokens: Maximum output tokens (per-provider from config).
            total_timeout: Hard ceiling on total wall time for this call.

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        # Retry with exponential backoff on 429 rate-limit errors
        max_retries = _PROVIDER_CONFIG.get("rate_limit_max_retries", 3)
        backoff_base = _PROVIDER_CONFIG.get("rate_limit_backoff_base_sec", 30)

        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self._do_cloud_request(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema=schema,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        timeout=timeout,
                        provider=provider,
                        max_tokens=max_tokens,
                    ),
                    timeout=total_timeout,
                )
                return result
            except TimeoutError:
                log.error(
                    "cloud_api_total_timeout",
                    provider=provider,
                    total_timeout_s=total_timeout,
                    attempt=attempt + 1,
                )
                return None
            except _RateLimitError as exc:
                delay = backoff_base * (2**attempt)
                log.warning(
                    "cloud_api_rate_limited_retrying",
                    provider=provider,
                    status_code=exc.status_code,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    backoff_s=delay,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "cloud_api_rate_limit_exhausted",
                        provider=provider,
                        attempts=max_retries,
                    )
                    return None

        return None

    async def _do_cloud_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        provider: str,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        """Execute a single cloud API request. Raises _RateLimitError on 429."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)
            ) as client:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=self._build_request_body(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema=schema,
                        max_tokens=max_tokens,
                        provider=provider,
                    ),
                )
                resp.raise_for_status()
                body = resp.json()
                raw_content = body["choices"][0]["message"]["content"]
                if raw_content is None:
                    log.error(
                        "cloud_api_empty_content",
                        provider=provider,
                        finish_reason=body["choices"][0].get("finish_reason"),
                    )
                    return None
                parsed = json.loads(raw_content)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise _RateLimitError(
                    status_code=429,
                    provider=provider,
                    body=exc.response.text[:500],
                ) from exc
            log.error(
                "cloud_api_call_failed",
                provider=provider,
                error=str(exc),
                status_code=exc.response.status_code,
                response_body=exc.response.text[:500],
                exc_type=type(exc).__name__,
            )
            return None
        except Exception as exc:
            log.error(
                "cloud_api_call_failed",
                provider=provider,
                error=str(exc) or repr(exc),
                exc_type=type(exc).__name__,
            )
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

    async def _detect_conflict_async(self, new_pattern: str, new_category: str = "") -> int | None:
        """Check whether the new pattern contradicts existing consolidations (async).

        Args:
            new_pattern: The new pattern text to check.
            new_category: Category of the new pattern (for same-category check).

        Returns:
            Row ID of conflicting consolidation, or None if no conflict.
        """
        existing = await get_active_consolidations_async()
        return self._check_conflicts(existing, new_pattern, new_category)

    def _check_conflicts(
        self,
        existing: list[dict[str, Any]],
        new_pattern: str,
        new_category: str = "",
    ) -> int | None:
        """Pure logic for conflict detection against a list of existing patterns.

        Args:
            existing: List of active consolidation dicts.
            new_pattern: The new pattern text to check.
            new_category: Category of the new pattern.

        Returns:
            Row ID of conflicting consolidation, or None if no conflict.
        """
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
