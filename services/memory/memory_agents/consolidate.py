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
    config_path = Path("/app/config/swingrl.yaml")
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

        return {"primary": primary, "backup": backup}

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
# Epoch memory aggregation (Tier 1 — local stats, no LLM)
# ---------------------------------------------------------------------------


def _aggregate_epoch_summaries(memories: list[dict[str, Any]]) -> list[str]:
    """Aggregate raw epoch memories into per-fold statistical summaries.

    Groups memories by run_id (algo_fold), computes stats for each fold,
    identifies per-fold P1/P99 outlier events, and returns human-readable
    summary strings suitable for a single LLM consolidation call.

    Each summary includes:
    - Overall stats (mean/min/max/last) computed from ALL memories in the fold
    - Per-fold P1/P99 outlier details (worst/best MDD events, KL spikes)
    - Reward weight trajectory (first → last → number of changes)

    Args:
        memories: List of memory dicts with 'text' field containing epoch data.

    Returns:
        List of summary strings, one per fold (algo_fold combination).
    """
    import re
    from collections import defaultdict

    # Parse metrics from memory text into per-fold groups
    folds: dict[str, list[dict[str, float]]] = defaultdict(list)
    fold_meta: dict[str, dict[str, str]] = {}
    fold_weights: dict[str, list[str]] = defaultdict(list)

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

        # Track reward weight changes
        weights_m = re.search(r"reward_weights=(\{[^}]+\})", text)
        if weights_m:
            fold_weights[run_id].append(weights_m.group(1))

    def _stat(
        epochs_list: list[dict[str, float]],
        key: str,
    ) -> tuple[float, float, float, float]:
        """Return (mean, min, max, last) for a metric across epochs."""
        vals = [e[key] for e in epochs_list if key in e]
        if not vals:
            return (0.0, 0.0, 0.0, 0.0)
        mean = sum(vals) / len(vals)
        return (mean, min(vals), max(vals), vals[-1])

    def _percentile(vals: list[float], pct: float) -> float:
        """Compute percentile from sorted values."""
        if not vals:
            return 0.0
        idx = int(len(vals) * pct / 100)
        return vals[min(idx, len(vals) - 1)]

    def _find_outlier_event(
        epochs_list: list[dict[str, float]],
        key: str,
        worst: bool = True,
    ) -> str:
        """Find the epoch with the most extreme value for a metric."""
        candidates = [(e.get(key, 0.0), e) for e in epochs_list if key in e]
        if not candidates:
            return "none"
        if worst:
            _, event = min(candidates, key=lambda x: x[0])
        else:
            _, event = max(candidates, key=lambda x: x[0])
        return (
            f"epoch={event.get('epoch', '?'):.0f} "
            f"{key}={event.get(key, 0):.4f} "
            f"reward={event.get('mean_reward', 0):.4f} "
            f"sharpe={event.get('rolling_sharpe_500', 0):.4f}"
        )

    # Build summaries per fold
    summaries: list[str] = []
    for run_id in sorted(folds.keys()):
        epochs = folds[run_id]
        meta = fold_meta.get(run_id, {})
        n = len(epochs)
        if n == 0:
            continue

        # Overall stats from ALL memories
        reward_mean, reward_min, reward_max, reward_last = _stat(epochs, "mean_reward")
        sharpe_mean, sharpe_min, sharpe_max, sharpe_last = _stat(epochs, "rolling_sharpe_500")
        mdd_mean, mdd_min, mdd_max, mdd_last = _stat(epochs, "rolling_mdd_500")
        ploss_mean, _, _, ploss_last = _stat(epochs, "policy_loss")
        vloss_mean, _, _, vloss_last = _stat(epochs, "value_loss")
        kl_mean, _, kl_max, _ = _stat(epochs, "approx_kl")
        winrate_mean, _, _, winrate_last = _stat(epochs, "rolling_win_rate_500")

        # Per-fold P1/P99 thresholds
        mdd_vals = sorted(e.get("rolling_mdd_500", 0) for e in epochs if "rolling_mdd_500" in e)
        kl_vals = sorted(e.get("approx_kl", 0) for e in epochs if "approx_kl" in e)
        mdd_p1 = _percentile(mdd_vals, 1)
        mdd_p99 = _percentile(mdd_vals, 99)
        kl_p99 = _percentile(kl_vals, 99)

        # Count outliers using per-fold thresholds
        mdd_bottom1 = sum(1 for v in mdd_vals if v < mdd_p1) if mdd_vals else 0
        mdd_top1 = sum(1 for v in mdd_vals if v > mdd_p99) if mdd_vals else 0
        kl_top1 = sum(1 for v in kl_vals if v > kl_p99) if kl_vals else 0

        # Outlier event details
        worst_mdd_event = _find_outlier_event(epochs, "rolling_mdd_500", worst=True)
        best_mdd_event = _find_outlier_event(epochs, "rolling_mdd_500", worst=False)
        max_kl_event = _find_outlier_event(epochs, "approx_kl", worst=False)

        # Epoch range
        epoch_nums = [e.get("epoch", 0) for e in epochs if "epoch" in e]
        first_epoch = int(min(epoch_nums)) if epoch_nums else 0
        last_epoch = int(max(epoch_nums)) if epoch_nums else 0

        # Reward weight trajectory
        weights_list = fold_weights.get(run_id, [])
        first_weights = weights_list[0] if weights_list else "unknown"
        last_weights = weights_list[-1] if weights_list else "unknown"
        unique_weights = len(set(weights_list))
        weight_changes = max(0, unique_weights - 1)

        summary = (
            f"FOLD SUMMARY: {run_id} algo={meta.get('algo', '?')} "
            f"env={meta.get('env', '?')} "
            f"epochs={first_epoch}-{last_epoch} snapshots={n}\n"
            f"  STATS (all epochs):\n"
            f"    reward: mean={reward_mean:.4f} min={reward_min:.4f} "
            f"max={reward_max:.4f} last={reward_last:.4f}\n"
            f"    rolling_sharpe: mean={sharpe_mean:.4f} min={sharpe_min:.4f} "
            f"max={sharpe_max:.4f} last={sharpe_last:.4f}\n"
            f"    rolling_mdd: mean={mdd_mean:.4f} worst={mdd_min:.4f} "
            f"best={mdd_max:.4f} last={mdd_last:.4f}\n"
            f"    policy_loss: mean={ploss_mean:.6f} last={ploss_last:.6f}\n"
            f"    value_loss: mean={vloss_mean:.6f} last={vloss_last:.6f}\n"
            f"    approx_kl: mean={kl_mean:.6f} max={kl_max:.6f}\n"
            f"    win_rate: mean={winrate_mean:.4f} last={winrate_last:.4f}\n"
            f"  OUTLIERS (per-fold P1/P99):\n"
            f"    mdd_bottom1%({mdd_bottom1} events, threshold={mdd_p1:.4f}): "
            f"worst={worst_mdd_event}\n"
            f"    mdd_top1%({mdd_top1} events, threshold={mdd_p99:.4f}): "
            f"best={best_mdd_event}\n"
            f"    kl_top1%({kl_top1} events, threshold={kl_p99:.6f}): "
            f"max={max_kl_event}\n"
            f"  REWARD WEIGHTS: start={first_weights} end={last_weights} "
            f"changes={weight_changes}"
        )
        summaries.append(summary)

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
            equity_patterns = await get_active_consolidations_async(env_name="equity", stage=1)
            crypto_patterns = await get_active_consolidations_async(env_name="crypto", stage=1)
            if equity_patterns and crypto_patterns:
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

            result = await self._call_llm_with_retry(texts_a)
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
        phase_b_total_created = 0
        phase_b_any = False

        # Collect ALL epoch/reward/run memory IDs and texts for this env
        all_phase_b: list[dict[str, Any]] = []
        for prefix in ("training_epoch", "reward_adjustment", "training_run"):
            # Fetch in chunks to avoid memory issues, but don't LLM-process each chunk
            offset = 0
            _FETCH_CHUNK = 10_000
            while True:
                candidates = await get_memories_by_source_prefix_async(
                    prefix=prefix,
                    limit=_FETCH_CHUNK,
                    archived=False,
                )
                if not candidates:
                    break
                filtered = [m for m in candidates if f"env={env_name}" in m.get("text", "")]
                all_phase_b.extend(filtered)
                if len(candidates) < _FETCH_CHUNK:
                    break  # No more pages
                # Archive this chunk so next fetch gets new rows
                chunk_ids = [m["id"] for m in candidates]
                await archive_memories_async(chunk_ids)
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
        # Split epoch memories from reward adjustment memories
        epoch_memories = [m for m in all_phase_b if "EPOCH SNAPSHOT" in m.get("text", "")]
        reward_memories = [m for m in all_phase_b if "REWARD_ADJUSTMENT" in m.get("text", "")]

        fold_summaries = _aggregate_epoch_summaries(epoch_memories)
        reward_summaries = _summarize_reward_adjustments(reward_memories, env_name)

        # Combine into single prompt
        parts = []
        if fold_summaries:
            parts.append("=== FOLD TRAINING SUMMARIES ===\n" + "\n\n".join(fold_summaries))
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
        result = await self._call_llm_with_retry(summary_text)
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

        row_id = await insert_consolidation_async(
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
            )
            if result is not None:
                return result
            log.warning(
                "consolidation_provider_failed",
                provider=cfg["provider"],
                tier=tier,
            )
        return None

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
    ) -> dict[str, Any] | None:
        """Call OpenAI-compatible cloud API with json_object response format.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            schema: JSON schema for structured output enforcement.
            base_url: API base URL.
            api_key: Bearer token for authorization.
            model: Model identifier string.
            timeout: Read timeout in seconds.
            provider: Provider name for logging.
            max_tokens: Maximum output tokens (per-provider from config).

        Returns:
            Parsed and validated dict, or None if invalid.
        """
        effective_max_tokens = max_tokens
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
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": effective_max_tokens,
                        "frequency_penalty": 0.0,
                        "response_format": {"type": "json_object"},
                    },
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
