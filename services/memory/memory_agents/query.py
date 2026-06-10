"""QueryAgent: advises RL training configuration using consolidated memory patterns.

Two public methods:
- advise_run_config: Returns hyperparameter suggestions + reward weights for a new run.
- advise_epoch: Returns reward weight adjustments + stop_training signal during a run.

Both methods:
- Pull active consolidations (filtered by confidence + status) and raw memories
- Prefer Stage 2 cross-env patterns over Stage 1 per-env patterns
- Call OpenRouter (nemotron-120b) as primary, NVIDIA NIM (kimi-k2.5) as backup
- Return safe clamped defaults on any failure (cold-start guard)
- XML-wrap all memory text injected into prompts
- Track which patterns were presented via pattern_presentations table

Clamp bounds are defined inline (matching src/swingrl/memory/training/bounds.py constants)
to avoid importing from src/swingrl/ (cross-container boundary, Pitfall 4).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

from db import (
    get_active_consolidations,
    get_active_consolidations_async,
    get_memories,
    get_memories_async,
    insert_audit_log,
    insert_pattern_presentation,
    insert_pattern_presentation_async,
)

log = structlog.get_logger(__name__)

# Config path: env var override for non-Docker environments, default for Docker.
_CONFIG_PATH = Path(os.environ.get("SWINGRL_CONFIG_PATH", "/app/config/swingrl.yaml"))

# ---------------------------------------------------------------------------
# Cloud provider 429 blocking (calendar-day reset)
# ---------------------------------------------------------------------------
# Tracks providers that returned a blocking HTTP code (e.g. 429).
# Key = provider name, value = {"count": int, "blocked_until": datetime}.
# Uses exponential backoff: 5min → 15min → 60min → rest-of-UTC-day.
_CLOUD_BLOCKED: dict[str, dict[str, Any]] = {}
_CLOUD_BLOCKED_LOCK = threading.Lock()

# Serialize Ollama calls to prevent concurrent requests overwhelming the instance
_OLLAMA_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(1)

# ---------------------------------------------------------------------------
# Constants (loaded from mounted config YAML, fallback to hardcoded defaults)
# ---------------------------------------------------------------------------

_FALLBACK_HYPERPARAM_BOUNDS: dict[str, tuple[Any, Any]] = {
    "learning_rate": (1e-5, 1e-3),
    "entropy_coeff": (0.0, 0.05),
    "clip_range": (0.1, 0.4),
    "n_epochs": (3, 20),
    "batch_size": (32, 512),
    "gamma": (0.95, 0.995),
    "target_kl": (0.01, 0.05),
    "gae_lambda": (0.85, 1.0),
    "gradient_steps": (1, 8),
    "target_entropy": (-9.0, -0.5),
}

_ALGO_GAMMA_BOUNDS: dict[str, tuple[float, float]] = {
    "ppo": (0.95, 0.995),
    "a2c": (0.95, 0.985),
    "sac": (0.95, 0.995),
}

_FALLBACK_REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}


def _load_bounds_from_config() -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[float, float]]]:
    """Load bounds from mounted swingrl.yaml, fallback to hardcoded defaults."""
    config_path = _CONFIG_PATH
    if config_path.exists():
        # yaml.safe_load OK here: memory service can't import swingrl.config.schema
        cfg = yaml.safe_load(config_path.read_text()) or {}
        training = cfg.get("training", {})
        bounds = training.get("bounds", {})
        hp_raw = bounds.get("hyperparam_bounds", {})
        rw_raw = bounds.get("reward_bounds", {})

        hyperparam_bounds: dict[str, tuple[Any, Any]] = dict(_FALLBACK_HYPERPARAM_BOUNDS)
        for key in hyperparam_bounds:
            if key in hp_raw and isinstance(hp_raw[key], (list, tuple)) and len(hp_raw[key]) == 2:
                hyperparam_bounds[key] = tuple(hp_raw[key])

        reward_bounds: dict[str, tuple[float, float]] = dict(_FALLBACK_REWARD_BOUNDS)
        for key in reward_bounds:
            if key in rw_raw and isinstance(rw_raw[key], (list, tuple)) and len(rw_raw[key]) == 2:
                reward_bounds[key] = (float(rw_raw[key][0]), float(rw_raw[key][1]))

        return hyperparam_bounds, reward_bounds
    return dict(_FALLBACK_HYPERPARAM_BOUNDS), dict(_FALLBACK_REWARD_BOUNDS)


_HYPERPARAM_BOUNDS: dict[str, tuple[Any, Any]]
_REWARD_BOUNDS: dict[str, tuple[float, float]]
_HYPERPARAM_BOUNDS, _REWARD_BOUNDS = _load_bounds_from_config()

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
    "provider": "none",
    "model": "none",
}

# ---------------------------------------------------------------------------
# Cloud API config (primary: OpenRouter, backup: NVIDIA NIM)
# ---------------------------------------------------------------------------


def _load_query_cloud_config() -> dict[str, Any]:
    """Load query agent cloud config for HP tuning (run_config).

    Uses the query_provider config field to determine primary provider.
    Supports gemini, openrouter, nvidia, and mistral. Falls back to
    openrouter as backup.
    """
    config_path = _CONFIG_PATH
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        providers = cons.get("providers", {})
        primary_name = mem.get("query_provider", "gemini")

        def _build_entry(name: str) -> dict[str, Any]:
            p = providers.get(name, {})
            env_key = f"{name.upper()}_API_KEY"
            return {
                "base_url": p.get("base_url", ""),
                "api_key": os.environ.get(env_key, p.get("api_key", "")),
                "model": p.get("default_model", ""),
                "timeout": float(p.get("timeout_sec", 600)),
                "max_tokens": int(p.get("max_tokens", 32768)),
                "provider": name,
            }

        primary = _build_entry(primary_name)
        backup_name = "openrouter" if primary_name != "openrouter" else "nvidia"
        backup = _build_entry(backup_name)

        return {"primary": primary, "backup": backup}

    return {
        "primary": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "model": "gemini-2.5-flash",
            "timeout": 60.0,
            "max_tokens": 65536,
            "provider": "gemini",
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


def _load_ollama_config() -> dict[str, Any]:
    """Load Ollama instances + epoch advice provider config.

    Returns dict with keys: query_provider, epoch_advice_provider,
    ollama_instances (list of {name, url, model, timeout}).
    Backward compatible: wraps single ollama_url/ollama_model as one-item list.
    """
    config_path = _CONFIG_PATH
    base: dict[str, Any] = {
        "query_provider": "openrouter",
        "epoch_advice_provider": "cerebras",
        "cloud_block_on_429": True,
        "cloud_block_codes": [429],
    }

    mem: dict[str, Any] = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        base["query_provider"] = mem.get("query_provider", "openrouter")
        base["epoch_advice_provider"] = mem.get("epoch_advice_provider", "cerebras")
        base["cloud_block_on_429"] = bool(mem.get("cloud_block_on_429", True))
        raw_codes = mem.get("cloud_block_codes", [429])
        base["cloud_block_codes"] = [int(c) for c in raw_codes] if raw_codes else [429]

        # New multi-instance format
        raw_instances = mem.get("ollama_instances")
        if raw_instances and isinstance(raw_instances, list):
            base["ollama_instances"] = [
                {
                    "name": inst.get("name", f"ollama-{idx}"),
                    "url": inst.get("url", ""),
                    "model": inst.get("model", ""),
                    "timeout": float(inst.get("timeout", 30)),
                }
                for idx, inst in enumerate(raw_instances)
                if inst.get("url")  # skip entries with empty url
            ]
            return base

    # Backward compat: single ollama_url/ollama_model → one-item list
    url = mem.get("ollama_url", "http://swingrl-ollama:11434")
    model = mem.get("ollama_model", "qwen2.5:14b")
    base["ollama_instances"] = [{"name": "default", "url": url, "model": model, "timeout": 30.0}]
    return base


def _load_epoch_provider_configs() -> list[dict[str, Any]]:
    """Load cloud provider configs for epoch advice with fallback chain.

    Chain: Cerebras qwen3-235b (1M TPD) → Groq llama-4-scout (500K TPD) →
    Ollama (unlimited, wired in advise_epoch()).
    Ollama is safety net in advise_epoch().
    """
    config_path = _CONFIG_PATH
    configs: list[dict[str, Any]] = []

    def _build(name: str, p: dict[str, Any], model_override: str = "") -> dict[str, Any]:
        env_key = f"{name.upper()}_API_KEY"
        return {
            "base_url": p.get("base_url", ""),
            "api_key": os.environ.get(env_key, p.get("api_key", "")),
            "model": model_override or p.get("default_model", ""),
            "timeout": float(p.get("timeout_sec", 30)),
            "max_tokens": int(p.get("max_tokens", 32768)),
            "provider": name,
        }

    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        providers = cons.get("providers", {})

        # 1. Cerebras qwen3-235b (primary — smartest model, 1M TPD)
        cerebras_cfg = providers.get("cerebras", {})
        configs.append(_build("cerebras", cerebras_cfg))

        # 2. Groq llama-4-scout (fast, different provider, 500K TPD)
        groq_cfg = providers.get("groq", {})
        configs.append(_build("groq", groq_cfg))

        return configs

    # Fallback defaults
    return [
        {
            "base_url": "https://api.cerebras.ai/v1",
            "api_key": os.environ.get("CEREBRAS_API_KEY", ""),
            "model": "qwen-3-235b-a22b-instruct-2507",
            "timeout": 30.0,
            "max_tokens": 65536,
            "provider": "cerebras",
        },
        {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "timeout": 30.0,
            "max_tokens": 30000,
            "provider": "groq",
        },
    ]


try:
    _EPOCH_PROVIDER_CONFIGS = _load_epoch_provider_configs()
except Exception:
    _EPOCH_PROVIDER_CONFIGS = []


try:
    _PROVIDER_CONFIG = _load_query_cloud_config()
except Exception as _cfg_exc:
    log.warning("query_cloud_config_load_failed", error=str(_cfg_exc))
    _PROVIDER_CONFIG = {
        "primary": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "timeout": 60.0,
            "provider": "openrouter",
        },
        "backup": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "",
            "model": "moonshotai/kimi-k2.5",
            "timeout": 1800.0,
            "provider": "nvidia",
        },
    }


try:
    _OLLAMA_CONFIG = _load_ollama_config()
except Exception as _ollama_exc:
    log.warning("ollama_config_load_failed", error=str(_ollama_exc))
    _OLLAMA_CONFIG = {
        "query_provider": "openrouter",
        "epoch_advice_provider": "ollama",
        "ollama_instances": [],
    }

_QUERY_PROVIDER: str = _OLLAMA_CONFIG["query_provider"]
_EPOCH_ADVICE_PROVIDER: str = _OLLAMA_CONFIG["epoch_advice_provider"]
_OLLAMA_INSTANCES: list[dict[str, Any]] = _OLLAMA_CONFIG.get("ollama_instances", [])
_CLOUD_BLOCK_ENABLED: bool = _OLLAMA_CONFIG.get("cloud_block_on_429", True)
_CLOUD_BLOCK_CODES: list[int] = _OLLAMA_CONFIG.get("cloud_block_codes", [429])

log.info(
    "query_provider_configured",
    hp_tuning_provider=_QUERY_PROVIDER,
    epoch_advice_provider=_EPOCH_ADVICE_PROVIDER,
    ollama_instance_count=len(_OLLAMA_INSTANCES),
    ollama_instances=[
        {"name": i["name"], "url": i["url"], "model": i["model"]} for i in _OLLAMA_INSTANCES
    ],
    cloud_block_on_429=_CLOUD_BLOCK_ENABLED,
    cloud_block_codes=_CLOUD_BLOCK_CODES,
)


def _is_provider_blocked(provider: str) -> bool:
    """Check if a cloud provider is currently blocked (exponential backoff).

    Auto-resets: if the current time is past the blocked_until timestamp,
    removes the block and returns False so the provider gets retried.
    """
    if not _CLOUD_BLOCK_ENABLED:
        return False
    with _CLOUD_BLOCKED_LOCK:
        entry = _CLOUD_BLOCKED.get(provider)
        if entry is None:
            return False
        now = datetime.now(tz=UTC)
        if now >= entry["blocked_until"]:
            del _CLOUD_BLOCKED[provider]
            log.info("cloud_provider_unblocked", provider=provider)
            return False
        return True


def _block_provider(provider: str, status_code: int) -> None:
    """Block a cloud provider with exponential backoff after a rate-limit response.

    Backoff schedule: 5min → 15min → 60min → rest-of-UTC-day.
    """
    now = datetime.now(tz=UTC)
    with _CLOUD_BLOCKED_LOCK:
        entry = _CLOUD_BLOCKED.get(provider, {"count": 0, "blocked_until": now})
        count = entry["count"] + 1
        if count == 1:
            blocked_until = now + timedelta(minutes=5)
        elif count == 2:
            blocked_until = now + timedelta(minutes=15)
        elif count == 3:
            blocked_until = now + timedelta(minutes=60)
        else:
            # Calendar-day block (rest of UTC day)
            tomorrow = now.date() + timedelta(days=1)
            blocked_until = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC)
        _CLOUD_BLOCKED[provider] = {"count": count, "blocked_until": blocked_until}
    log.warning(
        "cloud_provider_blocked",
        provider=provider,
        status_code=status_code,
        block_count=count,
        blocked_until=blocked_until.isoformat(),
        total_blocked=len(_CLOUD_BLOCKED),
    )


def validate_query_config() -> None:
    """Validate that query agent cloud config is usable at startup.

    Logs warnings for missing API keys or base URLs. Call from app lifespan.
    """
    primary = _PROVIDER_CONFIG["primary"]
    backup = _PROVIDER_CONFIG["backup"]
    log.info(
        "query_config_validated",
        primary_provider=primary["provider"],
        primary_model=primary["model"],
        primary_has_key=bool(primary["api_key"]),
        backup_provider=backup["provider"],
        backup_model=backup["model"],
        backup_has_key=bool(backup["api_key"]),
    )


def _build_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
) -> str:
    """Build the system prompt dynamically from config-loaded bounds.

    Args:
        hp_bounds: Hyperparameter bounds dict.
        rw_bounds: Reward weight bounds dict.

    Returns:
        System prompt string with bounds inlined.
    """
    hp_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in hp_bounds.items())
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        "- Algorithms: PPO (on-policy), A2C (on-policy), SAC (off-policy, "
        "entropy-maximizing)\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n"
        "- Market regimes: bull (0), bear (1), crisis (2) — detected by HMM "
        "from FRED indicators\n\n"
        f"Hyperparameter bounds (you MUST stay within these):\n{hp_lines}\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        "You will receive recent memory patterns from past training runs and "
        "queries about current training state.\n"
        "Provide specific, numerical advice grounded in the memory patterns.\n"
        "If memory patterns are insufficient, stay close to safe defaults "
        "and explain why."
    )


_HALLUCINATION_GUARD = (
    "IMPORTANT — Evidence-based changes only:\n"
    "- ONLY change a hyperparameter from its baseline when a specific pattern provides "
    "direct evidence for that change. State which pattern supports each change.\n"
    "- If no pattern provides evidence for changing a parameter, keep the baseline value "
    "and state 'keeping baseline — insufficient evidence' in rationale.\n"
    "- A single anomalous fold is an observation, not a pattern. Require consistent "
    "evidence across at least 3 folds or 2 environments before recommending a change.\n"
    "- Ask yourself: 'Could the observed behavior be explained by random variance?' "
    "If yes, keep the baseline.\n"
    "- It is better to return baseline values with honest rationale than to guess."
)

_ALGO_HP_GUIDES: dict[str, str] = {
    "ppo": (
        "PPO Hyperparameter Guide:\n"
        "- learning_rate: Controls gradient step size. Higher = chases noise, lower = safer. "
        "For overfitting: reduce from 3e-4 toward 5e-5.\n"
        "- n_epochs: MOST CRITICAL for PPO overfitting. Each epoch reuses same data with stale "
        "importance sampling. Reduce from 10 to 4-6 when overfit_gap is high.\n"
        "- clip_range: Limits policy change per sample. Narrow (0.1-0.15) prevents overfitting "
        "to any single rollout. Monitor clip_fraction: >0.3 means too narrow or too many epochs.\n"
        "- ent_coef: Entropy bonus = regularization. Prevents deterministic overfitted policies. "
        "Increase (0.01-0.02) when policy collapses.\n"
        "- batch_size: Smaller = implicit regularization via gradient noise. "
        "Reduce LR proportionally when reducing batch_size.\n"
        "- gamma: Effective planning horizon. Lower (0.95-0.97) for capital preservation.\n\n"
        "- target_kl: HIGHEST PRIORITY tunable for PPO overfitting. When set (e.g., 0.02),\n"
        "  PPO early-stops epoch iterations when approx_kl exceeds threshold. This prevents\n"
        "  the stale importance sampling overfitting that n_epochs alone cannot control.\n"
        "  Acts as adaptive n_epochs — stops at 2 epochs if KL already too high.\n"
        "  Range [0.01, 0.05]. Start at 0.02 for capital-preservation.\n\n"
        "PPO Symptom → Fix mapping:\n"
        "- High overfit_gap (IS >> OOS): Set target_kl=0.02 FIRST, then reduce n_epochs\n"
        "- Exploding approx_kl: Set target_kl=0.015 (hard stop on KL)\n"
        "- clip_fraction > 0.3: Reduce n_epochs or narrow clip_range\n"
        "- Policy collapse (entropy -> 0): Increase ent_coef\n"
        "- Control folds outperform treatment: Reduce reward adjustment frequency/magnitude"
    ),
    "a2c": (
        "A2C Hyperparameter Guide:\n"
        "CRITICAL: A2C has NO clipping protection (unlike PPO). It is fundamentally more "
        "sensitive to ALL HPs. learning_rate is the primary lever; gamma is a HIGH-RISK parameter.\n\n"
        "- learning_rate: Without clipping, LR directly controls how far the policy moves "
        "per update. For overfitting: reduce learning_rate FIRST. "
        "Safe range for financial data: 1e-4 to 3e-4.\n"
        "- gamma: HIGH-RISK parameter for A2C. With n_steps=5 (default), "
        "gamma=0.99 means the critic must predict a 100-step horizon from only 5 real "
        "reward steps — enormous prediction burden. gamma > 0.985 causes critic/n_steps "
        "mismatch and training instability. SAFE RANGE: 0.95-0.985. "
        "NEVER set gamma above 0.985 without also increasing n_steps proportionally. "
        "Higher gamma absolutely requires lower LR for stability.\n"
        "- ent_coef: Even more important than PPO because A2C has no clipping to limit "
        "policy changes. Entropy bonus is the main regularizer. Keep at 0.01-0.02.\n\n"
        "- gae_lambda: Controls bias-variance tradeoff for advantage estimation.\n"
        "  lambda=1.0 (pure MC) gives maximum variance with n_steps=5 — only 5 real steps.\n"
        "  lambda=0.92 adds bootstrapping that reduces variance ~40% without significant bias.\n"
        "  Range [0.85, 1.0]. Lower lambda = more bias, less variance, faster convergence.\n"
        "  MISMATCH CONSTRAINT: (1/(1-gamma)) / n_steps must be < 8. The system enforces\n"
        "  this automatically — gamma will be clamped down if mismatch exceeds 8.\n\n"
        "A2C Symptom → Fix mapping:\n"
        "- High overfit_gap: Reduce learning_rate FIRST, then check gamma\n"
        "- Erratic policy (whipsawing trades): Reduce learning_rate, reduce gae_lambda\n"
        "- Policy collapse: Increase ent_coef\n"
        "- Training degradation across folds: CHECK GAMMA FIRST — if gamma > 0.985, "
        "reduce it to 0.97-0.98\n"
        "- High variance across folds: Reduce gae_lambda from 1.0 toward 0.90-0.95\n"
        "- Myopic trading: Increase gamma (max 0.985) or increase n_steps"
    ),
    "sac": (
        "SAC Hyperparameter Guide:\n"
        "SAC is off-policy with replay buffer. Maximum entropy framework provides built-in "
        "exploration. Twin Q-networks reduce overestimation. Key concern: replay buffer "
        "can hold stale data from different market regimes.\n\n"
        "- learning_rate: Applied to actor, both critics, AND entropy coefficient optimizer. "
        "Lower than PPO/A2C is appropriate because replay buffer provides more gradient "
        "updates per step. For overfitting: reduce to 3e-5 to 1e-4.\n"
        "- batch_size: Smaller (64) = implicit regularization through gradient noise. "
        "When reducing batch_size, reduce LR proportionally.\n"
        "- gamma: High gamma + large replay buffer = critic learns average across regimes. "
        "Lower (0.95-0.97) helps with regime-specific accuracy.\n"
        "- ent_coef: 'auto_0.1' starts alpha at 0.1 (10× lower than default). Default\n"
        "  'auto' starts at alpha=1.0 which drowns small financial rewards for ~100K steps.\n"
        "  ent_coef is mathematically equivalent to the inverse of reward scale (Haarnoja 2019).\n"
        "- gradient_steps: Controls Update-to-Data ratio. Default 1 = one gradient update\n"
        "  per env step. Increasing to 4 extracts 4× more learning per transition.\n"
        "  MUST reduce learning_rate proportionally (halve it) when increasing gradient_steps.\n"
        "  Also reduce tau proportionally (~1/k for gradient_steps=k) to prevent target\n"
        "  network instability.\n"
        "- target_entropy: Default -dim(action_space) = -3 for crypto, -9 for equity.\n"
        "  Too high for portfolio allocation where the agent should be decisive.\n"
        "  -1.0 for crypto and -3.0 for equity are better starting points.\n\n"
        "SAC Symptom → Fix mapping:\n"
        "- Flat Sharpe across iterations: ent_coef drowning reward signal → use 'auto_0.1'\n"
        "- High overfit_gap: Reduce learning_rate, increase gradient_steps (better extraction)\n"
        "- Q-values exploding: Reduce learning_rate, reduce gradient_steps\n"
        "- Over-exploration (random trades): Lower target_entropy (e.g., -1.0 for crypto)\n"
        "- Regime confusion: Reduce gamma (shorter horizon = less regime averaging)\n"
        "- Persistent high MDD: Increase gradient_steps to 4 (better Q-value learning)"
    ),
}

_REWARD_WEIGHT_GUIDE = (
    "Reward Weight Adjustment Guide:\n"
    "- High MDD / frequent large losses: Increase drawdown weight (0.2 -> 0.35)\n"
    "  CAUTION: Excessive drawdown penalty can cause inactivity or position concentration,\n"
    "  paradoxically increasing tail risk (Goodhart's Law). If MDD does not improve after\n"
    "  increasing drawdown weight, the HP configuration may be the root cause.\n"
    "- Too many trades / high turnover: Increase turnover weight (0.05 -> 0.15)\n"
    "  CAUTION: High turnover penalty can cause the agent to hold losing positions.\n"
    "- Good Sharpe but low total return: Increase profit weight\n"
    "- Volatile returns: Increase sharpe weight (smooths returns)\n"
    "- Agent holds through drawdowns: Increase drawdown weight + check gamma isn't too low\n\n"
    "CRITICAL INTERACTIONS:\n"
    "- High ent_coef (>0.02) can wash out reward weight signals — entropy bonus dominates\n"
    "  small financial reward components. Consider ent_coef before adjusting weights.\n"
    "- SAC auto-temperature lags behind reward scale changes — transient mismatch expected.\n"
    "- PPO value function clipping interacts with reward scale — avoid large weight jumps.\n\n"
    "ONLY adjust reward weights when the current epoch metrics show a clear problem. "
    "If rolling_sharpe > 0 and rolling_mdd > -0.05, the current weights are working — keep them. "
    "Prefer small adjustments (delta < 0.1) over large swings. "
    "Gradual adaptation outperforms abrupt changes. Do NOT oscillate weights back and forth."
)

_ALGO_EPOCH_GUIDES: dict[str, str] = {
    "ppo": (
        "PPO Reward Adjustment Constraints:\n"
        "- PPO's value function needs 2+ rollout cycles to adapt after reward weight changes.\n"
        "  Large changes cause stale advantage estimates and degrade policy quality.\n"
        "- ONLY adjust when rolling_mdd > -10% or rolling_sharpe < -1.0 (severe conditions).\n"
        "- Maximum per-component weight delta: 0.03. Smaller changes are safer for PPO.\n"
        "- Crypto PPO: reward adjustments are DISABLED — control folds outperform treatment\n"
        "  by 29.5% across 4 iterations. If env=crypto, return current weights unchanged.\n"
        "- If metrics are decent (sharpe > 0, mdd > -5%), keep current weights — PPO benefits\n"
        "  more from stability than from fine-tuning reward weights mid-training."
    ),
    "a2c": (
        "A2C Reward Adjustment Constraints:\n"
        "- A2C has no clipping protection — reward changes take effect within 5 steps.\n"
        "  This makes A2C highly responsive but also fragile to overshooting.\n"
        "- EQUITY: Max delta 0.02. Equity is lower-variance; same weight change is a larger\n"
        "  relative perturbation. Only adjust if rolling_sharpe < 0 or rolling_mdd > -5%.\n"
        "- CRYPTO: Max delta 0.05. Crypto's higher variance makes it more tolerant of\n"
        "  reward weight changes. Treatment improves A2C crypto by +30.5% vs control.\n"
        "- With gae_lambda=1.0 and n_steps=5, A2C uses pure Monte Carlo returns on very\n"
        "  short windows. Reward changes compound quickly — prefer smaller, directional moves."
    ),
    "sac": (
        "SAC Reward Adjustment Constraints:\n"
        "- SAC's replay buffer contains transitions scored under OLD reward weights.\n"
        "  After a weight change, the Q-function trains on mixed-weight data until the\n"
        "  buffer rotates (~20K steps). Allow this lag before measuring effectiveness.\n"
        "- SAC's entropy coefficient interacts with reward scale. Changing weights effectively\n"
        "  shifts the entropy-reward balance. auto-alpha adapts but not instantaneously.\n"
        "- Maximum per-component weight delta: 0.02. SAC is sensitive because both replay\n"
        "  buffer staleness and entropy coupling amplify the impact of weight changes.\n"
        "- If drawdown weight was already increased and MDD did not improve after 20K steps,\n"
        "  the issue is likely HP configuration (ent_coef, gamma), not reward weights."
    ),
}

_RUN_CONFIG_INSTRUCTIONS = (
    "For each hyperparameter you recommend changing from baseline:\n"
    "1. Name the specific pattern (by content, not ID) that supports the change\n"
    "2. Cite the specific metric from that pattern (e.g., 'overfit_gap >0.30 in 9/13 folds')\n"
    "3. Explain why that pattern implies this specific HP change\n\n"
    "PRIORITIZE hp_effectiveness and iteration_regression patterns when present — they\n"
    "directly show which HP changes worked or failed in previous iterations. If an\n"
    "hp_effectiveness pattern says a specific HP change caused regression, do NOT repeat\n"
    "that change. If an iteration_regression pattern flags HP bounds violations, ensure\n"
    "your recommendations stay within documented safe ranges.\n\n"
    "If fewer than 2 patterns support HP changes, return mostly baseline values.\n"
    "It is better to return safe defaults with honest rationale than to over-tune.\n\n"
    "BEST-KNOWN CONFIG ANCHORING:\n"
    "The user message includes 'VS BEST PER-ALGO' and 'VS BEST ENSEMBLE' comparisons.\n"
    "When the best iteration is NOT the current iteration, ANCHOR your recommendations\n"
    "near the best-known HP config. Stay within ±20% of the best-known values for\n"
    "learning_rate and within ±0.02 of the best-known gamma, UNLESS you have strong\n"
    "pattern-based evidence (cited from >=3 folds or 2 environments) that diverging\n"
    "from the best config will improve performance. State explicitly when you are\n"
    "diverging from the best-known config and why."
)


def _build_algo_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
    algo_name: str,
) -> str:
    """Build algo-specific system prompt with filtered HP bounds and tuning guide.

    Args:
        hp_bounds: Hyperparameter bounds dict (all algos).
        rw_bounds: Reward weight bounds dict.
        algo_name: Algorithm name ('ppo', 'a2c', 'sac').

    Returns:
        System prompt string with algo-specific bounds, HP guide, and hallucination guards.
    """
    valid_hp = set(_ALGO_HP_FIELDS.get(algo_name, {}).keys())
    filtered_bounds = {k: v for k, v in hp_bounds.items() if k in valid_hp}
    # Override gamma bounds with algo-specific values when available
    if "gamma" in filtered_bounds and algo_name.lower() in _ALGO_GAMMA_BOUNDS:
        filtered_bounds["gamma"] = _ALGO_GAMMA_BOUNDS[algo_name.lower()]
    hp_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in filtered_bounds.items())
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    algo_upper = algo_name.upper()
    valid_fields = ", ".join(valid_hp)
    hp_guide = _ALGO_HP_GUIDES.get(algo_name, "")
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        f"You are advising hyperparameters for the {algo_upper} algorithm.\n"
        f"ONLY include these hyperparameter fields: {valid_fields}\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        f"- Algorithm: {algo_upper} ("
        f"{'on-policy' if algo_name in ('ppo', 'a2c') else 'off-policy, entropy-maximizing'}"
        f")\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n"
        "- Market regimes: bull (0), bear (1), crisis (2) — detected by HMM "
        "from FRED indicators\n\n"
        f"Hyperparameter bounds for {algo_upper} (you MUST stay within these):\n{hp_lines}\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        f"{_HALLUCINATION_GUARD}\n\n"
        f"{hp_guide}\n\n"
        f"{_RUN_CONFIG_INSTRUCTIONS}"
    )


def _build_epoch_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
    algo_name: str,
) -> str:
    """Build algo-specific system prompt for mid-training epoch advice.

    Includes the algo HP guide, reward weight adjustment guide, and hallucination guards.

    Args:
        hp_bounds: Hyperparameter bounds dict (all algos).
        rw_bounds: Reward weight bounds dict.
        algo_name: Algorithm name ('ppo', 'a2c', 'sac').

    Returns:
        System prompt for epoch_advice requests.
    """
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    algo_upper = algo_name.upper()
    hp_guide = _ALGO_HP_GUIDES.get(algo_name, "")
    epoch_guide = _ALGO_EPOCH_GUIDES.get(algo_name, "")
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        f"You are advising mid-training reward weight adjustments for {algo_upper}.\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        f"- Algorithm: {algo_upper} ("
        f"{'on-policy' if algo_name in ('ppo', 'a2c') else 'off-policy, entropy-maximizing'}"
        f")\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        "FOLD ADJUSTMENT HISTORY: The user message may include recent reward weight\n"
        "adjustments from THIS fold with measured sharpe/mdd deltas and effectiveness.\n"
        "Use this to avoid repeating ineffective adjustments. If a specific dimension\n"
        "change was ineffective, try a different dimension or keep current weights.\n\n"
        f"{_HALLUCINATION_GUARD}\n\n"
        f"{hp_guide}\n\n"
        f"{epoch_guide}\n\n"
        f"{_REWARD_WEIGHT_GUIDE}\n\n"
        "You MUST respond with a valid JSON object matching the schema."
    )


# ---------------------------------------------------------------------------
# Relevant categories per request type (R2)
# ---------------------------------------------------------------------------

_RELEVANT_CATEGORIES: dict[str, list[str]] = {
    "run_config": [
        "regime_performance",
        "overfit_diagnosis",
        "iteration_progression",
        "data_size_impact",
        "macro_transition",
        "cross_env",
        "cross_env_correlation",
        "hp_effectiveness",
        "iteration_regression",
    ],
    "epoch_advice": [
        "drawdown_recovery",
        "trade_quality",
        "reward_shaping",
        "overfit_diagnosis",
        "iteration_progression",
        "hp_effectiveness",
        "iteration_regression",
    ],
    "live_trading": [
        "live_cycle_gate",
        "live_blend_weights",
        "live_risk_thresholds",
        "live_position",
        "live_trade_veto",
        "cross_env_correlation",
    ],
}


def _parse_query_context(query: str) -> dict[str, str | int | None]:
    """Parse env_name, algo_name, iteration, and run_id from query string."""
    env_name: str | None = None
    algo_name: str | None = None
    iteration: int | None = None
    run_id: str | None = None
    for part in query.split():
        if part.startswith("env="):
            env_name = part.split("=", 1)[1].lower()
        elif part.startswith("algo="):
            algo_name = part.split("=", 1)[1].lower()
        elif part.startswith("iteration=") or part.startswith("iter="):
            try:
                iteration = int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif part.startswith("run_id="):
            run_id = part.split("=", 1)[1]
    return {
        "env_name": env_name,
        "algo_name": algo_name,
        "iteration": iteration,
        "run_id": run_id,
    }


def _load_min_confidence() -> float:
    """Load min_confidence_for_advice from config, default 0.4."""
    config_path = _CONFIG_PATH
    if config_path.exists():
        # yaml.safe_load OK here: memory service can't import swingrl.config.schema
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        return float(cons.get("min_confidence_for_advice", 0.4))
    return 0.4


_MIN_CONFIDENCE = _load_min_confidence()

# System prompt for training config / epoch advice endpoints.
# Built dynamically from config-loaded bounds (Item 8).
_SYSTEM_PROMPT = _build_system_prompt(_HYPERPARAM_BOUNDS, _REWARD_BOUNDS)

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

# Per-algo HP field definitions (algo-aware schema enforcement)
_ALGO_HP_FIELDS: dict[str, dict[str, dict[str, str]]] = {
    "ppo": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "clip_range": {"type": "number"},
        "n_epochs": {"type": "integer"},
        "batch_size": {"type": "integer"},
        "gamma": {"type": "number"},
        "target_kl": {"type": "number"},
    },
    "a2c": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "gamma": {"type": "number"},
        "gae_lambda": {"type": "number"},
    },
    "sac": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "batch_size": {"type": "integer"},
        "gamma": {"type": "number"},
        "gradient_steps": {"type": "integer"},
        "target_entropy": {"type": "number"},
    },
}


def _build_run_config_schema(algo_name: str) -> dict[str, Any]:
    """Build JSON schema for run_config filtered to algo-valid HP fields.

    Args:
        algo_name: Algorithm name ('ppo', 'a2c', 'sac').

    Returns:
        JSON schema dict with only the HP fields valid for the given algo.
    """
    hp_fields = _ALGO_HP_FIELDS.get(algo_name, _ALGO_HP_FIELDS["ppo"])
    properties: dict[str, Any] = dict(hp_fields)
    properties["reward_weights"] = {
        "type": "object",
        "properties": {
            "profit": {"type": "number"},
            "sharpe": {"type": "number"},
            "drawdown": {"type": "number"},
            "turnover": {"type": "number"},
        },
        "additionalProperties": False,
    }
    properties["rationale"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["rationale"],
        "additionalProperties": False,
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


def _clamp_run_config(raw: dict[str, Any], *, algo: str | None = None) -> dict[str, Any]:
    """Clamp hyperparameters from LLM response to safe ranges.

    Args:
        raw: Dict from LLM response.
        algo: Optional algorithm name for algo-specific gamma bounds.

    Returns:
        New dict with clamped values.
    """
    import math

    out: dict[str, Any] = {}
    for key, (lo, hi) in _HYPERPARAM_BOUNDS.items():
        if key in raw:
            eff_lo, eff_hi = lo, hi
            if key == "gamma" and algo and algo.lower() in _ALGO_GAMMA_BOUNDS:
                eff_lo, eff_hi = _ALGO_GAMMA_BOUNDS[algo.lower()]
            clamped = _clamp(float(raw[key]), float(eff_lo), float(eff_hi))
            if key == "batch_size":
                # Force to nearest power of 2
                clamped = int(2 ** round(math.log2(max(1, clamped))))
            elif key in ("n_epochs", "gradient_steps"):
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
    """Advises RL training configuration using memory patterns + cloud LLM."""

    _last_provider: str = "unknown"
    _last_model: str = "unknown"

    def _audit_log(
        self,
        call_type: str,
        prompt_text: str,
        response_text: str | None,
        response_parsed: str | None,
        latency_ms: int,
        success: bool,
        error_text: str | None = None,
        **context: Any,
    ) -> None:
        """Fire-and-forget audit log entry for LLM calls."""
        try:
            insert_audit_log(
                call_type=call_type,
                provider=self._last_provider,
                model_name=self._last_model,
                prompt_text=prompt_text[:50000],
                response_text=response_text,
                response_parsed=response_parsed,
                latency_ms=latency_ms,
                success=success,
                error_text=error_text,
                algo=context.get("algo"),
                env=context.get("env"),
                fold_number=context.get("fold_number"),
                iteration_number=context.get("iteration_number"),
                is_control_fold=context.get("is_control_fold"),
            )
        except Exception:
            log.warning("query_audit_log_failed", call_type=call_type)

    async def advise_run_config(self, query: str) -> dict[str, Any]:
        """Return LLM-advised run configuration hyperparameters.

        Builds a context prompt from active consolidations (filtered by confidence,
        env, algo, and category relevance), queries cloud LLM, clamps the response.
        Returns safe defaults on any failure.

        Args:
            query: Context string from MetaTrainingOrchestrator (algo, env, iteration info).

        Returns:
            Dict with keys: learning_rate, entropy_coeff, clip_range, n_epochs,
            batch_size, gamma, reward_weights (dict), rationale.
        """
        parsed = _parse_query_context(query)
        context, presented_ids = await self._build_context_async(
            env_name=parsed["env_name"],
            algo_name=parsed["algo_name"],
            request_type="run_config",
        )

        algo = parsed["algo_name"] or "ppo"
        algo_upper = algo.upper()
        valid_fields = ", ".join(_ALGO_HP_FIELDS.get(algo, {}).keys())

        user_content = (
            f"Training context: {query}\n\n"
            f"Relevant memory patterns:\n{context}\n\n"
            f"Algorithm: {algo_upper}\n"
            f"Valid hyperparameter fields for {algo_upper}: {valid_fields}\n\n"
            f"Based ONLY on the memory patterns above, recommend hyperparameters "
            f"for this {algo_upper} training run. Cite the specific pattern that "
            f"supports each recommendation."
        )

        schema = _build_run_config_schema(algo)
        system_prompt = _build_algo_system_prompt(_HYPERPARAM_BOUNDS, _REWARD_BOUNDS, algo)

        # HP tuning uses cloud (OpenRouter nemotron) — smart model, 6 calls/run
        _t0 = time.monotonic()
        result = await self._call_lm(user_content, schema, system_prompt=system_prompt)
        _latency = int((time.monotonic() - _t0) * 1000)
        self._audit_log(
            call_type="run_config",
            prompt_text=user_content,
            response_text=json.dumps(result) if result else None,
            response_parsed=json.dumps(result) if result else None,
            latency_ms=_latency,
            success=result is not None,
            algo=parsed.get("algo_name"),
            env=parsed.get("env_name"),
        )
        if result is None:
            log.warning("run_config_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_DEFAULTS)

        # Track which patterns were presented
        await self._track_presentations_async(presented_ids, query, "run_config", result)

        # Merge with defaults for any missing fields
        merged = dict(_SAFE_DEFAULTS)
        merged.update(result)
        clamped = _clamp_run_config(merged, algo=algo)

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
        parsed = _parse_query_context(query)
        algo = parsed["algo_name"] or "ppo"
        context, presented_ids = await self._build_context_async(
            env_name=parsed["env_name"],
            algo_name=algo,
            request_type="epoch_advice",
        )

        # Build within-fold adjustment history from recent OUTCOMEs
        fold_history = ""
        run_id = parsed.get("run_id")
        if run_id:
            from db import get_recent_outcomes_for_run_id_async  # noqa: PLC0415

            recent_outcomes = await get_recent_outcomes_for_run_id_async(run_id, limit=5)
            if recent_outcomes:
                import json as _json  # noqa: PLC0415
                import re as _re  # noqa: PLC0415

                lines = ["Recent adjustments in this fold (most recent first):"]
                for outcome in recent_outcomes:
                    txt = outcome.get("text", "")
                    ep_m = _re.search(r"epoch_triggered=(\d+)", txt)
                    sd_m = _re.search(r"post_adjustment_sharpe_delta=([0-9.e+-]+)", txt)
                    md_m = _re.search(r"post_adjustment_mdd_delta=([0-9.e+-]+)", txt)
                    eff_m = _re.search(r"adjustment_effective=(True|False)", txt)
                    bw_m = _re.search(r"weights_before=(\{[^}]+\})", txt)
                    aw_m = _re.search(r"weights_after=(\{[^}]+\})", txt)

                    # Build per-dimension change labels
                    dim_parts: list[str] = []
                    if bw_m and aw_m:
                        try:
                            bw = _json.loads(bw_m.group(1).replace("'", '"'))
                            aw = _json.loads(aw_m.group(1).replace("'", '"'))
                            for dim in ("profit", "sharpe", "drawdown", "turnover"):
                                d = aw.get(dim, 0.0) - bw.get(dim, 0.0)
                                if abs(d) > 0.001:
                                    dim_parts.append(f"{dim} {d:+.3f}")
                        except (ValueError, TypeError):
                            pass

                    ep = ep_m.group(1) if ep_m else "?"
                    sd = float(sd_m.group(1)) if sd_m else 0.0
                    md = float(md_m.group(1)) if md_m else 0.0
                    eff = "effective" if (eff_m and eff_m.group(1) == "True") else "ineffective"
                    dim_label = f"[{', '.join(dim_parts)}]" if dim_parts else ""

                    lines.append(
                        f"- epoch {ep}: {dim_label} "
                        f"sharpe_delta={sd:+.4f} mdd_delta={md:+.4f} ({eff})"
                    )
                lines.append(
                    "Use this history to avoid repeating ineffective adjustments. "
                    "If previous adjustments in the same dimension were ineffective, "
                    "try a different dimension or keep current weights."
                )
                fold_history = "\n".join(lines) + "\n\n"

        user_content = (
            f"Current epoch state: {query}\n\n"
            f"{fold_history}"
            f"Memory patterns:\n{context}\n\n"
            "Should I adjust reward weights or stop training early?"
        )

        epoch_prompt = _build_epoch_system_prompt(_HYPERPARAM_BOUNDS, _REWARD_BOUNDS, algo)

        # Parse fold_number and control status from query for audit
        _fold_num: int | None = None
        _is_ctrl: bool | None = None
        _run_match = re.search(r"run_id=\S+_fold(\d+)(_CTRL)?", query)
        if _run_match:
            _fold_num = int(_run_match.group(1))
            _is_ctrl = _run_match.group(2) is not None

        _t0 = time.monotonic()
        # Route based on epoch_advice_provider config
        if _EPOCH_ADVICE_PROVIDER == "ollama":
            result = await self._call_ollama(
                user_content,
                _EPOCH_ADVICE_SCHEMA,
                system_prompt=epoch_prompt,
            )
        elif _EPOCH_ADVICE_PROVIDER != "ollama":
            # Use dedicated cloud chain for epoch advice
            result = await self._call_epoch_cloud(
                user_content,
                _EPOCH_ADVICE_SCHEMA,
                system_prompt=epoch_prompt,
            )
            # Ollama safety net: if all cloud providers failed, try local Ollama instances
            if result is None and _OLLAMA_INSTANCES:
                log.info(
                    "epoch_advice_cloud_exhausted_trying_ollama",
                    instance_count=len(_OLLAMA_INSTANCES),
                )
                result = await self._call_ollama(
                    user_content,
                    _EPOCH_ADVICE_SCHEMA,
                    system_prompt=epoch_prompt,
                )
                if result is not None:
                    log.info("epoch_advice_ollama_fallback_succeeded")
        _latency = int((time.monotonic() - _t0) * 1000)
        self._audit_log(
            call_type="epoch_advice",
            prompt_text=user_content,
            response_text=json.dumps(result) if result else None,
            response_parsed=json.dumps(result) if result else None,
            latency_ms=_latency,
            success=result is not None,
            algo=algo,
            env=parsed.get("env_name"),
            fold_number=_fold_num,
            is_control_fold=_is_ctrl,
        )
        if result is None:
            log.warning("epoch_advice_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_EPOCH_DEFAULTS)

        # Track which patterns were presented
        await self._track_presentations_async(presented_ids, query, "epoch_advice", result)

        clamped_weights = _clamp_reward_weights(
            result.get("reward_weights", _SAFE_EPOCH_DEFAULTS["reward_weights"])
        )

        result_out = {
            "reward_weights": clamped_weights,
            "stop_training": bool(result.get("stop_training", False)),
            "rationale": str(result.get("rationale", "llm_advised")),
        }
        result_out["provider"] = self._last_provider
        result_out["model"] = self._last_model
        log.info(
            "epoch_advice_response",
            provider=self._last_provider,
            model=self._last_model,
            response_preview=json.dumps(result_out)[:500],
        )
        return result_out

    @staticmethod
    def _filter_and_format_context(
        all_consolidations: list[dict[str, Any]],
        raw_memories: list[dict[str, Any]],
        algo_name: str | None,
        env_name: str | None,
        request_type: str,
    ) -> tuple[str, list[int]]:
        """Filter consolidations by algo and format into a context string.

        Shared logic extracted from sync/async _build_context variants.

        Args:
            all_consolidations: Combined Stage 2 + Stage 1 consolidation rows.
            raw_memories: Raw memory rows (only used if consolidations < 3).
            algo_name: Post-filter by affected_algos field (None = all).
            env_name: Environment name (for logging only).
            request_type: Category relevance key (for logging only).

        Returns:
            Tuple of (formatted context string, list of consolidation IDs presented).
        """
        # Post-filter by algo_name: keep universal patterns + those matching
        if algo_name:
            algo_lower = algo_name.lower()
            consolidations = []
            for c in all_consolidations:
                algos = c.get("affected_algos")
                if not algos:
                    consolidations.append(c)
                elif isinstance(algos, list) and algo_lower in [a.lower() for a in algos]:
                    consolidations.append(c)
        else:
            consolidations = all_consolidations

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

        # R3: Only include raw memories if insufficient consolidations (cold start)
        if len(consolidations) < 3 and raw_memories:
            parts.append("=== Recent Raw Memories ===")
            _max_raw_chars = 20_000  # ~5K tokens, leaves room for system + response
            total_chars = 0
            for m in raw_memories:
                text = m["text"]
                if total_chars + len(text) > _max_raw_chars:
                    remaining = _max_raw_chars - total_chars
                    if remaining > 200:
                        parts.append(text[:remaining] + "...[truncated]")
                    break
                parts.append(text)
                total_chars += len(text)

        if not parts:
            return "<no_memories>No memories available yet (cold start)</no_memories>", []

        log.info(
            "context_built",
            env=env_name,
            algo=algo_name,
            request_type=request_type,
            pattern_count=len(consolidations),
            raw_included=len(consolidations) < 3,
        )
        return "\n".join(parts), presented_ids

    def _build_context(
        self,
        env_name: str | None = None,
        algo_name: str | None = None,
        request_type: str = "run_config",
    ) -> tuple[str, list[int]]:
        """Build a context string from active consolidations and raw memories.

        Filters consolidations by env, algo, category relevance, and confidence.
        Prefers Stage 2 (cross-env) over Stage 1 (per-env) when both available.
        Skips raw memories when sufficient consolidations exist (R3).

        Args:
            env_name: Filter patterns to this environment (None = all).
            algo_name: Post-filter by affected_algos field (None = all).
            request_type: Category relevance key ('run_config', 'epoch_advice', etc.).

        Returns:
            Tuple of (formatted context string, list of consolidation IDs presented).
        """
        categories = _RELEVANT_CATEGORIES.get(request_type)

        stage2 = get_active_consolidations(
            env_name=env_name,
            stage=2,
            min_confidence=_MIN_CONFIDENCE,
            categories=categories,
            limit_per_category=3,
        )
        stage1 = get_active_consolidations(
            env_name=env_name,
            stage=1,
            min_confidence=_MIN_CONFIDENCE,
            categories=categories,
            limit_per_category=3,
        )
        all_consolidations = stage2 + stage1

        raw_memories: list[dict[str, Any]] = []
        # Prefetch raw memories; _filter_and_format_context decides whether to use them
        memories_candidate = get_memories(archived=False, limit=10)
        if memories_candidate:
            raw_memories = memories_candidate

        return self._filter_and_format_context(
            all_consolidations,
            raw_memories,
            algo_name,
            env_name,
            request_type,
        )

    async def _build_context_async(
        self,
        env_name: str | None = None,
        algo_name: str | None = None,
        request_type: str = "run_config",
    ) -> tuple[str, list[int]]:
        """Async version of _build_context — uses async DB wrappers.

        Args:
            env_name: Filter patterns to this environment (None = all).
            algo_name: Post-filter by affected_algos field (None = all).
            request_type: Category relevance key ('run_config', 'epoch_advice', etc.).

        Returns:
            Tuple of (formatted context string, list of consolidation IDs presented).
        """
        categories = _RELEVANT_CATEGORIES.get(request_type)

        stage2 = await get_active_consolidations_async(
            env_name=env_name,
            stage=2,
            min_confidence=_MIN_CONFIDENCE,
            categories=categories,
            limit_per_category=3,
        )
        stage1 = await get_active_consolidations_async(
            env_name=env_name,
            stage=1,
            min_confidence=_MIN_CONFIDENCE,
            categories=categories,
            limit_per_category=3,
        )
        all_consolidations = stage2 + stage1

        raw_memories: list[dict[str, Any]] = []
        memories_candidate = await get_memories_async(archived=False, limit=10)
        if memories_candidate:
            raw_memories = memories_candidate

        return self._filter_and_format_context(
            all_consolidations,
            raw_memories,
            algo_name,
            env_name,
            request_type,
        )

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
        parsed = _parse_query_context(query)
        iteration = parsed["iteration"]
        env_name = parsed["env_name"]

        raw_rationale = result.get("rationale", "")
        if isinstance(raw_rationale, list):
            raw_rationale = " ".join(str(r) for r in raw_rationale)
        advice_summary = str(raw_rationale)[:200]

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
                log.warning("presentation_tracking_failed", consolidation_id=cid, error=str(exc))

    async def _track_presentations_async(
        self,
        consolidation_ids: list[int],
        query: str,
        request_type: str,
        result: dict[str, Any],
    ) -> None:
        """Async version of _track_presentations — uses async DB wrappers.

        Args:
            consolidation_ids: IDs of consolidations included in context.
            query: The query string (extract iteration/env from it).
            request_type: 'run_config' or 'epoch_advice'.
            result: The LLM response (for advice_response summary).
        """
        parsed = _parse_query_context(query)
        iteration = parsed["iteration"]
        env_name = parsed["env_name"]

        raw_rationale = result.get("rationale", "")
        if isinstance(raw_rationale, list):
            raw_rationale = " ".join(str(r) for r in raw_rationale)
        advice_summary = str(raw_rationale)[:200]

        for cid in consolidation_ids:
            try:
                await insert_pattern_presentation_async(
                    consolidation_id=cid,
                    iteration=iteration,
                    env_name=env_name,
                    request_type=request_type,
                    advice_response=advice_summary,
                )
            except Exception as exc:
                log.warning("presentation_tracking_failed", consolidation_id=cid, error=str(exc))

    async def _call_cloud_api(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        provider: str,
        system_prompt: str | None = None,
        max_tokens: int = 32768,
    ) -> dict[str, Any] | None:
        """Call cloud API (OpenAI-compatible) with structured JSON output.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output enforcement.
            base_url: Provider base URL.
            api_key: Bearer token for the provider.
            model: Model identifier string.
            timeout: Read timeout in seconds.
            provider: Provider name for logging.
            system_prompt: Override system prompt (defaults to module-level _SYSTEM_PROMPT).
            max_tokens: Maximum output tokens (per-provider from config).

        Returns:
            Parsed dict from LLM response, or None on failure.
        """
        self._last_provider = provider
        self._last_model = model
        effective_max_tokens = max_tokens
        effective_system_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)
            ) as client:
                request_body: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": effective_system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "max_tokens": effective_max_tokens,
                }

                # Provider-specific structured output enforcement
                if provider == "openrouter":
                    request_body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "run_config",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                elif provider == "nvidia":
                    request_body["response_format"] = {"type": "json_object"}
                    request_body["guided_json"] = schema
                elif provider == "gemini":
                    # Gemini supports json_schema via OpenAI-compatible endpoint
                    request_body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "run_config",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                elif provider == "mistral":
                    request_body["response_format"] = {"type": "json_object"}
                elif provider == "cerebras":
                    # Cerebras supports strict JSON schema
                    request_body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "run_config",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                elif provider == "groq":
                    request_body["response_format"] = {"type": "json_object"}
                else:
                    request_body["response_format"] = {"type": "json_object"}

                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                if resp.status_code >= 400:
                    log.error(
                        "query_cloud_http_error",
                        provider=provider,
                        status_code=resp.status_code,
                        response_body=resp.text[:500],
                    )
                    if resp.status_code in _CLOUD_BLOCK_CODES:
                        _block_provider(provider, resp.status_code)
                resp.raise_for_status()
                body = resp.json()
                raw_content = body["choices"][0]["message"]["content"]
                if raw_content is None:
                    log.error(
                        "query_cloud_empty_content",
                        provider=provider,
                        finish_reason=body["choices"][0].get("finish_reason"),
                    )
                    return None
                return json.loads(raw_content)
        except httpx.HTTPStatusError as exc:
            log.error(
                "query_cloud_call_failed",
                provider=provider,
                error=str(exc),
                status_code=exc.response.status_code,
                response_body=exc.response.text[:500],
                exc_type=type(exc).__name__,
            )
            return None
        except Exception as exc:
            log.error(
                "query_cloud_call_failed",
                provider=provider,
                error=str(exc) or repr(exc),
                exc_type=type(exc).__name__,
            )
            return None

    async def _call_lm(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Call primary (OpenRouter) then backup (NVIDIA) cloud API.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output enforcement.
            system_prompt: Override system prompt (defaults to module-level _SYSTEM_PROMPT).
        """
        for tier in ("primary", "backup"):
            cfg = _PROVIDER_CONFIG[tier]
            if not cfg["api_key"]:
                log.debug(
                    "query_provider_skipped_no_key",
                    provider=cfg["provider"],
                    tier=tier,
                )
                continue
            if _is_provider_blocked(cfg["provider"]):
                log.info(
                    "query_provider_skipped_blocked",
                    provider=cfg["provider"],
                    tier=tier,
                )
                continue
            result = await self._call_cloud_api(
                user_content,
                schema,
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                timeout=cfg["timeout"],
                provider=cfg["provider"],
                system_prompt=system_prompt,
                max_tokens=cfg.get("max_tokens", 32768),
            )
            if result is not None:
                return result
            log.warning(
                "query_provider_failed",
                provider=cfg["provider"],
                tier=tier,
            )
        return None

    async def _call_ollama(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Call Ollama API with structured JSON output, trying instances sequentially.

        Iterates ``_OLLAMA_INSTANCES`` in order (serialized via semaphore).
        Returns the first successful result or None if all instances fail.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output via format parameter.
            system_prompt: System prompt (defaults to module-level _SYSTEM_PROMPT).

        Returns:
            Parsed JSON dict on success, None on any error.
        """
        if not _OLLAMA_INSTANCES:
            log.debug("ollama_skipped_no_instances")
            return None

        effective_system = system_prompt or _SYSTEM_PROMPT

        for idx, instance in enumerate(_OLLAMA_INSTANCES):
            inst_url = instance["url"]
            inst_model = instance["model"]
            inst_name = instance["name"]
            self._last_provider = f"ollama:{inst_name}"
            self._last_model = inst_model
            inst_timeout = instance.get("timeout", 30.0)

            if idx > 0:
                log.info(
                    "ollama_trying_next_instance",
                    instance=inst_name,
                    url=inst_url,
                    model=inst_model,
                    position=idx + 1,
                    total=len(_OLLAMA_INSTANCES),
                )

            payload = {
                "model": inst_model,
                "messages": [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": user_content},
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512, "num_ctx": 8192},
            }

            try:
                log.debug("ollama_semaphore_waiting", instance=inst_name)
                async with _OLLAMA_SEMAPHORE:
                    async with httpx.AsyncClient(timeout=inst_timeout) as client:
                        resp = await client.post(
                            f"{inst_url}/api/chat",
                            json=payload,
                        )
                        resp.raise_for_status()
                        body = resp.json()

                raw_content = body.get("message", {}).get("content")
                if not raw_content:
                    log.warning(
                        "ollama_empty_content",
                        instance=inst_name,
                        model=inst_model,
                    )
                    continue

                result = json.loads(raw_content)
                total_ms = round(body.get("total_duration", 0) / 1e6)
                log.info(
                    "ollama_query_success",
                    instance=inst_name,
                    model=inst_model,
                    total_ms=total_ms,
                    eval_count=body.get("eval_count", 0),
                )
                return result
            except Exception as exc:
                log.warning(
                    "ollama_instance_failed",
                    instance=inst_name,
                    url=inst_url,
                    model=inst_model,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                    position=idx + 1,
                    total=len(_OLLAMA_INSTANCES),
                )
                continue

        log.warning(
            "ollama_all_instances_exhausted",
            instances_tried=len(_OLLAMA_INSTANCES),
        )
        return None

    async def _call_epoch_cloud(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Call epoch advice cloud provider with fallback.

        Primary: Cerebras (fast, 1M TPD). Fallback: Groq (14,400 RPD, 500K TPD).
        Combined: ~1.5M tokens/day. Fail-open: returns None if both fail.
        """
        total_providers = len(_EPOCH_PROVIDER_CONFIGS)
        for idx, cfg in enumerate(_EPOCH_PROVIDER_CONFIGS):
            if not cfg.get("api_key"):
                log.debug("epoch_cloud_skipped_no_key", provider=cfg.get("provider"))
                continue
            if _is_provider_blocked(cfg["provider"]):
                log.info(
                    "epoch_cloud_skipped_blocked",
                    provider=cfg["provider"],
                    position=idx + 1,
                    total=total_providers,
                )
                continue

            if idx > 0:
                log.info(
                    "epoch_cloud_trying_fallback",
                    provider=cfg["provider"],
                    model=cfg["model"],
                    position=idx + 1,
                    total=total_providers,
                )

            result = await self._call_cloud_api(
                user_content,
                schema,
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                timeout=cfg["timeout"],
                provider=cfg["provider"],
                system_prompt=system_prompt,
                max_tokens=cfg.get("max_tokens", 32768),
            )
            if result is not None:
                return result
            log.warning(
                "epoch_cloud_provider_failed",
                provider=cfg["provider"],
                model=cfg["model"],
                position=idx + 1,
                total=total_providers,
            )

        log.warning(
            "epoch_cloud_all_providers_exhausted",
            providers_tried=total_providers,
        )
        return None

    async def _call_query_lm(
        self,
        user_content: str,
        schema: dict[str, Any],
        *,
        system_prompt: str | None = None,
        timeout: float = 90.0,
    ) -> dict[str, Any] | None:
        """Route query to configured provider (Ollama or OpenRouter).

        Uses ``_QUERY_PROVIDER`` config to decide routing. Switch between
        providers by changing ``memory_agent.query_provider`` in config and
        restarting the memory service.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output enforcement.
            system_prompt: Override system prompt.
            timeout: Request timeout in seconds (applies to Ollama; OpenRouter
                uses its own configured timeout).

        Returns:
            Parsed JSON dict on success, None on any error.
        """
        if _QUERY_PROVIDER == "ollama":
            return await self._call_ollama(
                user_content,
                schema,
                system_prompt=system_prompt,
            )
        return await self._call_lm(
            user_content,
            schema,
            system_prompt=system_prompt,
        )
