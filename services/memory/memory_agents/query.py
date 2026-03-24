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

import json
import os
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
    insert_pattern_presentation,
    insert_pattern_presentation_async,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (loaded from mounted config YAML, fallback to hardcoded defaults)
# ---------------------------------------------------------------------------

_FALLBACK_HYPERPARAM_BOUNDS: dict[str, tuple[Any, Any]] = {
    "learning_rate": (1e-5, 1e-3),
    "entropy_coeff": (0.0, 0.05),
    "clip_range": (0.1, 0.4),
    "n_epochs": (3, 20),
    "batch_size": (32, 512),
    "gamma": (0.90, 0.9999),
}

_FALLBACK_REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}


def _load_bounds_from_config() -> tuple[dict[str, tuple[Any, Any]], dict[str, tuple[float, float]]]:
    """Load bounds from mounted swingrl.yaml, fallback to hardcoded defaults."""
    config_path = Path("/app/config/swingrl.yaml")
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
}

# ---------------------------------------------------------------------------
# Cloud API config (primary: OpenRouter, backup: NVIDIA NIM)
# ---------------------------------------------------------------------------


def _load_query_cloud_config() -> dict[str, Any]:
    """Load query agent cloud config with primary (OpenRouter) and backup (NVIDIA)."""
    config_path = Path("/app/config/swingrl.yaml")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        cons = mem.get("consolidation", {})
        providers = cons.get("providers", {})
        timeout = float(cons.get("timeout_sec", 1800))

        or_cfg = providers.get("openrouter", {})
        primary = {
            "base_url": or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            "api_key": os.environ.get("OPENROUTER_API_KEY", or_cfg.get("api_key", "")),
            "model": or_cfg.get(
                "default_model",
                "nvidia/nemotron-3-super-120b-a12b:free",
            ),
            "timeout": timeout,
            "provider": "openrouter",
        }

        nv_cfg = providers.get("nvidia", {})
        backup = {
            "base_url": nv_cfg.get("base_url", "https://integrate.api.nvidia.com/v1"),
            "api_key": os.environ.get("NVIDIA_API_KEY", nv_cfg.get("api_key", "")),
            "model": nv_cfg.get("default_model", "moonshotai/kimi-k2.5"),
            "timeout": timeout,
            "provider": "nvidia",
        }

        return {"primary": primary, "backup": backup}

    return {
        "primary": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "timeout": 60.0,
            "provider": "openrouter",
        },
        "backup": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": os.environ.get("NVIDIA_API_KEY", ""),
            "model": "moonshotai/kimi-k2.5",
            "timeout": 1800.0,
            "provider": "nvidia",
        },
    }


def _load_ollama_config() -> dict[str, str]:
    """Load Ollama config for local epoch advice routing.

    Returns dict with keys: query_provider, epoch_advice_provider, ollama_url, ollama_model.
    """
    config_path = Path("/app/config/swingrl.yaml")
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        mem = cfg.get("memory_agent", {})
        return {
            "query_provider": mem.get("query_provider", "openrouter"),
            "epoch_advice_provider": mem.get("epoch_advice_provider", "ollama"),
            "ollama_url": mem.get("ollama_url", "http://swingrl-ollama:11434"),
            "ollama_model": mem.get("ollama_model", "qwen3:1.7b"),
        }
    return {
        "query_provider": "openrouter",
        "epoch_advice_provider": "ollama",
        "ollama_url": "http://swingrl-ollama:11434",
        "ollama_model": "qwen3:1.7b",
    }


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
        "ollama_url": "",
        "ollama_model": "",
    }

_QUERY_PROVIDER: str = _OLLAMA_CONFIG["query_provider"]
_EPOCH_ADVICE_PROVIDER: str = _OLLAMA_CONFIG["epoch_advice_provider"]
_OLLAMA_URL: str = _OLLAMA_CONFIG["ollama_url"]
_OLLAMA_MODEL: str = _OLLAMA_CONFIG["ollama_model"]

log.info(
    "query_provider_configured",
    hp_tuning_provider=_QUERY_PROVIDER,
    epoch_advice_provider=_EPOCH_ADVICE_PROVIDER,
    ollama_url=_OLLAMA_URL,
    ollama_model=_OLLAMA_MODEL,
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
        "PPO Symptom → Fix mapping:\n"
        "- High overfit_gap (IS >> OOS): Reduce n_epochs (10->5), then reduce learning_rate\n"
        "- Exploding approx_kl: Reduce learning_rate\n"
        "- clip_fraction > 0.3: Reduce n_epochs or narrow clip_range\n"
        "- Policy collapse (entropy -> 0): Increase ent_coef"
    ),
    "a2c": (
        "A2C Hyperparameter Guide:\n"
        "CRITICAL: A2C has NO clipping protection (unlike PPO). Learning rate is the ONLY "
        "lever controlling update magnitude. A2C is fundamentally more sensitive to all HPs.\n\n"
        "- learning_rate: THE most sensitive A2C parameter. Without clipping, LR directly "
        "controls how far the policy moves per update. For overfitting: reduce learning_rate "
        "FIRST (this is A2C's primary lever, unlike PPO where n_epochs comes first). "
        "Reduce to 1e-4 to 3e-4 range for noisy financial data.\n"
        "- gamma: Lower to 0.95-0.97 to reduce the critic's prediction burden. "
        "Higher gamma needs lower LR for stability.\n"
        "- ent_coef: Even more important than PPO because A2C has no clipping to limit "
        "policy changes. Entropy bonus is the main regularizer. Keep at 0.01-0.02.\n\n"
        "A2C Symptom → Fix mapping:\n"
        "- High overfit_gap: Reduce learning_rate FIRST (A2C's only real lever)\n"
        "- Erratic policy (whipsawing trades): Reduce learning_rate\n"
        "- Policy collapse: Increase ent_coef\n"
        "- Myopic trading: Increase gamma (carefully — higher gamma needs lower LR)"
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
        "- ent_coef: 'auto' learns optimal entropy coefficient. Fixed 0.01 is more "
        "predictable. Default 'auto' starts at alpha=1.0 which can overwhelm small "
        "financial reward signals.\n\n"
        "SAC Symptom → Fix mapping:\n"
        "- High overfit_gap: Reduce learning_rate, consider smaller buffer_size\n"
        "- Q-values exploding: Reduce learning_rate\n"
        "- Over-exploration (random trades): Recommend fixed ent_coef=0.01 instead of 'auto'\n"
        "- Regime confusion: Reduce gamma (shorter horizon = less regime averaging)"
    ),
}

_REWARD_WEIGHT_GUIDE = (
    "Reward Weight Adjustment Guide:\n"
    "- High MDD / frequent large losses: Increase drawdown weight (0.2 -> 0.35)\n"
    "- Too many trades / high turnover: Increase turnover weight (0.05 -> 0.15)\n"
    "- Good Sharpe but low total return: Increase profit weight\n"
    "- Volatile returns: Increase sharpe weight (smooths returns)\n"
    "- Agent holds through drawdowns: Increase drawdown weight\n\n"
    "ONLY adjust reward weights when the current epoch metrics show a clear problem. "
    "If rolling_sharpe > 0 and rolling_mdd > -0.05, the current weights are working — keep them. "
    "Prefer small adjustments (delta < 0.1) over large swings."
)

_RUN_CONFIG_INSTRUCTIONS = (
    "For each hyperparameter you recommend changing from baseline:\n"
    "1. Name the specific pattern (by content, not ID) that supports the change\n"
    "2. Cite the specific metric from that pattern (e.g., 'overfit_gap >0.30 in 9/13 folds')\n"
    "3. Explain why that pattern implies this specific HP change\n\n"
    "If fewer than 2 patterns support HP changes, return mostly baseline values.\n"
    "It is better to return safe defaults with honest rationale than to over-tune."
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
        f"{_HALLUCINATION_GUARD}\n\n"
        f"{hp_guide}\n\n"
        f"{_REWARD_WEIGHT_GUIDE}"
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
    ],
    "epoch_advice": [
        "drawdown_recovery",
        "trade_quality",
        "overfit_diagnosis",
        "iteration_progression",
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
    """Parse env_name, algo_name, and iteration from query string."""
    env_name: str | None = None
    algo_name: str | None = None
    iteration: int | None = None
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
    return {"env_name": env_name, "algo_name": algo_name, "iteration": iteration}


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
    },
    "a2c": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "gamma": {"type": "number"},
    },
    "sac": {
        "learning_rate": {"type": "number"},
        "entropy_coeff": {"type": "number"},
        "batch_size": {"type": "integer"},
        "gamma": {"type": "number"},
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
    """Advises RL training configuration using memory patterns + cloud LLM."""

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
        result = await self._call_lm(user_content, schema, system_prompt=system_prompt)
        if result is None:
            log.warning("run_config_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_DEFAULTS)

        # Track which patterns were presented
        await self._track_presentations_async(presented_ids, query, "run_config", result)

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
        parsed = _parse_query_context(query)
        algo = parsed["algo_name"] or "ppo"
        context, presented_ids = await self._build_context_async(
            env_name=parsed["env_name"],
            algo_name=algo,
            request_type="epoch_advice",
        )
        user_content = (
            f"Current epoch state: {query}\n\n"
            f"Memory patterns:\n{context}\n\n"
            "Should I adjust reward weights or stop training early?"
        )

        epoch_prompt = _build_epoch_system_prompt(_HYPERPARAM_BOUNDS, _REWARD_BOUNDS, algo)
        # Epoch advice uses local Ollama (qwen3:1.7b) — fast, unlimited, 60s timeout
        result = await self._call_ollama(
            user_content,
            _EPOCH_ADVICE_SCHEMA,
            system_prompt=epoch_prompt,
            timeout=60.0,
        )
        if result is None:
            log.warning("epoch_advice_fallback_to_defaults", query=query[:100])
            return dict(_SAFE_EPOCH_DEFAULTS)

        # Track which patterns were presented
        await self._track_presentations_async(presented_ids, query, "epoch_advice", result)

        clamped_weights = _clamp_reward_weights(
            result.get("reward_weights", _SAFE_EPOCH_DEFAULTS["reward_weights"])
        )

        return {
            "reward_weights": clamped_weights,
            "stop_training": bool(result.get("stop_training", False)),
            "rationale": str(result.get("rationale", "llm_advised")),
        }

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
    ) -> dict[str, Any] | None:
        """Call cloud API (OpenAI-compatible) with structured JSON output.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output enforcement.
            base_url: Provider base URL.
            api_key: Bearer token for the provider.
            model: Model identifier string.
            timeout: Read timeout in seconds.
            provider: Provider name for logging ('openrouter' or 'nvidia').
            system_prompt: Override system prompt (defaults to module-level _SYSTEM_PROMPT).

        Returns:
            Parsed dict from LLM response, or None on failure.
        """
        effective_max_tokens = 32768
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
            result = await self._call_cloud_api(
                user_content,
                schema,
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                timeout=cfg["timeout"],
                provider=cfg["provider"],
                system_prompt=system_prompt,
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
        timeout: float = 90.0,
    ) -> dict[str, Any] | None:
        """Call local Ollama API with structured JSON output.

        Uses the Ollama /api/chat endpoint with ``format`` parameter for schema
        enforcement. Qwen3 thinking mode is disabled via /no_think prefix.

        Args:
            user_content: The user-turn message content.
            schema: JSON schema dict for structured output via format parameter.
            system_prompt: System prompt (defaults to module-level _SYSTEM_PROMPT).
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON dict on success, None on any error.
        """
        if not _OLLAMA_URL:
            log.debug("ollama_skipped_no_url")
            return None

        effective_system = system_prompt or _SYSTEM_PROMPT
        payload = {
            "model": _OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_content},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 512, "num_ctx": 8192},
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{_OLLAMA_URL}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()

            raw_content = body.get("message", {}).get("content")
            if not raw_content:
                log.warning("ollama_empty_content", model=_OLLAMA_MODEL)
                return None

            result = json.loads(raw_content)
            total_ms = round(body.get("total_duration", 0) / 1e6)
            log.info(
                "ollama_query_success",
                model=_OLLAMA_MODEL,
                total_ms=total_ms,
                eval_count=body.get("eval_count", 0),
            )
            return result
        except Exception as exc:
            log.warning(
                "ollama_query_failed",
                model=_OLLAMA_MODEL,
                error=str(exc),
                exc_type=type(exc).__name__,
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
                timeout=timeout,
            )
        return await self._call_lm(
            user_content,
            schema,
            system_prompt=system_prompt,
        )
