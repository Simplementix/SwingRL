"""SwingRL Pydantic v2 config schema.

Load validated config from a YAML file with SWINGRL_* env var overrides:

    from swingrl.config.schema import load_config
    config = load_config("config/swingrl.yaml")

Environment variable overrides (for Docker .env):
    SWINGRL_TRADING_MODE=live              (top-level field)
    SWINGRL_EQUITY__MAX_POSITION_SIZE=0.3  (nested field, double-underscore)

Raises pydantic.ValidationError on any invalid field value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from swingrl.utils.exceptions import ConfigError


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Load settings from a YAML file. Returns empty dict if file does not exist."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path) -> None:
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field: Any, field_name: str) -> Any:  # noqa: ANN401
        """Return None — values are loaded in bulk via __call__."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Load YAML file and return settings dict. Empty dict if file missing."""
        if not self._yaml_path.exists():
            return {}
        with self._yaml_path.open() as f:
            return yaml.safe_load(f) or {}


class EquityConfig(BaseModel):
    """Equity environment configuration."""

    symbols: list[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"]
    )
    hmm_proxy_symbol: str = Field(default="SPY")
    max_position_size: float = Field(default=0.25, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.10, gt=0.0, lt=1.0)
    daily_loss_limit_pct: float = Field(default=0.02, gt=0.0, lt=1.0)
    min_order_usd: float = Field(default=1.0, ge=1.0)  # Alpaca $1 floor for fractional shares

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate equity symbols list is non-empty."""
        if not v:
            raise ConfigError("equity.symbols must not be empty")
        return v

    @model_validator(mode="after")
    def daily_loss_below_drawdown(self) -> EquityConfig:
        """daily_loss_limit_pct must be less than max_drawdown_pct.

        The daily loss limit is a per-day circuit breaker; the drawdown limit is the
        total account DD ceiling. Allowing daily_loss >= max_drawdown would mean a
        single bad day could blow through the full drawdown gate — defeat in purpose.
        """
        if self.daily_loss_limit_pct >= self.max_drawdown_pct:
            raise ConfigError(
                f"equity.daily_loss_limit_pct ({self.daily_loss_limit_pct}) must be "
                f"less than equity.max_drawdown_pct ({self.max_drawdown_pct})"
            )
        return self

    @model_validator(mode="after")
    def hmm_proxy_in_symbols(self) -> EquityConfig:
        """hmm_proxy_symbol must be present in symbols list."""
        if self.hmm_proxy_symbol not in self.symbols:
            raise ConfigError(
                f"equity.hmm_proxy_symbol '{self.hmm_proxy_symbol}' "
                f"must be in equity.symbols {self.symbols}"
            )
        return self


class CryptoConfig(BaseModel):
    """Crypto environment configuration."""

    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    hmm_proxy_symbol: str = Field(default="BTCUSDT")
    max_position_size: float = Field(default=0.50, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.12, gt=0.0, lt=1.0)
    daily_loss_limit_pct: float = Field(default=0.03, gt=0.0, lt=1.0)
    min_order_usd: float = Field(default=10.0, ge=10.0)  # Binance.US $10 floor

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate crypto symbols list is non-empty."""
        if not v:
            raise ConfigError("crypto.symbols must not be empty")
        return v

    @model_validator(mode="after")
    def daily_loss_below_drawdown(self) -> CryptoConfig:
        """daily_loss_limit_pct must be less than max_drawdown_pct (same logic as equity)."""
        if self.daily_loss_limit_pct >= self.max_drawdown_pct:
            raise ConfigError(
                f"crypto.daily_loss_limit_pct ({self.daily_loss_limit_pct}) must be "
                f"less than crypto.max_drawdown_pct ({self.max_drawdown_pct})"
            )
        return self

    @model_validator(mode="after")
    def hmm_proxy_in_symbols(self) -> CryptoConfig:
        """hmm_proxy_symbol must be present in symbols list."""
        if self.hmm_proxy_symbol not in self.symbols:
            raise ConfigError(
                f"crypto.hmm_proxy_symbol '{self.hmm_proxy_symbol}' "
                f"must be in crypto.symbols {self.symbols}"
            )
        return self


class CapitalConfig(BaseModel):
    """Capital allocation by environment."""

    equity_usd: float = Field(default=400.0, gt=0.0)
    crypto_usd: float = Field(default=47.0, gt=0.0)


class PathsConfig(BaseModel):
    """Filesystem paths (relative to repo root)."""

    data_dir: str = Field(default="data/")
    db_dir: str = Field(default="db/")
    models_dir: str = Field(default="models/")
    logs_dir: str = Field(default="logs/")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_logs: bool = Field(default=False)  # True in production/Docker


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    # HMM regime detection
    equity_hmm_window: int = Field(default=1260, ge=100)
    crypto_hmm_window: int = Field(default=2000, ge=100)
    hmm_n_iter: int = Field(default=200, ge=10)
    hmm_n_inits: int = Field(default=5, ge=1)
    hmm_ridge: float = Field(default=1e-6, gt=0.0)

    # Normalization
    equity_zscore_window: int = Field(default=252, ge=50)
    crypto_zscore_window: int = Field(default=360, ge=50)

    # Correlation pruning
    correlation_threshold: float = Field(default=0.85, gt=0.0, le=1.0)

    # Z-score normalization epsilon (prevents division by zero)
    zscore_epsilon: float = Field(default=1e-8, gt=0.0)

    # Turbulence
    equity_turbulence_warmup: int = Field(default=252, ge=50)
    equity_turbulence_half_life: int = Field(default=126, ge=10)
    crypto_turbulence_window: int = Field(default=1080, ge=100)
    crypto_turbulence_warmup: int = Field(default=360, ge=50)


class EnvironmentConfig(BaseModel):
    """RL environment configuration for training."""

    initial_amount: float = Field(default=100_000.0, gt=0.0)
    equity_episode_bars: int = Field(default=252, ge=50)
    crypto_episode_bars: int = Field(default=540, ge=50)
    equity_transaction_cost_pct: float = Field(default=0.0006, ge=0.0)
    crypto_transaction_cost_pct: float = Field(default=0.0022, ge=0.0)
    signal_deadzone: float = Field(default=0.02, ge=0.0, le=0.1)
    position_penalty_coeff: float = Field(default=10.0, ge=0.0)
    drawdown_penalty_coeff: float = Field(default=5.0, ge=0.0)


class SystemConfig(BaseModel):
    """System-level database configuration."""

    database_url: str = Field(default="")  # Set via DATABASE_URL env var


class AlertingConfig(BaseModel):
    """Alert rate-limiting, threshold, and webhook configuration."""

    alert_cooldown_minutes: int = Field(default=30, ge=1)
    consecutive_failures_before_alert: int = Field(default=3, ge=1)
    alerts_webhook_url: str = Field(default="")
    daily_webhook_url: str = Field(default="")
    healthchecks_equity_url: str = Field(default="")
    healthchecks_crypto_url: str = Field(default="")


class SchedulerConfig(BaseModel):
    """APScheduler configuration."""

    enabled: bool = Field(default=True)
    apscheduler_db_path: str = Field(default="db/apscheduler_jobs.sqlite")
    misfire_grace_time: int = Field(default=300, ge=60)
    max_workers: int = Field(default=4, ge=1)


class BackupConfig(BaseModel):
    """Backup and retention configuration."""

    backup_retention_days: int = Field(default=14, ge=1)
    backup_dir: str = Field(default="backups/")
    offsite_host: str = Field(default="")
    offsite_path: str = Field(default="")


class OptionsSnapshotConfig(BaseModel):
    """One scheduled snapshot: label, the market moment it represents, the pull time,
    and its misfire grace (D8/D9 — pull time trails market time on a delayed feed)."""

    label: str = Field(default="decision")
    market_time_et: str = Field(default="15:45")  # the moment the data represents
    pull_time_et: str = Field(default="16:00")  # when the cron fires (delay-adjusted)
    misfire_grace_s: int = Field(default=900, gt=0)

    @field_validator("label")
    @classmethod
    def label_known(cls, v: str) -> str:
        """Recognized snapshot labels (config-driven; add times via YAML, not code)."""
        allowed = {"open", "decision", "close", "eod"}
        if v not in allowed:
            raise ConfigError(f"options snapshot label must be one of {sorted(allowed)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def pull_not_before_market(self) -> OptionsSnapshotConfig:
        """A delayed feed can never show the market time before it happened."""
        if self.pull_time_et < self.market_time_et:  # HH:MM strings compare lexically
            raise ConfigError(
                f"snapshot {self.label!r}: pull_time_et {self.pull_time_et} precedes "
                f"market_time_et {self.market_time_et}"
            )
        return self


class OptionsIntegrityConfig(BaseModel):
    """Silent-corruption guards + audit schedule (spec §10.5/§10.6, §17 C1)."""

    # CBOE has no truncation flag; the partial-chain guard is a contract-count drop
    # vs the previous same-label snapshot (fraction; 0.5 = warn on a >50% drop).
    contract_count_drop_warn_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    audit_day_of_month: int = Field(default=1, ge=1, le=28)
    audit_time_et: str = Field(default="18:00")
    # Cron always fires a little late (millisecond-scale jitter is normal). Only warn on
    # the decision snapshot once lateness exceeds this tolerance (D8 lookahead guard).
    late_warn_s: float = Field(default=30.0, ge=0.0)


class OptionsBackupConfig(BaseModel):
    """Offsite 3-2-1 backup of the un-backfillable capture (spec §13)."""

    enabled: bool = Field(default=True)
    rclone_remote: str = Field(default="b2:swingrl-options")
    time_et: str = Field(default="02:30")


class OptionsCollectorConfig(BaseModel):
    """EOD option-chain collector configuration (spec §5 as amended §17)."""

    enabled: bool = Field(default=True)
    provider: str = Field(default="cboe")
    endpoint_url_template: str = Field(
        default="https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    )
    index_symbols: list[str] = Field(default_factory=lambda: ["_SPX"])
    include_equity_symbols: bool = Field(default=True)
    output_dir: str = Field(default="data/options_eod/cboe")
    schema_version: str = Field(default="v1")
    snapshots: list[OptionsSnapshotConfig] = Field(
        default_factory=lambda: [
            OptionsSnapshotConfig(
                label="decision", market_time_et="15:45", pull_time_et="16:00", misfire_grace_s=900
            ),
            OptionsSnapshotConfig(
                label="eod", market_time_et="16:15", pull_time_et="16:35", misfire_grace_s=18000
            ),
        ]
    )
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    rate_limit_per_sec: float = Field(default=1.0, gt=0.0)
    health_check_time_et: str = Field(default="17:15")
    health_lookback_days: int = Field(default=3, ge=1)  # D9 lookback window
    apscheduler_db_path: str = Field(default="db/apscheduler_options.sqlite")
    # Keep raw_json in Postgres JSONB too (bulky). Default on; revisit at first-run (decision D5).
    postgres_store_raw_json: bool = Field(default=True)
    integrity: OptionsIntegrityConfig = Field(default_factory=OptionsIntegrityConfig)
    backup: OptionsBackupConfig = Field(default_factory=OptionsBackupConfig)


class ShadowConfig(BaseModel):
    """Shadow model evaluation configuration."""

    equity_eval_days: int = Field(default=10, ge=5)
    crypto_eval_cycles: int = Field(default=30, ge=10)
    auto_promote: bool = Field(default=True)
    mdd_tolerance_ratio: float = Field(default=1.2, gt=1.0)


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration (optional, requires transformers)."""

    enabled: bool = Field(default=False)
    model_name: str = Field(default="ProsusAI/finbert")
    max_headlines_per_asset: int = Field(default=10, ge=1)
    finnhub_api_key: str = Field(default="")


class SecurityConfig(BaseModel):
    """Security and key management configuration."""

    key_rotation_days: int = Field(default=90, ge=30)
    env_file_permissions: str = Field(default="600")


class MemoryLiveEndpointsConfig(BaseModel):
    """Live memory agent endpoint toggles (all disabled by default)."""

    obs_enrichment: bool = False
    blend_weights: bool = False
    position_advice: bool = False
    trade_veto: bool = False
    cycle_gate: bool = False
    risk_thresholds: bool = False


class ConsolidationProviderConfig(BaseModel):
    """Single consolidation LLM provider."""

    base_url: str
    api_key: str = ""  # Override via env var (e.g. NVIDIA_API_KEY)
    default_model: str
    timeout_sec: float = 600.0  # Per-provider read timeout (generous default)
    max_tokens: int = 32768  # Per-provider max output tokens


class ConsolidationConfig(BaseModel):
    """Multi-provider consolidation configuration."""

    provider: str = "nvidia"  # Key into providers map
    model: str = ""  # Override per-provider default; empty = use provider's default_model
    timeout_sec: float = 120.0
    min_confidence_for_advice: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for patterns to be used in query advice.",
    )
    max_patterns_per_merge: int = Field(
        default=2,
        ge=1,
        description="Maximum patterns per dedup/merge LLM call (pairwise limit).",
    )
    providers: dict[str, ConsolidationProviderConfig] = Field(
        default_factory=lambda: {
            "mistral": ConsolidationProviderConfig(
                base_url="https://api.mistral.ai/v1",
                default_model="mistral-large-latest",
                timeout_sec=600,
                max_tokens=128000,
            ),
            "cerebras": ConsolidationProviderConfig(
                base_url="https://api.cerebras.ai/v1",
                default_model="qwen-3-235b-a22b-instruct-2507",
                timeout_sec=30,
                max_tokens=65536,
            ),
            "groq": ConsolidationProviderConfig(
                base_url="https://api.groq.com/openai/v1",
                default_model="meta-llama/llama-4-scout-17b-16e-instruct",
                timeout_sec=30,
                max_tokens=30000,
            ),
            "gemini": ConsolidationProviderConfig(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                default_model="gemini-2.5-flash",
                timeout_sec=60,
                max_tokens=65536,
            ),
            "openrouter": ConsolidationProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                default_model="nvidia/nemotron-3-super-120b-a12b:free",
                timeout_sec=1800,
                max_tokens=32768,
            ),
            "nvidia": ConsolidationProviderConfig(
                base_url="https://integrate.api.nvidia.com/v1",
                default_model="moonshotai/kimi-k2.5",
                timeout_sec=600,
                max_tokens=32768,
            ),
        }
    )

    @model_validator(mode="after")
    def resolve_provider_api_keys_from_env(self) -> ConsolidationConfig:
        """Fall back to {PROVIDER}_API_KEY env vars for empty api_key fields."""
        for provider_name, provider_cfg in self.providers.items():
            if not provider_cfg.api_key:
                env_key = f"{provider_name.upper()}_API_KEY"
                env_val = os.environ.get(env_key, "")
                if env_val:
                    provider_cfg.api_key = env_val
        return self


class OllamaInstanceConfig(BaseModel):
    """Single Ollama instance for the fallback chain."""

    name: str = "default"
    url: str = ""
    model: str = ""
    timeout: float = 30.0


class MemoryAgentConfig(BaseModel):
    """Memory agent (LLM meta-trainer) configuration.

    All fields default to disabled/safe values. Existing CI and paper trading
    are completely unaffected until memory_agent.enabled is set to true.
    """

    enabled: bool = False
    base_url: str = "http://swingrl-memory:8889"
    timeout_sec: float = 3.0
    blend_strength: float = 0.30
    meta_training: bool = False
    meta_training_timeout_sec: float = 300.0
    min_run_history_for_meta: int = 3
    llm_backend: str = "openrouter"
    openai_model: str = "gpt-4o-mini"
    cloud_fast_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    cloud_smart_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    query_provider: str = "gemini"  # HP tuning provider: "gemini", "openrouter", or "ollama"
    epoch_advice_provider: str = (
        "cerebras"  # Epoch advice: "cerebras" (fast cloud), "ollama" (local), or "openrouter"
    )
    cloud_block_on_429: bool = (
        True  # Skip cloud providers that returned 429 today (resets daily UTC)
    )
    cloud_block_codes: list[int] = Field(
        default_factory=lambda: [429]
    )  # HTTP codes triggering block
    ollama_url: str = "http://swingrl-ollama:11434"  # Legacy single-instance (backward compat)
    ollama_model: str = "qwen2.5:1.5b"  # Legacy single-instance (backward compat)
    ollama_instances: list[OllamaInstanceConfig] = Field(default_factory=list)

    # Per-algo epoch cadence (read from yaml by epoch_callback per fold).
    # Normalized so each algo makes ~4-17 calls/fold regardless of n_steps.
    epoch_cadence_ppo: int = 20  # ~82 rollouts/fold → ~4 calls
    epoch_cadence_a2c: int = 2000  # ~33K rollouts/fold → ~17 calls
    epoch_cadence_sac: int = 10000  # ~167K rollouts/fold → ~17 calls
    epoch_cadence_default: int = 100  # Fallback for unknown algos

    # Epoch aggregation parameters (used by consolidate.py _aggregate_epoch_summaries)
    outlier_iqr_mild: float = 1.5  # Tukey mild fence multiplier (Q1 - k*IQR, Q3 + k*IQR)
    outlier_iqr_extreme: float = 3.0  # Tukey extreme fence multiplier
    max_outlier_events: int = 3  # Max extreme events reported per metric per fold
    skewness_min_n: int = 8  # Min snapshots to compute skewness (unreliable below this)
    confidence_n_low: int = 5  # N <= this → "low" confidence label
    confidence_n_high: int = 15  # N >= this → "high" confidence label

    consolidate_interval_min: int = 30
    inbox_dir: str = "/data/memory_inbox"
    api_key: str = ""  # Populated from SWINGRL_MEMORY_AGENT__API_KEY env var; empty = no auth
    live_endpoints: MemoryLiveEndpointsConfig = Field(default_factory=MemoryLiveEndpointsConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)

    # Control folds: fold indices that skip reward adjustments (epoch advice disabled)
    # to serve as a scientific baseline for measuring reward shaping impact.
    # Empty list = all folds are treatment (backward compatible).
    control_folds_equity: list[int] = Field(default_factory=list)
    control_folds_crypto: list[int] = Field(default_factory=list)


class HyperparamBoundsConfig(BaseModel):
    """Hyperparameter bounds for LLM-suggested training config clamping."""

    learning_rate: tuple[float, float] = (1e-5, 1e-3)
    entropy_coeff: tuple[float, float] = (0.0, 0.05)
    clip_range: tuple[float, float] = (0.1, 0.4)
    n_epochs: tuple[int, int] = (3, 20)
    batch_size: tuple[int, int] = (32, 512)
    gamma: tuple[float, float] = (0.95, 0.995)
    target_kl: tuple[float, float] = (0.01, 0.05)
    gae_lambda: tuple[float, float] = (0.85, 1.0)
    gradient_steps: tuple[int, int] = (1, 8)
    target_entropy: tuple[float, float] = (-9.0, -0.5)


class RewardBoundsConfig(BaseModel):
    """Reward weight bounds for LLM-suggested reward weight clamping."""

    profit: tuple[float, float] = (0.10, 0.70)
    sharpe: tuple[float, float] = (0.10, 0.60)
    drawdown: tuple[float, float] = (0.05, 0.50)
    turnover: tuple[float, float] = (0.00, 0.20)


class TrainingBoundsConfig(BaseModel):
    """Combined bounds config for hyperparameters, reward weights, and the L1 lever."""

    hyperparam_bounds: HyperparamBoundsConfig = Field(default_factory=HyperparamBoundsConfig)
    reward_bounds: RewardBoundsConfig = Field(default_factory=RewardBoundsConfig)
    max_reward_delta: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "ppo": {"equity": 0.0, "crypto": 0.0},
            "a2c": {"equity": 0.0, "crypto": 0.0},
            "sac": {"equity": 0.0, "crypto": 0.0},
        },
        description=(
            "D-T2.1: L1 lever (mid-fold reward-weight nudges from epoch advice) is BENCHED. "
            "Shipped default is 0.0 (disabled) for every algo/env pair. Do not raise any "
            "value above 0.0 without a passing lever-verification harness run — see "
            "spec Section 2.3 (.planning/research/algo-reward-shaping.md for prior research). "
            "Override via SWINGRL_TRAINING__BOUNDS__MAX_REWARD_DELTA."
        ),
    )
    adjustment_cooldown_steps: dict[str, int] = Field(
        default_factory=lambda: {"ppo": 24_576, "a2c": 500, "sac": 20_000},
        description=(
            "Minimum timesteps between successive L1 reward-weight adjustments per algo. "
            "PPO: 2 rollouts (n_steps=2048, n_envs=6) for value function recovery. "
            "A2C: ~100 short rollouts for stability window. SAC: replay buffer rotation. "
            "Override via SWINGRL_TRAINING__BOUNDS__ADJUSTMENT_COOLDOWN_STEPS."
        ),
    )


class TrainingWindowsConfig(BaseModel):
    """Percent-of-fold rolling window sizes for reward-wrapper diagnostics (spec §2.6).

    Replaces the fixed 500-step deque used for window MDD with two windows sized as
    fractions of the fold's ACTUAL total_timesteps (escalated runs resize correctly --
    MemoryVecRewardWrapper.configure_windows() is called once per fold from
    model._total_timesteps, not from these fractions directly).
    """

    short_pct_of_fold: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description=(
            "Acute-detector window (N1): fraction of the fold's total_timesteps. "
            "Override via SWINGRL_TRAINING__WINDOWS__SHORT_PCT_OF_FOLD."
        ),
    )
    trend_pct_of_fold: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description=(
            "Decision-basis window (N2): fraction of the fold's total_timesteps. Must "
            "cover at least one full adjustment-cooldown cycle for every algo -- enforced "
            "by a startup guard (MemoryEpochCallback._on_training_start), not by this "
            "schema. Override via SWINGRL_TRAINING__WINDOWS__TREND_PCT_OF_FOLD."
        ),
    )


class NotableEventsConfig(BaseModel):
    """Notable-event trigger thresholds for mid-fold epoch capture (spec §4.10, D-T3.19).

    Replaces the retired NOTABLE_KL_THRESHOLD / NOTABLE_MDD_THRESHOLD module constants
    in epoch_callback.py -- the MDD threshold was calibrated against a cumsum-of-shaped-
    rewards quantity and became quasi-permanently true for crypto SAC (F2 root cause).
    All five triggers are evaluated against Task 5's short (acute-detector) window
    (window_metrics("short")) plus the current epoch's approx_kl/mean_reward.
    """

    kl_spike_threshold: float = Field(
        default=0.10,
        gt=0.0,
        description=(
            "approx_kl above this fires kl_spike. Unchanged value from the retired "
            "NOTABLE_KL_THRESHOLD constant (well-defined, rare). Override via "
            "SWINGRL_TRAINING__NOTABLE_EVENTS__KL_SPIKE_THRESHOLD."
        ),
    )
    mdd_breach_frac: dict[str, float] = Field(
        default_factory=lambda: {"equity": 0.10, "crypto": 0.12},
        description=(
            "Per-env equity-fraction drawdown ceiling for window_metrics('short')"
            "['mdd_frac_worst'] (the worst-sub-env basis -- never mdd_frac_mean) that "
            "fires mdd_breach. Sane units vs. per-env risk caps -- replaces the dead "
            "-25.0 cumsum threshold. Override via "
            "SWINGRL_TRAINING__NOTABLE_EVENTS__MDD_BREACH_FRAC (JSON dict)."
        ),
    )
    trade_shy_ratio: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "trade_rate below this fraction of the locked baseline_trade_rate fires "
            "trade_shy (mid-fold activity collapse). Override via "
            "SWINGRL_TRAINING__NOTABLE_EVENTS__TRADE_SHY_RATIO."
        ),
    )
    churning_ratio: float = Field(
        default=3.0,
        gt=1.0,
        description=(
            "trade_rate above this multiple of the locked baseline_trade_rate fires "
            "churning (the opposite disease). Override via "
            "SWINGRL_TRAINING__NOTABLE_EVENTS__CHURNING_RATIO."
        ),
    )
    hard_cap_per_run: int = Field(
        default=50,
        gt=0,
        description=(
            "Total event rows (cadence heartbeats excluded) allowed per training run "
            "before further event rows drop and a capture_alarm fires once (D-T3.19 "
            "three-layer bounding -- expected x10 headroom above the ~3-4x structural "
            "maximum of a healthy run). Override via "
            "SWINGRL_TRAINING__NOTABLE_EVENTS__HARD_CAP_PER_RUN."
        ),
    )


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    bounds: TrainingBoundsConfig = Field(default_factory=TrainingBoundsConfig)
    windows: TrainingWindowsConfig = Field(default_factory=TrainingWindowsConfig)
    notable_events: NotableEventsConfig = Field(default_factory=NotableEventsConfig)
    sac_buffer_size: int = Field(
        default=500_000,
        gt=0,
        description=(
            "SAC replay buffer size. Default 500K fits within 24GB swingrl container. "
            "Override via SWINGRL_TRAINING__SAC_BUFFER_SIZE. Proven 200K works on constrained RAM."
        ),
    )
    n_envs: int = Field(
        default=6,
        ge=1,
        description=(
            "Number of parallel environments for vectorized training. "
            "1 uses DummyVecEnv (sequential), >1 uses SubprocVecEnv (parallel). "
            "Default 6 balances parallelism with 3-algo parallel workers on 20-thread homelab."
        ),
    )
    vecenv_backend: Literal["dummy", "subproc"] = Field(
        default="subproc",
        description=(
            "VecEnv backend: 'dummy' for single-process sequential, "
            "'subproc' for multiprocess parallel. SubprocVecEnv requires picklable envs."
        ),
    )


class SwingRLConfig(BaseSettings):
    """Root SwingRL configuration.

    Load via load_config(path) — do not instantiate directly in business logic.
    """

    model_config = SettingsConfigDict(
        env_prefix="SWINGRL_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    trading_mode: Literal["paper", "live"] = Field(default="paper")
    equity: EquityConfig = Field(default_factory=EquityConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    options_collector: OptionsCollectorConfig = Field(default_factory=OptionsCollectorConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory_agent: MemoryAgentConfig = Field(default_factory=MemoryAgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: Path | str = "config/swingrl.yaml") -> SwingRLConfig:
    """Load and validate SwingRL configuration from a YAML file.

    Environment variables with SWINGRL_ prefix override YAML values.
    Use double-underscore for nested fields: SWINGRL_EQUITY__MAX_POSITION_SIZE=0.3

    Args:
        path: Path to the YAML config file. Defaults to config/swingrl.yaml.
              If the file does not exist, defaults are used.

    Returns:
        Validated SwingRLConfig instance.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    yaml_path = Path(path)

    class _ConfigWithYaml(SwingRLConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            """Priority: env vars > yaml file > model defaults."""
            return (env_settings, YamlConfigSettingsSource(settings_cls, yaml_path))

    return _ConfigWithYaml()
