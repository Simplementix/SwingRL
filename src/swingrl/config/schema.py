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
    # Daily rebalance time (ET, HH:MM). 15:45 fires 15m before the 16:00 close so fills
    # land intraday, not post-close (review C2). The scheduler restricts it to weekdays.
    cycle_time_et: str = Field(default="15:45")
    # Gate the equity cycle on the Alpaca market clock (skip on closed/holiday). Fail-safe.
    market_calendar_gate: bool = Field(default=True)
    # Post-submit fill polling bounds (review C2): wait up to timeout, polling each interval,
    # before cancelling an unfilled order. Never a $0 trade.
    order_fill_timeout_s: int = Field(default=60, ge=1)
    order_poll_interval_s: int = Field(default=2, ge=1)

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate equity symbols list is non-empty."""
        if not v:
            raise ConfigError("equity.symbols must not be empty")
        return v

    @field_validator("cycle_time_et")
    @classmethod
    def cycle_time_is_hh_mm(cls, v: str) -> str:
        """Validate cycle_time_et parses as a 24h HH:MM clock time."""
        parts = v.split(":")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ConfigError(f"equity.cycle_time_et must be HH:MM, got {v!r}")
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ConfigError(f"equity.cycle_time_et out of range, got {v!r}")
        return v

    @model_validator(mode="after")
    def poll_interval_within_timeout(self) -> EquityConfig:
        """The poll interval must fit within the fill timeout (at least one poll)."""
        if self.order_poll_interval_s > self.order_fill_timeout_s:
            raise ConfigError(
                f"equity.order_poll_interval_s ({self.order_poll_interval_s}) must not exceed "
                f"equity.order_fill_timeout_s ({self.order_fill_timeout_s})"
            )
        return self

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
    # crypto_turbulence_window now governs the realized-vol percentile lookback
    # (the OR-gate history), not a rolling Mahalanobis window.
    crypto_turbulence_window: int = Field(default=1080, ge=100)
    crypto_turbulence_warmup: int = Field(default=1080, ge=50)
    crypto_turbulence_half_life: int = Field(default=750, ge=10)

    # Turbulence hard-halt baseline (F1 fix, method review 2026-07-07)
    turbulence_halt_percentile: float = Field(default=0.97, gt=0.0, lt=1.0)
    # Trailing window (in bars) the halt percentile is computed over. Equity =
    # ~3y of trading days; crypto = 0 (sentinel) meaning full history.
    turbulence_baseline_lookback_equity: int = Field(default=756, ge=50)
    turbulence_baseline_lookback_crypto: int = Field(default=0, ge=0)

    def turbulence_baseline_lookback_bars(self, env_name: str) -> int | None:
        """Trailing baseline window in bars for ``env_name`` (None = full history).

        Args:
            env_name: "equity" or "crypto".

        Returns:
            Number of trailing bars, or None when the configured value is 0
            (full-history baseline, used for crypto).
        """
        if env_name == "equity":
            return self.turbulence_baseline_lookback_equity
        value = self.turbulence_baseline_lookback_crypto
        return value if value > 0 else None


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
    zero_turbulence_obs: bool = Field(
        default=True,
        description=(
            "F1b: freeze the turbulence observation slot at 0.0 before inference. "
            "Era-0 models were trained with this slot frozen at 0.0, so feeding a real "
            "value would multiply it by untrained weights (pure noise). The real sensor "
            "value is still read out and kept for capture before the slot is zeroed. Flip "
            "to false once era-1 models (trained with a live turbulence input) deploy -- "
            "Plan B automates this off the models table. Override via "
            "SWINGRL_ENVIRONMENT__ZERO_TURBULENCE_OBS."
        ),
    )


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
    # Per-env misfire grace for the trading cycle jobs (A30 restart addendum). Equity 720s:
    # a restart shortly after 15:45 still runs the cycle late, but past the graced window it
    # cleanly skips (never a post-close submit). Crypto 3600s: an hour late is immaterial on
    # a 4H cadence. Missed-beyond-grace cycles are skipped, never replayed.
    misfire_grace_s: dict[str, int] = Field(default_factory=lambda: {"equity": 720, "crypto": 3600})
    max_workers: int = Field(default=4, ge=1)

    @field_validator("misfire_grace_s")
    @classmethod
    def misfire_grace_has_both_envs(cls, v: dict[str, int]) -> dict[str, int]:
        """Both cycle environments need a positive misfire grace."""
        for env in ("equity", "crypto"):
            if env not in v:
                raise ConfigError(f"scheduler.misfire_grace_s must include '{env}'")
            if v[env] <= 0:
                raise ConfigError(f"scheduler.misfire_grace_s['{env}'] must be positive")
        return v


class BackupConfig(BaseModel):
    """Backup and retention configuration."""

    backup_retention_days: int = Field(default=14, ge=1)
    backup_dir: str = Field(default="backups/")
    offsite_host: str = Field(default="")
    offsite_path: str = Field(default="")
    trader_backup_jobs_enabled: bool = Field(
        default=True,
        description=(
            "Register the trader's in-container backup jobs (daily_sqlite_backup, "
            "weekly_duckdb_backup, monthly_offsite). Default True preserves legacy "
            "behavior for deployments with a pg_dump binary; set False when backups are "
            "handled host-side (2026-07-19 ruling: trader image has no pg_dump)."
        ),
    )


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


class CandleJobsConfig(BaseModel):
    """Collector-scheduled OHLCV candle ingestion (equity daily + crypto 4H).

    Training used to keep candles fresh by running the ingest pipeline before every
    iteration; with training paused mid-redesign, the collector owns candle freshness so the
    paper trader never reads stale bars (USER RULING 2026-07-18). Both jobs call the EXISTING
    Alpaca/Binance ingestors — CBOE stays options-only.
    """

    enabled: bool = Field(default=True)
    # After the 16:35 EOD options snapshot and outside the 15:30–16:45 ET quiet window.
    equity_time_et: str = Field(default="16:50")  # HH:MM, Mon–Fri
    # Minute past each 4H UTC bar close (hours 0,4,8,12,16,20) — ahead of the trader's :05 cycles.
    crypto_minute: int = Field(default=1, ge=0, le=59)
    equity_misfire_grace_s: int = Field(default=21600, gt=0)
    crypto_misfire_grace_s: int = Field(default=10800, gt=0)


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
    candle_jobs: CandleJobsConfig = Field(default_factory=CandleJobsConfig)


class CalendarConfig(BaseModel):
    """Event-calendar ingest config (Plan A Task 11; spec §4 D-T3.14).

    Seeds ``calendar_events`` with macro release dates (FRED release/dates API) and FOMC
    meeting dates (forward schedule in ``fomc_dates`` yaml; historical dates in the
    ``fomc_backfill_csv`` seed). Windows are materialized at ingest from ``window_hours``.
    The weekly ingest + daily staleness jobs run in the swingrl-collector (amended
    2026-07-14), not the trader.
    """

    enabled: bool = Field(default=True)
    fred_api_base_url: str = Field(default="https://api.stlouisfed.org/fred")
    fred_release_ids: dict[str, int] = Field(
        default_factory=lambda: {"cpi": 10, "nfp": 50, "gdp": 53}
    )
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    release_fetch_limit: int = Field(default=30, ge=1)  # recent-mode desc limit
    fomc_dates: list[str] = Field(default_factory=list)  # forward ISO datetimes (ET)
    fomc_backfill_csv: str = Field(default="config/fomc_dates_historical.csv")
    window_hours: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "fomc": [24, 24],
            "cpi": [12, 12],
            "nfp": [12, 12],
            "gdp": [12, 12],
        }
    )
    min_future_days: int = Field(default=10, ge=1)
    backfill_start: str = Field(default="2015-01-01")  # covers min(ohlcv_daily)=2016-01-04
    # Collector scheduling (America/New_York); jobs register in scripts/collector_main.py.
    ingest_day_of_week: str = Field(default="sun")
    ingest_time_et: str = Field(default="06:30")
    staleness_check_time_et: str = Field(default="07:00")

    @field_validator("window_hours")
    @classmethod
    def windows_are_before_after_pairs(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        """Each event type maps to a [before_hours, after_hours] pair of non-negative ints."""
        for event_type, hours in v.items():
            if len(hours) != 2 or any(h < 0 for h in hours):
                raise ConfigError(
                    f"calendar.window_hours[{event_type!r}] must be [before, after] "
                    f"non-negative hours, got {hours!r}"
                )
        return v


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


class MetaTraderConfig(BaseModel):
    """Meta-Trader (trade-time coach) configuration — Task 12 rotation-gated skeleton.

    ``enabled`` is the runtime gate for the post-cycle MT commentary skeleton: when
    False (the default), the scheduler job is a provable no-op — it makes no memory
    call at all. Key rotation is complete (2026-07-07); Task 16's go/no-go is the
    remaining gate before this is switched on. Graders/verdicts land in Plan B, so
    day-one intents accumulate ungraded until then (documented, accepted).
    """

    enabled: bool = False
    commentary_provider: str = "cerebras"


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
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory_agent: MemoryAgentConfig = Field(default_factory=MemoryAgentConfig)
    meta_trader: MetaTraderConfig = Field(default_factory=MetaTraderConfig)
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
