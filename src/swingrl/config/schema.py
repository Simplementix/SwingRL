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

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


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
    max_position_size: float = Field(default=0.25, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.10, gt=0.0, lt=1.0)
    daily_loss_limit_pct: float = Field(default=0.02, gt=0.0, lt=1.0)

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate equity symbols list is non-empty."""
        if not v:
            raise ValueError("equity.symbols must not be empty")
        return v

    @model_validator(mode="after")
    def daily_loss_below_drawdown(self) -> EquityConfig:
        """daily_loss_limit_pct must be less than max_drawdown_pct.

        The daily loss limit is a per-day circuit breaker; the drawdown limit is the
        total account DD ceiling. Allowing daily_loss >= max_drawdown would mean a
        single bad day could blow through the full drawdown gate — defeat in purpose.
        """
        if self.daily_loss_limit_pct >= self.max_drawdown_pct:
            raise ValueError(
                f"equity.daily_loss_limit_pct ({self.daily_loss_limit_pct}) must be "
                f"less than equity.max_drawdown_pct ({self.max_drawdown_pct})"
            )
        return self


class CryptoConfig(BaseModel):
    """Crypto environment configuration."""

    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    max_position_size: float = Field(default=0.50, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.12, gt=0.0, lt=1.0)
    daily_loss_limit_pct: float = Field(default=0.03, gt=0.0, lt=1.0)
    min_order_usd: float = Field(default=10.0, ge=10.0)  # Binance.US $10 floor

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        """Validate crypto symbols list is non-empty."""
        if not v:
            raise ValueError("crypto.symbols must not be empty")
        return v

    @model_validator(mode="after")
    def daily_loss_below_drawdown(self) -> CryptoConfig:
        """daily_loss_limit_pct must be less than max_drawdown_pct (same logic as equity)."""
        if self.daily_loss_limit_pct >= self.max_drawdown_pct:
            raise ValueError(
                f"crypto.daily_loss_limit_pct ({self.daily_loss_limit_pct}) must be "
                f"less than crypto.max_drawdown_pct ({self.max_drawdown_pct})"
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
