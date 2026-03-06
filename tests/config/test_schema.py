"""Tests for SwingRL Pydantic v2 config schema and load_config()."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError


class TestLoadConfigValid:
    def test_load_config_returns_swingrlconfig(self, tmp_path: Path) -> None:
        """ENV-11: load_config() returns SwingRLConfig on valid YAML."""
        from swingrl.config.schema import SwingRLConfig, load_config

        yaml_file = tmp_path / "swingrl.yaml"
        yaml_file.write_text("trading_mode: paper\n")
        config = load_config(yaml_file)
        assert isinstance(config, SwingRLConfig)
        assert config.trading_mode == "paper"

    def test_load_config_defaults_when_file_missing(self, tmp_path: Path) -> None:
        """ENV-11: load_config() returns defaults (no FileNotFoundError) when file absent."""
        from swingrl.config.schema import load_config

        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.trading_mode == "paper"

    def test_load_config_all_section_defaults(self, tmp_path: Path) -> None:
        """ENV-11: Default sections exist with expected types."""
        from swingrl.config.schema import (
            CapitalConfig,
            CryptoConfig,
            EquityConfig,
            LoggingConfig,
            PathsConfig,
            load_config,
        )

        config = load_config(tmp_path / "empty.yaml")
        assert isinstance(config.equity, EquityConfig)
        assert isinstance(config.crypto, CryptoConfig)
        assert isinstance(config.capital, CapitalConfig)
        assert isinstance(config.paths, PathsConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_equity_default_symbols(self, tmp_path: Path) -> None:
        """ENV-11: equity.symbols has 8 default ETF tickers."""
        from swingrl.config.schema import load_config

        config = load_config(tmp_path / "empty.yaml")
        assert config.equity.symbols == ["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"]

    def test_crypto_min_order_usd_default(self, tmp_path: Path) -> None:
        """ENV-11: crypto.min_order_usd defaults to 10.0 (Binance.US floor)."""
        from swingrl.config.schema import load_config

        config = load_config(tmp_path / "empty.yaml")
        assert config.crypto.min_order_usd == 10.0

    def test_capital_defaults(self, tmp_path: Path) -> None:
        """ENV-11: capital defaults to equity_usd=400.0, crypto_usd=47.0."""
        from swingrl.config.schema import load_config

        config = load_config(tmp_path / "empty.yaml")
        assert config.capital.equity_usd == 400.0
        assert config.capital.crypto_usd == 47.0

    def test_logging_json_logs_default_false(self, tmp_path: Path) -> None:
        """ENV-11: logging.json_logs defaults to False (dev mode)."""
        from swingrl.config.schema import load_config

        config = load_config(tmp_path / "empty.yaml")
        assert config.logging.json_logs is False
        assert config.logging.level == "INFO"


class TestLoadConfigValidationErrors:
    def test_invalid_trading_mode_raises_validation_error(self, tmp_path: Path) -> None:
        """ENV-11: invalid trading_mode raises ValidationError with field name in message."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("trading_mode: invalid_mode\n")
        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file)
        assert "trading_mode" in str(exc_info.value)

    def test_negative_position_size_raises_validation_error(self, tmp_path: Path) -> None:
        """ENV-11: gt=0.0 violated on max_position_size raises ValidationError."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                equity:
                  max_position_size: -0.1
            """)
        )
        with pytest.raises(ValidationError):
            load_config(yaml_file)

    def test_equity_cross_field_daily_loss_gte_drawdown_raises(self, tmp_path: Path) -> None:
        """ENV-11: equity.daily_loss_limit_pct >= max_drawdown_pct raises ValidationError."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                equity:
                  daily_loss_limit_pct: 0.15
                  max_drawdown_pct: 0.10
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file)
        assert "daily_loss_limit_pct" in str(exc_info.value)

    def test_crypto_cross_field_daily_loss_gte_drawdown_raises(self, tmp_path: Path) -> None:
        """ENV-11: crypto.daily_loss_limit_pct >= max_drawdown_pct raises ValidationError."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                crypto:
                  daily_loss_limit_pct: 0.15
                  max_drawdown_pct: 0.10
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(yaml_file)
        assert "daily_loss_limit_pct" in str(exc_info.value)

    def test_empty_equity_symbols_raises_validation_error(self, tmp_path: Path) -> None:
        """ENV-11: equity.symbols empty list raises ValidationError."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("equity:\n  symbols: []\n")
        with pytest.raises(ValidationError):
            load_config(yaml_file)

    def test_invalid_logging_level_raises_validation_error(self, tmp_path: Path) -> None:
        """ENV-11: invalid logging.level raises ValidationError."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("logging:\n  level: VERBOSE\n")
        with pytest.raises(ValidationError):
            load_config(yaml_file)

    def test_crypto_min_order_below_10_raises_validation_error(self, tmp_path: Path) -> None:
        """ENV-11: crypto.min_order_usd < 10.0 raises ValidationError (Binance.US floor)."""
        from swingrl.config.schema import load_config

        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("crypto:\n  min_order_usd: 5.0\n")
        with pytest.raises(ValidationError):
            load_config(yaml_file)


class TestEnvVarOverrides:
    def test_env_var_overrides_trading_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENV-11: SWINGRL_TRADING_MODE=live overrides YAML trading_mode: paper."""
        from swingrl.config.schema import load_config

        monkeypatch.setenv("SWINGRL_TRADING_MODE", "live")
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("trading_mode: paper\n")
        config = load_config(yaml_file)
        assert config.trading_mode == "live"

    def test_env_var_overrides_nested_equity_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENV-11: SWINGRL_EQUITY__MAX_POSITION_SIZE=0.99 overrides nested field."""
        from swingrl.config.schema import load_config

        monkeypatch.setenv("SWINGRL_EQUITY__MAX_POSITION_SIZE", "0.99")
        config = load_config(tmp_path / "empty.yaml")
        assert config.equity.max_position_size == pytest.approx(0.99)


class TestImportability:
    def test_all_classes_importable_from_swingrl_config(self) -> None:
        """ENV-11: SwingRLConfig and all sub-configs importable from swingrl.config."""
        from swingrl.config import (  # noqa: F401
            CapitalConfig,
            CryptoConfig,
            EquityConfig,
            LoggingConfig,
            PathsConfig,
            SwingRLConfig,
            load_config,
        )
