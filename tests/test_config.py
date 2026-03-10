"""Config schema validation tests.

Tests the load_config() + SwingRLConfig roundtrip per ENV-11.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from swingrl.config.schema import load_config


def test_valid_config_loads(loaded_config: object) -> None:
    """ENV-11: Valid YAML must load and produce trading_mode='paper'."""
    from swingrl.config.schema import SwingRLConfig

    assert isinstance(loaded_config, SwingRLConfig)
    assert loaded_config.trading_mode == "paper"  # type: ignore[union-attr]


def test_invalid_trading_mode_raises(tmp_path: Path) -> None:
    """ENV-11: Invalid Literal value raises ValidationError mentioning field name."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("trading_mode: definitely_not_valid\n")
    with pytest.raises(ValidationError) as exc_info:
        load_config(bad_yaml)
    assert "trading_mode" in str(exc_info.value)


def test_invalid_position_size_raises(tmp_path: Path) -> None:
    """ENV-11: max_position_size <= 0.0 violates gt=0.0 constraint."""
    bad_yaml = tmp_path / "bad_pos.yaml"
    bad_yaml.write_text(
        textwrap.dedent("""\
            trading_mode: paper
            equity:
              max_position_size: -0.1
        """)
    )
    with pytest.raises(ValidationError) as exc_info:
        load_config(bad_yaml)
    assert "max_position_size" in str(exc_info.value)


def test_env_var_overrides_yaml(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-11: SWINGRL_TRADING_MODE env var overrides YAML trading_mode value."""
    monkeypatch.setenv("SWINGRL_TRADING_MODE", "live")
    config = load_config(tmp_config)
    assert config.trading_mode == "live"


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    """ENV-11: load_config() with nonexistent path uses model defaults (no FileNotFoundError)."""
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config.trading_mode == "paper"
    assert config.capital.equity_usd == 400.0


def test_model_validator_rejects_daily_loss_above_drawdown(tmp_path: Path) -> None:
    """ENV-11: @model_validator catches daily_loss_limit_pct >= max_drawdown_pct."""
    bad_yaml = tmp_path / "bad_cross.yaml"
    bad_yaml.write_text(
        textwrap.dedent("""\
            trading_mode: paper
            equity:
              daily_loss_limit_pct: 0.15
              max_drawdown_pct: 0.10
        """)
    )
    with pytest.raises(ValidationError) as exc_info:
        load_config(bad_yaml)
    assert "daily_loss_limit_pct" in str(exc_info.value)


# ── Phase 10 Config Extensions ──────────────────────────────────────────────


class TestBackupConfig:
    """HARD-02: BackupConfig sub-model validation."""

    def test_backup_defaults_load(self, tmp_config: Path) -> None:
        """HARD-02: BackupConfig loads with sensible defaults."""
        config = load_config(tmp_config)
        assert config.backup.sqlite_retention_days == 14
        assert config.backup.duckdb_rotate is False
        assert config.backup.backup_dir == "backups/"
        assert config.backup.offsite_host == ""
        assert config.backup.offsite_path == ""

    def test_backup_retention_days_minimum(self, tmp_path: Path) -> None:
        """HARD-02: sqlite_retention_days < 1 raises ValidationError."""
        bad_yaml = tmp_path / "bad_backup.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                backup:
                  sqlite_retention_days: 0
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "sqlite_retention_days" in str(exc_info.value)

    def test_backup_negative_retention_raises(self, tmp_path: Path) -> None:
        """HARD-02: Negative retention days raise ValidationError."""
        bad_yaml = tmp_path / "bad_backup_neg.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                backup:
                  sqlite_retention_days: -5
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "sqlite_retention_days" in str(exc_info.value)


class TestShadowConfig:
    """HARD-02: ShadowConfig sub-model validation."""

    def test_shadow_defaults_load(self, tmp_config: Path) -> None:
        """HARD-02: ShadowConfig loads with sensible defaults."""
        config = load_config(tmp_config)
        assert config.shadow.equity_eval_days == 10
        assert config.shadow.crypto_eval_cycles == 30
        assert config.shadow.auto_promote is True
        assert config.shadow.mdd_tolerance_ratio == 1.2

    def test_shadow_equity_eval_days_minimum(self, tmp_path: Path) -> None:
        """HARD-02: equity_eval_days < 5 raises ValidationError."""
        bad_yaml = tmp_path / "bad_shadow.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                shadow:
                  equity_eval_days: 3
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "equity_eval_days" in str(exc_info.value)

    def test_shadow_crypto_eval_cycles_minimum(self, tmp_path: Path) -> None:
        """HARD-02: crypto_eval_cycles < 10 raises ValidationError."""
        bad_yaml = tmp_path / "bad_shadow_crypto.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                shadow:
                  crypto_eval_cycles: 5
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "crypto_eval_cycles" in str(exc_info.value)

    def test_shadow_mdd_tolerance_must_exceed_one(self, tmp_path: Path) -> None:
        """HARD-02: mdd_tolerance_ratio <= 1.0 raises ValidationError."""
        bad_yaml = tmp_path / "bad_shadow_mdd.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                shadow:
                  mdd_tolerance_ratio: 1.0
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "mdd_tolerance_ratio" in str(exc_info.value)


class TestSentimentConfig:
    """HARD-05: SentimentConfig sub-model validation."""

    def test_sentiment_defaults_load(self, tmp_config: Path) -> None:
        """HARD-05: SentimentConfig loads with defaults (disabled)."""
        config = load_config(tmp_config)
        assert config.sentiment.enabled is False
        assert config.sentiment.model_name == "ProsusAI/finbert"
        assert config.sentiment.max_headlines_per_asset == 10
        assert config.sentiment.finnhub_api_key == ""

    def test_sentiment_max_headlines_minimum(self, tmp_path: Path) -> None:
        """HARD-05: max_headlines_per_asset < 1 raises ValidationError."""
        bad_yaml = tmp_path / "bad_sentiment.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                sentiment:
                  max_headlines_per_asset: 0
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "max_headlines_per_asset" in str(exc_info.value)


class TestSecurityConfig:
    """HARD-05: SecurityConfig sub-model validation."""

    def test_security_defaults_load(self, tmp_config: Path) -> None:
        """HARD-05: SecurityConfig loads with defaults."""
        config = load_config(tmp_config)
        assert config.security.key_rotation_days == 90
        assert config.security.env_file_permissions == "600"

    def test_security_key_rotation_minimum(self, tmp_path: Path) -> None:
        """HARD-05: key_rotation_days < 30 raises ValidationError."""
        bad_yaml = tmp_path / "bad_security.yaml"
        bad_yaml.write_text(
            textwrap.dedent("""\
                trading_mode: paper
                security:
                  key_rotation_days: 15
            """)
        )
        with pytest.raises(ValidationError) as exc_info:
            load_config(bad_yaml)
        assert "key_rotation_days" in str(exc_info.value)


def test_all_new_sections_load_from_yaml(tmp_path: Path) -> None:
    """HARD-02/05: Full YAML with all new sections loads and validates."""
    full_yaml = tmp_path / "full.yaml"
    full_yaml.write_text(
        textwrap.dedent("""\
            trading_mode: paper
            backup:
              sqlite_retention_days: 7
              duckdb_rotate: true
              backup_dir: /app/backups/
            shadow:
              equity_eval_days: 15
              crypto_eval_cycles: 50
              auto_promote: false
              mdd_tolerance_ratio: 1.5
            sentiment:
              enabled: true
              model_name: ProsusAI/finbert
              max_headlines_per_asset: 20
              finnhub_api_key: test_key
            security:
              key_rotation_days: 60
              env_file_permissions: "600"
        """)
    )
    config = load_config(full_yaml)
    assert config.backup.sqlite_retention_days == 7
    assert config.backup.duckdb_rotate is True
    assert config.shadow.equity_eval_days == 15
    assert config.shadow.auto_promote is False
    assert config.sentiment.enabled is True
    assert config.sentiment.max_headlines_per_asset == 20
    assert config.security.key_rotation_days == 60
