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
