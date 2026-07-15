# tests/test_options_config.py
from __future__ import annotations

import pytest

from swingrl.config.schema import (
    OptionsCollectorConfig,
    OptionsSnapshotConfig,
    load_config,
)


def test_options_collector_defaults_present() -> None:
    """OPT-CFG-1: OptionsCollectorConfig has spec §17 C1 defaults."""
    cfg = OptionsCollectorConfig()
    assert cfg.enabled is True
    assert cfg.provider == "cboe"
    assert cfg.endpoint_url_template == (
        "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    )
    assert cfg.index_symbols == ["_SPX"]
    assert cfg.include_equity_symbols is True
    assert cfg.output_dir == "data/options_eod/cboe"
    assert cfg.schema_version == "v1"
    assert cfg.apscheduler_db_path == "db/apscheduler_options.sqlite"
    assert cfg.postgres_store_raw_json is True
    assert cfg.health_lookback_days == 3


def test_options_snapshots_pull_vs_market_time() -> None:
    """OPT-CFG-2: decision pulls 16:00 for the 15:45 state; eod pulls 16:35 (D8)."""
    cfg = OptionsCollectorConfig()
    rows = [(s.label, s.market_time_et, s.pull_time_et, s.misfire_grace_s) for s in cfg.snapshots]
    assert rows == [("decision", "15:45", "16:00", 900), ("eod", "16:15", "16:35", 18000)]


def test_options_config_attached_to_root() -> None:
    """OPT-CFG-4: load_config exposes options_collector (spec §5)."""
    cfg = load_config("config/swingrl.yaml")
    assert cfg.options_collector.enabled is True
    assert cfg.options_collector.integrity.contract_count_drop_warn_frac == 0.5


def test_options_config_env_override() -> None:
    """OPT-CFG-5: nested env override works (spec §5)."""
    import os

    os.environ["SWINGRL_OPTIONS_COLLECTOR__ENABLED"] = "false"
    try:
        cfg = load_config("config/swingrl.yaml")
        assert cfg.options_collector.enabled is False
    finally:
        del os.environ["SWINGRL_OPTIONS_COLLECTOR__ENABLED"]


def test_snapshot_label_must_be_known() -> None:
    """OPT-CFG-6: snapshot label validated against the known set (spec §6.1, D4)."""
    from swingrl.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        OptionsSnapshotConfig(label="lunchtime", market_time_et="12:00", pull_time_et="12:15")


def test_snapshot_grace_positive_and_pull_not_before_market_time() -> None:
    """OPT-CFG-7: misfire grace > 0; pull_time_et >= market_time_et (delayed feed, D8)."""
    from swingrl.utils.exceptions import ConfigError

    with pytest.raises(ConfigError):
        OptionsSnapshotConfig(
            label="decision", market_time_et="15:45", pull_time_et="15:30", misfire_grace_s=900
        )
