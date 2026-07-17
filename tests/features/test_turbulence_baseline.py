"""Tests for the F1 turbulence halt-baseline plumbing (Plan A Task 6c).

``FeaturePipeline.turbulence_halt_baseline`` computes the hard-halt percentile
of the turbulence series from the OHLCV-derived returns via ``compute_series``.
The ``features_*`` tables never had a turbulence column (the F1 bug), so the
old ExecutionPipeline percentile query always returned 0.0 and the halt never
fired. These tests exercise the real OHLCV-backed path on a seeded test DB.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager
from swingrl.features.pipeline import FeaturePipeline

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")

_SYMBOLS = ["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"]


@pytest.fixture
def baseline_config(tmp_path: Path) -> SwingRLConfig:
    """Config with 8 equity symbols and the test DATABASE_URL wired in."""
    db_url = os.environ["DATABASE_URL"]
    yaml_content = f"""\
trading_mode: paper
equity:
  symbols: [{", ".join(_SYMBOLS)}]
  max_position_size: 0.25
  max_drawdown_pct: 0.10
  daily_loss_limit_pct: 0.02
crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50
  max_drawdown_pct: 0.12
  daily_loss_limit_pct: 0.03
  min_order_usd: 10.0
capital:
  equity_usd: 400.0
  crypto_usd: 47.0
paths:
  data_dir: data/
  db_dir: db/
  models_dir: models/
  logs_dir: logs/
logging:
  level: INFO
  json_logs: false
system:
  database_url: "{db_url}"
alerting:
  alert_cooldown_minutes: 30
  consecutive_failures_before_alert: 3
"""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(yaml_content)
    return load_config(config_file)


@pytest.fixture
def feature_pipeline_with_ohlcv(
    baseline_config: SwingRLConfig,
    equity_prices_array: np.ndarray,
) -> FeaturePipeline:
    """FeaturePipeline over a test DB seeded with >= warmup+10 bars of closes.

    Reuses ``equity_prices_array`` (300 x 8 realistic cumulative-return paths)
    from tests/conftest.py so the turbulence series has finite post-warmup
    values for all 8 configured symbols.
    """
    DatabaseManager.reset()
    db = DatabaseManager(baseline_config)
    db.init_schema()

    n_bars = equity_prices_array.shape[0]  # 300 >= 252 warmup + 10
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_bars)]
    now = datetime.now(tz=UTC)
    with db.connection() as conn:
        for col, symbol in enumerate(_SYMBOLS):
            for i in range(n_bars):
                close = float(equity_prices_array[i, col])
                conn.execute(
                    "INSERT INTO ohlcv_daily "
                    "(symbol, date, open, high, low, close, volume, adjusted_close, fetched_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    [
                        symbol,
                        dates[i],
                        close - 0.5,
                        close + 1.0,
                        close - 1.0,
                        close,
                        1_000_000,
                        close,
                        now,
                    ],
                )
    yield FeaturePipeline(baseline_config, db)
    DatabaseManager.reset()


def test_turbulence_halt_baseline_nonzero_on_synthetic(
    feature_pipeline_with_ohlcv: FeaturePipeline,
) -> None:
    """F1: historical baseline computed from the OHLCV series, not a phantom column."""
    baseline = feature_pipeline_with_ohlcv.turbulence_halt_baseline("equity", "2026-07-07")
    assert baseline > 0.0


def test_turbulence_halt_baseline_zero_on_missing_data(
    feature_pipeline_with_ohlcv: FeaturePipeline,
) -> None:
    """Genuine data absence (query before any bars) yields 0.0, not a crash."""
    baseline = feature_pipeline_with_ohlcv.turbulence_halt_baseline("equity", "2000-01-01")
    assert baseline == 0.0
