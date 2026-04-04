"""Shared fixtures for feature engineering tests.

Provides larger OHLCV DataFrames with enough warmup bars for SMA-200
and crypto SMA-360, plus DuckDB in-memory connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config


@pytest.fixture
def equity_ohlcv_250() -> pd.DataFrame:
    """300-row daily equity OHLCV (enough for 200-bar SMA warmup + 100 computed rows).

    SPY-like prices (~470), UTC timezone, business-day frequency.
    """
    dates = pd.date_range("2023-01-02", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(44)
    close = 470.0 + rng.normal(0, 2, 300).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, 300),
            "high": close + rng.uniform(0, 2, 300),
            "low": close - rng.uniform(0, 2, 300),
            "close": close,
            "volume": rng.integers(50_000_000, 100_000_000, 300).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def crypto_ohlcv_400() -> pd.DataFrame:
    """450-row 4H crypto OHLCV (enough for 360-bar warmup + 90 computed rows).

    BTC-like prices (~42000), UTC timezone, 4H frequency.
    """
    dates = pd.date_range("2024-01-01", periods=450, freq="4h", tz="UTC")
    rng = np.random.default_rng(45)
    close = 42_000.0 + rng.normal(0, 200, 450).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, 450),
            "high": close + rng.uniform(0, 150, 450),
            "low": close - rng.uniform(0, 150, 450),
            "close": close,
            "volume": rng.uniform(500, 2000, 450),
        },
        index=dates,
    )


@pytest.fixture
def feature_config(tmp_path: Path) -> SwingRLConfig:
    """Loaded SwingRLConfig with features section using defaults."""
    yaml_content = """\
trading_mode: paper
equity:
  symbols: [SPY, QQQ]
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
  database_url: "postgresql://test:test@localhost:5432/swingrl_test"  # pragma: allowlist secret
  duckdb_path: data/db/market_data.ddb
  sqlite_path: data/db/trading_ops.db
alerting:
  alert_cooldown_minutes: 30
  consecutive_failures_before_alert: 3
features:
  equity_zscore_window: 252
  crypto_zscore_window: 360
"""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(yaml_content)
    return load_config(config_file)


@pytest.fixture
def pg_conn() -> Any:
    """PostgreSQL connection for integration tests.

    Requires DATABASE_URL env var. Skips if not available.
    """
    import os  # noqa: PLC0415

    import psycopg  # noqa: PLC0415
    from psycopg.rows import dict_row  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available for testing")

    conn = psycopg.connect(db_url, row_factory=dict_row)
    yield conn
    conn.close()
