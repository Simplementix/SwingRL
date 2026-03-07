"""SwingRL shared test fixtures.

Scope rules:
- session: read-only objects where repeated construction is expensive (repo_root)
- function (default): anything that writes files, mutates state, or returns mutable objects
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def valid_config_yaml() -> str:
    """Minimal valid swingrl.yaml content for testing."""
    return textwrap.dedent("""\
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
          duckdb_path: data/db/market_data.ddb
          sqlite_path: data/db/trading_ops.db
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def tmp_config(tmp_path: Path, valid_config_yaml: str) -> Path:
    """Write valid YAML to a temp file; return the Path.

    Function-scoped: each test gets a fresh file.
    """
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(valid_config_yaml)
    return config_file


@pytest.fixture
def loaded_config(tmp_config: Path) -> SwingRLConfig:
    """Return a validated SwingRLConfig loaded from tmp_config.

    Function-scoped: fresh config per test; do not mutate the returned object.
    """
    return load_config(tmp_config)


@pytest.fixture
def equity_ohlcv() -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame for equity tests.

    20 business-day rows, SPY-like prices (~470), UTC timezone.
    Columns: open, high, low, close, volume.
    """
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = 470.0 + rng.normal(0, 2, 20).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, 20),
            "high": close + rng.uniform(0, 2, 20),
            "low": close - rng.uniform(0, 2, 20),
            "close": close,
            "volume": rng.integers(50_000_000, 100_000_000, 20).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def crypto_ohlcv() -> pd.DataFrame:
    """Synthetic 4H OHLCV DataFrame for crypto tests.

    40 rows at 4H frequency, BTC-like prices (~42000), UTC timezone.
    Columns: open, high, low, close, volume (in BTC units).
    """
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    rng = np.random.default_rng(43)
    close = 42_000.0 + rng.normal(0, 200, 40).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 50, 40),
            "high": close + rng.uniform(0, 150, 40),
            "low": close - rng.uniform(0, 150, 40),
            "close": close,
            "volume": rng.uniform(500, 2000, 40),
        },
        index=dates,
    )


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create a standard SwingRL directory tree in tmp_path.

    Returns a dict mapping name to Path. All directories auto-cleaned by pytest.
    Keys: "data", "db", "models/active", "models/shadow", "models/archive", "logs"
    """
    dirs: dict[str, Path] = {}
    for name in ["data", "db", "models/active", "models/shadow", "models/archive", "logs"]:
        d = tmp_path / name
        d.mkdir(parents=True)
        dirs[name] = d
    return dirs
