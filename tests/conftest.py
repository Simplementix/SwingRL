"""SwingRL shared test fixtures.

Scope rules:
- session: read-only objects where repeated construction is expensive (repo_root)
- function (default): anything that writes files, mutates state, or returns mutable objects
"""

from __future__ import annotations

import textwrap
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import psycopg
import pytest
import structlog

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager
from swingrl.envs.equity import StockTradingEnv
from tests.db_guard import SAFE_DB_NAMES, classify_db_url, resolve_target_db_url
from tests.db_marker import is_db_test
from tests.db_worker import activate_isolated_db, drop_isolated_db

# Autouse fixture — imported so it registers globally (wipes test DB after each test).
from tests.fixtures.db_cleanup import close_cleanup_conn, wipe_db_after_test  # noqa: F401
from tests.fixtures.schema_preflight import schema_integrity_errors

# ---------------------------------------------------------------------------
# Database safety guard
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run the suite unless the *resolved* DB target is a safe, healthy test DB.

    Layer 2 (name): resolves the target the same way DatabaseManager does (env
    DATABASE_URL, then config.system.database_url), so a blank env var with a
    production URL in YAML config cannot slip past the guard.
    Layer 4 (shape): a cheap pg_constraint fingerprint refuses a poisoned scratch
    DB — any canonical table present without its PRIMARY KEY aborts the session
    with the table named (2026-07-22 incident cost three suite-hours without it).
    """
    # Per-worker/per-session DB isolation (tests/db_worker.py). Must run FIRST:
    # the name guard and schema preflight below validate the rewritten URL.
    try:
        activate_isolated_db()
    except (RuntimeError, psycopg.Error) as exc:
        pytest.exit(str(exc), returncode=2)

    db_url = resolve_target_db_url()
    verdict, db_name = classify_db_url(db_url)
    if verdict == "blank":
        return
    if verdict == "unparseable":
        pytest.exit(
            "Resolved database URL has no parseable database name; refusing to run pytest.",
            returncode=2,
        )
    if verdict != "safe":
        pytest.exit(
            f"REFUSING TO RUN: resolved database {db_name!r} is not a test database. "
            f"Test fixtures TRUNCATE/DELETE tables. Use a name in "
            f"{sorted(SAFE_DB_NAMES)} or one ending in '_test'. In CI, "
            f"scripts/ci-homelab.sh overrides DATABASE_URL automatically.",
            returncode=2,
        )
    try:
        errors = schema_integrity_errors(db_url)
    except (psycopg.OperationalError, psycopg.ProgrammingError) as exc:
        pytest.exit(
            f"Cannot run the schema preflight against {db_name!r}: {exc}",
            returncode=2,
        )
    if errors:
        pytest.exit(
            "SCHEMA PREFLIGHT FAILED — refusing to run against a poisoned test DB:\n  "
            + "\n  ".join(errors),
            returncode=2,
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-stamp the ``db`` marker on every DB-capable test (tests/db_marker.py).

    Explicit ``@pytest.mark.db`` is honored as-is (checked first). Everything the
    conditional wipe and the fast/db lanes do keys off this marker.
    """
    for item in items:
        if item.get_closest_marker("db") is not None:
            continue
        if is_db_test(getattr(item, "fixturenames", []), str(item.path)):
            item.add_marker(pytest.mark.db)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Drop this process's isolated DB clone and close the cleanup conn (best-effort)."""
    drop_isolated_db()
    close_cleanup_conn()


@pytest.fixture(autouse=True)
def _reset_db_singleton() -> None:
    """Reset DatabaseManager singleton after every test to prevent pool thread leaks."""
    yield
    DatabaseManager.reset()


@pytest.fixture(autouse=True)
def _isolate_structlog_config() -> Generator[None, None, None]:
    """Snapshot structlog's global config before each test and restore it after.

    structlog.configure() mutates process-global state and is NOT scoped to the
    caller. `structlog.configure(**kwargs)` only overrides the params it is given —
    any omitted param (e.g. `processors`) is left as whatever is currently
    installed, not reset to structlog's own default. This bites us for real:
    `services/memory/app.py` calls
    `structlog.configure(wrapper_class=..., logger_factory=structlog.PrintLoggerFactory())`
    with no `processors=` argument. If an earlier test in the same pytest process
    called `swingrl.utils.logging.configure_logging()` (which installs
    `structlog.stdlib.add_logger_name` as a processor), that processor is still
    active when the memory service's partial `configure()` runs — but the logger
    factory is now `PrintLoggerFactory`. `add_logger_name` does
    `event_dict["logger"] = logger.name`, and `PrintLogger` has no `.name`
    attribute, so every log call raises
    `AttributeError: 'PrintLogger' object has no attribute 'name'`.
    This only shows up when the whole suite runs in one process (test order
    dependent) — never in production, where the memory service is its own
    process and never inherits another module's structlog config. Restoring the
    exact pre-test config after every test removes the cross-test leak.
    """
    saved = structlog.get_config()
    try:
        yield
    finally:
        structlog.configure(**saved)


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
        environment:
          initial_amount: 100000.0
          equity_episode_bars: 252
          crypto_episode_bars: 540
          equity_transaction_cost_pct: 0.0006
          crypto_transaction_cost_pct: 0.0022
          signal_deadzone: 0.02
          position_penalty_coeff: 10.0
          drawdown_penalty_coeff: 5.0
        system:
          database_url: ""
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
def equity_env_config_yaml() -> str:
    """Config YAML with 8 equity symbols matching equity_prices_array (8 columns)."""
    return textwrap.dedent("""\
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]
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
        environment:
          initial_amount: 100000.0
          equity_episode_bars: 252
          crypto_episode_bars: 540
          equity_transaction_cost_pct: 0.0006
          crypto_transaction_cost_pct: 0.0022
          signal_deadzone: 0.02
          position_penalty_coeff: 10.0
          drawdown_penalty_coeff: 5.0
        system:
          database_url: ""
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def equity_env_config(tmp_path: Path, equity_env_config_yaml: str) -> SwingRLConfig:
    """SwingRLConfig with 8 equity symbols for environment tests."""
    config_file = tmp_path / "swingrl_env.yaml"
    config_file.write_text(equity_env_config_yaml)
    return load_config(config_file)


@pytest.fixture
def equity_features_array() -> np.ndarray:
    """Synthetic equity feature array for environment tests.

    Shape (300, 164) float32 — enough for 252-step episodes with buffer.
    """
    rng = np.random.default_rng(44)
    return rng.standard_normal((300, 164)).astype(np.float32)


@pytest.fixture
def crypto_features_array() -> np.ndarray:
    """Synthetic crypto feature array for environment tests.

    Shape (600, 47) float32 — enough for 540-step episodes with buffer.
    """
    rng = np.random.default_rng(45)
    return rng.standard_normal((600, 47)).astype(np.float32)


@pytest.fixture
def equity_prices_array() -> np.ndarray:
    """Synthetic equity price array for environment tests.

    Shape (300, 8) float32 with realistic cumulative-return price paths.
    """
    rng = np.random.default_rng(46)
    base_prices = np.array([470, 400, 220, 140, 110, 80, 40, 190], dtype=np.float32)
    returns = 1.0 + rng.normal(0.0002, 0.01, (300, 8))
    return (base_prices * np.cumprod(returns, axis=0)).astype(np.float32)


@pytest.fixture
def equity_env(
    equity_features_array: np.ndarray,
    equity_prices_array: np.ndarray,
    equity_env_config: SwingRLConfig,
) -> StockTradingEnv:
    """StockTradingEnv wired with synthetic features/prices/config, not yet reset."""
    return StockTradingEnv(
        features=equity_features_array,
        prices=equity_prices_array,
        config=equity_env_config,
    )


@pytest.fixture
def crypto_prices_array() -> np.ndarray:
    """Synthetic crypto price array for environment tests.

    Shape (600, 2) float32 with BTC/ETH-like cumulative-return price paths.
    """
    rng = np.random.default_rng(47)
    base_prices = np.array([42_000.0, 2_500.0], dtype=np.float32)
    returns = 1.0 + rng.normal(0.0001, 0.02, (600, 2))
    return (base_prices * np.cumprod(returns, axis=0)).astype(np.float32)


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


# ---------------------------------------------------------------------------
# Shared database mock helpers
# ---------------------------------------------------------------------------


def make_mock_db(
    fetchone_returns: list[Any] | None = None,
    fetchall_returns: list[Any] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create a standard mock DatabaseManager with connection() context manager.

    Returns (db_mock, conn_mock) so callers can further configure the connection.
    The connection() method is a context manager yielding conn_mock.
    """
    db = MagicMock(spec=["connection", "close", "init_schema", "reset"])
    conn = MagicMock()

    @contextmanager
    def _connection_ctx() -> Generator[MagicMock, None, None]:
        yield conn

    db.connection.side_effect = _connection_ctx

    if fetchone_returns is not None:
        conn.execute.return_value.fetchone.side_effect = fetchone_returns
    if fetchall_returns is not None:
        conn.execute.return_value.fetchall.side_effect = fetchall_returns

    return db, conn


def make_mock_db_multi(
    connection_configs: list[dict[str, Any]],
) -> MagicMock:
    """Create a mock DatabaseManager where each connection() call returns a different mock.

    Each item in connection_configs is a dict with optional keys:
      - "fetchone": return value for cursor.execute().fetchone()
      - "fetchone_side_effect": side_effect for fetchone()
      - "fetchall": return value for cursor.execute().fetchall()
    """
    db = MagicMock(spec=["connection", "close", "init_schema", "reset"])
    contexts = []

    for cfg in connection_configs:
        conn = MagicMock()
        if "fetchone" in cfg:
            conn.execute.return_value.fetchone.return_value = cfg["fetchone"]
        if "fetchone_side_effect" in cfg:
            conn.execute.return_value.fetchone.side_effect = cfg["fetchone_side_effect"]
        if "fetchall" in cfg:
            conn.execute.return_value.fetchall.return_value = cfg["fetchall"]

        @contextmanager
        def _ctx(c: MagicMock = conn) -> Generator[MagicMock, None, None]:
            yield c

        contexts.append(_ctx)

    db.connection.side_effect = contexts
    return db


@pytest.fixture()
def mock_alerter() -> MagicMock:
    """Mock Alerter with send_alert and send_embed methods."""
    alerter = MagicMock()
    alerter.send_alert = MagicMock()
    alerter.send_embed = MagicMock()
    return alerter
