"""Tests for Phase 15: Training CLI observation assembly.

Validates that _load_features_prices produces correctly shaped observation
matrices using ObservationAssembler, and that --dry-run flag works correctly.

Requirements: TRAIN-01, TRAIN-02, TRAIN-03, VAL-01
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

# Add scripts/ to path so we can import train.py as a module
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from swingrl.config.schema import SwingRLConfig  # noqa: E402
from swingrl.features.assembler import CRYPTO_OBS_DIM, equity_obs_dim  # noqa: E402
from swingrl.features.schema import init_feature_schema  # noqa: E402

# EQUITY symbols matching equity_env_config fixture: 8 ETFs
_EQUITY_SYMBOLS = ["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"]
_CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Training dates for test data
_EQUITY_DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]
_CRYPTO_DATETIMES = [
    "2024-01-01 00:00:00",
    "2024-01-01 04:00:00",
    "2024-01-01 08:00:00",
]

_EQUITY_FEATURE_COLS = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "weekly_trend_dir",
    "weekly_rsi_14",
    "pe_zscore",
    "earnings_growth",
    "debt_to_equity",
    "dividend_yield",
]

_CRYPTO_FEATURE_COLS = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "daily_trend_dir",
    "daily_rsi_14",
    "four_h_rsi_14",
    "four_h_price_sma20_ratio",
]


def _make_equity_db(equity_symbols: list[str] | None = None) -> duckdb.DuckDBPyConnection:
    """Create in-memory DuckDB seeded with equity test data."""
    if equity_symbols is None:
        equity_symbols = _EQUITY_SYMBOLS

    conn = duckdb.connect(":memory:")
    init_feature_schema(conn)

    # Create ohlcv_daily
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)

    # Create macro_features
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value DOUBLE,
            PRIMARY KEY (series_id, date)
        )
    """)

    rng = np.random.default_rng(42)

    # Seed features_equity: 3 dates x 8 symbols
    for date in _EQUITY_DATES:
        for symbol in equity_symbols:
            feature_vals = rng.uniform(0.5, 1.5, len(_EQUITY_FEATURE_COLS)).tolist()
            col_str = ", ".join(_EQUITY_FEATURE_COLS)
            val_str = ", ".join(str(v) for v in feature_vals)
            conn.execute(
                f"INSERT OR REPLACE INTO features_equity (symbol, date, {col_str}) "  # nosec B608
                f"VALUES ('{symbol}', '{date}', {val_str})"  # nosec B608
            )

    # Seed ohlcv_daily: close prices for each symbol/date
    base_prices = {sym: rng.uniform(50.0, 500.0) for sym in equity_symbols}
    for date in _EQUITY_DATES:
        for symbol in equity_symbols:
            close = base_prices[symbol] * rng.uniform(0.99, 1.01)
            conn.execute(
                "INSERT OR REPLACE INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
                f"VALUES ('{symbol}', '{date}', {close}, {close}, {close}, {close}, 1000000.0)"
            )

    # Seed macro_features
    macro_series = {
        "VIXCLS": 18.5,
        "T10Y2Y": 0.5,
        "DFF": 5.25,
        "CPIAUCSL": 3.2,
        "UNRATE": 3.7,
    }
    for series_id, value in macro_series.items():
        for date in _EQUITY_DATES:
            conn.execute(
                f"INSERT OR REPLACE INTO macro_features (series_id, date, value) "  # nosec B608
                f"VALUES ('{series_id}', '{date}', {value})"  # nosec B608
            )

    # Seed hmm_state_history for equity
    for date in _EQUITY_DATES:
        conn.execute(
            "INSERT OR REPLACE INTO hmm_state_history "
            "(environment, date, p_bull, p_bear, log_likelihood, fitted_at) "
            f"VALUES ('equity', '{date}', 0.65, 0.35, -100.0, '2024-01-01 00:00:00')"
        )

    return conn


def _make_crypto_db() -> duckdb.DuckDBPyConnection:
    """Create in-memory DuckDB seeded with crypto test data."""
    conn = duckdb.connect(":memory:")
    init_feature_schema(conn)

    # Create ohlcv_4h
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_4h (
            symbol TEXT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, datetime)
        )
    """)

    # Create macro_features
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value DOUBLE,
            PRIMARY KEY (series_id, date)
        )
    """)

    rng = np.random.default_rng(43)

    # Seed features_crypto: 3 timestamps x 2 symbols
    for dt in _CRYPTO_DATETIMES:
        for symbol in _CRYPTO_SYMBOLS:
            feature_vals = rng.uniform(0.5, 1.5, len(_CRYPTO_FEATURE_COLS)).tolist()
            col_str = ", ".join(_CRYPTO_FEATURE_COLS)
            val_str = ", ".join(str(v) for v in feature_vals)
            conn.execute(
                f"INSERT OR REPLACE INTO features_crypto (symbol, datetime, {col_str}) "  # nosec B608
                f"VALUES ('{symbol}', '{dt}', {val_str})"  # nosec B608
            )

    # Seed ohlcv_4h: close prices per symbol/datetime
    base_prices = {"BTCUSDT": 42000.0, "ETHUSDT": 2500.0}
    for dt in _CRYPTO_DATETIMES:
        for symbol in _CRYPTO_SYMBOLS:
            close = base_prices[symbol] * rng.uniform(0.99, 1.01)
            conn.execute(
                "INSERT OR REPLACE INTO ohlcv_4h "
                "(symbol, datetime, open, high, low, close, volume) "
                f"VALUES ('{symbol}', '{dt}', {close}, {close}, {close}, {close}, 500.0)"
            )

    # Seed macro_features
    macro_series = {
        "VIXCLS": 20.0,
        "T10Y2Y": 0.3,
        "DFF": 5.25,
        "CPIAUCSL": 3.5,
        "UNRATE": 3.8,
    }
    for series_id, value in macro_series.items():
        conn.execute(
            f"INSERT OR REPLACE INTO macro_features (series_id, date, value) "  # nosec B608
            f"VALUES ('{series_id}', '2024-01-01', {value})"  # nosec B608
        )

    # Seed hmm_state_history for crypto
    conn.execute(
        "INSERT OR REPLACE INTO hmm_state_history "
        "(environment, date, p_bull, p_bear, log_likelihood, fitted_at) "
        "VALUES ('crypto', '2024-01-01', 0.55, 0.45, -200.0, '2024-01-01 00:00:00')"
    )

    return conn


class TestLoadFeaturesEquity:
    """TRAIN-01: _load_features_prices returns correctly shaped equity arrays."""

    def test_equity_features_shape(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-01: equity features shape is (N, 156) with N timesteps."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_equity_db()
        features, prices = _load_features_prices(conn, "equity", equity_env_config)

        n_dates = len(_EQUITY_DATES)
        expected_obs_dim = equity_obs_dim(
            equity_env_config.sentiment.enabled, len(equity_env_config.equity.symbols)
        )
        assert features.shape == (n_dates, expected_obs_dim), (
            f"Expected equity features shape ({n_dates}, {expected_obs_dim}), got {features.shape}"
        )
        conn.close()

    def test_equity_prices_shape(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-01: equity prices shape is (N, n_symbols)."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_equity_db()
        features, prices = _load_features_prices(conn, "equity", equity_env_config)

        n_dates = len(_EQUITY_DATES)
        n_symbols = len(equity_env_config.equity.symbols)
        assert prices.shape == (n_dates, n_symbols), (
            f"Expected equity prices shape ({n_dates}, {n_symbols}), got {prices.shape}"
        )
        conn.close()

    def test_equity_no_nans_in_features(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-01: equity features contain no NaN values."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_equity_db()
        features, prices = _load_features_prices(conn, "equity", equity_env_config)

        assert not np.any(np.isnan(features)), "Features contain NaN values"
        conn.close()

    def test_equity_prices_positive(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-01: equity close prices are positive floats."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_equity_db()
        features, prices = _load_features_prices(conn, "equity", equity_env_config)

        assert np.all(prices > 0.0), "Some prices are not positive"
        conn.close()


class TestLoadFeaturesCrypto:
    """TRAIN-02: _load_features_prices returns correctly shaped crypto arrays."""

    def test_crypto_features_shape(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-02: crypto features shape is (N, 45) with N timesteps."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_crypto_db()
        features, prices = _load_features_prices(conn, "crypto", equity_env_config)

        n_timestamps = len(_CRYPTO_DATETIMES)
        assert features.shape == (n_timestamps, CRYPTO_OBS_DIM), (
            f"Expected crypto features shape ({n_timestamps}, {CRYPTO_OBS_DIM}), "
            f"got {features.shape}"
        )
        conn.close()

    def test_crypto_prices_shape(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-02: crypto prices shape is (N, n_symbols)."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_crypto_db()
        features, prices = _load_features_prices(conn, "crypto", equity_env_config)

        n_timestamps = len(_CRYPTO_DATETIMES)
        n_symbols = len(equity_env_config.crypto.symbols)
        assert prices.shape == (n_timestamps, n_symbols), (
            f"Expected crypto prices shape ({n_timestamps}, {n_symbols}), got {prices.shape}"
        )
        conn.close()


class TestDryRunCLI:
    """VAL-01: --dry-run flag validates features without training."""

    def test_parser_accepts_dry_run(self) -> None:
        """VAL-01: build_parser() accepts --dry-run flag."""
        from train import build_parser  # type: ignore[import]

        parser = build_parser()
        args = parser.parse_args(["--env", "equity", "--dry-run"])
        assert args.dry_run is True

    def test_dry_run_flag_default_false(self) -> None:
        """VAL-01: --dry-run defaults to False."""
        from train import build_parser  # type: ignore[import]

        parser = build_parser()
        args = parser.parse_args(["--env", "equity"])
        assert args.dry_run is False

    def test_dry_run_skips_training(self, tmp_path: Path, equity_env_config: SwingRLConfig) -> None:
        """VAL-01: main() with --dry-run does not call orchestrator.train()."""
        from train import main  # type: ignore[import]

        conn = _make_equity_db()

        # Write config to tmp file
        import textwrap

        config_yaml = textwrap.dedent("""\
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
              duckdb_path: "{db_path}"
              sqlite_path: data/db/trading_ops.db
            alerting:
              alert_cooldown_minutes: 30
              consecutive_failures_before_alert: 3
        """)

        # Create a real DuckDB file seeded with test data
        db_path = tmp_path / "test_market_data.ddb"
        seed_conn = duckdb.connect(str(db_path))
        init_feature_schema(seed_conn)
        seed_conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, date)
            )
        """)
        seed_conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_features (
                series_id TEXT NOT NULL,
                date DATE NOT NULL,
                value DOUBLE,
                PRIMARY KEY (series_id, date)
            )
        """)

        rng = np.random.default_rng(99)
        equity_symbols = ["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"]
        for date in _EQUITY_DATES:
            for symbol in equity_symbols:
                feature_vals = rng.uniform(0.5, 1.5, len(_EQUITY_FEATURE_COLS)).tolist()
                col_str = ", ".join(_EQUITY_FEATURE_COLS)
                val_str = ", ".join(str(v) for v in feature_vals)
                seed_conn.execute(
                    f"INSERT OR REPLACE INTO features_equity (symbol, date, {col_str}) "  # nosec B608
                    f"VALUES ('{symbol}', '{date}', {val_str})"  # nosec B608
                )
                close = rng.uniform(50.0, 500.0)
                seed_conn.execute(
                    "INSERT OR REPLACE INTO ohlcv_daily "
                    "(symbol, date, open, high, low, close, volume) "
                    f"VALUES ('{symbol}', '{date}', {close}, {close}, {close}, {close}, 1000000.0)"
                )
        for series_id, value in {"VIXCLS": 18.5, "T10Y2Y": 0.5, "DFF": 5.25}.items():
            for date in _EQUITY_DATES:
                seed_conn.execute(
                    "INSERT OR REPLACE INTO macro_features (series_id, date, value) "
                    f"VALUES ('{series_id}', '{date}', {value})"
                )
        for date in _EQUITY_DATES:
            seed_conn.execute(
                "INSERT OR REPLACE INTO hmm_state_history "
                "(environment, date, p_bull, p_bear, log_likelihood, fitted_at) "
                f"VALUES ('equity', '{date}', 0.65, 0.35, -100.0, '2024-01-01 00:00:00')"
            )
        seed_conn.close()

        # Write the config with the real db path
        config_file = tmp_path / "swingrl.yaml"
        config_file.write_text(config_yaml.format(db_path=str(db_path)))

        with patch("train.TrainingOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch

            exit_code = main(["--env", "equity", "--dry-run", "--config", str(config_file)])

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
        mock_orch.train.assert_not_called()
        conn.close()


class TestFoldShapes:
    """TRAIN-03: Features and prices arrays have identical row counts (N aligned)."""

    def test_equity_row_alignment(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-03: equity features and prices have same N (no date misalignment)."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_equity_db()
        features, prices = _load_features_prices(conn, "equity", equity_env_config)

        assert features.shape[0] == prices.shape[0], (
            f"Row count mismatch: features has {features.shape[0]} rows, "
            f"prices has {prices.shape[0]} rows"
        )
        conn.close()

    def test_crypto_row_alignment(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-03: crypto features and prices have same N (no timestamp misalignment)."""
        from train import _load_features_prices  # type: ignore[import]

        conn = _make_crypto_db()
        features, prices = _load_features_prices(conn, "crypto", equity_env_config)

        assert features.shape[0] == prices.shape[0], (
            f"Row count mismatch: features has {features.shape[0]} rows, "
            f"prices has {prices.shape[0]} rows"
        )
        conn.close()


class TestEmptyTables:
    """TRAIN-01/02: RuntimeError raised when features table is empty."""

    def test_empty_equity_table_raises(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-01: _load_features_prices raises RuntimeError for empty equity table."""
        from train import _load_features_prices  # type: ignore[import]

        conn = duckdb.connect(":memory:")
        init_feature_schema(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol TEXT, date DATE, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, volume DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_features (
                series_id TEXT, date DATE, value DOUBLE
            )
        """)

        with pytest.raises(RuntimeError, match="No data found"):
            _load_features_prices(conn, "equity", equity_env_config)
        conn.close()

    def test_empty_crypto_table_raises(self, equity_env_config: SwingRLConfig) -> None:
        """TRAIN-02: _load_features_prices raises RuntimeError for empty crypto table."""
        from train import _load_features_prices  # type: ignore[import]

        conn = duckdb.connect(":memory:")
        init_feature_schema(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_4h (
                symbol TEXT, datetime TIMESTAMP, open DOUBLE, high DOUBLE,
                low DOUBLE, close DOUBLE, volume DOUBLE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_features (
                series_id TEXT, date DATE, value DOUBLE
            )
        """)

        with pytest.raises(RuntimeError, match="No data found"):
            _load_features_prices(conn, "crypto", equity_env_config)
        conn.close()
