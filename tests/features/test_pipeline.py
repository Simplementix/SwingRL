"""Tests for feature pipeline, CLI, and A/B comparison infrastructure.

Verifies FeaturePipeline orchestrates compute-normalize-store for both environments,
compare_features applies Sharpe-based acceptance, and CLI parses arguments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.features.pipeline import FeaturePipeline, compare_features
from swingrl.features.schema import init_feature_schema


@pytest.fixture
def pipeline_config(tmp_path: Path) -> SwingRLConfig:
    """Config with 8 equity + 2 crypto symbols for pipeline tests."""
    yaml_content = """\
trading_mode: paper
equity:
  symbols: [DIA, IWM, QQQ, SPY, VTI, XLE, XLF, XLK]
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
features:
  equity_zscore_window: 252
  crypto_zscore_window: 360
  equity_hmm_window: 200
  crypto_hmm_window: 200
  hmm_n_iter: 50
  hmm_n_inits: 10
  hmm_ridge: 1e-4
"""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(yaml_content)
    return load_config(config_file)


@pytest.fixture
def seeded_duckdb() -> Any:
    """In-memory DuckDB with OHLCV, macro data, and feature schema."""
    conn = duckdb.connect(":memory:")

    # Create base tables
    conn.execute("""
        CREATE TABLE ohlcv_daily (
            symbol TEXT, date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT, adjusted_close DOUBLE,
            fetched_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE ohlcv_4h (
            symbol TEXT, datetime TIMESTAMP,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, source TEXT,
            fetched_at TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, datetime)
        )
    """)
    conn.execute("""
        CREATE TABLE macro_features (
            date DATE, series_id TEXT,
            value DOUBLE, release_date DATE,
            PRIMARY KEY (date, series_id)
        )
    """)

    # Init feature schema
    init_feature_schema(conn)

    # Seed equity OHLCV data (300 bars per symbol for warmup)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=300)
    for symbol in ["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"]:
        base = 100.0 + rng.random() * 400
        if symbol == "SPY":
            # Clear bull-then-bear regime so HMM converges reliably across platforms
            # Bull: small positive returns with low vol; Bear: negative returns with high vol
            bull_returns = rng.normal(0.002, 0.005, 150)
            bear_returns = rng.normal(-0.003, 0.02, 150)
            all_returns = np.concatenate([bull_returns, bear_returns])
            close = base * np.exp(np.cumsum(all_returns))
        else:
            close = base + rng.normal(0, 2, 300).cumsum()
        for i, dt in enumerate(dates):
            conn.execute(
                "INSERT INTO ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    symbol,
                    dt.date(),
                    close[i] - 0.5,
                    close[i] + 1.0,
                    close[i] - 1.0,
                    close[i],
                    int(rng.integers(1_000_000, 10_000_000)),
                    close[i],
                    datetime.now(tz=UTC),
                ],
            )

    # Seed crypto OHLCV data (450 4H bars)
    crypto_dates = pd.date_range("2024-01-01", periods=450, freq="4h")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        base = 40_000 if "BTC" in symbol else 2_500
        close = base + rng.normal(0, 50, 450).cumsum()
        for i, dt in enumerate(crypto_dates):
            conn.execute(
                "INSERT INTO ohlcv_4h VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    symbol,
                    dt.to_pydatetime(),
                    close[i] - 10,
                    close[i] + 20,
                    close[i] - 20,
                    close[i],
                    rng.uniform(100, 1000),
                    "test",
                    datetime.now(tz=UTC),
                ],
            )

    # Seed macro data (VIX, T10Y2Y, DFF, CPIAUCSL, UNRATE)
    macro_dates = pd.bdate_range("2022-01-03", periods=600)
    series_map = {
        "VIXCLS": (20.0, 5.0),
        "T10Y2Y": (1.0, 0.5),
        "DFF": (4.5, 0.1),
        "CPIAUCSL": (300.0, 2.0),
        "UNRATE": (3.5, 0.2),
    }
    for series_id, (base_val, std) in series_map.items():
        vals = base_val + rng.normal(0, std, 600).cumsum() * 0.01
        for i, dt in enumerate(macro_dates):
            conn.execute(
                "INSERT INTO macro_features VALUES (?, ?, ?, ?)",
                [dt.date(), series_id, vals[i], dt.date()],
            )

    yield conn
    conn.close()


class TestFeaturePipelineEquity:
    """Tests for equity pipeline computation."""

    def test_compute_equity_runs(self, pipeline_config: SwingRLConfig, seeded_duckdb: Any) -> None:
        """FEAT-11: compute_equity runs full pipeline without error."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        result = pipeline.compute_equity()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_equity_writes_features_table(
        self, pipeline_config: SwingRLConfig, seeded_duckdb: Any
    ) -> None:
        """FEAT-11: Pipeline writes rows to features_equity DuckDB table."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        pipeline.compute_equity()
        count = seeded_duckdb.execute("SELECT COUNT(*) FROM features_equity").fetchone()[0]
        assert count > 0

    def test_equity_stores_hmm_state(
        self, pipeline_config: SwingRLConfig, seeded_duckdb: Any
    ) -> None:
        """FEAT-11: Pipeline fits HMM and stores state to hmm_state_history."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        pipeline.compute_equity()
        count = seeded_duckdb.execute(
            "SELECT COUNT(*) FROM hmm_state_history WHERE environment = 'equity'"
        ).fetchone()[0]
        assert count > 0


class TestFeaturePipelineCrypto:
    """Tests for crypto pipeline computation."""

    def test_compute_crypto_runs(self, pipeline_config: SwingRLConfig, seeded_duckdb: Any) -> None:
        """FEAT-11: compute_crypto runs full pipeline without error."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        result = pipeline.compute_crypto()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_crypto_writes_features_table(
        self, pipeline_config: SwingRLConfig, seeded_duckdb: Any
    ) -> None:
        """FEAT-11: Pipeline writes rows to features_crypto DuckDB table."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        pipeline.compute_crypto()
        count = seeded_duckdb.execute("SELECT COUNT(*) FROM features_crypto").fetchone()[0]
        assert count > 0


class TestObservationIntegration:
    """Tests for observation vector assembly via pipeline."""

    def test_get_equity_observation(
        self, pipeline_config: SwingRLConfig, seeded_duckdb: Any
    ) -> None:
        """FEAT-11: Pipeline produces (156,) equity observation."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        pipeline.compute_equity()
        # Get last date from features_equity
        last_date = seeded_duckdb.execute("SELECT MAX(date) FROM features_equity").fetchone()[0]
        obs = pipeline.get_observation("equity", str(last_date))
        assert obs.shape == (156,)
        assert not np.any(np.isnan(obs))

    def test_get_crypto_observation(
        self, pipeline_config: SwingRLConfig, seeded_duckdb: Any
    ) -> None:
        """FEAT-11: Pipeline produces (45,) crypto observation."""
        pipeline = FeaturePipeline(pipeline_config, seeded_duckdb)
        pipeline.compute_crypto()
        last_dt = seeded_duckdb.execute("SELECT MAX(datetime) FROM features_crypto").fetchone()[0]
        obs = pipeline.get_observation("crypto", str(last_dt))
        assert obs.shape == (45,)
        assert not np.any(np.isnan(obs))


class TestFeatureComparison:
    """Tests for feature A/B comparison infrastructure."""

    def test_accept_when_sharpe_improves(self) -> None:
        """FEAT-08: Accept when validation Sharpe improvement >= 0.05."""
        baseline = {"validation_sharpe": 1.0, "train_sharpe": 1.2}
        candidate = {"validation_sharpe": 1.06, "train_sharpe": 1.3}
        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is True
        assert result["sharpe_improvement"] == pytest.approx(0.06, abs=1e-6)

    def test_reject_when_sharpe_insufficient(self) -> None:
        """FEAT-08: Reject when Sharpe improvement < 0.05."""
        baseline = {"validation_sharpe": 1.0, "train_sharpe": 1.2}
        candidate = {"validation_sharpe": 1.03, "train_sharpe": 1.25}
        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is False

    def test_reject_overfitting(self) -> None:
        """FEAT-08: Reject when train Sharpe improves but validation decreases."""
        baseline = {"validation_sharpe": 1.0, "train_sharpe": 1.2}
        candidate = {"validation_sharpe": 0.95, "train_sharpe": 1.5}
        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is False
        assert "overfit" in result["reason"].lower()

    def test_accept_with_custom_threshold(self) -> None:
        """FEAT-08: Custom threshold changes acceptance criteria."""
        baseline = {"validation_sharpe": 1.0, "train_sharpe": 1.2}
        candidate = {"validation_sharpe": 1.02, "train_sharpe": 1.25}
        result = compare_features(baseline, candidate, threshold=0.01)
        assert result["accepted"] is True


class TestCLIParsing:
    """Tests for compute_features.py CLI argument parsing."""

    @pytest.fixture(autouse=True)
    def _load_cli_module(self) -> None:
        """Load compute_features module from scripts directory."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compute_features",
            Path(__file__).resolve().parents[2] / "scripts" / "compute_features.py",
        )
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._build_parser = mod.build_parser  # type: ignore[attr-defined]

    def test_cli_parses_environment(self) -> None:
        """FEAT-11: CLI accepts --environment argument."""
        parser = self._build_parser()
        args = parser.parse_args(["--environment", "equity"])
        assert args.environment == "equity"

    def test_cli_parses_both(self) -> None:
        """FEAT-11: CLI accepts --environment both."""
        parser = self._build_parser()
        args = parser.parse_args(["--environment", "both"])
        assert args.environment == "both"

    def test_cli_parses_date_range(self) -> None:
        """FEAT-11: CLI accepts --start and --end."""
        parser = self._build_parser()
        args = parser.parse_args(["--start", "2023-01-01", "--end", "2023-12-31"])
        assert args.start == "2023-01-01"
        assert args.end == "2023-12-31"

    def test_cli_parses_symbols(self) -> None:
        """FEAT-11: CLI accepts --symbols override."""
        parser = self._build_parser()
        args = parser.parse_args(["--symbols", "SPY", "QQQ"])
        assert args.symbols == ["SPY", "QQQ"]
