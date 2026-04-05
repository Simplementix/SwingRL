"""Tests for CrossSourceValidator — Alpaca vs yfinance price comparison.

DATA-10: Cross-source validation compares closing prices from PostgreSQL (Alpaca)
with yfinance as a reference source. Discrepancies beyond $0.05 tolerance
are flagged as warnings.
"""

from __future__ import annotations

import os
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from swingrl.config.schema import load_config
from swingrl.data.cross_source import CrossSourceResult, CrossSourceValidator
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Skip entire module if no PostgreSQL available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Fixed reference date for all tests (avoids date.today() drift)
_AS_OF = date(2024, 1, 10)


@pytest.fixture
def cs_config_yaml(tmp_path: Path) -> str:
    """Config YAML with system section pointing to DATABASE_URL."""
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://test:test@localhost:5432/swingrl_test"
    )  # pragma: allowlist secret
    return textwrap.dedent(f"""\
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
          database_url: "{db_url}"
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def cs_config(tmp_path: Path, cs_config_yaml: str) -> Any:
    """Load config with DATABASE_URL."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(cs_config_yaml)
    return load_config(config_file)


@pytest.fixture
def cs_db(cs_config: Any) -> DatabaseManager:
    """Create a DatabaseManager and ensure cleanup after test."""
    DatabaseManager.reset()
    mgr = DatabaseManager(cs_config)
    mgr.init_schema()
    yield mgr  # type: ignore[misc]
    # Truncate all tables for test isolation
    with mgr.connection() as conn:
        conn.execute(
            "DO $$ DECLARE r RECORD; BEGIN "
            "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
            "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
            "END LOOP; END $$"
        )
    DatabaseManager.reset()


def _insert_ohlcv_daily(
    db: DatabaseManager,
    symbol: str,
    dates: list[str],
    closes: list[float],
) -> None:
    """Insert test rows into ohlcv_daily."""
    with db.connection() as conn:
        for d, c in zip(dates, closes, strict=True):
            conn.execute(
                "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [symbol, d, c - 1.0, c + 1.0, c - 2.0, c, 50_000_000],
            )


def _make_yfinance_df(
    symbols: list[str],
    dates: list[str],
    adj_closes: dict[str, list[float]],
) -> pd.DataFrame:
    """Build a DataFrame matching yfinance.download() multi-ticker output format."""
    date_idx = pd.DatetimeIndex([datetime.strptime(d, "%Y-%m-%d") for d in dates])
    columns = pd.MultiIndex.from_tuples(
        [
            (col_type, sym)
            for col_type in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
            for sym in symbols
        ],
        names=["Price", "Ticker"],
    )
    data: dict[tuple[str, str], list[float]] = {}
    for sym in symbols:
        closes = adj_closes.get(sym, [0.0] * len(dates))
        data[("Adj Close", sym)] = closes
        data[("Close", sym)] = [c + 0.01 for c in closes]
        data[("High", sym)] = [c + 2.0 for c in closes]
        data[("Low", sym)] = [c - 2.0 for c in closes]
        data[("Open", sym)] = [c - 0.5 for c in closes]
        data[("Volume", sym)] = [50_000_000.0] * len(dates)
    return pd.DataFrame(data, index=date_idx, columns=columns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossSourceValidator:
    """DATA-10: Cross-source validation tests."""

    def test_matching_prices_return_ok(self, cs_db: DatabaseManager, cs_config: Any) -> None:
        """DATA-10: Prices within $0.05 tolerance return status='ok'."""
        dates = ["2024-01-04", "2024-01-05", "2024-01-08"]
        _insert_ohlcv_daily(cs_db, "SPY", dates, [473.0, 476.0, 477.0])

        yf_df = _make_yfinance_df(["SPY"], dates, {"SPY": [473.01, 475.98, 477.03]})

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            results = validator.validate_prices(symbols=["SPY"], lookback_days=7, as_of_date=_AS_OF)

        assert all(r.status == "ok" for r in results)

    def test_price_beyond_tolerance_returns_warning(
        self, cs_db: DatabaseManager, cs_config: Any
    ) -> None:
        """DATA-10: Prices differing by >$0.05 return status='warning'."""
        dates = ["2024-01-04", "2024-01-05"]
        _insert_ohlcv_daily(cs_db, "SPY", dates, [473.0, 476.0])

        # Second date has 0.50 discrepancy
        yf_df = _make_yfinance_df(["SPY"], dates, {"SPY": [473.0, 476.50]})

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            results = validator.validate_prices(symbols=["SPY"], lookback_days=7, as_of_date=_AS_OF)

        warnings = [r for r in results if r.status == "warning"]
        assert len(warnings) >= 1
        assert warnings[0].symbol == "SPY"
        assert abs(warnings[0].diff) > 0.05

    def test_missing_yfinance_symbol_returns_error(
        self, cs_db: DatabaseManager, cs_config: Any
    ) -> None:
        """DATA-10: Missing symbol in yfinance data returns status='error'."""
        dates = ["2024-01-04"]
        _insert_ohlcv_daily(cs_db, "SPY", dates, [473.0])
        _insert_ohlcv_daily(cs_db, "QQQ", dates, [400.0])

        # yfinance returns data only for SPY, not QQQ
        yf_df = _make_yfinance_df(["SPY"], dates, {"SPY": [473.0]})

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            results = validator.validate_prices(
                symbols=["SPY", "QQQ"], lookback_days=7, as_of_date=_AS_OF
            )

        errors = [r for r in results if r.status == "error"]
        assert len(errors) >= 1
        assert any(r.symbol == "QQQ" for r in errors)

    def test_per_symbol_summary_fields(self, cs_db: DatabaseManager, cs_config: Any) -> None:
        """DATA-10: Results include per-symbol comparison entries."""
        dates = ["2024-01-04", "2024-01-05", "2024-01-08"]
        _insert_ohlcv_daily(cs_db, "SPY", dates, [473.0, 476.0, 477.0])

        yf_df = _make_yfinance_df(["SPY"], dates, {"SPY": [473.02, 476.0, 477.03]})

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            results = validator.validate_prices(symbols=["SPY"], lookback_days=7, as_of_date=_AS_OF)

        assert len(results) == 3
        assert all(isinstance(r, CrossSourceResult) for r in results)

    def test_default_symbols_from_config(self, cs_db: DatabaseManager, cs_config: Any) -> None:
        """DATA-10: validate_prices() defaults to config equity symbols."""
        dates = ["2024-01-04"]
        for sym in cs_config.equity.symbols:
            _insert_ohlcv_daily(cs_db, sym, dates, [100.0])

        yf_df = _make_yfinance_df(
            list(cs_config.equity.symbols),
            dates,
            {sym: [100.0] for sym in cs_config.equity.symbols},
        )

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            results = validator.validate_prices(lookback_days=7, as_of_date=_AS_OF)

        assert len(results) >= 1

    def test_yfinance_called_with_auto_adjust_false(
        self, cs_db: DatabaseManager, cs_config: Any
    ) -> None:
        """DATA-10: yfinance is called with auto_adjust=False."""
        dates = ["2024-01-04"]
        _insert_ohlcv_daily(cs_db, "SPY", dates, [473.0])

        yf_df = _make_yfinance_df(["SPY"], dates, {"SPY": [473.0]})

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = yf_df
            validator = CrossSourceValidator(db=cs_db, config=cs_config)
            validator.validate_prices(symbols=["SPY"], lookback_days=7, as_of_date=_AS_OF)

        mock_yf.download.assert_called_once()
        call_kwargs = mock_yf.download.call_args
        assert call_kwargs.kwargs.get("auto_adjust") is False


class TestDataValidatorStep12:
    """DATA-10: DataValidator.validate_batch Step 12 integration."""

    def test_step12_calls_cross_source_for_equity(
        self, cs_db: DatabaseManager, cs_config: Any
    ) -> None:
        """DATA-10: Step 12 runs cross-source check for equity when db provided."""
        from swingrl.data.validation import DataValidator

        now = pd.Timestamp.now(tz="UTC").normalize()
        idx = pd.DatetimeIndex([now - pd.Timedelta(days=1), now], tz="UTC")
        df = pd.DataFrame(
            {
                "open": [472.0, 475.0],
                "high": [475.0, 478.0],
                "low": [470.0, 474.0],
                "close": [473.0, 476.0],
                "volume": [80_000_000, 75_000_000],
            },
            index=idx,
        )

        with patch("swingrl.data.cross_source.yfinance") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            validator = DataValidator(source="equity", db=cs_db, config=cs_config)
            result = validator.validate_batch(df, symbol="SPY")

        assert len(result) == 2
        # Verify yfinance was called (Step 12 ran)
        mock_yf.download.assert_called_once()

    def test_step12_skipped_without_db(self) -> None:
        """DATA-10: Step 12 logs 'skipped' when db is None (backward compatible)."""
        from swingrl.data.validation import DataValidator

        now = pd.Timestamp.now(tz="UTC").normalize()
        idx = pd.DatetimeIndex([now - pd.Timedelta(days=1), now], tz="UTC")
        df = pd.DataFrame(
            {
                "open": [472.0, 475.0],
                "high": [475.0, 478.0],
                "low": [470.0, 474.0],
                "close": [473.0, 476.0],
                "volume": [80_000_000, 75_000_000],
            },
            index=idx,
        )

        validator = DataValidator(source="equity")
        result = validator.validate_batch(df, symbol="SPY")
        assert len(result) == 2
