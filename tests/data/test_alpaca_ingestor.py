"""Tests for AlpacaIngestor — equity OHLCV data ingestion via Alpaca IEX feed.

All tests use mocked SDK client to avoid real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.data.alpaca import AlpacaIngestor
from swingrl.data.validation import DataValidator
from swingrl.utils.exceptions import DataError


@pytest.fixture
def alpaca_fixture_path() -> Path:
    """Path to the Alpaca bars fixture JSON."""
    return Path(__file__).parent / "fixtures" / "alpaca_bars_spy.json"


@pytest.fixture
def alpaca_fixture_data(alpaca_fixture_path: Path) -> dict:
    """Parsed Alpaca bars fixture data."""
    return json.loads(alpaca_fixture_path.read_text())


@pytest.fixture
def mock_bars_df() -> pd.DataFrame:
    """DataFrame mimicking what BarSet.df returns for SPY."""
    dates = pd.DatetimeIndex(
        [
            "2024-01-02T05:00:00+00:00",
            "2024-01-03T05:00:00+00:00",
            "2024-01-04T05:00:00+00:00",
            "2024-01-05T05:00:00+00:00",
            "2024-01-08T05:00:00+00:00",
        ],
        name="timestamp",
    )
    df = pd.DataFrame(
        {
            "open": [472.65, 471.20, 470.80, 469.45, 469.10],
            "high": [474.10, 473.42, 472.15, 470.88, 471.60],
            "low": [470.52, 469.85, 468.42, 467.92, 468.50],
            "close": [472.35, 470.95, 469.30, 468.75, 471.25],
            "volume": [54672100.0, 61238400.0, 58903200.0, 52187600.0, 55894300.0],
            "trade_count": [452310.0, 498720.0, 467890.0, 412350.0, 435670.0],
            "vwap": [472.01, 471.38, 470.02, 469.15, 470.22],
        },
        index=dates,
    )
    # Alpaca SDK returns a MultiIndex with (symbol, timestamp) — .df output
    df.index = pd.MultiIndex.from_arrays(
        [["SPY"] * len(df), df.index],
        names=["symbol", "timestamp"],
    )
    return df


@pytest.fixture
def alpaca_ingestor(
    loaded_config: SwingRLConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AlpacaIngestor:
    """Create an AlpacaIngestor with mocked env vars and tmp data dir."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-api-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret-key")
    # Override data dir to use tmp_path
    monkeypatch.setattr(loaded_config.paths, "data_dir", str(tmp_path / "data"))
    return AlpacaIngestor(loaded_config)


class TestAlpacaFetch:
    """Test AlpacaIngestor.fetch() method."""

    def test_fetch_returns_ohlcv_dataframe(
        self, alpaca_ingestor: AlpacaIngestor, mock_bars_df: pd.DataFrame
    ) -> None:
        """DATA-01: fetch() returns DataFrame with OHLCV columns and UTC timestamps."""
        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        with patch.object(alpaca_ingestor._client, "get_stock_bars", return_value=mock_barset):
            result = alpaca_ingestor.fetch("SPY")

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}
        assert result.index.tz is not None  # UTC timezone
        assert len(result) == 5

    def test_fetch_uses_iex_feed(
        self, alpaca_ingestor: AlpacaIngestor, mock_bars_df: pd.DataFrame
    ) -> None:
        """DATA-01: StockBarsRequest uses feed=DataFeed.IEX."""
        from alpaca.data.enums import DataFeed

        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        with patch.object(
            alpaca_ingestor._client, "get_stock_bars", return_value=mock_barset
        ) as mock_get:
            alpaca_ingestor.fetch("SPY")

        call_args = mock_get.call_args[0][0]  # First positional arg (StockBarsRequest)
        assert call_args.feed == DataFeed.IEX

    def test_fetch_adjustment_all(
        self, alpaca_ingestor: AlpacaIngestor, mock_bars_df: pd.DataFrame
    ) -> None:
        """DATA-01: StockBarsRequest uses adjustment=Adjustment.ALL (split+dividend)."""
        from alpaca.data.enums import Adjustment

        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        with patch.object(
            alpaca_ingestor._client, "get_stock_bars", return_value=mock_barset
        ) as mock_get:
            alpaca_ingestor.fetch("SPY")

        call_args = mock_get.call_args[0][0]
        assert call_args.adjustment == Adjustment.ALL


class TestAlpacaIncremental:
    """Test incremental and backfill fetch modes."""

    def test_incremental_start_from_parquet(
        self,
        alpaca_ingestor: AlpacaIngestor,
        mock_bars_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """DATA-01: Incremental mode starts from max(timestamp) + 1 day in Parquet."""
        # Create existing parquet with data up to 2024-01-10
        existing_dates = pd.date_range("2024-01-08", periods=3, freq="B", tz="UTC")
        existing_df = pd.DataFrame(
            {
                "open": [469.10, 470.20, 471.30],
                "high": [471.60, 472.50, 473.40],
                "low": [468.50, 469.60, 470.70],
                "close": [471.25, 471.80, 472.90],
                "volume": [55000000.0, 56000000.0, 57000000.0],
            },
            index=existing_dates,
        )
        equity_dir = tmp_path / "data" / "equity"
        equity_dir.mkdir(parents=True, exist_ok=True)
        existing_df.to_parquet(equity_dir / "SPY_daily.parquet")

        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        with patch.object(
            alpaca_ingestor._client, "get_stock_bars", return_value=mock_barset
        ) as mock_get:
            alpaca_ingestor.fetch("SPY", since="incremental")

        call_args = mock_get.call_args[0][0]
        # Start should be after 2024-01-10 (max date in existing parquet)
        assert call_args.start.date() > existing_dates[-1].date()

    def test_backfill_starts_from_2016(
        self, alpaca_ingestor: AlpacaIngestor, mock_bars_df: pd.DataFrame
    ) -> None:
        """DATA-01: Full backfill (since=None) starts from 2016-01-01."""
        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        with patch.object(
            alpaca_ingestor._client, "get_stock_bars", return_value=mock_barset
        ) as mock_get:
            alpaca_ingestor.fetch("SPY", since=None)

        call_args = mock_get.call_args[0][0]
        assert call_args.start.year == 2016
        assert call_args.start.month == 1
        assert call_args.start.day == 1


class TestAlpacaStoreAndValidate:
    """Test store() and validate() methods."""

    def test_store_writes_to_equity_dir(
        self, alpaca_ingestor: AlpacaIngestor, tmp_path: Path
    ) -> None:
        """DATA-01: store() writes to data/equity/{SYMBOL}_daily.parquet."""
        dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [472.65, 471.20, 470.80],
                "high": [474.10, 473.42, 472.15],
                "low": [470.52, 469.85, 468.42],
                "close": [472.35, 470.95, 469.30],
                "volume": [54672100.0, 61238400.0, 58903200.0],
            },
            index=dates,
        )
        result_path = alpaca_ingestor.store(df, "SPY")
        assert result_path.exists()
        assert result_path.name == "SPY_daily.parquet"
        assert "equity" in result_path.parts

    def test_validate_delegates_to_data_validator(self, alpaca_ingestor: AlpacaIngestor) -> None:
        """DATA-01: validate() uses DataValidator(source='equity')."""
        dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [472.65, 471.20, 470.80],
                "high": [474.10, 473.42, 472.15],
                "low": [470.52, 469.85, 468.42],
                "close": [472.35, 470.95, 469.30],
                "volume": [54672100.0, 61238400.0, 58903200.0],
            },
            index=dates,
        )
        with patch("swingrl.data.alpaca.DataValidator") as mock_validator_cls:
            mock_instance = MagicMock()
            mock_instance.validate_rows.return_value = (df, pd.DataFrame())
            mock_instance.validate_batch.return_value = df
            mock_validator_cls.return_value = mock_instance

            clean, quarantine = alpaca_ingestor.validate(df, "SPY")

        mock_validator_cls.assert_called_once_with(source="equity")
        mock_instance.validate_rows.assert_called_once()
        mock_instance.validate_batch.assert_called_once()


class TestAlpacaRetry:
    """Test retry logic on API failures."""

    def test_retry_on_api_error(self, alpaca_ingestor: AlpacaIngestor) -> None:
        """DATA-01: fetch retries 3 times with backoff then raises DataError."""
        from requests.exceptions import HTTPError

        with patch.object(
            alpaca_ingestor._client, "get_stock_bars", side_effect=HTTPError("500 Server Error")
        ):
            with patch("swingrl.data.alpaca.time.sleep") as mock_sleep:
                with pytest.raises(DataError, match="SPY"):
                    alpaca_ingestor.fetch("SPY")

        # Should have slept 3 times with exponential backoff: 2, 4, 8
        assert mock_sleep.call_count == 3
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_args == [2, 4, 8]


class TestAlpacaRunAll:
    """Test run_all() method for multi-symbol execution."""

    def test_run_multiple_symbols_partial_failure(
        self, alpaca_ingestor: AlpacaIngestor, mock_bars_df: pd.DataFrame
    ) -> None:
        """DATA-01: run_all() continues after one symbol fails, returns failed list."""
        mock_barset = MagicMock()
        mock_barset.df = mock_bars_df

        def side_effect(request):
            symbol = request.symbol_or_symbols
            if symbol == "XLE":
                raise DataError("XLE fetch failed")
            return mock_barset

        with patch.object(alpaca_ingestor._client, "get_stock_bars", side_effect=side_effect):
            with patch("swingrl.data.alpaca.time.sleep"):
                # Patch staleness check since fixture data is old
                with patch.object(DataValidator, "_check_staleness"):
                    failed = alpaca_ingestor.run_all(["SPY", "XLE", "QQQ"], since=None)

        assert "XLE" in failed
        assert "SPY" not in failed
        assert "QQQ" not in failed


class TestAlpacaCLI:
    """Test CLI argument parsing."""

    def test_cli_backfill_flag(self) -> None:
        """DATA-01: --backfill flag sets since=None for full fetch."""
        from swingrl.data.alpaca import _parse_args

        args = _parse_args(["--backfill"])
        assert args.backfill is True

    def test_cli_symbols_override(self) -> None:
        """DATA-01: --symbols SPY QQQ overrides config symbols."""
        from swingrl.data.alpaca import _parse_args

        args = _parse_args(["--symbols", "SPY", "QQQ"])
        assert args.symbols == ["SPY", "QQQ"]

    def test_cli_days_flag(self) -> None:
        """DATA-01: --days 30 sets day count."""
        from swingrl.data.alpaca import _parse_args

        args = _parse_args(["--days", "30"])
        assert args.days == 30
