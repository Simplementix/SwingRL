"""Tests for FREDIngestor — FRED macro data ingestor.

All tests use mocked fredapi to avoid real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from swingrl.data.fred import ALL_SERIES, FREDIngestor
from swingrl.utils.exceptions import DataError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_vixcls_series() -> pd.Series:
    """Load mock VIX series from fixture file."""
    with open(FIXTURES_DIR / "fred_vixcls_series.json") as f:
        data = json.load(f)
    return pd.Series(data, dtype=float, name="VIXCLS")


def _load_cpiaucsl_releases() -> pd.DataFrame:
    """Load mock CPI all-releases from fixture file.

    Matches fredapi.Fred.get_series_all_releases() return format:
    integer index with columns date, realtime_start, value.
    """
    with open(FIXTURES_DIR / "fred_cpiaucsl_releases.json") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


@pytest.fixture()
def fred_config(loaded_config):  # noqa: ANN001, ANN201
    """Config fixture for FRED tests."""
    return loaded_config


@pytest.fixture()
def mock_fred_api():  # noqa: ANN201
    """Create a mocked fredapi.Fred instance."""
    with patch("swingrl.data.fred.Fred") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture()
def ingestor(fred_config, mock_fred_api, monkeypatch, tmp_path):  # noqa: ANN001, ANN201
    """Create a FREDIngestor with mocked fredapi and temp data dir."""
    monkeypatch.setenv("FRED_API_KEY", "test-key-12345")
    # Override data dir to use tmp_path
    fred_config.paths.data_dir = str(tmp_path / "data")
    inst = FREDIngestor(fred_config)
    # Disable staleness check — fixture data uses 2024 dates which would trip 35-day threshold
    monkeypatch.setattr(inst._validator, "_check_staleness", lambda df, sym: None)
    return inst


class TestFetchDailySeries:
    """Tests for daily (non-vintage) FRED series fetching."""

    def test_fetch_daily_series_vixcls(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock
    ) -> None:
        """DATA-04: fetch daily series returns DataFrame with value column, no vintage_date."""
        mock_fred_api.get_series.return_value = _load_vixcls_series()

        result = ingestor.fetch("VIXCLS")

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "vintage_date" not in result.columns
        assert result.index.name == "observation_date"

    def test_observation_start_2016(self, ingestor: FREDIngestor, mock_fred_api: MagicMock) -> None:
        """DATA-04: get_series called with observation_start='2016-01-01' for backfill."""
        mock_fred_api.get_series.return_value = _load_vixcls_series()

        ingestor.fetch("VIXCLS", since=None)

        mock_fred_api.get_series.assert_called_once()
        call_kwargs = mock_fred_api.get_series.call_args
        assert call_kwargs[1].get("observation_start") == "2016-01-01" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "2016-01-01"
        )


class TestFetchVintageSeries:
    """Tests for vintage (ALFRED) FRED series fetching."""

    def test_fetch_vintage_series_cpiaucsl(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock
    ) -> None:
        """DATA-04: fetch vintage series returns DataFrame with vintage_date column."""
        mock_fred_api.get_series_all_releases.return_value = _load_cpiaucsl_releases()

        result = ingestor.fetch("CPIAUCSL")

        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns
        assert "vintage_date" in result.columns
        assert result.index.name == "observation_date"

    def test_vintage_date_stored(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock, tmp_path: Path
    ) -> None:
        """DATA-04: CPIAUCSL Parquet has vintage_date column."""
        mock_fred_api.get_series_all_releases.return_value = _load_cpiaucsl_releases()

        df = ingestor.fetch("CPIAUCSL")
        path = ingestor.store(df, "CPIAUCSL")

        stored = pd.read_parquet(path)
        assert "vintage_date" in stored.columns

    def test_daily_series_no_vintage(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock
    ) -> None:
        """DATA-04: VIXCLS Parquet does NOT have vintage_date column."""
        mock_fred_api.get_series.return_value = _load_vixcls_series()

        df = ingestor.fetch("VIXCLS")
        path = ingestor.store(df, "VIXCLS")

        stored = pd.read_parquet(path)
        assert "vintage_date" not in stored.columns


class TestRunAll:
    """Tests for run_all orchestration."""

    def test_all_five_series_paths(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock, tmp_path: Path
    ) -> None:
        """DATA-04: run_all creates 5 Parquet files in data/macro/."""
        mock_fred_api.get_series.return_value = _load_vixcls_series()
        mock_fred_api.get_series_all_releases.return_value = _load_cpiaucsl_releases()

        failed = ingestor.run_all()

        assert len(failed) == 0
        macro_dir = tmp_path / "data" / "macro"
        for series_id in ALL_SERIES:
            assert (macro_dir / f"{series_id}.parquet").exists(), f"{series_id}.parquet missing"

    def test_partial_failure_continues(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock, tmp_path: Path
    ) -> None:
        """DATA-04: if VIXCLS fails, still fetches remaining series."""

        def side_effect(series_id: str, **kwargs) -> pd.Series:  # noqa: ANN003, ARG001
            if series_id == "VIXCLS":
                raise ConnectionError("API timeout")
            return _load_vixcls_series()

        mock_fred_api.get_series.side_effect = side_effect
        mock_fred_api.get_series_all_releases.return_value = _load_cpiaucsl_releases()

        failed = ingestor.run_all()

        assert "VIXCLS" in failed
        # Other daily series should still have been fetched
        macro_dir = tmp_path / "data" / "macro"
        assert (macro_dir / "T10Y2Y.parquet").exists()
        assert (macro_dir / "DFF.parquet").exists()


class TestIncremental:
    """Tests for incremental data fetching."""

    def test_incremental_from_parquet(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock, tmp_path: Path
    ) -> None:
        """DATA-04: when Parquet exists, fetch starts from max(observation_date) + 1 day."""
        # Create existing Parquet with data up to 2024-01-15
        existing_data = {
            "2024-01-02": 13.20,
            "2024-01-03": 13.68,
            "2024-01-15": 12.52,
        }
        existing_series = pd.Series(existing_data, dtype=float, name="VIXCLS")
        existing_df = pd.DataFrame(
            {"value": existing_series.values}, index=pd.to_datetime(existing_series.index)
        )
        existing_df.index.name = "observation_date"

        macro_dir = tmp_path / "data" / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        existing_df.to_parquet(macro_dir / "VIXCLS.parquet")

        mock_fred_api.get_series.return_value = _load_vixcls_series()

        ingestor.fetch("VIXCLS", since="auto")

        call_kwargs = mock_fred_api.get_series.call_args[1]
        assert call_kwargs["observation_start"] == "2024-01-16"


class TestRetry:
    """Tests for retry logic."""

    def test_retry_on_api_error(self, ingestor: FREDIngestor, mock_fred_api: MagicMock) -> None:
        """DATA-04: 3 retries with backoff; raises DataError after exhaustion."""
        mock_fred_api.get_series.side_effect = ConnectionError("API error")

        with pytest.raises(DataError, match="VIXCLS"):
            ingestor.fetch("VIXCLS")

        assert mock_fred_api.get_series.call_count == 3


class TestCLI:
    """Tests for CLI entry point."""

    def test_cli_series_override(
        self,
        fred_config,
        mock_fred_api: MagicMock,
        monkeypatch,
        tmp_path: Path,  # noqa: ANN001
    ) -> None:
        """DATA-04: --series VIXCLS T10Y2Y overrides default 5 series."""
        monkeypatch.setenv("FRED_API_KEY", "test-key-12345")
        fred_config.paths.data_dir = str(tmp_path / "data")
        mock_fred_api.get_series.return_value = _load_vixcls_series()
        # Disable staleness for fixture data
        monkeypatch.setattr(
            "swingrl.data.validation.DataValidator._check_staleness", lambda self, df, sym: None
        )

        from swingrl.data.fred import _run_cli

        result = _run_cli(fred_config, series=["VIXCLS", "T10Y2Y"], backfill=False)

        assert mock_fred_api.get_series.call_count == 2
        assert result == 0

    def test_cli_backfill(
        self,
        fred_config,
        mock_fred_api: MagicMock,
        monkeypatch,
        tmp_path: Path,  # noqa: ANN001
    ) -> None:
        """DATA-04: --backfill forces observation_start='2016-01-01'."""
        monkeypatch.setenv("FRED_API_KEY", "test-key-12345")
        fred_config.paths.data_dir = str(tmp_path / "data")
        # Disable staleness for fixture data
        monkeypatch.setattr(
            "swingrl.data.validation.DataValidator._check_staleness", lambda self, df, sym: None
        )

        # Create existing parquet so incremental would normally skip
        macro_dir = tmp_path / "data" / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        existing_df = pd.DataFrame(
            {"value": [13.20]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="observation_date"),
        )
        existing_df.to_parquet(macro_dir / "VIXCLS.parquet")

        mock_fred_api.get_series.return_value = _load_vixcls_series()

        from swingrl.data.fred import _run_cli

        _run_cli(fred_config, series=["VIXCLS"], backfill=True)

        call_kwargs = mock_fred_api.get_series.call_args[1]
        assert call_kwargs["observation_start"] == "2016-01-01"


class TestValidation:
    """Tests for validation delegation."""

    def test_validate_delegates_to_fred_validator(
        self, ingestor: FREDIngestor, mock_fred_api: MagicMock
    ) -> None:
        """DATA-04: validate() uses DataValidator(source='fred')."""
        mock_fred_api.get_series.return_value = _load_vixcls_series()
        df = ingestor.fetch("VIXCLS")

        clean, quarantine = ingestor.validate(df, "VIXCLS")

        # Clean data should pass through
        assert len(clean) > 0
        assert isinstance(quarantine, pd.DataFrame)
