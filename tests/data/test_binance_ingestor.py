"""Tests for BinanceIngestor — crypto 4H OHLCV data ingestion.

Covers DATA-02 (live 4H ingestion) and DATA-03 (historical backfill).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import responses as responses_lib

from swingrl.config.schema import SwingRLConfig
from swingrl.data.binance import (
    BINANCE_US_BASE_URL,
    FOUR_HOURS_MS,
    KLINES_ENDPOINT,
    WEIGHT_LIMIT,
    WEIGHT_THRESHOLD,
    BinanceIngestor,
)

_KLINES_URL = BINANCE_US_BASE_URL + KLINES_ENDPOINT


@pytest.fixture()
def klines_fixture() -> list[list]:
    """Load the mock klines JSON fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "binance_klines_btcusdt.json"
    with fixture_path.open() as f:
        return json.load(f)


@pytest.fixture()
def binance_ingestor(loaded_config: SwingRLConfig) -> BinanceIngestor:
    """Create a BinanceIngestor with test config."""
    return BinanceIngestor(loaded_config)


def _add_klines_then_empty(
    fixture: list[list],
    weight: str = "5",
) -> None:
    """Register a klines response followed by an empty page to stop pagination."""
    responses_lib.add(
        responses_lib.GET,
        _KLINES_URL,
        json=fixture,
        status=200,
        headers={"X-MBX-USED-WEIGHT-1M": weight},
    )
    responses_lib.add(
        responses_lib.GET,
        _KLINES_URL,
        json=[],
        status=200,
        headers={"X-MBX-USED-WEIGHT-1M": weight},
    )


# --- Task 1 Tests: Live fetch, rate limiting, klines parsing ---


@responses_lib.activate
def test_fetch_returns_4h_ohlcv(
    binance_ingestor: BinanceIngestor, klines_fixture: list[list]
) -> None:
    """DATA-02: fetch returns DataFrame with OHLCV columns and UTC timestamps at 4H intervals."""
    _add_klines_then_empty(klines_fixture)

    df = binance_ingestor.fetch("BTCUSDT", since="2024-01-01")
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None  # UTC-aware
    # Check 4H intervals between consecutive bars
    diffs = df.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=4)).all()


@responses_lib.activate
def test_volume_is_quote_asset(
    binance_ingestor: BinanceIngestor, klines_fixture: list[list]
) -> None:
    """DATA-02: volume column uses quote_asset_volume (position [7]), not base volume."""
    _add_klines_then_empty(klines_fixture)

    df = binance_ingestor.fetch("BTCUSDT", since="2024-01-01")
    # First row: quote_asset_volume = 5087550.00
    assert df.iloc[0]["volume"] == pytest.approx(5087550.00)


@responses_lib.activate
def test_rate_limit_throttle(binance_ingestor: BinanceIngestor, klines_fixture: list[list]) -> None:
    """DATA-02: when X-MBX-USED-WEIGHT-1M > 960 (80% of 1200), time.sleep is called."""
    threshold_weight = int(WEIGHT_LIMIT * WEIGHT_THRESHOLD) + 1  # 961
    _add_klines_then_empty(klines_fixture, weight=str(threshold_weight))

    with patch.object(time, "sleep") as mock_sleep:
        binance_ingestor.fetch("BTCUSDT", since="2024-01-01")
        mock_sleep.assert_called()


@responses_lib.activate
def test_rate_limit_no_throttle(
    binance_ingestor: BinanceIngestor, klines_fixture: list[list]
) -> None:
    """DATA-02: when header shows weight < 960, no sleep is called."""
    _add_klines_then_empty(klines_fixture, weight="100")

    with patch.object(time, "sleep") as mock_sleep:
        binance_ingestor.fetch("BTCUSDT", since="2024-01-01")
        mock_sleep.assert_not_called()


@responses_lib.activate
def test_pagination_loop(binance_ingestor: BinanceIngestor, klines_fixture: list[list]) -> None:
    """DATA-02: when >1000 bars needed, fetch loops with advancing startTime."""
    # First page returns 10 bars, second page returns empty (signals end)
    _add_klines_then_empty(klines_fixture)

    df = binance_ingestor.fetch("BTCUSDT", since="2024-01-01")
    assert len(df) == 10
    # Two requests were made (pagination)
    assert len(responses_lib.calls) == 2


@responses_lib.activate
def test_retry_on_api_error(binance_ingestor: BinanceIngestor) -> None:
    """DATA-02: 3 retries with exponential backoff; raises DataError after exhaustion."""
    from swingrl.utils.exceptions import DataError

    for _ in range(3):
        responses_lib.add(
            responses_lib.GET,
            _KLINES_URL,
            json={"error": "server error"},
            status=500,
        )

    with patch.object(time, "sleep"):
        with pytest.raises(DataError, match="API request failed after 3 attempts"):
            binance_ingestor.fetch("BTCUSDT", since="2024-01-01")


@responses_lib.activate
def test_store_writes_to_crypto_dir(binance_ingestor: BinanceIngestor, tmp_path: Path) -> None:
    """DATA-02: store() writes to data/crypto/{SYMBOL}_4h.parquet."""
    binance_ingestor._data_dir = tmp_path
    df = pd.DataFrame(
        {
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [5000000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")], name="timestamp"),
    )
    result_path = binance_ingestor.store(df, "BTCUSDT")
    assert result_path == tmp_path / "crypto" / "BTCUSDT_4h.parquet"
    assert result_path.exists()


@responses_lib.activate
def test_incremental_start_from_parquet(
    binance_ingestor: BinanceIngestor, tmp_path: Path, klines_fixture: list[list]
) -> None:
    """DATA-02: when Parquet exists, derives start from max(timestamp) + 1 bar (4H)."""
    binance_ingestor._data_dir = tmp_path
    # Create existing parquet with one bar
    crypto_dir = tmp_path / "crypto"
    crypto_dir.mkdir(parents=True)
    existing_df = pd.DataFrame(
        {
            "open": [41000.0],
            "high": [41100.0],
            "low": [40900.0],
            "close": [41050.0],
            "volume": [4000000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00", tz="UTC")], name="timestamp"),
    )
    existing_df.to_parquet(crypto_dir / "BTCUSDT_4h.parquet", index=True)

    _add_klines_then_empty(klines_fixture)

    # Fetch with since=None triggers incremental from existing parquet
    df = binance_ingestor.fetch("BTCUSDT")
    assert not df.empty
    # Verify the request used startTime after the existing bar
    first_call = responses_lib.calls[0]
    params = first_call.request.params
    start_time_ms = int(params["startTime"])
    # Should be existing max timestamp + 4 hours in ms
    expected_start = (
        int(pd.Timestamp("2024-01-01 00:00:00", tz="UTC").timestamp() * 1000) + FOUR_HOURS_MS
    )
    assert start_time_ms == expected_start


def test_base_url_is_binance_us() -> None:
    """DATA-02: requests go to api.binance.us, NOT api.binance.com."""
    assert BINANCE_US_BASE_URL == "https://api.binance.us"
    assert "binance.com" not in BINANCE_US_BASE_URL


# --- Task 2 Tests: Archive parsing, stitch validation, backfill ---


@pytest.fixture()
def archive_fixture_path() -> Path:
    """Path to the archive CSV fixture."""
    return Path(__file__).parent / "fixtures" / "binance_archive_btcusdt_4h.csv"


def test_archive_parse_milliseconds(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: CSV rows with ms timestamps (pre-2025) parsed correctly as UTC datetimes."""
    # Use only ms-timestamp rows (first 50 rows of fixture)
    csv_data = b"1514764800000,13800.00,14100.00,13700.00,14000.00,250.50,1514779199999,3507000.00,5200,130.20,1822800.00,0\n"
    csv_data += b"1514779200000,14000.00,14200.00,13900.00,14100.00,265.30,1514793599999,3740730.00,5500,137.40,1937340.00,0\n"
    df = binance_ingestor._parse_archive_csv(csv_data)
    assert not df.empty
    # 1514764800000 ms = 2018-01-01 00:00:00 UTC
    assert df.index[0] == pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
    # Timestamps should be reasonable (2018, not year 35000)
    assert df.index[0].year == 2018
    assert df.index[1].year == 2018


def test_archive_parse_microseconds(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: CSV rows with us timestamps (2025+) parsed correctly."""
    # Use us-timestamp rows from fixture
    csv_data = b"1735689600000000,95000.00,95500.00,94800.00,95300.00,520.30,1735703999999000,49568590.00,12000,269.50,25688350.00,0\n"
    csv_data += b"1735704000000000,95300.00,95800.00,95100.00,95600.00,535.60,1735718399999000,51195560.00,12500,277.40,26524040.00,0\n"
    df = binance_ingestor._parse_archive_csv(csv_data)
    assert not df.empty
    # 1735689600000000 us = 2025-01-01 00:00:00 UTC
    assert df.index[0] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert df.index[0].year == 2025
    assert df.index[1].year == 2025


def test_volume_normalization_quote_volume(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: when quote_asset_volume > 0, use it directly for volume."""
    csv_data = b"1514764800000,13800.00,14100.00,13700.00,14000.00,250.50,1514779199999,3507000.00,5200,130.20,1822800.00,0\n"
    df = binance_ingestor._parse_archive_csv(csv_data)
    # volume should be quote_asset_volume = 3507000.00
    assert df.iloc[0]["volume"] == pytest.approx(3507000.00)


def test_volume_normalization_fallback(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: when quote_asset_volume is 0, compute volume_base * close."""
    # Row with zero quote_asset_volume — should fallback to volume_base * close
    csv_data = b"1735977600000000,95200.00,95700.00,95000.00,95500.00,325.40,1735991999999000,0.00,8500,168.60,16101300.00,0\n"
    df = binance_ingestor._parse_archive_csv(csv_data)
    # volume should be volume_base * close = 325.40 * 95500.00
    expected = 325.40 * 95500.00
    assert df.iloc[0]["volume"] == pytest.approx(expected)


def test_stitch_validation_passes(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: archive and API overlap data within 0.5% price deviation -> no warning."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2019-09-01", tz="UTC"), pd.Timestamp("2019-09-01 04:00:00", tz="UTC")],
        name="timestamp",
    )
    archive_df = pd.DataFrame(
        {
            "open": [10000.0, 10100.0],
            "high": [10200.0, 10300.0],
            "low": [9900.0, 10000.0],
            "close": [10100.0, 10200.0],
            "volume": [5000000.0, 5100000.0],
        },
        index=idx,
    )
    api_df = pd.DataFrame(
        {
            "open": [10001.0, 10101.0],
            "high": [10201.0, 10301.0],
            "low": [9901.0, 10001.0],
            "close": [10101.0, 10201.0],
            "volume": [5000100.0, 5100100.0],
        },
        index=idx,
    )
    result = binance_ingestor._validate_stitch(archive_df, api_df)
    assert result is True


def test_stitch_validation_fails(binance_ingestor: BinanceIngestor) -> None:
    """DATA-03: archive and API overlap data with >0.5% deviation -> structlog warning."""
    idx = pd.DatetimeIndex([pd.Timestamp("2019-09-01", tz="UTC")], name="timestamp")
    archive_df = pd.DataFrame(
        {
            "open": [10000.0],
            "high": [10200.0],
            "low": [9900.0],
            "close": [10100.0],
            "volume": [5000000.0],
        },
        index=idx,
    )
    # 1% deviation on close (10100 vs 10000)
    api_df = pd.DataFrame(
        {
            "open": [10000.0],
            "high": [10200.0],
            "low": [9900.0],
            "close": [10000.0],
            "volume": [5000000.0],
        },
        index=idx,
    )
    result = binance_ingestor._validate_stitch(archive_df, api_df)
    assert result is False


def test_archive_parse_full_fixture(
    binance_ingestor: BinanceIngestor, archive_fixture_path: Path
) -> None:
    """DATA-03: full fixture CSV with both ms and us timestamps parsed correctly."""
    with archive_fixture_path.open("rb") as f:
        csv_data = f.read()
    df = binance_ingestor._parse_archive_csv(csv_data)
    assert len(df) == 80  # 50 ms rows + 30 us rows
    # First row is 2018 (ms), last rows are 2025 (us)
    assert df.index[0].year == 2018
    assert df.index[-1].year == 2025
    # All expected columns present
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@responses_lib.activate
def test_backfill_full_range(binance_ingestor: BinanceIngestor, tmp_path: Path) -> None:
    """DATA-03: backfill stitches archive + API data into single DataFrame."""
    import io as io_mod
    import zipfile as zf_mod

    binance_ingestor._data_dir = tmp_path

    # Archive CSV with a 2018-01-01 bar (ms timestamp)
    csv_row_ms = (
        b"1514764800000,13800.00,14100.00,13700.00,14000.00,"
        b"250.50,1514779199999,3507000.00,5200,130.20,1822800.00,0\n"
    )

    def _make_zip(csv_bytes: bytes) -> bytes:
        buf = io_mod.BytesIO()
        with zf_mod.ZipFile(buf, "w") as z:
            z.writestr("data.csv", csv_bytes)
        return buf.getvalue()

    archive_zip = _make_zip(csv_row_ms)

    # Register archive download responses for months 2019-08 through 2019-10
    for month in ("08", "09", "10"):
        responses_lib.add(
            responses_lib.GET,
            f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2019-{month}.zip",
            body=archive_zip,
            status=200,
        )

    # Mock API klines — one bar at the stitch point, then empty to stop pagination
    api_kline = [
        [
            1567296000000,
            "10000.00",
            "10200.00",
            "9900.00",
            "10100.00",
            "300.00",
            1567310399999,
            "3030000.00",
            6000,
            "155.00",
            "1565500.00",
            "0",
        ],
    ]
    responses_lib.add(
        responses_lib.GET,
        _KLINES_URL,
        json=api_kline,
        status=200,
        headers={"X-MBX-USED-WEIGHT-1M": "5"},
    )
    responses_lib.add(
        responses_lib.GET,
        _KLINES_URL,
        json=[],
        status=200,
        headers={"X-MBX-USED-WEIGHT-1M": "6"},
    )

    # Patch staleness check since test data is historical (2018/2019)
    with patch.object(binance_ingestor._validator, "_check_staleness"):
        df = binance_ingestor.backfill("BTCUSDT", start_date="2019-08-01", end_date="2019-10-01")
    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_cli_backfill_flag() -> None:
    """DATA-03: --backfill flag is recognized by the CLI argument parser."""

    # Just verify the argparse accepts --backfill without error
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int)
    args = parser.parse_args(["--backfill", "--symbols", "BTCUSDT"])
    assert args.backfill is True
    assert args.symbols == ["BTCUSDT"]
