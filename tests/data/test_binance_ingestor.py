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
