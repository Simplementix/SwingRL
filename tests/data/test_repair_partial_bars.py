"""Tests for scripts/repair_partial_bars.py — partial 4H bar repair tool.

A partial bar is an ohlcv_4h row whose ``fetched_at`` falls inside its own bar
window ``[datetime, datetime + 4h)`` — proof the bar was stored before it closed.
The repair tool detects these, refuses ones whose window is still open, and (with
``--apply``) deletes + refetches the completed bar and rewrites the Parquet file.

All tests run WITHOUT a database: the DB is replaced by an in-memory fake gateway.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.repair_partial_bars import (
    PartialBarRepairer,
    classify_partial_bars,
)
from swingrl.config.schema import SwingRLConfig
from swingrl.data.binance import BinanceIngestor

_COLUMNS = [
    "symbol",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "fetched_at",
]


def _row(
    symbol: str,
    open_str: str,
    fetched_str: str,
    *,
    close: float = 100.0,
    source: str = "binance",
) -> dict[str, Any]:
    """Build one ohlcv_4h row dict (as read_ohlcv_4h would return it)."""
    return {
        "symbol": symbol,
        "datetime": pd.Timestamp(open_str, tz="UTC"),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
        "source": source,
        "fetched_at": pd.Timestamp(fetched_str, tz="UTC"),
    }


class _FakeGateway:
    """In-memory stand-in for the Postgres ohlcv_4h gateway (no DB)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.df = pd.DataFrame(rows, columns=_COLUMNS)
        self.deleted: list[tuple[str, pd.Timestamp]] = []
        self.inserted: list[tuple[str, pd.Timestamp, dict[str, Any], pd.Timestamp]] = []

    def read_ohlcv_4h(self) -> pd.DataFrame:
        return self.df.copy()

    def delete_bar(self, symbol: str, bar_open: pd.Timestamp) -> None:
        bar_open = pd.Timestamp(bar_open)
        self.deleted.append((symbol, bar_open))
        keep = ~((self.df["symbol"] == symbol) & (self.df["datetime"] == bar_open))
        self.df = self.df[keep].reset_index(drop=True)

    def insert_bar(
        self,
        symbol: str,
        bar_open: pd.Timestamp,
        values: dict[str, Any],
        fetched_at: datetime,
    ) -> None:
        bar_open = pd.Timestamp(bar_open)
        fetched_ts = pd.Timestamp(fetched_at)
        self.inserted.append((symbol, bar_open, dict(values), fetched_ts))
        new = {
            "symbol": symbol,
            "datetime": bar_open,
            "open": values["open"],
            "high": values["high"],
            "low": values["low"],
            "close": values["close"],
            "volume": values["volume"],
            "source": values.get("source"),
            "fetched_at": fetched_ts,
        }
        self.df = pd.concat([self.df, pd.DataFrame([new], columns=_COLUMNS)], ignore_index=True)


# --- classify_partial_bars (pure detection) ---


def test_classify_flags_partial_within_closed_window() -> None:
    """A row whose fetched_at is inside its closed window is repairable; a good row is not."""
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)
    rows = pd.DataFrame(
        [
            # partial: opened 00:00, fetched 00:01 (inside [00:00, 04:00)), window closed
            _row("BTCUSDT", "2026-07-19 00:00:00", "2026-07-19 00:01:00", close=100.0),
            # good: opened 04:00, fetched 08:05 (after window close 08:00)
            _row("BTCUSDT", "2026-07-19 04:00:00", "2026-07-19 08:05:00", close=200.0),
        ],
        columns=_COLUMNS,
    )
    repairable, unclosed = classify_partial_bars(rows, now)
    assert len(repairable) == 1
    assert len(unclosed) == 0
    assert repairable.iloc[0]["datetime"] == pd.Timestamp("2026-07-19 00:00:00", tz="UTC")


def test_classify_refuses_bar_whose_window_is_open() -> None:
    """A partial whose window has NOT closed yet is returned as unclosed, not repairable."""
    now = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    rows = pd.DataFrame(
        [
            # opened 08:00, fetched 08:01 (inside window); window closes 12:00 > now 10:00
            _row("BTCUSDT", "2026-07-19 08:00:00", "2026-07-19 08:01:00"),
        ],
        columns=_COLUMNS,
    )
    repairable, unclosed = classify_partial_bars(rows, now)
    assert len(repairable) == 0
    assert len(unclosed) == 1


# --- PartialBarRepairer (orchestration) ---


@pytest.fixture
def crypto_ingestor(loaded_config: SwingRLConfig, tmp_path: Path) -> BinanceIngestor:
    """Real BinanceIngestor writing Parquet to a tmp data dir (fetch patched per test)."""
    ingestor = BinanceIngestor(loaded_config)
    ingestor._data_dir = tmp_path
    return ingestor


def test_dry_run_reports_without_mutating() -> None:
    """Default dry-run reports the partial but never deletes/inserts or refetches."""
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)
    gateway = _FakeGateway(
        [
            _row("BTCUSDT", "2026-07-19 00:00:00", "2026-07-19 00:01:00", close=100.0),
            _row("BTCUSDT", "2026-07-19 04:00:00", "2026-07-19 08:05:00", close=200.0),
        ]
    )
    ingestor = MagicMock()
    repairer = PartialBarRepairer(gateway, ingestor, now_fn=lambda: now)

    records = repairer.run(apply=False)

    assert gateway.deleted == []
    assert gateway.inserted == []
    ingestor.fetch.assert_not_called()
    would = [r for r in records if r.status == "would_repair"]
    assert len(would) == 1
    assert would[0].bar_open == pd.Timestamp("2026-07-19 00:00:00", tz="UTC")


def test_apply_repairs_partial_and_rewrites_parquet(crypto_ingestor: BinanceIngestor) -> None:
    """--apply deletes the partial, reinserts the refetched completed bar, rewrites Parquet."""
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)
    bar_open = pd.Timestamp("2026-07-19 00:00:00", tz="UTC")

    gateway = _FakeGateway(
        [
            _row("BTCUSDT", "2026-07-19 00:00:00", "2026-07-19 00:01:00", close=100.0),
            _row("BTCUSDT", "2026-07-19 04:00:00", "2026-07-19 08:05:00", close=200.0),
        ]
    )

    # Seed the Parquet with the partial value (close=100) at the bad bar's open time.
    partial_parquet = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000.0]},
        index=pd.DatetimeIndex([bar_open], name="timestamp"),
    )
    crypto_ingestor.store(partial_parquet, "BTCUSDT")

    # Refetch returns the COMPLETED bar (corrected close=555) at the same open time.
    corrected = pd.DataFrame(
        {"open": [500.0], "high": [560.0], "low": [495.0], "close": [555.0], "volume": [9000.0]},
        index=pd.DatetimeIndex([bar_open], name="timestamp"),
    )
    crypto_ingestor.fetch = MagicMock(return_value=corrected)  # type: ignore[method-assign]

    repairer = PartialBarRepairer(gateway, crypto_ingestor, now_fn=lambda: now)
    records = repairer.run(apply=True)

    # DB: partial deleted, corrected bar reinserted.
    assert ("BTCUSDT", bar_open) in gateway.deleted
    assert len(gateway.inserted) == 1
    ins_symbol, ins_open, ins_values, _ins_fetched = gateway.inserted[0]
    assert ins_symbol == "BTCUSDT"
    assert ins_open == bar_open
    assert ins_values["close"] == pytest.approx(555.0)

    # Parquet rewritten to agree with the corrected value.
    written = pd.read_parquet(crypto_ingestor._parquet_path("BTCUSDT"))
    assert written.loc[bar_open, "close"] == pytest.approx(555.0)

    # The good bar (04:00) is untouched.
    good = gateway.df[gateway.df["datetime"] == pd.Timestamp("2026-07-19 04:00:00", tz="UTC")]
    assert len(good) == 1
    assert good.iloc[0]["close"] == pytest.approx(200.0)

    repaired = [r for r in records if r.status == "repaired"]
    assert len(repaired) == 1


def test_second_apply_is_noop(crypto_ingestor: BinanceIngestor) -> None:
    """After a repair, a second --apply run finds nothing to fix (idempotent)."""
    now = datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)
    bar_open = pd.Timestamp("2026-07-19 00:00:00", tz="UTC")
    gateway = _FakeGateway(
        [_row("BTCUSDT", "2026-07-19 00:00:00", "2026-07-19 00:01:00", close=100.0)]
    )
    corrected = pd.DataFrame(
        {"open": [500.0], "high": [560.0], "low": [495.0], "close": [555.0], "volume": [9000.0]},
        index=pd.DatetimeIndex([bar_open], name="timestamp"),
    )
    crypto_ingestor.fetch = MagicMock(return_value=corrected)  # type: ignore[method-assign]
    repairer = PartialBarRepairer(gateway, crypto_ingestor, now_fn=lambda: now)

    repairer.run(apply=True)
    deletes_after_first = len(gateway.deleted)
    inserts_after_first = len(gateway.inserted)

    records = repairer.run(apply=True)

    assert len(gateway.deleted) == deletes_after_first
    assert len(gateway.inserted) == inserts_after_first
    assert not [r for r in records if r.status == "repaired"]


def test_unclosed_window_refused_on_apply(crypto_ingestor: BinanceIngestor) -> None:
    """Even with --apply, a bar whose window has not closed is refused, not repaired."""
    now = datetime(2026, 7, 19, 10, 0, 0, tzinfo=UTC)
    gateway = _FakeGateway(
        [_row("BTCUSDT", "2026-07-19 08:00:00", "2026-07-19 08:01:00", close=100.0)]
    )
    crypto_ingestor.fetch = MagicMock()  # type: ignore[method-assign]
    repairer = PartialBarRepairer(gateway, crypto_ingestor, now_fn=lambda: now)

    records = repairer.run(apply=True)

    assert gateway.deleted == []
    assert gateway.inserted == []
    crypto_ingestor.fetch.assert_not_called()
    refused = [r for r in records if r.status == "refused_window_open"]
    assert len(refused) == 1
