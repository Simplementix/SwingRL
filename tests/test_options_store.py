# tests/test_options_store.py
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.chain_parser import ParsedChain
from swingrl.data.options.store import OptionsStore


def _store(tmp_path: Path) -> OptionsStore:
    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
    return OptionsStore(cfg)


def _parsed() -> ParsedChain:
    df = pd.DataFrame(
        [
            {
                "contract_symbol": "SPXW260718C05000000",
                "strike": 5000.0,
                "iv": 12.3,
                "raw_json": {"symbol": "SPXW260718C05000000", "strikePrice": 5000.0},
            }
        ]
    )
    header = {
        "underlying_symbol": "_SPX",
        "quote_date": date(2026, 7, 14),
        "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "number_of_contracts": 1,
        "is_early_close": False,
        "raw_header": {"symbol": "_SPX", "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=df)


def test_symbol_to_dir_strips_prefixes() -> None:
    """OPT-STORE-1: _SPX/$SPX -> dir SPX (spec §5, §17 C1)."""
    assert OptionsStore.symbol_to_dir("_SPX") == "SPX"
    assert OptionsStore.symbol_to_dir("$SPX") == "SPX"
    assert OptionsStore.symbol_to_dir("SPY") == "SPY"


def test_parquet_path_layout(tmp_path: Path) -> None:
    """OPT-STORE-2: one file per (symbol,date,label) (spec §8.1)."""
    p = _store(tmp_path).parquet_path("_SPX", date(2026, 7, 14), "decision")
    assert p.name == "2026-07-14_decision.parquet"
    assert p.parent.name == "SPX"


def test_write_then_exists(tmp_path: Path) -> None:
    """OPT-STORE-3: write makes snapshot_exists_parquet true (spec §10.1)."""
    store = _store(tmp_path)
    assert store.snapshot_exists_parquet("_SPX", date(2026, 7, 14), "decision") is False
    store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    assert store.snapshot_exists_parquet("_SPX", date(2026, 7, 14), "decision") is True


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    """OPT-STORE-4: no .tmp file remains after write (spec §8.1)."""
    store = _store(tmp_path)
    path = store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    assert not path.with_suffix(".parquet.tmp").exists()
    assert list(path.parent.glob("*.tmp")) == []


def test_roundtrip_restores_dicts_and_datetimes(tmp_path: Path) -> None:
    """OPT-STORE-5: read_snapshot restores raw_json dict + header datetimes (spec §8.1)."""
    store = _store(tmp_path)
    store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    back = store.read_snapshot("_SPX", date(2026, 7, 14), "decision")
    assert isinstance(back.contracts.iloc[0]["raw_json"], dict)
    assert back.contracts.iloc[0]["raw_json"]["strikePrice"] == 5000.0
    assert back.header["snapshot_time_utc"] == datetime(2026, 7, 14, 19, 45, tzinfo=UTC)
    assert back.header["raw_header"]["status"] == "SUCCESS"
