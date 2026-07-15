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


def _parsed_n(n: int) -> ParsedChain:
    """Same shape as _parsed() but with n contract rows (drives last_snapshot_row_count)."""
    df = pd.DataFrame(
        [
            {
                "contract_symbol": f"SPXW260718C0500000{i}",
                "strike": 5000.0 + i,
                "iv": 12.3,
                "raw_json": {"symbol": f"SPXW260718C0500000{i}", "strikePrice": 5000.0 + i},
            }
            for i in range(n)
        ]
    )
    header = {
        "underlying_symbol": "_SPX",
        "quote_date": date(2026, 7, 14),
        "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "number_of_contracts": n,
        "is_early_close": False,
        "raw_header": {"symbol": "_SPX", "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=df)


def test_db_value_maps_nat_to_none(tmp_path: Path) -> None:
    """OPT-STORE-8: a Parquet-roundtripped missing trade_time_utc (NaT) -> SQL NULL (C3)."""
    store = _store(tmp_path)
    df = pd.DataFrame(
        {
            "contract_symbol": ["A", "B"],
            "trade_time_utc": pd.to_datetime(
                [datetime(2026, 7, 14, 19, 59, 7, tzinfo=UTC), None], utc=True
            ),
            "raw_json": [{"x": 1}, {"x": 2}],
        }
    )
    header = {
        "underlying_symbol": "_SPX",
        "quote_date": date(2026, 7, 14),
        "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "number_of_contracts": 2,
        "is_early_close": False,
        "raw_header": {"symbol": "_SPX", "status": "SUCCESS"},
    }
    store.write_snapshot(ParsedChain(header=header, contracts=df), "_SPX", date(2026, 7, 14), "eod")
    back = store.read_snapshot("_SPX", date(2026, 7, 14), "eod")
    nat_value = back.contracts.iloc[1]["trade_time_utc"]
    assert pd.isna(nat_value)  # the roundtrip really does yield NaT, not None
    assert OptionsStore._db_value("trade_time_utc", nat_value) is None


def test_reconcile_isolates_a_bad_file(tmp_path: Path, monkeypatch) -> None:
    """OPT-STORE-9: a missing sidecar is skipped; other valid snapshots still load (C3)."""
    from contextlib import contextmanager

    class _FakeDB:
        @contextmanager
        def connection(self):
            yield object()

    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
    store = OptionsStore(cfg, db=_FakeDB())
    store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")
    store.write_snapshot(_parsed_n(2), "_SPX", date(2026, 7, 15), "decision")
    # Corrupt one snapshot: delete its header sidecar -> _read_header raises for it.
    store.header_path("_SPX", date(2026, 7, 15), "decision").unlink()

    written: list = []
    monkeypatch.setattr(store, "snapshot_exists_db", lambda conn, s, d, label: False)
    monkeypatch.setattr(store, "_write_db", lambda conn, parsed: written.append(parsed))
    loaded = store.reconcile()
    assert loaded == 1
    assert len(written) == 1


def test_last_snapshot_row_count_none_when_missing(tmp_path: Path) -> None:
    """OPT-STORE-6: no matching Parquet on disk -> None (spec §8.1)."""
    store = _store(tmp_path)
    assert store.last_snapshot_row_count("_SPX", "decision") is None


def test_last_snapshot_row_count_uses_latest_date_and_matches_label(tmp_path: Path) -> None:
    """OPT-STORE-7: returns the LATEST date's row count for the label; other
    labels (even same date) are not matched (spec §8.1, §17 C1)."""
    store = _store(tmp_path)
    store.write_snapshot(_parsed_n(1), "_SPX", date(2026, 7, 14), "decision")
    store.write_snapshot(_parsed_n(2), "_SPX", date(2026, 7, 15), "decision")
    # Same date as the latest "decision" file, but a different label — must not match.
    store.write_snapshot(_parsed_n(5), "_SPX", date(2026, 7, 15), "eod")

    assert store.last_snapshot_row_count("_SPX", "decision") == 2
    assert store.last_snapshot_row_count("_SPX", "eod") == 5
