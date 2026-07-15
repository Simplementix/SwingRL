# tests/test_options_collector.py
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.data.options.chain_parser import ParsedChain
from swingrl.data.options.collector import OptionsCollector, check_schema_drift
from swingrl.utils.exceptions import DataError


def _cfg() -> SwingRLConfig:
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY", "QQQ"]
    cfg.options_collector.index_symbols = ["_SPX"]
    cfg.options_collector.include_equity_symbols = True
    return cfg


def _raw(symbol: str, n: int = 3) -> dict:
    contract = {
        "option": "SPXW260724C07500000",
        "bid": 1.0,
        "ask": 1.1,
        "bid_size": 1,
        "ask_size": 1,
        "iv": 0.2,
        "open_interest": 5,
        "volume": 1,
        "delta": 0.5,
        "gamma": 0.0,
        "theta": -0.1,
        "vega": 0.2,
        "rho": 0.1,
    }
    return {
        "timestamp": "2026-07-14 20:00:00",
        "symbol": symbol,
        "data": {"current_price": 100.0, "options": [dict(contract) for _ in range(n)]},
    }


def _collector(client, store) -> tuple[OptionsCollector, MagicMock]:
    alerter = MagicMock()
    return OptionsCollector(_cfg(), client, store, alerter=alerter), alerter


def _store_mock() -> MagicMock:
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False
    store.last_snapshot_row_count.return_value = None
    return store


def test_symbols_combines_index_and_equity() -> None:
    """OPT-COLLECT-5: symbols = index + equity when enabled (spec §5)."""
    c, _ = _collector(MagicMock(), _store_mock())
    assert c.symbols() == ["_SPX", "SPY", "QQQ"]


def test_per_symbol_isolation_one_fails_others_succeed() -> None:
    """OPT-COLLECT-6: one symbol failing does not abort the rest (spec §10.2)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(DataError("boom")) if s == "SPY" else _raw(s)
    )
    c, _ = _collector(client, _store_mock())
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert "SPY" in result.failed
    assert set(result.succeeded) == {"_SPX", "QQQ"}


def test_per_symbol_isolation_non_dataerror() -> None:
    """OPT-COLLECT-13: a non-DataError (e.g. transport/pool error) is isolated too (C4)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(RuntimeError("cdn down")) if s == "SPY" else _raw(s)
    )
    c, _ = _collector(client, _store_mock())
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert "SPY" in result.failed
    assert set(result.succeeded) == {"_SPX", "QQQ"}


def test_postgres_sync_failure_is_warning_not_failure() -> None:
    """OPT-COLLECT-14: Parquet landed but DB sync failed -> succeeded + warning (C4)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    store = _store_mock()
    store.sync_to_postgres.side_effect = RuntimeError("pg pool exhausted")
    c, _ = _collector(client, store)
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert set(result.succeeded) == {"_SPX", "SPY", "QQQ"}
    assert not result.failed
    assert any("postgres sync failed" in w for w in result.warnings)


def test_skip_already_captured() -> None:
    """OPT-COLLECT-7: existing Parquet snapshot is skipped (spec §10.1)."""
    store = _store_mock()
    store.snapshot_exists_parquet.return_value = True
    client = MagicMock()
    c, _ = _collector(client, store)
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    client.get_option_chain.assert_not_called()
    assert set(result.skipped) == {"_SPX", "SPY", "QQQ"}


def test_all_symbols_fail_is_critical() -> None:
    """OPT-COLLECT-8: every symbol failing -> CRITICAL summary (spec §10.4)."""
    client = MagicMock()
    client.get_option_chain.side_effect = DataError("boom")
    c, alerter = _collector(client, _store_mock())
    c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert any(call.args[0] == "critical" for call in alerter.send_alert.call_args_list)


def test_schema_drift_detected() -> None:
    """OPT-COLLECT-9: missing expected field flagged (spec §10.5)."""
    raw = {"data": {"options": [{"option": "A", "bid": 1.0}]}}
    missing = check_schema_drift(raw)
    assert "delta" in missing and "open_interest" in missing


def test_contract_count_drop_warns() -> None:
    """OPT-COLLECT-10: >50% contract-count drop vs previous snapshot -> WARNING (§17 C1)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s, n=2)
    store = _store_mock()
    store.last_snapshot_row_count.return_value = 100  # previous run had 100 rows
    c, _ = _collector(client, store)
    result = c.run_snapshot("eod", now=datetime(2026, 7, 14, 20, 35, tzinfo=UTC))
    assert any("count" in w for w in result.warnings)


def test_late_decision_fire_warns_and_stamps() -> None:
    """OPT-COLLECT-11: decision fired past schedule -> late_by_s stamped + WARNING (D8)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    store = _store_mock()
    c, _ = _collector(client, store)
    result = c.run_snapshot(
        "decision",
        now=datetime(2026, 7, 14, 20, 10, tzinfo=UTC),  # fired 16:10 ET
        scheduled_pull_utc=datetime(2026, 7, 14, 20, 0, tzinfo=UTC),  # scheduled 16:00 ET
    )
    assert any("late" in w for w in result.warnings)
    # late_by_s reaches the stored header via parse_chain(late_by_s=600)
    _, kwargs = store.write_snapshot.call_args
    parsed: ParsedChain = store.write_snapshot.call_args.args[0]
    assert parsed.header["raw_header"]["late_by_s"] == 600.0


def test_unknown_snapshot_label_raises_data_error() -> None:
    """OPT-COLLECT-12: unknown snapshot_label raises typed DataError, not bare
    StopIteration (repo convention: never raise bare exceptions)."""
    c, _ = _collector(MagicMock(), _store_mock())
    with pytest.raises(DataError, match="bogus"):
        c.run_snapshot("bogus", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
