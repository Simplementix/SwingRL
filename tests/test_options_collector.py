# tests/test_options_collector.py
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.data.options.chain_parser import ParsedChain
from swingrl.data.options.collector import OptionsCollector, check_schema_drift
from swingrl.monitoring.alerter import Alerter
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


def test_late_fire_within_tolerance_no_warning() -> None:
    """OPT-COLLECT-18: late_by_s within the default 30s cron-jitter tolerance -> no
    lateness warning (user design 2026-07-16; the old code warned on any late_by_s > 0,
    which fired on routine millisecond cron jitter)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    c, _ = _collector(client, _store_mock())
    result = c.run_snapshot(
        "decision",
        now=datetime(2026, 7, 14, 20, 0, 5, tzinfo=UTC),  # 5s late
        scheduled_pull_utc=datetime(2026, 7, 14, 20, 0, 0, tzinfo=UTC),
    )
    assert not any("late" in w for w in result.warnings)


def test_late_fire_beyond_tolerance_warns() -> None:
    """OPT-COLLECT-19: late_by_s beyond the default 30s tolerance still warns, and the
    warning text includes the actual lateness in seconds (user design 2026-07-16)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    c, _ = _collector(client, _store_mock())
    result = c.run_snapshot(
        "decision",
        now=datetime(2026, 7, 14, 20, 0, 45, tzinfo=UTC),  # 45s late
        scheduled_pull_utc=datetime(2026, 7, 14, 20, 0, 0, tzinfo=UTC),
    )
    assert any("late" in w and "45" in w for w in result.warnings)


def test_captured_summary_single_alert_with_inline_warning() -> None:
    """OPT-COLLECT-15: all succeeded + one warning -> exactly ONE alerter call: an info
    'captured' message carrying both the succeeded list and the warning text inline.
    No separate 'completed with issues' call — warnings must never suppress or replace
    the captured message (user design 2026-07-16)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: _raw(s)
    store = _store_mock()

    def _sync(parsed: ParsedChain) -> None:
        if parsed.header["underlying_symbol"] == "SPY":
            raise RuntimeError("pg pool exhausted")

    store.sync_to_postgres.side_effect = _sync
    c, alerter = _collector(client, store)
    result = c.run_snapshot("eod", now=datetime(2026, 7, 14, 20, 35, tzinfo=UTC))

    assert set(result.succeeded) == {"_SPX", "SPY", "QQQ"}
    assert not result.failed
    assert len(result.warnings) == 1

    assert alerter.send_alert.call_count == 1
    level, title, message = alerter.send_alert.call_args.args[:3]
    assert level == "info"
    assert "captured" in title
    assert "SPY" in message  # succeeded list present
    assert "postgres sync failed" in message  # warning folded in inline


def test_mixed_success_and_failure_sends_both_alerts() -> None:
    """OPT-COLLECT-16: some failed + some succeeded -> BOTH an info 'captured' alert
    (successes) and a warning 'completed with issues' alert (failures) fire
    (user design 2026-07-16)."""
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(DataError("boom")) if s == "SPY" else _raw(s)
    )
    c, alerter = _collector(client, _store_mock())
    result = c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))

    assert "SPY" in result.failed
    assert set(result.succeeded) == {"_SPX", "QQQ"}

    assert alerter.send_alert.call_count == 2
    calls_by_level = {call.args[0]: call.args for call in alerter.send_alert.call_args_list}
    assert set(calls_by_level) == {"info", "warning"}

    info_title, info_message = calls_by_level["info"][1], calls_by_level["info"][2]
    assert "captured" in info_title
    assert "QQQ" in info_message and "_SPX" in info_message

    warning_title, warning_message = calls_by_level["warning"][1], calls_by_level["warning"][2]
    assert "completed with issues" in warning_title
    assert "SPY" in warning_message


def test_all_failed_sends_critical_only() -> None:
    """OPT-COLLECT-17: all-attempted-failed -> exactly one CRITICAL alert, no info or
    warning fallback (pinned regression guard, unchanged by the 2026-07-16 routing fix)."""
    client = MagicMock()
    client.get_option_chain.side_effect = DataError("boom")
    c, alerter = _collector(client, _store_mock())
    c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))
    assert alerter.send_alert.call_count == 1
    assert alerter.send_alert.call_args.args[0] == "critical"


def test_capture_failure_warning_bypasses_suppression_reaches_webhook(mocker: Any) -> None:
    """OPT-COLLECT-20 (2026-07-16, user-directed scope addition): with a REAL Alerter
    gated at consecutive_failures_before_alert=3, a single some-failed capture run still
    reaches the webhook on the FIRST occurrence. A capture failure is same-day,
    un-backfillable data loss (spec §13) -- it must not wait for 3 consecutive identical
    days like a routine warning would. Uses a real Alerter (not a MagicMock) so the
    suppression gate is actually exercised, not just recorded as "called"."""
    mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
    mock_post.return_value = MagicMock(status_code=204)
    mock_post.return_value.raise_for_status = MagicMock()

    # info_immediate=True matches the collector's real production wiring (PR #23) --
    # otherwise the info "captured" message buffers for a daily digest instead of
    # posting, which would be a different bug than the one under test here.
    real_alerter = Alerter(
        webhook_url="https://discord.com/api/webhooks/test/token",
        cooldown_minutes=30,
        consecutive_failures_before_alert=3,
        info_immediate=True,
    )
    client = MagicMock()
    client.get_option_chain.side_effect = lambda s: (
        (_ for _ in ()).throw(DataError("boom")) if s == "SPY" else _raw(s)
    )
    c = OptionsCollector(_cfg(), client, _store_mock(), alerter=real_alerter)

    c.run_snapshot("decision", now=datetime(2026, 7, 14, 20, 0, tzinfo=UTC))

    # Two posts on the FIRST run: the info "captured" (successes) and the warning
    # (failures) -- neither suppressed, despite threshold=3.
    assert mock_post.call_count == 2
    titles = [call[1]["json"]["embeds"][0]["title"] for call in mock_post.call_args_list]
    assert any("captured" in t for t in titles)
    assert any("completed with issues" in t for t in titles)
