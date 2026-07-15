from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from scripts.collector_main import (
    all_job_ids,
    boot_self_check,
    guarded_snapshot,
    register_jobs,
    run_health_check,
)

from swingrl.config.schema import SwingRLConfig


def _components(cfg: SwingRLConfig | None = None) -> dict:
    return {
        "config": cfg or SwingRLConfig(),
        "collector": MagicMock(),
        "store": MagicMock(),
        "alerter": MagicMock(),
        "db": MagicMock(),
    }


def test_register_jobs_registers_snapshots_plus_fixed() -> None:
    """OPT-SCHED-1: one job per snapshot + 3 fixed jobs, stable ids (D4, §17 C2)."""
    cfg = SwingRLConfig()
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    registered = {call.kwargs["id"] for call in scheduler.add_job.call_args_list}
    assert registered == set(all_job_ids(cfg))
    assert {"options_decision_snapshot", "options_eod_snapshot"} <= registered
    assert len(registered) == 5  # 2 default snapshots + 3 fixed (no token jobs)


def test_snapshot_jobs_use_pull_time_and_per_label_grace() -> None:
    """OPT-SCHED-2: decision fires at 16:00 with 900s grace; eod 16:35 with 18000s (D8/D9)."""
    scheduler = MagicMock()
    register_jobs(scheduler, _components())
    by_id = {c.kwargs["id"]: c.kwargs for c in scheduler.add_job.call_args_list}
    dec = by_id["options_decision_snapshot"]
    assert (dec["hour"], dec["minute"], dec["misfire_grace_time"]) == (16, 0, 900)
    eod = by_id["options_eod_snapshot"]
    assert (eod["hour"], eod["minute"], eod["misfire_grace_time"]) == (16, 35, 18000)


def test_guarded_snapshot_skips_non_trading_day(monkeypatch) -> None:
    """OPT-SCHED-3: holiday/weekend -> run_snapshot NOT called (spec §9.2)."""
    monkeypatch.setattr("scripts.collector_main.market_calendar.is_trading_day", lambda d: False)
    collector = MagicMock()
    guarded_snapshot(collector, "decision", now=datetime(2026, 12, 25, 21, 0, tzinfo=UTC))
    collector.run_snapshot.assert_not_called()


def test_guarded_snapshot_passes_schedule_through(monkeypatch) -> None:
    """OPT-SCHED-4: trading day -> run_snapshot called with scheduled_pull_utc (D8)."""
    monkeypatch.setattr("scripts.collector_main.market_calendar.is_trading_day", lambda d: True)
    collector = MagicMock()
    sched = datetime(2026, 7, 14, 20, 0, tzinfo=UTC)
    guarded_snapshot(
        collector,
        "decision",
        scheduled_pull_utc=sched,
        now=datetime(2026, 7, 14, 20, 1, tzinfo=UTC),
    )
    collector.run_snapshot.assert_called_once()
    assert collector.run_snapshot.call_args.kwargs["scheduled_pull_utc"] == sched


def test_health_check_scans_lookback_days(monkeypatch) -> None:
    """OPT-SCHED-5: a hole YESTERDAY is caught today -> CRITICAL (D9 lookback)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [date(2026, 7, 13), date(2026, 7, 14)],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    # Everything present on the 14th; NOTHING on the 13th.
    store.snapshot_exists_parquet.side_effect = lambda s, d, label: d == date(2026, 7, 14)
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    assert any(c.args[0] == "critical" for c in alerter.send_alert.call_args_list)


def test_health_check_partial_is_warning(monkeypatch) -> None:
    """OPT-SCHED-6: some symbols missing -> WARNING, not CRITICAL (spec §10.4)."""
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [date(2026, 7, 14)],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    store.snapshot_exists_parquet.side_effect = lambda s, d, label: s == "_SPX"
    alerter = MagicMock()
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 14, 21, 15, tzinfo=UTC))
    levels = [c.args[0] for c in alerter.send_alert.call_args_list]
    assert "warning" in levels and "critical" not in levels


def test_boot_self_check_runs_reconcile_and_health(monkeypatch) -> None:
    """OPT-SCHED-7: boot self-check = reconcile + lookback health check (D9)."""
    called = {"health": 0}
    monkeypatch.setattr(
        "scripts.collector_main.run_health_check",
        lambda *a, **k: called.__setitem__("health", called["health"] + 1),
    )
    components = _components()
    boot_self_check(components)
    components["store"].reconcile.assert_called_once()
    assert called["health"] == 1
