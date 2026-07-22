from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from scripts.collector_main import (
    all_job_ids,
    boot_self_check,
    guarded_snapshot,
    register_jobs,
    remove_stale_jobs,
    run_health_check,
    set_components,
    snapshot_job,
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
    # 2 default snapshots + 3 fixed + 2 calendar jobs (Plan A Task 11) + 2 candle jobs
    # (2026-07-18); all four optional jobs enabled by default.
    assert len(registered) == 9


def test_snapshot_jobs_use_pull_time_and_per_label_grace() -> None:
    """OPT-SCHED-2: decision fires at 09:46 with 900s grace; eod 16:35 with 18000s (D8/D9)."""
    scheduler = MagicMock()
    register_jobs(scheduler, _components())
    by_id = {c.kwargs["id"]: c.kwargs for c in scheduler.add_job.call_args_list}
    dec = by_id["options_decision_snapshot"]
    assert (dec["hour"], dec["minute"], dec["misfire_grace_time"]) == (9, 46, 900)
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


def test_boot_self_check_reconcile_failure_is_nonfatal(monkeypatch) -> None:
    """OPT-SCHED-10: reconcile raising must not stop the health check or the boot (C3)."""
    called = {"health": 0}
    monkeypatch.setattr(
        "scripts.collector_main.run_health_check",
        lambda *a, **k: called.__setitem__("health", called["health"] + 1),
    )
    components = _components()
    components["store"].reconcile.side_effect = RuntimeError("db down")
    boot_self_check(components)  # must not raise
    assert called["health"] == 1
    assert any(c.args[0] == "warning" for c in components["alerter"].send_alert.call_args_list)


def test_jobs_are_serializable_in_sqlalchemy_jobstore(tmp_path) -> None:
    """OPT-SCHED-8: real SQLAlchemyJobStore round-trip proves every job pickles (C1).

    This is the regression test for the crash-loop: closures / live-object args could not
    be serialized, so scheduler.start() raised ValueError at first boot.
    """
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.schedulers.background import BackgroundScheduler

    cfg = SwingRLConfig()
    scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{tmp_path / 'jobs.sqlite'}")},
        job_defaults={"coalesce": True, "max_instances": 1},
    )
    # Deliberately register with fakes that could NOT be pickled if they were job args —
    # they are only reachable via the registry, never serialized into the jobstore.
    set_components(
        {
            "config": cfg,
            "collector": object(),
            "store": object(),
            "alerter": object(),
            "db": object(),
        }
    )
    register_jobs(scheduler, {"config": cfg})
    scheduler.start(paused=True)  # flushes + serializes pending jobs to the jobstore
    try:
        ids = {job.id for job in scheduler.get_jobs()}
        assert ids == set(all_job_ids(cfg))
        assert len(ids) == 9  # + 2 calendar (Task 11) + 2 candle jobs (2026-07-18)
    finally:
        scheduler.shutdown(wait=False)


def test_remove_stale_jobs_drops_unknown_ids(tmp_path) -> None:
    """OPT-SCHED-9: a persisted job id no longer in the desired set is removed (D4, R1).

    Models the real production order (main()): a legacy job id is already persisted in the
    jobstore (e.g. from a prior process with a since-renamed snapshot label). register_jobs()
    runs first, then scheduler.start(), then remove_stale_jobs() — matching main(), since
    get_jobs() on a not-yet-started scheduler cannot see the persistent jobstore (R1).
    """
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.schedulers.background import BackgroundScheduler

    cfg = SwingRLConfig()
    scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{tmp_path / 'jobs.sqlite'}")},
        job_defaults={"coalesce": True, "max_instances": 1},
    )
    # Persist a legacy job id directly into the jobstore before anything else runs, standing
    # in for a stale job left behind by a prior process (e.g. a renamed YAML snapshot label).
    scheduler.add_job(
        snapshot_job,
        trigger="cron",
        hour=1,
        args=["decision", "16:00"],
        id="options_LEGACY_snapshot",
        replace_existing=True,
    )

    set_components(_components(cfg))
    register_jobs(scheduler, {"config": cfg})
    scheduler.start()
    try:
        removed = remove_stale_jobs(scheduler, cfg)
        assert removed == ["options_LEGACY_snapshot"]
        ids = {j.id for j in scheduler.get_jobs()}
        assert "options_LEGACY_snapshot" not in ids
        assert ids == set(all_job_ids(cfg))
        assert len(ids) == 9  # + 2 calendar (Task 11) + 2 candle jobs (2026-07-18)
    finally:
        scheduler.shutdown(wait=False)


def test_health_check_skips_today_not_yet_due(monkeypatch) -> None:
    """OPT-SCHED-11: a daytime boot before pull times does not flag today MISSED (I1)."""
    today = date(2026, 7, 14)
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [today],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False  # nothing captured yet today
    alerter = MagicMock()
    # 09:00 ET (EDT) = 13:00 UTC — before every label's pull_time + misfire grace
    # (decision due 09:46+900s=10:01 ET; eod due 16:35+18000s=21:35 ET).
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 14, 13, 0, tzinfo=UTC))
    alerter.send_alert.assert_not_called()


def test_health_check_flags_today_after_due(monkeypatch) -> None:
    """OPT-SCHED-12: once a label's window closes, a same-day hole is CRITICAL (I1 boundary)."""
    today = date(2026, 7, 14)
    monkeypatch.setattr(
        "scripts.collector_main.market_calendar.recent_sessions",
        lambda as_of, n: [today],
    )
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    collector = MagicMock()
    collector.symbols.return_value = ["_SPX", "SPY"]
    store = MagicMock()
    store.snapshot_exists_parquet.return_value = False
    alerter = MagicMock()
    # 21:45 ET on 2026-07-14 = 01:45 UTC 2026-07-15 — past eod pull 16:35 + 18000s grace.
    run_health_check(cfg, collector, store, alerter, now=datetime(2026, 7, 15, 1, 45, tzinfo=UTC))
    assert any(c.args[0] == "critical" for c in alerter.send_alert.call_args_list)


def test_build_app_wires_info_immediate_alerter(tmp_config, monkeypatch) -> None:
    """OPT-SCHED-13: build_app passes info_immediate=True — collector INFOs post at once.

    The collector has no digest-flush job, so buffered INFO alerts would never reach
    Discord (found live at T16 first capture, 2026-07-15).
    """
    import scripts.collector_main as cm

    alerter_cls = MagicMock()
    monkeypatch.setattr(cm, "Alerter", alerter_cls)
    monkeypatch.setattr(cm, "DatabaseManager", MagicMock())
    monkeypatch.setattr(cm, "CboeChainClient", MagicMock())
    monkeypatch.setattr(cm, "OptionsStore", MagicMock())
    monkeypatch.setattr(cm, "OptionsCollector", MagicMock())

    cm.build_app(str(tmp_config))

    assert alerter_cls.call_args.kwargs["info_immediate"] is True
