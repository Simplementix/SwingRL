# scripts/collector_main.py
"""Standalone market-data collector container entrypoint (spec §9, §17 C4).

Its OWN scheduler + jobstore. Never touches the trader (A30). No auth (C1).
Plan A Task 11's calendar-ingest jobs register here when that task lands (D10).
"""

from __future__ import annotations

import argparse
import signal
import subprocess  # nosec B404
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from swingrl.config.schema import OptionsSnapshotConfig, SwingRLConfig, load_config
from swingrl.data.calendar import CalendarIngestor, run_calendar_staleness_check
from swingrl.data.db import DatabaseManager
from swingrl.data.ingest_all import (
    detect_and_fill_crypto_gaps,
    run_crypto,
    run_equity,
    run_features,
)
from swingrl.data.options import market_calendar
from swingrl.data.options.audit import run_data_quality_audit
from swingrl.data.options.cboe_client import CboeChainClient
from swingrl.data.options.collector import OptionsCollector
from swingrl.data.options.store import OptionsStore
from swingrl.monitoring.alerter import Alerter
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

FIXED_JOB_IDS = [
    "options_health_check",
    "options_data_audit",
    "options_offsite_backup",
]

# Calendar-ingest jobs (Plan A Task 11, amended 2026-07-14): homed here so calendar-code
# updates never require a trader rebuild (A30/D10). Registered only when calendar.enabled.
CALENDAR_JOB_IDS = ["calendar_ingest", "calendar_staleness"]

# Candle-ingestion jobs (USER RULING 2026-07-18): the collector owns OHLCV freshness while
# training is paused, so the paper trader never reads stale bars. Registered only when
# options_collector.candle_jobs.enabled.
CANDLE_JOB_IDS = ["candles_equity", "candles_crypto"]


def _snapshot_job_id(label: str) -> str:
    """Stable APScheduler job id for a snapshot label."""
    return f"options_{label}_snapshot"


def all_job_ids(config: SwingRLConfig) -> list[str]:
    """One snapshot job per configured snapshot + the fixed jobs (D4).

    The two calendar jobs join the keep-set only when ``calendar.enabled`` — so a disabled
    calendar drops any previously persisted calendar job via ``remove_stale_jobs``.
    """
    ids = [_snapshot_job_id(s.label) for s in config.options_collector.snapshots] + FIXED_JOB_IDS
    if config.calendar.enabled:
        ids += CALENDAR_JOB_IDS
    if config.options_collector.candle_jobs.enabled:
        ids += CANDLE_JOB_IDS
    return ids


def _hhmm(time_et: str) -> tuple[int, int]:
    hour, minute = (int(x) for x in time_et.split(":"))
    return hour, minute


# Process-global component registry (C1). A persistent SQLAlchemy jobstore can only
# pickle a job's *func reference + primitive args* — never live objects that hold psycopg
# pools. So the scheduler jobs are module-level functions that resolve their heavy
# components from here at run time, mirroring the trader's arg-less job pattern
# (scripts/main.py). This is what makes the jobs serializable.
_components: dict[str, Any] | None = None


def set_components(components: dict[str, Any]) -> None:
    """Publish the built components for the module-level jobs to resolve (C1)."""
    global _components
    _components = components


def get_components() -> dict[str, Any]:
    """Return the components registry; raise if main() has not wired it yet (C1)."""
    if _components is None:
        log.error("options_collector_components_unset")
        raise RuntimeError("collector components not initialized — call set_components() first")
    return _components


def build_app(config_path: str) -> dict[str, Any]:
    """Load config and wire logging, DB, alerter, and all collector components."""
    config = load_config(config_path)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    db = DatabaseManager(config)
    alerter = Alerter(
        webhook_url=config.alerting.alerts_webhook_url,
        alerts_webhook_url=config.alerting.alerts_webhook_url,
        daily_webhook_url=config.alerting.daily_webhook_url,
        cooldown_minutes=config.alerting.alert_cooldown_minutes,
        consecutive_failures_before_alert=config.alerting.consecutive_failures_before_alert,
        db=db,
        # ~2 INFOs/day and no digest-flush job here — buffered INFO would never send.
        info_immediate=True,
    )
    client = CboeChainClient(config.options_collector)
    store = OptionsStore(config.options_collector, db=db)
    collector = OptionsCollector(config, client, store, alerter=alerter)
    return {
        "config": config,
        "db": db,
        "alerter": alerter,
        "client": client,
        "store": store,
        "collector": collector,
    }


def guarded_snapshot(
    collector: OptionsCollector,
    label: str,
    scheduled_pull_utc: datetime | None = None,
    now: datetime | None = None,
) -> None:
    """Run a snapshot only on NYSE trading days; thread the schedule through (D8)."""
    now = now or datetime.now(UTC)
    quote_date = now.astimezone(_ET).date()
    if not market_calendar.is_trading_day(quote_date):
        log.info(
            "options_snapshot_skipped_non_trading_day", label=label, date=quote_date.isoformat()
        )
        return
    collector.run_snapshot(label, now=now, scheduled_pull_utc=scheduled_pull_utc)


def _scheduled_pull_utc(pull_time_et: str, now: datetime) -> datetime:
    hh, mm = _hhmm(pull_time_et)
    local = now.astimezone(_ET)
    return local.replace(hour=hh, minute=mm, second=0, microsecond=0).astimezone(UTC)


def _snapshot_due(snap: OptionsSnapshotConfig, now_et: datetime) -> bool:
    """True once today's pull_time_et + misfire grace has elapsed (I1 alarm-fatigue guard).

    On a trading-day boot before a label's window closes, that label is not yet MISSED —
    it may still fire. Only past sessions and already-due labels get health-checked.
    """
    hh, mm = _hhmm(snap.pull_time_et)
    deadline = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(
        seconds=snap.misfire_grace_s
    )
    return now_et >= deadline


def run_health_check(
    config: SwingRLConfig,
    collector: OptionsCollector,
    store: OptionsStore,
    alerter: Alerter,
    now: datetime | None = None,
) -> None:
    """Verify snapshots over the last health_lookback_days sessions (D9 lookback)."""
    now = now or datetime.now(UTC)
    now_et = now.astimezone(_ET)
    as_of = now_et.date()
    sessions = market_calendar.recent_sessions(as_of, config.options_collector.health_lookback_days)
    symbols = collector.symbols()
    for session in sessions:
        for snap in config.options_collector.snapshots:
            # I1: today's labels that aren't due yet are not "missed" — skip to avoid
            # false CRITICALs on every daytime restart. Past sessions are unaffected.
            if session == as_of and not _snapshot_due(snap, now_et):
                continue
            present = [s for s in symbols if store.snapshot_exists_parquet(s, session, snap.label)]
            if not present:
                alerter.send_alert(
                    "critical",
                    f"Options {snap.label} MISSED",
                    f"No {snap.label} snapshot for any symbol on {session.isoformat()}.",
                )
            elif len(present) < len(symbols):
                missing = [s for s in symbols if s not in present]
                alerter.send_alert(
                    "warning",
                    f"Options {snap.label} incomplete",
                    f"Missing {snap.label} for {missing} on {session.isoformat()}.",
                )


def boot_self_check(components: dict[str, Any]) -> None:
    """D9 boot trio: reconcile unsynced Parquet + lookback health check, every start.

    Non-fatal (C3): a failure in either step is logged (and alerted when an alerter is
    present) but must never stop the scheduler from starting — a crash-looping collector
    captures nothing, which is the worst outcome for un-backfillable data.
    """
    alerter = components.get("alerter")
    try:
        components["store"].reconcile()
    except Exception as exc:
        log.error("options_boot_reconcile_failed", error=str(exc))
        if alerter is not None:
            alerter.send_alert("warning", "Options boot reconcile failed", str(exc))
    try:
        run_health_check(
            components["config"],
            components["collector"],
            components["store"],
            components["alerter"],
        )
    except Exception as exc:
        log.error("options_boot_health_check_failed", error=str(exc))
        if alerter is not None:
            alerter.send_alert("warning", "Options boot health check failed", str(exc))
    log.info("options_boot_self_check_done")


def run_offsite_backup(config: SwingRLConfig, alerter: Alerter | None = None) -> None:
    """Sync captured data offsite via rclone (3-2-1 backup; spec §13)."""
    backup = config.options_collector.backup
    if not backup.enabled:
        return
    cmd = ["rclone", "sync", config.options_collector.output_dir, backup.rclone_remote]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("options_offsite_backup_ok", remote=backup.rclone_remote)
    except Exception as exc:
        log.error("options_offsite_backup_failed", error=str(exc))
        if alerter is not None:
            alerter.send_alert("warning", "Options offsite backup failed", str(exc))


def snapshot_job(label: str, pull_time_et: str) -> None:
    """Picklable snapshot job (C1): resolve components, run a guarded snapshot (D8)."""
    components = get_components()
    now = datetime.now(UTC)
    guarded_snapshot(
        components["collector"],
        label,
        scheduled_pull_utc=_scheduled_pull_utc(pull_time_et, now),
        now=now,
    )


def health_check_job() -> None:
    """Picklable health-check job (C1): resolve components, run the lookback check (D9)."""
    components = get_components()
    run_health_check(
        components["config"],
        components["collector"],
        components["store"],
        components["alerter"],
    )


def data_audit_job() -> None:
    """Picklable monthly data-quality audit job (C1)."""
    components = get_components()
    run_data_quality_audit(
        config=components["config"],
        db=components["db"],
        alerter=components["alerter"],
    )


def offsite_backup_job() -> None:
    """Picklable offsite-backup job (C1)."""
    components = get_components()
    run_offsite_backup(components["config"], components["alerter"])


def calendar_ingest_job() -> None:
    """Picklable weekly calendar-ingest job (C1): FRED release dates + FOMC yaml (Task 11)."""
    components = get_components()
    config = components["config"]
    if not config.calendar.enabled:
        return
    CalendarIngestor(config, components["db"]).run()


def calendar_staleness_job() -> None:
    """Picklable daily calendar-staleness job (C1): warn when the forward calendar runs dry."""
    components = get_components()
    run_calendar_staleness_check(components["config"], components["db"], components["alerter"])


def candles_equity_job() -> None:
    """Picklable equity-candle job (C1): incremental daily OHLCV ingest, then features.

    Owns equity candle freshness while training is paused so the paper trader never reads
    stale bars (USER RULING 2026-07-18). ``run_features`` recomputes both envs (no env-scoped
    variant), so it runs only when new rows landed. Sends an INFO alert on every success
    (rows added) mirroring the options snapshot "captured" pattern (USER RULING 2026-07-19).
    Never raises — an ingestion failure is logged and alerted (the collector sends its own
    alerts) but the scheduler must survive.
    """
    components = get_components()
    config = components["config"]
    try:
        rows = run_equity(config, backfill=False)
        if rows > 0:
            run_features(config)
        log.info("candles_equity_job_done", rows_added=rows, features_ran=rows > 0)
        # USER RULING 2026-07-19: mirror the options snapshot "captured" INFO (D9-style
        # success visibility) — every successful ingest, not just failures, reaches
        # Discord. Own inner try: an alerter exception here is a notification failure,
        # not an ingest failure, so it must not fall into the except below and get
        # reported as one.
        try:
            components["alerter"].send_alert(
                "info", "Equity candles ingested", f"rows_added={rows}"
            )
        except Exception as exc:
            log.error("candles_equity_info_alert_failed", error=str(exc))
    except Exception as exc:
        log.error("candles_equity_job_failed", error=str(exc))
        components["alerter"].send_alert("warning", "Equity candle ingestion failed", str(exc))


def candles_crypto_job() -> None:
    """Picklable crypto-candle job (C1): incremental 4H OHLCV ingest, gap-fill, then features.

    Fires a minute past each 4H UTC bar close, ahead of the trader's :05 crypto cycles, so the
    trader reads fresh bars (USER RULING 2026-07-18). Gap-fill runs between ingest and features;
    ``run_features`` recomputes both envs (no env-scoped variant), so it runs when either new
    rows landed or gaps were filled. Sends an INFO alert on every success (rows added + gaps
    filled) mirroring the options snapshot "captured" pattern (USER RULING 2026-07-19). Never
    raises — a failure is logged + alerted only.
    """
    components = get_components()
    config = components["config"]
    try:
        rows = run_crypto(config, backfill=False)
        gap_results = detect_and_fill_crypto_gaps(config)
        gaps_filled = sum(1 for g in gap_results if g.filled)
        features_ran = rows > 0 or gaps_filled > 0
        if features_ran:
            run_features(config)
        log.info(
            "candles_crypto_job_done",
            rows_added=rows,
            gaps_filled=gaps_filled,
            features_ran=features_ran,
        )
        # USER RULING 2026-07-19: mirror the options snapshot "captured" INFO — see the
        # equity job's identical comment. Own inner try for the same reason.
        try:
            components["alerter"].send_alert(
                "info",
                "Crypto candles ingested",
                f"rows_added={rows} gaps_filled={gaps_filled}",
            )
        except Exception as exc:
            log.error("candles_crypto_info_alert_failed", error=str(exc))
    except Exception as exc:
        log.error("candles_crypto_job_failed", error=str(exc))
        components["alerter"].send_alert("warning", "Crypto candle ingestion failed", str(exc))


def remove_stale_jobs(scheduler: Any, config: SwingRLConfig) -> list[str]:
    """Drop any persisted job whose id is no longer in the desired set (D4).

    Defensive against a non-iterable scheduler (a MagicMock in unit tests): if get_jobs()
    is not a real iterable of jobs, there is nothing persisted to clean up.
    """
    keep = set(all_job_ids(config))
    removed: list[str] = []
    try:
        existing = list(scheduler.get_jobs())
    except TypeError:
        return removed
    for job in existing:
        job_id = getattr(job, "id", None)
        if isinstance(job_id, str) and job_id not in keep:
            scheduler.remove_job(job_id)
            removed.append(job_id)
    if removed:
        log.info("options_stale_jobs_removed", removed=removed)
    return removed


def register_jobs(scheduler: Any, components: dict[str, Any]) -> None:
    """Register per-snapshot + fixed cron jobs (D8/D9).

    Jobs are module-level functions with primitive-only args so the SQLAlchemy jobstore can
    serialize them (C1). Heavy components are resolved from the process registry at run time.

    Stale-job removal (D4) is NOT done here: on a not-yet-started scheduler, get_jobs() does
    not see the persistent jobstore, so a stale job would silently survive. Callers must invoke
    remove_stale_jobs() AFTER scheduler.start() (see main()).
    """
    config: SwingRLConfig = components["config"]
    oc = config.options_collector

    for snap in oc.snapshots:
        sh, sm = _hhmm(snap.pull_time_et)
        scheduler.add_job(
            snapshot_job,
            trigger="cron",
            day_of_week="mon-fri",
            hour=sh,
            minute=sm,
            timezone="America/New_York",
            args=[snap.label, snap.pull_time_et],
            id=_snapshot_job_id(snap.label),
            misfire_grace_time=snap.misfire_grace_s,
            replace_existing=True,
        )

    hh, hm = _hhmm(oc.health_check_time_et)
    scheduler.add_job(
        health_check_job,
        trigger="cron",
        day_of_week="mon-fri",
        hour=hh,
        minute=hm,
        timezone="America/New_York",
        id="options_health_check",
        replace_existing=True,
    )
    ah, am = _hhmm(oc.integrity.audit_time_et)
    scheduler.add_job(
        data_audit_job,
        trigger="cron",
        day=oc.integrity.audit_day_of_month,
        hour=ah,
        minute=am,
        timezone="America/New_York",
        id="options_data_audit",
        replace_existing=True,
    )
    bh, bm = _hhmm(oc.backup.time_et)
    scheduler.add_job(
        offsite_backup_job,
        trigger="cron",
        hour=bh,
        minute=bm,
        timezone="America/New_York",
        id="options_offsite_backup",
        replace_existing=True,
    )

    # Plan A Task 11 (amended 2026-07-14): weekly calendar ingest + daily staleness check.
    cal = config.calendar
    if cal.enabled:
        ih, im = _hhmm(cal.ingest_time_et)
        scheduler.add_job(
            calendar_ingest_job,
            trigger="cron",
            day_of_week=cal.ingest_day_of_week,
            hour=ih,
            minute=im,
            timezone="America/New_York",
            id="calendar_ingest",
            replace_existing=True,
        )
        ch, cm = _hhmm(cal.staleness_check_time_et)
        scheduler.add_job(
            calendar_staleness_job,
            trigger="cron",
            hour=ch,
            minute=cm,
            timezone="America/New_York",
            id="calendar_staleness",
            replace_existing=True,
        )

    # Candle ingestion (USER RULING 2026-07-18): the collector keeps OHLCV bars fresh while
    # training is paused (it used to run before every iteration), so the paper trader never
    # reads stale bars. Existing Alpaca/Binance ingestors — CBOE stays options-only.
    cj = oc.candle_jobs
    if cj.enabled:
        eh, em = _hhmm(cj.equity_time_et)
        scheduler.add_job(
            candles_equity_job,
            trigger="cron",
            day_of_week="mon-fri",
            hour=eh,
            minute=em,
            timezone="America/New_York",
            id="candles_equity",
            misfire_grace_time=cj.equity_misfire_grace_s,
            replace_existing=True,
        )
        scheduler.add_job(
            candles_crypto_job,
            trigger="cron",
            hour="0,4,8,12,16,20",
            minute=cj.crypto_minute,
            timezone="UTC",
            id="candles_crypto",
            misfire_grace_time=cj.crypto_misfire_grace_s,
            replace_existing=True,
        )


def _make_signal_handler(
    scheduler: Any, stop_event: threading.Event
) -> Callable[[int, object], None]:
    def handler(_signum: int, _frame: object) -> None:
        log.info("options_collector_shutting_down")
        scheduler.shutdown(wait=False)
        stop_event.set()

    return handler


def main() -> int:
    """Build, self-check, register jobs, start the scheduler, and block."""
    parser = argparse.ArgumentParser(description="SwingRL market-data collector")
    parser.add_argument("--config", default="config/swingrl.yaml")
    args = parser.parse_args()

    components = build_app(args.config)
    set_components(components)  # C1: module-level jobs resolve heavy components from here
    boot_self_check(components)  # D9: every restart is a self-audit

    scheduler = BackgroundScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{components['config'].options_collector.apscheduler_db_path}"
            )
        },
        executors={"default": ThreadPoolExecutor(max_workers=4)},
        job_defaults={"coalesce": True, "max_instances": 1},
    )
    register_jobs(scheduler, components)

    stop_event = threading.Event()
    handler = _make_signal_handler(scheduler, stop_event)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    scheduler.start()
    # D4: stale-job removal must run AFTER start() — get_jobs() on a not-yet-started
    # scheduler does not see the persistent jobstore, so a stale job (e.g. a renamed
    # snapshot label) would survive and keep firing forever (R1).
    remove_stale_jobs(scheduler, components["config"])
    log.info("options_collector_started", jobs=all_job_ids(components["config"]))
    stop_event.wait()
    log.info("options_collector_exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
