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
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager
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


def _snapshot_job_id(label: str) -> str:
    """Stable APScheduler job id for a snapshot label."""
    return f"options_{label}_snapshot"


def all_job_ids(config: SwingRLConfig) -> list[str]:
    """One snapshot job per configured snapshot + the fixed jobs (D4)."""
    return [_snapshot_job_id(s.label) for s in config.options_collector.snapshots] + FIXED_JOB_IDS


def _hhmm(time_et: str) -> tuple[int, int]:
    hour, minute = (int(x) for x in time_et.split(":"))
    return hour, minute


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


def run_health_check(
    config: SwingRLConfig,
    collector: OptionsCollector,
    store: OptionsStore,
    alerter: Alerter,
    now: datetime | None = None,
) -> None:
    """Verify snapshots over the last health_lookback_days sessions (D9 lookback)."""
    now = now or datetime.now(UTC)
    as_of = now.astimezone(_ET).date()
    sessions = market_calendar.recent_sessions(as_of, config.options_collector.health_lookback_days)
    symbols = collector.symbols()
    for session in sessions:
        for snap in config.options_collector.snapshots:
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
    """D9 boot trio: reconcile unsynced Parquet + lookback health check, every start."""
    components["store"].reconcile()
    run_health_check(
        components["config"],
        components["collector"],
        components["store"],
        components["alerter"],
    )
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
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.error("options_offsite_backup_failed", error=str(exc))
        if alerter is not None:
            alerter.send_alert("warning", "Options offsite backup failed", str(exc))


def register_jobs(scheduler: Any, components: dict[str, Any]) -> None:
    """Register per-snapshot + fixed cron jobs on the scheduler (D4, D8/D9)."""
    config: SwingRLConfig = components["config"]
    oc = config.options_collector
    collector = components["collector"]
    store = components["store"]
    alerter = components["alerter"]
    db = components["db"]

    for snap in oc.snapshots:
        sh, sm = _hhmm(snap.pull_time_et)

        def _job(label: str = snap.label, pull: str = snap.pull_time_et) -> None:
            now = datetime.now(UTC)
            guarded_snapshot(
                collector, label, scheduled_pull_utc=_scheduled_pull_utc(pull, now), now=now
            )

        scheduler.add_job(
            _job,
            trigger="cron",
            day_of_week="mon-fri",
            hour=sh,
            minute=sm,
            timezone="America/New_York",
            id=_snapshot_job_id(snap.label),
            misfire_grace_time=snap.misfire_grace_s,
            replace_existing=True,
        )

    hh, hm = _hhmm(oc.health_check_time_et)
    scheduler.add_job(
        run_health_check,
        trigger="cron",
        day_of_week="mon-fri",
        hour=hh,
        minute=hm,
        timezone="America/New_York",
        args=[config, collector, store, alerter],
        id="options_health_check",
        replace_existing=True,
    )
    ah, am = _hhmm(oc.integrity.audit_time_et)
    scheduler.add_job(
        run_data_quality_audit,
        trigger="cron",
        day=oc.integrity.audit_day_of_month,
        hour=ah,
        minute=am,
        timezone="America/New_York",
        kwargs={"config": config, "db": db, "alerter": alerter},
        id="options_data_audit",
        replace_existing=True,
    )
    bh, bm = _hhmm(oc.backup.time_et)
    scheduler.add_job(
        run_offsite_backup,
        trigger="cron",
        hour=bh,
        minute=bm,
        timezone="America/New_York",
        args=[config, alerter],
        id="options_offsite_backup",
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
    log.info("options_collector_started", jobs=all_job_ids(components["config"]))
    stop_event.wait()
    log.info("options_collector_exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
