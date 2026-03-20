"""SwingRL production entrypoint -- APScheduler with cron jobs and stop-price polling.

Initializes all components, registers 12 jobs (6 trading + 3 backup + 1 shadow + 1 trigger +
1 reconciliation), starts crypto stop-price polling daemon thread, and blocks until
SIGTERM/SIGINT.

Usage:
    python scripts/main.py --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb as duckdb_module
import structlog

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.execution.pipeline import ExecutionPipeline
from swingrl.features.pipeline import FeaturePipeline
from swingrl.monitoring.alerter import Alerter
from swingrl.scheduler.halt_check import init_emergency_flags
from swingrl.scheduler.jobs import (
    automated_trigger_check_job,
    crypto_cycle,
    daily_backup_job,
    daily_summary_job,
    equity_cycle,
    init_job_context,
    monthly_macro_job,
    monthly_offsite_job,
    reconciliation_job,
    shadow_promotion_check_job,
    stuck_agent_check_job,
    weekly_duckdb_backup_job,
    weekly_fundamentals_job,
)
from swingrl.scheduler.stop_polling import start_stop_polling_thread
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    from types import FrameType


log = structlog.get_logger(__name__)

# Lazy import to allow mocking in tests
try:
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:  # pragma: no cover
    BackgroundScheduler = None  # type: ignore[assignment,misc]
    SQLAlchemyJobStore = None  # type: ignore[assignment,misc]
    ThreadPoolExecutor = None  # type: ignore[assignment,misc]


def create_scheduler_and_register_jobs(
    scheduler: Any,
    config: Any,
) -> None:
    """Register all 12 jobs on the scheduler (6 trading + 3 backup + 1 shadow + 1 trigger + 1 reconciliation).

    Args:
        scheduler: APScheduler BackgroundScheduler instance.
        config: Validated SwingRLConfig.
    """
    scheduler.add_job(
        equity_cycle,
        trigger="cron",
        hour=16,
        minute=15,
        timezone="America/New_York",
        id="equity_cycle",
        replace_existing=True,
    )

    scheduler.add_job(
        crypto_cycle,
        trigger="cron",
        hour="0,4,8,12,16,20",
        minute=5,
        timezone="UTC",
        id="crypto_cycle",
        replace_existing=True,
    )

    scheduler.add_job(
        daily_summary_job,
        trigger="cron",
        hour=18,
        minute=0,
        timezone="America/New_York",
        id="daily_summary",
        replace_existing=True,
    )

    scheduler.add_job(
        stuck_agent_check_job,
        trigger="cron",
        hour=17,
        minute=30,
        timezone="America/New_York",
        id="stuck_agent_check",
        replace_existing=True,
    )

    scheduler.add_job(
        weekly_fundamentals_job,
        trigger="cron",
        day_of_week="sun",
        hour=18,
        minute=0,
        timezone="America/New_York",
        id="weekly_fundamentals",
        replace_existing=True,
    )

    scheduler.add_job(
        monthly_macro_job,
        trigger="cron",
        day=1,
        hour=18,
        minute=0,
        timezone="America/New_York",
        id="monthly_macro",
        replace_existing=True,
    )

    # Backup jobs (run even when trading is halted)
    scheduler.add_job(
        daily_backup_job,
        trigger="cron",
        hour=2,
        minute=0,
        timezone="America/New_York",
        id="daily_sqlite_backup",
        replace_existing=True,
    )

    scheduler.add_job(
        weekly_duckdb_backup_job,
        trigger="cron",
        day_of_week="sun",
        hour=3,
        minute=0,
        timezone="America/New_York",
        id="weekly_duckdb_backup",
        replace_existing=True,
    )

    scheduler.add_job(
        monthly_offsite_job,
        trigger="cron",
        day=1,
        hour=4,
        minute=0,
        timezone="America/New_York",
        id="monthly_offsite",
        replace_existing=True,
    )

    # Shadow promotion check (daily at 7 PM ET, after daily summary)
    scheduler.add_job(
        shadow_promotion_check_job,
        trigger="cron",
        hour=19,
        minute=0,
        timezone="America/New_York",
        id="shadow_promotion_check",
        replace_existing=True,
    )

    # Automated emergency trigger check (runs even when halted)
    scheduler.add_job(
        automated_trigger_check_job,
        trigger="interval",
        minutes=5,
        id="automated_trigger_check",
        replace_existing=True,
    )

    # Daily equity position reconciliation (5 PM ET, after 4:15 PM equity cycle)
    scheduler.add_job(
        reconciliation_job,
        trigger="cron",
        hour=17,
        minute=0,
        timezone="America/New_York",
        id="daily_reconciliation",
        replace_existing=True,
    )

    log.info("scheduler_jobs_registered", count=12)


def make_signal_handler(
    scheduler: Any,
    stop_event: threading.Event,
    duckdb_conn: Any = None,
) -> Any:
    """Create a signal handler that shuts down the scheduler and sets the stop event.

    Args:
        scheduler: APScheduler BackgroundScheduler instance.
        stop_event: Threading event to unblock main thread.
        duckdb_conn: Optional DuckDB connection to close on shutdown.

    Returns:
        Signal handler callable.
    """

    def handler(signum: int, frame: FrameType | None) -> None:
        sig_name = signal.Signals(signum).name
        log.info("shutdown_signal_received", signal=sig_name)
        scheduler.shutdown(wait=False)
        if duckdb_conn is not None:
            try:
                duckdb_conn.close()
                log.info("duckdb_connection_closed")
            except Exception:  # noqa: BLE001
                log.warning("duckdb_close_failed")
        stop_event.set()

    return handler


def build_app(config_path: str = "config/swingrl.yaml") -> dict[str, Any]:
    """Build all application components without starting the scheduler.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict with scheduler, stop_event, and config for the caller to start.
    """
    config = load_config(config_path)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    log.info(
        "swingrl_starting",
        config_path=config_path,
        trading_mode=config.trading_mode,
    )

    db = DatabaseManager(config)
    init_emergency_flags(db)

    alerter = Alerter(
        webhook_url=config.alerting.alerts_webhook_url,
        alerts_webhook_url=config.alerting.alerts_webhook_url,
        daily_webhook_url=config.alerting.daily_webhook_url,
        cooldown_minutes=config.alerting.alert_cooldown_minutes,
        consecutive_failures_before_alert=config.alerting.consecutive_failures_before_alert,
        db=db,
    )

    duckdb_path = Path(config.system.duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb_conn = duckdb_module.connect(str(duckdb_path))
    feature_pipeline = FeaturePipeline(config, duckdb_conn)

    pipeline = ExecutionPipeline(
        config=config,
        db=db,
        feature_pipeline=feature_pipeline,
        alerter=alerter,
        models_dir=Path(config.paths.models_dir),
    )

    init_job_context(config=config, db=db, pipeline=pipeline, alerter=alerter)

    jobstores = {
        "default": SQLAlchemyJobStore(url=f"sqlite:///{config.scheduler.apscheduler_db_path}"),
    }
    executors = {
        "default": ThreadPoolExecutor(max_workers=config.scheduler.max_workers),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": config.scheduler.misfire_grace_time,
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
    )

    create_scheduler_and_register_jobs(scheduler, config)

    stop_event = threading.Event()

    start_stop_polling_thread(config, db)

    log.info(
        "swingrl_app_built",
        jobstore_path=config.scheduler.apscheduler_db_path,
        job_count=12,
    )

    return {
        "scheduler": scheduler,
        "stop_event": stop_event,
        "config": config,
        "duckdb_conn": duckdb_conn,
    }


def main() -> None:
    """Parse args, build app, start scheduler, and block until signal."""
    parser = argparse.ArgumentParser(description="SwingRL production entrypoint")
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to SwingRL YAML config (default: config/swingrl.yaml)",
    )
    args = parser.parse_args()

    app = build_app(config_path=args.config)
    scheduler = app["scheduler"]
    stop_event = app["stop_event"]
    duckdb_conn = app["duckdb_conn"]

    handler = make_signal_handler(scheduler, stop_event, duckdb_conn=duckdb_conn)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    scheduler.start()
    log.info("scheduler_started")

    stop_event.wait()
    log.info("swingrl_exiting")
    sys.exit(0)


if __name__ == "__main__":
    main()
