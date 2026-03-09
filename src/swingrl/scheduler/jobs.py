"""Scheduled job functions for SwingRL trading cycles.

Each job follows the pattern: halt-check -> execute -> callbacks -> error-handle.
Jobs are wired to APScheduler in main.py (Plan 04).

Usage:
    from swingrl.scheduler.jobs import init_job_context, equity_cycle, crypto_cycle

    init_job_context(config=config, db=db, pipeline=pipeline, alerter=alerter)
    fills = equity_cycle()  # Called by APScheduler
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from swingrl.scheduler.halt_check import is_halted
from swingrl.scheduler.healthcheck_ping import ping_healthcheck

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.pipeline import ExecutionPipeline
    from swingrl.execution.types import FillResult
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


@dataclass
class JobContext:
    """Shared context for all scheduled jobs."""

    config: SwingRLConfig
    db: DatabaseManager
    pipeline: ExecutionPipeline
    alerter: Alerter


_ctx: JobContext | None = None


def init_job_context(
    config: Any,
    db: Any,
    pipeline: Any,
    alerter: Any,
) -> JobContext:
    """Initialize the module-level job context.

    Must be called before any job function. Typically called once at startup.

    Args:
        config: Validated SwingRLConfig.
        db: DatabaseManager instance.
        pipeline: ExecutionPipeline instance.
        alerter: Alerter instance.

    Returns:
        The initialized JobContext.
    """
    global _ctx  # noqa: PLW0603
    _ctx = JobContext(config=config, db=db, pipeline=pipeline, alerter=alerter)
    log.info("job_context_initialized")
    return _ctx


def _get_ctx() -> JobContext:
    """Return the module-level job context, failing fast if not initialized."""
    if _ctx is None:
        raise RuntimeError("Job context not initialized. Call init_job_context() first.")
    return _ctx


def equity_cycle() -> list[FillResult]:
    """Execute the equity trading cycle.

    Pattern: halt-check -> execute_cycle("equity") -> callbacks -> error-handle.

    Returns:
        List of FillResult from the cycle, or empty list on halt/error.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("equity_cycle_skipped", reason="halt_flag_active")
        return []

    try:
        fills = ctx.pipeline.execute_cycle("equity")
    except Exception:
        log.exception("equity_cycle_failed")
        try:
            ctx.alerter.send_alert(
                "critical", "Equity Cycle Failed", "Exception during equity trading cycle"
            )
        except Exception:
            log.exception("equity_cycle_alert_failed")
        return []

    # Post-cycle callbacks (each wrapped individually)
    try:
        ping_healthcheck(ctx.config.alerting.healthchecks_equity_url)
    except Exception:
        log.exception("equity_healthcheck_ping_failed")

    return fills


def crypto_cycle() -> list[FillResult]:
    """Execute the crypto trading cycle.

    Pattern: halt-check -> execute_cycle("crypto") -> callbacks -> error-handle.

    Returns:
        List of FillResult from the cycle, or empty list on halt/error.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("crypto_cycle_skipped", reason="halt_flag_active")
        return []

    try:
        fills = ctx.pipeline.execute_cycle("crypto")
    except Exception:
        log.exception("crypto_cycle_failed")
        try:
            ctx.alerter.send_alert(
                "critical", "Crypto Cycle Failed", "Exception during crypto trading cycle"
            )
        except Exception:
            log.exception("crypto_cycle_alert_failed")
        return []

    # Post-cycle callbacks
    try:
        ping_healthcheck(ctx.config.alerting.healthchecks_crypto_url)
    except Exception:
        log.exception("crypto_healthcheck_ping_failed")

    return fills


def daily_summary_job() -> None:
    """Query portfolio snapshots and send daily summary alert.

    Queries the latest portfolio_snapshots per environment and sends
    a summary via the alerter.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("daily_summary_skipped", reason="halt_flag_active")
        return

    try:
        with ctx.db.sqlite() as conn:
            rows = conn.execute(
                "SELECT environment, total_value, cash_balance, daily_pnl, drawdown_pct "
                "FROM portfolio_snapshots "
                "ORDER BY timestamp DESC LIMIT 2"
            ).fetchall()

        if not rows:
            log.info("daily_summary_no_data")
            return

        lines = []
        for row in rows:
            lines.append(
                f"**{row['environment'].title()}**: "
                f"${row['total_value']:.2f} "
                f"(PnL: ${row['daily_pnl']:.2f}, DD: {row['drawdown_pct']:.2%})"
            )

        summary = "\n".join(lines)
        ctx.alerter.send_alert("info", "Daily Portfolio Summary", summary)
        log.info("daily_summary_sent")
    except Exception:
        log.exception("daily_summary_failed")


def stuck_agent_check_job() -> None:
    """Detect if any environment has consecutive all-cash cycles.

    Equity: alert after 10 all-cash snapshots.
    Crypto: alert after 30 all-cash snapshots.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("stuck_agent_check_skipped", reason="halt_flag_active")
        return

    thresholds = {"equity": 10, "crypto": 30}

    try:
        for env, threshold in thresholds.items():
            with ctx.db.sqlite() as conn:
                rows = conn.execute(
                    "SELECT total_value, cash_balance FROM portfolio_snapshots "
                    "WHERE environment = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (env, threshold),
                ).fetchall()

            if len(rows) < threshold:
                continue

            all_cash = all(abs(row["cash_balance"] - row["total_value"]) < 0.01 for row in rows)

            if all_cash:
                ctx.alerter.send_alert(
                    "warning",
                    f"Stuck Agent: {env.title()}",
                    f"{env.title()} environment has been 100% cash "
                    f"for {threshold} consecutive cycles.",
                    environment=env,
                )
                log.warning("stuck_agent_detected", environment=env, cycles=threshold)
    except Exception:
        log.exception("stuck_agent_check_failed")


def weekly_fundamentals_job() -> None:
    """Refresh fundamental features (weekly schedule).

    Import-guarded: if the features pipeline is not available, logs warning.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("weekly_fundamentals_skipped", reason="halt_flag_active")
        return

    try:
        try:
            from swingrl.data.ingestors.fred import FredIngestor  # type: ignore[import-not-found]

            ingestor = FredIngestor(ctx.config, ctx.db)
            ingestor.refresh()
            log.info("weekly_fundamentals_refreshed")
        except ImportError:
            log.warning("weekly_fundamentals_import_unavailable")
    except Exception:
        log.exception("weekly_fundamentals_failed")


def monthly_macro_job() -> None:
    """Refresh FRED macro data (monthly schedule).

    Import-guarded: if the FRED ingestor is not available, logs warning.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("monthly_macro_skipped", reason="halt_flag_active")
        return

    try:
        try:
            from swingrl.data.ingestors.fred import FredIngestor  # noqa: F811

            ingestor = FredIngestor(ctx.config, ctx.db)
            ingestor.refresh()
            log.info("monthly_macro_refreshed")
        except ImportError:
            log.warning("monthly_macro_import_unavailable")
    except Exception:
        log.exception("monthly_macro_failed")
