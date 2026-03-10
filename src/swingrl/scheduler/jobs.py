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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from swingrl.scheduler.halt_check import is_halted
from swingrl.scheduler.healthcheck_ping import ping_healthcheck

try:
    from swingrl.monitoring.embeds import build_daily_summary_embed, build_trade_embed
except ImportError:  # pragma: no cover
    build_trade_embed = None  # type: ignore[assignment]
    build_daily_summary_embed = None  # type: ignore[assignment]

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
    if build_trade_embed is not None:
        for fill in fills:
            try:
                embed = build_trade_embed(fill)
                ctx.alerter.send_embed("info", embed)
            except Exception:
                log.exception("equity_trade_embed_failed", symbol=getattr(fill, "symbol", "?"))

    try:
        ping_healthcheck(ctx.config.alerting.healthchecks_equity_url)
    except Exception:
        log.exception("equity_healthcheck_ping_failed")

    # Shadow inference (non-blocking, never affects active)
    try:
        from swingrl.shadow.shadow_runner import run_shadow_inference  # noqa: PLC0415

        run_shadow_inference(ctx, "equity")
    except Exception:
        log.exception("shadow_inference_failed", environment="equity")

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
    if build_trade_embed is not None:
        for fill in fills:
            try:
                embed = build_trade_embed(fill)
                ctx.alerter.send_embed("info", embed)
            except Exception:
                log.exception("crypto_trade_embed_failed", symbol=getattr(fill, "symbol", "?"))

    try:
        ping_healthcheck(ctx.config.alerting.healthchecks_crypto_url)
    except Exception:
        log.exception("crypto_healthcheck_ping_failed")

    # Shadow inference (non-blocking, never affects active)
    try:
        from swingrl.shadow.shadow_runner import run_shadow_inference  # noqa: PLC0415

        run_shadow_inference(ctx, "crypto")
    except Exception:
        log.exception("shadow_inference_failed", environment="crypto")

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

        # Build snapshots per environment
        equity_snap = None
        crypto_snap = None
        for row in rows:
            snap = {
                "total_value": row["total_value"],
                "daily_pnl": row["daily_pnl"],
                "cash_balance": row["cash_balance"],
            }
            if row["environment"] == "equity":
                equity_snap = snap
            elif row["environment"] == "crypto":
                crypto_snap = snap

        if build_daily_summary_embed is not None:
            embed = build_daily_summary_embed(
                equity_snapshot=equity_snap,
                crypto_snapshot=crypto_snap,
                equity_trades_today=0,
                crypto_trades_today=0,
            )
            ctx.alerter.send_embed("info", embed)
        else:
            # Fallback: send plain text summary
            lines = []
            for row in rows:
                lines.append(
                    f"**{row['environment'].title()}**: "
                    f"${row['total_value']:.2f} "
                    f"(PnL: ${row['daily_pnl']:.2f}, DD: {row['drawdown_pct']:.2%})"
                )
            ctx.alerter.send_alert("info", "Daily Portfolio Summary", "\n".join(lines))

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


def daily_backup_job() -> None:
    """Run daily SQLite backup with integrity verification and rotation.

    Backups should run even when trading is halted (no halt check).
    Wraps in try/except to never crash the scheduler.
    """
    ctx = _get_ctx()

    try:
        from swingrl.backup.sqlite_backup import backup_sqlite

        success = backup_sqlite(ctx.config, ctx.alerter)
        log.info("daily_backup_job_complete", success=success)
    except Exception:
        log.exception("daily_backup_job_failed")


def weekly_duckdb_backup_job() -> None:
    """Run weekly DuckDB backup with table/row verification.

    Backups should run even when trading is halted (no halt check).
    Wraps in try/except to never crash the scheduler.
    """
    ctx = _get_ctx()

    try:
        from swingrl.backup.duckdb_backup import backup_duckdb

        success = backup_duckdb(ctx.config, ctx.alerter)
        log.info("weekly_duckdb_backup_job_complete", success=success)
    except Exception:
        log.exception("weekly_duckdb_backup_job_failed")


def monthly_offsite_job() -> None:
    """Run monthly off-site rsync via Tailscale.

    Backups should run even when trading is halted (no halt check).
    Wraps in try/except to never crash the scheduler.
    """
    ctx = _get_ctx()

    try:
        from swingrl.backup.offsite_sync import offsite_rsync

        success = offsite_rsync(ctx.config, ctx.alerter)
        log.info("monthly_offsite_job_complete", success=success)
    except Exception:
        log.exception("monthly_offsite_job_failed")


def shadow_promotion_check_job() -> None:
    """Evaluate shadow models for promotion daily at 7 PM ET.

    For each environment (equity, crypto), runs auto-promotion criteria
    evaluation. Import-guarded so shadow module is optional.
    Wraps in try/except to never crash the scheduler.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("shadow_promotion_check_skipped", reason="halt_flag_active")
        return

    try:
        from swingrl.shadow.lifecycle import ModelLifecycle  # noqa: PLC0415
        from swingrl.shadow.promoter import evaluate_shadow_promotion  # noqa: PLC0415

        models_dir = Path(ctx.config.paths.models_dir)
        lifecycle = ModelLifecycle(models_dir)

        for env_name in ("equity", "crypto"):
            try:
                promoted = evaluate_shadow_promotion(
                    config=ctx.config,
                    db=ctx.db,
                    env_name=env_name,
                    lifecycle=lifecycle,
                    alerter=ctx.alerter,
                )
                log.info(
                    "shadow_promotion_check_complete",
                    environment=env_name,
                    promoted=promoted,
                )
            except Exception:
                log.exception("shadow_promotion_check_env_failed", environment=env_name)
    except Exception:
        log.exception("shadow_promotion_check_job_failed")


def automated_trigger_check_job() -> None:
    """Check for automated emergency stop triggers every 5 minutes.

    Checks VIX+CB threshold, consecutive NaN inferences, and Binance.US IP ban.
    If any triggers detected, executes the full four-tier emergency stop.
    No halt check -- triggers must be evaluated even when halted (idempotent halt set).
    Wraps in try/except to never crash the scheduler.
    """
    ctx = _get_ctx()

    try:
        from swingrl.execution.emergency import check_automated_triggers, execute_emergency_stop

        triggers = check_automated_triggers(config=ctx.config, db=ctx.db)

        if triggers:
            reason = "; ".join(triggers)
            log.critical("automated_triggers_firing", triggers=triggers)
            execute_emergency_stop(
                config=ctx.config,
                db=ctx.db,
                alerter=ctx.alerter,
                reason=reason,
            )
        else:
            log.debug("automated_trigger_check_clear")
    except Exception:
        log.exception("automated_trigger_check_job_failed")
