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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
_reconciliation_failures: int = 0

# Placeholder flagged algo for the MT commentary skeleton when a cycle has no
# recorded per-algo proposals yet. The future Meta-Trader spec computes the real
# flagged algo from proposals-vs-blend geometry (§4.7).
_DEFAULT_FLAGGED_ALGO = "ppo"


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


def maybe_post_trade_commentary(ctx: JobContext, environment: str) -> None:
    """Post-cycle Meta-Trader commentary (Task 12 skeleton) — inert unless enabled.

    Runtime gate: ``meta_trader.enabled`` (config default False). When disabled this
    returns immediately without importing the client or touching the network — the
    skeleton is provably inert by default (Task 16's go/no-go is the remaining gate).
    When enabled, it looks up the just-executed cycle, builds a shadow-commentary
    payload, and POSTs it fail-open to the memory service's /trade/commentary. Never
    raises — shadow commentary must never disturb a trading cycle.

    Args:
        ctx: The job context (config, db, pipeline, alerter).
        environment: "equity" or "crypto".
    """
    meta_trader = getattr(ctx.config, "meta_trader", None)
    if meta_trader is None or not meta_trader.enabled:
        return

    try:
        with ctx.db.connection() as conn:
            row = conn.execute(
                "SELECT ic.cycle_id, ic.hmm_p_bull, ic.hmm_p_bear, ic.vix, ic.turbulence,"
                " ic.deployed_iteration,"
                " (SELECT cap.algorithm FROM cycle_algo_proposals cap"
                "  WHERE cap.cycle_id = ic.cycle_id"
                "  ORDER BY cap.weight_in_blend_frac DESC LIMIT 1) AS algorithm,"
                " (SELECT string_agg(cap.algorithm || ':'"
                "         || round(cap.weight_in_blend_frac::numeric, 3)::text, '; ')"
                "  FROM cycle_algo_proposals cap WHERE cap.cycle_id = ic.cycle_id)"
                "  AS proposals_summary"
                " FROM inference_cycles ic WHERE ic.environment = %s"
                " ORDER BY ic.cycle_ts DESC LIMIT 1",
                [environment],
            ).fetchone()

        if row is None:
            log.info("trade_commentary_skipped_no_cycle", environment=environment)
            return

        from swingrl.memory.client import MemoryClient  # noqa: PLC0415

        client = MemoryClient(
            base_url=ctx.config.memory_agent.base_url,
            default_timeout=ctx.config.memory_agent.timeout_sec,
            api_key=ctx.config.memory_agent.api_key,
        )
        payload = {
            "cycle_id": row["cycle_id"],
            "environment": environment,
            "algorithm": row.get("algorithm") or _DEFAULT_FLAGGED_ALGO,
            "deployed_iteration": row.get("deployed_iteration") or 0,
            "regime": {
                "hmm_p_bull": row.get("hmm_p_bull"),
                "hmm_p_bear": row.get("hmm_p_bear"),
                "vix": row.get("vix"),
                "turbulence": row.get("turbulence"),
            },
            "proposals_summary": row.get("proposals_summary") or "",
        }
        client.trade_commentary(payload)
        log.info("trade_commentary_posted", environment=environment, cycle_id=row["cycle_id"])
    except Exception:
        log.exception("trade_commentary_failed", environment=environment)


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

    # Post-cycle Meta-Trader commentary (Task 12 skeleton; inert unless enabled)
    maybe_post_trade_commentary(ctx, "equity")

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

    # Post-cycle Meta-Trader commentary (Task 12 skeleton; inert unless enabled)
    maybe_post_trade_commentary(ctx, "crypto")

    return fills


def risk_sweep_job() -> None:
    """Between-cycle risk sweep (D10): mark held positions + evaluate breakers, no trading.

    Trading cycles are sparse (crypto every 4h, equity once a day), so between them the
    system is blind to a crash in a held position. This interval job (config
    ``risk.sweep_interval_minutes``) closes that blind window: for each environment with
    held positions it fetches fresh prices (fail-open per symbol), marks the positions to
    market, and evaluates the per-env drawdown/daily-loss breaker; it then evaluates the
    global breaker across BOTH environments' current values.

    Two negatives are load-bearing (pinned test contracts): the sweep places NO orders
    (it never touches the broker's submit path) and writes NO ``portfolio_snapshots`` rows
    (snapshots stay cycle-cadence append-only so the daily-P&L baseline query is not
    polluted — it never calls ``record_snapshot``). A breach flows through the breaker's
    existing ``_trigger`` halt/alert path unchanged.

    Skips entirely when the halt flag is active. Each env's body is isolated in its own
    try/except so one env's failure (e.g. a broker adapter lazy-init error, an env-specific
    DB error) cannot stop the other — a persistent env-A fault must never re-open env-B's
    blind window. The global check is skipped if any env's value is missing (see below).
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("risk_sweep_skipped", reason="halt_flag_active")
        return

    pipeline = ctx.pipeline
    tracker = pipeline.position_tracker
    breakers = pipeline.circuit_breakers

    # Combined inputs for the global breaker. EVERY env is included — even a flat
    # cash-only env — because GlobalCircuitBreaker reconstructs the combined high-water
    # mark from ALL envs' persisted snapshots; omitting an env's current value would
    # understate total value and overstate combined drawdown, tripping a false global halt
    # of every environment.
    portfolio_values: dict[str, float] = {}
    daily_pnls: dict[str, float] = {}

    for env, breaker in breakers.items():
        # Per-env isolation: a failure anywhere in this env's body is logged and the loop
        # moves on to the next env, so a persistent fault in one environment cannot re-open
        # the between-cycle blind window this job exists to close for the other.
        try:
            held = [p for p in tracker.get_positions(env) if p["quantity"]]

            # Fetch fresh marks only when the env actually holds positions (no broker
            # call for a flat env). Fail-open per symbol: a fetch failure or non-positive
            # price warns and is skipped, so compute_portfolio_value falls back to the
            # stored last_price for that symbol — a sweep never blocks on one bad quote.
            prices: dict[str, float] = {}
            if held:
                adapter = pipeline.get_adapter(env)
                for pos in held:
                    symbol = pos["symbol"]
                    try:
                        price = adapter.get_current_price(symbol)
                    except Exception:
                        log.warning("sweep_price_fetch_failed", environment=env, symbol=symbol)
                        continue
                    if price is not None and price > 0:
                        prices[symbol] = price
                    else:
                        log.warning(
                            "sweep_price_non_positive", environment=env, symbol=symbol, price=price
                        )
                tracker.mark_positions(env, prices)

            # Mark-to-market value for this env (cash-only when flat). No snapshot written.
            value = tracker.compute_portfolio_value(env, prices)
            daily_pnl = tracker.compute_daily_pnl(env, value)
            portfolio_values[env] = value
            daily_pnls[env] = daily_pnl

            # Per-env breaker only matters when the env holds positions (a cash-only env
            # cannot draw down). HWM folds in this cycle's value explicitly: unlike the
            # trade path, the sweep writes no snapshot first, so the stored HWM does not
            # yet include ``value`` — max() reproduces the record_snapshot-then-check
            # semantics the cycle path gets for free.
            if held:
                hwm = max(tracker.get_high_water_mark(env), value)
                breaker.check_and_update(
                    portfolio_value=value,
                    high_water_mark=hwm,
                    daily_pnl=daily_pnl,
                )
        except Exception:
            log.exception("risk_sweep_env_failed", environment=env)

    # Global breaker across both envs' current values (combined drawdown/daily loss). Skip
    # when any env's value is missing — an env that errored before computing its value is
    # absent from the dict, and a partial dict would understate total value and risk the
    # same false global halt the flat-env inclusion avoids. A flat env is NOT missing: it
    # contributed its cash value above. Own try/except so a global failure never crashes
    # the scheduler.
    if set(portfolio_values) != set(breakers):
        log.warning("risk_sweep_global_skipped", evaluated=sorted(portfolio_values))
        return
    try:
        pipeline.global_cb.check_combined(portfolio_values, daily_pnls)
        log.info("risk_sweep_complete", envs=sorted(portfolio_values))
    except Exception:
        log.exception("risk_sweep_global_failed")


# D13 buy-and-hold: env -> latest-close query. Constant SQL (no interpolation) so
# bandit stays quiet; the ORDER BY column differs per env (ohlcv_daily.date vs
# ohlcv_4h.datetime), which is why this is a map of full statements, not an f-string.
_LATEST_CLOSE_SQL: dict[str, str] = {
    "equity": "SELECT close FROM ohlcv_daily WHERE symbol = %s ORDER BY date DESC LIMIT 1",
    "crypto": "SELECT close FROM ohlcv_4h WHERE symbol = %s ORDER BY datetime DESC LIMIT 1",
}


def _benchmark_value(conn: Any, env: str) -> float | None:
    """Equal-weight buy-and-hold value of the env's baselines, grown to latest close (D13).

    Returns Σ over the env's ``benchmark_baselines`` rows of
    ``(capital_usd / n_symbols) × latest_close / baseline_price`` — what an equal-weight
    passive hold of each symbol would be worth today. Returns None when the env has no
    baselines (pre epoch reset), so the digest omits the "Buy & Hold" fields and its
    pre-reset shape is unchanged. A symbol with no stored close is skipped (its slice
    contributes nothing) rather than failing the whole digest.

    Args:
        conn: An open DB connection (called inside the digest's connection block).
        env: "equity" or "crypto".

    Returns:
        The benchmark value in USD, or None when no baselines exist for ``env``.
    """
    close_sql = _LATEST_CLOSE_SQL.get(env)
    if close_sql is None:
        return None
    baselines = conn.execute(
        "SELECT symbol, baseline_price, capital_usd FROM benchmark_baselines"
        " WHERE environment = %s",
        (env,),
    ).fetchall()
    if not baselines:
        return None
    n_symbols = len(baselines)
    total = 0.0
    for row in baselines:
        close_row = conn.execute(close_sql, (row["symbol"],)).fetchone()
        if close_row is None or close_row["close"] is None:
            continue
        total += (row["capital_usd"] / n_symbols) * (close_row["close"] / row["baseline_price"])
    return total


def daily_summary_job() -> None:
    """Query portfolio snapshots and send daily summary alert.

    Queries the latest portfolio_snapshots per environment and sends
    a summary via the alerter.
    """
    ctx = _get_ctx()

    # A30 Discord wiring (Plan A Task E): flush any buffered INFO alerts as the daily
    # digest. Alerter.send_daily_digest() had ZERO production callers, so INFO-level
    # alerts accumulated in memory and were lost on the next restart. This end-of-day
    # job (18:00 ET) is the flush point. It runs BEFORE the halt gate so INFO buffered
    # on a halted day still reaches Discord. The trader keeps digest semantics rather
    # than info_immediate (unlike the low-volume collector) — see docs/training/deploy-process.md.
    try:
        ctx.alerter.send_daily_digest()
    except Exception:
        log.exception("daily_digest_flush_failed")

    if is_halted(ctx.db):
        log.warning("daily_summary_skipped", reason="halt_flag_active")
        return

    try:
        # Per-env count of TODAY's (ET) signal trades from the trades ledger. ET date on
        # both sides — a bare timestamp::date renders in the server session timezone,
        # which misses between 20:00 and 24:00 ET (same convention as
        # position_tracker.get_daily_pnl). Replaces the previously hardcoded zeros that
        # made every digest report "0 trades" (found live 2026-07-21).
        counts = {"equity": 0, "crypto": 0}
        # Buy-and-hold benchmark per env (D13). Computed in the SAME connection block as
        # the snapshots + trade counts (Task 3's wiring) — one connection per digest.
        # None until the epoch-reset recorder has written baselines, in which case the
        # embed omits the "Buy & Hold" fields (pre-reset shape unchanged).
        benchmarks: dict[str, float | None] = {"equity": None, "crypto": None}
        with ctx.db.connection() as conn:
            rows = conn.execute(
                "SELECT environment, total_value, cash_balance, daily_pnl, drawdown_pct "
                "FROM portfolio_snapshots "
                "ORDER BY timestamp DESC LIMIT 2"
            ).fetchall()
            trade_rows = conn.execute(
                "SELECT environment, count(*) AS n FROM trades "
                "WHERE trade_type = 'signal' "
                "AND (timestamp AT TIME ZONE 'America/New_York')::date = "
                "(now() AT TIME ZONE 'America/New_York')::date "
                "GROUP BY environment"
            ).fetchall()
            benchmarks["equity"] = _benchmark_value(conn, "equity")
            benchmarks["crypto"] = _benchmark_value(conn, "crypto")

        for r in trade_rows:
            counts[r["environment"]] = int(r["n"])

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
                equity_trades_today=counts["equity"],
                crypto_trades_today=counts["crypto"],
                equity_benchmark=benchmarks["equity"],
                crypto_benchmark=benchmarks["crypto"],
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
            with ctx.db.connection() as conn:
                rows = conn.execute(
                    "SELECT total_value, cash_balance FROM portfolio_snapshots "
                    "WHERE environment = %s "
                    "ORDER BY timestamp DESC LIMIT %s",
                    [env, threshold],
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
            from swingrl.data.fred import FREDIngestor  # noqa: PLC0415

            ingestor = FREDIngestor(ctx.config)
            ingestor.run_all()
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
            from swingrl.data.fred import FREDIngestor  # noqa: PLC0415

            ingestor = FREDIngestor(ctx.config)
            ingestor.run_all()
            log.info("monthly_macro_refreshed")
        except ImportError:
            log.warning("monthly_macro_import_unavailable")
    except Exception:
        log.exception("monthly_macro_failed")


def daily_backup_job() -> None:
    """Run daily PostgreSQL backup with integrity verification and rotation.

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
    """Run weekly PostgreSQL backup with table/row verification.

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


def reconciliation_job() -> None:
    """Reconcile DB equity positions against Alpaca broker state (daily at 5 PM ET).

    Runs equity-only reconciliation. Crypto uses virtual balance and has no
    broker-side positions to reconcile. Skips when halt flag is active.
    Tracks consecutive failures; 3+ consecutive failures escalate to critical alert.
    Resets failure counter on success. Never raises (always wraps in try/except).
    """
    global _reconciliation_failures  # noqa: PLW0603

    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("reconciliation_job_skipped", reason="halt_flag_active")
        return

    try:
        from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter  # noqa: PLC0415
        from swingrl.execution.reconciliation import PositionReconciler  # noqa: PLC0415

        adapter = AlpacaAdapter(config=ctx.config, alerter=ctx.alerter)
        reconciler = PositionReconciler(
            config=ctx.config,
            db=ctx.db,
            adapter=adapter,
            alerter=ctx.alerter,
        )
        adjustments = reconciler.reconcile("equity")
        _reconciliation_failures = 0
        log.info("reconciliation_job_complete", adjustments=len(adjustments))
    except Exception:
        _reconciliation_failures += 1
        level: Literal["critical", "warning"] = (
            "critical" if _reconciliation_failures >= 3 else "warning"
        )
        log.exception(
            "reconciliation_job_failed",
            consecutive_failures=_reconciliation_failures,
            level=level,
        )
        try:
            ctx.alerter.send_alert(
                level,
                "Reconciliation Job Failed",
                f"Daily equity reconciliation failed "
                f"(consecutive failures: {_reconciliation_failures})",
            )
        except Exception:
            log.exception("reconciliation_job_alert_failed")


def equity_fill_confirmation_job() -> None:
    """Confirm pre-open equity opening-auction fills (~09:35 ET, spec D11).

    The 09:15 pre-open cycle submits DAY market orders that Alpaca fills at the official
    opening print; this job (after the 09:30 open) turns those confirmed fills into real
    trades. It loads every unresolved ``pending_orders`` row (a restart-safe DB worklist —
    the orders survive a process restart between 09:15 and 09:35), polls each order's status
    by id, and for a filled order builds a real FillResult and records it through the SAME
    post-fill path the synchronous route uses (``pipeline.record_fill`` — trades + positions +
    fill_quality + realized-P&L attach), sends the trade embed, and stamps ``resolved_at``.

    An order still unfilled/canceled after the auction gets a warning alert and is LEFT
    unresolved for the next run — never silently dropped. Runs regardless of the halt flag:
    the auction orders already executed, so their fills must be recorded for the ledger to
    stay truthful (recording is bookkeeping, not new trading). Never raises.
    """
    ctx = _get_ctx()

    try:
        with ctx.db.connection() as conn:
            rows = conn.execute(
                "SELECT order_id, cycle_id, symbol, side, submitted_at FROM pending_orders "
                "WHERE resolved_at IS NULL ORDER BY submitted_at"
            ).fetchall()
    except Exception:
        log.exception("equity_fill_confirmation_load_failed")
        return

    if not rows:
        log.info("equity_fill_confirmation_no_pending")
        return

    try:
        adapter = ctx.pipeline.get_adapter("equity")
    except Exception:
        log.exception("equity_fill_confirmation_adapter_failed")
        return

    confirmed = 0
    for row in rows:
        try:
            if _confirm_one_pending_order(ctx, adapter, row):
                confirmed += 1
        except Exception:
            log.exception("equity_fill_confirmation_order_failed", order_id=row["order_id"])
    log.info("equity_fill_confirmation_complete", pending=len(rows), confirmed=confirmed)


def _confirm_one_pending_order(ctx: JobContext, adapter: Any, row: dict[str, Any]) -> bool:
    """Confirm a single pending pre-open order; return True if a fill was recorded (D11).

    A filled order is recorded via the pipeline's shared post-fill path and its
    ``pending_orders`` row is stamped resolved; anything else (unfilled/canceled) alerts a
    warning and leaves the row unresolved for the next run.

    Args:
        ctx: The job context.
        adapter: The equity exchange adapter (Alpaca), exposing ``get_order_status``.
        row: A ``pending_orders`` row (order_id, cycle_id, symbol, side, submitted_at).

    Returns:
        True if the auction fill was recorded, False otherwise.
    """
    from swingrl.execution.types import FillResult, SizedOrder  # noqa: PLC0415

    order_id = row["order_id"]
    order = adapter.get_order_status(order_id)
    status = str(getattr(order, "status", "") or "").lower()
    filled_price = getattr(order, "filled_avg_price", None)
    filled_qty = getattr(order, "filled_qty", None)
    filled_qty_val = _safe_float(filled_qty)

    is_fully_filled = status == "filled" and filled_price is not None and filled_qty is not None

    if not is_fully_filled:
        if filled_qty_val is not None and filled_qty_val > 0:
            # Review #1: a partial fill executed REAL shares but the order is not fully filled.
            # Whether to record partial fills as trades is plan-silent (a user decision), so we
            # do NOT record here — but the executed shares must be surfaced explicitly under a
            # distinct title so this can never be mistaken for a clean no-fill. Row stays
            # unresolved for the next run / user review.
            requested = getattr(order, "qty", None)
            if requested is None:
                requested = getattr(order, "notional", None)
            ctx.alerter.send_alert(
                level="warning",
                title="Equity auction order PARTIALLY filled — unrecorded",
                message=(
                    f"{row['symbol']} {row['side']} (order {order_id}) PARTIALLY filled: "
                    f"{filled_qty_val} of {requested} at {filled_price} "
                    f"({status or 'unknown'}) — not recorded, left pending for the next "
                    "confirmation run / user review."
                ),
                environment="equity",
            )
            log.warning(
                "pending_order_partial_fill",
                order_id=order_id,
                symbol=row["symbol"],
                filled_qty=filled_qty_val,
                requested=str(requested),
                status=status,
            )
            return False

        # Clean no-fill / canceled after the auction — warn and leave the row unresolved
        # (retry next run). Never silently dropped; capital-preservation visibility.
        ctx.alerter.send_alert(
            level="warning",
            title="Equity auction order unfilled",
            message=(
                f"{row['symbol']} {row['side']} (order {order_id}) is still "
                f"{status or 'unknown'} after the opening auction — not recorded, left "
                "pending for the next confirmation run."
            ),
            environment="equity",
        )
        log.warning(
            "equity_auction_order_unfilled", order_id=order_id, symbol=row["symbol"], status=status
        )
        return False

    # Review #2 crash-safety: a prior run may have recorded the trade (the broker order id is
    # the trades PK) but crashed before stamping resolved_at, leaving this row unresolved. Re-
    # recording would hit the TEXT PK inside record_fill and fire a false "Fill Executed But
    # Not Recorded" CRITICAL on every run forever. Detect the already-recorded fill, stamp
    # resolved_at quietly, and move on — no duplicate trade, no critical, no re-sent embed.
    if _trade_already_recorded(ctx, order_id):
        _stamp_pending_resolved(ctx, order_id)
        log.info("equity_auction_fill_already_recorded", order_id=order_id, symbol=row["symbol"])
        return True

    fill_price = _safe_float(filled_price)
    quantity = filled_qty_val
    if fill_price is None or quantity is None:
        # Defensive: a "filled" order should always carry numeric price/qty; if not, do not
        # fabricate a trade — warn and leave the row for the next run rather than crash.
        log.warning(
            "equity_auction_fill_unparseable_amounts",
            order_id=order_id,
            symbol=row["symbol"],
            filled_price=str(filled_price),
            filled_qty=str(filled_qty),
        )
        return False
    submitted_at = _to_iso(row.get("submitted_at"))
    filled_at = _to_iso(getattr(order, "filled_at", None)) or datetime.now(UTC).isoformat()

    fill = FillResult(
        trade_id=order_id,
        symbol=row["symbol"],
        side=row["side"],
        quantity=quantity,
        fill_price=fill_price,
        commission=0.0,
        slippage=0.0,
        environment="equity",
        broker="alpaca",
        status="filled",
        submitted_at=submitted_at,
        filled_at=filled_at,
    )
    # Minimal SizedOrder — equity carries no stops; process() reads side + stop/TP (None).
    sized_order = SizedOrder(
        symbol=row["symbol"],
        side=row["side"],
        quantity=quantity,
        dollar_amount=fill_price * quantity,
        stop_loss_price=None,
        take_profit_price=None,
        environment="equity",
    )

    # Same post-fill path as the synchronous route (trades + positions + fill_quality +
    # realized-P&L attach). decision_price is None — the 09:15 sizing price is not stored on
    # the pending row, so fill_quality slippage is NULL (the row is still written).
    recorded = ctx.pipeline.record_fill(
        fill,
        sized_order=sized_order,
        cycle_id=row.get("cycle_id"),
        decision_price=None,
        env_name="equity",
    )
    if recorded is None:
        # Recording failed — record_fill already sent a critical alert. Leave the row
        # unresolved so the next run retries rather than losing the fill.
        return False

    # Trade embed (same callback the synchronous equity_cycle fires per fill).
    if build_trade_embed is not None:
        try:
            embed = build_trade_embed(recorded)
            ctx.alerter.send_embed("info", embed)
        except Exception:
            log.exception("equity_fill_confirmation_embed_failed", order_id=order_id)

    _stamp_pending_resolved(ctx, order_id)
    log.info(
        "equity_auction_fill_confirmed",
        order_id=order_id,
        symbol=row["symbol"],
        fill_price=fill_price,
        quantity=quantity,
        cycle_id=row.get("cycle_id"),
    )
    return True


def _trade_already_recorded(ctx: JobContext, trade_id: str) -> bool:
    """Return True if a trades row already exists for ``trade_id`` (the broker order id)."""
    with ctx.db.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM trades WHERE trade_id = %s LIMIT 1", (trade_id,)
        ).fetchone()
    return row is not None


def _stamp_pending_resolved(ctx: JobContext, order_id: str) -> None:
    """Stamp a pending_orders row resolved (confirmation complete)."""
    with ctx.db.connection() as conn:
        conn.execute(
            "UPDATE pending_orders SET resolved_at = %s WHERE order_id = %s",
            (datetime.now(UTC).isoformat(), order_id),
        )


def _safe_float(value: Any) -> float | None:  # noqa: ANN401
    """Parse a broker numeric field (str/float/None) to float, or None if unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_iso(value: Any) -> str | None:  # noqa: ANN401
    """Normalize a DB/broker timestamp (datetime or str) to a UTC ISO string, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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
