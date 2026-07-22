"""Discord embed builder functions for all alert types.

Builds Discord webhook-compatible embed payloads for trade fills,
daily summaries, stuck agents, circuit breaker events, and iteration
completion (Phase 0.6 of the memory agent refocus).

Usage:
    from swingrl.monitoring.embeds import build_trade_embed
    payload = build_trade_embed(fill, stop_price=440.0, take_profit=470.0)
    alerter.send_embed("info", payload)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swingrl.execution.types import FillResult

# Discord embed sidebar colors used by the non-category embeds (stuck-agent, circuit
# breaker, iteration-completion). Trade-fill and digest colors now live in _CATEGORY_STYLE.
_COLOR_BUY = 0x00FF00
_COLOR_WARNING = 0xFFA500
_COLOR_CRITICAL = 0xFF0000

# Central category style map (STYLE-D15): category -> (sidebar color, title emoji).
# Single source of truth — alerter.py imports this and applies it in send_alert(category=...),
# and the embed builders below read it too, so there are no scattered hex/emoji literals to
# retheme in more than one place. NOTE: trade-fill colors are kept at their live values
# (buy 0x00FF00 / sell 0xFF4444), not the styling brief's 0x2ECC71 / 0xE74C3C recollection —
# the directive was "colors unchanged" and existing tests pin the live values; only the
# 🟢/🔴 title emoji is new.
_CATEGORY_STYLE: dict[str, tuple[int, str]] = {
    "ingest": (0x3498DB, "📥"),  # data ingests (candles)
    "digest": (0xF1C40F, "📊"),  # Daily Summary digest — gold
    "buy": (0x00FF00, "🟢"),  # trade fill — buy (color kept)
    "sell": (0xFF4444, "🔴"),  # trade fill — sell (color kept)
    "cycle": (0x9B59B6, "🔄"),  # cycle-orders ping + ops heartbeats — purple
    "warning": (0xFFA500, "⚠️"),  # warning (level default, unchanged)
    "critical": (0xFF0000, "🚨"),  # critical (level default, unchanged)
}


def build_trade_embed(
    fill: FillResult,
    stop_price: float | None = None,
    take_profit: float | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build a trade fill Discord embed.

    Args:
        fill: Broker fill confirmation with symbol, side, quantity, price.
        stop_price: Optional stop loss price level.
        take_profit: Optional take profit price level.

    Returns:
        Discord webhook payload dict with embeds list.
    """
    side_upper = fill.side.upper()
    # STYLE-D15: color unchanged (green buy / red sell); title gains a 🟢/🔴 prefix.
    color, side_emoji = _CATEGORY_STYLE["buy" if fill.side == "buy" else "sell"]
    notional = fill.quantity * fill.fill_price

    fields: list[dict[str, object]] = [
        {"name": "Side", "value": side_upper, "inline": True},
        {"name": "Quantity", "value": str(fill.quantity), "inline": True},
        {"name": "Fill Price", "value": f"${fill.fill_price:,.2f}", "inline": True},
        {"name": "Notional", "value": f"${notional:,.2f}", "inline": True},
        {"name": "Commission", "value": f"${fill.commission:,.2f}", "inline": True},
    ]

    # Realized P&L: populated only for sell fills (Task 6, PNL-D8) — buys never carry it.
    if fill.realized_pnl is not None:
        fields.append(
            {"name": "Realized P&L", "value": f"${fill.realized_pnl:+,.2f}", "inline": True}
        )

    fields.extend(
        [
            {
                "name": "Stop Loss",
                "value": f"${stop_price:,.2f}" if stop_price is not None else "N/A",
                "inline": True,
            },
            {
                "name": "Take Profit",
                "value": f"${take_profit:,.2f}" if take_profit is not None else "N/A",
                "inline": True,
            },
        ]
    )

    return {
        "embeds": [
            {
                "title": f"{side_emoji} {side_upper} {fill.symbol}",
                "color": color,
                "fields": fields,
                "footer": {
                    "text": f"SwingRL | {fill.environment.title()} | TRADE",
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


def _benchmark_fields(prefix: str, agent_value: float, benchmark: float) -> list[dict[str, object]]:
    """Buy & Hold + vs B&H fields for one env (Task 9, BENCH-D13).

    ``vs B&H`` is agent_value − benchmark as signed dollars and percent of the
    benchmark — the daily agent-vs-passive gap. Rendered only when the env has a
    recorded benchmark (post epoch reset); Task 10 restyles these (color + ✅/❌).
    """
    delta = agent_value - benchmark
    pct = (delta / benchmark * 100) if benchmark != 0 else 0.0
    mark = "✅" if delta >= 0 else "❌"  # STYLE-D15: beating / trailing buy-and-hold
    return [
        {"name": f"{prefix} Buy & Hold", "value": f"${benchmark:,.2f}", "inline": True},
        {
            "name": f"{prefix} vs B&H",
            "value": f"{mark} ${delta:+,.2f} ({pct:+.2f}%)",
            "inline": True,
        },
    ]


def build_daily_summary_embed(
    equity_snapshot: dict[str, float] | None,
    crypto_snapshot: dict[str, float] | None,
    equity_trades_today: int,
    crypto_trades_today: int,
    cb_status: dict[str, str] | None = None,
    equity_benchmark: float | None = None,
    crypto_benchmark: float | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build a daily summary Discord embed.

    Args:
        equity_snapshot: Dict with total_value, daily_pnl, cash_balance or None.
        crypto_snapshot: Dict with total_value, daily_pnl, cash_balance or None.
        equity_trades_today: Number of equity trades executed today.
        crypto_trades_today: Number of crypto trades executed today.
        cb_status: Dict of active circuit breaker details or None.
        equity_benchmark: Equal-weight buy-and-hold value of the equity baselines
            (Task 9, BENCH-D13), or None pre-reset — the digest then omits the
            "Buy & Hold" / "vs B&H" fields, leaving the pre-reset shape unchanged.
        crypto_benchmark: Same for the crypto env.

    Returns:
        Discord webhook payload dict with embeds list.
    """
    fields: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    if equity_snapshot is not None:
        ev = equity_snapshot["total_value"]
        epnl = equity_snapshot.get("daily_pnl", 0.0)
        total_value += ev
        total_pnl += epnl
        pct = (epnl / (ev - epnl) * 100) if (ev - epnl) != 0 else 0.0
        fields.extend(
            [
                {"name": "Equity Value", "value": f"${ev:,.2f}", "inline": True},
                {
                    "name": "Equity P&L",
                    "value": f"{_pnl_arrow(epnl)} ${epnl:+,.2f} ({pct:+.2f}%)",
                    "inline": True,
                },
                {"name": "Equity Trades", "value": str(equity_trades_today), "inline": True},
            ]
        )
        if equity_benchmark is not None:
            fields.extend(_benchmark_fields("Equity", ev, equity_benchmark))

    if crypto_snapshot is not None:
        cv = crypto_snapshot["total_value"]
        cpnl = crypto_snapshot.get("daily_pnl", 0.0)
        total_value += cv
        total_pnl += cpnl
        pct = (cpnl / (cv - cpnl) * 100) if (cv - cpnl) != 0 else 0.0
        fields.extend(
            [
                {"name": "Crypto Value", "value": f"${cv:,.2f}", "inline": True},
                {
                    "name": "Crypto P&L",
                    "value": f"{_pnl_arrow(cpnl)} ${cpnl:+,.2f} ({pct:+.2f}%)",
                    "inline": True,
                },
                {"name": "Crypto Trades", "value": str(crypto_trades_today), "inline": True},
            ]
        )
        if crypto_benchmark is not None:
            fields.extend(_benchmark_fields("Crypto", cv, crypto_benchmark))

    fields.append(
        {"name": "Total Portfolio Value", "value": f"${total_value:,.2f}", "inline": False}
    )
    fields.append(
        {
            "name": "Combined Daily P&L",
            "value": f"{_pnl_arrow(total_pnl)} ${total_pnl:+,.2f}",
            "inline": False,
        }
    )

    cb_text = "All Clear"
    if cb_status:
        cb_text = ", ".join(f"{k}: {v}" for k, v in cb_status.items())
    fields.append({"name": "CB Status", "value": cb_text, "inline": False})

    today_str = datetime.now(UTC).strftime("%Y-%m-%d")

    digest_color, _ = _CATEGORY_STYLE["digest"]  # STYLE-D15: gold
    return {
        "embeds": [
            {
                "title": "Daily Summary",
                "color": digest_color,
                "fields": fields,
                "footer": {"text": f"SwingRL | Daily Summary | {today_str}"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


def build_stuck_agent_embed(
    environment: str,
    consecutive_count: int,
    last_action_date: str | None,
    regime_state: str | None = None,
    turbulence_level: float | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build a stuck agent warning Discord embed.

    Args:
        environment: Environment name (equity or crypto).
        consecutive_count: Number of consecutive all-cash cycles.
        last_action_date: ISO date of last non-trivial action, or None.
        regime_state: Current market regime label (e.g., bull, bear).
        turbulence_level: Current turbulence index value.

    Returns:
        Discord webhook payload dict with embeds list.
    """
    fields: list[dict[str, object]] = [
        {"name": "Consecutive Cash Cycles", "value": str(consecutive_count), "inline": True},
        {
            "name": "Last Non-Trivial Action",
            "value": last_action_date or "Unknown",
            "inline": True,
        },
        {
            "name": "Current Regime",
            "value": regime_state or "Unknown",
            "inline": True,
        },
        {
            "name": "Turbulence Level",
            "value": f"{turbulence_level:.2f}" if turbulence_level is not None else "N/A",
            "inline": True,
        },
    ]

    description = (
        f"The {environment} agent has been holding 100% cash for "
        f"{consecutive_count} consecutive cycles. This may indicate the agent "
        f"is stuck in a risk-averse state or the market regime is preventing "
        f"any trade signals from passing risk checks."
    )

    return {
        "embeds": [
            {
                "title": f"Stuck Agent: {environment.title()}",
                "color": _COLOR_WARNING,
                "description": description,
                "fields": fields,
                "footer": {"text": f"SwingRL | {environment.title()} | WARNING"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


def build_circuit_breaker_embed(
    environment: str,
    trigger_type: str,
    current_drawdown: float,
    threshold: float,
    cooldown_end: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build a circuit breaker critical Discord embed.

    Args:
        environment: Environment name (equity or crypto).
        trigger_type: Type of circuit breaker trigger (e.g., daily_drawdown).
        current_drawdown: Current drawdown percentage as decimal.
        threshold: Drawdown threshold that was breached.
        cooldown_end: ISO timestamp when cooldown ends, or None.

    Returns:
        Discord webhook payload dict with embeds list.
    """
    fields: list[dict[str, object]] = [
        {"name": "Trigger Type", "value": trigger_type, "inline": True},
        {
            "name": "Current Drawdown",
            "value": f"{current_drawdown:.2%}",
            "inline": True,
        },
        {"name": "Threshold", "value": f"{threshold:.2%}", "inline": True},
        {
            "name": "Cooldown Until",
            "value": cooldown_end or "N/A",
            "inline": True,
        },
    ]

    return {
        "embeds": [
            {
                "title": f"Circuit Breaker: {environment.title()}",
                "color": _COLOR_CRITICAL,
                "fields": fields,
                "footer": {"text": f"SwingRL | {environment.title()} | CRITICAL"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


def build_iteration_completion_embed(
    summary: dict[str, Any],
) -> dict[str, list[dict[str, object]]]:
    """Build a Capital Preservation Score iteration completion Discord embed.

    Phase 0.6 of the memory agent refocus. Fired by ``train_pipeline.py``
    after each (env, iteration) finishes and CPS is computed/persisted.

    Color logic:
        - **Green**: no regression flagged AND ``cps_v1_delta_vs_prev >= 0``
          (or no prior baseline). The iteration moved CPS forward without
          regressing on any dimension.
        - **Yellow**: ``regression_flag`` is True but ``cps_v1_delta_vs_prev``
          is still positive. One dimension (e.g., worst_fold_mdd) jumped
          but the overall CPS still improved — partial regression worth
          watching but not a failure.
        - **Red**: ``cps_v1_delta_vs_prev < 0``. The primary CPS metric
          regressed — likely the "gave back returns for pass rate" trap
          that the refocus is designed to catch.

    Args:
        summary: Dict from ``compute_and_persist_iteration_cps``. Required
            keys: env, iteration, cps_v1_multiplicative, cps_v2_additive,
            cps_v3_sortino, cps_v1_delta_vs_prev, return_delta_vs_prev,
            worst_mdd_delta_vs_prev, median_return, mean_winner_sharpe,
            winners_count, chronic_failure_count, worst_fold_number,
            worst_fold_mdd, regression_flag, regression_dimensions,
            cps_v1_treatment_only, cps_v1_control_only, dedup_rows_dropped.

    Returns:
        Discord webhook payload dict ready for ``alerter.send_embed()``.
    """
    cps_v1_delta = summary.get("cps_v1_delta_vs_prev")
    regression_flag = bool(summary.get("regression_flag", False))

    # Color: red if v1 dropped, yellow if partial regression with v1 up,
    # green otherwise (including iter 0 with no baseline).
    if cps_v1_delta is not None and cps_v1_delta < 0:
        color = _COLOR_CRITICAL
    elif regression_flag:
        color = _COLOR_WARNING
    else:
        color = _COLOR_BUY

    fields: list[dict[str, object]] = []

    # Primary CPS values (3 formulas side-by-side)
    fields.append(
        {
            "name": "CPS v1 (multiplicative)",
            "value": _fmt_cps(summary.get("cps_v1_multiplicative"), delta=cps_v1_delta),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "CPS v2 (additive)",
            "value": _fmt_cps(summary.get("cps_v2_additive")),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "CPS v3 (sortino)",
            "value": _fmt_cps(summary.get("cps_v3_sortino")),
            "inline": True,
        }
    )

    # Median return + delta
    fields.append(
        {
            "name": "Median Return",
            "value": _fmt_pct(
                summary.get("median_return"),
                delta=summary.get("return_delta_vs_prev"),
            ),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "Winner Sharpe (mean)",
            "value": _fmt_float(summary.get("mean_winner_sharpe")),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "Winners",
            "value": _fmt_int(summary.get("winners_count")),
            "inline": True,
        }
    )

    # Worst-fold info
    fields.append(
        {
            "name": "Worst Fold",
            "value": _fmt_int(summary.get("worst_fold_number")),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "Worst Fold MDD",
            "value": _fmt_pct(
                summary.get("worst_fold_mdd"),
                delta=summary.get("worst_mdd_delta_vs_prev"),
            ),
            "inline": True,
        }
    )
    fields.append(
        {
            "name": "Chronic Failures",
            "value": _fmt_int(summary.get("chronic_failure_count")),
            "inline": True,
        }
    )

    # Treatment vs control split — only if populated (iter 3+ with control folds)
    treatment_v1 = summary.get("cps_v1_treatment_only")
    control_v1 = summary.get("cps_v1_control_only")
    if treatment_v1 is not None and control_v1 is not None:
        fields.append(
            {
                "name": "Treatment CPS v1",
                "value": _fmt_cps(treatment_v1),
                "inline": True,
            }
        )
        fields.append(
            {
                "name": "Control CPS v1",
                "value": _fmt_cps(control_v1),
                "inline": True,
            }
        )
        # Compute the ratio for the smoking-gun signal
        if treatment_v1 != 0:
            ratio = control_v1 / treatment_v1
            fields.append(
                {
                    "name": "Control / Treatment",
                    "value": f"{ratio:.2f}×",
                    "inline": True,
                }
            )

    # Regression dimensions list (only when regression_flag is True)
    regression_dims = summary.get("regression_dimensions") or []
    if regression_dims:
        fields.append(
            {
                "name": "Regression dimensions",
                "value": ", ".join(regression_dims),
                "inline": False,
            }
        )

    # Audit field — only show dedup count when nonzero (iter 1 equity)
    dedup = summary.get("dedup_rows_dropped", 0)
    if dedup:
        fields.append(
            {
                "name": "Dedup rows dropped",
                "value": str(int(dedup)),
                "inline": True,
            }
        )

    env = str(summary.get("env", "?"))
    iteration = int(summary.get("iteration", 0))
    title = f"Iteration {iteration} complete — {env.title()}"
    footer_marker = "REGRESSION" if regression_flag else "OK"

    return {
        "embeds": [
            {
                "title": title,
                "color": color,
                "fields": fields,
                "footer": {"text": f"SwingRL | {env.title()} | {footer_marker}"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


# ---------------------------------------------------------------------------
# Formatting helpers (kept private to embeds.py — used only by the
# iteration completion embed)
# ---------------------------------------------------------------------------


def _pnl_arrow(value: float) -> str:
    """Return the STYLE-D15 direction arrow for a P&L value (▲ for >= 0, ▼ for negative)."""
    return "▲" if value >= 0 else "▼"


def _fmt_cps(value: float | None, *, delta: float | None = None) -> str:
    """Format a CPS scalar (5 decimal places). Optionally append delta in parens."""
    if value is None:
        return "—"
    base = f"{value:.5f}"
    if delta is None:
        return base
    return f"{base} ({delta:+.5f})"


def _fmt_pct(value: float | None, *, delta: float | None = None) -> str:
    """Format a fractional value as percentage. Optionally append delta in pp."""
    if value is None:
        return "—"
    base = f"{value * 100:.2f}%"
    if delta is None:
        return base
    return f"{base} ({delta * 100:+.2f}pp)"


def _fmt_float(value: float | None) -> str:
    """Format a plain float to 3 decimals; em-dash for None."""
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_int(value: int | None) -> str:
    """Format an integer cell; em-dash for None."""
    if value is None:
        return "—"
    return str(int(value))
