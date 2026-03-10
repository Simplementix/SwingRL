"""Discord embed builder functions for all alert types.

Builds Discord webhook-compatible embed payloads for trade fills,
daily summaries, stuck agents, and circuit breaker events.

Usage:
    from swingrl.monitoring.embeds import build_trade_embed
    payload = build_trade_embed(fill, stop_price=440.0, take_profit=470.0)
    alerter.send_embed("info", payload)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swingrl.execution.types import FillResult

# Discord embed sidebar colors
_COLOR_BUY = 0x00FF00
_COLOR_SELL = 0xFF4444
_COLOR_BLUE = 0x3498DB
_COLOR_WARNING = 0xFFA500
_COLOR_CRITICAL = 0xFF0000


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
    color = _COLOR_BUY if fill.side == "buy" else _COLOR_SELL
    notional = fill.quantity * fill.fill_price

    fields: list[dict[str, object]] = [
        {"name": "Side", "value": side_upper, "inline": True},
        {"name": "Quantity", "value": str(fill.quantity), "inline": True},
        {"name": "Fill Price", "value": f"${fill.fill_price:,.2f}", "inline": True},
        {"name": "Notional", "value": f"${notional:,.2f}", "inline": True},
        {"name": "Commission", "value": f"${fill.commission:,.2f}", "inline": True},
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

    return {
        "embeds": [
            {
                "title": f"{side_upper} {fill.symbol}",
                "color": color,
                "fields": fields,
                "footer": {
                    "text": f"SwingRL | {fill.environment.title()} | TRADE",
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


def build_daily_summary_embed(
    equity_snapshot: dict[str, float] | None,
    crypto_snapshot: dict[str, float] | None,
    equity_trades_today: int,
    crypto_trades_today: int,
    cb_status: dict[str, str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Build a daily summary Discord embed.

    Args:
        equity_snapshot: Dict with total_value, daily_pnl, cash_balance or None.
        crypto_snapshot: Dict with total_value, daily_pnl, cash_balance or None.
        equity_trades_today: Number of equity trades executed today.
        crypto_trades_today: Number of crypto trades executed today.
        cb_status: Dict of active circuit breaker details or None.

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
                    "value": f"${epnl:+,.2f} ({pct:+.2f}%)",
                    "inline": True,
                },
                {"name": "Equity Trades", "value": str(equity_trades_today), "inline": True},
            ]
        )

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
                    "value": f"${cpnl:+,.2f} ({pct:+.2f}%)",
                    "inline": True,
                },
                {"name": "Crypto Trades", "value": str(crypto_trades_today), "inline": True},
            ]
        )

    fields.append(
        {"name": "Total Portfolio Value", "value": f"${total_value:,.2f}", "inline": False}
    )
    fields.append({"name": "Combined Daily P&L", "value": f"${total_pnl:+,.2f}", "inline": False})

    cb_text = "All Clear"
    if cb_status:
        cb_text = ", ".join(f"{k}: {v}" for k, v in cb_status.items())
    fields.append({"name": "CB Status", "value": cb_text, "inline": False})

    today_str = datetime.now(UTC).strftime("%Y-%m-%d")

    return {
        "embeds": [
            {
                "title": "Daily Summary",
                "color": _COLOR_BLUE,
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
