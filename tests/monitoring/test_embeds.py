"""Tests for Discord embed builder functions.

PAPER-13: Trade alert, daily summary, stuck agent, and circuit breaker embeds.
"""

from __future__ import annotations

from datetime import datetime

from swingrl.execution.types import FillResult
from swingrl.monitoring.embeds import (
    build_circuit_breaker_embed,
    build_daily_summary_embed,
    build_stuck_agent_embed,
    build_trade_embed,
)

# ---------------------------------------------------------------------------
# Test: build_trade_embed
# ---------------------------------------------------------------------------


class TestBuildTradeEmbed:
    """PAPER-13: Trade embed contains symbol, side, fill price, stop/TP levels."""

    def test_buy_embed_has_green_color(self) -> None:
        """Buy trade embed has green sidebar color 0x00FF00."""
        fill = FillResult(
            trade_id="t1",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.25,
            commission=0.0,
            slippage=0.01,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        assert embed["embeds"][0]["color"] == 0x00FF00

    def test_sell_embed_has_red_color(self) -> None:
        """Sell trade embed has red sidebar color 0xFF4444."""
        fill = FillResult(
            trade_id="t2",
            symbol="QQQ",
            side="sell",
            quantity=5.0,
            fill_price=380.00,
            commission=0.0,
            slippage=0.02,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        assert embed["embeds"][0]["color"] == 0xFF4444

    def test_title_contains_side_and_symbol(self) -> None:
        """Title is 'BUY SPY' or 'SELL SPY'."""
        fill = FillResult(
            trade_id="t3",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        assert embed["embeds"][0]["title"] == "BUY SPY"

    def test_fields_include_required_values(self) -> None:
        """Embed fields include side, quantity, fill price, notional, commission."""
        fill = FillResult(
            trade_id="t4",
            symbol="BTC",
            side="buy",
            quantity=0.5,
            fill_price=60000.0,
            commission=66.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )
        embed = build_trade_embed(fill)
        fields = embed["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]

        assert "Side" in field_names
        assert "Quantity" in field_names
        assert "Fill Price" in field_names
        assert "Notional" in field_names
        assert "Commission" in field_names

    def test_stop_and_tp_fields_present(self) -> None:
        """Stop loss and take profit fields present when provided."""
        fill = FillResult(
            trade_id="t5",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill, stop_price=440.0, take_profit=470.0)
        fields = embed["embeds"][0]["fields"]
        field_map = {f["name"]: f["value"] for f in fields}

        assert "$440.00" in str(field_map["Stop Loss"])
        assert "$470.00" in str(field_map["Take Profit"])

    def test_stop_and_tp_default_na(self) -> None:
        """Stop loss and take profit show N/A when not provided."""
        fill = FillResult(
            trade_id="t6",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        fields = embed["embeds"][0]["fields"]
        field_map = {f["name"]: f["value"] for f in fields}

        assert field_map["Stop Loss"] == "N/A"
        assert field_map["Take Profit"] == "N/A"

    def test_footer_contains_env_and_trade(self) -> None:
        """Footer is 'SwingRL | Equity | TRADE'."""
        fill = FillResult(
            trade_id="t7",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        footer = embed["embeds"][0]["footer"]["text"]
        assert "SwingRL" in footer
        assert "Equity" in footer
        assert "TRADE" in footer

    def test_timestamp_is_iso_format(self) -> None:
        """Embed includes a parseable ISO-8601 timestamp."""
        fill = FillResult(
            trade_id="t8",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )
        embed = build_trade_embed(fill)
        ts = embed["embeds"][0]["timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# Test: build_daily_summary_embed
# ---------------------------------------------------------------------------


class TestBuildDailySummaryEmbed:
    """PAPER-13: Daily summary embed shows per-env P&L and combined value."""

    def test_blue_color(self) -> None:
        """Daily summary has blue sidebar 0x3498DB."""
        equity_snap = {
            "total_value": 10500.0,
            "daily_pnl": 50.0,
            "cash_balance": 2000.0,
        }
        embed = build_daily_summary_embed(
            equity_snapshot=equity_snap,
            crypto_snapshot=None,
            equity_trades_today=2,
            crypto_trades_today=0,
        )
        assert embed["embeds"][0]["color"] == 0x3498DB

    def test_title_is_daily_summary(self) -> None:
        """Title is 'Daily Summary'."""
        embed = build_daily_summary_embed(
            equity_snapshot=None,
            crypto_snapshot=None,
            equity_trades_today=0,
            crypto_trades_today=0,
        )
        assert embed["embeds"][0]["title"] == "Daily Summary"

    def test_equity_fields_present(self) -> None:
        """Equity snapshot fields shown when provided."""
        equity_snap = {
            "total_value": 10500.0,
            "daily_pnl": 50.0,
            "cash_balance": 2000.0,
        }
        embed = build_daily_summary_embed(
            equity_snapshot=equity_snap,
            crypto_snapshot=None,
            equity_trades_today=3,
            crypto_trades_today=0,
        )
        fields = embed["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]

        assert any("Equity" in n and "Value" in n for n in field_names)

    def test_combined_total_present(self) -> None:
        """Combined total portfolio value field present."""
        equity_snap = {"total_value": 10000.0, "daily_pnl": 50.0, "cash_balance": 2000.0}
        crypto_snap = {"total_value": 500.0, "daily_pnl": -5.0, "cash_balance": 100.0}
        embed = build_daily_summary_embed(
            equity_snapshot=equity_snap,
            crypto_snapshot=crypto_snap,
            equity_trades_today=1,
            crypto_trades_today=1,
        )
        fields = embed["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert any("Total" in n for n in field_names)

    def test_cb_status_all_clear(self) -> None:
        """CB status field shows 'All Clear' when no active breakers."""
        embed = build_daily_summary_embed(
            equity_snapshot=None,
            crypto_snapshot=None,
            equity_trades_today=0,
            crypto_trades_today=0,
            cb_status=None,
        )
        fields = embed["embeds"][0]["fields"]
        field_map = {f["name"]: f["value"] for f in fields}
        assert "All Clear" in str(field_map.get("CB Status", ""))

    def test_footer_contains_daily_summary(self) -> None:
        """Footer contains 'Daily Summary'."""
        embed = build_daily_summary_embed(
            equity_snapshot=None,
            crypto_snapshot=None,
            equity_trades_today=0,
            crypto_trades_today=0,
        )
        footer = embed["embeds"][0]["footer"]["text"]
        assert "Daily Summary" in footer


# ---------------------------------------------------------------------------
# Test: build_stuck_agent_embed
# ---------------------------------------------------------------------------


class TestBuildStuckAgentEmbed:
    """PAPER-13: Stuck agent embed includes cycle count, diagnostics."""

    def test_orange_warning_color(self) -> None:
        """Stuck agent embed has orange warning color 0xFFA500."""
        embed = build_stuck_agent_embed(
            environment="equity",
            consecutive_count=15,
            last_action_date="2026-03-01",
        )
        assert embed["embeds"][0]["color"] == 0xFFA500

    def test_title_contains_env(self) -> None:
        """Title is 'Stuck Agent: Equity'."""
        embed = build_stuck_agent_embed(
            environment="equity",
            consecutive_count=15,
            last_action_date="2026-03-01",
        )
        assert embed["embeds"][0]["title"] == "Stuck Agent: Equity"

    def test_fields_include_diagnostics(self) -> None:
        """Fields include consecutive cash cycles, last action, regime, turbulence."""
        embed = build_stuck_agent_embed(
            environment="crypto",
            consecutive_count=35,
            last_action_date="2026-03-05",
            regime_state="bear",
            turbulence_level=2.5,
        )
        fields = embed["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]

        assert "Consecutive Cash Cycles" in field_names
        assert "Last Non-Trivial Action" in field_names
        assert "Current Regime" in field_names
        assert "Turbulence Level" in field_names

    def test_description_explains_stuck(self) -> None:
        """Description explains what a stuck agent is."""
        embed = build_stuck_agent_embed(
            environment="equity",
            consecutive_count=10,
            last_action_date=None,
        )
        assert "description" in embed["embeds"][0]
        assert len(embed["embeds"][0]["description"]) > 10


# ---------------------------------------------------------------------------
# Test: build_circuit_breaker_embed
# ---------------------------------------------------------------------------


class TestBuildCircuitBreakerEmbed:
    """PAPER-13: Circuit breaker embed contains trigger details and cooldown."""

    def test_red_critical_color(self) -> None:
        """Circuit breaker embed has red critical color 0xFF0000."""
        embed = build_circuit_breaker_embed(
            environment="equity",
            trigger_type="daily_drawdown",
            current_drawdown=0.045,
            threshold=0.04,
        )
        assert embed["embeds"][0]["color"] == 0xFF0000

    def test_title_contains_env(self) -> None:
        """Title is 'Circuit Breaker: Equity'."""
        embed = build_circuit_breaker_embed(
            environment="equity",
            trigger_type="daily_drawdown",
            current_drawdown=0.045,
            threshold=0.04,
        )
        assert embed["embeds"][0]["title"] == "Circuit Breaker: Equity"

    def test_fields_include_trigger_info(self) -> None:
        """Fields include trigger type, drawdown, threshold, cooldown."""
        embed = build_circuit_breaker_embed(
            environment="equity",
            trigger_type="daily_drawdown",
            current_drawdown=0.045,
            threshold=0.04,
            cooldown_end="2026-03-10T10:00:00Z",
        )
        fields = embed["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]

        assert "Trigger Type" in field_names
        assert "Current Drawdown" in field_names
        assert "Threshold" in field_names
        assert "Cooldown Until" in field_names

    def test_cooldown_na_when_not_provided(self) -> None:
        """Cooldown shows N/A when not provided."""
        embed = build_circuit_breaker_embed(
            environment="equity",
            trigger_type="daily_drawdown",
            current_drawdown=0.045,
            threshold=0.04,
        )
        fields = embed["embeds"][0]["fields"]
        field_map = {f["name"]: f["value"] for f in fields}
        assert field_map["Cooldown Until"] == "N/A"
