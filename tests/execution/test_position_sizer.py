"""Tests for PositionSizer -- quarter-Kelly sizing with ATR stops.

Covers Kelly formula, risk cap, ATR stops, crypto floor, and post-floor rejection.
"""

from __future__ import annotations

import pytest

from swingrl.config.schema import load_config
from swingrl.execution.position_sizer import PositionSizer
from swingrl.execution.types import TradeSignal


@pytest.fixture
def config(tmp_path):
    """Config for position sizer tests."""
    yaml_content = """\
trading_mode: paper
equity:
  symbols: [SPY, QQQ]
  max_position_size: 0.25
  max_drawdown_pct: 0.10
  daily_loss_limit_pct: 0.02
crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50
  max_drawdown_pct: 0.12
  daily_loss_limit_pct: 0.03
  min_order_usd: 10.0
capital:
  equity_usd: 400.0
  crypto_usd: 47.0
environment:
  signal_deadzone: 0.02
"""
    cfg_file = tmp_path / "swingrl.yaml"
    cfg_file.write_text(yaml_content)
    return load_config(cfg_file)


@pytest.fixture
def sizer(config):
    """PositionSizer with default conservative stats."""
    return PositionSizer(config)


@pytest.fixture
def buy_signal():
    """Sample equity buy signal."""
    return TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)


@pytest.fixture
def sell_signal():
    """Sample equity sell signal."""
    return TradeSignal(environment="equity", symbol="SPY", action="sell", raw_weight=0.05)


class TestKellyFormula:
    """PAPER-08: Quarter-Kelly position sizing."""

    def test_kelly_positive_with_default_stats(self, sizer: PositionSizer) -> None:
        """Default stats (WR=0.50, W=0.03, L=0.02) give positive Kelly."""
        # K = 0.50 - (1-0.50) / (0.03/0.02) = 0.50 - 0.50/1.50 = 0.50 - 0.333 = 0.167
        # Quarter-Kelly: 0.167 * 0.25 = 0.0417
        # Risk cap: min(0.0417, 0.02) = 0.02
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=470.0, atr_value=5.0, portfolio_value=10000.0)

        assert order is not None
        assert order.dollar_amount == pytest.approx(200.0)  # 0.02 * 10000

    def test_quarter_kelly_scaling(self, sizer: PositionSizer) -> None:
        """Quarter-Kelly reduces full Kelly by 75%."""
        # Full K = 0.167, Quarter = 0.0417
        # With 2% cap, the cap binds (0.02 < 0.0417)
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=470.0, atr_value=5.0, portfolio_value=10000.0)
        assert order is not None
        # 2% cap binds
        assert order.dollar_amount == pytest.approx(0.02 * 10000.0)

    def test_negative_kelly_returns_none(self, config) -> None:
        """When Kelly criterion is negative, skip trade."""
        sizer = PositionSizer(config, win_rate=0.30, avg_win=0.01, avg_loss=0.05)
        # K = 0.30 - 0.70 / (0.01/0.05) = 0.30 - 0.70/0.20 = 0.30 - 3.50 = -3.20
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=470.0, atr_value=5.0, portfolio_value=10000.0)
        assert order is None


class TestRiskCap:
    """PAPER-08: 2% max risk per trade."""

    def test_risk_cap_limits_dollar_amount(self, sizer: PositionSizer) -> None:
        """Dollar amount capped at 2% of portfolio value."""
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=470.0, atr_value=5.0, portfolio_value=50000.0)
        assert order is not None
        assert order.dollar_amount == pytest.approx(1000.0)  # 2% of 50k


class TestATRStops:
    """PAPER-08: ATR-based stop-loss and take-profit."""

    def test_buy_stop_loss(self, sizer: PositionSizer, buy_signal: TradeSignal) -> None:
        """Buy stop-loss at price - 2*ATR."""
        order = sizer.size(buy_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.stop_loss_price == pytest.approx(96.0)  # 100 - 2*2

    def test_buy_take_profit(self, sizer: PositionSizer, buy_signal: TradeSignal) -> None:
        """Buy take-profit at price + 4*ATR (2:1 R:R)."""
        order = sizer.size(buy_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.take_profit_price == pytest.approx(108.0)  # 100 + 4*2

    def test_sell_stop_loss(self, sizer: PositionSizer, sell_signal: TradeSignal) -> None:
        """Sell stop-loss at price + 2*ATR."""
        order = sizer.size(sell_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.stop_loss_price == pytest.approx(104.0)  # 100 + 2*2

    def test_sell_take_profit(self, sizer: PositionSizer, sell_signal: TradeSignal) -> None:
        """Sell take-profit at price - 4*ATR."""
        order = sizer.size(sell_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.take_profit_price == pytest.approx(92.0)  # 100 - 4*2


class TestCryptoFloor:
    """PAPER-11: Crypto $10 minimum order with post-floor risk check."""

    def test_crypto_floor_applied(self, sizer: PositionSizer) -> None:
        """Crypto orders below $10 are floor-adjusted to $10."""
        signal = TradeSignal(environment="crypto", symbol="BTCUSDT", action="buy", raw_weight=0.30)
        # With portfolio_value=47 and 2% risk cap: dollar_amount = 0.94
        # Floor to $10
        order = sizer.size(signal, current_price=42000.0, atr_value=500.0, portfolio_value=47.0)
        # $10 > max_position_size(0.50) * 47 = $23.5 -- $10 < $23.5, so it passes
        assert order is not None
        assert order.dollar_amount == pytest.approx(10.0)

    def test_crypto_post_floor_rejection(self, sizer: PositionSizer) -> None:
        """After floor to $10, reject if exceeds max_position_size * portfolio_value."""
        signal = TradeSignal(environment="crypto", symbol="BTCUSDT", action="buy", raw_weight=0.30)
        # portfolio_value=15, max_pos=0.50 => max $7.50, floor=$10 > $7.50 => reject
        order = sizer.size(signal, current_price=42000.0, atr_value=500.0, portfolio_value=15.0)
        assert order is None

    def test_equity_no_floor(self, sizer: PositionSizer) -> None:
        """Equity orders do not get crypto floor treatment."""
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=470.0, atr_value=5.0, portfolio_value=400.0)
        assert order is not None
        # 2% of 400 = $8, no floor applied
        assert order.dollar_amount == pytest.approx(8.0)


class TestQuantityCalculation:
    """Quantity is dollar_amount / current_price."""

    def test_quantity_computed(self, sizer: PositionSizer) -> None:
        """Quantity equals dollar_amount / price."""
        signal = TradeSignal(environment="equity", symbol="SPY", action="buy", raw_weight=0.30)
        order = sizer.size(signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        # dollar_amount = 200 (2% of 10k), quantity = 200/100 = 2.0
        assert order.quantity == pytest.approx(2.0)

    def test_order_side_matches_signal(self, sizer: PositionSizer, buy_signal: TradeSignal) -> None:
        """Order side matches signal action."""
        order = sizer.size(buy_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.side == "buy"

    def test_order_environment_matches_signal(
        self, sizer: PositionSizer, buy_signal: TradeSignal
    ) -> None:
        """Order environment matches signal environment."""
        order = sizer.size(buy_signal, current_price=100.0, atr_value=2.0, portfolio_value=10000.0)
        assert order is not None
        assert order.environment == "equity"
