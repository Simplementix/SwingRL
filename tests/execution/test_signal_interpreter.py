"""Tests for SignalInterpreter -- ensemble action to trade signal conversion.

Covers deadzone filtering, buy/sell classification, and symbol mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.execution.signal_interpreter import SignalInterpreter


@pytest.fixture
def config(tmp_path):
    """Minimal config for signal interpreter tests."""
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


class TestSignalInterpreter:
    """PAPER-07: Signal interpreter converts ensemble actions to trade signals."""

    def test_buy_signal_above_deadzone(self, config: SwingRLConfig) -> None:
        """Action above current weight + deadzone produces buy signal."""
        interp = SignalInterpreter(config)
        blended = np.array([0.30, 0.10])  # SPY target=0.30, QQQ target=0.10
        current = np.array([0.20, 0.10])  # SPY has 0.20, QQQ same

        signals = interp.interpret("equity", blended, current)

        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) == 1
        assert buy_signals[0].symbol == "SPY"
        assert buy_signals[0].raw_weight == pytest.approx(0.30)
        assert buy_signals[0].environment == "equity"

    def test_sell_signal_below_deadzone(self, config: SwingRLConfig) -> None:
        """Action below current weight - deadzone produces sell signal."""
        interp = SignalInterpreter(config)
        blended = np.array([0.05, 0.10])
        current = np.array([0.20, 0.10])

        signals = interp.interpret("equity", blended, current)

        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) == 1
        assert sell_signals[0].symbol == "SPY"
        assert sell_signals[0].action == "sell"

    def test_hold_within_deadzone(self, config: SwingRLConfig) -> None:
        """Action within +/-0.02 of current weight is hold (filtered out)."""
        interp = SignalInterpreter(config)
        blended = np.array([0.21, 0.09])  # within 0.02 of current
        current = np.array([0.20, 0.10])

        signals = interp.interpret("equity", blended, current)

        # Holds are not included in output
        assert len(signals) == 0

    def test_all_hold_returns_empty(self, config: SwingRLConfig) -> None:
        """When all actions are holds, return empty list."""
        interp = SignalInterpreter(config)
        blended = np.array([0.200, 0.100])
        current = np.array([0.200, 0.100])

        signals = interp.interpret("equity", blended, current)
        assert signals == []

    def test_crypto_symbols_mapping(self, config: SwingRLConfig) -> None:
        """Crypto environment uses crypto symbols from config."""
        interp = SignalInterpreter(config)
        blended = np.array([0.60, 0.10])
        current = np.array([0.30, 0.10])

        signals = interp.interpret("crypto", blended, current)

        assert len(signals) == 1
        assert signals[0].symbol == "BTCUSDT"
        assert signals[0].environment == "crypto"

    def test_mixed_signals(self, config: SwingRLConfig) -> None:
        """Multiple signals: one buy, one sell."""
        interp = SignalInterpreter(config)
        blended = np.array([0.30, 0.05])
        current = np.array([0.20, 0.20])

        signals = interp.interpret("equity", blended, current)

        actions = {s.symbol: s.action for s in signals}
        assert actions["SPY"] == "buy"
        assert actions["QQQ"] == "sell"

    def test_boundary_exactly_at_deadzone(self, config: SwingRLConfig) -> None:
        """Action exactly at current + deadzone boundary is still hold."""
        interp = SignalInterpreter(config)
        blended = np.array([0.22, 0.10])  # exactly +0.02
        current = np.array([0.20, 0.10])

        signals = interp.interpret("equity", blended, current)
        assert len(signals) == 0  # boundary is hold, not buy

    def test_just_beyond_deadzone(self, config: SwingRLConfig) -> None:
        """Action just beyond deadzone triggers signal."""
        interp = SignalInterpreter(config)
        blended = np.array([0.2201, 0.10])  # just past +0.02
        current = np.array([0.20, 0.10])

        signals = interp.interpret("equity", blended, current)
        assert len(signals) == 1
        assert signals[0].action == "buy"
