"""Tests for RL environment foundation components.

Tests cover PortfolioSimulator, RollingSharpeReward, process_actions (softmax + deadzone),
and EnvironmentConfig schema integration.
"""

from __future__ import annotations

import numpy as np
import pytest
from swingrl.envs.portfolio import PortfolioSimulator, process_actions
from swingrl.envs.rewards import RollingSharpeReward

# ---------------------------------------------------------------------------
# PortfolioSimulator
# ---------------------------------------------------------------------------


class TestPortfolioSimulatorInit:
    """PortfolioSimulator initializes with cash, zero shares."""

    def test_initial_cash(self) -> None:
        """TRAIN-07: Simulator starts with specified cash amount."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=8, transaction_cost_pct=0.0006)
        assert sim.cash == pytest.approx(100_000.0)

    def test_initial_shares_zero(self) -> None:
        """TRAIN-07: Simulator starts with zero shares in all assets."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=8, transaction_cost_pct=0.0006)
        assert np.all(sim.shares == 0.0)
        assert sim.shares.shape == (8,)

    def test_portfolio_value_equals_cash_at_start(self) -> None:
        """TRAIN-07: At start, portfolio_value == initial cash (no positions)."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.0022)
        prices = np.array([40_000.0, 2_500.0])
        assert sim.portfolio_value(prices) == pytest.approx(100_000.0)

    def test_empty_trade_log(self) -> None:
        """TRAIN-07: Trade log is empty at initialization."""
        sim = PortfolioSimulator(initial_cash=50_000.0, n_assets=3, transaction_cost_pct=0.001)
        assert sim.trade_log == []


class TestPortfolioSimulatorRebalance:
    """rebalance() moves positions to target weights and deducts costs."""

    def test_rebalance_allocates_positions(self) -> None:
        """TRAIN-07: After rebalance, shares reflect target weights."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.0)
        prices = np.array([100.0, 200.0])
        target_weights = np.array([0.5, 0.3])  # 20% cash remainder
        sim.rebalance(target_weights, prices)

        # target_values: [50_000, 30_000] => shares: [500, 150]
        assert sim.shares[0] == pytest.approx(500.0)
        assert sim.shares[1] == pytest.approx(150.0)

    def test_rebalance_deducts_transaction_cost(self) -> None:
        """TRAIN-07: Transaction cost is deducted from portfolio value."""
        cost_pct = 0.01  # 1% for easy math
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=cost_pct)
        prices = np.array([100.0, 200.0])
        target_weights = np.array([0.5, 0.3])

        cost = sim.rebalance(target_weights, prices)

        # Total traded = |50_000| + |30_000| = 80_000
        # Cost = 80_000 * 0.01 = 800
        assert cost == pytest.approx(800.0)
        total = sim.portfolio_value(prices)
        assert total == pytest.approx(100_000.0 - 800.0)

    def test_rebalance_logs_trades(self) -> None:
        """TRAIN-07: Trade log records symbol indices, side, shares, cost."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.001)
        prices = np.array([100.0, 200.0])
        target_weights = np.array([0.5, 0.3])
        sim.rebalance(target_weights, prices)

        assert len(sim.trade_log) > 0
        trade = sim.trade_log[0]
        assert "asset_idx" in trade
        assert "side" in trade
        assert "shares" in trade
        assert "cost" in trade

    def test_rebalance_second_call_adjusts(self) -> None:
        """TRAIN-07: Second rebalance adjusts from current position to new target."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.0)
        prices = np.array([100.0, 200.0])

        sim.rebalance(np.array([0.5, 0.3]), prices)
        sim.rebalance(np.array([0.3, 0.5]), prices)

        # After second rebalance, total value stays same (no costs)
        total = sim.portfolio_value(prices)
        assert total == pytest.approx(100_000.0)


class TestPortfolioSimulatorAssetWeights:
    """asset_weights returns current weight per asset."""

    def test_weights_after_rebalance(self) -> None:
        """TRAIN-07: asset_weights reflects current allocation."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.0)
        prices = np.array([100.0, 200.0])
        sim.rebalance(np.array([0.5, 0.3]), prices)

        weights = sim.asset_weights(prices)
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(0.3)

    def test_weights_sum_less_than_one(self) -> None:
        """TRAIN-07: asset_weights sum <= 1.0 (remainder is cash)."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.0)
        prices = np.array([100.0, 200.0])
        sim.rebalance(np.array([0.4, 0.3]), prices)

        weights = sim.asset_weights(prices)
        assert np.sum(weights) <= 1.0 + 1e-9


class TestPortfolioSimulatorReset:
    """reset() restores initial state."""

    def test_reset_clears_positions(self) -> None:
        """TRAIN-07: After reset, shares are zero and cash restored."""
        sim = PortfolioSimulator(initial_cash=100_000.0, n_assets=2, transaction_cost_pct=0.001)
        prices = np.array([100.0, 200.0])
        sim.rebalance(np.array([0.5, 0.3]), prices)
        sim.reset(100_000.0)

        assert sim.cash == pytest.approx(100_000.0)
        assert np.all(sim.shares == 0.0)
        assert sim.trade_log == []


# ---------------------------------------------------------------------------
# process_actions (softmax + deadzone)
# ---------------------------------------------------------------------------


class TestProcessActions:
    """Softmax conversion with deadzone for action processing."""

    def test_softmax_produces_positive_weights(self) -> None:
        """TRAIN-09: Softmax output is all positive."""
        raw = np.array([-1.0, 0.0, 1.0, 0.5])
        current = np.zeros(4)
        result = process_actions(raw, current, deadzone=0.0)
        assert np.all(result >= 0.0)

    def test_softmax_weights_sum_le_one(self) -> None:
        """TRAIN-09: Softmax weights sum to <= 1.0."""
        raw = np.array([0.3, -0.5, 0.8, 0.1, -0.2, 0.6, -0.1, 0.4])
        current = np.zeros(8)
        result = process_actions(raw, current, deadzone=0.0)
        assert np.sum(result) <= 1.0 + 1e-9

    def test_softmax_numerical_stability(self) -> None:
        """TRAIN-09: Large values handled without overflow (subtract-max trick)."""
        raw = np.array([1000.0, 1001.0, 999.0])
        current = np.zeros(3)
        result = process_actions(raw, current, deadzone=0.0)
        assert np.all(np.isfinite(result))
        assert np.sum(result) <= 1.0 + 1e-9

    def test_deadzone_suppresses_small_changes(self) -> None:
        """TRAIN-09: Changes below deadzone threshold keep current weight."""
        current = np.array([0.3, 0.3, 0.4])
        # Construct raw that produces softmax close to current
        # Softmax of [0, 0, 0] = [1/3, 1/3, 1/3] ~ [0.333, 0.333, 0.333]
        raw = np.zeros(3)
        process_actions(raw, current, deadzone=0.02)  # exercise path; tested below
        # The difference between 1/3 and 0.3 is ~0.033, above deadzone
        # The difference between 1/3 and 0.4 is ~0.067, above deadzone
        # So not all will be suppressed; test a clear suppression case:
        current2 = np.array([0.34, 0.33, 0.33])  # very close to 1/3
        result2 = process_actions(raw, current2, deadzone=0.02)
        # diff is ~0.006 for all => all suppressed => returns current
        np.testing.assert_array_almost_equal(result2, current2)

    def test_deadzone_allows_large_changes(self) -> None:
        """TRAIN-09: Changes above deadzone threshold are applied."""
        current = np.array([0.0, 0.0])
        raw = np.array([1.0, -1.0])  # softmax ~ [0.88, 0.12]
        result = process_actions(raw, current, deadzone=0.02)
        # Both changes > 0.02, so new weights should differ from current
        assert not np.allclose(result, current)


# ---------------------------------------------------------------------------
# RollingSharpeReward
# ---------------------------------------------------------------------------


class TestRollingSharpeReward:
    """Rolling 20-day Sharpe reward with expanding warmup."""

    def test_first_return_is_zero(self) -> None:
        """TRAIN-07: First data point returns 0.0 (< 2 returns)."""
        rsr = RollingSharpeReward(window=20)
        assert rsr.compute(0.01) == pytest.approx(0.0)

    def test_expanding_window_warmup(self) -> None:
        """TRAIN-07: Bars 2-19 use expanding window (not full 20)."""
        rsr = RollingSharpeReward(window=20)
        # Feed 5 returns
        for _ in range(4):
            rsr.compute(0.01)
        # 5th return should use expanding window (5 returns, ddof=1)
        result = rsr.compute(0.01)
        # 5 returns of 0.01: mean=0.01, std=0.0 => 0.0 (near-zero std guard)
        assert result == pytest.approx(0.0)

    def test_expanding_window_nonzero(self) -> None:
        """TRAIN-07: Expanding window returns nonzero for varied returns."""
        rsr = RollingSharpeReward(window=20)
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        results = [rsr.compute(r) for r in returns]
        # After 5 varied returns, should have a nonzero Sharpe
        assert results[-1] != 0.0

    def test_rolling_window_from_bar_20(self) -> None:
        """TRAIN-07: From bar 20+, uses rolling 20-day window."""
        rsr = RollingSharpeReward(window=20)
        rng = np.random.default_rng(99)
        returns = rng.normal(0.001, 0.01, 25)

        results = []
        for r in returns:
            results.append(rsr.compute(r))

        # Bar 20+ should use only last 20 returns
        assert results[-1] != 0.0  # Should produce a value

    def test_zero_std_returns_zero(self) -> None:
        """TRAIN-07: Returns 0.0 when standard deviation is near-zero."""
        rsr = RollingSharpeReward(window=20)
        # Feed identical returns
        for _ in range(5):
            result = rsr.compute(0.01)
        # All same => std ~ 0 => returns 0.0
        assert result == pytest.approx(0.0)

    def test_reset_clears_history(self) -> None:
        """TRAIN-07: reset() clears return history."""
        rsr = RollingSharpeReward(window=20)
        for _ in range(10):
            rsr.compute(0.01)
        rsr.reset()
        # After reset, first compute returns 0.0 again
        assert rsr.compute(0.01) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EnvironmentConfig schema
# ---------------------------------------------------------------------------


class TestEnvironmentConfig:
    """EnvironmentConfig fields accessible via SwingRLConfig."""

    def test_default_initial_amount(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-07: Default initial_amount is 100_000."""
        assert loaded_config.environment.initial_amount == pytest.approx(100_000.0)

    def test_default_equity_episode_bars(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-11: Default equity_episode_bars is 252."""
        assert loaded_config.environment.equity_episode_bars == 252

    def test_default_crypto_episode_bars(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-11: Default crypto_episode_bars is 540."""
        assert loaded_config.environment.crypto_episode_bars == 540

    def test_default_transaction_costs(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-07: Transaction cost defaults match spec."""
        assert loaded_config.environment.equity_transaction_cost_pct == pytest.approx(0.0006)
        assert loaded_config.environment.crypto_transaction_cost_pct == pytest.approx(0.0022)

    def test_default_signal_deadzone(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-09: Default signal_deadzone is 0.02."""
        assert loaded_config.environment.signal_deadzone == pytest.approx(0.02)

    def test_default_penalty_coefficients(self, loaded_config) -> None:  # type: ignore[no-untyped-def]
        """TRAIN-07: Default penalty coefficients are set."""
        assert loaded_config.environment.position_penalty_coeff == pytest.approx(10.0)
        assert loaded_config.environment.drawdown_penalty_coeff == pytest.approx(5.0)
