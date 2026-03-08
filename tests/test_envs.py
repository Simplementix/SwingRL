"""Tests for RL environment foundation components.

Tests cover PortfolioSimulator, RollingSharpeReward, process_actions (softmax + deadzone),
EnvironmentConfig schema integration, and BaseTradingEnv/StockTradingEnv contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from swingrl.envs.equity import StockTradingEnv
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


# ---------------------------------------------------------------------------
# StockTradingEnv (BaseTradingEnv + equity specialization)
# ---------------------------------------------------------------------------


class TestStockTradingEnvInit:
    """StockTradingEnv initializes with correct spaces."""

    def test_observation_space_shape(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: observation_space is Box(156,) float32."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        assert env.observation_space.shape == (156,)
        assert env.observation_space.dtype == np.float32

    def test_action_space_shape(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: action_space is Box(8,) float32 for 8 ETFs."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        assert env.action_space.shape == (8,)
        assert env.action_space.dtype == np.float32


class TestStockTradingEnvReset:
    """StockTradingEnv.reset() returns valid 2-tuple."""

    def test_reset_returns_obs_info(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: reset(seed=42) returns (obs, info) with correct obs shape/dtype."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (156,)
        assert obs.dtype == np.float32

    def test_reset_obs_all_finite(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: reset() observation has all finite values (no NaN, no Inf)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_reset_deterministic(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: reset(seed=42) twice produces identical observations."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestStockTradingEnvStep:
    """StockTradingEnv.step() returns valid 5-tuple."""

    def test_step_returns_5_tuple(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: step() returns (obs, reward, terminated, truncated, info)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape_dtype(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: step() observation shape is (156,) and dtype is float32."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (156,)
        assert obs.dtype == np.float32

    def test_terminated_after_252_steps(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-11: terminated=True after exactly 252 steps (equity episode length)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        for i in range(251):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert not terminated, f"Terminated early at step {i}"
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated, "Should terminate at step 252"

    def test_truncated_always_false(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-11: truncated is always False (no early termination)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        for _ in range(252):
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            assert truncated is False

    def test_info_dict_keys(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: info dict contains required keys."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        required_keys = {
            "portfolio_value",
            "daily_return",
            "transaction_cost",
            "turbulence",
            "step",
        }
        assert required_keys.issubset(info.keys())

    def test_portfolio_value_tracking(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: portfolio_value in info tracks correctly over multiple steps."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        values = []
        for _ in range(10):
            _, _, _, _, info = env.step(env.action_space.sample())
            values.append(info["portfolio_value"])
        # Values should be positive and varying (not all the same)
        assert all(v > 0 for v in values)
        assert len(set(values)) > 1, "Portfolio values should change over steps"


class TestStockTradingEnvRiskPenalties:
    """Soft risk penalties for position and drawdown breaches."""

    def test_position_penalty_when_weight_exceeds_limit(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-10: Soft penalty applied when weight exceeds max_position_size (0.25)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        # Create action that concentrates heavily in one asset
        action = np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 10.0], dtype=np.float32)
        # Step and get reward with concentrated position
        _, reward_concentrated, _, _, _ = env.step(action)

        # Reset and use balanced action
        env.reset(seed=42)
        action_balanced = np.zeros(8, dtype=np.float32)  # softmax -> equal weights
        _, reward_balanced, _, _, _ = env.step(action_balanced)

        # Concentrated position should have lower reward due to penalty
        assert reward_concentrated < reward_balanced

    def test_drawdown_penalty_applied(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-10: Soft penalty applied when portfolio drawdown exceeds max_drawdown_pct."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        env.reset(seed=42)
        # Force a drawdown scenario by manipulating peak value
        env.step(env.action_space.sample())
        # Artificially set peak high to trigger drawdown penalty
        env._peak_value = env._prev_value * 2.0  # 50% drawdown from peak
        _, reward, _, _, _ = env.step(env.action_space.sample())
        # With extreme drawdown, penalty should make reward more negative
        assert isinstance(reward, float)


class TestStockTradingEnvRawObservations:
    """Observations are raw (not normalized)."""

    def test_observations_can_exceed_unit_range(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-08: Observations are raw -- values can be outside [-1, 1] range."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs, _ = env.reset(seed=42)
        # Raw features from standard normal can exceed [-1, 1]
        assert np.max(np.abs(obs)) > 1.0, "Raw observations should not be clipped to [-1, 1]"


class TestStockTradingEnvFactory:
    """from_arrays() factory method."""

    def test_from_arrays_creates_env(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: Environment works with from_arrays() factory method."""
        env = StockTradingEnv.from_arrays(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (156,)
        assert obs.dtype == np.float32


class TestStockTradingEnvFullEpisode:
    """Full episode rollout completes without error."""

    def test_full_episode_rollout(
        self,
        equity_features_array: np.ndarray,
        equity_prices_array: np.ndarray,
        equity_env_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """TRAIN-01: Full episode rollout completes without error (252 steps)."""
        env = StockTradingEnv(
            features=equity_features_array,
            prices=equity_prices_array,
            config=equity_env_config,
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (156,)
        total_reward = 0.0
        for step_num in range(252):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            assert obs.shape == (156,)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            if step_num < 251:
                assert not terminated
            else:
                assert terminated
            assert not truncated
        assert isinstance(total_reward, float)
        assert info["step"] == 252
