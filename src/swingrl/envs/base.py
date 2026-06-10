"""Base Gymnasium environment for RL trading.

BaseTradingEnv provides shared portfolio simulation, reward computation, and
soft risk penalties for both equity and crypto environments. Subclasses
specialize episode length, observation dimensions, and start-step selection.

Observations are returned raw (not normalized) -- VecNormalize wraps the
environment externally during training (TRAIN-08).
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium
import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.envs.portfolio import PortfolioSimulator, process_actions
from swingrl.envs.rewards import RollingSharpeReward
from swingrl.features.assembler import (
    CRYPTO_OBS_DIM,
    CRYPTO_PORTFOLIO,
    EQUITY_PORTFOLIO,
    equity_obs_dim,
)

log = structlog.get_logger(__name__)


class BaseTradingEnv(gymnasium.Env):
    """Base Gymnasium environment for portfolio-based RL trading.

    Implements the shared step/reset contract, portfolio simulation via
    PortfolioSimulator, rolling Sharpe reward, and soft risk penalties.
    Subclasses override _select_start_step() for episode positioning.

    Args:
        features: Pre-loaded feature array, shape (n_steps, obs_dim), float32.
        prices: Pre-loaded close price array, shape (n_steps, n_assets), float32.
        config: Validated SwingRLConfig instance.
        environment: Which environment type ("equity" or "crypto").
        render_mode: Gymnasium render mode (optional).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: SwingRLConfig,
        environment: Literal["equity", "crypto"],
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._features = features
        self._prices = prices
        self._config = config
        self._environment = environment

        # Resolve environment-specific parameters
        if environment == "equity":
            self._obs_dim = equity_obs_dim(
                sentiment_enabled=config.sentiment.enabled,
                n_equity_symbols=len(config.equity.symbols),
            )
            self._portfolio_dim = EQUITY_PORTFOLIO
            self._n_assets = len(config.equity.symbols)
            self._episode_bars = config.environment.equity_episode_bars
            self._transaction_cost_pct = config.environment.equity_transaction_cost_pct
            self._max_position_size = config.equity.max_position_size
            self._max_drawdown_pct = config.equity.max_drawdown_pct
        else:
            self._obs_dim = CRYPTO_OBS_DIM
            self._portfolio_dim = CRYPTO_PORTFOLIO
            self._n_assets = len(config.crypto.symbols)
            self._episode_bars = config.environment.crypto_episode_bars
            self._transaction_cost_pct = config.environment.crypto_transaction_cost_pct
            self._max_position_size = config.crypto.max_position_size
            self._max_drawdown_pct = config.crypto.max_drawdown_pct

        self._initial_amount = config.environment.initial_amount
        self._deadzone = config.environment.signal_deadzone
        self._position_penalty_coeff = config.environment.position_penalty_coeff
        self._drawdown_penalty_coeff = config.environment.drawdown_penalty_coeff

        # Gymnasium spaces
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_assets + 1,),
            dtype=np.float32,
        )

        # Portfolio simulation components
        self._portfolio = PortfolioSimulator(
            initial_cash=self._initial_amount,
            n_assets=self._n_assets,
            transaction_cost_pct=self._transaction_cost_pct,
        )
        self._reward_fn = RollingSharpeReward(window=20)

        # Episode state (initialized in reset)
        self._current_step: int = 0
        self._max_step: int = 0
        self._peak_value: float = self._initial_amount
        self._prev_value: float = self._initial_amount
        self._step_count: int = 0
        self._last_trade_step: np.ndarray = np.zeros(self._n_assets, dtype=np.int32)
        self._last_cost: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to start of a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        self._current_step = self._select_start_step()
        # Cap max_step so step() never indexes past the last row of prices/features.
        # step() increments current_step then reads prices[current_step], so the
        # highest valid current_step after increment is len-1.
        self._max_step = min(
            self._current_step + self._episode_bars,
            len(self._features) - 1,
        )

        # Reset portfolio and reward
        self._portfolio.reset(self._initial_amount)
        self._reward_fn.reset()

        # Reset episode tracking
        self._peak_value = self._initial_amount
        self._prev_value = self._initial_amount
        self._step_count = 0
        self._last_trade_step = np.zeros(self._n_assets, dtype=np.int32)
        self._last_cost = 0.0

        # Build initial observation with 100% cash portfolio state
        prices = self._prices[self._current_step]
        portfolio_state = self._get_portfolio_state(prices)
        obs = self._build_observation(portfolio_state)

        info = self._build_info(
            portfolio_value=self._initial_amount,
            daily_return=0.0,
            transaction_cost=0.0,
        )

        log.debug(
            "episode_reset",
            environment=self._environment,
            start_step=self._current_step,
            episode_bars=self._episode_bars,
        )

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Raw agent action, shape (n_assets + 1,). Last element is cash preference.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        prices = self._prices[self._current_step]
        current_weights = self._portfolio.asset_weights(prices)

        # 1. Process actions via softmax + deadzone
        target_weights = process_actions(action, current_weights, self._deadzone)

        # 2. Execute trades at current prices
        cost = self._portfolio.rebalance(target_weights, prices)
        self._last_cost = cost

        # Track which assets were traded
        new_weights = self._portfolio.asset_weights(prices)
        for i in range(self._n_assets):
            if abs(new_weights[i] - current_weights[i]) > 1e-8:
                self._last_trade_step[i] = self._step_count

        # 3. Advance step
        self._current_step += 1
        self._step_count += 1

        # 4. Compute new portfolio value at new prices
        new_prices = self._prices[self._current_step]
        new_value = self._portfolio.portfolio_value(new_prices)

        # 5. Compute daily return
        daily_return = (
            (new_value - self._prev_value) / self._prev_value if self._prev_value > 0 else 0.0
        )

        # 6. Update peak value
        self._peak_value = max(self._peak_value, new_value)

        # 7. Compute reward = sharpe - risk penalty
        sharpe_reward = self._reward_fn.compute(daily_return)
        weights_after = self._portfolio.asset_weights(new_prices)
        risk_penalty = self._compute_risk_penalty(weights_after, new_value)
        reward = sharpe_reward - risk_penalty

        # 7b. Compute reward components for memory-guided shaping
        drawdown_frac = 0.0
        if self._peak_value > 0:
            drawdown_frac = (self._peak_value - new_value) / self._peak_value
        turnover_ratio = cost / self._prev_value if self._prev_value > 0 else 0.0

        reward_components = {
            "profit": daily_return,
            "sharpe": sharpe_reward,
            "drawdown": -drawdown_frac,
            "turnover": -turnover_ratio,
        }

        # 8. Build observation with current portfolio state
        portfolio_state = self._get_portfolio_state(new_prices)
        obs = self._build_observation(portfolio_state)

        # 9. Termination: when we reach the capped max_step boundary
        terminated = self._current_step >= self._max_step
        truncated = False

        # 10. Build info
        info = self._build_info(
            portfolio_value=new_value,
            daily_return=daily_return,
            transaction_cost=cost,
            reward_components=reward_components,
        )

        # Update previous value for next step
        self._prev_value = new_value

        if terminated:
            # Include trade log before DummyVecEnv auto-resets the env
            info["trade_log"] = list(self._portfolio.trade_log)
            log.info(
                "episode_complete",
                environment=self._environment,
                steps=self._step_count,
                final_value=new_value,
                total_return=(new_value - self._initial_amount) / self._initial_amount,
            )

        return obs, reward, terminated, truncated, info

    def _build_observation(self, portfolio_state: np.ndarray) -> np.ndarray:
        """Build observation vector with actual portfolio state.

        Takes the pre-computed features for the current step and replaces
        the last portfolio_dim elements with actual portfolio state.

        Args:
            portfolio_state: Current portfolio state vector.

        Returns:
            Full observation vector, shape (obs_dim,), float32.
        """
        obs = self._features[self._current_step].copy()
        obs[-self._portfolio_dim :] = portfolio_state
        result: np.ndarray = obs.astype(np.float32)
        return result

    def _get_portfolio_state(self, prices: np.ndarray) -> np.ndarray:
        """Compute portfolio state vector for observation.

        Layout: [cash_ratio, total_exposure, daily_return]
                + per-asset interleaved [weight, weight_deviation, unrealized_pnl, bars_since_trade]

        Args:
            prices: Current prices, shape (n_assets,).

        Returns:
            Portfolio state vector, shape (portfolio_dim,).
        """
        total_value = self._portfolio.portfolio_value(prices)
        weights = self._portfolio.asset_weights(prices)

        # Fixed components (3)
        cash_ratio = self._portfolio.cash / total_value if total_value > 0 else 1.0
        total_exposure = float(np.sum(weights))
        daily_return = (
            (total_value - self._prev_value) / self._prev_value if self._prev_value > 0 else 0.0
        )

        state = np.zeros(self._portfolio_dim, dtype=np.float32)
        state[0] = cash_ratio
        state[1] = total_exposure
        state[2] = daily_return

        # Per-asset components (4 * n_assets), interleaved
        for i in range(self._n_assets):
            base_idx = 3 + i * 4
            state[base_idx] = weights[i]
            # Weight deviation from equal-weight target
            state[base_idx + 1] = weights[i] - (1.0 / self._n_assets) if total_exposure > 0 else 0.0
            # Unrealized PnL pct from cost basis
            cost_basis = self._portfolio.cost_basis[i]
            if cost_basis > 0:
                state[base_idx + 2] = (prices[i] - cost_basis) / cost_basis
            else:
                state[base_idx + 2] = 0.0
            # Bars since last trade
            state[base_idx + 3] = float(self._step_count - self._last_trade_step[i])

        return state

    def _compute_risk_penalty(self, weights: np.ndarray, portfolio_value: float) -> float:
        """Compute soft risk penalty for position concentration and drawdown.

        Args:
            weights: Current asset weights, shape (n_assets,).
            portfolio_value: Current portfolio value in dollars.

        Returns:
            Total risk penalty (non-negative).
        """
        penalty = 0.0

        # Position concentration: quadratic penalty on weights exceeding max_position_size
        for w in weights:
            excess = max(0.0, float(w) - self._max_position_size)
            penalty += self._position_penalty_coeff * excess * excess

        # Drawdown: linear penalty on drawdown exceeding max_drawdown_pct
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
            excess_dd = max(0.0, drawdown - self._max_drawdown_pct)
            penalty += self._drawdown_penalty_coeff * excess_dd

        return penalty

    def _get_turbulence(self) -> float:
        """Return turbulence value for the current step.

        Default implementation returns 0.0. Turbulence data can be optionally
        passed in features or overridden by subclasses.
        """
        return 0.0

    def _select_start_step(self) -> int:
        """Select starting step for episode. Overridden by subclasses.

        Returns:
            Starting index into features/prices arrays.
        """
        return 0

    def _build_info(
        self,
        portfolio_value: float,
        daily_return: float,
        transaction_cost: float,
        reward_components: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build info dict for step/reset return.

        Args:
            portfolio_value: Current portfolio value.
            daily_return: Return since last step.
            transaction_cost: Cost incurred this step.
            reward_components: Optional decomposed reward components for
                memory-guided shaping (profit, sharpe, drawdown, turnover).

        Returns:
            Info dictionary with required keys.
        """
        info: dict[str, Any] = {
            "portfolio_value": portfolio_value,
            "daily_return": daily_return,
            "transaction_cost": transaction_cost,
            "turbulence": self._get_turbulence(),
            "step": self._step_count,
        }
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info
