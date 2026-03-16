"""Portfolio simulation and action processing for RL trading environments.

PortfolioSimulator tracks cash, positions, and portfolio value with dollar-based
accounting and transaction cost deduction. process_actions converts raw agent
outputs to target portfolio weights via numerically stable softmax + deadzone.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class PortfolioSimulator:
    """Dollar-based portfolio tracker with rebalancing and trade logging.

    Tracks cash, share positions, and total portfolio value. Rebalancing
    adjusts holdings to target percentage weights and deducts proportional
    transaction costs.

    Args:
        initial_cash: Starting cash amount in dollars.
        n_assets: Number of tradeable assets.
        transaction_cost_pct: Proportional cost per dollar traded (round-trip).
    """

    def __init__(
        self,
        initial_cash: float,
        n_assets: int,
        transaction_cost_pct: float,
    ) -> None:
        self.cash: float = initial_cash
        self.shares: np.ndarray = np.zeros(n_assets, dtype=np.float64)
        self.n_assets: int = n_assets
        self.transaction_cost_pct: float = transaction_cost_pct
        self.trade_log: list[dict[str, object]] = []

    def portfolio_value(self, prices: np.ndarray) -> float:
        """Compute total portfolio value: cash + sum(shares * prices).

        Args:
            prices: Current price per asset, shape (n_assets,).

        Returns:
            Total portfolio value in dollars.
        """
        return float(self.cash + np.sum(self.shares * prices))

    def asset_weights(self, prices: np.ndarray) -> np.ndarray:
        """Compute current weight of each asset in the portfolio.

        Args:
            prices: Current price per asset, shape (n_assets,).

        Returns:
            Per-asset weight array, shape (n_assets,). Sums to <= 1.0
            (remainder is cash weight).
        """
        total = self.portfolio_value(prices)
        if total <= 0.0:
            return np.zeros(self.n_assets, dtype=np.float64)
        weights: np.ndarray = (self.shares * prices) / total
        return weights

    def rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> float:
        """Rebalance portfolio to target asset weights.

        Computes the dollar-value trades needed to reach target_weights,
        deducts transaction costs proportional to traded volume, and updates
        share positions and cash.

        Args:
            target_weights: Desired weight per asset, shape (n_assets,).
                           Should sum to <= 1.0 (remainder stays as cash).
            prices: Current price per asset, shape (n_assets,).

        Returns:
            Total transaction cost deducted in dollars.
        """
        total_value = self.portfolio_value(prices)
        target_values = target_weights * total_value

        # Current dollar value per asset
        current_values = self.shares * prices

        # Dollar amount traded per asset
        deltas = target_values - current_values

        # Transaction cost on total traded volume
        cost = float(np.sum(np.abs(deltas))) * self.transaction_cost_pct

        # Safe division: avoid divide-by-zero for assets with price 0
        safe_prices = np.where(prices > 0.0, prices, 1.0)
        self.shares = target_values / safe_prices

        # Cash = total - invested - costs
        self.cash = total_value - float(np.sum(target_values)) - cost

        # Log individual trades
        for i in range(self.n_assets):
            delta_shares = deltas[i] / safe_prices[i]
            if abs(delta_shares) > 1e-10:
                side = "buy" if delta_shares > 0 else "sell"
                trade: dict[str, object] = {
                    "asset_idx": i,
                    "side": side,
                    "shares": float(abs(delta_shares)),
                    "price": float(safe_prices[i]),
                    "value": float(abs(deltas[i])),
                    "cost": float(abs(deltas[i])) * self.transaction_cost_pct,
                }
                self.trade_log.append(trade)
                log.debug(
                    "trade_executed",
                    asset_idx=i,
                    side=side,
                    shares=float(abs(delta_shares)),
                )

        return cost

    def reset(self, initial_cash: float) -> None:
        """Reset portfolio to initial state with full cash and zero positions.

        Args:
            initial_cash: Starting cash amount in dollars.
        """
        self.cash = initial_cash
        self.shares = np.zeros(self.n_assets, dtype=np.float64)
        self.trade_log = []


def process_actions(
    raw_actions: np.ndarray,
    current_weights: np.ndarray,
    deadzone: float,
) -> np.ndarray:
    """Convert raw agent actions to target portfolio weights.

    Applies numerically stable softmax (subtract-max trick) to convert
    unbounded agent outputs to positive weights summing to 1.0, then
    applies a deadzone filter to suppress small weight changes.

    Args:
        raw_actions: Raw agent output, shape (n_assets,).
        current_weights: Current portfolio weights, shape (n_assets,).
        deadzone: Minimum absolute weight change to trigger a trade.

    Returns:
        Target weights array, shape (n_assets,). Positive values summing
        to <= 1.0.
    """
    # Numerically stable softmax: subtract max to prevent overflow
    shifted = raw_actions - np.max(raw_actions)
    exp_vals = np.exp(shifted)
    softmax_weights = exp_vals / np.sum(exp_vals)

    # Apply deadzone: keep current weight where change is below threshold
    diff = np.abs(softmax_weights - current_weights)
    target_weights = np.where(diff >= deadzone, softmax_weights, current_weights)

    return target_weights
