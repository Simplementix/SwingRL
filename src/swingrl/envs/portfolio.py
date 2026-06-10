"""Portfolio simulation and action processing for RL trading environments.

PortfolioSimulator tracks cash, positions, cost basis, and portfolio value with
dollar-based accounting and transaction cost deduction. process_actions converts
raw agent outputs (including cash preference) to target portfolio weights via
numerically stable softmax + deadzone.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class PortfolioSimulator:
    """Dollar-based portfolio tracker with rebalancing and trade logging.

    Tracks cash, share positions, cost basis, and total portfolio value.
    Rebalancing adjusts holdings to target percentage weights and deducts
    proportional transaction costs.

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
        self.cost_basis: np.ndarray = np.zeros(n_assets, dtype=np.float64)
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
        share positions, cost basis, and cash.

        Args:
            target_weights: Desired weight per asset, shape (n_assets,).
                           Should sum to <= 1.0 (remainder stays as cash).
            prices: Current price per asset, shape (n_assets,).

        Returns:
            Total transaction cost deducted in dollars.
        """
        total_value = self.portfolio_value(prices)

        # Current dollar value per asset
        current_values = self.shares * prices

        # Estimate transaction cost BEFORE allocation to avoid over-investing
        est_deltas = target_weights * total_value - current_values
        est_cost = float(np.sum(np.abs(est_deltas))) * self.transaction_cost_pct
        adjusted_total = total_value - est_cost
        target_values = target_weights * adjusted_total

        # Recompute deltas with cost-adjusted target values
        deltas = target_values - current_values

        # Transaction cost on total traded volume
        cost = float(np.sum(np.abs(deltas))) * self.transaction_cost_pct

        # Safe division: avoid divide-by-zero for assets with price 0
        safe_prices = np.where(prices > 0.0, prices, 1.0)
        old_shares = self.shares.copy()
        new_shares = target_values / safe_prices

        # Update cost basis using weighted-average tracking
        for i in range(self.n_assets):
            if new_shares[i] > old_shares[i] and new_shares[i] > 0:
                # Buying: weighted average of old cost basis and new purchase price
                old_value = old_shares[i] * self.cost_basis[i]
                new_value = (new_shares[i] - old_shares[i]) * prices[i]
                self.cost_basis[i] = (old_value + new_value) / new_shares[i]
            elif new_shares[i] <= 0:
                self.cost_basis[i] = 0.0
            # If selling (new < old), cost basis unchanged

        self.shares = new_shares

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
        self.cost_basis = np.zeros(self.n_assets, dtype=np.float64)
        self.trade_log = []


def process_actions(
    raw_actions: np.ndarray,
    current_weights: np.ndarray,
    deadzone: float,
) -> np.ndarray:
    """Convert raw agent actions to target portfolio weights.

    Applies numerically stable softmax (subtract-max trick) across all
    dimensions (n_assets + 1, where last element is cash preference) to
    produce weights summing to 1.0. The cash dimension is then removed,
    leaving asset weights that sum to <= 1.0 (the implicit cash allocation
    is the remainder). A deadzone filter suppresses small weight changes.

    Args:
        raw_actions: Raw agent output, shape (n_assets + 1,). Last element
            is cash preference.
        current_weights: Current portfolio weights, shape (n_assets,).
        deadzone: Minimum absolute weight change to trigger a trade.

    Returns:
        Target weights array, shape (n_assets,). Positive values summing
        to <= 1.0.
    """
    # Numerically stable softmax across all dimensions (assets + cash)
    shifted = raw_actions - np.max(raw_actions)
    exp_vals = np.exp(shifted)
    all_weights = exp_vals / np.sum(exp_vals)

    # Remove cash dimension (last element); asset weights sum to < 1.0
    softmax_weights = all_weights[:-1]

    # Apply deadzone: keep current weight where change is below threshold
    diff = np.abs(softmax_weights - current_weights)
    target_weights = np.where(diff >= deadzone, softmax_weights, current_weights)

    # Clamp total weight to <= 1.0 (deadzone can preserve old weights that sum > 1)
    total = float(np.sum(target_weights))
    if total > 1.0:
        target_weights = target_weights * (1.0 / total)

    return target_weights
