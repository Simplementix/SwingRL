"""Order validator: cost gate and risk manager delegation.

Stage 3 of the execution pipeline. Validates sized orders against
transaction cost thresholds and delegates to RiskManager for policy checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from swingrl.execution.types import SizedOrder, ValidatedOrder
from swingrl.utils.exceptions import RiskVetoError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.execution.risk.risk_manager import RiskManager

log = structlog.get_logger(__name__)

# Cost gate threshold: reject orders where RT costs exceed this fraction
COST_GATE_THRESHOLD = 0.02  # 2%


class OrderValidator:
    """Validate orders against cost gate and risk rules before broker submission.

    Stage 3 applies two layers of validation:
    1. Cost gate: reject if estimated round-trip transaction costs exceed 2%
    2. Risk checks: delegate to RiskManager for position/exposure/drawdown limits
    """

    # Round-trip cost estimates by environment
    # Equity: SEC fee + TAF, no commission on Alpaca (~0.06%)
    EQUITY_RT_COST_PCT: float = 0.0006
    # Crypto: Binance.US 0.10% maker + 0.10% taker + slippage (~0.22%)
    CRYPTO_RT_COST_PCT: float = 0.0022

    def __init__(self, config: SwingRLConfig, risk_manager: RiskManager) -> None:
        """Initialize order validator.

        Args:
            config: SwingRL configuration.
            risk_manager: RiskManager instance for policy checks.
        """
        self._config = config
        self._risk_manager = risk_manager

    def validate(self, order: SizedOrder) -> ValidatedOrder:
        """Validate an order through cost gate and risk checks.

        Args:
            order: Sized order from PositionSizer.

        Returns:
            ValidatedOrder with list of passed check names.

        Raises:
            RiskVetoError: If cost gate or any risk check fails.
        """
        # Step 1: Cost gate
        cost_rate = self._get_cost_rate(order.environment)
        rt_cost = order.dollar_amount * cost_rate
        cost_pct = rt_cost / order.dollar_amount if order.dollar_amount > 0 else 1.0

        if cost_pct > COST_GATE_THRESHOLD:
            log.warning(
                "cost_gate_veto",
                symbol=order.symbol,
                dollar_amount=order.dollar_amount,
                rt_cost=rt_cost,
                cost_pct=cost_pct,
            )
            raise RiskVetoError(
                f"cost_gate: RT cost {cost_pct:.4%} exceeds {COST_GATE_THRESHOLD:.1%} threshold"
            )

        # Step 2: Risk manager delegation (may raise RiskVetoError or CircuitBreakerError)
        self._risk_manager.evaluate(order)

        # All checks passed
        passed_checks = [
            "cost_gate",
            "position_size",
            "exposure",
            "drawdown",
            "daily_loss",
            "global_aggregator",
        ]

        log.info(
            "order_validated",
            symbol=order.symbol,
            side=order.side,
            dollar_amount=order.dollar_amount,
            checks_passed=len(passed_checks),
        )

        return ValidatedOrder(
            order=order,
            risk_checks_passed=passed_checks,
        )

    def _get_cost_rate(self, environment: str) -> float:
        """Get round-trip cost rate for the environment."""
        if environment == "equity":
            return self.EQUITY_RT_COST_PCT
        return self.CRYPTO_RT_COST_PCT
