"""Two-tier risk veto layer.

Evaluates orders against per-environment and global portfolio risk limits.
All veto decisions are logged to risk_decisions table for audit.

Check order (Doc 04):
1. Circuit breaker state
2. Per-env position size vs max
3. Per-env exposure check
4. Per-env drawdown check
5. Per-env daily loss check
6. Global aggregator
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from swingrl.execution.risk.circuit_breaker import CBState
from swingrl.execution.types import RiskDecision, SizedOrder
from swingrl.utils.exceptions import CircuitBreakerError, RiskVetoError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.risk.circuit_breaker import CircuitBreaker, GlobalCircuitBreaker
    from swingrl.execution.risk.position_tracker import PositionTracker

log = structlog.get_logger(__name__)


class RiskManager:
    """Evaluate orders against risk rules and veto if policy violated.

    Args:
        config: SwingRLConfig with risk thresholds.
        db: DatabaseManager for decision logging.
        position_tracker: PositionTracker for portfolio state.
        circuit_breakers: Dict of env -> CircuitBreaker.
        global_cb: GlobalCircuitBreaker for combined checks.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        db: DatabaseManager,
        position_tracker: PositionTracker,
        circuit_breakers: dict[str, CircuitBreaker],
        global_cb: GlobalCircuitBreaker,
    ) -> None:
        """Initialize risk manager with all dependencies."""
        self._config = config
        self._db = db
        self._tracker = position_tracker
        self._circuit_breakers = circuit_breakers
        self._global_cb = global_cb

    def evaluate(self, order: SizedOrder) -> RiskDecision:
        """Evaluate an order against all risk rules.

        Checks in order: CB state, position size, exposure, drawdown,
        daily loss, global aggregator.

        Args:
            order: Sized order to evaluate.

        Returns:
            RiskDecision indicating approval.

        Raises:
            CircuitBreakerError: If CB is halted for this environment.
            RiskVetoError: If any risk check fails.
        """
        env = order.environment
        cb = self._circuit_breakers.get(env)

        # 1. Circuit breaker state check
        if cb is not None:
            state = cb.get_state()
            if state == CBState.HALTED:
                decision = self._make_decision(
                    order, "rejected", "circuit_breaker", "circuit breaker halted"
                )
                self._record_decision(decision)
                log.error(
                    "risk_veto_circuit_breaker",
                    environment=env,
                    symbol=order.symbol,
                )
                raise CircuitBreakerError(f"Circuit breaker halted for {env}; trading suspended")

        # Get portfolio state for remaining checks
        portfolio_value = self._tracker.get_portfolio_value(env)
        env_config = self._config.equity if env == "equity" else self._config.crypto

        # 2. Position size check
        if portfolio_value > 0:
            position_pct = order.dollar_amount / portfolio_value
            if position_pct > env_config.max_position_size:
                self._veto(
                    order,
                    "position_size",
                    f"position_size {position_pct:.4f} exceeds max {env_config.max_position_size}",
                )

        # 3. Exposure check
        current_exposure = self._tracker.get_exposure(env)
        new_exposure = current_exposure + (
            order.dollar_amount / portfolio_value if portfolio_value > 0 else 0.0
        )
        if new_exposure > 1.0:
            self._veto(
                order,
                "exposure",
                f"total exposure {new_exposure:.4f} would exceed 1.0",
            )

        # 4. Drawdown check
        hwm = self._tracker.get_high_water_mark(env)
        if hwm > 0:
            current_dd = 1.0 - portfolio_value / hwm
            if current_dd >= env_config.max_drawdown_pct:
                # Trigger CB and reject
                if cb is not None:
                    cb.check_and_update(
                        portfolio_value=portfolio_value,
                        high_water_mark=hwm,
                        daily_pnl=self._tracker.get_daily_pnl(env),
                    )
                decision = self._make_decision(
                    order,
                    "rejected",
                    "drawdown_circuit_breaker",
                    f"drawdown {current_dd:.4f} >= {env_config.max_drawdown_pct}",
                )
                self._record_decision(decision)
                raise CircuitBreakerError(
                    f"Drawdown {current_dd:.4f} triggered circuit breaker for {env}"
                )

        # 5. Daily loss check (use HWM as denominator — consistent with circuit_breaker)
        daily_pnl = self._tracker.get_daily_pnl(env)
        if daily_pnl < 0 and hwm > 0:
            daily_loss_pct = abs(daily_pnl) / hwm
            if daily_loss_pct >= env_config.daily_loss_limit_pct:
                # Trigger CB
                if cb is not None:
                    cb.check_and_update(
                        portfolio_value=portfolio_value,
                        high_water_mark=hwm,
                        daily_pnl=daily_pnl,
                    )
                self._veto(
                    order,
                    "daily_loss",
                    f"daily_loss {daily_loss_pct:.4f} >= {env_config.daily_loss_limit_pct}",
                )

        # 6. Global aggregator check
        eq_value = self._tracker.get_portfolio_value("equity")
        cr_value = self._tracker.get_portfolio_value("crypto")
        eq_pnl = self._tracker.get_daily_pnl("equity")
        cr_pnl = self._tracker.get_daily_pnl("crypto")
        global_triggered = self._global_cb.check_combined(
            portfolio_values={"equity": eq_value, "crypto": cr_value},
            daily_pnls={"equity": eq_pnl, "crypto": cr_pnl},
        )
        if global_triggered:
            decision = self._make_decision(
                order,
                "rejected",
                "global_circuit_breaker",
                "global portfolio limits breached",
            )
            self._record_decision(decision)
            raise CircuitBreakerError("Global circuit breaker triggered; all trading suspended")

        # Scale order if CB is ramping
        if cb is not None:
            capacity = cb.get_capacity_fraction()
            if capacity < 1.0:
                log.info(
                    "order_scaled_by_ramp",
                    environment=env,
                    capacity=capacity,
                    original_amount=order.dollar_amount,
                )

        # All checks passed
        decision = self._make_decision(order, order.side, "none", "approved")
        self._record_decision(decision)
        log.info(
            "risk_approved",
            symbol=order.symbol,
            side=order.side,
            dollar_amount=order.dollar_amount,
            environment=env,
        )
        return decision

    def check_turbulence(
        self,
        env: str,
        turbulence_value: float,
        historical_90th_pct: float,
    ) -> bool:
        """Check turbulence for crash protection.

        If turbulence exceeds 90th percentile, triggers CB for environment
        and signals liquidation.

        Args:
            env: "equity" or "crypto".
            turbulence_value: Current turbulence index value.
            historical_90th_pct: 90th percentile of historical turbulence.

        Returns:
            True if liquidation signal (turbulence exceeded), False otherwise.
        """
        if turbulence_value > historical_90th_pct:
            log.critical(
                "turbulence_crash_protection",
                environment=env,
                turbulence=turbulence_value,
                threshold=historical_90th_pct,
            )
            cb = self._circuit_breakers.get(env)
            if cb is not None:
                cb._trigger(
                    trigger_value=turbulence_value,
                    threshold=historical_90th_pct,
                    reason=f"turbulence_{turbulence_value:.4f}_exceeds_90th_pct",
                )
            return True
        return False

    def _veto(self, order: SizedOrder, rule: str, reason: str) -> None:
        """Record veto and raise RiskVetoError."""
        decision = self._make_decision(order, "rejected", rule, reason)
        self._record_decision(decision)
        log.warning(
            "risk_veto",
            environment=order.environment,
            symbol=order.symbol,
            rule=rule,
            reason=reason,
        )
        raise RiskVetoError(f"{rule}: {reason}")

    def _make_decision(
        self,
        order: SizedOrder,
        final_action: str,
        rule: str,
        reason: str,
    ) -> RiskDecision:
        """Create a RiskDecision record."""
        return RiskDecision(
            decision_id=str(uuid4()),
            timestamp=datetime.now(tz=UTC).isoformat(),
            environment=order.environment,
            symbol=order.symbol,
            proposed_action=order.side,
            final_action=final_action,
            risk_rule_triggered=rule,
            reason=reason,
        )

    def _record_decision(self, decision: RiskDecision) -> None:
        """Write decision to risk_decisions table."""
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO risk_decisions "
                "(decision_id, timestamp, environment, symbol, proposed_action, "
                "final_action, risk_rule_triggered, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    decision.decision_id,
                    decision.timestamp,
                    decision.environment,
                    decision.symbol,
                    decision.proposed_action,
                    decision.final_action,
                    decision.risk_rule_triggered,
                    decision.reason,
                ),
            )
