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

import dataclasses
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

    def evaluate(self, order: SizedOrder, portfolio_value: float | None = None) -> RiskDecision:
        """Evaluate an order against all risk rules.

        Checks in order: CB state, position size, exposure, drawdown,
        daily loss, global aggregator.

        Args:
            order: Sized order to evaluate.
            portfolio_value: Freshly computed mark-to-market portfolio value for this
                cycle (amendment 2026-07-16). When provided, the drawdown/daily-loss
                breakers measure this value — marked to the cycle's fetched prices — not
                the last stored snapshot, so a held-position drawdown with zero fills is
                visible at that cycle's risk evaluation. Falls back to the stored snapshot
                value (and stored daily P&L) when ``None``.

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

        # Get portfolio state for remaining checks. Prefer the freshly computed
        # mark-to-market value (and matching daily P&L) when the caller supplies it.
        if portfolio_value is None:
            portfolio_value = self._tracker.get_portfolio_value(env)
            daily_pnl = self._tracker.get_daily_pnl(env)
        else:
            daily_pnl = self._tracker.compute_daily_pnl(env, portfolio_value)
        env_config = self._config.equity if env == "equity" else self._config.crypto

        # 2. Position size check — buys only (ruling 2026-07-22 #5): a sell shrinks or
        # closes an existing position; vetoing it on size would block risk reduction.
        if order.side == "buy" and portfolio_value > 0:
            position_pct = order.dollar_amount / portfolio_value
            if position_pct > env_config.max_position_size:
                self._veto(
                    order,
                    "position_size",
                    f"position_size {position_pct:.4f} exceeds max {env_config.max_position_size}",
                )

        # 3. Exposure check — side-aware (ruling 2026-07-22 #5): a buy adds its dollar
        # amount to exposure, a sell subtracts it. A sell can therefore never breach the
        # 1.0 cap (D14 2026-07-22: the additive form computed 0.85 + 0.43 = 1.28 for a
        # full-position close and vetoed the exit — the guard pointed the wrong way).
        current_exposure = self._tracker.get_exposure(env)
        signed_amount = order.dollar_amount if order.side == "buy" else -order.dollar_amount
        new_exposure = current_exposure + (
            signed_amount / portfolio_value if portfolio_value > 0 else 0.0
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
                        daily_pnl=daily_pnl,
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

    def apply_ramp_capacity(self, order: SizedOrder) -> SizedOrder:
        """Scale an order to the environment's post-halt ramp capacity (review H4).

        The circuit breaker re-enters trading gradually after a cooldown (25% → 50%
        → 75% → 100% capacity). The old code logged the capacity but never applied
        it, so full-size orders went to the broker during ramp-up. This scales the
        order down by the current capacity fraction *before* validation and
        submission. Call it before ``validate``/``evaluate`` so both the risk checks
        and the submitted order see the scaled order.

        Both ``dollar_amount`` and ``quantity`` are scaled by the capacity fraction
        (user ruling, supersedes the dollar-only contract): the equity broker
        (Alpaca) fills by ``notional=dollar_amount`` while the crypto sim adapter
        (``BinanceSimAdapter``) fills by ``quantity`` — so the ramp only genuinely
        shrinks the order in *both* environments if it scales both fields. Scaling
        dollars alone would leave the crypto fill full size. An ACTIVE breaker
        (capacity 1.0) returns the order unchanged.

        Args:
            order: The sized order about to be validated.

        Returns:
            The order scaled to the current ramp capacity (unchanged when ACTIVE).
        """
        cb = self._circuit_breakers.get(order.environment)
        if cb is None:
            return order
        capacity = cb.get_capacity_fraction()
        if capacity >= 1.0:
            return order

        scaled = dataclasses.replace(
            order,
            dollar_amount=order.dollar_amount * capacity,
            quantity=order.quantity * capacity,
        )
        log.info(
            "order_scaled_by_ramp",
            environment=order.environment,
            symbol=order.symbol,
            capacity=capacity,
            original_amount=order.dollar_amount,
            scaled_amount=scaled.dollar_amount,
            original_quantity=order.quantity,
            scaled_quantity=scaled.quantity,
        )
        return scaled

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
