"""Two-tier risk veto layer.

Evaluates orders against per-environment and global portfolio risk limits.
All veto decisions are logged for audit.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

import structlog

from swingrl.execution.types import RiskDecision, SizedOrder

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class RiskManager:
    """Evaluate orders against risk rules and veto if policy violated."""

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize risk manager with config."""
        self._config = config

    def evaluate(self, order: SizedOrder) -> RiskDecision:
        """Evaluate an order against risk rules.

        Returns a RiskDecision indicating approval or raises RiskVetoError.
        Stub implementation -- full logic in Plan 01 execution.
        """
        import uuid
        from datetime import datetime

        decision = RiskDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=UTC).isoformat(),
            environment=order.environment,
            symbol=order.symbol,
            proposed_action=order.side,
            final_action=order.side,
            risk_rule_triggered="none",
            reason="approved",
        )
        log.info(
            "risk_approved",
            symbol=order.symbol,
            side=order.side,
            dollar_amount=order.dollar_amount,
        )
        return decision
