"""Tests for two-tier risk manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.risk.position_tracker import PositionTracker
    from swingrl.execution.risk.risk_manager import RiskManager
    from swingrl.execution.types import SizedOrder


@pytest.fixture
def risk_manager(
    mock_db: DatabaseManager,
    exec_config: SwingRLConfig,
    position_tracker: PositionTracker,
) -> RiskManager:
    """Risk manager wired to all dependencies."""
    from swingrl.execution.risk.circuit_breaker import CircuitBreaker, GlobalCircuitBreaker

    from swingrl.execution.risk.risk_manager import RiskManager

    eq_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
    cr_cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)
    global_cb = GlobalCircuitBreaker(
        circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config
    )
    return RiskManager(
        config=exec_config,
        db=mock_db,
        position_tracker=position_tracker,
        circuit_breakers={"equity": eq_cb, "crypto": cr_cb},
        global_cb=global_cb,
    )


def _make_order(
    symbol: str = "SPY",
    side: str = "buy",
    quantity: float = 1.0,
    dollar_amount: float = 100.0,
    environment: str = "equity",
) -> SizedOrder:
    """Create a SizedOrder for testing."""
    from swingrl.execution.types import SizedOrder

    return SizedOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        dollar_amount=dollar_amount,
        stop_loss_price=None,
        take_profit_price=None,
        environment=environment,
    )


class TestRiskManagerApproval:
    """PAPER-03: Risk manager approves valid orders."""

    def test_approves_small_order(self, risk_manager: RiskManager) -> None:
        """PAPER-03: Small order within all limits is approved."""
        from swingrl.execution.types import RiskDecision

        order = _make_order(dollar_amount=50.0)
        decision = risk_manager.evaluate(order)
        assert isinstance(decision, RiskDecision)
        assert decision.final_action == "buy"
        assert decision.risk_rule_triggered == "none"


class TestRiskManagerPositionSize:
    """PAPER-03: Position size limit enforcement."""

    def test_vetoes_oversized_position(self, risk_manager: RiskManager) -> None:
        """PAPER-03: Vetoes order exceeding 25% equity position size."""
        from swingrl.utils.exceptions import RiskVetoError

        # 25% of $400 = $100, so $150 exceeds it
        order = _make_order(dollar_amount=150.0)
        with pytest.raises(RiskVetoError, match="position_size"):
            risk_manager.evaluate(order)


class TestRiskManagerDrawdown:
    """PAPER-04: Drawdown check triggers circuit breaker."""

    def test_triggers_cb_on_drawdown(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
    ) -> None:
        """PAPER-04: Risk manager triggers CB when drawdown exceeds threshold."""
        from swingrl.utils.exceptions import CircuitBreakerError

        # Record a snapshot showing 12% drawdown (>10% equity threshold)
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, "
                "high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-03-09T10:00:00Z", "equity", 352.0, 50.0, 400.0, -10.0, 0.12),
            )

        order = _make_order(dollar_amount=50.0)
        with pytest.raises(CircuitBreakerError):
            risk_manager.evaluate(order)


class TestRiskManagerDailyLoss:
    """PAPER-04: Daily loss check enforcement."""

    def test_vetoes_on_daily_loss_exceeded(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
    ) -> None:
        """PAPER-04: Vetoes when daily loss exceeds limit."""
        from datetime import UTC, datetime

        from swingrl.utils.exceptions import CircuitBreakerError, RiskVetoError

        today_ts = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with mock_db.sqlite() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, "
                "high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (today_ts, "equity", 392.0, 200.0, 400.0, -8.5, 0.02),
            )

        order = _make_order(dollar_amount=50.0)
        with pytest.raises((RiskVetoError, CircuitBreakerError)):
            risk_manager.evaluate(order)


class TestRiskManagerDecisionLogging:
    """PAPER-06: Risk decisions logged to SQLite."""

    def test_decision_logged_on_approval(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
    ) -> None:
        """PAPER-06: Approved decision written to risk_decisions table."""
        order = _make_order(dollar_amount=50.0)
        risk_manager.evaluate(order)

        with mock_db.sqlite() as conn:
            rows = conn.execute("SELECT * FROM risk_decisions").fetchall()
        assert len(rows) >= 1
        assert rows[0]["final_action"] == "buy"


class TestRiskManagerTurbulence:
    """PAPER-20: Turbulence crash protection."""

    def test_turbulence_triggers_cb(self, risk_manager: RiskManager) -> None:
        """PAPER-20: High turbulence triggers CB and signals liquidation."""
        result = risk_manager.check_turbulence(
            env="equity", turbulence_value=5.0, historical_90th_pct=3.0
        )
        assert result is True  # signals liquidation


class TestRiskManagerCBInteraction:
    """PAPER-04: Risk manager checks CB state before evaluation."""

    def test_rejects_when_cb_halted(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """PAPER-04: Raises CircuitBreakerError when CB is halted."""
        from swingrl.utils.exceptions import CircuitBreakerError

        # Trigger the equity CB directly
        risk_manager._circuit_breakers["equity"].check_and_update(
            portfolio_value=350.0, high_water_mark=400.0, daily_pnl=0.0
        )

        order = _make_order(dollar_amount=50.0)
        with pytest.raises(CircuitBreakerError):
            risk_manager.evaluate(order)
