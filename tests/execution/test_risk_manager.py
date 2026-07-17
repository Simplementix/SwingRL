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
        circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config, db=mock_db
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
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, "
                "high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
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
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, "
                "high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
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

        with mock_db.connection() as conn:
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


class TestRiskManagerRampCapacity:
    """Review H4: RAMPING state scales orders to the breaker capacity fraction."""

    def _ramp_crypto_to_quarter(self, risk_manager: RiskManager, mock_db: DatabaseManager) -> None:
        """Trip the crypto CB and fast-forward ~1 day of the 3-day cooldown → 0.25 capacity."""
        from datetime import UTC, datetime, timedelta

        cb = risk_manager._circuit_breakers["crypto"]
        cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=1)).isoformat()
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s WHERE environment = 'crypto'",
                (triggered_at,),
            )
        assert cb.get_capacity_fraction() == pytest.approx(0.25)

    def test_ramp_scales_order_to_capacity(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
    ) -> None:
        """H4 (a): a $1,000 order at 25% ramp scales to $250 AND quantity to 25% (user ruling)."""
        self._ramp_crypto_to_quarter(risk_manager, mock_db)

        # quantity default 1.0 → both dollar_amount and quantity must shrink to 25%.
        order = _make_order(symbol="BTCUSDT", dollar_amount=1000.0, environment="crypto")
        scaled = risk_manager.apply_ramp_capacity(order)
        assert scaled.dollar_amount == pytest.approx(250.0)
        assert scaled.quantity == pytest.approx(0.25)

    def test_crypto_ramp_scales_the_filled_quantity(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """H4 (a): a crypto order at 25% ramp fills 25% of the unramped QUANTITY.

        Crypto sim fills size by ``quantity`` (BinanceSimAdapter), so the ramp must
        scale quantity too — this traces the ramped order all the way to the fill
        object and asserts the FILLED quantity is 25% of the pre-ramp quantity, not
        just the order-object field.
        """
        from unittest.mock import MagicMock, patch

        from swingrl.execution.adapters.binance_sim import BinanceSimAdapter
        from swingrl.execution.types import ValidatedOrder

        self._ramp_crypto_to_quarter(risk_manager, mock_db)

        order = _make_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.02,
            dollar_amount=1000.0,
            environment="crypto",
        )
        scaled = risk_manager.apply_ramp_capacity(order)
        assert scaled.quantity == pytest.approx(0.005)  # 25% of 0.02

        adapter = BinanceSimAdapter(config=exec_config, db=mock_db)
        book = MagicMock()
        book.raise_for_status = MagicMock()
        book.json.return_value = {"bids": [["50000", "1"]], "asks": [["50000", "1"]]}
        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=book):
            fill = adapter.submit_order(ValidatedOrder(order=scaled))
        assert fill.quantity == pytest.approx(0.005)  # filled size = 25% of unramped 0.02

    def test_active_breaker_returns_order_unscaled(self, risk_manager: RiskManager) -> None:
        """H4 (a): with an ACTIVE breaker (full capacity) the order is returned unchanged."""
        order = _make_order(
            symbol="BTCUSDT", quantity=0.02, dollar_amount=1000.0, environment="crypto"
        )
        result = risk_manager.apply_ramp_capacity(order)
        assert result.dollar_amount == pytest.approx(1000.0)
        assert result.quantity == pytest.approx(0.02)


class TestStopBreachDoesNotHaltSells:
    """User ruling: a stop-breach audit row must NOT halt the crypto breaker.

    A HALTED breaker vetoes ALL crypto orders — including the sell that would close
    the breached position — so the stop-breach record is audit + alert only and must
    leave ``get_state()`` unchanged.
    """

    def test_stop_breach_leaves_breaker_active_and_sell_passes(
        self,
        risk_manager: RiskManager,
        mock_db: DatabaseManager,
    ) -> None:
        """Ruling 2 (a)/(b): after a stop breach crypto is NOT HALTED, a sell passes, row audits."""
        from swingrl.execution.risk.circuit_breaker import (
            STOP_BREACH_REASON_MARKER,
            CBState,
        )
        from swingrl.scheduler.stop_polling import _record_stop_breach

        # Seed both env snapshots so the FULL risk evaluation can approve the sell —
        # isolating the crypto breaker-state gate as the only thing under test.
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp, environment, total_value, "
                "cash_balance, high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("2026-03-09T10:00:00Z", "equity", 400.0, 400.0, 400.0, 0.0, 0.0),
            )
            conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp, environment, total_value, "
                "cash_balance, high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("2026-03-09T10:00:00Z", "crypto", 47.0, 47.0, 47.0, 0.0, 0.0),
            )

        # Record a stop breach the same way the poller does (audit-only marker row).
        _record_stop_breach(mock_db, "BTCUSDT", 40000.0, 50000.0, None)

        # (a) breaker state is NOT halted despite the fresh, unresolved breach row.
        crypto_cb = risk_manager._circuit_breakers["crypto"]
        assert crypto_cb.get_state() == CBState.ACTIVE
        assert crypto_cb.get_capacity_fraction() == pytest.approx(1.0)

        # (a) a crypto SELL passes the breaker-state check (no CircuitBreakerError).
        sell = _make_order(
            symbol="BTCUSDT",
            side="sell",
            quantity=0.0002,
            dollar_amount=10.0,
            environment="crypto",
        )
        decision = risk_manager.evaluate(sell)
        assert decision.final_action == "sell"
        assert decision.risk_rule_triggered == "none"

        # (b) the audit row exists and is queryable.
        with mock_db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM circuit_breaker_events WHERE environment = 'crypto' AND reason = %s",
                (f"{STOP_BREACH_REASON_MARKER}BTCUSDT",),
            ).fetchall()
        assert len(rows) == 1
        assert float(rows[0]["trigger_value"]) == pytest.approx(40000.0)
        assert rows[0]["resumed_at"] is None  # append-only audit row stays open


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
