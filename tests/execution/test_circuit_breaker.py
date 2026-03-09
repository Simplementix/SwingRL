"""Tests for circuit breaker state machine with SQLite persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def equity_cb(mock_db: DatabaseManager, exec_config: SwingRLConfig) -> CircuitBreaker:
    """Equity circuit breaker instance."""
    from swingrl.execution.risk.circuit_breaker import CircuitBreaker

    return CircuitBreaker(environment="equity", db=mock_db, config=exec_config)


@pytest.fixture
def crypto_cb(mock_db: DatabaseManager, exec_config: SwingRLConfig) -> CircuitBreaker:
    """Crypto circuit breaker instance."""
    from swingrl.execution.risk.circuit_breaker import CircuitBreaker

    return CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)


class TestCBStates:
    """PAPER-04: Circuit breaker has 3 states."""

    def test_initial_state_is_active(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: CB starts in ACTIVE state."""
        from swingrl.execution.risk.circuit_breaker import CBState

        assert equity_cb.get_state() == CBState.ACTIVE

    def test_capacity_is_one_when_active(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: Full capacity when ACTIVE."""
        assert equity_cb.get_capacity_fraction() == pytest.approx(1.0)


class TestCBTriggers:
    """PAPER-04: CB triggers at correct thresholds per environment."""

    def test_equity_drawdown_trigger(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: Equity CB triggers at -10% drawdown."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # >10% drawdown: portfolio at 359, HWM at 400 (10.25% DD)
        state = equity_cb.check_and_update(
            portfolio_value=359.0, high_water_mark=400.0, daily_pnl=0.0
        )
        assert state == CBState.HALTED

    def test_equity_drawdown_below_threshold_no_trigger(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: No trigger when drawdown below threshold."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # 5% drawdown: portfolio at 380, HWM at 400
        state = equity_cb.check_and_update(
            portfolio_value=380.0, high_water_mark=400.0, daily_pnl=0.0
        )
        assert state == CBState.ACTIVE

    def test_equity_daily_loss_trigger(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: Equity CB triggers at -2% daily loss."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # -2% daily loss on $400 = -8.0
        state = equity_cb.check_and_update(
            portfolio_value=392.0, high_water_mark=400.0, daily_pnl=-8.0
        )
        assert state == CBState.HALTED

    def test_crypto_drawdown_trigger(self, crypto_cb: CircuitBreaker) -> None:
        """PAPER-04: Crypto CB triggers at -12% drawdown."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # 12% drawdown: portfolio at 41.36, HWM at 47
        state = crypto_cb.check_and_update(
            portfolio_value=41.36, high_water_mark=47.0, daily_pnl=0.0
        )
        assert state == CBState.HALTED

    def test_crypto_daily_loss_trigger(self, crypto_cb: CircuitBreaker) -> None:
        """PAPER-04: Crypto CB triggers at -3% daily loss."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # -3% daily loss on $47 = -1.41
        state = crypto_cb.check_and_update(
            portfolio_value=45.59, high_water_mark=47.0, daily_pnl=-1.41
        )
        assert state == CBState.HALTED


class TestCBPersistence:
    """PAPER-04: Halt state persists across DB close/reopen."""

    def test_halt_persists_after_reopen(
        self, equity_cb: CircuitBreaker, mock_db: DatabaseManager, exec_config: SwingRLConfig
    ) -> None:
        """PAPER-04: CB state is HALTED after DB close and reopen."""
        from swingrl.execution.risk.circuit_breaker import CBState, CircuitBreaker

        equity_cb.check_and_update(portfolio_value=350.0, high_water_mark=400.0, daily_pnl=0.0)
        assert equity_cb.get_state() == CBState.HALTED

        # Create a new CB instance (simulates process restart)
        new_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
        assert new_cb.get_state() == CBState.HALTED


class TestCBCooldown:
    """PAPER-04: Cooldown periods and ramp-up progression."""

    def test_capacity_zero_when_halted(self, equity_cb: CircuitBreaker) -> None:
        """PAPER-04: Capacity is 0.0 when HALTED (before cooldown starts)."""
        equity_cb.check_and_update(portfolio_value=350.0, high_water_mark=400.0, daily_pnl=0.0)
        assert equity_cb.get_capacity_fraction() == pytest.approx(0.0)

    def test_crypto_ramp_after_partial_cooldown(
        self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-04: Crypto ramps to 0.25 after ~25% of 3-day cooldown."""
        from swingrl.execution.risk.circuit_breaker import CBState

        crypto_cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)

        # Fast-forward triggered_at to ~1 day ago (33% of 3-day cooldown => stage 2 = 0.50)
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=1)).isoformat()
        with mock_db.sqlite() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = ? WHERE environment = 'crypto'",
                (triggered_at,),
            )

        state = crypto_cb.get_state()
        cap = crypto_cb.get_capacity_fraction()
        assert state == CBState.RAMPING
        assert cap > 0.0
        assert cap <= 0.50

    def test_ramp_progression(self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager) -> None:
        """PAPER-04: Ramp progresses through 0.25, 0.50, 0.75, 1.00."""
        crypto_cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)

        # Fast-forward to ~75% of 3-day cooldown (2.25 days)
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=2, hours=6)).isoformat()
        with mock_db.sqlite() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = ? WHERE environment = 'crypto'",
                (triggered_at,),
            )

        cap = crypto_cb.get_capacity_fraction()
        assert cap == pytest.approx(0.75)

    def test_auto_resume_after_full_cooldown(
        self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager
    ) -> None:
        """PAPER-04: CB auto-resumes when full cooldown elapsed."""
        from swingrl.execution.risk.circuit_breaker import CBState

        crypto_cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)

        # Fast-forward past 3-day cooldown
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=4)).isoformat()
        with mock_db.sqlite() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = ? WHERE environment = 'crypto'",
                (triggered_at,),
            )

        cap = crypto_cb.get_capacity_fraction()
        assert cap == pytest.approx(1.0)
        assert crypto_cb.get_state() == CBState.ACTIVE


class TestGlobalCircuitBreaker:
    """PAPER-04: Global CB aggregates across environments."""

    def test_global_combined_drawdown_trigger(
        self,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """PAPER-04: Global CB triggers at -15% combined drawdown."""
        from swingrl.execution.risk.circuit_breaker import (
            CircuitBreaker,
            GlobalCircuitBreaker,
        )

        eq_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
        cr_cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)
        global_cb = GlobalCircuitBreaker(
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config
        )

        # Total initial: 400 + 47 = 447
        # 15% of 447 = 67.05 -> portfolio must be below 379.95
        # 340 + 28 = 368 -> DD = 1 - 368/447 = 17.7% > 15%
        result = global_cb.check_combined(
            portfolio_values={"equity": 340.0, "crypto": 28.0},
            daily_pnls={"equity": 0.0, "crypto": 0.0},
        )
        assert result is True  # triggered

    def test_global_combined_daily_loss_trigger(
        self,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """PAPER-04: Global CB triggers at -3% combined daily loss."""
        from swingrl.execution.risk.circuit_breaker import (
            CircuitBreaker,
            GlobalCircuitBreaker,
        )

        eq_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
        cr_cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)
        global_cb = GlobalCircuitBreaker(
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config
        )

        # Total initial: 447, 3% = 13.41
        result = global_cb.check_combined(
            portfolio_values={"equity": 400.0, "crypto": 47.0},
            daily_pnls={"equity": -10.0, "crypto": -5.0},
        )
        assert result is True  # triggered
