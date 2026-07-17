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


class TestCBTimezoneDateBoundary:
    """PAPER-04: business-day cooldown must not conflate ET and UTC calendar dates.

    pg16's server timezone is America/New_York, so psycopg returns TIMESTAMPTZ values
    tz-aware in ET. Between 20:00 and 24:00 ET the UTC calendar has already rolled to
    the next day; comparing an ET-rendered .date() against now(UTC).date() counted a
    phantom business day, instantly ramping a fresh halt (found via CI red 2026-07-15).
    """

    def test_fresh_halt_in_late_evening_counts_zero_business_days(self) -> None:
        """PAPER-04: fraction is 0.0 one minute after a 22:00 ET trigger."""
        from unittest.mock import MagicMock
        from zoneinfo import ZoneInfo

        from swingrl.config.schema import SwingRLConfig
        from swingrl.execution.risk.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(environment="equity", db=MagicMock(), config=SwingRLConfig())
        # As returned by psycopg from a TIMESTAMPTZ under an ET server timezone:
        triggered_at = datetime(2026, 7, 15, 22, 0, tzinfo=ZoneInfo("America/New_York"))
        now = triggered_at.astimezone(UTC) + timedelta(minutes=1)  # 02:01 UTC 2026-07-16

        assert cb._business_day_fraction(triggered_at, now) == pytest.approx(0.0)


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
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s WHERE environment = 'crypto'",
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
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s WHERE environment = 'crypto'",
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
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s WHERE environment = 'crypto'",
                (triggered_at,),
            )

        cap = crypto_cb.get_capacity_fraction()
        assert cap == pytest.approx(1.0)
        assert crypto_cb.get_state() == CBState.ACTIVE


class TestCBAlerts:
    """Review M1: breaker trips and auto-resumes send Discord alerts."""

    def test_trip_sends_critical_alert(
        self, mock_db: DatabaseManager, exec_config: SwingRLConfig
    ) -> None:
        """M1 (b): a breaker trip sends a critical alert with state + trigger value."""
        from unittest.mock import MagicMock

        from swingrl.execution.risk.circuit_breaker import CircuitBreaker

        alerter = MagicMock()
        cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config, alerter=alerter)
        # 12.5% drawdown (portfolio 350, HWM 400) > 10% equity threshold → trips.
        cb.check_and_update(portfolio_value=350.0, high_water_mark=400.0, daily_pnl=0.0)

        alerter.send_alert.assert_called_once()
        _args, kwargs = alerter.send_alert.call_args
        assert kwargs["level"] == "critical"
        blob = f"{kwargs}"
        assert "HALTED" in blob  # breaker state in the embed
        assert "0.125" in blob  # trigger value 1 - 350/400 = 0.1250

    def test_auto_resume_sends_alert(
        self, mock_db: DatabaseManager, exec_config: SwingRLConfig
    ) -> None:
        """M1 (c): auto-resume after a full cooldown sends an alert."""
        from unittest.mock import MagicMock

        from swingrl.execution.risk.circuit_breaker import CBState, CircuitBreaker

        alerter = MagicMock()
        cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config, alerter=alerter)
        cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)

        # Fast-forward past the 3-day crypto cooldown → next state read auto-resumes.
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=4)).isoformat()
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s WHERE environment = 'crypto'",
                (triggered_at,),
            )
        alerter.reset_mock()  # discard the trip alert; only the resume alert matters here

        assert cb.get_state() == CBState.ACTIVE  # triggers resume()
        alerter.send_alert.assert_called()
        _args, kwargs = alerter.send_alert.call_args
        blob = f"{kwargs}"
        assert "esume" in blob or "ACTIVE" in blob  # "Auto-Resumed" title / ACTIVE state

    def test_global_trip_sends_single_alert(
        self, mock_db: DatabaseManager, exec_config: SwingRLConfig
    ) -> None:
        """M1 (b/global): a global-limit breach sends exactly one global alert.

        The per-env cascade (_trigger_all) is suppressed so the single global embed
        speaks for both environments rather than firing three redundant alerts.
        """
        from unittest.mock import MagicMock

        from swingrl.execution.risk.circuit_breaker import CircuitBreaker, GlobalCircuitBreaker

        alerter = MagicMock()
        eq_cb = CircuitBreaker(
            environment="equity", db=mock_db, config=exec_config, alerter=alerter
        )
        cr_cb = CircuitBreaker(
            environment="crypto", db=mock_db, config=exec_config, alerter=alerter
        )
        global_cb = GlobalCircuitBreaker(
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb},
            config=exec_config,
            db=mock_db,
            alerter=alerter,
        )
        # Combined 368 vs initial 447 → 17.7% drawdown > 15% global limit → trips.
        result = global_cb.check_combined(
            portfolio_values={"equity": 340.0, "crypto": 28.0},
            daily_pnls={"equity": 0.0, "crypto": 0.0},
        )
        assert result is True
        alerter.send_alert.assert_called_once()
        _args, kwargs = alerter.send_alert.call_args
        assert kwargs["level"] == "critical"
        assert "combined_drawdown" in f"{kwargs}"


class TestStopBreachAuditInert:
    """User ruling 2: audit-only stop-breach rows are inert to breaker state.

    A row whose ``reason`` starts with ``STOP_BREACH_REASON_MARKER`` records a crypto
    stop-loss breach for audit + alerting. It must NOT change ``get_state`` /
    ``get_capacity_fraction`` — even as the most recent event — because a HALTED
    breaker would veto the very sell that closes the breached position.
    """

    def _insert_marker(
        self,
        db: DatabaseManager,
        symbol: str = "BTCUSDT",
        when: datetime | None = None,
    ) -> None:
        """Insert an audit-only stop-breach row exactly like the poller does."""
        from uuid import uuid4

        from swingrl.execution.risk.circuit_breaker import STOP_BREACH_REASON_MARKER

        ts = (when or datetime.now(tz=UTC)).isoformat()
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_events "
                "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    str(uuid4()),
                    "crypto",
                    ts,
                    40000.0,
                    50000.0,
                    f"{STOP_BREACH_REASON_MARKER}{symbol}",
                ),
            )

    def test_marker_row_does_not_halt_state(
        self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager
    ) -> None:
        """Ruling 2 (a): a lone marker row leaves the breaker ACTIVE at full capacity."""
        from swingrl.execution.risk.circuit_breaker import CBState

        self._insert_marker(mock_db)
        assert crypto_cb.get_state() == CBState.ACTIVE
        assert crypto_cb.get_capacity_fraction() == pytest.approx(1.0)

    def test_genuine_halt_survives_a_later_marker_row(
        self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager
    ) -> None:
        """Ruling 2 (c): a genuine halt still HALTS when a marker row is the newest event."""
        from swingrl.execution.risk.circuit_breaker import CBState

        # Genuine trip → HALTED.
        crypto_cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)
        assert crypto_cb.get_state() == CBState.HALTED

        # A stop breach lands AFTER the genuine trip (newest triggered_at). Without the
        # marker exclusion this would become _latest_event and mask the live halt.
        self._insert_marker(mock_db, when=datetime.now(tz=UTC) + timedelta(minutes=5))

        assert crypto_cb.get_state() == CBState.HALTED
        assert crypto_cb.get_capacity_fraction() == pytest.approx(0.0)

    def test_resume_does_not_stamp_marker_row(
        self, crypto_cb: CircuitBreaker, mock_db: DatabaseManager
    ) -> None:
        """Ruling 2 (c): auto-resume closes the genuine halt but leaves the marker row open."""
        from swingrl.execution.risk.circuit_breaker import STOP_BREACH_REASON_MARKER, CBState

        crypto_cb.check_and_update(portfolio_value=40.0, high_water_mark=47.0, daily_pnl=0.0)
        self._insert_marker(mock_db)

        # Fast-forward ONLY the genuine trip past the 3-day cooldown → next read resumes.
        triggered_at = (datetime.now(tz=UTC) - timedelta(days=4)).isoformat()
        with mock_db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET triggered_at = %s "
                "WHERE environment = 'crypto' AND reason NOT LIKE %s",
                (triggered_at, f"{STOP_BREACH_REASON_MARKER}%"),
            )
        assert crypto_cb.get_state() == CBState.ACTIVE  # triggers resume()

        with mock_db.connection() as conn:
            rows = conn.execute(
                "SELECT resumed_at FROM circuit_breaker_events "
                "WHERE environment = 'crypto' AND reason LIKE %s",
                (f"{STOP_BREACH_REASON_MARKER}%",),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["resumed_at"] is None  # marker row untouched by resume()


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
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config, db=mock_db
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
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb}, config=exec_config, db=mock_db
        )

        # Total initial: 447, 3% = 13.41
        result = global_cb.check_combined(
            portfolio_values={"equity": 400.0, "crypto": 47.0},
            daily_pnls={"equity": -10.0, "crypto": -5.0},
        )
        assert result is True  # triggered


class TestGlobalCBHwmRestartSurvival:
    """Review C1: global high-water mark is read from persisted snapshots.

    The old GlobalCircuitBreaker held the combined HWM only in process memory
    (``self._total_hwm``), so a restart reset it to initial capital and a real
    drawdown from a prior peak went undetected. The HWM must be reconstructed
    from ``MAX(total_value)`` per environment across persisted snapshots.
    """

    def test_global_hwm_survives_restart_from_snapshots(
        self,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """C1 (e): a fresh GlobalCircuitBreaker re-reads the combined peak and trips."""
        from swingrl.execution.risk.circuit_breaker import (
            CircuitBreaker,
            GlobalCircuitBreaker,
        )

        # Persisted peaks: equity 500 + crypto 60 = combined HWM 560 (> initial 447).
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp, environment, total_value, "
                "cash_balance, high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("2026-03-09T10:00:00Z", "equity", 500.0, 0.0, 500.0, 0.0, 0.0),
            )
            conn.execute(
                "INSERT INTO portfolio_snapshots (timestamp, environment, total_value, "
                "cash_balance, high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("2026-03-09T10:00:00Z", "crypto", 60.0, 0.0, 60.0, 0.0, 0.0),
            )

        # Fresh instances → no in-memory peak; must reconstruct 560 from snapshots.
        eq_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
        cr_cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)
        global_cb = GlobalCircuitBreaker(
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb},
            config=exec_config,
            db=mock_db,
        )

        # Combined value 470 (equity 420 + crypto 50): 16.1% below the 560 peak,
        # which exceeds the 15% global drawdown limit → trips.
        # Against initial capital (447) 470 shows a gain and would NOT trip, so a
        # pass proves the peak was re-read from persisted snapshots, not reset.
        result = global_cb.check_combined(
            portfolio_values={"equity": 420.0, "crypto": 50.0},
            daily_pnls={"equity": 0.0, "crypto": 0.0},
        )
        assert result is True

    def test_global_no_snapshots_uses_initial_capital(
        self,
        mock_db: DatabaseManager,
        exec_config: SwingRLConfig,
    ) -> None:
        """C1 (e): with no snapshots the HWM floors at initial capital (no false trip)."""
        from swingrl.execution.risk.circuit_breaker import (
            CircuitBreaker,
            GlobalCircuitBreaker,
        )

        eq_cb = CircuitBreaker(environment="equity", db=mock_db, config=exec_config)
        cr_cb = CircuitBreaker(environment="crypto", db=mock_db, config=exec_config)
        global_cb = GlobalCircuitBreaker(
            circuit_breakers={"equity": eq_cb, "crypto": cr_cb},
            config=exec_config,
            db=mock_db,
        )
        # 470 > initial 447 → a gain, no drawdown → must not trigger.
        result = global_cb.check_combined(
            portfolio_values={"equity": 420.0, "crypto": 50.0},
            daily_pnls={"equity": 0.0, "crypto": 0.0},
        )
        assert result is False
