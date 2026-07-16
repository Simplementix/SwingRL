"""Circuit breaker state machine with PostgreSQL persistence.

Three states: ACTIVE -> HALTED -> RAMPING -> ACTIVE
Triggers on drawdown or daily loss threshold breach.
Persists halt events in circuit_breaker_events table.

Cooldown periods:
- Equity: 5 business days (NYSE calendar)
- Crypto: 3 calendar days

Ramp-up: 25% -> 50% -> 75% -> 100% capacity at equal intervals during cooldown.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

# Ramp stages: capacity fractions at each stage
RAMP_STAGES: list[float] = [0.25, 0.50, 0.75, 1.00]

# NYSE sessions are ET calendar dates; business-day math must use ET on BOTH sides.
# (pg16 returns TIMESTAMPTZ tz-aware in the server timezone — .date() without an
# explicit astimezone conflates ET and UTC calendars between 20:00 and 24:00 ET.)
_ET = ZoneInfo("America/New_York")


class CBState(enum.Enum):
    """Circuit breaker states."""

    ACTIVE = "active"
    HALTED = "halted"
    RAMPING = "ramping"


class CircuitBreaker:
    """Per-environment circuit breaker with PostgreSQL persistence.

    Args:
        environment: "equity" or "crypto".
        db: DatabaseManager for database access.
        config: SwingRLConfig for thresholds and initial capital.
        alerter: Optional Discord alerter. When provided, every trip and every
            auto-resume sends a critical alert (review M1). None disables alerting
            (unit tests, contexts without a webhook).
    """

    def __init__(
        self,
        environment: str,
        db: DatabaseManager,
        config: SwingRLConfig,
        alerter: Alerter | None = None,
    ) -> None:
        """Initialize circuit breaker for an environment."""
        self._environment = environment
        self._db = db
        self._config = config
        self._alerter = alerter

        if environment == "equity":
            self._max_dd = config.equity.max_drawdown_pct
            self._daily_limit = config.equity.daily_loss_limit_pct
            self._initial_capital = config.capital.equity_usd
            self._cooldown_days = 5  # business days
            self._use_business_days = True
        else:
            self._max_dd = config.crypto.max_drawdown_pct
            self._daily_limit = config.crypto.daily_loss_limit_pct
            self._initial_capital = config.capital.crypto_usd
            self._cooldown_days = 3  # calendar days
            self._use_business_days = False

    def check_and_update(
        self,
        portfolio_value: float,
        high_water_mark: float,
        daily_pnl: float,
    ) -> CBState:
        """Check drawdown and daily loss against thresholds; trigger if breached.

        Args:
            portfolio_value: Current total portfolio value.
            high_water_mark: Historical high water mark.
            daily_pnl: Today's P&L (negative = loss).

        Returns:
            Current CBState after evaluation.
        """
        # Check drawdown
        if high_water_mark > 0:
            drawdown = 1.0 - portfolio_value / high_water_mark
            if drawdown >= self._max_dd:
                self._trigger(
                    trigger_value=drawdown,
                    threshold=self._max_dd,
                    reason=f"drawdown_{drawdown:.4f}_exceeds_{self._max_dd}",
                )
                return CBState.HALTED

        # Check daily loss against high-water mark
        if daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / high_water_mark if high_water_mark > 0 else 0.0
            if daily_loss_pct >= self._daily_limit:
                self._trigger(
                    trigger_value=daily_loss_pct,
                    threshold=self._daily_limit,
                    reason=f"daily_loss_{daily_loss_pct:.4f}_exceeds_{self._daily_limit}",
                )
                return CBState.HALTED

        return self.get_state()

    def get_state(self) -> CBState:
        """Load current state from database.

        Returns:
            ACTIVE if no unresolved halt, HALTED if within initial cooldown,
            RAMPING if in ramp-up period.
        """
        event = self._latest_event()
        if event is None:
            return CBState.ACTIVE

        if event["resumed_at"] is not None:
            return CBState.ACTIVE

        # Still halted -- check if cooldown has progressed to ramping
        triggered_at = datetime.fromisoformat(str(event["triggered_at"]))
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=UTC)

        elapsed = self._elapsed_cooldown_fraction(triggered_at)
        if elapsed >= 1.0:
            # Full cooldown complete -- auto-resume
            self.resume()
            return CBState.ACTIVE

        # Ramp starts when capacity > 0.0 (after first 1/(n+1) of cooldown)
        n_stages = len(RAMP_STAGES)
        ramp_start = 1.0 / (n_stages + 1)
        if round(elapsed, 4) >= ramp_start:
            return CBState.RAMPING
        return CBState.HALTED

    def get_capacity_fraction(self) -> float:
        """Return current trading capacity fraction.

        Returns:
            1.0 when ACTIVE, 0.0 when HALTED, 0.25-1.0 when RAMPING.
        """
        event = self._latest_event()
        if event is None:
            return 1.0

        if event["resumed_at"] is not None:
            return 1.0

        triggered_at = datetime.fromisoformat(str(event["triggered_at"]))
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=UTC)

        elapsed_frac = self._elapsed_cooldown_fraction(triggered_at)
        if elapsed_frac >= 1.0:
            self.resume()
            return 1.0
        if elapsed_frac <= 0.0:
            return 0.0

        # Determine ramp stage (4 equal stages shifted by 1)
        # 0-25% elapsed: HALTED (0.0), 25-50%: 0.25, 50-75%: 0.50, 75-100%: 0.75
        # Full resume at 100%.
        n = len(RAMP_STAGES)
        # Round to 4 decimal places to avoid floating point boundary issues
        rounded_frac = round(elapsed_frac, 4)
        # Stage index: each stage spans 1/(n+1) of cooldown, first stage is HALTED
        stage_idx = int(rounded_frac * (n + 1)) - 1
        if stage_idx < 0:
            return 0.0
        stage_idx = min(stage_idx, n - 1)
        return RAMP_STAGES[stage_idx]

    def resume(self) -> None:
        """Mark latest halt event as resumed and alert on the auto-resume (review M1)."""
        now = datetime.now(tz=UTC).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE circuit_breaker_events SET resumed_at = %s "
                "WHERE environment = %s AND resumed_at IS NULL",
                (now, self._environment),
            )
        log.info("circuit_breaker_resumed", environment=self._environment)

        # Auto-resume is a state change operators must see. Critical is the only
        # level with guaranteed immediate delivery (warning is gated by the
        # consecutive-failure counter; info buffers into a digest that the trader
        # has no flush job for) — so a one-shot resume would otherwise be lost.
        if self._alerter is not None:
            self._alerter.send_alert(
                level="critical",
                title=f"Circuit Breaker Auto-Resumed — {self._environment}",
                message=(
                    f"State: ACTIVE\nCooldown complete — {self._environment} trading "
                    "capacity restored."
                ),
                environment=self._environment,
            )

    def _trigger(
        self,
        trigger_value: float,
        threshold: float,
        reason: str,
        *,
        alert: bool = True,
    ) -> None:
        """Insert a halt event into circuit_breaker_events and alert on the trip.

        Args:
            trigger_value: The breaching metric value (drawdown/loss fraction, or the
                turbulence value).
            threshold: The limit that was breached.
            reason: Machine-readable reason string persisted with the event.
            alert: Send a Discord alert for this trip. Set False for the per-env
                cascade of a global trip — the single global alert speaks for both
                environments (see ``GlobalCircuitBreaker._trigger_all``).
        """
        event_id = str(uuid4())
        triggered_at = datetime.now(tz=UTC).isoformat()

        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO circuit_breaker_events "
                "(event_id, environment, triggered_at, trigger_value, threshold, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (event_id, self._environment, triggered_at, trigger_value, threshold, reason),
            )

        log.critical(
            "circuit_breaker_triggered",
            environment=self._environment,
            trigger_value=trigger_value,
            threshold=threshold,
            reason=reason,
        )

        if alert and self._alerter is not None:
            self._alerter.send_alert(
                level="critical",
                title=f"Circuit Breaker Halted — {self._environment}",
                message=(
                    f"State: HALTED\nReason: {reason}\n"
                    f"Trigger value: {trigger_value:.4f} (threshold {threshold:.4f})"
                ),
                environment=self._environment,
            )

    def _latest_event(self) -> dict[str, str | float | None] | None:
        """Load the latest circuit breaker event for this environment."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM circuit_breaker_events "
                "WHERE environment = %s ORDER BY triggered_at DESC LIMIT 1",
                (self._environment,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def _elapsed_cooldown_fraction(self, triggered_at: datetime) -> float:
        """Compute fraction of cooldown elapsed since triggered_at.

        Args:
            triggered_at: UTC datetime when CB was triggered.

        Returns:
            Fraction between 0.0 and 1.0+.
        """
        now = datetime.now(tz=UTC)

        if self._use_business_days:
            return self._business_day_fraction(triggered_at, now)

        # Calendar days for crypto
        elapsed = now - triggered_at
        total_seconds = timedelta(days=self._cooldown_days).total_seconds()
        if total_seconds <= 0:
            return 1.0
        return elapsed.total_seconds() / total_seconds

    def _business_day_fraction(self, triggered_at: datetime, now: datetime) -> float:
        """Compute fraction of business day cooldown elapsed using NYSE calendar.

        Args:
            triggered_at: tz-aware datetime when CB was triggered.
            now: Current tz-aware datetime.

        Returns:
            Fraction between 0.0 and 1.0+.
        """
        import exchange_calendars

        nyse = exchange_calendars.get_calendar("XNYS")

        start_date = triggered_at.astimezone(_ET).date()
        end_date = now.astimezone(_ET).date()

        if end_date <= start_date:
            return 0.0

        # Count business days between trigger date and now
        sessions = nyse.sessions_in_range(
            start_date + timedelta(days=1),  # exclusive of trigger day
            end_date,
        )
        biz_days = len(sessions)

        return biz_days / self._cooldown_days


class GlobalCircuitBreaker:
    """Global portfolio aggregator circuit breaker.

    Triggers when combined portfolio drawdown exceeds -15% or
    combined daily loss exceeds -3% of total initial capital.

    The combined high-water mark is reconstructed from persisted snapshots on
    every check (``MAX(total_value)`` per environment, summed) so it survives a
    process restart — the old process-memory HWM reset to initial capital on
    restart and a real drawdown from a prior peak went undetected (review C1).

    Args:
        circuit_breakers: Dict mapping environment name to CircuitBreaker.
        config: SwingRLConfig for initial capital values.
        db: DatabaseManager for reading persisted portfolio snapshots.
        alerter: Optional Discord alerter. When provided, a global trip sends one
            critical alert (review M1). None disables alerting.
    """

    GLOBAL_MAX_DD: float = 0.15
    GLOBAL_DAILY_LIMIT: float = 0.03

    def __init__(
        self,
        circuit_breakers: dict[str, CircuitBreaker],
        config: SwingRLConfig,
        db: DatabaseManager,
        alerter: Alerter | None = None,
    ) -> None:
        """Initialize global circuit breaker."""
        self._circuit_breakers = circuit_breakers
        self._config = config
        self._db = db
        self._alerter = alerter
        self._total_initial = config.capital.equity_usd + config.capital.crypto_usd
        self._total_hwm: float = self._total_initial

    def _persisted_combined_hwm(self) -> float:
        """Reconstruct the combined high-water mark from persisted snapshots.

        Sums each environment's peak ``total_value`` and floors at total initial
        capital. Independent per-environment peaks may not have coincided, so the
        sum can overstate the true combined peak. That overstatement affects the
        two ``check_combined`` branches differently — it is NOT uniformly
        conservative:

        - **Drawdown branch** (``1 − total/hwm``): a larger HWM makes drawdown
          look larger, so this branch is conservative — it can only trip earlier
          or equal, never later.
        - **Daily-loss branch** (``|daily_pnl|/hwm``): the same HWM sits in the
          *denominator*, so a larger HWM makes the loss fraction look *smaller* —
          this branch can trip *later* than a true combined HWM would, not
          earlier. The effect is bounded by how far independent per-env peaks
          diverge, and is a secondary guard behind the drawdown branch and the
          unaffected per-env checks.

        Returns:
            Combined high-water mark as float.
        """
        envs = list(self._circuit_breakers.keys())
        if not envs:
            return self._total_initial
        placeholders = ", ".join(["%s"] * len(envs))
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT environment, MAX(total_value) AS peak FROM portfolio_snapshots "  # noqa: S608
                f"WHERE environment IN ({placeholders}) GROUP BY environment",  # nosec B608
                tuple(envs),
            ).fetchall()
        peak_sum = sum(float(r["peak"]) for r in rows if r["peak"] is not None)
        return max(self._total_initial, peak_sum)

    def check_combined(
        self,
        portfolio_values: dict[str, float],
        daily_pnls: dict[str, float],
    ) -> bool:
        """Check combined portfolio metrics against global limits.

        Args:
            portfolio_values: Dict of env -> current portfolio value.
            daily_pnls: Dict of env -> today's P&L.

        Returns:
            True if global limit breached (triggers all per-env CBs), False otherwise.
        """
        total_value = sum(portfolio_values.values())
        total_daily_pnl = sum(daily_pnls.values())

        # High-water mark from persisted snapshots (survives restart), folding in
        # the current combined value in case it is a fresh peak not yet persisted.
        self._total_hwm = max(self._persisted_combined_hwm(), total_value)

        # Check combined drawdown against high-water mark
        combined_dd = 1.0 - total_value / self._total_hwm if self._total_hwm > 0 else 0.0
        if combined_dd >= self.GLOBAL_MAX_DD:
            log.critical(
                "global_circuit_breaker_triggered",
                reason="combined_drawdown",
                combined_dd=combined_dd,
                threshold=self.GLOBAL_MAX_DD,
            )
            self._alert_global("combined_drawdown", combined_dd, self.GLOBAL_MAX_DD)
            self._trigger_all("combined_drawdown")
            return True

        # Check combined daily loss against high-water mark
        if total_daily_pnl < 0:
            combined_loss_pct = (
                abs(total_daily_pnl) / self._total_hwm if self._total_hwm > 0 else 0.0
            )
            if combined_loss_pct >= self.GLOBAL_DAILY_LIMIT:
                log.critical(
                    "global_circuit_breaker_triggered",
                    reason="combined_daily_loss",
                    combined_loss_pct=combined_loss_pct,
                    threshold=self.GLOBAL_DAILY_LIMIT,
                )
                self._alert_global(
                    "combined_daily_loss", combined_loss_pct, self.GLOBAL_DAILY_LIMIT
                )
                self._trigger_all("combined_daily_loss")
                return True

        return False

    def _alert_global(self, reason: str, trigger_value: float, threshold: float) -> None:
        """Send the single global-trip alert (review M1)."""
        if self._alerter is None:
            return
        self._alerter.send_alert(
            level="critical",
            title="Global Circuit Breaker Triggered",
            message=(
                f"State: HALTED (all environments)\nReason: {reason}\n"
                f"Trigger value: {trigger_value:.4f} (threshold {threshold:.4f})"
            ),
            environment="Global",
        )

    def _trigger_all(self, reason: str) -> None:
        """Trigger all per-environment circuit breakers.

        The per-env alerts are suppressed (``alert=False``): the single global
        alert emitted by ``check_combined`` already speaks for both environments,
        so cascading three redundant embeds would only add alarm fatigue.
        """
        for _env, cb in self._circuit_breakers.items():
            cb._trigger(
                trigger_value=0.0,
                threshold=0.0,
                reason=f"global_{reason}",
                alert=False,
            )
