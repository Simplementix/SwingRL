"""Feature health tracking for live inference pipeline.

Monitors consecutive failures for macro, HMM, and turbulence feature sources.
When failures exceed thresholds, signals the execution pipeline to block trading
rather than feed out-of-distribution observations to the agent.

Usage:
    from swingrl.features.health import FeatureHealthTracker

    tracker = FeatureHealthTracker()
    tracker.record_failure("macro")
    health = tracker.assess("equity")
    if health.should_block:
        # skip this trading cycle
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


@dataclass
class FeatureHealth:
    """Tracks consecutive failures for a single feature source."""

    name: str
    consecutive_failures: int = 0
    last_success_ts: float = field(default_factory=time.time)
    last_failure_ts: float = 0.0

    def record_success(self) -> None:
        """Record a successful feature fetch, resetting the failure counter."""
        if self.consecutive_failures > 0:
            log.info(
                "feature_recovered",
                feature=self.name,
                after_failures=self.consecutive_failures,
            )
        self.consecutive_failures = 0
        self.last_success_ts = time.time()

    def record_failure(self) -> None:
        """Record a feature fetch failure."""
        self.consecutive_failures += 1
        self.last_failure_ts = time.time()
        log.warning(
            "feature_failure",
            feature=self.name,
            consecutive=self.consecutive_failures,
        )


@dataclass
class ObservationHealth:
    """Health assessment for a single observation cycle."""

    macro_ok: bool = True
    hmm_ok: bool = True
    turbulence_ok: bool = True
    should_block: bool = False
    reason: str = ""


class FeatureHealthTracker:
    """Tracks feature pipeline health and decides whether to block trading.

    Three sources are tracked: macro, hmm, and turbulence. Macro and HMM failures
    block trading after BLOCK_THRESHOLD consecutive failures. Turbulence failures
    produce warnings only (0.0 = calm market assumption, suboptimal but not catastrophic).
    Macro staleness (no success in STALENESS_SECONDS) also blocks trading.
    """

    BLOCK_THRESHOLD: int = 3
    CRITICAL_THRESHOLD: int = 5
    STALENESS_SECONDS: float = 7 * 24 * 3600  # 7 days

    def __init__(self) -> None:
        self._sources: dict[str, FeatureHealth] = {
            "macro": FeatureHealth(name="macro"),
            "hmm": FeatureHealth(name="hmm"),
            "turbulence": FeatureHealth(name="turbulence"),
        }

    def record_success(self, source: str) -> None:
        """Record a successful fetch for the given source."""
        if source in self._sources:
            self._sources[source].record_success()

    def record_failure(self, source: str) -> None:
        """Record a failed fetch for the given source."""
        if source in self._sources:
            self._sources[source].record_failure()

    def get_health(self, source: str) -> FeatureHealth | None:
        """Get health state for a source."""
        return self._sources.get(source)

    def assess(self, env_name: str) -> ObservationHealth:
        """Assess whether trading should proceed for this environment.

        Args:
            env_name: Environment name (for logging context).

        Returns:
            ObservationHealth with should_block=True if features are degraded.
        """
        health = ObservationHealth()

        macro = self._sources["macro"]
        hmm = self._sources["hmm"]

        health.macro_ok = macro.consecutive_failures < self.BLOCK_THRESHOLD
        health.hmm_ok = hmm.consecutive_failures < self.BLOCK_THRESHOLD
        # Turbulence: warning only, never blocks (0.0 = calm, suboptimal not catastrophic)
        health.turbulence_ok = self._sources["turbulence"].consecutive_failures == 0

        # Check staleness
        now = time.time()
        macro_stale = (now - macro.last_success_ts) > self.STALENESS_SECONDS

        if not health.macro_ok:
            health.should_block = True
            health.reason = f"macro: {macro.consecutive_failures} consecutive failures"
        elif not health.hmm_ok:
            health.should_block = True
            health.reason = f"hmm: {hmm.consecutive_failures} consecutive failures"
        elif macro_stale:
            health.should_block = True
            health.reason = "macro data stale (>7 days since last success)"

        if health.should_block:
            log.error(
                "observation_health_degraded",
                env=env_name,
                reason=health.reason,
                macro_failures=macro.consecutive_failures,
                hmm_failures=hmm.consecutive_failures,
            )

        return health

    def is_critical(self, source: str) -> bool:
        """Check if a source has reached critical failure level."""
        h = self._sources.get(source)
        return h is not None and h.consecutive_failures >= self.CRITICAL_THRESHOLD
