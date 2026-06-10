"""Tests for FeatureHealthTracker — blocks trading on degraded features."""

from __future__ import annotations

import time

from swingrl.features.health import FeatureHealthTracker


class TestFeatureHealth:
    """Unit tests for FeatureHealthTracker."""

    def test_record_success_resets_failures(self) -> None:
        """HEALTH-01: After 3 failures + 1 success, consecutive_failures == 0."""
        tracker = FeatureHealthTracker()
        for _ in range(3):
            tracker.record_failure("macro")

        health = tracker.get_health("macro")
        assert health is not None
        assert health.consecutive_failures == 3

        tracker.record_success("macro")
        assert health.consecutive_failures == 0

    def test_assess_blocks_after_threshold(self) -> None:
        """HEALTH-02: 3 macro failures -> should_block == True."""
        tracker = FeatureHealthTracker()
        for _ in range(3):
            tracker.record_failure("macro")

        result = tracker.assess("equity")
        assert result.should_block is True
        assert not result.macro_ok
        assert "macro" in result.reason

    def test_assess_hmm_blocks(self) -> None:
        """HEALTH-03: 3 HMM failures -> should_block == True."""
        tracker = FeatureHealthTracker()
        for _ in range(3):
            tracker.record_failure("hmm")

        result = tracker.assess("equity")
        assert result.should_block is True
        assert not result.hmm_ok
        assert "hmm" in result.reason

    def test_turbulence_never_blocks(self) -> None:
        """HEALTH-04: 10 turbulence failures -> should_block == False."""
        tracker = FeatureHealthTracker()
        for _ in range(10):
            tracker.record_failure("turbulence")

        result = tracker.assess("equity")
        assert result.should_block is False
        assert not result.turbulence_ok

    def test_staleness_blocks(self) -> None:
        """HEALTH-05: Macro last_success_ts 8 days ago -> should_block == True."""
        tracker = FeatureHealthTracker()
        health = tracker.get_health("macro")
        assert health is not None
        # Set last success to 8 days ago
        health.last_success_ts = time.time() - (8 * 24 * 3600)

        result = tracker.assess("equity")
        assert result.should_block is True
        assert "stale" in result.reason

    def test_recovery_unblocks(self) -> None:
        """HEALTH-06: 3 failures -> record_success -> should_block == False."""
        tracker = FeatureHealthTracker()
        for _ in range(3):
            tracker.record_failure("macro")

        blocked = tracker.assess("equity")
        assert blocked.should_block is True

        tracker.record_success("macro")
        recovered = tracker.assess("equity")
        assert recovered.should_block is False

    def test_is_critical(self) -> None:
        """HEALTH-07: 5 failures -> is_critical == True."""
        tracker = FeatureHealthTracker()
        for _ in range(5):
            tracker.record_failure("hmm")

        assert tracker.is_critical("hmm") is True
        assert tracker.is_critical("macro") is False

    def test_below_threshold_no_block(self) -> None:
        """HEALTH-08: 2 failures (below threshold) -> should_block == False."""
        tracker = FeatureHealthTracker()
        for _ in range(2):
            tracker.record_failure("macro")

        result = tracker.assess("equity")
        assert result.should_block is False
        assert result.macro_ok is True

    def test_unknown_source_ignored(self) -> None:
        """HEALTH-09: Unknown source names are silently ignored."""
        tracker = FeatureHealthTracker()
        tracker.record_failure("nonexistent")
        tracker.record_success("nonexistent")
        assert tracker.get_health("nonexistent") is None
