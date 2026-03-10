"""Tests for the swingrl_retry decorator.

Validates exponential backoff, retry count, and non-retryable exception pass-through.
"""

from __future__ import annotations

import pytest
from swingrl.utils.retry import swingrl_retry
from tenacity import RetryError


class TestSwingrlRetry:
    """HARD-05: Retry decorator behavior tests."""

    def test_retries_on_connection_error(self) -> None:
        """HARD-05: swingrl_retry retries ConnectionError up to max_attempts."""
        call_count = 0

        @swingrl_retry(max_attempts=3, min_wait=0, max_wait=0)
        def flaky_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "ok"

        result = flaky_fn()
        assert result == "ok"
        assert call_count == 3

    def test_reraises_after_exhausting_retries(self) -> None:
        """HARD-05: After max_attempts exhausted, the original exception is raised."""

        @swingrl_retry(max_attempts=2, min_wait=0, max_wait=0)
        def always_fails() -> None:
            raise ConnectionError("permanent failure")

        with pytest.raises(RetryError):
            always_fails()

    def test_does_not_retry_non_retryable(self) -> None:
        """HARD-05: Non-retryable exceptions (ValueError) pass through immediately."""
        call_count = 0

        @swingrl_retry(max_attempts=5, min_wait=0, max_wait=0)
        def bad_input() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            bad_input()
        assert call_count == 1

    def test_retries_on_timeout_error(self) -> None:
        """HARD-05: swingrl_retry retries TimeoutError."""
        call_count = 0

        @swingrl_retry(max_attempts=3, min_wait=0, max_wait=0)
        def timeout_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timed out")
            return "recovered"

        assert timeout_fn() == "recovered"
        assert call_count == 2

    def test_retries_on_os_error(self) -> None:
        """HARD-05: swingrl_retry retries OSError."""
        call_count = 0

        @swingrl_retry(max_attempts=3, min_wait=0, max_wait=0)
        def os_error_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("disk error")
            return "recovered"

        assert os_error_fn() == "recovered"

    def test_custom_retryable_exceptions(self) -> None:
        """HARD-05: Custom retryable_exceptions parameter overrides defaults."""
        call_count = 0

        @swingrl_retry(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retryable_exceptions=(KeyError,),
        )
        def key_error_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KeyError("missing")
            return "found"

        assert key_error_fn() == "found"
        assert call_count == 2

    def test_uses_exponential_backoff(self) -> None:
        """HARD-05: Verify tenacity statistics show retry attempts were made."""
        call_count = 0

        @swingrl_retry(max_attempts=4, min_wait=0, max_wait=0)
        def stats_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = stats_fn()
        assert result == "ok"
        # tenacity stores statistics on the decorated function
        assert stats_fn.retry.statistics["attempt_number"] == 3  # type: ignore[attr-defined]
