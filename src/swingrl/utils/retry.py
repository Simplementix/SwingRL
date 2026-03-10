"""Tenacity-based retry decorator for external API calls.

Usage:
    from swingrl.utils.retry import swingrl_retry

    @swingrl_retry(max_attempts=5)
    def fetch_data() -> dict:
        ...

Retries ConnectionError, TimeoutError, and OSError with exponential backoff.
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_DEFAULT_RETRYABLE = (ConnectionError, TimeoutError, OSError)


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """Log each retry attempt with structured context."""
    log.warning(
        "retry_attempt",
        attempt=retry_state.attempt_number,
        fn=getattr(retry_state.fn, "__name__", str(retry_state.fn)),
        error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
    )


def swingrl_retry(
    *,
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 30,
    retryable_exceptions: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE,
) -> Any:
    """Create a retry decorator with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        min_wait: Minimum wait time in seconds between retries.
        max_wait: Maximum wait time in seconds between retries.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        A tenacity retry decorator.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retryable_exceptions),
        before_sleep=_log_before_sleep,
        reraise=False,
    )
