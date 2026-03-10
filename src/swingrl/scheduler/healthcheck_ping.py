"""Healthchecks.io ping utility for post-cycle success signals.

Sends an HTTP GET to a Healthchecks.io check URL after successful cycle
completion. Never raises on failure -- logs a warning and continues.

Usage:
    from swingrl.scheduler.healthcheck_ping import ping_healthcheck
    ping_healthcheck("https://hc-ping.com/<uuid>")
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger(__name__)


def ping_healthcheck(check_url: str) -> None:
    """Send a success ping to Healthchecks.io.

    No-op if check_url is empty. Never raises on failure.

    Args:
        check_url: Full Healthchecks.io ping URL (e.g., "https://hc-ping.com/<uuid>").
                   Empty string disables the ping.
    """
    if not check_url:
        return

    try:
        httpx.get(check_url, timeout=10.0)
        log.info("healthcheck_pinged", url=check_url)
    except Exception:
        log.warning("healthcheck_ping_failed", url=check_url, exc_info=True)
