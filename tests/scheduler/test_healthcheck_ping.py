"""Tests for Healthchecks.io ping utility.

PAPER-16: HC ping sends GET with timeout, handles failures gracefully.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
from swingrl.scheduler.healthcheck_ping import ping_healthcheck


class TestPingHealthcheck:
    """ping_healthcheck sends HTTP GET and handles failures."""

    def test_empty_url_is_noop(self) -> None:
        """PAPER-16: Empty URL returns immediately without HTTP call."""
        with patch("swingrl.scheduler.healthcheck_ping.httpx") as mock_httpx:
            ping_healthcheck("")
            mock_httpx.get.assert_not_called()

    def test_sends_get_with_timeout(self) -> None:
        """PAPER-16: Sends GET to the check URL with 10s timeout."""
        with patch("swingrl.scheduler.healthcheck_ping.httpx") as mock_httpx:
            ping_healthcheck("https://hc-ping.com/test-uuid")
            mock_httpx.get.assert_called_once_with("https://hc-ping.com/test-uuid", timeout=10.0)

    def test_network_failure_does_not_raise(self) -> None:
        """PAPER-16: Network error logs warning but does not raise."""
        with patch("swingrl.scheduler.healthcheck_ping.httpx") as mock_httpx:
            mock_httpx.get.side_effect = httpx.ConnectError("connection refused")
            # Must not raise
            ping_healthcheck("https://hc-ping.com/test-uuid")

    def test_timeout_does_not_raise(self) -> None:
        """PAPER-16: Timeout logs warning but does not raise."""
        with patch("swingrl.scheduler.healthcheck_ping.httpx") as mock_httpx:
            mock_httpx.get.side_effect = httpx.TimeoutException("timed out")
            ping_healthcheck("https://hc-ping.com/test-uuid")

    def test_http_error_does_not_raise(self) -> None:
        """PAPER-16: HTTP error response logs warning but does not raise."""
        with patch("swingrl.scheduler.healthcheck_ping.httpx") as mock_httpx:
            mock_httpx.get.side_effect = httpx.HTTPStatusError(
                "500", request=httpx.Request("GET", "http://x"), response=httpx.Response(500)
            )
            ping_healthcheck("https://hc-ping.com/test-uuid")
