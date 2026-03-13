"""Memory agent HTTP client with fail-open semantics.

Provides ingest(), ingest_training(), and consolidate() methods that POST
to the memory agent REST API. All methods return False (not raise) on any
connection or HTTP error so that training is never blocked by memory agent
unavailability.

Usage:
    from swingrl.memory.client import MemoryClient
    client = MemoryClient(base_url="http://swingrl-memory:8889")
    client.ingest({"text": "...", "source": "equity:historical"})
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)


class MemoryClient:
    """HTTP client for the SwingRL memory agent.

    Wraps all outbound calls with fail-open error handling: if the memory
    agent is unavailable, methods return False so training continues
    uninterrupted.

    Args:
        base_url: Base URL of the memory agent service.
        default_timeout: Default request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://swingrl-memory:8889",
        default_timeout: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = default_timeout

    def ingest(
        self,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> bool:
        """POST a payload to the /ingest endpoint.

        Fail-open: returns False on any network or HTTP error. Never raises.

        Args:
            payload: Dict to POST as JSON to /ingest.
            timeout: Request timeout in seconds. Defaults to default_timeout.

        Returns:
            True on HTTP 2xx success, False on any error.
        """
        import urllib.error
        import urllib.parse
        import urllib.request

        effective_timeout = timeout if timeout is not None else self._default_timeout
        url = f"{self._base_url}/ingest"

        try:
            import json

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:  # noqa: S310  # nosec B310
                status: int = resp.status
                log.debug("memory_ingest_ok", url=url, status=status)
                return 200 <= status < 300
        except Exception as exc:
            log.debug("memory_ingest_failed", url=url, error=str(exc))
            return False

    def ingest_training(
        self,
        text: str,
        source: str,
        timeout: float | None = None,
    ) -> bool:
        """Ingest a historical training narrative to the memory agent.

        Asserts that source ends with ':historical' to prevent accidental
        ingestion of live/real-time data through this path.

        Args:
            text: Text content to ingest.
            source: Source identifier — must end with ':historical'.
            timeout: Request timeout in seconds.

        Returns:
            True on success, False on any error.

        Raises:
            AssertionError: If source does not end with ':historical'.
        """
        assert source.endswith(":historical"), (  # noqa: S101  # nosec B101
            f"ingest_training source must end with ':historical', got: {source!r}"
        )
        payload: dict[str, Any] = {"text": text, "source": source}
        return self.ingest(payload, timeout=timeout)

    def consolidate(self, timeout: float = 60.0) -> bool:
        """POST to /consolidate to trigger memory consolidation.

        Fail-open: returns False on any network or HTTP error.

        Args:
            timeout: Request timeout in seconds (longer for consolidation).

        Returns:
            True on HTTP 2xx success, False on any error.
        """
        import json
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/consolidate"

        try:
            data = json.dumps({}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
                status: int = resp.status
                log.debug("memory_consolidate_ok", url=url, status=status)
                return 200 <= status < 300
        except Exception as exc:
            log.debug("memory_consolidate_failed", url=url, error=str(exc))
            return False
