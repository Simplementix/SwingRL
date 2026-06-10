"""Memory agent HTTP client with fail-open semantics.

Provides ingest(), ingest_training(), and consolidate() methods that POST
to the memory agent REST API. All methods return False (not raise) on any
connection or HTTP error so that training is never blocked by memory agent
unavailability.

Usage:
    from swingrl.memory.client import MemoryClient
    client = MemoryClient(base_url="http://swingrl-memory:8889")
    client.ingest({"text": "...", "source": "walk_forward:equity:ppo"})
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
        api_key: API key sent as X-API-Key header on every request.
                 Empty string (default) means no auth header — backward compatible.
    """

    def __init__(
        self,
        base_url: str = "http://swingrl-memory:8889",
        default_timeout: float = 5.0,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = default_timeout
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        """Base URL of the memory agent service."""
        return self._base_url

    @property
    def api_key(self) -> str:
        """API key for memory agent authentication."""
        return self._api_key

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
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:  # noqa: S310  # nosec B310
                status: int = resp.status
                log.debug("memory_ingest_ok", url=url, status=status)
                return 200 <= status < 300
        except Exception as exc:
            log.warning("memory_ingest_failed", url=url, error=str(exc))
            return False

    def ingest_training(
        self,
        text: str,
        source: str,
        timeout: float | None = None,
    ) -> bool:
        """Ingest a training narrative to the memory agent.

        Accepts any source tag (e.g. ``walk_forward:equity:ppo``,
        ``walk_forward:crypto:ensemble``).  The old ``:historical`` assertion
        was removed because walk-forward results use different tag formats.

        Args:
            text: Text content to ingest.
            source: Source identifier (e.g. ``walk_forward:equity:ppo``).
            timeout: Request timeout in seconds.

        Returns:
            True on success, False on any error.
        """
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
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
                status: int = resp.status
                log.debug("memory_consolidate_ok", url=url, status=status)
                return 200 <= status < 300
        except Exception as exc:
            log.warning("memory_consolidate_failed", url=url, error=str(exc))
            return False

    def epoch_advice(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """POST to /training/epoch_advice and return the JSON response.

        Fail-open: returns empty dict on any network or HTTP error.

        Args:
            payload: Dict to POST as JSON (must contain 'query' key).
            timeout: Request timeout in seconds. Defaults to client's
                default_timeout (which should be set to meta_training_timeout_sec
                for training clients, typically 120s for LLM calls).

        Returns:
            Parsed JSON response dict on success, empty dict on any error.
        """
        import json as _json
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/training/epoch_advice"
        effective_timeout = timeout if timeout is not None else self._default_timeout

        try:
            data = _json.dumps(payload).encode("utf-8")
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:  # noqa: S310  # nosec B310
                body: dict[str, Any] = _json.loads(resp.read().decode("utf-8"))
                return body
        except Exception as exc:
            log.warning("memory_epoch_advice_failed", url=url, error=str(exc))
            return {}

    def record_outcome(
        self,
        iteration: int,
        env_name: str,
        gate_passed: bool | None = None,
        sharpe: float | None = None,
        mdd: float | None = None,
        sortino: float | None = None,
        pnl: float | None = None,
        patterns_presented: list[int] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """POST to /training/record_outcome to record iteration results.

        Fail-open: returns False on any network or HTTP error.

        Args:
            iteration: Training iteration number.
            env_name: Environment name.
            gate_passed: Whether ensemble gate passed.
            sharpe: Ensemble Sharpe ratio.
            mdd: Maximum drawdown.
            sortino: Sortino ratio.
            pnl: Total PnL.
            patterns_presented: List of consolidation IDs presented.
            timeout: Request timeout in seconds.

        Returns:
            True on HTTP 2xx success, False on any error.
        """
        import json
        import urllib.error
        import urllib.request

        effective_timeout = timeout if timeout is not None else self._default_timeout
        url = f"{self._base_url}/training/record_outcome"

        payload: dict[str, Any] = {
            "iteration": iteration,
            "env_name": env_name,
        }
        if gate_passed is not None:
            payload["gate_passed"] = gate_passed
        if sharpe is not None:
            payload["sharpe"] = sharpe
        if mdd is not None:
            payload["mdd"] = mdd
        if sortino is not None:
            payload["sortino"] = sortino
        if pnl is not None:
            payload["pnl"] = pnl
        if patterns_presented is not None:
            payload["patterns_presented"] = patterns_presented

        try:
            data = json.dumps(payload).encode("utf-8")
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-API-Key"] = self._api_key
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:  # noqa: S310  # nosec B310
                status: int = resp.status
                log.debug("memory_record_outcome_ok", url=url, status=status)
                return 200 <= status < 300
        except Exception as exc:
            log.warning("memory_record_outcome_failed", url=url, error=str(exc))
            return False
