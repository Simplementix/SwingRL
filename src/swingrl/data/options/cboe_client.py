# src/swingrl/data/options/cboe_client.py
"""Thin, swappable client for CBOE's delayed-quotes chain endpoint (spec §17 C1).

Provider-quarantine seam (D7): fallback providers (Schwab, moomoo — spec §17 C2)
replace THIS module only; parser/store/collector stay provider-agnostic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from swingrl.utils.exceptions import DataError
from swingrl.utils.retry import swingrl_retry

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig

log = structlog.get_logger(__name__)


class CboeChainClient:
    """Fetch a full option chain as a raw dict — unauthenticated, throttled, retried."""

    def __init__(self, config: OptionsCollectorConfig) -> None:
        self._config = config
        self._min_interval_s = 1.0 / config.rate_limit_per_sec
        self._last_call_ts = 0.0

    def chain_url(self, symbol: str) -> str:
        """Endpoint URL for one underlying (symbols are config-controlled)."""
        return self._config.endpoint_url_template.format(symbol=symbol)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call_ts = time.monotonic()

    @swingrl_retry(
        max_attempts=4, retryable_exceptions=(httpx.TransportError, TimeoutError, OSError)
    )
    def _fetch(self, url: str) -> httpx.Response:
        self._throttle()
        return httpx.get(url, timeout=self._config.request_timeout_s)

    def get_option_chain(self, symbol: str) -> dict[str, Any]:
        """Fetch the full chain payload for one underlying (spec §17 C1)."""
        resp = self._fetch(self.chain_url(symbol))
        if resp.status_code != 200:
            log.error("cboe_chain_http_error", symbol=symbol, status=resp.status_code)
            raise DataError(f"CBOE chain HTTP {resp.status_code} for {symbol}")
        try:
            payload: dict[str, Any] = resp.json()
        except ValueError as exc:
            log.error("cboe_chain_bad_json", symbol=symbol, error=str(exc))
            raise DataError(f"CBOE chain returned non-JSON for {symbol}") from exc
        if not isinstance(payload.get("data"), dict) or "options" not in payload["data"]:
            log.error("cboe_chain_bad_shape", symbol=symbol, keys=sorted(payload)[:8])
            raise DataError(f"CBOE chain payload missing data.options for {symbol}")
        return payload
