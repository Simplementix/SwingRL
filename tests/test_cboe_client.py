# tests/test_cboe_client.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from swingrl.data.options.cboe_client import CboeChainClient

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.utils.exceptions import DataError


def _client() -> CboeChainClient:
    cfg = OptionsCollectorConfig()
    cfg.rate_limit_per_sec = 1000.0  # no real sleeping in tests
    return CboeChainClient(cfg)


def _response(payload: dict | None, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    if payload is None:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = payload
    return resp


def test_chain_url_from_template() -> None:
    """OPT-CLIENT-1: URL built from the config template (spec §17 C1)."""
    assert _client().chain_url("_SPX") == (
        "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    )


def test_get_option_chain_returns_payload() -> None:
    """OPT-CLIENT-2: happy path returns the parsed JSON dict (spec §17 C1)."""
    ok = {
        "timestamp": "2026-07-14 19:45:10",
        "symbol": "^SPX",
        "data": {"current_price": 7543.59, "options": [{"option": "SPXW260724C07500000"}]},
    }
    with patch(
        "swingrl.data.options.cboe_client.httpx.get", return_value=_response(ok)
    ) as fake_get:
        out = _client().get_option_chain("_SPX")
    assert out["data"]["current_price"] == 7543.59
    fake_get.assert_called_once()


def test_http_error_raises_dataerror() -> None:
    """OPT-CLIENT-3: non-200 -> DataError (spec §10.3)."""
    with patch(
        "swingrl.data.options.cboe_client.httpx.get", return_value=_response({}, status=503)
    ):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")


def test_missing_options_key_raises_dataerror() -> None:
    """OPT-CLIENT-4: payload without data.options -> DataError (spec §10.5)."""
    with patch("swingrl.data.options.cboe_client.httpx.get", return_value=_response({"data": {}})):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")


def test_non_json_raises_dataerror() -> None:
    """OPT-CLIENT-5: non-JSON body -> DataError (spec §10.3)."""
    with patch("swingrl.data.options.cboe_client.httpx.get", return_value=_response(None)):
        with pytest.raises(DataError):
            _client().get_option_chain("SPY")
