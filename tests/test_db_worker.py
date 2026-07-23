"""Per-worker isolated-DB derivation: pure tests (fast lane, no DB, no xdist).

Deliberately avoids the literal D-A-T-A-B-A-S-E_URL env-var name in source so
this module stays out of the auto-derived db lane (it never touches Postgres).
"""

from __future__ import annotations

import pytest

from tests.db_worker import derive_isolated_db_url, isolation_token

_BASE = "postgresql://u:pw@172.18.5.246:5432/swingrl_test"  # pragma: allowlist secret


def test_derive_gw0() -> None:
    """swingrl_test + gw0 -> swingrl_gw0_test (guard-compatible _test suffix)."""
    assert (
        derive_isolated_db_url(_BASE, "gw0")
        == "postgresql://u:pw@172.18.5.246:5432/swingrl_gw0_test"  # pragma: allowlist secret
    )


def test_derive_preserves_query_string() -> None:
    """URL options after the DB name survive the rewrite."""
    assert (
        derive_isolated_db_url(_BASE + "?sslmode=disable", "gw3")
        == "postgresql://u:pw@172.18.5.246:5432/swingrl_gw3_test?sslmode=disable"  # pragma: allowlist secret
    )


def test_derive_refuses_non_test_base() -> None:
    """A production-named base URL can never spawn clones."""
    with pytest.raises(RuntimeError, match="non-test base URL"):
        derive_isolated_db_url(
            "postgresql://u:pw@h:5432/swingrl",  # pragma: allowlist secret
            "gw0",
        )


def test_isolation_token_xdist_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """xdist workers always isolate, keyed by their worker id."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    assert isolation_token() == "gw2"


def test_isolation_token_serial_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial sessions isolate only when SWINGRL_TEST_ISOLATED_DB=1 (pid-keyed)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("SWINGRL_TEST_ISOLATED_DB", "1")
    token = isolation_token()
    assert token is not None
    assert token.startswith("main")


def test_isolation_token_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No xdist, no opt-in -> use the configured URL as-is (backward compatible)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("SWINGRL_TEST_ISOLATED_DB", raising=False)
    assert isolation_token() is None
