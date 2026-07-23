"""Per-worker isolated-DB derivation: pure tests (fast lane, no DB, no xdist).

Deliberately avoids the literal D-A-T-A-B-A-S-E_URL env-var name in source so
this module stays out of the auto-derived db lane (it never touches Postgres).
"""

from __future__ import annotations

import os

import pytest

import tests.db_worker as db_worker
from tests.db_worker import derive_isolated_db_url, isolation_token

_BASE = "postgresql://u:pw@172.18.5.246:5432/swingrl_test"  # pragma: allowlist secret

# Built from parts so this module never contains the literal env-var name and
# therefore stays out of the auto-marked db lane (see the module docstring).
_DB_ENV = "DATABASE" + "_URL"


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


def test_activate_resolves_base_url_via_db_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A YAML-only base URL (no env var) still triggers per-worker derivation.

    Regression: activate_isolated_db read the env var directly, so a blank env
    plus a ``*_test`` URL in config/swingrl.yaml isolated NO worker — every xdist
    worker then resolved the SAME YAML DB (shared mutable state). It must resolve
    the base the same way the guard does — via resolve_target_db_url (env first,
    then the memoized YAML fallback).
    """
    yaml_only = "postgresql://u:pw@h:5432/swingrl_test"  # pragma: allowlist secret
    expected = "postgresql://u:pw@h:5432/swingrl_gw0_test"  # pragma: allowlist secret

    # Sandbox the env var so activate_isolated_db's rewrite is fully reverted after
    # the test. setenv records an undo entry; delenv then removes the value so the
    # test runs as if the env var were absent (YAML-only). A bare delenv on an
    # already-absent var records NO undo, and activate's write would leak.
    monkeypatch.setenv(_DB_ENV, "sentinel-unused")
    monkeypatch.delenv(_DB_ENV, raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    # Stub the resolution seam to a YAML-only URL (no env coupling). raising=False
    # so the same test drives a clean behavioural RED against the pre-fix code,
    # which never referenced this symbol.
    monkeypatch.setattr(db_worker, "resolve_target_db_url", lambda: yaml_only, raising=False)
    monkeypatch.setattr(db_worker, "_active_worker_url", None)
    monkeypatch.setattr(db_worker, "_admin_base_url", None)

    seen: dict[str, str] = {}
    monkeypatch.setattr(
        db_worker,
        "ensure_isolated_db",
        lambda base, worker: seen.update(base=base, worker=worker),
    )

    db_worker.activate_isolated_db()

    assert seen == {"base": yaml_only, "worker": expected}
    assert db_worker._active_worker_url == expected
    assert os.environ[_DB_ENV] == expected
