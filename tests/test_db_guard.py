"""STAGE1: Pure DB-guard helpers — URL classification and target resolution."""

from __future__ import annotations

import pytest

import tests.db_guard as guard
from tests.db_guard import classify_db_url, resolve_target_db_url


@pytest.mark.parametrize(
    ("url", "expected_verdict", "expected_name"),
    [
        ("", "blank", None),
        ("   ", "blank", None),
        (
            "postgresql://swingrl:pw@pg16:5432/swingrl_test",
            "safe",
            "swingrl_test",
        ),  # pragma: allowlist secret
        (
            "postgresql://u:pw@h:5432/anything_test",
            "safe",
            "anything_test",
        ),  # pragma: allowlist secret
        (
            "postgresql://u:pw@h:5432/swingrl_test?sslmode=require",
            "safe",
            "swingrl_test",
        ),  # pragma: allowlist secret
        (
            "postgresql://swingrl:pw@pg16:5432/swingrl",
            "unsafe",
            "swingrl",
        ),  # pragma: allowlist secret
        ("postgresql://u:pw@h:5432/prod?x=1", "unsafe", "prod"),  # pragma: allowlist secret
        ("not-a-url", "unparseable", None),
        (
            "postgresql://u:pw@h:5432/swingrl#foo_test",
            "unsafe",
            "swingrl",
        ),  # pragma: allowlist secret
    ],
)
def test_classify_db_url(url: str, expected_verdict: str, expected_name: str | None) -> None:
    """REQ-STAGE1: classify_db_url labels blank/safe/unsafe/unparseable correctly."""
    verdict, name = classify_db_url(url)
    assert verdict == expected_verdict
    assert name == expected_name


def test_resolve_prefers_env_over_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-STAGE1: DATABASE_URL env wins over the config fallback."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:pw@h:5432/swingrl_test"
    )  # pragma: allowlist secret
    assert (
        resolve_target_db_url() == "postgresql://u:pw@h:5432/swingrl_test"
    )  # pragma: allowlist secret


def test_resolve_falls_back_to_config_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-STAGE1: blank env resolves to config.system.database_url (the hole the guard closes)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    class _Sys:
        database_url = "postgresql://u:pw@h:5432/swingrl"  # production!  # pragma: allowlist secret

    class _Cfg:
        system = _Sys()

    monkeypatch.setattr(guard, "load_config", lambda _path: _Cfg())
    resolved = resolve_target_db_url()
    assert resolved == "postgresql://u:pw@h:5432/swingrl"  # pragma: allowlist secret
    # And that resolved value classifies unsafe -> the suite would refuse to run.
    assert classify_db_url(resolved)[0] == "unsafe"


def test_resolve_returns_blank_on_config_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-STAGE1: config load failure resolves to blank so tests skip gracefully."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def _boom(_path: object) -> object:
        raise OSError("config missing")

    monkeypatch.setattr(guard, "load_config", _boom)
    assert resolve_target_db_url() == ""
