"""STAGE1: Pre-wipe re-check (safety layer #3) refuses non-test databases."""

from __future__ import annotations

import pytest

from tests.fixtures.db_cleanup import ensure_wipe_target_is_test_db


def test_refuses_production_db() -> None:
    """REQ-STAGE1: pre-wipe re-check raises on a production DB name."""
    with pytest.raises(RuntimeError, match="not a test database"):
        ensure_wipe_target_is_test_db("postgresql://u:pw@h:5432/swingrl")


def test_refuses_unparseable_db() -> None:
    """REQ-STAGE1: pre-wipe re-check raises when the DB name cannot be parsed."""
    with pytest.raises(RuntimeError, match="not a test database"):
        ensure_wipe_target_is_test_db("not-a-url")


def test_allows_test_db() -> None:
    """REQ-STAGE1: pre-wipe re-check permits a *_test database."""
    assert ensure_wipe_target_is_test_db("postgresql://u:pw@h:5432/swingrl_test") is True


def test_skips_when_blank() -> None:
    """REQ-STAGE1: no DB configured -> skip wipe (returns False, no raise)."""
    assert ensure_wipe_target_is_test_db("") is False
