"""A25/A30 cutover: swingrl-memory service refuses to start against a stale schema ledger.

Mock-based (no DATABASE_URL / real Postgres needed): the memory service cannot import
swingrl.* (separate container -- verified pattern at
services/memory/memory_agents/query.py:97, which reimplements config loading with
yaml.safe_load rather than importing swingrl.config.schema). The floor-semantics check
here is deliberately duplicated from src/swingrl/data/migration_runner.py rather than
imported -- see the _EXPECTED_SCHEMA_VERSION comment in services/memory/db.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap -- mirrors tests/test_memory_service.py exactly.
# Ensures services/memory/ is first on path so db can be imported without
# conflicting with any similarly-named module under scripts/.
# ---------------------------------------------------------------------------
_MEMORY_SERVICE_DIR = Path(__file__).parent.parent / "services" / "memory"
_MEMORY_MODULE_NAMES = [
    "app",
    "db",
    "auth",
    "memory_agents",
    "memory_agents.ingest",
    "memory_agents.consolidate",
    "memory_agents.query",
    "routers",
    "routers.core",
    "routers.training",
    "routers.debug",
]

if str(_MEMORY_SERVICE_DIR) in sys.path:
    sys.path.remove(str(_MEMORY_SERVICE_DIR))
sys.path.insert(0, str(_MEMORY_SERVICE_DIR))

for _mod in list(sys.modules.keys()):
    if any(_mod == name or _mod.startswith(name + ".") for name in _MEMORY_MODULE_NAMES):
        del sys.modules[_mod]

import db as memory_db_module  # noqa: E402


def _mock_conn(fetchone_returns: list[object]) -> MagicMock:
    """Mock psycopg connection: conn.execute(...).fetchone() consumes side_effect in order."""
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = fetchone_returns
    return conn


def test_assert_memory_schema_current_raises_when_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    """A30 floor semantics: ledger version behind the floor -> RuntimeError."""
    monkeypatch.setattr(memory_db_module, "_EXPECTED_SCHEMA_VERSION", 3)
    # 1st fetchone(): ledger table exists check -> truthy row. 2nd: max(version) -> 1.
    conn = _mock_conn([{"?column?": 1}, {"v": 1}])

    with pytest.raises(RuntimeError, match="behind expected"):
        memory_db_module._assert_memory_schema_current(conn)


def test_assert_memory_schema_current_warns_when_ahead(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A30 floor semantics: ledger version ahead of the floor warns, does not raise."""
    import logging

    monkeypatch.setattr(memory_db_module, "_EXPECTED_SCHEMA_VERSION", 1)
    conn = _mock_conn([{"?column?": 1}, {"v": 5}])

    with caplog.at_level(logging.WARNING):
        memory_db_module._assert_memory_schema_current(conn)  # must not raise


def test_assert_memory_schema_current_ok_when_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A30 floor semantics: ledger version exactly at the floor -> no raise."""
    monkeypatch.setattr(memory_db_module, "_EXPECTED_SCHEMA_VERSION", 2)
    conn = _mock_conn([{"?column?": 1}, {"v": 2}])

    memory_db_module._assert_memory_schema_current(conn)  # must not raise


def test_assert_memory_schema_current_treats_missing_ledger_as_version_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh-DB decision: no schema_migrations table at all == version 0 == behind.

    A database before any V-file migration has no ledger table. Task 3 treats this the
    same as an empty ledger (both resolve to version 0), which is behind whenever
    _EXPECTED_SCHEMA_VERSION > 0 -- mirroring migration_runner.current_schema_version(),
    which achieves the same result by CREATE TABLE IF NOT EXISTS-ing the (then-empty)
    ledger before querying it.
    """
    monkeypatch.setattr(memory_db_module, "_EXPECTED_SCHEMA_VERSION", 1)
    # 1st fetchone(): ledger table existence check -> None (table absent).
    # No 2nd fetchone(): the version query must be skipped when the table is absent.
    conn = _mock_conn([None])

    with pytest.raises(RuntimeError, match="behind expected"):
        memory_db_module._assert_memory_schema_current(conn)


def test_init_db_calls_schema_guard_after_local_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    """init_db() must call the ledger guard so the service refuses to start when stale."""
    monkeypatch.setattr(memory_db_module, "_EXPECTED_SCHEMA_VERSION", 1)

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []  # information_schema.columns probe
    # Ledger existence check -> table absent -> version 0 -> behind floor 1.
    conn.execute.return_value.fetchone.side_effect = [None]

    class _FakePool:
        def connection(self) -> MagicMock:
            cm = MagicMock()
            cm.__enter__.return_value = conn
            cm.__exit__.return_value = False
            return cm

    monkeypatch.setattr(memory_db_module, "_get_pool", lambda: _FakePool())

    with pytest.raises(RuntimeError, match="behind expected"):
        memory_db_module.init_db()
