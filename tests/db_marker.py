"""Auto-derivation of the ``db`` marker at collection time.

A test is ``db`` (touches the real PostgreSQL scratch database) when it requests
one of the known real-DB fixtures below, or when its module's source mentions
``DATABASE_URL`` anywhere. This converts the four existing skip spellings
(module pytestmark / per-test skipif / inline pytest.skip / fixture-level skip,
51 files total) into ONE selectable marker without editing any of those files.

Deliberately coarse: over-marking a mock-only test inside a DATABASE_URL-mentioning
module only excludes it from the fast lane; under-marking is impossible without
hardcoding a DB URL in a test, which the no-hardcoded-values rule forbids.
An explicit ``@pytest.mark.db`` always wins (conftest checks it first).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path

# Fixtures that hand tests a real PostgreSQL connection/manager (audited 2026-07-22).
# NOTE the two naming traps flagged by the practices review: ``mock_db`` (execution
# conftest) and ``seeded_duckdb`` (features test_pipeline) are BOTH real Postgres.
DB_FIXTURE_NAMES: frozenset[str] = frozenset(
    {
        "mock_db",  # tests/execution/conftest.py:73 — real DatabaseManager
        "pg_conn",  # tests/features/conftest.py:103
        "seeded_duckdb",  # tests/features/test_pipeline.py:104 — real Postgres
        "db_with_legacy_schema",  # tests/data/conftest.py:64
        "db_config_yaml",  # tests/data/conftest.py:25 — embeds DATABASE_URL
        "db",  # tests/data/test_migration_runner.py:28
        "db_manager",  # tests/data/test_db.py:78
        "memory_db_env",  # tests/test_memory_service.py:81
        "api_client",  # tests/test_memory_service.py:115 — real pool via init_db()
    }
)


def is_db_test(fixturenames: Iterable[str], module_path: str) -> bool:
    """Return True when a test can reach the real database (see module docstring)."""
    if DB_FIXTURE_NAMES.intersection(fixturenames):
        return True
    return module_mentions_database_url(module_path)


@cache
def module_mentions_database_url(path: str) -> bool:
    """Whether the module file at ``path`` mentions DATABASE_URL (cached per path)."""
    try:
        return "DATABASE_URL" in Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
