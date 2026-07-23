"""Auto-derived ``db`` marker: pure derivation-logic tests (fast lane, no DB).

The four historical DATABASE_URL gating spellings across 51 test files (module
pytestmark, per-test skipif, inline pytest.skip, fixture-level skip) collapse to
one collection-time signal: the test requests a known real-DB fixture, or its
module source mentions DATABASE_URL. See tests/db_marker.py for the rationale.
"""

from __future__ import annotations

from pathlib import Path

from tests.db_marker import DB_FIXTURE_NAMES, is_db_test, module_mentions_database_url


def test_real_db_fixture_triggers_db() -> None:
    """A test requesting a known real-DB fixture is a db test."""
    assert is_db_test(["tmp_path", "pg_conn"], "") is True


def test_module_mention_triggers_db() -> None:
    """A module that reads DATABASE_URL anywhere makes all its tests db tests."""
    source = 'db_url = os.environ.get("DATABASE_URL", "")'
    assert is_db_test(["tmp_path"], source) is True


def test_plain_test_is_not_db() -> None:
    """No DB fixture + no DATABASE_URL mention -> fast lane."""
    assert is_db_test(["tmp_path", "loaded_config"], "import pandas as pd") is False


def test_known_real_db_fixtures_registered() -> None:
    """The audited real-DB fixture names are all present (practices review §0)."""
    expected = {
        "mock_db",
        "pg_conn",
        "seeded_duckdb",
        "db_with_legacy_schema",
        "db_config_yaml",
        "db",
        "db_manager",
        "memory_db_env",
        "api_client",
    }
    assert expected <= DB_FIXTURE_NAMES


def test_module_mentions_database_url_reads_file(tmp_path: Path) -> None:
    """File-content check is exact and cached per path."""
    mentions = tmp_path / "test_mentions.py"
    mentions.write_text('URL = os.environ["DATABASE_URL"]\n')
    clean = tmp_path / "test_clean.py"
    clean.write_text("def test_ok():\n    assert True\n")
    assert module_mentions_database_url(str(mentions)) is True
    assert module_mentions_database_url(str(clean)) is False


def test_module_mentions_database_url_missing_file_is_false() -> None:
    """Unreadable path never crashes collection."""
    assert module_mentions_database_url("/nonexistent/never/test_x.py") is False
