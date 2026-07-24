"""Conditional-wipe semantics: the autouse wipe fires ONLY for db-marked tests.

Runs a miniature pytest suite in-process (pytester) using the REAL wipe fixture
and the REAL marker-derivation logic, with the TRUNCATE function stubbed to a
recorder — no PostgreSQL needed. The literal ``DATABASE_URL`` string below
auto-marks THIS module ``db`` (tests/db_marker.py), so ``-m "not db"`` deselects
it: these meta-tests are db-marked by their own literal and run in the db/full
lanes, not the fast lane. That marking is harmless — in the fast lane the wipe
no-ops on a blank URL, in the db lane it truncates an already-clean scratch DB.
"""

from __future__ import annotations

import pytest

from tests.fixtures import db_cleanup

_FAKE_TEST_URL = "postgresql://u:pw@host:5432/swingrl_test"  # pragma: allowlist secret

_INNER_CONFTEST = """
from __future__ import annotations

import pytest

from tests.db_marker import DB_FIXTURE_NAMES, module_mentions_database_url
from tests.fixtures.db_cleanup import wipe_db_after_test  # noqa: F401  (autouse)


# NOTE: mirrors tests/conftest.py pytest_collection_modifyitems — keep in sync (db-marker glue).
def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("db") is None and (
            DB_FIXTURE_NAMES.intersection(getattr(item, "fixturenames", []))
            or module_mentions_database_url(str(item.path))
        ):
            item.add_marker(pytest.mark.db)
"""


def test_wipe_fires_only_for_db_marked_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One explicitly db-marked test + one plain test -> exactly one TRUNCATE."""
    calls: list[str] = []
    monkeypatch.setattr(db_cleanup, "_truncate_all_public_tables", calls.append)
    monkeypatch.setenv("DATABASE_URL", _FAKE_TEST_URL)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        test_plain="def test_no_db():\n    assert 1 + 1 == 2\n",
        test_marked=("import pytest\n\n@pytest.mark.db\ndef test_db_marked():\n    assert True\n"),
    )
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-q")
    result.assert_outcomes(passed=2)
    assert calls == [_FAKE_TEST_URL]


def test_wipe_fires_for_auto_derived_db_module(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module mentioning DATABASE_URL is auto-marked -> its test gets wiped."""
    calls: list[str] = []
    monkeypatch.setattr(db_cleanup, "_truncate_all_public_tables", calls.append)
    monkeypatch.setenv("DATABASE_URL", _FAKE_TEST_URL)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        test_mentions=(
            'import os\n\ndef test_reads_env():\n    assert os.environ.get("DATABASE_URL")\n'
        ),
    )
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-q")
    result.assert_outcomes(passed=1)
    assert calls == [_FAKE_TEST_URL]
