"""Tests for the Phase 0.2 iteration_results CPS column extension.

CPS-SCHEMA-01: postgres_schema.py CREATE TABLE contains all 16 new CPS columns.
CPS-SCHEMA-02: scripts/migrations/add_cps_columns.py declares the same columns
               as the canonical DDL.
CPS-SCHEMA-03: Migration is idempotent — running apply_migration() against an
               existing iteration_results table adds new columns once and is
               safe to re-run.

The static schema tests run with no database. The idempotency test runs
against the homelab Postgres only when DATABASE_URL is set, otherwise it
is skipped.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from swingrl.data.postgres_schema import _ITERATION_RESULTS_DDL

# Mirror of scripts/migrations/add_cps_columns.py::_NEW_COLUMNS — kept here so
# the test is self-contained and would catch divergence if either side changes
# without the other.
_EXPECTED_CPS_COLUMNS: list[tuple[str, str]] = [
    ("cps_v1_multiplicative", "DOUBLE PRECISION"),
    ("cps_v2_additive", "DOUBLE PRECISION"),
    ("cps_v3_sortino", "DOUBLE PRECISION"),
    ("cps_v1_treatment_only", "DOUBLE PRECISION"),
    ("cps_v1_control_only", "DOUBLE PRECISION"),
    ("cps_components", "TEXT"),
    ("cps_formula_version", "TEXT"),
    ("worst_fold_number", "INTEGER"),
    ("worst_fold_mdd", "DOUBLE PRECISION"),
    ("worst_fold_max_single_loss", "DOUBLE PRECISION"),
    ("median_return", "DOUBLE PRECISION"),
    ("mean_winner_sharpe", "DOUBLE PRECISION"),
    ("winners_count", "INTEGER"),
    ("chronic_failure_count", "INTEGER"),
    ("return_regression_delta", "DOUBLE PRECISION"),
    ("dedup_rows_dropped", "INTEGER"),
]


# ---------------------------------------------------------------------------
# Static schema tests (no DB required)
# ---------------------------------------------------------------------------


class TestCanonicalDdlContainsCpsColumns:
    """CPS-SCHEMA-01: the canonical DDL string must declare every new column."""

    @pytest.mark.parametrize("col_name,col_type", _EXPECTED_CPS_COLUMNS)
    def test_column_present_in_ddl(self, col_name: str, col_type: str) -> None:
        """Each expected CPS column appears in the canonical CREATE TABLE."""
        assert col_name in _ITERATION_RESULTS_DDL, (
            f"Column {col_name} missing from _ITERATION_RESULTS_DDL"
        )
        # The type must also appear on the same line (loose check — exact
        # whitespace varies). We split on the column name and check the
        # next ~30 chars contain the type token.
        idx = _ITERATION_RESULTS_DDL.index(col_name)
        snippet = _ITERATION_RESULTS_DDL[idx : idx + 80]
        assert col_type.split()[0] in snippet, (
            f"Column {col_name} found but type {col_type} not on same line: {snippet!r}"
        )


class TestMigrationScriptColumnList:
    """CPS-SCHEMA-02: the migration script's _NEW_COLUMNS must match the
    canonical DDL exactly.

    Detects drift if either the DDL or the migration is updated without the
    other. Imports the script's module-level constant directly.
    """

    def test_migration_column_list_matches_expected(self) -> None:
        """add_cps_columns._NEW_COLUMNS == _EXPECTED_CPS_COLUMNS (this test file)."""
        # Import lazily so the test does not require DATABASE_URL.
        import sys
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))
        from scripts.migrations.add_cps_columns import _NEW_COLUMNS  # noqa: E402

        assert list(_NEW_COLUMNS) == _EXPECTED_CPS_COLUMNS

    def test_migration_column_count(self) -> None:
        """16 new CPS columns are declared."""
        import sys
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))
        from scripts.migrations.add_cps_columns import _NEW_COLUMNS  # noqa: E402

        assert len(_NEW_COLUMNS) == 16


# ---------------------------------------------------------------------------
# Live idempotency test against homelab pg16 (skipped if DATABASE_URL unset)
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_conn() -> psycopg.Connection:
    """Connect to pg16; skip the test if DATABASE_URL is not set."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set; skipping live Postgres test")
    return psycopg.connect(db_url)


class TestMigrationIdempotency:
    """CPS-SCHEMA-03: running the migration twice is safe and a no-op the second time."""

    def test_apply_migration_then_reapply_is_noop(self, pg_conn: psycopg.Connection) -> None:
        """First run adds 16 columns (or 0 if already migrated). Second run adds 0."""
        import sys
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))
        from scripts.migrations.add_cps_columns import apply_migration  # noqa: E402

        # First call: count goes either 16 (fresh) or 0 (already applied).
        first = apply_migration(pg_conn)
        pg_conn.commit()
        assert first["added"] + first["already_present"] == 16

        # Second call: must be a complete no-op (all already present).
        second = apply_migration(pg_conn)
        pg_conn.commit()
        assert second["added"] == 0
        assert second["already_present"] == 16
