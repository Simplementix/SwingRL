"""Schema-integrity preflight: PK fingerprint of canonical tables.

This check would have failed run 1 of 3 on 2026-07-22 with the poisoned table's
name in the message, instead of costing three suite-hours (robustness review P1).
"""

from __future__ import annotations

import os

import psycopg
import pytest

from swingrl.data.postgres_schema import _HMM_STATE_HISTORY_DDL
from tests.fixtures.schema_preflight import expected_pk_tables, schema_integrity_errors


def test_expected_pk_tables_cover_known_pk_tables() -> None:
    """Pure (fast lane): the canonical PK map includes the incident table and friends."""
    tables = expected_pk_tables()
    assert "hmm_state_history" in tables
    assert "fundamentals" in tables
    assert "model_metadata" in tables
    # Sanity: parsing found a substantial share of the 36 canonical tables.
    assert len(tables) >= 10


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_detects_pk_less_table_and_passes_when_canonical() -> None:
    """db lane: a PK-less hmm_state_history (the incident shape) is named; canonical is clean.

    This test intentionally performs the forbidden DROP+hand-recreate pattern —
    WITH a finally-block that restores the canonical DDL, which is exactly the
    teardown discipline the pattern was missing.
    """
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(
            "CREATE TABLE hmm_state_history ("
            " date DATE, environment VARCHAR,"
            " p_bull DOUBLE PRECISION, p_bear DOUBLE PRECISION, p_crisis DOUBLE PRECISION)"
        )
        errors = schema_integrity_errors(db_url)
        assert any("hmm_state_history" in e for e in errors), errors
    finally:
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(_HMM_STATE_HISTORY_DDL)
        conn.close()
    clean_errors = [e for e in schema_integrity_errors(db_url) if "hmm_state_history" in e]
    assert clean_errors == []
