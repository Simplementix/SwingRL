"""Persistent cleanup connection: reuse across wipes + transparent reconnect.

db-lane tests — gated on DATABASE_URL like every real-DB test.
"""

from __future__ import annotations

import os

import pytest

from tests.fixtures import db_cleanup

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


def test_cleanup_connection_is_reused_across_wipes() -> None:
    """Two consecutive wipes use the SAME connection object (no per-test connect)."""
    db_url = os.environ["DATABASE_URL"]
    first = db_cleanup._get_cleanup_conn(db_url)
    db_cleanup._truncate_all_public_tables(db_url)
    db_cleanup._truncate_all_public_tables(db_url)
    second = db_cleanup._get_cleanup_conn(db_url)
    assert first is second
    assert not first.closed


def test_cleanup_connection_reopens_after_close() -> None:
    """A dropped connection (server restart) is reopened transparently."""
    db_url = os.environ["DATABASE_URL"]
    db_cleanup._get_cleanup_conn(db_url).close()  # simulate a dropped connection
    db_cleanup._truncate_all_public_tables(db_url)  # must not raise
    assert not db_cleanup._get_cleanup_conn(db_url).closed
