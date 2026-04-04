"""Tests for emergency halt flag CRUD operations.

PAPER-12: Pre-cycle halt check returns True/False based on emergency flag state.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

from swingrl.scheduler.halt_check import clear_halt, init_emergency_flags, is_halted, set_halt

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock DatabaseManager backed by a real PostgreSQL connection."""
    db_url = os.environ["DATABASE_URL"]
    db = MagicMock()

    def _pg_ctx():
        """Context manager yielding a real PostgreSQL connection."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():  # type: ignore[no-untyped-def]
            conn = psycopg.connect(db_url, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return _ctx()

    db.sqlite = _pg_ctx
    return db


class TestInitEmergencyFlags:
    """init_emergency_flags creates the table if it does not exist."""

    def test_creates_table(self, mock_db: MagicMock) -> None:
        """Table emergency_flags exists after init."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            cursor = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                " AND tablename = 'emergency_flags'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent(self, mock_db: MagicMock) -> None:
        """Calling init twice does not error."""
        init_emergency_flags(mock_db)
        init_emergency_flags(mock_db)


class TestIsHalted:
    """is_halted returns True only when halt row has active=1."""

    def test_no_halt_row_returns_false(self, mock_db: MagicMock) -> None:
        """PAPER-12: No halt row means not halted."""
        assert is_halted(mock_db) is False

    def test_active_zero_returns_false(self, mock_db: MagicMock) -> None:
        """PAPER-12: halt row with active=0 means not halted."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("INSERT INTO emergency_flags (flag_name, active) VALUES ('halt', 0)")
        assert is_halted(mock_db) is False

    def test_active_one_returns_true(self, mock_db: MagicMock) -> None:
        """PAPER-12: halt row with active=1 means halted."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO emergency_flags (flag_name, active, reason, set_by) "
                "VALUES ('halt', 1, 'test', 'unit_test')"
            )
        assert is_halted(mock_db) is True


class TestSetHalt:
    """set_halt inserts/updates halt row with active=1."""

    def test_sets_halt_flag(self, mock_db: MagicMock) -> None:
        """PAPER-12: set_halt makes is_halted return True."""
        set_halt(mock_db, reason="manual stop", set_by="test")
        assert is_halted(mock_db) is True

    def test_records_reason_and_set_by(self, mock_db: MagicMock) -> None:
        """set_halt stores reason and set_by fields."""
        set_halt(mock_db, reason="drawdown limit", set_by="circuit_breaker")
        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT reason, set_by FROM emergency_flags WHERE flag_name='halt'"
            ).fetchone()
        assert row["reason"] == "drawdown limit"
        assert row["set_by"] == "circuit_breaker"

    def test_records_timestamp(self, mock_db: MagicMock) -> None:
        """set_halt stores a UTC set_at timestamp."""
        set_halt(mock_db, reason="test", set_by="test")
        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT set_at FROM emergency_flags WHERE flag_name='halt'"
            ).fetchone()
        assert row["set_at"] is not None


class TestClearHalt:
    """clear_halt sets active=0 on halt row."""

    def test_clears_halt_flag(self, mock_db: MagicMock) -> None:
        """PAPER-12: clear_halt makes is_halted return False."""
        set_halt(mock_db, reason="test", set_by="test")
        assert is_halted(mock_db) is True
        clear_halt(mock_db)
        assert is_halted(mock_db) is False

    def test_clear_when_not_set(self, mock_db: MagicMock) -> None:
        """clear_halt is a no-op when no halt row exists."""
        init_emergency_flags(mock_db)
        clear_halt(mock_db)  # should not raise
        assert is_halted(mock_db) is False
