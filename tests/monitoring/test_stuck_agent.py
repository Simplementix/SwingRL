"""Tests for stuck agent detection logic.

PAPER-14: Stuck agent detection uses trading days for equity, cycles for crypto.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row

from swingrl.monitoring.stuck_agent import check_stuck_agents

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> MagicMock:
    """Provide a mock DatabaseManager backed by PostgreSQL."""
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(db_url, row_factory=dict_row, autocommit=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp TEXT NOT NULL,
            environment TEXT NOT NULL,
            total_value DOUBLE PRECISION NOT NULL,
            equity_value DOUBLE PRECISION,
            crypto_value DOUBLE PRECISION,
            cash_balance DOUBLE PRECISION,
            high_water_mark DOUBLE PRECISION,
            daily_pnl DOUBLE PRECISION,
            drawdown_pct DOUBLE PRECISION,
            PRIMARY KEY (timestamp, environment)
        )
    """)
    conn.execute("DELETE FROM portfolio_snapshots")
    conn.commit()

    db = MagicMock()
    db.connection.return_value.__enter__ = MagicMock(return_value=conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)
    yield db
    conn.close()


def _insert_snapshots(
    conn: Any,
    env: str,
    count: int,
    *,
    all_cash: bool = True,
    total_value: float = 10000.0,
) -> None:
    """Insert test portfolio snapshots.

    Args:
        conn: SQLite connection.
        env: Environment name.
        count: Number of snapshots to insert.
        all_cash: If True, cash_balance == total_value. Otherwise cash < total.
        total_value: Portfolio total value.
    """
    base = datetime(2026, 3, 1, tzinfo=UTC)
    for i in range(count):
        ts = (base + timedelta(hours=i)).isoformat()
        cash = total_value if all_cash else total_value * 0.5
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp, environment, total_value, cash_balance) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT DO NOTHING",
            (ts, env, total_value, cash),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Test: stuck equity detection
# ---------------------------------------------------------------------------


class TestStuckEquity:
    """PAPER-14: Equity stuck agent detected at 10 consecutive all-cash snapshots."""

    def test_equity_stuck_at_threshold(self, mock_db: MagicMock) -> None:
        """10 consecutive all-cash equity snapshots triggers stuck alert."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "equity", 10, all_cash=True)

        alerts = check_stuck_agents(mock_db)
        equity_alerts = [a for a in alerts if a["environment"] == "equity"]
        assert len(equity_alerts) == 1
        assert equity_alerts[0]["consecutive_cash_cycles"] == 10

    def test_equity_not_stuck_below_threshold(self, mock_db: MagicMock) -> None:
        """9 all-cash snapshots does not trigger equity stuck alert."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "equity", 9, all_cash=True)

        alerts = check_stuck_agents(mock_db)
        equity_alerts = [a for a in alerts if a["environment"] == "equity"]
        assert len(equity_alerts) == 0


# ---------------------------------------------------------------------------
# Test: stuck crypto detection
# ---------------------------------------------------------------------------


class TestStuckCrypto:
    """PAPER-14: Crypto stuck agent detected at 30 consecutive all-cash snapshots."""

    def test_crypto_stuck_at_threshold(self, mock_db: MagicMock) -> None:
        """30 consecutive all-cash crypto snapshots triggers stuck alert."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "crypto", 30, all_cash=True)

        alerts = check_stuck_agents(mock_db)
        crypto_alerts = [a for a in alerts if a["environment"] == "crypto"]
        assert len(crypto_alerts) == 1
        assert crypto_alerts[0]["consecutive_cash_cycles"] == 30

    def test_crypto_not_stuck_below_threshold(self, mock_db: MagicMock) -> None:
        """29 all-cash snapshots does not trigger crypto stuck alert."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "crypto", 29, all_cash=True)

        alerts = check_stuck_agents(mock_db)
        crypto_alerts = [a for a in alerts if a["environment"] == "crypto"]
        assert len(crypto_alerts) == 0


# ---------------------------------------------------------------------------
# Test: not stuck when recent activity
# ---------------------------------------------------------------------------


class TestNotStuck:
    """PAPER-14: No alert when environments have recent non-cash activity."""

    def test_equity_not_stuck_with_positions(self, mock_db: MagicMock) -> None:
        """Equity with positions (cash != total) is not stuck."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "equity", 15, all_cash=False)

        alerts = check_stuck_agents(mock_db)
        equity_alerts = [a for a in alerts if a["environment"] == "equity"]
        assert len(equity_alerts) == 0

    def test_empty_snapshots_not_stuck(self, mock_db: MagicMock) -> None:
        """No snapshots at all returns empty list."""
        alerts = check_stuck_agents(mock_db)
        assert alerts == []

    def test_mixed_cash_and_positions_not_stuck(self, mock_db: MagicMock) -> None:
        """Recent position snapshot among cash snapshots is not stuck."""
        conn = mock_db.connection.return_value.__enter__.return_value
        base = datetime(2026, 3, 1, tzinfo=UTC)

        # Insert 9 all-cash then 1 with positions (most recent)
        for i in range(9):
            ts = (base + timedelta(hours=i)).isoformat()
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (ts, "equity", 10000.0, 10000.0),
            )
        ts_last = (base + timedelta(hours=9)).isoformat()
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp, environment, total_value, cash_balance) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT DO NOTHING",
            (ts_last, "equity", 10000.0, 5000.0),
        )
        conn.commit()

        alerts = check_stuck_agents(mock_db)
        equity_alerts = [a for a in alerts if a["environment"] == "equity"]
        assert len(equity_alerts) == 0


# ---------------------------------------------------------------------------
# Test: last_action_date in alert dict
# ---------------------------------------------------------------------------


class TestLastActionDate:
    """PAPER-14: Stuck alert includes last_action_date for diagnostics."""

    def test_stuck_alert_has_last_action_date(self, mock_db: MagicMock) -> None:
        """Stuck alert dict includes last_action_date key."""
        conn = mock_db.connection.return_value.__enter__.return_value
        _insert_snapshots(conn, "equity", 10, all_cash=True)

        alerts = check_stuck_agents(mock_db)
        assert len(alerts) == 1
        assert "last_action_date" in alerts[0]
