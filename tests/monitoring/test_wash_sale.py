"""Tests for wash sale scanner.

PAPER-17: Wash sale scanner flags buys within 30-day window of realized loss.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from swingrl.execution.types import FillResult
from swingrl.monitoring.wash_sale import record_realized_loss, scan_wash_sales

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_db() -> tuple[MagicMock, sqlite3.Connection]:
    """Create a mock DatabaseManager backed by in-memory SQLite."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE wash_sale_tracker (
            symbol TEXT NOT NULL,
            sale_date TEXT NOT NULL,
            loss_amount REAL NOT NULL,
            wash_window_end TEXT NOT NULL,
            triggered INTEGER DEFAULT 0,
            PRIMARY KEY (symbol, sale_date)
        )
    """)
    conn.commit()

    db = MagicMock()
    db.sqlite.return_value.__enter__ = MagicMock(return_value=conn)
    db.sqlite.return_value.__exit__ = MagicMock(return_value=False)
    return db, conn


# ---------------------------------------------------------------------------
# Test: record_realized_loss
# ---------------------------------------------------------------------------


class TestRecordRealizedLoss:
    """PAPER-17: record_realized_loss inserts wash window record."""

    def test_inserts_row_with_30_day_window(self) -> None:
        """Record creates row with wash_window_end = sale_date + 30 days."""
        db, conn = _make_mock_db()
        record_realized_loss("SPY", "2026-03-01", 150.0, db)

        rows = conn.execute("SELECT * FROM wash_sale_tracker").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["symbol"] == "SPY"
        assert row["sale_date"] == "2026-03-01"
        assert row["loss_amount"] == 150.0
        assert row["wash_window_end"] == "2026-03-31"
        assert row["triggered"] == 0

    def test_replaces_existing_record(self) -> None:
        """Same symbol+sale_date replaces previous record."""
        db, conn = _make_mock_db()
        record_realized_loss("SPY", "2026-03-01", 100.0, db)
        record_realized_loss("SPY", "2026-03-01", 200.0, db)

        rows = conn.execute("SELECT * FROM wash_sale_tracker").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["loss_amount"] == 200.0


# ---------------------------------------------------------------------------
# Test: scan_wash_sales — equity buy in window
# ---------------------------------------------------------------------------


class TestScanWashSalesEquity:
    """PAPER-17: Detects equity buys within 30-day wash window."""

    def test_flags_buy_within_window(self) -> None:
        """Equity buy of same symbol within wash window is flagged."""
        db, conn = _make_mock_db()
        # Loss sale on March 1, window ends March 31
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        window_end = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO wash_sale_tracker VALUES (?, ?, ?, ?, 0)",
            ("SPY", today, 150.0, window_end),
        )
        conn.commit()

        fill = FillResult(
            trade_id="t1",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        warnings = scan_wash_sales([fill], db)
        assert len(warnings) == 1
        assert warnings[0]["symbol"] == "SPY"

    def test_no_flag_after_window_expired(self) -> None:
        """Buy after wash window expired is not flagged."""
        db, conn = _make_mock_db()
        # Window already expired
        conn.execute(
            "INSERT INTO wash_sale_tracker VALUES (?, ?, ?, ?, 0)",
            ("SPY", "2026-01-01", 100.0, "2026-01-31"),
        )
        conn.commit()

        fill = FillResult(
            trade_id="t2",
            symbol="SPY",
            side="buy",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        warnings = scan_wash_sales([fill], db)
        assert len(warnings) == 0

    def test_updates_triggered_flag(self) -> None:
        """Matched wash sale sets triggered=1 in database."""
        db, conn = _make_mock_db()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        window_end = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO wash_sale_tracker VALUES (?, ?, ?, ?, 0)",
            ("QQQ", today, 75.0, window_end),
        )
        conn.commit()

        fill = FillResult(
            trade_id="t3",
            symbol="QQQ",
            side="buy",
            quantity=5.0,
            fill_price=380.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        scan_wash_sales([fill], db)

        row = conn.execute(
            "SELECT triggered FROM wash_sale_tracker WHERE symbol = ?", ("QQQ",)
        ).fetchone()
        assert dict(row)["triggered"] == 1


# ---------------------------------------------------------------------------
# Test: scan_wash_sales — ignores crypto
# ---------------------------------------------------------------------------


class TestScanWashSalesCrypto:
    """PAPER-17: Wash sale scanner ignores crypto fills."""

    def test_crypto_buy_ignored(self) -> None:
        """Crypto buy fill is not checked for wash sales."""
        db, conn = _make_mock_db()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        window_end = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO wash_sale_tracker VALUES (?, ?, ?, ?, 0)",
            ("BTC", today, 500.0, window_end),
        )
        conn.commit()

        fill = FillResult(
            trade_id="t4",
            symbol="BTC",
            side="buy",
            quantity=0.5,
            fill_price=60000.0,
            commission=0.0,
            slippage=0.0,
            environment="crypto",
            broker="binance_us",
        )

        warnings = scan_wash_sales([fill], db)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Test: scan_wash_sales — ignores sell fills
# ---------------------------------------------------------------------------


class TestScanWashSalesSellIgnored:
    """PAPER-17: Sell fills are not checked for wash sales."""

    def test_sell_fill_ignored(self) -> None:
        """Equity sell fill is not checked for wash sales."""
        db, conn = _make_mock_db()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        window_end = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO wash_sale_tracker VALUES (?, ?, ?, ?, 0)",
            ("SPY", today, 150.0, window_end),
        )
        conn.commit()

        fill = FillResult(
            trade_id="t5",
            symbol="SPY",
            side="sell",
            quantity=10.0,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
        )

        warnings = scan_wash_sales([fill], db)
        assert len(warnings) == 0
