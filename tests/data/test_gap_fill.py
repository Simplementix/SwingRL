"""Tests for gap detection and filling from alternate data sources.

Covers detect_crypto_gaps, detect_equity_gaps, fill_crypto_gap,
_parse_klines_to_df, and the orchestrator detect_and_fill_crypto_gaps.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from swingrl.data.gap_fill import (
    GapRecord,
    _parse_klines_to_df,
    detect_crypto_gaps,
    detect_equity_gaps,
    fill_crypto_gap,
)

# ---------------------------------------------------------------------------
# Skip entire module if no PostgreSQL available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg_conn() -> object:
    """Create a PostgreSQL connection with ohlcv_4h and ohlcv_daily tables."""
    import psycopg  # noqa: PLC0415
    from psycopg.rows import dict_row  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg.connect(db_url, row_factory=dict_row, autocommit=True)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ohlcv_4h ("
        "  symbol TEXT, datetime TIMESTAMP, open DOUBLE PRECISION, high DOUBLE PRECISION, "
        "  low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, source TEXT, "
        "  PRIMARY KEY (symbol, datetime))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ohlcv_daily ("
        "  symbol TEXT, date DATE, open DOUBLE PRECISION, high DOUBLE PRECISION, "
        "  low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, "
        "  PRIMARY KEY (symbol, date))"
    )
    conn.execute("TRUNCATE TABLE ohlcv_4h, ohlcv_daily CASCADE")
    return conn


def _insert_crypto_rows(
    conn: object,
    symbol: str,
    start: datetime,
    interval_hours: int,
    count: int,
    *,
    skip_indices: set[int] | None = None,
) -> None:
    """Insert crypto rows with optional gaps at specific indices."""
    ts = start
    for i in range(count):
        if skip_indices and i in skip_indices:
            ts += timedelta(hours=interval_hours)
            continue
        conn.execute(  # type: ignore[union-attr]
            "INSERT INTO ohlcv_4h (symbol, datetime, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [symbol, ts.strftime("%Y-%m-%d %H:%M:%S"), 100.0, 101.0, 99.0, 100.5, 1.0],
        )
        ts += timedelta(hours=interval_hours)


def _insert_equity_rows(
    conn: object,
    symbol: str,
    start_date: datetime,
    count: int,
    *,
    skip_indices: set[int] | None = None,
) -> None:
    """Insert daily equity rows with optional gaps."""
    dt = start_date
    for i in range(count):
        if skip_indices and i in skip_indices:
            dt += timedelta(days=1)
            continue
        conn.execute(  # type: ignore[union-attr]
            "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [symbol, dt.strftime("%Y-%m-%d"), 100.0, 101.0, 99.0, 100.5, 1000.0],
        )
        dt += timedelta(days=1)


def _cleanup_pg_conn(conn: object) -> None:
    """Truncate tables and close connection."""
    conn.execute("TRUNCATE TABLE ohlcv_4h, ohlcv_daily CASCADE")  # type: ignore[union-attr]
    conn.close()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# GapRecord
# ---------------------------------------------------------------------------


class TestGapRecord:
    """Tests for GapRecord dataclass."""

    def test_gap_hours_calculation(self) -> None:
        """Gap hours computed from duration."""
        gap = GapRecord(
            symbol="BTCUSDT",
            environment="crypto",
            gap_start=datetime(2024, 1, 1, tzinfo=UTC),
            gap_end=datetime(2024, 1, 2, tzinfo=UTC),
            gap_duration=timedelta(hours=24),
            filled=False,
            source="detected",
        )
        assert gap.gap_hours == 24.0

    def test_frozen_dataclass(self) -> None:
        """GapRecord is immutable."""
        gap = GapRecord(
            symbol="BTCUSDT",
            environment="crypto",
            gap_start=datetime(2024, 1, 1, tzinfo=UTC),
            gap_end=datetime(2024, 1, 2, tzinfo=UTC),
            gap_duration=timedelta(hours=24),
            filled=False,
            source="detected",
        )
        with pytest.raises(AttributeError):
            gap.filled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_crypto_gaps
# ---------------------------------------------------------------------------


class TestDetectCryptoGaps:
    """DATA-04: Detect gaps in crypto 4H data."""

    def test_no_gaps(self, loaded_config: object) -> None:
        """No gaps reported when data is continuous."""
        conn = _make_pg_conn()
        try:
            start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            _insert_crypto_rows(conn, "BTCUSDT", start, 4, 50)
            _insert_crypto_rows(conn, "ETHUSDT", start, 4, 50)

            with patch("swingrl.data.gap_fill.DatabaseManager") as mock_db:
                mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
                mock_db.return_value.connection.return_value.__exit__ = lambda s, *a: None
                gaps = detect_crypto_gaps(loaded_config)  # type: ignore[arg-type]

            assert len(gaps) == 0
        finally:
            _cleanup_pg_conn(conn)

    def test_detects_large_gap(self, loaded_config: object) -> None:
        """Detects a 48h gap (>24h threshold)."""
        conn = _make_pg_conn()
        try:
            start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            _insert_crypto_rows(conn, "BTCUSDT", start, 4, 50)
            _insert_crypto_rows(conn, "ETHUSDT", start, 4, 50, skip_indices=set(range(10, 22)))

            with patch("swingrl.data.gap_fill.DatabaseManager") as mock_db:
                mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
                mock_db.return_value.connection.return_value.__exit__ = lambda s, *a: None
                gaps = detect_crypto_gaps(loaded_config)  # type: ignore[arg-type]

            assert len(gaps) == 1
            assert gaps[0].symbol == "ETHUSDT"
            assert gaps[0].gap_hours > 24.0
        finally:
            _cleanup_pg_conn(conn)

    def test_ignores_small_gap(self, loaded_config: object) -> None:
        """Gaps under 24h (e.g. 12h) are not reported."""
        conn = _make_pg_conn()
        try:
            start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
            _insert_crypto_rows(conn, "BTCUSDT", start, 4, 50, skip_indices={10, 11})
            _insert_crypto_rows(conn, "ETHUSDT", start, 4, 50)

            with patch("swingrl.data.gap_fill.DatabaseManager") as mock_db:
                mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
                mock_db.return_value.connection.return_value.__exit__ = lambda s, *a: None
                gaps = detect_crypto_gaps(loaded_config)  # type: ignore[arg-type]

            assert len(gaps) == 0
        finally:
            _cleanup_pg_conn(conn)


# ---------------------------------------------------------------------------
# detect_equity_gaps
# ---------------------------------------------------------------------------


class TestDetectEquityGaps:
    """DATA-04: Detect gaps in equity daily data."""

    def test_no_gaps(self, loaded_config: object) -> None:
        """No gaps when consecutive trading days."""
        conn = _make_pg_conn()
        try:
            start = datetime(2024, 1, 1)
            for symbol in ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLE", "VNQ"]:
                _insert_equity_rows(conn, symbol, start, 30)

            with patch("swingrl.data.gap_fill.DatabaseManager") as mock_db:
                mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
                mock_db.return_value.connection.return_value.__exit__ = lambda s, *a: None
                gaps = detect_equity_gaps(loaded_config)  # type: ignore[arg-type]

            assert len(gaps) == 0
        finally:
            _cleanup_pg_conn(conn)

    def test_detects_large_gap(self, loaded_config: object) -> None:
        """Detects a 7-day gap (>5 day threshold)."""
        conn = _make_pg_conn()
        try:
            start = datetime(2024, 1, 1)
            _insert_equity_rows(conn, "SPY", start, 30, skip_indices=set(range(5, 12)))
            for symbol in ["QQQ", "IWM", "DIA", "TLT", "GLD", "XLE", "VNQ"]:
                _insert_equity_rows(conn, symbol, start, 30)

            with patch("swingrl.data.gap_fill.DatabaseManager") as mock_db:
                mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
                mock_db.return_value.connection.return_value.__exit__ = lambda s, *a: None
                gaps = detect_equity_gaps(loaded_config)  # type: ignore[arg-type]

            assert len(gaps) == 1
            assert gaps[0].symbol == "SPY"
        finally:
            _cleanup_pg_conn(conn)


# ---------------------------------------------------------------------------
# _parse_klines_to_df
# ---------------------------------------------------------------------------


class TestParseKlinesToDf:
    """Parse raw Binance klines response into DataFrame."""

    def test_parses_klines(self) -> None:
        """Parses standard klines response with correct columns and index."""
        raw = [
            [
                1704067200000,
                "42000.0",
                "42100.0",
                "41900.0",
                "42050.0",
                "1.5",
                1704081600000,
                "63000.0",
                100,
                "0.8",
                "33600.0",
                "0",
            ],
            [
                1704081600000,
                "42050.0",
                "42150.0",
                "41950.0",
                "42100.0",
                "2.0",
                1704096000000,
                "84200.0",
                150,
                "1.0",
                "42100.0",
                "0",
            ],
        ]
        df = _parse_klines_to_df(raw)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert df["volume"].iloc[0] == pytest.approx(63000.0)

    def test_empty_response(self) -> None:
        """Returns empty DataFrame for empty response."""
        df = _parse_klines_to_df([])
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# fill_crypto_gap
# ---------------------------------------------------------------------------


class TestFillCryptoGap:
    """Fill a single crypto gap from Binance Global."""

    def test_fill_returns_data(self) -> None:
        """Successfully fills a gap with data from Binance Global."""
        gap = GapRecord(
            symbol="BTCUSDT",
            environment="crypto",
            gap_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            gap_end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC),
            gap_duration=timedelta(hours=24),
            filled=False,
            source="detected",
        )

        mock_klines = [
            [
                1704081600000,
                "42000.0",
                "42100.0",
                "41900.0",
                "42050.0",
                "1.5",
                1704096000000,
                "63000.0",
                100,
                "0.8",
                "33600.0",
                "0",
            ],
        ]

        with patch(
            "swingrl.data.gap_fill._fetch_binance_global_klines",
            side_effect=[mock_klines, []],
        ):
            result = fill_crypto_gap(gap)

        assert len(result) == 1
        assert "close" in result.columns

    def test_fill_handles_api_failure(self) -> None:
        """Returns empty DataFrame when Binance Global API fails."""
        from swingrl.utils.exceptions import DataError

        gap = GapRecord(
            symbol="BTCUSDT",
            environment="crypto",
            gap_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            gap_end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC),
            gap_duration=timedelta(hours=24),
            filled=False,
            source="detected",
        )

        with patch(
            "swingrl.data.gap_fill._fetch_binance_global_klines",
            side_effect=DataError("API failed"),
        ):
            result = fill_crypto_gap(gap)

        assert result.empty
