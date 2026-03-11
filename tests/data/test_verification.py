"""Tests for data verification module — DuckDB quality gates.

Covers dataclasses, row checks, date range checks, write_report, print_summary,
crypto gap detection, observation vector validation, and the run_verification aggregator.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

from swingrl.data.verification import (
    CheckResult,
    VerificationResult,
    print_summary,
    run_verification,
    write_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_in_memory_db() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with required tables."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            adjusted_close DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_4h (
            symbol TEXT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, datetime)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            date DATE NOT NULL,
            series_id TEXT NOT NULL,
            value DOUBLE,
            PRIMARY KEY (date, series_id)
        )
    """)
    return conn


def _insert_equity_rows(conn: duckdb.DuckDBPyConnection, symbol: str, count: int) -> None:
    """Insert `count` rows for `symbol` into ohlcv_daily."""
    from datetime import date, timedelta

    base = date(2020, 1, 1)
    for i in range(count):
        d = (base + timedelta(days=i)).isoformat()
        conn.execute(
            "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [symbol, d, 100.0, 101.0, 99.0, 100.5, 1000000],
        )


def _insert_crypto_rows(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    start: datetime,
    interval_hours: int,
    count: int,
) -> None:
    """Insert `count` rows for `symbol` at `interval_hours` spacing into ohlcv_4h."""
    from datetime import timedelta

    ts = start
    for _ in range(count):
        conn.execute(
            "INSERT INTO ohlcv_4h (symbol, datetime, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [symbol, ts.strftime("%Y-%m-%d %H:%M:%S"), 42000.0, 42100.0, 41900.0, 42050.0, 1.5],
        )
        ts += timedelta(hours=interval_hours)


# ---------------------------------------------------------------------------
# Task 1: Dataclasses and row checks
# ---------------------------------------------------------------------------


class TestDataclasses:
    """DATA-04: VerificationResult and CheckResult dataclasses."""

    def test_check_result_can_be_constructed(self) -> None:
        """DATA-04: CheckResult accepts name, passed, detail fields."""
        result = CheckResult(name="test_check", passed=True, detail="all good")
        assert result.name == "test_check"
        assert result.passed is True
        assert result.detail == "all good"

    def test_check_result_serializes_to_dict(self) -> None:
        """DATA-04: CheckResult can be serialized to dict via dataclasses.asdict."""
        result = CheckResult(name="test_check", passed=False, detail="missing symbol")
        d = dataclasses.asdict(result)
        assert d == {"name": "test_check", "passed": False, "detail": "missing symbol"}

    def test_verification_result_can_be_constructed(self) -> None:
        """DATA-04: VerificationResult accepts passed bool and list of CheckResult."""
        checks = [
            CheckResult(name="check_a", passed=True, detail="ok"),
            CheckResult(name="check_b", passed=False, detail="fail"),
        ]
        result = VerificationResult(passed=False, checks=checks)
        assert result.passed is False
        assert len(result.checks) == 2

    def test_verification_result_serializes_to_dict(self) -> None:
        """DATA-04: VerificationResult serializes fully via dataclasses.asdict."""
        checks = [CheckResult(name="c1", passed=True, detail="detail")]
        result = VerificationResult(passed=True, checks=checks)
        d = dataclasses.asdict(result)
        assert "passed" in d
        assert "checks" in d
        assert d["checks"][0]["name"] == "c1"


class TestCheckEquityRows:
    """DATA-04: _check_equity_rows detects missing/present equity symbols."""

    def test_missing_equity_symbol_fails(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=False when configured symbol has no rows."""
        from swingrl.data.verification import _check_equity_rows

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        # Insert rows for only one symbol (SPY), QQQ is missing
        _insert_equity_rows(conn, "SPY", 150)
        result = _check_equity_rows(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is False
        assert "QQQ" in result.detail or "missing" in result.detail.lower()
        cursor.close()
        conn.close()

    def test_all_equity_symbols_present_passes(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=True when all configured equity symbols have >100 rows."""
        from swingrl.data.verification import _check_equity_rows

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        # loaded_config has symbols: [SPY, QQQ]
        _insert_equity_rows(conn, "SPY", 150)
        _insert_equity_rows(conn, "QQQ", 150)
        result = _check_equity_rows(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is True
        cursor.close()
        conn.close()


class TestCheckCryptoRows:
    """DATA-04: _check_crypto_rows detects missing crypto symbols."""

    def test_missing_crypto_symbol_fails(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=False when ohlcv_4h is missing a configured crypto symbol."""
        from swingrl.data.verification import _check_crypto_rows

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        # Insert BTCUSDT only — ETHUSDT missing
        _insert_crypto_rows(conn, "BTCUSDT", start, 4, 150)
        result = _check_crypto_rows(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is False
        assert "ETHUSDT" in result.detail or "missing" in result.detail.lower()
        cursor.close()
        conn.close()


class TestCheckMacroSeries:
    """DATA-04: _check_macro_series detects missing FRED series."""

    def test_missing_fred_series_fails(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=False when a configured FRED series is absent."""
        from swingrl.data.verification import _check_macro_series

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        # Insert only one series, others missing
        conn.execute(
            "INSERT INTO macro_features (date, series_id, value) VALUES ('2024-01-01', 'VIXCLS', 20.5)"
        )
        result = _check_macro_series(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is False
        cursor.close()
        conn.close()


class TestWriteReport:
    """DATA-04: write_report produces valid JSON at specified path."""

    def test_write_report_creates_valid_json(self, tmp_path: Path) -> None:
        """DATA-04: write_report writes JSON-parseable file with correct structure."""
        checks = [
            CheckResult(name="equity_rows", passed=True, detail="all present"),
            CheckResult(name="crypto_rows", passed=False, detail="ETHUSDT missing"),
        ]
        result = VerificationResult(passed=False, checks=checks)
        report_path = tmp_path / "reports" / "verification.json"
        write_report(result, report_path)
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "passed" in data
        assert "checks" in data
        assert len(data["checks"]) == 2

    def test_write_report_creates_parent_dirs(self, tmp_path: Path) -> None:
        """DATA-04: write_report creates parent directories automatically."""
        result = VerificationResult(passed=True, checks=[])
        report_path = tmp_path / "nested" / "deep" / "report.json"
        write_report(result, report_path)
        assert report_path.exists()


class TestPrintSummary:
    """DATA-04: print_summary outputs [PASS]/[FAIL] per check to stdout."""

    def test_print_summary_shows_pass_fail(self, capsys: pytest.CaptureFixture[str]) -> None:
        """DATA-04: Console output includes [PASS] and [FAIL] tags per check."""
        checks = [
            CheckResult(name="equity_rows", passed=True, detail="ok"),
            CheckResult(name="crypto_rows", passed=False, detail="missing ETHUSDT"),
        ]
        result = VerificationResult(passed=False, checks=checks)
        print_summary(result)
        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "[FAIL]" in captured.out
        assert "equity_rows" in captured.out
        assert "crypto_rows" in captured.out

    def test_print_summary_overall_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        """DATA-04: print_summary outputs overall PASS/FAIL at end."""
        checks = [CheckResult(name="check", passed=True, detail="ok")]
        result = VerificationResult(passed=True, checks=checks)
        print_summary(result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# Task 2: Crypto gap detection, obs vector checks, and run_verification
# ---------------------------------------------------------------------------


class TestCheckCryptoGaps:
    """DATA-04: _check_crypto_gaps detects gaps >8h in ohlcv_4h timestamps."""

    def test_no_gaps_passes(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=True when all consecutive timestamps are <=8h apart."""
        from swingrl.data.verification import _check_crypto_gaps

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            _insert_crypto_rows(conn, symbol, start, 4, 50)
        result = _check_crypto_gaps(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is True
        assert "no gaps" in result.detail.lower()
        cursor.close()
        conn.close()

    def test_gap_over_8h_fails(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=False when ohlcv_4h has a gap >8h for any symbol."""
        from swingrl.data.verification import _check_crypto_gaps

        conn = _make_in_memory_db()
        cursor = conn.cursor()
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        # Insert BTC with normal spacing
        _insert_crypto_rows(conn, "BTCUSDT", start, 4, 50)
        # Insert ETH with a 12h gap at position 10
        from datetime import timedelta

        ts = start
        for i in range(50):
            gap_hours = 12 if i == 10 else 4
            conn.execute(
                "INSERT INTO ohlcv_4h (symbol, datetime, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "ETHUSDT",
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    2500.0,
                    2510.0,
                    2490.0,
                    2505.0,
                    0.5,
                ],
            )
            ts += timedelta(hours=gap_hours)
        result = _check_crypto_gaps(cursor, loaded_config)  # type: ignore[arg-type]
        assert result.passed is False
        assert "ETHUSDT" in result.detail or "gap" in result.detail.lower()
        cursor.close()
        conn.close()


class TestCheckObsVector:
    """DATA-04: _check_obs_vector validates NaN and shape of observation vectors."""

    def test_nan_in_obs_vector_fails(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=False when observation vector contains NaN."""
        from swingrl.data.verification import _check_obs_vector

        pipeline = MagicMock()
        pipeline.get_observation.return_value = np.array(
            [1.0, 2.0, np.nan, 4.0] * 39,  # 156 elements with a NaN
            dtype=np.float32,
        )
        result = _check_obs_vector(pipeline, "equity", "2024-01-15")
        assert result.passed is False
        assert "nan" in result.detail.lower()

    def test_clean_equity_obs_vector_passes(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=True for clean 156-dim equity observation vector."""
        from swingrl.data.verification import _check_obs_vector

        pipeline = MagicMock()
        pipeline.get_observation.return_value = np.ones(156, dtype=np.float32)
        result = _check_obs_vector(pipeline, "equity", "2024-01-15")
        assert result.passed is True
        assert "156" in result.detail

    def test_clean_crypto_obs_vector_passes(self, loaded_config: object) -> None:
        """DATA-04: Returns passed=True for clean 45-dim crypto observation vector."""
        from swingrl.data.verification import _check_obs_vector

        pipeline = MagicMock()
        pipeline.get_observation.return_value = np.ones(45, dtype=np.float32)
        result = _check_obs_vector(pipeline, "crypto", "2024-01-15")
        assert result.passed is True
        assert "45" in result.detail


class TestRunVerification:
    """DATA-04: run_verification aggregates all checks into a VerificationResult."""

    def test_run_verification_fails_when_any_check_fails(
        self, loaded_config: object, tmp_path: Path
    ) -> None:
        """DATA-04: Returns VerificationResult with passed=False if any check fails."""
        with (
            patch("swingrl.data.verification.DatabaseManager") as mock_db_cls,
            patch("swingrl.data.verification.FeaturePipeline") as mock_pipeline_cls,
        ):
            # Mock DatabaseManager context manager
            mock_cursor = MagicMock()
            mock_cursor.execute.return_value.fetchall.return_value = []
            mock_cursor.execute.return_value.fetchdf.return_value = __import__("pandas").DataFrame()
            mock_db = MagicMock()
            mock_db.duckdb.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_db.duckdb.return_value.__exit__ = MagicMock(return_value=False)
            mock_db_cls.return_value = mock_db

            # Mock FeaturePipeline to return NaN vector (will fail obs check)
            mock_pipeline = MagicMock()
            mock_pipeline.get_observation.return_value = np.full(156, np.nan, dtype=np.float32)
            mock_pipeline_cls.return_value = mock_pipeline

            result = run_verification(loaded_config)  # type: ignore[arg-type]
            # At least one check should fail (empty DB means no rows)
            assert isinstance(result, VerificationResult)
            assert result.passed is False

    def test_run_verification_passes_when_all_checks_pass(
        self, loaded_config: object, tmp_path: Path
    ) -> None:
        """DATA-04: Returns passed=True when all individual checks pass."""
        with (
            patch("swingrl.data.verification.DatabaseManager") as mock_db_cls,
            patch("swingrl.data.verification.FeaturePipeline") as mock_pipeline_cls,
        ):
            # Build a cursor mock that returns adequate data for all checks
            mock_cursor = _build_passing_cursor_mock(loaded_config)
            mock_db = MagicMock()
            mock_db.duckdb.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_db.duckdb.return_value.__exit__ = MagicMock(return_value=False)
            mock_db_cls.return_value = mock_db

            # Pipeline returns clean observation vectors
            mock_pipeline = MagicMock()
            mock_pipeline.get_observation.return_value = np.ones(156, dtype=np.float32)
            mock_pipeline_cls.return_value = mock_pipeline

            result = run_verification(loaded_config)  # type: ignore[arg-type]
            assert isinstance(result, VerificationResult)
            # All checks should pass — result.passed = True
            all_passed = all(c.passed for c in result.checks)
            assert all_passed is True
            assert result.passed is True


# ---------------------------------------------------------------------------
# Helper for building a passing cursor mock
# ---------------------------------------------------------------------------


def _build_passing_cursor_mock(config: object) -> MagicMock:
    """Build a cursor mock that returns data sufficient to pass all checks."""

    from swingrl.config.schema import SwingRLConfig

    cfg: SwingRLConfig = config  # type: ignore[assignment]

    cursor = MagicMock()

    # We'll use side_effect to return different data based on the SQL
    equity_symbols = cfg.equity.symbols
    crypto_symbols = cfg.crypto.symbols
    fred_series = cfg.fred.series

    equity_rows = [(sym, 150) for sym in equity_symbols]
    crypto_rows = [(sym, 150) for sym in crypto_symbols]
    macro_rows = [(sid,) for sid in fred_series]

    # Build timestamp pairs for crypto gaps (all 4h apart — no gaps)
    from datetime import timedelta

    base_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts_rows_btc = [(base_ts + timedelta(hours=4 * i),) for i in range(50)]
    ts_rows_eth = [(base_ts + timedelta(hours=4 * i),) for i in range(50)]

    # equity min/max date
    equity_date_row = [("2020-01-01", "2024-01-01")]
    crypto_date_row = [("2020-01-01 00:00:00", "2024-01-15 00:00:00")]

    # Craft side_effect based on SQL patterns
    def execute_side_effect(query: str, params: list | None = None) -> MagicMock:
        result_mock = MagicMock()
        q = query.strip().lower()

        if "group by symbol" in q and "ohlcv_daily" in q:
            result_mock.fetchall.return_value = equity_rows
        elif "group by symbol" in q and "ohlcv_4h" in q:
            result_mock.fetchall.return_value = crypto_rows
        elif "distinct series_id" in q:
            result_mock.fetchall.return_value = macro_rows
        elif "min(date)" in q and "ohlcv_daily" in q:
            result_mock.fetchall.return_value = equity_date_row
        elif "min(" in q and "ohlcv_4h" in q and "datetime" in q and "symbol" not in (params or []):
            result_mock.fetchall.return_value = crypto_date_row
        elif "from ohlcv_4h where symbol" in q and "order by" in q:
            symbol = params[0] if params else "BTCUSDT"
            ts_data = ts_rows_btc if "BTC" in symbol.upper() else ts_rows_eth
            result_mock.fetchall.return_value = ts_data
        elif "max(" in q:
            result_mock.fetchone.return_value = ("2024-01-15",)
            result_mock.fetchall.return_value = [("2024-01-15",)]
        else:
            result_mock.fetchall.return_value = []
            result_mock.fetchone.return_value = None

        return result_mock

    cursor.execute.side_effect = execute_side_effect
    return cursor
