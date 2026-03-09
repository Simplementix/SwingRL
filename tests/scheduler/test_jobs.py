"""Tests for scheduler job functions.

PAPER-12, PAPER-16: Job functions wrap execute_cycle with halt checks,
error handling, and post-cycle callbacks.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from swingrl.scheduler.halt_check import init_emergency_flags, set_halt
from swingrl.scheduler.jobs import (
    JobContext,
    crypto_cycle,
    daily_summary_job,
    equity_cycle,
    init_job_context,
    monthly_macro_job,
    stuck_agent_check_job,
    weekly_fundamentals_job,
)


@pytest.fixture
def mock_db(tmp_path: Path) -> MagicMock:
    """Create a mock DatabaseManager backed by a real SQLite file."""
    db_path = tmp_path / "test_ops.db"
    db = MagicMock()

    def _sqlite_ctx() -> Any:
        """Context manager yielding a real SQLite connection."""

        @contextmanager
        def _ctx() -> Generator[sqlite3.Connection, None, None]:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return _ctx()

    db.sqlite = _sqlite_ctx
    return db


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock ExecutionPipeline."""
    pipeline = MagicMock()
    pipeline.execute_cycle.return_value = []
    return pipeline


@pytest.fixture
def mock_alerter() -> MagicMock:
    """Create a mock Alerter."""
    return MagicMock()


@pytest.fixture
def job_ctx(
    mock_db: MagicMock, mock_pipeline: MagicMock, mock_alerter: MagicMock, loaded_config: Any
) -> JobContext:
    """Initialize JobContext and return it."""
    ctx = init_job_context(
        config=loaded_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
    )
    return ctx


class TestEquityCycle:
    """equity_cycle wraps execute_cycle with halt check and error handling."""

    def test_returns_fills_on_success(self, job_ctx: JobContext, mock_pipeline: MagicMock) -> None:
        """PAPER-12: equity_cycle returns fills from pipeline."""
        mock_fill = MagicMock()
        mock_pipeline.execute_cycle.return_value = [mock_fill]
        fills = equity_cycle()
        assert fills == [mock_fill]
        mock_pipeline.execute_cycle.assert_called_once_with("equity")

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: equity_cycle returns empty when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        fills = equity_cycle()
        assert fills == []

    def test_catches_exception_returns_empty(
        self, job_ctx: JobContext, mock_pipeline: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """PAPER-12: equity_cycle catches exceptions and sends critical alert."""
        mock_pipeline.execute_cycle.side_effect = RuntimeError("broker down")
        fills = equity_cycle()
        assert fills == []
        mock_alerter.send_alert.assert_called()

    def test_pings_healthcheck_on_success(
        self, job_ctx: JobContext, mock_pipeline: MagicMock
    ) -> None:
        """PAPER-16: equity_cycle pings healthcheck after successful cycle."""
        mock_pipeline.execute_cycle.return_value = []
        with patch("swingrl.scheduler.jobs.ping_healthcheck") as mock_ping:
            equity_cycle()
            mock_ping.assert_called_once()


class TestCryptoCycle:
    """crypto_cycle wraps execute_cycle with halt check and error handling."""

    def test_returns_fills_on_success(self, job_ctx: JobContext, mock_pipeline: MagicMock) -> None:
        """PAPER-12: crypto_cycle returns fills from pipeline."""
        mock_fill = MagicMock()
        mock_pipeline.execute_cycle.return_value = [mock_fill]
        fills = crypto_cycle()
        assert fills == [mock_fill]
        mock_pipeline.execute_cycle.assert_called_once_with("crypto")

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: crypto_cycle returns empty when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        fills = crypto_cycle()
        assert fills == []

    def test_catches_exception_returns_empty(
        self, job_ctx: JobContext, mock_pipeline: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """PAPER-12: crypto_cycle catches exceptions and sends critical alert."""
        mock_pipeline.execute_cycle.side_effect = RuntimeError("API timeout")
        fills = crypto_cycle()
        assert fills == []
        mock_alerter.send_alert.assert_called()


class TestDailySummaryJob:
    """daily_summary_job queries portfolio_snapshots and sends summary."""

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: daily_summary_job skips when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        daily_summary_job()
        # Should not crash

    def test_sends_summary_with_data(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """daily_summary_job queries DB and calls alerter."""
        # Seed portfolio_snapshots table
        init_emergency_flags(mock_db)
        with mock_db.sqlite() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    equity_value REAL, crypto_value REAL, cash_balance REAL,
                    high_water_mark REAL, daily_pnl REAL, drawdown_pct REAL,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute(
                "INSERT INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("2026-03-09T12:00:00Z", "equity", 400.0, 300.0, 0.0, 100.0, 400.0, -5.0, 0.01),
            )
        daily_summary_job()
        # daily_summary_job now uses send_embed via build_daily_summary_embed
        mock_alerter.send_embed.assert_called()


class TestStuckAgentCheckJob:
    """stuck_agent_check_job detects consecutive all-cash cycles."""

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: stuck_agent_check_job skips when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        stuck_agent_check_job()

    def test_detects_stuck_equity(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """stuck_agent_check sends alert when all-cash for 10 equity cycles."""
        init_emergency_flags(mock_db)
        with mock_db.sqlite() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    equity_value REAL, crypto_value REAL, cash_balance REAL,
                    high_water_mark REAL, daily_pnl REAL, drawdown_pct REAL,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            # Insert 10 all-cash equity snapshots
            for i in range(10):
                conn.execute(
                    "INSERT INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"2026-03-0{i % 9 + 1}T12:00:0{i}Z",
                        "equity",
                        400.0,
                        0.0,
                        0.0,
                        400.0,
                        400.0,
                        0.0,
                        0.0,
                    ),
                )
        stuck_agent_check_job()
        mock_alerter.send_alert.assert_called()


class TestWeeklyFundamentalsJob:
    """weekly_fundamentals_job runs data refresh."""

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: weekly_fundamentals_job skips when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        weekly_fundamentals_job()

    def test_runs_without_error(self, job_ctx: JobContext) -> None:
        """weekly_fundamentals_job completes without raising."""
        weekly_fundamentals_job()


class TestMonthlyMacroJob:
    """monthly_macro_job runs FRED macro refresh."""

    def test_skips_when_halted(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """PAPER-12: monthly_macro_job skips when halted."""
        set_halt(mock_db, reason="test halt", set_by="test")
        monthly_macro_job()

    def test_runs_without_error(self, job_ctx: JobContext) -> None:
        """monthly_macro_job completes without raising."""
        monthly_macro_job()
