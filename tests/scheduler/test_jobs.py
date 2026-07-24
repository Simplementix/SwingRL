"""Tests for scheduler job functions.

PAPER-12, PAPER-16: Job functions wrap execute_cycle with halt checks,
error handling, and post-cycle callbacks.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from alpaca.trading.enums import OrderStatus
from psycopg.rows import dict_row

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.scheduler.halt_check import init_emergency_flags, set_halt
from swingrl.scheduler.jobs import (
    JobContext,
    _benchmark_value,
    crypto_cycle,
    daily_summary_job,
    equity_cycle,
    init_job_context,
    monthly_macro_job,
    risk_sweep_job,
    stuck_agent_check_job,
    weekly_fundamentals_job,
)


@pytest.fixture(autouse=True)
def _clean_test_tables() -> None:
    """Truncate test-affected tables before each test to avoid cross-test pollution."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return
    conn = psycopg.connect(db_url, autocommit=True)
    for _table in ("emergency_flags", "portfolio_snapshots"):
        try:
            # conn.execute(f"DELETE FROM {_table}")  # nosec B608
            pass
        except Exception:
            pass  # Table may not exist yet
    conn.close()


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock DatabaseManager backed by a real PostgreSQL connection."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set")
    db = MagicMock()

    def _pg_ctx() -> Any:
        """Context manager yielding a real PostgreSQL connection."""

        @contextmanager
        def _ctx() -> Generator[Any, None, None]:
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

    db.connection = _pg_ctx
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
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION,
                    high_water_mark DOUBLE PRECISION, daily_pnl DOUBLE PRECISION,
                    drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute(
                "INSERT INTO portfolio_snapshots VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                ("2026-03-09T12:00:00Z", "equity", 400.0, 300.0, 0.0, 100.0, 400.0, -5.0, 0.01),
            )
        daily_summary_job()
        # daily_summary_job now uses send_embed via build_daily_summary_embed
        mock_alerter.send_embed.assert_called()

    def test_counts_signal_trades_today(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D5: digest counts today's (ET) signal trades per env from the trades
        table — never the hardcoded zeros (found live 2026-07-21)."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION,
                    high_water_mark DOUBLE PRECISION, daily_pnl DOUBLE PRECISION,
                    drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    commission DOUBLE PRECISION DEFAULT 0.0,
                    slippage DOUBLE PRECISION DEFAULT 0.0,
                    environment TEXT NOT NULL,
                    broker TEXT, order_type TEXT, trade_type TEXT
                )
            """)
            # Isolate from any cross-test rows so counts are deterministic.
            conn.execute("DELETE FROM portfolio_snapshots")
            conn.execute("DELETE FROM trades")
            # One snapshot per env so build_daily_summary_embed renders BOTH
            # "Equity Trades" and "Crypto Trades" fields.
            # Same timestamp for both envs is fine — PK is (timestamp, environment).
            # Must be a valid TIMESTAMPTZ literal: the migrated schema's timestamp
            # column is TIMESTAMPTZ, so a "...Z-equity" suffix would fail to parse.
            for env in ("equity", "crypto"):
                conn.execute(
                    "INSERT INTO portfolio_snapshots VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT DO NOTHING",
                    ("2026-03-09T12:00:00Z", env, 400.0, 300.0, 0.0, 100.0, 400.0, 0.0, 0.0),
                )
            insert = (
                "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, price, "
                "environment, trade_type) VALUES (%s, {ts}, %s, %s, %s, %s, %s, %s)"
            )
            # 2 crypto signal trades TODAY -> counted.
            conn.execute(
                insert.format(ts="now()"),
                ("c1", "BTCUSDT", "buy", 0.001, 40000.0, "crypto", "signal"),
            )
            conn.execute(
                insert.format(ts="now()"),
                ("c2", "ETHUSDT", "buy", 0.01, 3000.0, "crypto", "signal"),
            )
            # Decoy: crypto NON-signal today -> excluded by trade_type filter.
            conn.execute(
                insert.format(ts="now()"),
                ("c3", "BTCUSDT", "sell", 0.001, 40000.0, "crypto", "rebalance"),
            )
            # Decoy: crypto signal from a PRIOR ET day -> excluded by date filter.
            conn.execute(
                insert.format(ts="now() - interval '2 days'"),
                ("c4", "BTCUSDT", "buy", 0.001, 40000.0, "crypto", "signal"),
            )
            # Equity: no trades today -> count 0.
        daily_summary_job()

        embed = mock_alerter.send_embed.call_args.args[1]
        fields = {f["name"]: f["value"] for f in embed["embeds"][0]["fields"]}
        assert fields["Crypto Trades"] == "2"
        assert fields["Equity Trades"] == "0"

    def test_latest_per_env_keeps_equity_and_newest_crypto(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: with an OLDER equity row and TWO crypto rows, the digest keeps the
        equity section AND the NEWEST crypto snapshot (not the older one)."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
            rows = [
                ("2026-07-23T13:15:00Z", "equity", 402.0, 300.0, 0.0, 100.0, 402.0, 2.0, 0.0),
                ("2026-07-23T16:05:00Z", "crypto", 48.50, 0.0, 48.5, 0.0, 48.5, 0.5, 0.0),
                ("2026-07-23T20:05:00Z", "crypto", 47.42, 0.0, 47.42, 0.0, 48.5, -1.08, 0.0),
            ]
            for r in rows:
                conn.execute(
                    "INSERT INTO portfolio_snapshots VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT DO NOTHING",
                    r,
                )
        daily_summary_job()
        embed = mock_alerter.send_embed.call_args.args[1]
        names = [f["name"] for f in embed["embeds"][0]["fields"]]
        values = {f["name"]: f["value"] for f in embed["embeds"][0]["fields"]}
        assert any(n.startswith("Equity Value") for n in names)  # equity NOT dropped
        assert "$47.42" in values[next(n for n in names if n.startswith("Crypto Value"))]

    def test_crypto_only_db_omits_equity_without_crashing(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: a DB with only crypto snapshots renders crypto, omits equity, no crash."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
            conn.execute(
                "INSERT INTO portfolio_snapshots VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT DO NOTHING",
                ("2026-07-23T16:05:00Z", "crypto", 48.5, 0.0, 48.5, 0.0, 48.5, 0.5, 0.0),
            )
        daily_summary_job()
        embed = mock_alerter.send_embed.call_args.args[1]
        names = [f["name"] for f in embed["embeds"][0]["fields"]]
        assert not any(n.startswith("Equity Value") for n in names)
        assert any(n.startswith("Crypto Value") for n in names)

    def test_empty_snapshots_returns_early(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: an empty portfolio_snapshots table hits the `if not rows` early return."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
        daily_summary_job()
        mock_alerter.send_embed.assert_not_called()


def _seed_benchmark_tables(conn: Any) -> None:
    """Create + isolate benchmark_baselines and ohlcv_4h (IF NOT EXISTS, then DELETE).

    Mirrors the digest tests' self-contained table setup so the benchmark tests run
    against either the migrated schema (V010) or a bare scratch DB.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_baselines (
            environment TEXT NOT NULL,
            symbol TEXT NOT NULL,
            baseline_date DATE NOT NULL,
            baseline_price DOUBLE PRECISION NOT NULL,
            capital_usd DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (environment, symbol)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_4h (
            symbol TEXT NOT NULL,
            datetime TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
            close DOUBLE PRECISION, volume DOUBLE PRECISION, source TEXT,
            fetched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, datetime)
        )
    """)
    conn.execute("DELETE FROM benchmark_baselines WHERE environment = 'crypto'")
    conn.execute("DELETE FROM ohlcv_4h WHERE symbol IN ('BTCUSDT', 'ETHUSDT')")


class TestBenchmarkValue:
    """BENCH-D13: _benchmark_value grows each equal-weight baseline slice to latest close."""

    def test_benchmark_value_equal_weight(self, job_ctx: JobContext, mock_db: MagicMock) -> None:
        """BENCH-D13: benchmark = equal-weight capital split grown by close/baseline per
        symbol. 47 capital, 2 symbols; BTC +10%, ETH -10% -> exactly 47.0.

        (Reconciled: the brief sketched a ``mock_ctx.set_baselines`` helper; the real
        digest scaffolding in this module is DB-backed, so this seeds the same numbers
        into benchmark_baselines + ohlcv_4h — the assertion contract is unchanged.)
        """
        with mock_db.connection() as conn:
            _seed_benchmark_tables(conn)
            for symbol, baseline_price in (("BTCUSDT", 60000.0), ("ETHUSDT", 2000.0)):
                conn.execute(
                    "INSERT INTO benchmark_baselines"
                    " (environment, symbol, baseline_date, baseline_price, capital_usd)"
                    " VALUES ('crypto', %s, '2026-07-22', %s, 47.0)",
                    (symbol, baseline_price),
                )
            for symbol, close in (("BTCUSDT", 66000.0), ("ETHUSDT", 1800.0)):
                conn.execute(
                    "INSERT INTO ohlcv_4h (symbol, datetime, close) VALUES (%s, now(), %s)",
                    (symbol, close),
                )
            value = _benchmark_value(conn, "crypto")
        assert value == pytest.approx(47.0)

    def test_benchmark_value_none_when_no_baselines(
        self, job_ctx: JobContext, mock_db: MagicMock
    ) -> None:
        """BENCH-D13: no baselines for the env -> None (digest omits the fields)."""
        with mock_db.connection() as conn:
            _seed_benchmark_tables(conn)
            value = _benchmark_value(conn, "crypto")
        assert value is None

    def test_benchmark_value_uses_latest_close(
        self, job_ctx: JobContext, mock_db: MagicMock
    ) -> None:
        """BENCH-D13: the newest bar (max datetime) is the current price, not an older one."""
        with mock_db.connection() as conn:
            _seed_benchmark_tables(conn)
            conn.execute(
                "INSERT INTO benchmark_baselines"
                " (environment, symbol, baseline_date, baseline_price, capital_usd)"
                " VALUES ('crypto', 'BTCUSDT', '2026-07-22', 100.0, 47.0)"
            )
            conn.execute(
                "INSERT INTO ohlcv_4h (symbol, datetime, close)"
                " VALUES ('BTCUSDT', now() - interval '8 hours', 100.0)"
            )
            conn.execute(
                "INSERT INTO ohlcv_4h (symbol, datetime, close) VALUES ('BTCUSDT', now(), 200.0)"
            )
            value = _benchmark_value(conn, "crypto")
        # single symbol: 47 capital * (200 latest / 100 baseline) = 94.0
        assert value == pytest.approx(94.0)


class TestDailySummaryDigestFlush:
    """A30 Discord wiring (Task E): daily_summary_job flushes the buffered INFO digest."""

    def test_flushes_info_digest_even_when_halted(self) -> None:
        """A30: send_daily_digest had zero callers; the EOD job must flush it.

        The flush runs before the halt gate so INFO buffered on a halted day still
        reaches Discord instead of dying in memory on the next restart. Built on plain
        mocks so it runs without a database (DATABASE_URL-independent).
        """
        alerter = MagicMock()
        init_job_context(config=MagicMock(), db=MagicMock(), pipeline=MagicMock(), alerter=alerter)
        with patch("swingrl.scheduler.jobs.is_halted", return_value=True):
            daily_summary_job()
        alerter.send_daily_digest.assert_called_once()


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
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION,
                    high_water_mark DOUBLE PRECISION, daily_pnl DOUBLE PRECISION,
                    drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            # Insert 10 all-cash equity snapshots
            for i in range(10):
                conn.execute(
                    "INSERT INTO portfolio_snapshots VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT DO NOTHING",
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


class TestReconciliationJob:
    """reconciliation_job wraps PositionReconciler with halt check and failure tracking."""

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_reconciliation_success(self) -> None:
        """PAPER-09: reconciliation_job calls reconcile('equity') on success path."""
        from swingrl.scheduler.jobs import init_job_context, reconciliation_job

        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        init_job_context(
            config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
        )

        mock_reconciler = MagicMock()
        mock_reconciler.reconcile.return_value = []

        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch(
                "swingrl.execution.reconciliation.PositionReconciler",
                return_value=mock_reconciler,
            ):
                with patch(
                    "swingrl.execution.adapters.alpaca_adapter.AlpacaAdapter"
                ) as mock_adapter_cls:
                    mock_adapter_cls.return_value = MagicMock()
                    reconciliation_job()

        mock_reconciler.reconcile.assert_called_once_with("equity")

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_skips_when_halted(self) -> None:
        """PAPER-09: reconciliation_job skips reconcile when halt flag is active."""
        from swingrl.scheduler.jobs import init_job_context, reconciliation_job

        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        init_job_context(
            config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
        )

        mock_reconciler = MagicMock()

        with patch("swingrl.scheduler.jobs.is_halted", return_value=True):
            with patch(
                "swingrl.execution.reconciliation.PositionReconciler",
                return_value=mock_reconciler,
            ):
                reconciliation_job()

        mock_reconciler.reconcile.assert_not_called()

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    @patch("swingrl.scheduler.jobs._reconciliation_failures", new=0)
    def test_consecutive_failures_escalate(self) -> None:
        """PAPER-09: 3+ consecutive failures send critical alert; success resets counter."""
        import swingrl.scheduler.jobs as jobs_module
        from swingrl.scheduler.jobs import init_job_context, reconciliation_job

        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        init_job_context(
            config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
        )

        # Reset failure counter
        jobs_module._reconciliation_failures = 0

        def raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("broker connection refused")

        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch(
                "swingrl.execution.adapters.alpaca_adapter.AlpacaAdapter"
            ) as mock_adapter_cls:
                mock_adapter_cls.return_value = MagicMock()
                with patch(
                    "swingrl.execution.reconciliation.PositionReconciler"
                ) as mock_reconciler_cls:
                    mock_reconciler_cls.return_value.reconcile.side_effect = raise_error

                    # First failure: warning
                    reconciliation_job()
                    first_alert_level = mock_alerter.send_alert.call_args_list[-1].args[0]
                    assert first_alert_level == "warning"

                    # Second failure: still warning
                    reconciliation_job()
                    second_alert_level = mock_alerter.send_alert.call_args_list[-1].args[0]
                    assert second_alert_level == "warning"

                    # Third failure: escalates to critical
                    reconciliation_job()
                    third_alert_level = mock_alerter.send_alert.call_args_list[-1].args[0]
                    assert third_alert_level == "critical"

                    # Now simulate success — counter resets
                    mock_reconciler_cls.return_value.reconcile.side_effect = None
                    mock_reconciler_cls.return_value.reconcile.return_value = []
                    reconciliation_job()

        assert jobs_module._reconciliation_failures == 0


class TestFredImportPath:
    """Verify FRED jobs import from correct module path and use correct API."""

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_weekly_fundamentals_imports_fred_correctly(self) -> None:
        """FEAT-11: weekly_fundamentals_job imports FREDIngestor from swingrl.data.fred."""
        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        init_job_context(
            config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
        )

        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch("swingrl.data.fred.FREDIngestor") as mock_fred_cls:
                mock_ingestor = MagicMock()
                mock_fred_cls.return_value = mock_ingestor

                weekly_fundamentals_job()

                # FREDIngestor constructed with only config (not config + db)
                mock_fred_cls.assert_called_once_with(mock_config)
                # run_all() called (not refresh())
                mock_ingestor.run_all.assert_called_once()

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_monthly_macro_imports_fred_correctly(self) -> None:
        """FEAT-11: monthly_macro_job imports FREDIngestor from swingrl.data.fred."""
        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        init_job_context(
            config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
        )

        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch("swingrl.data.fred.FREDIngestor") as mock_fred_cls:
                mock_ingestor = MagicMock()
                mock_fred_cls.return_value = mock_ingestor

                monthly_macro_job()

                # FREDIngestor constructed with only config (not config + db)
                mock_fred_cls.assert_called_once_with(mock_config)
                # run_all() called (not refresh())
                mock_ingestor.run_all.assert_called_once()


class _SweepCtx:
    """Adapts the brief's ``mock_ctx`` to the real JobContext -> ExecutionPipeline layout.

    The risk components the sweep touches (position tracker, per-env circuit breakers,
    global breaker, exchange adapters) live on ``ctx.pipeline``, not on ``ctx`` itself.
    The sweep reaches them through the pipeline's public accessors, so this helper hangs
    mocks off a mock pipeline and mirrors the brief's ``set_positions`` ergonomics.
    """

    def __init__(self) -> None:
        """Wire mock risk components onto a mock pipeline (crypto+equity breakers)."""
        self.adapter = MagicMock()
        self.adapter.get_current_price.return_value = 100.0
        self.circuit_breakers: dict[str, MagicMock] = {
            "equity": MagicMock(),
            "crypto": MagicMock(),
        }
        self.global_cb = MagicMock()
        self.tracker = MagicMock()
        self._positions: dict[str, list[dict[str, Any]]] = {"equity": [], "crypto": []}
        self.tracker.get_positions.side_effect = lambda env: self._positions.get(env, [])
        # Numeric returns so the sweep's max()/arithmetic works on real numbers.
        self.tracker.compute_portfolio_value.return_value = 5000.0
        self.tracker.compute_daily_pnl.return_value = -100.0
        self.tracker.get_high_water_mark.return_value = 10000.0
        self.pipeline = MagicMock()
        self.pipeline.position_tracker = self.tracker
        self.pipeline.circuit_breakers = self.circuit_breakers
        self.pipeline.global_cb = self.global_cb
        self.pipeline.get_adapter.return_value = self.adapter
        self.db = MagicMock()

    def set_positions(self, env: str, positions: dict[str, tuple[float, float]]) -> None:
        """Seed held positions for ``env`` as ``{symbol: (quantity, cost_basis)}``."""
        self._positions[env] = [
            {
                "symbol": symbol,
                "quantity": qty,
                "cost_basis": cost,
                "last_price": cost,
                "unrealized_pnl": 0.0,
                "updated_at": None,
            }
            for symbol, (qty, cost) in positions.items()
        ]


@pytest.fixture
def sweep_ctx() -> Generator[_SweepCtx, None, None]:
    """Build a risk-sweep JobContext on a mock pipeline; default ``is_halted`` -> False."""
    ctx = _SweepCtx()
    init_job_context(config=MagicMock(), db=ctx.db, pipeline=ctx.pipeline, alerter=MagicMock())
    with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
        yield ctx


class TestRiskSweepJob:
    """SWEEP-D10: between-cycle risk sweep marks positions + evaluates breakers, no trading."""

    def test_risk_sweep_trips_breaker_between_cycles(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: a crash between cycles is caught by the sweep — marks refresh and the
        drawdown breaker evaluates without any trading."""
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        sweep_ctx.adapter.get_current_price.return_value = 30000.0
        risk_sweep_job()
        assert sweep_ctx.circuit_breakers["crypto"].check_and_update.called
        assert sweep_ctx.adapter.submit_order.call_count == 0

    def test_risk_sweep_writes_no_snapshots(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: sweeps never write portfolio_snapshots (cycle-cadence stays clean for
        daily-P&L baselines).

        ``PositionTracker.record_snapshot`` is the sole writer of that table, so asserting
        it is never called is the faithful adaptation of the brief's
        ``executed_sql_matching("INSERT INTO portfolio_snapshots")`` contract onto the real
        component layout.
        """
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        risk_sweep_job()
        sweep_ctx.tracker.record_snapshot.assert_not_called()

    def test_risk_sweep_skips_when_halted(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: an active halt flag short-circuits the sweep — no marks, no breaker eval."""
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        with patch("swingrl.scheduler.jobs.is_halted", return_value=True):
            risk_sweep_job()
        sweep_ctx.tracker.mark_positions.assert_not_called()
        sweep_ctx.circuit_breakers["crypto"].check_and_update.assert_not_called()
        sweep_ctx.global_cb.check_combined.assert_not_called()

    def test_risk_sweep_skips_env_with_no_positions(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: an env with no held positions gets no per-env breaker eval (nothing can
        draw down), but the global breaker still runs across both envs."""
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        # equity left flat (no positions).
        risk_sweep_job()
        sweep_ctx.circuit_breakers["equity"].check_and_update.assert_not_called()
        sweep_ctx.circuit_breakers["crypto"].check_and_update.assert_called_once()
        sweep_ctx.global_cb.check_combined.assert_called_once()

    def test_risk_sweep_price_fetch_fail_open(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: a per-symbol price-fetch failure is warned and skipped (fail-open) — the
        sweep still marks/evaluates using the stored last_price fallback and never raises."""
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        sweep_ctx.adapter.get_current_price.side_effect = RuntimeError("broker down")
        risk_sweep_job()  # must not raise
        sweep_ctx.circuit_breakers["crypto"].check_and_update.assert_called_once()

    def test_risk_sweep_evaluates_global_breaker_across_envs(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: the global breaker sees BOTH envs' current values (a cash-only env must
        be included or its capital is omitted from the combined drawdown/daily-loss math)."""
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
        risk_sweep_job()
        sweep_ctx.global_cb.check_combined.assert_called_once()
        portfolio_values, daily_pnls = sweep_ctx.global_cb.check_combined.call_args.args
        assert set(portfolio_values.keys()) == {"equity", "crypto"}
        assert set(daily_pnls.keys()) == {"equity", "crypto"}

    def test_risk_sweep_per_env_failure_isolation(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: a failure in one env (e.g. adapter lazy-init) must NOT stop the other —
        env B's per-env breaker still evaluates, so a persistent env-A fault cannot re-open
        env-B's between-cycle blind window, the exact window this job exists to close."""
        sweep_ctx.set_positions("equity", {"SPY": (10.0, 400.0)})
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})

        def _adapter(env: str) -> MagicMock:
            if env == "equity":  # env A (first-iterated) fails at adapter init
                raise RuntimeError("alpaca client init failed")
            return sweep_ctx.adapter

        sweep_ctx.pipeline.get_adapter.side_effect = _adapter

        risk_sweep_job()  # must not raise

        sweep_ctx.circuit_breakers["equity"].check_and_update.assert_not_called()
        sweep_ctx.circuit_breakers["crypto"].check_and_update.assert_called_once()

    def test_risk_sweep_skips_global_when_env_value_missing(self, sweep_ctx: _SweepCtx) -> None:
        """SWEEP-D10: when an env errors BEFORE its value is computed, that value is missing
        from the combined dict — check_combined is SKIPPED (a partial dict would understate
        total value and risk a false global halt) and the skip is warned. A flat env is NOT
        missing: it still contributes its cash value."""
        from structlog.testing import capture_logs

        sweep_ctx.set_positions("equity", {"SPY": (10.0, 400.0)})
        sweep_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})

        def _adapter(env: str) -> MagicMock:
            if env == "equity":
                raise RuntimeError("alpaca client init failed")
            return sweep_ctx.adapter

        sweep_ctx.pipeline.get_adapter.side_effect = _adapter

        with capture_logs() as logs:
            risk_sweep_job()

        sweep_ctx.global_cb.check_combined.assert_not_called()
        assert any(entry.get("event") == "risk_sweep_global_skipped" for entry in logs)


class TestTradeCommentary:
    """Task 12: post-cycle MT commentary skeleton — inert by default (meta_trader.enabled)."""

    def test_noop_when_meta_trader_disabled(self) -> None:
        """Task 12: with meta_trader.enabled=False the skeleton makes NO memory call."""
        from swingrl.scheduler.jobs import init_job_context, maybe_post_trade_commentary

        config = MagicMock()
        config.meta_trader.enabled = False
        ctx = init_job_context(
            config=config, db=MagicMock(), pipeline=MagicMock(), alerter=MagicMock()
        )
        with patch("swingrl.memory.client.MemoryClient") as mock_client_cls:
            maybe_post_trade_commentary(ctx, "equity")
        mock_client_cls.assert_not_called()

    def test_posts_when_enabled_with_cycle(self) -> None:
        """Task 12: when enabled, POSTs the latest cycle's context to /trade/commentary."""
        from contextlib import contextmanager

        from swingrl.scheduler.jobs import init_job_context, maybe_post_trade_commentary

        config = MagicMock()
        config.meta_trader.enabled = True
        config.meta_trader.commentary_provider = "cerebras"
        config.memory_agent.base_url = "http://swingrl-memory:8889"
        config.memory_agent.api_key = ""
        config.memory_agent.timeout_sec = 3.0

        cycle_row = {
            "cycle_id": 77,
            "hmm_p_bull": 0.6,
            "hmm_p_bear": 0.4,
            "vix": 15.0,
            "turbulence": 1.2,
            "deployed_iteration": 5,
        }
        db = MagicMock()

        @contextmanager
        def _conn_ctx():
            conn = MagicMock()
            conn.execute.return_value.fetchone.return_value = cycle_row
            yield conn

        db.connection = _conn_ctx
        ctx = init_job_context(config=config, db=db, pipeline=MagicMock(), alerter=MagicMock())

        with patch("swingrl.memory.client.MemoryClient") as mock_client_cls:
            client = MagicMock()
            mock_client_cls.return_value = client
            maybe_post_trade_commentary(ctx, "equity")

        client.trade_commentary.assert_called_once()
        payload = client.trade_commentary.call_args.args[0]
        assert payload["cycle_id"] == 77
        assert payload["environment"] == "equity"

    def test_equity_cycle_inert_by_default(
        self, job_ctx: JobContext, mock_pipeline: MagicMock
    ) -> None:
        """Task 12: equity_cycle with default config makes no MemoryClient commentary call."""
        mock_pipeline.execute_cycle.return_value = []
        with patch("swingrl.memory.client.MemoryClient") as mock_client_cls:
            equity_cycle()
        mock_client_cls.assert_not_called()


class _MockAlpaca:
    """Stand-in Alpaca equity adapter for the fill-confirmation job (D11).

    ``order_status`` programs a per-order broker status; the confirmation job reads it back
    through the production ``get_order_status(order_id)`` call.
    """

    def __init__(self) -> None:
        self._statuses: dict[str, SimpleNamespace] = {}

    def order_status(self, order_id: str, **fields: Any) -> None:
        """Program the broker order state the job will observe for ``order_id``."""
        self._statuses[order_id] = SimpleNamespace(**fields)

    def get_order_status(self, order_id: str) -> SimpleNamespace:
        """Return the programmed order (default: an un-filled 'new' order)."""
        return self._statuses.get(order_id, SimpleNamespace(status="new"))


class _MockCtx:
    """Adapts the brief's ``mock_ctx`` onto the real JobContext -> ExecutionPipeline layout.

    Wires a real (migrated) DB, a real ExecutionPipeline (mock feature pipeline + mock
    alerter), and a mock Alpaca equity adapter injected into the pipeline's adapter cache
    so ``equity_fill_confirmation_job()`` and the D12 cycle-orders ping run end-to-end.
    """

    def __init__(self, db: DatabaseManager, config: Any, alerter: MagicMock) -> None:
        from swingrl.execution.pipeline import ExecutionPipeline

        self.db = db
        self.alerter = alerter
        self.alpaca = _MockAlpaca()
        self.pipeline = ExecutionPipeline(
            config=config,
            db=db,
            feature_pipeline=MagicMock(),
            alerter=alerter,
            models_dir=Path("/tmp/swingrl_task8_models"),
        )
        # Inject the mock equity adapter so get_adapter("equity") returns it (no broker).
        self.pipeline._adapters["equity"] = self.alpaca
        init_job_context(config=config, db=db, pipeline=self.pipeline, alerter=alerter)
        self._clear()

    def _clear(self) -> None:
        """Isolate the shared test DB — each table dropped in its own transaction so a
        not-yet-migrated table (RED) cannot abort the others."""
        for table in ("pending_orders", "fill_quality", "trades", "positions"):
            try:
                with self.db.connection() as conn:
                    conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed table names
            except Exception:  # noqa: BLE001 — table may not exist yet at RED
                pass

    def set_pending_order(
        self,
        order_id: str,
        cycle_id: int,
        symbol: str,
        side: str,
        decision_price: float | None = None,
    ) -> None:
        """Seed the FK parent inference_cycles row (forced id) + an unresolved pending order.

        ``decision_price`` seeds the 09:15 sizing price on the worklist row so the confirmation
        job can forward it into ``record_fill`` (RULING-3); defaults None for the pre-ruling
        rows the other tests use.
        """
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO inference_cycles (cycle_id, environment, mode, cycle_ts) "
                "OVERRIDING SYSTEM VALUE VALUES (%s, 'equity', 'paper', now()) "
                "ON CONFLICT (cycle_id) DO NOTHING",
                (cycle_id,),
            )
            conn.execute(
                "INSERT INTO pending_orders "
                "(order_id, cycle_id, symbol, side, submitted_at, decision_price) "
                "VALUES (%s, %s, %s, %s, now(), %s)",
                (order_id, cycle_id, symbol, side, decision_price),
            )

    def inserted_trade(self) -> dict[str, Any] | None:
        """Return the most recently inserted trades row."""
        with self.db.connection() as conn:
            return conn.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1").fetchone()

    def pending_row(self, order_id: str) -> dict[str, Any] | None:
        """Return the pending_orders row for ``order_id`` (to assert resolved_at state)."""
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT * FROM pending_orders WHERE order_id = %s", (order_id,)
            ).fetchone()

    def seed_trade(
        self,
        trade_id: str,
        cycle_id: int,
        symbol: str,
        side: str,
        price: float = 600.10,
        quantity: float = 0.0416,
    ) -> None:
        """Seed a trades row (simulates a prior confirmation run that already recorded it)."""
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, price, "
                "commission, slippage, environment, broker, order_type, trade_type, cycle_id) "
                "VALUES (%s, now(), %s, %s, %s, %s, 0.0, 0.0, 'equity', 'alpaca', 'market', "
                "'signal', %s)",
                (trade_id, symbol, side, quantity, price, cycle_id),
            )

    def trade_count(self, trade_id: str) -> int:
        """Return the number of trades rows for ``trade_id`` (duplicate-detection)."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM trades WHERE trade_id = %s", (trade_id,)
            ).fetchone()
        return int(row["n"])

    @staticmethod
    def _to_summaries(orders: list[tuple[Any, ...]]) -> list[Any]:
        """Convert the brief's order tuples into CycleOrderSummary descriptors.

        buy:  (symbol, "buy", notional)
        sell: (symbol, "sell", qty, approx_value)
        """
        from swingrl.execution.types import CycleOrderSummary

        summaries: list[Any] = []
        for order in orders:
            if order[1] == "buy":
                summaries.append(
                    CycleOrderSummary(
                        symbol=order[0], side="buy", notional_usd=float(order[2]), quantity=None
                    )
                )
            else:
                summaries.append(
                    CycleOrderSummary(
                        symbol=order[0],
                        side="sell",
                        notional_usd=float(order[3]),
                        quantity=float(order[2]),
                    )
                )
        return summaries

    def run_equity_cycle(self, orders: list[tuple[Any, ...]]) -> None:
        """Drive the pipeline's D12 cycle-orders INFO ping for the equity env."""
        self.pipeline._send_cycle_orders_ping("equity", self._to_summaries(orders))

    def run_crypto_cycle(self, orders: list[tuple[Any, ...]]) -> None:
        """Drive the pipeline's D12 cycle-orders INFO ping for the crypto env."""
        self.pipeline._send_cycle_orders_ping("crypto", self._to_summaries(orders))


@pytest.fixture
def mock_ctx(valid_config_yaml: str, tmp_path: Path) -> Generator[_MockCtx, None, None]:
    """Real-DB harness for the D11 fill-confirmation job + D12 cycle-orders ping."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available for testing")

    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(valid_config_yaml)
    config = load_config(config_file)
    config.system.database_url = db_url

    DatabaseManager.reset()
    db = DatabaseManager(config)
    db.init_schema()  # legacy tables idempotent; V001-V009 applied to the test DB externally
    ctx = _MockCtx(db, config, MagicMock())
    yield ctx
    DatabaseManager.reset()


class TestEquityFillConfirmationJob:
    """EXEC-D11: the 09:35 job turns confirmed pre-open auction fills into trades/embeds."""

    def test_fill_confirmation_records_auction_fill(self, mock_ctx: _MockCtx) -> None:
        """EXEC-D11: the 09:35 job converts a filled auction order into a trade row with the
        originating cycle_id, capture rows, and a trade embed."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o1", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "o1", status="filled", filled_avg_price=600.10, filled_qty=0.0416
        )

        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None
        assert trade["cycle_id"] == 42
        assert trade["price"] == pytest.approx(600.10)
        assert mock_ctx.alerter.send_embed.called  # trade embed fired
        # The pending order is stamped resolved so the next run does not re-record it.
        assert mock_ctx.pending_row("o1")["resolved_at"] is not None

    def test_fill_confirmation_enum_status_filled_closes_row(self, mock_ctx: _MockCtx) -> None:
        """Ruling 2026-07-23: a REAL alpaca-py OrderStatus.FILLED enum must close the row.

        Incident 2026-07-23: str(OrderStatus.FILLED).lower() == 'orderstatus.filled' never
        matched 'filled', so all 8 fully-filled auction orders were misclassified as
        still-live: false PARTIALLY-filled alerts, rows open forever, daily nag alerts.
        Production payloads carry the enum, not a string (proper-testing house rule).
        """
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="oenum1", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "oenum1",
            status=OrderStatus.FILLED,
            filled_avg_price=739.25,
            filled_qty=0.068434223,
        )
        equity_fill_confirmation_job()
        row = mock_ctx.pending_row("oenum1")
        assert row["resolved_at"] is not None, "enum FILLED must stamp resolved_at"
        assert row["disposition"] == "filled"
        titles = [
            c.kwargs.get("title", c.args[1] if len(c.args) > 1 else "")
            for c in mock_ctx.alerter.send_alert.call_args_list
        ]
        assert not any("PARTIALLY" in t for t in titles), titles

    def test_fill_confirmation_enum_status_partial_stays_open(self, mock_ctx: _MockCtx) -> None:
        """A REAL OrderStatus.PARTIALLY_FILLED enum takes the valid-partial path (row open)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="oenum2", cycle_id=42, symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status(
            "oenum2",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=694.63,
            filled_qty=0.05,
            qty=0.0973036,
        )
        equity_fill_confirmation_job()
        row = mock_ctx.pending_row("oenum2")
        assert row["resolved_at"] is None, "genuine partial must stay open"

    def test_fill_confirmation_enum_status_canceled_closes_terminal(
        self, mock_ctx: _MockCtx
    ) -> None:
        """A REAL OrderStatus.CANCELED enum takes the terminal-dead path."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="oenum3", cycle_id=42, symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status("oenum3", status=OrderStatus.CANCELED)
        equity_fill_confirmation_job()
        row = mock_ctx.pending_row("oenum3")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "canceled"

    def test_fill_confirmation_closes_canceled_order(self, mock_ctx: _MockCtx) -> None:
        """RULING-2: a canceled never-filled order gets ONE final warning and a terminal
        disposition — it must not stay open and re-warn daily forever."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o2", cycle_id=42, symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status("o2", status="canceled")

        equity_fill_confirmation_job()

        assert mock_ctx.inserted_trade() is None
        row = mock_ctx.pending_row("o2")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "canceled"
        assert mock_ctx.alerter.send_alert.called
        # Finding 2: the one-shot terminal-close warning must bypass the consecutive-failures
        # gate (count 1 < 3 would suppress it) and carry the symbol so the same-title cooldown
        # cannot swallow a sibling for another symbol the same morning.
        close_kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert close_kwargs["bypass_suppression"] is True
        assert "QQQ" in close_kwargs["title"]

        # Second run: the resolved row is no longer in the worklist — no repeat warning.
        mock_ctx.alerter.send_alert.reset_mock()
        equity_fill_confirmation_job()
        assert not mock_ctx.alerter.send_alert.called

    def test_fill_confirmation_partial_fill_recorded(self, mock_ctx: _MockCtx) -> None:
        """RULING-1: a partial auction fill IS recorded as a trade for the filled quantity
        at the broker's average price, embed fired, row left open for the remainder."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o5", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "o5", status="partially_filled", filled_avg_price=600.10, filled_qty=0.02, qty=0.05
        )

        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None
        assert float(trade["quantity"]) == pytest.approx(0.02)
        assert float(trade["price"]) == pytest.approx(600.10)
        assert trade["cycle_id"] == 42
        assert mock_ctx.alerter.send_embed.called
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert "partially" in kwargs["title"].lower()
        assert "recorded" in kwargs["title"].lower()
        # Remainder still working: row stays open, no disposition yet.
        assert mock_ctx.pending_row("o5")["resolved_at"] is None
        assert mock_ctx.pending_row("o5")["disposition"] is None

    def test_partial_fill_alert_is_per_symbol_and_unsuppressed(self, mock_ctx: _MockCtx) -> None:
        """Ruling 2026-07-23: every VALID partial notifies — symbol in title, no suppression.

        Two same-morning partials on different symbols are distinct events, not duplicates:
        the old shared title + consecutive-gate + cooldown delivered only 1 of 8.
        """
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="op1", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.set_pending_order(order_id="op2", cycle_id=42, symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status(
            "op1",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=739.25,
            filled_qty=0.03,
            qty=0.068434223,
        )
        mock_ctx.alpaca.order_status(
            "op2",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=694.63,
            filled_qty=0.05,
            qty=0.0973036,
        )
        equity_fill_confirmation_job()
        partial_calls = [
            c
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(partial_calls) == 2, partial_calls
        titles = sorted(c.kwargs["title"] for c in partial_calls)
        assert any("QQQ" in t for t in titles), titles
        assert any("SPY" in t for t in titles), titles
        assert all(c.kwargs.get("bypass_suppression") is True for c in partial_calls)

    def test_partial_fill_alert_notional_order_text(self, mock_ctx: _MockCtx) -> None:
        """Notional orders (qty=None) show '$X notional' instead of 'None requested'."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="op3", cycle_id=42, symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status(
            "op3",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=365.13,
            filled_qty=0.05,
            qty=None,
            notional=61.10,
        )
        equity_fill_confirmation_job()
        partial_calls = [
            c
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(partial_calls) == 1
        msg = partial_calls[0].kwargs["message"]
        assert "None requested" not in msg, msg
        assert "$61.10 notional" in msg, msg

    def test_partial_fill_alert_unknown_amount_when_qty_and_notional_none(
        self, mock_ctx: _MockCtx
    ) -> None:
        """DIGEST-D3: qty AND notional both None renders a neutral phrase, never '$None'."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="opn", cycle_id=42, symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status(
            "opn",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=100.0,
            filled_qty=0.05,
            qty=None,
            notional=None,
        )
        equity_fill_confirmation_job()
        msg = next(
            c.kwargs["message"]
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        )
        assert "$None" not in msg
        assert "an unknown amount" in msg, msg

    def test_fill_confirmation_idempotent_after_crash(self, mock_ctx: _MockCtx) -> None:
        """EXEC-D11 (review #2): a prior run that recorded the trade but crashed before
        stamping resolved_at must NOT re-record (duplicate TEXT PK) or fire a false
        'Fill Executed But Not Recorded' CRITICAL. The re-run detects the already-recorded
        fill, stamps resolved_at quietly, and moves on — no duplicate, no critical."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o6", cycle_id=42, symbol="SPY", side="buy")
        # Prior run recorded the trade (same broker order id = trades PK) then crashed before
        # UPDATE pending_orders SET resolved_at — so the row is still unresolved. The seeded
        # quantity equals the broker cumulative so this run's slice delta is exactly zero.
        mock_ctx.seed_trade(
            "o6", cycle_id=42, symbol="SPY", side="buy", price=600.10, quantity=0.0416
        )
        mock_ctx.alpaca.order_status(
            "o6", status="filled", filled_avg_price=600.10, filled_qty=0.0416
        )

        equity_fill_confirmation_job()  # must not raise

        # Quietly resolved — no re-record, no critical.
        assert mock_ctx.pending_row("o6")["resolved_at"] is not None
        assert mock_ctx.pending_row("o6")["disposition"] == "filled"
        assert mock_ctx.trade_count("o6") == 1  # no duplicate trade
        levels = [c.kwargs.get("level") for c in mock_ctx.alerter.send_alert.call_args_list]
        assert "critical" not in levels

    def test_fill_confirmation_passes_decision_price(self, mock_ctx: _MockCtx) -> None:
        """RULING-3: the confirmation job forwards the stored 09:15 decision_price into
        record_fill so fill_quality computes real auction slippage (was always NULL)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(
            order_id="odp1", cycle_id=42, symbol="SPY", side="buy", decision_price=600.00
        )
        mock_ctx.alpaca.order_status(
            "odp1", status="filled", filled_avg_price=600.60, filled_qty=0.05
        )

        equity_fill_confirmation_job()

        with mock_ctx.db.connection() as conn:
            fq = conn.execute(
                "SELECT decision_price_usd, slippage_frac FROM fill_quality WHERE trade_id = %s",
                ("odp1",),
            ).fetchone()
        assert fq is not None
        assert float(fq["decision_price_usd"]) == pytest.approx(600.00)
        assert fq["slippage_frac"] is not None  # 0.60/600 — measured, not NULL

    def test_fill_confirmation_second_slice_at_derived_price(self, mock_ctx: _MockCtx) -> None:
        """RULING-1: a later run records only the increment, priced so that recorded dollars
        match the broker's cumulative average exactly."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o7", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "o7", status="partially_filled", filled_avg_price=600.00, filled_qty=0.02, qty=0.05
        )
        equity_fill_confirmation_job()  # records slice 1: 0.02 @ 600.00

        # By the next run the order finished: cumulative 0.05 @ avg 600.60, then expired.
        mock_ctx.alpaca.order_status(
            "o7", status="expired", filled_avg_price=600.60, filled_qty=0.05, qty=0.05
        )
        equity_fill_confirmation_job()

        with mock_ctx.db.connection() as conn:
            slices = conn.execute(
                "SELECT trade_id, quantity, price FROM trades "
                "WHERE trade_id = %s OR trade_id LIKE %s ORDER BY trade_id",
                ("o7", "o7#%"),
            ).fetchall()
        assert len(slices) == 2
        assert slices[1]["trade_id"] == "o7#2"
        assert float(slices[1]["quantity"]) == pytest.approx(0.03)
        # Derived slice price: (600.60*0.05 - 600.00*0.02) / 0.03 = 601.00
        assert float(slices[1]["price"]) == pytest.approx(601.00)
        row = mock_ctx.pending_row("o7")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "expired"

    def test_fill_confirmation_expired_after_partial_stamps_terminal(
        self, mock_ctx: _MockCtx
    ) -> None:
        """RULING-2: terminal-with-partial closes the row ('expired'), keeps the recorded
        slice, and the final warning mentions both the recorded and unfilled parts."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o8", cycle_id=42, symbol="XLK", side="buy")
        mock_ctx.alpaca.order_status(
            "o8", status="expired", filled_avg_price=175.50, filled_qty=0.01, qty=0.04
        )

        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None and float(trade["quantity"]) == pytest.approx(0.01)
        row = mock_ctx.pending_row("o8")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "expired"

    def test_fill_confirmation_filled_unparseable_price_left_open(self, mock_ctx: _MockCtx) -> None:
        """Finding 1: broker says 'filled' but filled_avg_price is unusable (None) while
        shares are unrecorded -> the job must NOT stamp the row 'filled' and silently drop
        the fill. It warns and leaves the row open; a later run with a good price records the
        trade and closes the row (retry path proven end-to-end)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="ou1", cycle_id=42, symbol="SPY", side="buy")
        # First run: filled status but no usable average price -> books cannot match broker.
        mock_ctx.alpaca.order_status(
            "ou1", status="filled", filled_avg_price=None, filled_qty=0.0416
        )
        equity_fill_confirmation_job()

        # Nothing recorded, row left open, a WARNING alert fired (no false 'already recorded').
        assert mock_ctx.inserted_trade() is None
        row = mock_ctx.pending_row("ou1")
        assert row["resolved_at"] is None
        assert row["disposition"] is None
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert "unparseable" in kwargs["title"].lower()
        # Finding 2: one-shot warning must bypass suppression and name the symbol in the title.
        assert kwargs["bypass_suppression"] is True
        assert "SPY" in kwargs["title"]

        # Second run: the broker now returns a usable price -> the fill is recorded and the
        # row is stamped 'filled' (retry path works end-to-end).
        mock_ctx.alpaca.order_status(
            "ou1", status="filled", filled_avg_price=600.10, filled_qty=0.0416
        )
        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None
        assert float(trade["quantity"]) == pytest.approx(0.0416)
        assert float(trade["price"]) == pytest.approx(600.10)
        assert mock_ctx.alerter.send_embed.called
        resolved = mock_ctx.pending_row("ou1")
        assert resolved["resolved_at"] is not None
        assert resolved["disposition"] == "filled"

    def test_fill_confirmation_filled_zero_qty_left_open(self, mock_ctx: _MockCtx) -> None:
        """Finding 1: broker says 'filled' but filled_qty is unparseable (cum_qty computes 0)
        with nothing recorded -> the job warns and leaves the row open rather than stamping a
        phantom 'filled' backed by no trade."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="ou2", cycle_id=42, symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status(
            "ou2", status="filled", filled_avg_price=None, filled_qty="unparseable"
        )
        equity_fill_confirmation_job()

        assert mock_ctx.inserted_trade() is None
        row = mock_ctx.pending_row("ou2")
        assert row["resolved_at"] is None
        assert row["disposition"] is None
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert "unparseable" in kwargs["title"].lower()
        # Finding 2: one-shot warning must bypass suppression and name the symbol in the title.
        assert kwargs["bypass_suppression"] is True
        assert "QQQ" in kwargs["title"]

    def test_fill_confirmation_terminal_unparseable_left_open(self, mock_ctx: _MockCtx) -> None:
        """Finding 1: a terminal (expired) order reporting NEW executed shares but no parseable
        average price must NOT stamp the terminal disposition and silently unbook those shares —
        it warns (one-shot, bypass_suppression, symbol in title) and leaves the row open. A later
        run with a good price records the slice AND stamps the terminal disposition (retry path
        proven end-to-end)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="otu1", cycle_id=42, symbol="XLK", side="buy")
        # First run: expired with real filled shares but no usable average price.
        mock_ctx.alpaca.order_status(
            "otu1", status="expired", filled_avg_price=None, filled_qty=0.02, qty=0.05
        )
        equity_fill_confirmation_job()

        # Nothing recorded, row NOT stamped terminal, a one-shot WARNING fired.
        assert mock_ctx.inserted_trade() is None
        row = mock_ctx.pending_row("otu1")
        assert row["resolved_at"] is None
        assert row["disposition"] is None
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert "unparseable" in kwargs["title"].lower()
        assert "XLK" in kwargs["title"]
        assert kwargs["bypass_suppression"] is True

        # Second run: the broker now returns a usable price -> the slice is recorded AND the
        # terminal disposition is stamped (retry path works end-to-end).
        mock_ctx.alpaca.order_status(
            "otu1", status="expired", filled_avg_price=175.50, filled_qty=0.02, qty=0.05
        )
        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None
        assert float(trade["quantity"]) == pytest.approx(0.02)
        assert float(trade["price"]) == pytest.approx(175.50)
        resolved = mock_ctx.pending_row("otu1")
        assert resolved["resolved_at"] is not None
        assert resolved["disposition"] == "expired"

    def test_partial_fill_alert_titles_unique_per_order(self, mock_ctx: _MockCtx) -> None:
        """DIGEST-D2: two partials on the SAME symbol get DISTINCT titles (order_id in title)
        so the alerter's per-title 30-min cooldown cannot swallow the second (found 2026-07-23).
        """
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="sp1", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.set_pending_order(order_id="sp2", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "sp1",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=600.0,
            filled_qty=0.02,
            qty=0.05,
        )
        mock_ctx.alpaca.order_status(
            "sp2",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_avg_price=601.0,
            filled_qty=0.03,
            qty=0.05,
        )
        equity_fill_confirmation_job()
        titles = [
            c.kwargs["title"]
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(titles) == 2
        assert len(set(titles)) == 2, titles  # RED today: both identical
        assert any("sp1" in t for t in titles) and any("sp2" in t for t in titles)


class TestCycleOrdersInfoPing:
    """EXEC-D12: every cycle (both envs) ends with one INFO listing each order."""

    def test_cycle_orders_info_ping_both_envs_incl_sells(self, mock_ctx: _MockCtx) -> None:
        """EXEC-D12: EVERY cycle (equity AND crypto) ends with one INFO listing each order
        — buys as notional, SELLS as qty + approx value — or 'no orders — deadzone held'.
        Fires both ways (candle-alert full-parity precedent, user ruling)."""
        mock_ctx.run_equity_cycle(orders=[("SPY", "buy", 25.0), ("QQQ", "sell", 0.0416, 25.0)])
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert kwargs["level"] == "info" and "cycle orders" in kwargs["title"].lower()
        # STYLE-D15: aligned monospace columns (side left-justified to 4 -> "BUY  SPY").
        assert "BUY  SPY $25.00" in kwargs["message"]
        assert "SELL QQQ 0.0416" in kwargs["message"]  # sells never omitted

        mock_ctx.run_crypto_cycle(orders=[("BTCUSDT", "buy", 20.06)])
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert "crypto" in kwargs["title"].lower() and "BUY  BTCUSDT $20.06" in kwargs["message"]

        mock_ctx.run_crypto_cycle(orders=[])
        assert "deadzone" in mock_ctx.alerter.send_alert.call_args.kwargs["message"].lower()

    def test_cycle_ping_orders_render_in_code_block(self, mock_ctx: _MockCtx) -> None:
        """STYLE-D15: D12 cycle-ping order lists are monospace code blocks, aligned."""
        mock_ctx.run_equity_cycle(orders=[("SPY", "buy", 25.0), ("QQQ", "sell", 0.0416, 25.0)])
        msg = mock_ctx.alerter.send_alert.call_args.kwargs["message"]
        assert "```" in msg and "BUY  SPY" in msg
        # STYLE-D15: the ping is tagged category="cycle" (purple 🔄) for the embed styling.
        assert mock_ctx.alerter.send_alert.call_args.kwargs["category"] == "cycle"
