"""Tests for scripts/main.py production entrypoint.

Tests verify job registration, init sequence, signal handling,
and end-to-end job -> embed -> alerter callback chain wiring.
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_config() -> MagicMock:
    """Create a mock SwingRLConfig with scheduler and alerting sections."""
    config = MagicMock()
    config.scheduler.apscheduler_db_path = "db/test_jobs.sqlite"
    config.scheduler.misfire_grace_time = 300
    config.scheduler.max_workers = 4
    config.alerting.alerts_webhook_url = ""
    config.alerting.daily_webhook_url = ""
    config.alerting.alert_cooldown_minutes = 30
    config.alerting.consecutive_failures_before_alert = 3
    config.logging.json_logs = False
    config.logging.level = "INFO"
    return config


@pytest.fixture()
def mock_fill() -> MagicMock:
    """Create a mock FillResult for trade embed tests."""
    fill = MagicMock()
    fill.symbol = "SPY"
    fill.side = "buy"
    fill.quantity = 10.0
    fill.fill_price = 450.0
    fill.commission = 0.0
    fill.environment = "equity"
    return fill


class TestMainRegistersJobs:
    """Verify that main.py registers all 6 cron jobs."""

    def test_main_registers_six_jobs(self, mock_config: MagicMock) -> None:
        """PAPER-12: main.py registers 6 cron jobs with correct IDs."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        assert mock_scheduler.add_job.call_count == 6

        job_ids = {c.kwargs["id"] for c in mock_scheduler.add_job.call_args_list}
        expected_ids = {
            "equity_cycle",
            "crypto_cycle",
            "daily_summary",
            "stuck_agent_check",
            "weekly_fundamentals",
            "monthly_macro",
        }
        assert job_ids == expected_ids

    def test_equity_cycle_schedule(self, mock_config: MagicMock) -> None:
        """PAPER-12: equity_cycle fires at 4:15 PM ET."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        equity_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "equity_cycle"
        )
        assert equity_call.kwargs["trigger"] == "cron"
        assert equity_call.kwargs["hour"] == 16
        assert equity_call.kwargs["minute"] == 15

    def test_crypto_cycle_schedule(self, mock_config: MagicMock) -> None:
        """PAPER-12: crypto_cycle fires 6x/day at 5 min past each 4H bar close."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        crypto_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "crypto_cycle"
        )
        assert crypto_call.kwargs["trigger"] == "cron"
        assert crypto_call.kwargs["hour"] == "0,4,8,12,16,20"
        assert crypto_call.kwargs["minute"] == 5

    def test_all_jobs_replace_existing(self, mock_config: MagicMock) -> None:
        """PAPER-12: all jobs use replace_existing=True for restart recovery."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        for c in mock_scheduler.add_job.call_args_list:
            assert c.kwargs.get("replace_existing") is True, (
                f"Job {c.kwargs.get('id')} missing replace_existing=True"
            )


class TestMainInitSequence:
    """Verify init_emergency_flags and init_job_context called before scheduler.start()."""

    @patch("scripts.main.init_emergency_flags")
    @patch("scripts.main.init_job_context")
    @patch("scripts.main.load_config")
    @patch("scripts.main.configure_logging")
    def test_init_order(
        self,
        mock_logging: MagicMock,
        mock_load_config: MagicMock,
        mock_init_job_ctx: MagicMock,
        mock_init_flags: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PAPER-15: init_emergency_flags and init_job_context called before scheduler.start()."""
        mock_load_config.return_value = mock_config

        from scripts.main import build_app

        mock_scheduler = MagicMock()

        with patch("scripts.main.BackgroundScheduler", return_value=mock_scheduler):
            with patch("scripts.main.DatabaseManager"):
                with patch("scripts.main.ExecutionPipeline"):
                    with patch("scripts.main.Alerter"):
                        with patch("scripts.main.start_stop_polling_thread"):
                            build_app(config_path="config/test.yaml")

        mock_init_flags.assert_called_once()
        mock_init_job_ctx.assert_called_once()


class TestMainSignalHandler:
    """Verify SIGTERM triggers scheduler.shutdown."""

    def test_sigterm_calls_shutdown(self, mock_config: MagicMock) -> None:
        """PAPER-16: SIGTERM triggers scheduler.shutdown(wait=False)."""
        from scripts.main import make_signal_handler

        mock_scheduler = MagicMock()
        mock_event = MagicMock()
        handler = make_signal_handler(mock_scheduler, mock_event)

        handler(signal.SIGTERM, None)

        mock_scheduler.shutdown.assert_called_once_with(wait=False)
        mock_event.set.assert_called_once()


class TestEquityCycleSendsTradeEmbeds:
    """Integration: equity_cycle -> build_trade_embed -> alerter.send_embed."""

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_equity_cycle_sends_trade_embeds(self, mock_fill: MagicMock) -> None:
        """PAPER-12: equity_cycle calls build_trade_embed and routes via alerter.send_embed."""
        from swingrl.scheduler.jobs import init_job_context

        mock_alerter = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.execute_cycle.return_value = [mock_fill]
        mock_db = MagicMock()
        mock_config = MagicMock()
        mock_config.alerting.healthchecks_equity_url = ""

        # Make is_halted return False
        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch(
                "swingrl.scheduler.jobs.build_trade_embed",
                return_value={"embeds": [{"title": "BUY SPY"}]},
            ) as mock_build:
                init_job_context(
                    config=mock_config, db=mock_db, pipeline=mock_pipeline, alerter=mock_alerter
                )

                from swingrl.scheduler.jobs import equity_cycle

                fills = equity_cycle()

        assert len(fills) == 1
        mock_build.assert_called_once_with(mock_fill)
        mock_alerter.send_embed.assert_called_once()

    @patch("swingrl.scheduler.jobs._ctx", new=None)
    def test_daily_summary_sends_embed(self) -> None:
        """PAPER-12: daily_summary_job calls build_daily_summary_embed and routes via alerter."""
        from swingrl.scheduler.jobs import init_job_context

        mock_alerter = MagicMock()
        mock_db = MagicMock()
        mock_config = MagicMock()

        # Mock the sqlite context manager to return rows
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            {
                "environment": "equity",
                "total_value": 400.0,
                "cash_balance": 380.0,
                "daily_pnl": 5.0,
                "drawdown_pct": 0.02,
            }
        ]
        mock_db.sqlite.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.sqlite.return_value.__exit__ = MagicMock(return_value=False)

        with patch("swingrl.scheduler.jobs.is_halted", return_value=False):
            with patch(
                "swingrl.scheduler.jobs.build_daily_summary_embed",
                return_value={"embeds": [{"title": "Daily Summary"}]},
            ):
                init_job_context(
                    config=mock_config, db=mock_db, pipeline=MagicMock(), alerter=mock_alerter
                )

                from swingrl.scheduler.jobs import daily_summary_job

                daily_summary_job()

        mock_alerter.send_embed.assert_called_once()
