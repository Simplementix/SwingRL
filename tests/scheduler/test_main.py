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
    config.scheduler.misfire_grace_s = {"equity": 720, "crypto": 3600}
    config.scheduler.max_workers = 4
    config.risk.sweep_interval_minutes = 30
    config.equity.cycle_time_et = "15:45"
    config.equity.market_calendar_gate = True
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
    """Verify that main.py registers all 13 cron/interval jobs."""

    def test_main_registers_all_jobs(self, mock_config: MagicMock) -> None:
        """PAPER-12 + SWEEP-D10: main.py registers 13 jobs with correct IDs."""
        from scripts.main import create_scheduler_and_register_jobs

        # Explicit (review minor finding #4): an unconfigured MagicMock attribute is
        # truthy by default, so this passed before by coincidence, not because the flag
        # was actually exercised. TestBackupJobGating covers both explicit states.
        mock_config.backup.trader_backup_jobs_enabled = True
        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        assert mock_scheduler.add_job.call_count == 13

        job_ids = {c.kwargs["id"] for c in mock_scheduler.add_job.call_args_list}
        expected_ids = {
            "equity_cycle",
            "crypto_cycle",
            "daily_summary",
            "stuck_agent_check",
            "weekly_fundamentals",
            "monthly_macro",
            "daily_sqlite_backup",
            "weekly_duckdb_backup",
            "monthly_offsite",
            "shadow_promotion_check",
            "automated_trigger_check",
            "daily_reconciliation",
            "risk_sweep",
        }
        assert job_ids == expected_ids

    def test_risk_sweep_schedule(self, mock_config: MagicMock) -> None:
        """SWEEP-D10: risk_sweep fires on an interval trigger every
        config.risk.sweep_interval_minutes minutes (between-cycle blind-window guard)."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_config.risk.sweep_interval_minutes = 30
        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        sweep_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "risk_sweep"
        )
        assert sweep_call.kwargs["trigger"] == "interval"
        assert sweep_call.kwargs["minutes"] == 30
        assert sweep_call.kwargs["replace_existing"] is True

    def test_equity_cycle_schedule(self, mock_config: MagicMock) -> None:
        """(e) equity_cycle fires from config.equity.cycle_time_et on weekdays (review C2).

        The cron time is read from config (15:45 default, before close) — no hardcoded
        16:15 post-close hour — and restricted to Mon-Fri so it never runs on weekends.
        """
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        equity_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "equity_cycle"
        )
        assert equity_call.kwargs["trigger"] == "cron"
        assert equity_call.kwargs["hour"] == 15
        assert equity_call.kwargs["minute"] == 45
        assert equity_call.kwargs["day_of_week"] == "mon-fri"
        assert equity_call.kwargs["timezone"] == "America/New_York"

    def test_cycle_jobs_use_misfire_grace_from_config(self, mock_config: MagicMock) -> None:
        """(f) equity/crypto cycle jobs read misfire_grace_time from scheduler.misfire_grace_s.

        Restart addendum (A30): equity 720s (late-but-pre-close), crypto 3600s (4H cadence).
        """
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        equity_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "equity_cycle"
        )
        crypto_call = next(
            c for c in mock_scheduler.add_job.call_args_list if c.kwargs["id"] == "crypto_cycle"
        )
        assert equity_call.kwargs["misfire_grace_time"] == 720
        assert crypto_call.kwargs["misfire_grace_time"] == 3600

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

    def test_daily_reconciliation_schedule(self, mock_config: MagicMock) -> None:
        """PAPER-09: daily_reconciliation fires at 5:00 PM ET."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        recon_call = next(
            c
            for c in mock_scheduler.add_job.call_args_list
            if c.kwargs["id"] == "daily_reconciliation"
        )
        assert recon_call.kwargs["trigger"] == "cron"
        assert recon_call.kwargs["hour"] == 17
        assert recon_call.kwargs["minute"] == 0
        assert recon_call.kwargs["timezone"] == "America/New_York"


class TestBackupJobGating:
    """USER RULING 2026-07-19: the trader's in-container backup jobs (daily_sqlite_backup,
    weekly_duckdb_backup, monthly_offsite) are config-gated behind
    config.backup.trader_backup_jobs_enabled — the trader image has no pg_dump binary, so
    they failed nightly; host-side dumps + Duplicati already cover backups.
    """

    BACKUP_JOB_IDS = frozenset({"daily_sqlite_backup", "weekly_duckdb_backup", "monthly_offsite"})
    NON_BACKUP_JOB_IDS = frozenset(
        {
            "equity_cycle",
            "crypto_cycle",
            "daily_summary",
            "stuck_agent_check",
            "weekly_fundamentals",
            "monthly_macro",
            "shadow_promotion_check",
            "automated_trigger_check",
            "daily_reconciliation",
            "risk_sweep",
        }
    )

    def test_flag_true_registers_backup_jobs(self, mock_config: MagicMock) -> None:
        """Flag true (default): the three backup ids are registered alongside every other job."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_config.backup.trader_backup_jobs_enabled = True
        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        job_ids = {c.kwargs["id"] for c in mock_scheduler.add_job.call_args_list}
        assert job_ids == self.BACKUP_JOB_IDS | self.NON_BACKUP_JOB_IDS

    def test_flag_false_skips_backup_jobs(self, mock_config: MagicMock) -> None:
        """Flag false: none of the three backup ids are registered; every other job still is."""
        from scripts.main import create_scheduler_and_register_jobs

        mock_config.backup.trader_backup_jobs_enabled = False
        mock_scheduler = MagicMock()
        create_scheduler_and_register_jobs(mock_scheduler, mock_config)

        job_ids = {c.kwargs["id"] for c in mock_scheduler.add_job.call_args_list}
        assert job_ids == self.NON_BACKUP_JOB_IDS
        assert job_ids.isdisjoint(self.BACKUP_JOB_IDS)

    def test_remove_stale_backup_jobs_tolerant_of_absence(self, mock_config: MagicMock) -> None:
        """No general stale-job sweep exists in scripts/main.py (unlike the collector's
        remove_stale_jobs), so replace_existing=True alone would leave a previously
        persisted backup job behind once the flag flips to False. The targeted cleanup
        (called by main() AFTER scheduler.start() -- see TestMainStaleBackupCleanupOrder
        and test_flag_false_removes_persisted_stale_backup_job_across_restart below) must
        attempt removal of all three ids and swallow JobLookupError when a job id was never
        persisted (e.g. a fresh deployment)."""
        from apscheduler.jobstores.base import JobLookupError

        from scripts.main import _remove_stale_trader_backup_jobs

        mock_config.backup.trader_backup_jobs_enabled = False
        mock_scheduler = MagicMock()
        mock_scheduler.remove_job.side_effect = JobLookupError("missing")

        removed = _remove_stale_trader_backup_jobs(mock_scheduler, mock_config)

        removed_ids = {c.args[0] for c in mock_scheduler.remove_job.call_args_list}
        assert removed_ids == self.BACKUP_JOB_IDS
        # JobLookupError was swallowed for every id, so nothing is reported as removed.
        assert removed == []

    def test_remove_stale_backup_jobs_noop_when_flag_true(self, mock_config: MagicMock) -> None:
        """Flag true: the three ids are the desired state, not stale -- no removal attempt
        at all (would otherwise race against the jobs this same boot just registered)."""
        from scripts.main import _remove_stale_trader_backup_jobs

        mock_config.backup.trader_backup_jobs_enabled = True
        mock_scheduler = MagicMock()

        removed = _remove_stale_trader_backup_jobs(mock_scheduler, mock_config)

        mock_scheduler.remove_job.assert_not_called()
        assert removed == []

    def test_flag_false_removes_persisted_stale_backup_job_across_restart(self, tmp_path) -> None:
        """Reproduces the real production sequence across a redeploy/restart, the exact
        scenario the CRITICAL review finding proved broken.

        Process 1 (prior deploy): flag True, register, start, persist the three backup
        jobs to the on-disk jobstore file, shut down.

        Process 2 (this deploy): a FRESH BackgroundScheduler instance -- not the same
        object as process 1, mirroring build_app() constructing a brand-new scheduler on
        every boot -- registers with the flag False via create_scheduler_and_register_jobs()
        BEFORE scheduler.start() (exactly like build_app() calls it), then starts, then
        runs _remove_stale_trader_backup_jobs() AFTER start() (exactly like main() calls
        it). Only that ordering can see jobs process 1 persisted: calling the cleanup
        pre-start (the bug this fixes) would leave them behind, because remove_job() on a
        STOPPED scheduler only consults this process's in-memory pending-jobs list, never
        the jobstore file a prior process wrote to.
        """
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.schedulers.background import BackgroundScheduler

        from scripts.main import (
            _remove_stale_trader_backup_jobs,
            create_scheduler_and_register_jobs,
        )
        from swingrl.config.schema import SwingRLConfig

        jobstore_url = f"sqlite:///{tmp_path / 'jobs.sqlite'}"

        # --- Process 1 (prior deploy): flag True, register, start, persist, shut down. ---
        cfg1 = SwingRLConfig()
        cfg1.backup.trader_backup_jobs_enabled = True
        scheduler1 = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=jobstore_url)},
            job_defaults={"coalesce": True, "max_instances": 1},
        )
        create_scheduler_and_register_jobs(scheduler1, cfg1)
        scheduler1.start(paused=True)  # flushes pending jobs to the persistent jobstore
        ids_persisted = {job.id for job in scheduler1.get_jobs()}
        assert self.BACKUP_JOB_IDS <= ids_persisted
        scheduler1.shutdown(wait=False)

        # --- Process 2 (this deploy): FRESH scheduler, flag now False. Mirrors
        # build_app() (register before start) then main() (cleanup after start). ---
        cfg2 = SwingRLConfig()
        cfg2.backup.trader_backup_jobs_enabled = False
        scheduler2 = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=jobstore_url)},
            job_defaults={"coalesce": True, "max_instances": 1},
        )
        create_scheduler_and_register_jobs(scheduler2, cfg2)  # build_app(): before start()
        scheduler2.start(paused=True)
        try:
            removed = _remove_stale_trader_backup_jobs(scheduler2, cfg2)  # main(): after start()
            assert set(removed) == self.BACKUP_JOB_IDS

            ids_after = {job.id for job in scheduler2.get_jobs()}
            assert ids_after.isdisjoint(self.BACKUP_JOB_IDS), (
                f"stale backup ids survived a restart: {ids_after & self.BACKUP_JOB_IDS}"
            )
            assert self.NON_BACKUP_JOB_IDS <= ids_after

            # Nothing left to remove now — must not raise (absence tolerated).
            removed_again = _remove_stale_trader_backup_jobs(scheduler2, cfg2)
            assert removed_again == []
        finally:
            scheduler2.shutdown(wait=False)


class TestMainStaleBackupCleanupOrder:
    """CRITICAL fix regression guard: main() must invoke the stale-backup cleanup strictly
    AFTER scheduler.start(), never before. This pins the call order at the exact call site
    the review flagged, independent of the real-jobstore proof above."""

    def test_main_calls_cleanup_after_scheduler_start(
        self, mock_config: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A future refactor that moves the cleanup call back before scheduler.start()
        (or back inside create_scheduler_and_register_jobs()) would reintroduce the bug
        silently; this test fails immediately if that ordering regresses."""
        import scripts.main as main_mod

        call_order: list[str] = []
        mock_scheduler = MagicMock()
        mock_scheduler.start.side_effect = lambda *a, **k: call_order.append("start")
        mock_stop_event = MagicMock()

        monkeypatch.setattr(
            main_mod,
            "build_app",
            lambda config_path: {
                "scheduler": mock_scheduler,
                "stop_event": mock_stop_event,
                "config": mock_config,
            },
        )

        def fake_cleanup(scheduler: MagicMock, config: MagicMock) -> list[str]:
            call_order.append("cleanup")
            return []

        monkeypatch.setattr(main_mod, "_remove_stale_trader_backup_jobs", fake_cleanup)
        monkeypatch.setattr(main_mod.signal, "signal", MagicMock())
        monkeypatch.setattr(main_mod.sys, "argv", ["main.py"])

        with pytest.raises(SystemExit) as exc_info:
            main_mod.main()

        assert exc_info.value.code == 0
        assert call_order == ["start", "cleanup"]


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
        from tests.conftest import make_mock_db

        mock_scheduler = MagicMock()
        # Schema version far ahead of EXPECTED_SCHEMA_VERSION: assert_schema_current only
        # warns (never raises) when ahead (A30 floor semantics) — robust to future version
        # bumps, unlike EXPECTED_SCHEMA_VERSION's old accidental match with MagicMock's
        # default __int__() == 1.
        mock_db, _ = make_mock_db(fetchone_returns=[{"v": 999}])

        with patch("scripts.main.BackgroundScheduler", return_value=mock_scheduler):
            with patch("scripts.main.SQLAlchemyJobStore"):
                with patch("scripts.main.ThreadPoolExecutor"):
                    with patch("scripts.main.DatabaseManager", return_value=mock_db):
                        with patch("scripts.main.ExecutionPipeline"):
                            with patch("scripts.main.Alerter"):
                                with patch("scripts.main.start_stop_polling_thread"):
                                    build_app(config_path="config/test.yaml")

        mock_init_flags.assert_called_once()
        mock_init_job_ctx.assert_called_once()

    @patch("scripts.main.init_emergency_flags")
    @patch("scripts.main.init_job_context")
    @patch("scripts.main.load_config")
    @patch("scripts.main.configure_logging")
    def test_build_app_creates_pipeline_with_all_args(
        self,
        mock_logging: MagicMock,
        mock_load_config: MagicMock,
        mock_init_job_ctx: MagicMock,
        mock_init_flags: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PAPER-01: build_app creates ExecutionPipeline with all 5 required arguments."""
        mock_load_config.return_value = mock_config

        from scripts.main import build_app
        from tests.conftest import make_mock_db

        mock_scheduler = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_feature_pipeline_cls = MagicMock()
        # Schema version far ahead of EXPECTED_SCHEMA_VERSION: assert_schema_current only
        # warns (never raises) when ahead (A30 floor semantics).
        mock_db, _ = make_mock_db(fetchone_returns=[{"v": 999}])

        with patch("scripts.main.BackgroundScheduler", return_value=mock_scheduler):
            with patch("scripts.main.SQLAlchemyJobStore"):
                with patch("scripts.main.ThreadPoolExecutor"):
                    with patch("scripts.main.DatabaseManager", return_value=mock_db):
                        with patch("scripts.main.ExecutionPipeline", mock_pipeline_cls):
                            with patch("scripts.main.Alerter"):
                                with patch("scripts.main.start_stop_polling_thread"):
                                    with patch(
                                        "scripts.main.FeaturePipeline",
                                        mock_feature_pipeline_cls,
                                    ):
                                        build_app(config_path="config/test.yaml")

        # ExecutionPipeline must be called with all 5 keyword arguments
        mock_pipeline_cls.assert_called_once()
        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert "config" in call_kwargs
        assert "db" in call_kwargs
        assert "feature_pipeline" in call_kwargs
        assert "alerter" in call_kwargs
        assert "models_dir" in call_kwargs

        # FeaturePipeline must have been constructed
        mock_feature_pipeline_cls.assert_called_once()

    @patch("scripts.main.init_emergency_flags")
    @patch("scripts.main.init_job_context")
    @patch("scripts.main.load_config")
    @patch("scripts.main.configure_logging")
    def test_build_app_passes_bare_models_dir_to_pipeline(
        self,
        mock_logging: MagicMock,
        mock_load_config: MagicMock,
        mock_init_job_ctx: MagicMock,
        mock_init_flags: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """PAPER-02: build_app passes bare models_dir (no / 'active') to ExecutionPipeline."""
        from pathlib import Path

        mock_load_config.return_value = mock_config
        mock_config.paths.models_dir = "models"

        from scripts.main import build_app
        from tests.conftest import make_mock_db

        mock_scheduler = MagicMock()
        mock_pipeline_cls = MagicMock()
        # Schema version far ahead of EXPECTED_SCHEMA_VERSION: assert_schema_current only
        # warns (never raises) when ahead (A30 floor semantics).
        mock_db, _ = make_mock_db(fetchone_returns=[{"v": 999}])

        with patch("scripts.main.BackgroundScheduler", return_value=mock_scheduler):
            with patch("scripts.main.SQLAlchemyJobStore"):
                with patch("scripts.main.ThreadPoolExecutor"):
                    with patch("scripts.main.DatabaseManager", return_value=mock_db):
                        with patch("scripts.main.ExecutionPipeline", mock_pipeline_cls):
                            with patch("scripts.main.Alerter"):
                                with patch("scripts.main.start_stop_polling_thread"):
                                    with patch("scripts.main.FeaturePipeline"):
                                        build_app(config_path="config/test.yaml")

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        models_dir = call_kwargs["models_dir"]
        # Must be exactly Path("models") — no "/ active" suffix
        assert models_dir == Path("models"), (
            f"Expected Path('models'), got {models_dir!r}. "
            "Double 'active' nesting bug may be present."
        )

    @patch("scripts.main.reconciliation_job")
    @patch("scripts.main.init_emergency_flags")
    @patch("scripts.main.init_job_context")
    @patch("scripts.main.load_config")
    @patch("scripts.main.configure_logging")
    def test_startup_reconciliation_runs_once_at_boot(
        self,
        mock_logging: MagicMock,
        mock_load_config: MagicMock,
        mock_init_job_ctx: MagicMock,
        mock_init_flags: MagicMock,
        mock_reconciliation: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """(g) build_app runs the equity reconciliation once at boot (restart drift audit).

        Restart addendum (A30): downtime fill/position drift is audited immediately, not
        hours later at the 17:00 ET cron.
        """
        mock_load_config.return_value = mock_config

        from scripts.main import build_app
        from tests.conftest import make_mock_db

        mock_scheduler = MagicMock()
        # Schema version far ahead: assert_schema_current only warns when ahead (A30 floor).
        mock_db, _ = make_mock_db(fetchone_returns=[{"v": 999}])

        with patch("scripts.main.BackgroundScheduler", return_value=mock_scheduler):
            with patch("scripts.main.SQLAlchemyJobStore"):
                with patch("scripts.main.ThreadPoolExecutor"):
                    with patch("scripts.main.DatabaseManager", return_value=mock_db):
                        with patch("scripts.main.ExecutionPipeline"):
                            with patch("scripts.main.Alerter"):
                                with patch("scripts.main.start_stop_polling_thread"):
                                    with patch("scripts.main.FeaturePipeline"):
                                        build_app(config_path="config/test.yaml")

        # Reconciliation invoked exactly once, before the scheduler is started (build_app
        # never calls scheduler.start()).
        mock_reconciliation.assert_called_once()
        mock_scheduler.start.assert_not_called()


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

        # Mock the connection context manager. daily_summary_job issues TWO queries:
        # the portfolio_snapshots read and the per-env signal-trade count. Route by SQL
        # so the trades query returns rows carrying the `n` key the count loop reads.
        snapshot_rows = [
            {
                "environment": "equity",
                "total_value": 400.0,
                "cash_balance": 380.0,
                "daily_pnl": 5.0,
                "drawdown_pct": 0.02,
            }
        ]
        trade_count_rows = [{"environment": "equity", "n": 3}]

        def _execute(sql: str, *args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            if "FROM trades" in sql:
                result.fetchall.return_value = trade_count_rows
            else:
                result.fetchall.return_value = snapshot_rows
            return result

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = _execute
        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

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


def test_equity_cycle_default_is_preopen() -> None:
    """SCHED-D2: default equity cycle is 09:15 ET pre-open — decide on t-1 bars, submit
    before the 09:28 auction cutoff, fill at the open (spec D2/D11)."""
    from swingrl.config.schema import EquityConfig

    assert EquityConfig().cycle_time_et == "09:15"
