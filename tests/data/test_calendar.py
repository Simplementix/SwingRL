"""Tests for CalendarIngestor — FRED release dates + FOMC yaml/CSV -> calendar_events.

Plan A Task 11 (amended 2026-07-14: jobs live in the swingrl-collector, not the trader).

Step 0 (P-A4) — live FRED release-id verification, run 2026-07-16 against
https://api.stlouisfed.org/fred/release?release_id={id}:
    release_id 10 -> "Consumer Price Index"      (cpi)  ✓
    release_id 50 -> "Employment Situation"      (nfp)  ✓
    release_id 53 -> "Gross Domestic Product"    (gdp)  ✓
All three match, so calendar.fred_release_ids = {"cpi": 10, "nfp": 50, "gdp": 53}
is confirmed — no correction needed.

FOMC seed source — federalreserve.gov (recorded in config/fomc_dates_historical.csv):
    forward/recent : https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2027)
    historical     : https://www.federalreserve.gov/monetarypolicy/fomchistorical{2016..2020}.htm

backfill_start — live SELECT min(date) FROM ohlcv_daily = 2016-01-04, so the plan
default backfill_start 2015-01-01 covers the earliest bar; default kept.

DB-gated tests use the scratch DB (swingrl_plana_test @ V003); they skip when
DATABASE_URL is unset. Pure-unit tests (ET->UTC, staleness, collector wiring) run
without a database.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Pure-unit: ET -> UTC conversion across the DST boundary (zoneinfo, not fixed offset)
# ---------------------------------------------------------------------------


def test_et_to_utc_est_winter() -> None:
    """CAL-1: a January 08:30 ET print is EST (UTC-5) -> 13:30 UTC."""
    from swingrl.data.calendar import et_to_utc

    assert et_to_utc(date(2026, 1, 15), 8, 30) == datetime(2026, 1, 15, 13, 30, tzinfo=UTC)


def test_et_to_utc_edt_summer() -> None:
    """CAL-2: a July 08:30 ET print is EDT (UTC-4) -> 12:30 UTC."""
    from swingrl.data.calendar import et_to_utc

    assert et_to_utc(date(2026, 7, 15), 8, 30) == datetime(2026, 7, 15, 12, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pure-unit: staleness alarm (mocked db + alerter — no DB needed)
# ---------------------------------------------------------------------------


def _mock_db_with_max(max_sched: datetime | None) -> MagicMock:
    db = MagicMock()
    conn = db.connection.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = {"max_sched": max_sched}
    return db


def test_staleness_alerts_when_newest_event_within_min_future_days() -> None:
    """CAL-3: newest future event nearer than min_future_days -> warning send_alert."""
    from swingrl.data.calendar import run_calendar_staleness_check

    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    cfg = SwingRLConfig()
    cfg.calendar.min_future_days = 10
    db = _mock_db_with_max(now + timedelta(days=5))
    alerter = MagicMock()
    run_calendar_staleness_check(cfg, db, alerter, now=now)
    alerter.send_alert.assert_called_once()
    assert alerter.send_alert.call_args.args[0] == "warning"


def test_staleness_silent_when_events_far_future() -> None:
    """CAL-4: a healthy forward calendar (> min_future_days) does not alert."""
    from swingrl.data.calendar import run_calendar_staleness_check

    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    cfg = SwingRLConfig()
    cfg.calendar.min_future_days = 10
    db = _mock_db_with_max(now + timedelta(days=400))
    alerter = MagicMock()
    run_calendar_staleness_check(cfg, db, alerter, now=now)
    alerter.send_alert.assert_not_called()


def test_staleness_alerts_when_no_future_events() -> None:
    """CAL-5: an empty forward calendar (max is NULL) alerts."""
    from swingrl.data.calendar import run_calendar_staleness_check

    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    cfg = SwingRLConfig()
    db = _mock_db_with_max(None)
    alerter = MagicMock()
    run_calendar_staleness_check(cfg, db, alerter, now=now)
    alerter.send_alert.assert_called_once()


def test_staleness_noop_when_disabled() -> None:
    """CAL-6: disabled calendar never queries or alerts."""
    from swingrl.data.calendar import run_calendar_staleness_check

    cfg = SwingRLConfig()
    cfg.calendar.enabled = False
    db = MagicMock()
    alerter = MagicMock()
    run_calendar_staleness_check(cfg, db, alerter, now=datetime(2026, 7, 16, 12, 0, tzinfo=UTC))
    db.connection.assert_not_called()
    alerter.send_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Collector registration (no DB — MagicMock scheduler)
# ---------------------------------------------------------------------------


def _components(cfg: SwingRLConfig) -> dict:
    return {
        "config": cfg,
        "collector": MagicMock(),
        "store": MagicMock(),
        "alerter": MagicMock(),
        "db": MagicMock(),
    }


def test_calendar_jobs_registered_when_enabled() -> None:
    """CAL-7: weekly ingest + daily staleness jobs register when calendar.enabled."""
    from scripts.collector_main import all_job_ids, register_jobs

    cfg = SwingRLConfig()  # calendar.enabled defaults to True
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    registered = {c.kwargs["id"] for c in scheduler.add_job.call_args_list}
    assert {"calendar_ingest", "calendar_staleness"} <= registered
    assert registered == set(all_job_ids(cfg))


def test_calendar_jobs_absent_when_disabled() -> None:
    """CAL-8: disabled calendar registers no calendar jobs and drops them from the keep-set."""
    from scripts.collector_main import all_job_ids, register_jobs

    cfg = SwingRLConfig()
    cfg.calendar.enabled = False
    scheduler = MagicMock()
    register_jobs(scheduler, _components(cfg))
    registered = {c.kwargs["id"] for c in scheduler.add_job.call_args_list}
    assert not ({"calendar_ingest", "calendar_staleness"} & registered)
    assert "calendar_ingest" not in all_job_ids(cfg)


def test_calendar_ingest_job_runs_ingestor(monkeypatch) -> None:  # noqa: ANN001
    """CAL-9: the picklable job resolves components and runs CalendarIngestor.run()."""
    import scripts.collector_main as cm

    ran = {"n": 0}

    class _FakeIngestor:
        def __init__(self, config: SwingRLConfig, db: object) -> None:
            pass

        def run(self) -> int:
            ran["n"] += 1
            return 0

    monkeypatch.setattr(cm, "CalendarIngestor", _FakeIngestor)
    cm.set_components(_components(SwingRLConfig()))
    cm.calendar_ingest_job()
    assert ran["n"] == 1


def test_calendar_staleness_job_calls_check(monkeypatch) -> None:  # noqa: ANN001
    """CAL-10: the picklable staleness job delegates to run_calendar_staleness_check."""
    import scripts.collector_main as cm

    called = {"n": 0}
    monkeypatch.setattr(
        cm,
        "run_calendar_staleness_check",
        lambda config, db, alerter, **kw: called.__setitem__("n", called["n"] + 1),
    )
    cm.set_components(_components(SwingRLConfig()))
    cm.calendar_staleness_job()
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# DB-gated: real inserts against the scratch DB (swingrl_plana_test @ V003)
# ---------------------------------------------------------------------------


def _fred_http_mock(dates_by_release: dict[int, list[str]], captured: list | None = None):
    """Build a fake httpx.get returning FRED release/dates payloads per release_id."""

    def _get(url: str, params: dict | None = None, timeout: float | None = None):  # noqa: ANN202
        params = params or {}
        if captured is not None:
            captured.append(params)
        rid = int(params["release_id"])
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "release_dates": [{"release_id": rid, "date": d} for d in dates_by_release.get(rid, [])]
        }
        return resp

    return _get


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL scratch DB available",
)
class TestCalendarIngestDB:
    """Integration tests requiring the V003 scratch DB (calendar_events + data_ingestion_log)."""

    @pytest.fixture()
    def db(self) -> DatabaseManager:  # noqa: ANN201
        DatabaseManager.reset()
        return DatabaseManager(SwingRLConfig())

    @staticmethod
    def _rows(db: DatabaseManager) -> list[dict]:
        with db.connection() as conn:
            return conn.execute(
                "SELECT event_type, symbol, scheduled_at, window_start, window_end, "
                "importance, source FROM calendar_events ORDER BY event_type, scheduled_at"
            ).fetchall()

    def test_fred_ingest_inserts_rows_with_windows_and_source_fred(
        self, db: DatabaseManager, monkeypatch
    ) -> None:  # noqa: ANN001
        """Step 1(a): mocked FRED response inserts rows with materialized windows + source='fred'."""
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        monkeypatch.setattr(
            "swingrl.data.calendar.httpx.get",
            _fred_http_mock({10: ["2026-06-10"], 50: ["2026-06-05"], 53: ["2026-06-26"]}),
        )
        from swingrl.data.calendar import CalendarIngestor

        cfg = SwingRLConfig()
        cfg.calendar.fred_release_ids = {"cpi": 10, "nfp": 50, "gdp": 53}
        cfg.calendar.fomc_dates = []
        inserted = CalendarIngestor(cfg, db).run()
        assert inserted == 3

        rows = self._rows(db)
        assert {r["event_type"] for r in rows} == {"cpi", "nfp", "gdp"}
        assert all(r["source"] == "fred" for r in rows)
        assert all(r["symbol"] is None for r in rows)
        assert all(r["importance"] == "high" for r in rows)
        # CPI 2026-06-10 08:30 EDT (UTC-4) -> 12:30 UTC; window +/- 12h (materialized).
        cpi = next(r for r in rows if r["event_type"] == "cpi")
        assert cpi["scheduled_at"] == datetime(2026, 6, 10, 12, 30, tzinfo=UTC)
        assert cpi["window_start"] == datetime(2026, 6, 10, 0, 30, tzinfo=UTC)
        assert cpi["window_end"] == datetime(2026, 6, 11, 0, 30, tzinfo=UTC)

    def test_reingest_is_idempotent_zero_new_rows(self, db: DatabaseManager, monkeypatch) -> None:  # noqa: ANN001
        """Step 1(b): re-ingest of the same payload inserts 0 new rows (UNIQUE NULLS NOT DISTINCT)."""
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        monkeypatch.setattr(
            "swingrl.data.calendar.httpx.get",
            _fred_http_mock({10: ["2026-06-10", "2026-05-13"], 50: [], 53: []}),
        )
        from swingrl.data.calendar import CalendarIngestor

        cfg = SwingRLConfig()
        cfg.calendar.fred_release_ids = {"cpi": 10, "nfp": 50, "gdp": 53}
        cfg.calendar.fomc_dates = []
        first = CalendarIngestor(cfg, db).run()
        second = CalendarIngestor(cfg, db).run()
        assert first == 2
        assert second == 0
        assert len(self._rows(db)) == 2

    def test_fomc_yaml_ingest_source_config(self, db: DatabaseManager, monkeypatch) -> None:  # noqa: ANN001
        """Step 1(c): FOMC yaml dates ingest with source='config' at 14:00 ET -> UTC."""
        from swingrl.data.calendar import CalendarIngestor

        cfg = SwingRLConfig()
        cfg.calendar.fred_release_ids = {}  # FRED skipped — FOMC-only
        cfg.calendar.fomc_dates = ["2026-01-28T14:00:00", "2026-06-17T14:00:00"]
        inserted = CalendarIngestor(cfg, db).run()
        assert inserted == 2

        rows = [r for r in self._rows(db) if r["event_type"] == "fomc"]
        assert len(rows) == 2
        assert all(r["source"] == "config" for r in rows)
        assert all(r["symbol"] is None for r in rows)
        # Jan 28 14:00 EST (UTC-5) -> 19:00 UTC; window +/- 24h.
        jan = min(rows, key=lambda r: r["scheduled_at"])
        assert jan["scheduled_at"] == datetime(2026, 1, 28, 19, 0, tzinfo=UTC)
        assert jan["window_start"] == datetime(2026, 1, 27, 19, 0, tzinfo=UTC)
        assert jan["window_end"] == datetime(2026, 1, 29, 19, 0, tzinfo=UTC)

    def test_backfill_mode_asc_no_limit_and_csv(
        self, db: DatabaseManager, monkeypatch, tmp_path: Path
    ) -> None:  # noqa: ANN001
        """Step 0a: backfill uses sort_order=asc + no limit and loads the FOMC CSV seed."""
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        captured: list[dict] = []
        monkeypatch.setattr(
            "swingrl.data.calendar.httpx.get",
            _fred_http_mock({10: ["2016-01-16", "2016-02-19"]}, captured=captured),
        )
        csv_path = tmp_path / "fomc.csv"
        csv_path.write_text(
            "date,note\n2016-01-27T14:00:00,scheduled\n2016-03-16T14:00:00,scheduled\n"
        )
        from swingrl.data.calendar import CalendarIngestor

        cfg = SwingRLConfig()
        cfg.calendar.fred_release_ids = {"cpi": 10}
        cfg.calendar.fomc_dates = []
        cfg.calendar.fomc_backfill_csv = str(csv_path)
        cfg.calendar.backfill_start = "2015-01-01"
        inserted = CalendarIngestor(cfg, db).run(backfill=True)
        assert inserted == 4  # 2 cpi + 2 fomc

        assert captured, "httpx.get was not called"
        for params in captured:
            assert params["sort_order"] == "asc"
            assert "limit" not in params
            assert params["realtime_start"] == "2015-01-01"
        fomc = [r for r in self._rows(db) if r["event_type"] == "fomc"]
        assert len(fomc) == 2
        assert all(r["source"] == "config" for r in fomc)

    def test_run_writes_one_calendar_ingestion_log_row(
        self, db: DatabaseManager, monkeypatch
    ) -> None:  # noqa: ANN001
        """A run logs exactly one data_ingestion_log row with environment='calendar'."""
        from swingrl.data.calendar import CalendarIngestor

        cfg = SwingRLConfig()
        cfg.calendar.fred_release_ids = {}
        cfg.calendar.fomc_dates = ["2026-09-16T14:00:00"]
        CalendarIngestor(cfg, db).run()
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT environment, symbol, status, rows_inserted "
                "FROM data_ingestion_log WHERE environment = 'calendar'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "success"
        assert rows[0]["rows_inserted"] == 1
