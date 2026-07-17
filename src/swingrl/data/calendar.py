"""Event-calendar ingest: FRED release dates + FOMC yaml/CSV -> ``calendar_events``.

NOT a ``BaseIngestor`` subclass. ``BaseIngestor``'s fetch/validate/store contract is
per-symbol-OHLCV-and-Parquet centric (fetch a DataFrame -> write Parquet -> sync to an
ohlcv table). Event-calendar data is a small set of scheduled macro events written
straight to the relational ``calendar_events`` table with materialized
``[window_start, window_end]`` windows and ``ON CONFLICT DO NOTHING`` idempotency
(UNIQUE NULLS NOT DISTINCT on ``(event_type, symbol, scheduled_at)``). This class therefore
owns its own ``run()``/upsert path and logs one ``data_ingestion_log`` row with
``environment='calendar'``, reusing ``base.py:_log_ingestion``'s shape (spec §4 D-T3.14;
Plan A Task 11, amended 2026-07-14 — the ingest + staleness jobs live in the
swingrl-collector, and the trader only READS calendar_events at cycle time, Task 9).

Sources:
  * CPI / NFP / GDP — FRED ``release/dates`` API (release ids verified live, P-A4).
    Each release date maps to ``scheduled_at`` at the standard 08:30 ET print time,
    converted to UTC via ``zoneinfo`` (DST-correct), ``importance='high'``, ``source='fred'``.
  * FOMC — ISO datetimes from ``config.calendar.fomc_dates`` (forward schedule) in normal
    mode, or the committed historical CSV seed (``config.calendar.fomc_backfill_csv``) in
    backfill mode. A bare date defaults to the 14:00 ET statement release time.
    ``source='config'``.

Backfill (Step 0a) is model-neutral: events are not observation features (locked design
D-MT.2). It enables Plan B backtest-side event stamping, ``event_shock_sensitivity``
weakness-profile seeding (§4.8), and empirical window-size validation.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import httpx
import structlog

from swingrl.config.schema import SwingRLConfig

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_ENVIRONMENT = "calendar"
_LOG_SYMBOL = "calendar"  # data_ingestion_log.symbol is NOT NULL — macro events have no ticker
_MACRO_PRINT_TIME_ET = (8, 30)  # CPI/NFP/GDP 08:30 ET print
_FOMC_ANNOUNCE_TIME_ET = (14, 0)  # FOMC 14:00 ET statement release
_DEFAULT_WINDOW_HOURS = (12, 12)
_REALTIME_END_MAX = "9999-12-31"  # FRED realtime_end for full-history backfill

_INSERT_SQL = (
    "INSERT INTO calendar_events "
    "(event_type, symbol, scheduled_at, window_start, window_end, importance, source) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT DO NOTHING"
)

_MAX_FUTURE_SQL = (
    "SELECT max(scheduled_at) AS max_sched FROM calendar_events WHERE scheduled_at >= %s"
)

_LOG_INSERT_SQL = (
    "INSERT INTO data_ingestion_log "
    "(run_id, environment, symbol, status, rows_inserted, errors_count, duration_ms, "
    "binance_weight_used) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
)


def et_to_utc(day: date, hour: int, minute: int) -> datetime:
    """Convert an ET wall-clock time on ``day`` to a UTC datetime (DST-correct).

    Uses ``zoneinfo`` America/New_York — never a fixed offset — so EST (UTC-5) and EDT
    (UTC-4) prints both land on the correct UTC instant.
    """
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=_ET)
    return local.astimezone(UTC)


def _parse_fomc_datetime(value: str) -> datetime:
    """Parse an FOMC date string to a UTC datetime.

    Accepts a bare ISO date (``2026-01-28`` -> 14:00 ET statement time), an ET-naive ISO
    datetime (``2026-01-28T14:00:00`` -> localized to America/New_York), or a tz-aware ISO
    datetime (respected as given). Raises on anything unparseable.
    """
    value = value.strip()
    if "T" in value or " " in value:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed.astimezone(UTC)
        return et_to_utc(parsed.date(), parsed.hour, parsed.minute)
    return et_to_utc(date.fromisoformat(value), *_FOMC_ANNOUNCE_TIME_ET)


@dataclass(frozen=True)
class _Event:
    """One materialized calendar event ready to upsert."""

    event_type: str
    symbol: str | None
    scheduled_at: datetime
    window_start: datetime
    window_end: datetime
    importance: str
    source: str


class CalendarIngestor:
    """Standalone (non-BaseIngestor) ingestor for macro event calendars.

    Args:
        config: Validated SwingRLConfig (reads ``config.calendar``).
        db: DatabaseManager for the ``calendar_events`` upsert + ``data_ingestion_log``.
    """

    def __init__(self, config: SwingRLConfig, db: DatabaseManager) -> None:
        self._config = config
        self._cal = config.calendar
        self._db = db
        self._api_key = os.environ.get("FRED_API_KEY", "")

    def run(self, *, backfill: bool = False) -> int:
        """Ingest FRED + FOMC events into ``calendar_events``; return rows upserted.

        Fail-soft: a per-release FRED fetch error or a bad FOMC date is logged and counted
        in ``errors_count`` but does not abort the run. Every run logs exactly one
        ``data_ingestion_log`` row with ``environment='calendar'``.

        Args:
            backfill: If True, pull the full historical FRED series (``sort_order=asc``, no
                limit, from ``calendar.backfill_start``) and the FOMC CSV seed instead of the
                recent window + forward yaml (Step 0a).

        Returns:
            Number of new rows inserted (0 on a fully idempotent re-ingest).
        """
        run_id = uuid.uuid4().hex
        start = time.perf_counter()
        status = "success"
        inserted = 0
        errors = 0
        try:
            if not self._cal.enabled:
                log.info("calendar_ingest_disabled")
                status = "no_data"
                return 0
            events, errors = self._collect_events(backfill=backfill)
            if not events:
                status = "no_data"
                log.warning("calendar_ingest_no_events", backfill=backfill, errors=errors)
                return 0
            inserted = self._upsert(events)
            log.info(
                "calendar_ingest_complete",
                backfill=backfill,
                events=len(events),
                inserted=inserted,
                errors=errors,
            )
            return inserted
        except Exception as exc:
            status = "failed"
            errors += 1
            log.error("calendar_ingest_failed", error=str(exc))
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._log_ingestion(run_id, status, inserted, errors, duration_ms)

    # -- collection ---------------------------------------------------------

    def _collect_events(self, *, backfill: bool) -> tuple[list[_Event], int]:
        fred_events, fred_errors = self._fred_events(backfill=backfill)
        fomc_events, fomc_errors = self._fomc_events(backfill=backfill)
        return fred_events + fomc_events, fred_errors + fomc_errors

    def _fred_events(self, *, backfill: bool) -> tuple[list[_Event], int]:
        ids = self._cal.fred_release_ids
        if not ids:
            return [], 0
        if not self._api_key:
            log.warning("calendar_fred_no_api_key", releases=sorted(ids))
            return [], 1
        events: list[_Event] = []
        errors = 0
        for event_type, release_id in ids.items():
            try:
                dates = self._fetch_release_dates(int(release_id), backfill=backfill)
            except Exception as exc:  # noqa: BLE001 — fail-soft per release
                log.warning(
                    "calendar_fred_fetch_failed",
                    event_type=event_type,
                    release_id=release_id,
                    error=str(exc),
                )
                errors += 1
                continue
            for date_str in dates:
                try:
                    events.append(self._macro_event(event_type, date_str))
                except Exception as exc:  # noqa: BLE001 — skip a single bad date
                    log.warning(
                        "calendar_bad_release_date",
                        event_type=event_type,
                        value=date_str,
                        error=str(exc),
                    )
                    errors += 1
        return events, errors

    def _fetch_release_dates(self, release_id: int, *, backfill: bool) -> list[str]:
        """GET FRED ``release/dates`` and return the release-date strings (YYYY-MM-DD)."""
        url = f"{self._cal.fred_api_base_url}/release/dates"
        # include_release_dates_with_no_data=true (user ruling 2026-07-16): 'false' returned
        # only PAST release dates, defeating trade-time pre-stamping (D-T3.14) and leaving the
        # staleness alarm FOMC-only. 'true' adds the FORWARD scheduled release dates (verified
        # live: the only delta vs 'false' is the +5 future rows — no historical no-data noise,
        # no duplicates — so every returned date is a real release event and no filtering is
        # needed). Each future date ingests on the same path (08:30 ET->UTC window, source='fred').
        params: dict[str, str | int] = {
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
            "include_release_dates_with_no_data": "true",
        }
        if backfill:
            params["sort_order"] = "asc"
            params["realtime_start"] = self._cal.backfill_start
            params["realtime_end"] = _REALTIME_END_MAX
        else:
            params["sort_order"] = "desc"
            params["limit"] = self._cal.release_fetch_limit
        resp = httpx.get(url, params=params, timeout=self._cal.request_timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        return [row["date"] for row in payload.get("release_dates", [])]

    def _macro_event(self, event_type: str, date_str: str, source: str = "fred") -> _Event:
        scheduled_at = et_to_utc(date.fromisoformat(date_str.strip()), *_MACRO_PRINT_TIME_ET)
        window_start, window_end = self._window(event_type, scheduled_at)
        return _Event(event_type, None, scheduled_at, window_start, window_end, "high", source)

    def _fomc_events(self, *, backfill: bool) -> tuple[list[_Event], int]:
        raw = self._fomc_csv_dates() if backfill else list(self._cal.fomc_dates)
        events: list[_Event] = []
        errors = 0
        for value in raw:
            try:
                scheduled_at = _parse_fomc_datetime(value)
            except Exception as exc:  # noqa: BLE001 — skip a single bad date
                log.warning("calendar_bad_fomc_date", value=value, error=str(exc))
                errors += 1
                continue
            window_start, window_end = self._window("fomc", scheduled_at)
            events.append(
                _Event("fomc", None, scheduled_at, window_start, window_end, "high", "config")
            )
        return events, errors

    def _fomc_csv_dates(self) -> list[str]:
        path = Path(self._cal.fomc_backfill_csv)
        if not path.exists():
            log.warning("calendar_fomc_csv_missing", path=str(path))
            return []
        dates: list[str] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("date") or "").strip()
                if value and not value.startswith("#"):
                    dates.append(value)
        return dates

    def _window(self, event_type: str, scheduled_at: datetime) -> tuple[datetime, datetime]:
        before, after = self._cal.window_hours.get(event_type, list(_DEFAULT_WINDOW_HOURS))
        return scheduled_at - timedelta(hours=before), scheduled_at + timedelta(hours=after)

    # -- persistence --------------------------------------------------------

    def _upsert(self, events: list[_Event]) -> int:
        inserted = 0
        with self._db.connection() as conn:
            for event in events:
                cur = conn.execute(
                    _INSERT_SQL,
                    [
                        event.event_type,
                        event.symbol,
                        event.scheduled_at,
                        event.window_start,
                        event.window_end,
                        event.importance,
                        event.source,
                    ],
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += cur.rowcount
        return inserted

    def _log_ingestion(
        self, run_id: str, status: str, rows_inserted: int, errors_count: int, duration_ms: int
    ) -> None:
        """Log one ``data_ingestion_log`` row; a logging failure never crashes the ingest."""
        try:
            with self._db.connection() as conn:
                conn.execute(
                    _LOG_INSERT_SQL,
                    [
                        run_id,
                        _ENVIRONMENT,
                        _LOG_SYMBOL,
                        status,
                        rows_inserted,
                        errors_count,
                        duration_ms,
                        None,
                    ],
                )
        except Exception:  # noqa: BLE001
            log.warning("calendar_ingestion_log_failed", run_id=run_id, status=status)


def run_calendar_staleness_check(
    config: SwingRLConfig,
    db: DatabaseManager,
    alerter: Alerter,
    now: datetime | None = None,
) -> None:
    """Warn (Discord) when the forward calendar runs dry (spec §4 D-T3.14).

    Alerts when the newest future ``scheduled_at`` is nearer than
    ``calendar.min_future_days`` (or there are no future events at all) — a signal that the
    FOMC forward schedule / FRED ingest needs re-seeding. Fail-open: a query error is logged
    and swallowed (a broken query must not spam alerts). Uses ``bypass_suppression`` so a
    genuine gap reaches Discord on the first daily check rather than waiting N consecutive.

    Args:
        config: Validated SwingRLConfig (reads ``config.calendar``).
        db: DatabaseManager for the max(scheduled_at) probe.
        alerter: Collector Alerter for the warning (routing rule: collector sends).
        now: Injected clock for tests; defaults to ``datetime.now(UTC)``.
    """
    cal = config.calendar
    if not cal.enabled:
        return
    now = now or datetime.now(UTC)
    threshold = now + timedelta(days=cal.min_future_days)
    try:
        with db.connection() as conn:
            row = conn.execute(_MAX_FUTURE_SQL, [now]).fetchone()
        max_sched = row["max_sched"] if row else None
    except Exception:  # noqa: BLE001 — fail-open: never alert on a query fault
        log.warning("calendar_staleness_query_failed", exc_info=True)
        return
    if max_sched is not None and max_sched >= threshold:
        log.info(
            "calendar_fresh", newest=max_sched.isoformat(), min_future_days=cal.min_future_days
        )
        return
    detail = (
        "No future calendar events"
        if max_sched is None
        else f"Newest event {max_sched.isoformat()}"
    )
    alerter.send_alert(
        "warning",
        "Calendar events stale",
        f"{detail} is within {cal.min_future_days} days — re-ingest FRED/FOMC dates "
        "(collector calendar_ingest job).",
        environment="calendar",
        bypass_suppression=True,
    )
