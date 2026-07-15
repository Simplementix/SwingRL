# src/swingrl/data/options/market_calendar.py
"""NYSE (XNYS) trading-day and early-close helpers (spec §6.1, §9.2)."""

from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd

_CALENDAR_NAME = "XNYS"
_REGULAR_CLOSE_HOUR_ET = 16
_calendar: xcals.ExchangeCalendar | None = None


def _cal() -> xcals.ExchangeCalendar:
    global _calendar
    if _calendar is None:
        _calendar = xcals.get_calendar(_CALENDAR_NAME)
    return _calendar


def is_trading_day(quote_date: date) -> bool:
    """True if quote_date is an NYSE session (excludes weekends + holidays)."""
    return bool(_cal().is_session(pd.Timestamp(quote_date)))


def is_early_close(quote_date: date) -> bool:
    """True if quote_date is an NYSE half-day (regular close is 16:00 ET)."""
    session = pd.Timestamp(quote_date)
    if not _cal().is_session(session):
        return False
    close_et = _cal().session_close(session).tz_convert("America/New_York")
    return bool(close_et.hour < _REGULAR_CLOSE_HOUR_ET)


def recent_sessions(as_of: date, n: int) -> list[date]:
    """The last n NYSE sessions ending at (and including, if a session) as_of."""
    ts = pd.Timestamp(as_of)
    sessions = _cal().sessions_in_range(ts - pd.Timedelta(days=n * 3 + 10), ts)
    return [s.date() for s in sessions[-n:]]
