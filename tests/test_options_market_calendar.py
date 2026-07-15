# tests/test_options_market_calendar.py
from __future__ import annotations

from datetime import date

from swingrl.data.options import market_calendar as mc


def test_weekend_is_not_trading_day() -> None:
    """OPT-COLLECT-1: Saturday is not a trading day (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 7, 18)) is False  # Saturday


def test_regular_weekday_is_trading_day() -> None:
    """OPT-COLLECT-2: a normal Tuesday is a trading day (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 7, 14)) is True


def test_christmas_is_not_trading_day() -> None:
    """OPT-COLLECT-3: NYSE holiday skipped (spec §9.2)."""
    assert mc.is_trading_day(date(2026, 12, 25)) is False


def test_black_friday_is_early_close() -> None:
    """OPT-COLLECT-4: half-day detected as early close (spec §6.1)."""
    # 2026-11-27 (day after Thanksgiving) is a 13:00 ET early close.
    assert mc.is_early_close(date(2026, 11, 27)) is True
    assert mc.is_early_close(date(2026, 7, 14)) is False
