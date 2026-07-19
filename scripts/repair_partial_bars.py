"""Repair partial 4H bars — RED stub (implementation follows in GREEN)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

BAR_INTERVAL: timedelta = timedelta(hours=4)


def classify_partial_bars(
    rows: pd.DataFrame,
    now: datetime,
    interval: timedelta = BAR_INTERVAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ohlcv_4h rows into (repairable_partials, unclosed_partials)."""
    raise NotImplementedError


class PartialBarRepairer:
    """Detect and repair partial 4H bars (stub)."""

    def __init__(self, gateway: object, ingestor: object, **kwargs: object) -> None:
        raise NotImplementedError

    def run(self, *, apply: bool) -> list:
        raise NotImplementedError
