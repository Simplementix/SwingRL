"""Corporate action detection and recording.

Detects overnight price spikes that may indicate stock splits or other
corporate actions. Records actions to the corporate_actions table and
suppresses false-positive quarantine for known actions.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Overnight spike thresholds by environment
_EQUITY_SPIKE_THRESHOLD = 0.30  # 30% for equity
_CRYPTO_SPIKE_THRESHOLD = 0.40  # 40% for crypto


class CorporateActionDetector:
    """Detect and record corporate actions (splits, dividends).

    Uses overnight price spike heuristic to detect potential unrecorded
    corporate actions. Known actions in the corporate_actions table
    suppress false-positive quarantine.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def detect_overnight_spike(
        self,
        symbol: str,
        prev_close: float,
        curr_close: float,
        date: str,
        environment: Literal["equity", "crypto"] = "equity",
    ) -> bool:
        """Detect >threshold% overnight price change.

        Returns True if a spike is detected AND the action is NOT already
        recorded in the corporate_actions table.

        Args:
            symbol: Ticker symbol.
            prev_close: Previous day's closing price.
            curr_close: Current day's closing price.
            date: Date of the current bar (YYYY-MM-DD).
            environment: Trading environment for threshold selection.

        Returns:
            True if potential unrecorded corporate action detected.
        """
        if prev_close == 0:
            return False

        pct_change = abs((curr_close - prev_close) / prev_close)
        threshold = _EQUITY_SPIKE_THRESHOLD if environment == "equity" else _CRYPTO_SPIKE_THRESHOLD

        if pct_change <= threshold:
            return False

        # Check if this is a known action
        if self.is_known_action(symbol, date):
            log.info(
                "corporate_action_known",
                symbol=symbol,
                date=date,
                pct_change=round(pct_change, 4),
            )
            return False

        log.warning(
            "overnight_spike_detected",
            symbol=symbol,
            date=date,
            pct_change=round(pct_change, 4),
            threshold=threshold,
            environment=environment,
        )
        return True

    def record_action(
        self,
        symbol: str,
        action_type: str,
        effective_date: str,
        ratio: float | None = None,
        amount: float | None = None,
    ) -> None:
        """Insert a corporate action into the corporate_actions table.

        Args:
            symbol: Ticker symbol.
            action_type: Type of action (e.g., "split", "dividend").
            effective_date: Date the action takes effect (YYYY-MM-DD).
            ratio: Split ratio (e.g., 2.0 for 2:1 split).
            amount: Dividend amount per share.
        """
        action_id = str(uuid.uuid4())
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO corporate_actions "
                "(action_id, symbol, action_type, effective_date, ratio, amount, processed) "
                "VALUES (%s, %s, %s, %s, %s, %s, 0)",
                [action_id, symbol, action_type, effective_date, ratio, amount],
            )
        log.info(
            "corporate_action_recorded",
            action_id=action_id,
            symbol=symbol,
            action_type=action_type,
            effective_date=effective_date,
            ratio=ratio,
            amount=amount,
        )

    def is_known_action(self, symbol: str, date: str) -> bool:
        """Check if a corporate action exists for the given symbol and date.

        Args:
            symbol: Ticker symbol.
            date: Date to check (YYYY-MM-DD).

        Returns:
            True if a matching corporate action exists.
        """
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM corporate_actions WHERE symbol = %s AND effective_date = %s LIMIT 1",
                [symbol, date],
            ).fetchone()
        return row is not None

    def check_and_suppress(
        self,
        symbol: str,
        date: str,
        pct_change: float,
        environment: Literal["equity", "crypto"] = "equity",
    ) -> bool:
        """Check if an overnight spike should suppress quarantine.

        Returns True if the spike exceeds the threshold AND a known
        corporate action exists (meaning quarantine should be suppressed).

        Args:
            symbol: Ticker symbol.
            date: Date of the price change (YYYY-MM-DD).
            pct_change: Absolute percentage change (0.50 = 50%).
            environment: Trading environment for threshold selection.

        Returns:
            True if quarantine should be suppressed (known corporate action).
        """
        threshold = _EQUITY_SPIKE_THRESHOLD if environment == "equity" else _CRYPTO_SPIKE_THRESHOLD

        if pct_change <= threshold:
            return False

        if self.is_known_action(symbol, date):
            log.info(
                "quarantine_suppressed",
                symbol=symbol,
                date=date,
                pct_change=round(pct_change, 4),
                reason="known_corporate_action",
            )
            return True

        return False
