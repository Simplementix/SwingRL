"""Stuck agent detection logic.

Detects environments that have been holding 100% cash for an extended
period by querying portfolio_snapshots. Thresholds: 10 for equity
(trading day snapshots), 30 for crypto (4H cycle snapshots).

Usage:
    from swingrl.monitoring.stuck_agent import check_stuck_agents
    alerts = check_stuck_agents(db)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Thresholds: equity snapshots are per trading day, crypto per 4H cycle
_THRESHOLDS: dict[str, int] = {
    "equity": 10,
    "crypto": 30,
}


def check_stuck_agents(db: DatabaseManager) -> list[dict[str, object]]:
    """Check for environments stuck in all-cash positions.

    For each environment, queries the most recent N portfolio snapshots
    (where N = threshold). If all snapshots show cash_balance approximately
    equal to total_value, the environment is considered stuck.

    Args:
        db: DatabaseManager providing SQLite connection.

    Returns:
        List of alert dicts with environment, consecutive_cash_cycles,
        and last_action_date for each stuck environment.
    """
    alerts: list[dict[str, object]] = []

    with db.sqlite() as conn:
        for env, threshold in _THRESHOLDS.items():
            rows = conn.execute(
                "SELECT cash_balance, total_value FROM portfolio_snapshots "
                "WHERE environment = ? ORDER BY timestamp DESC LIMIT ?",
                (env, threshold),
            ).fetchall()

            if len(rows) < threshold:
                continue

            all_cash = all(abs(row["cash_balance"] - row["total_value"]) < 0.01 for row in rows)

            if not all_cash:
                continue

            # Query last non-cash snapshot date for diagnostics
            last_action_row = conn.execute(
                "SELECT timestamp FROM portfolio_snapshots "
                "WHERE environment = ? AND abs(cash_balance - total_value) >= 0.01 "
                "ORDER BY timestamp DESC LIMIT 1",
                (env,),
            ).fetchone()

            last_action_date: str | None = None
            if last_action_row is not None:
                last_action_date = last_action_row["timestamp"]

            log.warning(
                "stuck_agent_detected",
                environment=env,
                consecutive_cash_cycles=len(rows),
                last_action_date=last_action_date,
            )

            alerts.append(
                {
                    "environment": env,
                    "consecutive_cash_cycles": len(rows),
                    "last_action_date": last_action_date,
                }
            )

    return alerts
