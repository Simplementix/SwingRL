"""System Health -- traffic-light status, last trades, and heartbeat monitoring."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit pages need parent dir on path to import app.py helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Helper functions (pure logic, no st.* calls -- testable without Streamlit)
# ---------------------------------------------------------------------------

# Expected update windows in hours per environment
_EXPECTED_WINDOWS: dict[str, float] = {
    "equity": 26.0,
    "crypto": 5.0,
}


def get_traffic_light_status(last_snapshot_ts: str | None, env: str) -> str:
    """Return 'green', 'yellow', or 'red' based on snapshot staleness.

    Args:
        last_snapshot_ts: ISO-format UTC timestamp of latest portfolio snapshot,
            or None if no snapshots exist.
        env: Environment name ('equity' or 'crypto').

    Returns:
        Traffic-light color string: 'green', 'yellow', or 'red'.
    """
    if last_snapshot_ts is None:
        return "red"

    try:
        last_ts = datetime.fromisoformat(last_snapshot_ts.replace("Z", "+00:00"))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return "red"

    now = datetime.now(UTC)
    hours_ago = (now - last_ts).total_seconds() / 3600.0

    window = _EXPECTED_WINDOWS.get(env, 26.0)

    if hours_ago <= window:
        return "green"
    elif hours_ago <= window * 2:
        return "yellow"
    else:
        return "red"


def get_latest_trades(conn: Any, limit: int = 5) -> list[dict]:
    """Query trade_log for most recent trades.

    Args:
        conn: PostgreSQL connection to swingrl database.
        limit: Maximum number of trades to return.

    Returns:
        List of trade dicts ordered by timestamp descending.
    """
    cursor = conn.execute(
        "SELECT trade_id, timestamp, environment, symbol, side, quantity, fill_price "
        "FROM trade_log ORDER BY timestamp DESC LIMIT %s",
        (limit,),
    )
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------

st.header("System Health")

try:
    from app import get_pg_conn

    conn = get_pg_conn()

    # Traffic-light status per environment
    st.subheader("Environment Status")
    cursor = conn.execute(
        "SELECT environment, MAX(timestamp) AS last_ts "
        "FROM portfolio_snapshots GROUP BY environment"
    )
    env_timestamps = {row["environment"]: row["last_ts"] for row in cursor.fetchall()}

    if env_timestamps:
        cols = st.columns(len(env_timestamps))
        for i, (env, last_ts) in enumerate(env_timestamps.items()):
            status = get_traffic_light_status(last_ts, env)
            with cols[i]:
                st.markdown(f"### :{status}_circle: {env.capitalize()}")
                st.caption(f"Last update: {last_ts}")
    else:
        st.warning("No environment data available. Waiting for first trading cycle.")

    # Last heartbeat times
    st.subheader("Heartbeat")
    if env_timestamps:
        for env, last_ts in env_timestamps.items():
            st.metric(f"{env.capitalize()} Last Snapshot", last_ts or "N/A")
    else:
        st.info("No heartbeat data yet.")

    # System uptime indicator
    st.subheader("System Uptime")
    cursor = conn.execute(
        "SELECT MIN(timestamp) AS earliest, MAX(timestamp) AS latest FROM portfolio_snapshots"
    )
    row = cursor.fetchone()
    if row and row["earliest"] and row["latest"]:
        st.metric("Tracking Since", row["earliest"])
        st.metric("Latest Snapshot", row["latest"])
    else:
        st.info("No uptime data available.")

    # Last 5 trades
    st.subheader("Recent Trades")
    trades = get_latest_trades(conn, limit=5)
    if trades:
        import pandas as pd

        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    else:
        st.info("No trades recorded yet.")

    conn.close()

except Exception as exc:
    st.warning(f"Could not load system health data: {exc}")
