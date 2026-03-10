"""SwingRL Dashboard -- Multi-page Streamlit entry point with auto-refresh."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import duckdb
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Page config and auto-refresh
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SwingRL Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st_autorefresh(interval=300_000, key="dashboard_refresh")  # 5-minute refresh

# ---------------------------------------------------------------------------
# DB path configuration
# ---------------------------------------------------------------------------

DB_DIR: Path = Path(os.environ.get("SWINGRL_DB_DIR", "db"))


def get_sqlite_conn() -> sqlite3.Connection:
    """Return a read-only SQLite connection to trading_ops.db."""
    db_path = DB_DIR / "trading_ops.db"
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """Return a read-only DuckDB connection to market_data.ddb."""
    db_path = DB_DIR / "market_data.ddb"
    return duckdb.connect(str(db_path), read_only=True)


# ---------------------------------------------------------------------------
# Main page content
# ---------------------------------------------------------------------------

st.title("SwingRL Dashboard")

st.sidebar.header("Navigation")
st.sidebar.markdown(
    """
- **Portfolio** -- equity curves and P&L
- **Trade Log** -- filterable trade history
- **Risk Metrics** -- drawdown and circuit breakers
- **System Health** -- traffic-light status
"""
)

# Sidebar: quick-glance system status
st.sidebar.divider()
st.sidebar.subheader("System Status")

try:
    conn = get_sqlite_conn()
    cursor = conn.execute(
        "SELECT environment, MAX(timestamp) AS last_ts "
        "FROM portfolio_snapshots GROUP BY environment"
    )
    rows = cursor.fetchall()
    conn.close()
    if rows:
        for env, last_ts in rows:
            st.sidebar.text(f"{env.capitalize()}: {last_ts}")
    else:
        st.sidebar.info("No portfolio data yet")
except Exception:
    st.sidebar.warning("DB not available")

st.markdown(
    "Use the sidebar pages to explore portfolio performance, trade history, "
    "risk metrics, and system health."
)
