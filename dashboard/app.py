"""SwingRL Dashboard -- Multi-page Streamlit entry point with auto-refresh."""

from __future__ import annotations

import os

import psycopg
import streamlit as st
from psycopg.rows import dict_row
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
# DB connection
# ---------------------------------------------------------------------------


def get_pg_conn() -> psycopg.Connection:
    """Return a PostgreSQL connection to the swingrl database."""
    url = os.environ.get("DATABASE_URL", "postgresql://swingrl:changeme@localhost:5432/swingrl")
    return psycopg.connect(url, row_factory=dict_row)


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
    conn = get_pg_conn()
    cursor = conn.execute(
        "SELECT environment, MAX(timestamp) AS last_ts "
        "FROM portfolio_snapshots GROUP BY environment"
    )
    rows = cursor.fetchall()
    conn.close()
    if rows:
        for row in rows:
            st.sidebar.text(f"{row['environment'].capitalize()}: {row['last_ts']}")
    else:
        st.sidebar.info("No portfolio data yet")
except Exception:
    st.sidebar.warning("DB not available")

st.markdown(
    "Use the sidebar pages to explore portfolio performance, trade history, "
    "risk metrics, and system health."
)
