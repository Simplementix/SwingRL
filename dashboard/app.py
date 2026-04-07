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


def _open_pg_connection() -> psycopg.Connection:
    """Open a fresh PostgreSQL connection from the configured DATABASE_URL."""
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
    )
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


@st.cache_resource
def _cached_pg_conn() -> psycopg.Connection:
    """Streamlit cache_resource holder for the singleton connection."""
    return _open_pg_connection()


def get_pg_conn() -> psycopg.Connection:
    """Return a healthy PostgreSQL connection for the swingrl database.

    Uses Streamlit's cache_resource to reuse the connection across page renders
    when possible, but transparently reconnects if a previous page closed the
    cached singleton (which all pages 1-4 do at the end of their render). This
    self-heal removes a footgun without requiring the consumer pages to change.
    """
    conn = _cached_pg_conn()
    if conn.closed:
        # Pre-existing pages call ``conn.close()`` at the end of their render,
        # which closes the cached singleton. The next page load (or this one
        # after a re-run) gets the closed connection — recover by clearing
        # the cache and opening a fresh one.
        _cached_pg_conn.clear()
        conn = _cached_pg_conn()
    return conn


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
