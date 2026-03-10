"""Risk Metrics -- drawdown tracking, circuit breaker status, and risk decisions."""

from __future__ import annotations

import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Helper functions (pure logic, no st.* calls)
# ---------------------------------------------------------------------------


def get_current_drawdown(conn: sqlite3.Connection) -> dict[str, float]:
    """Return latest drawdown_pct per environment."""
    cursor = conn.execute(
        "SELECT environment, drawdown_pct FROM portfolio_snapshots "
        "WHERE (environment, timestamp) IN ("
        "  SELECT environment, MAX(timestamp) FROM portfolio_snapshots "
        "  GROUP BY environment"
        ")"
    )
    return {row[0]: row[1] or 0.0 for row in cursor.fetchall()}


def drawdown_color(pct: float) -> str:
    """Return color name based on drawdown severity."""
    abs_pct = abs(pct)
    if abs_pct < 5.0:
        return "green"
    elif abs_pct < 8.0:
        return "yellow"
    else:
        return "red"


def fetch_drawdown_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return drawdown_pct time series for all environments."""
    return pd.read_sql_query(
        "SELECT timestamp, environment, drawdown_pct "
        "FROM portfolio_snapshots "
        "WHERE drawdown_pct IS NOT NULL "
        "ORDER BY timestamp ASC",
        conn,
    )


def fetch_circuit_breaker_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return circuit breaker events, most recent first."""
    return pd.read_sql_query(
        "SELECT * FROM circuit_breaker_events ORDER BY timestamp DESC",
        conn,
    )


def get_cb_status(conn: sqlite3.Connection) -> dict[str, str]:
    """Return circuit breaker status per environment (Active with cooldown or All Clear)."""
    cursor = conn.execute(
        "SELECT environment, cooldown_end FROM circuit_breaker_events "
        "WHERE (environment, timestamp) IN ("
        "  SELECT environment, MAX(timestamp) FROM circuit_breaker_events "
        "  GROUP BY environment"
        ")"
    )
    status: dict[str, str] = {}
    for env, cooldown_end in cursor.fetchall():
        if cooldown_end:
            status[env] = f"Active (cooldown until {cooldown_end})"
        else:
            status[env] = "All Clear"
    return status


def fetch_risk_decisions(conn: sqlite3.Connection, limit: int = 20) -> pd.DataFrame:
    """Return most recent risk decisions."""
    return pd.read_sql_query(
        "SELECT * FROM risk_decisions ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(limit,),
    )


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------

st.header("Risk Metrics")

try:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import get_sqlite_conn

    conn = get_sqlite_conn()

    # Current drawdown per environment
    st.subheader("Current Drawdown")
    drawdowns = get_current_drawdown(conn)
    if drawdowns:
        cols = st.columns(len(drawdowns))
        for i, (env, dd) in enumerate(drawdowns.items()):
            color = drawdown_color(dd)
            with cols[i]:
                st.metric(f"{env.capitalize()} Drawdown", f"{dd:.2f}%")
                st.markdown(f":{color}_circle: Status")
    else:
        st.info("No drawdown data available yet.")

    # Drawdown history chart
    st.subheader("Drawdown History")
    dd_df = fetch_drawdown_history(conn)
    if not dd_df.empty:
        fig = px.line(
            dd_df,
            x="timestamp",
            y="drawdown_pct",
            color="environment",
            title="Drawdown Over Time",
            labels={"drawdown_pct": "Drawdown (%)", "timestamp": "Date"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drawdown history available.")

    # Circuit breaker status
    st.subheader("Circuit Breaker Status")
    cb_status = get_cb_status(conn)
    if cb_status:
        for env, status in cb_status.items():
            icon = ":green_circle:" if "All Clear" in status else ":red_circle:"
            st.markdown(f"{icon} **{env.capitalize()}**: {status}")
    else:
        st.success("No circuit breaker events recorded. All clear.")

    # Circuit breaker events table
    cb_df = fetch_circuit_breaker_events(conn)
    if not cb_df.empty:
        st.subheader("Circuit Breaker Events")
        st.dataframe(cb_df, use_container_width=True)

    # Risk decisions
    st.subheader("Recent Risk Decisions")
    risk_df = fetch_risk_decisions(conn)
    if not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
    else:
        st.info("No risk decisions recorded yet.")

    conn.close()

except Exception as exc:
    st.warning(f"Could not load risk data: {exc}")
