"""Portfolio Overview -- equity curves and P&L per environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit pages need parent dir on path to import app.py helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Helper functions (pure logic, no st.* calls)
# ---------------------------------------------------------------------------


def fetch_portfolio_snapshots(conn: Any) -> pd.DataFrame:
    """Return all portfolio_snapshots as a DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC",
        conn,
    )
    return df


def compute_summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute latest total value, daily P&L, and HWM per environment."""
    if df.empty:
        return {}
    metrics: dict[str, Any] = {}
    for env in df["environment"].unique():
        env_df = df[df["environment"] == env].sort_values("timestamp")
        latest = env_df.iloc[-1]
        metrics[env] = {
            "total_value": latest["total_value"],
            "daily_pnl": latest.get("daily_pnl", 0.0) or 0.0,
            "high_water_mark": latest.get("high_water_mark", 0.0) or 0.0,
        }
    return metrics


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------

st.header("Portfolio Overview")

try:
    from app import get_pg_conn

    conn = get_pg_conn()
    df = fetch_portfolio_snapshots(conn)
    conn.close()

    if df.empty:
        st.info("No portfolio data yet. Data will appear after the first trading cycle.")
    else:
        # Summary metrics
        metrics = compute_summary_metrics(df)
        cols = st.columns(len(metrics))
        for i, (env, m) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=f"{env.capitalize()} Value",
                    value=f"${m['total_value']:,.2f}",
                    delta=f"${m['daily_pnl']:,.2f}",
                )
                st.caption(f"HWM: ${m['high_water_mark']:,.2f}")

        # Equity curve per environment
        st.subheader("Equity Curves")
        fig_env = px.line(
            df,
            x="timestamp",
            y="total_value",
            color="environment",
            title="Portfolio Value by Environment",
            labels={"total_value": "Value ($)", "timestamp": "Date"},
        )
        st.plotly_chart(fig_env, use_container_width=True)

        # Combined portfolio value
        combined = df.groupby("timestamp")["total_value"].sum().reset_index()
        fig_combined = px.line(
            combined,
            x="timestamp",
            y="total_value",
            title="Combined Portfolio Value",
            labels={"total_value": "Value ($)", "timestamp": "Date"},
        )
        st.plotly_chart(fig_combined, use_container_width=True)

        # Daily P&L bar chart
        if "daily_pnl" in df.columns:
            pnl_df = df[df["daily_pnl"].notna()]
            if not pnl_df.empty:
                st.subheader("Daily P&L")
                fig_pnl = px.bar(
                    pnl_df,
                    x="timestamp",
                    y="daily_pnl",
                    color="environment",
                    title="Daily P&L by Environment",
                    labels={"daily_pnl": "P&L ($)", "timestamp": "Date"},
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

except Exception as exc:
    st.warning(f"Could not load portfolio data: {exc}")
