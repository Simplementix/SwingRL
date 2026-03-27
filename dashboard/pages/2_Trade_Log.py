"""Trade Log -- filterable trade history with summary statistics."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit pages need parent dir on path to import app.py helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Helper functions (pure logic, no st.* calls)
# ---------------------------------------------------------------------------


def fetch_trades(
    conn: sqlite3.Connection,
    environment: str | None = None,
    side: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Query trade_log with optional filters and return a DataFrame."""
    query = "SELECT * FROM trade_log WHERE 1=1"
    params: list[str] = []

    if environment and environment != "All":
        query += " AND environment = ?"
        params.append(environment)
    if side and side != "All":
        query += " AND side = ?"
        params.append(side)
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)

    query += " ORDER BY timestamp DESC"
    return pd.read_sql_query(query, conn, params=params)


def compute_trade_stats(df: pd.DataFrame) -> dict[str, object]:
    """Compute summary statistics from trade DataFrame."""
    if df.empty:
        return {"total_trades": 0, "symbols": [], "buy_count": 0, "sell_count": 0}
    return {
        "total_trades": len(df),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "buy_count": int((df["side"] == "buy").sum()),
        "sell_count": int((df["side"] == "sell").sum()),
    }


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------

st.header("Trade Log")

try:
    from app import get_sqlite_conn

    conn = get_sqlite_conn()

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        env_filter = st.selectbox("Environment", ["All", "equity", "crypto"])
    with col2:
        side_filter = st.selectbox("Side", ["All", "buy", "sell"])
    with col3:
        start_date = st.date_input("Start Date", value=None)
    with col4:
        end_date = st.date_input("End Date", value=None)

    df = fetch_trades(
        conn,
        environment=env_filter,
        side=side_filter,
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
    )
    conn.close()

    if df.empty:
        st.info("No trades recorded yet.")
    else:
        # Summary stats
        stats = compute_trade_stats(df)
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Total Trades", stats["total_trades"])
        with stat_cols[1]:
            st.metric("Buys", stats["buy_count"])
        with stat_cols[2]:
            st.metric("Sells", stats["sell_count"])

        if stats["symbols"]:
            st.caption(f"Symbols traded: {', '.join(stats['symbols'])}")

        # Trade table
        st.dataframe(df, use_container_width=True)

except Exception as exc:
    st.warning(f"Could not load trade data: {exc}")
