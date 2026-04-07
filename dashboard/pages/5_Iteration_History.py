"""Iteration History — CPS trends, per-fold heatmap, and regression panel.

Phase 0.7 of the memory agent refocus. The first dashboard page that reads
``iteration_results`` (until now the table was write-only).

Sections:
1. Iteration table — one row per (env, iter): CPS v1/v2/v3, deltas, return,
   worst MDD, winners, chronic count, pass rate, regression flag.
2. CPS trend chart — line chart per environment, three CPS formula lines
   plus a legacy pass-rate trace on a secondary axis. Visualizes the
   "pass rate vs CPS divergence" anti-pattern when it occurs.
3. Per-fold heatmap — fold_number on the y-axis, iteration_number on the
   x-axis, color = OOS Sharpe (red = negative, green = high). Chronic
   failure folds appear as horizontal red bands.
4. Treatment vs control panel — for iter 3+, shows treatment-only vs
   control-only CPS side-by-side. The smoking-gun finding from Phase 0.4
   (control consistently outperforms treatment) is immediately visible.
5. Regression panel — filters to iterations where ``regression_flag=True``,
   shows which dimensions tripped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Streamlit pages need parent dir on path to import app.py helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add src/ so the swingrl package is importable when running locally outside Docker
_REPO_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from swingrl.reporting.iteration_report import (  # noqa: E402
    compute_iter_deltas,
    detect_chronic_failures,
    detect_protected_winners,
    load_fold_history,
    load_iteration_history,
)

# ---------------------------------------------------------------------------
# Helper functions (pure logic, no st.* calls — unit-testable)
# ---------------------------------------------------------------------------


def fetch_iteration_history(conn: Any, env: str, n: int = 10) -> pd.DataFrame:
    """Read iteration_results for one environment (delegates to iteration_report)."""
    return load_iteration_history(conn, env=env, n=n)


def fetch_fold_history(conn: Any, env: str) -> pd.DataFrame:
    """Read deduped per-fold backtest_results for one environment."""
    return load_fold_history(conn, env=env)


def build_cps_trend_figure(history_with_deltas: pd.DataFrame, env: str) -> go.Figure | None:
    """Build a CPS trend chart with the three formulas plus legacy pass-rate axis.

    Returns None when there's no data to render.
    """
    env_df = history_with_deltas[history_with_deltas["environment"] == env]
    if env_df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Three CPS formulas on the primary axis
    fig.add_trace(
        go.Scatter(
            x=env_df["iteration_number"],
            y=env_df["cps_v1_multiplicative"],
            mode="lines+markers",
            name="CPS v1 (multiplicative)",
            line={"width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=env_df["iteration_number"],
            y=env_df["cps_v2_additive"],
            mode="lines+markers",
            name="CPS v2 (additive)",
            line={"dash": "dot"},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=env_df["iteration_number"],
            y=env_df["cps_v3_sortino"],
            mode="lines+markers",
            name="CPS v3 (sortino)",
            line={"dash": "dash"},
        ),
        secondary_y=False,
    )

    # Legacy gate pass rate on the secondary axis. ``gate_passed`` is a
    # bool per (iter, env) — for the trend we just plot 0/1 to show the
    # divergence pattern visually. Real per-iteration pass rate would
    # require aggregating per-fold pass/fail.
    if "gate_passed" in env_df.columns:
        fig.add_trace(
            go.Scatter(
                x=env_df["iteration_number"],
                y=env_df["gate_passed"].astype(float),
                mode="lines+markers",
                name="Gate passed (legacy)",
                line={"color": "gray", "dash": "dashdot"},
                opacity=0.6,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"CPS trend — {env}",
        xaxis_title="Iteration",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "left", "x": 0},
        height=420,
    )
    fig.update_yaxes(title_text="CPS value", secondary_y=False)
    fig.update_yaxes(
        title_text="Gate passed (0/1)",
        secondary_y=True,
        range=[-0.1, 1.1],
        showgrid=False,
    )
    return fig


def build_fold_heatmap(fold_history: pd.DataFrame, env: str) -> go.Figure | None:
    """Build a per-fold Sharpe heatmap (rows=fold_number, cols=iteration).

    Aggregates across algos by taking the MAX sharpe per (iteration, fold).
    Chronic-failure folds appear as horizontal red bands.
    """
    env_df = fold_history[fold_history["environment"] == env]
    if env_df.empty:
        return None

    # Per (iter, fold) max sharpe across algos
    pivot = (
        env_df.groupby(["iteration_number", "fold_number"])["sharpe"]
        .max()
        .reset_index()
        .pivot(index="fold_number", columns="iteration_number", values="sharpe")
        .sort_index()
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=4,
        aspect="auto",
        labels={"x": "Iteration", "y": "Fold", "color": "OOS Sharpe (max across algos)"},
    )
    fig.update_layout(
        title=f"Per-fold Sharpe heatmap — {env}",
        xaxis_title="Iteration",
        yaxis_title="Fold number",
        height=520,
    )
    return fig


def build_treatment_vs_control_table(history: pd.DataFrame) -> pd.DataFrame:
    """Extract a treatment-vs-control comparison table for iter 3+ rows.

    Returns a DataFrame with one row per (env, iter) where both columns are
    populated. Empty if no qualifying rows exist.
    """
    cols_required = ("cps_v1_treatment_only", "cps_v1_control_only")
    for col in cols_required:
        if col not in history.columns:
            return pd.DataFrame()

    mask = history["cps_v1_treatment_only"].notna() & history["cps_v1_control_only"].notna()
    subset = history.loc[
        mask,
        [
            "iteration_number",
            "environment",
            "cps_v1_treatment_only",
            "cps_v1_control_only",
            "cps_v1_multiplicative",
        ],
    ].copy()
    if subset.empty:
        return subset

    # Compute the divergence ratio (control / treatment) for the smoking-gun signal
    subset["control_over_treatment"] = (
        subset["cps_v1_control_only"] / subset["cps_v1_treatment_only"]
    )
    subset = subset.rename(
        columns={
            "iteration_number": "iter",
            "environment": "env",
            "cps_v1_treatment_only": "treatment_v1",
            "cps_v1_control_only": "control_v1",
            "cps_v1_multiplicative": "ensemble_v1",
        }
    )
    return subset.sort_values(["env", "iter"]).reset_index(drop=True)


def build_iteration_table(history_with_deltas: pd.DataFrame) -> pd.DataFrame:
    """Project the columns we want to display in the iteration table."""
    keep_cols = [
        "iteration_number",
        "environment",
        "cps_v1_multiplicative",
        "cps_v1_delta",
        "cps_v2_additive",
        "cps_v3_sortino",
        "median_return",
        "return_delta",
        "worst_fold_mdd",
        "worst_mdd_delta",
        "winners_count",
        "chronic_failure_count",
        "gate_passed",
        "regression_flag",
        "dedup_rows_dropped",
    ]
    present = [c for c in keep_cols if c in history_with_deltas.columns]
    return history_with_deltas[present].sort_values(["environment", "iteration_number"])


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------

st.header("Iteration History")
st.caption(
    "Capital Preservation Score trends, per-fold performance heatmap, and "
    "regression detection. Reads `iteration_results` and `backtest_results`."
)

try:
    from app import get_pg_conn

    conn = get_pg_conn()

    envs = ("equity", "crypto")
    histories = {env: fetch_iteration_history(conn, env=env, n=20) for env in envs}
    fold_histories = {env: fetch_fold_history(conn, env=env) for env in envs}

    combined_history = pd.concat(
        [df for df in histories.values() if not df.empty], ignore_index=True
    )

    if combined_history.empty:
        st.info("No iteration_results rows yet. Run training or the CPS backfill script.")
    else:
        history_with_deltas = compute_iter_deltas(combined_history)

        # ------------------------------------------------------------------
        # Section 1: iteration table
        # ------------------------------------------------------------------
        st.subheader("Iteration Table")
        table = build_iteration_table(history_with_deltas)
        st.dataframe(
            table,
            use_container_width=True,
            column_config={
                "cps_v1_multiplicative": st.column_config.NumberColumn("CPS v1", format="%.5f"),
                "cps_v1_delta": st.column_config.NumberColumn("CPS v1 Δ", format="%+.5f"),
                "cps_v2_additive": st.column_config.NumberColumn("CPS v2", format="%.5f"),
                "cps_v3_sortino": st.column_config.NumberColumn("CPS v3", format="%.5f"),
                "median_return": st.column_config.NumberColumn("Return", format="%.2%%"),
                "return_delta": st.column_config.NumberColumn("Return Δ", format="%+.2%%"),
                "worst_fold_mdd": st.column_config.NumberColumn("Worst MDD", format="%.2%%"),
                "worst_mdd_delta": st.column_config.NumberColumn("Worst MDD Δ", format="%+.2%%"),
                "regression_flag": st.column_config.CheckboxColumn("Regression"),
            },
        )

        # ------------------------------------------------------------------
        # Section 2: CPS trend charts (one per environment)
        # ------------------------------------------------------------------
        st.subheader("CPS Trend by Environment")
        st.markdown(
            "Three CPS formulas on the primary axis; legacy gate-passed flag "
            "on the secondary axis. **A divergence between CPS dropping and "
            "gate continuing to pass is the iter-5-style anti-pattern this "
            "page is designed to surface.**"
        )
        for env in envs:
            fig = build_cps_trend_figure(history_with_deltas, env)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # Section 3: per-fold heatmap
        # ------------------------------------------------------------------
        st.subheader("Per-Fold Sharpe Heatmap")
        st.markdown(
            "Each cell is the **max OOS Sharpe across algos** for one "
            "(iteration, fold). Chronic failure folds (e.g., equity 2/4/7/13/15) "
            "appear as horizontal red bands."
        )
        for env in envs:
            heatmap = build_fold_heatmap(fold_histories[env], env)
            if heatmap is not None:
                st.plotly_chart(heatmap, use_container_width=True)

        # ------------------------------------------------------------------
        # Section 4: chronic failures + protected winners
        # ------------------------------------------------------------------
        st.subheader("Chronic Failures & Protected Winners")
        col_chronic, col_winners = st.columns(2)
        with col_chronic:
            st.markdown("**Chronic failures (4+ of last 6 iters)**")
            for env in envs:
                chronics = detect_chronic_failures(fold_histories[env])
                folds = chronics.get(env, [])
                if folds:
                    st.markdown(f"- `{env}`: {folds}")
                else:
                    st.markdown(f"- `{env}`: _none detected_")
        with col_winners:
            st.markdown("**Protected winners (Sharpe>4 in 4+ iters)**")
            for env in envs:
                winners = detect_protected_winners(fold_histories[env])
                folds = winners.get(env, [])
                if folds:
                    st.markdown(f"- `{env}`: {folds}")
                else:
                    st.markdown(f"- `{env}`: _none detected_")

        # ------------------------------------------------------------------
        # Section 5: treatment vs control
        # ------------------------------------------------------------------
        st.subheader("Treatment vs Control Split (iter 3+)")
        st.markdown(
            "Iter 3 introduced control folds — folds that receive **no LLM "
            "advice** but are still trained. If the memory system is "
            "actively helping, treatment CPS should exceed control CPS. "
            "If the ratio control/treatment is **>1.0**, memory is hurting "
            "more than helping."
        )
        tc_table = build_treatment_vs_control_table(history_with_deltas)
        if tc_table.empty:
            st.info("No treatment/control split data yet — populated for iter 3+.")
        else:
            st.dataframe(
                tc_table,
                use_container_width=True,
                column_config={
                    "treatment_v1": st.column_config.NumberColumn("Treatment v1", format="%.5f"),
                    "control_v1": st.column_config.NumberColumn("Control v1", format="%.5f"),
                    "ensemble_v1": st.column_config.NumberColumn("Ensemble v1", format="%.5f"),
                    "control_over_treatment": st.column_config.NumberColumn(
                        "Control / Treatment", format="%.2fx"
                    ),
                },
            )
            # Highlight if any rows show control > treatment
            harm_rows = tc_table[tc_table["control_over_treatment"] > 1.0]
            if not harm_rows.empty:
                st.error(
                    f":warning: **{len(harm_rows)} of {len(tc_table)} iterations show "
                    "control CPS > treatment CPS** — empirical evidence the memory "
                    "system is hurting training. Highest harm ratio: "
                    f"{harm_rows['control_over_treatment'].max():.2f}× "
                    f"(env={harm_rows.loc[harm_rows['control_over_treatment'].idxmax(), 'env']}, "
                    f"iter={harm_rows.loc[harm_rows['control_over_treatment'].idxmax(), 'iter']})."
                )

        # ------------------------------------------------------------------
        # Section 6: regression panel
        # ------------------------------------------------------------------
        st.subheader("Regression Panel")
        regressions = history_with_deltas[history_with_deltas["regression_flag"]]
        if regressions.empty:
            st.success("No iterations have tripped the regression flag.")
        else:
            st.warning(
                f"{len(regressions)} iteration(s) flagged as regressions. "
                "Check `cps_v1_delta`, `return_delta`, and `worst_mdd_delta` "
                "to identify which dimensions failed."
            )
            reg_cols = [
                "environment",
                "iteration_number",
                "cps_v1_delta",
                "return_delta",
                "worst_mdd_delta",
                "median_return",
                "worst_fold_mdd",
            ]
            reg_present = [c for c in reg_cols if c in regressions.columns]
            st.dataframe(regressions[reg_present], use_container_width=True)

except Exception as exc:
    st.warning(f"Could not load iteration history: {exc}")
