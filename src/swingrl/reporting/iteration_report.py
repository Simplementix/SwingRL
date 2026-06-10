"""Iteration history loader, delta computer, and chronic-failure detector.

This module is the **first reader** of the ``iteration_results`` table in
production code. Until Phase 0.3 the table was write-only — backtest.py
wrote to it, but no code consumed it.

Architecture:
- **Loaders** (``load_iteration_history``, ``load_fold_history``) take a
  psycopg connection and return DataFrames. They are the only functions
  that touch I/O.
- **Pure functions** (``compute_iter_deltas``, ``detect_chronic_failures``,
  ``detect_protected_winners``, ``format_iteration_summary``) take
  DataFrames and return DataFrames or dicts. They have no DB dependencies
  so they are easy to test with synthetic fixtures.

The dedup logic in ``load_fold_history`` is the **post-fix selector** for
the iter 1 restart-with-fixes case (see plan §Data Audit). It uses
``DISTINCT ON ... ORDER BY ..., created_at DESC`` so the latest row per
``(iteration, environment, algorithm, fold_number)`` wins — that is the
post-fix run, which is what we want for CPS computation.

Regression flag thresholds are deliberately exposed as module-level
constants so tests and the dashboard can reuse the same numbers.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import psycopg
import structlog

from swingrl.metrics.cps import FoldMetrics, compute_all_cps

log = structlog.get_logger(__name__)

# Algorithms in the canonical CPS aggregation order. Per-algo CPS values
# stored under these keys in cps_components.per_algo for cross-iteration
# inspection.
_CPS_ALGORITHMS: tuple[str, ...] = ("ppo", "a2c", "sac")

# CPS formula version tag persisted alongside every computed row so future
# formula revisions can be audited without losing the old numbers.
_CPS_FORMULA_VERSION: str = "v1+v2+v3"

# ---------------------------------------------------------------------------
# Regression detection thresholds (module-level for test/dashboard reuse)
# ---------------------------------------------------------------------------

REGRESSION_RETURN_THRESHOLD: float = 0.02
"""A median-return drop greater than this fraction trips the regression flag."""

REGRESSION_WORST_MDD_THRESHOLD: float = 0.02
"""A worst-fold-MDD increase greater than this fraction trips the regression flag."""

# Chronic-failure detection defaults (4-of-6 hysteresis per the plan)
CHRONIC_DEFAULT_WINDOW: int = 6
CHRONIC_DEFAULT_MIN_FAILS: int = 4

# Protected-winner detection defaults
PROTECTED_DEFAULT_WINDOW: int = 6
PROTECTED_DEFAULT_MIN_WINS: int = 4
PROTECTED_DEFAULT_SHARPE_THRESHOLD: float = 4.0


# ---------------------------------------------------------------------------
# Loaders (DB I/O)
# ---------------------------------------------------------------------------


def _query_to_dataframe(
    conn: psycopg.Connection, sql: str, params: tuple[Any, ...]
) -> pd.DataFrame:
    """Execute a query and return rows as a DataFrame.

    Bypasses ``pd.read_sql_query`` because pandas-stubs does not type psycopg
    connections (it expects SQLAlchemy or sqlite3) and pandas itself emits a
    UserWarning at runtime for the same reason. We use cursor.fetchall +
    DataFrame.from_records which is fully typed and warning-free.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)  # type: ignore[arg-type]
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description] if cur.description else []
    return pd.DataFrame.from_records(rows, columns=columns)


def load_iteration_history(conn: psycopg.Connection, env: str, n: int = 10) -> pd.DataFrame:
    """Read the most recent ``n`` iterations of ``iteration_results`` for ``env``.

    Args:
        conn: Open psycopg connection.
        env: Environment name ('equity' or 'crypto').
        n: Maximum number of iterations to return (default 10).

    Returns:
        DataFrame ordered by ``iteration_number`` ascending. Columns include
        all rows from ``iteration_results`` plus the new CPS columns added
        in Phase 0.2.
    """
    query = (
        "SELECT * FROM iteration_results "
        "WHERE environment = %s "
        "ORDER BY iteration_number DESC "
        "LIMIT %s"
    )
    df = _query_to_dataframe(conn, query, (env, n))
    # Reverse to chronological order so deltas compute against the prior row.
    return df.sort_values("iteration_number").reset_index(drop=True)


def load_fold_history(conn: psycopg.Connection, env: str) -> pd.DataFrame:
    """Read per-fold backtest_results for one environment, dedup'd by latest run.

    Iter 1 contains 9 duplicate rows (6 A2C + 3 PPO equity folds) from a
    mid-iteration restart-with-fixes. We pick the **latest** row per
    ``(iteration, env, algorithm, fold_number)`` because the later row is
    the post-fix run — that is the version training considered final.

    Args:
        conn: Open psycopg connection.
        env: Environment name.

    Returns:
        DataFrame with one row per (iteration, algorithm, fold_number),
        de-duplicated. Includes columns needed by ``detect_chronic_failures``
        and ``detect_protected_winners``: iteration_number, environment,
        algorithm, fold_number, sharpe, mdd, overfitting_class, plus
        anything else in backtest_results.
    """
    query = (
        "SELECT DISTINCT ON (iteration_number, environment, algorithm, fold_number) "
        "  iteration_number, environment, algorithm, fold_number, "
        "  sharpe, sortino, calmar, mdd, total_return, profit_factor, "
        "  win_rate, total_trades, overfitting_class, max_single_loss, "
        "  hmm_p_bull, hmm_p_bear, vix_mean, is_control_fold, "
        "  train_start_date, test_start_date, test_end_date, created_at "
        "FROM backtest_results "
        "WHERE environment = %s "
        "ORDER BY iteration_number, environment, algorithm, fold_number, created_at DESC"
    )
    return _query_to_dataframe(conn, query, (env,))


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_iter_deltas(history: pd.DataFrame) -> pd.DataFrame:
    """Compute per-iteration deltas vs the previous iteration of the same env.

    The output preserves all input columns and appends:
        - cps_v1_delta
        - return_delta
        - worst_mdd_delta
        - winners_delta
        - regression_flag (bool)

    Iter 0 of each environment has no prior; its delta columns are NaN and
    ``regression_flag`` is False (no delta = no regression).

    The regression flag is True iff any of:
        - cps_v1_delta < 0
        - return_delta < -REGRESSION_RETURN_THRESHOLD
        - worst_mdd_delta > REGRESSION_WORST_MDD_THRESHOLD

    NULL CPS values (pre-backfill rows) yield NaN cps_v1_delta and do NOT
    contribute to the regression flag.

    Args:
        history: DataFrame from ``load_iteration_history`` (or a synthetic
            equivalent for tests).

    Returns:
        Copy of ``history`` with delta columns appended. Length is
        unchanged. Output is sorted by (environment, iteration_number).
    """
    if history.empty:
        return history.copy()

    out = history.sort_values(["environment", "iteration_number"]).reset_index(drop=True)

    # groupby().shift() gives us "previous row's value within each env group"
    grouped = out.groupby("environment", group_keys=False)

    out["cps_v1_delta"] = out["cps_v1_multiplicative"] - grouped["cps_v1_multiplicative"].shift(1)
    out["return_delta"] = out["median_return"] - grouped["median_return"].shift(1)
    out["worst_mdd_delta"] = out["worst_fold_mdd"] - grouped["worst_fold_mdd"].shift(1)
    if "winners_count" in out.columns:
        out["winners_delta"] = out["winners_count"] - grouped["winners_count"].shift(1)
    else:
        out["winners_delta"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    # Regression flag: True if any dimension regressed. NaN deltas (iter 0
    # or NULL CPS) do not contribute. Using fillna(False) to coerce NaN
    # comparisons to False.
    cps_regressed = (out["cps_v1_delta"] < 0).fillna(False)
    return_regressed = (out["return_delta"] < -REGRESSION_RETURN_THRESHOLD).fillna(False)
    mdd_regressed = (out["worst_mdd_delta"] > REGRESSION_WORST_MDD_THRESHOLD).fillna(False)

    out["regression_flag"] = (cps_regressed | return_regressed | mdd_regressed).astype(bool)
    return out


def detect_chronic_failures(
    fold_history: pd.DataFrame,
    window: int = CHRONIC_DEFAULT_WINDOW,
    min_fails: int = CHRONIC_DEFAULT_MIN_FAILS,
) -> dict[str, list[int]]:
    """Identify folds that have failed in ``min_fails`` of the last ``window`` iterations.

    A fold "failed" iff its ``overfitting_class`` is not ``'healthy'``
    (i.e., 'marginal' or 'reject') in that iteration. Aggregated across
    algorithms — if ANY algorithm passed for that fold in that iteration,
    the fold is considered to have passed for that iteration overall.

    Args:
        fold_history: DataFrame from ``load_fold_history`` (or synthetic).
        window: Number of recent iterations to look at (default 6).
        min_fails: Minimum failing iterations to flag the fold (default 4).

    Returns:
        Dict mapping environment name → sorted list of chronic fold numbers.
        Environments with no chronic failures are omitted.
    """
    if fold_history.empty:
        return {}

    # Aggregate across algorithms: a fold "passes" in an iteration iff at
    # least one algorithm produced a 'healthy' overfitting_class for it.
    fold_history = fold_history.copy()
    fold_history["passed"] = fold_history["overfitting_class"] == "healthy"

    # Per (env, iter, fold): max(passed) across algorithms
    per_iter_pass = (
        fold_history.groupby(["environment", "iteration_number", "fold_number"])["passed"]
        .max()
        .reset_index()
    )

    result: dict[str, list[int]] = {}
    for env in per_iter_pass["environment"].unique():
        env_df = per_iter_pass[per_iter_pass["environment"] == env]
        # Limit to the last `window` iterations
        latest_iters = sorted(env_df["iteration_number"].unique())[-window:]
        env_df = env_df[env_df["iteration_number"].isin(latest_iters)]

        # Per fold: count failing iterations
        fail_counts = (
            env_df.groupby("fold_number")["passed"]
            .apply(lambda s: int((~s).sum()))
            .reset_index(name="fails")
        )
        chronic = sorted(fail_counts[fail_counts["fails"] >= min_fails]["fold_number"].tolist())
        if chronic:
            result[env] = chronic
    return result


def detect_protected_winners(
    fold_history: pd.DataFrame,
    window: int = PROTECTED_DEFAULT_WINDOW,
    min_wins: int = PROTECTED_DEFAULT_MIN_WINS,
    sharpe_threshold: float = PROTECTED_DEFAULT_SHARPE_THRESHOLD,
) -> dict[str, list[int]]:
    """Identify folds with sharpe > threshold in ``min_wins`` of last ``window`` iterations.

    Aggregated across algorithms — a fold "won" in an iteration iff its
    MAX sharpe across algorithms exceeded the threshold.

    Args:
        fold_history: DataFrame from ``load_fold_history``.
        window: Recent iterations to consider.
        min_wins: Minimum winning iterations to flag the fold.
        sharpe_threshold: Sharpe threshold for "winner" classification.

    Returns:
        Dict mapping environment → sorted list of protected winner fold numbers.
    """
    if fold_history.empty:
        return {}

    fold_history = fold_history.copy()

    # Per (env, iter, fold): max sharpe across algorithms
    per_iter_max = (
        fold_history.groupby(["environment", "iteration_number", "fold_number"])["sharpe"]
        .max()
        .reset_index()
    )

    result: dict[str, list[int]] = {}
    for env in per_iter_max["environment"].unique():
        env_df = per_iter_max[per_iter_max["environment"] == env]
        latest_iters = sorted(env_df["iteration_number"].unique())[-window:]
        env_df = env_df[env_df["iteration_number"].isin(latest_iters)]

        win_counts = (
            env_df.groupby("fold_number")["sharpe"]
            .apply(lambda s: int((s > sharpe_threshold).sum()))
            .reset_index(name="wins")
        )
        winners = sorted(win_counts[win_counts["wins"] >= min_wins]["fold_number"].tolist())
        if winners:
            result[env] = winners
    return result


def format_iteration_summary(history: pd.DataFrame) -> str:
    """Render a markdown table of the iteration history with regression highlighting.

    Used by both Discord embeds (Phase 0.6) and CLI output from the backfill
    script (Phase 0.4). Iterations with ``regression_flag=True`` are marked
    with a ⚠ prefix so a quick visual scan finds them.

    Args:
        history: DataFrame from ``load_iteration_history`` or
            ``compute_iter_deltas``. If the input lacks delta columns,
            this function adds them via ``compute_iter_deltas``.

    Returns:
        Multi-line markdown string. Empty input returns the header only.
    """
    if history.empty:
        return "_(no iterations to summarize)_"

    if "regression_flag" not in history.columns:
        history = compute_iter_deltas(history)

    lines: list[str] = []
    lines.append(
        "| Iter | Env | CPS v1 | CPS Δ | Return | Return Δ | Worst MDD | Winners | Regression |"
    )
    lines.append(
        "|------|-----|--------|-------|--------|----------|-----------|---------|------------|"
    )
    for _, row in history.iterrows():
        marker = "⚠ " if bool(row.get("regression_flag", False)) else ""
        cps = _fmt(row.get("cps_v1_multiplicative"))
        cps_d = _fmt(row.get("cps_v1_delta"), signed=True)
        ret = _fmt_pct(row.get("median_return"))
        ret_d = _fmt_pct(row.get("return_delta"), signed=True)
        mdd = _fmt_pct(row.get("worst_fold_mdd"))
        winners = _fmt_int(row.get("winners_count"))
        flag = "REGRESSION" if bool(row.get("regression_flag", False)) else ""
        lines.append(
            f"| {marker}{int(row['iteration_number'])} | {row['environment']} | "
            f"{cps} | {cps_d} | {ret} | {ret_d} | {mdd} | {winners} | {flag} |"
        )
    return "\n".join(lines)


def _fmt(value: Any, *, signed: bool = False) -> str:
    """Format a CPS-scale float (5 decimal places) with optional sign."""
    if value is None or pd.isna(value):
        return "—"
    fmt = f"{value:+.5f}" if signed else f"{value:.5f}"
    return fmt


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    """Format a fractional value as a percentage with 2 decimal places."""
    if value is None or pd.isna(value):
        return "—"
    pct = float(value) * 100.0
    fmt = f"{pct:+.2f}%" if signed else f"{pct:.2f}%"
    return fmt


def _fmt_int(value: Any) -> str:
    """Format an integer cell, returning em-dash for missing values."""
    if value is None or pd.isna(value):
        return "—"
    return str(int(value))


# ---------------------------------------------------------------------------
# Per-iteration CPS aggregation (used by backfill script and Phase 0.5 hook)
# ---------------------------------------------------------------------------


# Initial training capital for backtests. Source: config/swingrl.yaml
# environment.initial_amount = 100000.0 (validated in
# src/swingrl/config/schema.py EnvironmentConfig.initial_amount default).
# Used to convert max_single_loss from dollars (as stored in pg16
# backtest_results) into the fraction expected by CPS formulas.
_BACKTEST_INITIAL_CAPITAL: float = 100_000.0


def _fold_row_to_metrics(row: pd.Series) -> FoldMetrics:
    """Convert a single backtest_results row (from load_fold_history) into FoldMetrics.

    The fold_history DataFrame columns map directly to FoldMetrics keys, but
    we must:

    1. Convert ``max_single_loss`` from dollars (pg16 storage) to a fraction
       of the initial training capital ($100K). The CPS formulas expect
       fractional units (e.g., -0.04 = lost 4% of equity on the worst trade).
       This is the unit-conversion boundary between the storage layer and the
       pure CPS module.
    2. Coerce numeric types and handle nullable max_single_loss explicitly.

    Note: ``total_return`` and ``mdd`` are already stored as fractions in
    pg16, so they need no conversion.
    """
    msl_dollars = row.get("max_single_loss")
    if msl_dollars is None or pd.isna(msl_dollars):
        msl_fraction: float | None = None
    else:
        msl_fraction = float(msl_dollars) / _BACKTEST_INITIAL_CAPITAL

    return {
        "fold_number": int(row["fold_number"]),
        "sharpe": float(row["sharpe"]),
        "mdd": float(row["mdd"]),
        "total_return": float(row["total_return"]),
        "profit_factor": float(row["profit_factor"]),
        "win_rate": float(row["win_rate"]),
        "total_trades": int(row["total_trades"]),
        "sortino": float(row["sortino"]),
        "max_single_loss": msl_fraction,
        "overfitting_class": str(row["overfitting_class"]),
        "is_control_fold": bool(row.get("is_control_fold", False)),
    }


def _safe_mean(values: list[float]) -> float | None:
    """Mean of a list, returning None for empty input."""
    if not values:
        return None
    return sum(values) / len(values)


def compute_iteration_cps(
    fold_history: pd.DataFrame,
    env: str,
    iteration: int,
    prev_iter_median_return: float | None,
    chronic_failure_count: int,
) -> dict[str, Any]:
    """Compute CPS for one (env, iteration) by aggregating per-algo CPS values.

    Strategy: each algorithm gets its own CPS computation against its slice
    of the fold rows. The aggregated primary value is the **mean** across
    algos (mean reflects ensemble-style deployment; the per-algo breakdown
    is preserved in the components JSON for inspection).

    Treatment-only and control-only CPS are computed by the same per-algo →
    mean recipe but with the fold rows pre-filtered. Iter 0-2 had no control
    folds, so treatment_only / control_only return None for those iterations.

    Args:
        fold_history: DataFrame from ``load_fold_history``, NOT yet filtered
            to a specific iteration. The function filters internally.
        env: Environment name.
        iteration: Iteration number to compute CPS for.
        prev_iter_median_return: Median return from the previous iteration
            (used by v3). Pass None for iter 0 or when unavailable.
        chronic_failure_count: Number of chronic-failure folds in the
            window ending at this iteration (used by v2).

    Returns:
        Dict with keys (matching iteration_results columns):
            cps_v1_multiplicative
            cps_v2_additive
            cps_v3_sortino
            cps_v1_treatment_only
            cps_v1_control_only
            cps_components (JSON-serializable dict including per_algo)
            worst_fold_number
            worst_fold_mdd
            worst_fold_max_single_loss
            median_return
            mean_winner_sharpe
            winners_count
            chronic_failure_count
            return_regression_delta
    """
    # Filter to this iteration + environment
    iter_folds = fold_history[
        (fold_history["iteration_number"] == iteration) & (fold_history["environment"] == env)
    ]

    if iter_folds.empty:
        return {
            "cps_v1_multiplicative": None,
            "cps_v2_additive": None,
            "cps_v3_sortino": None,
            "cps_v1_treatment_only": None,
            "cps_v1_control_only": None,
            "cps_components": {"per_algo": {}, "aggregation_strategy": "mean_across_algos"},
            "worst_fold_number": None,
            "worst_fold_mdd": None,
            "worst_fold_max_single_loss": None,
            "median_return": None,
            "mean_winner_sharpe": None,
            "winners_count": 0,
            "chronic_failure_count": chronic_failure_count,
            "return_regression_delta": 0.0,
        }

    # Per-algo CPS — one CPS dict per algorithm present
    per_algo: dict[str, dict[str, Any]] = {}
    for algo in _CPS_ALGORITHMS:
        algo_rows = iter_folds[iter_folds["algorithm"] == algo]
        if algo_rows.empty:
            continue
        algo_metrics = [_fold_row_to_metrics(row) for _, row in algo_rows.iterrows()]
        per_algo[algo] = compute_all_cps(
            algo_metrics, prev_iter_median_return, chronic_failure_count
        )

    # Aggregate primary CPS values via mean across algos
    v1_values = [v["cps_v1_multiplicative"] for v in per_algo.values()]
    v2_values = [v["cps_v2_additive"] for v in per_algo.values()]
    v3_values = [v["cps_v3_sortino"] for v in per_algo.values() if v["cps_v3_sortino"] is not None]

    cps_v1 = _safe_mean(v1_values)
    cps_v2 = _safe_mean(v2_values)
    cps_v3 = _safe_mean(v3_values) if v3_values else None

    # Treatment-only / control-only — only meaningful when control folds exist
    treatment_v1: float | None = None
    control_v1: float | None = None
    has_control_folds = bool(iter_folds["is_control_fold"].any())
    if has_control_folds:
        treatment_v1 = _aggregate_subset_v1(
            iter_folds[~iter_folds["is_control_fold"].astype(bool)],
            prev_iter_median_return,
            chronic_failure_count,
        )
        control_v1 = _aggregate_subset_v1(
            iter_folds[iter_folds["is_control_fold"].astype(bool)],
            prev_iter_median_return,
            chronic_failure_count,
        )

    # Worst-fold identification (across all algos): find the (fold, algo) row
    # with the highest MDD. We report the fold_number and its MDD value.
    worst_idx = iter_folds["mdd"].idxmax()
    worst_row = iter_folds.loc[worst_idx]
    worst_fold_number = int(worst_row["fold_number"])
    worst_fold_mdd = float(worst_row["mdd"])
    worst_msl_raw = worst_row.get("max_single_loss")
    worst_fold_msl = (
        float(worst_msl_raw) if worst_msl_raw is not None and not pd.isna(worst_msl_raw) else None
    )

    # Aggregate component stats — use the per-algo means as the canonical
    # iteration-level numbers
    median_returns = [v["median_return"] for v in per_algo.values()]
    mean_winner_sharpes = [v["mean_winner_sharpe"] for v in per_algo.values()]
    winners_counts = [v["winners_count"] for v in per_algo.values()]
    return_regression_deltas = [v["return_regression_delta"] for v in per_algo.values()]

    components = {
        "per_algo": per_algo,
        "aggregation_strategy": "mean_across_algos",
        "median_return_per_algo": dict(zip(per_algo.keys(), median_returns, strict=True)),
        "mean_winner_sharpe_per_algo": dict(zip(per_algo.keys(), mean_winner_sharpes, strict=True)),
        "winners_count_per_algo": dict(zip(per_algo.keys(), winners_counts, strict=True)),
        "v1_min_across_algos": min(v1_values) if v1_values else None,
        "v1_max_across_algos": max(v1_values) if v1_values else None,
    }

    return {
        "cps_v1_multiplicative": cps_v1,
        "cps_v2_additive": cps_v2,
        "cps_v3_sortino": cps_v3,
        "cps_v1_treatment_only": treatment_v1,
        "cps_v1_control_only": control_v1,
        "cps_components": components,
        "worst_fold_number": worst_fold_number,
        "worst_fold_mdd": worst_fold_mdd,
        "worst_fold_max_single_loss": worst_fold_msl,
        "median_return": _safe_mean(median_returns),
        "mean_winner_sharpe": _safe_mean(mean_winner_sharpes),
        "winners_count": int(_safe_mean([float(w) for w in winners_counts]) or 0),
        "chronic_failure_count": chronic_failure_count,
        "return_regression_delta": _safe_mean(return_regression_deltas) or 0.0,
    }


def _aggregate_subset_v1(
    subset_folds: pd.DataFrame,
    prev_iter_median_return: float | None,
    chronic_failure_count: int,
) -> float | None:
    """Compute mean v1 across algos for a pre-filtered subset (treatment or control)."""
    if subset_folds.empty:
        return None
    per_algo_v1: list[float] = []
    for algo in _CPS_ALGORITHMS:
        algo_rows = subset_folds[subset_folds["algorithm"] == algo]
        if algo_rows.empty:
            continue
        algo_metrics = [_fold_row_to_metrics(row) for _, row in algo_rows.iterrows()]
        result = compute_all_cps(algo_metrics, prev_iter_median_return, chronic_failure_count)
        per_algo_v1.append(result["cps_v1_multiplicative"])
    return _safe_mean(per_algo_v1)


# ---------------------------------------------------------------------------
# Persistence — UPSERT CPS values to iteration_results (used by backfill and
# train_pipeline.py integration hooks alike)
# ---------------------------------------------------------------------------


def persist_iteration_cps(
    conn: psycopg.Connection,
    env: str,
    iteration: int,
    cps_data: dict[str, Any],
    dedup_rows_dropped: int,
) -> None:
    """UPSERT CPS values into iteration_results for one (env, iter) row.

    Writes only the CPS-related columns so the existing ensemble_sharpe /
    per-algo means / hyperparams written by ``store_iteration_results_to_duckdb``
    are untouched. If the row does not yet exist (e.g., iter 5 being
    backfilled before training finishes), creates a minimal placeholder row
    with ``result_id = 'backfill-{env}-iter{N}'``.

    Args:
        conn: Open psycopg connection. Caller manages the transaction; this
            function does not commit.
        env: Environment name.
        iteration: Iteration number.
        cps_data: Output of ``compute_iteration_cps``.
        dedup_rows_dropped: Audit count of duplicate fold rows that were
            dedup'd by the load_fold_history DISTINCT ON query (0 for clean
            iterations, 9 for iter 1 equity per the restart-with-fixes case).
    """
    components_json = json.dumps(cps_data["cps_components"], default=_cps_json_default)

    with conn.cursor() as cur:
        # Try UPDATE first — the common path for iter 0-N rows already
        # written by the training pipeline.
        cur.execute(
            """
            UPDATE iteration_results SET
                cps_v1_multiplicative = %s,
                cps_v2_additive = %s,
                cps_v3_sortino = %s,
                cps_v1_treatment_only = %s,
                cps_v1_control_only = %s,
                cps_components = %s,
                cps_formula_version = %s,
                worst_fold_number = %s,
                worst_fold_mdd = %s,
                worst_fold_max_single_loss = %s,
                median_return = %s,
                mean_winner_sharpe = %s,
                winners_count = %s,
                chronic_failure_count = %s,
                return_regression_delta = %s,
                dedup_rows_dropped = %s
            WHERE iteration_number = %s
              AND environment = %s
              AND run_type = 'baseline'
            """,
            (
                cps_data["cps_v1_multiplicative"],
                cps_data["cps_v2_additive"],
                cps_data["cps_v3_sortino"],
                cps_data["cps_v1_treatment_only"],
                cps_data["cps_v1_control_only"],
                components_json,
                _CPS_FORMULA_VERSION,
                cps_data["worst_fold_number"],
                cps_data["worst_fold_mdd"],
                cps_data["worst_fold_max_single_loss"],
                cps_data["median_return"],
                cps_data["mean_winner_sharpe"],
                cps_data["winners_count"],
                cps_data["chronic_failure_count"],
                cps_data["return_regression_delta"],
                dedup_rows_dropped,
                iteration,
                env,
            ),
        )
        if cur.rowcount > 0:
            return

        # No existing row — INSERT a placeholder. Handles the case where CPS
        # is computed before training's store_iteration_results_to_duckdb
        # (shouldn't happen in the pipeline, but supports backfill of iter 5
        # before training finishes).
        result_id = f"backfill-{env}-iter{iteration}"
        cur.execute(
            """
            INSERT INTO iteration_results (
                result_id, iteration_number, environment, run_type,
                cps_v1_multiplicative, cps_v2_additive, cps_v3_sortino,
                cps_v1_treatment_only, cps_v1_control_only,
                cps_components, cps_formula_version,
                worst_fold_number, worst_fold_mdd, worst_fold_max_single_loss,
                median_return, mean_winner_sharpe, winners_count,
                chronic_failure_count, return_regression_delta,
                dedup_rows_dropped
            ) VALUES (
                %s, %s, %s, 'baseline',
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (iteration_number, environment, run_type) DO NOTHING
            """,
            (
                result_id,
                iteration,
                env,
                cps_data["cps_v1_multiplicative"],
                cps_data["cps_v2_additive"],
                cps_data["cps_v3_sortino"],
                cps_data["cps_v1_treatment_only"],
                cps_data["cps_v1_control_only"],
                components_json,
                _CPS_FORMULA_VERSION,
                cps_data["worst_fold_number"],
                cps_data["worst_fold_mdd"],
                cps_data["worst_fold_max_single_loss"],
                cps_data["median_return"],
                cps_data["mean_winner_sharpe"],
                cps_data["winners_count"],
                cps_data["chronic_failure_count"],
                cps_data["return_regression_delta"],
                dedup_rows_dropped,
            ),
        )


def _cps_json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy/pandas scalars embedded in cps_components."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _count_dedup_drops(conn: psycopg.Connection, env: str, iteration: int) -> int:
    """Count how many duplicate (algo, fold) pairs existed pre-dedup.

    Returns 0 for clean iterations and 9 for iter 1 equity (the
    restart-with-fixes case). Used purely for the ``dedup_rows_dropped``
    audit column.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT (algorithm, fold_number)) "
            "FROM backtest_results "
            "WHERE environment = %s AND iteration_number = %s",
            (env, iteration),
        )
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0


def _load_prev_iter_median_return(
    conn: psycopg.Connection, env: str, iteration: int
) -> float | None:
    """Read the previous iteration's median_return for the v3 regression baseline.

    Returns None when no prior iteration row exists (iter 0 or new env).
    """
    if iteration == 0:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT median_return FROM iteration_results "
            "WHERE iteration_number = %s AND environment = %s AND run_type = 'baseline'",
            (iteration - 1, env),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])


def compute_and_persist_iteration_cps(
    conn: psycopg.Connection, env: str, iteration: int
) -> dict[str, Any]:
    """Full orchestration: load fold data, compute CPS, persist, return summary.

    Used by both the Phase 0.4 backfill script and the Phase 0.5 training
    pipeline hook. The returned dict includes deltas vs the previous
    iteration so callers can fire ``iteration_regression_detected`` log
    events when appropriate.

    Args:
        conn: Open psycopg connection.
        env: Environment name.
        iteration: Iteration number to compute CPS for.

    Returns:
        Dict containing:
            cps_v1_multiplicative, cps_v2_additive, cps_v3_sortino (primary scalars)
            cps_v1_delta_vs_prev, return_delta_vs_prev, worst_mdd_delta_vs_prev
            regression_flag (bool)
            regression_dimensions (list[str])  — which dims tripped the flag
            winners_count, chronic_failure_count, worst_fold_number,
                worst_fold_mdd, median_return, mean_winner_sharpe,
                dedup_rows_dropped
            full_cps_data (dict from compute_iteration_cps for completeness)
    """
    fold_history = load_fold_history(conn, env=env)

    # Chronic-failure window ends at this iteration (hysteresis built in)
    history_through_now = fold_history[fold_history["iteration_number"] <= iteration]
    chronics = detect_chronic_failures(history_through_now)
    chronic_count = len(chronics.get(env, []))

    prev_median_return = _load_prev_iter_median_return(conn, env, iteration)

    cps_data = compute_iteration_cps(
        fold_history=fold_history,
        env=env,
        iteration=iteration,
        prev_iter_median_return=prev_median_return,
        chronic_failure_count=chronic_count,
    )

    dedup_dropped = _count_dedup_drops(conn, env, iteration)
    persist_iteration_cps(conn, env, iteration, cps_data, dedup_dropped)

    # Compute deltas vs prev iter (requires reading the prior row's CPS)
    prev_cps_v1, prev_worst_mdd = _load_prev_iter_cps_and_mdd(conn, env, iteration)
    cps_v1_delta: float | None = None
    return_delta: float | None = None
    worst_mdd_delta: float | None = None
    if prev_cps_v1 is not None and cps_data["cps_v1_multiplicative"] is not None:
        cps_v1_delta = cps_data["cps_v1_multiplicative"] - prev_cps_v1
    if prev_median_return is not None and cps_data["median_return"] is not None:
        return_delta = cps_data["median_return"] - prev_median_return
    if prev_worst_mdd is not None and cps_data["worst_fold_mdd"] is not None:
        worst_mdd_delta = cps_data["worst_fold_mdd"] - prev_worst_mdd

    regression_dimensions: list[str] = []
    if cps_v1_delta is not None and cps_v1_delta < 0:
        regression_dimensions.append("cps_v1")
    if return_delta is not None and return_delta < -REGRESSION_RETURN_THRESHOLD:
        regression_dimensions.append("median_return")
    if worst_mdd_delta is not None and worst_mdd_delta > REGRESSION_WORST_MDD_THRESHOLD:
        regression_dimensions.append("worst_fold_mdd")

    return {
        "env": env,
        "iteration": iteration,
        "cps_v1_multiplicative": cps_data["cps_v1_multiplicative"],
        "cps_v2_additive": cps_data["cps_v2_additive"],
        "cps_v3_sortino": cps_data["cps_v3_sortino"],
        "cps_v1_treatment_only": cps_data["cps_v1_treatment_only"],
        "cps_v1_control_only": cps_data["cps_v1_control_only"],
        "cps_v1_delta_vs_prev": cps_v1_delta,
        "return_delta_vs_prev": return_delta,
        "worst_mdd_delta_vs_prev": worst_mdd_delta,
        "regression_flag": bool(regression_dimensions),
        "regression_dimensions": regression_dimensions,
        "winners_count": cps_data["winners_count"],
        "chronic_failure_count": chronic_count,
        "worst_fold_number": cps_data["worst_fold_number"],
        "worst_fold_mdd": cps_data["worst_fold_mdd"],
        "median_return": cps_data["median_return"],
        "mean_winner_sharpe": cps_data["mean_winner_sharpe"],
        "dedup_rows_dropped": dedup_dropped,
        "full_cps_data": cps_data,
    }


def _load_prev_iter_cps_and_mdd(
    conn: psycopg.Connection, env: str, iteration: int
) -> tuple[float | None, float | None]:
    """Read (cps_v1_multiplicative, worst_fold_mdd) from the previous iter row."""
    if iteration == 0:
        return None, None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT cps_v1_multiplicative, worst_fold_mdd FROM iteration_results "
            "WHERE iteration_number = %s AND environment = %s AND run_type = 'baseline'",
            (iteration - 1, env),
        )
        row = cur.fetchone()
        if row is None:
            return None, None
        v1 = float(row[0]) if row[0] is not None else None
        mdd = float(row[1]) if row[1] is not None else None
        return v1, mdd
