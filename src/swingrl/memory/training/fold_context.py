"""Fold-scoped training context: role classification + context assembly.

classify_fold_role is the single source of truth for chronic-failure /
protected-winner determination (spec §5.2) — it delegates to the existing
detectors in iteration_report.py rather than reimplementing their windows.

load_fold_context is a thin I/O wrapper that assembles the per-fold advice
context dict consumed by the LLM epoch advice payload (Task 8). It fails
open: any DB error returns the neutral cold-start context so that advice
generation never blocks training.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import structlog

from swingrl.reporting.iteration_report import (
    detect_chronic_failures,
    detect_protected_winners,
    load_fold_history,
)

log = structlog.get_logger(__name__)

FoldRole = Literal["chronic_failure", "protected_winner", "neutral"]


def classify_fold_role(fold_history: pd.DataFrame, env: str, fold_number: int) -> FoldRole:
    """Classify a fold from its cross-iteration history.

    Delegates entirely to the existing detectors in iteration_report.py so
    this function and the iteration report always agree. Chronic is checked
    first — a fold somehow matching both detectors (theoretically impossible
    with real data but defensively handled) is treated as chronic_failure
    because it is the riskier classification.

    Args:
        fold_history: DataFrame shaped like load_fold_history() output
            (iteration_number, environment, algorithm, fold_number, sharpe,
            overfitting_class, ...). May be empty (iter 0 cold start).
        env: Environment name ('equity' or 'crypto').
        fold_number: Fold to classify.

    Returns:
        "chronic_failure" | "protected_winner" | "neutral". Chronic is checked
        first — a fold somehow matching both detectors is treated as chronic
        (the riskier classification wins, defensively).
    """
    if fold_history.empty:
        return "neutral"

    chronic = detect_chronic_failures(fold_history).get(env, [])
    if fold_number in chronic:
        return "chronic_failure"

    protected = detect_protected_winners(fold_history).get(env, [])
    if fold_number in protected:
        return "protected_winner"

    return "neutral"


def load_fold_context(database_url: str, env: str, fold_number: int) -> dict[str, Any]:
    """Assemble the per-fold advice context (thin I/O wrapper; spec §5.1).

    Connects to the database with a 5-second timeout so a dead or unreachable
    DB cannot stall training for minutes. Loads the full fold history for
    ``env``, classifies the fold's role, and reads the most recent
    ``cps_v1_multiplicative`` from ``iteration_results``.

    Returns dict with four keys:
        fold_role: "chronic_failure" | "protected_winner" | "neutral"
        chronic_failure_folds: list of chronic fold numbers for this env
        protected_winner_folds: list of protected winner fold numbers for this env
        prev_iter_cps_v1: float | None — None when no prior iteration exists

    Fails open: any DB exception (connection refused, timeout, query error)
    logs a warning and returns the neutral cold-start context so that advice
    generation never blocks training.

    Args:
        database_url: psycopg-compatible connection string.
        env: Environment name ('equity' or 'crypto').
        fold_number: Fold to classify.

    Returns:
        Context dict — always present, never raises.
    """
    import psycopg

    neutral: dict[str, Any] = {
        "fold_role": "neutral",
        "chronic_failure_folds": [],
        "protected_winner_folds": [],
        "prev_iter_cps_v1": None,
    }
    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:
            history = load_fold_history(conn, env)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT cps_v1_multiplicative FROM iteration_results "
                    "WHERE environment = %s ORDER BY iteration_number DESC LIMIT 1",
                    (env,),
                )
                row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 — fail-open by design (see docstring)
        log.warning("fold_context_load_failed", env=env, error=str(exc))
        return neutral

    chronic_folds = detect_chronic_failures(history).get(env, []) if not history.empty else []
    protected_folds = detect_protected_winners(history).get(env, []) if not history.empty else []

    return {
        "fold_role": classify_fold_role(history, env, fold_number),
        "chronic_failure_folds": chronic_folds,
        "protected_winner_folds": protected_folds,
        "prev_iter_cps_v1": row[0] if row else None,
    }


def record_fold_attribution(conn: Any, run_id: str, fold: dict[str, Any]) -> None:
    """Close the advice-attribution loop after a fold's backtest completes.

    Sets fold_cps_v1_after = single-fold CPS v1 and
    advice_was_effective = (after > before) on every advice row for this run_id.
    Rows with NULL fold_cps_v1_before keep NULL effectiveness (no baseline —
    iter 0 never has a before value).

    Does NOT swallow exceptions. Callers (e.g. the walk-forward loop) must
    wrap in try/except so that attribution failure never kills a fold.

    Args:
        conn: An open psycopg connection (or any object supporting .cursor()).
        run_id: Training run identifier matching reward_adjustments.run_id.
        fold: FoldMetrics-shaped dict for the completed fold. Must include at
            minimum: fold_number, sharpe, mdd, total_return, profit_factor,
            win_rate, total_trades, sortino, max_single_loss, overfitting_class,
            is_control_fold.
    """
    from swingrl.metrics.cps import FoldMetrics, compute_cps_v1_multiplicative

    # Cast to FoldMetrics so the type checker is satisfied; runtime is duck-typed.
    fold_metrics: FoldMetrics = fold  # type: ignore[assignment]
    cps_after = compute_cps_v1_multiplicative([fold_metrics])

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE reward_adjustments "
            "SET fold_cps_v1_after = %s, "
            "    advice_was_effective = CASE WHEN fold_cps_v1_before IS NULL "
            "        THEN NULL ELSE %s > fold_cps_v1_before END "
            "WHERE run_id = %s",
            (cps_after, cps_after, run_id),
        )

    log.info("fold_attribution_recorded", run_id=run_id, fold_cps_v1_after=cps_after)
