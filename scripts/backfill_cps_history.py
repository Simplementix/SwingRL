#!/usr/bin/env python3
"""Backfill Capital Preservation Score (CPS) values into iteration_results.

Phase 0.4 of the memory agent refocus. Reads ``backtest_results`` for iter
0-4 (or a user-specified range), dedups iter 1's restart-with-fixes
duplicates, computes per-iteration CPS via the per-algo aggregation in
``iteration_report.compute_iteration_cps``, and UPSERTs the values into
``iteration_results``.

Idempotent — safe to re-run. The UPSERT writes only the new CPS columns
plus an audit count of duplicate rows that were dedup'd; it does NOT touch
the existing ensemble_sharpe / ensemble_mdd / per-algo means written by the
training pipeline.

**Iter 5 is deliberately skipped** by default because training is still in
progress at the time of Phase 0.4. After iter 5 completes, re-run with
``--max-iter 5`` to backfill it as well.

Usage::

    export DATABASE_URL="$(grep DATABASE_URL .env | cut -d= -f2-)"
    uv run python scripts/backfill_cps_history.py

    # Or via SSH against homelab (DATABASE_URL is set in the container env):
    ssh homelab "docker exec swingrl python3 /app/scripts/backfill_cps_history.py"

    # Override iteration range:
    uv run python scripts/backfill_cps_history.py --max-iter 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
import structlog

# Allow running from repo root or from inside the container
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swingrl.reporting.iteration_report import (  # noqa: E402
    compute_and_persist_iteration_cps,
    load_fold_history,
)

log = structlog.get_logger(__name__)

# Default iteration range (inclusive). Iter 5 skipped because training is
# still running at the time Phase 0.4 was authored.
_DEFAULT_MIN_ITER = 0
_DEFAULT_MAX_ITER = 4

# Environments to backfill. Crypto is included because iter 0-4 crypto
# rows exist; Phase 0.4 covers both equity and crypto.
_ENVIRONMENTS: tuple[str, ...] = ("equity", "crypto")


def get_database_url() -> str:
    """Read DATABASE_URL from environment with a clear error message."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Source it from .env or the container env.")
    return url


def backfill_environment(
    conn: psycopg.Connection, env: str, min_iter: int, max_iter: int
) -> list[dict[str, Any]]:
    """Backfill all iterations in [min_iter, max_iter] for one environment.

    Delegates per-iteration work to ``compute_and_persist_iteration_cps``
    so the backfill and the training pipeline hook (Phase 0.5) share one
    code path. Returns a list of summary dicts (one per iteration) for the
    printed validation table.
    """
    log.info("backfill_environment_started", env=env, min_iter=min_iter, max_iter=max_iter)

    fold_history = load_fold_history(conn, env=env)
    if fold_history.empty:
        log.warning("backfill_no_fold_history", env=env)
        return []

    summaries: list[dict[str, Any]] = []

    for iteration in range(min_iter, max_iter + 1):
        # Skip iterations with no fold rows (e.g., iter 5 crypto before training)
        if fold_history[fold_history["iteration_number"] == iteration].empty:
            log.warning("backfill_iteration_no_folds", env=env, iteration=iteration)
            continue

        result = compute_and_persist_iteration_cps(conn, env, iteration)

        log.info(
            "backfill_iteration_complete",
            env=env,
            iteration=iteration,
            cps_v1=result["cps_v1_multiplicative"],
            cps_v2=result["cps_v2_additive"],
            cps_v3=result["cps_v3_sortino"],
            winners=result["winners_count"],
            chronic=result["chronic_failure_count"],
            dedup_dropped=result["dedup_rows_dropped"],
            regression=result["regression_flag"],
        )

        summaries.append(
            {
                "env": env,
                "iteration": iteration,
                "cps_v1": result["cps_v1_multiplicative"],
                "cps_v2": result["cps_v2_additive"],
                "cps_v3": result["cps_v3_sortino"],
                "treatment_v1": result["cps_v1_treatment_only"],
                "control_v1": result["cps_v1_control_only"],
                "median_return": result["median_return"],
                "winners": result["winners_count"],
                "chronic": result["chronic_failure_count"],
                "dedup_dropped": result["dedup_rows_dropped"],
                "worst_fold": result["worst_fold_number"],
                "worst_mdd": result["worst_fold_mdd"],
            }
        )

    conn.commit()
    return summaries


def print_validation_table(summaries: list[dict[str, Any]]) -> None:
    """Print the iteration-by-iteration CPS validation table to stdout."""
    if not summaries:
        print("(no iterations backfilled)")
        return

    # Group by environment for readable output
    by_env: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        by_env.setdefault(s["env"], []).append(s)

    print()
    print("=" * 100)
    print("CPS BACKFILL VALIDATION TABLE")
    print("=" * 100)

    for env, env_summaries in by_env.items():
        print()
        print(f"### {env.upper()}")
        print(
            f"{'Iter':>4}  {'CPS v1':>9}  {'CPS v2':>9}  {'CPS v3':>9}  "
            f"{'Treat v1':>9}  {'Ctrl v1':>9}  {'Return':>8}  {'Win':>3}  "
            f"{'Chr':>3}  {'Worst':>5}  {'WMdd':>6}  {'Dedup':>5}"
        )
        prev_v1: float | None = None
        for s in env_summaries:
            v1 = s["cps_v1"]
            v1_d = ""
            if v1 is not None and prev_v1 is not None:
                delta = v1 - prev_v1
                marker = " ⚠ REGRESSION" if delta < 0 else ""
                v1_d = f" (Δ{delta:+.5f}{marker})"
            prev_v1 = v1

            print(
                f"{s['iteration']:>4}  "
                f"{_fmt_cps(v1):>9}  "
                f"{_fmt_cps(s['cps_v2']):>9}  "
                f"{_fmt_cps(s['cps_v3']):>9}  "
                f"{_fmt_cps(s['treatment_v1']):>9}  "
                f"{_fmt_cps(s['control_v1']):>9}  "
                f"{_fmt_pct(s['median_return']):>8}  "
                f"{s['winners']:>3}  "
                f"{s['chronic']:>3}  "
                f"{_fmt_int(s['worst_fold']):>5}  "
                f"{_fmt_pct(s['worst_mdd']):>6}  "
                f"{s['dedup_dropped']:>5}"
                f"{v1_d}"
            )
    print()
    print("=" * 100)
    print()


def _fmt_cps(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.5f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "—"
    return str(int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-iter",
        type=int,
        default=_DEFAULT_MIN_ITER,
        help=f"First iteration to backfill (default {_DEFAULT_MIN_ITER})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=_DEFAULT_MAX_ITER,
        help=f"Last iteration to backfill, inclusive (default {_DEFAULT_MAX_ITER}; "
        "iter 5 skipped because training is still running)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment to backfill (default: both equity and crypto)",
    )
    args = parser.parse_args()

    db_url = get_database_url()
    log.info("backfill_started", min_iter=args.min_iter, max_iter=args.max_iter)

    envs = (args.env,) if args.env else _ENVIRONMENTS
    all_summaries: list[dict[str, Any]] = []

    with psycopg.connect(db_url) as conn:
        for env in envs:
            summaries = backfill_environment(conn, env, args.min_iter, args.max_iter)
            all_summaries.extend(summaries)

    print_validation_table(all_summaries)
    log.info("backfill_complete", iterations_processed=len(all_summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
