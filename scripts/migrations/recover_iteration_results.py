#!/usr/bin/env python3
"""Recover missing iteration_results rows from backtest_results.

Recovery for the silent rollback bug in train_pipeline.py:2283 where the
iteration_results write opens a connection without ``autocommit=True``,
``store_iteration_results_to_duckdb`` never calls ``commit()``, and the
``finally: conn_ens.close()`` rolls back the INSERT. The bug has been
silent since the postgres migration; iter 0-4 rows in pg16 came from the
one-time migration script.

This script reconstructs the iteration_results row(s) for a given
iteration directly from the data we still have:

- ``backtest_results`` (the per-fold OOS metrics) — used to compute
  per-algo means and total fold count.
- ``meta_decisions`` (the LLM HP suggestions per (env, algo)) — used to
  populate the ``ppo_hyperparams`` / ``a2c_hyperparams`` / ``sac_hyperparams``
  JSON columns.

It uses the SAME ensemble math as ``check_ensemble_gate`` and
``sharpe_softmax_weights`` so the recovered row matches what the live
pipeline would have written.

Usage::

    export DATABASE_URL="$(grep DATABASE_URL .env | cut -d= -f2-)"
    uv run python scripts/migrations/recover_iteration_results.py --iteration 5

    # Or via SSH (DATABASE_URL is set in the swingrl container env):
    ssh homelab "docker exec swingrl python3 /tmp/recover_iteration_results.py --iteration 5"

Idempotent: uses INSERT ... ON CONFLICT DO UPDATE so re-runs are safe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

import numpy as np
import psycopg
import structlog

log = structlog.get_logger(__name__)

# Mirror of swingrl.training.pipeline_helpers._GATE_MIN_SHARPE / _GATE_MAX_MDD
# (kept inline so the recovery script has zero swingrl-package dependencies
# beyond psycopg + numpy).
_GATE_MIN_SHARPE: float = 1.0
_GATE_MAX_MDD: float = 0.15

_ALGOS: tuple[str, ...] = ("ppo", "a2c", "sac")


def get_database_url() -> str:
    """Read DATABASE_URL from environment with a clear error message."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Source it from .env or the container env.")
    return url


def sharpe_softmax_weights(sharpe_ratios: dict[str, float]) -> dict[str, float]:
    """Numerically-stable softmax over per-algo Sharpe.

    Mirror of swingrl.training.ensemble.sharpe_softmax_weights so the
    recovery script doesn't pull in the SB3/torch dependency tree.
    """
    if not sharpe_ratios:
        return {}
    names = list(sharpe_ratios.keys())
    values = np.array([sharpe_ratios[n] for n in names])
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted)
    total = float(np.sum(exp_vals))
    return {name: float(w) / total for name, w in zip(names, exp_vals, strict=True)}


def load_per_algo_data(
    conn: psycopg.Connection, env: str, iteration: int
) -> dict[str, dict[str, float | int]]:
    """Read backtest_results for one (env, iter) and aggregate per algo.

    Applies the same DISTINCT ON dedup logic as ``load_fold_history`` to
    handle the iter 1 restart-with-fixes case (no-op for iter 5).

    Returns:
        Dict keyed by algorithm. Each value is a dict with:
            - mean_sharpe (float)
            - mean_mdd (float)
            - fold_count (int)
    """
    query = """
        SELECT algorithm, sharpe, mdd
        FROM (
            SELECT DISTINCT ON (iteration_number, environment, algorithm, fold_number)
                algorithm, sharpe, mdd
            FROM backtest_results
            WHERE iteration_number = %s AND environment = %s
            ORDER BY iteration_number, environment, algorithm, fold_number, created_at DESC
        ) deduped
    """
    with conn.cursor() as cur:
        cur.execute(query, (iteration, env))
        rows = cur.fetchall()

    by_algo: dict[str, list[tuple[float, float]]] = {}
    for algo, sharpe, mdd in rows:
        by_algo.setdefault(algo, []).append((float(sharpe), float(mdd)))

    result: dict[str, dict[str, float | int]] = {}
    for algo in _ALGOS:
        folds = by_algo.get(algo, [])
        if folds:
            sharpes = [s for s, _ in folds]
            mdds = [m for _, m in folds]
            result[algo] = {
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_mdd": float(np.mean(mdds)),
                "fold_count": len(folds),
            }
        else:
            result[algo] = {"mean_sharpe": 0.0, "mean_mdd": 0.0, "fold_count": 0}
    return result


def compute_ensemble_metrics(
    per_algo: dict[str, dict[str, float | int]],
    weights: dict[str, float],
) -> tuple[float, float]:
    """Reproduce check_ensemble_gate's weighted-average computation.

    Returns (ensemble_sharpe, ensemble_mdd).
    """
    weighted_sharpe_sum = 0.0
    weighted_mdd_sum = 0.0
    total_weight = 0.0
    for algo, stats in per_algo.items():
        if stats["fold_count"] == 0:
            continue
        w = weights.get(algo, 0.0)
        weighted_sharpe_sum += w * float(stats["mean_sharpe"])
        weighted_mdd_sum += w * float(stats["mean_mdd"])
        total_weight += w
    if total_weight <= 0:
        return 0.0, 0.0
    return weighted_sharpe_sum / total_weight, weighted_mdd_sum / total_weight


def load_hyperparams_from_meta_decisions(
    conn: psycopg.Connection, env: str, iteration: int
) -> dict[str, str | None]:
    """Read the most recent run_config decision per (env, algo) for this iteration.

    The meta_decisions.run_id encodes the env and algo (e.g.,
    'equity_ppo_20260406T233346Z'). We pick the latest decision per
    (env, algo) and serialize its decision_json as the hyperparams string
    that would have been written by the live pipeline.

    Returns:
        Dict mapping algo → JSON-string-or-None for the
        ``ppo_hyperparams`` / ``a2c_hyperparams`` / ``sac_hyperparams`` columns.
    """
    result: dict[str, str | None] = dict.fromkeys(_ALGOS)
    with conn.cursor() as cur:
        for algo in _ALGOS:
            cur.execute(
                """
                SELECT decision_json FROM meta_decisions
                WHERE env = %s AND algo = %s AND decision_type = 'run_config'
                  AND run_id LIKE %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (env, algo, f"{env}_{algo}_%"),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                # decision_json is text; if it's already valid JSON, pass through
                raw = row[0]
                if isinstance(raw, dict):
                    result[algo] = json.dumps(raw)
                else:
                    result[algo] = str(raw)
    return result


def upsert_iteration_results(
    conn: psycopg.Connection,
    iteration: int,
    env: str,
    ensemble_sharpe: float,
    ensemble_mdd: float,
    gate_passed: bool,
    weights: dict[str, float],
    per_algo: dict[str, dict[str, float | int]],
    hyperparams: dict[str, str | None],
    total_folds: int,
    memory_enabled: bool,
) -> None:
    """INSERT ... ON CONFLICT DO UPDATE for one iteration_results row.

    Schema columns mirror the live ``store_iteration_results_to_duckdb``
    plus an explicit ``conn.commit()`` at the end (the bug fix).
    """
    result_id = f"recovered-{env}-iter{iteration}-{uuid.uuid4().hex[:8]}"
    hp_source = "memory_advised" if any(v is not None for v in hyperparams.values()) else "baseline"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO iteration_results (
                result_id, iteration_number, environment,
                ensemble_sharpe, ensemble_mdd, gate_passed,
                ppo_weight, a2c_weight, sac_weight,
                ppo_mean_sharpe, a2c_mean_sharpe, sac_mean_sharpe,
                ppo_mean_mdd, a2c_mean_mdd, sac_mean_mdd,
                total_folds,
                ppo_hyperparams, a2c_hyperparams, sac_hyperparams, hp_source,
                run_type, wall_clock_s, memory_enabled
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (iteration_number, environment, run_type) DO UPDATE SET
                ensemble_sharpe = EXCLUDED.ensemble_sharpe,
                ensemble_mdd = EXCLUDED.ensemble_mdd,
                gate_passed = EXCLUDED.gate_passed,
                ppo_weight = EXCLUDED.ppo_weight,
                a2c_weight = EXCLUDED.a2c_weight,
                sac_weight = EXCLUDED.sac_weight,
                ppo_mean_sharpe = EXCLUDED.ppo_mean_sharpe,
                a2c_mean_sharpe = EXCLUDED.a2c_mean_sharpe,
                sac_mean_sharpe = EXCLUDED.sac_mean_sharpe,
                ppo_mean_mdd = EXCLUDED.ppo_mean_mdd,
                a2c_mean_mdd = EXCLUDED.a2c_mean_mdd,
                sac_mean_mdd = EXCLUDED.sac_mean_mdd,
                total_folds = EXCLUDED.total_folds,
                ppo_hyperparams = EXCLUDED.ppo_hyperparams,
                a2c_hyperparams = EXCLUDED.a2c_hyperparams,
                sac_hyperparams = EXCLUDED.sac_hyperparams,
                hp_source = EXCLUDED.hp_source,
                memory_enabled = EXCLUDED.memory_enabled
            """,
            (
                result_id,
                iteration,
                env,
                ensemble_sharpe,
                ensemble_mdd,
                gate_passed,
                weights.get("ppo"),
                weights.get("a2c"),
                weights.get("sac"),
                per_algo["ppo"]["mean_sharpe"] if per_algo["ppo"]["fold_count"] > 0 else None,
                per_algo["a2c"]["mean_sharpe"] if per_algo["a2c"]["fold_count"] > 0 else None,
                per_algo["sac"]["mean_sharpe"] if per_algo["sac"]["fold_count"] > 0 else None,
                per_algo["ppo"]["mean_mdd"] if per_algo["ppo"]["fold_count"] > 0 else None,
                per_algo["a2c"]["mean_mdd"] if per_algo["a2c"]["fold_count"] > 0 else None,
                per_algo["sac"]["mean_mdd"] if per_algo["sac"]["fold_count"] > 0 else None,
                total_folds,
                hyperparams.get("ppo"),
                hyperparams.get("a2c"),
                hyperparams.get("sac"),
                hp_source,
                "baseline",
                None,  # wall_clock_s — genuinely lost (not in any other source)
                memory_enabled,
            ),
        )
    conn.commit()


def recover_environment(
    conn: psycopg.Connection, env: str, iteration: int, memory_enabled: bool
) -> dict[str, object]:
    """Recover one (env, iteration) iteration_results row. Returns a summary."""
    log.info("recover_started", env=env, iteration=iteration)

    per_algo = load_per_algo_data(conn, env, iteration)
    total_folds = sum(int(stats["fold_count"]) for stats in per_algo.values())
    if total_folds == 0:
        log.warning("recover_no_data", env=env, iteration=iteration)
        return {"env": env, "iteration": iteration, "skipped": True}

    sharpes = {algo: float(stats["mean_sharpe"]) for algo, stats in per_algo.items()}
    weights = sharpe_softmax_weights(sharpes)

    ensemble_sharpe, ensemble_mdd = compute_ensemble_metrics(per_algo, weights)
    gate_passed = ensemble_sharpe > _GATE_MIN_SHARPE and abs(ensemble_mdd) < _GATE_MAX_MDD

    hyperparams = load_hyperparams_from_meta_decisions(conn, env, iteration)

    upsert_iteration_results(
        conn=conn,
        iteration=iteration,
        env=env,
        ensemble_sharpe=ensemble_sharpe,
        ensemble_mdd=ensemble_mdd,
        gate_passed=gate_passed,
        weights=weights,
        per_algo=per_algo,
        hyperparams=hyperparams,
        total_folds=total_folds,
        memory_enabled=memory_enabled,
    )

    log.info(
        "recover_complete",
        env=env,
        iteration=iteration,
        ensemble_sharpe=round(ensemble_sharpe, 4),
        ensemble_mdd=round(ensemble_mdd, 4),
        gate_passed=gate_passed,
        weights={k: round(v, 4) for k, v in weights.items()},
        total_folds=total_folds,
    )

    return {
        "env": env,
        "iteration": iteration,
        "ensemble_sharpe": ensemble_sharpe,
        "ensemble_mdd": ensemble_mdd,
        "gate_passed": gate_passed,
        "weights": weights,
        "total_folds": total_folds,
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Iteration number to recover (e.g., 5)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=("equity", "crypto"),
        help="Environment to recover (default: both)",
    )
    parser.add_argument(
        "--memory-enabled",
        action="store_true",
        default=True,
        help="Set memory_enabled=True on the recovered row (default true)",
    )
    args = parser.parse_args()

    db_url = get_database_url()
    envs = (args.env,) if args.env else ("equity", "crypto")

    print()
    print("=" * 70)
    print(f"RECOVERING iteration_results FOR ITER {args.iteration}")
    print("=" * 70)
    print()

    with psycopg.connect(db_url) as conn:
        for env in envs:
            summary = recover_environment(
                conn, env=env, iteration=args.iteration, memory_enabled=args.memory_enabled
            )
            if summary.get("skipped"):
                print(f"  {env}: SKIPPED (no fold data found)")
                continue
            weights_obj = summary["weights"]
            if not isinstance(weights_obj, dict):
                # Defensive — recover_environment always returns a dict here.
                continue
            weights: dict[str, float] = weights_obj
            sharpe_v = float(summary["ensemble_sharpe"])  # type: ignore[arg-type]
            mdd_v = float(summary["ensemble_mdd"])  # type: ignore[arg-type]
            total_v = int(summary["total_folds"])  # type: ignore[call-overload]
            print(
                f"  {env}: gate_passed={summary['gate_passed']}  "
                f"sharpe={sharpe_v:.4f}  mdd={mdd_v:.4f}  folds={total_v}"
            )
            print(
                f"         weights: ppo={weights['ppo']:.4f}  "
                f"a2c={weights['a2c']:.4f}  sac={weights['sac']:.4f}"
            )

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
