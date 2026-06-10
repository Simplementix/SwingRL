"""Pre-live validation script for memory agent data integrity.

Validates PostgreSQL schema tables and columns required for the memory agent.
Optionally checks memory agent connectivity when --memory-url is provided.

Schema-only mode (default):
    python scripts/validate_memory.py --config config/swingrl.yaml

With memory agent connectivity check:
    python scripts/validate_memory.py --config config/swingrl.yaml \
        --memory-url http://localhost:8889

Exit codes:
    0 — All checks passed
    1 — One or more checks failed (diagnostic printed to stderr)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import psycopg
import structlog
from psycopg.rows import dict_row

from swingrl.config.schema import load_config
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected PostgreSQL schema requirements
# ---------------------------------------------------------------------------

# Tables that must exist in market_data.db
REQUIRED_TABLES: list[str] = [
    "training_epochs",
    "meta_decisions",
    "reward_adjustments",
    "training_runs",
]

# Columns required in hmm_state_history (3-state schema)
HMM_REQUIRED_COLUMNS: list[str] = ["p_bull", "p_bear", "p_crisis"]

# Meta-training columns required in training_runs (added by ALTER TABLE migration)
TRAINING_RUNS_META_COLUMNS: list[str] = [
    "run_type",
    "meta_rationale",
    "dominant_regime",
]

# Minimum training run summaries for cold-start guard
MIN_TRAINING_RUNS: int = 3


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _check_table_exists(conn: Any, table_name: str) -> bool:
    """Check if a table exists in PostgreSQL.

    Args:
        conn: Active PostgreSQL connection.
        table_name: Table name to check.

    Returns:
        True if table exists.
    """
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s",
        [table_name],
    ).fetchone()
    return bool(result and result[0] > 0)


def _get_table_columns(conn: Any, table_name: str) -> list[str]:
    """Get column names for a table.

    Args:
        conn: Active PostgreSQL connection.
        table_name: Table name to inspect.

    Returns:
        List of column names (empty if table doesn't exist).
    """
    try:
        result = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            [table_name],
        ).fetchall()
        return [row[0] for row in result]
    except Exception:
        return []


def _count_rows(conn: Any, table_name: str) -> int:
    """Count rows in a table.

    Args:
        conn: Active PostgreSQL connection.
        table_name: Table name to count.

    Returns:
        Row count, or -1 if the query fails.
    """
    try:
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()  # noqa: S608  # nosec B608
        return int(result[0]) if result else 0
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Schema-only validation (no memory agent connectivity required)
# ---------------------------------------------------------------------------


def validate_schema(database_url: str) -> list[str]:
    """Validate PostgreSQL schema for memory agent requirements.

    Checks:
    - Required tables exist (training_epochs, meta_decisions, reward_adjustments,
      training_runs)
    - hmm_state_history has p_crisis column (3-state schema)
    - training_runs has meta-training columns (run_type, meta_rationale,
      dominant_regime)
    - At least MIN_TRAINING_RUNS training run summaries exist

    Args:
        database_url: PostgreSQL database connection URL.

    Returns:
        List of failure messages. Empty list means all checks passed.
    """
    failures: list[str] = []

    log.info("schema_validation_started", database_url=database_url)

    try:
        conn = psycopg.connect(database_url, row_factory=dict_row)
    except Exception as exc:
        failures.append(f"Cannot connect to PostgreSQL: {exc}")
        log.error("pg_connect_failed", database_url=database_url, error=str(exc))
        return failures

    try:
        # ── 1. Required tables ───────────────────────────────────────────────
        for table in REQUIRED_TABLES:
            if _check_table_exists(conn, table):
                row_count = _count_rows(conn, table)
                log.info("table_ok", table=table, rows=row_count)
            else:
                msg = f"Required table missing: {table}"
                failures.append(msg)
                log.error("table_missing", table=table)

        # ── 2. hmm_state_history 3-state schema ──────────────────────────────
        if _check_table_exists(conn, "hmm_state_history"):
            existing_cols = _get_table_columns(conn, "hmm_state_history")
            for col in HMM_REQUIRED_COLUMNS:
                if col in existing_cols:
                    log.info("hmm_column_ok", column=col)
                else:
                    msg = f"hmm_state_history missing 3-state column: {col}"
                    failures.append(msg)
                    log.error("hmm_column_missing", column=col, table="hmm_state_history")
        else:
            log.warning(
                "hmm_state_history_not_found",
                note="Table may not exist until first HMM run; skipping column check",
            )

        # ── 3. training_runs meta-training columns ───────────────────────────
        if _check_table_exists(conn, "training_runs"):
            existing_cols = _get_table_columns(conn, "training_runs")
            for col in TRAINING_RUNS_META_COLUMNS:
                if col in existing_cols:
                    log.info("training_runs_column_ok", column=col)
                else:
                    msg = f"training_runs missing meta column (ALTER TABLE not applied): {col}"
                    failures.append(msg)
                    log.error(
                        "training_runs_meta_column_missing",
                        column=col,
                        hint="Run ALTER TABLE training_runs ADD COLUMN migration",
                    )

            # ── 4. Cold-start guard: minimum training runs ──────────────────
            run_count = _count_rows(conn, "training_runs")
            if run_count >= 0:
                log.info("training_runs_count", count=run_count, minimum=MIN_TRAINING_RUNS)
                if run_count < MIN_TRAINING_RUNS:
                    # Informational warning — not a hard failure (first-time setup)
                    log.warning(
                        "training_runs_below_threshold",
                        count=run_count,
                        minimum=MIN_TRAINING_RUNS,
                        note="Cold-start guard will hold meta-trainer passive until threshold met",
                    )
            else:
                msg = "Cannot count rows in training_runs"
                failures.append(msg)
                log.error("training_runs_count_failed")

        # ── 5. Row counts for all present tables ─────────────────────────────
        all_tables = ["training_epochs", "meta_decisions", "reward_adjustments"]
        for table in all_tables:
            if _check_table_exists(conn, table):
                count = _count_rows(conn, table)
                log.info("table_row_count", table=table, rows=count)

    finally:
        conn.close()

    return failures


# ---------------------------------------------------------------------------
# Memory agent connectivity check (requires --memory-url)
# ---------------------------------------------------------------------------


def _query_memory_agent(memory_url: str, query: str, timeout: float = 5.0) -> dict[str, Any]:
    """Query the memory agent /query endpoint.

    Args:
        memory_url: Base URL of the memory agent.
        query: Query string to send.
        timeout: Request timeout in seconds.

    Returns:
        Dict with 'ok' (bool) and 'response' (any) keys.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({"q": query})
    url = f"{memory_url.rstrip('/')}/query?{params}"

    try:
        req = urllib.request.Request(url, method="GET")  # noqa: S310  # nosec B310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            status: int = resp.status
            if 200 <= status < 300:
                body = json.loads(resp.read().decode("utf-8"))
                return {"ok": True, "response": body}
            return {"ok": False, "response": f"HTTP {status}"}
    except Exception as exc:
        return {"ok": False, "response": str(exc)}


def _check_memory_health(memory_url: str, timeout: float = 5.0) -> bool:
    """Check memory agent /health endpoint.

    Args:
        memory_url: Base URL of the memory agent.
        timeout: Request timeout in seconds.

    Returns:
        True if health check passes.
    """
    import urllib.error
    import urllib.request

    url = f"{memory_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, method="GET")  # noqa: S310  # nosec B310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            status: int = resp.status
            return 200 <= status < 300
    except Exception as exc:
        log.debug("memory_health_failed", url=url, error=str(exc))
        return False


def validate_memory_agent(memory_url: str) -> list[str]:
    """Validate memory agent connectivity and training data population.

    Per memory_storage_spec.md Section 8: checks that training run summaries,
    backtest fills, and curriculum performance records exist.

    Args:
        memory_url: Base URL of the memory agent.

    Returns:
        List of failure messages. Empty list means all checks passed.
    """
    failures: list[str] = []

    log.info("memory_agent_validation_started", url=memory_url)

    # ── 1. Health endpoint ───────────────────────────────────────────────────
    if _check_memory_health(memory_url):
        log.info("memory_health_ok", url=memory_url)
    else:
        failures.append(f"Memory agent /health check failed: {memory_url}")
        log.error("memory_health_failed", url=memory_url)
        return failures  # No point continuing if agent is unreachable

    # ── 2. Training run summaries (cold-start guard threshold) ──────────────
    result = _query_memory_agent(memory_url, "TRAINING RUN SUMMARY")
    if result["ok"]:
        log.info("memory_query_training_runs_ok", has_response=True)
    else:
        failures.append(f"Memory query for TRAINING RUN SUMMARY failed: {result['response']}")
        log.error("memory_query_failed", query="TRAINING RUN SUMMARY", error=result["response"])

    # ── 3. Backtest fill records ─────────────────────────────────────────────
    result = _query_memory_agent(memory_url, "BACKTEST FILL")
    if result["ok"]:
        log.info("memory_query_backtest_fills_ok")
    else:
        log.warning(
            "memory_backtest_fills_empty",
            note="Expected if seed_memory_from_backtest.py has not been run yet",
        )

    # ── 4. Curriculum performance records ───────────────────────────────────
    result = _query_memory_agent(memory_url, "CURRICULUM PERFORMANCE")
    if result["ok"]:
        log.info("memory_query_curriculum_ok")
    else:
        log.warning(
            "memory_curriculum_empty",
            note="Expected if no training runs have completed yet",
        )

    # ── 5. Reward adjustment records ────────────────────────────────────────
    result = _query_memory_agent(memory_url, "REWARD_ADJUSTMENT_OUTCOME")
    if result["ok"]:
        log.info("memory_query_reward_adjustments_ok")
    else:
        log.info(
            "memory_reward_adjustments_empty",
            note="Expected if no weight changes occurred during training",
        )

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the validation CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pre-live validation script for SwingRL memory agent data integrity. "
            "Without --memory-url: schema-only checks against PostgreSQL. "
            "With --memory-url: additionally checks memory agent connectivity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/validate_memory.py --config config/swingrl.yaml
  python scripts/validate_memory.py --config config/swingrl.yaml \\
      --memory-url http://localhost:8889
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    parser.add_argument(
        "--memory-url",
        type=str,
        default=None,
        help=(
            "Memory agent base URL for connectivity checks. "
            "If omitted, only schema checks are performed (no network required)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run pre-live memory validation.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = all checks passed, 1 = one or more failures).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    database_url = config.system.database_url
    all_failures: list[str] = []

    # ── Schema validation (always runs) ─────────────────────────────────────
    log.info("validation_started", mode="schema_only" if not args.memory_url else "full")
    schema_failures = validate_schema(database_url)
    all_failures.extend(schema_failures)

    # ── Memory agent validation (only if --memory-url provided) ─────────────
    if args.memory_url:
        agent_failures = validate_memory_agent(args.memory_url)
        all_failures.extend(agent_failures)

    # ── Final result ─────────────────────────────────────────────────────────
    if all_failures:
        log.error(
            "validation_failed",
            failure_count=len(all_failures),
            failures=all_failures,
        )
        for msg in all_failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    log.info(
        "validation_passed",
        checks="schema_only" if not args.memory_url else "schema+memory_agent",
    )
    print("OK: All validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
