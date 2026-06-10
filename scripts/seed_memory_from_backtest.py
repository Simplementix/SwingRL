"""Seed memory agent with backtest fill records before live trading.

Queries evaluation fills from PostgreSQL, joins HMM regime labels,
formats each fill as plain text, and ingests via MemoryClient.ingest().
Triggers consolidation after all fills.

Run once before enabling live trading to give the memory agent historical
trading context.

Usage:
    python scripts/seed_memory_from_backtest.py \\
        --config config/swingrl.yaml \\
        --memory-url http://localhost:8889

Graceful handling:
    - Missing fills table or no evaluation fills → exit 0 with info message
    - Missing hmm_state_history → regime labeled "unknown"
    - Memory agent unavailable → fails fast with exit code 1 (seeding requires
      memory agent to be reachable; schema-only mode is not meaningful here)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import psycopg
import structlog
from psycopg.rows import dict_row

from swingrl.config.schema import load_config
from swingrl.memory.client import MemoryClient
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Fill loading helpers
# ---------------------------------------------------------------------------


def _load_fills(database_url: str) -> list[dict[str, Any]]:
    """Load evaluation fills from PostgreSQL.

    Queries fills joined with training_runs to get reward weights.
    Only ingests fills from run_type = 'evaluation' rows.

    Args:
        database_url: PostgreSQL connection URL.

    Returns:
        List of fill dicts. Empty list if no evaluation fills exist or
        the fills/training_runs tables are missing.
    """
    try:
        with psycopg.connect(database_url, row_factory=dict_row) as conn:
            # Verify fills table exists
            result = conn.execute(
                "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public' AND tablename = 'fills'"
            ).fetchone()
            if result[0] == 0:
                log.info("fills_table_missing", note="No fills to seed (table not yet created)")
                return []

            # Verify training_runs table exists
            result = conn.execute(
                "SELECT COUNT(*) FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename = 'training_runs'"
            ).fetchone()
            if result[0] == 0:
                log.info(
                    "training_runs_table_missing",
                    note="No fills to seed (training_runs table not yet created)",
                )
                return []

            # Verify run_type column exists in training_runs
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name = 'training_runs' AND column_name = 'run_type'"
            ).fetchone()
            has_run_type = result[0] > 0

            if not has_run_type:
                log.warning(
                    "training_runs_run_type_missing",
                    note=(
                        "run_type column not found in training_runs; "
                        "seeding all fills (no evaluation filter applied)"
                    ),
                )
                rows = conn.execute(
                    """
                    SELECT f.symbol, f.env, f.side, f.algo, f.run_id,
                           f.signal_confidence, f.pnl, f.hold_steps,
                           f.curriculum_window, f.timestamp,
                           r.reward_weight_profit, r.reward_weight_sharpe,
                           r.reward_weight_drawdown
                    FROM fills f
                    JOIN training_runs r ON f.run_id = r.run_id
                    ORDER BY f.timestamp ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT f.symbol, f.env, f.side, f.algo, f.run_id,
                           f.signal_confidence, f.pnl, f.hold_steps,
                           f.curriculum_window, f.timestamp,
                           r.reward_weight_profit, r.reward_weight_sharpe,
                           r.reward_weight_drawdown
                    FROM fills f
                    JOIN training_runs r ON f.run_id = r.run_id
                    WHERE r.run_type = 'evaluation'
                    ORDER BY f.timestamp ASC
                    """
                ).fetchall()

    except Exception as exc:
        log.warning("fills_query_failed", error=str(exc), note="No fills to seed")
        return []

    cols = [
        "symbol",
        "env",
        "side",
        "algo",
        "run_id",
        "signal_confidence",
        "pnl",
        "hold_steps",
        "curriculum_window",
        "timestamp",
        "rw_profit",
        "rw_sharpe",
        "rw_drawdown",
    ]
    fills = [dict(zip(cols, [row[c] for c in cols], strict=False)) for row in rows]
    log.info("fills_loaded", count=len(fills))
    return fills


def _derive_regime(p_bull: float, p_bear: float, p_crisis: float) -> tuple[str, float]:
    """Derive dominant regime and confidence from HMM state probabilities.

    Args:
        p_bull: Probability of bull regime.
        p_bear: Probability of bear regime.
        p_crisis: Probability of crisis regime.

    Returns:
        Tuple of (dominant_regime, confidence) based on argmax of probabilities.
    """
    probs = {"bull": p_bull, "bear": p_bear, "crisis": p_crisis}
    dominant = max(probs, key=probs.get)  # type: ignore[arg-type]
    return (dominant, probs[dominant])


def _lookup_regime(
    pg_conn: Any,
    timestamp: str | None,
    env: str,
) -> tuple[str, float]:
    """Look up the dominant HMM regime at a given timestamp.

    Queries hmm_state_history for the most recent row at or before the
    fill timestamp. Derives dominant regime from argmax of p_bull, p_bear,
    p_crisis probabilities.

    Args:
        pg_conn: Active PostgreSQL connection.
        timestamp: Fill timestamp string (ISO 8601 or compatible).
        env: Environment name ("equity" or "crypto").

    Returns:
        Tuple of (dominant_regime, confidence). Returns ("unknown", 0.0)
        if the table is missing, empty, or lookup fails.
    """
    if timestamp is None:
        return ("unknown", 0.0)

    try:
        # Check if hmm_state_history table exists
        exists = pg_conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'hmm_state_history'"
        ).fetchone()
        if not exists or exists[0] == 0:
            return ("unknown", 0.0)

        # Query actual schema columns: environment, date, p_bull, p_bear, p_crisis
        row = pg_conn.execute(
            """
            SELECT p_bull, p_bear, p_crisis
            FROM hmm_state_history
            WHERE environment = %s AND date <= %s
            ORDER BY date DESC LIMIT 1
            """,
            [env, timestamp],
        ).fetchone()

        if row:
            return _derive_regime(float(row[0]), float(row[1]), float(row[2] or 0.0))

        # If no row found for this env/timestamp, try most recent overall
        row = pg_conn.execute(
            "SELECT p_bull, p_bear, p_crisis FROM hmm_state_history ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            return _derive_regime(float(row[0]), float(row[1]), float(row[2] or 0.0))

        return ("unknown", 0.0)

    except Exception as exc:
        log.debug("regime_lookup_failed", error=str(exc))
        return ("unknown", 0.0)


def _format_fill_text(fill: dict[str, Any], dominant_regime: str, regime_conf: float) -> str:
    """Format a fill dict as plain text for memory agent ingestion.

    Per memory_storage_spec.md Section 4 required fields.

    Args:
        fill: Fill dict with symbol, env, side, algo, run_id, etc.
        dominant_regime: HMM dominant regime at fill time.
        regime_conf: HMM confidence at fill time.

    Returns:
        Formatted plain-text string.
    """
    signal_conf = fill.get("signal_confidence") or 0.0
    pnl = fill.get("pnl") or 0.0
    hold_steps = fill.get("hold_steps") or 0
    curriculum_window = fill.get("curriculum_window") or "unknown"
    rw_profit = fill.get("rw_profit") or 0.0
    rw_sharpe = fill.get("rw_sharpe") or 0.0
    rw_drawdown = fill.get("rw_drawdown") or 0.0

    return (
        f"BACKTEST FILL: symbol={fill['symbol']} env={fill['env']} "
        f"side={fill['side']} algo={fill['algo']} run_id={fill['run_id']} "
        f"signal_confidence={float(signal_conf):.2f} "
        f"simulated_pnl={float(pnl):.4f} "
        f"hold_duration_steps={int(hold_steps)} "
        f"regime_at_fill={dominant_regime} "
        f"regime_confidence={float(regime_conf):.2f} "
        f"curriculum_window={curriculum_window} "
        f"reward_weight_profit={float(rw_profit):.2f} "
        f"reward_weight_sharpe={float(rw_sharpe):.2f} "
        f"reward_weight_drawdown={float(rw_drawdown):.2f}"
    )


# ---------------------------------------------------------------------------
# Main seeding logic
# ---------------------------------------------------------------------------


def seed(config_path: str, memory_url: str, api_key: str = "") -> int:
    """Seed memory agent with backtest fill records.

    Args:
        config_path: Path to config YAML file.
        memory_url: Memory agent base URL.
        api_key: API key for memory agent authentication.

    Returns:
        Exit code (0 = success or no fills, 1 = error).
    """
    config = load_config(config_path)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    database_url = config.system.database_url

    log.info(
        "seeding_started",
        database_url=database_url,
        memory_url=memory_url,
    )

    # ── Load fills from PostgreSQL ───────────────────────────────────────────
    fills = _load_fills(database_url)

    if not fills:
        log.info(
            "no_fills_to_seed",
            note="No evaluation fills found. Memory seeding skipped (exit 0).",
        )
        print("INFO: No evaluation fills found. Memory seeding skipped.")
        return 0

    # ── Open PostgreSQL for regime lookups ─────────────────────────────────────
    pg_conn: Any = None
    try:
        pg_conn = psycopg.connect(database_url, row_factory=dict_row)
        log.info("pg_opened", database_url=database_url)
    except Exception as exc:
        log.warning("pg_open_failed", database_url=database_url, error=str(exc))
        pg_conn = None

    # ── Ingest fills via MemoryClient ─────────────────────────────────────────
    client = MemoryClient(base_url=memory_url, api_key=api_key)
    ingested = 0
    failed = 0

    try:
        for fill in fills:
            dominant_regime, regime_conf = (
                _lookup_regime(pg_conn, fill.get("timestamp"), fill.get("env", "equity"))
                if pg_conn is not None
                else ("unknown", 0.0)
            )

            text = _format_fill_text(fill, dominant_regime, regime_conf)

            ok = client.ingest({"text": text, "source": "backtest_fill:historical"})
            if ok:
                ingested += 1
                log.debug(
                    "fill_ingested",
                    symbol=fill["symbol"],
                    run_id=fill["run_id"],
                    regime=dominant_regime,
                )
            else:
                failed += 1
                log.warning(
                    "fill_ingest_failed",
                    symbol=fill.get("symbol"),
                    run_id=fill.get("run_id"),
                )

    finally:
        if pg_conn is not None:
            pg_conn.close()

    log.info("seeding_complete", ingested=ingested, failed=failed, total=len(fills))

    # ── Trigger consolidation ─────────────────────────────────────────────────
    if ingested > 0:
        log.info("consolidation_triggered")
        consolidated = client.consolidate()
        if consolidated:
            log.info("consolidation_ok")
        else:
            log.warning("consolidation_failed", note="Consolidation may run on next schedule")

    print(f"Seeded {ingested}/{len(fills)} backtest fills. Consolidation triggered.")

    if failed > 0:
        log.error("some_fills_failed", failed=failed)
        return 1

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the seeding CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Seed the memory agent with backtest fill records before live trading. "
            "Queries evaluation fills from PostgreSQL, joins HMM regime data, "
            "and ingests as plain text via MemoryClient."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/seed_memory_from_backtest.py \\
      --config config/swingrl.yaml \\
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
        default="http://localhost:8889",
        help="Memory agent base URL (default: http://localhost:8889).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for memory agent authentication (default: empty, no auth).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Seed memory agent with backtest fills.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success or no fills to seed, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return seed(args.config, args.memory_url, api_key=args.api_key)


if __name__ == "__main__":
    sys.exit(main())
