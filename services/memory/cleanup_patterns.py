"""One-time pattern cleanup migration for consolidation quality issues.

Retires stale/redundant/noisy patterns and restores falsely superseded ones.
Idempotent: checks current status before each update — safe to run multiple times.

Usage:
    docker exec swingrl-memory python3 cleanup_patterns.py
"""

from __future__ import annotations

import os

import psycopg
import structlog
from psycopg.rows import dict_row

log = structlog.get_logger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
)

# Patterns to retire with reasons
RETIRE_IDS: dict[int, str] = {
    # Stale iter-0 patterns (claim "no reward adjustments" — no longer true)
    62: "stale_iter0_no_reward_adjustments_equity",
    69: "stale_iter0_no_reward_adjustments_crypto",
    73: "stale_iter0_no_reward_adjustments_cross_env",
    # Redundant patterns (same signal as patterns needed for epoch_advice routing)
    134: "redundant_ppo_decline_duplicate_of_60",
    139: "redundant_ppo_decline_duplicate_of_68",
    110: "redundant_ppo_regression_duplicate_of_68",
    125: "redundant_yield_spread_duplicate_of_140",
    # Noise / miscategorized
    111: "noise_sac_crypto_trend_slopes_zero",
    130: "miscategorized_as_data_size_impact",
}

# Patterns to restore (falsely superseded by coarse conflict detection)
RESTORE_IDS: dict[int, str] = {
    127: "false_conflict_with_128_different_dimensions",
    132: "false_conflict_with_136_different_dimensions",
}


def run_cleanup() -> None:
    """Execute pattern cleanup migration."""
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    retired_count = 0
    restored_count = 0

    try:
        for row_id, reason in RETIRE_IDS.items():
            current = conn.execute(
                "SELECT status FROM consolidations WHERE id = %s", [row_id]
            ).fetchone()
            if current is None:
                log.warning("pattern_not_found", id=row_id)
                continue
            if current["status"] == "retired":
                log.info("pattern_already_retired", id=row_id)
                continue
            conn.execute(
                "UPDATE consolidations SET status = 'retired' WHERE id = %s",
                [row_id],
            )
            retired_count += 1
            log.info("pattern_retired", id=row_id, reason=reason)

        for row_id, reason in RESTORE_IDS.items():
            current = conn.execute(
                "SELECT status, superseded_by FROM consolidations WHERE id = %s",
                [row_id],
            ).fetchone()
            if current is None:
                log.warning("pattern_not_found", id=row_id)
                continue
            if current["status"] == "active":
                log.info("pattern_already_active", id=row_id)
                continue
            conn.execute(
                "UPDATE consolidations SET status = 'active', superseded_by = NULL, "
                "conflict_group_id = NULL WHERE id = %s",
                [row_id],
            )
            restored_count += 1
            log.info("pattern_restored", id=row_id, reason=reason)

        conn.commit()
        log.info(
            "cleanup_complete",
            retired=retired_count,
            restored=restored_count,
        )
    except Exception:
        conn.rollback()
        log.error("cleanup_failed", exc_info=True)
    finally:
        conn.close()


if __name__ == "__main__":
    run_cleanup()
