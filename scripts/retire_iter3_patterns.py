"""One-time script to retire invalid/misleading consolidation patterns from iter 3.

These patterns were produced when the epoch advice provider chain was completely
non-functional (0% crypto delivery, 28% equity delivery from 1.5b garbage model).
The [CTRL] vs [TREATMENT] labels were misleading because treatment folds received
no actual advice. The patterns incorrectly blame reward shaping for performance
gaps that are actually fold selection bias.

Pattern IDs to retire:
  155 — cross_env_correlation: "Control outperforms treatment in both envs" (INVALID)
  147 — iteration_regression: "PPO treatment regressed, reward adjustments hurt" (INVALID)
  142 — iteration_regression: "Control higher Sharpe for all equity algos" (INVALID)
  146 — reward_shaping: "SAC adjustments mixed effectiveness" (MISLEADING - garbage data)
  148 — hp_effectiveness: "A2C lr=0.0003 exceeds safe range" (WRONG - blames lr, misses gamma)
  154 — hp_effectiveness: "HP primary driver because weights stable" (WRONG reasoning)

Usage:
    # From homelab, run inside the memory service container:
    docker exec swingrl-memory python3 /tmp/retire_iter3_patterns.py

    # Or via SSH:
    scp scripts/retire_iter3_patterns.py homelab:/tmp/
    ssh homelab "docker cp /tmp/retire_iter3_patterns.py swingrl-memory:/tmp/ && \
        docker exec swingrl-memory python3 /tmp/retire_iter3_patterns.py"
"""

from __future__ import annotations

import os
import sys

import psycopg
from psycopg.rows import dict_row

# Pattern IDs to retire — all from iter 3 consolidation (2026-04-01)
RETIRE_IDS = [155, 147, 142, 146, 148, 154]
REASON = "iter3_advice_chain_failure"

# Memory service DB URL (inside container)
DB_URL = os.environ.get(
    "MEMORY_DATABASE_URL",
    "postgresql://swingrl:changeme@localhost:5432/memory",  # pragma: allowlist secret
)


def main() -> int:
    try:
        conn = psycopg.connect(DB_URL, row_factory=dict_row)
    except Exception as exc:
        print(f"ERROR: Cannot connect to database: {exc}")
        return 1

    # Show current state of patterns
    print("=== BEFORE ===")
    for pid in RETIRE_IDS:
        row = conn.execute(
            "SELECT id, category, confidence, status, pattern_text FROM consolidations WHERE id = %s",
            (pid,),
        ).fetchone()
        if row:
            print(
                f"  ID={row['id']} | {row['category']} | conf={row['confidence']} "
                f"| status={row['status']} | {row['pattern_text'][:80]}..."
            )
        else:
            print(f"  ID={pid} — NOT FOUND")

    # Retire patterns — use individual parameterized updates to avoid bandit B608
    retired_count = 0
    for pid in RETIRE_IDS:
        cursor = conn.execute(
            "UPDATE consolidations SET status = 'retired', superseded_by = %s WHERE id = %s",
            [REASON, pid],
        )
        retired_count += cursor.rowcount
    conn.commit()
    print(f"\n=== RETIRED {retired_count} patterns ===")

    # Verify
    print("\n=== AFTER ===")
    for pid in RETIRE_IDS:
        row = conn.execute(
            "SELECT id, status, superseded_by FROM consolidations WHERE id = %s",
            (pid,),
        ).fetchone()
        if row:
            print(
                f"  ID={row['id']} | status={row['status']} | superseded_by={row['superseded_by']}"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
