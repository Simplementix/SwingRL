"""One-time manual cross-algo consolidation.

Sends all WF memories for each env in a single LLM call (not per-algo),
then runs stage 2 for cross-env patterns. Uses OpenRouter directly.

Usage (from inside swingrl-memory container):
    python3 /app/scripts/manual_cross_algo_consolidation.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import ssl
import time
import urllib.request
from typing import Any

# Must run from services/memory context (inside container)
from memory_agents.consolidate import (
    _PHASE_A_FEW_SHOT_EXAMPLES,
    _PHASE_A_SYSTEM_PROMPT,
    ConsolidateAgent,
)

from db import (
    archive_memories_async,
    get_active_consolidations_async,
    init_capacity_limiters,
    init_db,
    insert_consolidation_async,
)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DB_PATH = os.environ.get("MEMORY_DB_PATH", "/app/db/memory.db")
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    """Call OpenRouter with 20 min timeout."""
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 32768,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    req = urllib.request.Request(BASE_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + API_KEY)

    start = time.time()
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=1200, context=ctx) as resp:  # nosec B310
            elapsed = time.time() - start
            body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            finish = body["choices"][0]["finish_reason"]
            usage = body.get("usage", {})
            print(f"  Time: {elapsed:.1f}s | Finish: {finish}")
            print(f"  Usage: {usage}")
            if content:
                return json.loads(content)
            print(f"  Empty content! finish={finish}")
            return None
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        print(f"  HTTP {e.code} after {elapsed:.1f}s: {e.read().decode()[:300]}")
        return None
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {type(e).__name__} after {elapsed:.1f}s: {e!s:.200}")
        return None


async def run() -> None:
    """Run cross-algo consolidation for both envs + stage 2."""
    init_db()
    init_capacity_limiters()
    agent = ConsolidateAgent()
    conn = sqlite3.connect(DB_PATH)

    for env_name in ("equity", "crypto"):
        rows = conn.execute(
            "SELECT id, text FROM memories WHERE source LIKE ? AND archived=0",
            (f"walk_forward:{env_name}%",),
        ).fetchall()

        if not rows:
            print(f"{env_name}: no unarchived memories, skipping")
            continue

        ids = [r[0] for r in rows]
        texts = "\n\n".join(f"- {r[1]}" for r in rows)
        print(f"\n{env_name}: {len(rows)} memories, {len(texts)} chars")

        user_prompt = (
            "--- FORMAT TEMPLATE (fictional data, for output structure only) ---\n"
            f"{_PHASE_A_FEW_SHOT_EXAMPLES}\n"
            "--- END FORMAT TEMPLATE ---\n\n"
            f"<training_data>\n{texts}\n</training_data>\n\n"
            "Analyze ONLY the data between the <training_data> tags. Return a JSON "
            "object matching the schema. 0-5 patterns. Empty array is acceptable."
        )

        total_chars = len(_PHASE_A_SYSTEM_PROMPT) + len(user_prompt)
        print(f"  Total input: {total_chars} chars (~{total_chars // 4} tokens)")

        parsed = call_openrouter(_PHASE_A_SYSTEM_PROMPT, user_prompt)
        if parsed is None:
            print(f"  {env_name} FAILED — no response")
            continue

        patterns = parsed.get("patterns", [])
        print(f"  Patterns: {len(patterns)}")

        for p in patterns:
            if not agent._validate_single_pattern(p):
                print("    SKIPPED invalid pattern")
                continue
            row_id = await insert_consolidation_async(
                pattern_text=p["pattern_text"],
                source_count=len(rows),
                category=p.get("category"),
                affected_algos=p.get("affected_algos"),
                affected_envs=p.get("affected_envs"),
                confidence=p.get("confidence"),
                evidence=p.get("evidence"),
                actionable_implication=p.get("actionable_implication"),
                stage=1,
                env_name=env_name,
            )
            cat = p.get("category", "?")
            conf = p.get("confidence", "?")
            txt = p.get("pattern_text", "")[:120]
            print(f"    Inserted #{row_id} [{cat}] conf={conf} {txt}")

        await archive_memories_async(ids)
        print(f"  Archived {len(ids)} memories")

    conn.close()

    # Stage 2: cross-env
    equity_pats = await get_active_consolidations_async(env_name="equity", stage=1)
    crypto_pats = await get_active_consolidations_async(env_name="crypto", stage=1)
    print(f"\nStage 2: equity={len(equity_pats)} crypto={len(crypto_pats)} stage1 patterns")

    if equity_pats and crypto_pats:
        count = await agent.run_stage2(equity_pats, crypto_pats)
        print(f"Stage 2 patterns created: {count}")
    else:
        print("Skipping stage 2 — need patterns from both envs")


if __name__ == "__main__":
    asyncio.run(run())
