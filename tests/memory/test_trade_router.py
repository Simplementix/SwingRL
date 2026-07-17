"""Task 12: /trade/commentary endpoint + trade-commentary DB writer (spec §4.4).

The memory service (services/memory/) is its own runtime root, so its modules
import each other bare (``from db import ...``, ``from routers.trade import ...``).
That dir is put on ``sys.path`` here so those imports resolve against the service
(mirroring the running container), not against the empty repo-root ``db/``
namespace package. DB-gated: needs DATABASE_URL pointing at a scratch PostgreSQL
(the writers exercise the real llm_calls/intent_records tables). No real LLM call
is ever made — the provider method is mocked.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

_MEMORY_DIR = str(Path(__file__).parents[2] / "services" / "memory")
if _MEMORY_DIR not in sys.path:
    sys.path.insert(0, _MEMORY_DIR)
# Force the service's db.py to win over the empty repo-root db/ namespace package.
if "db" in sys.modules and getattr(sys.modules["db"], "__file__", None) is None:
    del sys.modules["db"]

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


@pytest.fixture
def memory_db_at_v004() -> Generator[Any, None, None]:
    """Scratch DB with V001–V004 applied; memory pool reset to the same DATABASE_URL.

    Teardown drops V004→V001 artifacts and clears ledger rows 1–4 (mirrors
    tests/data/conftest.py's db_with_legacy_schema) so the persistent scratch DB
    stays re-runnable.
    """
    import textwrap

    import db as memory_db  # services/memory/db.py
    from swingrl.config.schema import load_config
    from swingrl.data.db import DatabaseManager
    from swingrl.data.migration_runner import apply_migrations

    db_url = os.environ["DATABASE_URL"]
    cfg_yaml = textwrap.dedent(f"""\
        trading_mode: paper
        system:
          database_url: "{db_url}"
    """)
    # Build a swingrl DatabaseManager for schema setup/teardown.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "swingrl.yaml"
        cfg_path.write_text(cfg_yaml)
        config = load_config(cfg_path)
        DatabaseManager.reset()
        mgr = DatabaseManager(config)
        mgr.init_schema()
        apply_migrations(mgr)

        # Point the memory service's lazy pool at this same DATABASE_URL.
        memory_db._pool = None

        try:
            yield memory_db
        finally:
            memory_db._pool = None
            with mgr.connection() as conn:
                conn.execute(
                    "ALTER TABLE ensemble_weight_history DROP CONSTRAINT IF EXISTS fk_ewh_intent"
                )
                conn.execute("DROP INDEX IF EXISTS uq_mt_commentary_per_cycle")
                conn.execute("DROP INDEX IF EXISTS uq_llm_commentary_cycle")
                conn.execute("DROP TABLE IF EXISTS intent_verdicts")
                conn.execute("DROP TABLE IF EXISTS intent_applications")
                conn.execute("DROP TABLE IF EXISTS intent_records")
                conn.execute("DROP TABLE IF EXISTS llm_calls")
                conn.execute("DROP TABLE IF EXISTS event_outcomes")
                conn.execute("DROP TABLE IF EXISTS calendar_events")
                conn.execute("DROP TABLE IF EXISTS fill_quality")
                conn.execute("ALTER TABLE trades DROP COLUMN IF EXISTS cycle_id")
                conn.execute("DROP TABLE IF EXISTS cycle_algo_proposals")
                conn.execute("DROP TABLE IF EXISTS inference_cycles")
                conn.execute("DROP TABLE IF EXISTS ensemble_weight_history")
                conn.execute("DROP TABLE IF EXISTS models")
                conn.execute("DROP TABLE IF EXISTS training_runs")
                conn.execute(
                    "ALTER TABLE backtest_results "
                    "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_id"
                )
                conn.execute(
                    "ALTER TABLE iteration_results "
                    "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_ensemble_id"
                )
                conn.execute("DROP TABLE IF EXISTS eras")
                conn.execute("DROP TABLE IF EXISTS gate_versions")
                conn.execute(
                    "DO $$ BEGIN "
                    "IF to_regclass('public.schema_migrations') IS NOT NULL THEN "
                    "DELETE FROM schema_migrations WHERE version IN (1, 2, 3, 4); "
                    "END IF; END $$;"
                )
            DatabaseManager.reset()


def _insert_cycle(memory_db: Any, environment: str = "equity") -> int:
    """Insert an inference_cycles row via the memory pool; return cycle_id."""
    with memory_db.get_connection() as conn:
        cycle_id = conn.execute(
            "INSERT INTO inference_cycles (environment, mode, cycle_ts)"
            " VALUES (%s, 'paper', now()) RETURNING cycle_id",
            [environment],
        ).fetchone()["cycle_id"]
        conn.commit()
    return int(cycle_id)


# ---------------------------------------------------------------------------
# DB writer: atomic llm_calls + intent_records
# ---------------------------------------------------------------------------


def test_writer_writes_llm_call_and_intent_atomically(memory_db_at_v004: Any) -> None:
    """Task 12: insert_trade_commentary writes both rows in one transaction."""
    memory_db = memory_db_at_v004
    cycle_id = _insert_cycle(memory_db)

    llm_call_id, intent_id = memory_db.insert_trade_commentary(
        cycle_id=cycle_id,
        environment="equity",
        algorithm="ppo",
        deployed_iteration=5,
        provider="cerebras",
        model="qwen-3-235b",
        prompt_version="mt-commentary-v0",
        prompt_text="context...",
        response_text=json.dumps({"diagnosis": "ok"}),
        response_parsed=json.dumps({"diagnosis": "ok"}),
        latency_ms=42,
        success=True,
        error_text=None,
        tokens_in=100,
        tokens_out=50,
        evidence=json.dumps({"regime": {"vix": 15.0}}),
        proposal=json.dumps({"change": "none", "rationale": "stable"}),
        bet_metric="cycle_pnl_frac",
        bet_direction="up",
        bet_baseline_value=0.0,
        horizon_spec=json.dumps({"type": "wall_clock_hours", "hours": 24}),
    )

    with memory_db.get_connection() as conn:
        call = conn.execute(
            "SELECT coach, call_type, cycle_id, provider, prompt_version FROM llm_calls"
            " WHERE llm_call_id = %s",
            [llm_call_id],
        ).fetchone()
        intent = conn.execute(
            "SELECT llm_call_id, coach, lever, mode, horizon_spec FROM intent_records"
            " WHERE intent_id = %s",
            [intent_id],
        ).fetchone()

    assert call["coach"] == "meta_trader"
    assert call["call_type"] == "trade_commentary"
    assert call["cycle_id"] == cycle_id
    assert call["prompt_version"] == "mt-commentary-v0"
    assert intent["llm_call_id"] == llm_call_id
    assert intent["coach"] == "meta_trader"
    assert intent["lever"] == "MT_commentary"
    assert intent["mode"] == "shadow"
    assert intent["horizon_spec"] == {"type": "wall_clock_hours", "hours": 24}


def test_writer_rolls_back_llm_call_when_intent_invalid(memory_db_at_v004: Any) -> None:
    """Task 12: a failing intent insert rolls back the llm_calls insert (atomic)."""
    memory_db = memory_db_at_v004
    cycle_id = _insert_cycle(memory_db)

    with pytest.raises(Exception):  # noqa: B017,PT011 — CheckViolation on bad bet_direction
        memory_db.insert_trade_commentary(
            cycle_id=cycle_id,
            environment="equity",
            algorithm="ppo",
            deployed_iteration=5,
            provider="cerebras",
            model="qwen-3",
            prompt_version="mt-commentary-v0",
            prompt_text="ctx",
            response_text=None,
            response_parsed=None,
            latency_ms=1,
            success=True,
            error_text=None,
            tokens_in=None,
            tokens_out=None,
            evidence=json.dumps({}),
            proposal=json.dumps({}),
            bet_metric="cycle_pnl_frac",
            bet_direction="sideways",  # violates CHECK (up, down)
            bet_baseline_value=0.0,
            horizon_spec=json.dumps({"type": "wall_clock_hours", "hours": 24}),
        )

    with memory_db.get_connection() as conn:
        orphan = conn.execute(
            "SELECT count(*) AS n FROM llm_calls WHERE cycle_id = %s", [cycle_id]
        ).fetchone()
    assert orphan["n"] == 0


# ---------------------------------------------------------------------------
# Endpoint: POST /trade/commentary
# ---------------------------------------------------------------------------


def _mount_trade_app() -> Any:
    """Build a minimal FastAPI app with only the trade router + auth overridden."""
    from auth import verify_api_key
    from fastapi import FastAPI
    from routers import trade
    from starlette.testclient import TestClient

    app = FastAPI()
    app.include_router(trade.router, prefix="/trade")
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    return TestClient(app)


def test_endpoint_writes_llm_call_and_intent(memory_db_at_v004: Any) -> None:
    """Task 12: POST /trade/commentary renders v0 prompt, calls provider (mocked),
    writes llm_calls + one MT_commentary intent with the env's horizon_spec."""
    memory_db = memory_db_at_v004
    cycle_id = _insert_cycle(memory_db, environment="crypto")

    fake_result = {
        "diagnosis": "trend-following into resistance",
        "matched_weakness": "trade_shy",
        "proposal": "hold current blend; no change",
        "falsifiable_bet": {
            "metric": "cycle_pnl_frac",
            "direction": "up",
            "baseline_value": 0.0,
        },
        "provider": "cerebras",
        "model": "qwen-3-235b",
    }

    client = _mount_trade_app()
    with patch(
        "routers.trade.QueryAgent.advise_trade_commentary",
        new=AsyncMock(return_value=fake_result),
    ):
        resp = client.post(
            "/trade/commentary",
            json={
                "cycle_id": cycle_id,
                "environment": "crypto",
                "algorithm": "sac",
                "deployed_iteration": 3,
                "regime": {"hmm_p_bull": 0.55, "vix": 18.0, "turbulence": 2.1},
                "proposals_summary": "sac: BTC +0.3; ppo: BTC +0.1",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent_id"] is not None

    with memory_db.get_connection() as conn:
        intent = conn.execute(
            "SELECT lever, mode, environment, algorithm, horizon_spec, bet_metric, bet_direction"
            " FROM intent_records WHERE intent_id = %s",
            [body["intent_id"]],
        ).fetchone()
        call = conn.execute(
            "SELECT call_type, cycle_id, prompt_version FROM llm_calls WHERE llm_call_id = %s",
            [body["llm_call_id"]],
        ).fetchone()

    assert intent["lever"] == "MT_commentary"
    assert intent["mode"] == "shadow"
    assert intent["environment"] == "crypto"
    assert intent["horizon_spec"] == {"type": "next_n_cycles", "n": 6}
    assert intent["bet_metric"] == "cycle_pnl_frac"
    assert intent["bet_direction"] == "up"
    assert call["call_type"] == "trade_commentary"
    assert call["cycle_id"] == cycle_id
    assert call["prompt_version"] == "mt-commentary-v0"


def test_endpoint_second_commentary_same_cycle_bounces(memory_db_at_v004: Any) -> None:
    """Task 12 (A14): the ≤1-intent-per-cycle cap holds end-to-end via the endpoint."""
    memory_db = memory_db_at_v004
    cycle_id = _insert_cycle(memory_db, environment="equity")

    fake_result = {
        "diagnosis": "d",
        "matched_weakness": "w",
        "proposal": "p",
        "falsifiable_bet": {"metric": "cycle_pnl_frac", "direction": "down", "baseline_value": 0.0},
        "provider": "cerebras",
        "model": "m",
    }
    client = _mount_trade_app()
    payload = {
        "cycle_id": cycle_id,
        "environment": "equity",
        "algorithm": "ppo",
        "deployed_iteration": 5,
        "regime": {},
        "proposals_summary": "ppo: SPY +0.1",
    }
    with patch(
        "routers.trade.QueryAgent.advise_trade_commentary",
        new=AsyncMock(return_value=fake_result),
    ):
        first = client.post("/trade/commentary", json=payload)
        second = client.post("/trade/commentary", json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 409, second.text

    with memory_db.get_connection() as conn:
        count = conn.execute(
            "SELECT count(*) AS n FROM intent_records ir JOIN llm_calls lc"
            " ON ir.llm_call_id = lc.llm_call_id WHERE lc.cycle_id = %s",
            [cycle_id],
        ).fetchone()
    assert count["n"] == 1
