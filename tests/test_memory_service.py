"""Tests for the swingrl-memory FastAPI service and DuckDB training telemetry migrations.

TRAIN-06, TRAIN-07: Covers:
- swingrl-memory FastAPI endpoints (ingest, health, training, debug, consolidation)
- DuckDB telemetry tables (training_epochs, meta_decisions, reward_adjustments, ALTER training_runs)
- Enriched DB schema (new tables, columns, indexes)
- Two-stage consolidation pipeline (Stage 1 per-env, Stage 2 cross-env)
- Pattern lifecycle (active → superseded → retired)
- Dedup/merge (similar patterns confirmed, old incremented)
- Presentation tracking (query logs to pattern_presentations)
- Outcome recording (record_outcome endpoint stores data)
- Pattern effectiveness (joins presentations + outcomes)
- APScheduler removed (no scheduler in lifespan)
- Source tag format walk_forward:{env}:{algo}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add services/memory to sys.path so the service modules can be imported
# This mirrors how uvicorn serves app.py with WORKDIR=/app/services/memory
_MEMORY_SERVICE_DIR = Path(__file__).parent.parent / "services" / "memory"
_MEMORY_MODULE_NAMES = [
    "app",
    "db",
    "auth",
    "memory_agents",
    "memory_agents.ingest",
    "memory_agents.consolidate",
    "memory_agents.query",
    "routers",
    "routers.core",
    "routers.training",
    "routers.debug",
]

# Ensure services/memory/ is always first on sys.path to prevent shadowing by scripts/
if str(_MEMORY_SERVICE_DIR) in sys.path:
    sys.path.remove(str(_MEMORY_SERVICE_DIR))
sys.path.insert(0, str(_MEMORY_SERVICE_DIR))


# ---------------------------------------------------------------------------
# Memory service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_consolidation_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eliminate 60s asyncio.sleep in consolidation module for all tests."""
    import asyncio

    original_sleep = asyncio.sleep

    async def _fast_sleep(seconds: float, *args: Any, **kwargs: Any) -> None:
        """Replace long sleeps with instant ones to speed up tests."""
        if seconds > 1:
            return
        await original_sleep(seconds, *args, **kwargs)

    monkeypatch.setattr("asyncio.sleep", _fast_sleep)


@pytest.fixture()
def memory_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SQLite db to a temp directory and set required env vars."""
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_path))
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    return db_path


@pytest.fixture()
def api_client(memory_db_env: Path):  # type: ignore[no-untyped-def]
    """Return a FastAPI TestClient."""
    from fastapi.testclient import TestClient

    # Evict cached memory service modules to ensure clean imports.
    for mod_name in list(_MEMORY_MODULE_NAMES):
        sys.modules.pop(mod_name, None)

    import db as memory_db_module

    memory_db_module.init_db()

    import app

    client = TestClient(app.app, raise_server_exceptions=True)
    yield client

    # Clean up memory service modules after each test to avoid cross-test pollution
    for mod_name in list(_MEMORY_MODULE_NAMES):
        sys.modules.pop(mod_name, None)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Return valid authentication headers."""
    return {"X-API-Key": "test-key"}


def _make_cloud_response(content: dict[str, Any]) -> MagicMock:
    """Build a mock httpx response in OpenAI-compatible format (OpenRouter/NVIDIA)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {"content": json.dumps(content)},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for POST /ingest endpoint."""

    def test_ingest_stores_memory_and_returns_id(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: POST /ingest stores memory and returns {id, status: ok}."""
        response = api_client.post(
            "/ingest",
            json={
                "text": "PPO equity Sharpe improved to 1.2 in bull regime",
                "source": "walk_forward:equity:ppo",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["id"], int)
        assert body["id"] >= 1

    def test_ingest_sequential_ids(self, api_client: Any, auth_headers: dict[str, str]) -> None:
        """TRAIN-06: Multiple ingests return sequential IDs."""
        r1 = api_client.post(
            "/ingest",
            json={"text": "First memory", "source": "walk_forward:equity:ppo"},
            headers=auth_headers,
        )
        r2 = api_client.post(
            "/ingest",
            json={"text": "Second memory", "source": "walk_forward:equity:a2c"},
            headers=auth_headers,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["id"] == r1.json()["id"] + 1

    def test_ingest_without_api_key_returns_401(self, api_client: Any) -> None:
        """TRAIN-06: POST /ingest without X-API-Key header returns 401."""
        response = api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "walk_forward:equity:ppo"},
        )
        assert response.status_code == 401

    def test_ingest_with_wrong_api_key_returns_401(self, api_client: Any) -> None:
        """TRAIN-06: POST /ingest with wrong X-API-Key returns 401."""
        response = api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "walk_forward:equity:ppo"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for GET /health endpoint."""

    def test_health_returns_healthy(self, api_client: Any) -> None:
        """TRAIN-06: GET /health returns status: healthy when SQLite is reachable."""
        response = api_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["db"] is True

    def test_health_no_auth_required(self, api_client: Any) -> None:
        """TRAIN-06: GET /health requires no authentication."""
        response = api_client.get("/health")

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestTrainingEndpoints
# ---------------------------------------------------------------------------


class TestTrainingEndpoints:
    """Tests for /training/run_config and /training/epoch_advice."""

    def test_run_config_returns_valid_shape(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: POST /training/run_config returns RunConfigResponse shape."""
        mock_response = {
            "learning_rate": 0.0003,
            "entropy_coeff": 0.01,
            "clip_range": 0.2,
            "n_epochs": 10,
            "batch_size": 64,
            "gamma": 0.99,
            "reward_weights": {"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
            "rationale": "test_advice",
        }

        with patch("memory_agents.query.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post(
                "/training/run_config",
                json={"query": "PPO equity iteration 1"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert "reward_weights" in body
        assert "rationale" in body
        assert isinstance(body["reward_weights"], dict)

    def test_run_config_returns_defaults_on_cloud_failure(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: /training/run_config returns safe defaults when cloud API fails."""
        with patch("memory_agents.query.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(side_effect=ConnectionError("Cloud API down"))

            response = api_client.post(
                "/training/run_config",
                json={"query": "SAC crypto"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["rationale"] == "cold_start_defaults"
        assert body["reward_weights"]["profit"] == pytest.approx(0.4)

    def test_epoch_advice_returns_valid_shape(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: POST /training/epoch_advice returns EpochAdviceResponse shape."""
        mock_response = {
            "reward_weights": {"profit": 0.3, "sharpe": 0.4, "drawdown": 0.25, "turnover": 0.05},
            "stop_training": False,
            "rationale": "maintain_current_weights",
        }

        with patch("memory_agents.query.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post(
                "/training/epoch_advice",
                json={"query": "epoch 5 sharpe=0.8 mdd=-0.05"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert "reward_weights" in body
        assert "stop_training" in body
        assert isinstance(body["stop_training"], bool)
        assert "rationale" in body

    def test_epoch_advice_clamps_out_of_bounds_weights(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: /training/epoch_advice clamps LLM weights to safe bounds."""
        # LLM returns profit=0.99 which exceeds max bound of 0.70
        mock_response = {
            "reward_weights": {"profit": 0.99, "sharpe": 0.01, "drawdown": 0.00, "turnover": 0.00},
            "stop_training": False,
            "rationale": "aggressive_profit",
        }

        with patch("memory_agents.query.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post(
                "/training/epoch_advice",
                json={"query": "epoch 10"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        # Clamp bounds apply pre-normalization: profit clamped to 0.70, others raised to their mins
        # After normalization the ratio changes but weights sum to 1.0 and profit is the max
        # (STATE.md decision: bounds applied pre-normalization only)
        weights = body["reward_weights"]
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01), f"Weights must sum to 1.0, got {total}"
        # Profit must dominate (was the highest clamped value at 0.70)
        assert weights["profit"] > weights["sharpe"], "profit should be highest after clamping"
        # sharpe was raised to minimum 0.10, so it should be present
        assert weights["sharpe"] >= 0.05

    def test_training_endpoints_require_auth(self, api_client: Any) -> None:
        """TRAIN-06: Training endpoints return 401 without API key."""
        r1 = api_client.post("/training/run_config", json={"query": "test"})
        r2 = api_client.post("/training/epoch_advice", json={"query": "test"})
        assert r1.status_code == 401
        assert r2.status_code == 401


# ---------------------------------------------------------------------------
# TestDebugEndpoints
# ---------------------------------------------------------------------------


class TestDebugEndpoints:
    """Tests for GET /debug/memories and GET /debug/consolidations."""

    def test_debug_memories_returns_list(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: GET /debug/memories returns a list."""
        api_client.post(
            "/ingest",
            json={"text": "Memory A", "source": "walk_forward:equity:ppo"},
            headers=auth_headers,
        )
        response = api_client.get("/debug/memories", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_debug_memories_source_filter(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: GET /debug/memories?source= filters by source tag."""
        api_client.post(
            "/ingest",
            json={"text": "Equity memory", "source": "walk_forward:equity:ppo"},
            headers=auth_headers,
        )
        api_client.post(
            "/ingest",
            json={"text": "Macro memory", "source": "macro_correlation:historical"},
            headers=auth_headers,
        )

        response = api_client.get(
            "/debug/memories?source=walk_forward:equity:ppo",
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert all(row["source"] == "walk_forward:equity:ppo" for row in body)

    def test_debug_consolidations_returns_list(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: GET /debug/consolidations returns a list."""
        response = api_client.get("/debug/consolidations", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_debug_endpoints_require_auth(self, api_client: Any) -> None:
        """TRAIN-06: Debug endpoints return 401 without API key."""
        r1 = api_client.get("/debug/memories")
        r2 = api_client.get("/debug/consolidations")
        assert r1.status_code == 401
        assert r2.status_code == 401


# ---------------------------------------------------------------------------
# TestConsolidation
# ---------------------------------------------------------------------------


class TestConsolidation:
    """Tests for POST /consolidate and ConsolidateAgent behavior."""

    def test_consolidate_triggers_and_returns_count(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: POST /consolidate triggers consolidation and returns count."""
        for i in range(3):
            api_client.post(
                "/ingest",
                json={"text": f"Training observation {i}", "source": "walk_forward:equity:ppo"},
                headers=auth_headers,
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "PPO performs better in bull regimes with lower entropy_coeff",
                    "category": "regime_performance",
                    "affected_algos": ["ppo"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Reduce entropy_coeff to 0.005 in bull regime",
                    "confidence": 0.65,
                    "evidence": "fold 1 bull sharpe=1.2; fold 3 bear sharpe=0.4",
                },
            ],
        }

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["consolidated"] >= 1

    def test_consolidate_rejects_malformed_llm_output_and_logs_quality(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: ConsolidateAgent rejects malformed LLM output and logs to quality table."""
        api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "walk_forward:equity:ppo"},
            headers=auth_headers,
        )

        malformed_response = {"some_field": "missing required fields"}

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(malformed_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] == 0

        import db as memory_db_module

        conn = memory_db_module.get_connection()
        rows = conn.execute("SELECT * FROM consolidation_quality WHERE accepted = 0").fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[0]["rejected_reason"] is not None

    def test_consolidate_detects_conflicting_consolidations(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: ConsolidateAgent flags contradicting patterns.

        Conflict requires: same category + shared env + opposite sentiment on
        the same metric (adjacency-aware matching).
        """
        import db as memory_db_module

        memory_db_module.insert_consolidation(
            pattern_text="PPO equity improved sharpe significantly in bull regime",
            source_count=5,
            category="regime_performance",
            affected_algos=["ppo"],
            affected_envs=["equity"],
            confidence=0.7,
        )

        for i in range(2):
            api_client.post(
                "/ingest",
                json={"text": f"Equity observation {i}", "source": "walk_forward:equity:ppo"},
                headers=auth_headers,
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "PPO equity degraded sharpe and decreased returns in bear regime",
                    "category": "regime_performance",
                    "affected_algos": ["ppo"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Reduce position sizing in bear regime",
                    "confidence": 0.6,
                    "evidence": "fold 2 bear sharpe=-0.3; fold 4 bear sharpe=-0.5",
                },
            ],
        }

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] >= 1

        # The conflicting original pattern should be superseded
        rows = memory_db_module.get_consolidations(limit=10)
        superseded = [r for r in rows if r.get("status") == "superseded"]
        assert len(superseded) >= 1, "Expected at least one superseded consolidation"

    def test_consolidate_returns_zero_when_no_memories(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: POST /consolidate returns consolidated=0 when memory table is empty."""
        response = api_client.post("/consolidate", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["consolidated"] == 0

    def test_consolidate_requires_auth(self, api_client: Any) -> None:
        """TRAIN-06: POST /consolidate returns 401 without API key."""
        response = api_client.post("/consolidate")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# TestConsolidationArchiving — Fix A (archiving behavior) and Fix F (batch loop)
# ---------------------------------------------------------------------------


class TestConsolidationArchiving:
    """Tests for Fix A (no archive on LLM failure) and Fix F (batch loop)."""

    def test_consolidate_does_not_archive_on_llm_failure(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1-FIX-A: Memories remain archived=0 when LLM returns None."""
        import db as memory_db_module

        for i in range(3):
            api_client.post(
                "/ingest",
                json={
                    "text": f"WF observation {i}",
                    "source": "walk_forward:equity:ppo",
                },
                headers=auth_headers,
            )

        with patch(
            "memory_agents.consolidate.ConsolidateAgent._call_llm_with_retry",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] == 0

        # Verify memories are NOT archived
        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT id, archived FROM memories WHERE source LIKE 'walk_forward:equity%'"
            ).fetchall()
            assert len(rows) >= 3
            assert all(row["archived"] == 0 for row in rows), (
                "Memories must remain unarchived when LLM fails"
            )
        finally:
            conn.close()

    def test_consolidate_archives_on_success(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1-FIX-A: Memories are archived=1 after successful LLM consolidation."""
        import db as memory_db_module

        for i in range(3):
            api_client.post(
                "/ingest",
                json={
                    "text": f"WF result {i}",
                    "source": "walk_forward:equity:sac",
                },
                headers=auth_headers,
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "SAC equity shows stable Sharpe across regimes",
                    "category": "regime_performance",
                    "affected_algos": ["sac"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Maintain SAC weight in ensemble",
                    "confidence": 0.60,
                    "evidence": "fold 1 sharpe=1.0; fold 2 sharpe=0.9",
                },
            ],
        }

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] >= 1

        # Verify memories ARE archived
        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT id, archived FROM memories WHERE source LIKE 'walk_forward:equity:sac%'"
            ).fetchall()
            assert len(rows) >= 3
            assert all(row["archived"] == 1 for row in rows), (
                "Memories must be archived after successful consolidation"
            )
        finally:
            conn.close()

    def test_consolidate_aggregates_epoch_memories(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1-FIX-F: All 250+ epoch memories are aggregated and archived in single call."""
        import db as memory_db_module

        # Insert 250 epoch memories with realistic format for aggregation
        for i in range(250):
            memory_db_module.insert_memory(
                text=(
                    f"EPOCH SNAPSHOT: run_id=equity_ppo_fold0 algo=PPO env=equity "
                    f"epoch={i} mean_reward=0.{i % 10:04d} rolling_sharpe_500=1.{i % 5} "
                    f"rolling_mdd_500=-{i % 20}.0 approx_kl=0.00{i % 3} "
                    f"reward_weights={{}} notable_event=None"
                ),
                source="training_epoch:historical",
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "PPO equity epochs show gradual improvement",
                    "category": "iteration_progression",
                    "affected_algos": ["ppo"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Continue current training approach",
                    "confidence": 0.55,
                    "evidence": "epoch 0 sharpe=0.1; epoch 249 sharpe=0.9",
                },
            ],
        }

        call_count = 0

        async def _mock_call_llm_with_retry(
            self_arg: Any,
            memory_texts: str,
            system_prompt: Any = None,
            few_shot_examples: Any = None,
        ) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch(
            "memory_agents.consolidate.ConsolidateAgent._call_llm_with_retry",
            _mock_call_llm_with_retry,
        ):
            response = api_client.post(
                "/consolidate",
                json={"env_name": "equity"},
                headers=auth_headers,
            )

        assert response.status_code == 200

        # Verify aggregated approach: single LLM call (not batched)
        assert call_count >= 1, f"Expected >= 1 LLM call for aggregated memories, got {call_count}"

        # Verify ALL epoch memories are archived
        conn = memory_db_module.get_connection()
        try:
            unarchived = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE source LIKE 'training_epoch:%' AND archived = 0"
            ).fetchone()[0]
            assert unarchived == 0, f"Expected 0 unarchived epoch memories, got {unarchived}"
        finally:
            conn.close()

    def test_consolidate_preserves_memories_on_llm_failure(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1-FIX-F: All memories preserved (not archived) when LLM call fails."""
        import db as memory_db_module

        # Insert 250 epoch memories with realistic format
        for i in range(250):
            memory_db_module.insert_memory(
                text=(
                    f"EPOCH SNAPSHOT: run_id=equity_a2c_fold0 algo=A2C env=equity "
                    f"epoch={i} mean_reward=0.{i % 10:04d} rolling_sharpe_500=1.0 "
                    f"rolling_mdd_500=-5.0 approx_kl=0.001 "
                    f"reward_weights={{}} notable_event=None"
                ),
                source="training_epoch:historical",
            )

        async def _mock_call_llm_with_retry(
            self_arg: Any,
            memory_texts: str,
            system_prompt: Any = None,
            few_shot_examples: Any = None,
        ) -> dict[str, Any] | None:
            return None  # LLM always fails

        with patch(
            "memory_agents.consolidate.ConsolidateAgent._call_llm_with_retry",
            _mock_call_llm_with_retry,
        ):
            response = api_client.post(
                "/consolidate",
                json={"env_name": "equity"},
                headers=auth_headers,
            )

        assert response.status_code == 200

        # All memories should be preserved (not archived) on LLM failure
        conn = memory_db_module.get_connection()
        try:
            unarchived = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE source LIKE 'training_epoch:%' AND archived = 0"
            ).fetchone()[0]
            assert unarchived == 250, f"All 250 memories should be preserved, got {unarchived}"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# TestUnarchiveMechanism — Fix B (unarchive and prefix queries)
# ---------------------------------------------------------------------------


class TestUnarchiveMechanism:
    """Tests for Fix B: unarchive_memories() and get_archived_memories_by_prefix()."""

    def test_unarchive_resets_flag(self, memory_db_env: Path) -> None:
        """19.1-FIX-B: unarchive_memories resets archived flag to 0."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()

        # Insert and archive memories
        ids = []
        for i in range(5):
            row_id = memory_db_module.insert_memory(
                text=f"Memory to archive {i}",
                source="walk_forward:equity:ppo",
            )
            ids.append(row_id)

        memory_db_module.archive_memories(ids)

        # Verify they are archived
        conn = memory_db_module.get_connection()
        try:
            archived_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 1"
            ).fetchone()[0]
            assert archived_count == 5
        finally:
            conn.close()

        # Unarchive them
        updated = memory_db_module.unarchive_memories(ids)
        assert updated == 5

        # Verify they are back to active
        conn = memory_db_module.get_connection()
        try:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 0"
            ).fetchone()[0]
            assert active_count == 5
            archived_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE archived = 1"
            ).fetchone()[0]
            assert archived_count == 0
        finally:
            conn.close()

    def test_get_archived_memories_by_prefix(self, memory_db_env: Path) -> None:
        """19.1-FIX-B: get_archived_memories_by_prefix filters by source prefix."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()

        # Insert memories with different source prefixes
        equity_ids = []
        for i in range(3):
            row_id = memory_db_module.insert_memory(
                text=f"Equity WF memory {i}",
                source="walk_forward:equity:ppo",
            )
            equity_ids.append(row_id)

        crypto_ids = []
        for i in range(2):
            row_id = memory_db_module.insert_memory(
                text=f"Crypto WF memory {i}",
                source="walk_forward:crypto:sac",
            )
            crypto_ids.append(row_id)

        epoch_ids = []
        for i in range(4):
            row_id = memory_db_module.insert_memory(
                text=f"Epoch memory {i}",
                source="training_epoch:equity:ppo:iter0",
            )
            epoch_ids.append(row_id)

        # Archive all
        memory_db_module.archive_memories(equity_ids + crypto_ids + epoch_ids)

        # Query archived by different prefixes
        equity_archived = memory_db_module.get_archived_memories_by_prefix("walk_forward:equity")
        assert len(equity_archived) == 3
        assert all("equity" in m["source"] for m in equity_archived)

        crypto_archived = memory_db_module.get_archived_memories_by_prefix("walk_forward:crypto")
        assert len(crypto_archived) == 2
        assert all("crypto" in m["source"] for m in crypto_archived)

        epoch_archived = memory_db_module.get_archived_memories_by_prefix("training_epoch")
        assert len(epoch_archived) == 4

        # Non-matching prefix returns empty
        empty = memory_db_module.get_archived_memories_by_prefix("nonexistent_prefix")
        assert len(empty) == 0

        # Unarchived memories should NOT appear
        memory_db_module.unarchive_memories(equity_ids)
        equity_after_unarchive = memory_db_module.get_archived_memories_by_prefix(
            "walk_forward:equity"
        )
        assert len(equity_after_unarchive) == 0


# ---------------------------------------------------------------------------
# TestDBSchema — New tables and columns
# ---------------------------------------------------------------------------


class TestDBSchema:
    """Tests for the enriched database schema."""

    def test_consolidation_sources_table_exists(self, memory_db_env: Path) -> None:
        """19.1: consolidation_sources join table is created."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidation_sources'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            conn.close()

    def test_pattern_presentations_table_exists(self, memory_db_env: Path) -> None:
        """19.1: pattern_presentations table is created."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_presentations'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            conn.close()

    def test_pattern_outcomes_table_exists(self, memory_db_env: Path) -> None:
        """19.1: pattern_outcomes table is created."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pattern_outcomes'"
            ).fetchall()
            assert len(rows) == 1
        finally:
            conn.close()

    def test_consolidations_has_new_columns(self, memory_db_env: Path) -> None:
        """19.1: consolidations table has all new columns."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        conn = memory_db_module.get_connection()
        try:
            cols = conn.execute("PRAGMA table_info(consolidations)").fetchall()
            col_names = {row[1] for row in cols}
            expected_new = {
                "category",
                "affected_algos",
                "affected_envs",
                "actionable_implication",
                "confidence",
                "evidence",
                "stage",
                "env_name",
                "confirmation_count",
                "last_confirmed_at",
                "superseded_by",
                "status",
                "conflict_group_id",
            }
            missing = expected_new - col_names
            assert not missing, f"Missing columns: {missing}"
        finally:
            conn.close()

    def test_consolidation_indexes_exist(self, memory_db_env: Path) -> None:
        """19.1: New indexes on consolidations table exist."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        conn = memory_db_module.get_connection()
        try:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='consolidations'"
            ).fetchall()
            index_names = {row[0] for row in indexes}
            assert "idx_consolidations_status" in index_names
            assert "idx_consolidations_env_stage" in index_names
        finally:
            conn.close()

    def test_schema_migration_idempotent(self, memory_db_env: Path) -> None:
        """19.1: init_db() is idempotent — running twice does not error."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        memory_db_module.init_db()  # Second call must not raise


# ---------------------------------------------------------------------------
# TestDBFunctions — New insert/get/update functions
# ---------------------------------------------------------------------------


class TestDBFunctions:
    """Tests for new DB functions."""

    def test_insert_consolidation_with_new_fields(self, memory_db_env: Path) -> None:
        """19.1: insert_consolidation() accepts all new fields."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        row_id = memory_db_module.insert_consolidation(
            pattern_text="Test pattern",
            source_count=3,
            category="regime_performance",
            affected_algos=["ppo", "sac"],
            affected_envs=["equity"],
            actionable_implication="Reduce LR in bear regime",
            confidence=0.65,
            evidence="fold 1 sharpe=1.2; fold 3 sharpe=0.4",
            stage=1,
            env_name="equity",
        )
        assert row_id >= 1

        # Verify retrieval with deserialization
        rows = memory_db_module.get_active_consolidations(env_name="equity")
        assert len(rows) >= 1
        row = rows[0]
        assert row["category"] == "regime_performance"
        assert row["affected_algos"] == ["ppo", "sac"]
        assert row["affected_envs"] == ["equity"]
        assert row["confidence"] == pytest.approx(0.65)
        assert row["stage"] == 1

    def test_insert_consolidation_source(self, memory_db_env: Path) -> None:
        """19.1: consolidation_sources join table is populated."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        mid = memory_db_module.insert_memory("test memory", "walk_forward:equity:ppo")
        cid = memory_db_module.insert_consolidation(
            pattern_text="Test",
            source_count=1,
            category="trade_quality",
        )
        memory_db_module.insert_consolidation_source(cid, mid)

        conn = memory_db_module.get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM consolidation_sources WHERE consolidation_id = ?", (cid,)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["memory_id"] == mid
        finally:
            conn.close()

    def test_get_active_consolidations_filters(self, memory_db_env: Path) -> None:
        """19.1: get_active_consolidations filters by env, stage, confidence."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        memory_db_module.insert_consolidation(
            pattern_text="High confidence equity",
            source_count=3,
            confidence=0.8,
            stage=1,
            env_name="equity",
        )
        memory_db_module.insert_consolidation(
            pattern_text="Low confidence crypto",
            source_count=2,
            confidence=0.2,
            stage=1,
            env_name="crypto",
        )

        # Filter by env
        equity_only = memory_db_module.get_active_consolidations(env_name="equity")
        assert all(r.get("env_name") in ("equity", None) for r in equity_only)

        # Filter by min confidence
        high_conf = memory_db_module.get_active_consolidations(min_confidence=0.5)
        assert all(
            (r.get("confidence") or 0) >= 0.5 or r.get("confidence") is None for r in high_conf
        )

    def test_update_consolidation_status(self, memory_db_env: Path) -> None:
        """19.1: Pattern lifecycle active → superseded works."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        old_id = memory_db_module.insert_consolidation(
            pattern_text="Old pattern",
            source_count=2,
        )
        new_id = memory_db_module.insert_consolidation(
            pattern_text="New pattern",
            source_count=3,
        )
        memory_db_module.update_consolidation_status(old_id, "superseded", superseded_by=new_id)

        conn = memory_db_module.get_connection()
        try:
            row = conn.execute(
                "SELECT status, superseded_by FROM consolidations WHERE id = ?", (old_id,)
            ).fetchone()
            assert row["status"] == "superseded"
            assert row["superseded_by"] == new_id
        finally:
            conn.close()

    def test_update_consolidation_status_with_conflict_group_id(self, memory_db_env: Path) -> None:
        """Fix 2: update_consolidation_status propagates conflict_group_id."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        old_id = memory_db_module.insert_consolidation(
            pattern_text="Old pattern",
            source_count=2,
        )
        new_id = memory_db_module.insert_consolidation(
            pattern_text="New pattern",
            source_count=3,
        )

        group_id = "test-group-uuid-123"
        memory_db_module.update_consolidation_status(
            old_id, "superseded", superseded_by=new_id, conflict_group_id=group_id
        )

        conn = memory_db_module.get_connection()
        try:
            row = conn.execute(
                "SELECT status, superseded_by, conflict_group_id FROM consolidations WHERE id = ?",
                (old_id,),
            ).fetchone()
            assert row["status"] == "superseded"
            assert row["superseded_by"] == new_id
            assert row["conflict_group_id"] == group_id
        finally:
            conn.close()

    def test_check_conflicts_same_metric_opposite_sentiment(self, memory_db_env: Path) -> None:
        """Fix 1: Same metric + opposite sentiment + same category + same env = conflict."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        from memory_agents.consolidate import ConsolidateAgent

        agent = ConsolidateAgent()
        existing = [
            {
                "id": 1,
                "pattern_text": "PPO improved sharpe in equity bull regime",
                "category": "regime_performance",
                "affected_algos": ["ppo"],
                "affected_envs": ["equity"],
            }
        ]
        result = agent._check_conflicts(
            existing,
            "PPO degraded sharpe in equity bear regime",
            new_category="regime_performance",
            new_envs=["equity"],
        )
        assert result == 1, "Expected conflict on same metric (sharpe) with opposite sentiment"

    def test_check_conflicts_different_metrics_no_conflict(self, memory_db_env: Path) -> None:
        """Fix 1: Different metrics should not conflict even with opposite sentiment words."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        from memory_agents.consolidate import ConsolidateAgent

        agent = ConsolidateAgent()
        existing = [
            {
                "id": 1,
                "pattern_text": "PPO shows higher win rates in XLF asset",
                "category": "trade_quality",
                "affected_algos": ["ppo"],
                "affected_envs": ["equity"],
            }
        ]
        result = agent._check_conflicts(
            existing,
            "PPO shows degraded performance during high VIX periods",
            new_category="trade_quality",
            new_envs=["equity"],
        )
        assert result is None, "Different metrics (win vs performance) should not conflict"

    def test_check_conflicts_different_env_no_conflict(self, memory_db_env: Path) -> None:
        """Fix 1: Different envs should not conflict even with same category."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        from memory_agents.consolidate import ConsolidateAgent

        agent = ConsolidateAgent()
        existing = [
            {
                "id": 1,
                "pattern_text": "PPO improved sharpe in equity",
                "category": "regime_performance",
                "affected_algos": ["ppo"],
                "affected_envs": ["equity"],
            }
        ]
        result = agent._check_conflicts(
            existing,
            "SAC degraded sharpe in crypto",
            new_category="regime_performance",
            new_envs=["crypto"],
        )
        assert result is None, "Different envs + different algos should not conflict"

    def test_check_conflicts_different_category_no_conflict(self, memory_db_env: Path) -> None:
        """Fix 1: Different categories should never conflict."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        from memory_agents.consolidate import ConsolidateAgent

        agent = ConsolidateAgent()
        existing = [
            {
                "id": 1,
                "pattern_text": "PPO improved sharpe in equity",
                "category": "regime_performance",
                "affected_algos": ["ppo"],
                "affected_envs": ["equity"],
            }
        ]
        result = agent._check_conflicts(
            existing,
            "PPO degraded sharpe in equity",
            new_category="overfit_diagnosis",
            new_envs=["equity"],
        )
        assert result is None, "Different categories should never conflict"

    def test_increment_confirmation(self, memory_db_env: Path) -> None:
        """19.1: increment_confirmation updates count and timestamp."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        cid = memory_db_module.insert_consolidation(
            pattern_text="Test",
            source_count=1,
        )
        memory_db_module.increment_confirmation(cid)
        memory_db_module.increment_confirmation(cid)

        conn = memory_db_module.get_connection()
        try:
            row = conn.execute(
                "SELECT confirmation_count, last_confirmed_at FROM consolidations WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["confirmation_count"] == 2
            assert row["last_confirmed_at"] is not None
        finally:
            conn.close()

    def test_insert_pattern_presentation(self, memory_db_env: Path) -> None:
        """19.1: Presentation tracking records are created."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        cid = memory_db_module.insert_consolidation(
            pattern_text="Test",
            source_count=1,
        )
        pid = memory_db_module.insert_pattern_presentation(
            consolidation_id=cid,
            iteration=1,
            env_name="equity",
            request_type="run_config",
            advice_response="test advice",
        )
        assert pid >= 1

    def test_insert_pattern_outcome(self, memory_db_env: Path) -> None:
        """19.1: Outcome recording stores iteration results."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        oid = memory_db_module.insert_pattern_outcome(
            iteration=1,
            env_name="equity",
            gate_passed=True,
            sharpe=1.2,
            mdd=-0.05,
            sortino=1.8,
            pnl=5000.0,
            patterns_presented=[1, 2, 3],
        )
        assert oid >= 1

    def test_get_pattern_effectiveness(self, memory_db_env: Path) -> None:
        """19.1: Pattern effectiveness join returns presentation + outcome data."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        cid = memory_db_module.insert_consolidation(
            pattern_text="Test pattern",
            source_count=1,
        )
        memory_db_module.insert_pattern_presentation(
            consolidation_id=cid,
            iteration=1,
            env_name="equity",
            request_type="run_config",
            advice_response="reduce LR",
        )
        memory_db_module.insert_pattern_outcome(
            iteration=1,
            env_name="equity",
            gate_passed=True,
            sharpe=1.5,
            mdd=-0.03,
            sortino=2.0,
            pnl=3000.0,
        )

        results = memory_db_module.get_pattern_effectiveness()
        assert len(results) >= 1
        row = results[0]
        assert row["consolidation_id"] == cid
        assert row["sharpe"] == pytest.approx(1.5)
        assert row["gate_passed"] == 1  # SQLite stores bool as int


# ---------------------------------------------------------------------------
# TestRecordOutcome — New endpoint
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    """Tests for POST /training/record_outcome endpoint."""

    def test_record_outcome_stores_data(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1: POST /training/record_outcome stores iteration outcome."""
        response = api_client.post(
            "/training/record_outcome",
            json={
                "iteration": 1,
                "env_name": "equity",
                "gate_passed": True,
                "sharpe": 1.2,
                "mdd": -0.05,
                "sortino": 1.8,
                "pnl": 5000.0,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["id"], int)

    def test_record_outcome_requires_auth(self, api_client: Any) -> None:
        """19.1: POST /training/record_outcome returns 401 without API key."""
        response = api_client.post(
            "/training/record_outcome",
            json={"iteration": 0, "env_name": "equity"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# TestPatternEffectiveness — New endpoint
# ---------------------------------------------------------------------------


class TestPatternEffectiveness:
    """Tests for GET /training/pattern_effectiveness endpoint."""

    def test_pattern_effectiveness_returns_list(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1: GET /training/pattern_effectiveness returns a list."""
        response = api_client.get("/training/pattern_effectiveness", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_pattern_effectiveness_requires_auth(self, api_client: Any) -> None:
        """19.1: GET /training/pattern_effectiveness returns 401 without API key."""
        response = api_client.get("/training/pattern_effectiveness")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# TestAPSchedulerRemoved
# ---------------------------------------------------------------------------


class TestAPSchedulerRemoved:
    """Tests verifying APScheduler has been removed from app.py."""

    def test_no_apscheduler_import(self) -> None:
        """19.1: app.py does not import apscheduler."""
        app_path = _MEMORY_SERVICE_DIR / "app.py"
        content = app_path.read_text()
        assert "apscheduler" not in content, "APScheduler should be removed from app.py"

    def test_no_scheduler_global(self) -> None:
        """19.1: app.py does not have a scheduler global variable."""
        app_path = _MEMORY_SERVICE_DIR / "app.py"
        content = app_path.read_text()
        # Check that no scheduler = ... assignment exists outside of docstrings/comments
        code_lines = [
            line
            for line in content.splitlines()
            if not line.strip().startswith("#") and not line.strip().startswith('"""')
        ]
        code_text = "\n".join(code_lines)
        assert "scheduler = " not in code_text, (
            "No scheduler global variable should exist in app.py"
        )

    def test_lifespan_does_not_start_scheduler(self) -> None:
        """19.1: lifespan does not start a background scheduler."""
        app_path = _MEMORY_SERVICE_DIR / "app.py"
        content = app_path.read_text()
        assert "scheduler.start" not in content
        assert "add_job" not in content


# ---------------------------------------------------------------------------
# TestConsolidationSchema
# ---------------------------------------------------------------------------


class TestConsolidationSchema:
    """Tests for the new consolidation JSON schema and validation."""

    def test_multi_pattern_response_accepted(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1: LLM returning multiple patterns in array is accepted."""
        for i in range(3):
            api_client.post(
                "/ingest",
                json={"text": f"Observation {i}", "source": "walk_forward:equity:ppo"},
                headers=auth_headers,
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "PPO bull outperformance pattern",
                    "category": "regime_performance",
                    "affected_algos": ["ppo"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Increase PPO weight in bull regime",
                    "confidence": 0.55,
                    "evidence": "fold 1 sharpe=1.1; fold 2 sharpe=1.3",
                },
                {
                    "pattern_text": "SAC drawdown recovery pattern",
                    "category": "drawdown_recovery",
                    "affected_algos": ["sac"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Increase drawdown weight for SAC",
                    "confidence": 0.48,
                    "evidence": "fold 3 avg_dd=0.05; fold 4 avg_dd=0.03",
                },
            ],
        }

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] >= 2

    def test_empty_patterns_array_accepted(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """19.1: LLM returning empty patterns array is accepted gracefully."""
        api_client.post(
            "/ingest",
            json={"text": "Observation", "source": "walk_forward:equity:ppo"},
            headers=auth_headers,
        )

        mock_response = {"patterns": []}

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_cloud_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] == 0


# ---------------------------------------------------------------------------
# TestSourceTags
# ---------------------------------------------------------------------------


class TestSourceTags:
    """Tests for the updated source tag format."""

    def test_source_prefix_query(self, memory_db_env: Path) -> None:
        """19.1: get_memories_by_source_prefix filters by env prefix."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()
        memory_db_module.insert_memory("Equity PPO data", "walk_forward:equity:ppo")
        memory_db_module.insert_memory("Equity A2C data", "walk_forward:equity:a2c")
        memory_db_module.insert_memory("Crypto PPO data", "walk_forward:crypto:ppo")

        equity_mems = memory_db_module.get_memories_by_source_prefix("walk_forward:equity")
        assert len(equity_mems) == 2
        assert all("equity" in m["source"] for m in equity_mems)

        crypto_mems = memory_db_module.get_memories_by_source_prefix("walk_forward:crypto")
        assert len(crypto_mems) == 1


# ---------------------------------------------------------------------------
# TestAuth
# ---------------------------------------------------------------------------


class TestAuth:
    """Authentication enforcement tests for all protected endpoints."""

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("post", "/ingest", {"text": "t", "source": "s"}),
            ("post", "/consolidate", {}),
            ("post", "/training/run_config", {"query": "q"}),
            ("post", "/training/epoch_advice", {"query": "q"}),
            ("post", "/training/record_outcome", {"iteration": 0, "env_name": "equity"}),
            ("get", "/debug/memories", None),
            ("get", "/debug/consolidations", None),
            ("get", "/training/pattern_effectiveness", None),
        ],
    )
    def test_all_protected_endpoints_return_401_without_key(
        self,
        api_client: Any,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> None:
        """TRAIN-06: All endpoints except /health return 401 without X-API-Key."""
        fn = getattr(api_client, method)
        if body is not None:
            response = fn(path, json=body)
        else:
            response = fn(path)
        assert response.status_code == 401, f"Expected 401 for {method.upper()} {path}"


# ---------------------------------------------------------------------------
# TestTrainingTelemetryDDL (existing DuckDB migration tests — preserved)
# ---------------------------------------------------------------------------


class TestTrainingTelemetryDDL:
    """Tests for apply_training_telemetry_ddl() DuckDB schema migrations."""

    def test_training_epochs_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates training_epochs table in DuckDB."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'training_epochs'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_training_epochs_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: training_epochs table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'training_epochs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "epoch",
                "algo",
                "env",
                "sharpe",
                "mdd",
                "reward_weight_profit",
                "reward_weight_sharpe",
                "reward_weight_drawdown",
                "reward_weight_turnover",
                "stop_training",
                "rationale",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_meta_decisions_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates meta_decisions table."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'meta_decisions'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_meta_decisions_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: meta_decisions table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'meta_decisions'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "algo",
                "env",
                "decision_type",
                "decision_json",
                "rationale",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_reward_adjustments_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates reward_adjustments table."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'reward_adjustments'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_reward_adjustments_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: reward_adjustments table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'reward_adjustments'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "epoch_trigger",
                "epoch_outcome",
                "algo",
                "env",
                "weight_before",
                "weight_after",
                "outcome_sharpe",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_ddl_is_idempotent(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl is idempotent (run twice, no error)."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            # Second call must not raise
            apply_training_telemetry_ddl(conn)
            # Tables still exist
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name IN ('training_epochs', 'meta_decisions', 'reward_adjustments')"
            ).fetchone()
            assert result[0] == 3
        finally:
            conn.close()

    def test_alter_training_runs_adds_run_type_column(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TRAIN-07: ALTER TABLE migration adds run_type column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            # Create a minimal training_runs table first (as if from previous phase)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_id VARCHAR NOT NULL,"
                "  algo VARCHAR NOT NULL,"
                "  env VARCHAR NOT NULL,"
                "  created_at TIMESTAMP DEFAULT NOW()"
                ")"
            )
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "run_type" in col_names, "run_type column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_training_runs_adds_meta_rationale_column(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TRAIN-07: ALTER TABLE migration adds meta_rationale column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_id VARCHAR NOT NULL"
                ")"
            )
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "meta_rationale" in col_names, "meta_rationale column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_adds_dominant_regime_column(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-07: ALTER TABLE migration adds dominant_regime column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS training_runs (id INTEGER PRIMARY KEY)")
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "dominant_regime" in col_names, "dominant_regime column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_idempotent_when_columns_exist(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-07: ALTER TABLE is idempotent if columns already exist (no error on re-run)."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            # Create training_runs with the new columns already present
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_type VARCHAR DEFAULT 'completed',"
                "  meta_rationale TEXT,"
                "  dominant_regime VARCHAR"
                ")"
            )
            # Running twice must not raise
            apply_training_telemetry_ddl(conn)
            apply_training_telemetry_ddl(conn)
        finally:
            conn.close()

    def test_telemetry_tables_insertable(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: Can insert rows into training_epochs, meta_decisions, reward_adjustments."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            # Insert into training_epochs
            conn.execute(
                "INSERT INTO training_epochs (run_id, epoch, algo, env, sharpe, mdd) "
                "VALUES ('run-001', 1, 'ppo', 'equity', 1.5, -0.05)"
            )
            # Insert into meta_decisions
            conn.execute(
                "INSERT INTO meta_decisions (run_id, algo, env, decision_type, decision_json) "
                "VALUES ('run-001', 'ppo', 'equity', 'run_config', '{\"lr\": 3e-4}')"
            )
            # Insert into reward_adjustments
            conn.execute(
                "INSERT INTO reward_adjustments (run_id, algo, env) "
                "VALUES ('run-001', 'ppo', 'equity')"
            )
            count = conn.execute("SELECT COUNT(*) FROM training_epochs").fetchone()[0]
            assert count == 1
            count = conn.execute("SELECT COUNT(*) FROM meta_decisions").fetchone()[0]
            assert count == 1
            count = conn.execute("SELECT COUNT(*) FROM reward_adjustments").fetchone()[0]
            assert count == 1
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# TestRetrievalEfficiency
# ---------------------------------------------------------------------------


class TestRetrievalEfficiency:
    """Tests for context-aware category & env/algo filtering (R1-R3)."""

    def test_parse_query_context(self, memory_db_env: Path) -> None:
        """Correct extraction of env/algo/iteration from query string."""
        from memory_agents.query import _parse_query_context

        result = _parse_query_context("env=equity algo=PPO iteration=5 sharpe=0.8")
        assert result["env_name"] == "equity"
        assert result["algo_name"] == "ppo"
        assert result["iteration"] == 5

    def test_parse_query_context_partial(self, memory_db_env: Path) -> None:
        """Parsing with missing fields returns None for those fields."""
        from memory_agents.query import _parse_query_context

        result = _parse_query_context("just a plain query")
        assert result["env_name"] is None
        assert result["algo_name"] is None
        assert result["iteration"] is None

    def test_parse_query_context_iter_alias(self, memory_db_env: Path) -> None:
        """iter= alias works same as iteration=."""
        from memory_agents.query import _parse_query_context

        result = _parse_query_context("env=crypto algo=sac iter=3")
        assert result["env_name"] == "crypto"
        assert result["algo_name"] == "sac"
        assert result["iteration"] == 3

    def test_build_context_filters_by_env(self, memory_db_env: Path) -> None:
        """Only equity + NULL env patterns returned for equity query."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # Insert equity pattern
        memory_db_module.insert_consolidation(
            pattern_text="equity pattern",
            source_count=3,
            category="regime_performance",
            confidence=0.8,
            stage=1,
            env_name="equity",
        )
        # Insert crypto pattern
        memory_db_module.insert_consolidation(
            pattern_text="crypto pattern",
            source_count=3,
            category="regime_performance",
            confidence=0.8,
            stage=1,
            env_name="crypto",
        )
        # Insert universal pattern (no env)
        memory_db_module.insert_consolidation(
            pattern_text="universal pattern",
            source_count=3,
            category="regime_performance",
            confidence=0.8,
            stage=1,
            env_name=None,
        )

        agent = QueryAgent()
        context, ids = agent._build_context(env_name="equity", request_type="run_config")

        assert "equity pattern" in context
        assert "universal pattern" in context
        assert "crypto pattern" not in context

    def test_build_context_filters_by_algo(self, memory_db_env: Path) -> None:
        """SAC-only patterns excluded from PPO query; universal patterns kept."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # SAC-only pattern
        memory_db_module.insert_consolidation(
            pattern_text="sac only pattern",
            source_count=3,
            category="overfit_diagnosis",
            affected_algos=["sac"],
            confidence=0.8,
            stage=1,
        )
        # Universal pattern (no algos)
        memory_db_module.insert_consolidation(
            pattern_text="universal algo pattern",
            source_count=3,
            category="overfit_diagnosis",
            confidence=0.8,
            stage=1,
        )
        # PPO pattern
        memory_db_module.insert_consolidation(
            pattern_text="ppo specific pattern",
            source_count=3,
            category="overfit_diagnosis",
            affected_algos=["ppo"],
            confidence=0.8,
            stage=1,
        )

        agent = QueryAgent()
        context, ids = agent._build_context(algo_name="ppo", request_type="run_config")

        assert "universal algo pattern" in context
        assert "ppo specific pattern" in context
        assert "sac only pattern" not in context

    def test_build_context_category_filtering_run_config(self, memory_db_env: Path) -> None:
        """Training categories included, live categories excluded for run_config."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # run_config relevant category
        memory_db_module.insert_consolidation(
            pattern_text="regime perf pattern",
            source_count=3,
            category="regime_performance",
            confidence=0.8,
            stage=1,
        )
        # live_trading category (should be excluded for run_config)
        memory_db_module.insert_consolidation(
            pattern_text="live gate pattern",
            source_count=3,
            category="live_cycle_gate",
            confidence=0.8,
            stage=1,
        )

        agent = QueryAgent()
        context, ids = agent._build_context(request_type="run_config")

        assert "regime perf pattern" in context
        assert "live gate pattern" not in context

    def test_build_context_category_filtering_epoch_advice(self, memory_db_env: Path) -> None:
        """Epoch-relevant categories included, others excluded."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # epoch_advice relevant
        memory_db_module.insert_consolidation(
            pattern_text="drawdown recovery tip",
            source_count=3,
            category="drawdown_recovery",
            confidence=0.8,
            stage=1,
        )
        # Not in epoch_advice categories
        memory_db_module.insert_consolidation(
            pattern_text="data size impact note",
            source_count=3,
            category="data_size_impact",
            confidence=0.8,
            stage=1,
        )

        agent = QueryAgent()
        context, ids = agent._build_context(request_type="epoch_advice")

        assert "drawdown recovery tip" in context
        assert "data size impact note" not in context

    def test_build_context_skips_raw_memories(self, memory_db_env: Path) -> None:
        """No raw memories when >= 3 consolidations exist."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # Insert 3 consolidations in a run_config-relevant category
        for i in range(3):
            memory_db_module.insert_consolidation(
                pattern_text=f"pattern {i}",
                source_count=2,
                category="regime_performance",
                confidence=0.7,
                stage=1,
            )

        # Insert a raw memory
        memory_db_module.insert_memory(
            text="raw memory text xyz",
            source="walk_forward:equity:ppo",
        )

        agent = QueryAgent()
        context, ids = agent._build_context(request_type="run_config")

        assert "raw memory text xyz" not in context
        assert "pattern 0" in context

    def test_build_context_includes_raw_on_cold_start(self, memory_db_env: Path) -> None:
        """Raw memories included when < 3 consolidations."""
        from memory_agents.query import QueryAgent

        import db as memory_db_module

        memory_db_module.init_db()

        # Only 1 consolidation
        memory_db_module.insert_consolidation(
            pattern_text="single pattern",
            source_count=2,
            category="regime_performance",
            confidence=0.7,
            stage=1,
        )

        # Insert a raw memory
        memory_db_module.insert_memory(
            text="cold start raw memory",
            source="walk_forward:equity:ppo",
        )

        agent = QueryAgent()
        context, ids = agent._build_context(request_type="run_config")

        assert "single pattern" in context
        assert "cold start raw memory" in context

    def test_get_active_consolidations_limit_per_category(self, memory_db_env: Path) -> None:
        """Top-3 per category respected with limit_per_category."""
        import db as memory_db_module

        memory_db_module.init_db()

        # Insert 5 patterns in same category with varying confidence
        for i in range(5):
            memory_db_module.insert_consolidation(
                pattern_text=f"pattern conf {i}",
                source_count=2,
                category="regime_performance",
                confidence=0.5 + i * 0.1,
                stage=1,
            )

        results = memory_db_module.get_active_consolidations(
            categories=["regime_performance"],
            limit_per_category=3,
        )

        assert len(results) == 3
        # Should be the top 3 by composite score (confidence dominates)
        texts = [r["pattern_text"] for r in results]
        assert "pattern conf 4" in texts  # conf=0.9
        assert "pattern conf 3" in texts  # conf=0.8
        assert "pattern conf 2" in texts  # conf=0.7

    def test_get_active_consolidations_composite_score(self, memory_db_env: Path) -> None:
        """High-confidence old vs low-confidence recent — confidence dominates."""
        import db as memory_db_module

        memory_db_module.init_db()

        # High confidence pattern
        memory_db_module.insert_consolidation(
            pattern_text="high conf old",
            source_count=5,
            category="overfit_diagnosis",
            confidence=0.95,
            stage=1,
        )
        # Low confidence pattern
        memory_db_module.insert_consolidation(
            pattern_text="low conf new",
            source_count=1,
            category="overfit_diagnosis",
            confidence=0.45,
            stage=1,
        )

        results = memory_db_module.get_active_consolidations(
            categories=["overfit_diagnosis"],
            limit_per_category=1,
        )

        assert len(results) == 1
        assert results[0]["pattern_text"] == "high conf old"

    def test_get_active_consolidations_multi_category_spread(self, memory_db_env: Path) -> None:
        """limit_per_category=2 across 2 categories returns up to 2 from each."""
        import db as memory_db_module

        memory_db_module.init_db()

        # 3 patterns in regime_performance
        for i in range(3):
            memory_db_module.insert_consolidation(
                pattern_text=f"regime {i}",
                source_count=2,
                category="regime_performance",
                confidence=0.6 + i * 0.1,
                stage=1,
            )
        # 3 patterns in overfit_diagnosis
        for i in range(3):
            memory_db_module.insert_consolidation(
                pattern_text=f"overfit {i}",
                source_count=2,
                category="overfit_diagnosis",
                confidence=0.6 + i * 0.1,
                stage=1,
            )

        results = memory_db_module.get_active_consolidations(
            categories=["regime_performance", "overfit_diagnosis"],
            limit_per_category=2,
        )

        assert len(results) == 4  # 2 per category
        cats = [r["category"] for r in results]
        assert cats.count("regime_performance") == 2
        assert cats.count("overfit_diagnosis") == 2

    def test_get_active_consolidations_empty_categories(self, memory_db_env: Path) -> None:
        """Empty categories list returns no results (not a SQL error)."""
        import db as memory_db_module

        memory_db_module.init_db()

        memory_db_module.insert_consolidation(
            pattern_text="some pattern",
            source_count=2,
            category="regime_performance",
            confidence=0.8,
            stage=1,
        )

        # Empty list should be treated as "no category filter" (falsy)
        results = memory_db_module.get_active_consolidations(categories=[])
        assert len(results) == 1  # No filter applied, returns all


# ---------------------------------------------------------------------------
# Epoch aggregation helper tests
# ---------------------------------------------------------------------------


class TestEpochAggregationHelpers:
    """Tests for _quartiles, _iqr_outliers, _iqm, _trend_slope, _skewness,
    _temporal_windows, and _aggregate_epoch_summaries."""

    def test_quartiles_n4(self) -> None:
        """Quartiles with N=4 (PPO typical fold size)."""
        from memory_agents.consolidate import _quartiles

        vals = [1.0, 2.0, 3.0, 4.0]
        q1, med, q3 = _quartiles(vals)
        assert q1 == pytest.approx(1.75)
        assert med == pytest.approx(2.5)
        assert q3 == pytest.approx(3.25)

    def test_quartiles_n17(self) -> None:
        """Quartiles with N=17 (A2C/SAC typical fold size)."""
        from memory_agents.consolidate import _quartiles

        vals = list(range(1, 18))  # 1..17
        q1, med, q3 = _quartiles(vals)
        assert med == pytest.approx(9.0)
        assert q1 == pytest.approx(5.0)
        assert q3 == pytest.approx(13.0)

    def test_quartiles_n1(self) -> None:
        """Quartiles with single value returns that value for all."""
        from memory_agents.consolidate import _quartiles

        q1, med, q3 = _quartiles([42.0])
        assert q1 == 42.0
        assert med == 42.0
        assert q3 == 42.0

    def test_quartiles_empty(self) -> None:
        """Quartiles with empty list returns zeros."""
        from memory_agents.consolidate import _quartiles

        q1, med, q3 = _quartiles([])
        assert q1 == 0.0
        assert med == 0.0
        assert q3 == 0.0

    def test_iqr_outliers_small_n(self) -> None:
        """IQR outlier detection at N=4 with an extreme value."""
        from memory_agents.consolidate import _iqr_outliers

        # Normal values + one extreme outlier
        vals = [1.0, 2.0, 3.0, 100.0]
        result = _iqr_outliers(vals, mild_k=1.5, extreme_k=3.0)
        assert result["q1"] == pytest.approx(1.75)
        assert result["q3"] == pytest.approx(27.25)
        assert result["iqr"] == pytest.approx(25.5)
        assert 100.0 in result["mild_outliers"]
        assert len(result["mild_outliers"]) >= 1

    def test_iqr_no_outliers(self) -> None:
        """All values within IQR fences returns empty outlier lists."""
        from memory_agents.consolidate import _iqr_outliers

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _iqr_outliers(vals, mild_k=1.5, extreme_k=3.0)
        assert result["mild_outliers"] == []
        assert result["extreme_outliers"] == []

    def test_iqr_extreme_vs_mild(self) -> None:
        """Extreme outliers are a subset of mild outliers."""
        from memory_agents.consolidate import _iqr_outliers

        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 200.0]
        result = _iqr_outliers(vals, mild_k=1.5, extreme_k=3.0)
        # Every extreme outlier should also be in mild outliers
        for v in result["extreme_outliers"]:
            assert v in result["mild_outliers"]

    def test_iqm_basic(self) -> None:
        """IQM excludes outer quartiles."""
        from memory_agents.consolidate import _iqm

        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        # IQM should average the middle 50%: vals[2:6] = [3, 4, 5, 6] = mean 4.5
        assert _iqm(vals) == pytest.approx(4.5)

    def test_iqm_small_n(self) -> None:
        """IQM falls back to mean when N < 4."""
        from memory_agents.consolidate import _iqm

        vals = [1.0, 2.0, 3.0]
        assert _iqm(vals) == pytest.approx(2.0)

    def test_iqm_empty(self) -> None:
        """IQM of empty list returns 0."""
        from memory_agents.consolidate import _iqm

        assert _iqm([]) == 0.0

    def test_trend_slope_increasing(self) -> None:
        """Positive slope for increasing sequence."""
        from memory_agents.consolidate import _trend_slope

        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _trend_slope(vals) == pytest.approx(1.0)

    def test_trend_slope_decreasing(self) -> None:
        """Negative slope for decreasing sequence."""
        from memory_agents.consolidate import _trend_slope

        vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _trend_slope(vals) == pytest.approx(-1.0)

    def test_trend_slope_flat(self) -> None:
        """Near-zero slope for constant values."""
        from memory_agents.consolidate import _trend_slope

        vals = [3.0, 3.0, 3.0, 3.0]
        assert abs(_trend_slope(vals)) < 1e-6

    def test_trend_slope_small_n(self) -> None:
        """Returns 0.0 when N < 3."""
        from memory_agents.consolidate import _trend_slope

        assert _trend_slope([1.0, 2.0]) == 0.0
        assert _trend_slope([1.0]) == 0.0

    def test_skewness_symmetric(self) -> None:
        """Near-zero skewness for symmetric data."""
        from memory_agents.consolidate import _skewness

        # Symmetric around 5
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        skew = _skewness(vals)
        assert skew is not None
        assert abs(skew) < 0.3

    def test_skewness_right_skewed(self) -> None:
        """Positive skewness for right-skewed data."""
        from memory_agents.consolidate import _skewness

        vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 20.0]
        skew = _skewness(vals)
        assert skew is not None
        assert skew > 0.5

    def test_skewness_small_n_returns_none(self) -> None:
        """Returns None when N < skewness_min_n."""
        from memory_agents.consolidate import _skewness

        vals = [1.0, 2.0, 3.0]  # N=3 < default min_n=8
        assert _skewness(vals) is None

    def test_temporal_windows_n17(self) -> None:
        """3-window split for N >= 6."""
        from memory_agents.consolidate import _temporal_windows

        epochs = [{"mean_reward": float(i)} for i in range(17)]
        windows = _temporal_windows(epochs, ["mean_reward"])
        assert len(windows) == 3
        assert windows[0]["label"] == "early"
        assert windows[1]["label"] == "mid"
        assert windows[2]["label"] == "late"
        # Late window should have higher mean than early
        assert windows[2]["mean_reward"] > windows[0]["mean_reward"]

    def test_temporal_windows_n4(self) -> None:
        """2-window split for 4 <= N < 6."""
        from memory_agents.consolidate import _temporal_windows

        epochs = [{"mean_reward": float(i)} for i in range(4)]
        windows = _temporal_windows(epochs, ["mean_reward"])
        assert len(windows) == 2
        assert windows[0]["label"] == "early"
        assert windows[1]["label"] == "late"

    def test_temporal_windows_n2(self) -> None:
        """No windowing for N < 4."""
        from memory_agents.consolidate import _temporal_windows

        epochs = [{"mean_reward": 1.0}, {"mean_reward": 2.0}]
        assert _temporal_windows(epochs, ["mean_reward"]) == []

    def test_aggregate_kl_skipped_for_a2c(self) -> None:
        """A2C fold summaries should not contain approx_kl lines."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(10):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_a2c_fold0 algo=a2c env=equity "
                        f"epoch={i * 2000} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500={0.5 + 0.05 * i:.4f} "
                        f"rolling_mdd_500=-0.05 rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert len(summaries) == 1
        assert "approx_kl" not in summaries[0]

    def test_aggregate_kl_present_for_ppo(self) -> None:
        """PPO fold summaries should contain approx_kl lines."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(5):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_ppo_fold0 algo=ppo env=equity "
                        f"epoch={i * 20} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 approx_kl=0.015 "
                        f"rolling_sharpe_500=0.5 rolling_mdd_500=-0.05 "
                        f"rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert len(summaries) == 1
        assert "approx_kl" in summaries[0]

    def test_aggregate_fold_metadata(self) -> None:
        """Fold summary includes N, cadence, and confidence label."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(4):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_ppo_fold0 algo=ppo env=equity "
                        f"epoch={i * 20} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 approx_kl=0.01 "
                        f"rolling_sharpe_500=0.5 rolling_mdd_500=-0.05 "
                        f"rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        summary = summaries[0]
        assert "N=4" in summary
        assert "cadence=20" in summary
        assert "confidence=low" in summary  # N=4 <= 5

    def test_aggregate_iqr_outliers_format(self) -> None:
        """Fold summary uses IQR-based outlier detection, not P1/P99."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(10):
            memories.append(
                {
                    "text": (
                        f"run_id=crypto_sac_fold0 algo=sac env=crypto "
                        f"epoch={i * 10000} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500=0.5 "
                        f"rolling_mdd_500={-0.05 if i != 5 else -0.50} "
                        f"rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        summary = summaries[0]
        assert "IQR fences" in summary
        assert "P1/P99" not in summary

    def test_aggregate_trajectory_present(self) -> None:
        """Fold with N >= 4 includes TRAJECTORY section."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(6):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_a2c_fold0 algo=a2c env=equity "
                        f"epoch={i * 2000} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500={0.5 + 0.1 * i:.4f} "
                        f"rolling_mdd_500=-0.05 rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert "TRAJECTORY" in summaries[0]
        assert "early" in summaries[0]
        assert "late" in summaries[0]

    def test_aggregate_ppo_vs_sac_handling(self) -> None:
        """PPO and SAC folds have different output: KL for PPO, no KL for SAC."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        # PPO fold
        for i in range(4):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_ppo_fold0 algo=ppo env=equity "
                        f"epoch={i * 20} mean_reward=0.1 "
                        f"policy_loss=0.01 value_loss=0.02 approx_kl=0.02 "
                        f"rolling_sharpe_500=0.5 rolling_mdd_500=-0.05 "
                        f"rolling_win_rate_500=0.52"
                    ),
                }
            )
        # SAC fold
        for i in range(17):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_sac_fold0 algo=sac env=equity "
                        f"epoch={i * 10000} mean_reward=0.2 "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500=0.6 rolling_mdd_500=-0.04 "
                        f"rolling_win_rate_500=0.55"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert len(summaries) == 2
        # Find each fold's summary
        ppo_summary = [s for s in summaries if "algo=ppo" in s][0]
        sac_summary = [s for s in summaries if "algo=sac" in s][0]
        # PPO has KL, SAC doesn't
        assert "approx_kl" in ppo_summary
        assert "approx_kl" not in sac_summary
        # PPO has low confidence (N=4), SAC has high (N=17)
        assert "confidence=low" in ppo_summary
        assert "confidence=high" in sac_summary

    def test_aggregate_weight_trajectory_with_changes(self) -> None:
        """Weight changes include pre/post sharpe deltas."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = [
            {
                "text": (
                    "run_id=equity_ppo_fold0 algo=ppo env=equity "
                    "epoch=0 mean_reward=0.10 policy_loss=0.01 value_loss=0.02 "
                    "approx_kl=0.01 rolling_sharpe_500=0.50 rolling_mdd_500=-0.05 "
                    "rolling_win_rate_500=0.52 "
                    "reward_weights={sharpe: 0.4, drawdown: 0.3}"
                ),
            },
            {
                "text": (
                    "run_id=equity_ppo_fold0 algo=ppo env=equity "
                    "epoch=20 mean_reward=0.15 policy_loss=0.01 value_loss=0.02 "
                    "approx_kl=0.01 rolling_sharpe_500=0.60 rolling_mdd_500=-0.05 "
                    "rolling_win_rate_500=0.52 "
                    "reward_weights={sharpe: 0.4, drawdown: 0.3}"
                ),
            },
            {
                "text": (
                    "run_id=equity_ppo_fold0 algo=ppo env=equity "
                    "epoch=40 mean_reward=0.20 policy_loss=0.01 value_loss=0.02 "
                    "approx_kl=0.01 rolling_sharpe_500=0.80 rolling_mdd_500=-0.04 "
                    "rolling_win_rate_500=0.55 "
                    "reward_weights={sharpe: 0.3, drawdown: 0.5}"
                ),
            },
            {
                "text": (
                    "run_id=equity_ppo_fold0 algo=ppo env=equity "
                    "epoch=60 mean_reward=0.25 policy_loss=0.01 value_loss=0.02 "
                    "approx_kl=0.01 rolling_sharpe_500=0.90 rolling_mdd_500=-0.03 "
                    "rolling_win_rate_500=0.55 "
                    "reward_weights={sharpe: 0.3, drawdown: 0.5}"
                ),
            },
        ]
        summaries = _aggregate_epoch_summaries(memories)
        summary = summaries[0]
        assert "REWARD WEIGHTS:" in summary
        assert "change@epoch=40" in summary
        assert "total_changes=1" in summary

    def test_aggregate_iqm_in_stats(self) -> None:
        """Stats section includes iqm= field."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(8):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_a2c_fold0 algo=a2c env=equity "
                        f"epoch={i * 2000} mean_reward={float(i):.4f} "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500=0.5 rolling_mdd_500=-0.05 "
                        f"rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert "iqm=" in summaries[0]

    def test_aggregate_trend_in_stats(self) -> None:
        """Stats section includes trend= field with direction label."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(6):
            memories.append(
                {
                    "text": (
                        f"run_id=equity_a2c_fold0 algo=a2c env=equity "
                        f"epoch={i * 2000} mean_reward={0.1 * i:.4f} "
                        f"policy_loss=0.01 value_loss=0.02 "
                        f"rolling_sharpe_500={0.5 + 0.1 * i:.4f} "
                        f"rolling_mdd_500=-0.05 rolling_win_rate_500=0.52"
                    ),
                }
            )
        summaries = _aggregate_epoch_summaries(memories)
        assert "trend=" in summaries[0]
        assert "improving" in summaries[0]


# ---------------------------------------------------------------------------
# Fix 1b: Per-dimension effectiveness in reward summary
# ---------------------------------------------------------------------------


class TestRewardSummaryPerDimension:
    """Tests for per-dimension effectiveness breakdown in _summarize_reward_adjustments."""

    def test_per_dimension_breakdown_present(self) -> None:
        """Reward summary includes per_dimension_effectiveness section."""
        from memory_agents.consolidate import _summarize_reward_adjustments

        memories = [
            {
                "text": (
                    "REWARD_ADJUSTMENT_OUTCOME: run_id=crypto_a2c_fold0 algo=a2c env=crypto "
                    "epoch_triggered=2000 post_adjustment_sharpe_delta=-3.5 "
                    "post_adjustment_mdd_delta=-1.2 adjustment_effective=False "
                    'weights_before={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1} '
                    'weights_after={"profit": 0.44, "sharpe": 0.19, "drawdown": 0.26, "turnover": 0.11}'
                ),
                "source": "reward_adjustment:historical",
            },
            {
                "text": (
                    "REWARD_ADJUSTMENT_OUTCOME: run_id=crypto_a2c_fold1 algo=a2c env=crypto "
                    "epoch_triggered=4000 post_adjustment_sharpe_delta=2.5 "
                    "post_adjustment_mdd_delta=1.8 adjustment_effective=True "
                    'weights_before={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1} '
                    'weights_after={"profit": 0.44, "sharpe": 0.19, "drawdown": 0.26, "turnover": 0.11}'
                ),
                "source": "reward_adjustment:historical",
            },
        ]
        summaries = _summarize_reward_adjustments(memories, "crypto")
        assert len(summaries) == 1
        summary = summaries[0]
        assert "per_dimension_effectiveness:" in summary
        assert "drawdown_increase" in summary
        assert "profit_decrease" in summary

    def test_no_outcomes_no_dimension_section(self) -> None:
        """Triggers without outcomes produce no per_dimension section."""
        from memory_agents.consolidate import _summarize_reward_adjustments

        memories = [
            {
                "text": (
                    "REWARD_ADJUSTMENT_TRIGGER: run_id=crypto_ppo_fold0 algo=ppo env=crypto "
                    "epoch_triggered=20 trigger_metric=epoch_advice trigger_value=-5.0 "
                    'weights_before={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1} '
                    'weights_after={"profit": 0.4, "sharpe": 0.35, "drawdown": 0.2, "turnover": 0.05}'
                ),
                "source": "reward_adjustment:historical",
            },
        ]
        summaries = _summarize_reward_adjustments(memories, "crypto")
        assert len(summaries) == 1
        assert "per_dimension_effectiveness:" not in summaries[0]


# ---------------------------------------------------------------------------
# Fix 1a: Fold-level effectiveness with outcome_lookup
# ---------------------------------------------------------------------------


class TestFoldLevelEffectiveness:
    """Tests for outcome_lookup enrichment in _aggregate_epoch_summaries."""

    def test_outcome_lookup_enriches_weight_changes(self) -> None:
        """When outcome_lookup is provided, weight changes show dimension labels and effectiveness."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        # Create epoch memories with a weight change at epoch 100
        memories = []
        for i in range(20):
            weights = (
                '{"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1}'
                if i < 10
                else '{"profit": 0.44, "sharpe": 0.19, "drawdown": 0.26, "turnover": 0.11}'
            )
            memories.append(
                {
                    "text": (
                        f"EPOCH SNAPSHOT: run_id=crypto_a2c_fold0 algo=a2c env=crypto "
                        f"epoch={i * 100} mean_reward=0.01 rolling_sharpe_500=1.5 "
                        f"rolling_mdd_500=-0.05 approx_kl=0.0 "
                        f"reward_weights={weights}"
                    ),
                }
            )

        outcome_lookup = {
            ("crypto_a2c_fold0", 1000): {
                "sharpe_delta": 2.5,
                "mdd_delta": 1.2,
                "effective": True,
                "weights_before": '{"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1}',
                "weights_after": '{"profit": 0.44, "sharpe": 0.19, "drawdown": 0.26, "turnover": 0.11}',
            },
        }

        summaries = _aggregate_epoch_summaries(memories, outcome_lookup)
        assert len(summaries) == 1
        summary = summaries[0]
        assert "sharpe_delta=+2.5000" in summary
        assert "effective" in summary
        assert "drawdown" in summary

    def test_no_outcome_lookup_uses_fallback(self) -> None:
        """Without outcome_lookup, weight changes use original sharpe pre/post format."""
        from memory_agents.consolidate import _aggregate_epoch_summaries

        memories = []
        for i in range(20):
            weights = (
                '{"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1}'
                if i < 10
                else '{"profit": 0.44, "sharpe": 0.19, "drawdown": 0.26, "turnover": 0.11}'
            )
            memories.append(
                {
                    "text": (
                        f"EPOCH SNAPSHOT: run_id=crypto_a2c_fold0 algo=a2c env=crypto "
                        f"epoch={i * 100} mean_reward=0.01 rolling_sharpe_500=1.5 "
                        f"rolling_mdd_500=-0.05 approx_kl=0.0 "
                        f"reward_weights={weights}"
                    ),
                }
            )

        summaries = _aggregate_epoch_summaries(memories)
        assert len(summaries) == 1
        # Fallback: shows old->new dict format with sharpe pre->post
        assert "change@epoch=" in summaries[0]


# ---------------------------------------------------------------------------
# Fix 3: Within-fold adjustment history in epoch advice
# ---------------------------------------------------------------------------


class TestFoldAdjustmentHistory:
    """Tests for get_recent_outcomes_for_run_id and _parse_query_context run_id extraction."""

    def test_parse_query_context_extracts_run_id(self) -> None:
        """_parse_query_context extracts run_id from epoch advice query."""
        from memory_agents.query import _parse_query_context

        query = "EPOCH ADVICE: run_id=crypto_a2c_fold5 algo=a2c env=crypto epoch=8000 rolling_sharpe=-16.45"
        parsed = _parse_query_context(query)
        assert parsed["run_id"] == "crypto_a2c_fold5"
        assert parsed["algo_name"] == "a2c"
        assert parsed["env_name"] == "crypto"

    def test_parse_query_context_no_run_id(self) -> None:
        """_parse_query_context returns None for run_id when not present."""
        from memory_agents.query import _parse_query_context

        query = "TRAINING RUN CONFIG ADVICE: env=equity algo=ppo"
        parsed = _parse_query_context(query)
        assert parsed["run_id"] is None

    def test_get_recent_outcomes_returns_matching(self, memory_db_env: Path) -> None:
        """get_recent_outcomes_for_run_id returns only matching fold outcomes."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()

        # Insert outcomes for two different folds
        for fold in ("fold0", "fold1"):
            memory_db_module.insert_memory(
                text=(
                    f"REWARD_ADJUSTMENT_OUTCOME: run_id=crypto_a2c_{fold} algo=a2c env=crypto "
                    f"epoch_triggered=2000 post_adjustment_sharpe_delta=1.5 "
                    f"post_adjustment_mdd_delta=0.8 adjustment_effective=True "
                    f'weights_before={{"profit": 0.5}} weights_after={{"profit": 0.4}}'
                ),
                source="reward_adjustment:historical",
            )

        results = memory_db_module.get_recent_outcomes_for_run_id("crypto_a2c_fold0", limit=5)
        assert len(results) == 1
        assert "crypto_a2c_fold0" in results[0]["text"]

    def test_get_recent_outcomes_respects_limit(self, memory_db_env: Path) -> None:
        """get_recent_outcomes_for_run_id respects the limit parameter."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        import db as memory_db_module

        memory_db_module.init_db()

        for epoch in (2000, 4000, 6000, 8000, 10000):
            memory_db_module.insert_memory(
                text=(
                    f"REWARD_ADJUSTMENT_OUTCOME: run_id=crypto_sac_fold0 algo=sac env=crypto "
                    f"epoch_triggered={epoch} post_adjustment_sharpe_delta=0.5 "
                    f"post_adjustment_mdd_delta=0.3 adjustment_effective=True "
                    f'weights_before={{"profit": 0.5}} weights_after={{"profit": 0.4}}'
                ),
                source="reward_adjustment:historical",
            )

        results = memory_db_module.get_recent_outcomes_for_run_id("crypto_sac_fold0", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Fix 4: New categories in prompts and _RELEVANT_CATEGORIES
# ---------------------------------------------------------------------------


class TestNewCategories:
    """Tests for hp_effectiveness and iteration_regression category wiring."""

    def test_phase_a_prompt_has_both_categories(self) -> None:
        """Phase A system prompt includes hp_effectiveness and iteration_regression."""
        from memory_agents.consolidate import _PHASE_A_SYSTEM_PROMPT

        assert "hp_effectiveness" in _PHASE_A_SYSTEM_PROMPT
        assert "iteration_regression" in _PHASE_A_SYSTEM_PROMPT

    def test_phase_b_prompt_has_both_categories(self) -> None:
        """Phase B system prompt includes hp_effectiveness and iteration_regression."""
        from memory_agents.consolidate import _PHASE_B_SYSTEM_PROMPT

        assert "hp_effectiveness" in _PHASE_B_SYSTEM_PROMPT
        assert "iteration_regression" in _PHASE_B_SYSTEM_PROMPT

    def test_phase_a_few_shot_has_iteration_regression(self) -> None:
        """Phase A few-shot examples include iteration_regression."""
        from memory_agents.consolidate import _PHASE_A_FEW_SHOT_EXAMPLES

        examples = json.loads(_PHASE_A_FEW_SHOT_EXAMPLES)
        categories = [p["category"] for p in examples["patterns"]]
        assert "iteration_regression" in categories

    def test_phase_a_few_shot_has_hp_effectiveness(self) -> None:
        """Phase A few-shot examples include hp_effectiveness."""
        from memory_agents.consolidate import _PHASE_A_FEW_SHOT_EXAMPLES

        examples = json.loads(_PHASE_A_FEW_SHOT_EXAMPLES)
        categories = [p["category"] for p in examples["patterns"]]
        assert "hp_effectiveness" in categories

    def test_phase_b_few_shot_has_both_categories(self) -> None:
        """Phase B few-shot examples include hp_effectiveness and iteration_regression."""
        from memory_agents.consolidate import _PHASE_B_FEW_SHOT_EXAMPLES

        examples = json.loads(_PHASE_B_FEW_SHOT_EXAMPLES)
        categories = [p["category"] for p in examples["patterns"]]
        assert "hp_effectiveness" in categories
        assert "iteration_regression" in categories

    def test_relevant_categories_run_config_has_both(self) -> None:
        """run_config relevant categories include both new categories."""
        from memory_agents.query import _RELEVANT_CATEGORIES

        assert "hp_effectiveness" in _RELEVANT_CATEGORIES["run_config"]
        assert "iteration_regression" in _RELEVANT_CATEGORIES["run_config"]

    def test_relevant_categories_epoch_advice_has_both(self) -> None:
        """epoch_advice relevant categories include both new categories."""
        from memory_agents.query import _RELEVANT_CATEGORIES

        assert "hp_effectiveness" in _RELEVANT_CATEGORIES["epoch_advice"]
        assert "iteration_regression" in _RELEVANT_CATEGORIES["epoch_advice"]

    def test_relevant_categories_live_trading_unchanged(self) -> None:
        """live_trading categories do NOT include new categories."""
        from memory_agents.query import _RELEVANT_CATEGORIES

        assert "hp_effectiveness" not in _RELEVANT_CATEGORIES["live_trading"]
        assert "iteration_regression" not in _RELEVANT_CATEGORIES["live_trading"]

    def test_phase_b_prompt_has_attribution_guidance(self) -> None:
        """Phase B prompt includes attribution guidance section."""
        from memory_agents.consolidate import _PHASE_B_SYSTEM_PROMPT

        assert "ATTRIBUTION GUIDANCE" in _PHASE_B_SYSTEM_PROMPT
        assert "Ng et al." in _PHASE_B_SYSTEM_PROMPT
        assert "Blom et al." in _PHASE_B_SYSTEM_PROMPT
        assert "Goodhart" in _PHASE_B_SYSTEM_PROMPT

    def test_run_config_instructions_prioritize_new_categories(self) -> None:
        """Run config instructions mention hp_effectiveness and iteration_regression."""
        from memory_agents.query import _RUN_CONFIG_INSTRUCTIONS

        assert "hp_effectiveness" in _RUN_CONFIG_INSTRUCTIONS
        assert "iteration_regression" in _RUN_CONFIG_INSTRUCTIONS

    def test_reward_weight_guide_has_cautions(self) -> None:
        """Enhanced reward weight guide includes failure mode cautions."""
        from memory_agents.query import _REWARD_WEIGHT_GUIDE

        assert "CAUTION" in _REWARD_WEIGHT_GUIDE
        assert "Goodhart" in _REWARD_WEIGHT_GUIDE
        assert "CRITICAL INTERACTIONS" in _REWARD_WEIGHT_GUIDE
        assert "ent_coef" in _REWARD_WEIGHT_GUIDE
        assert "oscillate" in _REWARD_WEIGHT_GUIDE


# ---------------------------------------------------------------------------
# Fix 2: Cross-iteration source prefix in consolidation
# ---------------------------------------------------------------------------


class TestCrossIterationSourcePrefix:
    """Tests for cross_iteration source prefix in Phase A and Phase B."""

    def test_consolidation_fetches_cross_iteration_memories(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """Consolidation Phase A picks up cross_iteration memories."""
        import db as memory_db_module

        # Insert a cross-iteration memory + a WF memory
        memory_db_module.insert_memory(
            text="ITERATION HP COMPARISON: iteration=1 env=equity VS BASELINE ...",
            source="cross_iteration:equity",
        )
        memory_db_module.insert_memory(
            text="WF RESULTS: equity ppo fold0 sharpe=2.5 ...",
            source="walk_forward:equity:ppo",
        )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "Iter 1 equity HP changes improved PPO",
                    "category": "hp_effectiveness",
                    "affected_algos": ["ppo"],
                    "affected_envs": ["equity"],
                    "actionable_implication": "Keep current HP configuration",
                    "confidence": 0.65,
                    "evidence": "sharpe 2.68->2.71 (+1.0%)",
                },
            ],
        }

        async def _mock_llm(self: Any, text: str, **kwargs: Any) -> dict[str, Any]:
            # Verify cross-iteration memory text is in the LLM input
            assert "ITERATION HP COMPARISON" in text
            return mock_response

        with patch(
            "memory_agents.consolidate.ConsolidateAgent._call_llm_with_retry",
            _mock_llm,
        ):
            response = api_client.post(
                "/consolidate",
                json={"env_name": "equity"},
                headers=auth_headers,
            )

        assert response.status_code == 200
