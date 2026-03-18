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


@pytest.fixture()
def memory_db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SQLite db to a temp directory and set required env vars."""
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_path))
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_URL", "http://mock-ollama:11434")
    return db_path


@pytest.fixture()
def api_client(memory_db_env: Path):  # type: ignore[no-untyped-def]
    """Return a FastAPI TestClient with Ollama wait mocked out."""
    from fastapi.testclient import TestClient

    # Evict cached memory service modules to ensure clean imports.
    for mod_name in list(_MEMORY_MODULE_NAMES):
        sys.modules.pop(mod_name, None)

    import db as memory_db_module

    memory_db_module.init_db()

    async def _mock_wait(*args: Any, **kwargs: Any) -> None:
        return

    with patch("app._wait_for_ollama", _mock_wait):
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


def _make_ollama_response(content: dict[str, Any]) -> MagicMock:
    """Build a mock httpx response that looks like Ollama /api/chat output."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": json.dumps(content)}}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_tags_response() -> MagicMock:
    """Build a mock httpx response for Ollama /api/tags (health check)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "qwen2.5:3b"}]}
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

    def test_health_returns_healthy_with_mocked_ollama(self, api_client: Any) -> None:
        """TRAIN-06: GET /health returns status: healthy when Ollama responds."""
        with patch("routers.core.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.get = AsyncMock(return_value=_make_tags_response())

            response = api_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["db"] is True

    def test_health_no_auth_required(self, api_client: Any) -> None:
        """TRAIN-06: GET /health requires no authentication."""
        with patch("routers.core.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.get = AsyncMock(return_value=_make_tags_response())

            response = api_client.get("/health")

        assert response.status_code == 200

    def test_health_degraded_when_ollama_down(self, api_client: Any) -> None:
        """TRAIN-06: GET /health returns status: degraded when Ollama is unreachable."""
        with patch("routers.core.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.get = AsyncMock(side_effect=ConnectionError("refused"))

            response = api_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["ollama"] is False
        assert body["db"] is True


# ---------------------------------------------------------------------------
# TestTrainingEndpoints
# ---------------------------------------------------------------------------


class TestTrainingEndpoints:
    """Tests for /training/run_config and /training/epoch_advice."""

    def test_run_config_returns_valid_shape_with_mocked_ollama(
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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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

    def test_run_config_returns_defaults_on_ollama_failure(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: /training/run_config returns safe defaults when Ollama fails."""
        with patch("memory_agents.query.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(side_effect=ConnectionError("Ollama down"))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(malformed_response))

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
        """TRAIN-06: ConsolidateAgent flags contradicting patterns."""
        import db as memory_db_module

        memory_db_module.insert_consolidation(
            pattern_text="PPO equity improved significantly in bull momentum regime",
            source_count=5,
            category="regime_performance",
            affected_algos=["ppo"],
            affected_envs=["equity"],
            confidence=0.7,
        )

        for i in range(2):
            api_client.post(
                "/ingest",
                json={"text": f"Crypto observation {i}", "source": "walk_forward:equity:sac"},
                headers=auth_headers,
            )

        mock_response = {
            "patterns": [
                {
                    "pattern_text": "SAC crypto degraded and decreased returns in bear crash regime",
                    "category": "regime_performance",
                    "affected_algos": ["sac"],
                    "affected_envs": ["crypto"],
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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

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
# TestWaitForOllama
# ---------------------------------------------------------------------------


class TestWaitForOllama:
    """Tests for the _wait_for_ollama startup helper."""

    @pytest.fixture(autouse=True)
    def evict_stale_modules(self) -> None:  # type: ignore[return]
        """Evict cached memory service modules for clean import."""
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)
        yield
        for mod_name in list(_MEMORY_MODULE_NAMES):
            sys.modules.pop(mod_name, None)

    @pytest.mark.anyio
    async def test_wait_for_ollama_raises_on_timeout(self) -> None:
        """TRAIN-06: _wait_for_ollama raises RuntimeError when Ollama never responds."""
        import app

        with patch("app.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.get = AsyncMock(side_effect=ConnectionError("refused"))

            with pytest.raises(RuntimeError, match="not healthy"):
                await app._wait_for_ollama(url="http://fake:11434", timeout_sec=1)

    @pytest.mark.anyio
    async def test_wait_for_ollama_returns_when_healthy(self) -> None:
        """TRAIN-06: _wait_for_ollama returns immediately when Ollama responds 200."""
        import app

        with patch("app.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.get = AsyncMock(return_value=_make_tags_response())

            await app._wait_for_ollama(url="http://mock:11434", timeout_sec=10)


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
