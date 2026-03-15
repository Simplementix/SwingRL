"""Tests for the swingrl-memory FastAPI service and DuckDB training telemetry migrations.

TRAIN-06, TRAIN-07: Covers:
- swingrl-memory FastAPI endpoints (ingest, health, training, debug, consolidation)
- DuckDB telemetry tables (training_epochs, meta_decisions, reward_adjustments, ALTER training_runs)
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
    mock_resp.json.return_value = {"models": [{"name": "qwen3:14b"}]}
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
                "source": "training_run:historical",
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
            json={"text": "First memory", "source": "training_run:historical"},
            headers=auth_headers,
        )
        r2 = api_client.post(
            "/ingest",
            json={"text": "Second memory", "source": "training_run:historical"},
            headers=auth_headers,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["id"] == r1.json()["id"] + 1

    def test_ingest_without_api_key_returns_401(self, api_client: Any) -> None:
        """TRAIN-06: POST /ingest without X-API-Key header returns 401."""
        response = api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "training_run:historical"},
        )
        assert response.status_code == 401

    def test_ingest_with_wrong_api_key_returns_401(self, api_client: Any) -> None:
        """TRAIN-06: POST /ingest with wrong X-API-Key returns 401."""
        response = api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "training_run:historical"},
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
            json={"text": "Memory A", "source": "training_run:historical"},
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
            json={"text": "Equity memory", "source": "training_run:historical"},
            headers=auth_headers,
        )
        api_client.post(
            "/ingest",
            json={"text": "Macro memory", "source": "macro_correlation:historical"},
            headers=auth_headers,
        )

        response = api_client.get(
            "/debug/memories?source=training_run:historical",
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert all(row["source"] == "training_run:historical" for row in body)

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
                json={"text": f"Training observation {i}", "source": "training_run:historical"},
                headers=auth_headers,
            )

        mock_response = {
            "pattern_text": "PPO performs better in bull regimes with lower entropy_coeff",
            "affected_algos": ["ppo"],
            "affected_envs": ["equity"],
            "actionable_implication": "Reduce entropy_coeff to 0.005 in bull regime",
            "confidence": 0.8,
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
        assert body["consolidated"] == 1

    def test_consolidate_rejects_malformed_llm_output_and_logs_quality(
        self, api_client: Any, auth_headers: dict[str, str]
    ) -> None:
        """TRAIN-06: ConsolidateAgent rejects malformed LLM output and logs to quality table."""
        api_client.post(
            "/ingest",
            json={"text": "Some memory", "source": "training_run:historical"},
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
        """TRAIN-06: ConsolidateAgent flags contradicting patterns with conflicting_with FK."""
        import db as memory_db_module

        memory_db_module.insert_consolidation(
            pattern_text="PPO equity improved significantly in bull momentum regime",
            source_count=5,
        )

        for i in range(2):
            api_client.post(
                "/ingest",
                json={"text": f"Crypto observation {i}", "source": "training_run:historical"},
                headers=auth_headers,
            )

        mock_response = {
            "pattern_text": "SAC crypto degraded and decreased returns in bear crash regime",
            "affected_algos": ["sac"],
            "affected_envs": ["crypto"],
            "actionable_implication": "Reduce position sizing in bear regime",
            "confidence": 0.7,
        }

        with patch("memory_agents.consolidate.httpx.AsyncClient") as mock_client_class:
            mock_cm = AsyncMock()
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_cm)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_cm.post = AsyncMock(return_value=_make_ollama_response(mock_response))

            response = api_client.post("/consolidate", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["consolidated"] == 1

        # At least one consolidation in the table should have conflicting_with set
        rows = memory_db_module.get_consolidations(limit=5)
        conflict_rows = [r for r in rows if r["conflicting_with"] is not None]
        assert len(conflict_rows) >= 1, (
            "Expected at least one consolidation with conflicting_with set"
        )

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
            ("get", "/debug/memories", None),
            ("get", "/debug/consolidations", None),
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
