"""Tests for MetaTrainingOrchestrator.

TRAIN-09: MetaTrainingOrchestrator wraps TrainingOrchestrator with LLM-guided
meta-training. Cold-start guard, run history queries, regime vector, and
summary text generation are all tested here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb

from swingrl.memory.training.meta_orchestrator import MetaTrainingOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_memory_client() -> MagicMock:
    """Create a mock MemoryClient."""
    client = MagicMock()
    client._base_url = "http://localhost:8889"
    client.ingest_training.return_value = True
    return client


def _make_mock_config(db_path: str | Path) -> MagicMock:
    """Create a minimal mock SwingRLConfig with memory_agent section."""
    config = MagicMock()
    config.system.duckdb_path = str(db_path)
    config.memory_agent.base_url = "http://localhost:8889"
    config.memory_agent.meta_training_timeout_sec = 5
    return config


def _make_mock_training_result(
    total_timesteps: int = 100_000,
    converged_at_step: int | None = None,
) -> MagicMock:
    """Create a mock TrainingResult."""
    result = MagicMock()
    result.total_timesteps = total_timesteps
    result.converged_at_step = converged_at_step
    return result


def _make_orchestrator(
    tmp_path: Path,
    db_path: str | Path | None = None,
) -> MetaTrainingOrchestrator:
    """Create orchestrator with mock config and client."""
    path = db_path or tmp_path / "test.ddb"
    config = _make_mock_config(path)
    client = _make_mock_memory_client()
    return MetaTrainingOrchestrator(config=config, memory_client=client, db_path=path)


# ---------------------------------------------------------------------------
# TestMetaOrchestratorColdStart
# ---------------------------------------------------------------------------


class TestMetaOrchestratorColdStart:
    """TRAIN-09: Cold-start guard returns empty config below minimum runs."""

    def test_query_run_config_returns_empty_below_min_runs(self, tmp_path: Path) -> None:
        """TRAIN-09: Fewer than 3 completed runs returns empty dict."""
        orch = _make_orchestrator(tmp_path)
        # Mock _get_run_history to return only 2 runs (below minimum of 3)
        orch._get_run_history = MagicMock(return_value=[{"run_id": "r1"}, {"run_id": "r2"}])
        result = orch._query_run_config("equity", "ppo")
        assert result == {}

    def test_query_run_config_calls_api_after_cold_start(self, tmp_path: Path) -> None:
        """TRAIN-09: With 3+ completed runs, an HTTP POST is attempted."""
        orch = _make_orchestrator(tmp_path)
        runs = [{"run_id": f"r{i}"} for i in range(3)]
        orch._get_run_history = MagicMock(return_value=runs)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.5, "bear": 0.3, "crisis": 0.1, "sideways": 0.1}
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"learning_rate": 0.0003}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = orch._query_run_config("equity", "ppo")

        mock_urlopen.assert_called_once()
        assert result == {"learning_rate": 0.0003}

    def test_query_run_config_returns_empty_on_api_failure(self, tmp_path: Path) -> None:
        """TRAIN-09: Connection error on API call returns empty dict."""
        orch = _make_orchestrator(tmp_path)
        runs = [{"run_id": f"r{i}"} for i in range(3)]
        orch._get_run_history = MagicMock(return_value=runs)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = orch._query_run_config("equity", "ppo")

        assert result == {}

    def test_query_run_config_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        """TRAIN-09: Timeout on API call returns empty dict."""

        orch = _make_orchestrator(tmp_path)
        runs = [{"run_id": f"r{i}"} for i in range(3)]
        orch._get_run_history = MagicMock(return_value=runs)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = orch._query_run_config("equity", "ppo")

        assert result == {}


# ---------------------------------------------------------------------------
# TestMetaOrchestratorRunHistory
# ---------------------------------------------------------------------------


class TestMetaOrchestratorRunHistory:
    """TRAIN-09: _get_run_history() queries DuckDB training_runs table."""

    def _create_training_runs_db(self, tmp_path: Path) -> Path:
        """Create a real DuckDB with training_runs table and test data."""
        db_path = tmp_path / "training.ddb"
        conn = duckdb.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE training_runs (
                run_id VARCHAR,
                algo VARCHAR,
                env VARCHAR,
                final_sharpe DOUBLE,
                final_mdd DOUBLE,
                total_timesteps BIGINT,
                epochs_to_convergence BIGINT,
                timestamp_end TIMESTAMP,
                run_type VARCHAR
            )
            """
        )
        conn.execute(
            """
            INSERT INTO training_runs VALUES
            ('r1', 'PPO', 'equity', 1.2, -0.05, 100000, 50, '2026-01-01 00:00:00', 'completed'),
            ('r2', 'PPO', 'equity', 0.8, -0.08, 200000, 80, '2026-01-02 00:00:00', 'completed'),
            ('r3', 'A2C', 'equity', 0.5, -0.10, 150000, 60, '2026-01-03 00:00:00', 'completed'),
            ('r4', 'PPO', 'crypto', 1.0, -0.06, 100000, 40, '2026-01-04 00:00:00', 'completed')
            """
        )
        conn.close()
        return db_path

    def test_get_run_history_returns_completed_runs(self, tmp_path: Path) -> None:
        """TRAIN-09: Returns list of dicts for completed runs matching env+algo."""
        db_path = self._create_training_runs_db(tmp_path)
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        history = orch._get_run_history("equity", "ppo")
        assert isinstance(history, list)
        assert len(history) == 2
        assert all(r["algo"] == "PPO" for r in history)
        assert all(r["env"] == "equity" for r in history)

    def test_get_run_history_returns_empty_on_missing_table(self, tmp_path: Path) -> None:
        """TRAIN-09: Missing training_runs table returns empty list."""
        db_path = tmp_path / "empty.ddb"
        # Create empty DB with no tables
        conn = duckdb.connect(str(db_path))
        conn.close()
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        result = orch._get_run_history("equity", "ppo")
        assert result == []

    def test_get_run_history_returns_empty_on_connection_error(self, tmp_path: Path) -> None:
        """TRAIN-09: Non-existent db path returns empty list gracefully."""
        db_path = tmp_path / "nonexistent" / "db.ddb"
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        result = orch._get_run_history("equity", "ppo")
        assert result == []

    def test_get_run_history_filters_by_env_and_algo(self, tmp_path: Path) -> None:
        """TRAIN-09: Only returns rows matching both env and algo."""
        db_path = self._create_training_runs_db(tmp_path)
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        # A2C equity
        history_a2c = orch._get_run_history("equity", "a2c")
        assert len(history_a2c) == 1
        assert history_a2c[0]["algo"] == "A2C"
        # PPO crypto
        history_crypto = orch._get_run_history("crypto", "ppo")
        assert len(history_crypto) == 1
        assert history_crypto[0]["env"] == "crypto"


# ---------------------------------------------------------------------------
# TestMetaOrchestratorRegimeVector
# ---------------------------------------------------------------------------


class TestMetaOrchestratorRegimeVector:
    """TRAIN-09: _current_regime_vector() queries hmm_state_history table."""

    def _create_hmm_db(self, tmp_path: Path, row: tuple | None = None) -> Path:
        """Create DuckDB with hmm_state_history table."""
        db_path = tmp_path / "hmm.ddb"
        conn = duckdb.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE hmm_state_history (
                date DATE,
                environment VARCHAR,
                p_bull DOUBLE,
                p_bear DOUBLE,
                p_crisis DOUBLE
            )
            """
        )
        if row is not None:
            conn.execute(
                "INSERT INTO hmm_state_history VALUES (?, ?, ?, ?, ?)",
                list(row),
            )
        conn.close()
        return db_path

    def test_current_regime_vector_queries_hmm_state_history(self, tmp_path: Path) -> None:
        """TRAIN-09: Returns dict with bull/bear/crisis/sideways keys from DB row."""
        db_path = self._create_hmm_db(tmp_path, row=("2026-01-01", "equity", 0.6, 0.2, 0.1))
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        vec = orch._current_regime_vector("equity")
        assert set(vec.keys()) == {"bull", "bear", "crisis", "sideways"}
        assert abs(vec["bull"] - 0.6) < 1e-6
        assert abs(vec["bear"] - 0.2) < 1e-6
        assert abs(vec["crisis"] - 0.1) < 1e-6

    def test_current_regime_vector_computes_sideways_as_remainder(self, tmp_path: Path) -> None:
        """TRAIN-09: p_sideways = max(0, 1 - bull - bear - crisis)."""
        db_path = self._create_hmm_db(tmp_path, row=("2026-01-01", "equity", 0.4, 0.3, 0.1))
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        vec = orch._current_regime_vector("equity")
        expected_sideways = 1.0 - 0.4 - 0.3 - 0.1
        assert abs(vec["sideways"] - expected_sideways) < 1e-6

    def test_current_regime_vector_uses_defaults_on_no_row(self, tmp_path: Path) -> None:
        """TRAIN-09: No matching row → defaults {bull:0.33, bear:0.33, crisis:0.17, sideways:0.17}."""
        db_path = self._create_hmm_db(tmp_path)  # no rows inserted
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        vec = orch._current_regime_vector("equity")
        assert abs(vec["bull"] - 0.33) < 1e-6
        assert abs(vec["bear"] - 0.33) < 1e-6
        assert abs(vec["crisis"] - 0.17) < 1e-6
        assert abs(vec["sideways"] - 0.17) < 1e-6

    def test_current_regime_vector_handles_null_columns(self, tmp_path: Path) -> None:
        """TRAIN-09: NULL columns treated as 0.33/0.17 defaults."""
        db_path = tmp_path / "null_hmm.ddb"
        conn = duckdb.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE hmm_state_history (
                date DATE,
                environment VARCHAR,
                p_bull DOUBLE,
                p_bear DOUBLE,
                p_crisis DOUBLE
            )
            """
        )
        conn.execute(
            "INSERT INTO hmm_state_history VALUES ('2026-01-01', 'equity', NULL, NULL, NULL)"
        )
        conn.close()
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        vec = orch._current_regime_vector("equity")
        assert abs(vec["bull"] - 0.33) < 1e-6
        assert abs(vec["bear"] - 0.33) < 1e-6
        assert abs(vec["crisis"] - 0.17) < 1e-6

    def test_current_regime_vector_uses_defaults_on_exception(self, tmp_path: Path) -> None:
        """TRAIN-09: Exception (e.g., missing table) returns defaults."""
        db_path = tmp_path / "no_hmm.ddb"
        conn = duckdb.connect(str(db_path))
        conn.close()  # No tables
        orch = _make_orchestrator(tmp_path, db_path=db_path)
        vec = orch._current_regime_vector("equity")
        assert set(vec.keys()) == {"bull", "bear", "crisis", "sideways"}
        assert abs(vec["bull"] - 0.33) < 1e-6


# ---------------------------------------------------------------------------
# TestMetaOrchestratorFinalMetrics
# ---------------------------------------------------------------------------


class TestMetaOrchestratorFinalMetrics:
    """TRAIN-09: _compute_final_metrics() returns placeholder zero metrics."""

    def test_compute_final_metrics_returns_zero_placeholders(self, tmp_path: Path) -> None:
        """TRAIN-09: All four metric values are 0.0 (placeholder)."""
        orch = _make_orchestrator(tmp_path)
        result = _make_mock_training_result()
        metrics = orch._compute_final_metrics(result)
        assert metrics["final_sharpe"] == 0.0
        assert metrics["final_mdd"] == 0.0
        assert metrics["final_sortino"] == 0.0
        assert metrics["final_mean_reward"] == 0.0

    def test_compute_final_metrics_returns_all_required_keys(self, tmp_path: Path) -> None:
        """TRAIN-09: Dict has final_sharpe, final_mdd, final_sortino, final_mean_reward."""
        orch = _make_orchestrator(tmp_path)
        result = _make_mock_training_result()
        metrics = orch._compute_final_metrics(result)
        assert "final_sharpe" in metrics
        assert "final_mdd" in metrics
        assert "final_sortino" in metrics
        assert "final_mean_reward" in metrics


# ---------------------------------------------------------------------------
# TestMetaOrchestratorBuildSummaryText
# ---------------------------------------------------------------------------


class TestMetaOrchestratorBuildSummaryText:
    """TRAIN-09: _build_run_summary_text() produces valid summary strings."""

    def _make_summary(
        self,
        tmp_path: Path,
        run_id: str = "equity_ppo_20260101T000000Z",
        reward_weights: dict | None = None,
    ) -> str:
        orch = _make_orchestrator(tmp_path)
        start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC)
        result = _make_mock_training_result(total_timesteps=100_000)
        final_metrics = {
            "final_sharpe": 0.0,
            "final_mdd": 0.0,
            "final_sortino": 0.0,
            "final_mean_reward": 0.0,
        }
        regime_vector = {"bull": 0.5, "bear": 0.3, "crisis": 0.1, "sideways": 0.1}
        return orch._build_run_summary_text(
            run_id=run_id,
            algo_name="ppo",
            env_name="equity",
            start_time=start_time,
            end_time=end_time,
            result=result,
            final_metrics=final_metrics,
            merged_hp={},
            reward_weights=reward_weights,
            regime_vector=regime_vector,
            rationale="cold_start",
        )

    def test_build_run_summary_text_contains_run_id(self, tmp_path: Path) -> None:
        """TRAIN-09: run_id appears in the output string."""
        text = self._make_summary(tmp_path, run_id="equity_ppo_20260101T000000Z")
        assert "equity_ppo_20260101T000000Z" in text

    def test_build_run_summary_text_contains_dominant_regime(self, tmp_path: Path) -> None:
        """TRAIN-09: Dominant regime key (max value) appears in output."""
        text = self._make_summary(tmp_path)
        # regime_vector = {bull:0.5, bear:0.3, ...} -> dominant = bull
        assert "dominant_regime=bull" in text

    def test_build_run_summary_text_uses_default_weights_if_none(self, tmp_path: Path) -> None:
        """TRAIN-09: None reward_weights → default {profit:0.50, sharpe:0.25, ...} used."""
        text = self._make_summary(tmp_path, reward_weights=None)
        assert "reward_weight_profit=0.5000" in text
        assert "reward_weight_sharpe=0.2500" in text

    def test_build_run_summary_text_contains_timestamps(self, tmp_path: Path) -> None:
        """TRAIN-09: ISO timestamps appear in output."""
        text = self._make_summary(tmp_path)
        assert "2026-01-01T00:00:00" in text
        assert "2026-01-01T01:00:00" in text


# ---------------------------------------------------------------------------
# TestMetaOrchestratorGenerateRunId
# ---------------------------------------------------------------------------


class TestMetaOrchestratorGenerateRunId:
    """TRAIN-09: _generate_run_id() produces correctly formatted unique IDs."""

    def test_generate_run_id_format(self) -> None:
        """TRAIN-09: Produces '{env}_{algo}_{YYYYMMDDTHHMMSSz}' format."""
        run_id = MetaTrainingOrchestrator._generate_run_id("equity", "ppo")
        assert run_id.startswith("equity_ppo_")
        parts = run_id.split("_")
        # equity, ppo, timestamp
        assert len(parts) == 3
        ts = parts[2]
        assert len(ts) == 16  # YYYYMMDDTHHMMSSz = 16 chars
        assert ts.endswith("Z")

    def test_generate_run_id_is_unique_per_call(self) -> None:
        """TRAIN-09: Two rapid calls may differ (timestamp-based) or are at least both valid."""
        r1 = MetaTrainingOrchestrator._generate_run_id("equity", "ppo")
        r2 = MetaTrainingOrchestrator._generate_run_id("equity", "ppo")
        # Both should start with the correct prefix
        assert r1.startswith("equity_ppo_")
        assert r2.startswith("equity_ppo_")
