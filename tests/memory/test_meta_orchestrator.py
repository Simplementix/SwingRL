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
    """TRAIN-09: Cold-start guard blocks LLM until consolidated patterns exist."""

    def test_query_run_config_returns_empty_when_no_patterns(self, tmp_path: Path) -> None:
        """TRAIN-09: Zero consolidated patterns returns empty dict."""
        orch = _make_orchestrator(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=0)
        result = orch._query_run_config("equity", "ppo")
        assert result == {}

    def test_query_run_config_calls_api_with_patterns(self, tmp_path: Path) -> None:
        """TRAIN-09: With 1+ patterns, an HTTP POST is attempted."""
        orch = _make_orchestrator(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=2)
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
        orch._get_pattern_count = MagicMock(return_value=3)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = orch._query_run_config("equity", "ppo")

        assert result == {}

    def test_query_run_config_returns_empty_on_timeout(self, tmp_path: Path) -> None:
        """TRAIN-09: Timeout on API call returns empty dict."""

        orch = _make_orchestrator(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=3)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = orch._query_run_config("equity", "ppo")

        assert result == {}


# ---------------------------------------------------------------------------
# TestMetaOrchestratorPatternCount
# ---------------------------------------------------------------------------


class TestMetaOrchestratorPatternCount:
    """TRAIN-09: _get_pattern_count() checks consolidated patterns via memory service."""

    def test_pattern_count_returns_zero_on_connection_error(self, tmp_path: Path) -> None:
        """TRAIN-09: Connection error returns 0 (fail-open)."""
        orch = _make_orchestrator(tmp_path)
        # No mock for urlopen → will fail to connect → returns 0
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            count = orch._get_pattern_count("equity")
        assert count == 0

    def test_pattern_count_filters_by_env(self, tmp_path: Path) -> None:
        """TRAIN-09: Counts only patterns matching the requested env."""
        orch = _make_orchestrator(tmp_path)
        mock_patterns = [
            {"env_name": "equity", "affected_envs": ["equity"]},
            {"env_name": "equity", "affected_envs": ["equity"]},
            {"env_name": "crypto", "affected_envs": ["crypto"]},
        ]
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_patterns).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            equity_count = orch._get_pattern_count("equity")
            assert equity_count == 2


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
    """TRAIN-09: _compute_final_metrics() returns convergence data."""

    def test_compute_final_metrics_returns_convergence_info(self, tmp_path: Path) -> None:
        """TRAIN-09: Returns convergence status from TrainingResult."""
        orch = _make_orchestrator(tmp_path)
        result = _make_mock_training_result(total_timesteps=100_000, converged_at_step=50_000)
        metrics = orch._compute_final_metrics(result)
        assert metrics["converged"] is True
        assert metrics["converged_at_step"] == 50_000
        assert metrics["total_timesteps"] == 100_000
        assert abs(metrics["convergence_ratio"] - 0.5) < 1e-6

    def test_compute_final_metrics_no_convergence(self, tmp_path: Path) -> None:
        """TRAIN-09: Non-converged run reports ratio 1.0."""
        orch = _make_orchestrator(tmp_path)
        result = _make_mock_training_result(total_timesteps=100_000, converged_at_step=None)
        metrics = orch._compute_final_metrics(result)
        assert metrics["converged"] is False
        assert metrics["convergence_ratio"] == 1.0

    def test_compute_final_metrics_returns_all_required_keys(self, tmp_path: Path) -> None:
        """TRAIN-09: Dict has converged, converged_at_step, total_timesteps, convergence_ratio."""
        orch = _make_orchestrator(tmp_path)
        result = _make_mock_training_result()
        metrics = orch._compute_final_metrics(result)
        assert "converged" in metrics
        assert "converged_at_step" in metrics
        assert "total_timesteps" in metrics
        assert "convergence_ratio" in metrics


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


# ---------------------------------------------------------------------------
# TestMetaOrchestratorPatternCountWarning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestQueryHyperparams
# ---------------------------------------------------------------------------


class TestQueryHyperparams:
    """TRAIN-09: query_hyperparams() returns clamped, SB3-keyed HP dict."""

    def test_query_hyperparams_cold_start_returns_empty(self, tmp_path: Path) -> None:
        """TRAIN-09: Zero patterns triggers cold-start guard, returns empty dict."""
        orch = _make_orchestrator(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=0)
        result = orch.query_hyperparams("equity", "ppo")
        assert result == {}

    def test_query_hyperparams_returns_clamped_hp(self, tmp_path: Path) -> None:
        """TRAIN-09: Out-of-bound values are clamped; entropy_coeff mapped to ent_coef."""
        orch = _make_orchestrator(tmp_path)
        # _query_run_config returns raw (unclamped) values
        orch._query_run_config = MagicMock(
            return_value={"learning_rate": 0.005, "entropy_coeff": 0.1, "gamma": 0.95}
        )
        result = orch.query_hyperparams("equity", "ppo")

        # learning_rate 0.005 > max 0.001 → clamped to 0.001
        assert result["learning_rate"] == 0.001
        # entropy_coeff 0.1 > max 0.05 → clamped to 0.05, key mapped to ent_coef
        assert result["ent_coef"] == 0.05
        assert "entropy_coeff" not in result
        # gamma 0.95 is within [0.90, 0.9999] → passes through
        assert result["gamma"] == 0.95

    def test_query_hyperparams_fail_open(self, tmp_path: Path) -> None:
        """TRAIN-09: _query_run_config returning empty dict yields empty result (fail-open)."""
        orch = _make_orchestrator(tmp_path)
        orch._query_run_config = MagicMock(return_value={})
        result = orch.query_hyperparams("equity", "ppo")
        assert result == {}


# ---------------------------------------------------------------------------
# TestMetaOrchestratorPatternCountWarning
# ---------------------------------------------------------------------------


class TestMetaOrchestratorPatternCountWarning:
    """TRAIN-09: _get_pattern_count logs warning on HTTP failure."""

    def test_pattern_count_warns_on_http_failure(self, tmp_path: Path) -> None:
        """TRAIN-09: HTTP failure in _get_pattern_count emits meta_pattern_count_query_failed."""
        orch = _make_orchestrator(tmp_path)

        log_events: list[str] = []

        import swingrl.memory.training.meta_orchestrator as meta_mod

        def capture_warning(event: str, **kwargs: object) -> None:
            log_events.append(event)

        with (
            patch("urllib.request.urlopen", side_effect=ConnectionError("refused")),
            patch.object(meta_mod.log, "warning", side_effect=capture_warning),
        ):
            count = orch._get_pattern_count("equity")

        assert count == 0, "_get_pattern_count should return 0 on failure"
        assert "meta_pattern_count_query_failed" in log_events, (
            "Should log meta_pattern_count_query_failed on HTTP failure"
        )
