"""Tests for MetaTrainingOrchestrator.

TRAIN-09: MetaTrainingOrchestrator wraps TrainingOrchestrator with LLM-guided
meta-training. Cold-start guard, run history queries, regime vector, and
summary text generation are all tested here.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import psycopg
import pytest

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


def _make_mock_config(database_url: str | None = None) -> MagicMock:
    """Create a minimal mock SwingRLConfig with memory_agent section."""
    config = MagicMock()
    config.system.database_url = database_url or ""
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
    database_url: str | None = None,
) -> MetaTrainingOrchestrator:
    """Create orchestrator with mock config and client."""
    config = _make_mock_config(database_url)
    client = _make_mock_memory_client()
    return MetaTrainingOrchestrator(config=config, memory_client=client, database_url=database_url)


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

    def _create_hmm_db(self, tmp_path: Path, row: tuple | None = None) -> str:
        """Create PostgreSQL hmm_state_history table. Returns DATABASE_URL."""
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            pytest.skip("DATABASE_URL not set")
        conn = psycopg.connect(db_url, autocommit=True)
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(
            """
            CREATE TABLE hmm_state_history (
                date DATE,
                environment VARCHAR,
                p_bull DOUBLE PRECISION,
                p_bear DOUBLE PRECISION,
                p_crisis DOUBLE PRECISION
            )
            """
        )
        if row is not None:
            conn.execute(
                "INSERT INTO hmm_state_history VALUES (%s, %s, %s, %s, %s)",
                list(row),
            )
        conn.close()
        return db_url

    def test_current_regime_vector_queries_hmm_state_history(self, tmp_path: Path) -> None:
        """TRAIN-09: Returns dict with bull/bear/crisis/sideways keys from DB row."""
        db_path = self._create_hmm_db(tmp_path, row=("2026-01-01", "equity", 0.6, 0.2, 0.1))
        orch = _make_orchestrator(tmp_path, database_url=str(db_path))
        vec = orch._current_regime_vector("equity")
        assert set(vec.keys()) == {"bull", "bear", "crisis", "sideways"}
        assert abs(vec["bull"] - 0.6) < 1e-6
        assert abs(vec["bear"] - 0.2) < 1e-6
        assert abs(vec["crisis"] - 0.1) < 1e-6

    def test_current_regime_vector_computes_sideways_as_remainder(self, tmp_path: Path) -> None:
        """TRAIN-09: p_sideways = max(0, 1 - bull - bear - crisis)."""
        db_path = self._create_hmm_db(tmp_path, row=("2026-01-01", "equity", 0.4, 0.3, 0.1))
        orch = _make_orchestrator(tmp_path, database_url=str(db_path))
        vec = orch._current_regime_vector("equity")
        expected_sideways = 1.0 - 0.4 - 0.3 - 0.1
        assert abs(vec["sideways"] - expected_sideways) < 1e-6

    def test_current_regime_vector_uses_defaults_on_no_row(self, tmp_path: Path) -> None:
        """TRAIN-09: No matching row → defaults {bull:0.33, bear:0.33, crisis:0.17, sideways:0.17}."""
        db_path = self._create_hmm_db(tmp_path)  # no rows inserted
        orch = _make_orchestrator(tmp_path, database_url=str(db_path))
        vec = orch._current_regime_vector("equity")
        assert abs(vec["bull"] - 0.33) < 1e-6
        assert abs(vec["bear"] - 0.33) < 1e-6
        assert abs(vec["crisis"] - 0.17) < 1e-6
        assert abs(vec["sideways"] - 0.17) < 1e-6

    def test_current_regime_vector_handles_null_columns(self, tmp_path: Path) -> None:
        """TRAIN-09: NULL columns treated as 0.33/0.17 defaults."""
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            pytest.skip("DATABASE_URL not set")
        conn = psycopg.connect(db_url, autocommit=True)
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(
            """
            CREATE TABLE hmm_state_history (
                date DATE,
                environment VARCHAR,
                p_bull DOUBLE PRECISION,
                p_bear DOUBLE PRECISION,
                p_crisis DOUBLE PRECISION
            )
            """
        )
        conn.execute(
            "INSERT INTO hmm_state_history VALUES ('2026-01-01', 'equity', NULL, NULL, NULL)"
        )
        conn.close()
        orch = _make_orchestrator(tmp_path, database_url=db_url)
        vec = orch._current_regime_vector("equity")
        assert abs(vec["bull"] - 0.33) < 1e-6
        assert abs(vec["bear"] - 0.33) < 1e-6
        assert abs(vec["crisis"] - 0.17) < 1e-6

    def test_current_regime_vector_uses_defaults_on_exception(self, tmp_path: Path) -> None:
        """TRAIN-09: Exception (e.g., missing table) returns defaults."""
        # Use a bogus URL to trigger connection failure (defaults)
        orch = _make_orchestrator(tmp_path, database_url="postgresql://bad:bad@localhost:9999/bad")
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


# ---------------------------------------------------------------------------
# TestRunConfigPayload — C1-PAYLOAD-02
# ---------------------------------------------------------------------------


def _make_fold_history_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal backtest_results-shaped DataFrame for tests."""
    return pd.DataFrame(rows)


def _make_urlopen_mock(response_body: dict) -> MagicMock:
    """Return a context-manager mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_body).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestRunConfigPayload:
    """C1-PAYLOAD-02: run-config payload carries fold context + diagnoses."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _orch_with_db(self, tmp_path: Path) -> MetaTrainingOrchestrator:
        """Orchestrator with a non-empty database_url so the context path fires."""
        return _make_orchestrator(tmp_path, database_url="postgresql://fake/db")

    def _orch_without_db(self, tmp_path: Path) -> MetaTrainingOrchestrator:
        """Orchestrator with no database_url (cold-start context)."""
        return _make_orchestrator(tmp_path, database_url=None)

    def _prev_iter_rows(self) -> list[dict]:
        """Two equity/ppo rows: one healthy-shaped, one trade_shy-shaped."""
        # equity/ppo baseline: p25_trades=446, med_return=0.0535
        # healthy: trades>=446, win_rate>=0.562 (p25), mdd<=0.20, return>=0.0535
        # trade_shy: trades<446 AND total_return<0.0535
        return [
            {
                "iteration_number": 5,
                "environment": "equity",
                "algorithm": "ppo",
                "fold_number": 3,
                "sharpe": 1.8,
                "mdd": 0.05,
                "total_return": 0.08,
                "profit_factor": 2.0,
                "win_rate": 0.65,
                "total_trades": 460,
                "sortino": 2.0,
                "max_single_loss": None,
                "overfitting_class": "healthy",
                "is_control_fold": False,
                "calmar": 1.5,
                "hmm_p_bull": 0.5,
                "hmm_p_bear": 0.2,
                "vix_mean": 18.0,
                "train_start_date": None,
                "test_start_date": None,
                "test_end_date": None,
                "created_at": None,
            },
            {
                "iteration_number": 5,
                "environment": "equity",
                "algorithm": "ppo",
                "fold_number": 7,
                "sharpe": 0.3,
                "mdd": 0.04,
                "total_return": 0.01,  # < 0.0535 median
                "profit_factor": 1.2,
                "win_rate": 0.60,
                "total_trades": 200,  # < p25=446 → trade_shy
                "sortino": 0.4,
                "max_single_loss": None,
                "overfitting_class": "reject",
                "is_control_fold": False,
                "calmar": 0.3,
                "hmm_p_bull": 0.5,
                "hmm_p_bear": 0.2,
                "vix_mean": 18.0,
                "train_start_date": None,
                "test_start_date": None,
                "test_end_date": None,
                "created_at": None,
            },
        ]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_payload_contains_context_json(self, tmp_path: Path) -> None:
        """C1-PAYLOAD-02: query embeds context with target_metric + fold lists +
        per-fold prev-iteration diagnoses."""
        orch = self._orch_with_db(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=2)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.5, "bear": 0.3, "crisis": 0.1, "sideways": 0.1}
        )

        fold_ctx = {
            "fold_role": "neutral",
            "chronic_failure_folds": [2],
            "protected_winner_folds": [7],
            "prev_iter_cps_v1": 0.034,
        }
        history_df = _make_fold_history_df(self._prev_iter_rows())

        captured_req: list[MagicMock] = []

        def fake_urlopen(req: MagicMock, timeout: float | None = None) -> MagicMock:
            captured_req.append(req)
            return _make_urlopen_mock({"learning_rate": 0.0003})

        with (
            patch(
                "swingrl.memory.training.meta_orchestrator.load_fold_context",
                return_value=fold_ctx,
            ),
            patch(
                "swingrl.memory.training.meta_orchestrator.load_fold_history",
                return_value=history_df,
            ),
            patch("swingrl.memory.training.meta_orchestrator.psycopg") as mock_psycopg,
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            mock_conn = MagicMock()
            mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

            orch._query_run_config("equity", "ppo", iteration=6)

        assert captured_req, "urlopen was never called"
        req = captured_req[0]
        raw_data = req.data.decode("utf-8")
        outer = json.loads(raw_data)
        query_str = outer["query"]
        context_str = query_str.split("context=", 1)[1]
        context = json.loads(context_str)

        assert context["target_metric"] == "cps_v1_multiplicative"
        assert context["chronic_failure_folds"] == [2]
        assert context["protected_winner_folds"] == [7]
        assert abs(context["prev_iter_cps_v1"] - 0.034) < 1e-9

        diagnoses = context["prev_iter_diagnoses"]
        # json.loads converts int keys to strings
        assert "3" in diagnoses
        assert "7" in diagnoses
        assert diagnoses["3"] == "healthy"
        assert diagnoses["7"] == "trade_shy"

    def test_no_database_url_minimal_context(self, tmp_path: Path) -> None:
        """C1-PAYLOAD-02: without database_url, context = {target_metric} only; no crash."""
        orch = self._orch_without_db(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=2)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        captured_req: list[MagicMock] = []

        def fake_urlopen(req: MagicMock, timeout: float | None = None) -> MagicMock:
            captured_req.append(req)
            return _make_urlopen_mock({"learning_rate": 0.0003})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            orch._query_run_config("equity", "ppo", iteration=1)

        assert captured_req, "urlopen was never called"
        raw_data = captured_req[0].data.decode("utf-8")
        query_str = json.loads(raw_data)["query"]
        context_str = query_str.split("context=", 1)[1]
        context = json.loads(context_str)

        assert context == {"target_metric": "cps_v1_multiplicative"}

    def test_context_failure_fails_open(self, tmp_path: Path) -> None:
        """C1-PAYLOAD-02: load_fold_context raising → payload still sent with minimal context."""
        orch = self._orch_with_db(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=2)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        captured_req: list[MagicMock] = []

        def fake_urlopen(req: MagicMock, timeout: float | None = None) -> MagicMock:
            captured_req.append(req)
            return _make_urlopen_mock({"learning_rate": 0.0003})

        with (
            patch(
                "swingrl.memory.training.meta_orchestrator.load_fold_context",
                side_effect=RuntimeError("db is dead"),
            ),
            patch("swingrl.memory.training.meta_orchestrator.psycopg") as mock_psycopg,
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            mock_conn = MagicMock()
            mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)
            result = orch._query_run_config("equity", "ppo", iteration=1)

        # Training was NOT blocked — result came back
        assert result == {"learning_rate": 0.0003}
        assert captured_req, "urlopen should have been called despite context failure"

        raw_data = captured_req[0].data.decode("utf-8")
        query_str = json.loads(raw_data)["query"]
        context_str = query_str.split("context=", 1)[1]
        context = json.loads(context_str)
        # Only the target_metric key — no fold lists, no diagnoses
        assert context == {"target_metric": "cps_v1_multiplicative"}

    def test_null_metric_row_skipped_not_fatal(self, tmp_path: Path) -> None:
        """C1-PAYLOAD-02: a prev-iter row with None profit_factor is skipped; others diagnosed."""
        orch = self._orch_with_db(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=2)
        orch._current_regime_vector = MagicMock(
            return_value={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
        )

        rows = self._prev_iter_rows()
        # Corrupt fold 3's profit_factor to None — diagnose_fold should raise DataError
        rows[0]["profit_factor"] = None
        history_df = _make_fold_history_df(rows)

        fold_ctx = {
            "fold_role": "neutral",
            "chronic_failure_folds": [],
            "protected_winner_folds": [],
            "prev_iter_cps_v1": 0.020,
        }

        captured_req: list[MagicMock] = []

        def fake_urlopen(req: MagicMock, timeout: float | None = None) -> MagicMock:
            captured_req.append(req)
            return _make_urlopen_mock({"learning_rate": 0.0003})

        with (
            patch(
                "swingrl.memory.training.meta_orchestrator.load_fold_context",
                return_value=fold_ctx,
            ),
            patch(
                "swingrl.memory.training.meta_orchestrator.load_fold_history",
                return_value=history_df,
            ),
            patch("swingrl.memory.training.meta_orchestrator.psycopg") as mock_psycopg,
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            mock_conn = MagicMock()
            mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)
            result = orch._query_run_config("equity", "ppo", iteration=6)

        # Training not blocked
        assert result == {"learning_rate": 0.0003}
        assert captured_req

        raw_data = captured_req[0].data.decode("utf-8")
        query_str = json.loads(raw_data)["query"]
        context_str = query_str.split("context=", 1)[1]
        context = json.loads(context_str)

        diagnoses = context["prev_iter_diagnoses"]
        # fold 3 was skipped (None profit_factor → DataError), fold 7 was diagnosed
        assert "3" not in diagnoses
        assert "7" in diagnoses
        assert diagnoses["7"] == "trade_shy"

    def test_cold_start_guard_short_circuits_before_context_assembly(self, tmp_path: Path) -> None:
        """C1-PAYLOAD-02: cold-start guard (pattern_count < min) returns {} before DB work."""
        orch = self._orch_with_db(tmp_path)
        orch._get_pattern_count = MagicMock(return_value=0)

        with (
            patch("swingrl.memory.training.meta_orchestrator.load_fold_context") as mock_lfc,
            patch("swingrl.memory.training.meta_orchestrator.psycopg") as mock_psycopg,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            result = orch._query_run_config("equity", "ppo", iteration=0)

        assert result == {}
        mock_lfc.assert_not_called()
        mock_psycopg.connect.assert_not_called()
        mock_urlopen.assert_not_called()
