"""Tests for SwingRL dashboard pages -- syntax validation and behavioral tests."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Repo root for AST-based file checks
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"

# ---------------------------------------------------------------------------
# A. Syntax and structure tests (AST-based, no Streamlit import needed)
# ---------------------------------------------------------------------------


class TestDashboardSyntax:
    """Verify all dashboard files parse without syntax errors."""

    def test_app_parses(self) -> None:
        """PAPER-15: app.py parses without syntax errors."""
        source = (DASHBOARD_DIR / "app.py").read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_portfolio_page_parses(self) -> None:
        """PAPER-15: Portfolio page parses and has expected functions."""
        source = (DASHBOARD_DIR / "pages" / "1_Portfolio.py").read_text()
        tree = ast.parse(source)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "fetch_portfolio_snapshots" in func_names
        assert "compute_summary_metrics" in func_names

    def test_trade_log_page_parses(self) -> None:
        """PAPER-15: Trade Log page parses and has expected functions."""
        source = (DASHBOARD_DIR / "pages" / "2_Trade_Log.py").read_text()
        tree = ast.parse(source)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "fetch_trades" in func_names
        assert "compute_trade_stats" in func_names

    def test_risk_metrics_page_parses(self) -> None:
        """PAPER-15: Risk Metrics page parses and has expected functions."""
        source = (DASHBOARD_DIR / "pages" / "3_Risk_Metrics.py").read_text()
        tree = ast.parse(source)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "get_current_drawdown" in func_names
        assert "drawdown_color" in func_names
        assert "fetch_circuit_breaker_events" in func_names

    def test_system_health_page_parses(self) -> None:
        """PAPER-15: System Health page parses and has get_traffic_light_status."""
        source = (DASHBOARD_DIR / "pages" / "4_System_Health.py").read_text()
        tree = ast.parse(source)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "get_traffic_light_status" in func_names
        assert "get_latest_trades" in func_names

    def test_iteration_history_page_parses(self) -> None:
        """Phase 0.7: Iteration History page parses with expected helpers."""
        source = (DASHBOARD_DIR / "pages" / "5_Iteration_History.py").read_text()
        tree = ast.parse(source)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name in (
            "fetch_iteration_history",
            "fetch_fold_history",
            "build_cps_trend_figure",
            "build_fold_heatmap",
            "build_treatment_vs_control_table",
            "build_iteration_table",
        ):
            assert name in func_names, f"missing helper: {name}"

    def test_dockerfile_exists(self) -> None:
        """PAPER-15: Dockerfile.dashboard exists."""
        assert (DASHBOARD_DIR / "Dockerfile.dashboard").exists()

    def test_requirements_lists_streamlit(self) -> None:
        """PAPER-15: requirements.txt includes streamlit."""
        reqs = (DASHBOARD_DIR / "requirements.txt").read_text()
        assert "streamlit" in reqs


# ---------------------------------------------------------------------------
# B. Behavioral tests for extracted helper functions
# ---------------------------------------------------------------------------

# Mock streamlit and streamlit_autorefresh before importing dashboard modules
_mock_st = MagicMock()
_mock_autorefresh = MagicMock()
sys.modules["streamlit"] = _mock_st
sys.modules["streamlit_autorefresh"] = _mock_autorefresh
# Mock duckdb to avoid import issues in app.py
if "duckdb" not in sys.modules:
    sys.modules["duckdb"] = MagicMock()
# Mock plotly.express + plotly.graph_objects + plotly.subplots for page imports
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.express", MagicMock())
sys.modules.setdefault("plotly.graph_objects", MagicMock())
sys.modules.setdefault("plotly.subplots", MagicMock())

# Add dashboard to path for imports
sys.path.insert(0, str(DASHBOARD_DIR))


def _load_module(name: str, filepath: Path) -> object:
    """Load a Python module from file path without triggering Streamlit calls."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register module before exec to handle self-imports
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the system health module to get testable functions
_health_module = _load_module(
    "dashboard.pages.system_health",
    DASHBOARD_DIR / "pages" / "4_System_Health.py",
)
get_traffic_light_status = _health_module.get_traffic_light_status  # type: ignore[attr-defined]
get_latest_trades = _health_module.get_latest_trades  # type: ignore[attr-defined]

# Load the iteration history module (Phase 0.7) to get its testable helpers.
# Note: importing this module will execute the page-rendering code which uses
# Streamlit calls — those are mocked above so the import is safe.
_iter_module = _load_module(
    "dashboard.pages.iteration_history",
    DASHBOARD_DIR / "pages" / "5_Iteration_History.py",
)
build_iteration_table = _iter_module.build_iteration_table  # type: ignore[attr-defined]
build_treatment_vs_control_table = _iter_module.build_treatment_vs_control_table  # type: ignore[attr-defined]


class TestTrafficLightStatus:
    """Test get_traffic_light_status logic for equity and crypto environments."""

    def test_traffic_light_green_equity(self) -> None:
        """PAPER-15: Equity snapshot 2 hours ago is green (within 26h window)."""
        ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert get_traffic_light_status(ts, "equity") == "green"

    def test_traffic_light_yellow_equity(self) -> None:
        """PAPER-15: Equity snapshot 27 hours ago is yellow (stale but exists)."""
        ts = (datetime.now(UTC) - timedelta(hours=27)).isoformat()
        assert get_traffic_light_status(ts, "equity") == "yellow"

    def test_traffic_light_red_equity(self) -> None:
        """PAPER-15: Equity snapshot 53+ hours ago is red (>2x 26h)."""
        ts = (datetime.now(UTC) - timedelta(hours=53)).isoformat()
        assert get_traffic_light_status(ts, "equity") == "red"

    def test_traffic_light_red_no_data(self) -> None:
        """PAPER-15: None timestamp returns red."""
        assert get_traffic_light_status(None, "equity") == "red"

    def test_traffic_light_green_crypto(self) -> None:
        """PAPER-15: Crypto snapshot 3 hours ago is green (within 5h window)."""
        ts = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        assert get_traffic_light_status(ts, "crypto") == "green"

    def test_traffic_light_yellow_crypto(self) -> None:
        """PAPER-15: Crypto snapshot 6 hours ago is yellow (stale but within 2x)."""
        ts = (datetime.now(UTC) - timedelta(hours=6)).isoformat()
        assert get_traffic_light_status(ts, "crypto") == "yellow"

    def test_traffic_light_red_crypto(self) -> None:
        """PAPER-15: Crypto snapshot 11+ hours ago is red (>2x 5h)."""
        ts = (datetime.now(UTC) - timedelta(hours=11)).isoformat()
        assert get_traffic_light_status(ts, "crypto") == "red"


class TestGetLatestTrades:
    """Test get_latest_trades helper with PostgreSQL."""

    @pytest.fixture()
    def trade_db(self) -> Any:
        """Create a PostgreSQL connection with trade_log table."""
        # ⚠️ DANGER: DELETE FROM trade_log disabled 2026-04-07 — prod incident.
        # See tests/agents/test_backtest.py:_create_backtest_schema for context.
        raise RuntimeError(
            "trade_db fixture is disabled pending prod-DB guard. "
            "Run via scripts/ci-homelab.sh which isolates against swingrl_test."
        )
        import psycopg
        from psycopg.rows import dict_row

        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            pytest.skip("DATABASE_URL not set")
        conn = psycopg.connect(db_url, row_factory=dict_row, autocommit=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                environment TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                fill_price DOUBLE PRECISION NOT NULL,
                commission DOUBLE PRECISION,
                slippage DOUBLE PRECISION,
                broker TEXT
            )
            """
        )
        # conn.execute("DELETE FROM trade_log")
        conn.commit()
        return conn

    def test_get_latest_trades_returns_limit(self, trade_db: Any) -> None:
        """PAPER-15: get_latest_trades returns at most `limit` rows ordered by timestamp DESC."""
        for i in range(10):
            trade_db.execute(
                "INSERT INTO trade_log (trade_id, timestamp, environment, symbol, side, "
                "quantity, fill_price) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (f"t{i:03d}", f"2026-03-0{i % 9 + 1}T12:00:00", "equity", "SPY", "buy", 10, 450.0),
            )
        trade_db.commit()

        result = get_latest_trades(trade_db, limit=5)
        assert len(result) == 5
        # Verify descending order
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_latest_trades_empty_table(self, trade_db: Any) -> None:
        """PAPER-15: get_latest_trades returns empty list when no trades exist."""
        result = get_latest_trades(trade_db, limit=5)
        assert result == []


# ---------------------------------------------------------------------------
# C. Iteration History page (Phase 0.7) — pure helper tests
# ---------------------------------------------------------------------------


class TestIterationHistoryHelpers:
    """Phase 0.7: testable pure helpers in 5_Iteration_History.py."""

    def _make_history_with_deltas(self) -> Any:
        import pandas as pd  # local import — module-level may not be loaded yet

        return pd.DataFrame(
            [
                {
                    "iteration_number": 0,
                    "environment": "equity",
                    "cps_v1_multiplicative": 0.0117,
                    "cps_v2_additive": 0.076,
                    "cps_v3_sortino": None,
                    "cps_v1_treatment_only": None,
                    "cps_v1_control_only": None,
                    "median_return": 0.0616,
                    "worst_fold_mdd": 0.41,
                    "winners_count": 11,
                    "chronic_failure_count": 0,
                    "gate_passed": True,
                    "regression_flag": False,
                    "cps_v1_delta": None,
                    "return_delta": None,
                    "worst_mdd_delta": None,
                    "dedup_rows_dropped": 0,
                },
                {
                    "iteration_number": 3,
                    "environment": "equity",
                    "cps_v1_multiplicative": 0.0134,
                    "cps_v2_additive": -0.43,
                    "cps_v3_sortino": 1.55,
                    "cps_v1_treatment_only": 0.0123,
                    "cps_v1_control_only": 0.0343,
                    "median_return": 0.0605,
                    "worst_fold_mdd": 0.36,
                    "winners_count": 12,
                    "chronic_failure_count": 5,
                    "gate_passed": True,
                    "regression_flag": False,
                    "cps_v1_delta": 0.002,
                    "return_delta": 0.001,
                    "worst_mdd_delta": -0.04,
                    "dedup_rows_dropped": 0,
                },
                {
                    "iteration_number": 4,
                    "environment": "crypto",
                    "cps_v1_multiplicative": 0.124,
                    "cps_v2_additive": 0.37,
                    "cps_v3_sortino": 4.89,
                    "cps_v1_treatment_only": 0.0832,
                    "cps_v1_control_only": 0.4206,
                    "median_return": 0.6016,
                    "worst_fold_mdd": 0.46,
                    "winners_count": 5,
                    "chronic_failure_count": 1,
                    "gate_passed": False,
                    "regression_flag": False,
                    "cps_v1_delta": 0.028,
                    "return_delta": 0.167,
                    "worst_mdd_delta": -0.02,
                    "dedup_rows_dropped": 0,
                },
            ]
        )

    def test_build_iteration_table_keeps_expected_columns(self) -> None:
        history = self._make_history_with_deltas()
        table = build_iteration_table(history)
        for col in (
            "iteration_number",
            "environment",
            "cps_v1_multiplicative",
            "cps_v2_additive",
            "winners_count",
            "regression_flag",
        ):
            assert col in table.columns

    def test_build_iteration_table_sorted_by_env_then_iter(self) -> None:
        history = self._make_history_with_deltas()
        table = build_iteration_table(history).reset_index(drop=True)
        envs = table["environment"].tolist()
        # crypto sorts before equity alphabetically
        assert envs[0] == "crypto"
        assert envs[1] == "equity"
        assert envs[2] == "equity"

    def test_treatment_vs_control_table_filters_iter0(self) -> None:
        """Iter 0 has NULL treatment/control — must be excluded."""
        history = self._make_history_with_deltas()
        tc = build_treatment_vs_control_table(history)
        assert (tc["iter"] != 0).all()
        assert len(tc) == 2  # equity iter 3 + crypto iter 4

    def test_treatment_vs_control_ratio_computed(self) -> None:
        """control_over_treatment column = control_v1 / treatment_v1."""
        history = self._make_history_with_deltas()
        tc = build_treatment_vs_control_table(history)
        crypto_row = tc[(tc["env"] == "crypto") & (tc["iter"] == 4)].iloc[0]
        # 0.4206 / 0.0832 ≈ 5.06
        assert crypto_row["control_over_treatment"] == pytest.approx(5.057, abs=0.01)

    def test_treatment_vs_control_table_handles_missing_columns(self) -> None:
        """If the input lacks the treatment/control columns entirely, return empty."""
        import pandas as pd

        empty = pd.DataFrame([{"iteration_number": 0, "environment": "equity"}])
        result = build_treatment_vs_control_table(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# D. app.get_pg_conn — self-heal regression test
# ---------------------------------------------------------------------------


class TestGetPgConnSelfHeal:
    """Phase 0.7 follow-up: app.get_pg_conn must reconnect when the cached
    singleton has been closed by another page (pages 1-4 all close it).

    Uses MagicMock for the connection so we don't need a live DB.
    """

    def test_get_pg_conn_reconnects_on_closed_cached_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If _cached_pg_conn returns a closed connection, get_pg_conn must
        clear the cache and re-call to obtain a fresh one.

        Mocks _cached_pg_conn directly because Streamlit's cache_resource
        decorator (which wraps it in production) is itself mocked at the
        top of this file.
        """
        app_module = _load_module("dashboard.app", DASHBOARD_DIR / "app.py")

        closed_conn = MagicMock()
        closed_conn.closed = True
        fresh_conn = MagicMock()
        fresh_conn.closed = False

        call_count = {"n": 0}

        def fake_cached_pg_conn() -> Any:
            call_count["n"] += 1
            # First call returns the stale (closed) singleton; subsequent
            # calls return a fresh open connection.
            return closed_conn if call_count["n"] == 1 else fresh_conn

        # cache_resource exposes a .clear() method in production
        fake_cached_pg_conn.clear = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setattr(app_module, "_cached_pg_conn", fake_cached_pg_conn)

        result = app_module.get_pg_conn()

        # Self-heal expectations:
        assert result is fresh_conn, "should return the fresh, non-closed connection"
        assert call_count["n"] == 2, "should re-invoke _cached_pg_conn after clearing"
        assert fake_cached_pg_conn.clear.called, "should call .clear() on the cache"

    def test_get_pg_conn_passthrough_when_cached_singleton_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: cached connection is healthy → returned as-is, no clear."""
        app_module = _load_module("dashboard.app", DASHBOARD_DIR / "app.py")

        healthy_conn = MagicMock()
        healthy_conn.closed = False

        def fake_cached_pg_conn() -> Any:
            return healthy_conn

        fake_cached_pg_conn.clear = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setattr(app_module, "_cached_pg_conn", fake_cached_pg_conn)

        result = app_module.get_pg_conn()
        assert result is healthy_conn
        assert not fake_cached_pg_conn.clear.called
