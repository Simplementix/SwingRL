"""Tests for SwingRL dashboard pages -- syntax validation and behavioral tests."""

from __future__ import annotations

import ast
import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
# Mock plotly.express for page imports
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.express", MagicMock())

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
    """Test get_latest_trades helper with in-memory SQLite."""

    @pytest.fixture()
    def trade_db(self) -> sqlite3.Connection:
        """Create an in-memory SQLite DB with trade_log table."""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE trade_log (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                environment TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                fill_price REAL NOT NULL,
                commission REAL,
                slippage REAL,
                broker TEXT
            )
            """
        )
        return conn

    def test_get_latest_trades_returns_limit(self, trade_db: sqlite3.Connection) -> None:
        """PAPER-15: get_latest_trades returns at most `limit` rows ordered by timestamp DESC."""
        for i in range(10):
            trade_db.execute(
                "INSERT INTO trade_log (trade_id, timestamp, environment, symbol, side, "
                "quantity, fill_price) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"t{i:03d}", f"2026-03-0{i % 9 + 1}T12:00:00", "equity", "SPY", "buy", 10, 450.0),
            )
        trade_db.commit()

        result = get_latest_trades(trade_db, limit=5)
        assert len(result) == 5
        # Verify descending order
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_get_latest_trades_empty_table(self, trade_db: sqlite3.Connection) -> None:
        """PAPER-15: get_latest_trades returns empty list when no trades exist."""
        result = get_latest_trades(trade_db, limit=5)
        assert result == []
