"""Tests for CorporateActionDetector — split heuristic and action tracking.

DATA-11: Corporate action detection uses overnight price spike heuristic
(30% equity, 40% crypto) and integrates with PostgreSQL corporate_actions table
to suppress false-positive quarantine.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from swingrl.config.schema import load_config
from swingrl.data.corporate_actions import CorporateActionDetector
from swingrl.data.db import DatabaseManager

# ---------------------------------------------------------------------------
# Skip entire module if no PostgreSQL available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ca_config_yaml(tmp_path: Path) -> str:
    """Config YAML with system section pointing to DATABASE_URL."""
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://test:test@localhost:5432/swingrl_test"
    )  # pragma: allowlist secret
    return textwrap.dedent(f"""\
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ]
          max_position_size: 0.25
          max_drawdown_pct: 0.10
          daily_loss_limit_pct: 0.02
        crypto:
          symbols: [BTCUSDT, ETHUSDT]
          max_position_size: 0.50
          max_drawdown_pct: 0.12
          daily_loss_limit_pct: 0.03
          min_order_usd: 10.0
        capital:
          equity_usd: 400.0
          crypto_usd: 47.0
        paths:
          data_dir: data/
          db_dir: db/
          models_dir: models/
          logs_dir: logs/
        logging:
          level: INFO
          json_logs: false
        system:
          database_url: "{db_url}"
          duckdb_path: data/db/market_data.ddb
          sqlite_path: data/db/trading_ops.db
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)


@pytest.fixture
def ca_config(tmp_path: Path, ca_config_yaml: str) -> Any:
    """Load config with DATABASE_URL."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(ca_config_yaml)
    return load_config(config_file)


@pytest.fixture
def ca_db(ca_config: Any) -> DatabaseManager:
    """Create a DatabaseManager with schema and ensure cleanup."""
    DatabaseManager.reset()
    mgr = DatabaseManager(ca_config)
    mgr.init_schema()
    yield mgr  # type: ignore[misc]
    # Truncate all tables for test isolation
    with mgr.connection() as conn:
        conn.execute(
            "DO $$ DECLARE r RECORD; BEGIN "
            "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
            "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
            "END LOOP; END $$"
        )
    DatabaseManager.reset()


# ---------------------------------------------------------------------------
# Tests: detect_overnight_spike
# ---------------------------------------------------------------------------


class TestDetectOvernightSpike:
    """DATA-11: Overnight price spike detection."""

    def test_equity_spike_above_30pct_detected(self, ca_db: DatabaseManager) -> None:
        """DATA-11: >30% equity overnight change is flagged as potential action."""
        detector = CorporateActionDetector(db=ca_db)
        # 2:1 split scenario: price drops from 200 to 100 (50% drop)
        result = detector.detect_overnight_spike(
            symbol="AAPL",
            prev_close=200.0,
            curr_close=100.0,
            date="2024-06-10",
            environment="equity",
        )
        assert result is True

    def test_equity_change_under_30pct_passes(self, ca_db: DatabaseManager) -> None:
        """DATA-11: <30% equity overnight change is NOT flagged."""
        detector = CorporateActionDetector(db=ca_db)
        # 20% drop — normal volatility
        result = detector.detect_overnight_spike(
            symbol="AAPL",
            prev_close=100.0,
            curr_close=80.0,
            date="2024-06-10",
            environment="equity",
        )
        assert result is False

    def test_crypto_spike_above_40pct_detected(self, ca_db: DatabaseManager) -> None:
        """DATA-11: >40% crypto overnight change is flagged."""
        detector = CorporateActionDetector(db=ca_db)
        # 45% drop
        result = detector.detect_overnight_spike(
            symbol="BTCUSDT",
            prev_close=40000.0,
            curr_close=22000.0,
            date="2024-06-10",
            environment="crypto",
        )
        assert result is True

    def test_crypto_change_under_40pct_passes(self, ca_db: DatabaseManager) -> None:
        """DATA-11: <40% crypto overnight change (35%) is NOT flagged."""
        detector = CorporateActionDetector(db=ca_db)
        # 35% drop — passes crypto threshold
        result = detector.detect_overnight_spike(
            symbol="BTCUSDT",
            prev_close=40000.0,
            curr_close=26000.0,
            date="2024-06-10",
            environment="crypto",
        )
        assert result is False

    def test_known_action_suppresses_detection(self, ca_db: DatabaseManager) -> None:
        """DATA-11: Known corporate action in table prevents spike detection."""
        detector = CorporateActionDetector(db=ca_db)
        # First record the action
        detector.record_action(
            symbol="AAPL",
            action_type="split",
            effective_date="2024-06-10",
            ratio=2.0,
        )
        # Now detect — should return False since action is known
        result = detector.detect_overnight_spike(
            symbol="AAPL",
            prev_close=200.0,
            curr_close=100.0,
            date="2024-06-10",
            environment="equity",
        )
        assert result is False


# ---------------------------------------------------------------------------
# Tests: record_action and is_known_action
# ---------------------------------------------------------------------------


class TestRecordAndQuery:
    """DATA-11: Recording and querying corporate actions."""

    def test_record_action_inserts_row(self, ca_db: DatabaseManager) -> None:
        """DATA-11: record_action() inserts a row into corporate_actions table."""
        detector = CorporateActionDetector(db=ca_db)
        detector.record_action(
            symbol="AAPL",
            action_type="split",
            effective_date="2024-06-10",
            ratio=2.0,
        )
        with ca_db.connection() as conn:
            row = conn.execute("SELECT * FROM corporate_actions WHERE symbol = 'AAPL'").fetchone()
        assert row is not None
        assert row["action_type"] == "split"
        assert row["ratio"] == 2.0
        assert row["processed"] is False or row["processed"] == 0

    def test_is_known_action_returns_true(self, ca_db: DatabaseManager) -> None:
        """DATA-11: is_known_action() returns True when action exists."""
        detector = CorporateActionDetector(db=ca_db)
        detector.record_action(
            symbol="AAPL",
            action_type="split",
            effective_date="2024-06-10",
            ratio=2.0,
        )
        assert detector.is_known_action("AAPL", "2024-06-10") is True

    def test_is_known_action_returns_false(self, ca_db: DatabaseManager) -> None:
        """DATA-11: is_known_action() returns False when no action exists."""
        detector = CorporateActionDetector(db=ca_db)
        assert detector.is_known_action("AAPL", "2024-06-10") is False

    def test_split_recorded_with_correct_type(self, ca_db: DatabaseManager) -> None:
        """DATA-11: A 2:1 split is recorded with action_type='split' and ratio=2.0."""
        detector = CorporateActionDetector(db=ca_db)
        detector.record_action(
            symbol="TSLA",
            action_type="split",
            effective_date="2024-08-25",
            ratio=2.0,
        )
        with ca_db.connection() as conn:
            row = conn.execute(
                "SELECT action_type, ratio FROM corporate_actions WHERE symbol = 'TSLA'"
            ).fetchone()
        assert row["action_type"] == "split"
        assert row["ratio"] == 2.0


# ---------------------------------------------------------------------------
# Tests: check_and_suppress
# ---------------------------------------------------------------------------


class TestCheckAndSuppress:
    """DATA-11: check_and_suppress integrates spike detection + known action lookup."""

    def test_suppress_known_action(self, ca_db: DatabaseManager) -> None:
        """DATA-11: check_and_suppress returns True for known action (suppress quarantine)."""
        detector = CorporateActionDetector(db=ca_db)
        detector.record_action(
            symbol="AAPL",
            action_type="split",
            effective_date="2024-06-10",
            ratio=2.0,
        )
        # 50% drop on a known split date
        result = detector.check_and_suppress(
            symbol="AAPL",
            date="2024-06-10",
            pct_change=0.50,
            environment="equity",
        )
        assert result is True

    def test_no_suppress_unknown_spike(self, ca_db: DatabaseManager) -> None:
        """DATA-11: check_and_suppress returns False for unknown spike."""
        detector = CorporateActionDetector(db=ca_db)
        result = detector.check_and_suppress(
            symbol="AAPL",
            date="2024-06-10",
            pct_change=0.50,
            environment="equity",
        )
        assert result is False

    def test_no_suppress_small_change(self, ca_db: DatabaseManager) -> None:
        """DATA-11: check_and_suppress returns False for small price change."""
        detector = CorporateActionDetector(db=ca_db)
        result = detector.check_and_suppress(
            symbol="AAPL",
            date="2024-06-10",
            pct_change=0.10,
            environment="equity",
        )
        assert result is False
