"""Tests for PositionReconciler.

Verifies position reconciliation logic: detecting mismatches between
DB positions and broker state, auto-correcting DB, and alerting.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from swingrl.execution.reconciliation import PositionReconciler


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Mock ExchangeAdapter for broker position reads."""
    adapter = MagicMock()
    adapter.get_positions.return_value = []
    adapter.get_current_price.return_value = 450.0
    return adapter


@pytest.fixture
def reconciler(
    exec_config: Any,
    mock_db: Any,
    mock_adapter: MagicMock,
    mock_alerter: MagicMock,
) -> PositionReconciler:
    """Create PositionReconciler with mocked dependencies."""
    return PositionReconciler(
        config=exec_config,
        db=mock_db,
        adapter=mock_adapter,
        alerter=mock_alerter,
    )


class TestReconcile:
    """Test reconciliation logic."""

    def test_no_mismatch_returns_empty(
        self, reconciler: PositionReconciler, mock_adapter: MagicMock
    ) -> None:
        """No positions on either side means no adjustments."""
        mock_adapter.get_positions.return_value = []
        result = reconciler.reconcile("equity")
        assert result == []

    def test_crypto_skips_reconciliation(self, reconciler: PositionReconciler) -> None:
        """Crypto uses virtual balance, no broker-side reconciliation."""
        result = reconciler.reconcile("crypto")
        assert result == []

    def test_quantity_mismatch_creates_adjustment(
        self,
        reconciler: PositionReconciler,
        mock_db: Any,
        mock_adapter: MagicMock,
        mock_alerter: MagicMock,
    ) -> None:
        """Quantity mismatch between DB and broker creates adjustment trade."""
        # DB has 5 shares of SPY
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("SPY", "equity", 5.0, 440.0, 450.0, 50.0, "2026-01-01T00:00:00"),
            )

        # Broker has 7 shares of SPY
        mock_adapter.get_positions.return_value = [
            {"symbol": "SPY", "quantity": 7.0, "avg_entry_price": 440.0, "market_value": 3150.0}
        ]

        result = reconciler.reconcile("equity")
        assert len(result) > 0
        # Alert should have been sent
        mock_alerter.send_alert.assert_called()

    def test_missing_db_position_creates_it(
        self,
        reconciler: PositionReconciler,
        mock_adapter: MagicMock,
        mock_alerter: MagicMock,
    ) -> None:
        """Position in broker but not in DB creates new DB position."""
        mock_adapter.get_positions.return_value = [
            {"symbol": "QQQ", "quantity": 3.0, "avg_entry_price": 380.0, "market_value": 1140.0}
        ]

        result = reconciler.reconcile("equity")
        assert len(result) > 0
        mock_alerter.send_alert.assert_called()

    def test_extra_db_position_deletes_it(
        self,
        reconciler: PositionReconciler,
        mock_db: Any,
        mock_adapter: MagicMock,
        mock_alerter: MagicMock,
    ) -> None:
        """Position in DB but not in broker gets deleted (broker is truth)."""
        # DB has a position
        with mock_db.connection() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, environment, quantity, cost_basis, "
                "last_price, unrealized_pnl, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("XLE", "equity", 2.0, 80.0, 85.0, 10.0, "2026-01-01T00:00:00"),
            )

        # Broker has no positions
        mock_adapter.get_positions.return_value = []

        result = reconciler.reconcile("equity")
        assert len(result) > 0

        # Verify DB position was deleted
        with mock_db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE symbol = %s AND environment = %s",
                ("XLE", "equity"),
            ).fetchone()
        assert row is None
        mock_alerter.send_alert.assert_called()
