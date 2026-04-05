"""Tests for crypto stop-price polling daemon thread.

Tests verify halt checking, exception recovery, and daemon thread properties.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestStopPollingChecksHalt:
    """Verify polling daemon respects halt flag."""

    def test_stop_polling_checks_halt(self) -> None:
        """PAPER-16: stop polling skips execution when halted."""
        from swingrl.scheduler.stop_polling import _poll_stop_prices

        mock_config = MagicMock()
        mock_db = MagicMock()
        call_count = 0

        def halt_side_effect(db: MagicMock) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise StopIteration("Break loop for test")
            return True

        with patch("swingrl.scheduler.stop_polling.is_halted", side_effect=halt_side_effect):
            with patch("swingrl.scheduler.stop_polling.time.sleep", side_effect=lambda _: None):
                with pytest.raises(StopIteration):
                    _poll_stop_prices(mock_config, mock_db)

        # is_halted was called at least once
        assert call_count >= 1


class TestStopPollingExceptionRecovery:
    """Verify polling daemon recovers from exceptions."""

    def test_stop_polling_exception_recovery(self) -> None:
        """PAPER-16: polling continues after exception."""
        from swingrl.scheduler.stop_polling import _poll_stop_prices

        mock_config = MagicMock()
        mock_db = MagicMock()
        call_count = 0

        def halt_side_effect(db: MagicMock) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False  # Not halted, proceed to body
            if call_count >= 3:
                raise StopIteration("Break loop for test")
            return False

        def connection_side_effect() -> MagicMock:
            raise ConnectionError("Test DB error")

        mock_db.connection.side_effect = connection_side_effect

        with patch("swingrl.scheduler.stop_polling.is_halted", side_effect=halt_side_effect):
            with patch("swingrl.scheduler.stop_polling.time.sleep", side_effect=lambda _: None):
                with pytest.raises(StopIteration):
                    _poll_stop_prices(mock_config, mock_db)

        # Made it past the first exception (call_count > 1)
        assert call_count >= 2


class TestStopPollingThread:
    """Verify daemon thread properties."""

    def test_start_stop_polling_thread_is_daemon(self) -> None:
        """PAPER-16: stop polling thread is daemon=True."""
        from swingrl.scheduler.stop_polling import start_stop_polling_thread

        mock_config = MagicMock()
        mock_db = MagicMock()

        with patch("swingrl.scheduler.stop_polling._poll_stop_prices"):
            thread = start_stop_polling_thread(mock_config, mock_db)

        assert thread.daemon is True
        # Clean up: thread won't actually run since _poll_stop_prices is mocked


class TestStopPollingUsesPositionsTable:
    """Verify stop_polling queries 'positions' table (not 'position_tracker')."""

    def test_query_references_positions_table(self) -> None:
        """INT-03: stop_polling SELECT references 'FROM positions' not 'FROM position_tracker'."""
        import inspect

        from swingrl.scheduler.stop_polling import _poll_stop_prices

        source = inspect.getsource(_poll_stop_prices)
        assert "FROM positions " in source
        assert "position_tracker" not in source

    def test_processes_rows_with_stop_levels(self) -> None:
        """INT-03: stop_polling processes rows with stop_loss_price and take_profit_price."""
        from swingrl.scheduler.stop_polling import _check_stop_levels

        mock_config = MagicMock()
        mock_db = MagicMock()

        # Row with stop_loss and take_profit
        row = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "stop_loss_price": 50000.0,
            "take_profit_price": 70000.0,
        }

        # Mock httpx (imported locally inside _check_stop_levels)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"price": "60000.0"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_resp):
            # Should not raise
            _check_stop_levels(row, mock_config, mock_db)

    def test_skips_row_without_stop_levels(self) -> None:
        """INT-03: stop_polling skips rows with no stop_loss_price or take_profit_price."""
        from swingrl.scheduler.stop_polling import _check_stop_levels

        mock_config = MagicMock()
        mock_db = MagicMock()

        row = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "stop_loss_price": None,
            "take_profit_price": None,
        }

        # Should return early without making any HTTP calls
        with patch("httpx.get") as mock_get:
            _check_stop_levels(row, mock_config, mock_db)
            mock_get.assert_not_called()
