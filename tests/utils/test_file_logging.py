"""Tests for file-based JSON logging with rotation.

Validates RotatingFileHandler writes JSON lines and rotates at threshold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import structlog

from swingrl.utils.logging import configure_logging


class TestFileLogging:
    """HARD-05: File-based JSON logging tests."""

    def test_log_file_created_with_json_lines(self, tmp_path: Path) -> None:
        """HARD-05: configure_logging with log_file creates a file with valid JSON lines."""
        log_file = tmp_path / "test.log"
        configure_logging(json_logs=False, log_level="INFO", log_file=log_file)

        log = structlog.get_logger("test_file_logging")
        log.info("test_event", key="value")

        # Flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert log_file.exists()
        lines = [line for line in log_file.read_text().strip().split("\n") if line]
        assert len(lines) >= 1

        # Each line must be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "event" in data

    def test_log_events_contain_expected_fields(self, tmp_path: Path) -> None:
        """HARD-05: JSON log events contain timestamp, level, and event fields."""
        log_file = tmp_path / "fields.log"
        configure_logging(json_logs=False, log_level="DEBUG", log_file=log_file)

        log = structlog.get_logger("test_fields")
        log.info("order_placed", symbol="SPY", quantity=10)

        for handler in logging.getLogger().handlers:
            handler.flush()

        lines = [line for line in log_file.read_text().strip().split("\n") if line]
        # Find our specific event
        found = False
        for line in lines:
            data = json.loads(line)
            if data.get("event") == "order_placed":
                assert "timestamp" in data
                assert data["symbol"] == "SPY"
                assert data["quantity"] == 10
                found = True
                break
        assert found, "Expected 'order_placed' event not found in log file"

    def test_rotating_file_handler_rotates(self, tmp_path: Path) -> None:
        """HARD-05: RotatingFileHandler rotates at max_bytes threshold."""
        log_file = tmp_path / "rotate.log"
        # Small max_bytes to force rotation quickly
        configure_logging(
            json_logs=False,
            log_level="DEBUG",
            log_file=log_file,
            max_bytes=500,
            backup_count=3,
        )

        log = structlog.get_logger("test_rotation")
        # Write enough to exceed 500 bytes and trigger rotation
        for i in range(50):
            log.info("rotation_test", iteration=i, padding="x" * 50)

        for handler in logging.getLogger().handlers:
            handler.flush()

        # Check that backup files were created
        backup_files = list(tmp_path.glob("rotate.log.*"))
        assert len(backup_files) >= 1, "Expected at least one rotated backup file"

    def test_file_handler_uses_json_regardless_of_json_logs_flag(self, tmp_path: Path) -> None:
        """HARD-05: File handler always writes JSON even when json_logs=False."""
        log_file = tmp_path / "always_json.log"
        configure_logging(json_logs=False, log_level="INFO", log_file=log_file)

        log = structlog.get_logger("test_always_json")
        log.info("json_check", data="test")

        for handler in logging.getLogger().handlers:
            handler.flush()

        lines = [line for line in log_file.read_text().strip().split("\n") if line]
        for line in lines:
            # Must be valid JSON (not console-formatted)
            json.loads(line)
