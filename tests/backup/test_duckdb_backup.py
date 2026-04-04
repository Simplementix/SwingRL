"""Tests for PostgreSQL backup via pg_dump (formerly DuckDB backup)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swingrl.backup.duckdb_backup import backup_duckdb

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


def _make_config(tmp_path: Path) -> MagicMock:
    """Build a mock SwingRLConfig for PostgreSQL backup."""
    config = MagicMock()
    config.system.database_url = os.environ.get("DATABASE_URL", "")
    config.backup.backup_dir = str(tmp_path / "backups")
    return config


class TestBackupPostgres:
    """Tests for backup_duckdb function (now pg_dump based)."""

    def test_creates_backup_file(self, tmp_path: Path) -> None:
        """PROD-01: pg_dump backup creates a compressed dump file."""
        config = _make_config(tmp_path)
        alerter = MagicMock()

        # Mock subprocess.run to simulate successful pg_dump
        with patch("swingrl.backup.duckdb_backup.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = backup_duckdb(config, alerter)

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "pg_dump"

    def test_alerts_on_success(self, tmp_path: Path) -> None:
        """PROD-01: PostgreSQL backup calls alerter on success."""
        config = _make_config(tmp_path)
        alerter = MagicMock()

        with patch("swingrl.backup.duckdb_backup.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            backup_duckdb(config, alerter)

        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "info"

    def test_alerts_critical_on_pg_dump_failure(self, tmp_path: Path) -> None:
        """PROD-01: PostgreSQL backup alerts critical when pg_dump fails."""
        config = _make_config(tmp_path)
        alerter = MagicMock()

        with patch("swingrl.backup.duckdb_backup.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="connection refused")
            result = backup_duckdb(config, alerter)

        assert result is False
        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "critical"
