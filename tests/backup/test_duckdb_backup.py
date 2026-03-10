"""Tests for DuckDB backup with table/row verification."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from swingrl.backup.duckdb_backup import backup_duckdb


@pytest.fixture()
def duckdb_env(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal DuckDB database with OHLCV tables for testing."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_path = db_dir / "market_data.ddb"

    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE ohlcv_daily ("
        "  symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, "
        "  close DOUBLE, volume BIGINT"
        ")"
    )
    conn.execute(
        "INSERT INTO ohlcv_daily VALUES ('SPY', '2026-03-01', 100, 101, 99, 100.5, 1000000)"
    )
    conn.execute(
        "CREATE TABLE ohlcv_4h ("
        "  symbol VARCHAR, timestamp TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, "
        "  close DOUBLE, volume BIGINT"
        ")"
    )
    conn.execute(
        "INSERT INTO ohlcv_4h VALUES "
        "('BTCUSDT', '2026-03-01 00:00:00', 50000, 50100, 49900, 50050, 500)"
    )
    conn.close()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    return {
        "db_path": db_path,
        "backup_dir": backup_dir,
        "tmp_path": tmp_path,
    }


def _make_config(duckdb_env: dict[str, Path]) -> MagicMock:
    """Build a mock SwingRLConfig for DuckDB backup."""
    config = MagicMock()
    config.system.duckdb_path = str(duckdb_env["db_path"])
    config.backup.backup_dir = str(duckdb_env["backup_dir"])
    config.backup.duckdb_rotate = False
    return config


class TestBackupDuckdb:
    """Tests for backup_duckdb function."""

    def test_creates_verified_copy(self, duckdb_env: dict[str, Path]) -> None:
        """PROD-01: DuckDB backup creates a copy with correct table count."""
        config = _make_config(duckdb_env)
        alerter = MagicMock()

        result = backup_duckdb(config, alerter)

        assert result is True
        duckdb_dir = duckdb_env["backup_dir"] / "duckdb"
        ddb_files = list(duckdb_dir.glob("market_data_*.ddb"))
        assert len(ddb_files) == 1

    def test_verifies_row_counts(self, duckdb_env: dict[str, Path]) -> None:
        """PROD-01: DuckDB backup verifies row counts for ohlcv tables."""
        config = _make_config(duckdb_env)
        alerter = MagicMock()

        backup_duckdb(config, alerter)

        duckdb_dir = duckdb_env["backup_dir"] / "duckdb"
        ddb_files = list(duckdb_dir.glob("market_data_*.ddb"))
        conn = duckdb.connect(str(ddb_files[0]), read_only=True)
        daily_count = conn.execute("SELECT COUNT(*) FROM ohlcv_daily").fetchone()[0]
        h4_count = conn.execute("SELECT COUNT(*) FROM ohlcv_4h").fetchone()[0]
        conn.close()
        assert daily_count > 0
        assert h4_count > 0

    def test_never_rotates(self, duckdb_env: dict[str, Path]) -> None:
        """PROD-01: DuckDB backups are never rotated (duckdb_rotate=False)."""
        from unittest.mock import patch

        config = _make_config(duckdb_env)
        alerter = MagicMock()

        # Create first backup with mocked timestamp
        from datetime import datetime

        ts1 = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)

        with patch("swingrl.backup.duckdb_backup.datetime") as mock_dt:
            mock_dt.now.return_value = ts1
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            backup_duckdb(config, alerter)

        with patch("swingrl.backup.duckdb_backup.datetime") as mock_dt:
            mock_dt.now.return_value = ts2
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            backup_duckdb(config, alerter)

        duckdb_dir = duckdb_env["backup_dir"] / "duckdb"
        ddb_files = list(duckdb_dir.glob("market_data_*.ddb"))
        assert len(ddb_files) == 2  # Both kept

    def test_alerts_on_success(self, duckdb_env: dict[str, Path]) -> None:
        """PROD-01: DuckDB backup calls alerter on success."""
        config = _make_config(duckdb_env)
        alerter = MagicMock()

        backup_duckdb(config, alerter)

        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "info"

    def test_alerts_critical_on_failure(self, tmp_path: Path) -> None:
        """PROD-01: DuckDB backup alerts critical on failure."""
        config = MagicMock()
        config.system.duckdb_path = str(tmp_path / "nonexistent.ddb")
        config.backup.backup_dir = str(tmp_path / "backups")
        config.backup.duckdb_rotate = False
        alerter = MagicMock()

        result = backup_duckdb(config, alerter)

        assert result is False
        alerter.send_alert.assert_called_once()
        call_args = alerter.send_alert.call_args
        assert call_args[0][0] == "critical"
