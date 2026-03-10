"""Tests for the disaster recovery script implementing 9-step quarterly checklist.

PROD-08: Validates backup restore, DB integrity verification, and full DR flow.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestRestoreFromBackup:
    """Verify DR restore copies backup to target directory."""

    def test_restore_sqlite_copies_backup(self, tmp_path: Path) -> None:
        """PROD-08: DR restore_from_backup copies SQLite backup to db/ directory."""
        from scripts.disaster_recovery import restore_sqlite_backup

        # Create backup source
        backup_dir = tmp_path / "backups" / "sqlite"
        backup_dir.mkdir(parents=True)
        backup_file = backup_dir / "trading_ops_20260310_120000.db"
        # Create a real SQLite DB as backup
        conn = sqlite3.connect(str(backup_file))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        # Create target db directory
        db_dir = tmp_path / "db"
        db_dir.mkdir()

        result = restore_sqlite_backup(backup_dir, db_dir / "trading_ops.db")
        assert result.passed is True
        assert (db_dir / "trading_ops.db").exists()

    def test_restore_sqlite_fails_no_backup(self, tmp_path: Path) -> None:
        """PROD-08: DR restore fails when no backup files found."""
        from scripts.disaster_recovery import restore_sqlite_backup

        backup_dir = tmp_path / "backups" / "sqlite"
        backup_dir.mkdir(parents=True)
        db_dir = tmp_path / "db"
        db_dir.mkdir()

        result = restore_sqlite_backup(backup_dir, db_dir / "trading_ops.db")
        assert result.passed is False


class TestVerifyDbIntegrity:
    """Verify DB integrity checks on restored databases."""

    def test_sqlite_integrity_check(self, tmp_path: Path) -> None:
        """PROD-08: DR verify_db_integrity runs PRAGMA integrity_check on SQLite."""
        from scripts.disaster_recovery import verify_sqlite_integrity

        db_path = tmp_path / "trading_ops.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE trades (id INTEGER, symbol TEXT)")
        conn.execute("INSERT INTO trades VALUES (1, 'SPY')")
        conn.commit()
        conn.close()

        result = verify_sqlite_integrity(db_path)
        assert result.passed is True
        assert "ok" in result.detail.lower()

    def test_duckdb_table_count(self, tmp_path: Path) -> None:
        """PROD-08: DR verify_db_integrity checks DuckDB table count."""
        import duckdb
        from scripts.disaster_recovery import verify_duckdb_integrity

        db_path = tmp_path / "market_data.ddb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE ohlcv_daily (ts TIMESTAMP, close DOUBLE)")
        conn.execute("INSERT INTO ohlcv_daily VALUES ('2026-01-01', 100.0)")
        conn.close()

        result = verify_duckdb_integrity(db_path)
        assert result.passed is True
        assert "table" in result.detail.lower()


class TestModelVerification:
    """Verify model loading check."""

    def test_verify_model_loading(self, tmp_path: Path) -> None:
        """PROD-08: DR verify_model_loading checks models/active/ has loadable models."""
        from scripts.disaster_recovery import verify_model_loading

        active_dir = tmp_path / "models" / "active"
        active_dir.mkdir(parents=True)
        # Create a dummy model file
        (active_dir / "ppo_equity.zip").write_bytes(b"fake model data")

        result = verify_model_loading(active_dir)
        assert result.passed is True

    def test_verify_model_loading_empty(self, tmp_path: Path) -> None:
        """PROD-08: DR verify_model_loading fails when no models present."""
        from scripts.disaster_recovery import verify_model_loading

        active_dir = tmp_path / "models" / "active"
        active_dir.mkdir(parents=True)

        result = verify_model_loading(active_dir)
        assert result.passed is False


class TestNineStepChecklist:
    """Verify the full 9-step DR checklist in dry-run mode."""

    @patch("scripts.disaster_recovery.subprocess")
    def test_dry_run_checklist(self, mock_subprocess: MagicMock, tmp_path: Path) -> None:
        """PROD-08: DR 9-step checklist runs all steps and reports pass/fail per step."""
        from scripts.disaster_recovery import StepResult, run_dr_checklist

        # Set up test directory structure
        backup_sqlite_dir = tmp_path / "backups" / "sqlite"
        backup_sqlite_dir.mkdir(parents=True)
        backup_duckdb_dir = tmp_path / "backups" / "duckdb"
        backup_duckdb_dir.mkdir(parents=True)
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        data_db_dir = tmp_path / "data" / "db"
        data_db_dir.mkdir(parents=True)
        models_dir = tmp_path / "models" / "active"
        models_dir.mkdir(parents=True)

        # Create SQLite backup
        sqlite_backup = backup_sqlite_dir / "trading_ops_20260310_120000.db"
        conn = sqlite3.connect(str(sqlite_backup))
        conn.execute("CREATE TABLE trades (id INTEGER)")
        conn.commit()
        conn.close()

        # Create DuckDB backup
        import duckdb

        duckdb_backup = backup_duckdb_dir / "market_data_20260310_120000.ddb"
        dconn = duckdb.connect(str(duckdb_backup))
        dconn.execute("CREATE TABLE ohlcv_daily (ts TIMESTAMP, close DOUBLE)")
        dconn.execute("INSERT INTO ohlcv_daily VALUES ('2026-01-01', 100.0)")
        dconn.close()

        # Create model file
        (models_dir / "ppo_equity.zip").write_bytes(b"fake model")

        results = run_dr_checklist(
            backup_dir=tmp_path / "backups",
            db_dir=db_dir,
            duckdb_target=data_db_dir / "market_data.ddb",
            sqlite_target=db_dir / "trading_ops.db",
            models_active_dir=models_dir,
            dry_run=True,
        )

        # Dry run should execute steps 3-7 (validation only)
        assert isinstance(results, list)
        assert len(results) >= 5  # Steps 3-7 at minimum
        assert all(isinstance(r, StepResult) for r in results)

        # In dry run, steps 1-2 and 8-9 should be skipped
        skipped = [r for r in results if r.skipped]
        assert len(skipped) >= 4  # Steps 1, 2, 8, 9

        # Steps 3-7 should have pass/fail
        validated = [r for r in results if not r.skipped]
        assert all(hasattr(r, "passed") for r in validated)
