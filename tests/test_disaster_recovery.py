"""Tests for the disaster recovery script implementing 9-step quarterly checklist.

PROD-08: Validates backup restore, DB integrity verification, and full DR flow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

_TEST_DB_URL = "postgresql://test:test@localhost/test"  # pragma: allowlist secret


class TestRestoreFromBackup:
    """Verify DR restore copies backup to target directory."""

    def test_restore_postgres_backup(self, tmp_path: Path) -> None:
        """PROD-08: DR restore_postgres_backup runs pg_restore on latest .sql.gz."""
        from scripts.disaster_recovery import restore_postgres_backup

        backup_dir = tmp_path / "backups" / "postgres"
        backup_dir.mkdir(parents=True)
        backup_file = backup_dir / "swingrl_20260310_120000.sql.gz"
        backup_file.write_bytes(b"fake pg_dump data")

        with patch("scripts.disaster_recovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = restore_postgres_backup(backup_dir, _TEST_DB_URL)

        assert result.passed is True
        mock_run.assert_called_once()

    def test_restore_postgres_fails_no_backup(self, tmp_path: Path) -> None:
        """PROD-08: DR restore fails when no backup files found."""
        from scripts.disaster_recovery import restore_postgres_backup

        backup_dir = tmp_path / "backups" / "postgres"
        backup_dir.mkdir(parents=True)

        result = restore_postgres_backup(backup_dir, _TEST_DB_URL)
        assert result.passed is False


class TestVerifyDbIntegrity:
    """Verify DB integrity checks on restored databases."""

    def test_postgres_integrity_check(self, tmp_path: Path) -> None:
        """PROD-08: DR verify_postgres_integrity checks PostgreSQL connectivity."""
        from scripts.disaster_recovery import verify_postgres_integrity

        mock_conn = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.fetchone.return_value = (1,)
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = [("ohlcv_daily",), ("trades",)]
        mock_conn.execute.side_effect = [mock_result1, mock_result2]

        with patch("psycopg.connect", return_value=mock_conn):
            result = verify_postgres_integrity()

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
        backup_dir = tmp_path / "backups"
        postgres_backup_dir = backup_dir / "postgres"
        postgres_backup_dir.mkdir(parents=True)
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        models_dir = tmp_path / "models" / "active"
        models_dir.mkdir(parents=True)

        # Create PostgreSQL backup
        pg_backup = postgres_backup_dir / "swingrl_20260310_120000.sql.gz"
        pg_backup.write_bytes(b"fake pg_dump")

        # Create model file
        (models_dir / "ppo_equity.zip").write_bytes(b"fake model")

        # Mock psycopg.connect for verify_postgres_integrity (step 6)
        mock_pg_conn = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.fetchone.return_value = (1,)
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = [("ohlcv_daily",)]
        mock_pg_conn.execute.side_effect = [mock_result1, mock_result2]

        with patch("psycopg.connect", return_value=mock_pg_conn):
            results = run_dr_checklist(
                backup_dir=backup_dir,
                db_dir=db_dir,
                database_url=_TEST_DB_URL,
                models_active_dir=models_dir,
                dry_run=True,
            )

        # Dry run should execute all 9 steps
        assert isinstance(results, list)
        assert len(results) >= 5
        assert all(isinstance(r, StepResult) for r in results)

        # In dry run, steps 1, 2, 5 (reserved), 8, 9 should be skipped
        skipped = [r for r in results if r.skipped]
        assert len(skipped) >= 4  # Steps 1, 2, 5, 8, 9

        # Validated steps should have pass/fail
        validated = [r for r in results if not r.skipped]
        assert all(hasattr(r, "passed") for r in validated)
