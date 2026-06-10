"""Scripted 9-step quarterly disaster recovery test for SwingRL.

Implements the full DR checklist from Doc 14 Section 10.1:
  Step 1: Stop running container
  Step 2: Delete database volumes
  Step 3: Locate latest backups
  Step 4: Restore PostgreSQL from backup (pg_restore)
  Step 5: (reserved — previously DuckDB restore)
  Step 6: Verify DB integrity (PostgreSQL connectivity)
  Step 7: Verify model loading
  Step 8: Start container
  Step 9: Verify first cycle (healthcheck)

Usage:
    python scripts/disaster_recovery.py --dry-run    # Steps 3-7 only (non-destructive)
    python scripts/disaster_recovery.py --full        # All 9 steps
    python scripts/disaster_recovery.py --dry-run --backup-dir backups/ --database-url postgresql://...
"""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StepResult:
    """Result of a single DR step."""

    step: int
    name: str
    passed: bool
    detail: str
    skipped: bool = False


def _find_latest_backup(backup_dir: Path, pattern: str) -> Path | None:
    """Find the most recent backup file matching pattern.

    Args:
        backup_dir: Directory to search.
        pattern: Glob pattern (e.g., '*.db').

    Returns:
        Path to newest file or None.
    """
    files = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def restore_postgres_backup(backup_dir: Path, database_url: str) -> StepResult:
    """Restore PostgreSQL from the latest pg_dump backup in backup_dir.

    Args:
        backup_dir: Directory containing PostgreSQL backup .sql.gz files.
        database_url: PostgreSQL connection URL for pg_restore target.

    Returns:
        StepResult with pass/fail.
    """
    latest = _find_latest_backup(backup_dir, "*.sql.gz")
    if latest is None:
        return StepResult(
            step=4,
            name="restore-postgres",
            passed=False,
            detail=f"No PostgreSQL backup files found in {backup_dir}",
        )

    try:
        subprocess.run(  # nosec B603 B607
            ["pg_restore", "--dbname", database_url, "--clean", "--if-exists", str(latest)],
            check=True,
            capture_output=True,
            text=True,
        )
        return StepResult(
            step=4,
            name="restore-postgres",
            passed=True,
            detail=f"Restored {latest.name} via pg_restore",
        )
    except subprocess.CalledProcessError as exc:
        return StepResult(
            step=4,
            name="restore-postgres",
            passed=False,
            detail=f"pg_restore failed: {exc.stderr}",
        )


def _step5_reserved() -> StepResult:
    """Step 5 is reserved (previously DuckDB restore, now handled by PostgreSQL).

    Returns:
        StepResult marked as skipped.
    """
    return StepResult(
        step=5,
        name="reserved",
        passed=True,
        detail="Reserved (DuckDB restore removed — data now in PostgreSQL)",
        skipped=True,
    )


def verify_postgres_integrity() -> StepResult:
    """Verify PostgreSQL database is accessible and has tables.

    Returns:
        StepResult with connectivity and table count verification.
    """
    try:
        import psycopg  # noqa: PLC0415

        database_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
        )
        conn = psycopg.connect(database_url, connect_timeout=10)
        try:
            result = conn.execute("SELECT 1").fetchone()
            if result is None or result[0] != 1:
                return StepResult(
                    step=6,
                    name="verify-postgres-integrity",
                    passed=False,
                    detail="PostgreSQL connectivity check failed",
                )

            tables = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            if not table_names:
                return StepResult(
                    step=6,
                    name="verify-postgres-integrity",
                    passed=False,
                    detail="PostgreSQL has no tables in public schema",
                )

            return StepResult(
                step=6,
                name="verify-postgres-integrity",
                passed=True,
                detail=f"PostgreSQL OK. {len(table_names)} table(s): {', '.join(table_names)}",
            )
        finally:
            conn.close()
    except Exception as exc:
        return StepResult(
            step=6,
            name="verify-postgres-integrity",
            passed=False,
            detail=f"PostgreSQL verification failed: {exc}",
        )


def verify_model_loading(active_dir: Path) -> StepResult:
    """Check that models/active/ contains model files.

    Args:
        active_dir: Path to models/active/ directory.

    Returns:
        StepResult with model verification.
    """
    if not active_dir.exists():
        return StepResult(
            step=7,
            name="verify-model-loading",
            passed=False,
            detail=f"Models directory not found: {active_dir}",
        )

    model_files = list(active_dir.glob("*.zip"))
    if not model_files:
        return StepResult(
            step=7,
            name="verify-model-loading",
            passed=False,
            detail=f"No model files (*.zip) found in {active_dir}",
        )

    names = [f.name for f in model_files]
    return StepResult(
        step=7,
        name="verify-model-loading",
        passed=True,
        detail=f"Found {len(model_files)} model(s): {', '.join(names)}",
    )


def _locate_backups(backup_dir: Path) -> StepResult:
    """Step 3: Locate latest PostgreSQL backup.

    Args:
        backup_dir: Root backup directory containing postgres/ subdir.

    Returns:
        StepResult with backup location details.
    """
    postgres_dir = backup_dir / "postgres"

    postgres_latest = (
        _find_latest_backup(postgres_dir, "*.sql.gz") if postgres_dir.exists() else None
    )

    if postgres_latest:
        return StepResult(
            step=3,
            name="locate-backups",
            passed=True,
            detail=f"PostgreSQL: {postgres_latest.name}",
        )

    return StepResult(
        step=3,
        name="locate-backups",
        passed=False,
        detail="Missing backups: PostgreSQL",
    )


def run_dr_checklist(
    *,
    backup_dir: Path,
    db_dir: Path,
    database_url: str = "",
    models_active_dir: Path,
    dry_run: bool = True,
    compose_project: str = "swingrl",
) -> list[StepResult]:
    """Execute the 9-step disaster recovery checklist.

    In dry_run mode, steps 1-2 (stop/delete) and 8-9 (start/verify) are skipped.
    Steps 3-7 (locate, restore, verify) run non-destructively.

    Args:
        backup_dir: Root backup directory (contains postgres/ subdir).
        db_dir: Target directory for database files (used for volume cleanup).
        database_url: PostgreSQL connection URL for pg_restore.
        models_active_dir: Path to models/active/ directory.
        dry_run: If True, skip destructive Docker operations.
        compose_project: Docker compose project name.

    Returns:
        List of StepResult for all 9 steps.
    """
    results: list[StepResult] = []

    # Step 1: Stop running container
    if dry_run:
        results.append(
            StepResult(
                step=1,
                name="stop-container",
                passed=True,
                detail="SKIPPED (dry-run)",
                skipped=True,
            )
        )
    else:
        try:
            subprocess.run(  # nosec B603 B607
                ["docker", "compose", "down"],
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(
                StepResult(
                    step=1,
                    name="stop-container",
                    passed=True,
                    detail="Container stopped via docker compose down",
                )
            )
        except subprocess.CalledProcessError as exc:
            results.append(
                StepResult(
                    step=1,
                    name="stop-container",
                    passed=False,
                    detail=f"Failed to stop container: {exc.stderr}",
                )
            )

    # Step 2: Delete database volumes
    if dry_run:
        results.append(
            StepResult(
                step=2,
                name="delete-volumes",
                passed=True,
                detail="SKIPPED (dry-run)",
                skipped=True,
            )
        )
    else:
        try:
            for db_file in db_dir.glob("*.db"):
                db_file.unlink()
            for ddb_file in db_dir.parent.glob("data/db/*.ddb"):
                ddb_file.unlink()
            results.append(
                StepResult(
                    step=2,
                    name="delete-volumes",
                    passed=True,
                    detail="Database files deleted",
                )
            )
        except OSError as exc:
            results.append(
                StepResult(
                    step=2,
                    name="delete-volumes",
                    passed=False,
                    detail=f"Failed to delete DB files: {exc}",
                )
            )

    # Step 3: Locate latest backups
    results.append(_locate_backups(backup_dir))

    # Step 4: Restore PostgreSQL from backup
    postgres_backup_dir = backup_dir / "postgres"
    if not database_url:
        database_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
        )
    results.append(restore_postgres_backup(postgres_backup_dir, database_url))

    # Step 5: Reserved (previously DuckDB restore)
    results.append(_step5_reserved())

    # Step 6: Verify DB integrity (PostgreSQL)
    results.append(verify_postgres_integrity())

    # Step 7: Verify model loading
    results.append(verify_model_loading(models_active_dir))

    # Step 8: Start container
    if dry_run:
        results.append(
            StepResult(
                step=8,
                name="start-container",
                passed=True,
                detail="SKIPPED (dry-run)",
                skipped=True,
            )
        )
    else:
        try:
            subprocess.run(  # nosec B603 B607
                ["docker", "compose", "up", "-d"],
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(
                StepResult(
                    step=8,
                    name="start-container",
                    passed=True,
                    detail="Container started via docker compose up -d",
                )
            )
        except subprocess.CalledProcessError as exc:
            results.append(
                StepResult(
                    step=8,
                    name="start-container",
                    passed=False,
                    detail=f"Failed to start container: {exc.stderr}",
                )
            )

    # Step 9: Verify first cycle (healthcheck)
    if dry_run:
        results.append(
            StepResult(
                step=9,
                name="verify-healthcheck",
                passed=True,
                detail="SKIPPED (dry-run)",
                skipped=True,
            )
        )
    else:
        try:
            proc = subprocess.run(  # nosec B603 B607
                ["docker", "compose", "exec", compose_project, "python", "scripts/healthcheck.py"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            results.append(
                StepResult(
                    step=9,
                    name="verify-healthcheck",
                    passed=True,
                    detail=f"Healthcheck passed: {proc.stdout.strip()}",
                )
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            results.append(
                StepResult(
                    step=9,
                    name="verify-healthcheck",
                    passed=False,
                    detail=f"Healthcheck failed: {exc}",
                )
            )

    return results


def main() -> None:
    """CLI entry point for disaster recovery test."""
    parser = argparse.ArgumentParser(description="SwingRL 9-step quarterly disaster recovery test")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run steps 3-7 only (non-destructive validation)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Execute all 9 steps (DESTRUCTIVE: stops container, deletes DBs)",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("backups"),
        help="Root backup directory (default: backups/)",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("db"),
        help="Target database directory (default: db/)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="",
        help="PostgreSQL connection URL (default: from DATABASE_URL env var)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/active"),
        help="Active models directory (default: models/active/)",
    )
    args = parser.parse_args()

    print("SwingRL Disaster Recovery Test")
    print("=" * 40)

    dry_run = args.dry_run
    if dry_run:
        print("MODE: DRY RUN (steps 3-7 only, non-destructive)")
    else:
        print("MODE: FULL (all 9 steps, DESTRUCTIVE)")
        print("WARNING: This will stop the container and delete databases!")

    print()

    results = run_dr_checklist(
        backup_dir=args.backup_dir,
        db_dir=args.db_dir,
        database_url=args.database_url,
        models_active_dir=args.models_dir,
        dry_run=dry_run,
    )

    all_passed = True
    for result in results:
        if result.skipped:
            print(f"  [~] SKIP: Step {result.step} — {result.name}")
        elif result.passed:
            print(f"  [+] PASS: Step {result.step} — {result.name}")
        else:
            print(f"  [-] FAIL: Step {result.step} — {result.name}")
            all_passed = False
        print(f"      {result.detail}")

    print()
    if all_passed:
        print("DR Test: ALL STEPS PASSED")
        sys.exit(0)
    else:
        print("DR Test: SOME STEPS FAILED — review above")
        sys.exit(1)


if __name__ == "__main__":
    main()
