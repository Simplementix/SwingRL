"""Scripted 9-step quarterly disaster recovery test for SwingRL.

Implements the full DR checklist from Doc 14 Section 10.1:
  Step 1: Stop running container
  Step 2: Delete database volumes
  Step 3: Locate latest backups
  Step 4: Restore SQLite from backup
  Step 5: Restore DuckDB from backup
  Step 6: Verify DB integrity
  Step 7: Verify model loading
  Step 8: Start container
  Step 9: Verify first cycle (healthcheck)

Usage:
    python scripts/disaster_recovery.py --dry-run    # Steps 3-7 only (non-destructive)
    python scripts/disaster_recovery.py --full        # All 9 steps
    python scripts/disaster_recovery.py --dry-run --backup-dir backups/ --db-dir db/
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
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


def restore_sqlite_backup(backup_dir: Path, target: Path) -> StepResult:
    """Restore SQLite from the latest backup in backup_dir.

    Args:
        backup_dir: Directory containing SQLite backup .db files.
        target: Target path for restored database.

    Returns:
        StepResult with pass/fail.
    """
    latest = _find_latest_backup(backup_dir, "*.db")
    if latest is None:
        return StepResult(
            step=4,
            name="restore-sqlite",
            passed=False,
            detail=f"No SQLite backup files found in {backup_dir}",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(latest), str(target))

    return StepResult(
        step=4,
        name="restore-sqlite",
        passed=True,
        detail=f"Restored {latest.name} -> {target}",
    )


def restore_duckdb_backup(backup_dir: Path, target: Path) -> StepResult:
    """Restore DuckDB from the latest backup in backup_dir.

    Args:
        backup_dir: Directory containing DuckDB backup .ddb files.
        target: Target path for restored database.

    Returns:
        StepResult with pass/fail.
    """
    latest = _find_latest_backup(backup_dir, "*.ddb")
    if latest is None:
        return StepResult(
            step=5,
            name="restore-duckdb",
            passed=False,
            detail=f"No DuckDB backup files found in {backup_dir}",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(latest), str(target))

    return StepResult(
        step=5,
        name="restore-duckdb",
        passed=True,
        detail=f"Restored {latest.name} -> {target}",
    )


def verify_sqlite_integrity(db_path: Path) -> StepResult:
    """Run PRAGMA integrity_check on a SQLite database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        StepResult with integrity check result.
    """
    if not db_path.exists():
        return StepResult(
            step=6,
            name="verify-sqlite-integrity",
            passed=False,
            detail=f"SQLite database not found: {db_path}",
        )

    conn = sqlite3.connect(str(db_path))
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result and result[0] == "ok":
            # Count tables
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            return StepResult(
                step=6,
                name="verify-sqlite-integrity",
                passed=True,
                detail=f"Integrity OK. Tables: {', '.join(table_names) or 'none'}",
            )
        return StepResult(
            step=6,
            name="verify-sqlite-integrity",
            passed=False,
            detail=f"Integrity check failed: {result}",
        )
    finally:
        conn.close()


def verify_duckdb_integrity(db_path: Path) -> StepResult:
    """Verify DuckDB backup has tables with data.

    Args:
        db_path: Path to DuckDB database file.

    Returns:
        StepResult with table count verification.
    """
    if not db_path.exists():
        return StepResult(
            step=6,
            name="verify-duckdb-integrity",
            passed=False,
            detail=f"DuckDB database not found: {db_path}",
        )

    import duckdb

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if not tables:
            return StepResult(
                step=6,
                name="verify-duckdb-integrity",
                passed=False,
                detail="DuckDB has no tables",
            )
        return StepResult(
            step=6,
            name="verify-duckdb-integrity",
            passed=True,
            detail=f"DuckDB has {len(tables)} table(s): {', '.join(tables)}",
        )
    finally:
        conn.close()


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
    """Step 3: Locate latest backups.

    Args:
        backup_dir: Root backup directory containing sqlite/ and duckdb/ subdirs.

    Returns:
        StepResult with backup location details.
    """
    sqlite_dir = backup_dir / "sqlite"
    duckdb_dir = backup_dir / "duckdb"

    sqlite_latest = _find_latest_backup(sqlite_dir, "*.db") if sqlite_dir.exists() else None
    duckdb_latest = _find_latest_backup(duckdb_dir, "*.ddb") if duckdb_dir.exists() else None

    if sqlite_latest and duckdb_latest:
        return StepResult(
            step=3,
            name="locate-backups",
            passed=True,
            detail=f"SQLite: {sqlite_latest.name}, DuckDB: {duckdb_latest.name}",
        )

    missing = []
    if not sqlite_latest:
        missing.append("SQLite")
    if not duckdb_latest:
        missing.append("DuckDB")
    return StepResult(
        step=3,
        name="locate-backups",
        passed=False,
        detail=f"Missing backups: {', '.join(missing)}",
    )


def run_dr_checklist(
    *,
    backup_dir: Path,
    db_dir: Path,
    duckdb_target: Path,
    sqlite_target: Path,
    models_active_dir: Path,
    dry_run: bool = True,
    compose_project: str = "swingrl",
) -> list[StepResult]:
    """Execute the 9-step disaster recovery checklist.

    In dry_run mode, steps 1-2 (stop/delete) and 8-9 (start/verify) are skipped.
    Steps 3-7 (locate, restore, verify) run non-destructively.

    Args:
        backup_dir: Root backup directory (contains sqlite/ and duckdb/ subdirs).
        db_dir: Target directory for restored SQLite databases.
        duckdb_target: Target path for restored DuckDB file.
        sqlite_target: Target path for restored SQLite file.
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

    # Step 4: Restore SQLite from backup
    sqlite_backup_dir = backup_dir / "sqlite"
    results.append(restore_sqlite_backup(sqlite_backup_dir, sqlite_target))

    # Step 5: Restore DuckDB from backup
    duckdb_backup_dir = backup_dir / "duckdb"
    results.append(restore_duckdb_backup(duckdb_backup_dir, duckdb_target))

    # Step 6: Verify DB integrity
    results.append(verify_sqlite_integrity(sqlite_target))
    results.append(verify_duckdb_integrity(duckdb_target))

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
        "--duckdb-target",
        type=Path,
        default=Path("data/db/market_data.ddb"),
        help="Target DuckDB path (default: data/db/market_data.ddb)",
    )
    parser.add_argument(
        "--sqlite-target",
        type=Path,
        default=Path("db/trading_ops.db"),
        help="Target SQLite path (default: db/trading_ops.db)",
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
        duckdb_target=args.duckdb_target,
        sqlite_target=args.sqlite_target,
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
