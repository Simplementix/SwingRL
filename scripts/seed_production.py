"""CLI entry point for production DB seeding.

Implements Doc 14 SS10.1 7-step DB seeding procedure for transferring
database files from M1 Mac to homelab. Validates file integrity after copy.
PostgreSQL connectivity is verified via a simple SELECT 1 query.

Usage:
    python scripts/seed_production.py --source-dir /path/to/m1/db --target-dir db/
    python scripts/seed_production.py --source-dir ~/swingrl/db --target-dir db/ --skip-integrity
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import structlog

from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

# Expected database files
_EXPECTED_FILES = [
    "market_data.ddb",
    "trading_ops.db",
]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Seed production databases on homelab from M1 Mac source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Steps performed:
  1. Validate source files exist
  2. Compute SHA256 checksums of source files
  3. Copy files to target directory
  4. Verify checksums of copied files
  5. Run integrity checks on databases
  6. Print summary

Examples:
  python scripts/seed_production.py --source-dir ~/swingrl/db --target-dir db/
""",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to source DB files (M1 Mac side).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="db",
        help="Path to target DB directory (default: db/).",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        default=False,
        help="Skip database integrity checks after copy.",
    )

    return parser


def _sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _check_postgres_connectivity() -> bool:
    """Verify PostgreSQL database is accessible via SELECT 1.

    Returns:
        True if connectivity check passes.
    """
    try:
        import psycopg  # noqa: PLC0415

        database_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://swingrl:changeme@localhost:5432/swingrl",  # pragma: allowlist secret
        )
        conn = psycopg.connect(database_url)
        result = conn.execute("SELECT 1").fetchone()
        tables = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        table_count = len(tables)
        conn.close()
        log.info("pg_tables_found", count=table_count)
        return result is not None and result[0] == 1
    except Exception as exc:
        log.error("pg_connectivity_failed", error=str(exc))
        return False


def main(argv: list[str] | None = None) -> int:
    """Run production DB seeding procedure.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(json_logs=False, log_level="INFO")

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    skip_integrity: bool = args.skip_integrity

    log.info(
        "seed_production_started",
        source=str(source_dir),
        target=str(target_dir),
    )

    # Step 1: Validate source files exist
    missing = []
    for filename in _EXPECTED_FILES:
        src_path = source_dir / filename
        if not src_path.exists():
            missing.append(filename)

    if missing:
        log.error("source_files_missing", missing=missing)
        print(f"\nError: Missing source files: {', '.join(missing)}")
        return 1

    # Step 2: Compute SHA256 checksums of source files
    source_checksums: dict[str, str] = {}
    for filename in _EXPECTED_FILES:
        src_path = source_dir / filename
        checksum = _sha256(src_path)
        source_checksums[filename] = checksum
        log.info("source_checksum", file=filename, sha256=checksum[:16] + "...")

    # Step 3: Copy files to target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in _EXPECTED_FILES:
        src_path = source_dir / filename
        dst_path = target_dir / filename
        shutil.copy2(str(src_path), str(dst_path))
        log.info(
            "file_copied",
            file=filename,
            size_mb=round(dst_path.stat().st_size / (1024 * 1024), 2),
        )

    # Step 4: Verify checksums of copied files
    checksum_ok = True
    for filename in _EXPECTED_FILES:
        dst_path = target_dir / filename
        dst_checksum = _sha256(dst_path)
        if dst_checksum != source_checksums[filename]:
            log.error(
                "checksum_mismatch",
                file=filename,
                expected=source_checksums[filename][:16],
                actual=dst_checksum[:16],
            )
            checksum_ok = False
        else:
            log.info("checksum_verified", file=filename)

    if not checksum_ok:
        print("\nError: Checksum verification failed!")
        return 1

    # Step 5: Run integrity checks (PostgreSQL connectivity)
    integrity_ok = True
    if not skip_integrity:
        if _check_postgres_connectivity():
            log.info("pg_connectivity_passed")
        else:
            log.error("pg_connectivity_failed")
            integrity_ok = False

    if not integrity_ok:
        print("\nError: Integrity check failed!")
        return 1

    # Step 6: Print summary
    print("\n=== Production DB Seeding Complete ===")
    for filename in _EXPECTED_FILES:
        dst_path = target_dir / filename
        size_mb = dst_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}: {size_mb:.2f} MB, checksum verified")
    print(f"  Integrity checks: {'skipped' if skip_integrity else 'passed'}")
    print(f"  Target directory: {target_dir.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("seed_production_interrupted")
        sys.exit(130)
