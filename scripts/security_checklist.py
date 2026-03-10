"""Automated security verification checklist for SwingRL production deployment.

Verifies non-root container, env_file permissions, resource limits,
and documents the 90-day staggered key rotation schedule.

Usage:
    python scripts/security_checklist.py
    python scripts/security_checklist.py --fix        # Auto-fix .env permissions
    python scripts/security_checklist.py --project-root /path/to/swingrl
"""

from __future__ import annotations

import argparse
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    """Result of a single security check."""

    name: str
    passed: bool
    detail: str


def check_non_root_user(dockerfile: Path) -> CheckResult:
    """Verify Dockerfile has a non-root USER directive.

    Args:
        dockerfile: Path to Dockerfile.

    Returns:
        CheckResult with pass/fail and detail.
    """
    if not dockerfile.exists():
        return CheckResult(
            name="non-root-user",
            passed=False,
            detail=f"Dockerfile not found: {dockerfile}",
        )

    content = dockerfile.read_text()
    # Find all USER directives (last one wins in Docker)
    user_lines = re.findall(r"^USER\s+(\S+)", content, re.MULTILINE)

    if not user_lines:
        return CheckResult(
            name="non-root-user",
            passed=False,
            detail="No USER directive found in Dockerfile (running as root)",
        )

    last_user = user_lines[-1]
    if last_user == "root":
        return CheckResult(
            name="non-root-user",
            passed=False,
            detail="Dockerfile USER is 'root' — must be non-root (e.g., trader)",
        )

    return CheckResult(
        name="non-root-user",
        passed=True,
        detail=f"Non-root USER '{last_user}' configured in Dockerfile",
    )


def check_env_file_reference(compose_file: Path) -> CheckResult:
    """Verify docker-compose.yml references env_file.

    Args:
        compose_file: Path to docker-compose.yml.

    Returns:
        CheckResult with pass/fail and detail.
    """
    if not compose_file.exists():
        return CheckResult(
            name="env-file-reference",
            passed=False,
            detail=f"docker-compose.yml not found: {compose_file}",
        )

    content = compose_file.read_text()
    if "env_file" in content:
        return CheckResult(
            name="env-file-reference",
            passed=True,
            detail="env_file referenced in docker-compose.yml",
        )

    return CheckResult(
        name="env-file-reference",
        passed=False,
        detail="No env_file directive found in docker-compose.yml — secrets may be exposed",
    )


def check_env_permissions(env_file: Path) -> CheckResult:
    """Verify .env file permissions are 600 (owner read/write only).

    Args:
        env_file: Path to .env file.

    Returns:
        CheckResult with pass/fail and detail.
    """
    if not env_file.exists():
        return CheckResult(
            name="env-permissions",
            passed=False,
            detail=f".env file not found: {env_file}",
        )

    mode = stat.S_IMODE(env_file.stat().st_mode)
    if mode == 0o600:
        return CheckResult(
            name="env-permissions",
            passed=True,
            detail=f".env permissions are {oct(mode)} (owner read/write only)",
        )

    return CheckResult(
        name="env-permissions",
        passed=False,
        detail=f".env permissions are {oct(mode)}, expected 0o600 — run with --fix to correct",
    )


def check_resource_limits(compose_file: Path) -> CheckResult:
    """Verify docker-compose.yml has resource limits (mem_limit, cpus).

    Args:
        compose_file: Path to docker-compose.yml.

    Returns:
        CheckResult with pass/fail and detail.
    """
    if not compose_file.exists():
        return CheckResult(
            name="resource-limits",
            passed=False,
            detail=f"docker-compose.yml not found: {compose_file}",
        )

    content = compose_file.read_text()
    has_mem = "mem_limit" in content
    has_cpus = "cpus" in content

    if has_mem and has_cpus:
        return CheckResult(
            name="resource-limits",
            passed=True,
            detail="mem_limit and cpus configured in docker-compose.yml",
        )

    missing = []
    if not has_mem:
        missing.append("mem_limit")
    if not has_cpus:
        missing.append("cpus")
    return CheckResult(
        name="resource-limits",
        passed=False,
        detail=f"Missing resource limits in docker-compose.yml: {', '.join(missing)}",
    )


def fix_env_permissions(env_file: Path) -> None:
    """Set .env file permissions to 600 (owner read/write only).

    Args:
        env_file: Path to .env file.
    """
    env_file.chmod(0o600)


def get_key_rotation_schedule() -> str:
    """Return the 90-day staggered key rotation schedule.

    Returns:
        Human-readable rotation schedule string.
    """
    return (
        "90-Day Staggered Key Rotation Schedule\n"
        "======================================\n"
        "\n"
        "Month 1: Alpaca API Key Rotation\n"
        "  1. Generate new API key in Alpaca dashboard\n"
        "  2. Update .env: ALPACA_API_KEY and ALPACA_API_SECRET\n"
        "  3. Restart container: docker compose restart swingrl\n"
        "  4. Verify: paper order submission succeeds\n"
        "  5. Delete old key in Alpaca dashboard\n"
        "\n"
        "Month 2: Binance.US API Key Rotation\n"
        "  1. Generate new API key in Binance.US dashboard\n"
        "     - Withdrawals: OFF\n"
        "     - IP allowlist: homelab IP only\n"
        "     - Permissions: Reading + Spot Trading only\n"
        "  2. Update .env: BINANCE_API_KEY and BINANCE_API_SECRET\n"
        "  3. Restart container: docker compose restart swingrl\n"
        "  4. Verify: price fetch succeeds\n"
        "  5. Delete old key in Binance.US dashboard\n"
        "\n"
        "Month 3: No rotation (buffer month)\n"
        "  - Review access logs for anomalies\n"
        "  - Verify IP allowlist is current\n"
    )


def run_checklist(project_root: Path, fix: bool = False) -> list[CheckResult]:
    """Run all security checks against the project.

    Args:
        project_root: Path to the SwingRL repository root.
        fix: If True, auto-fix .env permissions.

    Returns:
        List of CheckResult objects.
    """
    results: list[CheckResult] = []

    dockerfile = project_root / "Dockerfile"
    compose = project_root / "docker-compose.yml"
    env_file = project_root / ".env"

    results.append(check_non_root_user(dockerfile))
    results.append(check_env_file_reference(compose))

    if fix and env_file.exists():
        fix_env_permissions(env_file)

    results.append(check_env_permissions(env_file))
    results.append(check_resource_limits(compose))

    # Key rotation is documentation, always passes
    get_key_rotation_schedule()  # validate schedule is available
    results.append(
        CheckResult(
            name="key-rotation-schedule",
            passed=True,
            detail="90-day staggered rotation documented",
        )
    )

    return results


def main() -> None:
    """CLI entry point for security checklist."""
    parser = argparse.ArgumentParser(description="SwingRL security verification checklist")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to SwingRL repository root (default: cwd)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix .env permissions to 600",
    )
    args = parser.parse_args()

    print("SwingRL Security Checklist")
    print("=" * 40)
    print()

    results = run_checklist(args.project_root, fix=args.fix)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        icon = "[+]" if result.passed else "[-]"
        print(f"  {icon} {status}: {result.name}")
        print(f"      {result.detail}")
        if not result.passed:
            all_passed = False

    print()
    print(get_key_rotation_schedule())

    if all_passed:
        print("\nAll security checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome security checks FAILED. Review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
