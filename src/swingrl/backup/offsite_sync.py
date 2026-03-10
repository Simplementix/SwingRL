"""Monthly rsync via Tailscale to off-site NAS.

Runs rsync -avz to sync backup directory to a remote host over Tailscale.
Skips if offsite_host is not configured (empty string).

Usage:
    from swingrl.backup.offsite_sync import offsite_rsync
    success = offsite_rsync(config, alerter)
"""

from __future__ import annotations

import subprocess  # nosec B404 -- rsync invocation with fixed arguments
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


def offsite_rsync(config: SwingRLConfig, alerter: Alerter) -> bool:
    """Sync backup directory to off-site NAS via rsync over Tailscale.

    Skips if offsite_host is not configured (empty string).

    Args:
        config: Validated SwingRLConfig with backup.offsite_host and offsite_path.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True on success or skip, False on failure.
    """
    offsite_host = config.backup.offsite_host
    offsite_path = config.backup.offsite_path
    backup_dir = Path(config.backup.backup_dir)

    if not offsite_host:
        log.info("offsite_rsync_skipped", reason="offsite_host_not_configured")
        return True

    try:
        destination = f"{offsite_host}:{offsite_path}/"
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"{backup_dir}/",
            destination,
        ]

        log.info("offsite_rsync_starting", destination=destination)

        result = subprocess.run(  # noqa: S603  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"rsync failed with code {result.returncode}: {result.stderr}")

        log.info(
            "offsite_rsync_complete",
            destination=destination,
            stdout_lines=result.stdout.count("\n"),
        )

        alerter.send_alert(
            "info",
            "Off-site Sync Complete",
            f"Synced backups to {offsite_host}",
        )
        return True

    except Exception as exc:
        log.error("offsite_rsync_failed", error=str(exc), exc_info=True)
        alerter.send_alert(
            "critical",
            "Off-site Sync Failed",
            f"rsync failed: {exc}",
        )
        return False
