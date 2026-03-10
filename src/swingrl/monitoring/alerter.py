"""Discord webhook alerter with level-based routing and cooldown.

Provides alert infrastructure for ingestion failures, circuit breakers,
daily summaries, and stuck agents. Critical/warning alerts fire immediately;
info alerts buffer into a daily digest.

Supports two-webhook routing: critical/warning to alerts channel,
info/daily digest to daily channel. Falls back to single webhook_url
if specific URLs not provided.

Usage:
    from swingrl.monitoring.alerter import Alerter
    from swingrl.data.db import DatabaseManager

    alerter = Alerter(
        webhook_url="https://discord.com/api/webhooks/...",
        alerts_webhook_url="https://discord.com/api/webhooks/alerts/...",
        daily_webhook_url="https://discord.com/api/webhooks/daily/...",
        cooldown_minutes=30,
        db=DatabaseManager(config),
    )
    alerter.send_alert("critical", "Broker Down", "Alpaca API unreachable")
    alerter.send_daily_digest()
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import httpx
import structlog

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

AlertLevel = Literal["critical", "warning", "info"]

# Discord embed sidebar colors
_COLORS: dict[str, int] = {
    "critical": 0xFF0000,
    "warning": 0xFFA500,
    "info": 0x3498DB,
}


class Alerter:
    """Discord webhook alerter with level-based routing and cooldown.

    Critical and warning alerts are sent immediately via Discord webhook.
    Info alerts are buffered and flushed as a single daily digest embed.
    Cooldown prevents duplicate critical/warning alerts within a configurable window.
    Thread-safe for concurrent APScheduler calls.

    Supports two-webhook routing: alerts_webhook_url for critical/warning,
    daily_webhook_url for info/digest. Falls back to webhook_url if not set.
    """

    def __init__(
        self,
        webhook_url: str | None,
        cooldown_minutes: int = 30,
        consecutive_failures_before_alert: int = 3,
        db: DatabaseManager | None = None,
        alerts_webhook_url: str | None = None,
        daily_webhook_url: str | None = None,
    ) -> None:
        """Initialize alerter.

        Args:
            webhook_url: Discord webhook URL. Empty string or None disables sending.
            cooldown_minutes: Minimum minutes between identical critical/warning alerts.
            consecutive_failures_before_alert: Number of consecutive identical warnings
                before alerting. First N-1 occurrences are suppressed.
            db: Optional DatabaseManager for alert_log SQLite writes. If None,
                alert_log is skipped.
            alerts_webhook_url: Optional webhook URL for critical/warning alerts.
                Falls back to webhook_url if not provided.
            daily_webhook_url: Optional webhook URL for info alerts and daily digest.
                Falls back to webhook_url if not provided.
        """
        self._webhook_url: str = webhook_url or ""
        self._alerts_webhook_url: str = alerts_webhook_url or ""
        self._daily_webhook_url: str = daily_webhook_url or ""
        self._cooldown_minutes: int = cooldown_minutes
        self._consecutive_failures_before_alert: int = consecutive_failures_before_alert
        self._db: DatabaseManager | None = db
        self._info_buffer: deque[dict[str, str]] = deque()
        self._lock: threading.Lock = threading.Lock()
        self._last_alert_times: dict[str, datetime] = {}
        self._failure_counts: dict[str, int] = {}
        self._seen_hashes: set[str] = set()

    def _get_webhook_for_level(self, level: str) -> str:
        """Return the appropriate webhook URL for the given alert level.

        Args:
            level: Alert level string.

        Returns:
            Webhook URL string (may be empty if disabled).
        """
        if level in ("critical", "warning"):
            return self._alerts_webhook_url or self._webhook_url
        # info and daily digest
        return self._daily_webhook_url or self._webhook_url

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        environment: str | None = None,
    ) -> None:
        """Send an alert via Discord webhook or buffer for digest.

        Args:
            level: Alert severity -- "critical", "warning", or "info".
            title: Short alert title for the embed header.
            message: Alert message body.
            environment: Optional environment name (e.g., "Equity", "Crypto")
                to include in the embed footer.
        """
        msg_hash = self._compute_hash(title, message)

        if level == "info":
            self._buffer_info(title, message, msg_hash)
            return

        cooldown_key = f"{level}:{title}"

        # Check consecutive failures threshold for warnings
        if level == "warning":
            with self._lock:
                self._failure_counts[cooldown_key] = self._failure_counts.get(cooldown_key, 0) + 1
                count = self._failure_counts[cooldown_key]
            if count < self._consecutive_failures_before_alert:
                log.info(
                    "alert_suppressed_consecutive",
                    level=level,
                    title=title,
                    count=count,
                    threshold=self._consecutive_failures_before_alert,
                )
                return

        # Check cooldown
        with self._lock:
            last_sent = self._last_alert_times.get(cooldown_key)
        if last_sent is not None:
            elapsed = datetime.now(UTC) - last_sent
            if elapsed < timedelta(minutes=self._cooldown_minutes):
                log.info(
                    "alert_suppressed_cooldown",
                    level=level,
                    title=title,
                    cooldown_remaining_s=(
                        timedelta(minutes=self._cooldown_minutes) - elapsed
                    ).total_seconds(),
                )
                self._log_alert(level, title, msg_hash, sent=False)
                return

        # Build and send
        payload = self._build_embed(level, title, message, environment)
        webhook_url = self._get_webhook_for_level(level)
        success = self._post_webhook(payload, level, title, msg_hash, webhook_url=webhook_url)

        if success:
            with self._lock:
                self._last_alert_times[cooldown_key] = datetime.now(UTC)

    def send_embed(
        self,
        level: AlertLevel,
        embed: dict[str, list[dict[str, object]]],
        webhook_url: str | None = None,
    ) -> None:
        """Send a pre-built embed dict to the correct webhook.

        Args:
            level: Alert severity for routing to correct webhook.
            embed: Pre-built Discord webhook payload dict.
            webhook_url: Optional override webhook URL.
        """
        target_url = webhook_url or self._get_webhook_for_level(level)
        title = ""
        if embed.get("embeds"):
            title = str(embed["embeds"][0].get("title", ""))
        msg_hash = self._compute_hash(title, str(embed))
        self._post_webhook(embed, level, title, msg_hash, webhook_url=target_url)

    def send_daily_digest(self) -> None:
        """Flush buffered info alerts as a single Discord digest embed."""
        with self._lock:
            if not self._info_buffer:
                return
            items = list(self._info_buffer)
            self._info_buffer.clear()
            self._seen_hashes.clear()

        description = "\n".join(f"- **{item['title']}**: {item['message']}" for item in items)
        payload = self._build_embed("info", "Daily Digest", description)
        msg_hash = self._compute_hash("Daily Digest", description)
        webhook_url = self._get_webhook_for_level("info")
        self._post_webhook(payload, "info", "Daily Digest", msg_hash, webhook_url=webhook_url)

    def _buffer_info(self, title: str, message: str, msg_hash: str) -> None:
        """Add an info alert to the digest buffer if not duplicate.

        Args:
            title: Alert title.
            message: Alert message.
            msg_hash: Precomputed hash for dedup.
        """
        with self._lock:
            if msg_hash in self._seen_hashes:
                return
            self._seen_hashes.add(msg_hash)
            self._info_buffer.append({"title": title, "message": message})

    def _build_embed(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        environment: str | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Build a Discord webhook JSON payload with embed.

        Args:
            level: Alert severity for color and footer.
            title: Embed title.
            message: Embed description text.
            environment: Optional environment for footer prefix.

        Returns:
            Discord webhook payload dict.
        """
        footer_parts = ["SwingRL"]
        if environment:
            footer_parts.append(environment)
        footer_parts.append(level.upper())

        return {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": _COLORS[level],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "footer": {"text": " | ".join(footer_parts)},
                }
            ]
        }

    def _post_webhook(
        self,
        payload: dict[str, list[dict[str, object]]],
        level: str,
        title: str,
        message_hash: str,
        webhook_url: str | None = None,
    ) -> bool:
        """POST payload to Discord webhook and log to alert_log.

        Args:
            payload: Discord embed payload.
            level: Alert level string.
            title: Alert title for logging.
            message_hash: Hash for dedup tracking.
            webhook_url: Optional override URL. Falls back to _webhook_url.

        Returns:
            True on successful send, False on failure or disabled mode.
        """
        url = webhook_url or self._webhook_url
        if not url:
            log.info("alert_disabled", level=level, title=title, reason="no_webhook_url")
            return False

        try:
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            log.info("alert_sent", level=level, title=title)
            self._log_alert(level, title, message_hash, sent=True)
            return True
        except Exception:
            log.error("alert_send_failed", level=level, title=title, exc_info=True)
            self._log_alert(level, title, message_hash, sent=False)
            return False

    def _log_alert(self, level: str, title: str, message_hash: str, *, sent: bool) -> None:
        """Write a row to the alert_log SQLite table if DatabaseManager is available.

        Args:
            level: Alert level.
            title: Alert title.
            message_hash: Hash of title+message.
            sent: Whether the alert was successfully delivered.
        """
        if self._db is None:
            return

        try:
            with self._db.sqlite() as conn:
                conn.execute(
                    "INSERT INTO alert_log"
                    " (alert_id, timestamp, level, title, message_hash, sent)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        datetime.now(UTC).isoformat(),
                        level,
                        title,
                        message_hash,
                        1 if sent else 0,
                    ),
                )
        except Exception:
            log.error(
                "alert_log_write_failed",
                level=level,
                title=title,
                exc_info=True,
            )

    @staticmethod
    def _compute_hash(title: str, message: str) -> str:
        """Compute SHA-256 hash of title+message for dedup.

        Args:
            title: Alert title.
            message: Alert message body.

        Returns:
            Hex digest string.
        """
        return hashlib.sha256(f"{title}:{message}".encode()).hexdigest()
