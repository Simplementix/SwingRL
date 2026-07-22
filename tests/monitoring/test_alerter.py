"""Tests for Discord webhook alerter with level-based routing.

DATA-13: Alert infrastructure for ingestion failures, circuit breakers, daily summaries.
"""

from __future__ import annotations

import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.monitoring.alerter import Alerter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def webhook_url() -> str:
    """Test Discord webhook URL."""
    return "https://discord.com/api/webhooks/test/token"


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    """Provide a DatabaseManager with real PostgreSQL for alert_log tests."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available")
    config_yaml = textwrap.dedent(f"""\
        trading_mode: paper
        system:
          database_url: "{db_url}"
    """)
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    db = DatabaseManager(config)
    db.init_schema()
    # Clean stale data from prior test runs
    with db.connection() as conn:
        conn.execute("DELETE FROM alert_log")
    yield db
    DatabaseManager.reset()


@pytest.fixture
def alerter(webhook_url: str) -> Alerter:
    """Alerter without DatabaseManager (standalone mode)."""
    return Alerter(
        webhook_url=webhook_url, cooldown_minutes=30, consecutive_failures_before_alert=1
    )


@pytest.fixture
def alerter_with_db(webhook_url: str, db_manager: DatabaseManager) -> Alerter:
    """Alerter with DatabaseManager for alert_log writes."""
    return Alerter(
        webhook_url=webhook_url,
        cooldown_minutes=30,
        consecutive_failures_before_alert=3,
        db=db_manager,
    )


# ---------------------------------------------------------------------------
# Test: Critical alerts send immediately with red embed
# ---------------------------------------------------------------------------


class TestCriticalAlert:
    """DATA-13: Critical alerts post immediately with red Discord embed."""

    def test_critical_sends_red_embed(self, alerter: Alerter, mocker: Any) -> None:
        """Critical alert calls httpx.post with color=0xFF0000."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("critical", "Test Title", "Test message")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embed = payload["embeds"][0]
        assert embed["color"] == 0xFF0000
        assert embed["title"] == "Test Title"
        assert embed["description"] == "Test message"


# ---------------------------------------------------------------------------
# Test: Warning alerts send immediately with orange embed
# ---------------------------------------------------------------------------


class TestWarningAlert:
    """DATA-13: Warning alerts post immediately with orange Discord embed."""

    def test_warning_sends_orange_embed(self, alerter: Alerter, mocker: Any) -> None:
        """Warning alert calls httpx.post with color=0xFFA500."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("warning", "Warn Title", "Warn message")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embed = payload["embeds"][0]
        assert embed["color"] == 0xFFA500


# ---------------------------------------------------------------------------
# Test: Info alerts are buffered (not sent immediately)
# ---------------------------------------------------------------------------


class TestInfoAlert:
    """DATA-13: Info alerts buffer for daily digest, not sent immediately."""

    def test_info_does_not_send(self, alerter: Alerter, mocker: Any) -> None:
        """Info alert does NOT call httpx.post."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")

        alerter.send_alert("info", "Info Title", "Info message")

        mock_post.assert_not_called()

    def test_info_immediate_sends_blue_embed(self, webhook_url: str, mocker: Any) -> None:
        """OPT-ALERT-1: info_immediate=True posts info at once with color 0x3498DB."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()
        alerter = Alerter(
            webhook_url=webhook_url,
            cooldown_minutes=30,
            consecutive_failures_before_alert=1,
            info_immediate=True,
        )

        alerter.send_alert("info", "Info Title", "Info message")

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["embeds"][0]["color"] == 0x3498DB

    def test_info_immediate_does_not_buffer(self, webhook_url: str, mocker: Any) -> None:
        """OPT-ALERT-2: immediate info never lands in the digest buffer (no double-send)."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()
        alerter = Alerter(
            webhook_url=webhook_url,
            cooldown_minutes=30,
            consecutive_failures_before_alert=1,
            info_immediate=True,
        )
        alerter.send_alert("info", "Info Title", "Info message")
        mock_post.reset_mock()

        alerter.send_daily_digest()

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Daily digest
# ---------------------------------------------------------------------------


class TestDailyDigest:
    """DATA-13: send_daily_digest flushes info buffer as single embed."""

    def test_digest_sends_buffered_items(self, alerter: Alerter, mocker: Any) -> None:
        """Buffered info items posted as single embed, then buffer cleared."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("info", "Item 1", "First info")
        alerter.send_alert("info", "Item 2", "Second info")

        # Nothing sent yet
        mock_post.assert_not_called()

        alerter.send_daily_digest()

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        embed = payload["embeds"][0]
        assert embed["color"] == 0x3498DB
        assert "Item 1" in embed["description"]
        assert "Item 2" in embed["description"]

    def test_digest_empty_buffer_no_send(self, alerter: Alerter, mocker: Any) -> None:
        """Empty buffer does NOT call httpx.post."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")

        alerter.send_daily_digest()

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    """DATA-13: Alert cooldown prevents duplicate alerts within window."""

    def test_duplicate_suppressed_within_window(self, alerter: Alerter, mocker: Any) -> None:
        """Two identical critical alerts within 30 min — second suppressed."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("critical", "Same Title", "Same msg")
        alerter.send_alert("critical", "Same Title", "Same msg")

        assert mock_post.call_count == 1

    def test_different_titles_not_suppressed(self, alerter: Alerter, mocker: Any) -> None:
        """Different titles within cooldown window are independent."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("critical", "Title A", "msg")
        alerter.send_alert("critical", "Title B", "msg")

        assert mock_post.call_count == 2

    def test_alert_after_cooldown_expires(self, alerter: Alerter, mocker: Any) -> None:
        """Same alert after cooldown expires IS sent."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        now = datetime.now(UTC)
        past = now - timedelta(minutes=31)

        alerter.send_alert("critical", "Expire Test", "msg")

        # Manually set the cooldown time to the past
        alerter._last_alert_times["critical:Expire Test"] = past

        alerter.send_alert("critical", "Expire Test", "msg")

        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Test: Consecutive failures threshold
# ---------------------------------------------------------------------------


class TestConsecutiveFailures:
    """DATA-13: consecutive_failures_before_alert suppresses first N-1 warnings."""

    def test_first_n_minus_1_suppressed(self, mocker: Any) -> None:
        """First 2 failures suppressed, 3rd triggers alert (threshold=3)."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/test/token",
            cooldown_minutes=30,
            consecutive_failures_before_alert=3,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("warning", "Repeated Warn", "fail 1")
        a.send_alert("warning", "Repeated Warn", "fail 2")
        a.send_alert("warning", "Repeated Warn", "fail 3")

        # Only the 3rd should have triggered a post
        assert mock_post.call_count == 1

    def test_bypass_suppression_sends_on_first_occurrence(self, mocker: Any) -> None:
        """OPT-ALERT-3 (2026-07-16): bypass_suppression=True reaches the webhook on the
        FIRST identical warning, even under a threshold=3 gate. Same-day, un-backfillable
        data-loss warnings (e.g. an options-capture failure) must not wait for 3
        consecutive occurrences before alerting."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/test/token",
            cooldown_minutes=30,
            consecutive_failures_before_alert=3,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("warning", "Capture Failed", "SPY failed", bypass_suppression=True)

        mock_post.assert_called_once()

    def test_default_still_suppresses_first_occurrence(self, mocker: Any) -> None:
        """OPT-ALERT-4 (2026-07-16): default bypass_suppression=False leaves the
        consecutive-suppression gate unchanged for other callers (e.g. the backup job) --
        a single occurrence under threshold=3 is still suppressed."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/test/token",
            cooldown_minutes=30,
            consecutive_failures_before_alert=3,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("warning", "Backup Warn", "disk low")

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """DATA-13: Concurrent send_alert calls do not corrupt buffer."""

    def test_concurrent_info_alerts(self, alerter: Alerter, mocker: Any) -> None:
        """Multiple threads adding info alerts concurrently."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")

        def send(i: int) -> None:
            alerter.send_alert("info", f"Thread {i}", f"Message {i}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(send, range(20)))

        # Buffer should have 20 items, no corruption
        assert len(alerter._info_buffer) == 20
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: httpx failure handling
# ---------------------------------------------------------------------------


class TestWebhookFailure:
    """DATA-13: httpx.post failure is caught and logged, does not raise."""

    def test_connection_error_caught(self, alerter: Alerter, mocker: Any) -> None:
        """ConnectionError from httpx.post is caught; send_alert returns without raising."""
        import httpx

        mocker.patch(
            "swingrl.monitoring.alerter.httpx.post",
            side_effect=httpx.ConnectError("Connection refused"),
        )

        # Should not raise
        alerter.send_alert("critical", "Test", "msg")

    def test_http_status_error_does_not_leak_webhook_url_in_console_logs(self, mocker: Any) -> None:
        """SEC-15 (reviewer round): a failed webhook POST must not leak the URL via exc_info.

        httpx.HTTPStatusError's message embeds the full request URL, and Discord webhook
        URLs embed their secret token directly in that URL's path. _post_webhook's except
        clause used to log the exception with exc_info=True; in console-render mode
        (json_logs=False -- the currently-deployed homelab config), structlog's
        ConsoleRenderer expands the traceback text (including the exception's message),
        printing the token to stderr. Reproduced end-to-end: a real 401 response via
        httpx.MockTransport, driven through the real Alerter._post_webhook (not a
        synthetic exception injection) -- this is the most-likely-to-fire path in
        production (wrong/revoked/rate-limited token -> alerting about the failure
        prints the token that caused it).
        """
        import io
        import logging

        import httpx

        from swingrl.utils import configure_logging

        fake_token = "SEC15_REVIEWER_FIXTURE_TOKEN_zz9"  # noqa: S105 -- fixture, not real
        webhook_url = f"https://discord.com/api/webhooks/999999999/{fake_token}"

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Unauthorized"})

        def _fake_post(url: str, **kwargs: object) -> httpx.Response:
            with httpx.Client(transport=httpx.MockTransport(_handler)) as client:
                return client.post(url, **kwargs)  # type: ignore[arg-type]

        mocker.patch("swingrl.monitoring.alerter.httpx.post", side_effect=_fake_post)

        # Console-render mode -- matches the currently-deployed homelab config
        # (config/swingrl.yaml: json_logs: false).
        configure_logging(json_logs=False, log_level="INFO")
        buf = io.StringIO()
        root = logging.getLogger()
        for handler in root.handlers:
            handler.stream = buf

        test_alerter = Alerter(webhook_url=webhook_url, cooldown_minutes=30)
        test_alerter.send_alert("critical", "Test Alert", "test message")

        for handler in root.handlers:
            handler.flush()
        rendered = buf.getvalue()

        assert "alert_send_failed" in rendered, (
            "expected the alert_send_failed event to still be logged"
        )
        assert fake_token not in rendered, (
            f"webhook token leaked into rendered console log output: {rendered!r}"
        )


# ---------------------------------------------------------------------------
# Test: Embed payload structure
# ---------------------------------------------------------------------------


class TestEmbedPayload:
    """DATA-13: Embed payload includes timestamp, footer, correct color."""

    def test_embed_has_timestamp_and_footer(self, alerter: Alerter, mocker: Any) -> None:
        """Embed includes ISO-8601 UTC timestamp and footer with level."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter.send_alert("critical", "Payload Test", "payload msg")

        payload = mock_post.call_args[1]["json"]
        embed = payload["embeds"][0]

        # timestamp is ISO-8601
        assert "timestamp" in embed
        datetime.fromisoformat(embed["timestamp"])  # Should not raise

        # footer
        assert "footer" in embed
        assert "CRITICAL" in embed["footer"]["text"]
        assert "SwingRL" in embed["footer"]["text"]


# ---------------------------------------------------------------------------
# Test: Message hash dedup in info buffer
# ---------------------------------------------------------------------------


class TestInfoDedup:
    """DATA-13: Same title+message within digest window not duplicated in buffer."""

    def test_duplicate_info_not_added(self, alerter: Alerter, mocker: Any) -> None:
        """Same title+message added twice, buffer has only one."""
        mocker.patch("swingrl.monitoring.alerter.httpx.post")

        alerter.send_alert("info", "Dup Title", "Dup message")
        alerter.send_alert("info", "Dup Title", "Dup message")

        assert len(alerter._info_buffer) == 1


# ---------------------------------------------------------------------------
# Test: Dev/test mode (empty webhook URL)
# ---------------------------------------------------------------------------


class TestDevMode:
    """DATA-13: Empty webhook_url disables sending."""

    def test_empty_url_no_send(self, mocker: Any) -> None:
        """Alerter with empty URL does not call httpx.post."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        a = Alerter(webhook_url="", cooldown_minutes=30)

        a.send_alert("critical", "Test", "msg")

        mock_post.assert_not_called()

    def test_none_url_no_send(self, mocker: Any) -> None:
        """Alerter with None URL does not call httpx.post."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        a = Alerter(webhook_url=None, cooldown_minutes=30)  # type: ignore[arg-type]

        a.send_alert("critical", "Test", "msg")

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Test: alert_log SQLite writes
# ---------------------------------------------------------------------------


class TestAlertLog:
    """DATA-13: _post_webhook writes to alert_log table."""

    def test_successful_send_writes_alert_log(
        self, alerter_with_db: Alerter, db_manager: DatabaseManager, mocker: Any
    ) -> None:
        """Successful alert writes row with sent=1."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter_with_db.send_alert("critical", "Log Test", "log msg")

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT * FROM alert_log").fetchall()
            assert len(rows) == 1
            row = dict(rows[0])
            assert row["level"] == "critical"
            assert row["title"] == "Log Test"
            assert row["sent"] == 1

    def test_no_db_still_sends(self, alerter: Alerter, mocker: Any) -> None:
        """db=None mode sends but skips alert_log write."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        # Should not raise
        alerter.send_alert("critical", "No DB", "msg")
        mock_post.assert_called_once()

    def test_suppressed_alert_writes_sent_zero(
        self, alerter_with_db: Alerter, db_manager: DatabaseManager, mocker: Any
    ) -> None:
        """Suppressed alert (cooldown) writes alert_log with sent=0."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        alerter_with_db.send_alert("critical", "Cooldown Test", "msg")
        alerter_with_db.send_alert("critical", "Cooldown Test", "msg")

        with db_manager.connection() as conn:
            rows = conn.execute("SELECT * FROM alert_log ORDER BY timestamp").fetchall()
            assert len(rows) == 2
            sent_values = [dict(r)["sent"] for r in rows]
            assert 1 in sent_values
            assert 0 in sent_values


# ---------------------------------------------------------------------------
# Test: Two-webhook routing
# ---------------------------------------------------------------------------


class TestTwoWebhookRouting:
    """PAPER-13: Alerter routes critical/warning to alerts webhook, info/daily to daily webhook."""

    def test_critical_routes_to_alerts_webhook(self, mocker: Any) -> None:
        """Critical alert posts to alerts_webhook_url, not daily_webhook_url."""
        a = Alerter(
            webhook_url=None,
            alerts_webhook_url="https://discord.com/api/webhooks/alerts/token",
            daily_webhook_url="https://discord.com/api/webhooks/daily/token",
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("critical", "Test", "msg")

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        assert url == "https://discord.com/api/webhooks/alerts/token"

    def test_warning_routes_to_alerts_webhook(self, mocker: Any) -> None:
        """Warning alert posts to alerts_webhook_url."""
        a = Alerter(
            webhook_url=None,
            alerts_webhook_url="https://discord.com/api/webhooks/alerts/token",
            daily_webhook_url="https://discord.com/api/webhooks/daily/token",
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("warning", "Test", "msg")

        url = mock_post.call_args[0][0]
        assert url == "https://discord.com/api/webhooks/alerts/token"

    def test_daily_digest_routes_to_daily_webhook(self, mocker: Any) -> None:
        """Daily digest posts to daily_webhook_url."""
        a = Alerter(
            webhook_url=None,
            alerts_webhook_url="https://discord.com/api/webhooks/alerts/token",
            daily_webhook_url="https://discord.com/api/webhooks/daily/token",
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("info", "Item", "msg")
        a.send_daily_digest()

        url = mock_post.call_args[0][0]
        assert url == "https://discord.com/api/webhooks/daily/token"

    def test_fallback_to_single_webhook(self, mocker: Any) -> None:
        """Without specific URLs, falls back to webhook_url for all levels."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/single/token",
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("critical", "Test", "msg")

        url = mock_post.call_args[0][0]
        assert url == "https://discord.com/api/webhooks/single/token"


# ---------------------------------------------------------------------------
# Test: send_embed method
# ---------------------------------------------------------------------------


class TestSendEmbed:
    """PAPER-13: send_embed sends pre-built embed dict to correct webhook."""

    def test_send_embed_posts_payload(self, mocker: Any) -> None:
        """send_embed sends the embed dict directly as embeds payload."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/test/token",
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        embed = {"embeds": [{"title": "Test Embed", "color": 0xFF0000}]}
        a.send_embed("critical", embed)

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["embeds"][0]["title"] == "Test Embed"

    def test_send_embed_routes_by_level(self, mocker: Any) -> None:
        """send_embed routes to correct webhook based on level."""
        a = Alerter(
            webhook_url=None,
            alerts_webhook_url="https://discord.com/api/webhooks/alerts/token",
            daily_webhook_url="https://discord.com/api/webhooks/daily/token",
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        embed = {"embeds": [{"title": "Info Embed", "color": 0x3498DB}]}
        a.send_embed("info", embed)

        url = mock_post.call_args[0][0]
        assert url == "https://discord.com/api/webhooks/daily/token"


# ---------------------------------------------------------------------------
# Test: Category styling (STYLE-D15)
# ---------------------------------------------------------------------------


class TestCategoryStyling:
    """STYLE-D15: category selects sidebar color + emoji title prefix; routing unchanged."""

    def test_alert_category_sets_color_and_emoji(self, webhook_url: str, mocker: Any) -> None:
        """STYLE-D15: category selects sidebar color + emoji title prefix; ingest=blue 📥,
        cycle=purple 🔄; omitted category keeps today's level-default behavior."""
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()
        # info_immediate so info posts at once (mirrors the collector's candle-ingest path).
        a = Alerter(
            webhook_url=webhook_url,
            cooldown_minutes=30,
            consecutive_failures_before_alert=1,
            info_immediate=True,
        )

        a.send_alert(
            level="info", title="Equity candles ingested", message="rows=8", category="ingest"
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == 0x3498DB
        assert embed["title"].startswith("📥")

        a.send_alert(
            level="info", title="Cycle orders submitted — crypto", message="…", category="cycle"
        )
        embed = mock_post.call_args[1]["json"]["embeds"][0]
        assert embed["color"] == 0x9B59B6 and embed["title"].startswith("🔄")

        a.send_alert(level="info", title="plain", message="no category")
        assert mock_post.call_args[1]["json"]["embeds"][0]["color"] == 0x3498DB  # legacy path


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """PAPER-13: Existing single-webhook usage unchanged after extension."""

    def test_existing_constructor_still_works(self, mocker: Any) -> None:
        """Original constructor signature still works."""
        a = Alerter(
            webhook_url="https://discord.com/api/webhooks/test/token",
            cooldown_minutes=30,
            consecutive_failures_before_alert=1,
        )
        mock_post = mocker.patch("swingrl.monitoring.alerter.httpx.post")
        mock_post.return_value = MagicMock(status_code=204)
        mock_post.return_value.raise_for_status = MagicMock()

        a.send_alert("critical", "Compat", "msg")
        mock_post.assert_called_once()
