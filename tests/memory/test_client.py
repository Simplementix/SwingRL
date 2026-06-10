"""Tests for MemoryClient API key header and config schema additions.

TRAIN-06, TRAIN-07: MemoryClient auth header, MemoryAgentConfig.api_key,
and sac_buffer_size configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMemoryClientAuth:
    """Tests for MemoryClient X-API-Key header behaviour."""

    def test_ingest_sends_api_key_header_when_configured(self) -> None:
        """TRAIN-06: MemoryClient with api_key sends X-API-Key header on ingest()."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://example.com", api_key="test-secret")

        captured_requests: list = []

        def fake_urlopen(req: object, timeout: float | None = None) -> MagicMock:
            captured_requests.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.ingest({"text": "hello", "source": "equity:historical"})

        assert result is True
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.get_header("X-api-key") == "test-secret"

    def test_consolidate_sends_api_key_header_when_configured(self) -> None:
        """TRAIN-06: MemoryClient with api_key sends X-API-Key header on consolidate()."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://example.com", api_key="another-secret")

        captured_requests: list = []

        def fake_urlopen(req: object, timeout: float | None = None) -> MagicMock:
            captured_requests.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.consolidate(timeout=5.0)

        assert result is True
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.get_header("X-api-key") == "another-secret"

    def test_ingest_no_api_key_header_when_empty(self) -> None:
        """TRAIN-06: MemoryClient with empty api_key sends NO X-API-Key header (backward compatible)."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://example.com", api_key="")

        captured_requests: list = []

        def fake_urlopen(req: object, timeout: float | None = None) -> MagicMock:
            captured_requests.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.ingest({"text": "hello", "source": "equity:historical"})

        assert result is True
        req = captured_requests[0]
        # urllib Request.get_header returns None for missing headers
        assert req.get_header("X-api-key") is None

    def test_consolidate_no_api_key_header_when_empty(self) -> None:
        """TRAIN-06: MemoryClient with empty api_key sends NO X-API-Key header on consolidate()."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://example.com", api_key="")

        captured_requests: list = []

        def fake_urlopen(req: object, timeout: float | None = None) -> MagicMock:
            captured_requests.append(req)
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.consolidate(timeout=5.0)

        assert result is True
        req = captured_requests[0]
        assert req.get_header("X-api-key") is None

    def test_default_api_key_is_empty_string(self) -> None:
        """TRAIN-06: MemoryClient default api_key is empty string (backward compatible)."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://example.com")
        assert client._api_key == ""

    def test_fail_open_still_works_with_api_key(self) -> None:
        """TRAIN-06: MemoryClient with api_key still returns False on connection error (fail-open)."""
        from swingrl.memory.client import MemoryClient

        client = MemoryClient(base_url="http://localhost:99999", api_key="some-key")
        result = client.ingest({"text": "test", "source": "equity:historical"})
        assert result is False


class TestMemoryAgentConfigApiKey:
    """Tests for MemoryAgentConfig.api_key field."""

    def test_memory_agent_config_has_api_key_field(self) -> None:
        """TRAIN-06: MemoryAgentConfig has api_key field with default empty string."""
        from swingrl.config.schema import MemoryAgentConfig

        cfg = MemoryAgentConfig()
        assert hasattr(cfg, "api_key")
        assert cfg.api_key == ""

    def test_memory_agent_config_api_key_settable(self) -> None:
        """TRAIN-06: MemoryAgentConfig api_key can be set to a non-empty value."""
        from swingrl.config.schema import MemoryAgentConfig

        cfg = MemoryAgentConfig(api_key="my-secret-key")  # pragma: allowlist secret
        assert cfg.api_key == "my-secret-key"  # pragma: allowlist secret


class TestSacBufferSizeConfig:
    """Tests for sac_buffer_size configuration field."""

    def test_training_config_has_sac_buffer_size(self) -> None:
        """TRAIN-06: SwingRLConfig has sac_buffer_size accessible with default 500_000."""
        # Load from nonexistent path to use defaults
        import tempfile
        from pathlib import Path

        from swingrl.config.schema import load_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp) / "nonexistent.yaml")
            assert hasattr(cfg, "training")
            assert hasattr(cfg.training, "sac_buffer_size")
            assert cfg.training.sac_buffer_size == 500_000

    def test_sac_buffer_size_configurable_via_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: sac_buffer_size can be set via YAML config."""
        from pathlib import Path

        from swingrl.config.schema import load_config

        yaml_path = Path(str(tmp_path)) / "test.yaml"  # type: ignore[arg-type]
        yaml_path.write_text("training:\n  sac_buffer_size: 200000\n")
        cfg = load_config(yaml_path)
        assert cfg.training.sac_buffer_size == 200_000
