"""Shared fixtures for memory service tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_memory_client() -> MagicMock:
    """Standard mock MemoryClient for memory service tests."""
    client = MagicMock()
    client._base_url = "http://localhost:8889"
    client.ingest_training.return_value = True
    return client
