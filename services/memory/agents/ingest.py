"""IngestAgent: accepts raw text and stores it to the memory SQLite database.

Text is wrapped with XML delimiters before storage for prompt injection safety.
No LLM calls are made during ingestion — this is a pure write path.
"""

from __future__ import annotations

import structlog

from db import insert_memory

log = structlog.get_logger(__name__)

_XML_OPEN = "<memory>"
_XML_CLOSE = "</memory>"


class IngestAgent:
    """Stores memory text records to SQLite with XML delimiter wrapping."""

    def store(self, text: str, source: str) -> int:
        """Wrap text in XML delimiters and store to memories table.

        Args:
            text: Raw memory text to store.
            source: Source tag identifying the origin (e.g. 'training_run:historical').

        Returns:
            Row ID of the inserted memory record.
        """
        wrapped = f"{_XML_OPEN}{text}{_XML_CLOSE}"
        row_id = insert_memory(wrapped, source)
        log.info("memory_stored", source=source, row_id=row_id, length=len(text))
        return row_id
