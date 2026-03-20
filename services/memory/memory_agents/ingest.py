"""IngestAgent: accepts raw text and stores it to the memory SQLite database.

Text is wrapped with XML delimiters before storage for prompt injection safety.
No LLM calls are made during ingestion — this is a pure write path.
"""

from __future__ import annotations

import re

import structlog

from db import insert_memory, insert_memory_async

log = structlog.get_logger(__name__)

_XML_OPEN = "<memory>"
_XML_CLOSE = "</memory>"

# Regex to strip only XML-unsafe control characters (keeps & < > " ' as-is
# since they appear naturally in metric text like "sharpe=0.42 & mdd=-0.08"
# and the wrapping tags are our own delimiters, not arbitrary XML).
# Only characters truly invalid in XML 1.0 are removed: U+0000-U+0008,
# U+000B-U+000C, U+000E-U+001F (tab, newline, carriage return are fine).
_XML_UNSAFE_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_for_xml(text: str) -> str:
    """Remove XML-unsafe control characters without mangling content.

    Unlike html.escape(), this preserves ampersands, angle brackets, and
    quotes that appear naturally in metric text (e.g. ``sharpe=0.42``).
    Only strips C0 control characters that are invalid in XML 1.0.

    Args:
        text: Raw text to sanitize.

    Returns:
        Text safe for XML wrapping.
    """
    return _XML_UNSAFE_CHARS.sub("", text)


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
        safe_text = _sanitize_for_xml(text)
        wrapped = f"{_XML_OPEN}{safe_text}{_XML_CLOSE}"
        row_id = insert_memory(wrapped, source)
        log.info("memory_stored", source=source, row_id=row_id, length=len(text))
        return row_id

    async def store_async(self, text: str, source: str) -> int:
        """Async version of store — runs insert in the live thread pool.

        Args:
            text: Raw memory text to store.
            source: Source tag identifying the origin (e.g. 'training_run:historical').

        Returns:
            Row ID of the inserted memory record.
        """
        safe_text = _sanitize_for_xml(text)
        wrapped = f"{_XML_OPEN}{safe_text}{_XML_CLOSE}"
        row_id = await insert_memory_async(wrapped, source)
        log.info("memory_stored", source=source, row_id=row_id, length=len(text))
        return row_id
