"""SwingRL data ingestion package.

Public API:
    BaseIngestor — ABC for all data ingestors
    DataValidator — 12-step validation checklist
    ParquetStore — Parquet read/upsert/write helpers
"""

from __future__ import annotations

from swingrl.data.base import BaseIngestor
from swingrl.data.parquet_store import ParquetStore
from swingrl.data.validation import DataValidator

__all__ = ["BaseIngestor", "DataValidator", "ParquetStore"]
