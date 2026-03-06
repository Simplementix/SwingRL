"""SwingRL data ingestion package.

Public API:
    AlpacaIngestor — Alpaca equity OHLCV data ingestor
    BaseIngestor — ABC for all data ingestors
    BinanceIngestor — Binance.US crypto 4H OHLCV ingestor
    DataValidator — 12-step validation checklist
    FREDIngestor — FRED macro data ingestor
    ParquetStore — Parquet read/upsert/write helpers
"""

from __future__ import annotations

from swingrl.data.alpaca import AlpacaIngestor
from swingrl.data.base import BaseIngestor
from swingrl.data.binance import BinanceIngestor
from swingrl.data.fred import FREDIngestor
from swingrl.data.parquet_store import ParquetStore
from swingrl.data.validation import DataValidator

__all__ = [
    "AlpacaIngestor",
    "BaseIngestor",
    "BinanceIngestor",
    "DataValidator",
    "FREDIngestor",
    "ParquetStore",
]
