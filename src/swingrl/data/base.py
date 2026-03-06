"""BaseIngestor ABC — shared contract for all data ingestors.

All concrete ingestors (Alpaca, Binance, FRED) inherit from BaseIngestor
and implement the fetch/validate/store protocol.
"""

from __future__ import annotations

import abc
from datetime import date
from pathlib import Path

import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class BaseIngestor(abc.ABC):
    """Abstract base for all data ingestors.

    Constructor takes a SwingRLConfig. Concrete subclasses implement
    fetch(), validate(), and store(). The run() method orchestrates
    the fetch -> validate -> store pipeline for a single symbol.

    Args:
        config: Validated SwingRLConfig instance.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        self._config = config
        self._data_dir = Path(config.paths.data_dir)

    @abc.abstractmethod
    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "SPY", "BTCUSDT").
            since: ISO timestamp for incremental mode; None for full fetch.

        Returns:
            DataFrame with OHLCV columns and UTC DatetimeIndex.
        """
        ...

    @abc.abstractmethod
    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate data and separate clean rows from quarantine.

        Args:
            df: Raw OHLCV DataFrame.
            symbol: Ticker symbol for logging context.

        Returns:
            Tuple of (clean_df, quarantine_df). Never raises for bad rows.
        """
        ...

    @abc.abstractmethod
    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert DataFrame into Parquet storage.

        Args:
            df: Validated OHLCV DataFrame.
            symbol: Ticker symbol used to determine file path.

        Returns:
            Path to the written Parquet file.
        """
        ...

    def run(self, symbol: str, since: str | None = None) -> None:
        """Orchestrate fetch -> validate -> store for one symbol.

        Args:
            symbol: Ticker symbol to ingest.
            since: ISO timestamp for incremental mode; None for full fetch.
        """
        log.info("ingestor_run_start", symbol=symbol, since=since)
        raw = self.fetch(symbol, since)
        if raw.empty:
            log.warning("ingestor_no_data", symbol=symbol)
            return
        clean, quarantine = self.validate(raw, symbol)
        if not quarantine.empty:
            self._store_quarantine(quarantine, symbol)
            log.warning(
                "rows_quarantined",
                symbol=symbol,
                count=len(quarantine),
            )
        if not clean.empty:
            self.store(clean, symbol)
        log.info("ingestor_run_complete", symbol=symbol, rows=len(clean))

    def _store_quarantine(self, df: pd.DataFrame, symbol: str) -> Path:
        """Write quarantined rows to data/quarantine/.

        Args:
            df: DataFrame of quarantined rows with a 'reason' column.
            symbol: Symbol for filename context.

        Returns:
            Path to the quarantine Parquet file.
        """
        q_dir = self._data_dir / "quarantine"
        q_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        path = q_dir / f"{symbol}_{today}.parquet"
        df.to_parquet(path, index=True)
        log.info("quarantine_written", path=str(path), rows=len(df))
        return path
