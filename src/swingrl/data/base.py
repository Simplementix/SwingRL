"""BaseIngestor ABC — shared contract for all data ingestors.

All concrete ingestors (Alpaca, Binance, FRED) inherit from BaseIngestor
and implement the fetch/validate/store protocol. After Parquet write,
data is auto-synced to DuckDB and every run is logged to data_ingestion_log.
"""

from __future__ import annotations

import abc
import json
import time
import uuid
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Column mappings for DuckDB sync per table
_OHLCV_DAILY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_OHLCV_4H_COLUMNS = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
_MACRO_COLUMNS = ["date", "series_id", "value"]

_TABLE_COLUMN_MAP: dict[str, list[str]] = {
    "ohlcv_daily": _OHLCV_DAILY_COLUMNS,
    "ohlcv_4h": _OHLCV_4H_COLUMNS,
    "macro_features": _MACRO_COLUMNS,
}


class BaseIngestor(abc.ABC):
    """Abstract base for all data ingestors.

    Constructor takes a SwingRLConfig. Concrete subclasses implement
    fetch(), validate(), and store(). The run() method orchestrates
    the fetch -> validate -> store -> DuckDB sync -> log pipeline.

    Subclasses must set class attributes:
        _environment: str — "equity", "crypto", or "macro"
        _duckdb_table: str — target DuckDB table name

    Args:
        config: Validated SwingRLConfig instance.
    """

    _environment: str = ""
    _duckdb_table: str = ""
    _binance_weight_used: int | None = None

    def __init__(self, config: SwingRLConfig) -> None:
        self._config = config
        self._data_dir = Path(config.paths.data_dir)
        self._db: DatabaseManager | None = None

    def _get_db(self) -> DatabaseManager | None:
        """Lazy-initialize DatabaseManager from config.

        Returns None if DatabaseManager cannot be created (e.g., DB not set up).
        """
        if self._db is not None:
            return self._db
        try:
            from swingrl.data.db import DatabaseManager

            self._db = DatabaseManager(self._config)
            return self._db
        except Exception:  # noqa: BLE001
            log.warning("database_manager_unavailable")
            return None

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
        """Orchestrate fetch -> validate -> store -> DuckDB sync -> log.

        Every run creates a row in data_ingestion_log with timing, counts,
        and status. Data is synced to DuckDB after Parquet write.

        Args:
            symbol: Ticker symbol to ingest.
            since: ISO timestamp for incremental mode; None for full fetch.
        """
        run_id = str(uuid.uuid4())
        start = time.perf_counter()
        rows_inserted = 0
        errors_count = 0
        status = "success"

        try:
            log.info("ingestor_run_start", symbol=symbol, since=since)
            raw = self.fetch(symbol, since)
            if raw.empty:
                status = "no_data"
                log.warning("ingestor_no_data", symbol=symbol)
                return
            clean, quarantine = self.validate(raw, symbol)
            errors_count = len(quarantine)
            if not quarantine.empty:
                self._store_quarantine(quarantine, symbol)
                log.warning(
                    "rows_quarantined",
                    symbol=symbol,
                    count=len(quarantine),
                )
            if not clean.empty:
                self.store(clean, symbol)
                rows_inserted = self._sync_to_duckdb(clean, symbol)
            log.info("ingestor_run_complete", symbol=symbol, rows=len(clean))
        except Exception as e:
            status = "failed"
            log.error("ingestor_run_failed", symbol=symbol, error=str(e))
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._log_ingestion(run_id, symbol, status, rows_inserted, errors_count, duration_ms)

    def _sync_to_duckdb(self, clean: pd.DataFrame, symbol: str) -> int:
        """Sync validated data to DuckDB table.

        Uses INSERT OR IGNORE for idempotency (primary key dedup).

        Args:
            clean: Validated DataFrame to sync.
            symbol: Ticker symbol for column mapping.

        Returns:
            Number of rows inserted into DuckDB.
        """
        db = self._get_db()
        if db is None:
            log.warning("duckdb_sync_skipped", reason="no database manager")
            return 0

        table = self._duckdb_table
        columns = _TABLE_COLUMN_MAP.get(table)
        if columns is None:
            log.warning("duckdb_sync_skipped", reason="unknown table", table=table)
            return 0

        try:
            with db.duckdb() as cursor:
                # Build a DataFrame with the columns DuckDB expects
                sync_df = self._build_sync_df(clean, symbol, table)  # noqa: F841 — used by DuckDB replacement scan

                # Get row count before insert
                before = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608  # nosec B608

                # Use DuckDB's ability to query pandas DataFrames directly
                col_list = ", ".join(columns)
                cursor.execute(
                    f"INSERT OR IGNORE INTO {table} ({col_list}) "  # noqa: S608  # nosec B608
                    f"SELECT {col_list} FROM sync_df"
                )

                after = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608  # nosec B608
                inserted = after - before

            log.info("duckdb_sync_complete", symbol=symbol, table=table, rows=inserted)
            return int(inserted)
        except Exception:  # noqa: BLE001
            log.warning("duckdb_sync_failed", symbol=symbol, table=table)
            return 0

    def _build_sync_df(self, clean: pd.DataFrame, symbol: str, table: str) -> pd.DataFrame:
        """Build a DataFrame matching the DuckDB table schema.

        Args:
            clean: Source DataFrame with OHLCV data.
            symbol: Ticker symbol to add as column.
            table: Target table name for column mapping.

        Returns:
            DataFrame with columns matching the DuckDB table schema.
        """
        sync_df = clean.copy()

        if table == "ohlcv_daily":
            sync_df["symbol"] = symbol
            sync_df["date"] = pd.DatetimeIndex(sync_df.index).date
            return sync_df[_OHLCV_DAILY_COLUMNS]

        if table == "ohlcv_4h":
            sync_df["symbol"] = symbol
            sync_df["datetime"] = sync_df.index
            return sync_df[_OHLCV_4H_COLUMNS]

        if table == "macro_features":
            dt_idx = pd.DatetimeIndex(sync_df.index)
            sync_df["date"] = dt_idx.date if hasattr(dt_idx, "date") else sync_df.index
            sync_df["series_id"] = symbol
            return sync_df[_MACRO_COLUMNS]

        return sync_df

    def _log_ingestion(
        self,
        run_id: str,
        symbol: str,
        status: str,
        rows_inserted: int,
        errors_count: int,
        duration_ms: int,
    ) -> None:
        """Log ingestion run to data_ingestion_log table.

        Failure to log does not crash the ingestor.

        Args:
            run_id: UUID for this run.
            symbol: Ticker symbol.
            status: "success", "no_data", or "failed".
            rows_inserted: Number of rows inserted into DuckDB.
            errors_count: Number of quarantined rows.
            duration_ms: Wall-clock duration in milliseconds.
        """
        db = self._get_db()
        if db is None:
            return

        try:
            with db.duckdb() as cursor:
                cursor.execute(
                    "INSERT INTO data_ingestion_log "
                    "(run_id, environment, symbol, status, rows_inserted, "
                    "errors_count, duration_ms, binance_weight_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        run_id,
                        self._environment,
                        symbol,
                        status,
                        rows_inserted,
                        errors_count,
                        duration_ms,
                        self._binance_weight_used,
                    ],
                )
        except Exception:  # noqa: BLE001
            log.warning(
                "ingestion_log_failed",
                run_id=run_id,
                symbol=symbol,
                status=status,
            )

    def _store_quarantine(self, df: pd.DataFrame, symbol: str) -> Path:
        """Write quarantined rows to data/quarantine/ and DuckDB data_quarantine.

        Parquet write is retained for backward compatibility. DuckDB write
        serializes each row to JSON with failure reason and severity.

        Args:
            df: DataFrame of quarantined rows with a 'reason' column.
            symbol: Symbol for filename context.

        Returns:
            Path to the quarantine Parquet file.
        """
        # Parquet write (backward compatibility)
        q_dir = self._data_dir / "quarantine"
        q_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        path = q_dir / f"{symbol}_{today}.parquet"
        df.to_parquet(path, index=True)
        log.info("quarantine_written", path=str(path), rows=len(df))

        # DuckDB write
        db = self._get_db()
        if db is not None:
            try:
                with db.duckdb() as cursor:
                    for _, row in df.iterrows():
                        # Serialize the row data to JSON (exclude 'reason' column)
                        data_cols = [c for c in df.columns if c != "reason"]
                        row_data = {c: _serialize_value(row[c]) for c in data_cols}
                        raw_json = json.dumps(row_data)
                        reason = str(row.get("reason", "unknown"))

                        cursor.execute(
                            "INSERT INTO data_quarantine "
                            "(source, symbol, raw_data_json, failure_reason, severity) "
                            "VALUES (?, ?, ?, ?, ?)",
                            [self._environment, symbol, raw_json, reason, "warning"],
                        )
                log.info("quarantine_duckdb_written", symbol=symbol, rows=len(df))
            except Exception:  # noqa: BLE001
                log.warning("quarantine_duckdb_failed", symbol=symbol)

        return path


def _serialize_value(val: object) -> object:
    """Convert non-JSON-serializable values to strings.

    Args:
        val: Value to serialize.

    Returns:
        JSON-safe value.
    """
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    return str(val)


def sync_parquet_to_duckdb(
    parquet_path: Path,
    table: str,
    symbol: str,
    db: DatabaseManager,
) -> int:
    """Bulk-load a Parquet file into a DuckDB table.

    Uses DuckDB's read_parquet() for fast bulk loading with INSERT OR IGNORE
    for idempotency (primary key dedup).

    Args:
        parquet_path: Path to the Parquet file to load.
        table: Target DuckDB table name (e.g. "ohlcv_daily", "ohlcv_4h").
        symbol: Ticker symbol to add as column value.
        db: DatabaseManager instance with initialized schema.

    Returns:
        Number of rows loaded into DuckDB.
    """
    columns = _TABLE_COLUMN_MAP.get(table)
    if columns is None:
        log.warning("parquet_sync_skipped", reason="unknown table", table=table)
        return 0

    with db.duckdb() as cursor:
        before = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608  # nosec B608

        # Read parquet and add symbol column, then insert
        # Parquet stores pandas index as __index_level_0__ by default
        idx = "__index_level_0__"
        if table == "ohlcv_daily":
            cursor.execute(
                f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "  # noqa: S608  # nosec B608
                f'SELECT ?, "{idx}"::DATE, open, high, low, close, volume '
                f"FROM read_parquet('{parquet_path}')",
                [symbol],
            )
        elif table == "ohlcv_4h":
            cursor.execute(
                f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "  # noqa: S608  # nosec B608
                f'SELECT ?, "{idx}"::TIMESTAMP, open, high, low, close, volume '
                f"FROM read_parquet('{parquet_path}')",
                [symbol],
            )
        elif table == "macro_features":
            cursor.execute(
                f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "  # noqa: S608  # nosec B608
                f'SELECT "{idx}"::DATE, ?, value '
                f"FROM read_parquet('{parquet_path}')",
                [symbol],
            )

        after = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608  # nosec B608
        loaded = after - before

    log.info("parquet_to_duckdb_complete", path=str(parquet_path), table=table, rows=loaded)
    return int(loaded)
