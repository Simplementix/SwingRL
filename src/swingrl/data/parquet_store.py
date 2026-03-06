"""ParquetStore — Parquet read/upsert/write helpers.

Provides atomic upsert (read-merge-dedup-write) for single-symbol Parquet files.
All three ingestors delegate storage to this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class ParquetStore:
    """Read and upsert Parquet files with deduplication on index.

    Upsert semantics: if the file exists, read it, concatenate with new data,
    deduplicate on index (keep latest values), sort, and write back atomically.
    """

    def read(self, path: Path) -> pd.DataFrame:
        """Read a Parquet file, returning an empty DataFrame if missing.

        Args:
            path: Path to the Parquet file.

        Returns:
            DataFrame with data, or empty DataFrame if file does not exist.
        """
        if not path.exists():
            log.debug("parquet_file_missing", path=str(path))
            return pd.DataFrame()
        return pd.read_parquet(path)

    def upsert(self, path: Path, new_df: pd.DataFrame) -> None:
        """Upsert new_df into existing Parquet at path.

        If the file exists, reads existing data, merges with new_df,
        deduplicates on index (keeping the latest/new values), sorts,
        and writes back. Uses atomic write via temp file + rename.

        Args:
            path: Target Parquet file path.
            new_df: New data to upsert.
        """
        if new_df.empty:
            log.debug("parquet_upsert_empty", path=str(path))
            return

        if path.exists():
            existing = pd.read_parquet(path)
            # Concat with new data last so it wins on dedup
            combined = pd.concat([existing, new_df])
            # Keep last occurrence (new data) for duplicate indices
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_df.sort_index()

        # Atomic write: write to temp file, then replace
        tmp_path = path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=True, compression="snappy")
        tmp_path.replace(path)
        log.info("parquet_written", path=str(path), rows=len(combined))
