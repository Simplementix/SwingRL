# src/swingrl/data/options/store.py
"""Durable options-snapshot storage: Parquet-first, then Postgres (spec §8)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain

if TYPE_CHECKING:
    from swingrl.config.schema import OptionsCollectorConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

_HEADER_DT_FIELDS = ("snapshot_time_utc", "pulled_at_utc")
_HEADER_DATE_FIELDS = ("quote_date",)


class OptionsStore:
    """Writes each snapshot as an atomic Parquet + header sidecar (spec §8.1)."""

    def __init__(self, config: OptionsCollectorConfig, db: DatabaseManager | None = None) -> None:
        self._config = config
        self._db = db
        self._root = Path(config.output_dir)

    @staticmethod
    def symbol_to_dir(symbol: str) -> str:
        """Filesystem-safe directory name for a symbol (_SPX/$SPX -> SPX)."""
        return symbol.lstrip("$_")

    def parquet_path(self, symbol: str, quote_date: date, snapshot_label: str) -> Path:
        """Path to the contract Parquet for one (symbol, date, snapshot)."""
        return (
            self._root
            / self.symbol_to_dir(symbol)
            / f"{quote_date.isoformat()}_{snapshot_label}.parquet"
        )

    def header_path(self, symbol: str, quote_date: date, snapshot_label: str) -> Path:
        """Path to the header sidecar for one (symbol, date, snapshot)."""
        return self.parquet_path(symbol, quote_date, snapshot_label).with_suffix(".header.json")

    def snapshot_exists_parquet(self, symbol: str, quote_date: date, snapshot_label: str) -> bool:
        """True if the Parquet file for this snapshot already exists (skip unit)."""
        return self.parquet_path(symbol, quote_date, snapshot_label).exists()

    def write_snapshot(
        self, parsed: ParsedChain, symbol: str, quote_date: date, snapshot_label: str
    ) -> Path:
        """Atomically write the header sidecar and the contract Parquet (spec §8.1)."""
        pq_path = self.parquet_path(symbol, quote_date, snapshot_label)
        pq_path.parent.mkdir(parents=True, exist_ok=True)

        # Header sidecar (atomic).
        hdr_path = self.header_path(symbol, quote_date, snapshot_label)
        hdr_tmp = hdr_path.with_suffix(".json.tmp")
        hdr_tmp.write_text(json.dumps(parsed.header, default=str, indent=2))
        hdr_tmp.replace(hdr_path)

        # Contracts Parquet (atomic); raw_json -> JSON string for a stable columnar type.
        df = parsed.contracts.copy()
        df["raw_json"] = df["raw_json"].map(lambda d: json.dumps(d, default=str))
        pq_tmp = pq_path.with_suffix(".parquet.tmp")
        df.to_parquet(pq_tmp, index=False, compression="snappy")
        pq_tmp.replace(pq_path)
        log.info(
            "options_snapshot_written",
            symbol=symbol,
            quote_date=quote_date.isoformat(),
            snapshot_label=snapshot_label,
            rows=len(df),
        )
        return pq_path

    def read_snapshot(self, symbol: str, quote_date: date, snapshot_label: str) -> ParsedChain:
        """Read a snapshot back, restoring raw_json dicts and header datetimes."""
        df = pd.read_parquet(self.parquet_path(symbol, quote_date, snapshot_label))
        df["raw_json"] = df["raw_json"].map(json.loads)
        header = self._read_header(self.header_path(symbol, quote_date, snapshot_label))
        # Reindex to the canonical column order only when the full contract grain is
        # present (real snapshots from chain_parser.parse_chain). Narrower frames (e.g.
        # unit-test stubs with a handful of columns) keep whatever was actually stored —
        # selecting CONTRACT_COLUMNS unconditionally would KeyError on those.
        if all(col in df.columns for col in CONTRACT_COLUMNS):
            df = df[CONTRACT_COLUMNS]
        return ParsedChain(header=header, contracts=df)

    @staticmethod
    def _read_header(path: Path) -> dict[str, Any]:
        header: dict[str, Any] = json.loads(path.read_text())
        for field in _HEADER_DT_FIELDS:
            if header.get(field):
                header[field] = datetime.fromisoformat(header[field])
        for field in _HEADER_DATE_FIELDS:
            if header.get(field):
                header[field] = date.fromisoformat(header[field])
        return header
