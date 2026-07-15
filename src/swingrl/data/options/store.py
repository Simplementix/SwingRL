# src/swingrl/data/options/store.py
"""Durable options-snapshot storage: Parquet-first, then Postgres (spec §8)."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog
from psycopg.types.json import Jsonb

from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain
from swingrl.data.options.schema import ensure_monthly_partition, ensure_options_schema

if TYPE_CHECKING:
    import psycopg

    from swingrl.config.schema import OptionsCollectorConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

_HEADER_DT_FIELDS = ("snapshot_time_utc", "pulled_at_utc")
_HEADER_DATE_FIELDS = ("quote_date",)

DB_SNAPSHOT_COLUMNS: list[str] = [
    "underlying_symbol",
    "quote_date",
    "snapshot_label",
    "snapshot_time_utc",
    "pulled_at_utc",
    "underlying_price",
    "is_delayed",
    "is_early_close",
    "interest_rate",
    "dividend_yield",
    "underlying_volatility",
    "number_of_contracts",
    "status",
    "source",
    "schema_version",
    "raw_header",
]
DB_CHAIN_COLUMNS: list[str] = [
    "underlying_symbol",
    "quote_date",
    "snapshot_label",
    "contract_symbol",
    "option_root",
    "expiration",
    "dte",
    "strike",
    "option_right",
    "expiration_type",
    "settlement_type",
    "exercise_type",
    "multiplier",
    "in_the_money",
    "bid",
    "ask",
    "last",
    "mark",
    "bid_size",
    "ask_size",
    "last_size",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "net_change",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
    "theoretical_value",
    "time_value",
    "intrinsic_value",
    "extrinsic_value",
    "underlying_price",
    "is_delayed",
    "quote_time_utc",
    "trade_time_utc",
    "pulled_at_utc",
    "source",
    "schema_version",
    "raw_json",
]
_DATE_DB_COLS = {"quote_date", "expiration"}


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
        pq_path = self.parquet_path(symbol, quote_date, snapshot_label)
        df = pd.read_parquet(pq_path)
        df["raw_json"] = df["raw_json"].map(json.loads)
        header = self._read_header(self.header_path(symbol, quote_date, snapshot_label))
        # Reindex to the canonical column order only when the full contract grain is
        # present (real snapshots from chain_parser.parse_chain). Narrower frames (e.g.
        # unit-test stubs with a handful of columns) keep whatever was actually stored —
        # selecting CONTRACT_COLUMNS unconditionally would KeyError on those.
        if all(col in df.columns for col in CONTRACT_COLUMNS):
            df = df[CONTRACT_COLUMNS]
        else:
            log.warning(
                "options_snapshot_partial_columns",
                path=str(pq_path),
                missing=sorted(set(CONTRACT_COLUMNS) - set(df.columns)),
            )
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

    def snapshot_exists_db(
        self,
        conn: psycopg.Connection[Any],
        symbol: str,
        quote_date: date,
        snapshot_label: str,
    ) -> bool:
        """True if the parent snapshot row already exists in Postgres."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM options_snapshots "
                "WHERE underlying_symbol=%s AND quote_date=%s AND snapshot_label=%s",
                (symbol, quote_date, snapshot_label),
            )
            return cur.fetchone() is not None

    def sync_to_postgres(self, parsed: ParsedChain) -> None:
        """Upsert parent + child rows for one snapshot (idempotent, JSONB; spec §8.2)."""
        if self._db is None:
            return
        with self._db.connection() as conn:
            self._write_db(conn, parsed)

    def _write_db(self, conn: psycopg.Connection[Any], parsed: ParsedChain) -> None:
        hdr = parsed.header
        ensure_options_schema(conn)
        ensure_monthly_partition(conn, hdr["quote_date"])
        with conn.cursor() as cur:
            parent = tuple(self._db_value(k, hdr.get(k)) for k in DB_SNAPSHOT_COLUMNS)
            placeholders = ", ".join(["%s"] * len(DB_SNAPSHOT_COLUMNS))
            cur.execute(
                # Column list is the module-level DB_SNAPSHOT_COLUMNS constant, never
                # user input — safe to interpolate (same convention as schema.py:85).
                f"INSERT INTO options_snapshots ({', '.join(DB_SNAPSHOT_COLUMNS)}) "  # noqa: S608  # nosec B608
                f"VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                parent,
            )
            df = parsed.contracts
            store_raw = self._config.postgres_store_raw_json
            records = [
                tuple(
                    None
                    if (col == "raw_json" and not store_raw)
                    else self._db_value(col, row.get(col))
                    for col in DB_CHAIN_COLUMNS
                )
                for row in df.to_dict("records")
            ]
            child_placeholders = ", ".join(["%s"] * len(DB_CHAIN_COLUMNS))
            cur.executemany(
                # Column list is the module-level DB_CHAIN_COLUMNS constant, never
                # user input — safe to interpolate (same convention as schema.py:85).
                f"INSERT INTO options_chains ({', '.join(DB_CHAIN_COLUMNS)}) "  # noqa: S608  # nosec B608
                f"VALUES ({child_placeholders}) ON CONFLICT DO NOTHING",
                records,
            )
        log.info(
            "options_snapshot_synced",
            underlying_symbol=hdr["underlying_symbol"],
            quote_date=hdr["quote_date"].isoformat(),
            snapshot_label=hdr["snapshot_label"],
            rows=len(parsed.contracts),
        )

    @staticmethod
    def _db_value(column: str, value: Any) -> Any:
        """Adapt a Python/pandas value for psycopg (JSONB, NaN->NULL, date coercion)."""
        if column in ("raw_json", "raw_header"):
            return Jsonb(value if isinstance(value, dict) else json.loads(value))
        if isinstance(value, float) and math.isnan(value):
            return None
        if value is None:
            return None
        if column in _DATE_DB_COLS and isinstance(value, datetime):
            return value.date()
        if column in _DATE_DB_COLS and isinstance(value, pd.Timestamp):
            return value.date()
        return value

    def reconcile(self) -> int:
        """Load any Parquet snapshot with no parent DB row; self-heals outages (spec §8.2)."""
        if self._db is None:
            return 0
        loaded = 0
        with self._db.connection() as conn:
            for pq in sorted(self._root.glob("*/*.parquet")):
                hdr = self._read_header(pq.with_suffix(".header.json"))
                sym, qdate, label = (
                    hdr["underlying_symbol"],
                    hdr["quote_date"],
                    hdr["snapshot_label"],
                )
                if self.snapshot_exists_db(conn, sym, qdate, label):
                    continue
                self._write_db(conn, self.read_snapshot(sym, qdate, label))
                loaded += 1
        log.info("options_reconcile_done", loaded=loaded)
        return loaded
