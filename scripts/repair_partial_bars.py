"""Repair partial 4H bars frozen by the inclusive-endTime ingestion defect.

Background
----------
``BinanceIngestor.fetch`` used to pass the last-completed-bar boundary as Binance's
``endTime``, which is INCLUSIVE of a kline whose openTime equals it. A collector run
firing a minute past the boundary therefore stored the just-opened bar with ~1 minute
of data; the next incremental run started AFTER it, freezing the partial values
permanently. The boundary itself is fixed in ``binance.py``; this one-shot tool repairs
rows already written partial.

Detection
---------
An ``ohlcv_4h`` row is *partial* when its ``fetched_at`` falls inside its own bar window
``[datetime, datetime + 4h)`` — the fetch happened before the bar closed. A partial is
only *repairable* once its window has CLOSED (``datetime + 4h <= now``); a bar still
forming is refused.

For each repairable row the tool deletes the DB row, refetches the completed bar via the
now-fixed ingestor path, reinserts it, and rewrites the Parquet file through the sanctioned
``BinanceIngestor.store`` (ParquetStore) path so Parquet and Postgres agree.

Scope: ``ohlcv_4h`` only. The equity ``ohlcv_daily`` path is structurally immune to
partial daily bars (Alpaca ``end`` is pinned to 00:00 UTC of the current day while daily
bars are ET-session-anchored at 04:00/05:00 UTC — today's bar always exceeds ``end``), so
there is deliberately no ``ohlcv_daily`` branch here.

Safety: ``--dry-run`` is the DEFAULT; mutations require ``--apply``. Run gated by the
controller, never ad-hoc against a live database.

Usage:
    python scripts/repair_partial_bars.py            # dry-run (default)
    python scripts/repair_partial_bars.py --apply    # apply repairs
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
import structlog

from swingrl.config.schema import load_config
from swingrl.data.binance import BinanceIngestor
from swingrl.data.pg_helpers import fetchdf
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# 4H bar width — the window length used for partial detection.
BAR_INTERVAL: timedelta = timedelta(hours=4)

_OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class OhlcvGateway(Protocol):
    """Minimal ohlcv_4h data access the repairer needs (keeps psycopg out of tests)."""

    def read_ohlcv_4h(self) -> pd.DataFrame:
        """Return every ohlcv_4h row with symbol, datetime, OHLCV, source, fetched_at."""
        ...

    def delete_bar(self, symbol: str, bar_open: pd.Timestamp) -> None:
        """Delete the row keyed by (symbol, bar_open)."""
        ...

    def insert_bar(
        self,
        symbol: str,
        bar_open: pd.Timestamp,
        values: Mapping[str, Any],
        fetched_at: datetime,
    ) -> None:
        """Insert/replace the row keyed by (symbol, bar_open) with corrected values."""
        ...


@dataclass
class RepairRecord:
    """One bar's disposition in a repair run.

    Attributes:
        symbol: Trading pair.
        bar_open: Bar open time (UTC).
        status: "would_repair" | "repaired" | "refused_window_open" | "refetch_missing".
        before: Stored (partial) values.
        after: Corrected values, or None when not applied.
    """

    symbol: str
    bar_open: pd.Timestamp
    status: str
    before: dict[str, Any]
    after: dict[str, Any] | None


def _to_utc(value: Any) -> pd.Timestamp:
    """Coerce a timestamp-like value to a UTC-aware pandas Timestamp."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def classify_partial_bars(
    rows: pd.DataFrame,
    now: datetime,
    interval: timedelta = BAR_INTERVAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ohlcv_4h rows into (repairable_partials, unclosed_partials).

    A row is *partial* when ``fetched_at`` is within ``[datetime, datetime + interval)``.
    A partial is *repairable* only once its window has closed
    (``datetime + interval <= now``); partials whose window is still open are returned
    separately so callers can refuse them.

    Args:
        rows: DataFrame with at least 'symbol', 'datetime', 'fetched_at' columns.
        now: Current time (UTC-aware or naive-UTC).
        interval: Bar width (default 4H).

    Returns:
        Tuple of (repairable, unclosed) DataFrame slices of the input rows.
    """
    if rows.empty:
        return rows.iloc[0:0], rows.iloc[0:0]

    dt = pd.to_datetime(rows["datetime"], utc=True)
    fetched = pd.to_datetime(rows["fetched_at"], utc=True)
    window_end = dt + interval
    now_ts = _to_utc(now)

    partial = (fetched >= dt) & (fetched < window_end)
    closed = window_end <= now_ts
    return rows[partial & closed], rows[partial & ~closed]


class PartialBarRepairer:
    """Detect and (optionally) repair partial 4H bars in ohlcv_4h + Parquet."""

    def __init__(
        self,
        gateway: OhlcvGateway,
        ingestor: BinanceIngestor,
        *,
        now_fn: Callable[[], datetime] | None = None,
        interval: timedelta = BAR_INTERVAL,
    ) -> None:
        """Initialize the repairer.

        Args:
            gateway: ohlcv_4h data access (read/delete/insert).
            ingestor: Fixed BinanceIngestor used to refetch and to rewrite Parquet.
            now_fn: Callable returning current time (injectable for tests).
            interval: Bar width (default 4H).
        """
        self._gateway = gateway
        self._ingestor = ingestor
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._interval = interval

    def run(self, *, apply: bool) -> list[RepairRecord]:
        """Find partial bars and, when ``apply`` is True, repair the closed-window ones.

        Args:
            apply: When True, delete+refetch+reinsert and rewrite Parquet. When False,
                report only (no mutation).

        Returns:
            One RepairRecord per detected partial bar.
        """
        now = self._now_fn()
        rows = self._gateway.read_ohlcv_4h()
        repairable, unclosed = classify_partial_bars(rows, now, self._interval)

        records: list[RepairRecord] = []
        for _, row in repairable.iterrows():
            records.append(self._handle_repairable(row, now=now, apply=apply))
        for _, row in unclosed.iterrows():
            symbol, bar_open = str(row["symbol"]), _to_utc(row["datetime"])
            log.warning("partial_bar_window_open", symbol=symbol, bar_open=str(bar_open))
            records.append(
                RepairRecord(symbol, bar_open, "refused_window_open", _before_values(row), None)
            )
        return records

    def _handle_repairable(self, row: pd.Series, *, now: datetime, apply: bool) -> RepairRecord:
        """Report or repair a single closed-window partial bar."""
        symbol = str(row["symbol"])
        bar_open = _to_utc(row["datetime"])
        before = _before_values(row)

        if not apply:
            return RepairRecord(symbol, bar_open, "would_repair", before, None)

        bar_df = self._refetch_bar(symbol, bar_open)
        if bar_df is None:
            log.warning("partial_bar_refetch_missing", symbol=symbol, bar_open=str(bar_open))
            return RepairRecord(symbol, bar_open, "refetch_missing", before, None)

        corrected = {field: float(bar_df.iloc[0][field]) for field in _OHLCV_FIELDS}
        values: dict[str, Any] = {**corrected, "source": _optional(row.get("source"))}

        self._gateway.delete_bar(symbol, bar_open)
        self._gateway.insert_bar(symbol, bar_open, values, now)
        # Sanctioned Parquet write path — upsert overwrites the partial row (keep last).
        self._ingestor.store(bar_df, symbol)

        log.info(
            "partial_bar_repaired",
            symbol=symbol,
            bar_open=str(bar_open),
            close_before=before.get("close"),
            close_after=corrected["close"],
        )
        return RepairRecord(symbol, bar_open, "repaired", before, corrected)

    def _refetch_bar(self, symbol: str, bar_open: pd.Timestamp) -> pd.DataFrame | None:
        """Refetch the completed bar at ``bar_open`` via the fixed ingestor path.

        Returns a single-row DataFrame (indexed by ``bar_open``) or None if the bar is
        not present in the refetched range.
        """
        refetched = self._ingestor.fetch(symbol, since=bar_open.isoformat())
        if refetched is None or refetched.empty:
            return None
        mask = refetched.index == bar_open
        if not bool(mask.any()):
            return None
        return refetched[mask]


def _before_values(row: pd.Series) -> dict[str, Any]:
    """Extract stored values (OHLCV + fetched_at) from a DB row for reporting."""
    values: dict[str, Any] = {field: row.get(field) for field in _OHLCV_FIELDS}
    values["fetched_at"] = str(row.get("fetched_at"))
    return values


def _optional(value: Any) -> Any:
    """Return None for NaN/NaT, else the value unchanged."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


class PostgresOhlcvGateway:
    """Concrete ohlcv_4h gateway backed by the production PostgreSQL database."""

    _SELECT = (
        "SELECT symbol, datetime, open, high, low, close, volume, source, fetched_at "
        "FROM ohlcv_4h ORDER BY symbol, datetime"
    )
    _DELETE = "DELETE FROM ohlcv_4h WHERE symbol = %s AND datetime = %s"
    _INSERT = (
        "INSERT INTO ohlcv_4h "
        "(symbol, datetime, open, high, low, close, volume, source, fetched_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (symbol, datetime) DO UPDATE SET "
        "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
        "close = EXCLUDED.close, volume = EXCLUDED.volume, source = EXCLUDED.source, "
        "fetched_at = EXCLUDED.fetched_at"
    )

    def __init__(self, db: DatabaseManager) -> None:
        """Wrap a DatabaseManager for ohlcv_4h read/delete/insert."""
        self._db = db

    def read_ohlcv_4h(self) -> pd.DataFrame:
        """Read every ohlcv_4h row into a DataFrame."""
        with self._db.connection() as conn:
            cur: Any = conn.execute(self._SELECT)
            return fetchdf(cur)

    def delete_bar(self, symbol: str, bar_open: pd.Timestamp) -> None:
        """Delete the row keyed by (symbol, bar_open)."""
        with self._db.connection() as conn:
            conn.execute(self._DELETE, [symbol, bar_open.to_pydatetime()])

    def insert_bar(
        self,
        symbol: str,
        bar_open: pd.Timestamp,
        values: Mapping[str, Any],
        fetched_at: datetime,
    ) -> None:
        """Insert/replace the corrected bar keyed by (symbol, bar_open)."""
        with self._db.connection() as conn:
            conn.execute(
                self._INSERT,
                [
                    symbol,
                    bar_open.to_pydatetime(),
                    values["open"],
                    values["high"],
                    values["low"],
                    values["close"],
                    values["volume"],
                    values.get("source"),
                    fetched_at,
                ],
            )


def _print_summary(records: list[RepairRecord], *, apply: bool) -> None:
    """Print a human-readable per-symbol summary of the run."""
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Partial 4H bar repair ({mode}) ===")

    if not records:
        print("No partial bars found. Nothing to repair.")
        return

    repaired = [r for r in records if r.status in ("repaired", "would_repair")]
    refused = [r for r in records if r.status == "refused_window_open"]
    missing = [r for r in records if r.status == "refetch_missing"]

    by_symbol: dict[str, list[RepairRecord]] = {}
    for rec in repaired:
        by_symbol.setdefault(rec.symbol, []).append(rec)

    verb = "repaired" if apply else "would repair"
    for symbol in sorted(by_symbol):
        recs = by_symbol[symbol]
        print(f"\n{symbol}: {len(recs)} {verb}")
        for rec in recs:
            before_close = rec.before.get("close")
            after_close = rec.after.get("close") if rec.after else "(refetch pending)"
            print(f"  {rec.bar_open}  close {before_close} -> {after_close}")
            if rec.after:
                deltas = ", ".join(
                    f"{f} {rec.before.get(f)}->{rec.after.get(f)}" for f in _OHLCV_FIELDS
                )
                print(f"    {deltas}")

    print(f"\nRefused (window still open): {len(refused)}")
    for rec in refused:
        print(f"  {rec.symbol} {rec.bar_open}")
    if missing:
        print(f"Refetch produced no bar for: {len(missing)}")
        for rec in missing:
            print(f"  {rec.symbol} {rec.bar_open}")
    print(f"\nTotal partial bars found: {len(records)}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Repair partial 4H bars in ohlcv_4h (and their Parquet files).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs (delete+refetch+reinsert, rewrite Parquet). Default: dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; never mutate (the default). Wins over --apply if both are given.",
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the partial-bar repair tool.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    apply = bool(args.apply) and not bool(args.dry_run)

    from swingrl.data.db import DatabaseManager  # noqa: PLC0415  # lazy: avoids DB import cost

    db = DatabaseManager(config)
    gateway = PostgresOhlcvGateway(db)
    ingestor = BinanceIngestor(config)
    try:
        repairer = PartialBarRepairer(gateway, ingestor)
        records = repairer.run(apply=apply)
    finally:
        ingestor.close()

    _print_summary(records, apply=apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
