# src/swingrl/data/options/collector.py
"""EOD collector orchestration: per-symbol fetch->parse->store with guards (spec §6, §10, §17)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

from swingrl.data.options import market_calendar
from swingrl.data.options.chain_parser import parse_chain
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.options.cboe_client import CboeChainClient
    from swingrl.data.options.store import OptionsStore
    from swingrl.monitoring.alerter import Alerter, AlertLevel

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

EXPECTED_CONTRACT_FIELDS = {
    "option",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "iv",
    "open_interest",
    "volume",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
}


@dataclass
class SnapshotResult:
    """Outcome of one snapshot run across all symbols."""

    label: str
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_schema_drift(raw: dict[str, Any]) -> list[str]:
    """Return expected contract fields missing from the first contract (spec §10.5)."""
    options = raw.get("data", {}).get("options", [])
    if not options:
        return []
    return sorted(EXPECTED_CONTRACT_FIELDS - set(options[0]))


class OptionsCollector:
    """Runs one snapshot across all configured symbols with per-symbol isolation."""

    def __init__(
        self,
        config: SwingRLConfig,
        client: CboeChainClient,
        store: OptionsStore,
        alerter: Alerter | None = None,
    ) -> None:
        self._config = config
        self._oc = config.options_collector
        self._client = client
        self._store = store
        self._alerter = alerter

    def symbols(self) -> list[str]:
        """Index symbols first, then equity symbols if enabled (spec §5)."""
        symbols = list(self._oc.index_symbols)
        if self._oc.include_equity_symbols:
            symbols.extend(self._config.equity.symbols)
        return symbols

    def _market_time_utc(self, snapshot_label: str, quote_date: date) -> datetime:
        """The MARKET moment this label represents (D8) — never the pull time."""
        snap = next((s for s in self._oc.snapshots if s.label == snapshot_label), None)
        if snap is None:
            log.error("options_unknown_snapshot_label", label=snapshot_label)
            raise DataError(f"No snapshot config for label {snapshot_label!r}")
        hh, mm = (int(x) for x in snap.market_time_et.split(":"))
        return datetime(
            quote_date.year, quote_date.month, quote_date.day, hh, mm, tzinfo=_ET
        ).astimezone(UTC)

    def run_snapshot(
        self,
        snapshot_label: str,
        now: datetime | None = None,
        scheduled_pull_utc: datetime | None = None,
    ) -> SnapshotResult:
        """Fetch+store every symbol's chain for one snapshot; alert on the summary."""
        now = now or datetime.now(UTC)
        quote_date = now.astimezone(_ET).date()
        result = SnapshotResult(label=snapshot_label)

        late_by_s = 0.0
        if scheduled_pull_utc is not None:
            late_by_s = max(0.0, (now - scheduled_pull_utc).total_seconds())
        # late_by_s is still stamped into the stored snapshot provenance below regardless
        # of this tolerance check (parse_chain(late_by_s=...) in _capture_one) -- only the
        # Discord warning is gated on exceeding the cron-jitter tolerance.
        if late_by_s > self._oc.integrity.late_warn_s and snapshot_label == "decision":
            result.warnings.append(
                f"decision snapshot fired late by {late_by_s:.0f}s — market state is "
                f"NOT the {snapshot_label} moment (lookahead guard, D8)"
            )

        early_close = market_calendar.is_early_close(quote_date)
        market_time_utc = self._market_time_utc(snapshot_label, quote_date)

        for symbol in self.symbols():
            if self._store.snapshot_exists_parquet(symbol, quote_date, snapshot_label):
                result.skipped.append(symbol)
                continue
            try:
                self._capture_one(
                    symbol,
                    snapshot_label,
                    quote_date,
                    market_time_utc,
                    early_close,
                    late_by_s,
                    result,
                )
                result.succeeded.append(symbol)
            except Exception as exc:
                # Broaden beyond DataError (C4): httpx.TransportError (CDN down post-retry),
                # psycopg/pool errors, OSError must not abort the whole run without a summary.
                log.error("options_symbol_failed", symbol=symbol, error=str(exc))
                result.failed.append(symbol)

        self._route_summary_alert(result)
        return result

    def _capture_one(
        self,
        symbol: str,
        snapshot_label: str,
        quote_date: date,
        market_time_utc: datetime,
        early_close: bool,
        late_by_s: float,
        result: SnapshotResult,
    ) -> None:
        raw = self._client.get_option_chain(symbol)
        missing = check_schema_drift(raw)
        if missing:
            result.warnings.append(f"{symbol}: schema drift, missing {missing}")
        parsed = parse_chain(
            raw,
            underlying_symbol=symbol,
            snapshot_label=snapshot_label,
            quote_date=quote_date,
            snapshot_time_utc=market_time_utc,
            pulled_at_utc=datetime.now(UTC),
            schema_version=self._oc.schema_version,
            is_early_close=early_close,
            late_by_s=late_by_s,
        )
        previous = self._store.last_snapshot_row_count(symbol, snapshot_label)
        threshold = self._oc.integrity.contract_count_drop_warn_frac
        if previous and len(parsed.contracts) < previous * (1.0 - threshold):
            result.warnings.append(
                f"{symbol}: contract count dropped {previous} -> {len(parsed.contracts)} "
                f"(possible partial chain — CBOE has no truncation flag)"
            )
        self._store.write_snapshot(parsed, symbol, quote_date, snapshot_label)
        # Parquet-first design (C4): the durable capture already landed above. A Postgres
        # sync failure is a WARNING, not a symbol failure — the next boot reconcile heals it.
        try:
            self._store.sync_to_postgres(parsed)
        except Exception as exc:
            log.warning("options_postgres_sync_failed", symbol=symbol, error=str(exc))
            result.warnings.append(f"{symbol}: postgres sync failed ({exc}) — reconcile will heal")

    def _route_summary_alert(self, result: SnapshotResult) -> None:
        """Route the summary alert (user design 2026-07-16).

        Any symbol succeeding always sends the info "captured" message (with warnings
        folded in inline) -- warnings must never suppress or replace it. Any symbol
        failing additionally sends a warning listing the failures. All-attempted-failed
        remains critical-only (unchanged).
        """
        attempted = len(result.succeeded) + len(result.failed)
        if attempted > 0 and not result.succeeded:
            self._alert(
                "critical", f"Options {result.label}: ALL symbols failed", f"failed={result.failed}"
            )
            return

        if result.succeeded:
            message = f"succeeded={result.succeeded} skipped={result.skipped}"
            if result.warnings:
                message += f" | warnings: {'; '.join(result.warnings)}"
            self._alert("info", f"Options {result.label} captured", message)

        if result.failed:
            # bypass_suppression=True: a capture failure is same-day, un-backfillable
            # data loss (spec §13) -- it must reach Discord on the FIRST occurrence,
            # not wait for consecutive_failures_before_alert identical days.
            self._alert(
                "warning",
                f"Options {result.label} completed with issues",
                f"failed={result.failed} warnings={result.warnings}",
                bypass_suppression=True,
            )

    def _alert(
        self, level: AlertLevel, title: str, message: str, *, bypass_suppression: bool = False
    ) -> None:
        if self._alerter is not None:
            self._alerter.send_alert(level, title, message, bypass_suppression=bypass_suppression)
