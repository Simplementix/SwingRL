# src/swingrl/data/options/audit.py
"""Data-quality audit over captured options (the slow-rot net; spec §10.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from swingrl.data.pg_helpers import fetchdf

if TYPE_CHECKING:
    from typing import Any

    import psycopg

    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")


@dataclass
class AuditResult:
    """Result of a data-quality audit run."""

    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    symbols_checked: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, float | int | None]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when no hard quality failures were found."""
        return not self.failures


def audit_dataframe(df: pd.DataFrame) -> list[str]:
    """Return quality failures for one symbol's recent contracts (spec §10.6)."""
    failures: list[str] = []
    delta_raw = df["delta"] if "delta" in df else pd.Series(dtype=float)
    delta = pd.to_numeric(delta_raw, errors="coerce").dropna()
    if len(delta) and bool(((delta < -1.0) | (delta > 1.0)).any()):
        failures.append("delta outside [-1, 1]")
    both = df.dropna(subset=["bid", "ask"])
    if len(both) and bool((both["ask"] < both["bid"]).any()):
        failures.append("ask < bid (crossed market)")
    if "open_interest" in df and int(df["open_interest"].notna().sum()) == 0:
        failures.append("open_interest entirely null")
    return failures


def oi_stability_failures(df: pd.DataFrame) -> list[str]:
    """OI must match across same-day snapshots for a contract (T-1/once-daily; decision D6)."""
    required = {"quote_date", "contract_symbol", "snapshot_label", "open_interest"}
    if not required <= set(df.columns):
        return []
    oi = df.dropna(subset=["open_interest"])
    per_contract_day = oi.groupby(["quote_date", "contract_symbol"])["open_interest"].nunique()
    bad = int((per_contract_day > 1).sum())
    return [f"OI differs across same-day snapshots on {bad} contract-days"] if bad else []


def descriptive_stats(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Lightweight monthly digest stats for one symbol (spec §10.6)."""
    iv = pd.to_numeric(df["iv"], errors="coerce").dropna() if "iv" in df else pd.Series(dtype=float)
    spread = (
        (df["ask"] - df["bid"]).dropna()
        if {"ask", "bid"} <= set(df.columns)
        else pd.Series(dtype=float)
    )
    return {
        "rows": int(len(df)),
        "median_iv": round(float(iv.median()), 2) if len(iv) else None,
        "median_spread": round(float(spread.median()), 4) if len(spread) else None,
    }


def audit_symbols(config: SwingRLConfig) -> list[str]:
    """Index symbols + equity symbols (when enabled) covered by the audit."""
    symbols = list(config.options_collector.index_symbols)
    if config.options_collector.include_equity_symbols:
        symbols.extend(config.equity.symbols)
    return symbols


def _load_recent(conn: psycopg.Connection[Any], symbol: str, cutoff: date) -> pd.DataFrame:
    """Fetch the trailing-window rows for one symbol from options_chains."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT quote_date, snapshot_label, contract_symbol, delta, bid, ask, iv, "
            "open_interest FROM options_chains "
            "WHERE underlying_symbol = %s AND quote_date >= %s",
            (symbol, cutoff),
        )
        return fetchdf(cur)


def run_data_quality_audit(
    config: SwingRLConfig,
    db: DatabaseManager,
    *,
    since_days: int = 30,
    now: datetime | None = None,
    alerter: Alerter | None = None,
) -> AuditResult:
    """Audit the trailing window per symbol; CRITICAL-alert on any failure (spec §10.6)."""
    now = now or datetime.now(UTC)
    cutoff = now.astimezone(_ET).date() - timedelta(days=since_days)
    result = AuditResult()
    with db.connection() as conn:
        for symbol in audit_symbols(config):
            df = _load_recent(conn, symbol, cutoff)
            if df.empty:
                result.notes.append(f"{symbol}: no data in trailing {since_days}d")
                continue
            result.symbols_checked.append(symbol)
            for message in audit_dataframe(df) + oi_stability_failures(df):
                result.failures.append(f"{symbol}: {message}")
            result.stats[symbol] = descriptive_stats(df)
    log.info(
        "options_audit_complete",
        passed=result.passed,
        failures=len(result.failures),
        symbols=len(result.symbols_checked),
    )
    if alerter is not None:
        if not result.passed:
            alerter.send_alert(
                "critical",
                "Options data-quality audit FAILED",
                "; ".join(result.failures[:20]),
            )
        else:
            digest = "; ".join(
                f"{s}: rows={st['rows']} iv~{st['median_iv']}" for s, st in result.stats.items()
            )
            alerter.send_alert("info", "Options monthly audit summary", digest or "no data")
    return result
