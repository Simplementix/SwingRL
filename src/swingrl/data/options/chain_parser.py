"""Parse a raw CBOE chain dict into typed contract rows + raw_json (spec §6, §17 C1)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_SENTINELS = {-999.0, -999}
_OSI_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

# Ordered typed columns — the grain (spec §6.3). Columns absent from CBOE payloads
# (quote_time_utc, settlement/exercise/multiplier fields) stay in the T9 DDL as NULLs.
CONTRACT_COLUMNS: list[str] = [
    "underlying_symbol",
    "quote_date",
    "snapshot_label",
    "underlying_price",
    "is_delayed",
    "trade_time_utc",
    "pulled_at_utc",
    "source",
    "schema_version",
    "contract_symbol",
    "option_root",
    "expiration",
    "dte",
    "strike",
    "option_right",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "last",
    "volume",
    "open_interest",
    "net_change",
    "prev_day_close",
    "open",
    "high",
    "low",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
    "theoretical_value",
    "raw_json",
]


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain: snapshot-level header + one DataFrame row per contract."""

    header: dict[str, Any]
    contracts: pd.DataFrame


def parse_osi(symbol: str) -> tuple[str, date, str, float]:
    """Split an OSI id into (root, expiration, CALL|PUT, strike) — CBOE sends no fields."""
    m = _OSI_RE.match(symbol)
    if not m:
        log.error("options_osi_unparseable", symbol=symbol)
        raise DataError(f"Unparseable OSI option symbol: {symbol!r}")
    root, yymmdd, right, strike_milli = m.groups()
    expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    return root, expiration, ("CALL" if right == "C" else "PUT"), int(strike_milli) / 1000.0


def parse_cboe_ts(value: str | None) -> datetime | None:
    """CBOE last_trade_time strings -> tz-aware UTC.

    The naive timestamp is treated as America/New_York, not UTC: the fixtures cluster in
    market hours (e.g. an SPY last trade of 16:00:00 = the ET close), so it must be an ET
    wall-clock. This convention is inferred from fixture evidence and is to be CONFIRMED by
    the T6 trading-day probe — it is not yet empirically verified. Parse naive, localize ET,
    then convert to UTC.
    """
    if not value:
        return None
    normalized = value.replace("T", " ")
    try:
        naive = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        log.warning("cboe_ts_unparsed", value=value)
        return None
    return naive.replace(tzinfo=_ET).astimezone(UTC)


def clean_sentinel(value: float | int | None, *, zero_is_missing: bool = False) -> float:
    """Map -999 / NaN / None (and optionally 0.0, for iv) to real NaN (spec §6.3)."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")
    if value in _SENTINELS:
        return float("nan")
    if zero_is_missing and float(value) == 0.0:
        return float("nan")
    return float(value)


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _i(value: Any) -> int | None:
    return None if value is None else int(value)


def _row(contract: dict, *, quote_date: date, base: dict) -> dict:
    symbol = contract.get("option", "")
    root, expiration, right, strike = parse_osi(symbol)
    row = dict(base)
    row.update(
        contract_symbol=symbol,
        option_root=root,
        expiration=expiration,
        dte=(expiration - quote_date).days,
        strike=strike,
        option_right=right,
        bid=_f(contract.get("bid")),
        ask=_f(contract.get("ask")),
        bid_size=_i(contract.get("bid_size")),
        ask_size=_i(contract.get("ask_size")),
        last=_f(contract.get("last_trade_price")),
        volume=_i(contract.get("volume")),
        open_interest=_i(contract.get("open_interest")),
        net_change=_f(contract.get("change")),
        prev_day_close=_f(contract.get("prev_day_close")),
        open=_f(contract.get("open")),
        high=_f(contract.get("high")),
        low=_f(contract.get("low")),
        delta=clean_sentinel(contract.get("delta")),
        gamma=clean_sentinel(contract.get("gamma")),
        theta=clean_sentinel(contract.get("theta")),
        vega=clean_sentinel(contract.get("vega")),
        rho=clean_sentinel(contract.get("rho")),
        # CBOE illiquid convention observed 2026-07-14: iv == 0.0 means "no IV".
        iv=clean_sentinel(contract.get("iv"), zero_is_missing=True),
        theoretical_value=_f(contract.get("theo")),
        trade_time_utc=parse_cboe_ts(contract.get("last_trade_time")),
        raw_json=contract,
    )
    return row


def parse_chain(
    raw: dict[str, Any],
    *,
    underlying_symbol: str,
    snapshot_label: str,
    quote_date: date,
    snapshot_time_utc: datetime,
    pulled_at_utc: datetime,
    schema_version: str,
    is_early_close: bool,
    late_by_s: float = 0.0,
    source: str = "cboe",
) -> ParsedChain:
    """Flatten a raw CBOE chain to typed rows + raw_json and build the header (spec §6).

    snapshot_time_utc = the MARKET moment the data represents (D8), never the pull time.
    """
    data = raw.get("data", {})
    options = data.get("options", [])
    base = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "underlying_price": _f(data.get("current_price")),
        "is_delayed": True,  # constant: this IS the delayed feed (spec §17 C1)
        "pulled_at_utc": pulled_at_utc,
        "source": source,
        "schema_version": schema_version,
    }
    rows = [_row(c, quote_date=quote_date, base=base) for c in options]
    if not rows:
        log.error("options_empty_chain", underlying_symbol=underlying_symbol)
        raise DataError(f"Empty option chain for {underlying_symbol}")
    contracts_df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)

    raw_header = {k: v for k, v in data.items() if k != "options"}
    raw_header["payload_timestamp"] = raw.get("timestamp")
    raw_header["late_by_s"] = late_by_s
    header = {
        "underlying_symbol": underlying_symbol,
        "quote_date": quote_date,
        "snapshot_label": snapshot_label,
        "snapshot_time_utc": snapshot_time_utc,
        "pulled_at_utc": pulled_at_utc,
        "underlying_price": _f(data.get("current_price")),
        "is_delayed": True,
        "is_early_close": is_early_close,
        "interest_rate": None,  # not provided by CBOE; FRED covers recomputation
        "dividend_yield": None,  # not provided by CBOE
        "underlying_volatility": _f(data.get("iv30")),
        "number_of_contracts": len(rows),  # computed — CBOE has no header count
        "status": "SUCCESS",
        "source": source,
        "schema_version": schema_version,
        "raw_header": raw_header,
    }
    return ParsedChain(header=header, contracts=contracts_df)
