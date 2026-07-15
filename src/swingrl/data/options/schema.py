# src/swingrl/data/options/schema.py
"""Additive Postgres schema for options capture (spec §8.2, decision D1)."""

from __future__ import annotations

from datetime import date
from typing import Any

import psycopg
import structlog

log = structlog.get_logger(__name__)

OPTIONS_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS options_snapshots (
  underlying_symbol   text NOT NULL,
  quote_date          date NOT NULL,
  snapshot_label      text NOT NULL,
  snapshot_time_utc   timestamptz NOT NULL,
  pulled_at_utc       timestamptz NOT NULL,
  underlying_price    double precision,
  is_delayed          boolean,
  is_early_close      boolean,
  interest_rate       double precision,
  dividend_yield      double precision,
  underlying_volatility double precision,
  number_of_contracts integer,
  status              text,
  source              text NOT NULL DEFAULT 'cboe',
  schema_version      text NOT NULL,
  raw_header          jsonb NOT NULL,
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label)
)
"""

OPTIONS_CHAINS_DDL = """
CREATE TABLE IF NOT EXISTS options_chains (
  underlying_symbol text NOT NULL,
  quote_date        date NOT NULL,
  snapshot_label    text NOT NULL,
  contract_symbol   text NOT NULL,
  option_root text, expiration date, dte integer, strike double precision,
  option_right text, expiration_type text, settlement_type text, exercise_type text,
  multiplier double precision, in_the_money boolean,
  bid double precision, ask double precision, last double precision, mark double precision,
  bid_size integer, ask_size integer, last_size integer,
  open double precision, high double precision, low double precision, close double precision,
  volume bigint, open_interest bigint, net_change double precision,
  delta double precision, gamma double precision, theta double precision,
  vega double precision, rho double precision,
  iv double precision, theoretical_value double precision,
  time_value double precision, intrinsic_value double precision, extrinsic_value double precision,
  underlying_price double precision, is_delayed boolean,
  quote_time_utc timestamptz, trade_time_utc timestamptz,
  pulled_at_utc timestamptz NOT NULL,
  source text NOT NULL DEFAULT 'cboe', schema_version text NOT NULL,
  raw_json jsonb,   -- nullable: NULL when postgres_store_raw_json=false (decision D5; Parquet always keeps it)
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label, contract_symbol)
) PARTITION BY RANGE (quote_date)
"""


def monthly_partition_bounds(quote_date: date) -> tuple[str, date, date]:
    """Return (partition_name, lo_inclusive, hi_exclusive) for quote_date's month."""
    lo = quote_date.replace(day=1)
    hi = date(lo.year + 1, 1, 1) if lo.month == 12 else date(lo.year, lo.month + 1, 1)
    name = f"options_chains_{lo.year:04d}_{lo.month:02d}"
    return name, lo, hi


def ensure_options_schema(conn: psycopg.Connection[Any]) -> None:
    """Idempotently create both options tables (additive; A30-safe, spec §8.2)."""
    with conn.cursor() as cur:
        cur.execute(OPTIONS_SNAPSHOTS_DDL)
        cur.execute(OPTIONS_CHAINS_DDL)
    log.info("options_schema_ensured")


def ensure_monthly_partition(conn: psycopg.Connection[Any], quote_date: date) -> str:
    """Create the monthly partition of options_chains if absent; return its name."""
    name, lo, hi = monthly_partition_bounds(quote_date)
    with conn.cursor() as cur:
        cur.execute(
            # name is derived from quote_date (YYYY_MM), never user input — safe to interpolate.
            f"CREATE TABLE IF NOT EXISTS {name} "  # noqa: S608  # nosec B608
            f"PARTITION OF options_chains FOR VALUES FROM (%s) TO (%s)",
            (lo, hi),
        )
    log.info("options_partition_ensured", partition=name)
    return name
