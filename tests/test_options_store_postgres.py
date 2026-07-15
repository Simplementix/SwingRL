# tests/test_options_store_postgres.py
from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import psycopg
import pytest

from swingrl.config.schema import OptionsCollectorConfig
from swingrl.data.options.chain_parser import CONTRACT_COLUMNS, ParsedChain
from swingrl.data.options.schema import ensure_options_schema
from swingrl.data.options.store import OptionsStore

_DB_URL = os.environ.get("DATABASE_URL")
_needs_db = pytest.mark.skipif(not _DB_URL, reason="DATABASE_URL not set")


class _FakeDB:
    """Hands out one connection whose commits are swallowed for test isolation."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def connection(self):  # mimics DatabaseManager.connection() contextmanager
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self._conn  # no commit — test rolls back at the end

        return _cm()


def _parsed(symbol: str = "_SPX") -> ParsedChain:
    row = dict.fromkeys(CONTRACT_COLUMNS)
    row.update(
        underlying_symbol=symbol,
        quote_date=date(2026, 7, 14),
        snapshot_label="decision",
        contract_symbol="SPXW260718C05000000",
        strike=5000.0,
        dte=4,
        option_right="CALL",
        delta=0.55,
        iv=12.3,
        underlying_price=5001.2,
        is_delayed=False,
        pulled_at_utc=datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        expiration=date(2026, 7, 18),
        source="cboe",
        schema_version="v1",
        raw_json={"symbol": "SPXW260718C05000000", "strikePrice": 5000.0},
    )
    header = {
        "underlying_symbol": symbol,
        "quote_date": date(2026, 7, 14),
        "snapshot_label": "decision",
        "snapshot_time_utc": datetime(2026, 7, 14, 19, 45, tzinfo=UTC),
        "pulled_at_utc": datetime(2026, 7, 14, 19, 45, 3, tzinfo=UTC),
        "underlying_price": 5001.2,
        "is_delayed": False,
        "is_early_close": False,
        "interest_rate": 5.0,
        "dividend_yield": 1.3,
        "underlying_volatility": 13.0,
        "number_of_contracts": 1,
        "status": "SUCCESS",
        "source": "cboe",
        "schema_version": "v1",
        "raw_header": {"symbol": symbol, "status": "SUCCESS"},
    }
    return ParsedChain(header=header, contracts=pd.DataFrame([row])[CONTRACT_COLUMNS])


def _store(tmp_path: Path, conn: psycopg.Connection) -> OptionsStore:
    cfg = OptionsCollectorConfig()
    cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
    return OptionsStore(cfg, db=_FakeDB(conn))


@_needs_db
def test_sync_inserts_parent_and_child(tmp_path: Path) -> None:
    """OPT-STORE-6: sync writes parent snapshot + child contract rows (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_snapshots")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM options_chains")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT raw_json->>'strikePrice' FROM options_chains")
            assert cur.fetchone()[0] == "5000.0"
        conn.rollback()


@_needs_db
def test_sync_is_idempotent(tmp_path: Path) -> None:
    """OPT-STORE-7: re-sync -> ON CONFLICT DO NOTHING, no duplicate (spec §10.1)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.sync_to_postgres(_parsed())
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_chains")
            assert cur.fetchone()[0] == 1
        conn.rollback()


@_needs_db
def test_sync_respects_raw_json_flag(tmp_path: Path) -> None:
    """OPT-STORE-9: postgres_store_raw_json=False stores NULL raw_json (decision D5)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        cfg = OptionsCollectorConfig()
        cfg.output_dir = str(tmp_path / "options_eod" / "cboe")
        cfg.postgres_store_raw_json = False
        store = OptionsStore(cfg, db=_FakeDB(conn))
        store.sync_to_postgres(_parsed())
        with conn.cursor() as cur:
            cur.execute("SELECT raw_json FROM options_chains")
            assert cur.fetchone()[0] is None
        conn.rollback()


@_needs_db
def test_reconcile_loads_unsynced_parquet(tmp_path: Path) -> None:
    """OPT-STORE-8: reconcile loads a Parquet with no parent row (spec §8.2)."""
    with psycopg.connect(_DB_URL) as conn:
        ensure_options_schema(conn)
        store = _store(tmp_path, conn)
        store.write_snapshot(_parsed(), "_SPX", date(2026, 7, 14), "decision")  # Parquet only
        loaded = store.reconcile()
        assert loaded == 1
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM options_snapshots")
            assert cur.fetchone()[0] == 1
        conn.rollback()
