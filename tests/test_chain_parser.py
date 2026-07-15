from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from swingrl.data.options.chain_parser import (
    CONTRACT_COLUMNS,
    clean_sentinel,
    parse_chain,
    parse_osi,
)


def _raw() -> dict:
    """Representative CBOE payload (shape verified live 2026-07-14)."""
    liquid_call = {
        "option": "SPXW260724C07500000",
        "bid": 12.3,
        "bid_size": 10.0,
        "ask": 12.7,
        "ask_size": 8.0,
        "iv": 0.1234,
        "open_interest": 15000.0,
        "volume": 4200.0,
        "delta": 0.55,
        "gamma": 0.01,
        "vega": 1.2,
        "theta": -0.9,
        "rho": 0.3,
        "theo": 12.45,
        "change": 0.1,
        "open": 12.0,
        "high": 13.0,
        "low": 11.8,
        "tick": "up",
        "last_trade_price": 12.5,
        "last_trade_time": "2026-07-14T15:59:07",
        "percent_change": 0.8,
        "prev_day_close": 12.4,
    }
    illiquid_put = {
        **liquid_call,
        "option": "SPX260918P04000000",
        "iv": 0.0,
        "delta": -999.0,
        "open_interest": 0.0,
        "last_trade_price": 0.0,
        "last_trade_time": None,
    }
    return {
        "timestamp": "2026-07-14 19:45:10",
        "symbol": "^SPX",
        "data": {
            "symbol": "^SPX",
            "security_type": "index",
            "current_price": 7543.59,
            "bid": 7543.0,
            "ask": 7544.0,
            "iv30": 13.2,
            "seqno": 12345,
            "last_trade_time": "2026-07-14T15:59:59",
            "options": [liquid_call, illiquid_put],
        },
    }


def _parse():
    return parse_chain(
        _raw(),
        underlying_symbol="_SPX",
        snapshot_label="decision",
        quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 19, 45, tzinfo=UTC),  # 15:45 ET market state
        pulled_at_utc=datetime(2026, 7, 14, 20, 0, 3, tzinfo=UTC),  # pulled 16:00 ET (D8)
        schema_version="v1",
        is_early_close=False,
        late_by_s=0.0,
    )


def test_parse_osi_call_and_put() -> None:
    """OPT-PARSE-1: root/expiry/right/strike parsed from the OSI symbol (§17 C1)."""
    assert parse_osi("SPXW260724C07500000") == ("SPXW", date(2026, 7, 24), "CALL", 7500.0)
    assert parse_osi("SPX260918P04000000") == ("SPX", date(2026, 9, 18), "PUT", 4000.0)
    assert parse_osi("SPY260821C00650000") == ("SPY", date(2026, 8, 21), "CALL", 650.0)


def test_clean_sentinel_maps_to_nan() -> None:
    """OPT-PARSE-2: -999 / NaN / None -> NaN; iv zero-is-missing rule (§6.3, §17 C1)."""
    assert math.isnan(clean_sentinel(-999.0))
    assert math.isnan(clean_sentinel(None))
    assert math.isnan(clean_sentinel(0.0, zero_is_missing=True))
    assert clean_sentinel(0.0) == 0.0
    assert clean_sentinel(0.55) == 0.55


def test_contracts_flattened_one_row_per_contract() -> None:
    """OPT-PARSE-3: grain = one row per contract (spec §6.3)."""
    df = _parse().contracts
    assert len(df) == 2
    assert list(df.columns) == CONTRACT_COLUMNS


def test_identity_columns_derived() -> None:
    """OPT-PARSE-4: right/strike/expiration/dte derived from OSI + quote_date."""
    df = _parse().contracts.set_index("contract_symbol")
    row = df.loc["SPXW260724C07500000"]
    assert row["option_right"] == "CALL"
    assert row["strike"] == 7500.0
    assert row["expiration"] == date(2026, 7, 24)
    assert int(row["dte"]) == 10  # 2026-07-24 minus quote_date 2026-07-14


def test_iv_fraction_preserved_and_zero_becomes_nan() -> None:
    """OPT-PARSE-5: iv stored as the CBOE decimal fraction; 0.0 -> NaN (illiquid)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert df.loc["SPXW260724C07500000", "iv"] == pytest.approx(0.1234)
    assert math.isnan(df.loc["SPX260918P04000000", "iv"])


def test_sentinel_greeks_become_nan() -> None:
    """OPT-PARSE-6: -999 greeks stored as NaN, never -999 (spec §6.3)."""
    df = _parse().contracts.set_index("contract_symbol")
    assert math.isnan(df.loc["SPX260918P04000000", "delta"])


def test_trade_time_localized_from_et() -> None:
    """OPT-PARSE-10: last_trade_time is ET; stored trade_time_utc converts to UTC (C2)."""
    df = _parse().contracts.set_index("contract_symbol")
    ts = df.loc["SPXW260724C07500000", "trade_time_utc"]
    # Fixture last_trade_time "2026-07-14T15:59:07" is 15:59:07 ET (EDT) = 19:59:07 UTC.
    assert pd.Timestamp(ts) == pd.Timestamp("2026-07-14 19:59:07", tz="UTC")


def test_parse_cboe_ts_localizes_et_to_utc() -> None:
    """OPT-PARSE-11: naive CBOE timestamp is ET, not UTC; None passes through (C2)."""
    from swingrl.data.options.chain_parser import parse_cboe_ts

    # 16:00:00 ET (EDT) close = 20:00:00 UTC — the +4h shift proves ET localization.
    assert parse_cboe_ts("2026-07-14T16:00:00") == datetime(2026, 7, 14, 20, 0, 0, tzinfo=UTC)
    assert parse_cboe_ts(None) is None


def test_raw_json_populated_per_row() -> None:
    """OPT-PARSE-7: full original contract dict kept in raw_json (spec §6.2)."""
    df = _parse().contracts.set_index("contract_symbol")
    raw = df.loc["SPXW260724C07500000", "raw_json"]
    assert isinstance(raw, dict) and raw["theo"] == 12.45


def test_header_denormalized_context() -> None:
    """OPT-PARSE-8: header carries market context + D8 provenance (spec §6.4, §17 C3)."""
    parsed = _parse()
    assert parsed.header["underlying_price"] == 7543.59
    assert parsed.header["is_delayed"] is True  # constant: delayed feed (§17 C1)
    assert parsed.header["number_of_contracts"] == 2  # computed = len(options)
    assert parsed.header["raw_header"]["payload_timestamp"] == "2026-07-14 19:45:10"
    assert parsed.header["raw_header"]["late_by_s"] == 0.0
    assert "options" not in parsed.header["raw_header"]
    assert (parsed.contracts["underlying_price"] == 7543.59).all()


def test_empty_chain_raises_dataerror() -> None:
    """OPT-PARSE-9: no contracts -> DataError (spec §10.3)."""
    from swingrl.utils.exceptions import DataError

    empty = _raw()
    empty["data"]["options"] = []
    with pytest.raises(DataError):
        parse_chain(
            empty,
            underlying_symbol="_SPX",
            snapshot_label="eod",
            quote_date=date(2026, 7, 14),
            snapshot_time_utc=datetime(2026, 7, 14, 20, 15, tzinfo=UTC),
            pulled_at_utc=datetime(2026, 7, 14, 20, 35, tzinfo=UTC),
            schema_version="v1",
            is_early_close=False,
        )
