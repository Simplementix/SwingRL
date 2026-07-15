# tests/test_chain_parser_real_fixture.py
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from swingrl.data.options.chain_parser import parse_chain
from swingrl.data.options.collector import check_schema_drift

_FIXTURE = Path("tests/fixtures/cboe_chain_spx.json")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="real fixture not yet captured (T6)")
def test_parse_real_spx_fixture_no_schema_drift() -> None:
    """OPT-PARSE-10: parser handles the REAL captured chain, no drift (spec §12, §17 C1)."""
    raw = json.loads(_FIXTURE.read_text())
    assert check_schema_drift(raw) == [], "real payload field names differ — update the mapping"
    parsed = parse_chain(
        raw,
        underlying_symbol="_SPX",
        snapshot_label="eod",
        quote_date=date(2026, 7, 14),
        snapshot_time_utc=datetime(2026, 7, 14, 20, 15, tzinfo=UTC),
        pulled_at_utc=datetime(2026, 7, 14, 20, 35, tzinfo=UTC),
        schema_version="v1",
        is_early_close=False,
    )
    assert len(parsed.contracts) > 0
    assert parsed.contracts["iv"].notna().any()
    assert parsed.contracts["strike"].gt(0).all()
    assert parsed.header["number_of_contracts"] == len(parsed.contracts)
