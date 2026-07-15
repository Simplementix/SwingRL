from __future__ import annotations

import pandas as pd
from swingrl.data.options.audit import (
    AuditResult,
    audit_dataframe,
    audit_symbols,
    descriptive_stats,
    oi_stability_failures,
)

from swingrl.config.schema import SwingRLConfig


def test_clean_frame_has_no_failures() -> None:
    """OPT-AUDIT-1: sane greeks/spreads/OI pass (spec §10.6)."""
    df = pd.DataFrame(
        {"delta": [0.5, -0.4], "bid": [1.0, 2.0], "ask": [1.1, 2.2], "open_interest": [10, 20]}
    )
    assert audit_dataframe(df) == []


def test_delta_out_of_range_fails() -> None:
    """OPT-AUDIT-2: |delta| > 1 flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [1.5], "bid": [1.0], "ask": [1.1], "open_interest": [10]})
    assert any("delta" in f for f in audit_dataframe(df))


def test_crossed_market_fails() -> None:
    """OPT-AUDIT-3: ask < bid flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [0.5], "bid": [2.0], "ask": [1.0], "open_interest": [10]})
    assert any("bid" in f.lower() or "ask" in f.lower() for f in audit_dataframe(df))


def test_all_oi_null_fails() -> None:
    """OPT-AUDIT-4: open_interest entirely null flagged (spec §10.6)."""
    df = pd.DataFrame({"delta": [0.5], "bid": [1.0], "ask": [1.1], "open_interest": [None]})
    assert any("open_interest" in f for f in audit_dataframe(df))


def test_audit_symbols_combines_index_and_equity() -> None:
    """OPT-AUDIT-5: audit covers index + equity symbols (spec §5)."""
    cfg = SwingRLConfig()
    cfg.equity.symbols = ["SPY"]
    cfg.options_collector.index_symbols = ["_SPX"]
    assert audit_symbols(cfg) == ["_SPX", "SPY"]


def test_audit_result_passed_flag() -> None:
    """OPT-AUDIT-6: passed is True iff no failures (spec §10.6)."""
    assert AuditResult(failures=[]).passed is True
    assert AuditResult(failures=["x: delta"]).passed is False


def test_oi_stability_passes_when_identical() -> None:
    """OPT-AUDIT-7: identical OI across same-day snapshots passes (decision D6)."""
    df = pd.DataFrame(
        {
            "quote_date": ["2026-07-14", "2026-07-14"],
            "contract_symbol": ["C", "C"],
            "snapshot_label": ["decision", "eod"],
            "open_interest": [100, 100],
        }
    )
    assert oi_stability_failures(df) == []


def test_oi_stability_flags_intraday_change() -> None:
    """OPT-AUDIT-8: OI differing across same-day snapshots is flagged (decision D6)."""
    df = pd.DataFrame(
        {
            "quote_date": ["2026-07-14", "2026-07-14"],
            "contract_symbol": ["C", "C"],
            "snapshot_label": ["decision", "eod"],
            "open_interest": [100, 120],
        }
    )
    assert oi_stability_failures(df) != []


def test_descriptive_stats_shape() -> None:
    """OPT-AUDIT-9: monthly digest stats computed (spec §10.6)."""
    df = pd.DataFrame({"iv": [10.0, 20.0], "bid": [1.0, 2.0], "ask": [1.2, 2.3]})
    stats = descriptive_stats(df)
    assert stats["rows"] == 2
    assert stats["median_iv"] == 15.0
