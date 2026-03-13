"""DuckDB DDL for Phase 5 feature tables.

Creates four tables: features_equity, features_crypto, fundamentals,
and hmm_state_history. All CREATE TABLE IF NOT EXISTS for idempotency.

Usage:
    import duckdb
    from swingrl.features.schema import init_feature_schema

    conn = duckdb.connect("data/db/market_data.ddb")
    init_feature_schema(conn)
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)

_FEATURES_EQUITY_DDL = """
CREATE TABLE IF NOT EXISTS features_equity (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    price_sma50_ratio DOUBLE,
    price_sma200_ratio DOUBLE,
    rsi_14 DOUBLE,
    macd_line DOUBLE,
    macd_histogram DOUBLE,
    bb_position DOUBLE,
    atr_14_pct DOUBLE,
    volume_sma20_ratio DOUBLE,
    adx_14 DOUBLE,
    weekly_trend_dir DOUBLE,
    weekly_rsi_14 DOUBLE,
    pe_zscore DOUBLE,
    earnings_growth DOUBLE,
    debt_to_equity DOUBLE,
    dividend_yield DOUBLE,
    PRIMARY KEY (symbol, date)
)
"""

_FEATURES_CRYPTO_DDL = """
CREATE TABLE IF NOT EXISTS features_crypto (
    symbol TEXT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    price_sma50_ratio DOUBLE,
    price_sma200_ratio DOUBLE,
    rsi_14 DOUBLE,
    macd_line DOUBLE,
    macd_histogram DOUBLE,
    bb_position DOUBLE,
    atr_14_pct DOUBLE,
    volume_sma20_ratio DOUBLE,
    adx_14 DOUBLE,
    daily_trend_dir DOUBLE,
    daily_rsi_14 DOUBLE,
    four_h_rsi_14 DOUBLE,
    four_h_price_sma20_ratio DOUBLE,
    PRIMARY KEY (symbol, datetime)
)
"""

_FUNDAMENTALS_DDL = """
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    pe_ratio DOUBLE,
    earnings_growth DOUBLE,
    debt_to_equity DOUBLE,
    dividend_yield DOUBLE,
    sector TEXT,
    fetched_at TIMESTAMP,
    PRIMARY KEY (symbol, date)
)
"""

_HMM_STATE_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS hmm_state_history (
    environment TEXT NOT NULL,
    date DATE NOT NULL,
    p_bull DOUBLE,
    p_bear DOUBLE,
    p_crisis DOUBLE DEFAULT 0.0,
    log_likelihood DOUBLE,
    fitted_at TIMESTAMP,
    PRIMARY KEY (environment, date)
)
"""


def init_feature_schema(conn: Any) -> None:
    """Create all Phase 5 feature tables in DuckDB.

    Args:
        conn: DuckDB connection (or cursor) to execute DDL on.
    """
    for ddl in [
        _FEATURES_EQUITY_DDL,
        _FEATURES_CRYPTO_DDL,
        _FUNDAMENTALS_DDL,
        _HMM_STATE_HISTORY_DDL,
    ]:
        conn.execute(ddl)

    log.info("feature_schema_initialized", tables=4)
