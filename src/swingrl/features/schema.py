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


_TRAINING_EPOCHS_DDL = """
CREATE TABLE IF NOT EXISTS training_epochs (
    id              INTEGER PRIMARY KEY DEFAULT nextval('training_epochs_id_seq'),
    run_id          VARCHAR NOT NULL,
    epoch           INTEGER NOT NULL,
    algo            VARCHAR NOT NULL,
    env             VARCHAR NOT NULL,
    sharpe          DOUBLE,
    mdd             DOUBLE,
    reward_weight_profit   DOUBLE,
    reward_weight_sharpe   DOUBLE,
    reward_weight_drawdown DOUBLE,
    reward_weight_turnover DOUBLE,
    stop_training   BOOLEAN DEFAULT FALSE,
    rationale       TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
)
"""

_META_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS meta_decisions (
    id              INTEGER PRIMARY KEY DEFAULT nextval('meta_decisions_id_seq'),
    run_id          VARCHAR NOT NULL,
    algo            VARCHAR NOT NULL,
    env             VARCHAR NOT NULL,
    decision_type   VARCHAR NOT NULL,
    decision_json   TEXT NOT NULL,
    rationale       TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
)
"""

_REWARD_ADJUSTMENTS_DDL = """
CREATE TABLE IF NOT EXISTS reward_adjustments (
    id              INTEGER PRIMARY KEY DEFAULT nextval('reward_adjustments_id_seq'),
    run_id          VARCHAR NOT NULL,
    epoch_trigger   INTEGER,
    epoch_outcome   INTEGER,
    algo            VARCHAR NOT NULL,
    env             VARCHAR NOT NULL,
    weight_before   TEXT,
    weight_after    TEXT,
    outcome_sharpe  DOUBLE,
    created_at      TIMESTAMP DEFAULT NOW()
)
"""

# Sequences for auto-incrementing PKs (CREATE IF NOT EXISTS is idempotent)
_TRAINING_TELEMETRY_SEQUENCES: list[str] = [
    "CREATE SEQUENCE IF NOT EXISTS training_epochs_id_seq",
    "CREATE SEQUENCE IF NOT EXISTS meta_decisions_id_seq",
    "CREATE SEQUENCE IF NOT EXISTS reward_adjustments_id_seq",
]

# Idempotent ALTER TABLE statements for training_runs (DuckDB supports IF NOT EXISTS)
_ALTER_TRAINING_RUNS_DDL: list[str] = [
    "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS run_type VARCHAR DEFAULT 'completed'",
    "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS meta_rationale TEXT",
    "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS dominant_regime VARCHAR",
]


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


def apply_training_telemetry_ddl(conn: Any) -> None:
    """Create training telemetry tables and migrate training_runs in DuckDB.

    Creates three new telemetry tables (training_epochs, meta_decisions,
    reward_adjustments) and applies idempotent ALTER TABLE statements to
    add meta-training columns to training_runs.

    All DDL uses CREATE TABLE IF NOT EXISTS and ADD COLUMN IF NOT EXISTS
    for idempotency — safe to call multiple times on the same connection.

    Args:
        conn: DuckDB connection (or cursor) to execute DDL on.
              Must be open in read-write mode.
    """
    for seq_ddl in _TRAINING_TELEMETRY_SEQUENCES:
        conn.execute(seq_ddl)

    for ddl in [
        _TRAINING_EPOCHS_DDL,
        _META_DECISIONS_DDL,
        _REWARD_ADJUSTMENTS_DDL,
    ]:
        conn.execute(ddl)

    log.info(
        "training_telemetry_tables_created",
        tables=["training_epochs", "meta_decisions", "reward_adjustments"],
    )

    # Apply ALTER TABLE only when training_runs exists (may not exist on fresh DB)
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'training_runs'"
    ).fetchone()
    if result and result[0] > 0:
        for alter_ddl in _ALTER_TRAINING_RUNS_DDL:
            conn.execute(alter_ddl)
        log.info(
            "training_runs_migrated",
            columns=["run_type", "meta_rationale", "dominant_regime"],
        )
    else:
        log.debug(
            "training_runs_not_found_skipping_alter",
            note="ALTER TABLE skipped; training_runs table does not exist yet",
        )
