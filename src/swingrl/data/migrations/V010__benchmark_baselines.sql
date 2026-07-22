-- Spec D13: buy-and-hold benchmark baselines for the daily digest. At each epoch reset the
-- recorder (scripts/record_benchmark_baselines.py) snapshots one row per env symbol: the
-- latest stored close as baseline_price, the reset day as baseline_date, and the env's TOTAL
-- configured capital as capital_usd. The digest's _benchmark_value() later equal-weights that
-- capital across the env's symbols and grows each slice by latest_close / baseline_price, so
-- "agent vs passive buy-and-hold" is visible every day.
--
-- capital_usd persists the env total AT RECORD TIME so later config drift never silently moves
-- the benchmark — the snapshot, not config, is the source of truth afterward. baseline_price and
-- capital_usd are NOT NULL because a baseline with no anchor price or no capital cannot value a
-- benchmark. PRIMARY KEY (environment, symbol) makes the recorder's ON CONFLICT upsert idempotent
-- (re-running at the same reset overwrites in place rather than duplicating).
CREATE TABLE benchmark_baselines (
    environment    TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    baseline_date  DATE NOT NULL,
    baseline_price DOUBLE PRECISION NOT NULL,
    capital_usd    DOUBLE PRECISION NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (environment, symbol)
);
