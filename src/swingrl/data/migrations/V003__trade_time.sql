-- Spec §4.7 (D-T3.13/14/15) + A27 turbulence stamp.

CREATE TABLE inference_cycles (
    cycle_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    environment        TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    mode               TEXT NOT NULL CHECK (mode IN ('paper', 'live')),
    cycle_ts           TIMESTAMPTZ NOT NULL,
    deployed_iteration SMALLINT,            -- derived/display only (A20)
    hmm_p_bull         DOUBLE PRECISION,    -- probability 0-1
    hmm_p_bear         DOUBLE PRECISION,    -- probability 0-1
    vix                DOUBLE PRECISION,    -- index points
    turbulence         DOUBLE PRECISION,    -- A27: decision-time sensor value (pre-zeroing)
    active_event_ids   BIGINT[],
    blended_actions    JSONB,               -- {"schema_version":1,"raw":{sym:val},"target_weights_frac":{sym:val}}
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_inference_cycles_env_ts ON inference_cycles (environment, cycle_ts);

CREATE TABLE cycle_algo_proposals (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cycle_id             BIGINT NOT NULL REFERENCES inference_cycles (cycle_id),
    model_id             TEXT NOT NULL REFERENCES models (model_id),
    algorithm            TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    proposed_actions     JSONB NOT NULL,    -- {"schema_version":1,"raw":{sym:val}} — same shape as blend
    weight_in_blend_frac DOUBLE PRECISION NOT NULL,  -- snapshotted (D-T3.13)
    UNIQUE (cycle_id, model_id)
);

ALTER TABLE trades ADD COLUMN cycle_id BIGINT REFERENCES inference_cycles (cycle_id);
CREATE INDEX idx_trades_cycle ON trades (cycle_id);

CREATE TABLE fill_quality (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trade_id                TEXT NOT NULL UNIQUE REFERENCES trades (trade_id),
    decision_price_usd      NUMERIC(18, 8),
    expected_fill_price_usd NUMERIC(18, 8),
    fill_price_usd          NUMERIC(18, 8) NOT NULL,
    slippage_frac           DOUBLE PRECISION,
    expected_cost_frac      DOUBLE PRECISION,   -- snapshotted from config in force
    realized_cost_frac      DOUBLE PRECISION,
    time_to_fill_ms         INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON COLUMN fill_quality.slippage_frac IS
    'fraction, signed side-aware: positive = adverse to the order';

CREATE TABLE calendar_events (
    event_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type   TEXT NOT NULL CHECK (event_type IN ('fomc', 'cpi', 'nfp', 'gdp')),
    symbol       TEXT,                        -- NULL for macro (all current types)
    scheduled_at TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,        -- materialized at ingest (D-T3.14)
    window_end   TIMESTAMPTZ NOT NULL,
    importance   TEXT NOT NULL CHECK (importance IN ('high', 'medium', 'low')),
    source       TEXT NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (event_type, symbol, scheduled_at)   -- pg16; idempotent re-ingest
);
CREATE INDEX idx_calendar_events_sched ON calendar_events (scheduled_at);

CREATE TABLE event_outcomes (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id    BIGINT NOT NULL REFERENCES calendar_events (event_id),
    payload     JSONB NOT NULL,               -- units per key, schema_version inside
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
