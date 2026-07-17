-- Spec §4.4 Coach records (D-T3.8/D-T3.9/D-T3.10/D-T3.11).
-- Replaces llm_audit_log; retires meta_decisions/reward_adjustments' two-pass core.
-- One conversation event -> two typed records (llm_calls + intent_records), joined by key.

-- llm_calls: the transcript of every conversation with either coach.
CREATE TABLE llm_calls (
    llm_call_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    coach            TEXT NOT NULL CHECK (coach IN ('meta_trainer', 'meta_trader', 'consolidator')),
    call_type        TEXT NOT NULL CHECK (call_type IN
        ('run_config', 'epoch_advice', 'consolidate_stage1', 'consolidate_stage2',
         'harness_replay', 'trade_commentary', 'trade_alarm', 'event_significance')),
    run_pk           BIGINT REFERENCES training_runs (run_pk),      -- fold-scoped calls
    cycle_id         BIGINT REFERENCES inference_cycles (cycle_id), -- trade-time calls (A15)
    iteration_number SMALLINT,
    environment      TEXT CHECK (environment IN ('equity', 'crypto')),
    algorithm        TEXT CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    provider         TEXT NOT NULL,           -- e.g. cerebras, openrouter
    model            TEXT NOT NULL,           -- exact model ID
    prompt_version   TEXT NOT NULL,           -- §2.3 production-identical prompts = checkable equality
    prompt_text      TEXT,
    response_text    TEXT,
    response_parsed  JSONB,
    success          BOOLEAN,                 -- fail-open but counted (F3)
    error            TEXT,
    latency_ms       INTEGER,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- A15 identity CHECK matrix (all 8 call_types): NULL legal only where the call
    -- type genuinely has no such context (the F3 fix). Every enumerated call_type is
    -- covered by exactly one branch; harness_replay is linked via harness_replays
    -- (§4.6), so it carries no column-level NOT NULL requirement here.
    CONSTRAINT ck_llm_calls_identity CHECK (
        CASE call_type
            WHEN 'epoch_advice'        THEN run_pk IS NOT NULL
            WHEN 'run_config'          THEN iteration_number IS NOT NULL AND environment IS NOT NULL
            WHEN 'consolidate_stage1'  THEN iteration_number IS NOT NULL AND environment IS NOT NULL
            WHEN 'consolidate_stage2'  THEN iteration_number IS NOT NULL
            WHEN 'harness_replay'      THEN true
            WHEN 'trade_commentary'    THEN cycle_id IS NOT NULL
            WHEN 'trade_alarm'         THEN cycle_id IS NOT NULL
            WHEN 'event_significance'  THEN cycle_id IS NOT NULL
            ELSE false
        END
    )
);
CREATE INDEX idx_llm_calls_run ON llm_calls (run_pk);
CREATE INDEX idx_llm_calls_cycle ON llm_calls (cycle_id);

-- intent_records: the §2.4 five-block bet slip, blocks 1-4 (written once at call time,
-- no UPDATEs). mode + lever discriminate shadow L1 / U1 stop / live L2 / trade-time MT.
CREATE TABLE intent_records (
    intent_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    llm_call_id        BIGINT NOT NULL REFERENCES llm_calls (llm_call_id),
    coach              TEXT NOT NULL CHECK (coach IN ('meta_trainer', 'meta_trader')),
    lever              TEXT NOT NULL CHECK (lever IN
        ('L1_reward_weights', 'L2_hyperparams', 'U1_stop',
         'MT_commentary', 'MT_alarm', 'MT_pre_event')),   -- A17 inclusion rule
    mode               TEXT NOT NULL CHECK (mode IN ('shadow', 'live')),
    run_pk             BIGINT REFERENCES training_runs (run_pk),  -- mid-fold; NULL for L2 + trade-time
    iteration_number   SMALLINT,
    environment        TEXT CHECK (environment IN ('equity', 'crypto')),
    algorithm          TEXT CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    epoch              INTEGER,             -- mid-fold only, dual-unit
    timestep           BIGINT,              -- mid-fold only, dual-unit
    pct_complete       DOUBLE PRECISION,    -- mid-fold only, dual-unit
    -- Block 2 Evidence: self-contained snapshot (graders never join).
    evidence           JSONB NOT NULL,
    -- Block 3 Proposal: the change or explicit no-change + rationale (applied lives in sidecar, A13).
    proposal           JSONB NOT NULL,
    -- Block 4 Bet.
    bet_metric         TEXT NOT NULL,       -- fixed menu; registry declares units per metric (A9)
    bet_direction      TEXT NOT NULL CHECK (bet_direction IN ('up', 'down')),
    bet_baseline_value DOUBLE PRECISION NOT NULL,  -- metric at pull time
    horizon_spec       JSONB NOT NULL,      -- system-written, never coach-chosen (D-T2.8)
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Per-lever identity CHECK (spec §4.4 block 1). Trade-time levers carry env +
    -- flagged algo + deployed iteration and NULL run_pk; L2 is season-level (NULL
    -- run_pk, iteration+env); L1/U1 are mid-fold (run_pk NOT NULL).
    CONSTRAINT ck_intent_lever_identity CHECK (
        CASE lever
            WHEN 'L1_reward_weights' THEN run_pk IS NOT NULL
            WHEN 'U1_stop'           THEN run_pk IS NOT NULL
            WHEN 'L2_hyperparams'    THEN run_pk IS NULL
                                          AND iteration_number IS NOT NULL
                                          AND environment IS NOT NULL
            WHEN 'MT_commentary'     THEN run_pk IS NULL
                                          AND environment IS NOT NULL
                                          AND algorithm IS NOT NULL
                                          AND iteration_number IS NOT NULL
            WHEN 'MT_alarm'          THEN run_pk IS NULL
                                          AND environment IS NOT NULL
                                          AND algorithm IS NOT NULL
                                          AND iteration_number IS NOT NULL
            WHEN 'MT_pre_event'      THEN run_pk IS NULL
                                          AND environment IS NOT NULL
                                          AND algorithm IS NOT NULL
                                          AND iteration_number IS NOT NULL
            ELSE false
        END
    )
);
CREATE INDEX idx_intent_records_llm_call ON intent_records (llm_call_id);
CREATE INDEX idx_intent_records_lever ON intent_records (coach, lever, mode);

-- intent_applications (A13): restores no-UPDATE. The proposal is written at call time;
-- the application lands later in a different process (trainer start for L2), after clamps.
-- Proposal != applied stays visible by design; no application row = rejected/never-landed.
CREATE TABLE intent_applications (
    intent_id  BIGINT PRIMARY KEY REFERENCES intent_records (intent_id),  -- UNIQUE via PK
    applied    JSONB NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- intent_verdicts (block 5, grader script only, append-only). Every intent gets a
-- verdict eventually (A16 terminal-verdict guarantee); regrades are new rows.
CREATE TABLE intent_verdicts (
    verdict_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- surrogate PK (A16)
    intent_id       BIGINT NOT NULL REFERENCES intent_records (intent_id),
    grader_version  SMALLINT NOT NULL,
    actual_value    DOUBLE PRECISION,        -- metric at horizon
    direction_match BOOLEAN,                 -- intent-aware verdict (D-T1.5)
    menu_consistent BOOLEAN,                 -- diagnosis -> correction-menu check
    excluded        BOOLEAN NOT NULL DEFAULT false,
    excluded_reason TEXT CHECK (excluded_reason IS NULL OR excluded_reason IN
        ('new_fold_residue', 'event_shock', 'horizon_unreachable')),  -- §2.6 + §3.7
    graded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (intent_id, grader_version)       -- A16: regrades are new rows, never UPDATEs
);
CREATE INDEX idx_intent_verdicts_intent ON intent_verdicts (intent_id);

-- ensemble_weight_history.intent_id gained its column in V002 without the FK (the FK
-- target intent_records did not exist yet). Backfill it now.
ALTER TABLE ensemble_weight_history
    ADD CONSTRAINT fk_ewh_intent FOREIGN KEY (intent_id) REFERENCES intent_records (intent_id);

-- A14 volume cap: MT commentary is capped at <=1 intent record per inference cycle.
-- Enforced by two partial UNIQUE indexes: one MT_commentary intent per llm_call, and
-- one trade_commentary llm_call per cycle -> at most one MT_commentary intent per cycle.
CREATE UNIQUE INDEX uq_mt_commentary_per_cycle
    ON intent_records (llm_call_id)
    WHERE lever = 'MT_commentary';
CREATE UNIQUE INDEX uq_llm_commentary_cycle
    ON llm_calls (cycle_id) WHERE call_type = 'trade_commentary';
