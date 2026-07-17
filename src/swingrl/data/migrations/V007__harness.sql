-- Spec §4.6 Harness records (D-T3.12). Stage-1/2 harness runs are already spine-tagged
-- (run_type); these tables add the experiment layer: grouping, pre-registration, verdicts.
-- All JSONB payloads carry a "schema_version" key (A8), enforced by the writers.

-- harness_experiments: the pre-registered plan. pull_spec is written before any run
-- starts (D-T2.3's "judgment w/o training" separation depends on this existing first).
CREATE TABLE harness_experiments (
    experiment_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lever                TEXT NOT NULL CHECK (lever IN
        ('L1_reward_weights', 'L2_hyperparams', 'U1_stop',
         'MT_commentary', 'MT_alarm', 'MT_pre_event')),  -- same enum as intent_records.lever (§4.4)
    stage                SMALLINT NOT NULL CHECK (stage IN (1, 2)),  -- 1 mechanics, 2 judgment
    environment          TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    algorithm            TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    fold_number          SMALLINT NOT NULL,
    fold_role            TEXT NOT NULL CHECK (fold_role IN
        ('neutral', 'chronic_failure', 'disaster')),  -- neutral gate / chronic tryout (§2.3);
                                          --   same enum as fold_results.fold_role (V005)
    pull_spec            JSONB NOT NULL,  -- scripted pull direction/magnitude + expected:{metric,
                                          --   direction} — written before any run starts (pre-reg)
    min_run_length_steps BIGINT NOT NULL, -- floor per (algo, lever) from the implementation plan
    passed               BOOLEAN,         -- nullable; set at verdict time
    verdict_detail        JSONB,          -- per-seed-pair agreement + trade-activity-collapse check
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at          TIMESTAMPTZ
);
CREATE INDEX idx_harness_experiments_lever ON harness_experiments (lever);
CREATE INDEX idx_harness_experiments_env_algo ON harness_experiments (environment, algorithm);

-- harness_experiment_runs (Stage 1): one row per arm/seed-pair run. run_pk is the PK
-- (not a composite with experiment_id) — a training run is created for exactly one
-- experiment arm, so this also enforces "a run cannot belong to two experiments."
-- "Majority of seed-pairs agree" is computable straight from these keys.
CREATE TABLE harness_experiment_runs (
    experiment_id BIGINT NOT NULL REFERENCES harness_experiments (experiment_id),
    run_pk        BIGINT PRIMARY KEY REFERENCES training_runs (run_pk),
    arm           TEXT NOT NULL CHECK (arm IN ('pull', 'control')),
    seed_pair     SMALLINT NOT NULL
);
CREATE INDEX idx_harness_experiment_runs_experiment
    ON harness_experiment_runs (experiment_id, seed_pair);

-- harness_replays (Stage 2 film-room): a scripted quiz situation + expected judgment,
-- graded against the coach's actual llm_calls response. llm_call_id is a MANDATORY FK
-- (same never-orphaned-record discipline as V006's pattern_presentations) — with
-- prompt_version + model living on the linked llm_calls row, a flunked quiz identifies
-- which prompt or model.
CREATE TABLE harness_replays (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    experiment_id      BIGINT NOT NULL REFERENCES harness_experiments (experiment_id),
    llm_call_id        BIGINT NOT NULL REFERENCES llm_calls (llm_call_id),
    situation          JSONB NOT NULL,   -- the scripted scenario presented to the coach
    expected_response  JSONB NOT NULL,   -- the pre-registered correct judgment
    graded_consistent  BOOLEAN,          -- nullable; set at grading time
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_harness_replays_experiment ON harness_replays (experiment_id);
CREATE INDEX idx_harness_replays_llm_call ON harness_replays (llm_call_id);
