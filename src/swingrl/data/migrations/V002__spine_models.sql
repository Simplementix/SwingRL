-- Spec §4.1 (D-T3.1, A4/A5/A6/A12) + §4.4 models/ensemble_weight_history (D-T3.10, A22)

CREATE TABLE training_runs (
    run_pk            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    iteration_number  SMALLINT NOT NULL,
    environment       TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    algorithm         TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    fold_number       SMALLINT NOT NULL,
    run_type          TEXT NOT NULL CHECK (run_type IN
        ('season', 'reference', 'harness_stage1', 'harness_stage2', 'final_train', 'l1_reearn_control')),
    seed              INTEGER NOT NULL,       -- -1 sentinel allowed for era-0 backfill only (P-A1)
    attempt           SMALLINT NOT NULL DEFAULT 1,
    status            TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'aborted')),
    era_id            SMALLINT NOT NULL REFERENCES eras (era_id),
    code_version      TEXT NOT NULL,
    config_hash       TEXT,
    config_snapshot   JSONB,
    data_fingerprint  TEXT NOT NULL,
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    UNIQUE (iteration_number, environment, algorithm, fold_number, run_type, attempt)
);
CREATE INDEX idx_training_runs_iter_env_algo ON training_runs (iteration_number, environment, algorithm);
CREATE INDEX idx_training_runs_era ON training_runs (era_id);

CREATE TABLE models (
    model_id                       TEXT PRIMARY KEY,
    run_pk                         BIGINT NOT NULL REFERENCES training_runs (run_pk),
    artifact_path                  TEXT NOT NULL,
    vecnormalize_path              TEXT NOT NULL,
    artifact_sha256                TEXT,        -- NOT NULL from era 1 (A22); nullable for era-0 backfill
    vecnormalize_sha256            TEXT,
    training_window_start          DATE,
    training_window_end            DATE,
    converged_at_step              BIGINT,
    ensemble_weight_at_train_frac  DOUBLE PRECISION,
    status                         TEXT NOT NULL CHECK (status IN ('active', 'shadow', 'archived')),
    promoted_at                    TIMESTAMPTZ,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_models_run ON models (run_pk);

CREATE TABLE ensemble_weight_history (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id       TEXT NOT NULL REFERENCES models (model_id),
    weight_frac    DOUBLE PRECISION NOT NULL,
    set_by         TEXT NOT NULL CHECK (set_by IN ('training', 'meta_trader', 'human')),
    intent_id      BIGINT,   -- FK to intent_records added by V004
    effective_from TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_ewh_model_from ON ensemble_weight_history (model_id, effective_from DESC);
