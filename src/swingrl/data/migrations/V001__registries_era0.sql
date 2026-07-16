-- Spec §4.1 (D-T3.4, A7, A7b). Registries are written ONLY by human-approved migrations.

CREATE TABLE gate_versions (
    gate_version_id     SMALLINT PRIMARY KEY,          -- surrogate (P-A2 / proposed A29)
    gate_type           TEXT NOT NULL CHECK (gate_type IN ('per_fold', 'ensemble')),
    version_number      SMALLINT NOT NULL,
    definition          JSONB NOT NULL,                -- machine-readable rules, units per key
    derivation_evidence TEXT,
    approved_by         TEXT NOT NULL,
    approved_at         TIMESTAMPTZ NOT NULL,
    UNIQUE (gate_type, version_number)
);

CREATE TABLE eras (
    era_id                 SMALLINT PRIMARY KEY,
    reason                 TEXT NOT NULL,
    gate_version_per_fold  SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),
    gate_version_ensemble  SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),
    first_iteration        SMALLINT NOT NULL,
    started_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- A7 monotonicity: enforced procedurally — the runner is the only writer and each new
-- era migration must assert first_iteration > max(existing). (CHECK cannot see other rows.)

INSERT INTO gate_versions
    (gate_version_id, gate_type, version_number, definition, derivation_evidence, approved_by, approved_at)
VALUES
    (0, 'per_fold', 0,
     '{"schema_version": 1, "sharpe_min": 0.7, "mdd_max_frac": 0.15, "profit_factor_min": 1.5, "overfitting_gap_max": 0.20}',
     'Pre-CPS legacy per-fold gate (spec §2.8); retro-registered at era-0 bootstrap',
     'era0-bootstrap-migration', now()),
    (1, 'ensemble', 0,
     '{"schema_version": 1, "sharpe_min": 1.0, "mdd_abs_max_frac": 0.15}',
     'Pre-CPS legacy ensemble gate (spec §2.8); retro-registered at era-0 bootstrap',
     'era0-bootstrap-migration', now());

INSERT INTO eras (era_id, reason, gate_version_per_fold, gate_version_ensemble, first_iteration)
VALUES (0, 'Pre-redesign legacy: iterations 0-4 under the pre-CPS gate (D-T3.18 kept evidence)', 0, 1, 0);

-- Back-stamp the kept rows (574 in production; count-agnostic by design).
ALTER TABLE backtest_results
    ADD COLUMN era_id SMALLINT NOT NULL DEFAULT 0 REFERENCES eras (era_id),
    ADD COLUMN gate_version_id SMALLINT NOT NULL DEFAULT 0 REFERENCES gate_versions (gate_version_id);
ALTER TABLE iteration_results
    ADD COLUMN era_id SMALLINT NOT NULL DEFAULT 0 REFERENCES eras (era_id),
    ADD COLUMN gate_version_ensemble_id SMALLINT NOT NULL DEFAULT 1 REFERENCES gate_versions (gate_version_id);
