-- Spec §4.5 Patterns + lifecycle (D-T3.6, D-T3.7). The playbook + its lineage DAG.
-- Replaces consolidations/consolidation_sources + the superseded_by/conflicting_with
-- self-pointer columns + pattern_outcomes. Raw `memories` is retired: provenance points
-- only at STRUCTURED records. All JSONB payloads carry a "schema_version" key (A8),
-- enforced by the writers (not a DB constraint — same as V005).

-- patterns: playbook notes with a machine-readable core (claim JSONB NOT NULL).
CREATE TABLE patterns (
    pattern_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_iteration   SMALLINT,             -- iteration that produced the note
    environment         TEXT CHECK (environment IN ('equity', 'crypto')),  -- NULL = stage-2 cross-env
    stage               SMALLINT CHECK (stage IN (1, 2)),  -- 1 = per-env, 2 = cross-env
    era_id              SMALLINT NOT NULL REFERENCES eras (era_id),  -- old-rulebook notes identifiable
    category            TEXT CHECK (category IN
        ('trade_shy', 'poor_selection', 'single_disaster', 'churning')),
                                              -- cps_diagnosis.py taxonomy; spec trailing "…" left open
    claim               JSONB NOT NULL,       -- {scope:{env,algo,fold_role}, condition:{...},
                                              --   effect:{metric,direction,magnitude_frac}} — units per key
    prompt_text         TEXT,                 -- prompt rendering (allowed alongside the structured claim)
    confidence          REAL,                 -- 0-1; min 0.4 for prompt eligibility (application-enforced)
    qa_passed           BOOLEAN NOT NULL DEFAULT false,  -- not prompt-eligible until true (C6 §7 QA)
    qa_checks           JSONB,                -- per-criterion QA working
    status              TEXT NOT NULL CHECK (status IN
        ('active', 'conflicted', 'superseded', 'retired')),
    confirmation_count  INTEGER NOT NULL DEFAULT 0,  -- script-maintained (season-close claim re-check)
    contradiction_count INTEGER NOT NULL DEFAULT 0,  -- script-maintained; LLM never grades its own notes
    conflict_group_id   UUID,                 -- the dispute case file (NULL unless conflicted)
    resolved_at         TIMESTAMPTZ,
    resolution_method   TEXT CHECK (resolution_method IN ('evidence_dominance', 'scope_split', 'human')),
    retired_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_patterns_era ON patterns (era_id);
CREATE INDEX idx_patterns_status ON patterns (status);
CREATE INDEX idx_patterns_conflict_group ON patterns (conflict_group_id);

-- pattern_sources: provenance points at STRUCTURED records only (never raw memories).
-- source_id is polymorphic (addresses a row in the named table) — no FK, target varies (A11).
CREATE TABLE pattern_sources (
    pattern_id   BIGINT NOT NULL REFERENCES patterns (pattern_id),
    source_table TEXT NOT NULL CHECK (source_table IN
        ('fold_results', 'epoch_snapshots', 'intent_records', 'backtest_trades')),
    source_id    BIGINT NOT NULL,   -- polymorphic addressability; no FK — the target table varies
    PRIMARY KEY (pattern_id, source_table, source_id)
);
CREATE INDEX idx_pattern_sources_source ON pattern_sources (source_table, source_id);  -- reverse S4 trace

-- pattern_links: the lineage DAG (replaces the superseded_by/conflicting_with single
-- self-pointer — cannot express N-parent merges or M->N splits). Edges written only by
-- the consolidation/resolution scripts, atomically with the status change (no orphaned gens).
CREATE TABLE pattern_links (
    parent_pattern_id BIGINT NOT NULL REFERENCES patterns (pattern_id),
    child_pattern_id  BIGINT NOT NULL REFERENCES patterns (pattern_id),
    link_type         TEXT NOT NULL CHECK (link_type IN
        ('merged_into', 'split_into', 'refined_into')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (parent_pattern_id, child_pattern_id),
    CONSTRAINT ck_pattern_links_no_self CHECK (parent_pattern_id <> child_pattern_id)
);
CREATE INDEX idx_pattern_links_child ON pattern_links (child_pattern_id);  -- recursive ancestry

-- pattern_presentations: identity inherited through MANDATORY FKs — the NULL-iteration
-- failure class (4,933/9,575 rows) is structurally impossible. pattern_outcomes is retired
-- (effectiveness is a computed view: presentations -> llm_calls -> season/fold results).
CREATE TABLE pattern_presentations (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pattern_id   BIGINT NOT NULL REFERENCES patterns (pattern_id),
    llm_call_id  BIGINT NOT NULL REFERENCES llm_calls (llm_call_id),
    presented_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_pattern_presentations_pattern ON pattern_presentations (pattern_id);
CREATE INDEX idx_pattern_presentations_llm_call ON pattern_presentations (llm_call_id);
