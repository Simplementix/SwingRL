-- Spec §4.8 weakness profiles (D-T3.17) + operator_actions (N14, §4.14 misc decision)
-- + the six derived views (§4.2 rule 5: never store the derivable). All JSONB payloads
-- carry a "schema_version" key (A8), enforced by the writers (not a DB constraint).

-- ---------------------------------------------------------------------------
-- §4.8 weakness_profiles — the shared scouting reports. One row per
-- (environment, algorithm, failure_mode, version); append-only versioning
-- (revisions are new rows, latest version active). Writers: maintenance script +
-- consolidation pipeline only — no LLM response mutates a profile (D-MT.6).
-- ---------------------------------------------------------------------------
CREATE TABLE weakness_profiles (
    weakness_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    environment     TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    algorithm       TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    failure_mode    TEXT NOT NULL CHECK (failure_mode IN
        ('trade_shy', 'poor_selection', 'single_disaster', 'churning',
         'slippage_sensitivity', 'event_shock_sensitivity', 'regime_transition_lag')),
                                          -- cps_diagnosis.py taxonomy + live-side modes;
                                          --   spec trailing "…" left open (one vocabulary)
    signature       JSONB NOT NULL,       -- conditions + identifying metric pattern, units per key
    early_indicators JSONB,               -- what the Meta-Trader watches to catch it early
    confidence      REAL,                 -- 0-1; script-computed from evidence support
    version         SMALLINT NOT NULL DEFAULT 1,   -- append-only versioning; revisions are new rows
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
                                          -- trained-out weaknesses retire, never delete
    seed_provenance JSONB,                -- doc-seeded entries: {"doc": ..., "section": ...}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (environment, algorithm, failure_mode, version)   -- append-only versioning key
);
CREATE INDEX idx_weakness_profiles_env_algo ON weakness_profiles (environment, algorithm);

-- weakness_evidence: same polymorphic shape as pattern_sources (§4.5). source_id
-- addresses a row in the named table — no FK, the target varies (A11/A16/A21). May
-- point at structured records AND patterns (a confirmed pattern graduating into the
-- career file keeps its full lineage unbroken).
CREATE TABLE weakness_evidence (
    weakness_id  BIGINT NOT NULL REFERENCES weakness_profiles (weakness_id),
    source_table TEXT NOT NULL CHECK (source_table IN
        ('fold_results', 'backtest_trades', 'intent_verdicts', 'fill_quality',
         'inference_cycles', 'patterns')),
    source_id    BIGINT NOT NULL,   -- polymorphic addressability; no FK — the target table varies
    PRIMARY KEY (weakness_id, source_table, source_id)
);
CREATE INDEX idx_weakness_evidence_source ON weakness_evidence (source_table, source_id);

-- ---------------------------------------------------------------------------
-- N14 operator_actions — append-only record of human interventions taken outside
-- the pre-built slots (ladder demotions/benches, manual promotions, config
-- actuations approved off a Discord recommendation). Append-only (§4.2 rule 4).
-- ---------------------------------------------------------------------------
CREATE TABLE operator_actions (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor        TEXT NOT NULL,           -- who performed the intervention
    action_type  TEXT NOT NULL,           -- e.g. demote, bench, promote, config_change
    target_table TEXT,                    -- what it acted on (nullable — some actions are global)
    target_id    BIGINT,                  -- polymorphic row reference (nullable)
    reason       TEXT NOT NULL,           -- every intervention carries its why
    payload      JSONB,                   -- optional structured detail (schema_version inside)
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_operator_actions_created ON operator_actions (created_at);

-- ===========================================================================
-- Derived views (§4.2 rule 5 — a stored copy can drift; a view cannot).
-- ===========================================================================

-- v_consolidation_corpus (§4.6 / A19 — the structural quarantine boundary): the
-- ONLY surface consolidation reads, defined per record type. Run-scoped tables are
-- restricted to CANONICAL runs (A6: highest completed attempt) with run_type IN
-- ('season','reference'); a non-run-scoped allowlist adds canonical season_results,
-- L2 intents + verdicts, and trade-time records; everything harness-tagged is
-- excluded (Stage 1 via run_type, Stage-2 replays via call_type = 'harness_replay').
-- S4 criterion 5 keys off this definition. source_id is TEXT so the uniform
-- (source_table, source_id) address covers both BIGINT-keyed rows and trades.trade_id.
CREATE VIEW v_consolidation_corpus AS
WITH canonical_run AS (
    SELECT DISTINCT ON (iteration_number, environment, algorithm, fold_number, run_type)
        run_pk, run_type
    FROM training_runs
    WHERE status = 'completed'
    ORDER BY iteration_number, environment, algorithm, fold_number, run_type, attempt DESC
),
corpus_run AS (   -- canonical season/reference runs only (A6 + §4.6 run-scoped rule)
    SELECT run_pk FROM canonical_run WHERE run_type IN ('season', 'reference')
)
-- run-scoped tables: epoch_snapshots
SELECT 'epoch_snapshots'::text AS source_table, es.id::text AS source_id,
       es.run_pk, tr.iteration_number, tr.environment, tr.algorithm
FROM epoch_snapshots es
JOIN corpus_run cr ON cr.run_pk = es.run_pk
JOIN training_runs tr ON tr.run_pk = es.run_pk
UNION ALL
SELECT 'fold_results', fr.id::text, fr.run_pk, tr.iteration_number, tr.environment, tr.algorithm
FROM fold_results fr
JOIN corpus_run cr ON cr.run_pk = fr.run_pk
JOIN training_runs tr ON tr.run_pk = fr.run_pk
UNION ALL
SELECT 'backtest_trades', bt.id::text, bt.run_pk, tr.iteration_number, tr.environment, tr.algorithm
FROM backtest_trades bt
JOIN corpus_run cr ON cr.run_pk = bt.run_pk
JOIN training_runs tr ON tr.run_pk = bt.run_pk
UNION ALL
-- run-scoped llm_calls (epoch_advice etc.); harness_replay explicitly excluded
SELECT 'llm_calls', lc.llm_call_id::text, lc.run_pk, tr.iteration_number, tr.environment,
       tr.algorithm
FROM llm_calls lc
JOIN corpus_run cr ON cr.run_pk = lc.run_pk
JOIN training_runs tr ON tr.run_pk = lc.run_pk
WHERE lc.call_type <> 'harness_replay'
UNION ALL
-- run-scoped intent_records (L1/U1 mid-fold — run_pk NOT NULL)
SELECT 'intent_records', ir.intent_id::text, ir.run_pk, tr.iteration_number, tr.environment,
       tr.algorithm
FROM intent_records ir
JOIN corpus_run cr ON cr.run_pk = ir.run_pk
JOIN training_runs tr ON tr.run_pk = ir.run_pk
UNION ALL
-- non-run-scoped allowlist: season_results (canonical result_version only)
SELECT 'season_results', sr.id::text, NULL::bigint, sr.iteration_number, sr.environment,
       CASE WHEN sr.scope IN ('ppo', 'a2c', 'sac') THEN sr.scope ELSE NULL END
FROM season_results sr
WHERE sr.result_version = (
    SELECT max(s2.result_version) FROM season_results s2
    WHERE s2.iteration_number = sr.iteration_number
      AND s2.environment = sr.environment
      AND s2.scope = sr.scope
)
UNION ALL
-- non-run-scoped allowlist: L2 intent_records (run_pk NULL, lever L2_hyperparams)
SELECT 'intent_records', ir.intent_id::text, NULL::bigint, ir.iteration_number, ir.environment,
       ir.algorithm
FROM intent_records ir
WHERE ir.run_pk IS NULL AND ir.lever = 'L2_hyperparams'
UNION ALL
-- non-run-scoped allowlist: L2 intent_verdicts (verdicts of L2 intents)
SELECT 'intent_verdicts', iv.verdict_id::text, NULL::bigint, ir.iteration_number,
       ir.environment, ir.algorithm
FROM intent_verdicts iv
JOIN intent_records ir ON ir.intent_id = iv.intent_id
WHERE ir.run_pk IS NULL AND ir.lever = 'L2_hyperparams'
UNION ALL
-- trade-time records: inference_cycles (mode-tagged)
SELECT 'inference_cycles', ic.cycle_id::text, NULL::bigint, ic.deployed_iteration,
       ic.environment, NULL::text
FROM inference_cycles ic
UNION ALL
-- trade-time records: cycle_algo_proposals
SELECT 'cycle_algo_proposals', cap.id::text, NULL::bigint, ic.deployed_iteration,
       ic.environment, cap.algorithm
FROM cycle_algo_proposals cap
JOIN inference_cycles ic ON ic.cycle_id = cap.cycle_id
UNION ALL
-- trade-time records: trades with a cycle_id (adjustment/reconciliation rows excluded)
SELECT 'trades', t.trade_id, NULL::bigint, ic.deployed_iteration, ic.environment, NULL::text
FROM trades t
JOIN inference_cycles ic ON ic.cycle_id = t.cycle_id
WHERE t.cycle_id IS NOT NULL
UNION ALL
-- trade-time records: fill_quality
SELECT 'fill_quality', fq.id::text, NULL::bigint, ic.deployed_iteration, ic.environment,
       NULL::text
FROM fill_quality fq
JOIN trades t ON t.trade_id = fq.trade_id
JOIN inference_cycles ic ON ic.cycle_id = t.cycle_id;

-- v_l2_settings_history (§4.4 / D-T2.11): one row per (environment, algorithm,
-- iteration). Base = canonical algo-scope season_results (ensemble excluded — no
-- hyperparams_used). source_intent_id links the live L2 pull that drove the HPs
-- (NULL = baseline/reference season, which runs coach-free). hp_delta is surfaced as
-- the (hyperparams_used, prev_hyperparams_used) pair — a JSONB structural diff is not
-- a single clean SQL expression; the K-season prompt digest renders the delta.
CREATE VIEW v_l2_settings_history AS
WITH l2_seasons AS (
    SELECT sr.*
    FROM season_results sr
    WHERE sr.scope IN ('ppo', 'a2c', 'sac')
      AND sr.result_version = (
          SELECT max(s2.result_version) FROM season_results s2
          WHERE s2.iteration_number = sr.iteration_number
            AND s2.environment = sr.environment
            AND s2.scope = sr.scope
      )
)
SELECT
    ls.environment,
    ls.scope AS algorithm,
    ls.iteration_number,
    ls.era_id,
    ls.coach_config,
    ls.hyperparams_used,
    lag(ls.hyperparams_used) OVER w AS prev_hyperparams_used,
    ls.cps_v1,
    ls.cps_v2,
    ls.cps_v3,
    ls.cps_components,
    ls.cps_v1 - lag(ls.cps_v1) OVER w AS cps_v1_delta_vs_prev,
    li.intent_id AS source_intent_id,
    li.proposal AS proposed_change,
    li.applied_change,
    li.direction_match,
    li.actual_value
FROM l2_seasons ls
LEFT JOIN LATERAL (
    SELECT ir.intent_id, ir.proposal, ia.applied AS applied_change,
           iv.direction_match, iv.actual_value
    FROM intent_records ir
    LEFT JOIN intent_applications ia ON ia.intent_id = ir.intent_id
    LEFT JOIN intent_verdicts iv ON iv.intent_id = ir.intent_id
    WHERE ir.lever = 'L2_hyperparams'
      AND ir.mode = 'live'
      AND ir.iteration_number = ls.iteration_number
      AND ir.environment = ls.environment
      AND (ir.algorithm = ls.scope OR ir.algorithm IS NULL)
    ORDER BY ir.intent_id DESC, iv.grader_version DESC NULLS LAST
    LIMIT 1
) li ON true
WINDOW w AS (PARTITION BY ls.environment, ls.scope ORDER BY ls.iteration_number);

-- v_lever_track_record (§4.4 / S8): aggregation over intent_records ⋈ intent_verdicts
-- grouped by (coach, lever, scope) — scope = (environment, algorithm). coach_config is
-- pulled through from the canonical ensemble season_results for the intent's
-- (iteration, env) — a season-wide staircase stamp (identical across scopes), joined
-- once via LATERAL to avoid fan-out; NULL for mid-fold intents with no iteration.
-- Feeds the coach's prompts each iteration and the §2.7 ladder.
CREATE VIEW v_lever_track_record AS
SELECT
    ir.coach,
    ir.lever,
    ir.environment,
    ir.algorithm,
    cc.coach_config,
    count(*)                                                        AS total_verdicts,
    count(*) FILTER (WHERE iv.excluded)                             AS excluded_count,
    count(*) FILTER (WHERE NOT iv.excluded)                         AS graded_count,
    count(*) FILTER (WHERE NOT iv.excluded AND iv.direction_match)  AS direction_match_count,
    count(*) FILTER (WHERE NOT iv.excluded AND iv.menu_consistent)  AS menu_consistent_count,
    avg(CASE WHEN NOT iv.excluded AND iv.direction_match THEN 1.0
             WHEN NOT iv.excluded THEN 0.0 END)                     AS direction_match_frac
FROM intent_records ir
JOIN intent_verdicts iv ON iv.intent_id = ir.intent_id
LEFT JOIN LATERAL (
    SELECT s.coach_config
    FROM season_results s
    WHERE s.iteration_number = ir.iteration_number
      AND s.environment = ir.environment
      AND s.scope = 'ensemble'
    ORDER BY s.result_version DESC
    LIMIT 1
) cc ON true
GROUP BY ir.coach, ir.lever, ir.environment, ir.algorithm, cc.coach_config;

-- v_consolidator_quality (§4.4 / A18): the third LLM's track record. Pattern
-- confirmation/contradiction tallies grouped by the producing consolidator call's
-- prompt_version + model. patterns carry no FK to their producing call, so the join is
-- reconstructed on (created_iteration, environment) against the consolidator's
-- consolidate_stage1/stage2 calls — the natural producer per A15's identity matrix.
-- Sustained contradiction dominance ⇒ patterns withheld from prompts (application layer).
CREATE VIEW v_consolidator_quality AS
SELECT
    lc.prompt_version,
    lc.model,
    count(DISTINCT p.pattern_id)                      AS pattern_count,
    sum(p.confirmation_count)                         AS total_confirmations,
    sum(p.contradiction_count)                        AS total_contradictions,
    CASE WHEN sum(p.confirmation_count) + sum(p.contradiction_count) > 0
         THEN sum(p.confirmation_count)::double precision
              / (sum(p.confirmation_count) + sum(p.contradiction_count))
         ELSE NULL END                                AS confirmation_ratio
FROM patterns p
JOIN llm_calls lc
    ON lc.coach = 'consolidator'
   AND lc.call_type IN ('consolidate_stage1', 'consolidate_stage2')
   AND lc.iteration_number = p.created_iteration
   AND (lc.environment = p.environment
        OR (lc.environment IS NULL AND p.environment IS NULL))
GROUP BY lc.prompt_version, lc.model;

-- v_pattern_effectiveness (§4.5): presentations → llm_calls → season/fold results.
-- Per pattern: how often it was presented and how the seasons/folds it was presented
-- into performed. A presenting call is fold-scoped (run_pk → fold_results) and/or
-- season-scoped (iteration+env → the canonical ensemble season_results). Replaces the
-- retired pattern_outcomes table.
CREATE VIEW v_pattern_effectiveness AS
SELECT
    pp.pattern_id,
    count(*)                          AS presentation_count,
    count(DISTINCT pp.llm_call_id)    AS distinct_call_count,
    avg(fr.oos_sharpe_annualized)     AS mean_presented_fold_oos_sharpe,
    avg(sr.cps_v2)                    AS mean_presented_season_cps_v2
FROM pattern_presentations pp
JOIN llm_calls lc ON lc.llm_call_id = pp.llm_call_id
LEFT JOIN fold_results fr ON fr.run_pk = lc.run_pk
LEFT JOIN season_results sr
    ON sr.iteration_number = lc.iteration_number
   AND sr.environment = lc.environment
   AND sr.scope = 'ensemble'
   AND sr.result_version = (
       SELECT max(s2.result_version) FROM season_results s2
       WHERE s2.iteration_number = sr.iteration_number
         AND s2.environment = sr.environment
         AND s2.scope = sr.scope
   )
GROUP BY pp.pattern_id;

-- v_live_transfer (§4.7): "did the edge survive contact?" Per (env, algo,
-- deployed_iteration): backtest trade-rate + per-trade distributions (from
-- backtest_trades) vs live participation (from cycle_algo_proposals ⋈ cycles), and
-- expected vs realized cost. Live per-trade cost is cycle-level (the blended fill), not
-- per-algo attributable — the "which player drove this trade" formula is the future
-- Meta-Trader spec's; here live cost joins on (env, deployed_iteration).
CREATE VIEW v_live_transfer AS
WITH bt_agg AS (
    SELECT tr.environment, tr.algorithm, tr.iteration_number,
           count(*)                        AS backtest_trade_count,
           avg(bt.cost_frac)               AS backtest_mean_cost_frac,
           avg(bt.realized_pnl_frac)       AS backtest_mean_realized_pnl_frac
    FROM backtest_trades bt
    JOIN training_runs tr ON tr.run_pk = bt.run_pk
    GROUP BY tr.environment, tr.algorithm, tr.iteration_number
),
live_agg AS (
    SELECT ic.environment, cap.algorithm, ic.deployed_iteration AS iteration_number,
           count(DISTINCT ic.cycle_id)     AS live_cycle_count,
           avg(cap.weight_in_blend_frac)   AS live_mean_blend_weight_frac
    FROM cycle_algo_proposals cap
    JOIN inference_cycles ic ON ic.cycle_id = cap.cycle_id
    GROUP BY ic.environment, cap.algorithm, ic.deployed_iteration
),
live_cost AS (   -- cycle-level (blended fill) cost, per (env, deployed_iteration)
    SELECT ic.environment, ic.deployed_iteration AS iteration_number,
           avg(fq.realized_cost_frac)      AS live_mean_realized_cost_frac,
           avg(fq.expected_cost_frac)      AS live_mean_expected_cost_frac
    FROM fill_quality fq
    JOIN trades t ON t.trade_id = fq.trade_id
    JOIN inference_cycles ic ON ic.cycle_id = t.cycle_id
    GROUP BY ic.environment, ic.deployed_iteration
)
SELECT
    coalesce(b.environment, l.environment)         AS environment,
    coalesce(b.algorithm, l.algorithm)             AS algorithm,
    coalesce(b.iteration_number, l.iteration_number) AS deployed_iteration,
    b.backtest_trade_count,
    b.backtest_mean_cost_frac,
    b.backtest_mean_realized_pnl_frac,
    l.live_cycle_count,
    l.live_mean_blend_weight_frac,
    lc.live_mean_realized_cost_frac,
    lc.live_mean_expected_cost_frac
FROM bt_agg b
FULL OUTER JOIN live_agg l
    ON l.environment = b.environment
   AND l.algorithm = b.algorithm
   AND l.iteration_number = b.iteration_number
LEFT JOIN live_cost lc
    ON lc.environment = coalesce(l.environment, b.environment)
   AND lc.iteration_number = coalesce(l.iteration_number, b.iteration_number);
