-- Spec §4.3 (D-T3.3, D-T3.5, D-T3.16). Training-record tables.
-- All JSONB payloads carry a "schema_version" key (A8), enforced by the writers.

CREATE TABLE epoch_snapshots (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_pk           BIGINT NOT NULL REFERENCES training_runs (run_pk),  -- full identity via one join
    epoch            INTEGER,             -- count
    timestep         BIGINT,              -- absolute steps (dual-unit pair w/ pct_complete, D-T2.7)
    pct_complete     REAL,                -- fraction 0-1
    mean_reward      DOUBLE PRECISION,    -- shaped-reward units (reward_wrapper pathway)
    learner_metrics  JSONB NOT NULL,      -- per-algo key contract (PPO kl/clip; SAC actor/critic
                                          --   loss, ent_coef) — kills the PPO-only-keys bug
    window_short     JSONB,               -- {pct, steps, sharpe_annualized, mdd_frac, win_rate,
                                          --   trade_rate} — acute detector
    window_trend     JSONB,               -- same shape; decision-basis window (§2.6)
    reward_weights   JSONB,               -- fractions, sum 1.0
    notable_event    TEXT,                -- nullable; enum §4.10, rate-limited per (type, trend window)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_epoch_snapshots_run ON epoch_snapshots (run_pk);

CREATE TABLE fold_results (
    id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- surrogate PK (A11)
    run_pk                 BIGINT NOT NULL UNIQUE REFERENCES training_runs (run_pk),
                                          -- one box score per run; read-time dedup dies
    era_id                 SMALLINT NOT NULL REFERENCES eras (era_id),                    -- explicit stamp
    gate_version_id        SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),  -- A29 surrogate
    seed                   INTEGER NOT NULL,        -- denorm from the spine (A11 / D-T2.11)
    fold_role              TEXT CHECK (fold_role IN ('neutral', 'chronic_failure', 'disaster')),
                                          -- §2.3 fold selection; spec enum trailing "…" left open
    fold_start_ts          TIMESTAMPTZ,  -- UTC; TIMESTAMPTZ not DATE — crypto 4H folds intra-day (A11)
    fold_end_ts            TIMESTAMPTZ,  -- UTC
    oos_return_frac        DOUBLE PRECISION,   -- fraction
    oos_sharpe_annualized  DOUBLE PRECISION,   -- annualized
    oos_sortino_annualized DOUBLE PRECISION,   -- annualized
    oos_calmar             DOUBLE PRECISION,   -- ratio
    oos_mdd_frac           DOUBLE PRECISION,   -- fraction >= 0; feeds CPS max_mdd, S2
    is_sharpe_annualized   DOUBLE PRECISION,   -- annualized; overfit check input
    is_return_frac         DOUBLE PRECISION,   -- fraction; overfit check input
    overfitting_gap        DOUBLE PRECISION,   -- IS - OOS Sharpe
    overfitting_class      TEXT,
    profit_factor          DOUBLE PRECISION,   -- ratio
    trade_count            INTEGER,            -- count; feeds the activity floor
    win_rate_frac          DOUBLE PRECISION,   -- fraction
    max_single_loss_frac   DOUBLE PRECISION,   -- fraction of capital (CPS-v2 units fix; dollars dropped)
    initial_capital_usd    DOUBLE PRECISION,   -- USD; dollars recoverable as frac x base
    gate_passed            BOOLEAN,
    gate_components        JSONB,     -- per key: each §2.8 requirement w/ threshold+actual+pass
    hmm_p_bull             DOUBLE PRECISION,   -- probability 0-1
    hmm_p_bear             DOUBLE PRECISION,   -- probability 0-1
    vix_mean               DOUBLE PRECISION,   -- index points
    turbulence_mean        DOUBLE PRECISION,   -- Mahalanobis-distance score (A27); era-1 comparability
                                          --   with the trade-time stamp (§4.7)
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_fold_results_run ON fold_results (run_pk);
CREATE INDEX idx_fold_results_era ON fold_results (era_id);
CREATE INDEX idx_fold_results_fold_start_ts ON fold_results (fold_start_ts);

CREATE TABLE season_results (
    id                            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    iteration_number              SMALLINT NOT NULL,
    environment                   TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    scope                         TEXT NOT NULL CHECK (scope IN ('ppo', 'a2c', 'sac', 'ensemble')),
    result_version                SMALLINT NOT NULL DEFAULT 1,  -- recomputes/re-runs are new rows,
                                          --   never UPDATEs (A10); canonical row = highest version
    era_id                        SMALLINT NOT NULL REFERENCES eras (era_id),  -- explicit stamp (§4.1)
    gate_version_per_fold         SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),  -- A29
    gate_version_ensemble         SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),  -- A29
    gate_passed                   BOOLEAN,
    gate_components               JSONB,   -- ensemble-scope rows: §2.8 ensemble-gate verdict + working
    coach_config                  JSONB NOT NULL,  -- staircase stamp: {"l1":"benched","l2":"live",
                                          --   "patterns_in_prompt":false,"reference_season":false} (D-T2.6)
    cps_v1                        DOUBLE PRECISION,   -- score
    cps_v2                        DOUBLE PRECISION,   -- score
    cps_v3                        DOUBLE PRECISION,   -- score; nullable
    cps_components                JSONB,   -- median_return_frac, max_mdd_frac, mean_winner_sharpe,
                                          --   pass_ratio, winners, fold_count
    worst_fold_number             SMALLINT,           -- joins to fold_results
    worst_fold_mdd_frac           DOUBLE PRECISION,   -- fraction; S2 reads this
    return_regression_delta_frac  DOUBLE PRECISION,   -- fraction; vs previous season, same env+scope
    hyperparams_used              JSONB,   -- per key; algo scopes only
    reward_weights_used           JSONB,   -- fractions
    ensemble_weights              JSONB,   -- fractions; ensemble scope only
    wall_clock_seconds            INTEGER,            -- seconds
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (iteration_number, environment, scope, result_version)
);
CREATE INDEX idx_season_results_era ON season_results (era_id);
CREATE INDEX idx_season_results_iter_env ON season_results (iteration_number, environment);

CREATE TABLE backtest_trades (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_pk               BIGINT NOT NULL REFERENCES training_runs (run_pk),  -- identity + era + quarantine
    bar_ts               TIMESTAMPTZ NOT NULL,   -- UTC; joinable to regime data
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    weight_delta_frac    DOUBLE PRECISION,   -- fraction; rebalance delta (target-weight system)
    price_usd            DOUBLE PRECISION,   -- USD
    cost_frac            DOUBLE PRECISION,   -- fraction; modeled transaction cost
    position_after_frac  DOUBLE PRECISION,   -- fraction
    realized_pnl_frac    DOUBLE PRECISION,   -- fraction; nullable; on position-reducing trades only
    UNIQUE (run_pk, bar_ts, symbol)   -- physical ceiling — duplicates bounce off the schema (§4.10)
);
CREATE INDEX idx_backtest_trades_run ON backtest_trades (run_pk);
