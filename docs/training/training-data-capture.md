# Training Data Capture Reference

Living reference for SwingRL's pg16 schema. Source-of-truth for every table's purpose, writer hot-path, write cadence, and reader. Update when any DDL or writer/reader changes.

**Last verified against code:** 2026-04-16

**Schema source-of-truth:** `src/swingrl/data/postgres_schema.py` (837 lines, 36 tables, ~10 strategic indexes).

**Honest-gap policy:** every concrete claim is `file:line`-cited. Cardinality is stated as "writer cadence × dimension"; actual row counts are marked `UNKNOWN — needs DB query`. Tables with no SELECT in `src/` or `services/` are flagged `no readers found in src/` rather than imputing a likely consumer.

## Tables at a glance

36 tables, 4 logical clusters. Click each cluster header to jump.

| Cluster | Tables | Hot-path writer location |
|---------|--------|--------------------------|
| [Market data & features](#market-data--features-10-tables) | 10 | `data/`, `features/` |
| [Training & evaluation](#training--evaluation-6-tables) | 6 | `training/`, `agents/backtest.py`, `memory/training/` |
| [Memory system](#memory-system-7-tables) | 7 | `services/memory/` |
| [Live trading & observability](#live-trading--observability-13-tables) | 13 | `execution/`, `execution/risk/`, `scheduler/`, `shadow/`, `monitoring/` |

## Market data & features (10 tables)

### `ohlcv_daily`
- **Purpose:** Equity daily OHLCV. PK `(symbol, date)`. (`postgres_schema.py:39-53`)
- **Writers:**
  - `data/base.py:190-192` — `executemany_from_df(..., on_conflict="DO NOTHING")` from each ingestor's `_sync_to_db()` — fires once per scheduled incremental ingestion per symbol.
  - `data/gap_fill.py:345` — gap-fill backfill batches (equity).
- **Cardinality cadence:** 1 row per `(symbol, date)`. Total UNKNOWN — needs DB query.
- **Readers:** `data/verification.py:85,175,381`; `training/data_loader.py:178,181-182`; `features/pipeline.py:567,648`; `features/macro.py:43`; `data/cross_source.py:73`; `data/gap_fill.py:142`.

### `ohlcv_4h`
- **Purpose:** Crypto 4-hour OHLCV with `source` column. PK `(symbol, datetime)`. (`postgres_schema.py:54-67`)
- **Writers:** `data/base.py:190-192` (primary); `data/gap_fill.py:345` (gap-fill, `source='binance_global'`).
- **Cardinality cadence:** 1 row per `(symbol, 4H bar)`. Total UNKNOWN.
- **Readers:** `data/verification.py:118,197,228,383`; `training/data_loader.py:305,308-309`; `features/pipeline.py:609,612-614,670`; `features/macro.py:87`; `data/gap_fill.py:94`.

### `macro_features`
- **Purpose:** FRED macro series (VIXCLS, T10Y2Y, DFF, CPIAUCSL, UNRATE) with `release_date` for look-ahead-bias prevention. PK `(date, series_id)`. (`postgres_schema.py:69-77`)
- **Writers:** `data/fred.py:47::FREDIngestor` registers `_duckdb_table = "macro_features"` (L58). Writes flow via inherited `BaseIngestor.run()` → `data/base.py:383::executemany_from_df(..., on_conflict="DO NOTHING")` after Parquet sync. Macro-specific column mapping at `data/base.py:377-380`.
- **Cardinality cadence:** 1 row per `(date, series_id)`. Total UNKNOWN.
- **Readers:** `training/data_loader.py:48-51`; `features/pipeline.py:427-431`; `features/macro.py:44-66` (LATERAL JOIN); `execution/emergency.py:450`.

### `data_quarantine`
- **Purpose:** Validation-failed rows quarantined during ingestion (raw JSON + reason). PK `id` IDENTITY. (`postgres_schema.py:79-89`)
- **Writers:** `data/base.py:314-319::_store_quarantine()` — fires per quarantined row when ingestion validation fails.
- **Cardinality cadence:** 0..N rows per ingestion run, depending on validation failures. Total UNKNOWN.
- **Readers:** **no readers found in src/**. Write-only audit table.

### `data_ingestion_log`
- **Purpose:** One row per ingestor run with status, rows_inserted, errors_count, duration. PK `run_id` (TEXT). (`postgres_schema.py:91-103`)
- **Writers:** `data/base.py:257-272::_log_ingestion()` — once per `ingestor.run()` call.
- **Cardinality cadence:** 1 row per ingestor run per symbol. Total UNKNOWN.
- **Readers:** **no readers found in src/**.

### `fundamentals`
- **Purpose:** P/E, earnings growth, debt/equity, dividend yield, sector. PK `(symbol, date)`. (`postgres_schema.py:268-280`)
- **Writers:** `features/fundamentals.py:331::store_fundamentals()` — `executemany_from_df` with `ON CONFLICT (symbol,date) DO UPDATE`.
- **Cardinality cadence:** 1 row per `(symbol, date)` per fetch. Trigger frequency for `fetch_all()`: not traced.
- **Readers:** **no SELECT FROM fundamentals found in src/**. Note: `features/pipeline.py:174-175` stub-fills the four fundamental columns of `features_equity` with `0.0` rather than reading this table — the fetcher exists but is not yet wired into the feature pipeline. Cross-link: see `feature-catalog.md` "Known issues / fundamentals freshness".

### `corporate_actions`
- **Purpose:** Tracks splits / overnight gaps detected by spike-detection (`action_type`, `effective_date`, `ratio`, `processed`). PK `action_id` (TEXT). (`postgres_schema.py:438-448`)
- **Writers:** `data/corporate_actions.py:107-112::record_action()` — fires when `detect_overnight_spike()` flags an unrecorded spike (data-dependent, not scheduled).
- **Cardinality cadence:** 1 row per detected action. Total UNKNOWN.
- **Readers:** `data/corporate_actions.py:134-137::is_known_action()` — read-only existence check.

### `features_equity`
- **Purpose:** Pre-computed equity feature row per `(symbol, date)` — every column listed in `feature-catalog.md`. PK `(symbol, date)`. (`postgres_schema.py:224-245`)
- **Writers:** `features/pipeline.py:699-701::_store_equity_features()` — `executemany_from_df` with `ON CONFLICT DO UPDATE`. Trigger: invoked by `pipeline.compute_equity()` (orchestrator call-site not traced here — likely a scheduled job).
- **Cardinality cadence:** 1 row per `(symbol, date)`. Total UNKNOWN.
- **Readers:** `training/data_loader.py:157-169`; `features/pipeline.py:342-345`.

### `features_crypto`
- **Purpose:** Pre-computed crypto feature row per `(symbol, datetime)`. PK `(symbol, datetime)`. (`postgres_schema.py:247-266`)
- **Writers:** `features/pipeline.py:724-726::_store_crypto_features()` — `executemany_from_df` with `ON CONFLICT DO UPDATE`.
- **Cardinality cadence:** 1 row per `(symbol, 4H bar)`. Total UNKNOWN.
- **Readers:** `training/data_loader.py:284-296`; `features/pipeline.py:386-389`.

### `hmm_state_history`
- **Purpose:** Persisted HMM regime probabilities `(p_bull, p_bear, p_crisis, log_likelihood)` per refit. PK `(environment, date)`. Note `p_crisis` is **stored but not exposed** to the agent's observation (only `p_bull, p_bear` are — see `feature-catalog.md`). (`postgres_schema.py:282-297`)
- **Writers:** `features/hmm_regime.py:293::store_hmm_state()` (INSERT at `:317-321`) — fires per HMM refit. Refit cadence per env (proxy symbol window) not traced here.
- **Cardinality cadence:** 1 row per `(environment, date)` per refit. Total UNKNOWN.
- **Readers:** `training/data_loader.py:89-103`; `features/pipeline.py:469-482`; `memory/training/meta_orchestrator.py:406`.

## Training & evaluation (6 tables)

### `model_metadata`
- **Purpose:** Trained-model artifacts: paths, version, training window, `converged_at_step`, `ensemble_weight`, `validation_sharpe`. PK `model_id` (TEXT). (`postgres_schema.py:105-121`)
- **Writers:** `scripts/train_pipeline.py:1612::_write_model_metadata()` (idempotent INSERT … ON CONFLICT DO UPDATE; non-fatal); `scripts/train.py:488` (legacy path, same pattern).
- **Cardinality cadence:** ~6 rows per training cycle (3 algos × 2 envs). Total UNKNOWN.
- **Readers:** `execution/pipeline.py:422` (loads `algorithm`, `ensemble_weight` per inference cycle).
- **Honest gaps:** `validation_sharpe` always written as `None` — purpose unverified; no other writer found.

### `backtest_results`
- **Purpose:** Per-fold backtest metrics (IS + OOS Sharpe/Sortino/Calmar/MDD, profit factor, trade stats, regime context, overfitting class). PK `result_id` (UUID, TEXT). 41 columns. (`postgres_schema.py:124-169`)
- **Writers:**
  - `agents/backtest.py:598::_store_results()` — plain INSERT, fires per fold (baseline walk-forward).
  - `agents/backtest.py:724::store_fold_results_to_duckdb()` — richer iteration-N row including regime context (`hmm_p_bull/bear`, `vix_mean`, `yield_spread_mean`), in-sample surrogates, overfitting gap/class, dates, `is_control_fold` flag. (Function name says duckdb but actually writes pg16; legacy name kept.)
- **Cardinality cadence:** 1 row per `(iteration, environment, algorithm, fold_number)`. Per-iteration count varies by walk-forward fold count. Total UNKNOWN — historical reference: per memory, **564 backtest_results rows for iter 0-4** as of 2026-04-07.
- **Readers:** `reporting/iteration_report.py:114::load_fold_history()` (dedup'd latest per fold for CPS computation), `iteration_report.py:753` (pre-dedup count for audit column).
- **Honest gaps:** No DDL index on `(environment, iteration_number)` despite being a frequent query filter.

### `iteration_results`
- **Purpose:** Iteration-level ensemble row: ensemble Sharpe/MDD, gate pass/fail, per-algo weights & means, hyperparams JSON, CPS v1/v2/v3 scores, worst-fold markers, regression deltas. Composite uniqueness `(iteration_number, environment, run_type)` enables `ON CONFLICT DO UPDATE` for re-runs. 55 columns. (`postgres_schema.py:171-222`)
- **Writers:**
  - `agents/backtest.py:882::store_iteration_results_to_duckdb()` — base row with ensemble metrics, gate, weights, hyperparams, wall-clock, `memory_enabled`. (Function name says duckdb but actually writes pg16; legacy name kept.)
  - `reporting/iteration_report.py:612::persist_iteration_cps()` — UPDATE-then-INSERT pass that fills CPS columns (`cps_v1_multiplicative`, treatment/control splits, components JSON, worst-fold markers, regression delta, dedup audit).
- **Cardinality cadence:** 1 row per `(iteration_number, environment, run_type)`. Total UNKNOWN — historical reference: per memory, **10 rows for iter 0-4 × {equity, crypto}** as of 2026-04-07.
- **Readers:** `reporting/iteration_report.py:90::load_iteration_history()` (full SELECT for last N iters); `:772` (median_return for CPS v3 baseline); `:880` (CPS v1 + worst_fold_mdd for regression deltas).
- **Honest gaps:** UPDATE path in `persist_iteration_cps` does not touch `created_at` — CPS recompute timestamp is implicit / unobservable.

### `training_epochs`
- **Purpose:** Per-epoch training snapshot when cadence matches OR notable event fires. Schema and cadence already covered in [`reward-shaping.md`](reward-shaping.md). (`postgres_schema.py:300-322`)
- **Writers:** `memory/training/epoch_callback.py:252` — **buffered** in `_epoch_queue` during training, flushed by `flush_telemetry()` per fold (`epoch_callback.py:223-266`).
- **Cardinality cadence:** rows per epoch when `epoch % cadence == 0` OR `notable_event != None` (KL > 0.10 / MDD < -25.0). Per-fold yield: PPO ~1.4 (cadence 60), A2C ~4 (cadence 8000), SAC ~4 (cadence 40000) per the `epoch_callback.py:37-39` comment. Plus notable events. Total UNKNOWN.
- **Readers:** **no readers found in src/**. Consumed externally (dashboards / notebooks).

### `meta_decisions`
- **Purpose:** Per-decision audit of LLM HP-tuning / reward-weight / stop-training directives. `decision_json` is JSON-stringified TEXT. (`postgres_schema.py:325-336`)
- **Writers:** `memory/training/meta_orchestrator.py:538::_log_decision()` — direct synchronous INSERT after each LLM call.
- **Cardinality cadence:** 1 row per LLM decision event. Total UNKNOWN.
- **Readers:** **no readers found in src/**. Audit table for human review.
- **Honest gaps:** No index on `(run_id, algo, env)`. No DDL-level schema for `decision_json`.

### `reward_adjustments`
- **Purpose:** Two-pass trigger/outcome tracking for LLM-driven reward-weight adjustments. Schema, two-pass mechanics, and `effective` rule already covered in [`reward-shaping.md`](reward-shaping.md). (`postgres_schema.py:339-359`)
- **Writers:** `memory/training/epoch_callback.py:271` (trigger queue), `:286` (outcome queue) — **buffered**, flushed per-fold by `flush_telemetry()`.
- **Cardinality cadence:** rows per LLM-approved adjustment (gated by per-algo cooldown — see `reward-shaping.md`). Total UNKNOWN.
- **Readers:** **no readers found in src/**.

## Memory system (7 tables)

The memory subsystem lives largely in `services/memory/` (FastAPI service in its own container — `swingrl-memory`), with a thin client at `src/swingrl/memory/client.py`. Most writers and readers therefore reside under `services/memory/`.

### `memories`
- **Purpose:** Raw ingested narrative text, XML-wrapped, source-tagged (e.g., `training_epoch:equity:ppo`, `reward_adjustment:crypto:sac`). PK `id` IDENTITY; `archived` flag; indexes `idx_memories_source`, `idx_memories_created`. (`postgres_schema.py:551-559`)
- **Writers:** `services/memory/db.py:309-328::insert_memory()` (sync + async); called via HTTP POST `/ingest` (`services/memory/routers/core.py:66-77`) from `src/swingrl/memory/client.py:57-122` and `memory/training/epoch_callback.py`. Direct write per call (no batching).
- **Cardinality cadence:** rows per `(source, event)`. Per memory in `MEMORY.md`: **historical reference ~688K crypto SAC memories** observed in past iters (not extrapolation, observed). Per-iteration count UNKNOWN — needs DB query.
- **Readers:** `services/memory/db.py:379-443::get_memories()` / `get_memories_by_source_prefix()`; `services/memory/memory_agents/consolidate.py` (batches 200 unarchived memories for LLM consolidation, then calls `archive_memories()` at `db.py:445-465`); `services/memory/memory_agents/query.py` (pattern context retrieval).

### `consolidations`
- **Purpose:** LLM-synthesized patterns derived from raw memories. Stage 1 = per-env, Stage 2 = cross-env. Tracks `status` ∈ {`active`, `superseded`, `retired`}, `confidence`, `confirmation_count`, `conflicting_with`, `superseded_by`, `conflict_group_id`. Indexes `idx_consolidations_status`, `idx_consolidations_env_stage`, `idx_consolidations_category_status` (`postgres_schema.py:756-759`). (`postgres_schema.py:561-582`)
- **Writers:** `services/memory/db.py:506-574::insert_consolidation()`; `:703-730::update_consolidation_status()`; `:733-745::increment_confirmation()`. Trigger: `services/memory/memory_agents/consolidate.py:2135` after LLM synthesis; HTTP POST `/consolidate` invoked from `scripts/train_pipeline.py:567` after each iteration.
- **Cardinality cadence:** rows per pattern emitted by Stage-1 (per env) + Stage-2 (cross-env) per consolidation run. Yield is LLM-dependent. Total UNKNOWN.
- **Readers:** `services/memory/db.py:609-700::get_active_consolidations()`; `services/memory/memory_agents/consolidate.py` (dedup/conflict detection); `services/memory/memory_agents/query.py:35-42` (pattern context for `run_config` + `epoch_advice` advice); `src/swingrl/memory/training/meta_orchestrator.py:48-52` (count-only — gates LLM advice on `≥1 active pattern`).

### `consolidation_quality`
- **Purpose:** Per-LLM-call attempt count + accept/reject outcome for consolidation synthesis batches. (`postgres_schema.py:584-592`)
- **Writers:** `services/memory/db.py:850-873::log_consolidation_quality()` (fire-and-forget; exceptions swallowed).
- **Cardinality cadence:** ~1-2 rows per `/consolidate` call (Stage 1 per env + Stage 2). Total UNKNOWN.
- **Readers:** **no readers found in src/ or services/**. Audit-only.

### `consolidation_sources`
- **Purpose:** Many-to-many join from `consolidations.id` → `memories.id`. Composite PK `(consolidation_id, memory_id)`. Idempotent `ON CONFLICT DO NOTHING`. (`postgres_schema.py:594-600`)
- **Writers:** `services/memory/db.py:577-606::insert_consolidation_source()` / `insert_consolidation_sources()` (batch). Called from `services/memory/memory_agents/consolidate.py:2152` immediately after `insert_consolidation()`.
- **Cardinality cadence:** ~5-50 source rows per consolidation (size of LLM context batch). Total UNKNOWN.
- **Readers:** **no readers found in src/ or services/**. Audit-only / traceability.

### `pattern_presentations`
- **Purpose:** Records every time a consolidation pattern is presented to the LLM during `advise_run_config()` or `advise_epoch()`. Used to attribute outcomes back to specific patterns. (`postgres_schema.py:602-612`)
- **Writers:** `services/memory/db.py:748-777::insert_pattern_presentation()`; trigger `services/memory/memory_agents/query.py:1466,1502` per advice query.
- **Cardinality cadence:** ~5-10 patterns presented per `run_config` query, ~1-5 per `epoch_advice` query. Per-iteration total UNKNOWN.
- **Readers:** `services/memory/db.py:827-842::get_pattern_effectiveness()` (LEFT JOIN with `pattern_outcomes` on `(iteration, env_name)`); `services/memory/routers/training.py:177-186` (`/training/pattern_effectiveness` GET endpoint).

### `pattern_outcomes`
- **Purpose:** Iteration-level realized metrics (`gate_passed`, `sharpe`, `mdd`, `sortino`, `pnl`) plus `patterns_presented` JSON list. Joined with `pattern_presentations` to measure pattern effectiveness. (`postgres_schema.py:614-627`)
- **Writers:** `services/memory/db.py:780-824::insert_pattern_outcome()`; trigger HTTP POST `/training/record_outcome` (`services/memory/routers/training.py:148-174`) from `scripts/train_pipeline.py:576`.
- **Cardinality cadence:** 2 rows per iteration (equity + crypto). For iters 0-N: 2N rows.
- **Readers:** same as `pattern_presentations` (joined view).
- **Honest gaps:** No UNIQUE constraint on `(iteration, env_name)` — accidental double-write would create duplicates silently.

### `llm_audit_log`
- **Purpose:** Audit of every LLM call inside the memory service: prompt, response, latency, success/failure, training context (algo, env, fold, iteration, control flag). Indexes `idx_audit_call_type`, `idx_audit_timestamp` (`postgres_schema.py:760-761`). (`postgres_schema.py:629-652`)
- **Writers:** `services/memory/db.py:331-376::insert_audit_log()` (fire-and-forget). Triggered from `services/memory/memory_agents/consolidate.py` (synthesis calls) and `services/memory/memory_agents/query.py` (`run_config`, `epoch_advice` advice calls).
- **Call types observed:** `consolidate_stage1_{env}`, `consolidate_stage2`, `run_config`, `epoch_advice`. Other call types: not traced.
- **Cardinality cadence:** 3-5 per iteration baseline (consolidate 2-4 + advice 1-2). Plus per-fold advice calls (cadence in `reward-shaping.md`). Total UNKNOWN.
- **Readers:** **no readers found in src/ or services/**. Audit-only.

## Live trading & observability (13 tables)

`corporate_actions` is logically a market-data table; documented in cluster 1 above.

### `trades`
- **Purpose:** Persistent log of all fills (live + simulated) with commission/slippage/`trade_type`. PK `trade_id` (TEXT). Index `idx_trades_symbol_env` (`postgres_schema.py:743`). (`postgres_schema.py:365-380`)
- **Writers:** `execution/fill_processor.py:115::_record_trade()` (post-fill, INSERT at `:125`); `:64::record_adjustment()` (`trade_type='adjustment'`, INSERT at `:87`).
- **Cardinality cadence:** 1 row per fill event. Total UNKNOWN.
- **Readers:** `execution/risk/position_tracker.py:263::_days_since_trade()`.
- **Status:** Active in live execution path.

### `positions`
- **Purpose:** Active portfolio per `(symbol, environment)`: quantity, weighted-average cost basis, unrealized P&L, stop/TP. PK composite. Index `idx_positions_symbol_env` (`postgres_schema.py:744`). (`postgres_schema.py:382-396`)
- **Writers:** `execution/fill_processor.py:145::_update_position()` (INSERT new buy / UPSERT averaging / UPSERT on sell / DELETE on sell-to-zero); `execution/reconciliation.py:213` (broker reconciliation); `execution/adapters/binance_sim.py:166` (liquidations).
- **Cardinality cadence:** ≤ 1 active row per `(symbol, env)`. Total UNKNOWN.
- **Readers:** `execution/risk/position_tracker.py:87`; `execution/fill_processor.py:167`; `scheduler/stop_polling.py:82`; `execution/adapters/binance_sim.py:128`; `execution/reconciliation.py:245`.
- **Status:** Active — core live state.

### `risk_decisions`
- **Purpose:** Per-order audit of risk-rule evaluations (proposed action vs. final, rule triggered, reason). PK `decision_id` (TEXT). (`postgres_schema.py:398-409`)
- **Writers:** `execution/risk/risk_manager.py:273::_record_decision()` — fires per order check.
- **Cardinality cadence:** 1 row per risk evaluation. Total UNKNOWN.
- **Readers:** **no readers found in src/**. Compliance audit table.

### `portfolio_snapshots`
- **Purpose:** Time-series of portfolio total / cash / P&L / high-water-mark / drawdown. PK `(timestamp, environment)`. Index `idx_portfolio_snapshots_env_ts` DESC (`postgres_schema.py:751`). (`postgres_schema.py:411-424`)
- **Writers:** `execution/risk/position_tracker.py:162::record_snapshot()` — fires on scheduled snapshot calls.
- **Cardinality cadence:** 1 row per snapshot per env. Cadence depends on scheduler config (likely hourly or per cycle — not traced here).
- **Readers:** `execution/risk/position_tracker.py:60,105,133`; `shadow/promoter.py:197,232`; `execution/emergency.py:457`; `scheduler/jobs.py:204,268`; `monitoring/stuck_agent.py:49,64`.
- **Status:** Active — observability + emergency + shadow inputs.

### `circuit_breaker_events`
- **Purpose:** Each CB trigger logged with `triggered_at`, `trigger_type`, `value`, `threshold`, `reason`, `resumed_at` (NULL until resume). PK `event_id` (TEXT). Index `idx_cb_env_resumed` (`postgres_schema.py:745`). (`postgres_schema.py:461-471`)
- **Writers:** `execution/risk/circuit_breaker.py:189::_trigger()` — fires on threshold breach. Resume not explicitly logged; the `resumed_at` column is updated elsewhere or remains NULL (path not traced).
- **Cardinality cadence:** 1 row per trigger. Total UNKNOWN.
- **Readers:** `execution/risk/circuit_breaker.py:210::_latest_event()` (cooldown elapsed check); `shadow/promoter.py:384` (counts CB events during shadow period).
- **Status:** Active — critical safety mechanism.

### `emergency_flags`
- **Purpose:** Runtime kill-switch flags (currently only `'halt'`): `active`, `set_at`, `set_by`, `reason`. PK `flag_name` (TEXT). (`postgres_schema.py:537-549`)
- **Writers:** `scheduler/halt_check.py:75::set_halt()` (UPSERT active=1); `:100::clear_halt()` (UPDATE active=0).
- **Cardinality cadence:** 1 row per flag (currently 1 flag total).
- **Readers:** `scheduler/halt_check.py:49::is_halted()`.
- **Status:** Active — emergency kill switch.

### `wash_sale_tracker`
- **Purpose:** Equity tax compliance. Records loss-sale windows; `triggered` flag flipped on violation. PK `(symbol, sale_date)`. (`postgres_schema.py:450-459`)
- **Writers:** `monitoring/wash_sale.py:28::record_realized_loss()` (INSERT at `:48`, ON CONFLICT … DO UPDATE).
- **Cardinality cadence:** 1 row per loss sale per symbol. Total UNKNOWN.
- **Readers:** `monitoring/wash_sale.py:66::scan_wash_sales()` (active-window check on each buy fill).
- **Status:** Active — equity tax rule path.

### `alert_log`
- **Purpose:** Audit of every alert sent (level, title, `message_hash` for dedup, `sent` flag). PK `alert_id` (TEXT). (`postgres_schema.py:506-515`)
- **Writers:** `monitoring/alerter.py:314::_record_alert()` per `send_alert()` call.
- **Cardinality cadence:** 1 row per alert event. Total UNKNOWN.
- **Readers:** **no readers found in src/**. Audit-only.

### `api_errors`
- **Purpose:** Broker API failure log (broker, status_code, endpoint, error). PK `id` IDENTITY. (`postgres_schema.py:526-535`)
- **Writers:** `execution/adapters/binance_sim.py:283` — INSERT on price-fetch failure after max retries. **Honest gap:** no writer found for live-broker (Alpaca / Binance.US live) failures — only the simulator currently writes here. Whether live brokers should also feed this table is unclear from code.
- **Cardinality cadence:** 1 row per detected API failure. Total UNKNOWN.
- **Readers:** `execution/emergency.py:494` (recent-error count for emergency-ban check).
- **Status:** Active for paper / sim path; production wiring unverified.

### `inference_outcomes`
- **Purpose:** Per-inference health flag (`had_nan`, `environment`, `timestamp`). PK `id` IDENTITY. (`postgres_schema.py:517-524`)
- **Writers:** `execution/pipeline.py:186::run_inference()` — INSERT after each inference cycle.
- **Cardinality cadence:** 1 row per inference. Total UNKNOWN.
- **Readers:** `execution/emergency.py:478` (NaN-rate emergency check).
- **Status:** Active.

### `shadow_trades`
- **Purpose:** Hypothetical trades from a shadow model — same shape as `trades` plus `model_version`. PK `trade_id` (TEXT). (`postgres_schema.py:488-504`)
- **Writers:** `shadow/shadow_runner.py:282::_record_shadow_trades()` — batch INSERT per shadow inference cycle.
- **Cardinality cadence:** N rows per shadow cycle (N = number of hypothetical trades). Total UNKNOWN.
- **Readers:** `shadow/promoter.py:72` (eligibility — count + earliest); `:232` (return series for Sharpe/MDD computation).
- **Status:** Active — gates promotion (see `agent-architecture.md` "Shadow promotion").

### `system_events`
- **Purpose:** General system-event log (`level, module, event_type, message, metadata_json`). PK `event_id` (TEXT). (`postgres_schema.py:426-436`)
- **Writers:** **no writer found in src/**. Table is **scaffold-only** — DDL exists but no Python writer yet.
- **Readers:** **no readers found in src/**.
- **Status:** Scaffold-only.

### `options_positions`
- **Purpose:** SPX-style options spread tracking (`underlying`, `strategy`, `expiration`, strikes, greeks). PK `spread_id` (TEXT). (`postgres_schema.py:473-486`)
- **Writers:** **no writer found in src/**. Table is **scaffold-only** (options module deferred — see `MEMORY.md` "Charles Schwab (SPX options) deferred to Phase 3").
- **Readers:** **no readers found in src/**.
- **Status:** Scaffold-only / deferred.

## Cross-cutting concerns

### Schema bootstrap

`postgres_schema.py:654+` exposes `_ALL_TABLE_DDL: list[str]` — every DDL string in apply order. The bootstrap routine creates tables and indexes idempotently (`CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`). Trigger: container start-up + ad-hoc migrations.

### Indexes (declared in DDL section)

Strategic indexes confirmed at `postgres_schema.py:743-761`:

| Table | Index |
|-------|-------|
| `trades` | `(symbol, environment)` |
| `positions` | `(symbol, environment)` |
| `circuit_breaker_events` | `(environment, resumed_at)` |
| `portfolio_snapshots` | `(environment, timestamp DESC)` |
| `memories` | `(source)`, `(created_at)` |
| `consolidations` | `(status)`, `(env_name, stage)`, `(category, status)` |
| `llm_audit_log` | `(call_type)`, `(timestamp)` |

**Honest gap — missing strategic indexes:** `backtest_results(environment, iteration_number)`, `training_epochs(run_id, epoch)`, `meta_decisions(run_id)`, `reward_adjustments(run_id, epoch_trigger)` — all are common query filters in dashboards / `iteration_report.py`. Worth measuring before adding.

### Idempotency patterns

| Pattern | Tables using it |
|---------|-----------------|
| `ON CONFLICT (pk) DO NOTHING` | `ohlcv_daily`, `ohlcv_4h`, `consolidation_sources` |
| `ON CONFLICT (pk) DO UPDATE` | `fundamentals`, `features_equity`, `features_crypto`, `model_metadata`, `iteration_results`, `wash_sale_tracker`, `emergency_flags` (UPSERT shape), `positions` (UPSERT shape) |
| Plain INSERT (PK uniqueness) | `trades`, `risk_decisions`, `circuit_breaker_events`, `alert_log`, `shadow_trades`, `corporate_actions` |
| IDENTITY auto-PK (no idempotency by design) | `data_quarantine`, `training_epochs`, `meta_decisions`, `reward_adjustments`, `memories`, `consolidations`, `consolidation_quality`, `pattern_presentations`, `pattern_outcomes`, `llm_audit_log`, `api_errors`, `inference_outcomes` |

### Buffered vs direct writers

Buffered (queued during work, single transaction flush):

- `training_epochs` — `_epoch_queue` flushed per fold by `epoch_callback.py::flush_telemetry`.
- `reward_adjustments` — `_adjustment_trigger_queue` + `_adjustment_outcome_queue` flushed per fold.

All other writers fire one INSERT/UPDATE per call. `llm_audit_log` and `consolidation_quality` are **fire-and-forget** — exceptions swallowed by design so memory-service work never aborts on audit failures.

### Pg helpers

Connection management lives in `data/pg_helpers.py` and `data/db.py`. The `executemany_from_df(conn, table, df, columns, on_conflict=...)` helper (`pg_helpers.py`) is the canonical multi-row writer used by ingestion + feature pipelines.

### Recovery / migrations

Per `MEMORY.md` (2026-04-07 incident): `add_cps_columns.py` and `backfill_cps_history.py` are present at `/app/scripts/migrations/` and `/app/scripts/` inside the live `swingrl` container — survive restart but not recreation. **Honest gap:** these scripts' source location in the repo and their orchestrator are not traced here.

## Configurable values (yaml)

PG connection knobs are sourced from `config/swingrl.yaml` `database.*` and `.env` (the latter for credentials only). Specific schema field paths and validators are in `src/swingrl/config/schema.py`. **Honest gap:** the exact `database.*` block was not exhaustively re-audited for this doc — re-verify before changing any connection / pool setting.

## Hardcoded values (not yaml-tunable — code edit required)

| Value | Location |
|-------|----------|
| Every table name + PK + DDL | `data/postgres_schema.py` per-table DDL string |
| Index names + columns | `postgres_schema.py:743-761` |
| Notable-event thresholds (KL > 0.10, MDD < -25.0) — gate writes to `training_epochs` | `memory/training/epoch_callback.py:76-77` |
| `ADJUSTMENT_RESOLVE_EPOCHS = 10` — gates the `reward_adjustments` outcome write | `memory/training/epoch_callback.py:79` |
| `consolidate.py` LLM batch size = 200 — sets max source rows per `consolidation_sources` write | `services/memory/memory_agents/consolidate.py` (line not traced) |
| Audit `call_type` enum (`consolidate_stage1_*`, `consolidate_stage2`, `run_config`, `epoch_advice`) | call sites in `services/memory/memory_agents/{consolidate,query}.py` |

## Invariants

- All `created_at` / `set_at` / `recorded_at` / `timestamp` columns are `TIMESTAMPTZ` and stored in **UTC** per project convention. ET only at presentation layer (Discord, dashboards).
- JSON-shaped data is stored as `TEXT` (no `JSONB`) — consumers must `json.loads` themselves; no schema validation at the DB layer.
- pg16 specifically (no pg14/15 features assumed). The `db/` directory mounts a versioned data volume.
- Idempotent writers (`ON CONFLICT DO UPDATE`) re-run safely; IDENTITY-PK writers (audit / append-only) do not — re-running creates duplicates.
- No DB-level foreign keys are declared. `consolidation_sources` references `consolidations.id` and `memories.id` by convention only. Application-level cascades only.
- Memory-service tables (`memories`, `consolidations`, `consolidation_*`, `pattern_*`, `llm_audit_log`) are written exclusively by the **`swingrl-memory` container**; the main `swingrl` container talks to them via HTTP, never SQL.

## Known issues / open questions

- **`macro_features` writer not located in src/** — likely a script outside `src/`; source-of-truth for FRED ingestion is unconfirmed.
- **`fundamentals` table is orphaned from the feature pipeline** — fetcher writes rows, but `features/pipeline.py:174-175` stub-fills the four fundamental columns with `0.0`. Either the table should be wired in or the fetcher disabled.
- **`system_events` and `options_positions` are scaffold-only** — DDL with no Python writer. Document or drop in next milestone.
- **Many audit tables have no readers in src/** — `data_quarantine`, `data_ingestion_log`, `training_epochs`, `meta_decisions`, `reward_adjustments`, `risk_decisions`, `alert_log`, `consolidation_quality`, `consolidation_sources`, `llm_audit_log`. All consumed externally (notebooks / dashboards) or write-only by design. Worth confirming each consumer for ops handover.
- **Missing strategic indexes** on `backtest_results`, `training_epochs`, `meta_decisions`, `reward_adjustments` — frequent query filters with no index. EXPLAIN before adding.
- **Pg16 incident 2026-04-07 (per `MEMORY.md`)** — iter 0-5 production data wiped by tests running against the production pg16 instance. Plan A recovery restored iter 0-4 from duckdb backup; iter 5 declared lost. The conftest guard from Plan B is the structural fix.
- **`api_errors` is only written by the paper simulator** — production-broker error path either uses a different table or doesn't persist API errors. Worth a wiring audit before live trading.
- **`pattern_outcomes` has no UNIQUE constraint on `(iteration, env_name)`** — accidental double-write would silently duplicate. Consider promoting to UNIQUE in the next migration.

## Source of truth

| Concern | File |
|---------|------|
| All DDL + indexes | `src/swingrl/data/postgres_schema.py` |
| Connection helpers | `src/swingrl/data/pg_helpers.py`, `src/swingrl/data/db.py` |
| OHLCV ingestion writers | `src/swingrl/data/base.py`, `src/swingrl/data/gap_fill.py` |
| Feature writers | `src/swingrl/features/pipeline.py`, `src/swingrl/features/hmm_regime.py`, `src/swingrl/features/fundamentals.py` |
| Training-side writers | `src/swingrl/agents/backtest.py`, `src/swingrl/memory/training/epoch_callback.py`, `src/swingrl/memory/training/meta_orchestrator.py`, `scripts/train_pipeline.py` |
| Reporting / CPS readers | `src/swingrl/reporting/iteration_report.py` |
| Live-trading writers | `src/swingrl/execution/{fill_processor.py, pipeline.py, reconciliation.py, emergency.py}`, `src/swingrl/execution/risk/{risk_manager.py, position_tracker.py, circuit_breaker.py}`, `src/swingrl/execution/adapters/binance_sim.py`, `src/swingrl/scheduler/{halt_check.py, stop_polling.py, jobs.py}`, `src/swingrl/shadow/{shadow_runner.py, promoter.py}`, `src/swingrl/monitoring/{alerter.py, stuck_agent.py, wash_sale.py}` |
| Memory-service writers / readers | `services/memory/db.py`, `services/memory/routers/*.py`, `services/memory/memory_agents/{consolidate,query,ingest}.py` |
| Memory thin-client | `src/swingrl/memory/client.py` |

## Changelog

- **2026-04-16** — Initial version.
- **2026-05-15** — Wholesale path-citation cleanup. Fixed every `safety/*` (directory doesn't exist), `tax/*`, `alerts/*`, `paper/*`, `risk/*` (sans `execution/`) reference. Corrected line numbers and function names in `agents/backtest.py` (L598 `_store_results`, L724 `store_fold_results_to_duckdb`, L882 `store_iteration_results_to_duckdb`), `reporting/iteration_report.py` (L90 `load_iteration_history`, L114 `load_fold_history`, L612 `persist_iteration_cps`), `features/hmm_regime.py` (L293), `scripts/train_pipeline.py` (L1612), `execution/fill_processor.py` (L64/L115 labels were swapped), `execution/risk/position_tracker.py` (L162, L263 private `_days_since_trade`), `execution/risk/risk_manager.py` (L273), `monitoring/wash_sale.py` (L28 `record_realized_loss`, L66 `scan_wash_sales`). Resolved `macro_features` HONEST GAP — writer is `FREDIngestor` via `base.py:383`. Function rename: `check_halt()` → `is_halted()`.
