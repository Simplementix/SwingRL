# Phase 19.1 — Complete Change Log

**Date:** 2026-03-19 to 2026-03-20
**Branch:** `gsd/phase-19.1-memory-agent-infrastructure-and-training`
**Commits:** `966d693..b635134` (6 commits, 86 files, +5,251/-1,435 lines)

This document catalogs every significant change made during the Phase 19.1 code review and meta-trainer integration fix session. Reference this when planning Phases 20-22.

---

## 1. Capital Safety Fixes

### 1.1 Tier 2 Crypto Liquidation — Was a No-Op
- **Files:** `execution/emergency.py`, `execution/adapters/binance_sim.py`
- **Problem:** `_tier2_liquidate_crypto()` logged positions but never submitted sell orders
- **Fix:** Added `BinanceSimAdapter.emergency_sell()` method and wired actual sell call in tier 2
- **Phase 20 note:** Real `BinanceAdapter` for live crypto orders still needed (deferred, noted in 20-CONTEXT.md)

### 1.2 AlpacaAdapter — Bracket Orders Required Even Without SL/TP
- **File:** `execution/adapters/alpaca_adapter.py`
- **Problem:** `submit_order()` raised `BrokerError` when SL/TP prices were None
- **Fix:** Submits `OrderClass.SIMPLE` when no SL/TP, `OrderClass.BRACKET` when provided
- **Impact:** Weight-based rebalancing (no SL/TP by design) now works with Alpaca

### 1.3 Circuit Breaker — Daily Loss Used Initial Capital
- **File:** `execution/risk/circuit_breaker.py`
- **Problem:** Per-env CB used `self._initial_capital` and GlobalCB used `self._total_initial` for daily loss %
- **Fix:** Both now use high-water mark — daily loss is relative to peak portfolio value
- **Phase 20 note:** 20-CONTEXT.md updated to reference this change

### 1.4 Extended Hours — Missed 9:00-9:29 AM Pre-Market
- **File:** `execution/emergency.py`
- **Problem:** Pre-market check was `4 <= hour < 9`, missing the 9:00-9:29 window
- **Fix:** Added `(hour == 9 and minute < 30)` check

### 1.5 Tier 3 Equity Liquidation — Identical Branches
- **File:** `execution/emergency.py`
- **Problem:** Three strategy branches all called `close_all_positions(cancel_orders=True)`
- **Fix:** Single call with strategy logged

### 1.6 Shadow Runner — Hardcoded Fallback Prices
- **File:** `shadow/shadow_runner.py`
- **Problem:** Used `$100` equity / `$50,000` BTC fallback when DB price unavailable
- **Fix:** Skips the signal entirely and logs warning
- **Also:** Removed empty `TYPE_CHECKING` block and unused import

### 1.7 Max Drawdown — Missed Initial Investment Drop
- **File:** `agents/metrics.py`
- **Problem:** `np.cumprod(1 + returns)` starts at `1+r[0]`, not 1.0 — misses drawdown from initial value
- **Fix:** Prepend 1.0: `np.concatenate([[1.0], np.cumprod(1 + returns)])` in `max_drawdown()`, `avg_drawdown()`, `max_drawdown_duration()`
- **Tests updated:** `tests/agents/test_metrics.py` — expected values recomputed with prepended 1.0

---

## 2. Turbulence Overhaul

### 2.1 Split Into Two Specialized Calculators
- **File:** `features/turbulence.py` (complete rewrite)
- **Problem:** Single `TurbulenceCalculator` used same algorithm for both equity and crypto
- **Fix:**
  - `EquityTurbulenceCalculator` — EWMA-weighted Mahalanobis distance, 126-day half-life, 252-bar warmup
  - `CryptoTurbulenceCalculator` — Vol z-score × correlation spike composite, 360-bar warmup, 1080-bar window
  - `BaseTurbulenceCalculator` ABC with `min_warmup` property
  - `TurbulenceCalculator()` factory function (backward-compatible callable name)

### 2.2 Fixed 90th Percentile Query Column
- **File:** `execution/pipeline.py`
- **Problem:** Queried `QUANTILE(atr_14_pct, 0.9)` instead of `QUANTILE(turbulence, 0.9)`
- **Fix:** Changed to `turbulence` column

---

## 3. Memory Service Fixes

### 3.1 Deduplicated `_build_context` (Sync/Async)
- **File:** `services/memory/memory_agents/query.py`
- **Problem:** ~90 lines of identical logic in sync and async versions
- **Fix:** Extracted `_filter_and_format_context()` static helper + `_build_system_prompt()` for dynamic prompt from config

### 3.2 CapacityLimiter TOCTOU Race
- **File:** `services/memory/db.py`
- **Problem:** Lazy init of anyio CapacityLimiters had a race condition
- **Fix:** `init_capacity_limiters()` called eagerly in `app.py` lifespan

### 3.3 Consolidation Config Validation
- **File:** `services/memory/memory_agents/consolidate.py`, `services/memory/app.py`
- **Problem:** Module-level config silently fell back to defaults
- **Fix:** `validate_consolidation_config()` called at startup

### 3.4 Dead Sync `_detect_conflict()` Removed
- **File:** `services/memory/memory_agents/consolidate.py`
- **Problem:** Sync version of conflict detection was dead code
- **Fix:** Removed method and unused sync DB imports

### 3.5 MemoryClient Public Properties
- **File:** `memory/client.py`
- **Problem:** `epoch_callback.py` and `meta_orchestrator.py` accessed private `_base_url`/`_api_key` with `# noqa: SLF001`
- **Fix:** Added `base_url` and `api_key` `@property` methods. Callers updated.

### 3.6 MemoryClient `epoch_advice()` Method
- **File:** `memory/client.py`, `memory/training/epoch_callback.py`
- **Problem:** Epoch callback built raw urllib requests, duplicating HTTP logic
- **Fix:** Added `epoch_advice(payload)` method to MemoryClient. Callback uses it.

### 3.7 Bisect Date Lookups in Curriculum
- **File:** `memory/training/curriculum.py`
- **Problem:** O(n) linear scan for date lookups
- **Fix:** Uses `bisect.bisect_left`/`bisect_right` for O(log n)

### 3.8 Bounds to Config YAML
- **Files:** `memory/training/bounds.py`, `config/schema.py`, `config/swingrl.yaml`, `services/memory/memory_agents/query.py`
- **Problem:** Hardcoded bounds constants in system prompt
- **Fix:** Bounds loaded from `config.training.bounds`, dynamic system prompt built from config values

### 3.9 XML Sanitization Fix
- **File:** `services/memory/memory_agents/ingest.py`
- **Problem:** `html.escape()` mangled `&`, `<`, `>` in metric text
- **Fix:** Regex strips only XML-invalid C0 control characters

### 3.10 VecNormalize Unwrap Check
- **File:** `training/trainer.py`
- **Problem:** No verification after unwrap loop
- **Fix:** Raises `ModelError` if result is not `VecNormalize` (replaced bare assert for bandit compliance)

### 3.11 Exception Chaining in Retry
- **File:** `scripts/train_pipeline.py`
- **Problem:** Retry swallowed original exception
- **Fix:** `exc2.__cause__ = exc` chains exceptions

---

## 4. Code Quality & Architecture

### 4.1 Public `compute_turbulence()` on FeaturePipeline
- **Files:** `features/pipeline.py`, `execution/pipeline.py`
- **Problem:** Execution pipeline called private `_compute_turbulence_*` methods
- **Fix:** Public `compute_turbulence(env_name, date)` method delegates to private implementations

### 4.2 Proper VecNormalize Env Stub
- **File:** `execution/pipeline.py`
- **Problem:** `DummyVecEnv([lambda: None])` for VecNormalize loading
- **Fix:** Minimal gymnasium.Env stub with correct observation/action spaces from loaded model

### 4.3 Public Accessors on ExecutionPipeline
- **Files:** `execution/pipeline.py`, `shadow/shadow_runner.py`
- **Problem:** Shadow runner accessed private `_config`, `_feature_pipeline`, `_db`
- **Fix:** Added `config`, `feature_pipeline`, `db` properties

### 4.4 DataError on Feature Storage Failure
- **File:** `features/pipeline.py`
- **Problem:** Feature storage silently failed on DuckDB error
- **Fix:** Re-raises `DataError` with chained exception

### 4.5 Narrow OperationalError Catch
- **File:** `data/db.py`
- **Problem:** Broad `except OperationalError: pass` for ALTER TABLE
- **Fix:** Checks for "duplicate column" in error message before suppressing

### 4.6 Data Loader Module
- **Files:** `src/swingrl/training/data_loader.py` (new), `scripts/train_pipeline.py`, `scripts/train.py`
- **Problem:** Fragile dynamic import via importlib in train_pipeline
- **Fix:** Canonical `load_features_prices()` in proper module. Scripts delegate to it. Raises `DataError` instead of `RuntimeError`.

### 4.7 Cached Adapters in ExecutionPipeline
- **File:** `execution/pipeline.py`
- **Problem:** Created new broker adapter per cycle (connection leak)
- **Fix:** `_get_adapter()` caches in `self._adapters` dict, reuses on subsequent calls

### 4.8 Session Cleanup in Data Ingestors
- **Files:** `data/binance.py`, `data/gap_fill.py`, `data/ingest_all.py`
- **Problem:** HTTP sessions never closed (TCP leak)
- **Fix:** Context managers and explicit `close()` calls

### 4.9 Dead Code Removal
- **Files:** `shadow/promoter.py`, `shadow/shadow_runner.py`, `data/verification.py`
- **Problem:** Empty `TYPE_CHECKING` blocks with `pass`
- **Fix:** Removed blocks and unused `TYPE_CHECKING` imports
- **Also:** Cleaned EnsembleBlender dead code reference in docstring (`training/ensemble.py`)

---

## 5. Performance Improvements

### 5.1 Macro Query Optimization
- **File:** `features/pipeline.py`
- **Problem:** `iterrows()` over DataFrame to build dict
- **Fix:** Vectorized `dict(zip(...))` approach

### 5.2 Duplicate OHLCV Read Elimination
- **File:** `features/pipeline.py`
- **Problem:** SPY/BTC OHLCV read twice (once for features, once for turbulence)
- **Fix:** Turbulence caching — `_cache_equity_turbulence()`/`_cache_crypto_turbulence()` compute from already-loaded data during feature pipeline. `_compute_turbulence_*()` checks cache before DB query.

---

## 6. Infrastructure

### 6.1 APScheduler to Main Dependencies
- **File:** `pyproject.toml`
- **Problem:** APScheduler was in dev deps but used in production `main.py`
- **Fix:** Moved to main dependencies

### 6.2 SQLAlchemy Added to Dependencies
- **File:** `pyproject.toml`, `uv.lock`
- **Problem:** APScheduler's `SQLAlchemyJobStore` needs SQLAlchemy — container crashed on startup
- **Fix:** Added `sqlalchemy>=2.0,<3` to main dependencies

### 6.3 Memory Healthcheck Python Path
- **File:** `docker-compose.prod.yml`
- **Problem:** Memory service healthcheck used bare `python`
- **Fix:** Uses `python3`

### 6.4 DuckDB Shutdown Cleanup
- **File:** `scripts/main.py`
- **Problem:** DuckDB connection never closed on shutdown
- **Fix:** *(Superseded by 7.1 — persistent connection eliminated entirely)*

### 6.5 Seed Script API Key Argument
- **File:** `scripts/seed_memory_from_backtest.py`
- **Problem:** MemoryClient created without api_key
- **Fix:** Added `--api-key` CLI argument

### 6.6 deploy_model.sh $HOME Fix
- **File:** `scripts/deploy_model.sh`
- **Problem:** `$HOME` evaluated locally instead of on remote host
- **Fix:** Uses `~/swingrl` (tilde expands on remote)

### 6.7 Per-Future Error Handling
- **File:** `scripts/train_pipeline.py`
- **Problem:** ProcessPoolExecutor futures not individually error-handled
- **Fix:** Each `future.result()` wrapped in try/except with per-algo error logging

### 6.8 `min_order_usd` Config Field
- **Files:** `config/schema.py`, `config/swingrl.yaml`
- **Problem:** Hardcoded $5 equity minimum order
- **Fix:** `min_order_usd` field on `EquityConfig` (default $1.00) aligned with existing `CryptoConfig.min_order_usd` ($10.00)

### 6.9 Docker CPU Limit Bump (Temporary)
- **File:** `docker-compose.prod.yml`
- **Problem:** 8 CPU soft limit tight for training + other homelab services
- **Fix:** Bumped to 10 CPUs for 6-iteration training run
- **Phase 20 note:** Revert to 8 CPUs after training (noted in 20-CONTEXT.md)

### 6.10 .gitignore Updates
- **File:** `.gitignore`
- **Fix:** Added `coverage.json` and `MagicMock/` (test artifacts)

---

## 7. Meta-Trainer Integration Fixes

These fixes enable the memory-training feedback loop to actually function.

### 7.1 Short-Lived DuckDB Connections in FeaturePipeline
- **Files:** `features/pipeline.py`, `features/macro.py`, `scripts/main.py`
- **Problem:** `main.py` opened a persistent DuckDB connection for `FeaturePipeline`, holding the write lock for the entire process lifetime. Training subprocess couldn't access DuckDB.
- **Fix:**
  - `FeaturePipeline` accepts `duckdb_path` parameter, uses `_db()` context manager (open/close per operation)
  - `MacroFeatureAligner` same pattern
  - `main.py` passes `duckdb_path=` instead of a live connection
  - Scheduler and training coexist in same container without lock conflicts
- **Impact:** Eliminates the need for separate training container. `DatabaseManager.duckdb()` was already short-lived; this was the last persistent connection.

### 7.2 Reward Component Decomposition
- **File:** `envs/base.py`
- **Problem:** `MemoryVecRewardWrapper._shape_rewards()` expected `info["reward_components"]` but environments never emitted them — reward shaping was a complete no-op
- **Fix:** `step()` computes 4 components (profit=daily_return, sharpe=rolling_sharpe, drawdown=-drawdown_frac, turnover=-turnover_ratio) and passes to `_build_info()` which includes them in the info dict
- **Impact:** LLM reward weight advice now actually shapes the rewards the RL agent sees

### 7.3 Reward Wrapper Placement
- **File:** `training/trainer.py`
- **Problem:** Wrapper was OUTSIDE VecNormalize (`Base → VecNormalize → Wrapper`), so shaped rewards bypassed normalization — training instability
- **Fix:** Wrapper now BETWEEN base and VecNormalize (`Base → Wrapper → VecNormalize`). Shaped rewards get normalized before reaching the RL algo.
- **Also:** `trainer.train()` accepts `initial_reward_weights` parameter

### 7.4 Initial Reward Weights Passthrough
- **File:** `memory/training/meta_orchestrator.py`
- **Problem:** Meta-orchestrator computed `reward_weights` from LLM advice and clamped them, but never passed them to `trainer.train()` — wrapper always used `DEFAULT_WEIGHTS`
- **Fix:** `trainer.train(initial_reward_weights=reward_weights)` passes LLM-advised initial weights

### 7.5 `entropy_coeff` → `ent_coef` Naming Fix
- **File:** `memory/training/meta_orchestrator.py`
- **Problem:** LLM returns `entropy_coeff` but SB3 expects `ent_coef`. The wrong key was silently ignored.
- **Fix:** Key mapping in hp extraction: `key = "ent_coef" if k == "entropy_coeff" else k`

### 7.6 `MIN_EPOCHS_BEFORE_STOP` Enforcement
- **File:** `memory/training/epoch_callback.py`
- **Problem:** `MIN_EPOCHS_BEFORE_STOP = 10` was defined in bounds.py but never checked — LLM could stop training at epoch 1
- **Fix:** Epoch callback checks `self._epoch < MIN_EPOCHS_BEFORE_STOP` before honoring `stop_training` advice

### 7.7 `stop_training` Signal in `_on_step()`
- **File:** `memory/training/epoch_callback.py`
- **Problem:** `_on_step()` always returned True, ignoring `model.stop_training`
- **Fix:** Returns `not getattr(self.model, "stop_training", False)`

### 7.8 Cold-Start Guard — Pattern Count Instead of training_runs
- **File:** `memory/training/meta_orchestrator.py`
- **Problem:** `_query_run_config()` checked `training_runs` DuckDB table (never populated) — cold-start guard permanently blocked LLM advice
- **Fix:** `_get_pattern_count()` queries memory service `/debug/consolidations` endpoint, filters by env_name. Guard passes when ≥1 consolidated pattern exists.
- **Removed:** `_get_run_history()` method (dead code after this change)

### 7.9 Consolidation After Baseline Iteration
- **File:** `scripts/train_pipeline.py`
- **Problem:** `memory_client.consolidate()` only ran `if i > 0` — baseline WF data was never consolidated, so iteration 1 had no patterns
- **Fix:** Consolidation runs after ALL iterations including iteration 0

### 7.10 Real Convergence Metrics in Run Summaries
- **File:** `memory/training/meta_orchestrator.py`
- **Problem:** `_compute_final_metrics()` returned all zeros (placeholder)
- **Fix:** Returns real convergence data: `converged` flag, `converged_at_step`, `total_timesteps`, `convergence_ratio`
- **Note:** Walk-forward evaluation metrics are ingested separately via `walk_forward:{env}:{algo}` source tags

### 7.11 Iteration Number in WF Memories
- **File:** `scripts/train_pipeline.py`
- **Problem:** `_ingest_wf_results_to_memory()` called with `iteration_number=0` hardcoded
- **Fix:** `run_environment()` accepts `iteration_number` parameter, `run_all_iterations()` passes loop index

---

## 8. Context File Updates

### 8.1 Phase 20 Context (20-CONTEXT.md)
Added:
- PositionSizer removal (dead code after weight-based rebalancing)
- Data pipeline overhaul (Tiingo, raw prices) deferred to Phase 23+
- BinanceAdapter for live crypto orders needed
- Test coverage gaps to address as pre-Phase 20 work
- CPU limit revert reminder (10 → 8 CPUs)

### 8.2 Phase 22 Context (22-CONTEXT.md)
Added:
- MemoryClient connection pooling (intentional fail-open, revisit if latency issues)
- `step_wait` 4-tuple SB3 API dependency (may move to 5-tuple in future SB3)

---

## 9. Known Remaining Gaps (Deferred)

| Gap | Severity | Target Phase |
|-----|----------|-------------|
| Curriculum sampler exists but never wired | Dead code | Phase 22+ |
| Ensemble weights have no memory hook | Missing | Phase 20 live |
| Risk parameters (max_dd, position limits) not LLM-controllable | Missing | Phase 20 live |
| Env config (tx costs, deadzone) not LLM-controllable | Missing | Phase 22+ |
| Non-WF memories (`training_epoch:historical`, etc.) accumulate unarchived | Accumulation | Phase 22 |
| Dedup requires exact algo list match (fragile) | Quality | Phase 22+ |
| Conflict detection uses bag-of-words keywords (fragile) | Quality | Phase 22+ |
| `_compute_final_metrics` lacks WF eval metrics (only convergence) | Quality | Phase 22 |
| Eval env uses raw rewards while training uses shaped rewards | By design | Monitor |
| Epoch counting semantics differ across PPO/A2C/SAC | Inherent to SB3 | Document only |
| Tiingo data source for equity back to 2002 | Missing | Phase 23+ |
| Persistent DuckDB in `FeaturePipeline` via legacy `conn` param | Deprecated | Remove in Phase 22 |

---

## 10. Test Impact

- **Final test count:** 1,187 passed (was 905 at start of Phase 19.1)
- **New test files:** `test_health.py`, `test_curriculum.py`, `test_epoch_callback_extended.py`, `test_meta_orchestrator.py`, `test_trainer_memory_wiring.py`
- **Updated tests:** `test_metrics.py` (drawdown formula), `test_phase15.py` (DataError), `test_shadow_runner.py` (public accessors), `test_trainer_memory_wiring.py` (stop_training, mock updates)
- **All checks pass:** ruff, ruff format, mypy, bandit, detect-secrets
