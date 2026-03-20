# Phase 22: Automated Retraining - Context

**Gathered:** 2026-03-16 (updated 2026-03-17 — DuckDB contention + SubprocVecEnv + fold log enrichment + pre-phase fixes, 2026-03-20 — Phase 20 deferrals added)
**Status:** Ready for planning

<domain>
## Phase Boundary

Automated retraining for equity (monthly) and crypto (biweekly) via APScheduler subprocesses with hybrid scheduling (calendar + performance trigger). Walk-forward validation gates retrained models before shadow deployment. Bootstrap guard prevents spurious promotion on fresh deployments. Memory agent integration via MetaTrainingOrchestrator (configurable toggle). Rolling Sharpe calculation pulled forward from Phase 23 for performance-triggered retraining. Operator CLI for manual retrain, status, history, enable/disable. Docker resource limits already applied in Phase 19.1 (swingrl 24GB/8CPU, Ollama 4GB/4CPU cpuset=0-3).

</domain>

<decisions>
## Implementation Decisions

### Training Approach
- **Configurable toggle**: memory-enhanced (MetaTrainingOrchestrator) or baseline training. Config field `retraining.use_memory_agent` (default true). Operator can disable if LLM guidance degrades quality
- **Single run per retrain cycle** — not multi-iteration like Phase 19.1. MetaTrainingOrchestrator uses accumulated memories to guide one attempt. Simpler, memories improve naturally over time
- **Data freshness verification before retrain**: equity <2 calendar days stale, crypto <12 hours stale. If stale, auto-run incremental ingest before training proceeds
- **Same walk-forward fold parameters as initial training**: equity (test_bars=63, embargo=10, min_train=252), crypto (test_bars=540, embargo=130, min_train=2190). Robust validation; extra 30-60 min irrelevant for overnight runs
- **Same ensemble gates**: Sharpe > 1.0, MDD < 15%. Sortino as ranking metric, Calmar as tiebreaker (consistent with Phase 19.1 best-model selection)
- **All 3 algos every cycle**: PPO, A2C, SAC retrained sequentially for the target environment. Fresh Sharpe-softmax ensemble weights recalculated from new walk-forward OOS results
- **Memory agent guides training, not weights**: MetaTrainingOrchestrator controls hyperparams, curriculum, reward shaping. Ensemble weights stay mathematical (Sharpe-softmax formula). Memory influences model quality, which indirectly affects weights through better Sharpe

### Scheduling (Hybrid: Calendar + Performance Trigger)
- **Calendar schedule**: equity monthly (default Saturday 2 AM ET), crypto biweekly (default Sunday 3 AM ET). Both configurable via cron expressions in config
- **Performance trigger**: rolling 30-day Sharpe calculated per environment. Retrain triggered if Sharpe declines >20% from deployment baseline OR absolute Sharpe drops below 0.5. Both thresholds configurable
- **Rolling Sharpe calculation pulled forward from Phase 23** into Phase 22 — needed for performance-triggered retraining
- **Sequential lock**: only one retrain runs at a time. File lock prevents overlap. If equity retrain is still running when crypto is due, crypto waits until next scheduled window
- **Duration tracking**: actual training duration stored in DuckDB per retrain (total + per-algo breakdown). Discord retrain-started embed includes estimated duration from last run
- **Configurable start times**: cron expressions in config with defaults above

### Failure Recovery
- **3 consecutive crash failures**: auto-disable retrain schedule for that environment. Fire CRITICAL Discord alert. Operator must re-enable via CLI after investigating
- **Gate failure != crash failure**: a retrain that completes training but fails the ensemble gate is NOT a "failure" for the counter. Only crashes/exceptions/timeouts count. Gate failure means the model isn't good enough — that's normal, not an error
- **Partial algo failure = full retrain failure**: if any algo (PPO, A2C, or SAC) fails, the entire retrain is marked as failed. Don't deploy partial results to shadow. All 3 must succeed for ensemble consistency. Discord alert specifies which algo failed
- **Timeout**: configurable, default 72 hours. Timeout = full failure, no salvage of completed algos. Counts toward the 3-failure auto-disable counter
- **Re-enable**: CLI command `docker exec swingrl python scripts/retrain.py --enable equity|crypto`. Sets flag in SQLite. No container restart required
- **--dry-run mode**: full training pipeline runs but skips deploy_to_shadow(). Logs results, fires Discord 'completed' embed with [DRY RUN] tag. Operator verifies pipeline works before enabling auto
- **SAC buffer cap (FIXED in Phase 19.1)**: hardcoded 1M buffer_size removed from HYPERPARAMS, now injected from `config.training.sac_buffer_size` (500K default) at runtime. If OOM occurs, caught as crash failure
- **Memory agent ingests failure context**: on retrain failure, ingest error details + context (crash type, which algo, system state) to memory. Memory agent can reference this in future run_config suggestions

### Bootstrap Guard
- **Dual condition**: minimum 50 trades AND 14 calendar days since shadow deployment (both configurable per env via config). Both must be met before promotion check can run
- **Independent per environment**: equity and crypto have separate trade counters, day counters, and promotion decisions. Crypto trades more frequently (every 4H) so it may reach thresholds faster
- **Better candidate replaces current**: if a new retrain produces a model with better WF OOS Sortino than the current shadow, replace shadow model. Bootstrap timer resets
- **Reuse existing shadow_promotion_check_job**: add bootstrap guard check to the existing daily 7 PM ET promoter job. If guard not met, log "bootstrap: X/50 trades, Y/14 days" and skip
- **Rejected models archived**: move to models/archive/ with timestamp, clear shadow slot. Discord "shadow rejected" embed fires
- **Promotion events triple-logged**: memory ingest (with comparison metrics + market conditions) + Discord embed (gold, Phase 21 design) + DuckDB record (model_metadata table)

### Resource Contention
- **Subprocess inside swingrl container**: spawned via `subprocess.Popen` (NOT `multiprocessing.Process`). Popen + exec = fresh Python interpreter, no inherited threads/connections from main.py (APScheduler, DB handles). Consistent with REQUIREMENTS.md ("subprocess isolation sufficient")
- **swingrl container has no mem_limit / 8CPU** (applied in Phase 19.1 — 24GB hard cap caused SAC memory thrashing). No docker-compose.prod.yml change needed. Other containers use ~5.5GB of 64GB homelab
- **swingrl-memory stays at 1GB/1CPU** per Phase 20 plan
- **Concurrent with live trading**: retrain subprocess runs at **nice +10** (lower priority). Live trading runs at normal priority (nice 0), gets CPU preference during its brief 2-5 min cycles. Trading never paused during retrain.
- **SubprocVecEnv(n_envs=6) + 3 algo workers**: same parallelization as initial training (proven at ~800% CPU). `nice +10` applied via `preexec_fn=lambda: os.nice(10)` in Popen, inherited by all SubprocVecEnv fork workers
- **Memory estimate**: retrain peak ~12-15GB (3 algo workers × 6+6 SubprocVecEnv forks, SAC 500K buffer ~4GB for equity). No mem_limit — OS manages memory on 64GB homelab
- **Fail-open on Ollama**: if Ollama/memory agent unavailable during retrain, MetaTrainingOrchestrator falls back to defaults. Retrain continues without LLM guidance
- **Memory agent concurrent access safe**: live trading uses qwen2.5:3b for /live/ endpoints (local Ollama, pinned P-cores). Retrain uses qwen2.5:3b for /training/run_config (same local Ollama). FastAPI async handles concurrency. Fail-open + timeouts handle delays
- **Memory consolidation via cloud API (ALREADY IMPLEMENTED in Phase 19.1)**: local qwen3:14b consolidation replaced with OpenAI-compatible cloud API (NVIDIA NIM default, OpenRouter + Ollama fallback). Config in `memory_agent.consolidation` with multi-provider map. `ConsolidationProviderConfig` + `ConsolidationConfig` in schema.py. `consolidate.py` reads mounted config YAML, API keys via env vars. NVIDIA API key tested and working on homelab. No Phase 22 work needed for consolidation infrastructure
- **Schedule consolidation outside trading windows**: even with cloud consolidation, avoid concurrent Ollama requests during live endpoint calls. Consolidation runs in gaps between trading cycles (equity 4:15 PM ET, crypto every 4H). Configurable via cron or interval with excluded windows
- **Ollama resource reduction (ALREADY APPLIED in Phase 19.1)**: Ollama reduced to 4GB/4CPU with cpuset=0-3 (2 P-cores). Only serves qwen2.5:3b (~2.4GB). No Phase 22 work needed
- **No checkpoint resume**: if retrain dies (crash/power outage), counts as crash failure. Restarts from scratch on next scheduled window. 3-failure auto-disable prevents infinite retries

### DuckDB Concurrent Access (RESOLVED in Phase 19.1)
- **ALREADY DONE**: `FeaturePipeline` uses short-lived DuckDB connections via `duckdb_path` + `_db()` context manager (Phase 19.1 fix §7.1). No persistent connection. No Phase 22 work needed for feature pipeline.
- **Verify `DatabaseManager.duckdb()` in `data/db.py`**: confirm it also uses short-lived connections (may still have persistent singleton pattern). If so, apply same fix.
- **No other database conflicts**: SQLite (trading_ops.db) only accessed by main.py. SQLite (memory.db) accessed via HTTP through memory service container. DuckDB is the only concern.
- **DuckDB WAL crash recovery**: auto-recovers from WAL on next connect. No data corruption risk.
- **Config serialization for ProcessPoolExecutor**: `load_config()` creates unpicklable local class `_ConfigWithYaml`. Fix: pass `config.model_dump()` dict to workers, reconstruct via `SwingRLConfig(**config_dict)`. Already implemented in `train_pipeline.py`.
- **SubprocVecEnv start method**: `fork` (not `forkserver` or `spawn`). Fork works on Linux CPU-only Docker. Tested and proven during Phase 19.1.

### Memory Agent Alignment
- **Consolidate after every retrain**: success or failure. POST /consolidate call synthesizes patterns across all training runs. Helps future retrains benefit from accumulated knowledge
- **Per-algo + combined summary ingestion**: ingest individual PPO/A2C/SAC results (Sharpe, MDD, Sortino per algo) + overall ensemble summary. Memory agent learns algo-specific patterns
- **Cold-start guard kept**: passive until 3 completed runs in DuckDB. Already satisfied by 6+ Phase 19/19.1 runs. No code change needed
- **New ':retrain' source tag suffix**: training_run:retrain, training_epoch:retrain, etc. Distinguishes from initial training in memory. Enables pattern detection: "retrains after bear markets tend to underperform"
- **Pass trigger reason to run_config**: include trigger_reason in run_config request — 'scheduled_monthly', 'scheduled_biweekly', 'performance_degradation (Sharpe dropped 22%)'. Memory agent tailors advice by trigger type
- **Always ingest results even in baseline mode**: when toggle switches to baseline, only run_config and epoch_advice calls are skipped. Ingestion continues to keep knowledge base growing
- **Auto-seed backtest fills after retrain**: after successful retrain + walk-forward, automatically ingest new backtest fills (trade entries/exits, P&L) to memory with ':retrain' source tag
- **Model lineage in both DuckDB + memory**: DuckDB for structured queries (training_runs table), memory for narrative context (consolidation builds lineage narrative). Two-tier storage pattern
- **Same timeouts as Phase 19**: epoch_advice=8s, run_config=15s, consolidation=60s. Consistency simplifies code
- **Clamp audit already implemented**: meta_decisions DuckDB table (Phase 19) stores proposed vs clamped config. No additional Phase 22 work needed
- **Live trading data already flows to memory**: Phase 20 implements post-trade ingest. Memory agent accumulates live performance data between retrains naturally

### Eval Env Memory Optimization (Pre-Phase Fix)
- **Problem**: `TrainingOrchestrator.train()` creates eval_vec_env via `_create_env()` which uses SubprocVecEnv(n_envs=6). With 3 parallel algos, that's 18 forked eval workers sitting idle 99%+ of the time — they only activate for ~2 seconds per eval round (10 rounds per fold). Wastes **~5.4 GB** of RAM
- **Root cause**: `trainer.py:168` calls `self._create_env()` for eval, same as train. No separate eval path
- **Fix**: Add a `_create_eval_env()` method (or `eval=True` flag on `_create_env()`) that always uses DummyVecEnv with `n_envs=1`. EvalCallback runs 5 sequential episodes of 252 inference steps — pure predict+step, no gradients. Parallelism provides zero speedup (IPC overhead actually makes SubprocVecEnv slower for this workload)
- **Impact**: 36 forked processes → 18. Peak memory drops ~5.4 GB. No speed loss (eval takes ~2s either way). No functionality loss (same convergence detection, same 5-episode mean reward)
- **Files**: `src/swingrl/training/trainer.py` (add eval env method, change line 168), `tests/training/test_trainer.py` (update if needed)
- **Verify**: memory monitor log should show lower peak during SAC folds after this change

### Operator CLI
- **Single script**: `scripts/retrain.py` with subcommands via flags
  - `--env equity|crypto|all` — target environment
  - `--manual` — immediate trigger with confirmation prompt ("Last retrain was X days ago, current Sharpe is Y. Proceed? [y/N]")
  - `--dry-run` — full pipeline without shadow deployment
  - `--status` — comprehensive status: last retrain (date, env, result, duration), next scheduled, current model (deployed date, WF Sharpe), shadow (status + bootstrap progress X/50 trades, Y/14 days), schedule enabled/disabled per env, last 5 retrain Sharpe trend (directional arrow)
  - `--enable equity|crypto` / `--disable equity|crypto` — per-env schedule toggle
  - `--history` — last 10 retrains (default), `--limit N` to override. Table format default, `--json` for scripting
- **Manual retrain requires confirmation**: interactive prompt showing current state. Prevents accidental triggers. Still runs walk-forward gates before shadow deployment

### Retrain Notifications
- **Retrain completed embed includes memory summary**: what the LLM suggested (hyperparams, curriculum), whether suggestions were clamped, result comparison ("Sharpe improved 12% vs last retrain")
- **Retrain failed embed includes memory analysis**: memory agent analyzes WHY training failed, provides recommended next steps. Phase 21 specified this; Phase 22 implements the analysis call
- **Performance-triggered retrain gets WARNING embed to #alerts**: "Sharpe degradation detected (1.5 → 1.1, -27%). Automated retrain triggered for equity." Calendar retrains get normal INFO "retrain started" to #daily
- All embeds follow Phase 21's severity matrix and branding (footer: "SwingRL v1.1 | {env} | {TYPE}")

### Retrain History & Reporting
- **Same training_runs table + run_type column**: add run_type ('initial', 'retrain_scheduled', 'retrain_triggered', 'manual') and trigger_reason columns. All training data queryable from one table
- **Always compare with active model**: store active model's Sharpe/MDD/Sortino/Calmar at retrain time. Report delta: "New model Sharpe 1.8 (+0.3 vs active 1.5)". Enables trend analysis
- **Generation counter**: monotonically increasing version number per environment. equity_gen_1 (initial), equity_gen_2 (first retrain), etc. Stored in DuckDB. Clear lineage tracking
- **Duration tracked granularly**: total + PPO + A2C + SAC durations per retrain. Helps identify slow algos. Discord completed embed shows total + slowest algo
- **Trend summary in --status**: last 5 retrains with directional Sharpe arrow (improving/declining)

### Verification Monitoring Items
- **Ollama latency under retrain load**: qwen2.5:3b serves 5 live trading endpoints with 3s timeout. Phase 19.1 pins Ollama to cpuset=0-3 (2 P-cores). Verify during Phase 22 homelab smoke test that qwen2.5:3b responds <3s during concurrent retrain+live. With dedicated P-cores and qwen3:14b removed from Ollama, contention should be eliminated
- **Cloud consolidation connectivity (VERIFIED in Phase 19.1)**: NVIDIA NIM endpoint tested and working from homelab. `POST /consolidate` returned `consolidated:1` on first attempt. No Phase 22 verification needed

### Testing Strategy
- **Mock training, real orchestration**: mock SB3 trainer.learn() to return dummy models in seconds. Test full orchestration: data freshness check → training loop → walk-forward validation → gate check → deploy to shadow → Discord alerts → memory ingest
- **Real model files**: mock side-effect creates real SB3 model.zip + vec_normalize.pkl (tiny networks). Tests verify deployment, shadow copy, smoke tests, promotion — the whole downstream chain. Consistent with Phase 19 TDD pattern
- **Homelab smoke test**: `--dry-run --timesteps 1000` as part of deployment verification. Full pipeline on real hardware in ~5-10 min. Doesn't deploy to shadow

### Fold Log Enrichment (DONE in Phase 19.1)
- **`algo_name` and `env_name` added to `fold_complete` log event** in `src/swingrl/agents/backtest.py`. Already committed and deployed. No Phase 22 work needed
- See `.planning/phases/22-automated-retraining/22-BACKLOG-fold-log-enrichment.md` for context

### Pre-Phase 22 Fixes Already Landed (branch: gsd/phase-19.1)
These fixes are prerequisites for Phase 22 retraining and are already merged:
- **Profit factor gate fix**: `_reconstruct_round_trips()` FIFO matcher reconstructs round-trip trades from `PortfolioSimulator.trade_log`. Portfolio now logs `price` and `value` per order. Trade log captured in terminal info dict before `DummyVecEnv` auto-reset clears it
- **WF performance ingestion to memory**: `_ingest_wf_results_to_memory()` in `train_pipeline.py` sends per-algo OOS metrics (sharpe, mdd, sortino, profit_factor, win_rate, trade count) to memory agent after ensemble gate check. Gates on `api_key` presence (not `enabled` flag) so baseline iteration results are also ingested
- **ProcessPoolExecutor test fix**: Pipeline tests use `ThreadPoolExecutor` to avoid mock object pickling across process boundaries

### Claude's Discretion
- Exact cron expression syntax and APScheduler job configuration
- Rolling Sharpe calculation implementation details (window size, minimum observations)
- File lock implementation for sequential retrain guard
- Confirmation prompt UX for manual retrain
- DuckDB schema details for new columns (run_type, trigger_reason, generation)
- Exact memory ingestion format for retrain-specific events
- How to detect "performance degradation resolved" state
- ~~Nice level implementation~~ DECIDED: `preexec_fn=lambda: os.nice(10)` in subprocess.Popen
- ~~Retrain subprocess spawning mechanism~~ DECIDED: `subprocess.Popen` (not multiprocessing.Process). Clean exec isolation from main.py

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training Pipeline
- `scripts/train_pipeline.py` — Full training pipeline with walk-forward, ensemble gate, tuning grid. Phase 22 wraps this in RetrainingOrchestrator
- `src/swingrl/training/trainer.py` — TrainingOrchestrator with locked hyperparams, VecNormalize, 6 smoke tests
- `src/swingrl/training/ensemble.py` — EnsembleBlender with sharpe_softmax_weights
- `src/swingrl/training/pipeline_helpers.py` — check_ensemble_gate, compute_ensemble_weights_from_wf, decide_final_timesteps
- `src/swingrl/memory/training/meta_orchestrator.py` — MetaTrainingOrchestrator wrapping TrainingOrchestrator with LLM guidance

### Shadow Promotion
- `src/swingrl/shadow/promoter.py` — evaluate_shadow_promotion() with 3-criterion check + Discord alerts
- `src/swingrl/shadow/lifecycle.py` — ModelLifecycle: deploy_to_shadow, promote, rollback, archive
- `src/swingrl/shadow/shadow_runner.py` — Shadow mode evaluation runner

### Scheduler
- `src/swingrl/scheduler/jobs.py` — All 12 scheduler jobs including shadow_promotion_check_job
- `scripts/main.py` — APScheduler setup, job registration, signal handling

### Walk-Forward Validation
- `src/swingrl/agents/backtest.py` — WalkForwardBacktester with fold generation, per-fold retraining
- `src/swingrl/agents/validation.py` — check_validation_gates (4 gates), diagnose_overfitting

### Memory Agent
- `src/swingrl/memory/client.py` — MemoryClient using stdlib urllib, fail-open pattern
- `src/swingrl/memory/training/bounds.py` — Hard clamp for all LLM outputs before trainer contact
- `services/memory/` — FastAPI service with /ingest, /consolidate, /training/* endpoints

### DuckDB Concurrent Access
- `.planning/phases/22-automated-retraining/22-DUCKDB-CONTENTION.md` — Full analysis of DuckDB lock contention, resolution options, SubprocVecEnv findings
- `src/swingrl/data/db.py` — DatabaseManager. FeaturePipeline already uses short-lived connections (Phase 19.1). Verify DatabaseManager.duckdb() also uses short-lived pattern.

### Config & Schema
- `src/swingrl/config/schema.py` — SwingRLConfig, ShadowConfig, MemoryAgentConfig, TrainingConfig (n_envs, vecenv_backend added)
- `config/swingrl.yaml` — Dev config defaults (n_envs=6, vecenv_backend=subproc)

### Prior Phase Context
- `.planning/phases/19-model-training/19-CONTEXT.md` — Memory agent architecture, safety bounds, meta-trainer scope
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/19.1-CONTEXT.md` — Memory service deployment, multi-iteration training
- `.planning/phases/20-production-deployment/20-CONTEXT.md` — Docker stack, resource limits, memory live enrichment
- `.planning/phases/21-discord-alert-suite/21-CONTEXT.md` — 4 retrain embed designs, severity matrix, channel routing

### Requirements
- `.planning/REQUIREMENTS.md` — RETRAIN-01 through RETRAIN-05

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/train_pipeline.py` (53KB): Full training pipeline — Phase 22 wraps this in RetrainingOrchestrator callable from APScheduler
- `src/swingrl/shadow/promoter.py`: evaluate_shadow_promotion() with 3-criterion check — add bootstrap guard here
- `src/swingrl/shadow/lifecycle.py`: ModelLifecycle.deploy_to_shadow(), promote(), rollback(), archive() — complete lifecycle management
- `src/swingrl/scheduler/jobs.py`: 12 existing jobs, JobContext singleton, is_halted() pattern — add automated_retraining_job()
- `src/swingrl/memory/training/meta_orchestrator.py`: MetaTrainingOrchestrator — reuse for memory-enhanced retraining
- `src/swingrl/memory/client.py`: MemoryClient with fail-open pattern — reuse for retrain ingestion
- `src/swingrl/agents/backtest.py`: WalkForwardBacktester — reuse for retrain validation
- `src/swingrl/training/pipeline_helpers.py`: check_ensemble_gate(), compute_ensemble_weights_from_wf() — reuse for gate checks

### Established Patterns
- Docker exec for operator commands: `docker exec swingrl python scripts/...`
- APScheduler with SQLAlchemyJobStore + ThreadPoolExecutor (but retrain uses subprocess, not thread pool)
- Memory calls fail-open with try/except → safe defaults
- Config loaded via load_config() → SwingRLConfig with env var overrides
- structlog with keyword args for all logging
- TDD: mock side-effects create real SB3 model files (Phase 19 pattern)

### Integration Points
- `src/swingrl/data/db.py` — Verify DatabaseManager.duckdb() uses short-lived connections (FeaturePipeline already fixed in Phase 19.1)
- `src/swingrl/scheduler/jobs.py` — add automated_retraining_job() using subprocess.Popen, wire in main.py
- `src/swingrl/shadow/promoter.py` — add bootstrap guard to evaluate_shadow_promotion()
- `src/swingrl/config/schema.py` — add RetrainingConfig class to SwingRLConfig
- `docker-compose.prod.yml` — already at 24GB/8CPU (Phase 19.1), no change needed
- `src/swingrl/training/` — new retraining.py for RetrainingOrchestrator
- `scripts/retrain.py` — new operator CLI
- DuckDB training_runs table — add run_type, trigger_reason, generation columns

</code_context>

<specifics>
## Specific Ideas

- Training on homelab takes ~11.5hr per env with SubprocVecEnv(6) + 3 algo workers (~23hr per iteration for both envs). Old estimate of 23+ hours was at 1 CPU; parallelization cut this significantly
- Configurable toggle for memory-enhanced vs baseline gives operator safety valve without losing infrastructure
- Performance-triggered retraining is more compute-efficient than fixed calendar when training is expensive
- Rolling Sharpe pulled forward from Phase 23 — Phase 23 builds monitoring/alerting on top of this calculation
- Generation counter gives clear model lineage: "promoted Gen 5 after bear market retrain"
- Memory agent learning from failures is a unique advantage — "SAC OOM'd with these params, try smaller buffer"
- The ':retrain' source tag suffix enables memory agent to distinguish initial vs retrain patterns
- Confirmation prompt on manual retrain prevents "oops I triggered a 24hr training job by accident"

</specifics>

<deferred>
## Deferred Ideas

- Phase 25 (Dashboard updates): retrain history visualization, model generation timeline, Sharpe trend charts
- Obs space expansion for memory dims — PARTIALLY DONE: Phase 19.1 expanded equity 156→164, crypto 45→47 (4 per-asset fields + cash action dim). Any further memory-specific obs dims would need additional retraining
- Per-algo retraining (only retrain the weakest algo) — considered but rejected for ensemble consistency
- Rolling/one-algo-at-a-time retraining — too complex for initial implementation
- Separate retrain container — rejected per REQUIREMENTS.md, subprocess isolation sufficient
- Dynamic Docker resource limits during retrain — fragile, unnecessary with uncapped swingrl on 64GB homelab

### Deferred from Phase 20 (2026-03-20)
- **Training execution timing shift**: Move training env to use next-bar open price for execution (bar t+1 open instead of bar t close). Changes `PortfolioSimulator.rebalance()` to take `execution_prices` separately from `evaluation_prices`. Mild look-ahead bias acceptable for swing trading but should be fixed when retraining. Retraining required afterward.
- **Shadow agent verification**: Full verification of shadow gates (pass + fail cases), promotion process, rejection process, and alignment between shadow trading and paper trading pipeline. Include in Phase 22 smoke test.

### Carryover from Phase 19.1 Consolidated Code Review
- **MemoryClient connection pooling** (I17) — MemoryClient currently uses per-request urllib connections. Intentional fail-open design. Consider adding connection pooling if latency becomes an issue under retrain load.
- **step_wait 4-tuple SB3 API dependency** (I18) — MemoryVecRewardWrapper.step_wait() returns 4-tuple (obs, rewards, dones, infos). SB3 may move to 5-tuple (adding truncated) in future versions. Add comment noting this dependency.

</deferred>

---

*Phase: 22-automated-retraining*
*Context gathered: 2026-03-16*
