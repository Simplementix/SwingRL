# Roadmap: SwingRL

## Milestones

- ✅ **v1.0 MVP** — Phases 1-17 (shipped 2026-03-11)
- 🚧 **v1.1 Operational Deployment** — Phases 18-25 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-17) — SHIPPED 2026-03-11</summary>

- [x] Phase 1: Dev Foundation (3/3 plans) — completed 2026-03-06
- [x] Phase 2: Developer Experience (4/4 plans) — completed 2026-03-06
- [x] Phase 3: Data Ingestion (4/4 plans) — completed 2026-03-06
- [x] Phase 4: Data Storage and Validation (4/4 plans) — completed 2026-03-06
- [x] Phase 5: Feature Engineering (5/5 plans) — completed 2026-03-06
- [x] Phase 6: RL Environments (3/3 plans) — completed 2026-03-07
- [x] Phase 7: Agent Training and Validation (3/3 plans) — completed 2026-03-08
- [x] Phase 8: Paper Trading Core (5/5 plans) — completed 2026-03-09
- [x] Phase 9: Automation and Monitoring (4/4 plans) — completed 2026-03-09
- [x] Phase 10: Production Hardening (8/8 plans) — completed 2026-03-10
- [x] Phase 11: Production Startup Wiring (1/1 plan) — completed 2026-03-10
- [x] Phase 12: Schema Alignment and Emergency Triggers (1/1 plan) — completed 2026-03-10
- [x] Phase 13: Model Path Fix and Reconciliation Scheduling (1/1 plan) — completed 2026-03-10
- [x] Phase 14: Feature Pipeline Wiring (1/1 plan) — completed 2026-03-10
- [x] Phase 15: Training CLI Observation Assembly (1/1 plan) — completed 2026-03-11
- [x] Phase 16: Crypto Stop Price Persistence (1/1 plan) — completed 2026-03-11
- [x] Phase 17: Doc Housekeeping (1/1 plan) — completed 2026-03-10

</details>

### 🚧 v1.1 Operational Deployment (In Progress)

**Milestone Goal:** SwingRL runs hands-off on the homelab in paper trading mode with automated retraining, Discord alerts, and complete operator documentation.

- [x] **Phase 18: Data Ingestion** - Populate homelab DuckDB with maximum historical depth, aligned observation vectors (completed 2026-03-11)
- [x] **Phase 19: Model Training** - Train and validate PPO/A2C/SAC ensemble on homelab CPU, pass all walk-forward gates (completed 2026-03-13)
- [x] **Phase 19.1: Memory Agent Infrastructure and Training** - Deploy Ollama + memory service, run 6 training iterations with best-model deployment (completed 2026-03-15)
- [ ] **Phase 20: Production Deployment** - Docker stack running on homelab with paper trading firing on schedule
- [ ] **Phase 21: Discord Alert Suite** - Full alert coverage wired and smoke-tested across all severity channels
- [ ] **Phase 22: Automated Retraining** - Equity monthly + crypto biweekly retraining with validated shadow promotion
- [ ] **Phase 23: Monitoring and Observability** - Degradation alerts, structured validation logs, and dead man's switch active
- [ ] **Phase 24: Operator Runbook** - Comprehensive step-by-step documentation covering all workflows and failure modes
- [ ] **Phase 25: Dashboard updates** - Dashboard updates

## Phase Details

### Phase 18: Data Ingestion
**Goal**: Homelab DuckDB is populated with maximum available historical OHLCV, macro, and feature data — all observation vector dimensions non-NaN and ready for training
**Depends on**: Phase 17 (v1.0 codebase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. DuckDB on homelab contains equity bars for all 8 ETFs at maximum Alpaca history depth with no date gaps
  2. DuckDB contains crypto 4H bars for BTC/ETH at maximum Binance.US history depth, with pre-2019 archive stitched
  3. FRED macro series (VIX, T10Y2Y, DFF, CPI, UNRATE) are aligned to OHLCV date ranges with no NaN in shared windows
  4. Feature pipeline produces complete 156-dim equity and 45-dim crypto observation vectors with zero NaN columns
  5. Data ingestion runs end-to-end by executing commands on the homelab Docker container — no M1 Mac involvement required
**Plans:** 2/2 plans complete
Plans:
- [ ] 18-01-PLAN.md — Verification module with DuckDB quality checks
- [ ] 18-02-PLAN.md — Ingest-all orchestrator CLI with pipeline stages and tests

### Phase 19: Model Training
**Goal**: PPO/A2C/SAC agents for equity and crypto are trained on homelab CPU with LLM-powered meta-training loop, pass all walk-forward validation gates, and are deployed to models/active/ alongside their VecNormalize statistics
**Depends on**: Phase 18
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07
**Success Criteria** (what must be TRUE):
  1. All 3 algorithms (PPO, A2C, SAC) complete training for equity (3-year rolling window) and crypto (1-year rolling window) on homelab CPU with no MPS dependency
  2. Walk-forward validation records Sharpe, max drawdown, and profit factor for each algorithm and the ensemble blend
  3. Sharpe-weighted ensemble blending produces non-placeholder weight assignments derived from actual walk-forward Sharpe ratios
  4. Each trained model.zip has a corresponding vec_normalize.pkl present in models/active/{env}/{algo}/
  5. Ensemble passes the training success gate: Sharpe > 1.0 and max drawdown < 15% before any deployment step proceeds
**Plans:** 3/3 plans complete
Plans:
- [ ] 19-01-PLAN.md — Memory agent foundation (config, client, bounds) + pipeline helper functions with tests
- [ ] 19-02-PLAN.md — Memory training modules (HMM 3-state, reward wrapper, meta-orchestrator) + train_pipeline.py CLI
- [ ] 19-03-PLAN.md — Memory validation scripts + homelab training execution and verification checkpoint

### Phase 19.1: Memory Agent Infrastructure and Training (INSERTED)

**Goal:** Ollama runs as a Docker service on the homelab serving Qwen models (qwen2.5:3b, qwen3:14b — no embeddings), the swingrl-memory FastAPI service exposes /ingest, /consolidate, /health, /training/run_config, /training/epoch_advice, and debug endpoints with API key auth, model weights persist in a named Docker volume, and 6 training iterations complete (1 baseline + 5 memory-enhanced) with automated comparison and best-per-algo deployment (Sortino rank, Calmar tiebreak)
**Requirements**: TRAIN-06, TRAIN-07
**Depends on:** Phase 19
**Plans:** 4/4 plans complete

Plans:
- [ ] 19.1-01-PLAN.md — swingrl-memory FastAPI service + Docker compose + Ollama setup + unit tests
- [ ] 19.1-02-PLAN.md — DuckDB migrations + config schema updates + MemoryClient API key support
- [ ] 19.1-03-PLAN.md — Multi-iteration training loop with resumable state + best-model deployment
- [ ] 19.1-04-PLAN.md — CI updates + homelab deployment + 6 training iterations + verification checkpoint

### Phase 20: Production Deployment
**Goal**: The homelab Docker stack runs both containers healthy, paper trading fires on schedule for equity (4:15 PM ET) and crypto (every 4H), and end-to-end trade execution completes without error
**Depends on**: Phase 19
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05
**Success Criteria** (what must be TRUE):
  1. Both Docker containers (swingrl + dashboard) report healthy status on homelab with all bind-mount directories writable by the container user
  2. Homelab .env contains all required API keys and Discord webhook URLs, and startup validation confirms broker credentials and DB tables on container boot
  3. APScheduler fires the equity cycle at 4:15 PM ET and the crypto cycle every 4 hours — verified by inspecting next-run times in scheduler logs, not UTC times
  4. Paper trading executes a complete end-to-end cycle for both environments: signal generation → position sizing → risk validation → order submission → fill confirmation logged
**Plans:** 5 plans

Plans:
- [ ] 20-01-PLAN.md — Production dependencies, config schema extensions, startup validation, healthcheck
- [ ] 20-02-PLAN.md — Memory live endpoints (5 /live/ routes in swingrl-memory service)
- [ ] 20-03-PLAN.md — LiveMemoryClient, execute_cycle() memory hooks, performance tracking, CB overrides
- [ ] 20-04-PLAN.md — MacroWatcher, operational scripts, Docker compose, main.py wiring, PositionSizer removal + shadow runner rewrite
- [ ] 20-05-PLAN.md — CI updates, homelab deployment, memory seeding, smoke test verification

### Phase 21: Discord Alert Suite
**Goal**: All Discord alert types are wired, routed to the correct channels, and confirmed delivered — including four new retrain lifecycle embeds required by Phase 22
**Depends on**: Phase 20
**Requirements**: DISC-01, DISC-02, DISC-03, DISC-04
**Success Criteria** (what must be TRUE):
  1. Discord webhook URLs for both channels (#alerts for critical/warning, #daily for summaries) are configured and respond to a smoke-test ping from the running container
  2. A simulated circuit breaker trigger, broker auth failure, and stuck agent each produce a critical embed in the #alerts channel
  3. The daily summary embed posts with Sharpe, P&L, and position summary fields populated from actual paper trading data
  4. Four retrain embed types are implemented and render correctly: retrain started, retrain completed (old vs new Sharpe), retrain failed, and shadow promoted/rejected
**Plans:** 3 plans

Plans:
- [ ] 21-01-PLAN.md — AlertingConfig extension, Alerter upgrade (trades webhook, escalating cooldown, rate-limit queue, type/severity filtering)
- [ ] 21-02-PLAN.md — 9 new embed builders + v1.1 footer upgrade + daily summary rewrite with rolling metrics
- [ ] 21-03-PLAN.md — Scheduler wiring, lifecycle alerts, startup validation, smoke test Discord section, alert history dashboard

### Phase 22: Automated Retraining
**Goal**: Equity retraining runs monthly and crypto retraining runs biweekly as APScheduler subprocesses — each running walk-forward validation before shadow deployment, with a bootstrap guard preventing spurious promotion on fresh deployments
**Depends on**: Phase 21
**Requirements**: RETRAIN-01, RETRAIN-02, RETRAIN-03, RETRAIN-04, RETRAIN-05
**Success Criteria** (what must be TRUE):
  1. equity_retrain_job fires monthly on Saturday at 2 AM ET and crypto_retrain_job fires biweekly on Sunday at 3 AM ET — both scheduled as subprocesses (not threads) with misfire_grace_time configured
  2. Each retraining job explicitly runs all 4 walk-forward validation gates before calling deploy_to_shadow() — a gate failure aborts shadow deployment and fires a retrain-failed Discord alert
  3. After shadow deployment, the existing shadow_promotion_check_job evaluates the new model the following day using the 3-criteria promotion logic
  4. The shadow promoter bootstrap guard returns False and skips auto-promotion when the trades table contains fewer than the minimum required rows, preventing spurious promotion on fresh deployments
**Plans:** 4 plans

Plans:
- [ ] 22-01-PLAN.md — DuckDB short-lived connection fix, RetrainingConfig schema, DB migrations, filelock dependency
- [ ] 22-02-PLAN.md — RetrainingOrchestrator with data freshness, walk-forward gating, failure counter, rolling Sharpe, memory integration
- [ ] 22-03-PLAN.md — Bootstrap guard in evaluate_shadow_promotion() with dual-condition check
- [ ] 22-04-PLAN.md — APScheduler retrain jobs, operator CLI, Discord alerts, performance trigger, verification checkpoint

### Phase 23: Monitoring and Observability
**Goal**: Paper trading performance is continuously tracked with degradation alerts, structured validation logs, and a dead man's switch confirming each cycle completed
**Depends on**: Phase 22
**Requirements**: MON-01, MON-02, MON-03
**Success Criteria** (what must be TRUE):
  1. A rolling Sharpe degradation alert fires in the #alerts Discord channel when paper trading Sharpe drops below the configured threshold — threshold is configurable via swingrl.yaml, not hardcoded
  2. Every trading cycle writes a structured log entry with daily P&L, current drawdown, and fill latency in JSON format readable by structlog
  3. Healthchecks.io receives a ping after each completed trading cycle — a missed ping generates an external notification confirming the dead man's switch is live
**Plans**: TBD

### Phase 24: Operator Runbook
**Goal**: A complete operator runbook documents every workflow and failure mode with step-by-step procedures derived from verified system behavior — enabling the operator to run SwingRL without source code archaeology
**Depends on**: Phase 23
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05
**Success Criteria** (what must be TRUE):
  1. Runbook covers initial setup end-to-end: homelab prerequisites, Docker installation, .env population, data ingestion commands, and expected output at each step
  2. Runbook covers the first training run with expected log output, validation gate pass/fail criteria, and the command to verify models/active/ is populated correctly
  3. Runbook covers the retraining workflow including how to interpret shadow promotion decisions and when to manually intervene vs. let auto-promotion proceed
  4. Runbook covers emergency stop (all 4 tiers), model rollback procedure, and disaster recovery steps with explicit commands and expected outputs
  5. Runbook covers common failure diagnosis: at least 8 failure scenarios with symptoms, root cause, and resolution steps drawn from research pitfalls
**Plans**: TBD

### Phase 25: Dashboard updates

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 24
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 25 to break down)

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Dev Foundation | v1.0 | 3/3 | Complete | 2026-03-06 |
| 2. Developer Experience | v1.0 | 4/4 | Complete | 2026-03-06 |
| 3. Data Ingestion | v1.0 | 4/4 | Complete | 2026-03-06 |
| 4. Data Storage and Validation | v1.0 | 4/4 | Complete | 2026-03-06 |
| 5. Feature Engineering | v1.0 | 5/5 | Complete | 2026-03-06 |
| 6. RL Environments | v1.0 | 3/3 | Complete | 2026-03-07 |
| 7. Agent Training and Validation | v1.0 | 3/3 | Complete | 2026-03-08 |
| 8. Paper Trading Core | v1.0 | 5/5 | Complete | 2026-03-09 |
| 9. Automation and Monitoring | v1.0 | 4/4 | Complete | 2026-03-09 |
| 10. Production Hardening | v1.0 | 8/8 | Complete | 2026-03-10 |
| 11. Production Startup Wiring | v1.0 | 1/1 | Complete | 2026-03-10 |
| 12. Schema Alignment | v1.0 | 1/1 | Complete | 2026-03-10 |
| 13. Model Path Fix | v1.0 | 1/1 | Complete | 2026-03-10 |
| 14. Feature Pipeline Wiring | v1.0 | 1/1 | Complete | 2026-03-10 |
| 15. Training CLI Obs Assembly | v1.0 | 1/1 | Complete | 2026-03-11 |
| 16. Crypto Stop Persistence | v1.0 | 1/1 | Complete | 2026-03-11 |
| 17. Doc Housekeeping | v1.0 | 1/1 | Complete | 2026-03-10 |
| 18. Data Ingestion | v1.1 | 2/2 | Complete | 2026-03-11 |
| 19. Model Training | v1.1 | 3/3 | Complete | 2026-03-13 |
| 19.1. Memory Agent Infra | v1.1 | 4/4 | Complete | 2026-03-15 |
| 20. Production Deployment | v1.1 | 0/5 | Planned | - |
| 21. Discord Alert Suite | v1.1 | 0/3 | Planned | - |
| 22. Automated Retraining | v1.1 | 0/4 | Planned | - |
| 23. Monitoring and Observability | v1.1 | 0/TBD | Not started | - |
| 24. Operator Runbook | v1.1 | 0/TBD | Not started | - |
| 25. Dashboard updates | v1.1 | 0/TBD | Not started | - |

**Full v1.0 details:** `.planning/milestones/v1.0-ROADMAP.md`
