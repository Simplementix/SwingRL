# Roadmap: SwingRL

## Overview

SwingRL builds an automated RL swing trading system in 10 phases: starting from a validated cross-platform dev environment, progressing through data ingestion, feature engineering, RL environment and agent construction, walk-forward validation, paper trading with full risk management, and culminating in a production-hardened homelab deployment with model lifecycle management. Every phase ends with a passing ci-homelab.sh run on the homelab server.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Dev Foundation** - Reproducible Python 3.11 environment with Docker validated on x86 homelab (completed 2026-03-06)
- [x] **Phase 2: Developer Experience** - Claude skills, config schema, smoke tests, and models scaffold (completed 2026-03-06)
- [x] **Phase 3: Data Ingestion** - Raw OHLCV and macro data flowing from Alpaca, Binance.US, and FRED (completed 2026-03-06)
- [x] **Phase 4: Data Storage and Validation** - DuckDB/SQLite schema operational with validation, quarantine, and alerting (completed 2026-03-06)
- [x] **Phase 5: Feature Engineering** - Full 156-dim equity and 45-dim crypto observation vectors assembled and verified
- [x] **Phase 6: RL Environments** - Gymnasium-compatible trading environments passing step/reset contracts
- [x] **Phase 7: Agent Training and Validation** - PPO/A2C/SAC ensemble trained and walk-forward validated against performance gates
- [x] **Phase 8: Paper Trading Core** - Broker connections live with risk management veto layer and execution middleware (completed 2026-03-09)
- [x] **Phase 9: Automation and Monitoring** - Scheduled execution, Discord alerting, dashboard, and dead man's switch running (completed 2026-03-09)
- [x] **Phase 10: Production Hardening** - Backup, model deployment pipeline, shadow mode, security review, and disaster recovery verified (completed 2026-03-10)
- [x] **Phase 11: Production Startup Wiring** - Fix main.py ExecutionPipeline constructor, wire feature table init, fix FRED import path (gap closure, completed 2026-03-10)
- [x] **Phase 12: Schema Alignment and Emergency Triggers** - Fix stop_polling table reference, fix emergency trigger queries against missing tables (gap closure) (completed 2026-03-10)
- [x] **Phase 13: Model Path Fix and Reconciliation Scheduling** - Fix model path double-nesting bug and wire PositionReconciler to scheduler (gap closure) (completed 2026-03-10)
- [x] **Phase 14: Feature Pipeline Wiring** - Wire compare_features() consumer and integrate sentiment features into observation vector (gap closure) (completed 2026-03-10)
- [x] **Phase 15: Training CLI Observation Assembly** - Fix train.py to use ObservationAssembler instead of raw DuckDB columns (gap closure) (completed 2026-03-11)
- [ ] **Phase 16: Crypto Stop Price Persistence** - Persist stop/TP prices from FillProcessor to positions table for stop_polling (gap closure)
- [ ] **Phase 17: Doc Housekeeping** - Fix stale counters, descriptions, and plan counts in REQUIREMENTS.md and ROADMAP.md (gap closure)

## Phase Details

### Phase 1: Dev Foundation
**Goal**: A reproducible, cross-platform environment where code committed on M1 Mac builds and tests correctly on the x86 homelab via Docker
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, ENV-06, ENV-07, ENV-08
**Success Criteria** (what must be TRUE):
  1. `python --version` returns 3.11.x and `import torch; torch.backends.mps.is_available()` returns True on M1 Mac
  2. `uv` is installed and functional on both M1 Mac and homelab (Ubuntu 24.04)
  3. GitHub repo exists with canonical directory structure (src/, config/, data/, db/, models/, tests/, scripts/, status/) visible on remote
  4. `pre-commit run --all-files` passes ruff, black, mypy, detect-secrets, and bandit checks on a clean commit
  5. `bash ci-homelab.sh` via `ssh homelab` completes — docker compose build, pytest, and cleanup all green
**Plans:** 3/3 plans complete

Plans:
- [ ] 01-01-PLAN.md — Repository scaffold, pyproject.toml, dependency resolution, smoke tests
- [ ] 01-02-PLAN.md — Pre-commit hooks (ruff, black, mypy, bandit, detect-secrets)
- [ ] 01-03-PLAN.md — Dockerfile, docker-compose.yml, ci-homelab.sh with homelab validation

### Phase 2: Developer Experience
**Goal**: A developer can onboard to SwingRL conventions immediately, the config schema is enforced at startup, and a smoke test confirms all core packages import correctly
**Depends on**: Phase 1
**Requirements**: ENV-09, ENV-10, ENV-11, ENV-12, ENV-13
**Success Criteria** (what must be TRUE):
  1. CLAUDE.md exists at repo root describing SwingRL conventions (TDD, snake_case, type hints, Pydantic, pathlib, no hardcoded keys)
  2. `.claude/commands/` contains SwingRL-specific skill files usable from Claude Code
  3. Loading swingrl.yaml through the Pydantic v2 config schema raises a `ValidationError` on any invalid field and passes on a valid config
  4. `models/active/`, `models/shadow/`, and `models/archive/` directories exist with `.gitkeep` files committed
  5. `pytest tests/test_smoke.py` passes — all core package imports succeed and conftest fixtures load without error
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — Remove darwin constraint, add structlog + pydantic-settings, create utils/exceptions.py and utils/logging.py
- [ ] 02-02-PLAN.md — Create CLAUDE.md conventions document and .claude/commands/ skill files
- [ ] 02-03-PLAN.md — Pydantic v2 config schema (schema.py), config/swingrl.yaml dev defaults, prod example
- [ ] 02-04-PLAN.md — Expand tests/conftest.py fixtures, create tests/test_config.py, add ENV-09/10/12 smoke tests

### Phase 3: Data Ingestion
**Goal**: Raw OHLCV bars and macro indicators flow reliably from all three sources into the system, with historical crypto backfill complete
**Depends on**: Phase 2
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. Running the Alpaca ingestor fetches OHLCV bars for all 8 equity ETFs and writes them to disk without error
  2. Running the Binance.US ingestor fetches 4H bars for BTC/USD and ETH/USD, with rate-limit headers monitored and logged
  3. The crypto historical backfill produces a continuous series from 2017 to present, with volume normalized to USD units and no gaps at the 2019 stitch point
  4. The FRED macro pipeline fetches all 5 Tier 1 series using ALFRED vintage data (no look-ahead bias) and forward-fills correctly
  5. Feeding a row with a known data defect through the 12-step validation checklist results in that row being written to the data_quarantine table rather than the main tables
**Plans**: 4 plans

Plans:
- [ ] 03-01-PLAN.md — Foundation: BaseIngestor ABC, DataValidator (12-step checklist), ParquetStore, deps install
- [ ] 03-02-PLAN.md — Alpaca equity ingestor for 8 ETFs with IEX feed, incremental/backfill modes, CLI
- [ ] 03-03-PLAN.md — Binance.US crypto ingestor with 4H bars, rate limiting, historical backfill, stitch validation
- [ ] 03-04-PLAN.md — FRED macro ingestor for 5 Tier 1 series with ALFRED vintage data, CLI

### Phase 4: Data Storage and Validation
**Goal**: DuckDB and SQLite databases are operational with incremental schema, cross-source validation, ingestion logging, and Discord alerting wired up
**Depends on**: Phase 3
**Requirements**: DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11, DATA-12, DATA-13
**Success Criteria** (what must be TRUE):
  1. DuckDB (market_data.ddb) and SQLite (trading_ops.db) databases exist with all Phase 4 tables created and queryable
  2. A DuckDB query using the sqlite_scanner extension joins a market_data table with a trading_ops table and returns correct results
  3. Weekly closing prices for a test equity differ by less than 0.1% between Alpaca and yfinance, confirmed by the cross-source validator
  4. After an ingestion run, a new row exists in data_ingestion_log with correct run_id, timing, and row counts
  5. Calling `alerter.send()` with a test message routes a Discord webhook notification to the correct channel based on alert level
**Plans**: 4 plans

Plans:
- [ ] 04-01-PLAN.md — DatabaseManager singleton, DuckDB/SQLite schema, aggregation views, config additions, init_db.py
- [ ] 04-02-PLAN.md — BaseIngestor DuckDB sync, ingestion logging, quarantine migration to DuckDB
- [ ] 04-03-PLAN.md — Discord alerter module with level-based routing, cooldown, daily digest
- [ ] 04-04-PLAN.md — Cross-source validation (Alpaca vs yfinance) and corporate action detection

### Phase 5: Feature Engineering
**Goal**: The complete 156-dimension equity and 45-dimension crypto observation vectors are computed correctly from raw bars, with normalization and correlation pruning applied
**Depends on**: Phase 4
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04, FEAT-05, FEAT-06, FEAT-07, FEAT-08, FEAT-09, FEAT-10, FEAT-11
**Success Criteria** (what must be TRUE):
  1. The feature pipeline produces an observation array of shape (N, 156) for equity and (N, 45) for crypto with no NaN values after the warmup period
  2. Fundamental features (P/E z-score, earnings growth, debt-to-equity, dividend yield) are present in the equity feature table and update on a quarterly schedule
  3. HMM regime detection produces two continuous probabilities (P(bull), P(bear)) that sum to 1.0 for every bar in both environments
  4. Rolling z-score normalization uses a 252-bar window for equity and a 360-bar window for crypto, and the per-environment feature tables exist in DuckDB (features_equity, features_crypto)
  5. A new candidate feature added to the pipeline is rejected when its A/B Sharpe improvement is less than 0.05, and accepted when it meets the threshold
**Plans:** 4/5 plans executed

Plans:
- [ ] 05-01-PLAN.md — Config schema (FeaturesConfig), DuckDB DDL (4 tables), technical indicator calculator, test infrastructure
- [ ] 05-02-PLAN.md — Fundamental data fetcher (yfinance + Alpha Vantage fallback), macro feature aligner (ASOF JOIN)
- [ ] 05-03-PLAN.md — HMM regime detector (2-state Gaussian per environment), turbulence index calculator
- [ ] 05-04-PLAN.md — Rolling z-score normalization, correlation pruning with domain-driven rules
- [ ] 05-05-PLAN.md — Observation assembler (156/45 dims), feature pipeline orchestrator, CLI, A/B comparison infrastructure

### Phase 6: RL Environments
**Goal**: Gymnasium-compatible trading environments for both equity and crypto pass the step/reset contract and produce valid observations and rewards
**Depends on**: Phase 5
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11
**Success Criteria** (what must be TRUE):
  1. `gym.make('StockTradingEnv-v0')` initializes and `env.reset()` returns an observation of shape (156,) with all values finite
  2. `gym.make('CryptoTradingEnv-v0')` initializes and `env.reset()` returns an observation of shape (45,) with all values finite
  3. The rolling 20-day Sharpe reward function returns an expanding-window warmup value for the first 19 bars, then a proper Sharpe ratio thereafter
  4. Actions within +/-0.02 of zero result in a "hold" (no trade executed), confirmed by inspecting the trade log after a random rollout
  5. Equity episodes run for 252-day segments; crypto episodes run for 540 4H bars with a random start — confirmed by running 10 episodes each and checking lengths
**Plans:** 3 plans

Plans:
- [ ] 06-01-PLAN.md — EnvironmentConfig schema, PortfolioSimulator, RollingSharpeReward, action processing utilities, test fixtures
- [ ] 06-02-PLAN.md — BaseTradingEnv base class and StockTradingEnv equity environment with step/reset contracts
- [ ] 06-03-PLAN.md — CryptoTradingEnv with random-start episodes, Gymnasium registration, SB3 check_env integration tests

### Phase 7: Agent Training and Validation
**Goal**: PPO, A2C, and SAC agents train on both environments, the Sharpe-weighted ensemble is blended, and walk-forward backtesting confirms all four performance gates pass
**Depends on**: Phase 6
**Requirements**: TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-12, VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06, VAL-07, VAL-08
**Success Criteria** (what must be TRUE):
  1. Training runs for PPO, A2C, and SAC complete without error on both environments using the specified hyperparameters, producing saved model artifacts in models/active/
  2. The Sharpe-weighted softmax ensemble produces blended action weights using 63-bar equity and 126-bar crypto validation windows
  3. Walk-forward backtesting produces results across at least 3 non-overlapping 3-month folds with a 200-bar purge gap verified between each fold
  4. All four validation gates pass: Sharpe > 0.7 per environment, MDD < 15%, Profit Factor > 1.5, overfitting gap < 20%
  5. Backtest results are stored in the DuckDB backtest_results table with per-model and per-fold rows queryable via SQL
**Plans:** 2/3 plans executed

Plans:
- [ ] 07-01-PLAN.md — Performance metrics (Sharpe, Sortino, Calmar, Rachev, MDD, trade metrics), validation gates, overfitting detection, DuckDB DDL
- [ ] 07-02-PLAN.md — ConvergenceCallback (SB3 early stopping), TrainingOrchestrator (PPO/A2C/SAC with locked hyperparams, model saving, smoke tests)
- [ ] 07-03-PLAN.md — Walk-forward backtester (growing window, purge gaps), Sharpe-weighted ensemble blender, CLI scripts (train.py, backtest.py)

### Phase 8: Paper Trading Core
**Goal**: Equity and crypto paper trading connections are live, orders flow through the full 5-stage execution middleware, and the two-tier risk management veto layer blocks out-of-policy trades
**Depends on**: Phase 7
**Requirements**: PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10, PAPER-11, PAPER-18, PAPER-19, PAPER-20
**Success Criteria** (what must be TRUE):
  1. A test equity signal submitted to Alpaca paper API results in a filled order visible in the Alpaca paper dashboard
  2. A test crypto signal results in a simulated fill recorded locally via Binance.US real-time price, with the trade logged in trading_ops.db
  3. Submitting an order that would breach the per-environment equity drawdown limit (-10% DD) is vetoed by the risk layer before reaching the exchange adapter
  4. Triggering a circuit breaker writes a row to circuit_breaker_events in SQLite, and the system still reads that halt state correctly after a container restart
  5. An order sized below $10 for crypto is automatically floor-adjusted to $10.00 before submission, and an order where round-trip costs exceed 2.0% of order value is rejected by the cost gate
**Plans:** 5/5 plans complete

Plans:
- [ ] 08-01-PLAN.md — Pipeline data types, risk infrastructure (two-tier risk manager, circuit breakers, position tracker)
- [ ] 08-02-PLAN.md — Middleware stages 1-3 (signal interpreter, position sizer, order validator with cost gate)
- [ ] 08-03-PLAN.md — Exchange adapters (Alpaca paper trading, Binance.US simulated fills) and fill processor
- [ ] 08-04-PLAN.md — ExecutionPipeline orchestrator, position reconciliation, CLI scripts (run_cycle, reconcile, seed_production)
- [ ] 08-05-PLAN.md — Multi-stage Docker production build, HEALTHCHECK, production docker-compose for homelab

### Phase 9: Automation and Monitoring
**Goal**: The system runs autonomously on schedule, sends Discord alerts for all significant events, and the operator can monitor system health from the Streamlit dashboard and Healthchecks.io
**Depends on**: Phase 8
**Requirements**: PAPER-12, PAPER-13, PAPER-14, PAPER-15, PAPER-16, PAPER-17
**Success Criteria** (what must be TRUE):
  1. APScheduler triggers the equity inference cycle at 4:15 PM ET and the crypto cycle every 4H at 5 min past bar close, with pre-cycle halt checks running before each execution
  2. A test trade execution, a circuit breaker trigger, and the 6 PM ET daily summary each produce a correctly formatted Discord webhook message in the target channel
  3. If the equity environment stays 100% cash for 10 consecutive trading days, a "stuck agent" Discord alert fires
  4. The Streamlit dashboard displays traffic-light status for both environments and the last 5 trade executions without manual refresh
  5. Healthchecks.io receives a heartbeat ping within the 70-minute crypto window and the 25-hour equity window; missing either sends a Discord alert
**Plans:** 4/4 plans complete

Plans:
- [ ] 09-01-PLAN.md — Scheduler infrastructure: config extensions, halt check, job functions, HC ping, emergency scripts
- [ ] 09-02-PLAN.md — Discord alerting: embed builders, two-webhook Alerter, stuck agent detection, wash sale scanner
- [ ] 09-03-PLAN.md — Streamlit multi-page dashboard with separate Docker container
- [ ] 09-04-PLAN.md — main.py entrypoint wiring, stop-price polling, Dockerfile CMD, production compose with dashboard

### Phase 10: Production Hardening
**Goal**: The system is fully hardened for sustained operation — backups automated, models deploy via script, shadow mode validates new models before promotion, security reviewed, and disaster recovery tested
**Depends on**: Phase 9
**Requirements**: HARD-01, HARD-02, HARD-03, HARD-04, HARD-05, PROD-01, PROD-02, PROD-03, PROD-04, PROD-05, PROD-06, PROD-07, PROD-08, PROD-09
**Success Criteria** (what must be TRUE):
  1. Daily SQLite backup, weekly DuckDB backup, and monthly off-site rsync via Tailscale all run on schedule and produce verifiable archives
  2. Running `deploy_model.sh` from M1 Mac SCPs a model to homelab, verifies integrity, and runs a smoke test — the new model appears in models/active/ on homelab
  3. A new model running in shadow mode produces hypothetical trades in parallel with the active model for 10 equity days / 30 crypto cycles; auto-promotion fires when all three promotion criteria are met
  4. Running `emergency_stop.py` halts all jobs, cancels open orders, and liquidates crypto immediately (equity queued for market open) — confirmed by checking exchange state and trading_ops.db
  5. Stopping the container, deleting all volumes, restoring from backup, and restarting completes the 9-step disaster recovery checklist with the system resuming paper trading correctly
**Plans:** 8/8 plans complete

Plans:
- [x] 10-01-PLAN.md — Config extensions (Backup/Shadow/Sentiment/Security), dependencies, retry decorator, file-based JSON logging
- [x] 10-02-PLAN.md — Backup automation: daily SQLite, weekly DuckDB, monthly off-site rsync, APScheduler jobs
- [x] 10-03-PLAN.md — Model deployment pipeline (deploy_model.sh), model lifecycle state machine, 6-point smoke test
- [x] 10-04-PLAN.md — Shadow mode: parallel inference, hypothetical trades, auto-promotion with 3 criteria
- [x] 10-05-PLAN.md — FinBERT sentiment pipeline, news fetcher (Alpaca + Finnhub), A/B experiment infrastructure
- [x] 10-06-PLAN.md — Four-tier emergency stop protocol, 3 automated triggers, CLI update
- [x] 10-07-PLAN.md — Security review checklist, disaster recovery test script, key rotation runbook, Jupyter notebooks
- [x] 10-08-PLAN.md — [GAP CLOSURE] Wire _generate_hypothetical_trades stub to feature pipeline, model inference, and signal interpretation

### Phase 11: Production Startup Wiring
**Goal**: The production entrypoint (main.py) starts without error, feature tables are initialized as part of standard DB setup, and scheduler jobs import from correct module paths
**Depends on**: Phase 10
**Requirements**: PAPER-01, PAPER-02, PAPER-12, FEAT-11, DATA-09, DATA-04, FEAT-04
**Gap Closure:** Closes INT-01, INT-02, INT-05, Flow 1 from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. `python scripts/main.py` with a valid config starts without TypeError — ExecutionPipeline receives all 5 required arguments
  2. `DatabaseManager.init_schema()` creates feature tables (features_equity, features_crypto) without requiring a prior `compute_features.py` run
  3. Weekly fundamentals and monthly macro scheduler jobs import from `swingrl.data.fred` without ImportError

Plans:
- [x] 11-01-PLAN.md — Fix main.py ExecutionPipeline constructor, wire feature table init into DatabaseManager, fix FRED import path

### Phase 12: Schema Alignment and Emergency Triggers
**Goal**: Stop-price polling queries the correct table with correct columns, and all three automated emergency triggers query existing tables
**Depends on**: Phase 11
**Requirements**: PAPER-10, PROD-07
**Gap Closure:** Closes INT-03, INT-04, Flow 2 from v1.0 audit
**Success Criteria** (what must be TRUE):
  1. `stop_polling.py` queries the `positions` table with correct column names and processes stop/TP prices without error
  2. All three automated emergency triggers in `check_automated_triggers()` query existing tables and return valid results
  3. The automated emergency trigger flow fires `execute_emergency_stop()` when trigger conditions are met

Plans:
- [ ] 12-01-PLAN.md — Fix stop_polling table/column references, fix or create tables for emergency trigger queries

### Phase 13: Model Path Fix and Reconciliation Scheduling
**Goal**: The production trading cycle finds trained models and executes trades, and position reconciliation runs daily to prevent drift between DB and broker state
**Depends on**: Phase 12
**Requirements**: PAPER-02, PAPER-09
**Gap Closure:** Closes PAPER-02 (model path), PAPER-09 (reconciler scheduling), integration gap (main.py → pipeline.py), broken flow (Production Trading Cycle). Also fixes PAPER-01, TRAIN-06, PROD-02 (same root cause).
**Success Criteria** (what must be TRUE):
  1. `ExecutionPipeline._load_models()` finds models at `models/active/{env}/{algo}/model.zip` — no double "active" in path
  2. `execute_cycle("equity")` with a trained model present produces non-zero fills (model loaded → ensemble → signals → orders)
  3. A reconciliation job is registered in APScheduler and calls `PositionReconciler.reconcile()` daily after equity market close

Plans:
- [ ] 13-01-PLAN.md — Fix model path double-nesting in main.py, add reconciler job to scheduler

### Phase 14: Feature Pipeline Wiring
**Goal**: Feature A/B comparison has a production consumer and sentiment features integrate into the observation vector when enabled
**Depends on**: Phase 13
**Requirements**: FEAT-09, FEAT-10
**Gap Closure:** Closes FEAT-09 (compare_features consumer), FEAT-10 (sentiment integration)
**Success Criteria** (what must be TRUE):
  1. `compare_features()` is callable from a CLI entrypoint or scheduled job that produces A/B comparison results
  2. When `SentimentConfig.enabled=True`, `get_sentiment_features()` output is included in the equity observation vector assembly
  3. When `SentimentConfig.enabled=False` (default), observation vector dimensions remain unchanged (156 equity, 45 crypto)

Plans:
- [ ] 14-01-PLAN.md — Wire compare_features() consumer and integrate sentiment into observation assembly

### Phase 15: Training CLI Observation Assembly
**Goal**: The training CLI (train.py) produces correctly shaped observation matrices by using ObservationAssembler, so SB3 model construction succeeds on both environments
**Depends on**: Phase 14
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, VAL-01
**Gap Closure:** Closes INT-GAP-01 (train.py → env obs_space shape mismatch), Flow "Training Pipeline (CLI → Model)"
**Success Criteria** (what must be TRUE):
  1. `train.py _load_features_prices()` calls `ObservationAssembler.assemble_equity()` / `assemble_crypto()` to produce full observation matrices
  2. Equity observations have shape (N, 156) and crypto observations have shape (N, 45) — matching env observation_space
  3. `python scripts/train.py --env equity --dry-run` completes model construction without shape mismatch error

Plans:
- [ ] 15-01-PLAN.md — Fix train.py to use ObservationAssembler for feature-to-observation transformation

### Phase 16: Crypto Stop Price Persistence
**Goal**: Stop-loss and take-profit prices flow from the execution pipeline through FillProcessor into the positions table, enabling stop_polling to enforce crypto stops
**Depends on**: Phase 15
**Requirements**: PAPER-07, PAPER-10
**Gap Closure:** Closes INT-GAP-02 (FillProcessor → positions table → stop_polling), Flow "Crypto Stop-Loss Enforcement"
**Success Criteria** (what must be TRUE):
  1. `FillProcessor._update_position()` writes `stop_loss_price` and `take_profit_price` to the positions table
  2. After a crypto fill, `SELECT stop_loss_price, take_profit_price FROM positions WHERE environment='crypto'` returns non-NULL values
  3. `stop_polling._check_stop_levels()` reads valid stop/TP prices and can trigger liquidation when breached

Plans:
- [ ] 16-01-PLAN.md — Update FillProcessor to persist stop/TP prices and verify stop_polling integration

### Phase 17: Doc Housekeeping
**Goal**: Planning documents accurately reflect the current state of all 74 requirements, phase completion, and gap closure work
**Depends on**: Phase 16
**Requirements**: None (documentation only)
**Gap Closure:** Closes tech debt items from v1.0 re-audit
**Success Criteria** (what must be TRUE):
  1. REQUIREMENTS.md coverage counter matches actual state (all 74 complete)
  2. REQUIREMENTS.md descriptions for FEAT-09/10 and PAPER-02/09 accurately reflect gap closure work
  3. ROADMAP.md progress table shows correct plan counts for all phases

Plans:
- [ ] 17-01-PLAN.md — Fix stale counters, descriptions, and plan counts in planning docs

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dev Foundation | 3/3 | Complete   | 2026-03-06 |
| 2. Developer Experience | 4/4 | Complete   | 2026-03-06 |
| 3. Data Ingestion | 4/4 | Complete   | 2026-03-06 |
| 4. Data Storage and Validation | 4/4 | Complete   | 2026-03-06 |
| 5. Feature Engineering | 4/5 | In Progress|  |
| 6. RL Environments | 0/3 | Not started | - |
| 7. Agent Training and Validation | 2/3 | In Progress|  |
| 8. Paper Trading Core | 5/5 | Complete   | 2026-03-09 |
| 9. Automation and Monitoring | 4/4 | Complete   | 2026-03-09 |
| 10. Production Hardening | 8/8 | Complete    | 2026-03-10 |
| 11. Production Startup Wiring | 1/1 | Complete    | 2026-03-10 |
| 12. Schema Alignment and Emergency Triggers | 1/1 | Complete    | 2026-03-10 |
| 13. Model Path Fix and Reconciliation Scheduling | 1/1 | Complete    | 2026-03-10 |
| 14. Feature Pipeline Wiring | 1/1 | Complete    | 2026-03-10 |
| 15. Training CLI Observation Assembly | 1/1 | Complete   | 2026-03-11 |
| 16. Crypto Stop Price Persistence | 0/1 | Not started | - |
| 17. Doc Housekeeping | 0/1 | Not started | - |
