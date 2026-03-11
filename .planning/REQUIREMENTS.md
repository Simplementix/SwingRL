# Requirements: SwingRL

**Defined:** 2026-03-10
**Core Value:** Capital preservation through disciplined, automated risk management — the system must never lose more than it can recover from, prioritizing survival over returns.

## v1.1 Requirements

Requirements for operational deployment milestone. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: System ingests maximum available equity OHLCV history from Alpaca (all 8 ETFs) into homelab DuckDB
- [x] **DATA-02**: System ingests maximum available crypto 4H OHLCV history from Binance.US (BTC/ETH) into homelab DuckDB
- [x] **DATA-03**: System ingests FRED macro data (VIX, T10Y2Y, DFF, CPI, UNRATE) aligned to OHLCV date ranges
- [x] **DATA-04**: All observation vector dimensions are populated without NaN (156-dim equity, 45-dim crypto) after feature pipeline runs
- [x] **DATA-05**: Data ingestion runs directly on the homelab Docker container (no M1 Mac dependency)

### Training

- [ ] **TRAIN-01**: Baseline training runs with default hyperparameters (no tuning) for all 3 algos on equity (3-year rolling window)
- [ ] **TRAIN-02**: Baseline training runs with default hyperparameters (no tuning) for all 3 algos on crypto (1-year rolling window)
- [ ] **TRAIN-03**: Walk-forward validation records Sharpe, max drawdown, and profit factor per agent and ensemble
- [ ] **TRAIN-04**: If baseline ensemble Sharpe < 0.5, targeted tuning phase runs (PPO first, then A2C/SAC)
- [ ] **TRAIN-05**: Sharpe-weighted ensemble blending produces valid ensemble weights from backtest results
- [ ] **TRAIN-06**: Trained models deploy to `models/active/{env}/{algo}/` with VecNormalize files
- [ ] **TRAIN-07**: Training success gate: ensemble Sharpe > 1.0 and MDD < 15% before proceeding to paper trading

### Deployment

- [ ] **DEPLOY-01**: Docker production stack starts on homelab with both services healthy (swingrl + dashboard)
- [ ] **DEPLOY-02**: Homelab `.env` configured with all required API keys and webhook URLs
- [ ] **DEPLOY-03**: Startup validation runs on container boot: DB tables, active models, broker creds, Discord ping
- [ ] **DEPLOY-04**: APScheduler fires equity cycle (16:15 ET) and crypto cycle (every 4H) correctly with timezone handling
- [ ] **DEPLOY-05**: Paper trading executes end-to-end: signal → size → validate → submit → fill for both environments

### Discord

- [ ] **DISC-01**: Discord webhook URLs created and configured for critical/warning and daily summary channels
- [ ] **DISC-02**: Critical alerts fire on circuit breaker trigger, broker auth failure, stuck agent
- [ ] **DISC-03**: Daily summary embed posts with Sharpe, P&L, and position summary
- [ ] **DISC-04**: Retraining alerts fire: started, completed (with old vs new Sharpe), failed, shadow promoted/rejected

### Retraining

- [ ] **RETRAIN-01**: Equity retraining job runs monthly via APScheduler as subprocess (not in thread pool)
- [ ] **RETRAIN-02**: Crypto retraining job runs biweekly via APScheduler as subprocess
- [ ] **RETRAIN-03**: Walk-forward validation gate runs on newly trained model before shadow deployment
- [ ] **RETRAIN-04**: New model deploys to shadow, existing shadow promotion logic evaluates and auto-promotes if better
- [ ] **RETRAIN-05**: Shadow promoter bootstrap guard prevents promotion when trades table is empty

### Monitoring

- [ ] **MON-01**: Rolling Sharpe degradation alert fires when paper trading Sharpe drops below configurable threshold
- [ ] **MON-02**: Paper trading validation log tracks daily P&L, drawdown, and fill latency in structured format
- [ ] **MON-03**: Healthchecks.io dead man's switch pings after each trading cycle

### Documentation

- [ ] **DOC-01**: Operator runbook covers initial setup (homelab, Docker, .env, data ingestion)
- [ ] **DOC-02**: Operator runbook covers first training run with expected output and validation
- [ ] **DOC-03**: Operator runbook covers retraining workflow and shadow promotion decisions
- [ ] **DOC-04**: Operator runbook covers emergency stop, model rollback, and disaster recovery
- [ ] **DOC-05**: Operator runbook covers common failure diagnosis with troubleshooting steps

## Future Requirements

Deferred to v1.2+ milestones.

### Live Trading

- **LIVE-01**: Per-environment trading mode toggles (paper/live independent switching)
- **LIVE-02**: Live trading deployment at $447 ($47 crypto + $400 equity) with Phase 1 risk overrides
- **LIVE-03**: Capital scaling strategy: deposit-driven growth from $447 to $1K+

### Advanced Monitoring

- **ADVMON-01**: Model performance attribution by asset (which ETFs drive returns)
- **ADVMON-02**: Feature importance tracking across retraining cycles

## Out of Scope

| Feature | Reason |
|---------|--------|
| SPX options / IBKR integration | Requires $2K+ capital (Phase 3) |
| GPU training on homelab | Intel i5 has no GPU; CPU training viable for SB3 |
| Continuous/online retraining | Swing trading frequency too low for online RL; noise not signal |
| Separate training Docker container | Single-node homelab; subprocess isolation sufficient |
| MLflow / experiment tracking server | DuckDB model_runs table sufficient for single operator |
| Kubernetes / Docker Swarm | Single-node server; Docker Compose sufficient |
| Automated paper → live promotion | Capital at risk; live promotion must be manual operator decision |
| M1 Mac training dependency | Homelab CPU trains directly; no cross-device model transfer |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 18 | Complete |
| DATA-02 | Phase 18 | Complete |
| DATA-03 | Phase 18 | Complete |
| DATA-04 | Phase 18 | Complete |
| DATA-05 | Phase 18 | Complete |
| TRAIN-01 | Phase 19 | Pending |
| TRAIN-02 | Phase 19 | Pending |
| TRAIN-03 | Phase 19 | Pending |
| TRAIN-04 | Phase 19 | Pending |
| TRAIN-05 | Phase 19 | Pending |
| TRAIN-06 | Phase 19 | Pending |
| TRAIN-07 | Phase 19 | Pending |
| DEPLOY-01 | Phase 20 | Pending |
| DEPLOY-02 | Phase 20 | Pending |
| DEPLOY-03 | Phase 20 | Pending |
| DEPLOY-04 | Phase 20 | Pending |
| DEPLOY-05 | Phase 20 | Pending |
| DISC-01 | Phase 21 | Pending |
| DISC-02 | Phase 21 | Pending |
| DISC-03 | Phase 21 | Pending |
| DISC-04 | Phase 21 | Pending |
| RETRAIN-01 | Phase 22 | Pending |
| RETRAIN-02 | Phase 22 | Pending |
| RETRAIN-03 | Phase 22 | Pending |
| RETRAIN-04 | Phase 22 | Pending |
| RETRAIN-05 | Phase 22 | Pending |
| MON-01 | Phase 23 | Pending |
| MON-02 | Phase 23 | Pending |
| MON-03 | Phase 23 | Pending |
| DOC-01 | Phase 24 | Pending |
| DOC-02 | Phase 24 | Pending |
| DOC-03 | Phase 24 | Pending |
| DOC-04 | Phase 24 | Pending |
| DOC-05 | Phase 24 | Pending |

**Coverage:**
- v1.1 requirements: 34 total
- Mapped to phases: 34
- Unmapped: 0

---
*Requirements defined: 2026-03-10*
*Last updated: 2026-03-10 after roadmap creation*
