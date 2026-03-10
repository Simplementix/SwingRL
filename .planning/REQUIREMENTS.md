# Requirements: SwingRL

**Defined:** 2026-03-06
**Core Value:** Capital preservation through disciplined, automated risk management

## v1 Requirements

Requirements for M0-M6 software build (pre-live-trading).

### Environment & Tooling

- [x] **ENV-01**: Python 3.11 development environment with PyTorch MPS acceleration verified on M1 Mac
- [x] **ENV-02**: uv package manager installed on M1 Mac and homelab (Ubuntu 24.04)
- [x] **ENV-03**: GitHub repository with canonical directory structure (src/, config/, data/, db/, models/, tests/, scripts/, status/)
- [x] **ENV-04**: pyproject.toml with pinned dependencies and tool configuration (pytest, ruff, black, mypy)
- [x] **ENV-05**: Pre-commit hooks enforcing ruff, black, mypy, detect-secrets, and bandit on every commit
- [x] **ENV-06**: Dockerfile using python:3.11-slim with CPU-only PyTorch and non-root trader user (UID 1000)
- [x] **ENV-07**: docker-compose.yml with resource limits (2.5g mem, 1 cpu), bind mounts, env_file secrets, TZ=America/New_York
- [x] **ENV-08**: ci-homelab.sh script (git pull + docker compose build --no-cache + pytest + cleanup) validated via ssh homelab
- [x] **ENV-09**: CLAUDE.md project context file with SwingRL conventions (TDD, snake_case, type hints, Pydantic, pathlib, no hardcoded keys)
- [x] **ENV-10**: .claude/commands/ with SwingRL-specific skills
- [x] **ENV-11**: Pydantic v2 config schema (swingrl.yaml) implementing Doc 05 §10.7 with Field constraints, Literal enums, and model_validators
- [x] **ENV-12**: Scaffolded models/ directory with active/, shadow/, archive/ subdirectories and .gitkeep files
- [x] **ENV-13**: tests/test_smoke.py verifying all core package imports and tests/conftest.py with shared fixtures

### Data Pipeline

- [x] **DATA-01**: Alpaca OHLCV ingestion for 8 equity ETFs (SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK) via IEX feed
- [x] **DATA-02**: Binance.US 4-hour bar ingestion for BTC/USD and ETH/USD with rate limit monitoring (X-MBX-USED-WEIGHT headers)
- [x] **DATA-03**: 8+ year crypto historical backfill stitching international Binance archives (2017-2019) with Binance.US API (2019+), normalizing volume from base units to USD
- [x] **DATA-04**: FRED macro pipeline for Tier 1 series (VIX, T10Y2Y spread, DFF, CPI, unemployment) using ALFRED vintage data to prevent look-ahead bias
- [x] **DATA-05**: 12-step data validation checklist with quarantine to data_quarantine table for failed data
- [x] **DATA-06**: DuckDB analytical database (market_data.ddb) with tables created incrementally per milestone
- [x] **DATA-07**: SQLite operational database (trading_ops.db) with tables created incrementally per milestone
- [x] **DATA-08**: Cross-database joins via DuckDB sqlite_scanner extension
- [x] **DATA-09**: "Store lowest, aggregate up" strategy — only store daily/4H bars, compute weekly/monthly on-the-fly via DuckDB
- [x] **DATA-10**: Corporate action handling (stock splits, dividends) with corporate_actions table
- [x] **DATA-11**: Cross-source validation (Alpaca vs yfinance closing prices, weekly)
- [x] **DATA-12**: Data ingestion logging to data_ingestion_log table with run_id, timing, and row counts
- [x] **DATA-13**: Alerter module (src/monitoring/alerter.py) with Discord webhook integration, level-based routing, and daily digest batching

### Feature Engineering

- [x] **FEAT-01**: Technical indicators via pandas_ta: SMA(50,200) as price ratios, RSI(14), MACD line + histogram, Bollinger Band position (0-1), ATR(14) as % of price, Volume/Volume_SMA(20) ratio
- [x] **FEAT-02**: Derived features: log returns (1d, 5d, 20d), Bollinger Band width
- [x] **FEAT-03**: Fundamental features (equities only): P/E ratio (sector-relative z-score), earnings growth, debt-to-equity, dividend yield — updated quarterly
- [x] **FEAT-04**: Macro regime features (shared): VIX z-score, yield curve spread/direction, Fed Funds 90-day change, CPI YoY, unemployment 3-month direction — forward-filled via ASOF JOIN
- [x] **FEAT-05**: HMM regime detection: 2-state Gaussian HMM per environment (SPY for equity, BTC for crypto), producing P(bull) and P(bear) continuous probabilities
- [x] **FEAT-06**: Rolling z-score normalization with per-environment windows (252 bars equity, 360 bars crypto)
- [x] **FEAT-07**: Observation space assembly: 156 dimensions equity, 45 dimensions crypto
- [x] **FEAT-08**: Feature addition protocol: new features kept only if validation Sharpe improves by at least 0.05 in A/B test
- [x] **FEAT-09**: Correlation pruning: remove features with pairwise Pearson r > 0.85
- [x] **FEAT-10**: Weekly-derived features (SMA trend direction, weekly RSI-14) computed from aggregated weekly bars, not fetched separately
- [x] **FEAT-11**: Per-environment feature tables in DuckDB: features_equity (DATE key), features_crypto (TIMESTAMP key)

### RL Training

- [x] **TRAIN-01**: StockTradingEnv — Gymnasium-compatible, daily bars, 8 ETFs, 156-dim observation, continuous action space (target portfolio weights)
- [x] **TRAIN-02**: CryptoTradingEnv — Gymnasium-compatible, 4H bars, BTC/ETH, 45-dim observation, continuous action space
- [x] **TRAIN-03**: PPO agent training with hyperparameters: lr=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01, clip_range=0.2, gamma=0.99
- [x] **TRAIN-04**: A2C agent training with hyperparameters: lr=0.0007, n_steps=5, vf_coef=0.5, ent_coef=0.01, gamma=0.99
- [x] **TRAIN-05**: SAC agent training with hyperparameters: lr=0.0003, batch_size=256, tau=0.005, ent_coef="auto", gamma=0.99
- [x] **TRAIN-06**: Sharpe-weighted softmax ensemble blending using per-environment validation windows (63 trading days equity, 126 4H bars crypto)
- [x] **TRAIN-07**: Rolling 20-day Sharpe ratio reward function with expanding-window warmup for first 19 bars
- [x] **TRAIN-08**: VecNormalize statistics handling — frozen (training=False) during inference to prevent train/serve skew
- [x] **TRAIN-09**: Signal deadzone: actions within +/-0.02 of zero mapped to "hold" to suppress meaningless tiny trades
- [x] **TRAIN-10**: Adaptive validation windows: shrink by 50% during high turbulence, expand during calm
- [x] **TRAIN-11**: Episode structure: equity 252-day segments, crypto 540 4H bars (3 months) with random start within training window
- [x] **TRAIN-12**: Model metadata and ensemble weights stored in DuckDB for audit trail

### Validation

- [x] **VAL-01**: Walk-forward backtesting framework with 3-month test folds for both environments
- [x] **VAL-02**: 200-bar purge gap and 1% embargo between training and test folds to prevent data leakage
- [x] **VAL-03**: Performance metric calculators: Sharpe, Sortino, Calmar, Rachev ratios, MDD, average drawdown, drawdown duration
- [x] **VAL-04**: Trade-level metrics: win rate, Profit Factor, trade frequency (2-10 trades per week)
- [x] **VAL-05**: ConvergenceCallback for Stable Baselines3 — early stopping if mean reward improvement < 1% over 10 evaluations
- [x] **VAL-06**: Overfitting detection: in-sample vs out-of-sample Sharpe gap (< 20% healthy, 20-50% marginal, > 50% reject)
- [x] **VAL-07**: Validation gates: Sharpe > 0.7 per env, MDD < 15%, Profit Factor > 1.5, overfitting gap < 20%
- [x] **VAL-08**: Backtest results stored in DuckDB backtest_results table per model and fold

### Paper Trading & Deployment

- [x] **PAPER-01**: Alpaca paper trading connection for equity environment
- [x] **PAPER-02**: Binance.US simulated fills for crypto environment (real-time prices, local fill recording)
- [x] **PAPER-03**: Two-tier risk management veto layer: Tier 1 per-environment budgets + Tier 2 global portfolio constraints
- [x] **PAPER-04**: Circuit breakers: equity -10% DD / -2% daily, crypto -12% DD / -3% daily, global -15% DD / -3% combined
- [x] **PAPER-05**: Circuit breaker cooldown: 5 business days equity, 3 calendar days crypto, with 25%/50%/75%/100% ramp-up
- [x] **PAPER-06**: Halt-state persistence in circuit_breaker_events table across container restarts
- [x] **PAPER-07**: Position sizing: modified Kelly criterion (quarter-Kelly Phase 1), 2% max risk per trade, ATR(2x) stop-losses
- [x] **PAPER-08**: Binance.US $10 minimum order floor: max(kelly_sized_amount, $10.00)
- [x] **PAPER-09**: 5-stage execution middleware: Signal Interpreter, Position Sizer, Order Validator, Exchange Adapter, Fill Processor
- [x] **PAPER-10**: Bracket orders: OTO for Alpaca (ATR stop-loss, R:R take-profit), two-step OCO for Binance.US
- [x] **PAPER-11**: Cost gate: reject orders where estimated round-trip transaction costs exceed 2.0% of order value
- [x] **PAPER-12**: APScheduler: equity daily at 4:15 PM ET, crypto every 4H at 5 min past bar close, pre-cycle halt checks
- [x] **PAPER-13**: Discord webhook alerting: trade executions, circuit breakers, daily summary (6 PM ET), stuck agent detection
- [x] **PAPER-14**: Stuck agent detection: alert if environment stays 100% cash for 10 equity days or 30 crypto cycles
- [x] **PAPER-15**: Streamlit dashboard for system health monitoring with traffic-light status
- [x] **PAPER-16**: Healthchecks.io dead man's switch (70-min crypto window, 25-hr equity window)
- [x] **PAPER-17**: Wash sale tracker: log realized losses and flag 30-day wash sale violations
- [x] **PAPER-18**: Production database seeding from M1 Mac historical archive (7-step procedure)
- [x] **PAPER-19**: Full Docker production deployment on homelab with TRADING_MODE=paper
- [x] **PAPER-20**: Turbulence index crash protection: liquidate and halt if turbulence > 90th percentile

### Hardening & Sentiment

- [ ] **HARD-01**: Jupyter analysis notebooks for weekly performance review (portfolio curves, trade logs, risk metrics)
- [x] **HARD-02**: Error handling with retry logic and exponential backoff for API timeouts and data feed gaps
- [ ] **HARD-03**: FinBERT sentiment pipeline (ProsusAI/finbert): score daily headlines from Alpaca News + Finnhub
- [ ] **HARD-04**: A/B sentiment experiment: train identical agents with/without sentiment, evaluate Sharpe improvement
- [x] **HARD-05**: Structured JSON logging to bind-mounted logs/ volume with comprehensive audit trails

### Production Hardening

- [x] **PROD-01**: Backup automation: daily SQLite (trading_ops.db), weekly DuckDB (market_data.ddb), monthly off-site rsync via Tailscale
- [x] **PROD-02**: deploy_model.sh: SCP model transfer from M1 Mac to homelab with integrity verification and automated smoke test
- [ ] **PROD-03**: Shadow mode: new models run parallel with active model generating hypothetical trades (10 equity days, 30 crypto cycles)
- [ ] **PROD-04**: Shadow mode auto-promotion criteria: Shadow Sharpe >= Active Sharpe, Shadow MDD <= 120% Active MDD, no circuit breakers
- [x] **PROD-05**: Model lifecycle: Training -> Shadow -> Active -> Archive -> Deletion
- [ ] **PROD-06**: Security review: non-root containers, env_file (chmod 600), Binance.US IP allowlisting, 90-day key rotation (staggered)
- [ ] **PROD-07**: emergency_stop.py: four-tier kill switch — halt jobs, cancel orders, liquidate crypto immediately, queue equity for market open
- [ ] **PROD-08**: Disaster recovery test: stop container, delete volumes, restore from backup, verify system resumes correctly
- [ ] **PROD-09**: 9-step quarterly recovery checklist (starting M6)

## v2 Requirements

Deferred to future milestones (M7+).

### Live Trading (M7)

- **LIVE-01**: Transition to real money at $447 ($47 crypto + $400 equity) with Phase 1 risk overrides (10% max DD, quarter-Kelly)
- **LIVE-02**: Per-environment trading mode toggles (paper/live independent switching)

### Capital Scaling (Phase 2)

- **SCALE-01**: Phase 2 transition criteria: Sharpe > 0.7 for 3 consecutive months, no circuit breakers for 60 days
- **SCALE-02**: Deposit-driven scaling to $1,000+ total with model retraining for updated capital

### Options Environment (Phase 3, M8-M11)

- **OPT-01**: IBKR Pro integration and IB Gateway Docker container
- **OPT-02**: CreditSpreadEnv for SPX options with multi-discrete action space
- **OPT-03**: OptionsRiskManager with VIX-based throttling
- **OPT-04**: Three-environment capital split (55% equity / 20% crypto / 25% options)
- **OPT-05**: Options-derived features for equity environment (IV rank, put/call OI, VRP, GEX, skew)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time WebSocket streaming | Scheduled batch processing sufficient for swing trading timeframes |
| On-chain crypto metrics | Price-based features only for v1; may add in v2+ |
| Mobile app | Streamlit dashboard + Discord alerts sufficient for monitoring |
| Multi-user support | Single operator system by design |
| Per-ticker specialist models | Generalist pooled model with 8x more training data outperforms |
| Stochastic / Williams %R / CCI | Redundant with RSI-14 — excluded per Doc 08 §12 |
| OAuth / social login | Not applicable — system has no user-facing web interface |
| Real-time chat / notifications beyond Discord | Discord webhooks are the canonical alert channel |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| ENV-04 | Phase 1 | Complete |
| ENV-05 | Phase 1 | Complete |
| ENV-06 | Phase 1 | Complete |
| ENV-07 | Phase 1 | Complete |
| ENV-08 | Phase 1 | Complete |
| ENV-09 | Phase 2 | Complete |
| ENV-10 | Phase 2 | Complete |
| ENV-11 | Phase 2 | Complete |
| ENV-12 | Phase 2 | Complete |
| ENV-13 | Phase 2 | Complete |
| DATA-01 | Phase 3 | Complete |
| DATA-02 | Phase 3 | Complete |
| DATA-03 | Phase 3 | Complete |
| DATA-04 | Phase 3 | Complete |
| DATA-05 | Phase 3 | Complete |
| DATA-06 | Phase 4 | Complete |
| DATA-07 | Phase 4 | Complete |
| DATA-08 | Phase 4 | Complete |
| DATA-09 | Phase 4 | Complete |
| DATA-10 | Phase 4 | Complete |
| DATA-11 | Phase 4 | Complete |
| DATA-12 | Phase 4 | Complete |
| DATA-13 | Phase 4 | Complete |
| FEAT-01 | Phase 5 | Complete |
| FEAT-02 | Phase 5 | Complete |
| FEAT-03 | Phase 5 | Complete |
| FEAT-04 | Phase 5 | Complete |
| FEAT-05 | Phase 5 | Complete |
| FEAT-06 | Phase 5 | Complete |
| FEAT-07 | Phase 5 | Complete |
| FEAT-08 | Phase 5 | Complete |
| FEAT-09 | Phase 5 | Complete |
| FEAT-10 | Phase 5 | Complete |
| FEAT-11 | Phase 5 | Complete |
| TRAIN-01 | Phase 6 | Complete |
| TRAIN-02 | Phase 6 | Complete |
| TRAIN-07 | Phase 6 | Complete |
| TRAIN-08 | Phase 6 | Complete |
| TRAIN-09 | Phase 6 | Complete |
| TRAIN-10 | Phase 6 | Complete |
| TRAIN-11 | Phase 6 | Complete |
| TRAIN-03 | Phase 7 | Complete |
| TRAIN-04 | Phase 7 | Complete |
| TRAIN-05 | Phase 7 | Complete |
| TRAIN-06 | Phase 7 | Complete |
| TRAIN-12 | Phase 7 | Complete |
| VAL-01 | Phase 7 | Complete |
| VAL-02 | Phase 7 | Complete |
| VAL-03 | Phase 7 | Complete |
| VAL-04 | Phase 7 | Complete |
| VAL-05 | Phase 7 | Complete |
| VAL-06 | Phase 7 | Complete |
| VAL-07 | Phase 7 | Complete |
| VAL-08 | Phase 7 | Complete |
| PAPER-01 | Phase 8 | Complete |
| PAPER-02 | Phase 8 | Complete |
| PAPER-03 | Phase 8 | Complete |
| PAPER-04 | Phase 8 | Complete |
| PAPER-05 | Phase 8 | Complete |
| PAPER-06 | Phase 8 | Complete |
| PAPER-07 | Phase 8 | Complete |
| PAPER-08 | Phase 8 | Complete |
| PAPER-09 | Phase 8 | Complete |
| PAPER-10 | Phase 8 | Complete |
| PAPER-11 | Phase 8 | Complete |
| PAPER-18 | Phase 8 | Complete |
| PAPER-19 | Phase 8 | Complete |
| PAPER-20 | Phase 8 | Complete |
| PAPER-12 | Phase 9 | Complete |
| PAPER-13 | Phase 9 | Complete |
| PAPER-14 | Phase 9 | Complete |
| PAPER-15 | Phase 9 | Complete |
| PAPER-16 | Phase 9 | Complete |
| PAPER-17 | Phase 9 | Complete |
| HARD-01 | Phase 10 | Pending |
| HARD-02 | Phase 10 | Complete |
| HARD-03 | Phase 10 | Pending |
| HARD-04 | Phase 10 | Pending |
| HARD-05 | Phase 10 | Complete |
| PROD-01 | Phase 10 | Complete |
| PROD-02 | Phase 10 | Complete |
| PROD-03 | Phase 10 | Pending |
| PROD-04 | Phase 10 | Pending |
| PROD-05 | Phase 10 | Complete |
| PROD-06 | Phase 10 | Pending |
| PROD-07 | Phase 10 | Pending |
| PROD-08 | Phase 10 | Pending |
| PROD-09 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 74 total
- Mapped to phases: 74
- Unmapped: 0

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-06 after roadmap creation — traceability complete*
