# SwingRL

## What This Is

An automated reinforcement learning swing trading system that deploys PPO/A2C/SAC Sharpe-weighted ensembles across US equities (daily bars) and cryptocurrency (4-hour bars). Designed as a hands-off, capital-preserving trading platform for a solo operator, running on a two-machine architecture: M1 Mac for training and an always-on homelab server for production inference and execution.

## Core Value

Capital preservation through disciplined, automated risk management — the system must never lose more than it can recover from, prioritizing survival over returns.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Dev environment with Python 3.11, PyTorch MPS, uv package manager, and full tooling (ruff, black, mypy, bandit, detect-secrets, pre-commit)
- [ ] GitHub repo with canonical directory structure (src/, config/, data/, db/, models/, tests/, scripts/, status/)
- [ ] Docker scaffold (python:3.11-slim, CPU-only PyTorch, non-root trader user) validated on x86 homelab
- [ ] Project CLAUDE.md and .claude/commands/ skills for SwingRL conventions
- [ ] Cross-platform validation via ci-homelab.sh (git pull + docker build + pytest) after every milestone
- [ ] Data ingestion pipeline for Alpaca (equities, IEX feed), Binance.US (crypto 4H bars), and FRED (macro indicators)
- [ ] 8+ year crypto historical backfill stitching international Binance archives (2017-2019) with Binance.US API (2019+)
- [ ] 12-step data validation checklist with quarantine for failed data
- [ ] Feature engineering: technical indicators (SMA, RSI, MACD, Bollinger, ATR), fundamentals (equities), macro regime (FRED), HMM regime probabilities (2-state per environment)
- [ ] Observation space assembly: 156 dimensions (equity), 45 dimensions (crypto)
- [ ] Gymnasium-compatible environments: StockTradingEnv (daily, 8 ETFs) and CryptoTradingEnv (4H, BTC/ETH)
- [ ] PPO/A2C/SAC agent training with specified hyperparameters per environment
- [ ] Sharpe-weighted softmax ensemble blending (validation windows: 63 trading days equity, 126 4H bars crypto)
- [ ] Walk-forward backtesting with 3-month test folds, 200-bar purge gap, and convergence diagnostics
- [ ] Performance validation gates: Sharpe > 0.7 per env, MDD < 15%, Profit Factor > 1.5, overfitting gap < 20%
- [ ] Paper trading connections: Alpaca paper API (equities), Binance.US simulated fills (crypto)
- [ ] Two-tier risk management veto layer: per-environment limits + global portfolio constraints
- [ ] Circuit breakers with cooldown periods and graduated ramp-up on resume
- [ ] Position sizing: modified Kelly criterion (quarter-Kelly Phase 1), 2% max risk per trade, ATR-based stops
- [ ] APScheduler automation: equity daily 4:15 PM ET, crypto every 4 hours, pre-cycle halt checks
- [ ] Discord webhook alerting: trade executions, circuit breakers, daily summary, stuck agent detection
- [ ] Streamlit dashboard for system health monitoring
- [ ] Healthchecks.io dead man's switch (70min crypto, 25hr equity alert windows)
- [ ] Jupyter analysis notebooks for weekly performance review
- [ ] Error handling with retry logic and exponential backoff for API timeouts
- [ ] FinBERT sentiment pipeline (ProsusAI/finbert) with A/B experiment
- [ ] Backup automation: daily SQLite, weekly DuckDB, monthly off-site rsync
- [ ] Model deployment workflow: deploy_model.sh (M1 Mac to homelab via SCP + smoke test)
- [ ] Security review: non-root containers, env_file secrets, Binance.US IP allowlisting, 90-day key rotation
- [ ] emergency_stop.py: four-tier manual/automated kill switch
- [ ] Disaster recovery test: stop container, delete volumes, restore from backup, verify resume

### Out of Scope

- Live trading deployment (M7+) — this scope covers M0-M6 software build only
- SPX options environment and IBKR integration (Phase 3, M8-M11)
- Mobile app or web frontend beyond Streamlit dashboard
- Multi-user support — single operator system
- Real-time streaming data (WebSocket) — scheduled batch processing only
- On-chain crypto metrics — price-based features only for v1

## Context

**Architecture:** Two-machine setup. M1 MacBook Pro (32GB RAM) handles development, backtesting, and model training via PyTorch MPS acceleration. Intel i5-13th gen homelab server (64GB RAM) runs production inference and execution in Docker 24/7. Models are platform-independent (PyTorch CPU tensors) and transfer between ARM and x86 without conversion.

**Brokers:** Dedicated broker per asset class — Alpaca for equities (commission-free, paper + live), Binance.US for crypto (0.10% maker/taker, $10 minimum order floor). IBKR Pro for SPX options deferred to Phase 3.

**Database:** SQLite (trading_ops.db) for transactional OLTP workloads. DuckDB (market_data.ddb) for columnar OLAP analytics. Cross-database joins via DuckDB's sqlite_scanner. Schema defined incrementally per milestone (28 total tables: 10 SQLite + 18 DuckDB).

**Capital:** Phase 1 starts at $447 ($47 crypto + $400 equity). Deposit-driven scaling to $1K+ (Phase 2) and $2K+ (Phase 3). Growth through returns alone is negligible at this scale.

**Existing specs:** 15 specification documents in NotebookLM notebook "SwingRL - RL Swing Trading System" covering project overview, framework comparison, RL fundamentals, performance metrics, architecture/roadmap, key questions, security/costs, data architecture, monitoring, Claude Code skills, gap analysis, specification summary, SPX options, database schema, and failure runbook.

## Constraints

- **Python version**: 3.11 — FinRL compatibility constraint (pyfolio dependency breaks on 3.12+)
- **Package manager**: uv — must be installed on both M1 Mac and homelab (Ubuntu 24.04)
- **Training hardware**: M1 Mac only (MPS acceleration). Homelab does NOT train models.
- **Budget**: Free data sources only (Alpaca IEX, yfinance, FRED, CoinGecko). $295 reserved for DiscountOptionData in Phase 3.
- **Homelab validation**: Every milestone must pass ci-homelab.sh (Docker build + pytest on x86) via `ssh homelab`
- **Config schema**: Pydantic v2 typed schema from Doc 05 §10.7 — implement as-is, not redesign
- **Milestones sequential**: M2 finishes before M3 starts (no interleaving)
- **Dashboard**: Streamlit (not static HTML)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python 3.11 over 3.12+ | FinRL/pyfolio dependency compatibility | — Pending |
| uv over pip/poetry | Fast, modern, pyproject.toml-native | — Pending |
| Streamlit over static HTML | Interactive dashboard, better for deep analysis | — Pending |
| Docker only for CI (no native validation) | Production runs in Docker, validate what you ship | — Pending |
| git pull for ci-homelab.sh | Fast, validates committed code, Docker cache works | — Pending |
| Sequential M2→M3 (no interleaving) | Cleaner milestone boundaries | — Pending |
| Incremental DB schema | Create tables as each milestone needs them, not all 28 upfront | — Pending |
| FinBERT included in M5 | Observation space feature, A/B experiment to validate value | — Pending |
| Project-scoped Claude skills | CLAUDE.md + .claude/commands/ specific to SwingRL | — Pending |
| Scaffold models/ in M0 | Empty active/shadow/archive dirs with .gitkeep | — Pending |

---
*Last updated: 2026-03-06 after initialization*
