# SwingRL

## What This Is

An automated reinforcement learning swing trading system deploying PPO/A2C/SAC Sharpe-weighted ensembles across US equities (8 ETFs, daily bars) and cryptocurrency (BTC/ETH, 4-hour bars). Runs as a hands-off, capital-preserving paper trading platform for a solo operator on a two-machine architecture: M1 Mac for training and an always-on homelab server for production inference and execution via Docker.

## Core Value

Capital preservation through disciplined, automated risk management — the system must never lose more than it can recover from, prioritizing survival over returns.

## Requirements

### Validated

- ✓ Dev environment: Python 3.11, PyTorch MPS, uv, full tooling (ruff, mypy, bandit, detect-secrets, pre-commit) — v1.0
- ✓ GitHub repo with canonical directory structure and Docker CI on x86 homelab — v1.0
- ✓ Pydantic v2 config schema with CLAUDE.md conventions and .claude/commands/ skills — v1.0
- ✓ Multi-source data pipeline: Alpaca (8 equity ETFs), Binance.US (BTC/ETH 4H), FRED (5 macro series) — v1.0
- ✓ 8+ year crypto backfill stitching Binance archives (2017-2019) with Binance.US API (2019+) — v1.0
- ✓ 12-step data validation with quarantine, DuckDB/SQLite dual-database storage — v1.0
- ✓ Feature engineering: 156-dim equity and 45-dim crypto observation vectors — v1.0
- ✓ HMM regime detection, rolling z-score normalization, correlation pruning — v1.0
- ✓ Gymnasium-compatible RL environments: StockTradingEnv and CryptoTradingEnv — v1.0
- ✓ PPO/A2C/SAC training with Sharpe-weighted softmax ensemble blending — v1.0
- ✓ Walk-forward backtesting with 200-bar purge gap and 4 validation gates — v1.0
- ✓ Paper trading: Alpaca paper API (equity), Binance.US simulated fills (crypto) — v1.0
- ✓ 5-stage execution middleware with two-tier risk veto and circuit breakers — v1.0
- ✓ Position sizing: quarter-Kelly, ATR stops, $10 crypto floor, cost gate — v1.0
- ✓ APScheduler automation: equity daily 4:15 PM ET, crypto every 4H — v1.0
- ✓ Discord alerting, Streamlit dashboard, Healthchecks.io dead man's switch — v1.0
- ✓ FinBERT sentiment pipeline with A/B experiment infrastructure — v1.0
- ✓ Backup automation, deploy_model.sh, shadow mode with auto-promotion — v1.0
- ✓ Security review, emergency_stop.py (4-tier), disaster recovery tested — v1.0

### Active

- [ ] Historical data ingestion with complete dimension alignment (OHLCV + macro + features)
- [ ] Agent training pipeline: PPO/A2C/SAC for equity and crypto with walk-forward validation
- [ ] Homelab Docker deployment with production config and environment setup
- [ ] Discord webhook setup with full alert suite (critical, warning, daily summary)
- [ ] Automated retraining: equity monthly, crypto biweekly with shadow promotion
- [ ] Operator runbook with detailed walkthroughs for all workflows
- [ ] 6-month paper trading validation period on homelab

### Out of Scope

- SPX options environment and Charles Schwab integration (Phase 3, M8-M11) — requires $2K+ capital
- Mobile app or web frontend beyond Streamlit dashboard — Discord + Streamlit sufficient
- Multi-user support — single operator system by design
- Real-time streaming data (WebSocket) — scheduled batch processing sufficient for swing trading
- On-chain crypto metrics — price-based features only for v1
- Per-ticker specialist models — generalist pooled model with 8x more training data outperforms

## Context

**Shipped v1.0** with 33,316 LOC Python (15,288 src/ + 18,028 tests/) across 848+ tests.

**Tech stack:** Python 3.11, Stable Baselines3 (PPO/A2C/SAC), Gymnasium, DuckDB, SQLite, APScheduler, Streamlit, structlog, Pydantic v2, Docker, uv.

**Architecture:** Two-machine setup. M1 MacBook Pro (32GB RAM) for development, backtesting, model training (MPS). Intel i5-13th gen homelab (30-36GB FREE) for 24/7 Docker production. Models transfer ARM→x86 without conversion.

**Brokers:** Alpaca (equities, commission-free), Binance.US (crypto, 0.10% maker/taker, $10 floor). Charles Schwab (SPX options) deferred.

**Database:** SQLite (trading_ops.db, OLTP) + DuckDB (market_data.ddb, OLAP). 28 tables (10 SQLite + 18 DuckDB). Cross-database joins via sqlite_scanner.

**Capital:** Phase 1: $447 ($47 crypto + $400 equity). Deposit-driven scaling to $1K+ (Phase 2), $2K+ (Phase 3).

**Known tech debt:**

- pipeline.py sets fundamental columns to 0.0 (live FundamentalFetcher integration deferred)
- Placeholder Sharpe ratios in train.py for initial ensemble weights (backtest.py provides real values)

## Constraints

- **Python version**: 3.11 — FinRL/pyfolio compatibility (breaks on 3.12+)
- **Package manager**: uv — installed on both M1 Mac and homelab
- **Training hardware**: Homelab CPU (Intel i5, 30-36GB RAM FREE). CPU-only training for automated retraining.
- **Budget**: Free data sources only. $295 reserved for DiscountOptionData (Phase 3).
- **Homelab validation**: Every milestone must pass ci-homelab.sh
- **Config schema**: Pydantic v2 typed schema (Doc 05 §10.7)
- **Dashboard**: Streamlit

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python 3.11 over 3.12+ | FinRL/pyfolio dependency compatibility | ✓ Good — no compatibility issues |
| uv over pip/poetry | Fast, modern, pyproject.toml-native | ✓ Good — seamless cross-platform |
| Streamlit over static HTML | Interactive dashboard, deep analysis | ✓ Good — separate container, read-only DB |
| Docker only for CI | Validate what ships to production | ✓ Good — caught ARM/x86 issues |
| Incremental DB schema | Create tables per phase, not all 28 upfront | ✓ Good — clean phase boundaries |
| stockstats over pandas_ta | pandas_ta requires Python 3.12+ | ✓ Good — FinRL-native TA library |
| ruff-format replaces black in pre-commit | Avoid formatting conflicts between tools | ✓ Good — single formatter |
| Dedicated broker per asset class | Alpaca equities, Binance.US crypto, Charles Schwab options | ✓ Good — clean separation |
| CPU-only torch in Docker | Homelab has no GPU; MPS stays on M1 Mac | ✓ Good — smaller image |
| FinBERT as optional dep group | 2GB+ install when disabled via `[sentiment]` | ✓ Good — default install stays light |
| Shadow mode zero portfolio weights | Shadow has no real positions | ✓ Good — clean hypothetical trades |
| Quarter-Kelly position sizing | Conservative Phase 1 risk management | — Pending live validation |

| Homelab trains (CPU) | Fully hands-off operation; M1 Mac not required for retraining | — Pending |

## Current Milestone: v1.1 Operational Deployment

**Goal:** Get SwingRL running on the homelab in paper trading mode with automated retraining, Discord alerts, and operator documentation — fully hands-off after initial setup.

**Target features:**

- Complete data ingestion pipeline (max historical depth, aligned dimensions)
- Trained PPO/A2C/SAC agents for both equity and crypto
- Homelab Docker deployment with paper trading
- Discord webhook with full alert suite
- Automated retraining (equity monthly, crypto biweekly) with shadow promotion
- Comprehensive operator runbook with detailed walkthroughs

---
*Last updated: 2026-03-10 after v1.1 milestone start*
