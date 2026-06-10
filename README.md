# SwingRL

RL-based swing trading system using PPO/A2C/SAC ensemble agents on two environments: equity daily (8 ETFs via Alpaca) and crypto 4H (BTC/ETH via Binance.US). Built on Stable Baselines3 with capital preservation as the primary constraint.

**Status:** v1.0 MVP shipped — 74/74 requirements met across 17 phases. 33K+ LOC Python, 848+ tests, 85% coverage minimum.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Paper Trading](#paper-trading)
  - [Production Deployment](#production-deployment)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)
- [Scripts Reference](#scripts-reference)
- [License](#license)

## Features

- **Dual-environment trading** — equity daily bars (8 ETFs: SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK) and crypto 4H bars (BTC/ETH)
- **Three-algorithm ensemble** — PPO, A2C, and SAC with Sharpe-ratio weighted softmax blending
- **5-stage execution pipeline** — signal interpretation, position sizing (Kelly + ATR stops), order validation, broker submission, fill processing
- **Risk management** — dual circuit breakers (per-environment + global), position tracking, wash-sale detection
- **Feature engineering** — technical indicators (stockstats), FRED macro data, HMM regime detection, turbulence index, correlation pruning, rolling z-score normalization
- **Dual-database storage** — DuckDB for market data (OLAP), SQLite for trading operations (OLTP)
- **Shadow model lifecycle** — A/B test new models against active, auto-promote on Sharpe improvement with MDD tolerance
- **Discord alerting** — critical/warning/info alerts with rate limiting and healthchecks.io integration
- **Automated scheduling** — APScheduler with 12 jobs covering trading cycles, backups, macro updates, and monitoring
- **Docker-ready** — multi-stage Dockerfile (CI + production targets), non-root container, healthcheck endpoint
- **Disaster recovery** — automated backups (daily SQLite, weekly DuckDB, monthly offsite rsync) with restore tooling

## Architecture

```
Market APIs (Alpaca, Binance.US, FRED)
    │
    ▼
┌─────────────────────────────────────┐
│  Data Ingestion (BaseIngestor)      │
│  Alpaca │ Binance │ FRED            │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  Storage Layer                      │
│  DuckDB (market data, features)     │
│  SQLite (trades, positions, risk)   │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  Feature Pipeline                   │
│  Technical │ Macro │ HMM │ Turb.   │
│  Normalization │ Correlation Prune  │
│  Assembler (156-dim equity,         │
│             152-dim crypto)         │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  RL Training (Stable Baselines3)    │
│  PPO │ A2C │ SAC                    │
│  Gymnasium Envs + VecNormalize      │
│  Ensemble Blender (Sharpe-softmax)  │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  Execution Pipeline (5 stages)      │
│  Signal → Size → Validate →        │
│  Submit → Fill                      │
│                                     │
│  Risk: Circuit Breaker │ Position   │
│  Tracker │ Wash-Sale Detector       │
└────────────┬────────────────────────┘
             ▼
┌─────────────────────────────────────┐
│  Monitoring & Automation            │
│  APScheduler (12 jobs)              │
│  Discord Alerts │ Healthchecks.io   │
│  Backup & Disaster Recovery         │
└─────────────────────────────────────┘
```

## Prerequisites

- **Python 3.11** (3.11.x only — not 3.12+)
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **Docker** and **Docker Compose** (for containerized deployment)
- **API keys** (see [Configuration](#configuration)):
  - [Alpaca](https://alpaca.markets/) — equities (paper + live)
  - [Binance.US](https://www.binance.us/) — crypto
  - [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) — macro economic data
  - [Discord webhook](https://discord.com/developers/docs/resources/webhook) — alerts (optional)

## Installation

```bash
# Clone the repository
git clone https://github.com/Simplementix/SwingRL.git
cd SwingRL

# Install dependencies with uv
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run pytest tests/ -v
```

### Optional: Sentiment Analysis

```bash
uv sync --extra sentiment
```

This installs `transformers` for FinBERT-based news sentiment scoring.

## Configuration

### 1. Environment Variables

```bash
cp .env.example .env
chmod 600 .env
# Edit .env with your API keys
```

Required keys:

| Variable | Service | Purpose |
|----------|---------|---------|
| `ALPACA_API_KEY` | Alpaca | Equity trading (paper + live) |
| `ALPACA_SECRET_KEY` | Alpaca | Equity trading (paper + live) |
| `BINANCE_API_KEY` | Binance.US | Crypto trading |
| `BINANCE_SECRET_KEY` | Binance.US | Crypto trading |
| `FRED_API_KEY` | FRED | Macro economic data |
| `DISCORD_WEBHOOK_URL` | Discord | Alert notifications (optional) |

### 2. Application Config

The main config file is `config/swingrl.yaml`. Key sections:

```yaml
trading_mode: paper          # "paper" | "live"

equity:
  symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]
  max_position_size: 0.25    # max 25% of equity capital in one position
  max_drawdown_pct: 0.10     # circuit breaker at 10% drawdown
  daily_loss_limit_pct: 0.02 # circuit breaker at 2% daily loss

crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50
  max_drawdown_pct: 0.12
  daily_loss_limit_pct: 0.03
  min_order_usd: 10.0        # Binance.US minimum

capital:
  equity_usd: 400.0
  crypto_usd: 47.0
```

See `config/swingrl.prod.yaml.example` for production reference values with annotations.

### 3. Environment Variable Overrides

Any config field can be overridden via environment variables with the `SWINGRL_` prefix:

```bash
SWINGRL_TRADING_MODE=live                    # top-level field
SWINGRL_EQUITY__MAX_POSITION_SIZE=0.30       # nested field (double-underscore)
SWINGRL_LOGGING__JSON_LOGS=true              # enable JSON logging for Docker
```

### Config Loading in Code

```python
from swingrl.config.schema import load_config

config = load_config("config/swingrl.yaml")
print(config.trading_mode)       # "paper"
print(config.equity.symbols)     # ["SPY", "QQQ", ...]
print(config.capital.equity_usd) # 400.0
```

## Usage

### Training

Train RL agents (PPO, A2C, SAC) on historical data:

```bash
# Train all algorithms with default settings
uv run python scripts/train.py --config config/swingrl.yaml

# Compute features before training
uv run python scripts/compute_features.py --config config/swingrl.yaml
```

Trained models are saved to `models/active/{env}/{algo}/` with weights, normalization parameters, and metadata.

### Backtesting

Evaluate trained agents on historical data:

```bash
uv run python scripts/backtest.py --config config/swingrl.yaml
```

### Paper Trading

Run a single trading cycle to test the execution pipeline:

```bash
# Single equity cycle
uv run python scripts/run_cycle.py --config config/swingrl.yaml --env equity

# Single crypto cycle
uv run python scripts/run_cycle.py --config config/swingrl.yaml --env crypto
```

### Production Deployment

#### Docker Compose (recommended)

```bash
# Copy and configure environment
cp .env.example .env
chmod 600 .env
# Fill in API keys in .env

# Initialize databases
uv run python scripts/init_db.py --config config/swingrl.yaml
uv run python scripts/seed_production.py --config config/swingrl.yaml

# Start production daemon + dashboard
docker compose up -d

# Monitor logs
docker logs -f swingrl
```

The production container runs `scripts/main.py` as an APScheduler daemon with these scheduled jobs:

| Job | Schedule | Description |
|-----|----------|-------------|
| `equity_cycle` | 16:15 ET daily | Post-market equity trading cycle |
| `crypto_cycle` | Every 4 hours | Crypto trading cycle (UTC-aligned) |
| `reconciliation_job` | 18:00 ET daily | Position reconciliation vs. broker |
| `shadow_promotion_check` | 17:00 ET daily | Evaluate shadow model promotion |
| `stuck_agent_check` | Every 30 min | Detect stale agent signals |
| `daily_summary` | 20:00 ET daily | Discord summary (Sharpe, P&L) |
| `daily_backup` | 21:00 ET daily | SQLite backup (14-day retention) |
| `weekly_duckdb_backup` | Sunday 22:00 ET | DuckDB snapshot |
| `monthly_offsite` | 1st of month 23:00 ET | Offsite rsync |
| `monthly_macro` | 1st of month 09:00 ET | FRED economic data update |

#### Resource Requirements

| Container | RAM | CPU |
|-----------|-----|-----|
| swingrl | 2.5 GB | 1.0 |
| swingrl-dashboard | 512 MB | 0.5 |

### Operational Scripts

```bash
# Emergency: halt all trading
uv run python scripts/emergency_stop.py --config config/swingrl.yaml

# Resume after circuit breaker halt
uv run python scripts/reset_halt.py --config config/swingrl.yaml

# End-of-day reconciliation
uv run python scripts/reconcile.py --config config/swingrl.yaml

# Restore from backup
uv run python scripts/disaster_recovery.py --config config/swingrl.yaml

# Security audit
uv run python scripts/security_checklist.py --config config/swingrl.yaml
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=swingrl --cov-report=term-missing

# Run specific test module
uv run pytest tests/config/test_schema.py -v

# Run by pattern
uv run pytest -k "test_equity_env" -v

# Skip integration tests (require live API keys)
uv run pytest -m "not integration" -v
```

### Test Structure

| Directory | Coverage |
|-----------|----------|
| `tests/config/` | Config loading, schema validation, env overrides |
| `tests/data/` | Ingestors (Alpaca, Binance, FRED), DuckDB/SQLite storage |
| `tests/features/` | Technical indicators, HMM, macro, normalization |
| `tests/envs/` | Gymnasium environments, step/reset contracts |
| `tests/training/` | Agent training, ensemble blending, callbacks |
| `tests/execution/` | Order pipeline, position sizing, risk veto |
| `tests/execution/risk/` | Circuit breaker, position tracker, risk manager |
| `tests/scheduler/` | Job scheduling, halt checks |
| `tests/monitoring/` | Discord alerts, wash-sale detection |
| `tests/backup/` | Backup/restore operations |
| `tests/shadow/` | Shadow model lifecycle, promotion |

### Test Conventions

- **File naming:** `test_<module>.py`
- **Function naming:** `test_<behavior>()`
- **Docstrings:** `"""REQ-ID: What is being tested."""`
- **Coverage minimum:** 85% (enforced in CI)

## CI/CD

CI runs on a homelab server via `scripts/ci-homelab.sh`:

```bash
# Run CI locally
bash scripts/ci-homelab.sh

# Clean build (after lockfile changes)
bash scripts/ci-homelab.sh --no-cache

# Remote CI (from dev machine)
ssh homelab "cd ~/swingrl && git fetch origin && git checkout <branch> && git pull && bash scripts/ci-homelab.sh --no-cache"
```

### CI Pipeline Stages

| Stage | Tool | Check |
|-------|------|-------|
| 1 | git | Fast-forward pull |
| 2 | Docker | Build CI image |
| 3 | pytest | Run tests (85% coverage minimum) |
| 4 | ruff + mypy | Lint, format check, type check |
| 5 | Docker | Cleanup containers and images |

### Pre-commit Hooks

Installed automatically and run on every commit:

| Hook | Purpose |
|------|---------|
| ruff | Lint + auto-fix + format check |
| mypy | Type checking (`disallow_untyped_defs = true`) |
| bandit | Security scanning |
| detect-secrets | Prevent secret leaks |

## Project Structure

```
SwingRL/
├── config/
│   ├── swingrl.yaml              # Dev defaults (paper mode)
│   └── swingrl.prod.yaml.example # Production reference
├── src/swingrl/
│   ├── config/       # Pydantic v2 config schema + loader
│   ├── data/         # Market data ingestors + DuckDB/SQLite storage
│   ├── features/     # Technical, macro, HMM, normalization, assembler
│   ├── envs/         # Gymnasium RL environments (equity + crypto)
│   ├── training/     # SB3 trainer + ensemble blending
│   ├── execution/    # 5-stage pipeline + broker adapters
│   │   ├── adapters/ # Alpaca, Binance broker adapters
│   │   └── risk/     # Circuit breaker, position tracker, risk manager
│   ├── agents/       # Backtest metrics + validation
│   ├── scheduler/    # APScheduler job registration
│   ├── monitoring/   # Discord alerter, wash-sale checker
│   ├── backup/       # DuckDB/SQLite backup + offsite sync
│   ├── shadow/       # Shadow model lifecycle + promotion
│   ├── sentiment/    # FinBERT sentiment (optional)
│   └── utils/        # Exceptions, logging, retry
├── tests/            # 67 test modules, 848+ test cases
├── scripts/          # CLI entry points + operational tools
├── dashboard/        # Streamlit web dashboard
├── data/             # Raw OHLCV market data
├── db/               # DuckDB + SQLite databases
├── models/           # Trained agents (active/shadow/archive)
├── logs/             # Structured application logs
├── docs/             # 15 specification documents
└── .planning/        # Milestone planning + roadmap
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/main.py` | Production APScheduler daemon (12 jobs + stop-price poller) |
| `scripts/train.py` | Train PPO/A2C/SAC agents |
| `scripts/backtest.py` | Backtest agent performance |
| `scripts/compute_features.py` | Run feature engineering pipeline |
| `scripts/run_cycle.py` | Execute single trading cycle |
| `scripts/reconcile.py` | End-of-day position reconciliation |
| `scripts/init_db.py` | Initialize DuckDB + SQLite schemas |
| `scripts/seed_production.py` | Bootstrap production databases |
| `scripts/emergency_stop.py` | Halt all trading (circuit breaker) |
| `scripts/reset_halt.py` | Resume trading after halt |
| `scripts/disaster_recovery.py` | Restore from backups |
| `scripts/security_checklist.py` | API key rotation + env validation |
| `scripts/healthcheck.py` | Docker HEALTHCHECK endpoint |
| `scripts/ci-homelab.sh` | CI pipeline (test + lint + type check) |
| `scripts/deploy_model.sh` | Deploy model to active/shadow |
| `scripts/key_rotation_runbook.sh` | Interactive key rotation guide |

## License

MIT License. Copyright (c) 2026 Simplementix Inc. See [LICENSE](LICENSE) for details.
