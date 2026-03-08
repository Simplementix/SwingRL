# SwingRL — Claude Code Conventions

## Project Overview

SwingRL is an RL-based swing trading system using PPO/A2C/SAC ensemble agents (Stable Baselines3)
on two environments: equity daily (8 ETFs via Alpaca) and crypto 4H (BTC/ETH via Binance.US).
Capital preservation is the primary constraint — never lose more than you can recover from.
Phases: 1 Foundation → 2 DX → 3 Data → 4 Storage → 5 Features → 6 Envs → 7 Training →
8 Paper Trading → 9 Automation → 10 Hardening.

## Git Workflow — Per Phase (Closeout Checklist)

1. **Start of phase execution** (Claude): Create feature branch `gsd/phase-{N}-{slug}` from `main`.
2. **All phase commits** go to the feature branch.
3. **After verification passes** (Claude): Fix ROADMAP.md checkbox (`- [x]`) and push the feature branch to origin.
4. **Run homelab CI** (Claude): SSH into homelab and run CI. Must pass before PR creation.
   ```bash
   ssh homelab "cd ~/swingrl && git fetch origin && git checkout {branch} && git pull origin {branch} && bash scripts/ci-homelab.sh --no-cache"
   ```
5. **CI passes → Create PR** (Claude): Create PR to `main` with phase summary. Only after CI passes.
6. **User merges** (branch protection requires approval).
7. **After merge** (User): `git checkout main && git pull origin main` before starting next phase.

## Critical Rules — Never Violate

- **No hardcoded values**: Never hardcode ticker symbols, API keys, file paths, or dollar amounts.
  Use `SwingRLConfig` fields or named constants in `src/swingrl/config/schema.py`.
- **Config via load_config()**: Call `load_config(path)` to obtain a validated `SwingRLConfig`.
  Never call `yaml.safe_load()` directly in business logic. No side-effect imports.
- **UTC internally**: All timestamps stored and computed in UTC.
  Convert to ET only for display output and Discord alerts.
- **Broker middleware only**: All order submission goes through `src/swingrl/execution/`.
  Never call broker APIs (Alpaca, Binance.US) directly from envs, agents, or training code.
- **Tests first**: Write the failing test before writing production code (TDD for all logic with
  defined I/O). Commit RED test, then GREEN implementation.
- **Never skip pre-commit**: Fix the hook failure; do not pass `--no-verify`.

## Python Style

- Python 3.11 only. snake_case for all identifiers (modules, functions, variables, parameters).
- Type hints on **all** function signatures — `disallow_untyped_defs = true` in mypy config.
- `pathlib.Path` for all file operations. Never `os.path` or raw string paths.
- Line length: 100 characters (`ruff` + `black` both configured to 100).
- Docstrings: one-line summary required for all public functions; full docstring with Args/Raises
  for public API methods.
- `from __future__ import annotations` at top of every module (deferred evaluation).

## Imports (isort order, enforced by ruff)

```python
from __future__ import annotations  # always first

# stdlib
import logging
from pathlib import Path
from typing import Literal

# third-party
import pandas as pd
import structlog
from pydantic import Field

# first-party (absolute, never relative for src/ modules)
from swingrl.config.schema import load_config
from swingrl.utils.exceptions import BrokerError
```

Never use relative imports (`from .schema import`) in `src/swingrl/`.

## Error Handling

Raise typed subclasses of `SwingRLError` (`src/swingrl/utils/exceptions.py`):

| Exception | When to Raise |
|-----------|---------------|
| `ConfigError` | Config file missing, unreadable, or fails Pydantic validation |
| `BrokerError` | Broker API auth failure, order rejection, rate limit |
| `DataError` | Missing bars, validation failure, quarantine trigger |
| `ModelError` | Model file missing, shape mismatch, load failure |
| `CircuitBreakerError` | Circuit breaker active; trading halted |
| `RiskVetoError` | Risk layer vetoed trade; order not submitted |

Never raise bare `Exception` or `ValueError` for application-level errors.
Log the error with structlog **before** raising. Callers catch specific subclasses.

## Logging (structlog)

Call `configure_logging()` once at application entry point (before any log calls):

```python
from swingrl.utils.logging import configure_logging
configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
```

Get a logger per module:
```python
import structlog
log = structlog.get_logger(__name__)
```

Always pass context as keyword args — never f-strings:
```python
log.info("order_submitted", symbol="SPY", quantity=10, side="buy", env="equity")
log.error("broker_error", error=str(e), order_id=order_id)
```

`json_logs=False` (dev): colored console. `json_logs=True` (Docker): JSON to stdout.

## Config Access

```python
from swingrl.config.schema import load_config, SwingRLConfig

config: SwingRLConfig = load_config("config/swingrl.yaml")
# Access fields:
mode = config.trading_mode          # "paper" | "live"
symbols = config.equity.symbols     # list[str]
cap = config.capital.equity_usd     # float
```

Environment variable overrides (Docker .env):
- Top-level: `SWINGRL_TRADING_MODE=live`
- Nested: `SWINGRL_EQUITY__MAX_POSITION_SIZE=0.30` (double-underscore for nested)

## Testing Conventions

- Test file: `tests/test_<module>.py`. Test function: `test_<behavior>`.
- Docstring: `"""REQ-ID: What is being tested."""`
- Use fixtures from `tests/conftest.py` — never construct real file paths or config inline.
- Key fixtures: `tmp_config` (valid YAML in temp dir), `loaded_config` (SwingRLConfig),
  `equity_ohlcv` (20-row DataFrame), `crypto_ohlcv` (40-row DataFrame), `tmp_dirs` (dir tree).
- Function scope by default. Session scope only for read-only, expensive setup.
- Run before commit: `uv run pytest tests/ -v`

## Key Paths (from repo root)

| Purpose | Path |
|---------|------|
| Config YAML | `config/swingrl.yaml` |
| Production config reference | `config/swingrl.prod.yaml.example` |
| Source packages | `src/swingrl/` |
| Tests | `tests/` |
| Data (raw bars) | `data/` |
| Databases | `db/` |
| Models (active/shadow/archive) | `models/` |
| Logs | `logs/` |
| CI script | `scripts/ci-homelab.sh` |

## Phase 2 Artifacts Added

- `src/swingrl/utils/exceptions.py` — SwingRLError hierarchy (7 classes)
- `src/swingrl/utils/logging.py` — configure_logging() with JSON/console switching
- `src/swingrl/config/schema.py` — Pydantic v2 config schema + load_config()
- `config/swingrl.yaml` — Dev defaults (paper mode, conservative limits)
- `config/swingrl.prod.yaml.example` — Production reference values (annotated)
- `.claude/commands/` — Dev workflow slash commands
