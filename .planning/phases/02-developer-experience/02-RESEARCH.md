# Phase 2: Developer Experience - Research

**Researched:** 2026-03-06
**Domain:** CLAUDE.md conventions, Claude Code skills (.claude/commands/), Pydantic v2 config schema, pytest fixtures, models scaffold
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**CLAUDE.md Conventions**
- Comprehensive rulebook (~100-150 lines) covering all major patterns: imports, error handling, logging, config access, test structure, naming, docstrings
- Zero ambiguity for new sessions — no need to look up spec docs for coding patterns
- Custom exception hierarchy: SwingRLError base class with typed subclasses (BrokerError, DataError, ConfigError, etc.)
- structlog for structured JSON logging — pairs with HARD-05 requirement, works well with Docker log aggregation
- Include SwingRL domain rules: never hardcode ticker symbols, UTC internally with ET display conversion, broker interactions through execution middleware, config via Pydantic model (never raw YAML)

**Claude Skills (.claude/commands/)**
- Only create skills usable today — dev workflow skills only (test, lint, typecheck, Docker build)
- Data/training/ops skills added in the phases that need them (no dead stubs)
- Native execution by default, with --docker flag option for container-based execution
- Include a local CI mirror skill that runs the same 5 stages as ci-homelab.sh but natively on Mac (no SSH, no Docker build) for quick pre-push validation

**Config Schema and Defaults**
- Dev defaults in config/swingrl.yaml (paper mode, conservative limits, local paths), committed to repo
- Separate config/swingrl.prod.yaml.example with production values commented and annotated, committed as reference
- Environment variable overrides via Pydantic v2 SettingsConfigDict with SWINGRL_ prefix — perfect for Docker where .env sets trading mode
- Config file lives in config/swingrl.yaml (existing config/ directory)
- Explicit load pattern: call load_config(path) to get validated config — no side effects on import, tests can pass custom configs easily, fail-fast with clear ValidationError
- Schema itself follows Doc 05 §10.7 as-is (locked constraint from PROJECT.md)

**Smoke Tests and Fixtures**
- Keep existing 8 smoke tests, add config validation tests in new tests/test_config.py
- conftest.py provides: config fixtures (tmp_config with valid YAML in temp dir), sample data fixtures (OHLCV DataFrames for equity daily + crypto 4H), and temp directory fixtures (data/, db/, models/, logs/ with auto-cleanup)
- Function-scoped fixtures by default (fresh per test, prevents coupling). Session-scope only for expensive read-only setup like config loading
- Must-have test scenario: config validation roundtrip — valid YAML loads, invalid YAML raises ValidationError with clear message, env vars override YAML values

### Claude's Discretion
- Exact structlog configuration and formatter setup
- Exception hierarchy granularity (which subclasses beyond the obvious)
- Specific dev skill implementations (exact commands, flags)
- Sample data fixture values (realistic but synthetic OHLCV data)
- Fixture helper utilities in conftest.py

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-09 | CLAUDE.md project context file with SwingRL conventions (TDD, snake_case, type hints, Pydantic, pathlib, no hardcoded keys) | CLAUDE.md format is free-form Markdown; Claude Code reads it from repo root on startup; content locked in CONTEXT.md decisions |
| ENV-10 | .claude/commands/ with SwingRL-specific skills | Claude Code slash commands are Markdown files in .claude/commands/; format documented; only dev workflow skills today |
| ENV-11 | Pydantic v2 config schema (swingrl.yaml) implementing Doc 05 §10.7 with Field constraints, Literal enums, and model_validators | Pydantic v2 BaseSettings with SettingsConfigDict; yaml_settings_source pattern; SWINGRL_ env var prefix; ValidationError on invalid fields |
| ENV-12 | Scaffolded models/ directory with active/, shadow/, archive/ subdirectories and .gitkeep files | Already exists with .gitkeep files — verify committed, test coverage confirms directories present |
| ENV-13 | tests/test_smoke.py verifying all core package imports and tests/conftest.py with shared fixtures | Keep 8 existing smoke tests; expand conftest.py (currently 225 bytes); add tests/test_config.py for config validation roundtrip |
</phase_requirements>

---

## Summary

Phase 2 establishes the developer ergonomics layer that makes every future session consistent and productive. The work divides into four independent streams: (1) CLAUDE.md project conventions document, (2) .claude/commands/ skill files, (3) Pydantic v2 config schema for swingrl.yaml, and (4) expanded test fixtures with config validation tests.

The most technically interesting piece is the Pydantic v2 config schema (ENV-11). Pydantic v2 introduced a different BaseSettings API from v1 — `model_config = SettingsConfigDict(...)` replaces the inner `Config` class, and custom settings sources (for YAML file loading) must implement the `PydanticBaseSettingsSource` protocol. The schema itself is specified in Doc 05 §10.7 and is a locked constraint — implement it faithfully, not redesign it.

The .claude/commands/ skills are Markdown files with a specific frontmatter format that Claude Code parses to create slash commands. The format is straightforward but precise: `description:` in frontmatter becomes the help text, and the Markdown body is the prompt template. Skills must be immediately executable with the project's current tooling (uv, ruff, mypy, pytest, Docker) — no stubs for future phases.

ENV-12 is already complete — models/active/, models/shadow/, and models/archive/ all exist with .gitkeep files. The phase just needs tests confirming this and the pyproject.toml darwin constraint removed (deferred action from Phase 1 noted in STATE.md).

**Primary recommendation:** Implement config schema first (ENV-11) as it anchors conftest.py fixtures. Build CLAUDE.md and skills in parallel. Verify ENV-12 as a fast check-off task. Expand smoke tests and conftest last once config loading is working.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | >=2.5 (already in pyproject.toml) | Config schema validation, BaseSettings | Already installed; v2 API with SettingsConfigDict; ValidationError with field-level messages |
| pyyaml | >=6.0 (already in pyproject.toml) | YAML file loading for swingrl.yaml | Already installed; yaml.safe_load() is the standard |
| structlog | >=24.0 | Structured JSON logging | Locked by CONTEXT.md for HARD-05; pairs naturally with Docker log aggregation |
| pytest | >=8.0 (already in pyproject.toml) | Test runner + fixtures | Already installed; conftest.py fixtures via @pytest.fixture |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic-settings | >=2.2 | BaseSettings with env var support | Required for SWINGRL_ prefix env var overrides; separate package from pydantic in v2 |
| tmp_path (pytest builtin) | N/A | Temp directory fixture | Built into pytest 8.x; use for config file fixtures instead of manual tempfile |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pydantic-settings BaseSettings | OmegaConf / Hydra | Pydantic already in stack; CONTEXT.md locks this choice |
| structlog | Python stdlib logging | structlog provides JSON formatting + context binding natively; stdlib requires custom formatters |
| pyyaml safe_load | ruamel.yaml | pyyaml already installed; ruamel only needed for round-trip YAML editing (not needed here) |

**Installation (add to pyproject.toml):**
```bash
uv add structlog pydantic-settings
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)

```
SwingRL/
├── CLAUDE.md                       # NEW: ~100-150 lines, SwingRL conventions
├── .claude/
│   └── commands/                   # NEW: slash command skill files
│       ├── test.md                 # /project:test — run pytest natively
│       ├── lint.md                 # /project:lint — ruff check + ruff format --check + mypy
│       ├── typecheck.md            # /project:typecheck — mypy src/ only
│       ├── docker-build.md         # /project:docker-build — docker compose build
│       └── ci-local.md             # /project:ci-local — 5-stage native CI mirror
├── config/
│   ├── swingrl.yaml                # NEW: dev defaults (paper mode, conservative limits)
│   └── swingrl.prod.yaml.example   # NEW: production values annotated as reference
├── src/swingrl/
│   └── config/
│       ├── __init__.py             # (exists, empty)
│       └── schema.py               # NEW: Pydantic v2 config schema + load_config()
├── tests/
│   ├── conftest.py                 # EXPAND: add config/data/tmp fixtures
│   ├── test_smoke.py               # KEEP: existing 8 tests
│   └── test_config.py              # NEW: config validation roundtrip tests
└── models/
    ├── active/.gitkeep             # EXISTS: verify committed
    ├── shadow/.gitkeep             # EXISTS: verify committed
    └── archive/.gitkeep            # EXISTS: verify committed
```

### Pattern 1: Pydantic v2 BaseSettings with YAML + Env Var Override

**What:** Config class that loads from YAML file, then allows SWINGRL_* environment variables to override individual fields
**When to use:** The canonical config loading pattern for SwingRL — call load_config(path) at startup

```python
# Source: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
# src/swingrl/config/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Loads settings from a YAML file path provided at instantiation."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_path: Path) -> None:
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field: Any, field_name: str) -> Any:
        # Not used directly — overriding __call__ instead
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self._yaml_path.exists():
            return {}
        with self._yaml_path.open() as f:
            return yaml.safe_load(f) or {}


class SwingRLConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWINGRL_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    trading_mode: Literal["paper", "live"] = Field(default="paper")
    # ... (fields from Doc 05 §10.7)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        secrets_dir: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: env vars > yaml file > defaults
        # YamlConfigSettingsSource injected via load_config()
        return (env_settings,)


def load_config(path: Path | str = "config/swingrl.yaml") -> SwingRLConfig:
    """Load and validate SwingRL config from YAML file with env var overrides.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    yaml_path = Path(path)

    class _ConfigWithYaml(SwingRLConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            secrets_dir: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (env_settings, YamlConfigSettingsSource(settings_cls, yaml_path))

    return _ConfigWithYaml()
```

**Key design decisions:**
- `load_config(path)` is the single entry point — no side effects on module import
- ValidationError is raised immediately with field-level detail (pydantic v2 default)
- Env vars always win over YAML values (Docker `.env` overrides local yaml)
- SWINGRL__ double-underscore syntax for nested fields (e.g., `SWINGRL_EQUITY__MAX_POSITION_SIZE`)

### Pattern 2: CLAUDE.md Structure for Claude Code Projects

**What:** Markdown file read by Claude Code at session start to establish project context and conventions
**When to use:** Repo root CLAUDE.md for all Claude Code projects

```markdown
# SwingRL — Claude Code Conventions

## Project Overview
[2-3 sentence summary of what SwingRL is]

## Critical Rules (Never Violate)
- Never hardcode ticker symbols, API keys, file paths, or dollar amounts
- Config access only through load_config() — never yaml.safe_load() directly in business logic
- UTC timestamps internally; convert to ET only for display/alerts
- All broker interactions through execution middleware (src/swingrl/execution/)
- Tests must pass before any commit (pre-commit enforces this)

## Code Style
- Python 3.11, snake_case for all identifiers
- Type hints on all function signatures (disallow_untyped_defs = true in mypy)
- pathlib.Path for all file operations — never os.path or raw strings
- Line length: 100 characters (ruff + black config)
- Docstrings: one-line summary required; full docstring for public API

## Imports
[standard ordering pattern]

## Error Handling
[exception hierarchy, when to raise vs log]

## Logging
[structlog pattern, required context fields]

## Testing
[TDD expectation, fixture patterns, requirement ID in docstring]

## Config Access
[load_config() pattern, never raw YAML]

## Key Paths
[repo-relative paths for data, db, models, config]
```

### Pattern 3: Claude Code Skill (.claude/commands/) Format

**What:** Markdown files in .claude/commands/ that become available as /project:[name] slash commands
**When to use:** Repeatable dev tasks that benefit from consistent flags and context

```markdown
---
description: Run pytest suite natively on Mac (fast, no Docker)
---

Run the SwingRL test suite natively using uv:

```bash
uv run pytest tests/ -v $ARGUMENTS
```

Options:
- No args: full suite
- `tests/test_smoke.py`: smoke tests only
- `-k test_name`: single test
- `-x`: stop on first failure
```

**Key insight about .claude/commands/ format:**
- YAML frontmatter with `description:` is the help text shown in Claude Code
- The body is the prompt template Claude uses when invoked
- `$ARGUMENTS` captures anything typed after the slash command
- Files are `.md` extension, no special registration needed — Claude Code auto-discovers them

### Pattern 4: conftest.py Fixture Expansion

**What:** Shared pytest fixtures providing config, OHLCV sample data, and temp directories
**When to use:** All test files that need configured objects without real filesystem dependencies

```python
# tests/conftest.py
from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def valid_config_yaml() -> str:
    """Minimal valid swingrl.yaml content for testing."""
    return textwrap.dedent("""
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ]
          max_position_size: 0.25
        crypto:
          symbols: [BTCUSDT, ETHUSDT]
          max_position_size: 0.50
    """)


@pytest.fixture
def tmp_config(tmp_path: Path, valid_config_yaml: str) -> Path:
    """Write valid YAML to a temp file; return the Path."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(valid_config_yaml)
    return config_file


@pytest.fixture
def loaded_config(tmp_config: Path) -> SwingRLConfig:
    """Return a validated SwingRLConfig loaded from tmp_config."""
    return load_config(tmp_config)


@pytest.fixture
def equity_ohlcv() -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame for equity tests (20 rows, SPY-like values)."""
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = 470.0 + rng.normal(0, 2, 20).cumsum()
    return pd.DataFrame({
        "open": close - rng.uniform(0, 1, 20),
        "high": close + rng.uniform(0, 2, 20),
        "low": close - rng.uniform(0, 2, 20),
        "close": close,
        "volume": rng.integers(50_000_000, 100_000_000, 20).astype(float),
    }, index=dates)


@pytest.fixture
def crypto_ohlcv() -> pd.DataFrame:
    """Synthetic 4H OHLCV DataFrame for crypto tests (40 rows, BTC-like values)."""
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    rng = np.random.default_rng(43)
    close = 42_000.0 + rng.normal(0, 200, 40).cumsum()
    return pd.DataFrame({
        "open": close - rng.uniform(0, 50, 40),
        "high": close + rng.uniform(0, 150, 40),
        "low": close - rng.uniform(0, 150, 40),
        "close": close,
        "volume": rng.uniform(500, 2000, 40),
    }, index=dates)


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create a standard SwingRL directory tree in tmp_path. Auto-cleaned by pytest."""
    dirs = {}
    for name in ["data", "db", "models/active", "models/shadow", "models/archive", "logs"]:
        d = tmp_path / name
        d.mkdir(parents=True)
        dirs[name] = d
    return dirs
```

### Pattern 5: Config Validation Test Structure

**What:** Tests confirming that load_config() enforces schema, rejects invalid YAML, and honors env var overrides
**When to use:** tests/test_config.py — the canonical pattern for all future config validation work

```python
# tests/test_config.py
import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from swingrl.config.schema import load_config


def test_valid_config_loads(tmp_config: Path) -> None:
    """ENV-11: Valid YAML must load without error."""
    config = load_config(tmp_config)
    assert config.trading_mode == "paper"


def test_invalid_trading_mode_raises(tmp_path: Path) -> None:
    """ENV-11: Invalid Literal value raises ValidationError."""
    bad_yaml = tmp_path / "swingrl.yaml"
    bad_yaml.write_text("trading_mode: live_aggressive\n")
    with pytest.raises(ValidationError) as exc_info:
        load_config(bad_yaml)
    assert "trading_mode" in str(exc_info.value)


def test_env_var_overrides_yaml(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-11: SWINGRL_ env vars override YAML values."""
    monkeypatch.setenv("SWINGRL_TRADING_MODE", "live")
    config = load_config(tmp_config)
    assert config.trading_mode == "live"
```

### Pattern 6: Custom Exception Hierarchy

**What:** SwingRLError base class with domain-typed subclasses for structured error handling
**When to use:** All SwingRL modules raise typed subclasses; callers catch specific types

```python
# src/swingrl/utils/exceptions.py
class SwingRLError(Exception):
    """Base class for all SwingRL exceptions."""


class ConfigError(SwingRLError):
    """Config file missing, unreadable, or fails validation."""


class BrokerError(SwingRLError):
    """Broker API errors: auth failure, order rejection, rate limit."""


class DataError(SwingRLError):
    """Data pipeline errors: missing bars, validation failure, quarantine."""


class ModelError(SwingRLError):
    """Model load/inference errors: file missing, shape mismatch."""


class CircuitBreakerError(SwingRLError):
    """Circuit breaker is active; trading halted."""


class RiskVetoError(SwingRLError):
    """Risk management layer vetoed the trade."""
```

### Pattern 7: structlog Configuration (JSON for Docker, Colored for Dev)

**What:** structlog setup that outputs JSON in production (Docker) and colored console output locally
**When to use:** src/swingrl/utils/logging.py — imported once at startup

```python
# src/swingrl/utils/logging.py
# Source: https://www.structlog.org/en/stable/configuration.html
import logging
import sys

import structlog


def configure_logging(json_logs: bool = False, log_level: str = "INFO") -> None:
    """Configure structlog for SwingRL.

    Args:
        json_logs: True for JSON output (production/Docker), False for colored console (dev).
        log_level: Python logging level string.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
```

**Usage pattern:**
```python
import structlog
log = structlog.get_logger(__name__)
log.info("order_submitted", symbol="SPY", quantity=10, side="buy")
# JSON output: {"event": "order_submitted", "symbol": "SPY", "quantity": 10, ...}
```

### Anti-Patterns to Avoid

- **Importing config at module level with side effects:** `from swingrl.config import CONFIG` at module top triggers file I/O on import, breaks testing. Use `load_config()` explicitly.
- **Using Python's `logging.basicConfig` alongside structlog:** structlog replaces stdlib logging formatting; calling basicConfig before configure_logging() creates duplicate/garbled output.
- **Pydantic v1 style `class Config:` inside models:** In Pydantic v2, use `model_config = SettingsConfigDict(...)` at class level. The inner `Config` class is deprecated and silently ignored in v2.
- **Creating .claude/commands/ files without testing them:** Slash commands must work with the actual project tooling. Test each command manually before committing.
- **Fixtures with session scope that mutate state:** session-scoped fixtures are shared across the entire test run. Any fixture that writes files or modifies env vars must be function-scoped.
- **YAML with unsafe_load:** Always use `yaml.safe_load()` — unsafe_load can execute arbitrary Python objects embedded in YAML.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config validation | Custom dict validator with isinstance() checks | Pydantic v2 BaseSettings | Field-level error messages, Literal enums, nested model validation, free env var override support |
| YAML-to-env merge | Manual os.environ lookups + yaml merge | pydantic-settings settings sources | Priority chain (env > yaml > defaults) handled automatically |
| Structured logging | Custom JSON formatter on stdlib logging | structlog | Context binding, contextvars propagation, async-safe, Docker-native JSON output |
| Temp test directories | Manual tempfile.mkdtemp() + shutil.rmtree() | pytest's tmp_path fixture | Auto-cleanup on test teardown, no teardown yield needed, thread-safe |
| Config file discovery | Walk filesystem looking for swingrl.yaml | Explicit path parameter in load_config() | Explicit > implicit; tests pass any path; no hidden global state |
| Exception stringification | str(e) parsing | Pydantic v2 ValidationError.errors() | Structured list of field errors with loc, msg, type — directly parseable |

**Key insight:** The entire Phase 2 stack (Pydantic v2 + structlog + pytest fixtures) is designed to eliminate boilerplate. Any custom solution for config validation, logging, or test setup adds maintenance cost with no benefit.

---

## Common Pitfalls

### Pitfall 1: pydantic-settings Is a Separate Package in v2

**What goes wrong:** `from pydantic import BaseSettings` raises ImportError — works in Pydantic v1 but fails in v2.
**Why it happens:** Pydantic v2 extracted BaseSettings to its own package: `pydantic-settings`.
**How to avoid:** Add `pydantic-settings>=2.2` to pyproject.toml dependencies. Import from `pydantic_settings`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
```
**Warning signs:** `ImportError: cannot import name 'BaseSettings' from 'pydantic'`

### Pitfall 2: SWINGRL__ Double-Underscore for Nested Fields

**What goes wrong:** Env var `SWINGRL_EQUITY_MAX_POSITION_SIZE` doesn't override the nested field.
**Why it happens:** pydantic-settings uses `env_nested_delimiter` to map env vars to nested models. With `env_nested_delimiter="__"`, the env var must use double-underscore: `SWINGRL_EQUITY__MAX_POSITION_SIZE`.
**How to avoid:** Document this in CLAUDE.md and in the .env.example file. Single underscore matches only top-level fields.
**Warning signs:** Env var set in Docker .env but config still shows YAML value.

### Pitfall 3: Pydantic v2 model_validators Replace v1 @validator

**What goes wrong:** `@validator('field')` from Pydantic v1 raises AttributeError or doesn't fire in v2.
**Why it happens:** Pydantic v2 replaced `@validator` with `@field_validator` (per-field) and `@model_validator` (cross-field). Old decorators are not imported.
**How to avoid:** Use `@field_validator('field_name', mode='before')` for field-level validation, `@model_validator(mode='after')` for cross-field validation.
**Warning signs:** ValidationError not raised when expected; import error for `@validator`.

### Pitfall 4: structlog Requires Processor Configuration Before First Use

**What goes wrong:** structlog output is unformatted or fails silently if `configure_logging()` is called after the first log call.
**Why it happens:** structlog caches the processor chain on first use (`cache_logger_on_first_use=True`). First use before configuration gets the default (bare string) renderer.
**How to avoid:** Call `configure_logging()` at the application entry point before any `structlog.get_logger()` calls. In tests, call it in conftest.py session setup or use `structlog.reset_defaults()` between tests if testing logging output.
**Warning signs:** Log output is a plain dict repr instead of JSON or colored console.

### Pitfall 5: .claude/commands/ Slash Commands Not Auto-Discovered

**What goes wrong:** Slash command doesn't appear in Claude Code after creating the .md file.
**Why it happens:** The .claude/ directory must be at the repo root (same level as CLAUDE.md). If .claude/ is nested, Claude Code won't find it.
**How to avoid:** Create `.claude/commands/` at the exact repo root level. Restart Claude Code session after adding new commands.
**Warning signs:** `/project:test` shows "unknown command" in Claude Code.

### Pitfall 6: conftest.py Function-Scope vs Session-Scope Fixture Coupling

**What goes wrong:** Test isolation breaks — one test's side effects leak into another.
**Why it happens:** Session-scoped fixtures are shared across the entire test run. If a fixture returns a mutable object (like a config instance or DataFrame), mutations in one test persist.
**How to avoid:** Function scope by default. Session scope only for: (1) read-only objects, (2) expensive setup that must not be repeated (like loading a large model). The `repo_root` fixture is legitimately session-scoped; `loaded_config` should be function-scoped.
**Warning signs:** Test passes when run alone but fails when run in suite; test ordering matters.

### Pitfall 7: pyproject.toml darwin Constraint Must Be Removed Before Phase 2

**What goes wrong:** Docker/Linux lockfile generation fails because pyproject.toml has `tool.uv.environments` constraining to darwin.
**Why it happens:** Phase 1 deferred Linux lockfile generation (noted in STATE.md: "Remove constraint before Phase 2 Plan 1").
**How to avoid:** First task in Phase 2 Wave 0 — remove `tool.uv.environments` darwin constraint from pyproject.toml and run `uv lock` to regenerate cross-platform lockfile.
**Warning signs:** `uv sync --locked` in Docker fails with platform constraint error.

---

## Code Examples

Verified patterns from official sources:

### swingrl.yaml Dev Defaults Structure

```yaml
# config/swingrl.yaml — Development defaults (paper mode, conservative limits)
# Override any field with SWINGRL_ env var (e.g., SWINGRL_TRADING_MODE=live)
# Nested fields use double-underscore: SWINGRL_EQUITY__MAX_POSITION_SIZE=0.3

trading_mode: paper  # "paper" | "live"

equity:
  symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]
  max_position_size: 0.25       # max fraction of equity capital in one position
  max_drawdown_pct: 0.10        # circuit breaker: 10% DD halts equity env
  daily_loss_limit_pct: 0.02    # circuit breaker: 2% daily loss

crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50       # BTC can be up to 50% of crypto capital
  max_drawdown_pct: 0.12        # circuit breaker: 12% DD halts crypto env
  daily_loss_limit_pct: 0.03    # circuit breaker: 3% daily loss
  min_order_usd: 10.0           # Binance.US $10 floor

capital:
  equity_usd: 400.0
  crypto_usd: 47.0

paths:
  data_dir: data/
  db_dir: db/
  models_dir: models/
  logs_dir: logs/

logging:
  level: INFO
  json_logs: false  # true in production/Docker
```

### Pydantic v2 Field Constraints Example

```python
# Source: https://docs.pydantic.dev/latest/concepts/fields/
from pydantic import Field
from typing import Literal

class EquityConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    max_position_size: float = Field(default=0.25, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.10, gt=0.0, lt=1.0)
    daily_loss_limit_pct: float = Field(default=0.02, gt=0.0, lt=1.0)

class SwingRLConfig(BaseSettings):
    trading_mode: Literal["paper", "live"] = "paper"
    equity: EquityConfig = Field(default_factory=EquityConfig)
    # ...
```

### Claude Code Skill File Format

```markdown
---
description: Run pytest suite natively (no Docker). Use: /project:test [path] [-k filter] [-x]
---

Run SwingRL tests natively using uv. Faster than Docker for development iteration.

```bash
uv run pytest $ARGUMENTS -v
```

Common patterns:
- Full suite: `/project:test`
- Smoke only: `/project:test tests/test_smoke.py`
- Config tests: `/project:test tests/test_config.py`
- Single test: `/project:test -k test_valid_config_loads`
- Stop on first fail: `/project:test -x`
```

### ci-local Skill (5-Stage Native Mirror)

```markdown
---
description: Run full CI quality gate natively on Mac (mirrors ci-homelab.sh stages 2-5, no Docker build)
---

Run the same quality checks as ci-homelab.sh but natively for fast pre-push validation.
Stages: [1] tests, [2] ruff lint, [3] ruff format check, [4] mypy types, [5] summary.

```bash
set -e
echo "=== [1/4] Tests ==="
uv run pytest tests/ -v

echo "=== [2/4] Lint ==="
uv run ruff check .

echo "=== [3/4] Format check ==="
uv run ruff format --check .

echo "=== [4/4] Type check ==="
uv run mypy src/

echo "=== CI LOCAL PASSED ==="
```

Note: Stage [2/5] Docker build is skipped — run actual ci-homelab.sh via ssh for full validation before merging.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pydantic v1 `class Config:` inner class | Pydantic v2 `model_config = SettingsConfigDict(...)` | Pydantic 2.0 (2023) | Old syntax silently ignored in v2 |
| `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` | Pydantic 2.0 (2023) | ImportError in v2 without separate package |
| Pydantic v1 `@validator` | Pydantic v2 `@field_validator` / `@model_validator` | Pydantic 2.0 (2023) | Different import, different decorator signature |
| stdlib `logging.basicConfig(format=...)` | structlog with JSON/console renderer | 2021-present | Structured context binding, Docker-native JSON |
| Raw YAML dict access | Pydantic model field access | Pydantic v1+ | Type safety, validation at load time, IDE autocomplete |

**Deprecated/outdated:**
- `pydantic.BaseSettings`: Moved to `pydantic-settings` package in v2. Old import raises ImportError.
- `@validator` decorator: Replaced by `@field_validator` in Pydantic v2. Use `mode='before'` or `mode='after'` parameter.
- `conftest.py` with `yield`-only teardown: pytest's `tmp_path` fixture makes manual teardown unnecessary for temp file cleanup.

---

## Open Questions

1. **Doc 05 §10.7 Exact Schema Fields**
   - What we know: The schema spec lives in a NotebookLM document that is not in the repo. The CONTEXT.md states "implement Doc 05 §10.7 as-is." The config/swingrl.yaml structure can be inferred from PROJECT.md, REQUIREMENTS.md, and memory files (trading_mode, equity/crypto capital, broker settings, circuit breaker thresholds, etc.)
   - What's unclear: The exact field names, nesting structure, and which fields have `model_validators` vs simple `Field` constraints per Doc 05 §10.7
   - Recommendation: The planner should create a Wave 0 task to review Doc 05 §10.7 (in NotebookLM) and transcribe the schema fields into a task-level specification before implementing `schema.py`. Use the swingrl.yaml structure shown in REQUIREMENTS.md and memory files as a starting point — it covers all known fields.

2. **structlog Version Compatibility with Python 3.11**
   - What we know: structlog >=24.0 is available on PyPI; Python 3.11 is fully supported
   - What's unclear: Whether there are any conflicts with the existing dependency set (FinRL pulls various dependencies)
   - Recommendation: Run `uv add structlog pydantic-settings` and verify `uv sync` resolves cleanly. This is a Wave 0 task.

3. **pyproject.toml darwin Constraint Removal**
   - What we know: STATE.md explicitly flags this: "Remove `tool.uv.environments` darwin constraint from pyproject.toml before Phase 2 Plan 1"
   - What's unclear: Confirmed action needed — not an open question
   - Recommendation: First task in Phase 2 Wave 0, before any code changes.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `uv run pytest tests/test_smoke.py tests/test_config.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-09 | CLAUDE.md exists at repo root with conventions | smoke | `pytest tests/test_smoke.py::test_claude_md_exists -v` | ❌ Wave 0 (add to test_smoke.py) |
| ENV-10 | .claude/commands/ exists with at least one skill file | smoke | `pytest tests/test_smoke.py::test_claude_commands_exist -v` | ❌ Wave 0 (add to test_smoke.py) |
| ENV-11 | load_config() validates YAML, raises ValidationError on bad input, honors SWINGRL_ env vars | unit | `pytest tests/test_config.py -v` | ❌ Wave 0 (create test_config.py) |
| ENV-12 | models/active/, shadow/, archive/ exist with .gitkeep | smoke | `pytest tests/test_smoke.py::test_models_directories_exist -v` | ❌ Wave 0 (add to test_smoke.py) |
| ENV-13 | All core package imports succeed; conftest fixtures load | smoke + unit | `pytest tests/test_smoke.py tests/test_config.py -v` | Partial (test_smoke.py exists, conftest.py minimal) |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_smoke.py -v`
- **Per wave merge:** `uv run pytest tests/ -v && uv run ruff check . && uv run mypy src/`
- **Phase gate:** Full suite green + `ssh homelab "bash ~/swingrl/scripts/ci-homelab.sh"` before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_config.py` — covers ENV-11 (config validation roundtrip, ValidationError, env var override)
- [ ] `tests/conftest.py` expansion — add `tmp_config`, `loaded_config`, `equity_ohlcv`, `crypto_ohlcv`, `tmp_dirs` fixtures
- [ ] `src/swingrl/config/schema.py` — Pydantic v2 config schema + `load_config()` (required before test_config.py can run)
- [ ] `src/swingrl/utils/exceptions.py` — SwingRLError hierarchy (referenced in CLAUDE.md conventions)
- [ ] `src/swingrl/utils/logging.py` — structlog `configure_logging()` (referenced in CLAUDE.md conventions)
- [ ] Add ENV-09, ENV-10, ENV-12 assertion tests to `tests/test_smoke.py`
- [ ] Dependency additions: `uv add structlog pydantic-settings`
- [ ] Remove `tool.uv.environments` darwin constraint from `pyproject.toml` + `uv lock` regeneration

---

## Sources

### Primary (HIGH confidence)

- [Pydantic v2 Settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — BaseSettings, SettingsConfigDict, env_prefix, custom settings sources, PydanticBaseSettingsSource protocol
- [Pydantic v2 Fields docs](https://docs.pydantic.dev/latest/concepts/fields/) — Field constraints, gt/le/lt, Literal type, model_validators
- [Pydantic v2 Migration guide](https://docs.pydantic.dev/latest/migration/) — v1 → v2 changes: BaseSettings moved, @validator replaced
- [structlog configuration docs](https://www.structlog.org/en/stable/configuration.html) — configure(), ProcessorFormatter, JSONRenderer, ConsoleRenderer
- [pytest fixtures docs](https://docs.pytest.org/en/stable/reference/fixtures.html) — tmp_path, scope, conftest.py discovery
- Direct inspection of `pyproject.toml`, `tests/conftest.py`, `tests/test_smoke.py`, `scripts/ci-homelab.sh` in this repo — HIGH confidence, authoritative

### Secondary (MEDIUM confidence)

- [Claude Code documentation on CLAUDE.md](https://docs.anthropic.com/en/docs/claude-code) — project context file format, .claude/commands/ discovery
- [pydantic-settings PyPI](https://pypi.org/project/pydantic-settings/) — version availability for Python 3.11

### Tertiary (LOW confidence)

- Inferred Doc 05 §10.7 schema fields from REQUIREMENTS.md and memory files — not directly verified; actual schema document is in NotebookLM outside this repo

---

## Metadata

**Confidence breakdown:**
- Standard stack (Pydantic v2, structlog, pytest fixtures): HIGH — all verified against official docs and existing pyproject.toml
- Architecture (CLAUDE.md format, .claude/commands/ format, config loading pattern): HIGH — established patterns, verified against Claude Code docs
- Pitfalls (pydantic-settings separate package, env_nested_delimiter, v2 migration): HIGH — verified against Pydantic v2 migration guide
- Doc 05 §10.7 exact schema fields: LOW — document not in repo; inferred from other sources

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (Pydantic v2 API is stable; structlog API is stable; only Claude Code .claude/commands/ format could change)
