---
phase: 02-developer-experience
plan: 03
subsystem: config
tags: [pydantic, pydantic-settings, yaml, config, schema, env-vars]

# Dependency graph
requires:
  - phase: 02-developer-experience-01
    provides: pydantic-settings>=2.2 installed, pyproject.toml configured, pre-commit hooks established

provides:
  - SwingRLConfig Pydantic v2 BaseSettings with SWINGRL_ env prefix and __ nested delimiter
  - load_config(path) entry point — every future phase calls this to get a typed, validated config object
  - EquityConfig, CryptoConfig, CapitalConfig, PathsConfig, LoggingConfig sub-config models
  - Cross-field model_validators enforcing daily_loss_limit_pct < max_drawdown_pct on both envs
  - config/swingrl.yaml — dev defaults (paper mode, 8 ETFs, $10 Binance.US floor, 400/47 USD)
  - config/swingrl.prod.yaml.example — annotated production reference with Docker absolute paths

affects:
  - All future phases (3-10) that call load_config() for typed configuration
  - Phase 5 (features), 6 (envs), 7 (training) — read equity.symbols and crypto.symbols
  - Phase 8 (paper trading) — reads trading_mode, capital.equity_usd, capital.crypto_usd
  - Phase 9 (automation) — reads logging.json_logs for Docker JSON log aggregation

# Tech tracking
tech-stack:
  added:
    - types-PyYAML (dev dep — mypy stubs for pyyaml)
  patterns:
    - YamlConfigSettingsSource pattern — custom PydanticBaseSettingsSource for YAML loading
    - load_config() factory pattern using inner _ConfigWithYaml subclass to bind yaml_path
    - Priority chain: env vars > YAML file > Pydantic field defaults
    - Cross-field @model_validator(mode="after") for risk parameter consistency

key-files:
  created:
    - src/swingrl/config/schema.py
    - config/swingrl.yaml
    - config/swingrl.prod.yaml.example
    - tests/config/test_schema.py
    - tests/config/__init__.py
  modified:
    - src/swingrl/config/__init__.py
    - pyproject.toml
    - uv.lock
    - .pre-commit-config.yaml

key-decisions:
  - "pydantic-settings 2.13.1 uses file_secret_settings (not secrets_dir) as the 4th arg in settings_customise_sources — fixed to match actual API"
  - "types-PyYAML added to both pyproject.toml dev deps and pre-commit mypy additional_dependencies — pre-commit mypy uses isolated env"
  - "load_config() uses inner _ConfigWithYaml subclass pattern to bind yaml_path at call time, not at class definition time"

patterns-established:
  - "Config loading pattern: load_config(path) is the single entry point — never instantiate SwingRLConfig directly in business logic"
  - "Env var override pattern: SWINGRL_ prefix top-level, SWINGRL_SECTION__FIELD for nested"
  - "Risk parameter invariant: daily_loss_limit_pct must always be < max_drawdown_pct (model_validator enforces)"

requirements-completed:
  - ENV-11

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 2 Plan 03: Config Schema Summary

**Pydantic v2 BaseSettings SwingRLConfig with YAML loading, SWINGRL_ env override, and cross-field risk validators enforcing daily_loss < max_drawdown**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-03-06T17:42:59Z
- **Completed:** 2026-03-06T17:48:20Z
- **Tasks:** 2 (Task 1 TDD, Task 2 YAML files)
- **Files modified:** 8

## Accomplishments

- SwingRLConfig Pydantic v2 BaseSettings with 6 sub-config models and full type annotations
- load_config() entry point with priority chain: env vars > YAML file > defaults; returns {} if file missing
- Cross-field model_validators on EquityConfig and CryptoConfig reject daily_loss_limit_pct >= max_drawdown_pct
- config/swingrl.yaml committed with paper mode, 8 ETFs, Binance.US $10 floor, conservative risk limits
- config/swingrl.prod.yaml.example with annotated production values and Docker absolute paths
- 17 TDD tests covering valid config, invalid field validation, env var overrides, and importability

## Task Commits

Each task was committed atomically:

1. **TDD RED: test_schema.py** - `fc5fa9f` (test)
2. **Task 1: schema.py + __init__.py** - `db502ea` (feat)
3. **Task 2: swingrl.yaml + prod example** - `720e03e` (feat)

## Files Created/Modified

- `src/swingrl/config/schema.py` - Full Pydantic v2 config schema with load_config() and YamlConfigSettingsSource
- `src/swingrl/config/__init__.py` - Re-exports all 7 public symbols from swingrl.config
- `config/swingrl.yaml` - Dev defaults: paper mode, 8 ETFs, crypto $10 floor, 400/47 USD capital
- `config/swingrl.prod.yaml.example` - Production reference: live mode, Docker paths, json_logs: true
- `tests/config/test_schema.py` - 17 TDD tests for schema behavior
- `tests/config/__init__.py` - Package init for test discovery
- `pyproject.toml` - Added types-pyyaml to dev dependencies
- `.pre-commit-config.yaml` - Added types-PyYAML to mypy hook additional_dependencies

## Decisions Made

- **pydantic-settings 2.13.1 API**: The `settings_customise_sources` method signature uses `file_secret_settings` as the 4th parameter (not `secrets_dir` as shown in some older docs). Fixed to match actual installed version.
- **Pre-commit mypy isolation**: Pre-commit mypy runs in its own isolated virtualenv — types-PyYAML must be in `additional_dependencies`, not just uv dev deps. Same pattern as structlog and pydantic-settings.
- **load_config() design**: Inner `_ConfigWithYaml` subclass binds the yaml_path at call time, keeping `SwingRLConfig` importable and instantiable independently for testing while still supporting the factory pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed pydantic-settings 2.13.1 API signature mismatch**
- **Found during:** Task 1 (schema.py implementation, GREEN phase)
- **Issue:** Plan's `settings_customise_sources` used `secrets_dir` as parameter name but pydantic-settings 2.13.1 uses `file_secret_settings` — all 17 tests failed on first run
- **Fix:** Changed `secrets_dir` to `file_secret_settings` in the inner class method signature
- **Files modified:** `src/swingrl/config/schema.py`
- **Verification:** All 17 schema tests passed after fix
- **Committed in:** `db502ea` (Task 1 feat commit)

**2. [Rule 2 - Missing Critical] Added types-PyYAML to pre-commit mypy hook**
- **Found during:** Task 1 commit (pre-commit mypy hook failure)
- **Issue:** Pre-commit mypy runs in isolated env — types-PyYAML installed in uv venv not visible; `import yaml` caused `import-untyped` error blocking commit
- **Fix:** Added `types-PyYAML` to `.pre-commit-config.yaml` mypy `additional_dependencies` and to `pyproject.toml` dev deps
- **Files modified:** `.pre-commit-config.yaml`, `pyproject.toml`
- **Verification:** `uv run pre-commit run mypy` passed after env reinstall
- **Committed in:** `db502ea` (Task 1 feat commit)

---

**Total deviations:** 2 auto-fixed (1 blocking API mismatch, 1 missing critical type stubs)
**Impact on plan:** Both fixes essential for correctness and type safety. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- load_config() is the established single entry point for all future phases
- All sub-config models (EquityConfig, CryptoConfig, etc.) ready for use in Phase 3+ data fetching
- config/swingrl.yaml committed with dev defaults — developers get correct paper mode automatically
- Type checking fully operational: mypy passes on config module, pre-commit hooks pass

---
*Phase: 02-developer-experience*
*Completed: 2026-03-06*
