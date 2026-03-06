---
phase: 02-developer-experience
plan: "04"
subsystem: testing
tags: [pytest, conftest, fixtures, pydantic, config-validation, tdd, smoke-tests]

# Dependency graph
requires:
  - phase: 02-developer-experience-plan-03
    provides: "SwingRLConfig Pydantic v2 schema with load_config() and cross-field model_validators"
  - phase: 02-developer-experience-plan-01
    provides: "models/active|shadow|archive .gitkeep files + .claude/commands/ skill files"
  - phase: 02-developer-experience-plan-02
    provides: "CLAUDE.md at repo root"
provides:
  - "7 shared test fixtures in conftest.py (valid_config_yaml, tmp_config, loaded_config, equity_ohlcv, crypto_ohlcv, tmp_dirs, repo_root)"
  - "6 config validation tests in test_config.py covering ENV-11 requirements"
  - "3 new smoke assertions in test_smoke.py for ENV-09, ENV-10, ENV-12"
  - "Full Phase 2 test suite: 45 tests, zero failures"
affects:
  - Phase 3 and beyond (all phases inherit fixture patterns from conftest.py)
  - Any test that exercises config validation

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "conftest.py fixture hierarchy: session-scoped repo_root + function-scoped config/data fixtures"
    - "TDD: write failing tests first, then confirm they pass against existing implementation"
    - "tmp_path-based config testing: each test gets isolated YAML file via tmp_config fixture"
    - "monkeypatch.setenv for env var override tests (no global state pollution)"

key-files:
  created:
    - tests/test_config.py
  modified:
    - tests/conftest.py
    - tests/test_smoke.py

key-decisions:
  - "conftest.py fixture scopes: session for repo_root only; function-scoped for everything else to prevent cross-test state mutation"
  - "valid_config_yaml as separate string fixture allows test_config.py to construct bad YAML variations independently"
  - "equity_ohlcv uses seed 42, crypto_ohlcv uses seed 43 — deterministic but distinct random states"

patterns-established:
  - "Fixture pattern: tmp_config writes YAML to tmp_path; loaded_config calls load_config(tmp_config)"
  - "OHLCV fixture pattern: pd.date_range with tz=UTC + numpy.random.default_rng for deterministic data"
  - "Bad YAML tests: use tmp_path directly (not tmp_config fixture) to construct invalid variants"

requirements-completed: [ENV-11, ENV-12]

# Metrics
duration: 2min
completed: "2026-03-06"
---

# Phase 2 Plan 04: Test Suite Expansion Summary

**pytest conftest.py with 7 shared fixtures + 6 config validation tests (ENV-11) + 3 smoke assertions (ENV-09/10/12), establishing the fixture baseline all future phases inherit**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-06T17:51:19Z
- **Completed:** 2026-03-06T17:53:16Z
- **Tasks:** 2 completed
- **Files modified:** 3 (conftest.py, test_config.py, test_smoke.py)

## Accomplishments

- Expanded conftest.py from 1 fixture (repo_root) to 7 fixtures covering config, OHLCV data, and directory tree
- Created test_config.py with 6 tests validating the full load_config() + SwingRLConfig roundtrip including @model_validator cross-field guard
- Added 3 smoke assertions to test_smoke.py confirming ENV-09 (CLAUDE.md), ENV-10 (.claude/commands/), and ENV-12 (models/.gitkeep) requirements
- Full suite gate: 45 tests pass, ruff clean, mypy clean — Phase 2 complete

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand tests/conftest.py with shared fixtures** - `b653414` (feat)
2. **Task 2: Create tests/test_config.py and add smoke test assertions for ENV-09/10/12** - `cd56b49` (feat)

## Files Created/Modified

- `tests/conftest.py` - Expanded from 1 to 7 fixtures: valid_config_yaml, tmp_config, loaded_config, equity_ohlcv (20-row UTC daily), crypto_ohlcv (40-row UTC 4H), tmp_dirs (6-dir tree), repo_root
- `tests/test_config.py` - New: 6 config validation tests (ENV-11): valid load, invalid trading_mode, invalid position_size, env var override, missing file defaults, cross-field model_validator
- `tests/test_smoke.py` - Appended 3 tests: test_claude_md_exists (ENV-09), test_claude_commands_exist (ENV-10), test_models_directories_exist (ENV-12)

## Decisions Made

- conftest.py fixture scopes follow the rule: session-scoped only for `repo_root` (read-only, expensive); function-scoped for all others to prevent mutation-based cross-test interference
- `valid_config_yaml` is a separate string fixture (not inlined in tmp_config) — this allows individual tests to write their own bad YAML without depending on the valid template
- bad YAML tests use `tmp_path` directly rather than the `tmp_config` fixture to keep invalid configs independent of the valid baseline

## Deviations from Plan

None — plan executed exactly as written. ruff-format auto-applied formatting corrections during pre-commit (addBlank line after module docstring), which is expected behavior per project conventions.

## Issues Encountered

None. ruff-format reformatted both new test files during pre-commit hooks (blank line after module docstring). Re-staged and committed on the second attempt — standard project workflow.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 complete: all 4 plans done, 45 tests green, ruff + mypy clean
- conftest.py fixtures are ready for immediate use in Phase 3+ test files
- tmp_config / loaded_config pattern is the canonical way to test anything that touches SwingRLConfig
- equity_ohlcv / crypto_ohlcv are ready for Phase 5 (Feature Engineering) and Phase 6 (Environments)

---
*Phase: 02-developer-experience*
*Completed: 2026-03-06*
