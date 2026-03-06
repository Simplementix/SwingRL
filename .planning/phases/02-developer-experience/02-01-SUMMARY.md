---
phase: 02-developer-experience
plan: 01
subsystem: infra
tags: [structlog, pydantic-settings, exceptions, logging, uv, mypy, pre-commit]

# Dependency graph
requires:
  - phase: 01-dev-foundation
    provides: pyproject.toml, uv lockfile, pre-commit hooks, src/swingrl package skeleton
provides:
  - structlog and pydantic-settings in project dependencies and uv.lock
  - SwingRLError hierarchy (7 typed exception classes) in src/swingrl/utils/exceptions.py
  - configure_logging() structlog setup function in src/swingrl/utils/logging.py
  - Public re-exports from src/swingrl/utils/__init__.py
  - models/active, models/shadow, models/archive .gitkeep files (ENV-12)
affects:
  - 02-developer-experience (plans 02-04 all import from swingrl.utils)
  - All future phases using exceptions or logging

# Tech tracking
tech-stack:
  added:
    - structlog>=24.0 (structured logging library)
    - pydantic-settings>=2.2 (settings management with env var support)
  patterns:
    - TDD red-green: failing tests committed before implementation
    - Exception hierarchy: all errors inherit from SwingRLError base class
    - Structlog ProcessorFormatter: stdlib logging bridge for consistent output
    - json_logs flag: False=ConsoleRenderer (dev), True=JSONRenderer (prod/Docker)

key-files:
  created:
    - src/swingrl/utils/exceptions.py
    - src/swingrl/utils/logging.py
    - tests/utils/test_utils.py
  modified:
    - pyproject.toml (added structlog, pydantic-settings deps; mypy overrides updated)
    - uv.lock (regenerated with new dependencies)
    - src/swingrl/utils/__init__.py (re-exports all 7 exceptions + configure_logging)
    - .pre-commit-config.yaml (added structlog + pydantic-settings to mypy additional_dependencies)

key-decisions:
  - "structlog.* added to mypy ignore_missing_imports — no PEP 561 stubs available for structlog in pre-commit isolated env"
  - "Pre-commit mypy hook needs structlog + pydantic-settings as additional_dependencies — isolated env does not inherit venv packages"
  - "darwin constraint was already absent from pyproject.toml — deferred action from STATE.md was a no-op"
  - "configure_logging uses stdlib logging bridge (ProcessorFormatter) so third-party libraries also emit structured logs"

patterns-established:
  - "Exception pattern: raise BrokerError('msg') — never bare Exception"
  - "Logging pattern: configure_logging() once at startup, structlog.get_logger(__name__) per module"
  - "TDD pattern: commit failing tests (RED), then commit implementation (GREEN)"

requirements-completed: [ENV-12, ENV-13]

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 2 Plan 01: Dev Foundation Utilities Summary

**structlog + pydantic-settings added, cross-platform lockfile regenerated, 7-class SwingRLError hierarchy and configure_logging() implemented via TDD, models/ .gitkeep files confirmed (ENV-12)**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-03-06T17:33:46Z
- **Completed:** 2026-03-06T17:39:55Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Removed darwin lockfile constraint (was already absent; no-op confirmed)
- Added structlog>=24.0 and pydantic-settings>=2.2 dependencies via `uv add`, lockfile regenerated
- Created `exceptions.py` with `SwingRLError` base and 6 typed subclasses (ConfigError, BrokerError, DataError, ModelError, CircuitBreakerError, RiskVetoError)
- Created `logging.py` with `configure_logging(json_logs, log_level)` using structlog ProcessorFormatter bridge
- Updated `utils/__init__.py` to re-export all 7 exception classes + `configure_logging`
- 11 TDD tests pass (RED committed first, GREEN committed with implementation)
- All 8 existing smoke tests remain green

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove darwin constraint, add deps, ensure models/ .gitkeep** - `5e42722` (chore)
2. **Task 2 RED: Failing tests for exception hierarchy and logging** - `e8c8e26` (test)
3. **Task 2 GREEN: SwingRLError hierarchy and configure_logging implementation** - `3871e63` (feat, part of 02-02 session)

_Note: TDD task 2 has RED and GREEN commits as per TDD protocol._

## Files Created/Modified

- `src/swingrl/utils/exceptions.py` - SwingRLError base + 6 typed subclasses
- `src/swingrl/utils/logging.py` - configure_logging() with structlog ConsoleRenderer/JSONRenderer
- `src/swingrl/utils/__init__.py` - Public re-exports for all 7 exceptions + configure_logging
- `tests/utils/test_utils.py` - 11 tests covering exception hierarchy and logging
- `pyproject.toml` - Added structlog>=24.0, pydantic-settings>=2.2; updated mypy overrides
- `uv.lock` - Regenerated with new dependencies (cross-platform, no darwin constraint)
- `.pre-commit-config.yaml` - Added structlog + pydantic-settings to mypy additional_dependencies

## Decisions Made

- **structlog.* mypy override:** structlog has py.typed but the pre-commit mypy isolated env requires explicit `additional_dependencies` — both `structlog>=24.0` and `pydantic-settings>=2.2` added to the mypy hook config.
- **darwin constraint:** pyproject.toml already had no `[tool.uv.environments]` section; the deferred action from STATE.md was effectively a no-op (the platform routing via `[tool.uv.sources]` markers was already cross-platform).
- **stdlib bridge:** `configure_logging()` uses `structlog.stdlib.ProcessorFormatter` so third-party libraries that emit via stdlib logging also get structured output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added structlog and pydantic-settings to pre-commit mypy additional_dependencies**
- **Found during:** Task 2 GREEN (implementation commit)
- **Issue:** Pre-commit mypy hook uses an isolated env that doesn't inherit venv packages; import of `structlog` failed with `import-not-found` in the hook even though mypy passed when run directly via `uv run mypy`
- **Fix:** Added `structlog>=24.0` and `pydantic-settings>=2.2` to `additional_dependencies` in `.pre-commit-config.yaml` mypy hook
- **Files modified:** `.pre-commit-config.yaml`
- **Verification:** `uv run pre-commit run mypy --files src/swingrl/utils/logging.py` passes; full commit hook passes
- **Committed in:** Included in implementation commit (part of 02-02 HEAD)

**2. [Rule 1 - Bug] Replaced B017 blind-exception assert in test**
- **Found during:** Task 2 RED commit
- **Issue:** `with pytest.raises(Exception): raise SwingRLError(...)` triggers ruff B017 (don't assert blind exception)
- **Fix:** Changed test to `assert issubclass(SwingRLError, Exception)` — semantically equivalent, ruff-clean
- **Files modified:** `tests/utils/test_utils.py`
- **Verification:** Ruff passes; test still validates SwingRLError is an Exception subclass
- **Committed in:** `e8c8e26`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness and CI compliance. No scope creep.

## Issues Encountered

- Pre-commit mypy isolated environment doesn't inherit venv packages — structlog import-not-found despite structlog being installed. Resolved by adding to `additional_dependencies`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All utility modules ready for import by Phase 2 Plans 02-04 and all future phases
- `from swingrl.utils import SwingRLError, configure_logging` works without extra setup
- ENV-12 satisfied: models/active, models/shadow, models/archive .gitkeep files tracked
- ENV-13 satisfied: structlog structured logging configured and tested

---
*Phase: 02-developer-experience*
*Completed: 2026-03-06*
