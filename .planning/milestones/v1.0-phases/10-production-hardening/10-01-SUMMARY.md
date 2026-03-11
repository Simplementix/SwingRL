---
phase: 10-production-hardening
plan: 01
subsystem: config, infra
tags: [pydantic, tenacity, structlog, retry, logging, config-schema]

# Dependency graph
requires:
  - phase: 02-developer-experience
    provides: "SwingRLConfig schema, configure_logging(), exceptions hierarchy"
provides:
  - "BackupConfig, ShadowConfig, SentimentConfig, SecurityConfig sub-models"
  - "swingrl_retry tenacity-based decorator with exponential backoff"
  - "File-based JSON logging with RotatingFileHandler"
  - "tenacity, finnhub-python, matplotlib dependencies"
  - "transformers optional dependency group (sentiment)"
affects: [10-02, 10-03, 10-04, 10-05]

# Tech tracking
tech-stack:
  added: [tenacity, finnhub-python, matplotlib, transformers (optional)]
  patterns: [retry-decorator-factory, rotating-file-json-logging]

key-files:
  created:
    - src/swingrl/utils/retry.py
    - tests/test_retry.py
    - tests/utils/__init__.py
    - tests/utils/test_file_logging.py
  modified:
    - src/swingrl/config/schema.py
    - src/swingrl/utils/logging.py
    - config/swingrl.yaml
    - config/swingrl.prod.yaml.example
    - pyproject.toml

key-decisions:
  - "tenacity added to mypy ignore_missing_imports (no type stubs available)"
  - "swingrl_retry returns Any to avoid mypy no-any-return with tenacity's untyped decorator"
  - "transformers in optional [sentiment] dep group to avoid 2GB+ install when disabled"
  - "File handler always uses JSONRenderer regardless of json_logs flag for machine-parseable logs"

patterns-established:
  - "Retry factory pattern: swingrl_retry(**kwargs) returns tenacity decorator with logging"
  - "File logging: configure_logging(log_file=path) adds RotatingFileHandler alongside stream"

requirements-completed: [HARD-02, HARD-05]

# Metrics
duration: 5min
completed: 2026-03-10
---

# Phase 10 Plan 01: Foundation Extensions Summary

**Config sub-models (backup, shadow, sentiment, security), tenacity retry decorator, and rotating file-based JSON logging for production hardening**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-10T03:19:55Z
- **Completed:** 2026-03-10T03:25:01Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Four new Pydantic config sub-models (BackupConfig, ShadowConfig, SentimentConfig, SecurityConfig) with validated defaults and boundary constraints
- swingrl_retry decorator wrapping tenacity with exponential backoff, configurable retryable exceptions, and structlog retry logging
- File-based JSON logging via RotatingFileHandler with configurable rotation size and backup count
- 29 total tests passing (18 config + 7 retry + 4 file logging)

## Task Commits

Each task was committed atomically:

1. **Task 1: Config schema extensions and dependency installation**
   - `de89a5d` (test: RED - failing config sub-model tests)
   - `c965cb9` (feat: GREEN - config sub-models, deps, YAML updates)
2. **Task 2: Retry decorator and file-based JSON logging**
   - `fbd45e1` (test: RED - failing retry and file logging tests)
   - `15f7e0d` (feat: GREEN - retry decorator and file logging implementation)

## Files Created/Modified
- `src/swingrl/config/schema.py` - BackupConfig, ShadowConfig, SentimentConfig, SecurityConfig sub-models
- `src/swingrl/utils/retry.py` - swingrl_retry tenacity-based decorator factory
- `src/swingrl/utils/logging.py` - Extended configure_logging with optional file handler
- `config/swingrl.yaml` - New sections with dev defaults
- `config/swingrl.prod.yaml.example` - New sections with production values
- `pyproject.toml` - tenacity, finnhub-python, matplotlib deps; transformers optional; tenacity mypy ignore
- `tests/test_config.py` - 12 new tests for config sub-models
- `tests/test_retry.py` - 7 tests for retry decorator behavior
- `tests/utils/__init__.py` - Package init
- `tests/utils/test_file_logging.py` - 4 tests for file logging

## Decisions Made
- tenacity added to mypy ignore_missing_imports (no type stubs available)
- swingrl_retry returns `Any` type to work around tenacity's untyped decorator return
- transformers placed in `[project.optional-dependencies] sentiment` group to avoid heavy install when sentiment is disabled
- File handler always uses JSONRenderer regardless of `json_logs` flag for consistent machine-parseable file output

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed tenacity statistics test assertion**
- **Found during:** Task 2 (retry test verification)
- **Issue:** Test assumed tenacity stores `attempt_number` in `statistics` dict, but statistics is empty without explicit configuration
- **Fix:** Changed test to verify retry count via call_count and existence of `.retry` attribute
- **Files modified:** tests/test_retry.py
- **Verification:** All 7 retry tests pass
- **Committed in:** 15f7e0d

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor test assertion fix. No scope creep.

## Issues Encountered
- Pre-commit ruff hooks auto-fixed import ordering in test files (expected behavior, no manual intervention needed)
- mypy flagged tenacity imports and Any return type -- resolved by adding to ignore_missing_imports and adjusting return annotation

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Config sub-models ready for all Phase 10 plans (backup, shadow, sentiment, security)
- Retry decorator available for broker API, data ingestion, and external service calls
- File logging ready for production log aggregation setup
- No blockers for Plan 02

---
*Phase: 10-production-hardening*
*Completed: 2026-03-10*
