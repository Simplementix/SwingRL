---
phase: 11-production-startup-wiring
plan: 01
subsystem: execution
tags: [apscheduler, duckdb, fred, feature-pipeline, wiring-bugs]

# Dependency graph
requires:
  - phase: 08-paper-trading
    provides: ExecutionPipeline with 5-arg constructor
  - phase: 05-features
    provides: init_feature_schema DDL for feature tables
  - phase: 03-data-ingestion
    provides: FREDIngestor at swingrl.data.fred
  - phase: 09-automation
    provides: main.py APScheduler entrypoint with 11 jobs
provides:
  - Fixed production entrypoint (main.py) that starts without TypeError
  - Feature tables created during standard DB init (no manual script needed)
  - Correct FRED import path and API in scheduler jobs
affects: [12-schema-alignment-and-emergency-triggers, paper-trading-launch]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Local import in _init_duckdb_schema to avoid circular dependency with features.schema"
    - "Alerter and FeaturePipeline constructed before ExecutionPipeline in build_app"

key-files:
  created: []
  modified:
    - scripts/main.py
    - src/swingrl/data/db.py
    - src/swingrl/scheduler/jobs.py
    - tests/scheduler/test_main.py
    - tests/scheduler/test_jobs.py
    - tests/data/test_db.py

key-decisions:
  - "models_dir points to models/active (Phase 10 convention for active model directory)"
  - "Local import of init_feature_schema inside _init_duckdb_schema to avoid circular imports"

patterns-established:
  - "FeaturePipeline constructed with raw DuckDB conn from db._get_duckdb_conn()"

requirements-completed: [PAPER-01, PAPER-02, PAPER-12, FEAT-11, DATA-09, DATA-04, FEAT-04]

# Metrics
duration: 9min
completed: 2026-03-10
---

# Phase 11 Plan 01: Production Startup Wiring Summary

**Fixed three cross-phase wiring bugs: ExecutionPipeline 5-arg constructor, feature table init in DB schema, and FRED import path correction in scheduler jobs**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-10T14:58:56Z
- **Completed:** 2026-03-10T15:08:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- ExecutionPipeline now receives all 5 required arguments (config, db, feature_pipeline, alerter, models_dir) -- eliminates TypeError on startup
- init_schema() now creates features_equity, features_crypto, fundamentals, and hmm_state_history tables during standard DB initialization
- FRED scheduler jobs import FREDIngestor from the correct module path (swingrl.data.fred), use correct class name, single config arg, and run_all() method

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests for all three bug fixes (RED)** - `537e14a` (test)
2. **Task 2: Fix main.py constructor, db.py feature init, and jobs.py FRED imports (GREEN)** - `e833175` (feat)

_TDD: Task 1 = RED phase (failing tests), Task 2 = GREEN phase (implementation)_

## Files Created/Modified
- `scripts/main.py` - Added FeaturePipeline import, reordered build_app construction sequence, pass all 5 args to ExecutionPipeline
- `src/swingrl/data/db.py` - Added init_feature_schema(cursor) call at end of _init_duckdb_schema()
- `src/swingrl/scheduler/jobs.py` - Fixed FRED import path and API in weekly_fundamentals_job and monthly_macro_job
- `tests/scheduler/test_main.py` - Added test_build_app_creates_pipeline_with_all_args
- `tests/scheduler/test_jobs.py` - Added TestFredImportPath class with two tests
- `tests/data/test_db.py` - Added TestFeatureTableInit class

## Decisions Made
- models_dir points to `models/active` subdirectory (Phase 10 convention for active models)
- Local import of init_feature_schema inside _init_duckdb_schema to avoid circular dependency risk between data.db and features.schema

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Production entrypoint (main.py) can now start without TypeError
- Feature tables are created automatically during standard DB initialization
- FRED scheduler jobs will import and execute correctly
- All 795 tests pass with zero regressions

## Self-Check: PASSED

---
*Phase: 11-production-startup-wiring*
*Completed: 2026-03-10*
