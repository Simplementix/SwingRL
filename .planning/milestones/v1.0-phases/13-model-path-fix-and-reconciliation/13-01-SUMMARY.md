---
phase: 13-model-path-fix-and-reconciliation
plan: 01
subsystem: scheduler
tags: [apscheduler, reconciliation, model-loading, equity, jobs, tdd]

# Dependency graph
requires:
  - phase: 11-production-startup-wiring
    provides: ExecutionPipeline, models_dir convention, APScheduler job pattern
  - phase: 12-schema-alignment-and-emergency-triggers
    provides: PositionReconciler, AlpacaAdapter, Alerter
provides:
  - Fixed model path: ExecutionPipeline receives bare models_dir (no double 'active' nesting)
  - reconciliation_job function in jobs.py with halt check and consecutive failure escalation
  - daily_reconciliation APScheduler job registered at 5 PM ET (id='daily_reconciliation')
  - 12-job scheduler replacing previous 11-job configuration
affects: [phase-14, production-deployment, model-loading, daily-equity-cycle]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Literal type annotation for alert level variable to satisfy mypy on send_alert calls
    - Module-level integer counter (_reconciliation_failures) for consecutive failure tracking in job functions
    - local import pattern for AlpacaAdapter and PositionReconciler inside reconciliation_job (noqa: PLC0415)

key-files:
  created: []
  modified:
    - scripts/main.py
    - src/swingrl/scheduler/jobs.py
    - tests/scheduler/test_main.py
    - tests/scheduler/test_jobs.py
    - tests/execution/test_pipeline.py

key-decisions:
  - "reconciliation_job uses in-memory _reconciliation_failures counter (not DB) -- matches existing _ctx pattern and avoids DB write on every job"
  - "Literal['critical', 'warning'] type annotation required for mypy on send_alert level argument"
  - "patch target for PPO in test is stable_baselines3.PPO (local import inside _load_models), not swingrl.execution.pipeline.PPO"

patterns-established:
  - "Consecutive failure tracking: module-level integer counter + global declaration + Literal type for level variable"
  - "Job count updates: module docstring + create_scheduler_and_register_jobs docstring + log.info count + swingrl_app_built job_count"

requirements-completed: [PAPER-02, PAPER-09]

# Metrics
duration: 9min
completed: 2026-03-10
---

# Phase 13 Plan 01: Model Path Fix and Reconciliation Scheduling Summary

**Fixed models_dir double-nesting bug in main.py and wired PositionReconciler as a daily APScheduler job at 5 PM ET with consecutive failure escalation**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-10T19:12:54Z
- **Completed:** 2026-03-10T19:21:37Z
- **Tasks:** 2 (1 TDD, 1 validation)
- **Files modified:** 5

## Accomplishments
- Removed `/ "active"` from `models_dir` in `build_app()` -- pipeline's `_load_models()` was already appending `active/`, causing `models/active/active/{env}/{algo}/model.zip` double-nesting that prevented all model loads
- Added `reconciliation_job()` to `src/swingrl/scheduler/jobs.py` with halt check, local adapter/reconciler import, equity-only reconciliation, and consecutive failure tracking (warning for 1-2, critical for 3+)
- Registered `daily_reconciliation` at 17:00 ET in `create_scheduler_and_register_jobs()`, updating job count from 11 to 12 across all references (docstrings, log.info calls, module docstring)
- Full test suite: 817 tests pass, zero failures, ruff clean, mypy clean on changed files

## Task Commits

1. **Task 1 (RED): Failing tests** - `bada5c7` (test)
2. **Task 1 (GREEN): Implementation** - `f5be9d5` (feat)

_Task 2 (full suite validation) confirmed passing inline; no additional commit needed (no code changes)._

**Plan metadata:** _(pending docs commit)_

## Files Created/Modified
- `scripts/main.py` - Removed `/ "active"` from models_dir, added reconciliation_job import, registered daily_reconciliation job, updated job counts (11->12) in 4 places
- `src/swingrl/scheduler/jobs.py` - Added `_reconciliation_failures: int = 0` counter, `reconciliation_job()` function with full halt/failure/alerting logic, added `Literal` to imports
- `tests/scheduler/test_main.py` - Updated job count assertions (11->12), added `daily_reconciliation` to expected_ids, added `test_daily_reconciliation_schedule` and `test_build_app_passes_bare_models_dir_to_pipeline`
- `tests/scheduler/test_jobs.py` - Added `TestReconciliationJob` class (3 tests: success, halt-skip, consecutive failure escalation)
- `tests/execution/test_pipeline.py` - Added `test_load_models_path_no_double_active` verifying no double active/ nesting

## Decisions Made
- Consecutive failure counter is in-memory (`_reconciliation_failures` module global) -- consistent with `_ctx` pattern, no DB overhead per run
- `level` variable annotated as `Literal["critical", "warning"]` to satisfy mypy's strict Literal constraint on `Alerter.send_alert()`
- Patched `stable_baselines3.PPO` (not `swingrl.execution.pipeline.PPO`) in pipeline path test -- PPO is a local import inside `_load_models()`, not a module-level attribute

## Deviations from Plan

None - plan executed exactly as written. One minor fix in test patching (stable_baselines3.PPO vs swingrl.execution.pipeline.PPO) was discovered during RED phase before any commit.

## Issues Encountered
- `patch("swingrl.execution.pipeline.PPO")` raised `AttributeError` -- PPO is imported locally inside `_load_models()`, not at module level. Fixed before RED commit by patching `stable_baselines3.PPO` instead.
- `mypy` flagged `level = "critical" if ... else "warning"` as `str` not `Literal["critical", "warning"]` for `send_alert`. Fixed by adding explicit `Literal["critical", "warning"]` type annotation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Model path bug closed: production models will now load correctly from `models/active/{env}/{algo}/model.zip`
- Daily reconciliation will detect equity position drift between DB and Alpaca every day at 5 PM ET
- Production trading cycle fully wired: equity_cycle (4:15 PM) -> reconciliation_job (5:00 PM) -> daily_summary (6:00 PM)
- No blockers for next phase

## Self-Check: PASSED

- SUMMARY.md: FOUND
- scripts/main.py: FOUND (models_dir bug fixed, reconciliation_job imported, 12 jobs)
- src/swingrl/scheduler/jobs.py: FOUND (reconciliation_job defined)
- Commit bada5c7: FOUND (RED tests)
- Commit f5be9d5: FOUND (GREEN implementation)
- Full test suite: 817 passed, 0 failed

---
*Phase: 13-model-path-fix-and-reconciliation*
*Completed: 2026-03-10*
