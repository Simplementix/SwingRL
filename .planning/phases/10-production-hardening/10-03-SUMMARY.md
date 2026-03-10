---
phase: 10-production-hardening
plan: 03
subsystem: model-lifecycle, deployment
tags: [lifecycle-state-machine, smoke-test, scp, shadow-first, model-deployment]

# Dependency graph
requires:
  - phase: 10-production-hardening
    provides: "ShadowConfig, PathsConfig, ModelError exception"
provides:
  - "ModelLifecycle state machine (deploy_to_shadow, promote, archive, rollback, delete_archived, get_state)"
  - "smoke_test_model 6-point validation function"
  - "deploy_model.sh SCP + SHA256 + remote smoke test script"
affects: [10-04, 10-05]

# Tech tracking
tech-stack:
  added: []
  patterns: [shadow-first-deployment, lifecycle-state-machine, 6-point-smoke-test]

key-files:
  created:
    - src/swingrl/shadow/__init__.py
    - src/swingrl/shadow/lifecycle.py
    - scripts/deploy_model.sh
    - tests/shadow/__init__.py
    - tests/shadow/test_lifecycle.py
    - tests/test_deploy_smoke.py
  modified: []

key-decisions:
  - "type: ignore[attr-defined] for SB3 algo_cls.load() (dict-based algo dispatch)"
  - "type: ignore[call-arg] for VecNormalize.load() missing venv positional arg in stubs"
  - "VecNormalize .pkl absence treated as pass (not all models use normalization)"
  - "Archive naming uses UTC timestamp suffix for chronological sort ordering"

patterns-established:
  - "Shadow-first deployment: all models land in models/shadow/ before promotion"
  - "6-point smoke test: deserialization, shape, non-degenerate, speed, VecNormalize, no NaN"
  - "SHA256 integrity check: local compute -> SCP transfer -> remote verify -> abort on mismatch"

requirements-completed: [PROD-02, PROD-05]

# Metrics
duration: 5min
completed: 2026-03-10
---

# Phase 10 Plan 03: Model Deployment Pipeline Summary

**Model lifecycle state machine (Training->Shadow->Active->Archive->Deletion) with 6-point smoke test and SCP deploy script for shadow-first deployment**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-10T03:27:25Z
- **Completed:** 2026-03-10T03:32:48Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- ModelLifecycle class managing full lifecycle transitions with proper error handling via ModelError
- 6-point smoke_test_model function validating deserialization, output shape, non-degenerate actions, inference speed, VecNormalize, and NaN detection
- deploy_model.sh script with SCP transfer, SHA256 integrity verification, and remote smoke test execution
- 30 tests covering all lifecycle transitions and smoke test checks

## Task Commits

Each task was committed atomically:

1. **Task 1: Model lifecycle state machine and smoke test**
   - `cde65d9` (test: RED - failing lifecycle and smoke test tests)
   - `5ab0af6` (feat: GREEN - lifecycle state machine and smoke test implementation)
2. **Task 2: deploy_model.sh shell script**
   - `b144ae5` (feat: deploy script with SCP + SHA256 + smoke test)

## Files Created/Modified
- `src/swingrl/shadow/__init__.py` - Package init for shadow module
- `src/swingrl/shadow/lifecycle.py` - ModelState enum, ModelLifecycle class, smoke_test_model function
- `scripts/deploy_model.sh` - SCP model transfer with integrity check and remote smoke test
- `tests/shadow/__init__.py` - Test package init
- `tests/shadow/test_lifecycle.py` - 20 lifecycle transition tests
- `tests/test_deploy_smoke.py` - 10 smoke test validation tests

## Decisions Made
- `type: ignore[attr-defined]` for SB3 `algo_cls.load()` since dict-based dispatch loses type info
- `type: ignore[call-arg]` for `VecNormalize.load()` due to incomplete SB3 type stubs
- VecNormalize `.pkl` file absence treated as pass (not all models use normalization)
- Archive files get UTC timestamp suffix (`_YYYYMMDD_HHMMSS`) for chronological sort ordering

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing staged files from other plans (sentiment module) caused pre-commit failures until unstaged; resolved by staging only task-related files

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ModelLifecycle ready for shadow evaluation (Plan 04) and promotion workflows
- smoke_test_model available for both local and remote validation
- deploy_model.sh ready for M1 Mac -> homelab transfers via Tailscale
- No blockers for Plan 04

---
*Phase: 10-production-hardening*
*Completed: 2026-03-10*
