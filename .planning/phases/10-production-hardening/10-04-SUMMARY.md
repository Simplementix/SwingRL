---
phase: 10-production-hardening
plan: 04
subsystem: shadow-mode, model-evaluation
tags: [shadow-inference, auto-promotion, sharpe-ratio, mdd, lifecycle]

# Dependency graph
requires:
  - phase: 10-production-hardening
    provides: "ShadowConfig, ModelLifecycle state machine, smoke_test_model"
provides:
  - "run_shadow_inference() for parallel hypothetical trade generation"
  - "evaluate_shadow_promotion() with 3-criteria auto-promotion logic"
  - "shadow_trades SQLite table for hypothetical trade storage"
  - "archive_shadow() lifecycle method for failed shadow models"
  - "shadow_promotion_check_job in APScheduler at 7 PM ET daily"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [shadow-parallel-inference, 3-criteria-auto-promotion, import-guarded-shadow]

key-files:
  created:
    - src/swingrl/shadow/shadow_runner.py
    - src/swingrl/shadow/promoter.py
    - tests/shadow/test_shadow_runner.py
    - tests/shadow/test_promoter.py
  modified:
    - src/swingrl/shadow/lifecycle.py
    - src/swingrl/data/db.py
    - src/swingrl/scheduler/jobs.py
    - scripts/main.py

key-decisions:
  - "nosec B608 on promoter SQL table name interpolation (constrained to shadow_trades/trades)"
  - "Sharpe computed from sequential price returns (simple percentage change)"
  - "MDD computed from peak-to-trough on price series"
  - "archive_shadow method added to ModelLifecycle for shadow->archive transition"

patterns-established:
  - "Shadow inference integrated via try/except after active cycle (never affects fills)"
  - "Import-guarded shadow modules: if shadow code unavailable, system continues unaffected"
  - "3-criteria promotion: Sharpe >= active, MDD <= tolerance * active MDD, no CB triggers"

requirements-completed: [PROD-03, PROD-04]

# Metrics
duration: 9min
completed: 2026-03-10
---

# Phase 10 Plan 04: Shadow Mode Summary

**Shadow parallel inference with hypothetical trade recording and 3-criteria auto-promotion (Sharpe, MDD tolerance, CB check) integrated into APScheduler trading cycles**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-10T03:39:26Z
- **Completed:** 2026-03-10T03:48:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Shadow inference runner that loads shadow models, generates hypothetical trades, and records them in shadow_trades SQLite table
- Auto-promotion evaluator with 3 criteria: Sharpe ratio comparison, MDD tolerance check (120% default), and circuit breaker event check
- Full scheduler integration: shadow inference after every active cycle, daily promotion check at 7 PM ET
- Complete exception isolation ensuring shadow failures never affect active trading

## Task Commits

Each task was committed atomically:

1. **Task 1: Shadow runner and promoter with shadow_trades DDL**
   - `e650d4c` (feat: shadow runner, promoter, lifecycle archive_shadow, DDL, tests)
2. **Task 2: Integrate shadow inference into scheduler jobs**
   - `359f2cc` (feat: scheduler integration with shadow inference and promotion check)

## Files Created/Modified
- `src/swingrl/shadow/shadow_runner.py` - Shadow inference runner with no-op, model loading, trade recording
- `src/swingrl/shadow/promoter.py` - 3-criteria evaluation with Sharpe, MDD, CB checks
- `src/swingrl/shadow/lifecycle.py` - Added archive_shadow() method for failed shadow models
- `src/swingrl/data/db.py` - shadow_trades table DDL in init_sqlite_schema()
- `src/swingrl/scheduler/jobs.py` - Shadow inference in cycle jobs, shadow_promotion_check_job
- `scripts/main.py` - Registered shadow_promotion_check_job, job count 10 -> 11
- `tests/shadow/test_shadow_runner.py` - 4 tests for runner behavior
- `tests/shadow/test_promoter.py` - 10 tests for promoter criteria evaluation

## Decisions Made
- `nosec B608` on promoter SQL table name interpolation (table name is always "shadow_trades" or "trades", never user input)
- Sharpe ratio computed from simple sequential price returns (percentage change between consecutive trades)
- MDD computed as peak-to-trough on the price series from trades
- Added `archive_shadow()` to ModelLifecycle since existing `archive()` only handles active->archive, not shadow->archive

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test data for Sharpe comparison**
- **Found during:** Task 1 (promoter test verification)
- **Issue:** Linear ascending prices with different deltas produced counterintuitive Sharpe ratios (smaller delta = higher Sharpe due to lower relative volatility)
- **Fix:** Replaced linear price generators with explicit price series: steady uptrend vs noisy flat for clear Sharpe differentiation
- **Files modified:** tests/shadow/test_promoter.py
- **Verification:** All 14 shadow tests pass
- **Committed in:** e650d4c

---

**Total deviations:** 1 auto-fixed (1 bug in test data)
**Impact on plan:** Test data fix. No scope creep.

## Issues Encountered
- Pre-commit stash conflicts between concurrent plan executors (10-04 and 10-06) caused file staging issues; resolved by using Write tool for full file replacement
- bandit B608 required `# nosec B608` comment (not just ruff's `# noqa: S608`)
- mypy `no-any-return` on Sharpe ratio computation; fixed with explicit `float()` cast

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Shadow inference pipeline ready for production shadow model evaluation
- Promotion criteria configurable via config.shadow (eval days, mdd_tolerance_ratio, auto_promote)
- Lifecycle transitions (promote, archive_shadow) tested and integrated
- No blockers

---
*Phase: 10-production-hardening*
*Completed: 2026-03-10*
