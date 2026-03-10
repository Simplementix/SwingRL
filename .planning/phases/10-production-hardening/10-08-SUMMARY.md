---
phase: 10-production-hardening
plan: 08
subsystem: shadow-trading
tags: [shadow-inference, feature-pipeline, signal-interpreter, position-sizer, trade-generation]

# Dependency graph
requires:
  - phase: 10-production-hardening-04
    provides: "Shadow runner skeleton with stub _generate_hypothetical_trades"
  - phase: 08-paper-trading
    provides: "SignalInterpreter, PositionSizer, ExecutionPipeline, FeaturePipeline"
provides:
  - "Fully wired _generate_hypothetical_trades producing trade dicts from model predictions"
  - "Shadow mode can now populate shadow_trades table with hypothetical trades"
affects: [shadow-promotion, auto-promotion, production-readiness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy imports inside function body for circular dependency prevention"
    - "Zero-weight current_weights for shadow positions (no real holdings)"
    - "try/except wrapping entire shadow trade generation (never crash active cycle)"

key-files:
  created: []
  modified:
    - "src/swingrl/shadow/shadow_runner.py"
    - "tests/shadow/test_shadow_runner.py"

key-decisions:
  - "Shadow uses zero portfolio weights since it has no real positions"
  - "VecNormalize loaded via model.get_vec_normalize_env() if available"
  - "Estimated prices use static fallbacks (100.0 equity, 50000.0 crypto) for sizing"
  - "ATR fallback of 0.02 for shadow position sizing"

patterns-established:
  - "Shadow pipeline mirrors active pipeline stages but without ensemble blending"
  - "Lazy imports (PLC0415) for execution layer classes inside shadow functions"

requirements-completed: [PROD-03, PROD-04]

# Metrics
duration: 5min
completed: 2026-03-10
---

# Phase 10 Plan 08: Shadow Trade Generation Summary

**Replaced shadow trade stub with full pipeline wiring: get_observation -> model.predict -> SignalInterpreter.interpret -> PositionSizer.size -> shadow_trades dicts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-10T04:33:04Z
- **Completed:** 2026-03-10T04:38:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Replaced the `return []` stub in `_generate_hypothetical_trades` with real pipeline wiring
- Shadow inference now produces trade dicts matching the shadow_trades schema when models predict non-hold actions
- Added 5 new tests verifying trade generation end-to-end with mocked pipeline components
- All 39 shadow tests pass (9 runner + 30 promoter), all 4 original tests unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for _generate_hypothetical_trades** - `fedb6d3` (test)
2. **Task 1 GREEN: Implement _generate_hypothetical_trades** - `07e22b6` (feat)

_Note: TDD task with RED then GREEN commits_

## Files Created/Modified
- `src/swingrl/shadow/shadow_runner.py` - Replaced stub with full pipeline wiring: observation, predict, interpret, size, trade dict creation
- `tests/shadow/test_shadow_runner.py` - Added 5 new tests in TestGenerateHypotheticalTrades class

## Decisions Made
- Shadow uses zero portfolio weights (np.zeros) since shadow model has no real positions -- diff from zero drives all signal interpretation
- VecNormalize attempted via model.get_vec_normalize_env() with fallback to raw observation
- Estimated prices use static fallbacks (100.0 equity, 50000.0 crypto) since shadow doesn't need exact market prices for hypothetical sizing
- ATR fallback of 0.02 (conservative) since shadow doesn't query live ATR values
- Lazy imports (inside function body) for SignalInterpreter and PositionSizer to prevent circular dependency

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Test patches target source modules instead of shadow_runner**
- **Found during:** Task 1 GREEN
- **Issue:** Plan suggested patching `swingrl.shadow.shadow_runner.SignalInterpreter` but lazy imports inside the function body mean the attribute doesn't exist on the module
- **Fix:** Changed patches to target `swingrl.execution.signal_interpreter.SignalInterpreter` and `swingrl.execution.position_sizer.PositionSizer`
- **Files modified:** tests/shadow/test_shadow_runner.py
- **Verification:** All 9 runner tests pass
- **Committed in:** 07e22b6

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Patch target adjustment necessary due to lazy import pattern. No scope creep.

## Issues Encountered
- Pre-existing test failure in `tests/scheduler/test_main.py::test_main_registers_six_jobs` (unrelated to shadow changes, verified by running on stashed state)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Shadow mode can now produce hypothetical trades from model predictions
- shadow_trades table will be populated during shadow inference cycles
- Auto-promotion evaluator can now assess real shadow performance data

---
*Phase: 10-production-hardening*
*Completed: 2026-03-10*
