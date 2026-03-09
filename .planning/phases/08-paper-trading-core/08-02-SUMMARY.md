---
phase: 08-paper-trading-core
plan: 02
subsystem: execution
tags: [kelly-criterion, atr-stops, cost-gate, signal-interpreter, position-sizer, order-validator]

# Dependency graph
requires:
  - phase: 07-agent-training
    provides: ensemble blender producing blended action arrays
  - phase: 08-paper-trading-core plan 01
    provides: pipeline types (TradeSignal, SizedOrder, ValidatedOrder) and RiskManager
provides:
  - SignalInterpreter: ensemble action arrays to discrete TradeSignal list
  - PositionSizer: quarter-Kelly sizing with ATR stops and crypto $10 floor
  - OrderValidator: cost gate (2% RT threshold) + RiskManager delegation
affects: [08-paper-trading-core plan 03, 08-paper-trading-core plan 04]

# Tech tracking
tech-stack:
  added: []
  patterns: [quarter-kelly-sizing, atr-stop-placement, cost-gate-validation]

key-files:
  created:
    - src/swingrl/execution/signal_interpreter.py
    - src/swingrl/execution/position_sizer.py
    - src/swingrl/execution/order_validator.py
    - tests/execution/test_signal_interpreter.py
    - tests/execution/test_position_sizer.py
    - tests/execution/test_order_validator.py
  modified:
    - src/swingrl/execution/types.py

key-decisions:
  - "Deadzone boundary (exactly +/-0.02) treated as hold, not trade -- strict inequality for buy/sell"
  - "Cost gate uses proportional rate (cost_rate = RT_PCT), not fixed dollar -- equity 0.06% and crypto 0.22% always pass under 2% threshold"
  - "ValidatedOrder.order field name (not sized_order) per linter-applied frozen dataclass convention"

patterns-established:
  - "Three-stage middleware pipeline: interpret -> size -> validate"
  - "Quarter-Kelly with 2% risk cap as conservative default for paper trading"
  - "ATR(2x) stop-loss with 2:1 R:R take-profit ratio"

requirements-completed: [PAPER-07, PAPER-08, PAPER-11, PAPER-09]

# Metrics
duration: 5min
completed: 2026-03-09
---

# Phase 8 Plan 02: Execution Middleware Summary

**Three-stage execution middleware: signal interpreter with deadzone, quarter-Kelly position sizer with ATR stops and crypto floor, and order validator with 2% cost gate**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-09T04:05:38Z
- **Completed:** 2026-03-09T04:11:13Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- SignalInterpreter converts ensemble blended actions to discrete buy/sell/hold TradeSignals with +/-0.02 deadzone filtering
- PositionSizer implements quarter-Kelly criterion (K*0.25) with 2% max risk cap, ATR(2x) stop-losses, 2:1 R:R take-profit, and crypto $10 Binance.US floor with post-floor risk rejection
- OrderValidator enforces 2% round-trip cost gate (equity 0.06%, crypto 0.22%) and delegates to RiskManager for position/exposure/drawdown/daily-loss/global checks
- 29 tests total covering all edge cases: deadzone boundaries, negative Kelly skip, crypto floor adjustment and rejection, cost gate rejection, risk veto propagation

## Task Commits

Each task was committed atomically:

1. **Task 1: Signal interpreter and position sizer** - `8b95057` (feat) -- Note: committed in prior session alongside Plan 03 adapter work
2. **Task 2: Order validator with cost gate** - `67ee7b3` (feat)

_Note: TDD tasks with RED->GREEN flow_

## Files Created/Modified
- `src/swingrl/execution/signal_interpreter.py` - Stage 1: ensemble actions to TradeSignal list with deadzone
- `src/swingrl/execution/position_sizer.py` - Stage 2: quarter-Kelly sizing with ATR stops and crypto floor
- `src/swingrl/execution/order_validator.py` - Stage 3: cost gate + RiskManager delegation
- `src/swingrl/execution/types.py` - Pipeline DTOs updated to frozen dataclasses with Literal types
- `tests/execution/test_signal_interpreter.py` - 8 tests: deadzone, buy/sell, crypto mapping, boundaries
- `tests/execution/test_position_sizer.py` - 14 tests: Kelly, risk cap, ATR stops, crypto floor, quantity
- `tests/execution/test_order_validator.py` - 7 tests: cost gate, crypto rate, delegation, veto propagation

## Decisions Made
- Deadzone boundary at exactly +/-0.02 is treated as hold (strict inequality for buy/sell trigger) -- prevents marginal trades
- Cost gate uses proportional rate check (cost_pct = cost_rate constant), meaning standard broker rates (0.06% equity, 0.22% crypto) always pass; the gate protects against future fee increases or alternative brokers
- ValidatedOrder field renamed from `sized_order` to `order` by linter to match frozen dataclass convention from types.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created Plan 01 prerequisite types and RiskManager stub**
- **Found during:** Task 1 (Signal interpreter)
- **Issue:** Plan 02 depends on types.py and risk_manager.py from Plan 01, but Plan 01 was not executed
- **Fix:** Created types.py with all 5 pipeline dataclasses and RiskManager stub with approve-all behavior
- **Files modified:** src/swingrl/execution/types.py, src/swingrl/execution/risk/risk_manager.py, src/swingrl/execution/risk/__init__.py
- **Verification:** All 29 tests pass with stub types
- **Committed in:** 8b95057 (Task 1 commit)

**2. [Rule 1 - Bug] Adapted to linter-modified types.py (frozen dataclasses)**
- **Found during:** Task 2 (Order validator)
- **Issue:** Linter changed types.py to use frozen=True, Literal types, and renamed ValidatedOrder.sized_order to .order
- **Fix:** Updated OrderValidator to use `order=` kwarg and tests to assert `result.order`
- **Files modified:** src/swingrl/execution/order_validator.py, tests/execution/test_order_validator.py
- **Verification:** All 29 tests pass
- **Committed in:** 67ee7b3 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for execution. No scope creep.

## Issues Encountered
- Prior session had partially executed Plans 02 and 03 together in a single commit (8b95057). Task 1 files were already committed but needed Task 2 as a separate atomic commit.
- Pre-commit hooks were picking up unstaged Plan 01 files (conftest.py, test_position_tracker.py) causing ruff failures. Resolved by carefully staging only Task 2 files.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 3 middleware stages (interpret, size, validate) are composable and ready for pipeline wiring
- RiskManager stub needs full implementation in Plan 01 execution
- Stages can be tested independently or chained: SignalInterpreter -> PositionSizer -> OrderValidator

---
*Phase: 08-paper-trading-core*
*Completed: 2026-03-09*
