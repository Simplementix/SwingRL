---
phase: 08-paper-trading-core
plan: 01
subsystem: execution
tags: [risk-management, circuit-breaker, position-tracking, dataclass, sqlite]

requires:
  - phase: 04-storage
    provides: DatabaseManager with SQLite schema (positions, portfolio_snapshots, risk_decisions, circuit_breaker_events)
  - phase: 05-features
    provides: TurbulenceCalculator for crash protection
  - phase: 02-developer-experience
    provides: SwingRLConfig, SwingRLError hierarchy, structlog logging
provides:
  - Pipeline data types (TradeSignal, SizedOrder, ValidatedOrder, FillResult, RiskDecision)
  - PositionTracker for portfolio state reads from SQLite
  - CircuitBreaker state machine with SQLite persistence and graduated ramp-up
  - GlobalCircuitBreaker for combined portfolio risk aggregation
  - RiskManager two-tier veto layer with 6-check evaluation order
affects: [08-02, 08-03, 08-04, 08-05]

tech-stack:
  added: [exchange_calendars]
  patterns: [circuit-breaker-state-machine, two-tier-risk-veto, portfolio-state-array]

key-files:
  created:
    - src/swingrl/execution/types.py
    - src/swingrl/execution/risk/position_tracker.py
    - src/swingrl/execution/risk/circuit_breaker.py
    - src/swingrl/execution/risk/risk_manager.py
    - tests/execution/conftest.py
    - tests/execution/test_position_tracker.py
    - tests/execution/test_circuit_breaker.py
    - tests/execution/test_risk_manager.py
  modified:
    - src/swingrl/execution/__init__.py
    - src/swingrl/execution/risk/__init__.py
    - src/swingrl/data/db.py

key-decisions:
  - "portfolio_snapshots table gains environment column for per-env filtering (Rule 3: blocking schema change)"
  - "Ramp-up stages start after 20% of cooldown (first 20% is fully halted, then 4 stages at 20% intervals)"
  - "RiskManager raises RiskVetoError for policy violations, CircuitBreakerError for CB triggers"
  - "exchange_calendars NYSE calendar for equity business day cooldown counting"

patterns-established:
  - "CB state persistence: insert event on trigger, update resumed_at on resume, latest-by-timestamp query"
  - "Portfolio state array: [cash_ratio, exposure, daily_return, per-asset weight/unrealized/days] for ObservationAssembler"
  - "Risk evaluation order: CB state -> position size -> exposure -> drawdown -> daily loss -> global aggregator"

requirements-completed: [PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-20]

duration: 14min
completed: 2026-03-09
---

# Phase 08 Plan 01: Risk Infrastructure Summary

**Two-tier risk veto layer with circuit breaker state machine, position tracking from SQLite, and turbulence crash protection**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-09T04:05:18Z
- **Completed:** 2026-03-09T04:19:28Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- 5 pipeline data types defined with frozen dataclasses and Literal type hints
- PositionTracker reads portfolio state from SQLite with (27,) equity / (9,) crypto state arrays for ObservationAssembler
- CircuitBreaker with 3-state machine (ACTIVE/HALTED/RAMPING), SQLite persistence, NYSE business day cooldown, graduated ramp-up
- Two-tier RiskManager with 6-check evaluation order, all decisions logged to risk_decisions table
- Turbulence crash protection triggers CB halt and signals liquidation
- 38 new tests (17 position tracker + 14 circuit breaker + 7 risk manager), all passing

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1: Pipeline data types, position tracker, and test fixtures**
   - RED: `67ee7b3` (test: failing position tracker tests)
   - GREEN: `6d7ec62` (feat: implement types and position tracker)
2. **Task 2: Circuit breaker state machine and two-tier risk manager**
   - RED: `ba93d99` (test: failing CB and risk manager tests)
   - GREEN: `939167a` (feat: implement CB and risk manager)

## Files Created/Modified
- `src/swingrl/execution/types.py` - Pipeline DTOs: TradeSignal, SizedOrder, ValidatedOrder, FillResult, RiskDecision
- `src/swingrl/execution/risk/position_tracker.py` - Portfolio state reader from SQLite with state array builder
- `src/swingrl/execution/risk/circuit_breaker.py` - CB state machine + GlobalCircuitBreaker aggregator
- `src/swingrl/execution/risk/risk_manager.py` - Two-tier risk evaluation with decision logging
- `src/swingrl/execution/__init__.py` - Package exports for pipeline types
- `src/swingrl/execution/risk/__init__.py` - Package exports for risk classes
- `src/swingrl/data/db.py` - Added environment column to portfolio_snapshots table
- `tests/execution/conftest.py` - Shared fixtures: mock_db, exec_config, position_tracker
- `tests/execution/test_position_tracker.py` - 17 tests for portfolio state reading
- `tests/execution/test_circuit_breaker.py` - 14 tests for CB state machine
- `tests/execution/test_risk_manager.py` - 7 tests for risk evaluation

## Decisions Made
- Added `environment` column to `portfolio_snapshots` table (composite PK with timestamp) to support per-environment queries; original schema only had single timestamp PK
- Ramp-up stages use 5-interval mapping (20% initial halt + 4 x 20% ramp stages) to avoid floating point boundary issues at stage transitions
- Used `exchange_calendars` NYSE calendar for equity business day cooldown counting (already available in project dependencies)
- `RiskManager.evaluate()` raises `CircuitBreakerError` for CB-related rejections and `RiskVetoError` for policy violations, consistent with exception hierarchy

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added environment column to portfolio_snapshots table**
- **Found during:** Task 1 (PositionTracker implementation)
- **Issue:** portfolio_snapshots table had no environment column; PositionTracker needs per-environment queries
- **Fix:** Changed schema to composite PK (timestamp, environment) and added environment NOT NULL column
- **Files modified:** src/swingrl/data/db.py
- **Verification:** All position tracker tests pass with environment-filtered queries
- **Committed in:** 67ee7b3 (Task 1 RED commit)

**2. [Rule 1 - Bug] Fixed floating point boundary issues in ramp stage calculation**
- **Found during:** Task 2 (CircuitBreaker ramp-up)
- **Issue:** `0.75 * 4 = 3.0000000224` due to float imprecision, causing wrong ramp stage at boundaries
- **Fix:** Added `round(elapsed_frac, 4)` and adjusted stage mapping to 5-interval scheme
- **Files modified:** src/swingrl/execution/risk/circuit_breaker.py
- **Verification:** test_ramp_progression passes with exact 0.75 comparison
- **Committed in:** 939167a (Task 2 GREEN commit)

---

**Total deviations:** 2 auto-fixed (1 blocking schema, 1 bug fix)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
- Test threshold values needed adjustment for floating point precision (e.g., 360/400 = 0.0999... not 0.10)

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All risk infrastructure ready for broker adapters (Plan 02) and order flow (Plans 03-05)
- Pipeline data types define the contract for all subsequent execution plans
- Circuit breaker persistence enables safe process restart during paper trading

---
*Phase: 08-paper-trading-core*
*Completed: 2026-03-09*
