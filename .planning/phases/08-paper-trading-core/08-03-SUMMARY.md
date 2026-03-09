---
phase: 08-paper-trading-core
plan: 03
subsystem: execution
tags: [alpaca-py, binance-us, bracket-orders, fill-processing, sqlite, protocol]

# Dependency graph
requires:
  - phase: 08-paper-trading-core
    provides: "Pipeline types (TradeSignal, SizedOrder, ValidatedOrder, FillResult)"
provides:
  - "ExchangeAdapter Protocol (runtime_checkable broker abstraction)"
  - "AlpacaAdapter with bracket orders and notional fractional shares"
  - "BinanceSimAdapter with order book mid-price and slippage simulation"
  - "FillProcessor with trade recording and position management"
affects: [08-paper-trading-core, 09-automation]

# Tech tracking
tech-stack:
  added: [alpaca-py TradingClient, requests (Binance.US REST)]
  patterns: [Protocol-based adapter abstraction, exponential backoff retry, weighted-average cost basis]

key-files:
  created:
    - src/swingrl/execution/adapters/base.py
    - src/swingrl/execution/adapters/alpaca_adapter.py
    - src/swingrl/execution/adapters/binance_sim.py
    - src/swingrl/execution/fill_processor.py
    - src/swingrl/execution/adapters/__init__.py
    - tests/execution/test_alpaca_adapter.py
    - tests/execution/test_binance_sim.py
    - tests/execution/test_fill_processor.py
  modified: []

key-decisions:
  - "ValidatedOrder.order field name (not sized_order) -- matched existing frozen dataclass in types.py"
  - "broker Literal 'binance_us' (not 'binance_sim') -- matches types.py Literal constraint"
  - "Binance params use string '5' for limit to satisfy requests type signature"

patterns-established:
  - "ExchangeAdapter Protocol: 4-method interface (submit_order, get_positions, cancel_order, get_current_price)"
  - "Retry pattern: _retry(fn, max_attempts=3) with exponential backoff (1s, 2s, 4s) and alerter notification on final failure"
  - "FillProcessor position lifecycle: create -> cost basis average -> reduce -> delete on zero"

requirements-completed: [PAPER-01, PAPER-02, PAPER-10]

# Metrics
duration: 7min
completed: 2026-03-09
---

# Phase 08 Plan 03: Exchange Adapters and Fill Processor Summary

**Protocol-based broker abstraction with Alpaca bracket orders (notional/fractional), Binance.US simulated fills (mid-price + slippage + commission), and SQLite position tracking with weighted-average cost basis**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-09T04:05:24Z
- **Completed:** 2026-03-09T04:12:40Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- ExchangeAdapter runtime_checkable Protocol providing clean broker abstraction for the execution pipeline
- AlpacaAdapter submits bracket orders via alpaca-py with notional amounts for fractional share support, 3-retry exponential backoff, and Discord critical alert on failure
- BinanceSimAdapter fetches real order book from Binance.US, computes mid-price, applies 0.03% slippage and 0.10% commission per side, with wide spread (>0.5%) warning
- FillProcessor records all fills to SQLite trades table and maintains positions with weighted-average cost basis, sell-to-zero deletion, and reconciliation adjustment support
- 20 tests passing with fully mocked broker APIs (no real API calls)

## Task Commits

Each task was committed atomically:

1. **Task 1: ExchangeAdapter Protocol and AlpacaAdapter** - `8b95057` (feat)
   - RED test: included in commit (TDD)
2. **Task 2: BinanceSimAdapter and FillProcessor** - `71b3c3f` (feat)
   - RED test: `6ed06d2` (test)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `src/swingrl/execution/adapters/base.py` - ExchangeAdapter Protocol with 4 methods
- `src/swingrl/execution/adapters/alpaca_adapter.py` - Alpaca paper trading with bracket orders and retry
- `src/swingrl/execution/adapters/binance_sim.py` - Binance.US simulated fills with order book mid-price
- `src/swingrl/execution/fill_processor.py` - Trade recording and position management in SQLite
- `src/swingrl/execution/adapters/__init__.py` - Package exports
- `tests/execution/test_alpaca_adapter.py` - 6 tests: Protocol, bracket, notional, retry
- `tests/execution/test_binance_sim.py` - 7 tests: Protocol, mid-price, slippage, commission, spread, retry
- `tests/execution/test_fill_processor.py` - 7 tests: trade recording, position lifecycle

## Decisions Made
- Used `ValidatedOrder.order` field (not `sized_order`) to match the existing frozen dataclass in `types.py` that was created by Plan 01 execution
- Broker literal `"binance_us"` (not `"binance_sim"`) to conform to the `Literal["alpaca", "binance_us"]` constraint in `FillResult`
- Binance API limit param as string `"5"` to satisfy `requests.get()` type signature for mypy

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted to existing types.py frozen dataclass field names**
- **Found during:** Task 1 and Task 2
- **Issue:** Plan specified `ValidatedOrder.sized_order` and `broker="binance_sim"` but the existing `types.py` (from Plan 01) uses `ValidatedOrder.order` and `Literal["alpaca", "binance_us"]`
- **Fix:** Updated all adapter code and tests to use `.order` and `"binance_us"`
- **Files modified:** All adapter and test files
- **Verification:** All 20 tests pass
- **Committed in:** 8b95057, 71b3c3f

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary adaptation to match existing codebase types. No scope creep.

## Issues Encountered
None beyond the field name adaptation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Exchange adapters ready for integration into the execution pipeline (Plan 04/05)
- FillProcessor can be wired into the pipeline after adapter fills
- Both adapters satisfy ExchangeAdapter Protocol for pipeline-agnostic usage

---
*Phase: 08-paper-trading-core*
*Completed: 2026-03-09*
