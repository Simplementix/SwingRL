---
phase: 16-crypto-stop-price-persistence
plan: 01
subsystem: execution
tags: [fill-processor, positions, stop-loss, take-profit, sqlite, tdd]

requires:
  - phase: 08-paper-trading
    provides: FillProcessor, pipeline, PositionSizer, SizedOrder type with stop/TP fields
  - phase: 12-schema-alignment-emergency-triggers
    provides: positions table stop_loss_price, take_profit_price, side columns (ALTER TABLE)

provides:
  - FillProcessor.process() accepts optional SizedOrder parameter to persist stop/TP/side
  - Buy fills with sized_order write non-NULL stop_loss_price, take_profit_price, side to positions
  - Partial sells carry forward existing stop/TP values (not overwrite with sell-side data)
  - Pipeline passes sized_order context through to FillProcessor at Stage 5 call site

affects:
  - 17-stop-polling (stop_polling._check_stop_levels() now receives real stop/TP prices for crypto)
  - Any phase that reads positions.stop_loss_price or positions.take_profit_price

tech-stack:
  added: []
  patterns:
    - sized_order=None optional parameter pattern for backward-compatible extension of FillProcessor
    - carry-forward pattern for sell-path: read existing stop/TP before INSERT OR REPLACE

key-files:
  created: []
  modified:
    - src/swingrl/execution/fill_processor.py
    - src/swingrl/execution/pipeline.py
    - tests/execution/test_fill_processor.py

key-decisions:
  - "SizedOrder imported at runtime (not TYPE_CHECKING) since it appears in the method signature"
  - "Sell path uses existing stop/TP from SELECT result, not sized_order -- sell-side sized_order would carry sell-side stops, not position buy-side stops"
  - "second buy updates stop/TP to new sized_order values (not averaged) -- reflects new risk assessment"

patterns-established:
  - "Carry-forward on partial sell: SELECT stop/TP before INSERT OR REPLACE on sell path"
  - "Optional SizedOrder threading: process(fill, sized_order=None) -> _update_position(fill, sized_order)"

requirements-completed: [PAPER-07, PAPER-10]

duration: 12min
completed: 2026-03-10
---

# Phase 16 Plan 01: Crypto Stop Price Persistence Summary

**FillProcessor now threads SizedOrder stop/TP prices into the positions table on buy fills, and carries them forward on partial sells, closing the data flow gap that left stop_loss_price and take_profit_price always NULL for crypto positions.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-10T21:40:00Z
- **Completed:** 2026-03-10T21:52:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `FillProcessor.process()` now accepts `sized_order: SizedOrder | None = None`; threads it to `_update_position()`
- Buy fills with `sized_order` write `stop_loss_price`, `take_profit_price`, and `side` to the positions table
- Partial sells carry forward existing stop/TP values from the SELECT result (not overwritten)
- `ExecutionPipeline.execute_cycle()` Stage 5 call updated: `process(fill, sized_order=sized_order)`
- 6 new `TestStopTPPersistence` tests validate all persistence behaviors (buy, backward compat, second buy, partial sell, full sell, side column)
- All 854 tests pass; no regressions

## Task Commits

1. **RED (failing tests)** - `a8afcd7` (test): add failing TestStopTPPersistence class
2. **GREEN (FillProcessor)** - `1412bd5` (feat): persist stop/TP prices in FillProcessor._update_position()
3. **Task 2: Pipeline wire-up** - `7b3e371` (feat): wire pipeline to pass sized_order to fill_processor

## Files Created/Modified

- `src/swingrl/execution/fill_processor.py` - SizedOrder runtime import; updated process() and _update_position() signatures; expanded INSERT/INSERT OR REPLACE to include 10 columns; carry-forward on partial sell
- `src/swingrl/execution/pipeline.py` - Stage 5 call updated to pass sized_order=sized_order
- `tests/execution/test_fill_processor.py` - Added TestStopTPPersistence with 6 tests

## Decisions Made

- `SizedOrder` imported at runtime (not `TYPE_CHECKING`) because it appears in the method signature and is needed at call time.
- On the sell path, stop/TP is read from the existing position row via `SELECT`, not from `sized_order`. Using sell-side `sized_order` stop prices would overwrite buy-side stops with incorrect values.
- Second buy updates stop/TP to the new `sized_order` values (replaces, does not average) — reflects the new risk assessment at the time of the additional buy.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The positions table already had `stop_loss_price`, `take_profit_price`, and `side` columns from Phase 12. The `SizedOrder` type was already defined in `types.py`. The only work was threading the parameter and expanding SQL column lists.

## Next Phase Readiness

- `stop_polling._check_stop_levels()` will now receive real non-NULL stop/TP prices for crypto positions created after this change
- Phase 17 (stop polling enforcement) can proceed with confidence that the data flow gap is closed
- Equity fills without `sized_order` still produce NULL stop/TP (backward compatible — Alpaca handles stops at the exchange level)

---
*Phase: 16-crypto-stop-price-persistence*
*Completed: 2026-03-10*

## Self-Check: PASSED

- fill_processor.py: FOUND
- pipeline.py: FOUND
- test_fill_processor.py: FOUND
- 16-01-SUMMARY.md: FOUND
- Commit a8afcd7 (RED tests): FOUND
- Commit 1412bd5 (GREEN implementation): FOUND
- Commit 7b3e371 (pipeline wire-up): FOUND
