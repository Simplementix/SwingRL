---
phase: 12-schema-alignment-and-emergency-triggers
plan: 01
subsystem: database, execution
tags: [sqlite, duckdb, emergency-stop, schema-migration, stop-polling]

# Dependency graph
requires:
  - phase: 08-paper-trading
    provides: "positions table, portfolio_snapshots table, emergency.py, pipeline.py"
  - phase: 09-automation
    provides: "stop_polling.py daemon thread, scheduler infrastructure"
provides:
  - "inference_outcomes and api_errors SQLite tables for automated trigger tracking"
  - "Corrected stop_polling.py querying positions table"
  - "Working VIX+CB, NaN inference, and IP ban emergency triggers against real schema"
  - "NaN observation tracking wired into ExecutionPipeline.execute_cycle"
  - "API error tracking wired into BinanceSimAdapter price fetch failure path"
affects: [production-hardening, monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Idempotent ALTER TABLE with try/except for column additions"
    - "Independent DB connections per trigger (no shared outer context manager)"
    - "DuckDB tuple indexing vs SQLite dict-key access pattern for mixed queries"

key-files:
  created: []
  modified:
    - src/swingrl/data/db.py
    - src/swingrl/scheduler/stop_polling.py
    - src/swingrl/execution/emergency.py
    - src/swingrl/execution/pipeline.py
    - src/swingrl/execution/adapters/binance_sim.py
    - tests/test_emergency_stop.py
    - tests/scheduler/test_stop_polling.py
    - tests/data/test_db.py

key-decisions:
  - "_CB_DRAWDOWN_TRIGGER changed from -13.0 to 0.13 (positive fraction matching drawdown_pct storage)"
  - "Each trigger manages its own DB connection independently for fault isolation"
  - "MAX(drawdown_pct) across all environments for worst-case drawdown in VIX+CB trigger"

patterns-established:
  - "Idempotent ALTER TABLE: try/except sqlite3.OperationalError for column additions"
  - "Mixed DuckDB/SQLite triggers: tuple indexing for DuckDB, dict keys for SQLite"

requirements-completed: [PAPER-10, PROD-07]

# Metrics
duration: 6min
completed: 2026-03-10
---

# Phase 12 Plan 01: Schema Alignment and Emergency Triggers Summary

**Fixed stop_polling table reference and rewrote 3 emergency triggers to query real schema (macro_features, portfolio_snapshots, inference_outcomes, api_errors)**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-10T16:34:23Z
- **Completed:** 2026-03-10T16:40:30Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Added inference_outcomes and api_errors SQLite tables for automated trigger data population
- Added stop_loss_price, take_profit_price, side columns to positions table (idempotent)
- Fixed stop_polling.py to query `positions` instead of non-existent `position_tracker`
- Rewrote all 3 emergency triggers: VIX from DuckDB macro_features, drawdown from SQLite portfolio_snapshots, NaN from inference_outcomes, IP ban from api_errors
- Wired NaN observation tracking into ExecutionPipeline.execute_cycle
- Wired API error tracking into BinanceSimAdapter price fetch failure path

## Task Commits

Each task was committed atomically:

1. **Task 1: Schema migrations and stop_polling fix** - `2bb46e2` (feat)
2. **Task 2: Emergency trigger rewrites and NaN/API-error population** - `482c8ba` (feat)

## Files Created/Modified
- `src/swingrl/data/db.py` - Added inference_outcomes, api_errors tables and positions ALTER columns
- `src/swingrl/scheduler/stop_polling.py` - Fixed FROM position_tracker to FROM positions
- `src/swingrl/execution/emergency.py` - Rewrote check_automated_triggers with real data sources
- `src/swingrl/execution/pipeline.py` - Added NaN observation tracking INSERT
- `src/swingrl/execution/adapters/binance_sim.py` - Added api_errors INSERT on price fetch failure
- `tests/test_emergency_stop.py` - Rewrote trigger tests for new query patterns (23 tests)
- `tests/scheduler/test_stop_polling.py` - Added table reference and stop level tests (6 tests)
- `tests/data/test_db.py` - Added schema migration tests (6 tests)

## Decisions Made
- _CB_DRAWDOWN_TRIGGER changed from -13.0 to 0.13 (positive fraction matching how drawdown_pct is stored in portfolio_snapshots)
- Each trigger manages its own DB connection independently -- prevents one trigger failure from blocking others
- Used MAX(drawdown_pct) across all environments for worst-case drawdown detection in VIX+CB trigger

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- INT-03 and INT-04 integration gaps closed
- All emergency triggers query real data sources
- Full test suite green (811 tests)

---
*Phase: 12-schema-alignment-and-emergency-triggers*
*Completed: 2026-03-10*
