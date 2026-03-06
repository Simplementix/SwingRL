---
phase: 04-data-storage-and-validation
plan: 04
subsystem: database
tags: [yfinance, cross-source, corporate-actions, duckdb, sqlite, validation]

# Dependency graph
requires:
  - phase: 04-01
    provides: "DatabaseManager with DuckDB/SQLite dual-database layer and corporate_actions table"
provides:
  - "CrossSourceValidator comparing Alpaca vs yfinance equity closing prices"
  - "CorporateActionDetector with overnight spike heuristic and action recording"
  - "DataValidator Step 12 cross-source consistency check (no longer deferred)"
affects: [05-features, 06-envs, 08-paper-trading]

# Tech tracking
tech-stack:
  added: [yfinance]
  patterns: [cross-source-validation, corporate-action-detection, false-positive-suppression]

key-files:
  created:
    - src/swingrl/data/cross_source.py
    - src/swingrl/data/corporate_actions.py
    - tests/data/test_cross_source.py
    - tests/data/test_corporate_actions.py
  modified:
    - src/swingrl/data/validation.py
    - pyproject.toml

key-decisions:
  - "yfinance Adj Close used as reference price for cross-source comparison"
  - "as_of_date parameter added to validate_prices for testability"
  - "Step 12 cross-source check is warning-only, never quarantines"
  - "Corporate action thresholds: 30% equity, 40% crypto"

patterns-established:
  - "Cross-source validation: compare DuckDB prices against external source with tolerance threshold"
  - "Corporate action suppression: check corporate_actions table before flagging price spikes"

requirements-completed: [DATA-10, DATA-11]

# Metrics
duration: 13min
completed: 2026-03-06
---

# Phase 4 Plan 04: Cross-Source Validation and Corporate Actions Summary

**CrossSourceValidator comparing Alpaca vs yfinance with $0.05 tolerance, CorporateActionDetector with 30%/40% spike heuristic, and DataValidator Step 12 implementation**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-06T23:23:48Z
- **Completed:** 2026-03-06T23:37:34Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- CrossSourceValidator compares DuckDB Alpaca prices with yfinance Adj Close for equity symbols
- CorporateActionDetector identifies overnight price spikes and records actions to SQLite
- DataValidator Step 12 implemented: runs cross-source check for equity when db available, skips gracefully when not
- Known corporate actions in corporate_actions table suppress false-positive quarantine

## Task Commits

Each task was committed atomically:

1. **Task 1: Cross-source validator (Alpaca vs yfinance)** - `425daa7` (feat)
2. **Task 2: Corporate action detector with split heuristic** - `e743984` (feat)

## Files Created/Modified
- `src/swingrl/data/cross_source.py` - CrossSourceValidator, CrossSourceResult dataclass, yfinance parsing
- `src/swingrl/data/corporate_actions.py` - CorporateActionDetector with spike detection, action recording, suppression
- `src/swingrl/data/validation.py` - Step 12 implementation with optional db/config params
- `tests/data/test_cross_source.py` - 8 tests: price matching, warnings, errors, defaults, auto_adjust, Step 12
- `tests/data/test_corporate_actions.py` - 12 tests: spike detection, thresholds, recording, querying, suppression
- `pyproject.toml` - Added yfinance dependency and mypy ignore

## Decisions Made
- Used yfinance Adj Close (not raw Close) as reference price for accurate split-adjusted comparison
- Added `as_of_date` parameter to `validate_prices()` for deterministic testing with historical dates
- DataValidator Step 12 uses try/except to gracefully handle cross-source check failures without blocking validation
- Corporate action check_and_suppress returns True only when BOTH spike threshold exceeded AND action is known

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed date range mismatch in cross-source tests**
- **Found during:** Task 1 (cross-source validator tests)
- **Issue:** Tests used 2024 dates but validate_prices() queried from date.today() (2026), finding no matching data
- **Fix:** Added `as_of_date` parameter to validate_prices() for explicit date range control in tests
- **Files modified:** src/swingrl/data/cross_source.py, tests/data/test_cross_source.py
- **Verification:** All 8 cross-source tests pass
- **Committed in:** 425daa7 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed staleness check in DataValidator Step 12 tests**
- **Found during:** Task 1 (DataValidator Step 12 integration test)
- **Issue:** Tests used 2024 dates which triggered Step 10 staleness DataError before reaching Step 12
- **Fix:** Used pd.Timestamp.now() for recent dates in Step 12 tests
- **Files modified:** tests/data/test_cross_source.py
- **Verification:** Both Step 12 tests pass without staleness errors
- **Committed in:** 425daa7 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correct test date handling. No scope creep.

## Issues Encountered
- Pre-existing test failure in test_ingestion_logging.py (from plan 04-02) — out of scope, not caused by this plan's changes

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Cross-source validation and corporate action detection complete
- Phase 4 data storage and validation pipeline fully implemented
- Ready for Phase 5 (feature engineering) which will consume validated data

## Self-Check: PASSED

All files verified present, all commits verified in git history.

---
*Phase: 04-data-storage-and-validation*
*Completed: 2026-03-06*
