---
phase: 03-data-ingestion
plan: 04
subsystem: data
tags: [fred, fredapi, alfred, macro, vintage-data, parquet, cli]

# Dependency graph
requires:
  - phase: 03-data-ingestion
    provides: BaseIngestor ABC, DataValidator 12-step checklist, ParquetStore
provides:
  - FREDIngestor with daily and ALFRED vintage series fetching for 5 Tier 1 macro series
  - CLI entry point via python -m swingrl.data.fred with --series and --backfill flags
  - FRED-specific validation (null/dedup/gap/stale, no OHLCV checks)
affects: [05-features, 06-envs]

# Tech tracking
tech-stack:
  added: []
  patterns: [FRED-specific validation bypassing OHLCV steps, ALFRED vintage tracking for revised series, lambda-based retry with exponential backoff]

key-files:
  created:
    - src/swingrl/data/fred.py
    - tests/data/test_fred_ingestor.py
    - tests/data/fixtures/fred_cpiaucsl_releases.json
    - tests/data/fixtures/fred_vixcls_series.json
  modified:
    - src/swingrl/data/__init__.py

key-decisions:
  - "FRED validate() implements custom null-only row check instead of delegating to validate_rows() which assumes OHLCV columns"
  - "Staleness check remains in validate_batch() (35-day threshold for FRED monthly series)"
  - "run_all() uses BaseIngestor.run() per series with try/except for partial failure resilience"

patterns-established:
  - "FRED-specific validation: null value check (step 1) + batch-level dedup/gap/stale (steps 8-10), skipping OHLCV steps 2-7"
  - "ALFRED vintage tracking: realtime_start renamed to vintage_date for revised series (CPI, UNRATE)"
  - "Callable retry pattern: lambda wrapping fredapi calls with exponential backoff"

requirements-completed: [DATA-04]

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 3 Plan 04: FRED Macro Ingestor Summary

**FREDIngestor fetching 5 Tier 1 macro series with ALFRED vintage tracking for CPI/UNRATE and daily fetch for VIX/yield-curve/Fed-Funds**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T19:59:35Z
- **Completed:** 2026-03-06T20:07:28Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments
- FREDIngestor inheriting BaseIngestor with all abstract methods implemented
- ALFRED vintage data for CPIAUCSL and UNRATE storing vintage_date alongside observation_date (prevents look-ahead bias)
- Daily series (VIXCLS, T10Y2Y, DFF) fetched without vintage tracking
- Incremental mode reads max date from existing Parquet, backfill forces 2016-01-01 start
- CLI via `python -m swingrl.data.fred` with --series and --backfill flags
- 12 passing tests with mocked fredapi

## Task Commits

Each task was committed atomically (TDD flow):

1. **Task 1 RED: Failing tests for FREDIngestor** - `5700677` (test)
2. **Task 1 GREEN: Implement FREDIngestor** - `0a4aab1` (feat)

## Files Created/Modified
- `src/swingrl/data/fred.py` - FREDIngestor with daily/vintage fetching, retry, CLI, FRED-specific validation
- `tests/data/test_fred_ingestor.py` - 12 unit tests covering fetch, validate, store, run_all, retry, CLI
- `tests/data/fixtures/fred_cpiaucsl_releases.json` - Mock ALFRED all-releases response (20 entries with vintage dates)
- `tests/data/fixtures/fred_vixcls_series.json` - Mock daily VIX series response (20 entries)
- `src/swingrl/data/__init__.py` - Added FREDIngestor to public API exports

## Decisions Made
- FRED validate() uses custom null-only row-level check instead of delegating to DataValidator.validate_rows() which hardcodes OHLCV column assumptions (open/high/low/close/volume). Batch-level checks (dedup, gap detection, staleness) still delegate to DataValidator.validate_batch().
- Test fixtures use 2024-era dates with staleness check monkeypatched out, keeping fixture data stable and deterministic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FRED validate() cannot use DataValidator.validate_rows()**
- **Found during:** Task 1 GREEN (run_all tests failing)
- **Issue:** DataValidator.validate_rows() hardcodes OHLCV columns (open, high, low, close, volume). FRED data only has value + optional vintage_date, causing KeyError.
- **Fix:** Implemented FRED-specific validate() that does null value check inline (step 1) then delegates only batch-level checks to DataValidator.validate_batch() (steps 8-10).
- **Files modified:** src/swingrl/data/fred.py
- **Verification:** All 12 tests pass, validation test confirms delegation works
- **Committed in:** 0a4aab1 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary fix for correctness. DataValidator was designed for OHLCV data; FRED data has a fundamentally different schema. No scope creep.

## Issues Encountered
- Binance ingestor test (from Plan 03) hangs during full `tests/data/` suite run. Pre-existing issue, not related to this plan. FRED and all other data tests pass (43 total).

## User Setup Required
None - FRED_API_KEY environment variable is required for production use but all tests use mocked fredapi.

## Next Phase Readiness
- All 3 concrete ingestors complete (Alpaca, Binance, FRED)
- Phase 3 data ingestion layer fully implemented
- Ready for Phase 4 (Storage) or Phase 5 (Features)

---
*Phase: 03-data-ingestion*
*Completed: 2026-03-06*
