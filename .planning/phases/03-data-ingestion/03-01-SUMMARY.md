---
phase: 03-data-ingestion
plan: 01
subsystem: data
tags: [parquet, validation, exchange-calendars, pyarrow, abc, data-pipeline]

# Dependency graph
requires:
  - phase: 02-developer-experience
    provides: SwingRLConfig schema, DataError exception, structlog logging, conftest fixtures
provides:
  - BaseIngestor ABC with fetch/validate/store/run protocol
  - DataValidator 12-step checklist (row-level quarantine + batch-level checks)
  - ParquetStore with atomic read/upsert/write
  - Defect sample fixture library for validation testing
affects: [03-02-alpaca, 03-03-binance, 03-04-fred, 04-storage]

# Tech tracking
tech-stack:
  added: [fredapi, exchange-calendars, pyarrow, responses]
  patterns: [BaseIngestor ABC contract, vectorized pandas validation, atomic Parquet upsert, defect fixture factories]

key-files:
  created:
    - src/swingrl/data/base.py
    - src/swingrl/data/validation.py
    - src/swingrl/data/parquet_store.py
    - tests/data/test_validation.py
    - tests/data/test_parquet_store.py
    - tests/data/fixtures/defect_samples.py
  modified:
    - pyproject.toml
    - src/swingrl/data/__init__.py
    - uv.lock

key-decisions:
  - "pyarrow added as explicit dep (not transitive via pandas as research assumed)"
  - "responses moved to dev dependency group (test-only library)"
  - "Staleness threshold: 4 calendar days equity, 8H crypto, 35 days FRED"
  - "Validation reasons stored as semicolon-delimited string in quarantine reason column"

patterns-established:
  - "BaseIngestor ABC: fetch->validate->store contract with quarantine side-channel"
  - "DataValidator source-aware: equity/crypto/fred literal switches gap detection and zero-volume logic"
  - "ParquetStore atomic write: write to .tmp then Path.replace()"
  - "Defect fixture pattern: one factory function per defect type, returns small DataFrame"

requirements-completed: [DATA-05]

# Metrics
duration: 8min
completed: 2026-03-06
---

# Phase 3 Plan 01: Data Ingestion Foundation Summary

**BaseIngestor ABC, 12-step DataValidator with NYSE-aware gap detection, and atomic ParquetStore upsert with snappy compression**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-06T19:48:17Z
- **Completed:** 2026-03-06T19:56:30Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- BaseIngestor ABC defining fetch/validate/store contract for all three concrete ingestors
- 12-step DataValidator: 7 row-level checks (null, price>0, volume>=0, OHLC ordering, bounds tolerance, price spike >50%, zero-volume equity) + 5 batch-level checks (dedup, NYSE/crypto gap detection, staleness, row count, cross-source placeholder)
- ParquetStore with atomic upsert (read-merge-dedup-write) using snappy compression
- 19 passing tests with reusable defect fixture factory library

## Task Commits

Each task was committed atomically:

1. **Task 1: Install deps, BaseIngestor ABC, ParquetStore, DataValidator** - `b071a5f` (feat)
2. **Task 2: Comprehensive test suite with defect fixtures** - `cb31da0` (test)

## Files Created/Modified
- `src/swingrl/data/base.py` - BaseIngestor ABC with fetch/validate/store/run protocol
- `src/swingrl/data/validation.py` - DataValidator with 12-step checklist, exchange_calendars NYSE integration
- `src/swingrl/data/parquet_store.py` - ParquetStore with atomic read/upsert/write helpers
- `src/swingrl/data/__init__.py` - Public re-exports: BaseIngestor, DataValidator, ParquetStore
- `tests/data/test_validation.py` - 15 validation tests (9 row-level, 6 batch-level)
- `tests/data/test_parquet_store.py` - 4 parquet store tests
- `tests/data/fixtures/defect_samples.py` - 8 defect factory functions for test data
- `pyproject.toml` - Phase 3 deps, mypy overrides, integration marker
- `uv.lock` - Updated lockfile

## Decisions Made
- pyarrow added as explicit dependency (research assumed transitive via pandas, but it was not installed)
- responses moved to dev dependency group since it is test-only
- Staleness thresholds set to 4 calendar days for equity (covers weekends), 8H for crypto, 35 days for FRED monthly series
- Quarantine reasons stored as semicolon-delimited strings in a "reason" column for multiple-failure transparency

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added pyarrow as explicit dependency**
- **Found during:** Task 1 (ParquetStore tests)
- **Issue:** pandas.to_parquet() requires pyarrow or fastparquet; neither was installed despite research claiming pyarrow was transitive via pandas
- **Fix:** `uv add "pyarrow>=14.0"`
- **Files modified:** pyproject.toml, uv.lock
- **Verification:** All parquet tests pass
- **Committed in:** b071a5f (Task 1 commit)

**2. [Rule 1 - Bug] Fixed test data freshness for staleness checks**
- **Found during:** Task 1 (batch-level tests)
- **Issue:** Test helper functions used hardcoded 2024 dates, triggering the staleness check (Step 10) unintentionally
- **Fix:** Changed helpers to generate data ending near "today" using pd.Timestamp.now()
- **Files modified:** tests/data/test_validation.py
- **Verification:** All batch tests pass without false DataError
- **Committed in:** b071a5f (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
- conftest.py equity_ohlcv fixture generates data that violates OHLC bounds (open can fall outside low/high due to independent random generation) — used dedicated clean data helpers instead of conftest fixture for "valid rows pass" test

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BaseIngestor, DataValidator, ParquetStore ready for concrete ingestor implementations
- Plan 02 (AlpacaIngestor) can import and use all three modules directly
- exchange_calendars and pyarrow installed and verified

---
*Phase: 03-data-ingestion*
*Completed: 2026-03-06*
