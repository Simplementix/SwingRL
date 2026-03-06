---
phase: 03-data-ingestion
plan: 02
subsystem: data
tags: [alpaca, equity, ohlcv, iex, parquet, cli, retry]

# Dependency graph
requires:
  - phase: 03-data-ingestion plan 01
    provides: BaseIngestor ABC, DataValidator, ParquetStore
provides:
  - AlpacaIngestor concrete implementation for equity daily OHLCV
  - CLI entrypoint for manual and automated equity data ingestion
  - Incremental and 10-year backfill fetch modes
affects: [03-data-ingestion plan 03, 03-data-ingestion plan 04, 04-storage, 09-automation]

# Tech tracking
tech-stack:
  added: [alpaca-py BarSet/StockBarsRequest, DataFeed.IEX, Adjustment.ALL]
  patterns: [SDK client mocking with MagicMock, exponential backoff retry, MultiIndex flattening]

key-files:
  created:
    - src/swingrl/data/alpaca.py
    - tests/data/test_alpaca_ingestor.py
    - tests/data/fixtures/alpaca_bars_spy.json
  modified:
    - src/swingrl/data/__init__.py

key-decisions:
  - "Mock SDK client directly rather than HTTP responses -- cleaner tests for alpaca-py BarSet interface"
  - "cast(BarSet, ...) for strict mypy since get_stock_bars returns Union[BarSet, dict]"
  - "Staleness check patched in run_all test -- fixture data is intentionally old, staleness is tested elsewhere"

patterns-established:
  - "Ingestor TDD pattern: mock SDK client, verify request params, check DataFrame output"
  - "run_all() partial failure: try/except per symbol, collect failures, return list"

requirements-completed: [DATA-01]

# Metrics
duration: 5min
completed: 2026-03-06
---

# Phase 3 Plan 02: Alpaca Equity Ingestor Summary

**AlpacaIngestor with IEX feed, split+dividend adjustment, 3-retry backoff, incremental/backfill modes, and CLI entrypoint**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-06T19:59:26Z
- **Completed:** 2026-03-06T20:04:37Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments
- AlpacaIngestor inherits BaseIngestor, implements fetch/validate/store for 8 equity ETFs
- IEX feed with Adjustment.ALL (split+dividend adjusted) daily bars
- Incremental mode derives start from max(timestamp) in existing Parquet + 1 day
- Backfill mode fetches from 2016-01-01 (10-year history covering multiple market regimes)
- 3-retry exponential backoff (2s, 4s, 8s) raising DataError on exhaustion
- CLI with --symbols, --days, --backfill flags via argparse
- 12 unit tests passing with mocked SDK client, strict mypy clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AlpacaIngestor with fetch, validate, store, and CLI (TDD)**
   - RED: `bc31ec5` (test: add failing tests for AlpacaIngestor)
   - GREEN: `b370604` (feat: implement AlpacaIngestor with IEX feed, retry, CLI)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `src/swingrl/data/alpaca.py` - AlpacaIngestor concrete implementation + CLI main()
- `tests/data/test_alpaca_ingestor.py` - 12 unit tests with mocked SDK client
- `tests/data/fixtures/alpaca_bars_spy.json` - 5-bar SPY daily mock fixture
- `src/swingrl/data/__init__.py` - Added AlpacaIngestor to package exports

## Decisions Made
- Mock SDK client directly (MagicMock on StockHistoricalDataClient.get_stock_bars) rather than HTTP responses -- alpaca-py handles HTTP internally, mocking at SDK level is cleaner and more stable
- Used `cast(BarSet, ...)` for strict mypy compliance since `get_stock_bars` returns `Union[BarSet, dict[str, Any]]` but we never use raw_data mode
- Patched `DataValidator._check_staleness` in run_all partial failure test because fixture data is intentionally from 2024 (staleness validation is covered by dedicated tests in test_validation.py)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed staleness check interfering with run_all test**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** run_all test used fixture data from 2024-01 which triggered DataValidator staleness check, causing SPY and QQQ to appear as failed symbols
- **Fix:** Patched `DataValidator._check_staleness` in the run_all test since staleness is tested separately
- **Files modified:** tests/data/test_alpaca_ingestor.py
- **Verification:** All 12 tests pass
- **Committed in:** b370604

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minimal -- test fixture adjustment to avoid false failure from staleness check on old test data.

## Issues Encountered
None beyond the staleness fixture issue documented above.

## User Setup Required
None - no external service configuration required. API keys are read from environment variables at runtime.

## Next Phase Readiness
- AlpacaIngestor ready for integration with Phase 9 automation scheduler
- Parquet output at data/equity/{SYMBOL}_daily.parquet ready for Phase 4 DuckDB ingestion
- Pattern established for BinanceIngestor (Plan 03) to follow same ABC contract

---
*Phase: 03-data-ingestion*
*Completed: 2026-03-06*
