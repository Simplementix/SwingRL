---
phase: 03-data-ingestion
plan: 03
subsystem: data
tags: [binance, crypto, ohlcv, rate-limiting, backfill, archive, parquet]

# Dependency graph
requires:
  - phase: 03-data-ingestion
    provides: BaseIngestor ABC, DataValidator, ParquetStore (Plan 01)
provides:
  - BinanceIngestor with 4H klines fetch from api.binance.us
  - Rate limit monitoring via X-MBX-USED-WEIGHT-1M header
  - Historical backfill from data.binance.vision archives (2017-2019)
  - Archive/API stitch validation with 0.5% deviation threshold
  - Automatic ms/us timestamp detection for archive CSVs
  - CLI with --backfill, --symbols, --days flags
affects: [04-storage, 06-environments, 07-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [Binance klines pagination, rate limit throttling, archive timestamp detection, stitch validation]

key-files:
  created:
    - src/swingrl/data/binance.py
    - tests/data/test_binance_ingestor.py
    - tests/data/fixtures/binance_klines_btcusdt.json
    - tests/data/fixtures/binance_archive_btcusdt_4h.csv
  modified:
    - src/swingrl/data/__init__.py

key-decisions:
  - "All klines fetched from api.binance.us (not api.binance.com) per Binance.US broker architecture"
  - "Stitch point at 2019-09-01 (Binance.US launch) separates archive from API data"
  - "Microsecond threshold at 2_000_000_000_000 to detect 2025+ archive timestamp format"

patterns-established:
  - "Pagination loop: advance startTime by last_open_time + FOUR_HOURS_MS until empty response"
  - "Rate limit monitoring: check X-MBX-USED-WEIGHT-1M header, sleep(60) at 80% of 1200 limit"
  - "Archive CSV parsing: per-row timestamp unit detection for ms vs us formats"
  - "Volume normalization: quote_asset_volume (col 7) with fallback to volume_base * close"
  - "Stitch validation: compare close prices in 30-day overlap window, warn if >0.5% deviation"

requirements-completed: [DATA-02, DATA-03]

# Metrics
duration: 13min
completed: 2026-03-06
---

# Phase 3 Plan 03: Binance Crypto Ingestor Summary

**BinanceIngestor fetching 4H klines from api.binance.us with rate limit monitoring, archive backfill with ms/us timestamp detection, and 30-day stitch validation**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-06T19:59:43Z
- **Completed:** 2026-03-06T20:12:39Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- BinanceIngestor implementing full BaseIngestor protocol (fetch/validate/store) for crypto 4H OHLCV
- Rate limit monitoring with proactive throttling at 80% of 1200 weight/minute
- Historical backfill from data.binance.vision archives with automatic ms/us timestamp detection
- 30-day stitch validation confirming <0.5% price deviation between archive and API data
- 18 passing unit tests covering all live-fetch and backfill behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1: BinanceIngestor with 4H fetch, rate limiting, and klines parsing** - `c85fd4f` (feat)
2. **Task 2: Historical backfill with archive parsing and stitch validation** - `77bb46f` (feat)

## Files Created/Modified
- `src/swingrl/data/binance.py` - BinanceIngestor with fetch, backfill, rate limiting, archive parsing, CLI
- `tests/data/test_binance_ingestor.py` - 18 unit tests covering live-fetch and backfill behaviors
- `tests/data/fixtures/binance_klines_btcusdt.json` - 10-entry mock klines API response
- `tests/data/fixtures/binance_archive_btcusdt_4h.csv` - 80-row archive fixture (50 ms + 30 us timestamps)
- `src/swingrl/data/__init__.py` - Added BinanceIngestor to public exports

## Decisions Made
- All klines fetched from api.binance.us (not api.binance.com) per the Binance.US broker architecture
- Stitch point set to 2019-09-01 (Binance.US launch date) as the boundary between archive and API data
- Microsecond threshold at 2_000_000_000_000 to detect 2025+ archive timestamp format per research findings

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed pagination in tests — missing empty-response termination**
- **Found during:** Task 1 (test_fetch_returns_4h_ohlcv)
- **Issue:** Tests only registered one responses mock for klines, but fetch() loops until empty response. The retry logic caught the ConnectionError from unregistered responses and hung waiting on real sleep()
- **Fix:** Added `_add_klines_then_empty()` helper that registers both the data response and an empty termination response for all fetch tests
- **Files modified:** tests/data/test_binance_ingestor.py
- **Verification:** All 9 Task 1 tests pass without hanging
- **Committed in:** c85fd4f (Task 1 commit)

**2. [Rule 1 - Bug] Fixed mypy strict type errors on generic list parameters**
- **Found during:** Task 2 (mypy verification)
- **Issue:** `list[list]` without inner type parameters fails mypy --strict with [type-arg] error
- **Fix:** Changed all occurrences to `list[list[str | int]]` to match the klines data structure
- **Files modified:** src/swingrl/data/binance.py
- **Verification:** `mypy --strict` passes with zero errors
- **Committed in:** 77bb46f (Task 2 commit)

**3. [Rule 1 - Bug] Fixed timezone warning in _download_archives**
- **Found during:** Task 2 (test_backfill_full_range)
- **Issue:** `pd.Timestamp.to_period("M")` emits UserWarning when called on tz-aware timestamps
- **Fix:** Added `.tz_localize(None)` before `.to_period("M")` conversion
- **Files modified:** src/swingrl/data/binance.py
- **Verification:** No UserWarning in test output
- **Committed in:** 77bb46f (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 bugs)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- `responses` library version installed does not have `add_passthrough()` method — used explicit URL mocking per month instead

## User Setup Required
None - no external service configuration required (API keys not needed for unit tests, which use mocked HTTP).

## Next Phase Readiness
- BinanceIngestor ready for integration with storage layer (Phase 4)
- CLI supports manual backfill: `python -m swingrl.data.binance --backfill --symbols BTCUSDT ETHUSDT`
- All three data ingestors (Alpaca, Binance, FRED) now implemented

---
*Phase: 03-data-ingestion*
*Completed: 2026-03-06*
