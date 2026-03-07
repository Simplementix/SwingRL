---
phase: 05-feature-engineering
plan: 02
subsystem: features
tags: [yfinance, alpha-vantage, duckdb, asof-join, fundamentals, macro, z-score]

# Dependency graph
requires:
  - phase: 04-data-storage-and-validation
    provides: DuckDB macro_features table with FRED series and release_date, ohlcv_daily, ohlcv_4h tables
provides:
  - FundamentalFetcher class with yfinance + Alpha Vantage fallback for equity ETF fundamentals
  - MacroFeatureAligner class producing 6 look-ahead-bias-free macro features via ASOF JOIN
  - Sector-relative z-score computation for P/E ratio
  - DuckDB storage for fundamentals table
affects: [05-feature-engineering, 06-rl-environments, 07-training]

# Tech tracking
tech-stack:
  added: [alpha-vantage]
  patterns: [ASOF JOIN macro alignment, fundamental data fallback chain, sector-relative z-scoring]

key-files:
  created:
    - src/swingrl/features/fundamentals.py
    - src/swingrl/features/macro.py
    - tests/features/test_fundamentals.py
    - tests/features/test_macro.py
  modified:
    - pyproject.toml
    - uv.lock
    - .secrets.baseline

key-decisions:
  - "DuckDB ASOF JOIN on release_date prevents look-ahead bias for all 5 FRED macro series"
  - "Alpha Vantage FundamentalData is lazy-imported; fallback disabled if API key missing"
  - "Only P/E gets sector-relative z-score; other fundamentals remain as raw values per spec"
  - "Crypto macro lag periods scaled by 6x (4H bars vs daily) for equivalent time windows"

patterns-established:
  - "ASOF JOIN pattern: chained ASOF JOINs against macro_features with release_date for look-ahead-free alignment"
  - "Fundamental fallback chain: yfinance primary -> Alpha Vantage -> NaN defaults"
  - "DuckDB replacement scan for DataFrame-to-table sync (fundamentals table)"

requirements-completed: [FEAT-03, FEAT-04]

# Metrics
duration: 9min
completed: 2026-03-07
---

# Phase 5 Plan 2: Fundamentals and Macro Alignment Summary

**FundamentalFetcher with yfinance/AV fallback and MacroFeatureAligner producing 6 derived features via DuckDB ASOF JOIN**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-07T02:48:26Z
- **Completed:** 2026-03-07T02:58:14Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- FundamentalFetcher retrieves 4 metrics per ETF (P/E, earnings growth, D/E, dividend yield) with graceful missing-field handling
- Alpha Vantage fallback activates automatically when yfinance fails and API key is set
- MacroFeatureAligner produces 6 look-ahead-bias-free macro features for both equity and crypto environments
- ASOF JOIN on release_date ensures FRED data is only visible after publication date

## Task Commits

Each task was committed atomically:

1. **Task 1: Fundamental data fetcher** - `70eb325` (test) + `ca83790` (feat)
2. **Task 2: Macro feature aligner** - `d771861` (test) + `0bcaf47` (feat)

_Note: TDD tasks have separate RED (test) and GREEN (feat) commits._

## Files Created/Modified
- `src/swingrl/features/fundamentals.py` - FundamentalFetcher: fetch_symbol, validate, fetch_all, sector z-scores, store to DuckDB (305 lines)
- `src/swingrl/features/macro.py` - MacroFeatureAligner: ASOF JOIN alignment, 6 derived macro features, equity + crypto paths (302 lines)
- `tests/features/test_fundamentals.py` - 9 tests: fetching, validation, z-scores, DuckDB storage (296 lines)
- `tests/features/test_macro.py` - 10 tests: ASOF JOIN, derived macros, binary features, crypto alignment (286 lines)
- `pyproject.toml` - Added alpha-vantage dependency
- `uv.lock` - Updated lockfile
- `.secrets.baseline` - Updated for test API key patterns

## Decisions Made
- DuckDB ASOF JOIN on release_date prevents look-ahead bias for all 5 FRED macro series (VIXCLS, T10Y2Y, DFF, CPIAUCSL, UNRATE)
- Alpha Vantage FundamentalData is lazy-imported; fallback disabled gracefully if API key not set
- Only P/E ratio gets sector-relative z-score; earnings growth, D/E, and dividend yield remain raw per CONTEXT.md spec
- Crypto macro lag periods scaled by 6x for 4H bar equivalence (e.g., 90 days = 540 bars, 252 days = 1512 bars)
- Negative P/E and D/E values are set to NaN (not zero) during validation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-commit detect-secrets flagged test API key strings; resolved with `pragma: allowlist secret` inline comments
- Prior session had partially committed files (hmm_regime.py, schema.py, technical.py) on the branch; these were from 05-01 and 05-03 plans executed out-of-order

## User Setup Required

None - no external service configuration required. Alpha Vantage API key is optional (fallback disabled without it).

## Next Phase Readiness
- Fundamental and macro feature modules ready for observation vector assembly (Plan 05-05)
- 19/19 plan tests passing, 243/243 full suite green
- MacroFeatureAligner ready to feed both equity and crypto observation vectors

## Self-Check: PASSED

All 4 files verified present. All 4 commit hashes verified in git log.

---
*Phase: 05-feature-engineering*
*Completed: 2026-03-07*
