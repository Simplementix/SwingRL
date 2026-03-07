---
phase: 05-feature-engineering
plan: 01
subsystem: features
tags: [stockstats, duckdb, pydantic, technical-indicators, sma, rsi, macd, bollinger, atr, adx]

# Dependency graph
requires:
  - phase: 04-data-storage
    provides: DatabaseManager with DuckDB context manager, ohlcv_daily/ohlcv_4h tables
  - phase: 02-developer-experience
    provides: SwingRLConfig Pydantic schema, load_config(), structlog logging
provides:
  - FeaturesConfig in SwingRLConfig with normalization, HMM, correlation params
  - DuckDB DDL for features_equity, features_crypto, fundamentals, hmm_state_history tables
  - TechnicalIndicatorCalculator class with 9 price-action indicators
  - Weekly-derived feature computation (trend direction + weekly RSI-14)
  - Crypto multi-timeframe feature computation (daily + 4H indicators)
  - Derived features for HMM input (log returns, BB width)
  - Shared test fixtures (equity_ohlcv_250, crypto_ohlcv_400, feature_config, duckdb_conn)
affects: [05-02-fundamentals, 05-03-hmm-regime, 05-04-normalization, 05-05-assembler]

# Tech tracking
tech-stack:
  added: []
  patterns: [stockstats-retype-copy-safety, np-where-for-safe-division, duckdb-ddl-idempotent]

key-files:
  created:
    - src/swingrl/features/__init__.py
    - src/swingrl/features/schema.py
    - src/swingrl/features/technical.py
    - tests/features/__init__.py
    - tests/features/conftest.py
    - tests/features/test_config_and_schema.py
    - tests/features/test_technical.py
  modified:
    - src/swingrl/config/schema.py
    - config/swingrl.yaml
    - config/swingrl.prod.yaml.example

key-decisions:
  - "FeaturesConfig already existed from prior research commit; added zscore_epsilon field"
  - "init_feature_schema takes raw DuckDB conn (not DatabaseManager) for testability with in-memory connections"
  - "np.where handles zero-division in volume ratio and BB position without suppressing numpy warnings"

patterns-established:
  - "StockDataFrame.retype(ohlcv.copy()) pattern for copy-safe indicator computation"
  - "init_feature_schema(conn) accepts raw DuckDB connection for both production and test use"
  - "Weekly features forward-filled to daily index via pandas reindex(method='ffill')"

requirements-completed: [FEAT-01, FEAT-02, FEAT-10, FEAT-11]

# Metrics
duration: 5min
completed: 2026-03-07
---

# Phase 5 Plan 01: Feature Engineering Foundation Summary

**9 price-action indicators via stockstats, DuckDB DDL for 4 feature tables, FeaturesConfig schema, and shared test infrastructure for all Phase 5 plans**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-07T02:48:22Z
- **Completed:** 2026-03-07T02:53:49Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- TechnicalIndicatorCalculator computes all 9 price-action indicators (SMA ratios, RSI-14, MACD line/histogram, BB position, ATR%, volume ratio, ADX-14) with verified correctness
- Weekly-derived features (trend direction + RSI-14) and crypto multi-timeframe features (daily + 4H) compute from resampled bars
- DuckDB DDL creates 4 feature tables (features_equity, features_crypto, fundamentals, hmm_state_history) with idempotent CREATE TABLE IF NOT EXISTS
- FeaturesConfig with all normalization, HMM, correlation, and turbulence parameters added to SwingRLConfig
- 23 tests covering config loading, DDL correctness, indicator computation, edge cases (zero volume, copy safety, warmup NaN behavior)

## Task Commits

Each task was committed atomically:

1. **Task 1: Config schema + DuckDB DDL + test infrastructure** - `e9c8c64` (feat)
2. **Task 2: Technical indicator calculator** - `ca83790` (feat)

## Files Created/Modified
- `src/swingrl/features/__init__.py` - Feature engineering package init
- `src/swingrl/features/schema.py` - DuckDB DDL for 4 feature tables
- `src/swingrl/features/technical.py` - TechnicalIndicatorCalculator with compute_price_action, compute_weekly_features, compute_crypto_multi_timeframe, compute_derived
- `src/swingrl/config/schema.py` - Added zscore_epsilon to FeaturesConfig
- `config/swingrl.yaml` - Added features section with all params
- `config/swingrl.prod.yaml.example` - Added features section with all params
- `tests/features/__init__.py` - Test package init
- `tests/features/conftest.py` - Shared fixtures (equity_ohlcv_250, crypto_ohlcv_400, feature_config, duckdb_conn)
- `tests/features/test_config_and_schema.py` - 8 tests for config and DDL
- `tests/features/test_technical.py` - 15 tests for technical indicators

## Decisions Made
- FeaturesConfig already existed from a prior research-phase commit (162c680); only zscore_epsilon was added
- init_feature_schema() accepts a raw DuckDB connection rather than DatabaseManager, enabling in-memory DuckDB for tests without needing the full singleton
- np.where used for safe division (volume ratio and BB position) to handle zero denominators without try/except

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing RED test files from future plans (test_fundamentals.py, test_hmm_regime.py, test_turbulence.py) cause import errors when running full suite; these are expected failures for plans 05-02 through 05-05

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TechnicalIndicatorCalculator ready for use by normalization (Plan 04) and assembler (Plan 05)
- Feature tables ready for data insertion by fundamentals (Plan 02) and HMM (Plan 03)
- Shared test fixtures available for all subsequent Phase 5 test files
- No blockers

---
*Phase: 05-feature-engineering*
*Completed: 2026-03-07*
