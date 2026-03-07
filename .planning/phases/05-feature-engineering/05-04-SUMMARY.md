---
phase: 05-feature-engineering
plan: 04
subsystem: features
tags: [z-score, normalization, correlation, pruning, pandas-rolling, domain-rules]

# Dependency graph
requires:
  - phase: 05-feature-engineering
    provides: FeaturesConfig with zscore_epsilon, correlation_threshold; TechnicalIndicatorCalculator; shared test fixtures
provides:
  - RollingZScoreNormalizer with per-environment windows and bounds validation
  - CorrelationPruner with domain-driven drop rules and pruning report
affects: [05-05-assembler]

# Tech tracking
tech-stack:
  added: []
  patterns: [rolling-zscore-epsilon-floor, domain-priority-correlation-pruning, sma-exception-threshold]

key-files:
  created:
    - src/swingrl/features/normalization.py
    - src/swingrl/features/correlation.py
    - tests/features/test_normalization.py
    - tests/features/test_correlation.py
  modified: []

key-decisions:
  - "Epsilon floor via clip(lower=epsilon) on rolling std prevents inf with flat prices"
  - "Domain priority dict with integer scores drives deterministic drop selection"
  - "SMA exception threshold (0.90) higher than general correlation threshold (0.85)"
  - "Equal-priority tiebreaker uses alphabetical order for determinism"

patterns-established:
  - "Rolling z-score: (x - rolling_mean) / rolling_std.clip(lower=epsilon)"
  - "CorrelationPruner as one-time pre-training analysis tool, not runtime component"
  - "Domain-driven KEEP_PRIORITY dict for correlation pruning drop selection"

requirements-completed: [FEAT-06, FEAT-09]

# Metrics
duration: 4min
completed: 2026-03-07
---

# Phase 5 Plan 04: Normalization & Correlation Pruning Summary

**Rolling z-score normalization with 252/360-bar windows and correlation pruning with domain-driven drop priority rules (RSI > MACD histogram > SMAs > VIX z-score)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T03:00:54Z
- **Completed:** 2026-03-07T03:04:57Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- RollingZScoreNormalizer handles per-environment windows (252 equity, 360 crypto) with epsilon floor preventing inf on flat prices
- CorrelationPruner identifies r > 0.85 pairs and applies domain-driven drop rules preserving RSI-14, MACD histogram, VIX z-score, and both SMA ratios (unless r > 0.90)
- validate_bounds() smoke test verifies >90% of z-scored values fall within [-3, 3] range
- Human-readable pruning report documents all flagged pairs, dropped columns, and domain rules applied
- 23 tests covering normalization edge cases, domain rules, SMA exception, and pruning pipeline

## Task Commits

Each task was committed atomically:

1. **Task 1: Rolling z-score normalization** - `e369a81` (feat)
2. **Task 2: Correlation pruning with domain-driven rules** - `ccf3a40` (feat)

## Files Created/Modified
- `src/swingrl/features/normalization.py` - RollingZScoreNormalizer with normalize() and validate_bounds()
- `src/swingrl/features/correlation.py` - CorrelationPruner with analyze(), find_correlated_pairs(), select_drops(), prune(), report()
- `tests/features/test_normalization.py` - 10 tests for z-score normalization
- `tests/features/test_correlation.py` - 13 tests for correlation pruning

## Decisions Made
- Epsilon floor applied via `clip(lower=epsilon)` on rolling std rather than conditional logic -- simpler and handles all-NaN edge case
- Domain priority as integer dict (10 for RSI, 9 for MACD histogram, 8 for SMAs, 7 for VIX z-score) -- extensible for future features
- SMA exception threshold at 0.90 (higher than general 0.85) per CONTEXT.md spec
- Equal-priority tiebreaker uses alphabetical column name ordering for deterministic, reproducible pruning results
- mypy type ignore on `corr_matrix.iloc[i, j]` due to pandas stub limitations with scalar indexing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- mypy flagged `float(corr_matrix.iloc[i, j])` as incompatible type due to pandas stubs union type; resolved with targeted type: ignore[arg-type]
- mypy flagged `np.asarray` needed for `.values` return type in validate_bounds; resolved by explicit `np.asarray(col_vals, dtype=float)` cast

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- RollingZScoreNormalizer ready for observation vector assembler (Plan 05) to normalize technical features before assembly
- CorrelationPruner ready for pre-training feature selection pipeline
- No blockers

---
*Phase: 05-feature-engineering*
*Completed: 2026-03-07*
