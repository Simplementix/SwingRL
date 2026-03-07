---
phase: 05-feature-engineering
plan: 05
subsystem: features
tags: [assembler, pipeline, observation-vector, cli, a-b-comparison, duckdb, numpy]

# Dependency graph
requires:
  - phase: 05-feature-engineering
    provides: TechnicalIndicatorCalculator, FundamentalFetcher, MacroFeatureAligner, HMMRegimeDetector, TurbulenceCalculator, RollingZScoreNormalizer, DuckDB feature schema
provides:
  - ObservationAssembler with assemble_equity() returning (156,) and assemble_crypto() returning (45,)
  - FeaturePipeline orchestrating full compute-normalize-store flow for both environments
  - compare_features() for A/B feature comparison with Sharpe threshold and overfitting guard
  - CLI script compute_features.py for manual feature computation
  - Feature name generators for debugging and documentation
affects: [06-rl-environments, 07-training, 09-automation]

# Tech tracking
tech-stack:
  added: []
  patterns: [deterministic-alpha-sorted-assembly, default-portfolio-state, sharpe-based-ab-comparison]

key-files:
  created:
    - src/swingrl/features/assembler.py
    - src/swingrl/features/pipeline.py
    - scripts/compute_features.py
    - tests/features/test_assembler.py
    - tests/features/test_pipeline.py
  modified: []

key-decisions:
  - "Observation assembly: [per-asset alpha-sorted] + [macro] + [HMM] + [turbulence] + [overnight crypto-only] + [portfolio state]"
  - "Default portfolio state = 100% cash (cash_ratio=1.0, rest zeros) until Phase 6 provides real positions"
  - "NaN detection post-assembly raises DataError rather than silently continuing"
  - "Feature A/B comparison uses validation Sharpe >= 0.05 threshold with train-vs-validation overfitting guard"
  - "Pipeline stores normalized features to DuckDB via replacement scan pattern"

patterns-established:
  - "ObservationAssembler as contract between feature engineering and RL environments"
  - "FeaturePipeline as orchestrator coordinating all feature modules"
  - "compare_features() standalone function for A/B testing infrastructure"

requirements-completed: [FEAT-07, FEAT-08, FEAT-11]

# Metrics
duration: 9min
completed: 2026-03-07
---

# Phase 5 Plan 05: Assembler, Pipeline, CLI, and A/B Comparison Summary

**Observation vector assembler producing (156,)/(45,) arrays, full pipeline orchestrating compute-normalize-store, CLI for manual computation, and Sharpe-based A/B comparison infrastructure**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-07T03:07:48Z
- **Completed:** 2026-03-07T03:16:37Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- ObservationAssembler produces deterministic (156,) equity and (45,) crypto observation vectors with alpha-sorted per-asset features, NaN guard, and default portfolio state
- FeaturePipeline orchestrates full compute-normalize-store flow for both environments: reads OHLCV from DuckDB, computes technical indicators, normalizes via rolling z-score, stores to feature tables, fits HMM
- Feature A/B comparison infrastructure with Sharpe improvement threshold (>= 0.05) and overfitting guard (rejects when train improves but validation decreases)
- CLI script compute_features.py with --environment, --symbols, --start, --end, --check-fundamentals arguments
- 32 tests total (17 assembler + 15 pipeline) all passing, 298 full suite tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Observation vector assembler** - `771d2c3` (feat)
2. **Task 2: Feature pipeline + CLI + A/B comparison** - `81243cc` (feat)

## Files Created/Modified
- `src/swingrl/features/assembler.py` - ObservationAssembler with assemble_equity/crypto, feature name generators, default portfolio state
- `src/swingrl/features/pipeline.py` - FeaturePipeline with compute_equity/crypto, get_observation, compare_features
- `scripts/compute_features.py` - CLI entry point for manual feature computation
- `tests/features/test_assembler.py` - 17 tests for shape, order, defaults, NaN guard, feature names
- `tests/features/test_pipeline.py` - 15 tests for pipeline, DuckDB storage, observation assembly, A/B comparison, CLI

## Decisions Made
- Assembly order: [per-asset alpha-sorted] + [macro] + [HMM] + [turbulence] + [overnight crypto-only] + [portfolio state]
- Default portfolio state uses 100% cash (cash_ratio=1.0, rest zeros) since Phase 5 has no real position data
- NaN detection in assembled vectors raises DataError immediately -- post-warmup NaN is a pipeline bug that must be surfaced
- Feature A/B comparison is infrastructure only -- actual A/B testing requires trained agents (Phase 7+)
- Pipeline stores normalized (z-scored) features, not raw values, to DuckDB feature tables
- CLI loads module via importlib.util in tests since `scripts/` is not a Python package

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] HMM convergence with synthetic test data**
- **Found during:** Task 2 (pipeline tests)
- **Issue:** HMM initial_fit failed on 300-bar synthetic random walk data with default 1260-bar window and 5 inits
- **Fix:** Test config uses reduced hmm window (200), increased inits (10), and higher ridge (1e-4) for reliable convergence with limited test data
- **Files modified:** tests/features/test_pipeline.py
- **Committed in:** 81243cc (Task 2 commit)

**2. [Rule 3 - Blocking] CLI import path for tests**
- **Found during:** Task 2 (CLI tests)
- **Issue:** `from scripts.compute_features import build_parser` failed because `scripts/` is not a Python package
- **Fix:** Used `importlib.util.spec_from_file_location` to load the CLI module dynamically in tests
- **Files modified:** tests/features/test_pipeline.py
- **Committed in:** 81243cc (Task 2 commit)

**3. [Rule 3 - Blocking] mypy and bandit pre-commit failures**
- **Found during:** Task 2 commit
- **Issue:** mypy flagged `.tz`/`.tz_localize` on generic Index; bandit flagged parameterized SQL as injection
- **Fix:** Explicit pd.DatetimeIndex cast; nosec B608 on safe parameterized queries and hardcoded column lists
- **Files modified:** src/swingrl/features/pipeline.py
- **Committed in:** 81243cc (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All fixes necessary for test reliability and pre-commit compliance. No scope creep.

## Issues Encountered
- Pre-commit ruff-format reformatted files on first Task 1 commit attempt -- re-staged and committed successfully

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ObservationAssembler ready for Phase 6 RL environments to consume (156,)/(45,) observation vectors
- FeaturePipeline ready for Phase 9 automation to trigger after ingestion
- Feature A/B comparison infrastructure ready for Phase 7+ feature evaluation
- CLI enables manual feature seeding before training starts
- Full Phase 5 feature engineering complete: all 5 plans delivered
- No blockers

## Self-Check: PASSED

- All 5 created files verified present on disk
- Both task commits (771d2c3, 81243cc) verified in git log
- 298/298 tests passing in full suite

---
*Phase: 05-feature-engineering*
*Completed: 2026-03-07*
