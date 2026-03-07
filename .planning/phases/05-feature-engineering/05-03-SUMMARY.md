---
phase: 05-feature-engineering
plan: 03
subsystem: features
tags: [hmm, hmmlearn, turbulence, mahalanobis, regime-detection, numpy]

# Dependency graph
requires:
  - phase: 02-developer-experience
    provides: SwingRLConfig schema with load_config(), structlog logging
provides:
  - HMMRegimeDetector class with fit/predict_proba/warm_start_refit
  - TurbulenceCalculator class with expanding (equity) and rolling (crypto) lookback
  - FeaturesConfig in SwingRLConfig with HMM, normalization, and turbulence params
affects: [06-rl-environments, 07-training, 05-04, 05-05]

# Tech tracking
tech-stack:
  added: []
  patterns: [TDD for feature modules, ridge regularization for HMM warm-start, pinv for near-singular covariance]

key-files:
  created:
    - src/swingrl/features/__init__.py
    - src/swingrl/features/hmm_regime.py
    - src/swingrl/features/turbulence.py
    - tests/features/__init__.py
    - tests/features/test_hmm_regime.py
    - tests/features/test_turbulence.py
  modified:
    - src/swingrl/config/schema.py

key-decisions:
  - "FeaturesConfig added to SwingRLConfig with HMM window, n_iter, n_inits, ridge, normalization, and turbulence params"
  - "HMM label ordering via mean-return sort ensures bull=state0 consistency across refits"
  - "Ridge regularization (1e-6) on covariance matrices prevents ValueError on warm-start"

patterns-established:
  - "HMM warm-start pattern: copy params, add ridge to covars, fit with init_params=''"
  - "Turbulence via pinv: np.linalg.pinv for near-singular crypto covariance"
  - "Feature module TDD: test file first, implementation second, both in features/ package"

requirements-completed: [FEAT-05]

# Metrics
duration: 5min
completed: 2026-03-07
---

# Phase 5 Plan 3: HMM Regime & Turbulence Summary

**Two-state Gaussian HMM regime detector with warm-start ridge regularization and Mahalanobis turbulence index with expanding/rolling lookback**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-07T02:48:26Z
- **Completed:** 2026-03-07T02:54:22Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- HMMRegimeDetector with initial_fit, cold_start_fit, warm_start_refit producing P(bull)/P(bear) that sum to 1.0
- TurbulenceCalculator with Mahalanobis distance using expanding (equity) and rolling 1080-bar (crypto) lookback
- FeaturesConfig added to SwingRLConfig with all HMM, normalization, correlation, and turbulence parameters
- 24 tests covering both modules (14 HMM + 10 turbulence), full suite 233 tests green

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1: HMM regime detector** - `162c680` (test) + `3e8a19d` (feat)
2. **Task 2: Turbulence index calculator** - `e1eb054` (test) + `cd09322` (feat)

_TDD tasks have separate test and implementation commits._

## Files Created/Modified
- `src/swingrl/features/__init__.py` - Features package init
- `src/swingrl/features/hmm_regime.py` - HMMRegimeDetector with 8 methods (322 lines)
- `src/swingrl/features/turbulence.py` - TurbulenceCalculator with compute/compute_series (100 lines)
- `tests/features/__init__.py` - Test package init
- `tests/features/test_hmm_regime.py` - 14 tests across 6 test classes
- `tests/features/test_turbulence.py` - 10 tests across 4 test classes
- `src/swingrl/config/schema.py` - Added FeaturesConfig with 12 configuration fields

## Decisions Made
- FeaturesConfig added as new config section with sensible defaults matching CONTEXT.md specs
- Cold-start uses informed priors (positive mean/low vol = bull) for < 500 bars
- Turbulence compute_series uses loop over bars (simple, correct; vectorization deferred if needed)
- store_hmm_state accepts generic db connection with execute method for testability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed mypy type errors for numpy/pandas interop**
- **Found during:** Task 1 (HMM implementation)
- **Issue:** np.log() returns ndarray which mypy doesn't see .rolling() on; predict_proba return typed as Any
- **Fix:** Explicit pd.Series construction for log returns; explicit type annotation on predict_proba return
- **Files modified:** src/swingrl/features/hmm_regime.py
- **Verification:** mypy passes, all tests still green
- **Committed in:** 3e8a19d (Task 1 feat commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Type annotation fix necessary for mypy strict mode. No scope creep.

## Issues Encountered
- detect-secrets baseline needed update due to unstaged test_fundamentals.py from another plan containing test API key string. Updated .secrets.baseline to resolve.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- HMM and turbulence modules ready for observation vector assembly (Plan 05)
- FeaturesConfig established for all subsequent feature plans
- Both modules are self-contained with no cross-dependencies on other feature modules

---
*Phase: 05-feature-engineering*
*Completed: 2026-03-07*
