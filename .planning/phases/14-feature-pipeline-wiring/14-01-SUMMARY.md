---
phase: 14-feature-pipeline-wiring
plan: 01
subsystem: features
tags: [feature-pipeline, sentiment, assembler, cli, a-b-testing, observation-vector]

# Dependency graph
requires:
  - phase: 05-features
    provides: assembler.py ObservationAssembler, pipeline.py compare_features/get_sentiment_features
  - phase: 06-environments
    provides: base.py BaseTradingEnv EQUITY_OBS_DIM import
  - phase: 10-production-hardening
    provides: SentimentConfig in schema.py, finbert/news_fetcher sentiment modules

provides:
  - scripts/compare_features.py: CLI for A/B feature comparison (FEAT-09)
  - equity_obs_dim()/equity_per_asset_dim() helpers: config-aware dimension computation (FEAT-10)
  - assemble_equity(sentiment_features=...) producing (172,) vector when enabled (FEAT-10)
  - BaseTradingEnv observation_space.shape dynamically driven by SentimentConfig.enabled

affects:
  - training (env observation_space shape)
  - paper-trading (live observation assembly)
  - model inference (observation dimensions must match trained model)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - equity_obs_dim(sentiment_enabled, n_symbols) always derived, never hardcoded
    - assemble_equity() backward-compatible via default None sentinel for new kwarg
    - CLI scripts follow build_parser()/main(argv)/KeyboardInterrupt pattern from train.py

key-files:
  created:
    - scripts/compare_features.py
  modified:
    - src/swingrl/features/assembler.py
    - src/swingrl/envs/base.py
    - src/swingrl/features/pipeline.py
    - tests/test_phase14.py
    - tests/training/test_trainer.py

key-decisions:
  - "equity_obs_dim(False, 8)==156 and equity_obs_dim(True, 8)==172 -- sentiment adds 2 features per asset"
  - "EQUITY_OBS_DIM=156 and EQUITY_PER_ASSET=15 kept as backward-compat aliases at module level"
  - "assemble_equity() sentinel None for sentiment_features preserves existing callers unchanged"
  - "BaseTradingEnv._obs_dim derived from config.sentiment.enabled at __init__ time, not import time"
  - "tiny_equity_features fixture updated to derive shape from equity_obs_dim() dynamically"

patterns-established:
  - "Never hardcode 156 or 172 -- always call equity_obs_dim(sentiment_enabled, n_symbols)"
  - "Sentiment integration: get_sentiment_features() called in pipeline._get_equity_observation() only when enabled"
  - "Feature dimension helpers live in assembler.py as module-level pure functions, not class methods"

requirements-completed: [FEAT-09, FEAT-10]

# Metrics
duration: 11min
completed: 2026-03-10
---

# Phase 14 Plan 01: Feature Pipeline Wiring Summary

**compare_features CLI (FEAT-09) and config-aware sentiment integration producing (172,) equity observations (FEAT-10)**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-10T21:51:41Z
- **Completed:** 2026-03-10T22:02:51Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Delivered `scripts/compare_features.py` CLI that reads JSON metrics files, calls `compare_features()`, prints accepted/rejected verdict with Sharpe improvement, exits 0/1 — enabling shell scripting of A/B feature experiments
- Added `equity_obs_dim(sentiment_enabled, n_symbols)` and `equity_per_asset_dim(sentiment_enabled)` helpers that eliminate all hardcoded dimension constants from production paths
- Wired `get_sentiment_features()` into `assemble_equity()` via `sentiment_features` kwarg: returns `(172,)` when enabled, `(156,)` when disabled, with backward-compat `None` default
- Updated `BaseTradingEnv` to derive `observation_space.shape` from `config.sentiment.enabled` at init time
- All 18 phase-14 tests pass; full suite 835 passed, 0 failures

## Task Commits

Each task was committed atomically:

1. **RED (test(14-01)): failing tests for FEAT-09 and FEAT-10** - `decce89`
2. **GREEN (feat(14-01)): implement FEAT-09 and FEAT-10** - `5a75d93`
3. **Rule 1 fix (fix(14-01)): update tiny_equity_features fixture** - `3114a9a`

_TDD task: test commit (RED) -> implementation commit (GREEN) -> regression fix_

## Files Created/Modified

- `scripts/compare_features.py` - CLI entrypoint: build_parser(), main(), reads JSON metrics, calls compare_features(), exits 0/1
- `src/swingrl/features/assembler.py` - Added EQUITY_PER_ASSET_BASE, SENTIMENT_FEATURES_PER_ASSET, equity_per_asset_dim(), equity_obs_dim(), _SENTIMENT_FEATURE_NAMES; updated assemble_equity() and get_feature_names_equity()
- `src/swingrl/envs/base.py` - Replaced `EQUITY_OBS_DIM` import with `equity_obs_dim` function call; `self._obs_dim` now config-aware
- `src/swingrl/features/pipeline.py` - Added `import os` and `equity_per_asset_dim` import; `_get_equity_observation()` calls `get_sentiment_features()` when enabled and passes result as `sentiment_features` kwarg
- `tests/test_phase14.py` - 18 tests across 5 test classes covering all behaviors
- `tests/training/test_trainer.py` - Fixed `tiny_equity_features` fixture to derive shape from `equity_obs_dim()`

## Decisions Made

- `equity_obs_dim(False, 8) == 156` and `equity_obs_dim(True, 8) == 172` -- sentiment adds `SENTIMENT_FEATURES_PER_ASSET=2` per asset
- Backward-compat aliases `EQUITY_OBS_DIM=156` and `EQUITY_PER_ASSET=15` kept at module level so existing importers continue working
- `assemble_equity()` uses `sentiment_features: dict[str, tuple[float, float]] | None = None` sentinel pattern — callers without sentiment pass nothing
- `get_feature_names_equity(sentiment_enabled=True)` parameter matches `assemble_equity()` pattern — same toggle, same position in signature
- `BaseTradingEnv._obs_dim` computed from `config.sentiment.enabled` at `__init__` time so observation_space always matches actual feature array dimensions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed tiny_equity_features fixture using wrong shape for 2-symbol config**
- **Found during:** Task 2 (full suite validation)
- **Issue:** The `tiny_equity_features` fixture in `tests/training/test_trainer.py` used shape `(60, 156)` for a 2-symbol equity config. With the old hardcoded `EQUITY_OBS_DIM=156`, `BaseTradingEnv.observation_space.shape` was always `(156,)` regardless of symbol count, masking the mismatch. With the new `equity_obs_dim(False, 2) = 66` computation, SB3 threw `ValueError: could not broadcast input array from shape (156,) into shape (66,)`.
- **Fix:** Updated fixture to compute shape dynamically via `equity_obs_dim(sentiment_enabled=False, n_equity_symbols=2)`.
- **Files modified:** `tests/training/test_trainer.py`
- **Verification:** All 13 trainer tests pass; full suite 835 passed.
- **Committed in:** `3114a9a`

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test fixture)
**Impact on plan:** The fix was necessary to maintain correctness — the fixture dimension was latently wrong and our change correctly exposed it. No scope creep.

## Issues Encountered

- `uv run pytest` was not picking up the `swingrl` package (the `.pth` file mechanism wasn't activating). Resolved by running with `PYTHONPATH=/path/to/src` prefix. This is consistent with how previous phase tests were run.

## Next Phase Readiness

- FEAT-09 and FEAT-10 complete — all v1.0 requirements satisfied
- Phase 14 is the final execution phase; ready for CI validation and PR creation
- CI homelab run should confirm full suite passes on Linux

---
*Phase: 14-feature-pipeline-wiring*
*Completed: 2026-03-10*
