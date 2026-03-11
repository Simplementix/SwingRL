---
phase: 07-agent-training-and-validation
plan: 03
subsystem: agents
tags: [walk-forward, backtest, ensemble, softmax, validation, cli]

# Dependency graph
requires:
  - phase: 07-agent-training-and-validation
    provides: "Performance metrics (Plan 01) and TrainingOrchestrator (Plan 02)"
  - phase: 04-storage-and-validation
    provides: "DatabaseManager with DuckDB backtest_results table"
provides:
  - "Walk-forward backtester with growing-window fold generation and purge gaps"
  - "Sharpe-weighted softmax ensemble blender with adaptive validation windows"
  - "CLI scripts for training (scripts/train.py) and backtesting (scripts/backtest.py)"
affects: [08-paper-trading]

# Tech tracking
tech-stack:
  added: []
  patterns: [growing-window walk-forward folds, softmax ensemble weighting, adaptive turbulence windows]

key-files:
  created:
    - src/swingrl/agents/backtest.py
    - src/swingrl/training/ensemble.py
    - scripts/train.py
    - scripts/backtest.py
    - tests/agents/test_backtest.py
    - tests/training/test_ensemble.py
  modified: []

key-decisions:
  - "Default turbulence threshold 1.0 for adaptive window shrink (configurable via EnsembleBlender constructor)"
  - "Ensemble validation windows: 63 bars equity, 126 bars crypto (from CONTEXT.md spec)"
  - "Train script writes placeholder equal weights initially; actual validation Sharpe comes from backtest.py"
  - "nosec B608 on DuckDB table name interpolation (env_name is CLI enum, not user input)"

patterns-established:
  - "generate_folds pure function: stateless fold generation testable without SB3"
  - "WalkForwardBacktester lazy-imports TrainingOrchestrator to avoid circular deps"
  - "CLI scripts follow compute_features.py pattern: build_parser + main(argv) + __name__ guard"

requirements-completed: [VAL-01, VAL-02, TRAIN-06]

# Metrics
duration: 6min
completed: 2026-03-08
---

# Phase 7 Plan 03: Walk-Forward Backtesting and Ensemble Summary

**Growing-window walk-forward backtester with embargo purge gaps, Sharpe-weighted softmax ensemble blender with adaptive turbulence windows, and CLI scripts for training/backtesting pipelines**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-08T14:06:15Z
- **Completed:** 2026-03-08T14:12:21Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Walk-forward fold generator with growing training windows, per-environment embargo gaps (10 bars equity, 130 bars crypto), and minimum 3-fold validation
- Sharpe-weighted softmax ensemble blender with numerical stability (max-subtraction) and adaptive 50% window shrink during high turbulence
- WalkForwardBacktester class orchestrating per-fold retraining, IS/OOS evaluation, metric computation, validation gates, overfitting detection, and DuckDB result storage
- CLI scripts/train.py (--env/--algo/--timesteps) and scripts/backtest.py (--env/--timesteps) following existing CLI patterns
- 21 tests covering fold generation (count, leakage, growing window, crypto params), softmax weights (sum, ordering, negative, equal), and ensemble blending

## Task Commits

Each task was committed atomically (TDD red-green):

1. **Task 1 (RED):** Failing tests for backtester and ensemble - `e6aa014` (test)
2. **Task 1 (GREEN):** Walk-forward backtester and ensemble blender - `0810226` (feat)
3. **Task 2:** CLI scripts for training and backtesting - `0d3266a` (feat)

## Files Created/Modified
- `src/swingrl/agents/backtest.py` - generate_folds, FoldResult, WalkForwardBacktester (fold gen, train, eval, store)
- `src/swingrl/training/ensemble.py` - sharpe_softmax_weights, EnsembleBlender (adaptive validation windows)
- `scripts/train.py` - CLI for PPO/A2C/SAC training with DuckDB model_metadata writes
- `scripts/backtest.py` - CLI for walk-forward validation with summary tables and gate reporting
- `tests/agents/test_backtest.py` - 10 tests for fold generation and FoldResult
- `tests/training/test_ensemble.py` - 11 tests for softmax weights, blending, adaptive windows

## Decisions Made
- Default turbulence threshold set to 1.0 (configurable); matches historical 90th percentile heuristic
- Ensemble validation windows hardcoded as constants (63 equity, 126 crypto) per CONTEXT.md spec
- Train script uses placeholder equal Sharpe ratios for initial ensemble weights; actual validation comes from backtest.py
- Used `nosec B608` for DuckDB table name interpolation since env_name is constrained by CLI enum choices

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed ruff B905 strict zip parameter**
- **Found during:** Task 1 (ensemble.py commit)
- **Issue:** ruff requires explicit `strict=` parameter on `zip()` calls
- **Fix:** Added `strict=True` to zip in sharpe_softmax_weights
- **Files modified:** src/swingrl/training/ensemble.py
- **Verification:** Pre-commit passes
- **Committed in:** 0810226

**2. [Rule 3 - Blocking] Fixed bandit B608 SQL injection warning**
- **Found during:** Task 2 (CLI scripts commit)
- **Issue:** bandit flags f-string SQL as injection risk; `# noqa: S608` only suppresses ruff, not bandit
- **Fix:** Added `# nosec B608` alongside existing `# noqa: S608`
- **Files modified:** scripts/train.py, scripts/backtest.py
- **Verification:** bandit passes
- **Committed in:** 0d3266a

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Lint suppression fixes, no scope creep.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 7 (Agent Training and Validation) fully complete: metrics, validation gates, training pipeline, walk-forward backtester, ensemble blender, CLI scripts
- Ready for Phase 8 (Paper Trading): trained models can be loaded, ensemble weights computed, walk-forward validation results stored in DuckDB
- All CLI scripts importable and functional (train.py, backtest.py)

## Self-Check: PASSED

- All 6 created files exist on disk
- All 3 commit hashes verified in git log

---
*Phase: 07-agent-training-and-validation*
*Completed: 2026-03-08*
