---
phase: 07-agent-training-and-validation
plan: 02
subsystem: training
tags: [stable-baselines3, ppo, a2c, sac, vecnormalize, callbacks, early-stopping]

# Dependency graph
requires:
  - phase: 06-environments
    provides: "StockTradingEnv, CryptoTradingEnv, Gymnasium registration"
  - phase: 05-features
    provides: "Feature assembler with EQUITY_OBS_DIM=156, CRYPTO_OBS_DIM=45"
provides:
  - "ConvergenceCallback for SB3 early stopping based on reward stagnation"
  - "TrainingOrchestrator for PPO/A2C/SAC training with locked hyperparams"
  - "HYPERPARAMS, SEED_MAP, ALGO_MAP constants"
  - "TrainingResult dataclass with paths and convergence metadata"
  - "6 post-training smoke tests (deserialize, shape, non-degenerate, inference, VecNormalize, NaN)"
affects: [07-agent-training-and-validation, 08-paper-trading]

# Tech tracking
tech-stack:
  added: [stable-baselines3 callbacks, VecNormalize, DummyVecEnv, EvalCallback]
  patterns: [TDD red-green for training pipeline, locked hyperparameter constants]

key-files:
  created:
    - src/swingrl/training/callbacks.py
    - src/swingrl/training/trainer.py
    - tests/training/test_callbacks.py
    - tests/training/test_trainer.py
  modified:
    - pyproject.toml
    - .pre-commit-config.yaml

key-decisions:
  - "stable_baselines3 added to mypy ignore_missing_imports and pre-commit additional_dependencies"
  - "type: ignore[assignment] for SB3 VecEnv step/reset type stubs mismatches"
  - "Episode bars set to 50 (minimum allowed by schema) in test fixtures for fast training"

patterns-established:
  - "ConvergenceCallback as EvalCallback child: parent.last_mean_reward access pattern"
  - "TrainingOrchestrator.train() returns TrainingResult dataclass with model/VecNormalize paths"
  - "Smoke test pattern: 6 checks after every training run before model is considered valid"
  - "Tiny arrays (60 steps) + low timesteps (500) for pipeline tests under 10 seconds"

requirements-completed: [VAL-05, TRAIN-03, TRAIN-04, TRAIN-05]

# Metrics
duration: 11min
completed: 2026-03-08
---

# Phase 7 Plan 02: Training Pipeline Summary

**ConvergenceCallback early stopping and TrainingOrchestrator with locked PPO/A2C/SAC hyperparams, VecNormalize wrapping, model saving, and 6 post-training smoke tests**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-08T13:52:02Z
- **Completed:** 2026-03-08T14:03:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- ConvergenceCallback stops training after patience consecutive stagnant evaluations with relative improvement threshold and absolute fallback for zero/negative rewards
- TrainingOrchestrator trains PPO/A2C/SAC with exact locked hyperparameters, net_arch=[64,64], DummyVecEnv + VecNormalize wrapping, and tensorboard logging
- Model + VecNormalize saved to models/active/{env}/{algo}/ directory structure
- 6 post-training smoke tests verify model integrity after every training run
- All 22 tests pass in under 7 seconds using tiny arrays (60 steps, 500 timesteps)

## Task Commits

Each task was committed atomically (TDD red-green):

1. **Task 1: ConvergenceCallback (RED)** - `3f26625` (test)
2. **Task 1: ConvergenceCallback (GREEN)** - `4476d1e` (feat)
3. **Task 2: TrainingOrchestrator (RED)** - `0537f97` (test)
4. **Task 2: TrainingOrchestrator (GREEN)** - `231459e` (feat)

## Files Created/Modified
- `src/swingrl/training/callbacks.py` - ConvergenceCallback BaseCallback subclass for early stopping
- `src/swingrl/training/trainer.py` - TrainingOrchestrator, HYPERPARAMS, SEED_MAP, ALGO_MAP, TrainingResult
- `tests/training/test_callbacks.py` - 9 tests for callback behavior (mock-based, no SB3 models)
- `tests/training/test_trainer.py` - 13 tests for hyperparams, seeds, training pipeline, smoke tests
- `pyproject.toml` - Added stable_baselines3 to mypy ignore_missing_imports
- `.pre-commit-config.yaml` - Added stable-baselines3 to mypy additional_dependencies

## Decisions Made
- Used `type: ignore[attr-defined]` for `parent.last_mean_reward` since mypy resolves parent as BaseCallback (not EvalCallback)
- Used `type: ignore[assignment]` for VecEnv step/reset return types due to SB3 type stub mismatches
- Replaced bare `assert` with conditional return to satisfy bandit B101 in production code
- Removed `log_interval` parameter from `model.learn()` -- passing None/0 causes ZeroDivisionError in A2C on-policy algorithm

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ZeroDivisionError in A2C with log_interval=0**
- **Found during:** Task 2 (TrainingOrchestrator implementation)
- **Issue:** `log_interval=None` intended to suppress logging, but SB3 on-policy `learn()` does `iteration % log_interval` causing ZeroDivisionError when 0
- **Fix:** Removed log_interval parameter entirely (SB3 defaults to safe value)
- **Files modified:** src/swingrl/training/trainer.py
- **Verification:** All 22 tests pass including A2C training
- **Committed in:** 231459e (Task 2 GREEN commit)

**2. [Rule 3 - Blocking] Added stable_baselines3 to mypy overrides and pre-commit deps**
- **Found during:** Task 1 (ConvergenceCallback implementation)
- **Issue:** mypy cannot find stable_baselines3.common.callbacks type stubs, pre-commit hook fails
- **Fix:** Added to pyproject.toml mypy overrides and .pre-commit-config.yaml additional_dependencies
- **Files modified:** pyproject.toml, .pre-commit-config.yaml
- **Verification:** Pre-commit mypy passes
- **Committed in:** 4476d1e (Task 1 GREEN commit)

**3. [Rule 1 - Bug] Fixed episode_bars minimum validation in test fixtures**
- **Found during:** Task 2 (TrainingOrchestrator tests)
- **Issue:** Test config used episode_bars=30 but schema requires >=50
- **Fix:** Changed to episode_bars=50 and arrays to 60 rows
- **Files modified:** tests/training/test_trainer.py
- **Verification:** Config loads successfully, all tests pass
- **Committed in:** 231459e (Task 2 GREEN commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- Pre-commit stash conflict during commit caused trainer.py to be deleted from working tree after commit; restored via `git checkout HEAD`
- metrics.py from plan 07-01 was accidentally included in 07-02 commit due to pre-commit stash conflict recovery (no functional impact)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Training pipeline ready for ensemble validation (plan 07-03)
- ConvergenceCallback and TrainingOrchestrator can be used to train all 6 models (3 algos x 2 envs)
- Smoke tests ensure model integrity before any model is used in production

---
*Phase: 07-agent-training-and-validation*
*Completed: 2026-03-08*
