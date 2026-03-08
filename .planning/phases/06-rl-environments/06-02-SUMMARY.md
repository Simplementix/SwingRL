---
phase: 06-rl-environments
plan: 02
subsystem: environments
tags: [gymnasium, base-trading-env, stock-trading-env, portfolio-state, risk-penalties]

# Dependency graph
requires:
  - phase: 06-rl-environments
    provides: PortfolioSimulator, RollingSharpeReward, process_actions, EnvironmentConfig
  - phase: 05-feature-engineering
    provides: ObservationAssembler with EQUITY_OBS_DIM=156, EQUITY_PORTFOLIO=27
provides:
  - BaseTradingEnv(gymnasium.Env) with shared step/reset contract, portfolio simulation, reward, penalties
  - StockTradingEnv(BaseTradingEnv) for 8-ETF equity trading with 252-step episodes
  - Portfolio state injection into observation vectors (cash ratio, exposure, per-asset weights)
  - Soft risk penalties (quadratic position, linear drawdown)
  - from_arrays() factory with shape validation
affects: [06-03, 07-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [gymnasium-env-contract, portfolio-state-injection, soft-risk-penalties]

key-files:
  created:
    - src/swingrl/envs/base.py
    - src/swingrl/envs/equity.py
  modified:
    - src/swingrl/envs/__init__.py
    - tests/test_envs.py
    - tests/conftest.py
    - .pre-commit-config.yaml

key-decisions:
  - "step_count-based termination instead of current_step comparison avoids off-by-one in episode length"
  - "equity_env_config fixture with 8 symbols separate from loaded_config (2 symbols) prevents fixture coupling"
  - "gymnasium added to pre-commit mypy additional_dependencies for type checking in isolated hook environment"

patterns-established:
  - "BaseTradingEnv pattern: shared Gymnasium env with environment literal switching equity/crypto params"
  - "Portfolio state injection: last N elements of observation replaced with live portfolio state at each step"
  - "Soft risk penalty pattern: quadratic for position concentration, linear for drawdown excess"

requirements-completed: [TRAIN-01, TRAIN-08, TRAIN-10, TRAIN-11]

# Metrics
duration: 6min
completed: 2026-03-08
---

# Phase 6 Plan 02: BaseTradingEnv and StockTradingEnv Summary

**Gymnasium-compatible BaseTradingEnv with portfolio state injection and StockTradingEnv for 8-ETF 252-step equity episodes with soft risk penalties**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-08T03:30:03Z
- **Completed:** 2026-03-08T03:36:32Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 6

## Accomplishments
- BaseTradingEnv implements full Gymnasium step/reset contract with 5-tuple/2-tuple returns
- StockTradingEnv runs 252-step equity episodes with 8 ETFs, deterministic reset, and portfolio tracking
- Portfolio state (cash ratio, exposure, per-asset weights/PnL/bars) injected into observation vectors
- Soft risk penalties: quadratic position concentration penalty + linear drawdown penalty
- 44 total tests passing (28 from Plan 01 + 16 new StockTradingEnv tests)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `2603f2f` (test)
2. **Task 1 GREEN: Implementation** - `e9b5e58` (feat)

_TDD task: test committed first, then implementation._

## Files Created/Modified
- `src/swingrl/envs/base.py` - BaseTradingEnv with shared step/reset, portfolio state, risk penalties
- `src/swingrl/envs/equity.py` - StockTradingEnv specialization for 8-ETF equity trading
- `src/swingrl/envs/__init__.py` - Updated package docstring
- `tests/test_envs.py` - 16 new tests for StockTradingEnv contract, penalties, raw obs, full episode
- `tests/conftest.py` - equity_env_config fixture with 8 symbols for environment tests
- `.pre-commit-config.yaml` - Added gymnasium to mypy additional_dependencies

## Decisions Made
- Used `_step_count >= _episode_bars` for termination instead of `_current_step >= _max_step - 1` to avoid off-by-one
- Created separate `equity_env_config` fixture with 8 symbols (matching 8-column price array) rather than changing existing `loaded_config` fixture (2 symbols) to avoid breaking other tests
- Added `gymnasium>=1.0` to pre-commit mypy additional_dependencies since mypy hook runs in isolated environment

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed off-by-one in episode termination**
- **Found during:** Task 1 GREEN (implementation)
- **Issue:** `_current_step >= _max_step - 1` terminated at step 251 instead of 252
- **Fix:** Changed to `_step_count >= _episode_bars` for exact episode length control
- **Files modified:** src/swingrl/envs/base.py
- **Verification:** test_terminated_after_252_steps passes
- **Committed in:** e9b5e58

**2. [Rule 3 - Blocking] Added gymnasium to pre-commit mypy dependencies**
- **Found during:** Task 1 GREEN (commit attempt)
- **Issue:** Pre-commit mypy hook failed with `Cannot find module named "gymnasium"`
- **Fix:** Added `gymnasium>=1.0` to mypy additional_dependencies in .pre-commit-config.yaml
- **Files modified:** .pre-commit-config.yaml
- **Verification:** Pre-commit mypy passes
- **Committed in:** e9b5e58

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness and CI. No scope creep.

## Issues Encountered
- Test fixture mismatch: `loaded_config` has 2 equity symbols but `equity_prices_array` has 8 columns; resolved by creating dedicated `equity_env_config` fixture with 8 symbols

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- BaseTradingEnv ready for CryptoTradingEnv (Plan 03) to inherit with crypto-specific overrides
- StockTradingEnv passes all Gymnasium contract tests, ready for SB3 training in Phase 7
- Portfolio state injection pattern established for both environments

---
*Phase: 06-rl-environments*
*Completed: 2026-03-08*
