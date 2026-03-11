---
phase: 06-rl-environments
plan: 01
subsystem: environments
tags: [gymnasium, portfolio-simulator, sharpe-reward, softmax, pydantic]

# Dependency graph
requires:
  - phase: 05-feature-engineering
    provides: ObservationAssembler with EQUITY_OBS_DIM=156, CRYPTO_OBS_DIM=45
  - phase: 02-developer-experience
    provides: SwingRLConfig schema, load_config(), structlog logging
provides:
  - PortfolioSimulator with dollar-based rebalancing and transaction cost deduction
  - RollingSharpeReward with 20-day rolling window and expanding warmup
  - process_actions softmax + deadzone action conversion
  - EnvironmentConfig in SwingRLConfig with episode bars, costs, penalties
  - Test fixtures for synthetic feature/price arrays
affects: [06-02, 06-03, 07-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [dollar-based-portfolio-tracking, softmax-action-conversion, rolling-sharpe-reward]

key-files:
  created:
    - src/swingrl/envs/portfolio.py
    - src/swingrl/envs/rewards.py
  modified:
    - src/swingrl/config/schema.py
    - src/swingrl/envs/__init__.py
    - tests/conftest.py
    - tests/test_envs.py
    - config/swingrl.yaml

key-decisions:
  - "deque(maxlen=window) for RollingSharpeReward provides automatic rolling window without manual index tracking"
  - "np.where(prices > 0, prices, 1.0) safety guard prevents divide-by-zero in portfolio rebalancing"
  - "mypy run without --strict to match project pyproject.toml config (strict=false)"

patterns-established:
  - "PortfolioSimulator pattern: dollar-based tracking with percentage weight targets"
  - "process_actions pattern: subtract-max softmax + deadzone filter for action conversion"
  - "RollingSharpeReward pattern: expanding warmup for first window-1 bars, then rolling"

requirements-completed: [TRAIN-07, TRAIN-09]

# Metrics
duration: 4min
completed: 2026-03-08
---

# Phase 6 Plan 01: Environment Foundation Summary

**PortfolioSimulator with dollar-based rebalancing, RollingSharpeReward with 20-day expanding warmup, and softmax+deadzone action processing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-08T03:24:08Z
- **Completed:** 2026-03-08T03:28:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 7

## Accomplishments
- PortfolioSimulator tracks cash, positions, portfolio value with trade logging and transaction cost deduction
- RollingSharpeReward computes expanding-window Sharpe for bars 1-19, rolling 20-day Sharpe from bar 20+
- process_actions converts raw agent output via numerically stable softmax with deadzone filter
- EnvironmentConfig added to SwingRLConfig with all training environment parameters
- 28 tests covering all components + 6 existing config tests still passing

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `ae85533` (test)
2. **Task 1 GREEN: Implementation** - `f37fc2c` (feat)

_TDD task: test committed first, then implementation._

## Files Created/Modified
- `src/swingrl/envs/portfolio.py` - PortfolioSimulator class and process_actions function
- `src/swingrl/envs/rewards.py` - RollingSharpeReward with expanding warmup
- `src/swingrl/envs/__init__.py` - Package docstring
- `src/swingrl/config/schema.py` - EnvironmentConfig model added to SwingRLConfig
- `config/swingrl.yaml` - Environment section with defaults
- `tests/test_envs.py` - 28 tests for portfolio, rewards, actions, config
- `tests/conftest.py` - New fixtures: equity/crypto features and prices arrays, environment config YAML

## Decisions Made
- Used `collections.deque(maxlen=window)` for RollingSharpeReward to get automatic rolling window behavior
- Safe price division with `np.where(prices > 0, prices, 1.0)` to prevent NaN in portfolio rebalancing
- Ran mypy without `--strict` flag to match project pyproject.toml configuration (strict=false)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff flagged unused variable in deadzone test on first commit attempt; fixed inline before re-commit

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- PortfolioSimulator, RollingSharpeReward, and process_actions ready for Plans 02 (StockTradingEnv) and 03 (CryptoTradingEnv)
- EnvironmentConfig provides all configurable parameters needed by both environments
- Test fixtures for synthetic feature/price arrays available for environment tests

---
*Phase: 06-rl-environments*
*Completed: 2026-03-08*
