---
phase: 06-rl-environments
verified: 2026-03-08T04:00:00Z
status: passed
score: 5/5 success criteria verified
re_verification: false
---

# Phase 6: RL Environments Verification Report

**Phase Goal:** Gymnasium-compatible trading environments for both equity and crypto pass the step/reset contract and produce valid observations and rewards
**Verified:** 2026-03-08T04:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | gym.make('StockTradingEnv-v0') initializes and env.reset() returns (156,) finite observation | VERIFIED | test_gym_make_equity, test_reset_returns_obs_info, test_reset_obs_all_finite all pass; Gymnasium registration in __init__.py with register() |
| 2 | gym.make('CryptoTradingEnv-v0') initializes and env.reset() returns (45,) finite observation | VERIFIED | test_gym_make_crypto, TestCryptoTradingEnvReset tests all pass; CryptoTradingEnv registered in __init__.py |
| 3 | Rolling 20-day Sharpe reward returns expanding-window warmup for first 19 bars, then proper Sharpe | VERIFIED | test_first_return_is_zero, test_expanding_window_warmup, test_expanding_window_nonzero, test_rolling_window_from_bar_20 all pass; RollingSharpeReward uses deque(maxlen=20) |
| 4 | Actions within +/-0.02 of zero result in hold (no trade) | VERIFIED | test_signal_deadzone_hold passes; process_actions applies deadzone filter, test_deadzone_suppresses_small_changes confirms threshold behavior |
| 5 | Equity episodes run 252 steps; crypto episodes run 540 steps with random start -- confirmed by 10 episodes each | VERIFIED | test_equity_10_episodes_252_steps and test_crypto_10_episodes_540_steps both pass; step_count-based termination in BaseTradingEnv |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/swingrl/envs/portfolio.py` | PortfolioSimulator with rebalance, portfolio_value, reset, trade logging | VERIFIED | 163 lines, full implementation with dollar-based tracking, trade logging, safe division |
| `src/swingrl/envs/rewards.py` | RollingSharpeReward with warmup and 20-day window | VERIFIED | 55 lines, deque-based rolling window, expanding warmup, near-zero std guard |
| `src/swingrl/envs/base.py` | BaseTradingEnv(gymnasium.Env) with shared step/reset, portfolio state injection, risk penalties | VERIFIED | 362 lines, full Gymnasium contract, observation building, soft risk penalties |
| `src/swingrl/envs/equity.py` | StockTradingEnv for 8 ETFs, 252-step episodes | VERIFIED | 105 lines, subclass with from_arrays factory, sequential start |
| `src/swingrl/envs/crypto.py` | CryptoTradingEnv with random start, 540-step episodes | VERIFIED | 114 lines, np_random-based random start, from_arrays factory |
| `src/swingrl/envs/__init__.py` | Gymnasium registration for both environments | VERIFIED | 28 lines, register() calls before imports, __all__ exports |
| `src/swingrl/config/schema.py` | EnvironmentConfig in SwingRLConfig | VERIFIED | EnvironmentConfig at line 155 with all 8 fields, wired into SwingRLConfig at line 201 |
| `config/swingrl.yaml` | Environment section with defaults | VERIFIED | environment: section present with all 8 config values |
| `tests/test_envs.py` | Tests for all components | VERIFIED | 1103 lines, 67 tests covering portfolio, rewards, actions, config, both envs, registration, SB3 check_env, episode structure, integration |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| base.py | portfolio.py | `from swingrl.envs.portfolio import PortfolioSimulator, process_actions` | WIRED | Line 20; used in __init__ and step() |
| base.py | rewards.py | `from swingrl.envs.rewards import RollingSharpeReward` | WIRED | Line 21; used in __init__ and step() |
| base.py | assembler.py | `from swingrl.features.assembler import EQUITY_OBS_DIM, CRYPTO_OBS_DIM, EQUITY_PORTFOLIO, CRYPTO_PORTFOLIO` | WIRED | Lines 22-27; used in __init__ for space/state dimensions |
| equity.py | base.py | `class StockTradingEnv(BaseTradingEnv)` | WIRED | Line 20; inherits and calls super().__init__(environment="equity") |
| crypto.py | base.py | `class CryptoTradingEnv(BaseTradingEnv)` | WIRED | Line 21; inherits and overrides _select_start_step with random start |
| __init__.py | gymnasium | `register(id="StockTradingEnv-v0", ...) + register(id="CryptoTradingEnv-v0", ...)` | WIRED | Lines 15-22; gym.make() tested and passing |
| test_envs.py | portfolio.py | `from swingrl.envs.portfolio import PortfolioSimulator, process_actions` | WIRED | Direct import and 16+ tests |
| test_envs.py | equity.py, crypto.py | `from swingrl.envs.equity import StockTradingEnv` / `from swingrl.envs.crypto import CryptoTradingEnv` | WIRED | Direct imports with full env lifecycle tests |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAIN-01 | 06-02 | StockTradingEnv -- Gymnasium-compatible, daily bars, 8 ETFs, 156-dim observation | SATISFIED | StockTradingEnv passes check_env, obs shape (156,), action space (8,) |
| TRAIN-02 | 06-03 | CryptoTradingEnv -- Gymnasium-compatible, 4H bars, BTC/ETH, 45-dim observation | SATISFIED | CryptoTradingEnv passes check_env, obs shape (45,), action space (2,) |
| TRAIN-07 | 06-01 | Rolling 20-day Sharpe ratio reward with expanding-window warmup | SATISFIED | RollingSharpeReward with deque(maxlen=20), 6 dedicated tests |
| TRAIN-08 | 06-02, 06-03 | VecNormalize stats -- frozen during inference (raw obs from env) | SATISFIED | Observations returned raw (not normalized), test_raw_observations_not_normalized passes, docstring in base.py references TRAIN-08 |
| TRAIN-09 | 06-01, 06-03 | Signal deadzone: +/-0.02 maps to hold | SATISFIED | process_actions deadzone filter, test_signal_deadzone_hold integration test |
| TRAIN-10 | 06-02, 06-03 | Adaptive validation windows: turbulence exposed | SATISFIED | info dict contains turbulence key, test_info_dict_turbulence passes; _get_turbulence() returns 0.0 default (actual turbulence data passed via features for Phase 7) |
| TRAIN-11 | 06-02, 06-03 | Episode structure: equity 252-day, crypto 540 4H bars with random start | SATISFIED | 10-episode rollouts confirm exact lengths, crypto uses np_random.integers for random start |

No orphaned requirements found -- all 7 IDs from ROADMAP Phase 6 are covered by plan requirements fields.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected in any env source files |

No TODO, FIXME, placeholder, stub, or empty implementation patterns found in any of the 6 source files.

### Human Verification Required

None required. All success criteria are programmatically verifiable and have been verified via the 67-test suite including SB3 check_env() validation.

### Gaps Summary

No gaps found. All 5 success criteria verified, all 9 artifacts exist and are substantive, all 8 key links wired, all 7 requirements satisfied, and no anti-patterns detected. 67/67 tests pass including SB3 check_env for both environments and 10-episode rollout stress tests.

---

_Verified: 2026-03-08T04:00:00Z_
_Verifier: Claude (gsd-verifier)_
