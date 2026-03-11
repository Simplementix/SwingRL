---
phase: 07-agent-training-and-validation
plan: 01
subsystem: agents
tags: [sharpe, sortino, calmar, rachev, drawdown, validation-gates, overfitting, duckdb]

# Dependency graph
requires:
  - phase: 04-storage-and-validation
    provides: DatabaseManager with DuckDB/SQLite dual-database layer
  - phase: 02-developer-experience
    provides: SwingRLError exception hierarchy
provides:
  - Performance metric calculators (Sharpe, Sortino, Calmar, Rachev, MDD, avg DD, DD duration, trade metrics)
  - Validation gates with 4-threshold pass/fail checking
  - Overfitting detection with healthy/marginal/reject classification
  - DuckDB DDL for model_metadata and backtest_results tables
affects: [07-02-training-pipeline, 07-03-backtesting, 08-paper-trading]

# Tech tracking
tech-stack:
  added: []
  patterns: [pure-function metric calculators with numpy, dataclass-based gate results]

key-files:
  created:
    - src/swingrl/agents/metrics.py
    - src/swingrl/agents/validation.py
    - tests/agents/test_metrics.py
    - tests/agents/test_validation.py
  modified:
    - src/swingrl/data/db.py

key-decisions:
  - "Sortino uses sqrt(mean(neg^2)) for downside deviation (full lower partial moment, not just std of negatives)"
  - "Boundary: gap==0.20 falls into marginal, gap==0.50 falls into marginal (strict < for healthy, strict > for reject)"
  - "compute_trade_metrics treats zero-PnL trades as losses for win_rate calculation"

patterns-established:
  - "Pure-function pattern: stateless metric calculators with numpy arrays, NaN for invalid inputs"
  - "GateResult dataclass: structured pass/fail with details dict for threshold reporting"

requirements-completed: [VAL-03, VAL-04, VAL-06, VAL-07, TRAIN-12, VAL-08]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 7 Plan 01: Metrics and Validation Summary

**Pure-math performance metrics (Sharpe, Sortino, Calmar, Rachev, MDD, trade metrics), 4-threshold validation gates, overfitting detector, and DuckDB model/backtest DDL**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08T13:51:55Z
- **Completed:** 2026-03-08T13:59:30Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- 8 pure-function performance metric calculators with full edge case handling (NaN for invalid inputs)
- Validation gates checking 4 thresholds (Sharpe>0.7, MDD<0.15, PF>1.5, overfit gap<0.20) with structured GateResult
- Overfitting detector classifying IS/OOS Sharpe gap into healthy/marginal/reject bands
- DuckDB DDL for model_metadata (13 columns) and backtest_results (22 columns) tables added to init_schema
- 50 tests covering all functions, edge cases, and boundary conditions

## Task Commits

Each task was committed atomically:

1. **Task 1: Performance metric calculators** (TDD)
   - RED: `da8ddfc` (test) - Failing tests for 8 metric functions
   - GREEN: `4476d1e` (feat) - Implementation of all metric calculators (from prior session)
2. **Task 2: Validation gates, overfitting detection, and DuckDB DDL** (TDD)
   - RED: `668c385` (test) - Failing tests for validation and DDL
   - GREEN: `85d9444` (feat) - Validation gates, overfitting detector, DDL extension

_Note: Task 1 implementation was committed in a prior session (4476d1e). Tests and verification confirmed correctness._

## Files Created/Modified
- `src/swingrl/agents/metrics.py` - 8 pure-function metric calculators (Sharpe, Sortino, Calmar, Rachev, MDD, avg DD, DD duration, trade metrics)
- `src/swingrl/agents/validation.py` - GateResult dataclass, diagnose_overfitting, check_validation_gates
- `src/swingrl/data/db.py` - Extended _init_duckdb_schema with model_metadata and backtest_results CREATE TABLE
- `tests/agents/test_metrics.py` - 31 tests for metric calculators
- `tests/agents/test_validation.py` - 19 tests for validation and DDL

## Decisions Made
- Sortino uses sqrt(mean(neg^2)) for downside deviation (full lower partial moment)
- Boundary classification: gap==0.20 is marginal (strict < for healthy), gap==0.50 is marginal (strict > for reject)
- compute_trade_metrics treats zero-PnL trades as losses for win_rate calculation
- Rachev ratio uses max(1, int(n * alpha)) for tail sample count to ensure at least 1 observation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed floating-point boundary test assertions**
- **Found during:** Task 2 (validation gate tests)
- **Issue:** Exact float equality (gap == 0.15) failed due to floating-point representation; boundary tests at 0.20 and 0.50 used IS/OOS values that caused ambiguous float gaps
- **Fix:** Changed to abs(gap - expected) < 1e-10 for near-equality; used IS/OOS values producing unambiguous gaps for boundary tests
- **Files modified:** tests/agents/test_validation.py
- **Verification:** All 19 validation tests pass
- **Committed in:** 85d9444

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test precision fix, no scope creep.

## Issues Encountered
- Pre-existing mypy error in callbacks.py (attr-defined on BaseCallback.last_mean_reward) and bandit assert warning -- both from prior session's Plan 07-02 work, out of scope for this plan

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Metrics and validation foundations ready for Plan 02 (training pipeline) and Plan 03 (backtesting)
- All metric functions importable from swingrl.agents.metrics
- Validation gates importable from swingrl.agents.validation
- DuckDB tables created via DatabaseManager.init_schema()

## Self-Check: PASSED

- All 5 created/modified files exist on disk
- All 4 commit hashes verified in git log

---
*Phase: 07-agent-training-and-validation*
*Completed: 2026-03-08*
