---
phase: 19-model-training
plan: 03
subsystem: training
tags: [memory-agent, duckdb, sqlite, validation, seeding, backtest-fills, hmm, schema]

# Dependency graph
requires:
  - phase: 19-02
    provides: "train_pipeline.py, 3-state HMM, memory training modules, model deployment"
  - phase: 19-01
    provides: "MemoryClient, bounds.py, pipeline_helpers.py, MemoryAgentConfig"

provides:
  - "scripts/validate_memory.py: pre-live DuckDB schema and memory agent validation CLI (exit 0/1)"
  - "scripts/seed_memory_from_backtest.py: seeds memory agent with SQLite evaluation fills + DuckDB HMM regime"
  - "Training pipeline running on homelab (in progress — Task 2 checkpoint active)"

affects:
  - "phase-20-deployment: validate_memory.py run before enabling live endpoints; models/active/ consumed"
  - "phase-22-retraining: seed_memory_from_backtest.py run after each retrain evaluation"
  - "phase-23-monitoring: training_report.json provides ensemble gate result baseline"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Exit-0/Exit-1 validation pattern: validate_memory.py returns machine-parseable pass/fail"
    - "Fail-open seeding: seed_memory_from_backtest.py exits 0 (not 1) when no fills exist"
    - "Inline import consolidation: no blank lines between consecutive first-party imports in test methods"

key-files:
  created:
    - scripts/validate_memory.py
    - scripts/seed_memory_from_backtest.py
  modified:
    - src/swingrl/features/schema.py
    - tests/training/test_pipeline_helpers.py

key-decisions:
  - "validate_memory.py exits 1 with diagnostic for any schema gap — includes missing tables, missing p_crisis column, missing meta columns; exits 0 only when all checks pass"
  - "seed_memory_from_backtest.py exits 0 when no fills exist (expected pre-training state); only exits 1 on actual ingest failures"
  - "hmm_state_history base DDL in schema.py updated to include p_crisis DOUBLE DEFAULT 0.0 — fixes BinderException when compute_equity() tries to INSERT the 3-state HMM output"
  - "Training initiated on homelab via docker compose run --rm in background; CI passed before training launch"

patterns-established:
  - "Pattern 1: validate_memory.py is the canonical pre-live check gate — run before enabling live endpoints"
  - "Pattern 2: seed_memory_from_backtest.py gracefully skips if fills table absent — safe to call before first trade"

requirements-completed:
  - TRAIN-01
  - TRAIN-02
  - TRAIN-03
  - TRAIN-06
  - TRAIN-07

# Metrics
duration: 50min
completed: 2026-03-13
---

# Phase 19 Plan 03: Memory Validation Scripts + Homelab Training Summary

**DuckDB schema validation CLI (validate_memory.py), backtest fill seeder (seed_memory_from_backtest.py), schema DDL p_crisis fix, homelab CI pass, and training pipeline launched in background**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-03-13T15:16:31Z
- **Completed:** 2026-03-13T16:06:18Z (Task 1 complete; Task 2 checkpoint — training running)
- **Tasks:** 1 complete, 1 at checkpoint (training in progress on homelab)
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments

- `validate_memory.py`: exit 0/1 schema validation for DuckDB memory agent tables; checks hmm_state_history 3-state p_crisis column, training_runs meta columns, cold-start guard count; optional --memory-url for full connectivity validation per memory_storage_spec.md Section 8
- `seed_memory_from_backtest.py`: queries evaluation fills from SQLite joined with DuckDB HMM regime, formats as plain text per Section 4 spec, ingests via MemoryClient; triggers consolidation; gracefully handles missing tables (exit 0)
- Fixed pre-existing schema bug: `_HMM_STATE_HISTORY_DDL` lacked `p_crisis` column (3-state HMM from 19-02 never reflected in base DDL) — was causing `BinderException` in `test_compute_equity_runs`; 981 tests now all pass
- Fixed ruff I001 import ordering in `tests/training/test_pipeline_helpers.py` — homelab CI was failing on lint step
- Homelab CI passed; training pipeline launched via `docker compose run --rm` in background

## Task Commits

1. **Task 1: Memory validation and backtest seeding scripts** - `c2ddae2` (feat)
2. **Auto-fix: ruff I001 import ordering in test_pipeline_helpers** - `275d6df` (fix)
3. **Task 2: Homelab training** - IN PROGRESS (checkpoint reached; training running on homelab)

## Files Created/Modified

- `scripts/validate_memory.py` — Schema-only DuckDB validation + optional memory agent connectivity checks; follows memory_storage_spec.md Section 8 curl checks
- `scripts/seed_memory_from_backtest.py` — SQLite fills + DuckDB HMM regime join, formats per Section 4, ingest via MemoryClient.ingest(), triggers consolidate()
- `src/swingrl/features/schema.py` — Added `p_crisis DOUBLE DEFAULT 0.0` to `_HMM_STATE_HISTORY_DDL` base DDL (was missing since 3-state expansion in 19-02)
- `tests/training/test_pipeline_helpers.py` — Removed blank lines between consecutive inline imports in TestDecideFinalTimesteps methods to fix ruff I001

## Decisions Made

- `validate_memory.py` schema-only mode requires no memory agent connectivity — valid use case for pre-training DB checks
- `seed_memory_from_backtest.py` falls back to ingesting all fills (no `run_type = 'evaluation'` filter) when `run_type` column is missing — graceful degradation if ALTER TABLE migration hasn't been applied
- Training launched via `docker compose run --rm --no-deps --entrypoint '' swingrl uv run python scripts/train_pipeline.py --env all` (not `docker exec`) since swingrl container is not yet long-running (Docker stack deployment is Phase 20)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing p_crisis column in hmm_state_history base DDL**
- **Found during:** Task 1 verification (full test suite run)
- **Issue:** `_HMM_STATE_HISTORY_DDL` in `schema.py` only had `p_bull` and `p_bear` columns. After 19-02 expanded HMM to 3 states, `hmm_regime.py` INSERT now includes `p_crisis` but the CREATE TABLE DDL never added it. Test `test_compute_equity_runs` was failing with `BinderException: Table "hmm_state_history" does not have a column with name "p_crisis"`
- **Fix:** Added `p_crisis DOUBLE DEFAULT 0.0` to `_HMM_STATE_HISTORY_DDL`
- **Files modified:** `src/swingrl/features/schema.py`
- **Verification:** `test_compute_equity_runs` now passes; 981/981 tests pass
- **Committed in:** `c2ddae2` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed ruff I001 import ordering in test_pipeline_helpers.py**
- **Found during:** Homelab CI Stage 4 (lint check)
- **Issue:** Three test methods in `TestDecideFinalTimesteps` had blank lines separating consecutive `from swingrl.training.*` imports, causing ruff to treat them as separate import blocks and report I001 import-sort violations
- **Fix:** Removed blank lines to merge into single import blocks
- **Files modified:** `tests/training/test_pipeline_helpers.py`
- **Verification:** `ruff check tests/training/test_pipeline_helpers.py` passes; homelab CI Stage 4 passes
- **Committed in:** `275d6df`

---

**Total deviations:** 2 auto-fixed (2 Rule 1 - Bug)
**Impact on plan:** Both auto-fixes necessary for correctness and CI compliance. No scope creep.

## Issues Encountered

- `swingrl` Docker container is not long-running yet (Phase 20 deploys the stack). Training was launched via `docker compose run --rm` instead of `docker exec swingrl`. Training container is named `swingrl-swingrl-run-74ed8b9de020` and is running as of 2026-03-13T16:04Z.
- Training log: `~/swingrl/logs/training_pipeline.log` on homelab

## User Setup Required

**Training is running on homelab in background.** Verify completion:
```bash
# Check training is still running
ssh homelab "docker ps --filter name=swingrl"

# Monitor log
ssh homelab "tail -f ~/swingrl/logs/training_pipeline.log"

# After training completes (12-48h), verify models:
ssh homelab "ls ~/swingrl/models/active/equity/*/model.zip ~/swingrl/models/active/crypto/*/model.zip | wc -l"
# Expected: 6

# Check training report
ssh homelab "python3 -c \"import json; r=json.load(open('swingrl/data/training_report.json')); print('equity gate:', r.get('equity', {}).get('ensemble_gate')); print('crypto gate:', r.get('crypto', {}).get('ensemble_gate'))\""
```

## Next Phase Readiness

- `validate_memory.py` is ready to run post-training; will pass once training creates the required DuckDB tables
- `seed_memory_from_backtest.py` is ready to run once training fills the SQLite fills table
- Training is running on homelab; Phase 20 (Paper Trading) can begin once training completes and ensemble gate passes
- All 981 tests passing locally and on homelab CI

## Self-Check: PASSED

Files created:
- `scripts/validate_memory.py` — present
- `scripts/seed_memory_from_backtest.py` — present

Files modified:
- `src/swingrl/features/schema.py` — present
- `tests/training/test_pipeline_helpers.py` — present

Commits:
- `c2ddae2` — Task 1 (feat: validate + seed scripts)
- `275d6df` — Auto-fix (ruff I001)

---
*Phase: 19-model-training*
*Completed: 2026-03-13 (Task 1; Task 2 in progress)*
