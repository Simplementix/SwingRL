---
phase: 11-production-startup-wiring
verified: 2026-03-10T15:30:00Z
status: passed
score: 3/3 must-haves verified
re_verification: false
---

# Phase 11: Production Startup Wiring Verification Report

**Phase Goal:** The production entrypoint (main.py) starts without error, feature tables are initialized as part of standard DB setup, and scheduler jobs import from correct module paths
**Verified:** 2026-03-10T15:30:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | main.py build_app() completes without TypeError -- ExecutionPipeline receives all 5 required arguments | VERIFIED | `scripts/main.py:245-251` passes config, db, feature_pipeline, alerter, models_dir. FeaturePipeline constructed at line 243, Alerter at lines 233-240 (both before pipeline). Test `test_build_app_creates_pipeline_with_all_args` asserts all 5 kwargs. |
| 2 | DatabaseManager.init_schema() creates feature tables (features_equity, features_crypto, fundamentals, hmm_state_history) without requiring a prior compute_features.py run | VERIFIED | `src/swingrl/data/db.py:300-303` calls `init_feature_schema(cursor)` at end of `_init_duckdb_schema()` via local import. Test `TestFeatureTableInit.test_init_schema_creates_feature_tables` asserts all 4 tables present after init_schema(). |
| 3 | weekly_fundamentals_job and monthly_macro_job import FREDIngestor from the correct module path, use the correct class name, and call the correct method | VERIFIED | `src/swingrl/scheduler/jobs.py:304` and `328` both use `from swingrl.data.fred import FREDIngestor`. Constructor receives only `ctx.config` (not `ctx.config, ctx.db`). Method called is `run_all()` (not `refresh()`). Tests `TestFredImportPath` verify all three aspects via mock patching. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/main.py` | Fixed ExecutionPipeline constructor call with feature_pipeline, alerter, models_dir | VERIFIED | Lines 233-251: Alerter built first, FeaturePipeline constructed with DuckDB conn, ExecutionPipeline receives all 5 args. Contains "FeaturePipeline" import and usage. |
| `src/swingrl/data/db.py` | Feature table init wired into _init_duckdb_schema() | VERIFIED | Lines 300-303: local import of `init_feature_schema` and call with cursor. Contains "init_feature_schema". |
| `src/swingrl/scheduler/jobs.py` | Corrected FRED import path, class name, constructor args, and method call | VERIFIED | Lines 304 and 328: `from swingrl.data.fred import FREDIngestor`. No `type: ignore[import-not-found]` comments remain. |
| `tests/scheduler/test_main.py` | Updated job count assertion and ExecutionPipeline constructor test | VERIFIED | Line 54: `call_count == 11`. Lines 57-70: 11 expected job IDs. Lines 151-191: `test_build_app_creates_pipeline_with_all_args` checks all 5 kwargs. |
| `tests/scheduler/test_jobs.py` | FRED import path resolution tests | VERIFIED | Lines 252-301: `TestFredImportPath` class with `test_weekly_fundamentals_imports_fred_correctly` and `test_monthly_macro_imports_fred_correctly`. Both assert single-arg constructor and `run_all()` call. |
| `tests/data/test_db.py` | Feature table creation test | VERIFIED | Lines 244-254: `TestFeatureTableInit.test_init_schema_creates_feature_tables` asserts all 4 feature tables exist after `init_schema()`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/main.py` | `src/swingrl/execution/pipeline.py` | `ExecutionPipeline(config, db, feature_pipeline, alerter, models_dir)` | WIRED | main.py:245-251 passes all 5 keyword args. pipeline.py:55-62 constructor signature matches. |
| `src/swingrl/data/db.py` | `src/swingrl/features/schema.py` | `init_feature_schema(cursor)` call in `_init_duckdb_schema` | WIRED | db.py:301 imports and db.py:303 calls `init_feature_schema(cursor)`. schema.py:93 defines `init_feature_schema(conn: Any)`. |
| `src/swingrl/scheduler/jobs.py` | `src/swingrl/data/fred.py` | `from swingrl.data.fred import FREDIngestor` | WIRED | jobs.py:304 and 328 import from correct path. fred.py:47 defines `class FREDIngestor`. Constructor takes config only, method is `run_all()` at line 242. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PAPER-01 | 11-01-PLAN | Alpaca paper trading connection for equity environment | SATISFIED | ExecutionPipeline now receives all required args including feature_pipeline -- equity cycle can execute. |
| PAPER-02 | 11-01-PLAN | Binance.US simulated fills for crypto environment | SATISFIED | Same fix -- pipeline starts without TypeError, enabling crypto cycle execution. |
| PAPER-12 | 11-01-PLAN | APScheduler with all jobs registered | SATISFIED | 11 jobs registered (test asserts call_count == 11 with all IDs). |
| FEAT-11 | 11-01-PLAN | Per-environment feature tables in DuckDB | SATISFIED | init_schema() now creates features_equity, features_crypto, fundamentals, hmm_state_history. |
| DATA-09 | 11-01-PLAN | Store lowest, aggregate up strategy | SATISFIED | Feature tables created during init_schema() alongside aggregation views. |
| DATA-04 | 11-01-PLAN | FRED macro pipeline | SATISFIED | FRED jobs import from correct module path and call correct API. |
| FEAT-04 | 11-01-PLAN | Macro regime features shared | SATISFIED | FRED ingestor correctly wired into scheduler for weekly/monthly refresh. |

No orphaned requirements found -- all 7 IDs in REQUIREMENTS.md Phase 11 mapping match the PLAN.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | -- | -- | -- | No anti-patterns detected. No TODO/FIXME/PLACEHOLDER/HACK. No `type: ignore[import-not-found]` on FRED imports. No stub implementations. |

### Human Verification Required

### 1. Production Startup Smoke Test

**Test:** Run `python scripts/main.py --config config/swingrl.yaml` on homelab with Docker environment
**Expected:** Process starts without TypeError, logs "scheduler_started", and registers all 11 jobs
**Why human:** Requires real config file, database paths, and running APScheduler -- cannot verify programmatically in CI without a full integration environment

### 2. Feature Table Persistence After Restart

**Test:** Start container, verify feature tables exist in DuckDB, restart container, verify tables persist
**Expected:** All 4 feature tables (features_equity, features_crypto, fundamentals, hmm_state_history) survive container restart
**Why human:** Requires Docker container lifecycle testing with persistent volumes

### Gaps Summary

No gaps found. All three observable truths verified with supporting artifacts at all three levels (exists, substantive, wired). All 7 requirements satisfied. TDD discipline followed (RED commit 537e14a, GREEN commit e833175). No anti-patterns detected.

---

_Verified: 2026-03-10T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
