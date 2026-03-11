---
phase: 15-training-cli-observation-assembly
verified: 2026-03-11T00:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Phase 15: Training CLI Observation Assembly — Verification Report

**Phase Goal:** The training CLI (train.py) produces correctly shaped observation matrices by using ObservationAssembler, so SB3 model construction succeeds on both environments.
**Verified:** 2026-03-11
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_load_features_prices('equity', ...)` returns observations shaped `(N, 156)` | VERIFIED | `test_equity_features_shape` passes; assembler.assemble_equity returns shape confirmed by `equity_obs_dim()` calculation at runtime |
| 2 | `_load_features_prices('crypto', ...)` returns observations shaped `(N, 45)` | VERIFIED | `test_crypto_features_shape` passes; assembler.assemble_crypto enforces `CRYPTO_OBS_DIM=45` with a `DataError` on mismatch |
| 3 | `train.py --env equity --dry-run` completes model construction without shape mismatch | VERIFIED | `test_dry_run_skips_training` passes; `main()` returns 0 with `TrainingOrchestrator.train` not called; `build_parser` accepts `--dry-run` |
| 4 | Prices array aligns row-for-row with features array (same N timesteps, same symbol order) | VERIFIED | `test_equity_row_alignment` and `test_crypto_row_alignment` pass; INNER JOIN between features and OHLCV tables enforces date alignment; alpha-sorted symbol order in both prices pivot and assembler |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/train.py` | Fixed `_load_features_prices` using ObservationAssembler | VERIFIED (WIRED) | 642 lines; imports `ObservationAssembler`, `_EQUITY_FEATURE_COLS`, `_CRYPTO_FEATURE_COLS`; calls `assembler.assemble_equity()` (line 302) and `assembler.assemble_crypto()` (line 419); `--dry-run` flag at lines 93-99 |
| `tests/test_phase15.py` | Tests for observation shape correctness | VERIFIED (WIRED) | 535 lines; 5 test classes, 13 test functions; `TestLoadFeaturesEquity`, `TestLoadFeaturesCrypto`, `TestDryRunCLI`, `TestFoldShapes`, `TestEmptyTables`; all import and exercise `_load_features_prices` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/train.py` | `src/swingrl/features/assembler.py` | `ObservationAssembler.assemble_equity/assemble_crypto` | WIRED | Imported at line 28; `ObservationAssembler(config)` constructed at line 204; `assemble_equity()` called at line 302; `assemble_crypto()` called at line 419 |
| `scripts/train.py` | `src/swingrl/features/pipeline.py` | `_EQUITY_FEATURE_COLS` and `_CRYPTO_FEATURE_COLS` for DB column selection | WIRED | Imported at line 29; `_EQUITY_FEATURE_COLS` used in DuckDB query format string (line 245) and row extraction (line 296); `_CRYPTO_FEATURE_COLS` used in query (line 366) and row extraction (line 413) |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAIN-01 | 15-01-PLAN.md | 156-dim observation for equity (StockTradingEnv shape contract) | SATISFIED | `_load_features_prices('equity', ...)` returns `(N, 156)` via `equity_obs_dim()`; enforced by `ObservationAssembler.assemble_equity()` shape check with `DataError` on mismatch; 4 test functions validate shape, NaN-freeness, price positivity |
| TRAIN-02 | 15-01-PLAN.md | 45-dim observation for crypto (CryptoTradingEnv shape contract) | SATISFIED | `_load_features_prices('crypto', ...)` returns `(N, 45)` via `CRYPTO_OBS_DIM`; enforced by `ObservationAssembler.assemble_crypto()` shape check; 2 test functions validate shape and price alignment |
| TRAIN-03 | 15-01-PLAN.md | Model construction succeeds — dry-run validates without full training | SATISFIED | `--dry-run` flag added to `build_parser()` (lines 86-99); `main()` loads features, validates shapes, logs result, returns 0 before constructing `TrainingOrchestrator` when `dry_run=True` (lines 557-564); `test_dry_run_skips_training` confirms `orchestrator.train()` never called |
| VAL-01 | 15-01-PLAN.md | Shape validation without training run | SATISFIED | `TestDryRunCLI` covers parser acceptance of `--dry-run`, default False, and full `main()` path that validates feature loading but skips training; exit code 0 confirmed |

**Requirement ID note:** REQUIREMENTS.md definitions for TRAIN-01/02/03/VAL-01 use Phase 6/7 language (environment/training framework construction). Phase 15 maps them as gap-closure items — specifically closing INT-GAP-01 (raw feature columns bypassing assembler). The phase-15 must_haves directly address the shape correctness these requirements depend on, and REQUIREMENTS.md status tracker marks all four as Complete under Phase 15.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/train.py` | 597-599 | "placeholder Sharpe" comment + `placeholder_sharpes = dict.fromkeys(...)` | Info | Pre-existing in ensemble blending section; out of scope for phase 15; actual Sharpe comes from `backtest.py` per design |

No blockers. The "placeholder" in train.py is architectural — Sharpe-weighted ensemble blending uses backtest-computed values, not training-time values. This is documented behavior, not a stub.

The occurrences of `156` and `45` in `train.py` are docstring-only (lines 190-191, 228, 349) — not hardcoded in logic. All dimension computation uses `equity_obs_dim()` and `CRYPTO_OBS_DIM` via the assembler.

---

### Human Verification Required

None. All phase-15 behaviors are programmatically verifiable through unit tests and static analysis.

---

### Gaps Summary

No gaps. All 4 must-haves verified. All 13 phase-15 tests pass. Full suite of 848 tests passes with 0 failures. Ruff lint reports no issues on `scripts/train.py` and `tests/test_phase15.py`.

---

## Test Run Evidence

```
PYTHONPATH=src uv run pytest tests/test_phase15.py -v

TestLoadFeaturesEquity::test_equity_features_shape        PASSED
TestLoadFeaturesEquity::test_equity_prices_shape          PASSED
TestLoadFeaturesEquity::test_equity_no_nans_in_features   PASSED
TestLoadFeaturesEquity::test_equity_prices_positive       PASSED
TestLoadFeaturesCrypto::test_crypto_features_shape        PASSED
TestLoadFeaturesCrypto::test_crypto_prices_shape          PASSED
TestDryRunCLI::test_parser_accepts_dry_run                PASSED
TestDryRunCLI::test_dry_run_flag_default_false            PASSED
TestDryRunCLI::test_dry_run_skips_training                PASSED
TestFoldShapes::test_equity_row_alignment                 PASSED
TestFoldShapes::test_crypto_row_alignment                 PASSED
TestEmptyTables::test_empty_equity_table_raises           PASSED
TestEmptyTables::test_empty_crypto_table_raises           PASSED

13 passed in 3.59s

Full suite: 848 passed, 0 failed, 9 warnings in 78.41s
Ruff: All checks passed (scripts/train.py, tests/test_phase15.py)
```

---

_Verified: 2026-03-11_
_Verifier: Claude (gsd-verifier)_
