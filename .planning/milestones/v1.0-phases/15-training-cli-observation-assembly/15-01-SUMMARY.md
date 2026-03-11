---
phase: 15-training-cli-observation-assembly
plan: "01"
subsystem: training-cli
tags: [training, observation-assembly, duckdb, dry-run, tdd]
dependency_graph:
  requires:
    - "14-01: ObservationAssembler with equity_obs_dim/CRYPTO_OBS_DIM"
    - "05: features_equity / features_crypto DuckDB tables"
  provides:
    - "Working _load_features_prices() producing (N,156)/(N,45) observation arrays"
    - "--dry-run CLI flag for shape validation without training"
  affects:
    - "scripts/train.py: all callers of _load_features_prices"
tech_stack:
  added: []
  patterns:
    - "INNER JOIN features with OHLCV dates for row-aligned array extraction"
    - "Bulk macro/HMM pre-load to avoid N+1 queries per timestep"
    - "ObservationAssembler per-timestep assembly with default portfolio state"
key_files:
  created:
    - tests/test_phase15.py
  modified:
    - scripts/train.py
decisions:
  - "INNER JOIN between features_equity/features_crypto and OHLCV tables ensures date alignment (not outer join)"
  - "turbulence=0.0 default for training — circuit breaker usage, not observation quality"
  - "Bulk macro/HMM pre-load into dict keyed by date_str avoids N+1 DuckDB round-trips"
  - "Missing symbols filled with np.zeros(per_asset_dim) — graceful handling of sparse data"
  - "--dry-run placed before orchestrator construction so shape validation is pure I/O check"
  - "Removed unused CRYPTO_OBS_DIM / equity_obs_dim imports from train.py (assembler uses them internally)"
metrics:
  duration: "6 min 25 sec"
  completed: "2026-03-11"
  tasks_completed: 2
  files_changed: 2
  tests_added: 13
  tests_passing: 848
---

# Phase 15 Plan 01: Training CLI Observation Assembly Summary

**One-liner:** Fixed `_load_features_prices()` to use `ObservationAssembler` for pre-assembled `(N,156)`/`(N,45)` observation matrices plus `--dry-run` shape validation flag, closing INT-GAP-01.

## What Was Built

### Problem Closed: INT-GAP-01

The training CLI was passing raw DuckDB feature columns (15 equity / 13 crypto) directly to
`TrainingOrchestrator`, causing SB3 `observation_space` shape mismatch on model construction.
The `BaseTradingEnv` expects `(N, obs_dim)` arrays where `obs_dim=156` (equity) or `45` (crypto).

### Solution

Rewrote `_load_features_prices()` in `scripts/train.py` to:

1. **Added `config: SwingRLConfig` parameter** — provides symbol lists and sentiment flag needed to
   derive the correct observation dimension via `equity_obs_dim()`.

2. **Split into `_load_equity()` and `_load_crypto()` helpers** — each uses an INNER JOIN between
   the features table and OHLCV table so only dates present in both are included. This prevents
   `IndexError` from mismatched array lengths.

3. **Per-timestep assembly** — for each date/datetime, builds `per_asset` dict from feature rows,
   fetches bulk-preloaded macro (6-dim) and HMM (2-dim) context, and calls
   `assembler.assemble_equity()` / `assembler.assemble_crypto()` with `turbulence=0.0` and
   `portfolio_state=None` (defaults to 100% cash).

4. **Price extraction** — `ohlcv_daily` / `ohlcv_4h` close prices pivoted to `(N, n_symbols)`
   with alpha-sorted symbol column order matching the assembler's deterministic ordering.

5. **Added `--dry-run` flag** — placed before `TrainingOrchestrator` construction: loads features,
   validates shapes, logs results, and returns exit code 0 without training.

### Shape Guarantees

| Environment | Features | Prices |
|-------------|----------|--------|
| equity      | `(N, 156)` float32 | `(N, 8)` float32 |
| crypto      | `(N, 45)` float32  | `(N, 2)` float32 |

No hardcoded 156 or 45 — all dimensions derived from `equity_obs_dim(config.sentiment.enabled, len(config.equity.symbols))` and `CRYPTO_OBS_DIM` inside the assembler.

## TDD Execution

**RED commit:** `4965212` — 13 tests covering TRAIN-01/02/03 and VAL-01 requirements; all failed due to `TypeError: _load_features_prices() takes 2 positional arguments but 3 were given`

**GREEN commit:** `62de68d` — implementation passing all 13 tests; full suite of 848 tests passes

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Tests for observation assembly in train.py | `4965212` | `tests/test_phase15.py` |
| 2 (GREEN) | Fix _load_features_prices and add --dry-run flag | `62de68d` | `scripts/train.py` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Removed unused imports from train.py**
- **Found during:** Task 2, ruff lint check
- **Issue:** `CRYPTO_OBS_DIM` and `equity_obs_dim` were imported in `train.py` but not used directly — the assembler uses them internally
- **Fix:** Removed both from the import block
- **Files modified:** `scripts/train.py`
- **Commit:** `62de68d` (included in GREEN commit)

## Verification

All success criteria confirmed:

- `_load_features_prices("equity", ...)` returns `(N, 156)` features — PASS
- `_load_features_prices("crypto", ...)` returns `(N, 45)` features — PASS
- Prices arrays are `(N, 8)` equity / `(N, 2)` crypto, row-aligned — PASS
- `--dry-run` flag exists and skips training — PASS
- No hardcoded 156 or 45 — uses assembler dimension functions — PASS
- Full test suite: 848 passed, 0 failed — PASS
- `ruff check scripts/train.py tests/test_phase15.py` — no errors — PASS
- `mypy scripts/train.py` — no issues — PASS

## Self-Check: PASSED
