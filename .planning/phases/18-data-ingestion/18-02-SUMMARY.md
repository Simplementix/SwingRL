---
phase: 18-data-ingestion
plan: 02
subsystem: data
tags: [ingestion, orchestration, pipeline, cli, tdd, data-quality]

# Dependency graph
requires:
  - phase: 18-01
    provides: run_verification, write_report, print_summary, VerificationResult
  - phase: 03-data
    provides: AlpacaIngestor.run_all, BinanceIngestor.backfill/run_all, FREDIngestor.run_all
  - phase: 04-storage
    provides: DatabaseManager.duckdb() context manager
  - phase: 05-features
    provides: FeaturePipeline.compute_equity/compute_crypto
provides:
  - Single-command ingestion orchestrator: python -m swingrl.data.ingest_all --backfill
  - run_equity/run_crypto/run_macro stage functions with fail-fast DataError
  - resolve_crypto_gaps forward-fill (<=2 bars) with DataError for larger gaps
  - run_features conditional computation gate (skipped when rows_added==0)
  - run_pipeline sequential orchestration with verification final gate
  - CLI with --backfill and --config flags
affects: [19-training, 20-deployment, docker-startup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD red-green with per-commit isolation (RED at 03c9f55, GREEN at 7727978)
    - Explicit PYTHONPATH injection in subprocess tests for uv venv compatibility
    - nosec B608 for internal table name interpolation in DuckDB COUNT(*) helper

key-files:
  created:
    - src/swingrl/data/ingest_all.py
    - tests/data/test_ingest_all.py
  modified: []

key-decisions:
  - "Subprocess CLI tests inject PYTHONPATH=src/ explicitly because uv-managed Python does not process .pth files when spawned as subprocess"
  - "resolve_crypto_gaps uses pd.DataFrame.ffill(limit=2) — gaps >2 bars raise DataError requiring manual repair"
  - "run_pipeline returns int exit code (0/1), not None, enabling sys.exit() integration"
  - "Feature computation gate checks total rows across all three stages (equity+crypto+macro delta sum)"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04, DATA-05]

# Metrics
duration: 11min
completed: 2026-03-11
---

# Phase 18 Plan 02: Ingest All Orchestrator Summary

**Single-command ingestion pipeline wiring AlpacaIngestor, BinanceIngestor, FREDIngestor, FeaturePipeline, and verification gate into a fail-fast sequential orchestrator with --backfill CLI flag**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-11T11:44:20Z
- **Completed:** 2026-03-11T11:55:33Z
- **Tasks:** 2 (Task 1 TDD + Task 2 smoke tests)
- **Files created:** 2

## Accomplishments

- `ingest_all.py` (308 lines) with 8 public functions: `_count_rows`, `run_equity`, `run_crypto`, `run_macro`, `resolve_crypto_gaps`, `run_features`, `run_pipeline`, `main`
- Fail-fast pipeline: any ingestor returning failed symbols/series raises `DataError` immediately
- Conditional feature computation: `FeaturePipeline.compute_equity/compute_crypto` only run when `total_rows_added > 0`
- Verification always runs as final gate via `run_verification(config)` → `write_report` → `print_summary`
- `resolve_crypto_gaps` forward-fills up to 2 consecutive NaN bars; raises `DataError` for gaps >2 bars
- CLI: `python -m swingrl.data.ingest_all [--backfill] [--config PATH]`
- 19 tests covering all stage functions, gap resolution, pipeline flow, and CLI invocation
- Full suite: 892/892 tests pass, ruff/bandit/mypy clean

## Task Commits

| Task | Type | Commit | Description |
|------|------|--------|-------------|
| 1 RED | test | 03c9f55 | Failing tests for ingest_all orchestrator (19 tests) |
| 1 GREEN | feat | 7727978 | Implement ingest_all orchestrator + ruff/bandit fixes |
| 2 fix | fix | da35f11 | Fix import ordering and PYTHONPATH for subprocess tests |

## Files Created

- `src/swingrl/data/ingest_all.py` (308 lines) — Full orchestrator with CLI
- `tests/data/test_ingest_all.py` (637 lines) — 19 tests with all dependencies mocked

## Decisions Made

- Subprocess CLI tests use explicit `PYTHONPATH=src/` injection because uv-managed Python does not always process `.pth` editable install files when spawned as subprocesses. This is the same root cause documented in Plan 01's "Issues Encountered".
- `resolve_crypto_gaps` uses `pd.DataFrame.ffill(limit=2)`: forward-fill is the correct approach for 4H crypto data where brief exchange downtime creates 1-2 bar gaps that should be bridged from the last valid price.
- `run_pipeline` returns an int (0/1) rather than None so `sys.exit()` receives the correct exit code without extra logic in `main`.
- The `_count_rows` helper uses f-string interpolation for the table name (internal constant only, never user input) with `# noqa: S608 # nosec B608` to satisfy both ruff and bandit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed subprocess tests failing due to uv .pth non-processing**
- **Found during:** Task 2 smoke tests (full suite run)
- **Issue:** `subprocess.run([sys.executable, "-m", "swingrl.data.ingest_all", ...])` failed with `ModuleNotFoundError` in the full suite because uv-managed venv Python does not process `.pth` files for editable installs when spawned as a subprocess inside the pytest process
- **Fix:** Added `_subprocess_env()` helper that injects `PYTHONPATH=src/` into subprocess environment. Both `TestCliHelp` and `TestCliHelp2` now pass consistently across all 892 tests
- **Files modified:** tests/data/test_ingest_all.py
- **Commit:** da35f11

**2. [Rule 2 - Lint] Fixed unused imports and ruff E402 ordering**
- **Found during:** Pre-commit hook on GREEN commit
- **Issue:** Unused `contextmanager` and `Generator` imports in `ingest_all.py`; module-level constants placed before third-party imports in test file (E402)
- **Fix:** Removed unused imports from production file; moved `_REPO_ROOT`/`_SRC_PATH` constants below third-party imports in test file
- **Files modified:** src/swingrl/data/ingest_all.py, tests/data/test_ingest_all.py
- **Committed in:** 7727978, da35f11

**3. [Rule 2 - Security] Added nosec B608 for internal table name interpolation**
- **Found during:** Pre-commit bandit check on GREEN commit
- **Issue:** bandit B608 flagged `f"SELECT COUNT(*) FROM {table}"` as SQL injection vector
- **Fix:** Added `# nosec B608` (table is always an internal constant, never user input). Also retained `# noqa: S608` for ruff compatibility
- **Files modified:** src/swingrl/data/ingest_all.py
- **Committed in:** 7727978

---

**Total deviations:** 3 auto-fixed (Rules 1+2+2)
**Impact on plan:** All fixes were correctness/environment issues, not scope changes.

## Next Phase Readiness

- `python -m swingrl.data.ingest_all --backfill` runs the full pipeline when executed in Docker with proper env vars (ALPACA_API_KEY, ALPACA_SECRET_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, FRED_API_KEY)
- Plan 03 (or next phase) can invoke this command from the Docker entrypoint or Makefile
- Phase 18 is complete — verification gate + ingestion orchestrator both delivered

---
*Phase: 18-data-ingestion*
*Completed: 2026-03-11*

## Self-Check: PASSED

- FOUND: src/swingrl/data/ingest_all.py
- FOUND: tests/data/test_ingest_all.py
- FOUND: .planning/phases/18-data-ingestion/18-02-SUMMARY.md
- FOUND: commit 03c9f55 (test RED)
- FOUND: commit 7727978 (feat GREEN)
- FOUND: commit da35f11 (fix subprocess tests)
