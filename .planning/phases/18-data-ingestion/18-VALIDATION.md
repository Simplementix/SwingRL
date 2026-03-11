---
phase: 18
slug: data-ingestion
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (>=7.x, configured via pyproject.toml) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/data/test_verification.py tests/data/test_ingest_all.py -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run the relevant test file `-v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | DATA-04 | unit (tmp DuckDB) | `uv run pytest tests/data/test_verification.py::test_check_equity_rows_missing -x` | W0 (Plan 01 Task 1) | pending |
| 18-01-02 | 01 | 1 | DATA-04 | unit (tmp DuckDB) | `uv run pytest tests/data/test_verification.py::test_check_crypto_gaps_detected -x` | W0 (Plan 01 Task 2) | pending |
| 18-01-03 | 01 | 1 | DATA-04 | unit (mock) | `uv run pytest tests/data/test_verification.py::test_check_obs_vector_nan -x` | W0 (Plan 01 Task 2) | pending |
| 18-01-04 | 01 | 1 | DATA-04 | unit | `uv run pytest tests/data/test_verification.py::test_write_report_json -x` | W0 (Plan 01 Task 1) | pending |
| 18-01-05 | 01 | 1 | DATA-04 | unit | `uv run pytest tests/data/test_verification.py::test_run_verification_aggregates -x` | W0 (Plan 01 Task 2) | pending |
| 18-02-01 | 02 | 2 | DATA-01 | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_equity_backfill -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-02 | 02 | 2 | DATA-02 | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_crypto_backfill -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-03 | 02 | 2 | DATA-03 | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_macro_backfill -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-04 | 02 | 2 | DATA-04 | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_verification_nan_detection -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-05 | 02 | 2 | DATA-04 | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_verification_obs_shapes -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-06 | 02 | 2 | DATA-05 | smoke | `uv run python -m swingrl.data.ingest_all --help` | W0 (Plan 02 Task 1) | pending |
| 18-02-07 | 02 | 2 | DATA-01-03 | unit | `uv run pytest tests/data/test_ingest_all.py::test_fail_fast_on_equity_error -x` | W0 (Plan 02 Task 1) | pending |
| 18-02-08 | 02 | 2 | DATA-01-03 | unit | `uv run pytest tests/data/test_ingest_all.py::test_skip_features_on_no_new_data -x` | W0 (Plan 02 Task 1) | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/data/test_verification.py` — stubs for DATA-04 checks (Plan 01 creates this)
- [ ] `tests/data/test_ingest_all.py` — stubs for DATA-01 through DATA-05 (Plan 02 creates this)
- [ ] `src/swingrl/data/verification.py` — verification module (Plan 01 creates this)
- [ ] `src/swingrl/data/ingest_all.py` — the orchestrator module (Plan 02 creates this)

*Existing infrastructure covers framework and fixture needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Homelab Docker exec runs full pipeline | DATA-05 | Requires homelab hardware + Docker | `ssh homelab` then `docker exec swingrl python -m swingrl.data.ingest_all --backfill` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
