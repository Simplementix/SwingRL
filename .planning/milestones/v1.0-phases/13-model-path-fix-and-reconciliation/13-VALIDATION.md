---
phase: 13
slug: model-path-fix-and-reconciliation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/scheduler/test_main.py tests/scheduler/test_jobs.py tests/execution/test_pipeline.py -x -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/scheduler/test_main.py tests/scheduler/test_jobs.py tests/execution/test_pipeline.py -x -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | PAPER-02 | unit | `uv run pytest tests/execution/test_pipeline.py -k model_path -x` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | PAPER-02 | unit | `uv run pytest tests/scheduler/test_main.py::TestMainInitSequence::test_build_app_creates_pipeline_with_all_args -x` | ✅ | ⬜ pending |
| 13-01-03 | 01 | 1 | PAPER-09 | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob -x` | ❌ W0 | ⬜ pending |
| 13-01-04 | 01 | 1 | PAPER-09 | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob::test_skips_when_halted -x` | ❌ W0 | ⬜ pending |
| 13-01-05 | 01 | 1 | PAPER-09 | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob::test_consecutive_failures_escalate -x` | ❌ W0 | ⬜ pending |
| 13-01-06 | 01 | 1 | PAPER-09 | unit | `uv run pytest tests/scheduler/test_main.py::TestMainRegistersJobs -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/execution/test_pipeline.py::test_load_models_path_construction` — verifies `models_dir / "active" / env / algo / "model.zip"` (no double-active)
- [ ] `tests/scheduler/test_jobs.py::TestReconciliationJob` — new test class (3 tests: success, halt, consecutive failures)
- [ ] Update `tests/scheduler/test_main.py::TestMainRegistersJobs::test_main_registers_all_jobs` — change `call_count == 11` to `== 12`, add `"daily_reconciliation"` to `expected_ids`

*Existing infrastructure covers framework install — pytest already installed.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
