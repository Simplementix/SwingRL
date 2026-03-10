---
phase: 11
slug: production-startup-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/scheduler/ tests/data/test_db.py -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/scheduler/ tests/data/test_db.py -v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | PAPER-01, PAPER-02 | unit | `uv run pytest tests/scheduler/test_main.py -x -k "init"` | Yes (needs update) | ⬜ pending |
| 11-01-02 | 01 | 1 | FEAT-11, DATA-09 | unit | `uv run pytest tests/data/test_db.py -x -k "feature"` | Partial | ⬜ pending |
| 11-01-03 | 01 | 1 | PAPER-12, DATA-04, FEAT-04 | unit | `uv run pytest tests/scheduler/test_jobs.py -x -k "fred"` | Needs new test | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/scheduler/test_main.py` — test for ExecutionPipeline receiving all 5 constructor args
- [ ] `tests/data/test_db.py` — test that init_schema() creates feature tables (features_equity, features_crypto)
- [ ] `tests/scheduler/test_jobs.py` — test for FRED import path resolution (correct module, class, and method)

*Existing infrastructure covers framework and fixtures; Wave 0 adds missing test stubs only.*

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
