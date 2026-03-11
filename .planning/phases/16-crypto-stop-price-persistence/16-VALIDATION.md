---
phase: 16
slug: crypto-stop-price-persistence
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/execution/test_fill_processor.py tests/scheduler/test_stop_polling.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/execution/test_fill_processor.py tests/scheduler/test_stop_polling.py -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | PAPER-07 | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k stop` | ❌ W0 | ⬜ pending |
| 16-01-02 | 01 | 1 | PAPER-07 | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k partial_sell` | ❌ W0 | ⬜ pending |
| 16-01-03 | 01 | 1 | PAPER-07 | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k side` | ❌ W0 | ⬜ pending |
| 16-01-04 | 01 | 1 | PAPER-10 | integration | `uv run pytest tests/scheduler/test_stop_polling.py -v` | ❌ W0 | ⬜ pending |
| 16-01-05 | 01 | 1 | PAPER-10 | unit | `uv run pytest tests/execution/test_pipeline.py -v` | partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `TestStopTPPersistence` class in `tests/execution/test_fill_processor.py` — covers PAPER-07 (stop/TP write on buy, carry-forward on partial sell, side column)
- [ ] New test in `tests/scheduler/test_stop_polling.py` — end-to-end: fill processor writes stop/TP, stop_polling reads and triggers

*Existing test infrastructure covers config and fixture needs.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
