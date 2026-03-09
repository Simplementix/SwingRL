---
phase: 9
slug: automation-and-monitoring
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-09
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | PAPER-12, PAPER-16 | unit | `uv run pytest tests/scheduler/test_halt_check.py tests/scheduler/test_healthcheck_ping.py -x` | Yes (TDD) | pending |
| 09-01-02 | 01 | 1 | PAPER-12 | unit | `uv run pytest tests/scheduler/test_jobs.py -x` | Yes (TDD) | pending |
| 09-02-01 | 02 | 1 | PAPER-13, PAPER-14 | unit | `uv run pytest tests/monitoring/test_embeds.py tests/monitoring/test_stuck_agent.py -x` | Yes (TDD) | pending |
| 09-02-02 | 02 | 1 | PAPER-13, PAPER-17 | unit | `uv run pytest tests/monitoring/test_alerter.py tests/monitoring/test_wash_sale.py -x` | Yes (TDD) | pending |
| 09-03-01 | 03 | 1 | PAPER-15 | unit | `uv run pytest tests/dashboard/test_pages.py -x` | Yes (TDD) | pending |
| 09-03-02 | 03 | 1 | PAPER-15 | unit | `uv run pytest tests/dashboard/test_pages.py -x` | Yes (TDD) | pending |
| 09-04-01 | 04 | 2 | PAPER-12, PAPER-16 | unit | `uv run pytest tests/scheduler/test_main.py tests/scheduler/test_stop_polling.py -x` | Yes (TDD) | pending |
| 09-04-02 | 04 | 2 | PAPER-15 | config | `grep 'main.py' Dockerfile && grep 'swingrl-dashboard' docker-compose.prod.yml` | N/A | pending |
| 09-04-03 | 04 | 2 | PAPER-12,13,14,15,16,17 | integration + checkpoint | `uv run pytest tests/ -v` | N/A | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

All test files are created by their respective TDD tasks (RED phase creates the test file first):

- [x] `tests/scheduler/__init__.py` — created by Plan 01 Task 1
- [x] `tests/scheduler/test_halt_check.py` — created by Plan 01 Task 1 (TDD RED)
- [x] `tests/scheduler/test_healthcheck_ping.py` — created by Plan 01 Task 1 (TDD RED)
- [x] `tests/scheduler/test_jobs.py` — created by Plan 01 Task 2 (TDD RED)
- [x] `tests/monitoring/test_embeds.py` — created by Plan 02 Task 1 (TDD RED)
- [x] `tests/monitoring/test_stuck_agent.py` — created by Plan 02 Task 1 (TDD RED)
- [x] `tests/monitoring/test_alerter.py` — created by Plan 02 Task 2 (TDD RED)
- [x] `tests/monitoring/test_wash_sale.py` — created by Plan 02 Task 2 (TDD RED)
- [x] `tests/dashboard/__init__.py` — created by Plan 03 Task 2
- [x] `tests/dashboard/test_pages.py` — created by Plan 03 Task 2 (with behavioral tests)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Discord webhook message formatting | PAPER-13 | Requires live Discord channel | Send test embed to dev webhook URL, verify appearance |
| Healthchecks.io ping receipt | PAPER-16 | Requires live HC account | Check HC dashboard shows green after test ping |
| Streamlit dashboard visual layout | PAPER-15 | Visual inspection needed | Run `streamlit run dashboard/app.py`, verify traffic-light rendering |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
