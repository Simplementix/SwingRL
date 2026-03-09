---
phase: 9
slug: automation-and-monitoring
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 09-01-01 | 01 | 1 | PAPER-12 | unit | `uv run pytest tests/scheduler/test_jobs.py -x` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | PAPER-12 | unit | `uv run pytest tests/scheduler/test_halt_check.py -x` | ❌ W0 | ⬜ pending |
| 09-02-01 | 02 | 1 | PAPER-13 | unit | `uv run pytest tests/monitoring/test_embeds.py -x` | ❌ W0 | ⬜ pending |
| 09-02-02 | 02 | 1 | PAPER-13 | unit | `uv run pytest tests/monitoring/test_embeds.py::test_daily_summary_embed -x` | ❌ W0 | ⬜ pending |
| 09-02-03 | 02 | 1 | PAPER-13 | unit | `uv run pytest tests/monitoring/test_alerter.py::test_channel_routing -x` | ❌ W0 | ⬜ pending |
| 09-03-01 | 03 | 1 | PAPER-14 | unit | `uv run pytest tests/monitoring/test_stuck_agent.py -x` | ❌ W0 | ⬜ pending |
| 09-03-02 | 03 | 1 | PAPER-14 | unit | `uv run pytest tests/monitoring/test_stuck_agent.py::test_no_false_positive -x` | ❌ W0 | ⬜ pending |
| 09-04-01 | 04 | 2 | PAPER-15 | unit | `uv run pytest tests/dashboard/test_pages.py -x` | ❌ W0 | ⬜ pending |
| 09-05-01 | 05 | 1 | PAPER-16 | unit | `uv run pytest tests/monitoring/test_healthcheck_ping.py -x` | ❌ W0 | ⬜ pending |
| 09-05-02 | 05 | 1 | PAPER-16 | unit | `uv run pytest tests/monitoring/test_healthcheck_ping.py::test_no_ping_on_failure -x` | ❌ W0 | ⬜ pending |
| 09-06-01 | 06 | 1 | PAPER-17 | unit | `uv run pytest tests/monitoring/test_wash_sale.py -x` | ❌ W0 | ⬜ pending |
| 09-06-02 | 06 | 1 | PAPER-17 | unit | `uv run pytest tests/monitoring/test_wash_sale.py::test_crypto_exempt -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/scheduler/__init__.py` — package init
- [ ] `tests/scheduler/test_jobs.py` — stubs for PAPER-12 job registration and halt checks
- [ ] `tests/scheduler/test_halt_check.py` — stubs for emergency flag CRUD
- [ ] `tests/monitoring/test_embeds.py` — stubs for PAPER-13 embed builders
- [ ] `tests/monitoring/test_alerter.py` — stubs for PAPER-13 channel routing
- [ ] `tests/monitoring/test_stuck_agent.py` — stubs for PAPER-14
- [ ] `tests/monitoring/test_healthcheck_ping.py` — stubs for PAPER-16
- [ ] `tests/monitoring/test_wash_sale.py` — stubs for PAPER-17
- [ ] `tests/dashboard/__init__.py` — package init
- [ ] `tests/dashboard/test_pages.py` — stubs for PAPER-15 (Streamlit AppTest)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Discord webhook message formatting | PAPER-13 | Requires live Discord channel | Send test embed to dev webhook URL, verify appearance |
| Healthchecks.io ping receipt | PAPER-16 | Requires live HC account | Check HC dashboard shows green after test ping |
| Streamlit dashboard visual layout | PAPER-15 | Visual inspection needed | Run `streamlit run src/swingrl/dashboard/app.py`, verify traffic-light rendering |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
