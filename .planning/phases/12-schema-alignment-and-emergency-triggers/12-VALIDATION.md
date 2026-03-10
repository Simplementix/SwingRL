---
phase: 12
slug: schema-alignment-and-emergency-triggers
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_emergency_stop.py tests/scheduler/test_stop_polling.py -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~360 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_emergency_stop.py tests/scheduler/test_stop_polling.py -v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | PAPER-10 | unit | `uv run pytest tests/scheduler/test_stop_polling.py -v -x` | ✅ (needs update) | ⬜ pending |
| 12-01-02 | 01 | 1 | PROD-07 | unit | `uv run pytest tests/test_emergency_stop.py::TestAutomatedTriggers -v -x` | ✅ (needs update) | ⬜ pending |
| 12-01-03 | 01 | 1 | PROD-07 | unit | `uv run pytest tests/test_emergency_stop.py -v -x` | ✅ (mock-based) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Update `tests/test_emergency_stop.py` — current tests use MagicMock for all DB access; update trigger tests to reflect new query structure (DuckDB for VIX, corrected column names)
- [ ] Update `tests/scheduler/test_stop_polling.py` — add test that verifies correct table name `positions` in query
- [ ] Add integration-style test with real SQLite DB verifying `inference_outcomes` and `api_errors` DDL + query

*Existing infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Emergency stop fires on live trigger conditions | PROD-07 | Requires live market data + real drawdown state | Paper trading session with simulated drawdown |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
