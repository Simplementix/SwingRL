---
phase: 22
slug: automated-retraining
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 22 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/ -x -q --timeout=30` |
| **Full suite command** | `uv run pytest tests/ -v --timeout=120` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q --timeout=30`
- **After every plan wave:** Run `uv run pytest tests/ -v --timeout=120`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 22-01-01 | 01 | 1 | RETRAIN-01 | unit | `uv run pytest tests/test_retraining.py -k duckdb_short_lived -v` | ❌ W0 | ⬜ pending |
| 22-01-02 | 01 | 1 | RETRAIN-01 | unit | `uv run pytest tests/test_retraining.py -k config_schema -v` | ❌ W0 | ⬜ pending |
| 22-02-01 | 02 | 1 | RETRAIN-02 | unit | `uv run pytest tests/test_retraining.py -k rolling_sharpe -v` | ❌ W0 | ⬜ pending |
| 22-02-02 | 02 | 1 | RETRAIN-02 | unit | `uv run pytest tests/test_retraining.py -k retrain_orchestrator -v` | ❌ W0 | ⬜ pending |
| 22-03-01 | 03 | 2 | RETRAIN-03 | unit | `uv run pytest tests/test_retraining.py -k scheduler_job -v` | ❌ W0 | ⬜ pending |
| 22-03-02 | 03 | 2 | RETRAIN-03 | unit | `uv run pytest tests/test_retraining.py -k failure_recovery -v` | ❌ W0 | ⬜ pending |
| 22-04-01 | 04 | 2 | RETRAIN-04 | unit | `uv run pytest tests/test_retraining.py -k bootstrap_guard -v` | ❌ W0 | ⬜ pending |
| 22-04-02 | 04 | 2 | RETRAIN-04 | unit | `uv run pytest tests/test_retraining.py -k memory_ingest -v` | ❌ W0 | ⬜ pending |
| 22-05-01 | 05 | 3 | RETRAIN-05 | unit | `uv run pytest tests/test_retraining.py -k operator_cli -v` | ❌ W0 | ⬜ pending |
| 22-05-02 | 05 | 3 | RETRAIN-05 | integration | `uv run pytest tests/test_retraining.py -k end_to_end -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_retraining.py` — test stubs for all RETRAIN requirements
- [ ] `tests/conftest.py` — add retrain-specific fixtures (mock trainer, retrain config)

*Existing infrastructure covers framework and runner — only test files needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Retrain completes on homelab hardware | RETRAIN-01 | Requires real GPU/CPU resources | `docker exec swingrl python scripts/retrain.py --env equity --dry-run --timesteps 1000` |
| Discord embeds render correctly | RETRAIN-03 | Visual verification needed | Check #daily and #alerts channels after test retrain |
| Nice priority reduces CPU contention | RETRAIN-01 | Requires concurrent load testing | Run retrain + live trading simultaneously, check `top` output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
