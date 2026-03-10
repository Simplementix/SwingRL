---
phase: 10
slug: production-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml [tool.pytest] |
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
| 10-01-01 | 01 | 1 | PROD-01 | unit | `uv run pytest tests/backup/test_sqlite_backup.py -x` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | PROD-01 | unit | `uv run pytest tests/backup/test_duckdb_backup.py -x` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 1 | PROD-02 | unit | `uv run pytest tests/test_deploy_smoke.py -x` | ❌ W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | PROD-03 | unit | `uv run pytest tests/shadow/test_shadow_runner.py -x` | ❌ W0 | ⬜ pending |
| 10-03-02 | 03 | 2 | PROD-04 | unit | `uv run pytest tests/shadow/test_promoter.py -x` | ❌ W0 | ⬜ pending |
| 10-03-03 | 03 | 2 | PROD-05 | unit | `uv run pytest tests/shadow/test_lifecycle.py -x` | ❌ W0 | ⬜ pending |
| 10-04-01 | 04 | 3 | HARD-03 | unit | `uv run pytest tests/sentiment/test_finbert.py -x` | ❌ W0 | ⬜ pending |
| 10-04-02 | 04 | 3 | HARD-04 | unit | `uv run pytest tests/sentiment/test_ab_experiment.py -x` | ❌ W0 | ⬜ pending |
| 10-05-01 | 05 | 3 | PROD-07 | unit | `uv run pytest tests/test_emergency_stop.py -x` | ❌ W0 | ⬜ pending |
| 10-06-01 | 06 | 4 | PROD-06 | integration | `uv run pytest tests/test_security.py -x` | ❌ W0 | ⬜ pending |
| 10-06-02 | 06 | 4 | PROD-08 | integration | `uv run pytest tests/test_disaster_recovery.py -x` | ❌ W0 | ⬜ pending |
| 10-06-03 | 06 | 4 | HARD-02 | unit | `uv run pytest tests/test_retry.py -x` | ❌ W0 | ⬜ pending |
| 10-06-04 | 06 | 4 | HARD-05 | unit | `uv run pytest tests/utils/test_file_logging.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backup/__init__.py` — package init
- [ ] `tests/backup/test_sqlite_backup.py` — stubs for PROD-01 SQLite
- [ ] `tests/backup/test_duckdb_backup.py` — stubs for PROD-01 DuckDB
- [ ] `tests/shadow/__init__.py` — package init
- [ ] `tests/shadow/test_shadow_runner.py` — stubs for PROD-03
- [ ] `tests/shadow/test_promoter.py` — stubs for PROD-04
- [ ] `tests/shadow/test_lifecycle.py` — stubs for PROD-05
- [ ] `tests/sentiment/__init__.py` — package init
- [ ] `tests/sentiment/test_finbert.py` — stubs for HARD-03
- [ ] `tests/sentiment/test_ab_experiment.py` — stubs for HARD-04
- [ ] `tests/test_deploy_smoke.py` — stubs for PROD-02
- [ ] `tests/test_emergency_stop.py` — stubs for PROD-07 (extend existing)
- [ ] `tests/test_security.py` — stubs for PROD-06
- [ ] `tests/test_disaster_recovery.py` — stubs for PROD-08/09
- [ ] `tests/test_retry.py` — stubs for HARD-02
- [ ] `tests/utils/test_file_logging.py` — stubs for HARD-05

*Framework install: N/A — pytest already configured*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Jupyter notebook execution | HARD-01 | Requires Jupyter kernel and interactive widgets | Open each notebook in `notebooks/`, run all cells, verify outputs render |
| Monthly off-site rsync via Tailscale | PROD-01 | Requires remote Tailscale host connectivity | Run rsync command manually, verify files appear on remote |
| Emergency stop full integration | PROD-07 | Requires live broker connections | Run `emergency_stop.py` in paper mode, verify via exchange dashboards |
| Disaster recovery end-to-end | PROD-08/09 | Requires container stop/restart cycle | Follow 9-step DR checklist, verify system resumes correctly |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
