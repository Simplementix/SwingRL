---
phase: 4
slug: data-storage-and-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/test_storage/ -v --tb=short` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_storage/ -v --tb=short`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| *Populated during planning* | | | | | | | |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_storage/` — test directory for Phase 4
- [ ] `tests/test_storage/conftest.py` — shared fixtures (tmp DuckDB/SQLite, mock Discord webhook)
- [ ] `tests/test_storage/test_schema.py` — stubs for DATA-06 (DuckDB schema), DATA-07 (SQLite schema)
- [ ] `tests/test_storage/test_cross_source.py` — stubs for DATA-09 (cross-source validation)
- [ ] `tests/test_storage/test_ingestion_log.py` — stubs for DATA-10 (ingestion logging)
- [ ] `tests/test_storage/test_alerter.py` — stubs for DATA-11 (Discord alerting)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Discord webhook delivery | DATA-11 | Requires live webhook URL | 1. Set `SWINGRL_DISCORD__WEBHOOK_URL` env var 2. Run `uv run python -m swingrl.alerts.discord --test` 3. Verify message in Discord channel |
| Cross-DB join (DuckDB + SQLite) | DATA-08 | Requires both DBs populated with real data | 1. Run ingestors for equity + crypto 2. Execute join query from success criteria #2 3. Verify result set |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
