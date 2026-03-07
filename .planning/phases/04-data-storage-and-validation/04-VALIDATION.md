---
phase: 4
slug: data-storage-and-validation
status: active
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-06
updated: 2026-03-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/data/ tests/monitoring/ -v --tb=short` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/data/ tests/monitoring/ -v --tb=short`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Test File | Status |
|---------|------|------|-------------|-----------|-------------------|-----------|--------|
| 01-T1 | 04-01 | 1 | DATA-06, DATA-07 | unit | `uv run pytest tests/data/test_db.py -x -v -k "singleton or reset or duckdb_context or sqlite_context or config"` | tests/data/test_db.py | pending |
| 01-T2 | 04-01 | 1 | DATA-06, DATA-07, DATA-08, DATA-09 | unit+integration | `uv run pytest tests/data/test_db.py -x -v` | tests/data/test_db.py | pending |
| 02-T1 | 04-02 | 2 | DATA-12 | unit | `uv run pytest tests/data/test_ingestion_logging.py tests/data/test_parquet_to_duckdb.py -x -v` | tests/data/test_ingestion_logging.py, tests/data/test_parquet_to_duckdb.py | pending |
| 03-T1 | 04-03 | 2 | DATA-13 | unit | `uv run pytest tests/monitoring/test_alerter.py -x -v` | tests/monitoring/test_alerter.py | pending |
| 04-T1 | 04-04 | 2 | DATA-11 | unit | `uv run pytest tests/data/test_cross_source.py -x -v` | tests/data/test_cross_source.py | pending |
| 04-T2 | 04-04 | 2 | DATA-10 | unit | `uv run pytest tests/data/test_corporate_actions.py -x -v` | tests/data/test_corporate_actions.py | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:
- `tests/data/` directory exists from Phase 3
- `tests/conftest.py` has shared fixtures (tmp_config, loaded_config, tmp_dirs)
- `tests/monitoring/` directory and `__init__.py` created in Plan 03
- No additional Wave 0 setup needed — all test files created within their respective plan tasks

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Discord webhook delivery | DATA-13 | Requires live webhook URL and Discord server | 1. Set `DISCORD_WEBHOOK_URL` env var 2. Create Alerter with the URL 3. Call `send_alert("critical", "Test", "Integration test")` 4. Verify message appears in Discord channel |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] No Wave 0 MISSING references (all test files created within tasks)
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready
