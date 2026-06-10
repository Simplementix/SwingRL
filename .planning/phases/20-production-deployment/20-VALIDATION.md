---
phase: 20
slug: production-deployment
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/ -v -k "test_startup or test_live or test_fill or test_macro" -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_startup_checks.py tests/execution/test_fill_ingestor.py tests/memory/test_live_client.py -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 20-01-01 | 01 | 1 | DEPLOY-01 | smoke | `docker compose -f docker-compose.prod.yml build` | ❌ W0 (CI step) | ⬜ pending |
| 20-01-02 | 01 | 1 | DEPLOY-01 | unit | `uv run pytest tests/test_deploy_config.py::test_resource_limits -x` | ❌ W0 | ⬜ pending |
| 20-02-01 | 02 | 1 | DEPLOY-02 | unit | `uv run pytest tests/test_startup_checks.py::test_env_key_validation -x` | ❌ W0 | ⬜ pending |
| 20-02-02 | 02 | 1 | DEPLOY-03 | unit | `uv run pytest tests/test_startup_checks.py::test_broker_credential_hard_fail -x` | ❌ W0 | ⬜ pending |
| 20-02-03 | 02 | 1 | DEPLOY-03 | unit | `uv run pytest tests/test_startup_checks.py::test_memory_empty_warn -x` | ❌ W0 | ⬜ pending |
| 20-03-01 | 03 | 1 | DEPLOY-04 | unit | `uv run pytest tests/scheduler/test_jobs.py::test_equity_cycle_timezone -x` | ❌ W0 | ⬜ pending |
| 20-03-02 | 03 | 1 | DEPLOY-04 | smoke | `uv run pytest tests/test_show_schedule.py -x` | ❌ W0 | ⬜ pending |
| 20-04-01 | 04 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_fill_ingestor.py::test_alpaca_fill_ingest -x` | ❌ W0 | ⬜ pending |
| 20-04-02 | 04 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_fill_ingestor.py::test_binance_stub_paper -x` | ❌ W0 | ⬜ pending |
| 20-05-01 | 05 | 2 | DEPLOY-05 | unit | `uv run pytest tests/memory/test_live_client.py::test_cycle_gate_fail_open -x` | ❌ W0 | ⬜ pending |
| 20-05-02 | 05 | 2 | DEPLOY-05 | unit | `uv run pytest tests/memory/test_live_client.py::test_trade_veto_fail_open -x` | ❌ W0 | ⬜ pending |
| 20-06-01 | 06 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_pipeline.py::test_cycle_gate_blocks_cycle -x` | ❌ W0 | ⬜ pending |
| 20-06-02 | 06 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_pipeline.py::test_trade_veto_skips_signal -x` | ❌ W0 | ⬜ pending |
| 20-06-03 | 06 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_apply_overrides_tighten_only -x` | ❌ W0 | ⬜ pending |
| 20-06-04 | 06 | 2 | DEPLOY-05 | unit | `uv run pytest tests/execution/test_pipeline.py::test_memory_decisions_logged -x` | ❌ W0 | ⬜ pending |
| 20-07-01 | 07 | 3 | DEPLOY-05 | integration | Manual / docker exec | Manual-only | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_startup_checks.py` — stubs for DEPLOY-02, DEPLOY-03
- [ ] `tests/test_show_schedule.py` — stubs for DEPLOY-04
- [ ] `tests/test_deploy_config.py` — stubs for DEPLOY-01 resource limits
- [ ] `tests/execution/test_fill_ingestor.py` — stubs for DEPLOY-05 FillIngestors
- [ ] `tests/memory/test_live_client.py` — stubs for DEPLOY-05 live endpoint fail-open
- [ ] `tests/execution/test_circuit_breaker.py` — extend with apply_overrides tests
- [ ] `tests/execution/test_pipeline.py` — extend with memory hook integration tests
- [ ] Add SQLAlchemy, requests, python-binance to `pyproject.toml [project.dependencies]`; move apscheduler from dev to prod deps

*Existing test infrastructure covers scheduler, pipeline, and circuit breaker basics — new test files cover production deployment code paths.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| smoke_test.py full pipeline on homelab | DEPLOY-05 | Requires running Docker stack + Alpaca paper API | `docker exec swingrl python scripts/smoke_test.py` — verify exit 0, all checklist items pass |
| Containers healthy on homelab | DEPLOY-01 | Requires homelab hardware | `docker compose -f docker-compose.prod.yml ps` — both show "healthy" |
| .env validation on homelab | DEPLOY-02 | Requires real API keys | `docker exec swingrl python scripts/startup_checks.py` — no exit(1) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
