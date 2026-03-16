---
phase: 21
slug: discord-alert-suite
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.0 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/monitoring/ -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/monitoring/ -v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | DISC-01 | unit | `uv run pytest tests/test_config.py -k "alerting" -x` | extend | ⬜ pending |
| 21-01-02 | 01 | 1 | DISC-02 | unit | `uv run pytest tests/monitoring/test_alerter.py -k "trades_webhook or escalating or rate_limit or enabled_types" -x` | extend | ⬜ pending |
| 21-02-01 | 02 | 1 | DISC-02 | unit | `uv run pytest tests/monitoring/test_embeds.py -k "broker_auth or lifecycle or health or resolved" -x` | extend | ⬜ pending |
| 21-02-02 | 02 | 1 | DISC-04 | unit | `uv run pytest tests/monitoring/test_embeds.py -k "retrain" -x` | ❌ W0 | ⬜ pending |
| 21-03-01 | 03 | 2 | DISC-03 | unit | `uv run pytest tests/monitoring/test_embeds.py -k "daily_summary" -x` | extend | ⬜ pending |
| 21-03-02 | 03 | 2 | DISC-03 | unit | `uv run pytest tests/monitoring/test_embeds.py -k "memory_digest" -x` | ❌ W0 | ⬜ pending |
| 21-04-01 | 04 | 3 | DISC-01 | unit | `uv run pytest tests/monitoring/ tests/dashboard/ -x` | extend | ⬜ pending |
| 21-04-02 | 04 | 3 | DISC-01 | smoke | `ssh homelab "docker exec swingrl python scripts/smoke_test.py"` | manual | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Extend `tests/monitoring/test_embeds.py` — stubs for 8 new embed builders (retrain x4, broker auth, lifecycle x2, system health, resolved)
- [ ] Extend `tests/monitoring/test_alerter.py` — stubs for trades webhook routing, escalating cooldown, rate limit queue, enabled_types filter
- [ ] Extend `tests/test_config.py` — stubs for new AlertingConfig fields
- [ ] `tests/dashboard/test_pages.py` — stub for alert history page

*Existing test infrastructure covers alerter, embeds, and config basics — new tests extend those files.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Smoke test ping to all 3 Discord channels | DISC-01 | Requires real Discord webhook URLs | `docker exec swingrl python scripts/smoke_test.py` — verify purple embed appears in all 3 channels |
| Daily summary renders correctly in Discord | DISC-03 | Visual verification of embed layout | Post test summary, verify fields readable in Discord mobile + desktop |
| Retrain embeds render with correct colors | DISC-04 | Visual verification | Use test script to post all 4 retrain embed types, verify colors + fields |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
