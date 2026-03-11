---
phase: 02
slug: developer-experience
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/test_smoke.py tests/test_config.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_smoke.py tests/test_config.py -v`
- **After every plan wave:** Run `uv run pytest tests/ -v && uv run ruff check . && uv run mypy src/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | ENV-09 | smoke | `pytest tests/test_smoke.py::test_claude_md_exists -v` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | ENV-10 | smoke | `pytest tests/test_smoke.py::test_claude_commands_exist -v` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | ENV-11 | unit | `pytest tests/test_config.py -v` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | ENV-12 | smoke | `pytest tests/test_smoke.py::test_models_directories_exist -v` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 3 | ENV-13 | smoke+unit | `pytest tests/test_smoke.py tests/test_config.py -v` | Partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_config.py` — stubs for ENV-11 (config validation roundtrip, ValidationError, env var override)
- [ ] `tests/test_smoke.py` — add test_claude_md_exists, test_claude_commands_exist, test_models_directories_exist
- [ ] `tests/conftest.py` — expand with tmp_config, loaded_config, equity_ohlcv, crypto_ohlcv, tmp_dirs fixtures

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Claude skills usable from Claude Code | ENV-10 | Requires interactive Claude Code session | Open Claude Code, verify `/swingrl:test` and other skills appear and execute |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
