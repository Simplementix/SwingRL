---
phase: 15
slug: training-cli-observation-assembly
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_phase15.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_phase15.py -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | TRAIN-01 | unit | `uv run pytest tests/test_phase15.py::TestLoadFeaturesEquity -x` | ❌ W0 | ⬜ pending |
| 15-01-02 | 01 | 1 | TRAIN-02 | unit | `uv run pytest tests/test_phase15.py::TestLoadFeaturesCrypto -x` | ❌ W0 | ⬜ pending |
| 15-01-03 | 01 | 1 | TRAIN-03 | integration | `uv run pytest tests/test_phase15.py::TestDryRunCLI -x` | ❌ W0 | ⬜ pending |
| 15-01-04 | 01 | 1 | VAL-01 | unit | `uv run pytest tests/test_phase15.py::TestFoldShapes -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase15.py` — stubs for TRAIN-01, TRAIN-02, TRAIN-03, VAL-01; uses in-memory DuckDB with seeded feature rows

*Existing infrastructure — `conftest.py` fixtures `loaded_config`, `equity_env_config` — covers all config needs; only the test file itself is missing.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
