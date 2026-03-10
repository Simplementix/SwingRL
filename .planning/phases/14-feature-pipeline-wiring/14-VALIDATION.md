---
phase: 14
slug: feature-pipeline-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, configured in pyproject.toml) |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/test_phase14.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_phase14.py -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | FEAT-09 | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript -x` | ❌ W0 | ⬜ pending |
| 14-01-02 | 01 | 1 | FEAT-09 | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_returns_1_on_reject -x` | ❌ W0 | ⬜ pending |
| 14-01-03 | 01 | 1 | FEAT-09 | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_reads_json_files -x` | ❌ W0 | ⬜ pending |
| 14-01-04 | 01 | 1 | FEAT-09 | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_json_format -x` | ❌ W0 | ⬜ pending |
| 14-01-05 | 01 | 1 | FEAT-09 | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_missing_key_exits_1 -x` | ❌ W0 | ⬜ pending |
| 14-01-06 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestEquityDimHelpers::test_disabled_returns_156 -x` | ❌ W0 | ⬜ pending |
| 14-01-07 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestEquityDimHelpers::test_enabled_returns_172 -x` | ❌ W0 | ⬜ pending |
| 14-01-08 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_shape_172 -x` | ❌ W0 | ⬜ pending |
| 14-01-09 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_shape_156_when_disabled -x` | ❌ W0 | ⬜ pending |
| 14-01-10 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_sentiment_values_in_correct_positions -x` | ❌ W0 | ⬜ pending |
| 14-01-11 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestEnvObsSpace::test_equity_env_obs_172_when_sentiment_enabled -x` | ❌ W0 | ⬜ pending |
| 14-01-12 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestEnvObsSpace::test_equity_env_obs_156_default -x` | ❌ W0 | ⬜ pending |
| 14-01-13 | 01 | 1 | FEAT-10 | unit | `uv run pytest tests/test_phase14.py::TestFeatureNames::test_feature_names_172_when_enabled -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase14.py` — stubs for FEAT-09 and FEAT-10 (does not exist yet)

*Existing pytest + conftest.py fixtures apply — no new infrastructure needed.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
