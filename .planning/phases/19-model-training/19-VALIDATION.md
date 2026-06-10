---
phase: 19
slug: model-training
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-11
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, configured in pyproject.toml) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/training/ tests/agents/ -v -x` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/training/ tests/agents/ -v -x`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 19-01-01 | 01 | 1 | TRAIN-01, TRAIN-04 | unit | `uv run pytest tests/memory/test_bounds.py -x` | Yes (Plan 01 Task 1) | ⬜ pending |
| 19-01-02 | 01 | 1 | TRAIN-02, TRAIN-05 | unit | `uv run pytest tests/training/test_pipeline_helpers.py -x` | Yes (Plan 01 Task 2) | ⬜ pending |
| 19-02-01 | 02 | 2 | TRAIN-03 | unit | `uv run pytest tests/memory/test_reward_wrapper.py tests/features/test_hmm_regime.py -x` | Yes (Plan 02 Task 1) | ⬜ pending |
| 19-02-02 | 02 | 2 | TRAIN-01 | integration | `uv run pytest tests/training/test_pipeline.py::test_equity_baseline_training -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-03 | 02 | 2 | TRAIN-02 | integration | `uv run pytest tests/training/test_pipeline.py::test_crypto_baseline_training -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-04 | 02 | 2 | TRAIN-03 | unit | `uv run pytest tests/training/test_pipeline.py::test_wf_metrics_recorded -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-05 | 02 | 2 | TRAIN-04 | unit | `uv run pytest tests/training/test_pipeline.py::test_tuning_triggers_on_low_sharpe -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-06 | 02 | 2 | TRAIN-05 | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_weights_from_wf_sharpe -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-07 | 02 | 2 | TRAIN-06 | unit | `uv run pytest tests/training/test_pipeline.py::test_model_files_deployed -x` | Yes (Plan 02 Task 2) | ⬜ pending |
| 19-02-08 | 02 | 2 | TRAIN-07 | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_gate_blocks_deployment -x` | Yes (Plan 02 Task 2) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

No separate Wave 0 plan needed. Each plan's TDD tasks create their own test files inline:

- Plan 01 Task 1 creates `tests/memory/test_bounds.py`
- Plan 01 Task 2 creates `tests/training/test_pipeline_helpers.py`
- Plan 02 Task 1 creates `tests/memory/test_reward_wrapper.py`
- Plan 02 Task 2 creates `tests/training/test_pipeline.py`

*Existing infrastructure covers all other phase requirements — conftest.py and existing fixtures are sufficient.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Homelab CPU training completes in <24h | TRAIN-01/02 | Wall-clock runtime on actual hardware | SSH to homelab, run `python scripts/train_pipeline.py`, verify completion |
| Model files usable by paper trading | TRAIN-06 | End-to-end integration with paper trading phase | Load model in paper trading env, verify predictions |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (wave 0 handled by TDD pattern within each plan)
