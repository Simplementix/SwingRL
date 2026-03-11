---
phase: 19
slug: model-training
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 19-01-01 | 01 | 1 | TRAIN-01 | integration | `uv run pytest tests/training/test_pipeline.py::test_equity_baseline_training -x` | ❌ W0 | ⬜ pending |
| 19-01-02 | 01 | 1 | TRAIN-02 | integration | `uv run pytest tests/training/test_pipeline.py::test_crypto_baseline_training -x` | ❌ W0 | ⬜ pending |
| 19-01-03 | 01 | 1 | TRAIN-03 | unit | `uv run pytest tests/training/test_pipeline.py::test_wf_metrics_recorded -x` | ❌ W0 | ⬜ pending |
| 19-01-04 | 01 | 1 | TRAIN-04 | unit | `uv run pytest tests/training/test_pipeline.py::test_tuning_triggers_on_low_sharpe -x` | ❌ W0 | ⬜ pending |
| 19-01-05 | 01 | 1 | TRAIN-05 | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_weights_from_wf_sharpe -x` | ❌ W0 | ⬜ pending |
| 19-01-06 | 01 | 1 | TRAIN-06 | unit | `uv run pytest tests/training/test_pipeline.py::test_model_files_deployed -x` | ❌ W0 | ⬜ pending |
| 19-01-07 | 01 | 1 | TRAIN-07 | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_gate_blocks_deployment -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/training/test_pipeline.py` — stubs for TRAIN-01 through TRAIN-07

*Existing infrastructure covers all other phase requirements — conftest.py and existing fixtures are sufficient.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Homelab CPU training completes in <24h | TRAIN-01/02 | Wall-clock runtime on actual hardware | SSH to homelab, run `python scripts/train_pipeline.py`, verify completion |
| Model files usable by paper trading | TRAIN-06 | End-to-end integration with paper trading phase | Load model in paper trading env, verify predictions |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
