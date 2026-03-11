---
phase: 7
slug: agent-training-and-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/test_training/ -v --tb=short` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_training/ -v --tb=short`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | TRAIN-03 | unit | `uv run pytest tests/test_training/test_trainer.py -v` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | TRAIN-04 | unit | `uv run pytest tests/test_training/test_convergence.py -v` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 1 | TRAIN-05 | unit | `uv run pytest tests/test_training/test_model_io.py -v` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | TRAIN-06 | unit | `uv run pytest tests/test_agents/test_ensemble.py -v` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | VAL-01,VAL-02 | unit | `uv run pytest tests/test_training/test_metrics.py -v` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 3 | VAL-03,VAL-04,VAL-05 | integration | `uv run pytest tests/test_training/test_walkforward.py -v` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 3 | VAL-06,VAL-07 | integration | `uv run pytest tests/test_training/test_validation_gates.py -v` | ❌ W0 | ⬜ pending |
| 07-03-03 | 03 | 3 | VAL-08 | integration | `uv run pytest tests/test_training/test_backtest_storage.py -v` | ❌ W0 | ⬜ pending |
| 07-03-04 | 03 | 3 | TRAIN-12 | unit | `uv run pytest tests/test_training/test_smoke.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_training/` — directory structure for training tests
- [ ] `tests/test_agents/` — directory structure for agent tests
- [ ] `tests/test_training/conftest.py` — shared fixtures (mock envs, dummy models, tmp model dirs)
- [ ] `tests/test_agents/conftest.py` — shared fixtures for ensemble tests

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard training curves | TRAIN-03 | Visual inspection of convergence | Open `tensorboard --logdir logs/tensorboard/`, verify loss decreases |
| Full 1M-step training time | TRAIN-03 | Long-running, not suitable for CI | Run `scripts/train.py --env equity --algo ppo`, verify completes in <30min |

*All other behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
