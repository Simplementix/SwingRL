---
phase: 6
slug: rl-environments
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured in pyproject.toml) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_envs.py -x -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_envs.py -x -v`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | TRAIN-01 | unit | `uv run pytest tests/test_envs.py::test_equity_env_reset_observation_shape -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | TRAIN-01 | unit | `uv run pytest tests/test_envs.py::test_equity_env_step_contract -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | TRAIN-01 | integration | `uv run pytest tests/test_envs.py::test_equity_env_check_env -x` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | TRAIN-02 | unit | `uv run pytest tests/test_envs.py::test_crypto_env_reset_observation_shape -x` | ❌ W0 | ⬜ pending |
| 06-01-05 | 01 | 1 | TRAIN-02 | unit | `uv run pytest tests/test_envs.py::test_crypto_env_step_contract -x` | ❌ W0 | ⬜ pending |
| 06-01-06 | 01 | 1 | TRAIN-02 | integration | `uv run pytest tests/test_envs.py::test_crypto_env_check_env -x` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | TRAIN-07 | unit | `uv run pytest tests/test_envs.py::test_rolling_sharpe_warmup -x` | ❌ W0 | ⬜ pending |
| 06-02-02 | 02 | 1 | TRAIN-07 | unit | `uv run pytest tests/test_envs.py::test_rolling_sharpe_full_window -x` | ❌ W0 | ⬜ pending |
| 06-03-01 | 03 | 1 | TRAIN-08 | unit | `uv run pytest tests/test_envs.py::test_env_raw_observations -x` | ❌ W0 | ⬜ pending |
| 06-03-02 | 03 | 1 | TRAIN-09 | unit | `uv run pytest tests/test_envs.py::test_signal_deadzone_hold -x` | ❌ W0 | ⬜ pending |
| 06-03-03 | 03 | 1 | TRAIN-10 | unit | `uv run pytest tests/test_envs.py::test_info_dict_turbulence -x` | ❌ W0 | ⬜ pending |
| 06-04-01 | 04 | 2 | TRAIN-11 | unit | `uv run pytest tests/test_envs.py::test_equity_episode_length -x` | ❌ W0 | ⬜ pending |
| 06-04-02 | 04 | 2 | TRAIN-11 | unit | `uv run pytest tests/test_envs.py::test_crypto_episode_length_random_start -x` | ❌ W0 | ⬜ pending |
| 06-05-01 | 05 | 2 | TRAIN-01 | integration | `uv run pytest tests/test_envs.py::test_gym_make_equity -x` | ❌ W0 | ⬜ pending |
| 06-05-02 | 05 | 2 | TRAIN-02 | integration | `uv run pytest tests/test_envs.py::test_gym_make_crypto -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_envs.py` — stubs for TRAIN-01, TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11
- [ ] Test fixtures in `tests/conftest.py` — synthetic feature arrays (252+ rows equity, 540+ rows crypto) and price arrays for environment construction
- [ ] No new framework install needed — pytest already configured

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
