---
phase: 1
slug: dev-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/test_smoke.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_smoke.py -v`
- **After every plan wave:** Run `uv run pytest tests/ -v && pre-commit run --all-files`
- **Before `/gsd:verify-work`:** Full suite green + `ssh homelab "bash ~/swingrl/ci-homelab.sh"`
- **Max feedback latency:** 10 seconds (local), 120 seconds (homelab CI)

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists | Status |
|--------|----------|-----------|-------------------|-------------|--------|
| ENV-01 | Python 3.11 + MPS available on Mac | unit/smoke | `pytest tests/test_smoke.py::test_python_version -v` | ❌ W0 | ⬜ pending |
| ENV-02 | uv installed on both machines | manual | `uv --version` on M1 Mac + homelab | Manual verify | ⬜ pending |
| ENV-03 | Canonical directory structure exists | smoke | `pytest tests/test_smoke.py::test_directory_structure -v` | ❌ W0 | ⬜ pending |
| ENV-04 | pyproject.toml + uv.lock + tool config | smoke | `uv sync --locked` | ❌ W0 | ⬜ pending |
| ENV-05 | pre-commit passes all hooks | integration | `pre-commit run --all-files` | ❌ W0 | ⬜ pending |
| ENV-06 | Dockerfile builds + CPU torch works | integration | `docker build . && docker run --rm swingrl python -c "import torch"` | ❌ W0 | ⬜ pending |
| ENV-07 | docker-compose.yml resource limits | integration | `docker compose config` + `docker compose up -d` | ❌ W0 | ⬜ pending |
| ENV-08 | ci-homelab.sh runs end-to-end | e2e | `ssh homelab "bash ~/swingrl/ci-homelab.sh"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_smoke.py` — stubs for ENV-01, ENV-03 (import smoke + directory structure)
- [ ] `tests/conftest.py` — shared fixtures (minimal for Phase 1)
- [ ] Framework install: `uv add --group dev pytest>=8.0`
- [ ] `.secrets.baseline` — must be generated before pre-commit hooks work: `uv run detect-secrets scan > .secrets.baseline`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| uv installed on homelab | ENV-02 | Requires SSH to remote machine | `ssh homelab "uv --version"` — verify 0.6.x+ |
| MPS available on M1 Mac | ENV-01 | Platform-specific, can't test in Docker | `python -c "import torch; print(torch.backends.mps.is_available())"` — must return True |
| ci-homelab.sh end-to-end | ENV-08 | Requires homelab reachable via SSH | `ssh homelab "cd ~/swingrl && bash ci-homelab.sh"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s (local)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
