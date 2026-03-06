---
phase: 01-dev-foundation
plan: 01
subsystem: infra
tags: [uv, python, torch, pytorch, mps, stable-baselines3, gymnasium, finrl, pytest, pyproject-toml]

# Dependency graph
requires: []
provides:
  - Python 3.11 venv with all project dependencies resolved (uv.lock committed)
  - Importable swingrl package with 8 subpackages (src-layout)
  - All 8 canonical top-level directories with .gitkeep files
  - Smoke test suite validating env, structure, and imports (8 tests passing)
  - pyproject.toml with full tool configuration (ruff, black, mypy, bandit, pytest)
affects: [02-dev-foundation, all subsequent phases — all code builds on this venv and package structure]

# Tech tracking
tech-stack:
  added:
    - "uv 0.10.7 — package manager and venv"
    - "torch 2.3.1 — ML backend with MPS on M1 Mac"
    - "torchvision 0.18.1"
    - "stable-baselines3 2.7.1 — PPO/A2C/SAC implementations"
    - "gymnasium 1.2.3 — RL environment interface"
    - "finrl 0.3.7 — RL finance framework"
    - "pandas 2.3.3 — data handling"
    - "numpy 1.26.4 — numerical computing"
    - "hmmlearn — HMM regime detection"
    - "stockstats 0.6.8 — technical indicators (replaces pandas-ta for Python 3.11)"
    - "alpaca-py — Alpaca broker API"
    - "pydantic — config validation"
    - "pyyaml — YAML config loading"
    - "pytest 9.0.2 — test runner"
    - "ruff, black, mypy, bandit, detect-secrets — dev tools"
  patterns:
    - "uv + pyproject.toml as single source of truth for deps and tool config"
    - "src-layout: src/swingrl/ with subpackages, tests/ mirrors subpackage structure"
    - "[tool.uv.sources] for platform-specific PyTorch (darwin vs linux)"
    - "py.typed marker for PEP 561 compliance — all code requires type annotations"
    - "tool.uv.environments constrained to darwin for Phase 1 (linux deferred to Phase 2)"

key-files:
  created:
    - "pyproject.toml — full dependency spec, tool configs, build system"
    - "uv.lock — reproducible lockfile (macOS/darwin only, Phase 1)"
    - "src/swingrl/__init__.py — package root, __version__ = '0.1.0'"
    - "src/swingrl/py.typed — PEP 561 type marker"
    - "src/swingrl/{data,envs,agents,training,execution,monitoring,config,utils}/__init__.py — 8 subpackages"
    - "tests/test_smoke.py — 8 smoke tests (ENV-01, ENV-03, ENV-04)"
    - "tests/conftest.py — repo_root session fixture"
    - ".gitignore — comprehensive Python/ML project ignore rules"
    - ".env.example — 6 API key placeholders (Alpaca, Binance, Discord, FRED)"
    - ".planning/phases/01-dev-foundation/deferred-items.md — pandas-ta and linux-lock issues documented"
  modified: []

key-decisions:
  - "pandas-ta removed from Phase 1 deps — PyPI 0.4.x requires Python>=3.12, 0.3.14b no longer on PyPI, original GitHub repo deleted. stockstats>=0.4.0 used as replacement (FinRL's native TA library). Re-evaluation scheduled for Phase 6."
  - "tool.uv.environments constrained to sys_platform=darwin — avoids Linux split resolution failure caused by pandas-ta removal. Linux/Docker lockfile generation deferred to Phase 2 (ENV-06 Dockerfile task)."
  - "stockstats 0.6.8 replaces pandas_ta in smoke test imports — API difference noted in test docstring, deferred-items.md created."

patterns-established:
  - "uv sync --group dev: standard command for venv setup on M1 Mac"
  - "uv run pytest tests/test_smoke.py -v: Phase 1 test baseline"
  - "Per-task git commits with feat(01-01): prefix"

requirements-completed: [ENV-01, ENV-02, ENV-03, ENV-04]

# Metrics
duration: 11min
completed: 2026-03-06
---

# Phase 1 Plan 1: Repository Scaffold and Dependency Resolution Summary

**Python 3.11 venv with torch 2.3.1 (MPS enabled), SB3 2.7.1, FinRL 0.3.7 resolved via uv; 8-subpackage swingrl package importable; 8/8 smoke tests passing on M1 Mac.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-06T15:43:45Z
- **Completed:** 2026-03-06T15:54:36Z
- **Tasks:** 2
- **Files modified:** 35 (25 in Task 1, 10 in Task 2)

## Accomplishments

- All 8 top-level canonical directories created with .gitkeep (config/, data/, db/, models/, scripts/, status/, logs/)
- src/swingrl/ package importable with __version__ = "0.1.0", py.typed marker, and 8 subpackages each with __init__.py
- pyproject.toml with full dependency set resolved via uv sync (torch 2.3.1 + MPS, SB3 2.7.1, FinRL 0.3.7, gymnasium 1.2.3, numpy 1.26.4, pandas 2.3.3, stockstats 0.6.8, hmmlearn, alpaca-py, pydantic)
- uv.lock committed (91 packages, macOS/darwin platform, reproducible)
- 8/8 smoke tests pass — Python 3.11 verified, MPS confirmed True, all imports clean, structure validated

## Task Commits

1. **Task 1: Repository scaffold and pyproject.toml** - `b965261` (feat)
2. **Task 2: Smoke tests and conftest** - `71fa410` (feat)

**Plan metadata:** (this commit — docs)

## Files Created/Modified

- `pyproject.toml` — full project spec with all deps, tool configs (ruff, black, mypy, bandit, pytest)
- `uv.lock` — 91-package lockfile, committed, darwin-only for Phase 1
- `src/swingrl/__init__.py` — package root with `__version__ = "0.1.0"`
- `src/swingrl/py.typed` — PEP 561 type marker (empty)
- `src/swingrl/{data,envs,agents,training,execution,monitoring,config,utils}/__init__.py` — 8 empty subpackage inits
- `tests/test_smoke.py` — 8 smoke tests (57 lines)
- `tests/conftest.py` — session-scoped repo_root fixture
- `.gitignore` — comprehensive Python/ML ignore rules (venv, caches, data dirs, secrets)
- `.env.example` — 6 API key placeholders with comments
- `config/, data/, db/, models/, scripts/, status/, logs/` — 7 top-level dirs with .gitkeep
- `tests/{data,envs,agents,training,execution,monitoring,config,utils}/.gitkeep` — 8 test subdirs
- `.planning/phases/01-dev-foundation/deferred-items.md` — pandas-ta and linux-lock issues documented

## Decisions Made

- **pandas-ta removed:** PyPI 0.4.x requires Python>=3.12; 0.3.14b removed from PyPI; original GitHub repo deleted. All maintained forks also require numpy>=2.0 (incompatible with Python 3.11 ecosystem). stockstats (FinRL's native indicator library) used as Phase 1-5 replacement. Phase 6 re-evaluation planned.
- **tool.uv.environments=darwin:** Without this, uv attempts Linux split resolution which fails due to unavailable pandas-ta. Linux Docker lockfile generation is explicitly Phase 2 scope (ENV-06). Correct constraint for Phase 1.
- **FinRL 0.3.7 from PyPI resolves cleanly:** Despite being labeled "low-maintenance" in research, the actual uv resolver found no conflicts with SB3 2.7.1, gymnasium 1.2.3, and the full dependency graph.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed pandas-ta from dependencies**
- **Found during:** Task 1 (uv sync --group dev)
- **Issue:** pandas-ta 0.4.x requires Python>=3.12 and numpy>=2.0. The 0.3.14b version is no longer on PyPI. The original GitHub repo (twopirllc/pandas-ta) has been deleted. All maintained forks also require numpy>=2.0, conflicting with numpy<2 needed for Python 3.11.
- **Fix:** Replaced with stockstats>=0.4.0 (FinRL's native TA library, Python 3.11 compatible). Added deferred-items.md documenting the issue and Phase 6 resolution path.
- **Files modified:** pyproject.toml, tests/test_smoke.py (test_core_imports updated), deferred-items.md (created)
- **Verification:** uv sync --group dev succeeds; import stockstats passes; all 8 smoke tests green
- **Committed in:** b965261 (Task 1), 71fa410 (Task 2)

**2. [Rule 3 - Blocker] Added tool.uv.environments = darwin constraint**
- **Found during:** Task 1 (uv sync error on Linux platform split)
- **Issue:** Without the darwin-only environment constraint, uv attempts to resolve the Linux split for pytorch-cpu, which fails because pandas-ta is unavailable for that platform/version combination.
- **Fix:** Added `[tool.uv] environments = ["sys_platform == 'darwin'"]` to pyproject.toml, limiting lockfile generation to macOS for Phase 1.
- **Files modified:** pyproject.toml
- **Verification:** uv sync --locked succeeds; Docker/Linux deferred to Phase 2 (deferred-items.md item 2)
- **Committed in:** b965261 (Task 1)

---

**Total deviations:** 2 auto-fixed (1 Bug, 1 Blocker)
**Impact on plan:** Both fixes necessary to unblock dependency resolution on Python 3.11. No scope creep. Docker/Linux lockfile generation explicitly deferred to Phase 2 where it belongs (ENV-06 task).

## Issues Encountered

- **FinRL 0.3.7 PyPI resolution:** Resolved cleanly — the research concern about elegantrl/alpaca_trade_api conflicts did not materialize. FinRL 0.3.7 resolves against SB3 2.7.1 and gymnasium 1.2.3 without issues.
- **torch version:** Got 2.3.1 (within >=2.2,<2.4 range) — correct.
- **MPS confirmed:** torch.backends.mps.is_available() returns True on M1 Mac with standard PyPI torch.

## User Setup Required

None — no external service configuration required for Phase 1 scaffold.

## Next Phase Readiness

- Python 3.11 venv fully functional on M1 Mac — ready for Phase 1 Plan 2 (pre-commit, Dockerfile, docker-compose)
- **Blocker for Phase 2:** deferred-items.md item 2 — must remove `tool.uv.environments` constraint when adding Docker support, and regenerate uv.lock with Linux platform entries
- pyproject.toml tool configs (ruff, black, mypy, bandit) ready for pre-commit hook integration in Plan 2

---
*Phase: 01-dev-foundation*
*Completed: 2026-03-06*

## Self-Check: PASSED

All key files verified:
- pyproject.toml: FOUND
- uv.lock: FOUND
- src/swingrl/__init__.py: FOUND
- src/swingrl/py.typed: FOUND
- tests/test_smoke.py: FOUND
- tests/conftest.py: FOUND
- .gitignore: FOUND
- .env.example: FOUND
- .planning/phases/01-dev-foundation/01-01-SUMMARY.md: FOUND

Commits verified:
- b965261 (Task 1: scaffold + pyproject.toml): FOUND
- 71fa410 (Task 2: smoke tests): FOUND

Final pytest: 8/8 PASSED (uv run pytest tests/test_smoke.py -v)
