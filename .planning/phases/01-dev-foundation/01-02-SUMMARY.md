---
phase: 01-dev-foundation
plan: 02
subsystem: infra
tags: [pre-commit, ruff, mypy, bandit, detect-secrets, code-quality, security]

# Dependency graph
requires:
  - phase: 01-dev-foundation
    plan: 01
    provides: "pyproject.toml with [tool.bandit], [tool.mypy], [tool.ruff] config; installed dev dependencies (ruff, black, mypy, bandit, detect-secrets, pre-commit)"
provides:
  - .pre-commit-config.yaml with 5 blocking hooks (ruff, ruff-format, mypy, bandit, detect-secrets)
  - .secrets.baseline committed — detect-secrets ready for baseline auditing
  - pre-commit installed at .git/hooks/pre-commit
  - Verified: bandit B101 skipped in tests/, flagged in src/ (via pyproject.toml config)
  - Verified: detect-secrets blocks hardcoded secret strings before commit
affects: [all subsequent phases — every commit now runs all 5 quality gates]

# Tech tracking
tech-stack:
  added:
    - "pre-commit 4.5.1 — git hook manager"
    - "ruff-pre-commit v0.15.5 — lint + format hooks"
    - "mirrors-mypy v1.19.1 — type checking hook"
    - "bandit 1.9.4 — security linting hook (blocking)"
    - "detect-secrets v1.5.0 — secret scanning hook (blocking)"
  patterns:
    - "ruff-format replaces black in pre-commit (black config retained in pyproject.toml for CLI)"
    - "bandit[toml] additional_dependency required for pyproject.toml exclusion config"
    - "detect-secrets baseline generated and committed BEFORE pre-commit install"
    - "mypy files: ^src/ — only checks production code, not tests"

key-files:
  created:
    - ".pre-commit-config.yaml — 5-hook pre-commit configuration (ruff, ruff-format, mypy, bandit, detect-secrets)"
    - ".secrets.baseline — detect-secrets scan baseline (committed 8271806)"
  modified: []

key-decisions:
  - "ruff-format replaces black in pre-commit to avoid formatting conflicts; black remains in pyproject.toml for direct CLI use"
  - "mypy files: ^src/ constraint added to avoid running strict type checks on test files"
  - "bandit[toml] additional_dependency is required — without it bandit cannot read pyproject.toml exclusions and would falsely flag B101 in test files"

patterns-established:
  - "Critical setup sequence: generate baseline → commit baseline → install hooks → run --all-files"
  - "bandit exclude: ^tests/ in hook + [tool.bandit.assert_used] skips in pyproject.toml (belt-and-suspenders)"

requirements-completed: [ENV-05]

# Metrics
duration: 3min
completed: 2026-03-06
---

# Phase 1 Plan 2: Pre-commit Hooks Summary

**5 blocking pre-commit hooks (ruff, ruff-format, mypy, bandit, detect-secrets) installed and passing on full codebase, with bandit correctly excluding test assert statements and detect-secrets blocking hardcoded secret patterns.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-06T15:58:22Z
- **Completed:** 2026-03-06T16:01:42Z
- **Tasks:** 1
- **Files modified:** 2 (created .pre-commit-config.yaml, .secrets.baseline)

## Accomplishments

- .pre-commit-config.yaml created with 5 hooks using latest stable revisions (ruff v0.15.5, mypy v1.19.1, bandit 1.9.4, detect-secrets v1.5.0)
- .secrets.baseline generated and committed before pre-commit install (correct critical ordering)
- pre-commit installed at .git/hooks/pre-commit — hooks run on every commit automatically
- Verified: bandit B101 (assert_used) correctly skipped in tests/, flagged in src/ — confirming pyproject.toml [tool.bandit] config is read via bandit[toml]
- Verified: detect-secrets blocks commits containing hardcoded API key patterns (tested with fake key literal)
- All 5 hooks pass on full codebase; 8/8 smoke tests still pass after hook installation

## Task Commits

1. **Baseline commit:** `8271806` (chore: add detect-secrets baseline) — committed separately per plan requirement
2. **Task 1: Configure pre-commit hooks and generate secrets baseline** - `bcaa4fe` (feat)

**Plan metadata:** (this commit — docs)

## Files Created/Modified

- `.pre-commit-config.yaml` — 5-hook pre-commit config (36 lines)
- `.secrets.baseline` — detect-secrets scan baseline with 1 known false positive (plan doc example)

## Decisions Made

- **ruff-format replaces black in pre-commit:** Using both causes formatting conflicts. ruff-format is the recommended approach. black remains in pyproject.toml for direct `uv run black` usage.
- **mypy files: ^src/ constraint:** Prevents mypy from running strict type checks on test files, which have different typing needs (fixtures, assert patterns).
- **bandit[toml] is required:** Without it, bandit silently ignores pyproject.toml config and would flag B101 in test files. Verified working via round-trip test.

## Deviations from Plan

None - plan executed exactly as written. The critical ordering (baseline before hooks) was followed precisely.

## Issues Encountered

None. All hooks installed and passed on first run after correct setup sequence.

## User Setup Required

None — no external service configuration required for pre-commit hooks.

## Next Phase Readiness

- Pre-commit hooks fully operational — every future commit auto-enforces code quality and security
- Phase 1 Plan 3 (Dockerfile + docker-compose.yml + ci-homelab.sh) can proceed
- Reminder from Plan 1: remove `tool.uv.environments = darwin` constraint when adding Docker/Linux support in Plan 3

---
*Phase: 01-dev-foundation*
*Completed: 2026-03-06*
