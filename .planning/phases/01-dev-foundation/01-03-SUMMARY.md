---
phase: 01-dev-foundation
plan: 03
subsystem: infra
tags: [docker, dockerfile, docker-compose, ci, pytest, ruff, mypy, uv, python]

# Dependency graph
requires:
  - phase: 01-dev-foundation plan 01
    provides: pyproject.toml, uv.lock, test scaffold (tests/test_smoke.py)
  - phase: 01-dev-foundation plan 02
    provides: pre-commit hooks, ruff/mypy configured in pyproject.toml
provides:
  - Dockerfile with CPU-only torch and non-root trader user (UID 1000)
  - docker-compose.yml with 2.5g mem_limit, 1 CPU, bind mounts, env_file, TZ
  - scripts/ci-homelab.sh — 5-stage CI quality gate (git pull, docker build, pytest, ruff/mypy, cleanup)
  - Phase gate validated: full CI pipeline passes on x86 homelab via SSH
affects: [02-data-pipeline, 03-equity-env, 04-crypto-env, all future phases using Docker CI]

# Tech tracking
tech-stack:
  added: [docker, docker-compose, python:3.11-slim base image, CPU-only torch in container]
  patterns: [multi-stage Docker layer caching via uv sync, non-root container user (UID 1000), 5-stage CI gate pattern]

key-files:
  created:
    - Dockerfile
    - docker-compose.yml
    - .dockerignore
    - scripts/ci-homelab.sh
  modified:
    - pyproject.toml (uv.lock regenerated)

key-decisions:
  - "CPU-only torch in Docker — homelab has no GPU; MPS stays on M1 Mac natively"
  - "Single-stage Dockerfile with dev deps — no production split until Phase 8+"
  - "ruff format --check replaces standalone black check inside container — ruff formatter is drop-in black replacement"
  - "5-stage CI gate: git pull → docker build → pytest → ruff check + ruff format --check + mypy → cleanup"
  - "ci-homelab.sh supports --no-cache flag for clean builds when needed"

patterns-established:
  - "Pattern 1: Docker layer cache — uv sync deps-only layer before COPY . /app, keeps rebuilds fast"
  - "Pattern 2: Non-root trader user (UID 1000) — all container workloads run as trader, never root"
  - "Pattern 3: CI gate script — bash scripts/ci-homelab.sh is the canonical CI runner for all future phases"
  - "Pattern 4: bind mounts for runtime data — data/, db/, models/, logs/, status/ bind-mounted from host"

requirements-completed: [ENV-06, ENV-07, ENV-08]

# Metrics
duration: ~2 sessions (checkpoint between Task 2 and Task 3)
completed: 2026-03-06
---

# Phase 1 Plan 03: Docker + CI Pipeline Summary

**Dockerfile (CPU-only torch, UID 1000 non-root trader user), docker-compose.yml (2.5g/1CPU resource limits, bind mounts), and ci-homelab.sh (5-stage full quality gate) validated end-to-end on x86 homelab via SSH**

## Performance

- **Duration:** ~2 sessions (checkpoint between Task 2 local validation and Task 3 homelab verification)
- **Started:** 2026-03-06
- **Completed:** 2026-03-06T16:23:36Z
- **Tasks:** 3 (including 1 checkpoint:human-verify)
- **Files modified:** 5 (Dockerfile, docker-compose.yml, .dockerignore, scripts/ci-homelab.sh, pyproject.toml/uv.lock)

## Accomplishments
- Dockerfile builds CPU-only torch on python:3.11-slim with non-root trader user (UID 1000), supporting both M1 Mac and x86 homelab
- docker-compose.yml enforces 2.5g memory cap and 1 CPU limit with bind mounts for all runtime data directories and .env injection
- ci-homelab.sh implements the canonical 5-stage CI quality gate used throughout all future phases: git pull, docker build, pytest, ruff+mypy lint/typecheck, cleanup
- Full CI pipeline validated on x86 homelab via SSH — all 5 stages passed, "CI PASSED" confirmed

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Dockerfile and docker-compose.yml** - `a0c4af3` (feat)
2. **Task 2: Create ci-homelab.sh with full quality gate and validate locally** - `61abe1e` (feat)
3. **Task 3: Verify ci-homelab.sh on homelab via SSH** - verified by user (checkpoint:human-verify, no code commit required)

**Plan metadata:** (docs commit — this SUMMARY.md)

## Files Created/Modified
- `Dockerfile` — python:3.11-slim base, CPU-only torch via uv sync, non-root trader user UID 1000, uv layer-cache pattern
- `docker-compose.yml` — swingrl service with mem_limit=2.5g, cpus=1.0, TZ=America/New_York, env_file, 5 bind mounts (data, db, models, logs, status)
- `.dockerignore` — excludes .git, __pycache__, .env, .venv, node_modules, *.pyc from build context
- `scripts/ci-homelab.sh` — 5-stage CI: git pull, docker build (--no-cache flag supported), pytest, ruff check + ruff format --check + mypy inside container, docker compose down + image prune
- `pyproject.toml` / `uv.lock` — updated during Docker build validation

## Decisions Made
- **CPU-only torch in Docker:** Homelab has no GPU. MPS remains available natively on M1 Mac. Docker image uses CPU-only to stay lean and portable.
- **Single-stage Dockerfile with all deps:** No production/dev split yet — that's Phase 8+. Dev deps needed in CI so all are included in the single stage.
- **ruff format --check replaces black in container:** ruff's formatter is a drop-in black replacement. Satisfies the "ruff/black/mypy" quality gate requirement without adding black as a separate tool.
- **5-stage CI pattern established as canonical:** All future phases append to the same ci-homelab.sh or reference it as the baseline runner.

## Deviations from Plan

None — plan executed exactly as written. Task 3 was a checkpoint:human-verify; user confirmed all 5 CI stages passed on the x86 homelab.

## Issues Encountered

None. Docker build, pytest, ruff, and mypy all passed on first attempt on homelab.

## User Setup Required

None — all setup was automated. Homelab directory structure (`~/swingrl/{data,db,models,logs,status}`) and `.env` file were prepared as part of the checkpoint verification steps.

## Next Phase Readiness

- Phase 1 (Dev Foundation) complete — all 3 plans done, ENV-06/ENV-07/ENV-08 validated
- Docker CI pipeline is the canonical CI runner for all future phases
- Phase 2 (Data Pipeline) can begin: data/ and db/ bind mounts are in place, DuckDB and SQLite schemas are next
- Constraint to remove before Phase 2 Plan 1: `tool.uv.environments` darwin constraint in pyproject.toml (deferred from Phase 1)

---
*Phase: 01-dev-foundation*
*Completed: 2026-03-06*
