---
phase: 09-automation-and-monitoring
plan: 04
subsystem: infra
tags: [apscheduler, docker, entrypoint, stop-polling, daemon-thread, compose]

# Dependency graph
requires:
  - phase: 09-01
    provides: "Scheduler config, halt_check, job functions, healthcheck ping, emergency scripts"
  - phase: 09-02
    provides: "Embed builders, two-webhook Alerter, stuck agent detection, wash sale scanner"
  - phase: 09-03
    provides: "Streamlit dashboard with separate Dockerfile"
provides:
  - "Production entrypoint (scripts/main.py) with APScheduler and 6 cron jobs"
  - "Crypto stop-price polling daemon thread"
  - "Dockerfile CMD pointing to real entrypoint (no more sleep loop)"
  - "docker-compose.prod.yml with swingrl-dashboard service"
affects: [phase-10-production-hardening]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "APScheduler BackgroundScheduler with SQLAlchemy jobstore for persistent cron jobs"
    - "Daemon thread pattern for stop-price polling alongside scheduler"
    - "SIGTERM/SIGINT handler for graceful scheduler shutdown"
    - "threading.Event().wait() as cross-platform main block"

key-files:
  created:
    - scripts/main.py
    - src/swingrl/scheduler/stop_polling.py
    - tests/scheduler/test_main.py
    - tests/scheduler/test_stop_polling.py
  modified:
    - Dockerfile
    - docker-compose.prod.yml
    - src/swingrl/scheduler/jobs.py
    - pyproject.toml

key-decisions:
  - "threading.Event().wait() over signal.pause() for cross-platform main blocking"
  - "Stop-price polling runs as daemon thread (dies with main process) at 60s intervals"
  - "Dashboard service gets 512MB/0.5 CPU with read-only DB mounts"

patterns-established:
  - "Production entrypoint pattern: load config -> init components -> register jobs -> start scheduler -> block"
  - "Daemon thread pattern for periodic polling alongside APScheduler"

requirements-completed: [PAPER-12, PAPER-15, PAPER-16]

# Metrics
duration: 8min
completed: 2026-03-09
---

# Phase 9 Plan 04: Entrypoint Wiring Summary

**APScheduler main.py entrypoint with 6 cron jobs, crypto stop-price daemon thread, Dockerfile CMD update, and production compose with dashboard service**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-09
- **Completed:** 2026-03-09
- **Tasks:** 3 (2 auto + 1 checkpoint)
- **Files modified:** 9

## Accomplishments
- Production entrypoint (scripts/main.py) initializes APScheduler with SQLAlchemy jobstore and registers 6 cron jobs with correct schedules (equity 4:15 PM ET, crypto 4H+5m UTC, daily summary 6 PM ET, stuck agent 5:30 PM ET, weekly fundamentals Sun 6 PM ET, monthly macro 1st 6 PM ET)
- Crypto stop-price polling daemon thread checks halt flag and monitors positions at 60s intervals with full exception recovery
- Job-to-embed-to-alerter callback chain wired end-to-end: equity_cycle and daily_summary_job produce embeds and route through Alerter.send_embed
- Dockerfile production CMD updated from sleep loop placeholder to scripts/main.py
- docker-compose.prod.yml extended with swingrl-dashboard service (512MB/0.5 CPU, read-only DB mounts, port 8501)

## Task Commits

Each task was committed atomically:

1. **Task 1: main.py entrypoint, stop-price polling, config defaults, integration tests** - `1929a80` (test: RED), `ba6dce8` (feat: GREEN)
2. **Task 2: Dockerfile CMD update, production compose with dashboard** - `c9e6d2f` (feat)
3. **Task 3: Verify full Phase 9 automation stack** - checkpoint approved by user

## Files Created/Modified
- `scripts/main.py` - Production entrypoint: APScheduler + daemon thread + signal handler
- `src/swingrl/scheduler/stop_polling.py` - Crypto stop-price polling daemon thread
- `src/swingrl/scheduler/jobs.py` - Extended with embed callback wiring in equity_cycle and daily_summary_job
- `tests/scheduler/test_main.py` - Tests for job registration, init sequence, SIGTERM, callback chain
- `tests/scheduler/test_stop_polling.py` - Tests for halt check, exception recovery
- `Dockerfile` - Production CMD updated to scripts/main.py
- `docker-compose.prod.yml` - Added swingrl-dashboard service with resource limits
- `pyproject.toml` - Added zoneinfo dependency

## Decisions Made
- Used threading.Event().wait() over signal.pause() for cross-platform main blocking (works on both Mac and Linux)
- Stop-price polling daemon thread with 60s interval and daemon=True (auto-terminates with main process)
- Dashboard service constrained to 512MB RAM / 0.5 CPU with read-only volume mounts for security

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 9 complete: all 4 plans executed (scheduler infra, Discord alerting, dashboard, entrypoint wiring)
- System ready for homelab CI validation and Phase 10 production hardening
- Full automation stack: APScheduler cron jobs, stop-price polling, Discord alerts, Streamlit dashboard, Docker production compose

## Self-Check: PASSED

All 6 files verified present. All 3 commit hashes verified in git log.

---
*Phase: 09-automation-and-monitoring*
*Completed: 2026-03-09*
