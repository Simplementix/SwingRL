---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Operational Deployment
status: executing
stopped_at: "18-02-PLAN.md complete"
last_updated: "2026-03-11T11:55:33Z"
last_activity: 2026-03-11 — Phase 18 Plan 02 complete (ingest_all orchestrator, 19 tests, 892 total passing)
progress:
  total_phases: 7
  completed_phases: 0
  total_plans: 0
  completed_plans: 2
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Capital preservation through disciplined, automated risk management
**Current focus:** v1.1 Operational Deployment — Phase 18 (Data Ingestion)

## Current Position

Phase: 18 of 24 (Data Ingestion)
Plan: 02 complete — Phase 18 complete (both plans done)
Status: Executing
Last activity: 2026-03-11 — 18-02 ingest_all orchestrator complete (19 new tests, 892 total passing, ruff/bandit/mypy clean)

Progress: [░░░░░░░░░░] 0% (v1.1)

## Accumulated Context

### Decisions

- [18-01]: Use ALL_SERIES constant from swingrl.data.fred rather than config.fred.series (field does not exist on SwingRLConfig).
- [18-01]: Crypto gap threshold is timedelta(hours=8) — 2x the 4H bar cadence.
- [18-01]: Date range checks are always-pass (informational) — avoid false failures when data is sparse.
- [18-02]: Subprocess CLI tests inject PYTHONPATH=src/ explicitly; uv-managed Python does not process .pth editable install files in subprocesses.
- [18-02]: resolve_crypto_gaps uses ffill(limit=2) — gaps >2 bars raise DataError requiring manual repair.
- [18-02]: Feature computation gate sums all three stage deltas (equity+crypto+macro); features skip only when total==0.
- [v1.1 Roadmap]: Phases are strictly sequential — no parallelism. Data (18) → Training (19) → Deployment (20) → Discord (21) → Retraining (22) → Monitoring (23) → Runbook (24).
- [v1.1 Roadmap]: MON requirements assigned to Phase 23 (dedicated monitoring phase) rather than merged into Phase 20, consistent with fine granularity.
- [v1.0]: CPU-only training on homelab (no MPS transfer). DummyVecEnv over SubprocVecEnv. SAC buffer_size=200_000 for Docker.
- [v1.0]: APScheduler must be pinned <4.0. SQLAlchemy>=2.0,<3 required as companion dep.

### Pending Todos

None.

### Blockers/Concerns

- [Research]: APScheduler and SQLAlchemy are missing from pyproject.toml — must be added in Phase 20 before Docker stack deployment.
- [Research]: SAC buffer_size=200_000 cap is conservative but unvalidated against crypto policy quality. Monitor shadow Sharpe after first retrain cycle.
- [Research]: Ensemble Sharpe weights are placeholder in train.py (known tech debt). Phase 19 must resolve this before paper trading begins.

## Session Continuity

Last session: 2026-03-11T11:55:33Z
Stopped at: "Completed 18-02-PLAN.md"
Resume file: .planning/phases/18-data-ingestion/18-02-SUMMARY.md
