---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Operational Deployment
status: executing
stopped_at: Phase 19.1 context gathered
last_updated: "2026-03-14T20:22:15.540Z"
last_activity: 2026-03-13 — 19-03 validate_memory.py, seed_memory_from_backtest.py, schema p_crisis fix, homelab CI pass, training pipeline launched (981 tests passing)
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-10)

**Core value:** Capital preservation through disciplined, automated risk management
**Current focus:** v1.1 Operational Deployment — Phase 19 (Model Training)

## Current Position

Phase: 19 of 24 (Model Training)
Plan: 03 at checkpoint — homelab training running in background (docker compose run --rm)
Status: Executing
Last activity: 2026-03-13 — 19-03 validate_memory.py, seed_memory_from_backtest.py, schema p_crisis fix, homelab CI pass, training pipeline launched (981 tests passing)

Progress: [██████████] 100% (v1.1 — all 5 plans written; training in progress)

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
- [Phase 19-01]: ollama_smart_model defaults to qwen3:14b per CONTEXT.md override (not qwen2.5:14b from spec)
- [Phase 19-01]: MemoryClient uses stdlib urllib (fail-open, no new dependency); nosec annotations for bandit B310/B101
- [Phase 19-01]: clamp_reward_weights: bounds applied pre-normalization only; post-normalized values may exceed bound (correct behavior)
- [Phase 19-01]: TUNING_GRID variants never include net_arch — all variants inherit [64,64] from TrainingOrchestrator
- [Phase 19-01]: check_ensemble_gate: MDD stored as negative float; gate checks abs(mdd) < 0.15
- [Phase 19-02]: hmmlearn stores diag covariances as (n,d,d) internally; covars_ setter expects (n,d) — extract np.diag() before setting
- [Phase 19-02]: HMM _ensure_label_order applies ridge regularization during reorder to prevent near-singular matrix errors from setter validation
- [Phase 19-02]: train_pipeline.py walk-forward blocks are per-DuckDB-connection (open/close per algo), not held pipeline-wide
- [Phase 19-02]: Checkpoint resume uses model.zip existence check (not DuckDB), --force bypasses
- [Phase 19-02]: MetaTrainingOrchestrator.run() delegates actual train() call to TrainingOrchestrator (no SB3 internals injected yet)
- [Phase 19-02]: train_pipeline._load_features_prices imports scripts/train.py via importlib.util to avoid circular import
- [Phase 19-02]: TDD mock_train_side_effect creates real files as side effect so _verify_deployment() passes on real filesystem
- [Phase 19-03]: validate_memory.py exits 1 with diagnostic for any schema gap; exits 0 only when all checks pass
- [Phase 19-03]: hmm_state_history base DDL now includes p_crisis DOUBLE DEFAULT 0.0 — 3-state schema fully reflected in schema.py

### Pending Todos

None.

### Roadmap Evolution

- Phase 19.1 inserted after Phase 19: Memory Agent Infrastructure and Training (URGENT) — Deploy Ollama with Qwen models as Docker service, build swingrl-memory REST API, re-run training with memory_agent.enabled=true

### Blockers/Concerns

- [Research]: APScheduler and SQLAlchemy are missing from pyproject.toml — must be added in Phase 20 before Docker stack deployment.
- [Research]: SAC buffer_size=200_000 cap is conservative but unvalidated against crypto policy quality. Monitor shadow Sharpe after first retrain cycle.
- [Research]: Ensemble Sharpe weights placeholder (TRAIN-05) RESOLVED in Phase 19-02 — real WF OOS Sharpe used.

## Session Continuity

Last session: 2026-03-14T20:22:15.538Z
Stopped at: Phase 19.1 context gathered
Resume file: .planning/phases/19.1-memory-agent-infrastructure-and-training/19.1-CONTEXT.md
