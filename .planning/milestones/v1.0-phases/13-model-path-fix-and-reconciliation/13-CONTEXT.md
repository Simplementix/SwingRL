# Phase 13: Model Path Fix and Reconciliation Scheduling - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the model path double-nesting bug so `ExecutionPipeline._load_models()` resolves models at `models/active/{env}/{algo}/model.zip`, and wire `PositionReconciler.reconcile()` as a daily APScheduler job. This phase closes PAPER-02 (model path), PAPER-09 (reconciler scheduling), the integration gap (main.py -> pipeline.py), and the broken Production Trading Cycle flow.

</domain>

<decisions>
## Implementation Decisions

### Model path fix
- Fix `main.py:250` only — remove `/ "active"` so it passes `Path(config.paths.models_dir)` to `ExecutionPipeline`
- `pipeline._load_models()` is the single place that appends `active/` to the path — this is correct and stays as-is
- `run_cycle.py`, `backtest.py`, and `train.py` already pass bare `models` — no changes needed
- Existing `model_not_found` log in `_load_models()` already logs the full resolved path — sufficient for debugging

### Reconciliation timing
- Run daily at ~5:00 PM ET, after the 4:15 PM equity trade cycle completes
- Run every day including weekends — catches broker-side changes or corporate action settlements
- Startup reconciliation (equity-only, before first trade) remains as-is from Phase 8

### Reconciliation alerting
- Discord alert only when mismatches are detected (silent on success)
- Daily summary job should include reconciliation status (last run time, any recent mismatches)

### Reconciliation failure behavior
- Alert-only on failure, trading continues — mismatch is a bookkeeping issue, not a trading risk
- Unexpected positions (symbols not in config) should be recorded in DB so the system is aware of capital allocation
- Consecutive failures (3+) escalate from warning to critical-level Discord alert

### Claude's Discretion
- Exact reconciliation job function signature and job registration pattern (follow existing job conventions in main.py)
- How consecutive failure tracking is implemented (counter in DB, in-memory, or config)
- Test structure and fixture design for the new reconciliation job
- Whether to add the reconciliation job count to the existing `scheduler_jobs_registered` log (11 -> 12)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PositionReconciler` (src/swingrl/execution/reconciliation.py): Already implemented and tested, just not scheduled
- `scripts/reconcile.py`: CLI entrypoint for manual reconciliation — pattern reference for the scheduled job
- `scripts/run_cycle.py:130-144`: Startup reconciliation pattern — import and instantiation reference
- 11 existing APScheduler jobs in `main.py:61-186`: Pattern for job registration (cron trigger, timezone, replace_existing)

### Established Patterns
- Job functions use module-level `_ctx: JobContext | None` pattern (Phase 9 decision)
- All scheduled jobs registered in `create_scheduler_and_register_jobs()` function
- Discord alerting via `Alerter` class from job context
- Equity-only reconciliation — crypto uses virtual balance, no broker-side reconciliation (Phase 8 tests confirm)

### Integration Points
- `main.py:61` — `create_scheduler_and_register_jobs()` where the new job gets registered
- `src/swingrl/scheduler/jobs.py` — where job functions are defined (needs new reconciliation job function)
- Daily summary job — needs to include reconciliation status

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard bug fix and job wiring following existing codebase patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 13-model-path-fix-and-reconciliation*
*Context gathered: 2026-03-10*
