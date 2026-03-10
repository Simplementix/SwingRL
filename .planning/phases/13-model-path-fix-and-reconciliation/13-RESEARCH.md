# Phase 13: Model Path Fix and Reconciliation Scheduling - Research

**Researched:** 2026-03-10
**Domain:** Bug fix (path double-nesting) + APScheduler job wiring (PositionReconciler)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Model path fix:**
- Fix `scripts/main.py:250` only — remove `/ "active"` so it passes `Path(config.paths.models_dir)` to `ExecutionPipeline`
- `pipeline._load_models()` is the single place that appends `active/` to the path — this is correct and stays as-is
- `run_cycle.py`, `backtest.py`, and `train.py` already pass bare `models` — no changes needed
- Existing `model_not_found` log in `_load_models()` already logs the full resolved path — sufficient for debugging

**Reconciliation timing:**
- Run daily at ~5:00 PM ET, after the 4:15 PM equity trade cycle completes
- Run every day including weekends — catches broker-side changes or corporate action settlements
- Startup reconciliation (equity-only, before first trade) remains as-is from Phase 8

**Reconciliation alerting:**
- Discord alert only when mismatches are detected (silent on success)
- Daily summary job should include reconciliation status (last run time, any recent mismatches)

**Reconciliation failure behavior:**
- Alert-only on failure, trading continues — mismatch is a bookkeeping issue, not a trading risk
- Unexpected positions (symbols not in config) should be recorded in DB so the system is aware of capital allocation
- Consecutive failures (3+) escalate from warning to critical-level Discord alert

### Claude's Discretion
- Exact reconciliation job function signature and job registration pattern (follow existing job conventions in main.py)
- How consecutive failure tracking is implemented (counter in DB, in-memory, or config)
- Test structure and fixture design for the new reconciliation job
- Whether to add the reconciliation job count to the existing `scheduler_jobs_registered` log (11 -> 12)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-02 | Binance.US simulated fills for crypto environment (real-time prices, local fill recording) — blocked by broken Production Trading Cycle (model path bug prevents any model loading) | Model path fix in `scripts/main.py:250` unblocks the full cycle; `BinanceSimAdapter` already exists |
| PAPER-09 | 5-stage execution middleware: Signal Interpreter, Position Sizer, Order Validator, Exchange Adapter, Fill Processor — all 5 stages exist and work; blocked only by model loading failure and missing reconciliation scheduling | Path fix closes PAPER-02/09 together; reconciliation job wiring closes PAPER-09's scheduling gap |
</phase_requirements>

---

## Summary

Phase 13 is a precision gap-closure phase with two tightly scoped tasks. The first is a one-line bug fix: `scripts/main.py:250` passes `Path(config.paths.models_dir) / "active"` to `ExecutionPipeline`, but `ExecutionPipeline._load_models()` already appends `active/{env}/{algo}/model.zip` internally. This causes a double-nesting path of `models/active/active/{env}/{algo}/model.zip`, which never exists. The fix is to pass just `Path(config.paths.models_dir)` — no other callers are affected.

The second task wires `PositionReconciler.reconcile("equity")` as a daily APScheduler job in `create_scheduler_and_register_jobs()`. `PositionReconciler` is fully implemented and tested; it just has no scheduled caller. A new `reconciliation_job()` function goes in `src/swingrl/scheduler/jobs.py` following the exact same `_ctx: JobContext | None` pattern as the 11 existing jobs. The job registers in `scripts/main.py` at 5:00 PM ET daily. The job count log changes from 11 to 12.

With both tasks complete, the Production Trading Cycle works end-to-end: model loads, ensemble blends, signals generate, orders submit via adapters, fills record in DB, and reconciliation catches any drift daily. PAPER-02 and PAPER-09 close; PAPER-01, TRAIN-06, and PROD-02 (same root cause) also benefit.

**Primary recommendation:** Fix the single line in `scripts/main.py`, then add `reconciliation_job()` to `jobs.py` and register it in `create_scheduler_and_register_jobs()`. Follow existing job patterns exactly — no new patterns required.

---

## Standard Stack

### Core (already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| APScheduler | 3.x (pinned) | Cron job scheduling | Already in use for 11 jobs; `CronTrigger` pattern established |
| structlog | pinned | Structured logging per module | Project standard; all job functions use it |
| pathlib.Path | stdlib | File path manipulation | Project convention; `models_dir` is always `Path` |

### No new dependencies required.

**Installation:** None.

---

## Architecture Patterns

### Established Job Registration Pattern

All 11 existing jobs in `scripts/main.py` follow this exact structure:

```python
# In create_scheduler_and_register_jobs():
scheduler.add_job(
    reconciliation_job,           # function from swingrl.scheduler.jobs
    trigger="cron",
    hour=17,
    minute=0,
    timezone="America/New_York",  # always ET for trading-hours jobs
    id="daily_reconciliation",
    replace_existing=True,        # required for restart recovery
)
```

### Established Job Function Pattern

All 11 existing job functions in `src/swingrl/scheduler/jobs.py` follow this structure:

```python
def reconciliation_job() -> None:
    """Reconcile DB positions against broker state (equity only).

    Runs daily at 5 PM ET after equity trade cycle. Alert-only on mismatch;
    trading continues regardless. Consecutive failures (3+) escalate to critical.
    """
    ctx = _get_ctx()

    if is_halted(ctx.db):
        log.warning("reconciliation_job_skipped", reason="halt_flag_active")
        return

    try:
        from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter  # noqa: PLC0415
        from swingrl.execution.reconciliation import PositionReconciler       # noqa: PLC0415

        adapter = AlpacaAdapter(config=ctx.config, alerter=ctx.alerter)
        reconciler = PositionReconciler(
            config=ctx.config,
            db=ctx.db,
            adapter=adapter,
            alerter=ctx.alerter,
        )
        adjustments = reconciler.reconcile("equity")
        log.info("reconciliation_job_complete", adjustments=len(adjustments))
    except Exception:
        log.exception("reconciliation_job_failed")
```

Key conventions observed in existing jobs:
- Import-guarded local imports (`noqa: PLC0415`) for optional/lazy dependencies
- `try/except Exception` wraps everything to never crash the scheduler
- `is_halted` check at the top (reconciliation DOES check halt — unlike backup jobs)
- No return value (`-> None`) for maintenance jobs
- `_get_ctx()` call at the top

### Halt Check Decision

Reconciliation SHOULD check the halt flag (unlike backup jobs). Rationale: reconciliation is a trading support function; if the system is halted by emergency stop, there is no active trading to drift from. The existing startup reconciliation in `scripts/run_cycle.py:130-144` only runs when `not dry_run`, following the same conservative logic.

### Consecutive Failure Tracking

The CONTEXT.md leaves this to Claude's discretion. The simplest approach matching the codebase's in-memory patterns (Phase 9 decision: module-level `_ctx`):

```python
# Module-level counter (resets on container restart — acceptable for this use case)
_reconciliation_failures: int = 0

def reconciliation_job() -> None:
    global _reconciliation_failures  # noqa: PLW0603
    ctx = _get_ctx()
    ...
    try:
        ...
        _reconciliation_failures = 0  # reset on success
    except Exception:
        _reconciliation_failures += 1
        level = "critical" if _reconciliation_failures >= 3 else "warning"
        ctx.alerter.send_alert(level, "Reconciliation Failed", ...)
        log.exception("reconciliation_job_failed", consecutive=_reconciliation_failures)
```

This is consistent with how `consecutive_failures_before_alert` is tracked in `Alerter` (also in-memory, per `config.alerting.consecutive_failures_before_alert`). No DB table required.

### Model Path Fix — Exact Change

**Before (line 250 in `scripts/main.py`):**
```python
pipeline = ExecutionPipeline(
    ...
    models_dir=Path(config.paths.models_dir) / "active",
)
```

**After:**
```python
pipeline = ExecutionPipeline(
    ...
    models_dir=Path(config.paths.models_dir),
)
```

**Why this is the only change needed:**
`ExecutionPipeline._load_models()` in `src/swingrl/execution/pipeline.py:305` already builds the full path:
```python
model_path = self._models_dir / "active" / env_name / algo_name / "model.zip"
```
With the fix, `self._models_dir = Path("models/")`, so the resolved path becomes `models/active/equity/ppo/model.zip` — correct.

With the bug, `self._models_dir = Path("models/active")`, so the resolved path becomes `models/active/active/equity/ppo/model.zip` — never exists.

### Daily Summary Job — Reconciliation Status

CONTEXT.md says the daily summary job should include reconciliation status. The simplest approach: add a `_last_reconciliation_time: datetime | None` module-level variable updated after each successful reconcile, and include it in the daily summary embed/message.

Alternatively (simpler, no state), the `daily_summary_job()` can query the SQLite `trades` table for any adjustment trades from the last 24 hours with `reason LIKE 'reconcile:%'` to include a count. This approach is stateless and survives container restarts.

### Anti-Patterns to Avoid

- **Do not add `/ "active"` anywhere else.** Only `pipeline._load_models()` should construct the full path. All callers pass the bare `models_dir`.
- **Do not add retry logic to the reconciliation job.** The job is called again tomorrow; alert-only is the specified behavior.
- **Do not block trading on reconciliation.** It is a maintenance job, not a gate.
- **Do not run crypto reconciliation.** `PositionReconciler.reconcile("crypto")` immediately returns `[]` by design — crypto uses virtual balance. Only schedule `reconcile("equity")`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cron scheduling | Custom threading timer | APScheduler `add_job(trigger="cron")` | Already in use; 11 examples in `create_scheduler_and_register_jobs()` |
| Job context access | Pass deps as function args | `_get_ctx()` module-level pattern | Phase 9 decision; all jobs use it |
| Position reconciliation logic | Custom DB/broker diff | `PositionReconciler.reconcile()` | Fully implemented and tested in Phase 8; 5 tests covering all mismatch cases |
| Broker position fetch | Direct Alpaca API call | `AlpacaAdapter.get_positions()` | Broker middleware pattern; never call broker APIs outside `execution/` |

---

## Common Pitfalls

### Pitfall 1: Updating Job Count in Two Places

**What goes wrong:** The job count appears in two places: `log.info("scheduler_jobs_registered", count=11)` in `create_scheduler_and_register_jobs()` AND `log.info("swingrl_app_built", job_count=11)` in `build_app()`. If only one is updated to 12, the other stays stale.

**Why it happens:** The count is hardcoded in two log calls. When adding a job, it's easy to update only one.

**How to avoid:** Update both log calls from 11 to 12 in `scripts/main.py`.

**Warning signs:** The test `test_main_registers_all_jobs` checks `mock_scheduler.add_job.call_count == 11`. This test MUST be updated to `== 12` when the new job is added — otherwise it will fail and surface the discrepancy.

### Pitfall 2: Missing Import in `scripts/main.py`

**What goes wrong:** `reconciliation_job` is defined in `jobs.py` but the `from swingrl.scheduler.jobs import ...` block at the top of `scripts/main.py` must include it. Forgetting this causes a `NameError` at startup.

**How to avoid:** Add `reconciliation_job` to the import block at `scripts/main.py:27-40` alongside the other 11 job functions.

### Pitfall 3: Reconciliation Job Skips Halt Check (Unlike Backup Jobs)

**What goes wrong:** Backup jobs intentionally skip `is_halted()` (they run even when trading is halted). A developer might follow the backup job pattern for reconciliation and omit the halt check.

**Why it happens:** Two patterns exist in `jobs.py`: jobs with halt checks and backup jobs without. The reconciliation job should use the halt-check pattern (same as `equity_cycle`, `daily_summary_job`, `stuck_agent_check_job`).

**How to avoid:** Follow `equity_cycle`/`daily_summary_job` pattern, not `daily_backup_job` pattern.

### Pitfall 4: `build_app` References Stale `log.info` with job_count=11

**What goes wrong:** `build_app()` at line 283 logs `job_count=11`. This is a code clarity issue, not a functional bug — but it creates confusion in logs and potentially fails any log-based monitoring that checks job count.

**How to avoid:** Update `job_count=11` to `job_count=12` in `build_app()`.

### Pitfall 5: models_dir Fix Breaks `test_build_app_creates_pipeline_with_all_args`

**What goes wrong:** The existing test in `tests/scheduler/test_main.py::TestMainInitSequence::test_build_app_creates_pipeline_with_all_args` verifies that `ExecutionPipeline` is called with a `models_dir` kwarg, but the mock config's `paths.models_dir` is a `MagicMock`. After the fix, `Path(config.paths.models_dir)` (no `/ "active"`) will behave differently depending on what `MagicMock.__truediv__` returns. The test should still pass since it only checks `"models_dir" in call_kwargs`.

**How to avoid:** Run the existing test suite after the fix and confirm it passes.

---

## Code Examples

### Verified Pattern: Adding a Job to `create_scheduler_and_register_jobs()`

Source: `scripts/main.py:166-184` (shadow_promotion_check, automated_trigger_check — most recently added jobs)

```python
# In create_scheduler_and_register_jobs():
scheduler.add_job(
    reconciliation_job,
    trigger="cron",
    hour=17,
    minute=0,
    timezone="America/New_York",
    id="daily_reconciliation",
    replace_existing=True,
)

log.info("scheduler_jobs_registered", count=12)  # was 11
```

### Verified Pattern: Import Block Extension in `scripts/main.py`

Source: `scripts/main.py:27-40`

```python
from swingrl.scheduler.jobs import (
    automated_trigger_check_job,
    crypto_cycle,
    daily_backup_job,
    daily_summary_job,
    equity_cycle,
    init_job_context,
    monthly_macro_job,
    monthly_offsite_job,
    reconciliation_job,          # ADD THIS
    shadow_promotion_check_job,
    stuck_agent_check_job,
    weekly_duckdb_backup_job,
    weekly_fundamentals_job,
)
```

### Verified Pattern: PositionReconciler Instantiation

Source: `scripts/reconcile.py:91-106` (CLI reference pattern)

```python
from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter
from swingrl.execution.reconciliation import PositionReconciler

adapter = AlpacaAdapter(config=ctx.config, alerter=ctx.alerter)
reconciler = PositionReconciler(
    config=ctx.config,
    db=ctx.db,
    adapter=adapter,
    alerter=ctx.alerter,
)
adjustments = reconciler.reconcile("equity")
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Manual `scripts/reconcile.py` only | APScheduler daily job at 5 PM ET | Automatic daily reconciliation without operator intervention |
| `models_dir = Path(models_dir) / "active"` in `build_app()` | `models_dir = Path(models_dir)` (pipeline handles `active/` internally) | Model loading works; Production Trading Cycle unblocked |

---

## Open Questions

1. **Daily summary reconciliation status inclusion**
   - What we know: CONTEXT.md says daily summary should include reconciliation status (last run time, any recent mismatches)
   - What's unclear: Whether to add a module-level `_last_reconciliation_time` variable or query the `trades` table for adjustment records
   - Recommendation: Query-based approach (no module-level state) — query `trades` WHERE `reason LIKE 'reconcile:%'` AND `timestamp > now - 24h`. Stateless and survives restarts. Update `build_daily_summary_embed` if it accepts an extra kwarg, or append to the plain-text fallback in `daily_summary_job`.

2. **`daily_summary_job` embed signature**
   - What we know: `build_daily_summary_embed(equity_snapshot, crypto_snapshot, equity_trades_today, crypto_trades_today)` — current signature in `jobs.py:231`
   - What's unclear: Whether the embed builder accepts a `reconciliation_status` field or if reconciliation status should only go in the plain-text fallback
   - Recommendation: Add reconciliation mismatch count to the plain-text fallback path only (lowest risk, no embed schema change). Treat embed path as out of scope for this phase.

---

## Validation Architecture

`nyquist_validation` is enabled in `.planning/config.json`.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/scheduler/test_main.py tests/scheduler/test_jobs.py tests/execution/test_pipeline.py -x -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-02 | model path resolves to `models/active/{env}/{algo}/model.zip` (no double `active`) | unit | `uv run pytest tests/scheduler/test_main.py::TestMainInitSequence::test_build_app_creates_pipeline_with_all_args -x` | ✅ (needs update) |
| PAPER-09 | `reconciliation_job` registered at 5 PM ET with id `daily_reconciliation` | unit | `uv run pytest tests/scheduler/test_main.py::TestMainRegistersJobs -x` | ✅ (needs update) |
| PAPER-09 | `reconciliation_job()` calls `PositionReconciler.reconcile("equity")` | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob -x` | ❌ Wave 0 |
| PAPER-09 | job skips when halt flag is active | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob::test_skips_when_halted -x` | ❌ Wave 0 |
| PAPER-09 | consecutive failures (3+) escalate to critical alert | unit | `uv run pytest tests/scheduler/test_jobs.py::TestReconciliationJob::test_consecutive_failures_escalate -x` | ❌ Wave 0 |
| PAPER-02 | `ExecutionPipeline._load_models()` path construction (no double nesting) | unit | `uv run pytest tests/execution/test_pipeline.py -k model_path -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/scheduler/ tests/execution/test_pipeline.py -x -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/scheduler/test_jobs.py::TestReconciliationJob` — new test class for `reconciliation_job()` (3 tests: success, halt, consecutive failures)
- [ ] `tests/execution/test_pipeline.py::test_load_models_path_construction` — verifies `models_dir / "active" / env / algo / "model.zip"` (not double-active)
- [ ] Update `tests/scheduler/test_main.py::TestMainRegistersJobs::test_main_registers_all_jobs` — change `call_count == 11` to `== 12`, add `"daily_reconciliation"` to `expected_ids`
- [ ] Framework install: None — pytest already installed

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `scripts/main.py:250` — bug confirmed at the exact line cited in CONTEXT.md
- Direct code inspection: `src/swingrl/execution/pipeline.py:305` — `_load_models()` appends `"active"` internally, confirmed
- Direct code inspection: `src/swingrl/execution/reconciliation.py` — `PositionReconciler` fully implemented, 4 methods, all mismatch cases handled
- Direct code inspection: `src/swingrl/scheduler/jobs.py` — 11 job functions, `_ctx` pattern, import-guarded local imports confirmed
- Direct code inspection: `scripts/main.py:61-186` — `create_scheduler_and_register_jobs()` with all 11 registrations, exact kwargs pattern confirmed

### Secondary (MEDIUM confidence)
- `tests/scheduler/test_main.py` — test expectations for job count (11) and job IDs confirmed; these tests will need updates
- `tests/execution/test_reconciliation.py` — reconciliation test patterns confirmed; no scheduled-job test exists yet
- `tests/scheduler/test_jobs.py` — job function test class patterns confirmed (TestEquityCycle, TestDailySummaryJob, etc.)

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all from existing codebase
- Architecture: HIGH — patterns read directly from source files; no ambiguity
- Pitfalls: HIGH — all identified from direct code inspection (two-location job count, import block, halt check branching)
- Test gaps: HIGH — existing test structure is clear; missing tests are precisely identified

**Research date:** 2026-03-10
**Valid until:** Stable indefinitely (internal codebase research; no external library dependencies added)
