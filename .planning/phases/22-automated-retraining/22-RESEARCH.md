# Phase 22: Automated Retraining - Research

**Researched:** 2026-03-16
**Domain:** APScheduler subprocess orchestration, walk-forward validation gating, bootstrap guard, DuckDB short-lived connections, rolling Sharpe, operator CLI
**Confidence:** HIGH — all findings drawn from existing codebase analysis and decisions locked in 22-CONTEXT.md

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training Approach**
- Configurable toggle: memory-enhanced (MetaTrainingOrchestrator) or baseline training. Config field `retraining.use_memory_agent` (default true). Operator can disable if LLM guidance degrades quality
- Single run per retrain cycle — not multi-iteration. MetaTrainingOrchestrator uses accumulated memories to guide one attempt
- Data freshness verification before retrain: equity less than 2 calendar days stale, crypto less than 12 hours stale. If stale, auto-run incremental ingest before training proceeds
- Same walk-forward fold parameters as initial training: equity (test_bars=63, embargo=10, min_train=252), crypto (test_bars=540, embargo=130, min_train=2190)
- Same ensemble gates: Sharpe > 1.0, MDD < 15%. Sortino as ranking metric, Calmar as tiebreaker
- All 3 algos every cycle: PPO, A2C, SAC retrained sequentially for the target environment. Fresh Sharpe-softmax ensemble weights recalculated from new walk-forward OOS results
- Memory agent guides training, not weights: ensemble weights stay mathematical (Sharpe-softmax formula)

**Scheduling (Hybrid: Calendar + Performance Trigger)**
- Calendar schedule: equity monthly (default Saturday 2 AM ET), crypto biweekly (default Sunday 3 AM ET). Both configurable via cron expressions in config
- Performance trigger: rolling 30-day Sharpe per environment. Retrain triggered if Sharpe declines >20% from deployment baseline OR absolute Sharpe drops below 0.5. Both thresholds configurable
- Rolling Sharpe calculation pulled forward from Phase 23 into Phase 22
- Sequential lock: only one retrain runs at a time. File lock prevents overlap. If equity retrain is still running when crypto is due, crypto waits until next scheduled window
- Duration tracking: actual training duration stored in DuckDB per retrain. Discord retrain-started embed includes estimated duration from last run
- Configurable start times: cron expressions in config with defaults above

**Failure Recovery**
- 3 consecutive crash failures: auto-disable retrain schedule for that environment. Fire CRITICAL Discord alert. Operator must re-enable via CLI after investigating
- Gate failure != crash failure: a retrain that completes training but fails the ensemble gate is NOT counted as failure for the auto-disable counter
- Partial algo failure = full retrain failure: if any algo (PPO, A2C, or SAC) fails, entire retrain is marked failed. Don't deploy partial results to shadow
- Timeout: configurable, default 72 hours. Timeout = full failure, counts toward 3-failure counter
- Re-enable: CLI command `docker exec swingrl python scripts/retrain.py --enable equity|crypto`. Sets flag in SQLite. No container restart required
- --dry-run mode: full training pipeline runs but skips deploy_to_shadow(). Logs results, fires Discord completed embed with [DRY RUN] tag
- SAC buffer cap: uses existing sac_buffer_size from TrainingConfig (500K). OOM caught as crash failure
- Memory agent ingests failure context on retrain failure

**Bootstrap Guard**
- Dual condition: minimum 50 trades AND 14 calendar days since shadow deployment (both configurable per env). Both must be met before promotion check can run
- Independent per environment: equity and crypto have separate trade counters, day counters, and promotion decisions
- Better candidate replaces current: if new retrain produces model with better WF OOS Sortino than current shadow, replace shadow model. Bootstrap timer resets
- Reuse existing shadow_promotion_check_job: add bootstrap guard check to the existing daily 7 PM ET promoter job
- Rejected models archived: move to models/archive/ with timestamp, clear shadow slot. Discord shadow rejected embed fires
- Promotion events triple-logged: memory ingest + Discord embed (gold) + DuckDB record

**Resource Contention**
- Subprocess inside swingrl container: spawned via subprocess.Popen (NOT multiprocessing.Process). Popen + exec = fresh Python interpreter, no inherited threads/connections
- swingrl container at 16GB/8CPU (already configured in docker-compose.prod.yml)
- Concurrent with live trading: retrain subprocess runs at nice +10. Applied via preexec_fn in Popen
- SubprocVecEnv(n_envs=6) + 3 algo workers with fork start method (proven)
- Fail-open on Ollama: if Ollama/memory agent unavailable during retrain, falls back to defaults

**DuckDB Concurrent Access (CRITICAL)**
- Change DatabaseManager.duckdb() context manager to open fresh connection and close on exit. Persistent singleton removed
- One file change in src/swingrl/data/db.py — all callers benefit
- DuckDB WAL auto-recovers from crash mid-write

**Memory Agent Alignment**
- Consolidate after every retrain: success or failure. POST /consolidate call
- Per-algo + combined summary ingestion: individual PPO/A2C/SAC results + overall ensemble summary
- New ':retrain' source tag suffix: training_run:retrain, training_epoch:retrain, etc.
- Pass trigger reason to run_config: 'scheduled_monthly', 'scheduled_biweekly', 'performance_degradation (Sharpe dropped 22%)'
- Always ingest results even in baseline mode
- Auto-seed backtest fills after retrain: ingest new backtest fills with ':retrain' source tag
- Same timeouts as Phase 19: epoch_advice=8s, run_config=15s, consolidation=60s

**Operator CLI**
- Single script scripts/retrain.py with subcommands via flags: --env, --manual, --dry-run, --status, --enable/--disable, --history
- Manual retrain requires interactive confirmation prompt
- --status shows comprehensive state including bootstrap progress

**Retrain Notifications**
- Retrain completed embed includes memory summary (hyperparams suggested, whether clamped, Sharpe delta)
- Retrain failed embed includes memory analysis of why training failed
- Performance-triggered retrain gets WARNING embed to #alerts; calendar retrains get INFO to #daily
- All embeds follow Phase 21 severity matrix and branding

**Retrain History and Reporting**
- Same training_runs table + run_type column: add run_type ('initial', 'retrain_scheduled', 'retrain_triggered', 'manual') and trigger_reason columns
- Always compare with active model: store active model Sharpe/MDD/Sortino/Calmar at retrain time
- Generation counter: monotonically increasing version number per environment (equity_gen_1, equity_gen_2, etc.)
- Duration tracked granularly: total + PPO + A2C + SAC durations

**Testing Strategy**
- Mock training, real orchestration: mock SB3 trainer.learn() to return dummy models in seconds
- Real model files: mock side-effect creates real SB3 model.zip + vec_normalize files
- Homelab smoke test: --dry-run --timesteps 1000 as part of deployment verification

### Claude's Discretion
- Exact cron expression syntax and APScheduler job configuration
- Rolling Sharpe calculation implementation details (window size, minimum observations)
- File lock implementation for sequential retrain guard
- Confirmation prompt UX for manual retrain
- DuckDB schema details for new columns (run_type, trigger_reason, generation)
- Exact memory ingestion format for retrain-specific events
- How to detect "performance degradation resolved" state

### Deferred Ideas (OUT OF SCOPE)
- Phase 25 (Dashboard updates): retrain history visualization, model generation timeline, Sharpe trend charts
- Obs space expansion (156→164 equity, 45→53 crypto)
- Per-algo retraining (only retrain the weakest algo)
- Rolling/one-algo-at-a-time retraining
- Separate retrain container
- Dynamic Docker resource limits during retrain
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RETRAIN-01 | Equity retraining job runs monthly via APScheduler as subprocess (not in thread pool) | APScheduler CronTrigger with day_of_week="sat", hour=2, timezone="America/New_York"; subprocess.Popen with preexec_fn=os.nice(10) |
| RETRAIN-02 | Crypto retraining job runs biweekly via APScheduler as subprocess | APScheduler CronTrigger or IntervalTrigger for Sunday 3 AM ET biweekly; same Popen pattern |
| RETRAIN-03 | Walk-forward validation gate runs on newly trained model before shadow deployment | Reuse existing check_validation_gates() from src/swingrl/agents/validation.py; check_ensemble_gate() from pipeline_helpers.py |
| RETRAIN-04 | New model deploys to shadow, existing shadow promotion logic evaluates and auto-promotes if better | Reuse ModelLifecycle.deploy_to_shadow(); existing shadow_promotion_check_job already wired; add bootstrap guard |
| RETRAIN-05 | Shadow promoter bootstrap guard prevents promotion when trades table is empty | Add dual-condition check (trades count + days since deploy) in evaluate_shadow_promotion() before 3-criterion logic |
</phase_requirements>

---

## Summary

Phase 22 is an orchestration phase — the training and shadow promotion infrastructure already exists. The work is: (1) wrapping scripts/train_pipeline.py in a RetrainingOrchestrator callable from APScheduler, (2) adding two cron-scheduled subprocess jobs to jobs.py and main.py, (3) bolting a bootstrap guard onto the existing evaluate_shadow_promotion(), (4) fixing DuckDB's persistent singleton to use short-lived connections, (5) pulling forward rolling Sharpe for performance-triggered retraining, and (6) writing the operator CLI script.

The critical prerequisite is the DuckDB singleton fix. DatabaseManager._get_duckdb_conn() currently opens a persistent connection that is never closed. When main.py holds this connection and a retrain subprocess also tries to open a write connection to the same .ddb file, DuckDB raises a conflicting lock error. Changing duckdb() to open and close a fresh connection per context manager call resolves this entirely — the lock is held for milliseconds, collision probability on 4H trading cycles is negligible.

The bootstrap guard in evaluate_shadow_promotion() is a straightforward dual-condition check added before the existing 3-criterion logic. The promoter reads the shadow deployment timestamp from SQLite retrain_state and counts live trades for the environment in the trades table. If either condition is unmet it logs progress and returns False early — the 3-criterion evaluation never runs.

**Primary recommendation:** Implement in this wave order: (W0) DuckDB fix + RetrainingConfig schema + DB migrations, (W1) RetrainingOrchestrator + rolling Sharpe, (W2) APScheduler job registration + subprocess spawning, (W3) bootstrap guard + promoter updates, (W4) operator CLI + Discord embeds + TDD coverage, (W5) integration test + homelab smoke.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| apscheduler | <4.0 (pinned) | Cron job scheduling | Already in use; STATE.md pin confirmed. APScheduler 4.x has breaking API changes |
| subprocess (stdlib) | Python 3.11 | Spawn retrain process | Clean interpreter isolation; no inherited DuckDB handles or threads from main.py |
| duckdb | current | Market data read/write | Already in use; fix is to short-lived connections only |
| filelock | current | Sequential retrain guard | Prevents two retrains overlapping; file-based so subprocess-safe |
| sqlite3 (stdlib) | Python 3.11 | Retrain state flags (enabled/disabled), shadow deployment timestamps | Already in trading_ops.db |
| structlog | current | Structured logging | Project-wide standard |
| pydantic v2 | current | RetrainingConfig schema | Consistent with all other config classes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| filelock | current | Cross-subprocess lock for sequential retrain guard | Required for subprocess.Popen isolation; threading.Lock does not work across process boundaries |
| datetime / zoneinfo | stdlib | ET timezone handling, calendar day calculations | Already used throughout project |
| argparse | stdlib | Operator CLI subcommand parsing | Consistent with scripts/train_pipeline.py pattern |
| os.nice() | stdlib | Lower retrain process CPU priority | Applied via preexec_fn in Popen |

**Note on filelock:** The sequential retrain guard must be a filesystem lock, not a threading.Lock, because the retrain subprocess is a separate OS process (subprocess.Popen). Threading locks are intra-process only. filelock.FileLock is simpler and cross-platform versus fcntl.flock (Linux-only).

**Installation:**
```bash
uv add filelock
```

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| subprocess.Popen | multiprocessing.Process | Popen gives clean exec isolation (no inherited DuckDB connections, no APScheduler threads); multiprocessing.Process shares file descriptors with parent via fork |
| filelock | fcntl.flock | fcntl is Linux-only and lower-level; filelock is Pythonic and the standard choice |
| Short-lived DuckDB connections | DuckDB WAL mode | WAL only helps multiple readers; single-writer lock is still enforced per DuckDB semantics |

---

## Architecture Patterns

### Recommended Project Structure

New files for Phase 22:

```
src/swingrl/
├── training/
│   └── retraining.py          # RetrainingOrchestrator + run_retrain() entry point
├── scheduler/
│   └── jobs.py                # ADD: equity_retrain_job(), crypto_retrain_job()
└── shadow/
    └── promoter.py            # MODIFY: add bootstrap guard to evaluate_shadow_promotion()

scripts/
└── retrain.py                 # Operator CLI (--manual, --status, --enable, --disable, --history)

tests/
├── training/
│   └── test_retraining.py     # RetrainingOrchestrator unit tests (new file)
├── scheduler/
│   └── test_jobs.py           # ADD retrain job tests (file exists, add cases)
└── shadow/
    └── test_promoter.py       # ADD bootstrap guard tests (file exists, add cases)
```

Modified files:

```
src/swingrl/
├── data/db.py                 # CRITICAL: change duckdb() from persistent singleton to short-lived
├── config/schema.py           # ADD: RetrainingConfig class to SwingRLConfig
└── shadow/promoter.py         # MODIFY: bootstrap guard before 3-criterion check
```

DuckDB schema migration (in DatabaseManager._init_duckdb_schema, idempotent ALTER):

```
training_runs table additions:
  run_type TEXT DEFAULT 'initial'     -- initial | retrain_scheduled | retrain_triggered | manual
  trigger_reason TEXT                  -- 'scheduled_monthly', 'performance_degradation (Sharpe -22%)', etc.
  generation INTEGER DEFAULT 1         -- monotonically increasing per environment
  total_duration_sec DOUBLE
  ppo_duration_sec DOUBLE
  a2c_duration_sec DOUBLE
  sac_duration_sec DOUBLE
```

SQLite schema additions (in trading_ops.db):

```sql
CREATE TABLE IF NOT EXISTS retrain_state (
    environment TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    consecutive_failures INTEGER DEFAULT 0,
    last_failure_reason TEXT,
    last_retrain_at TEXT,
    shadow_deployed_at TEXT,
    baseline_sharpe REAL,
    current_generation INTEGER DEFAULT 1
);
```

### Pattern 1: RetrainingOrchestrator Entry Point

The APScheduler job calls a thin wrapper that spawns the retrain subprocess and holds the file lock until completion:

```python
# src/swingrl/training/retraining.py (conceptual)
import os
import subprocess
from pathlib import Path

def spawn_retrain_subprocess(env_name: str, config_path: Path, dry_run: bool = False) -> int:
    """Spawn retrain as subprocess with nice +10 priority. Returns exit code."""
    cmd = [
        "/app/.venv/bin/python",
        "scripts/retrain.py",
        "--env", env_name,
        "--config", str(config_path),
        "--scheduler-mode",  # skip interactive confirmation
    ]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.Popen(
        cmd,
        preexec_fn=lambda: os.nice(10),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc.wait()  # blocks APScheduler worker thread until done
```

The APScheduler job acquires the file lock first. If lock is already held, it logs "retrain already running" and returns immediately. It does NOT spawn a new process. The scheduler thread remains blocked for the duration of training — this is intentional. The scheduler's ThreadPoolExecutor handles other jobs concurrently.

### Pattern 2: DuckDB Short-Lived Connection Fix

**This is the highest-priority change in Phase 22.** The current _get_duckdb_conn() returns a persistent connection held for the lifetime of the DatabaseManager singleton. The fix:

```python
# src/swingrl/data/db.py — modified duckdb() method
@contextlib.contextmanager
def duckdb(self) -> Generator[Any, None, None]:
    """Open fresh DuckDB connection, yield, close. No persistent state.

    Lock is held only for the duration of the context block (milliseconds
    for reads, seconds for writes). Enables concurrent subprocess access.
    """
    self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(self._duckdb_path))
    try:
        yield conn
    finally:
        conn.close()
```

All existing callers use `with db.duckdb() as cursor:` — the API surface is unchanged. The _duckdb_conn field and _get_duckdb_conn() method are removed. The close() method no longer needs to close a DuckDB connection.

**Impact on existing callers:** Zero API changes required. train_pipeline.py already opens DuckDB connections inline (deferred pattern from Phase 19.1), so those are unaffected.

### Pattern 3: Bootstrap Guard in evaluate_shadow_promotion()

Insert as the first check in evaluate_shadow_promotion(), before the existing shadow_count check:

```python
# src/swingrl/shadow/promoter.py — addition at top of evaluate_shadow_promotion()
def _check_bootstrap_guard(
    config: Any, db: Any, env_name: str
) -> tuple[bool, str]:
    """Check dual conditions: min trades (live) AND min days since shadow deploy.

    Uses LIVE trades table (not shadow_trades) — bootstrap guards against
    spurious promotion before enough real market performance data exists.
    """
    min_trades = getattr(config.retraining, f"bootstrap_min_trades_{env_name}", 50)
    min_days = getattr(config.retraining, f"bootstrap_min_days_{env_name}", 14)

    with db.sqlite() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE environment = ?",
            (env_name,),
        ).fetchone()
        trade_count = row["cnt"] if row else 0

        row2 = conn.execute(
            "SELECT shadow_deployed_at FROM retrain_state WHERE environment = ?",
            (env_name,),
        ).fetchone()

    if trade_count < min_trades:
        return False, f"bootstrap: {trade_count}/{min_trades} trades"

    shadow_deployed_at = row2["shadow_deployed_at"] if row2 else None
    if shadow_deployed_at is None:
        return False, "bootstrap: no shadow deployment recorded"

    from datetime import UTC, datetime  # noqa: PLC0415
    deployed = datetime.fromisoformat(shadow_deployed_at).replace(tzinfo=UTC)
    days_elapsed = (datetime.now(tz=UTC) - deployed).days
    if days_elapsed < min_days:
        return False, f"bootstrap: {days_elapsed}/{min_days} days elapsed"

    return True, ""
```

### Pattern 4: Rolling Sharpe for Performance Trigger

Rolling 30-day Sharpe on live trade returns, computed from the trades table:

```python
def compute_rolling_sharpe(
    db: Any,
    env_name: str,
    window_days: int = 30,
    min_observations: int = 10,
) -> float | None:
    """Compute rolling Sharpe from live trades over last window_days.

    Aggregates fills to daily P&L series. Annualizes with sqrt(252) for equity,
    sqrt(1460) for crypto (6 bars/day * 365.25 days). Returns None if fewer
    than min_observations non-zero P&L days exist in window.
    """
```

Key implementation detail: aggregate by trading day (not per fill). A 30-day window on 4H fills gives ~180 fills for crypto but a meaningful 30-point daily P&L series. For equity, days with no trades produce P&L = 0. Require 10 non-zero P&L days before returning valid Sharpe — this prevents an all-cash period from triggering a retrain due to Sharpe = 0.

Baseline Sharpe storage: when a model is promoted to active, record its walk-forward OOS Sharpe in retrain_state.baseline_sharpe. The performance trigger compares current rolling Sharpe against this value.

### Pattern 5: Sequential Lock via filelock

```python
import filelock
from pathlib import Path

_RETRAIN_LOCK_PATH = Path("/tmp/swingrl_retrain.lock")  # noqa: S108

def try_acquire_retrain_lock() -> filelock.FileLock | None:
    """Non-blocking lock acquisition. Returns None if already held."""
    lock = filelock.FileLock(str(_RETRAIN_LOCK_PATH))
    try:
        lock.acquire(timeout=0)
        return lock
    except filelock.Timeout:
        return None
```

The lock is acquired by the APScheduler job thread before spawning the subprocess. The thread holds the lock for the full duration of subprocess execution (proc.wait()). On timeout or crash, the lock is released in a finally block.

### Pattern 6: Failure Counter in SQLite

The 3-failure auto-disable uses the retrain_state table's consecutive_failures column. Reset to 0 on any successful retrain (gate pass or gate fail — both count as non-crash). Increment on crash or timeout only. Gate failures do NOT increment. When consecutive_failures >= 3, set enabled = 0 and fire CRITICAL Discord alert.

**Return code convention for RetrainingOrchestrator:**
- EXIT 0: training succeeded, gate passed, deployed to shadow
- EXIT 1: gate failed (not a crash; do not increment failure counter)
- EXIT 2: crash/exception during training
- EXIT 3: timeout
- EXIT 4: data freshness check failed (not a crash; do not increment)

### Anti-Patterns to Avoid

- **Holding DuckDB connection across training loop:** Training takes 11.5h. Opening a write connection for 11.5h blocks all other access. Open only for brief read (features) and write (results) operations.
- **Spawning retrain as multiprocessing.Process:** Shares file descriptors with parent. DuckDB connection inherited from main.py causes lock conflict in child. Use subprocess.Popen.
- **Threading lock for sequential guard:** threading.Lock is per-process. subprocess.Popen creates a new OS process. Only filesystem locks (filelock) work cross-process.
- **Counting shadow_trades for bootstrap guard:** The bootstrap guard should count LIVE trades (trades table), not shadow_trades. Shadow trades are simulated; the guard's purpose is to ensure enough real market performance data exists.
- **Not waiting for subprocess in APScheduler job:** The job function must call proc.wait() while holding the file lock. Returning before subprocess terminates releases the lock prematurely.
- **Incrementing failure counter on gate failure:** Gate failure (model quality) is not a system failure. Only crash (exception), OOM, or timeout increments the counter.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Subprocess file lock | Custom lock file with PID | filelock.FileLock | Handles stale lock cleanup, cross-platform, well-tested |
| Cron scheduling | Custom time.sleep() loop | APScheduler CronTrigger | Already wired in main.py, SQLAlchemy persistence, timezone-aware |
| Walk-forward validation | Re-implement gate logic | check_validation_gates() + check_ensemble_gate() from pipeline_helpers.py | Already tested, exact same gates used in initial training |
| Shadow deployment | Custom file copy logic | ModelLifecycle.deploy_to_shadow() | Already handles path creation, error logging |
| DuckDB WAL recovery | Custom recovery logic | DuckDB built-in WAL | DuckDB auto-recovers from WAL file on next connection open |
| Ensemble weight calculation | Custom Sharpe formula | compute_ensemble_weights_from_wf() | Already tested with Sharpe-softmax formula |
| Memory ingestion | Direct HTTP calls | MemoryClient with fail-open | Already handles urllib, timeouts, fail-open pattern |

---

## Common Pitfalls

### Pitfall 1: DuckDB Persistent Connection Not Fixed Before Retrain
**What goes wrong:** Retrain subprocess raises IOException: Conflicting lock when trying to open DuckDB. First retrain attempt fails. Auto-disable counter increments.
**Why it happens:** DatabaseManager._get_duckdb_conn() holds persistent connection. Not fixed before retrain jobs go live.
**How to avoid:** DuckDB fix MUST be Wave 0 (first wave). Add regression test: verify duckdb() context manager opens fresh connection each call (no _duckdb_conn field retained).
**Warning signs:** duckdb.IOException or Conflicting lock in retrain subprocess stderr.

### Pitfall 2: Counting Shadow Trades Instead of Live Trades for Bootstrap
**What goes wrong:** Bootstrap guard returns True immediately on fresh deploy because shadow_trades accumulate quickly during shadow evaluation.
**Why it happens:** Confusion between shadow_trades table (simulated) and trades table (live).
**How to avoid:** Bootstrap guard explicitly queries trades table (live fills). Document this clearly in docstring.
**Warning signs:** Bootstrap guard passes after only a few hours of shadow evaluation.

### Pitfall 3: APScheduler Biweekly Schedule Syntax
**What goes wrong:** Using weeks=2 on a CronTrigger which APScheduler does not support directly. IntervalTrigger with weeks=2 fires every 2 weeks from first run, not on a fixed day of week.
**Why it happens:** "Biweekly on Sunday" requires CronTrigger with week='*/2' (every other week) or IntervalTrigger anchored to a specific Sunday start_date.
**How to avoid:** Use CronTrigger(day_of_week='sun', week='*/2', hour=3, timezone='America/New_York') or use IntervalTrigger(weeks=2, start_date='<next Sunday 3 AM ET>'). Verify with APScheduler <4.0 docs — week parameter uses ISO week numbers. Add unit test calling get_next_fire_time() to verify correct cadence.
**Warning signs:** Crypto retrain fires every week instead of every two weeks.

### Pitfall 4: Gate Failure Incrementing Crash Counter
**What goes wrong:** A retrain that trains all 3 algos successfully but fails Sharpe > 1.0 gate increments consecutive_failures. After 3 such cycles, retrain is auto-disabled even though the training pipeline is healthy.
**Why it happens:** Conflating gate failure (model quality issue) with crash failure (system issue).
**How to avoid:** RetrainingOrchestrator must use distinct exit codes: 0=success, 1=gate_failed, 2=crash, 3=timeout. Only exit codes 2 and 3 increment consecutive_failures. Validate with explicit test.
**Warning signs:** retrain_state.enabled flipped to 0 even though no exceptions occurred.

### Pitfall 5: Sequential Lock Released Before Subprocess Terminates
**What goes wrong:** APScheduler job spawns retrain subprocess, immediately returns (releasing lock), and a second retrain job fires before the first subprocess finishes.
**Why it happens:** subprocess.Popen is non-blocking. If the job returns before calling proc.wait(), lock window is milliseconds.
**How to avoid:** The APScheduler job thread MUST call proc.wait() while holding the file lock. The job runs in a ThreadPoolExecutor thread, so it can block safely. Verify scheduler max_workers (default 4) is sufficient so other jobs are not starved.

### Pitfall 6: Config Serialization for Subprocess Workers
**What goes wrong:** Attempting to pass SwingRLConfig to ProcessPoolExecutor workers inside the retrain subprocess fails because _ConfigWithYaml local class is not serializable.
**Why it happens:** load_config() creates a local subclass that cannot be serialized by ProcessPoolExecutor.
**How to avoid:** Already solved in train_pipeline.py — workers receive config as a plain dict, reconstruct via SwingRLConfig(**config_dict). The retrain subprocess receives the config PATH only; it calls load_config(path) fresh — no serialization issue for the subprocess.Popen launch itself.
**Note:** Only relevant for ProcessPoolExecutor workers WITHIN the retrain subprocess, not for the Popen spawn.

### Pitfall 7: Generation Counter Race Condition
**What goes wrong:** Two retrains fire concurrently (race condition before lock is acquired), both read the same current generation, both write generation N+1.
**Why it happens:** Time-of-check to time-of-use gap in generation counter read/write.
**How to avoid:** Read and increment generation inside a SQLite BEGIN EXCLUSIVE transaction. SQLite serializes writes. The sequential file lock is the primary protection; SQLite transaction is defense-in-depth.

---

## Code Examples

Verified patterns from existing codebase:

### APScheduler CronTrigger Registration (from scripts/main.py)
```python
# Source: scripts/main.py lines 73-197 (existing pattern to follow)
scheduler.add_job(
    equity_retrain_job,
    trigger="cron",
    day_of_week="sat",
    hour=2,
    minute=0,
    timezone="America/New_York",
    id="equity_retrain",
    replace_existing=True,
    misfire_grace_time=config.scheduler.misfire_grace_time,
)
```

### Subprocess Spawn with CPU Priority (from CONTEXT.md decision)
```python
# Source: 22-CONTEXT.md decision: "preexec_fn=lambda: os.nice(10) in subprocess.Popen"
import os
import subprocess

proc = subprocess.Popen(
    ["/app/.venv/bin/python", "scripts/retrain.py", "--env", "equity", "--scheduler-mode"],
    preexec_fn=lambda: os.nice(10),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)
return_code = proc.wait()
```

### DuckDB Short-Lived Connection Fix
```python
# Source: analysis of src/swingrl/data/db.py lines 99-110 (current pattern to replace)
@contextlib.contextmanager
def duckdb(self) -> Generator[Any, None, None]:
    """Open fresh connection, yield, close. No persistent state."""
    self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(self._duckdb_path))
    try:
        yield conn
    finally:
        conn.close()
```

### check_ensemble_gate Usage (from pipeline_helpers.py)
```python
# Source: src/swingrl/training/pipeline_helpers.py (existing functions)
from swingrl.training.pipeline_helpers import check_ensemble_gate, compute_ensemble_weights_from_wf

gate_ok, sharpe, mdd = check_ensemble_gate(wf_results, env_name)
if not gate_ok:
    log.warning("retrain_gate_failed", env=env_name, sharpe=sharpe, mdd=mdd)
    # fire Discord retrain-failed embed, do NOT call deploy_to_shadow()
    return RetrainResult(status="gate_failed")

weights = compute_ensemble_weights_from_wf(wf_results, env_name)
```

### evaluate_shadow_promotion() Bootstrap Guard Insertion Point
```python
# Source: src/swingrl/shadow/promoter.py lines 29-51 (existing function to modify)
# Bootstrap guard is inserted BEFORE the existing shadow_count check:
def evaluate_shadow_promotion(config, db, env_name, lifecycle, alerter) -> bool:
    # NEW Phase 22: bootstrap guard
    guard_passed, guard_reason = _check_bootstrap_guard(config, db, env_name)
    if not guard_passed:
        log.info("bootstrap_guard_not_met", env=env_name, reason=guard_reason)
        return False

    # EXISTING: min eval threshold check (kept unchanged)
    min_trades = config.shadow.equity_eval_days if env_name == "equity" ...
```

### Reconciliation Failure Counter as Model (from jobs.py)
```python
# Source: src/swingrl/scheduler/jobs.py lines 432-479 — existing pattern to mirror
# The reconciliation_job uses _reconciliation_failures module-level counter.
# For retrain, use SQLite retrain_state table instead (persists across restarts).
_reconciliation_failures: int = 0  # module-level — resets on restart
# vs.
# retrain_state.consecutive_failures — SQLite — survives container restart
```

### RetrainingConfig Schema (new Pydantic model)
```python
# src/swingrl/config/schema.py — new class following existing patterns
class RetrainingConfig(BaseModel):
    """Automated retraining schedule and threshold configuration."""

    use_memory_agent: bool = True
    equity_cron_day_of_week: str = "sat"
    equity_cron_hour: int = Field(default=2, ge=0, le=23)
    crypto_cron_day_of_week: str = "sun"
    crypto_cron_hour: int = Field(default=3, ge=0, le=23)
    timeout_hours: int = Field(default=72, ge=1)
    sharpe_decline_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    sharpe_floor: float = Field(default=0.5, gt=0.0)
    rolling_sharpe_window_days: int = Field(default=30, ge=7)
    rolling_sharpe_min_observations: int = Field(default=10, ge=3)
    bootstrap_min_trades_equity: int = Field(default=50, ge=1)
    bootstrap_min_trades_crypto: int = Field(default=50, ge=1)
    bootstrap_min_days_equity: int = Field(default=14, ge=1)
    bootstrap_min_days_crypto: int = Field(default=14, ge=1)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DummyVecEnv(1), 1 CPU, SAC buffer 200K | SubprocVecEnv(6), 3 parallel workers, SAC buffer 500K | Phase 19.1 | Training time reduced from 23h+ to ~11.5h per env |
| Persistent DuckDB singleton in DatabaseManager | Short-lived open/close per context manager | Phase 22 (to implement) | Enables concurrent subprocess retraining without lock conflicts |
| No automated retraining | APScheduler subprocess jobs for monthly equity + biweekly crypto | Phase 22 (to implement) | Models stay current without operator intervention |
| No bootstrap guard | Dual-condition guard (50 trades + 14 days) before promotion | Phase 22 (to implement) | Prevents spurious promotion on fresh deployment |

**Deprecated/outdated:**
- DatabaseManager._get_duckdb_conn() persistent pattern: replaced by per-call open/close in Phase 22
- train_pipeline.py multi-iteration logic: NOT reused for automated retraining — single run per cycle per CONTEXT.md decision

---

## Open Questions

1. **APScheduler biweekly cron exact syntax**
   - What we know: APScheduler <4.0 supports week parameter in CronTrigger. week='*/2' should fire every other ISO week.
   - What's unclear: Whether week='*/2' anchors to ISO week 1 (January) or first trigger date. Behavior depends on when system is first started.
   - Recommendation: Use IntervalTrigger(weeks=2, start_date='<next Sunday 3 AM ET>') for predictable biweekly firing. Add unit test calling get_next_fire_time() to verify correct cadence before shipping.

2. **Rolling Sharpe aggregation unit for minimum observations**
   - What we know: Window is 30 calendar days. Crypto trades every 4H (up to 6 fills/day). Equity trades less frequently (0-3 fills/day).
   - What's unclear: Should 10 observations mean 10 daily P&L points or 10 individual fills? For equity with infrequent trading, all-cash days produce P&L = 0.
   - Recommendation: Aggregate by trading day (not fill). Require 10 non-zero P&L days before returning valid Sharpe. Days with no trades produce 0 daily P&L and are excluded from the count. This prevents an all-cash period from triggering a retrain due to Sharpe = 0.

3. **shadow_deployed_at source of truth**
   - What we know: model_metadata table exists in DuckDB (OLAP). retrain_state in SQLite (operational). Bootstrap guard needs shadow_deployed_at for every evaluation run.
   - What's unclear: Whether to store shadow_deployed_at in DuckDB model_metadata or SQLite retrain_state.
   - Recommendation: Store in SQLite retrain_state (operational state, fast reads, no DuckDB lock concern for the frequent daily promoter job). DuckDB model_metadata records the deployment event for historical queries. Both are written at shadow deploy time.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml (existing) |
| Quick run command | `uv run pytest tests/training/test_retraining.py tests/shadow/test_promoter.py tests/scheduler/test_jobs.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RETRAIN-01 | equity_retrain_job spawns subprocess (not thread), uses Popen | unit | `uv run pytest tests/scheduler/test_jobs.py::test_equity_retrain_job_spawns_subprocess -x` | ❌ Wave 0 |
| RETRAIN-02 | crypto_retrain_job spawns subprocess biweekly via APScheduler | unit | `uv run pytest tests/scheduler/test_jobs.py::test_crypto_retrain_job_spawns_subprocess -x` | ❌ Wave 0 |
| RETRAIN-03 | Walk-forward gates checked before deploy_to_shadow(); gate failure aborts | unit | `uv run pytest tests/training/test_retraining.py::test_gate_failure_aborts_shadow_deployment -x` | ❌ Wave 0 |
| RETRAIN-04 | Successful retrain calls deploy_to_shadow(); shadow promoter evaluates next day | integration | `uv run pytest tests/training/test_retraining.py::test_successful_retrain_deploys_to_shadow -x` | ❌ Wave 0 |
| RETRAIN-05 | bootstrap guard returns False when live trades < min_required | unit | `uv run pytest tests/shadow/test_promoter.py::test_bootstrap_guard_insufficient_trades -x` | ❌ Wave 0 |
| RETRAIN-05b | bootstrap guard returns False when fewer than min_days since shadow deploy | unit | `uv run pytest tests/shadow/test_promoter.py::test_bootstrap_guard_insufficient_days -x` | ❌ Wave 0 |
| RETRAIN-05c | bootstrap guard returns True when both conditions met; 3-criterion check runs | unit | `uv run pytest tests/shadow/test_promoter.py::test_bootstrap_guard_passes_both_conditions -x` | ❌ Wave 0 |
| DuckDB fix | duckdb() opens fresh connection each call; no persistent _duckdb_conn | unit | `uv run pytest tests/data/test_db.py::test_duckdb_short_lived_connection -x` | ❌ Wave 0 |
| failure counter | gate failure does NOT increment consecutive_failures | unit | `uv run pytest tests/training/test_retraining.py::test_gate_failure_does_not_increment_counter -x` | ❌ Wave 0 |
| failure counter | 3 crash failures sets enabled=False and fires CRITICAL alert | unit | `uv run pytest tests/training/test_retraining.py::test_three_crashes_disables_retrain -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/training/test_retraining.py tests/shadow/test_promoter.py tests/scheduler/test_jobs.py -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/training/test_retraining.py` — new file; covers RETRAIN-03, RETRAIN-04, failure counter, gate vs crash distinction
- [ ] `tests/data/test_db.py::test_duckdb_short_lived_connection` — add to existing file; verifies DuckDB fix
- [ ] `retrain_state` SQLite table DDL added to `DatabaseManager._init_sqlite_schema()`
- [ ] DuckDB `training_runs` table ALTER columns for run_type, trigger_reason, generation, duration fields
- [ ] `RetrainingConfig` class added to `src/swingrl/config/schema.py` and wired to `SwingRLConfig`
- [ ] `filelock` added to `pyproject.toml` dependencies

*(Existing files tests/shadow/test_promoter.py and tests/scheduler/test_jobs.py already exist — new test functions are added to those files, not new files created)*

---

## Sources

### Primary (HIGH confidence)
- `src/swingrl/data/db.py` — DatabaseManager singleton pattern, duckdb() persistent connection (lines 91-110); current problem identified
- `src/swingrl/shadow/promoter.py` — evaluate_shadow_promotion() full implementation, 3-criterion logic
- `src/swingrl/scheduler/jobs.py` — 12 existing jobs, JobContext pattern, reconciliation failure counter (module-level) as model
- `scripts/main.py` — APScheduler setup, CronTrigger syntax, all 12 job registrations
- `src/swingrl/agents/validation.py` — check_validation_gates() 4 gates, GateResult dataclass
- `src/swingrl/training/pipeline_helpers.py` — check_ensemble_gate(), compute_ensemble_weights_from_wf(), gate constants
- `src/swingrl/shadow/lifecycle.py` — deploy_to_shadow(), promote(), archive_shadow()
- `src/swingrl/config/schema.py` — SwingRLConfig, all existing config classes as model for RetrainingConfig
- `.planning/phases/22-automated-retraining/22-CONTEXT.md` — All locked decisions
- `.planning/phases/22-automated-retraining/22-DUCKDB-CONTENTION.md` — DuckDB lock analysis, resolution options, SubprocVecEnv findings
- `.planning/STATE.md` — APScheduler must be pinned <4.0 decision (line 45)

### Secondary (MEDIUM confidence)
- `scripts/train_pipeline.py` — Existing pipeline structure that RetrainingOrchestrator wraps; config dict serialization fix pattern already in use
- `src/swingrl/memory/training/meta_orchestrator.py` — MetaTrainingOrchestrator API for memory-enhanced retrain path
- `src/swingrl/memory/client.py` — MemoryClient fail-open pattern and timeout handling

### Tertiary (LOW confidence)
- APScheduler week='*/2' biweekly behavior: requires validation with get_next_fire_time() in unit test; not directly verified against APScheduler <4.0 source code

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project; only filelock is new
- Architecture: HIGH — patterns derived directly from existing codebase analysis; no speculation
- Pitfalls: HIGH — most pitfalls discovered during Phase 19.1 DuckDB contention investigation (documented in 22-DUCKDB-CONTENTION.md) or are direct consequences of subprocess vs thread semantics
- Rolling Sharpe: MEDIUM — implementation details are Claude's discretion; window/aggregation logic is reasonable but not battle-tested against live data

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable domain; APScheduler <4.0 pin already set; DuckDB semantics are stable)
