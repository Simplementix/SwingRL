# Training Pipeline Reference

Living reference for SwingRL's training pipeline — the multi-iteration walk-forward orchestrator that drives baseline + memory-enhanced training, manages per-env state, triggers consolidation, and records outcomes. The pipeline lives in a single ~3K-line script (`scripts/train_pipeline.py`) plus the per-fold backtester (`src/swingrl/agents/backtest.py`); everything else is a helper module pulled in at function-call time.

**Last verified against code:** 2026-05-07

**Honest-gap policy:** every concrete claim is `file:line`-cited. Cross-table schema, source-tag taxonomy, and consolidation internals are deliberately not restated — see [`training-data-capture.md`](training-data-capture.md), [`memory-system.md`](memory-system.md), [`reward-shaping.md`](reward-shaping.md), and [`agent-architecture.md`](agent-architecture.md).

## Architecture at a glance

| Layer | File | Role |
|-------|------|------|
| Iteration loop | `scripts/train_pipeline.py:348` (`run_all_iterations`) | Iterate 0..N, configure baseline vs memory-enhanced, dispatch per env, consolidate after each |
| Per-env driver | `scripts/train_pipeline.py:2057` (`run_environment`) | Walk-forward + ensemble + gate + persistence per (iteration, env) |
| Per-fold backtester | `src/swingrl/agents/backtest.py:240` (`WalkForwardBacktester`) | Train/eval one algo across N folds; mark control vs treatment |
| Fold generator | `src/swingrl/agents/backtest.py:181` (`generate_folds`) | Growing-window non-overlapping folds with embargo gap |
| CLI parser | `scripts/train_pipeline.py:2775` (`build_parser`) | Argparse surface for the trainer |
| Wrapper script | `scripts/run_iterations.sh` | Shell helper that picks venv vs `uv run` |
| Multi-iter state | `data/training_state.json` (atomic write at `:85`) | Resume support across crashes |
| Production CMD | `Dockerfile:97` → `python scripts/main.py` | APScheduler daemon (trading + monitoring jobs only — **no training job**) |

**Trigger model:** training is **manual-only**. The production container's CMD launches an APScheduler with 12 cron jobs (`scripts/main.py:73-197`), none of which calls `train_pipeline.py`. Iterations are kicked off via `docker exec` (see [Entry points & invocation](#entry-points--invocation)).

## Iteration anatomy

### What is an "iteration"?

An iteration is one full pass through walk-forward training across both envs, ending with consolidation + outcome recording. Iterations are numbered `0..N` where `N = --iterations` CLI flag. Boundary marker: each iteration's `(env_name → result)` dict is checkpointed under `state[f"iteration_{i}_result"]` after both envs complete (`scripts/train_pipeline.py:500-510`).

### Baseline vs memory-enhanced

- **Iteration 0 — baseline.** `cfg.memory_agent.enabled = False`, `cfg.memory_agent.meta_training = False` (`scripts/train_pipeline.py:388-392`). Memory service is not contacted; HPs come from yaml + per-algo defaults.
- **Iterations 1..N — memory-enhanced.** Both flags flipped to `True` (`scripts/train_pipeline.py:394-397`). Before training starts, the memory service is health-probed (`check_memory_service_health` at `:285`); if down, `wait_for_memory_service` (`:304`) retries; if still unreachable the iteration **silently falls back to baseline** for that pass (`:401-410`, log line `memory_service_unavailable_falling_back_to_baseline`).

The iter-0 → iter-1 transition is load-bearing: per the docstring at `:552-554`, "Baseline WF data must be consolidated into patterns before iteration 1 starts so the meta-trainer has context to work with."

### Per-env loop

Both envs run **sequentially** within one iteration: `for env_name in ["equity", "crypto"]:` at `scripts/train_pipeline.py:438-499`. There is no parallel-envs path. Inside this loop:

1. Resume check — if `state[f"iteration_{i}_env_{env_name}"]` exists from a prior crash, skip (`:438-447`).
2. `run_environment()` (`:2057`) — full WF + ensemble + gate + persistence for this `(iteration, env)`.
3. Per-env checkpoint — `state[partial_key] = env_result; save_training_state(...)` (`:460-461`, atomic via `os.replace` at `:97`).
4. Retry-once on exception — second `run_environment()` call wrapped in inner try/except (`:466-498`); on second failure the env's slot becomes `{"error": ..., "original_error": ...}` and the iteration continues to the next env.

### Phase order within one iteration

```
iter i ─┬─ memory health gate (i > 0 only)            train_pipeline.py:401-410
        ├─ data ingestion (run_ingestion_pipeline)    train_pipeline.py:411-431
        ├─ for env in [equity, crypto]:
        │    └─ run_environment(env, iter=i):         train_pipeline.py:2057
        │         ├─ feature/price load via data_loader
        │         ├─ meta-orchestrator advice (iter ≥ 1)
        │         ├─ walk-forward per algo (PPO/A2C/SAC, parallel)
        │         ├─ ensemble weights + gate (Sharpe/MDD)
        │         ├─ ingest WF results (per-algo + ensemble)
        │         ├─ ingest trading patterns (per-algo)
        │         ├─ ingest run summaries (per-algo)
        │         ├─ persist iteration_results to pg16 (autocommit=True)
        │         └─ compute + persist CPS
        ├─ cross-iteration narrative ingest (iter > 0) train_pipeline.py:539
        ├─ /consolidate (Stage 1 per-env + Stage 2)   train_pipeline.py:567
        └─ /training/record_outcome × 2 (one per env)  train_pipeline.py:576-583
```

Each step's main entrypoint is cited above; the chronological order matches the source. Cross-link to [`reward-shaping.md`](reward-shaping.md) for what happens *inside* each fold's training loop.

### Iteration counter & resume

- Counter source: `--iterations` CLI flag (`scripts/train_pipeline.py:2821-2828`). The loop variable `i in range(iterations + 1)` is the canonical iteration number (`:380`).
- Persistence: `data/training_state.json` (default `_DEFAULT_STATE_PATH` at `:70`) — JSON dict with `completed_iterations: list[int]` and `current_iteration: int`.
- Resume: at iteration boundary, if `i in state["completed_iterations"]` the loop logs `iteration_skipped_checkpointed` and continues (`:381-383`). Per-env partial keys (`iteration_{i}_env_{env_name}`) are popped after the iteration's full result is saved (`:511-512`).

## Walk-forward fold structure

### Fold definition

`FoldResult` dataclass (`src/swingrl/agents/backtest.py:62-90`) — fields: `fold_number, train_range, test_range, in_sample_metrics, out_of_sample_metrics, trades, gate_result, overfitting, converged_at_step, total_timesteps, is_control_fold, advice_stats`.

### `generate_folds()` — growing window with embargo

`src/swingrl/agents/backtest.py:181-237`. Pure function; no DB or env coupling.

```
fold k:  train = range(0, train_end_k)        # always starts at 0 → growing
         test  = range(test_start_k, test_start_k + test_bars)
         test_start_{k+1} = test_start_k + test_bars + embargo_bars
```

The training window always starts at index 0 (`backtest.py:214`); only the test window slides forward by `test_bars + embargo_bars` per step (`:218`). `min_folds = 3`, raised as `DataError` if not met (`:220-227`).

### Per-env params (hardcoded — not yaml-tunable)

| Env | `test_bars` | `min_train_bars` | `embargo_bars` | `periods_per_year` | `bars_per_week` |
|-----|-------------|------------------|----------------|--------------------|-----------------|
| equity | 63 (~3 mo daily) | 252 (1 yr) | 10 (~2 wk) | 252 | 5 |
| crypto | 540 (~3 mo 4H) | 2190 (~1 yr) | 130 (~1 mo) | 2191.5 | 42 |

Source: `ENV_PARAMS` dict at `src/swingrl/agents/backtest.py:44-59`. Same `generate_folds()` code path is used for both envs; the only difference is the constant pack passed in.

### Per-fold execution order (one fold of one algo)

`WalkForwardBacktester.run()` at `src/swingrl/agents/backtest.py:300-470`:

1. Slice train + test data (`:357-360`).
2. `is_control = fold_idx in valid_control_set` (`:346`).
3. `orchestrator.train(...)` with `run_id=f"{env_name}_{algo_name}_fold{fold_idx}{ctrl_suffix}"` and `is_control_fold=is_control` (`:367-379`). Control folds get `advice_enabled=False` (`:363`).
4. `_evaluate_fold` on train data → in-sample metrics (`:382-390`).
5. `_evaluate_fold` on test data → out-of-sample metrics (`:393-401`).
6. `diagnose_overfitting(is_sharpe, oos_sharpe)` (`:404-407`).
7. `check_validation_gates(oos_sharpe, oos_mdd, profit_factor, overfit_gap)` (`:410-415`).
8. Build `FoldResult` (`:417-430`); enqueue to `fold_queue` for real-time DB write (`:434-439`); legacy DB write via `self._db._store_results` (`:441-444`).

### Control vs treatment folds

Yaml-driven via `memory_agent.control_folds_equity` / `control_folds_crypto` at `config/swingrl.yaml:114-115`:

- equity: `[0, 5, 10, 15, 20]` — 5 control folds
- crypto: `[0, 4, 9, 13]` — 4 control folds

Validated at runtime against actual fold count (`backtest.py:317-325`); out-of-range indices are logged (`control_fold_indices_out_of_range`) and dropped. Marking happens once per fold (`:346`) and propagates to `FoldResult.is_control_fold` (`:428`).

### `run_id` format

`{env_name}_{algo_name}_fold{fold_idx}[_CTRL]` — `src/swingrl/agents/backtest.py:375`. The `_CTRL` suffix is added when `is_control = True` (`:364`). Downstream consumers parse this with the regex `r"run_id=\S+_fold(\d+)(_CTRL)?"` — see [`memory-system.md`](memory-system.md) "Side effects" under `/training/epoch_advice`.

### Fold metric aggregation

After all folds + algos finish for one env, ensemble weights are computed via `compute_ensemble_weights_from_wf` (Sharpe-weighted softmax) and gated via `check_ensemble_gate` — both in `src/swingrl/training/pipeline_helpers.py`. Gate thresholds: `_GATE_MIN_SHARPE = 1.0`, `_GATE_MAX_MDD = 0.15` (`pipeline_helpers.py:57-58`).

## Entry points & invocation

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `--env` | `all` | One of `equity`, `crypto`, `all` (sequential) — `:2791-2795` |
| `--config` | `config/swingrl.yaml` | Path to config YAML — `:2797-2801` |
| `--models-dir` | `models` | Root directory for model storage — `:2803-2807` |
| `--report` | `data/training_report.json` | JSON training report path — `:2809-2813` |
| `--force` | `False` | Re-run even if checkpoints exist — `:2815-2819` |
| `--iterations` | `0` | N memory-enhanced iterations after baseline (0 = baseline only) — `:2821-2828` |
| `--state-path` | `data/training_state.json` | Resume state file — `:2830-2834` |
| `--comparison-path` | `data/training_comparison.json` | Cross-iteration comparison output — `:2836-2840` |
| `--skip-ingest` | `False` | Skip pre-iteration data ingestion — `:2842-2846` |

All from `build_parser()` at `scripts/train_pipeline.py:2775-2848`. `main()` at `:2851` dispatches: `--iterations > 0` → `run_all_iterations()`; otherwise the legacy single-pass per-env loop at `:2944-2972`.

### Documented commands

From `docs/TRAINING_RUNBOOK.md` and `scripts/run_iterations.sh`:

```bash
# Baseline + 5 memory iterations (homelab)
docker exec -d swingrl python scripts/train_pipeline.py --env all --iterations 5

# Wrapper script — defaults to ITERATIONS=5, picks /app/.venv (Docker) or uv (local)
ITERATIONS=3 bash scripts/run_iterations.sh

# Local dev (outside Docker)
uv run python scripts/train_pipeline.py --env all --iterations 5 --force
```

`scripts/run_iterations.sh:18-21` — the venv-vs-uv branch is the only useful logic in the wrapper; everything else passes straight through.

### Container model

Production target in `Dockerfile:59-97`:

- `FROM python:3.11-slim AS production` (`:59`)
- Non-root `trader` user, UID 1000 (`:63`)
- `uv sync --locked --no-install-project` against bind-mounted `uv.lock` + `pyproject.toml` (`:71-74`)
- `COPY --chown=trader:trader src/ /app/src/`, `config/`, `scripts/`, plus root `pyproject.toml` + `uv.lock` (`:77-80`)
- Final `uv sync --locked` to install the project itself (`:83-84`)
- HEALTHCHECK: `python scripts/healthcheck.py` (DB connectivity check) every 60s, 10s timeout, 3 retries (`:93-94`)
- **CMD: `["python", "scripts/main.py"]`** (`:97`) — APScheduler daemon

The CI target (`:15-48`) is separate and ends in `ENTRYPOINT ["uv", "run", "python", "-m", "swingrl"]` (`:48`). It exists for `ci-homelab.sh` and is overridden by `docker compose run --entrypoint ""` for tests.

### Compose service

`swingrl` service at `docker-compose.yml:63-96`:

- `target: production` (`:67`)
- `cpus: 10.0` (`:71`); no `mem_limit` (`:69-70` comment notes ~12-15 GB peak under SAC + SubprocVecEnv)
- `env_file: .env` (`:75`)
- `environment: SWINGRL_TRADING_MODE=paper, TZ=America/New_York` (`:76-78`)
- Bind mounts: `./data, ./db, ./models, ./logs, ./config, ./status` (`:81-86`)
- `depends_on: swingrl-memory` with `condition: service_started` — soft dependency, app handles unavailability gracefully (`:87-90`)

### `scripts/train.py` — legacy single-algo path

`scripts/train.py` is the older single-pass trainer, kept for reference and ad-hoc model rebuilds. A grep across `src/`, `scripts/`, and `tests/` for `from scripts.train` or `import scripts.train` returns no hits — nothing in the active codebase imports it. It is not invoked by `train_pipeline.py`, the scheduler, or any docker/compose target. Treat it as documentation of the older API; do not extend it.

### Healthcheck

Defined at `Dockerfile:93-94`; implementation in `scripts/healthcheck.py`. Verifies pg16 reachability via `SELECT 1`. Exit code 0 = healthy, non-zero = unhealthy. Does NOT check training liveness — only DB.

## Memory ingest call sites within the lifecycle

Schema, source-tag taxonomy, and consolidation phase semantics are documented in [`memory-system.md`](memory-system.md). This section gives only the *timing* — when in the iteration each ingest fires.

| Source prefix | Ingest call site | Trigger function | When in iteration |
|---------------|------------------|------------------|-------------------|
| `walk_forward:{env}:{algo}` | `train_pipeline.py:1283` | `_ingest_wf_results_to_memory` (def at `:991`) | After all folds + ensemble gate, per env |
| `walk_forward:{env}:ensemble` | `train_pipeline.py:1376` | (same function) | Immediately after per-algo block |
| `trading_pattern:{env}:{algo}` | `train_pipeline.py:1583` | `_ingest_trading_patterns_to_memory` (def at `:1383`) | After WF results ingest |
| `training_run:{env}` | `train_pipeline.py:1859` | `_ingest_run_summaries_to_memory` (def at `:1779`) | End of `run_environment()` for this env |
| `cross_iteration:{env}` | `train_pipeline.py:977` | `_ingest_cross_iteration_comparison` (def at `:760`, called from iteration loop at `:539`) | After both envs done, before `/consolidate` (iter > 0 only) |
| `training_epoch:{env}:{algo}` | inside SB3 epoch callback | `epoch_callback.py:436` | Per-cadence during fold training |
| `reward_adjustment:{env}:{algo}` | inside SB3 epoch callback | `epoch_callback.py:507-508` (trigger), `:562-563` (outcome) | When reward weights change |
| `curriculum_performance:historical` | inside curriculum loop | `curriculum.py:200` | At training-fold end |

`/consolidate` fires once per iteration at `train_pipeline.py:567` with `timeout = cfg.memory_agent.consolidation.timeout_sec` (default 1800s). `/training/record_outcome` fires twice (one per env) at `:576-583`, gated on `api_key` being non-empty.

## Failure modes & idempotency

### Crash recovery

| Crash point | Survives in pg16? | Survives in state file? | On restart |
|-------------|-------------------|--------------------------|------------|
| Mid-fold (e.g., fold 3 of 10) | None — `iteration_results` is written end-of-env (`:2305`) | Per-env partial-key written only on env success (`:460-461`) | Entire env re-runs from fold 0 |
| Mid-env (between folds, before result aggregation) | Per-fold rows in `backtest_results` if fold_queue worker drained (real-time write at `backtest.py:434-439`) | None for this env | Env re-runs from fold 0 — **fold-level results may be re-ingested into pg16 on retry** (no per-fold dedup) |
| After env A success, mid-env B | A's result in `state[iteration_{i}_env_equity]` + pg16 `iteration_results` row | Per-env partial key for A | Iteration loop resumes by skipping A (`:438-447`) and starting B fresh |
| After both envs, mid-`/consolidate` | Both envs' `iteration_results` rows + WF / trading-pattern memories | Per-env partial keys exist | On restart, iteration is **not** marked complete (no `completed_iterations` entry); the iteration retries from env A (which then no-ops via per-env resume) and re-fires `/consolidate`. See Phase A/B asymmetry below. |
| After `/consolidate` success, mid-`record_outcome` | All consolidation output | Per-env partial keys exist | Iteration not marked complete; retry can produce duplicate `pattern_outcomes` rows (no UNIQUE — see Known issues) |

### Per-env retry-once

Each env's `run_environment()` call is wrapped in two try/except blocks (`scripts/train_pipeline.py:466-498`). On first failure, the original exception is logged and the call is repeated. On second failure, the env's slot becomes `{"error": str(exc2), "original_error": str(exc)}` and the loop moves on. The original exception is chained via `exc2.__cause__ = exc` for traceback context.

### `iteration_results` autocommit caveat

The end-of-env pg16 write opens `psycopg.connect(database_url, ..., autocommit=True)` at `scripts/train_pipeline.py:2305`. The source comment at `:2295-2304` documents the iter-5 silent-rollback bug: the helper `store_iteration_results_to_duckdb` issues an INSERT but does not call `conn.commit()`; without `autocommit=True`, the `finally: conn_ens.close()` path rolls the INSERT back silently. Per-env retries are safe because the table has `UNIQUE (iteration_number, environment, run_type)` (`postgres_schema.py:216`) — a duplicate retry write fails the unique constraint cleanly. CPS uses a separate connection that explicitly commits (`:2330-2335`).

### `/consolidate` Phase A/B asymmetry on retry

Per [`memory-system.md`](memory-system.md) "Stage 1 Phase A — archive timing": Phase A archives source memories (`archived = 1`) only on LLM success (`services/memory/memory_agents/consolidate.py:1747`). Phase B archives independently. Consequence on retry:

- **Phase A re-run after success** — finds zero unarchived `walk_forward:`, `trading_pattern:`, `cross_iteration:`, `training_run:` memories. Phase A produces zero patterns. Effectively idempotent.
- **Phase A success → Phase B failure → retry** — Phase A re-run is a no-op as above; Phase B re-run hits the same `training_epoch:` / `reward_adjustment:` rows again. Dedup at `consolidate.py:2090-2108` is **exact match on `(category, affected_algos)`**; a near-duplicate LLM response with slightly different wording will land as a new active pattern. **Risk: Phase B duplicate patterns on retry.**

There is no distributed transaction across the four memory tables (`consolidations, consolidation_sources, llm_audit_log, consolidation_quality`). Each `insert_*` commits independently — see [`memory-system.md`](memory-system.md) "Persistence" under Stage 1 Phase A.

### Test/prod DB separation

`tests/conftest.py:36-60` enforces a `pytest_configure` guard: pytest exits with returncode 2 if `DATABASE_URL` doesn't end in `_test` and the DB name isn't in `_SAFE_DB_NAMES = {"swingrl_test"}`. The CI script `scripts/ci-homelab.sh:48-54` does override the URL — it parses the prod password out of `.env`, builds `TEST_DB_URL=postgresql://swingrl:${PG_PASS}@pg16:5432/swingrl_test`, drops + recreates the DB, then invokes pytest via `docker compose run --rm --entrypoint "" -e DATABASE_URL="$TEST_DB_URL" swingrl uv run pytest tests/ -v`.

The structural fix described in `MEMORY.md` Active Session State as "Plan B" has landed: the conftest guard exists, and the 18 `raise RuntimeError` test fixture disables noted in the same memory entry are no longer present. Per the 2026-04-07 incident summary in `MEMORY.md`, iter 0-4 production data was restored from duckdb backup; iter 5 was declared lost.

## Configurable values (yaml)

Per `config/swingrl.yaml` (and validated by `src/swingrl/config/schema.py`):

- `training.*` — n_envs, total_timesteps caps, sb3 verbose level
- `memory_agent.enabled` / `meta_training` — flipped per-iteration by `run_all_iterations` (so yaml value is the *baseline* state)
- `memory_agent.base_url` / `timeout_sec` / `api_key` — memory-service client config
- `memory_agent.consolidation.timeout_sec` — `/consolidate` wall-clock cap (default 1800)
- `memory_agent.control_folds_equity` / `control_folds_crypto` — control fold index lists (`:114-115`)
- `memory_agent.epoch_cadence_ppo` / `_a2c` / `_sac` / `_default` — per-algo advice cadence (`:108-110`); cross-link [`reward-shaping.md`](reward-shaping.md) and [`agent-architecture.md`](agent-architecture.md)
- Per-provider LLM config blocks — see [`memory-system.md`](memory-system.md) "Configurable values"

CLI flags override yaml defaults for: `--config` path itself, `--iterations`, `--env`, `--state-path`, `--comparison-path`, `--skip-ingest`, `--force`, `--models-dir`, `--report`.

**Honest gap:** `training.*` yaml field names were not exhaustively re-audited against `src/swingrl/config/schema.py` for this doc — re-verify field names and validators before changing any production value.

## Hardcoded values (not yaml-tunable — code edit required)

| Value | Location |
|-------|----------|
| `ENV_PARAMS["equity"]` = `{test_bars: 63, min_train_bars: 252, embargo_bars: 10, periods_per_year: 252, bars_per_week: 5}` | `src/swingrl/agents/backtest.py:44-51` |
| `ENV_PARAMS["crypto"]` = `{test_bars: 540, min_train_bars: 2190, embargo_bars: 130, periods_per_year: 2191.5, bars_per_week: 42}` | `src/swingrl/agents/backtest.py:52-58` |
| `min_folds = 3` | `src/swingrl/agents/backtest.py:186` |
| `RECENT_WINDOW_BARS["equity"] = 252 * 3`, `["crypto"] = 2191` | `src/swingrl/training/pipeline_helpers.py:39-42` |
| `DEFAULT_TIMESTEPS = {"equity": 1_000_000, "crypto": 500_000}` | `pipeline_helpers.py:45-48` |
| `ESCALATED_TIMESTEPS = {"equity": 2_000_000, "crypto": 1_000_000}` | `pipeline_helpers.py:51-54` |
| `_GATE_MIN_SHARPE = 1.0`, `_GATE_MAX_MDD = 0.15` | `pipeline_helpers.py:57-58` |
| `_DEFAULT_STATE_PATH = "data/training_state.json"` | `train_pipeline.py:70` |
| `_DEFAULT_COMPARISON_PATH = "data/training_comparison.json"` | `train_pipeline.py:71` |
| `iter_log_file = logs/training_iter{N}.log` | `train_pipeline.py:2867` |
| `run_id` format `{env}_{algo}_fold{N}[_CTRL]` | `backtest.py:375` |
| Per-env loop order `["equity", "crypto"]` | `train_pipeline.py:438` and `:2881` |
| Healthcheck cadence `60s / 10s / 3 retries` | `Dockerfile:93` |

## Invariants

- Iteration 0 is always baseline (`memory_agent.enabled = False`). The yaml value of `memory_agent.enabled` is a starting state; `run_all_iterations` flips it per iteration.
- Both envs run sequentially within one iteration; no parallel-envs path exists.
- Per-iteration boundary is persisted (`completed_iterations` list) only after both envs and `/consolidate` succeed; partial-iteration crashes leave the iteration marked incomplete and resumable.
- State file writes are POSIX-atomic via write-to-temp-then-rename (`scripts/train_pipeline.py:96-97`).
- Walk-forward training window is **growing** (always starts at index 0); only the test window slides forward.
- `min_folds = 3` enforced by `generate_folds()`; below this, `DataError` is raised and the env aborts.
- `--force` controls per-env directory cleanup, not iteration-level state. Per-iteration directories under `models/iterations/iter_{N}/` are isolated by construction (force-cleared per env, see `:455`'s `force=True`).
- Production container CMD never invokes `train_pipeline.py` — training is always manual via `docker exec`. The APScheduler in `scripts/main.py` runs trading + monitoring only.
- The trainer client is fail-open across all memory operations (see [`memory-system.md`](memory-system.md) "Fail-open client"). Memory-service downtime never blocks training.

## Known issues / open questions

- **No per-fold checkpointing.** Mid-fold crash loses fold 0..k-1 results from this env's run; the env restarts from fold 0 on retry. Per-iteration and per-env checkpoints exist; per-fold does not.
- **`pattern_outcomes` has no UNIQUE constraint.** The DDL at `src/swingrl/data/postgres_schema.py:615` defines only `id` as PK. Sibling table `iteration_results` has `UNIQUE (iteration_number, environment, run_type)` at `:216`; `pattern_outcomes` does not. Already noted in [`training-data-capture.md`](training-data-capture.md). On `record_outcome` retry the duplicate INSERT silently lands; downstream `/training/pattern_effectiveness` LEFT JOINs on `(iteration, env_name)` and would double-count.
- **`/consolidate` Phase B duplicate-pattern risk on retry.** Phase A is effectively idempotent across retries (archive flips on success → re-run finds zero sources). Phase B archives independently; if Phase B failed mid-run, retry re-ingests the same `training_epoch:` rows into a new LLM call. Dedup at `consolidate.py:2090-2108` is exact match on `(category, affected_algos)`; near-duplicate phrasings can land as fresh active patterns. Cross-link [`memory-system.md`](memory-system.md) Known issues.
- **Empirical: memory hurts training (iter 0-4).** Per `MEMORY.md` Active Session State, control folds outperformed treatment folds by 2.7-5.1× across iter 3-4 (treatment Sharpe 0.012-0.083 vs. control 0.034-0.42). Cross-link [`reward-shaping.md`](reward-shaping.md) and [`memory-system.md`](memory-system.md) Known issues.
- **2026-04-07 production-data wipe (resolved structurally).** Tests against pg16 wiped iter 0-5 production data. Plan A recovery restored iter 0-4 from duckdb backup; iter 5 was declared lost. Plan B's structural fix — `tests/conftest.py:36-60` `pytest_configure` guard — has landed. CI script `scripts/ci-homelab.sh:48-54` overrides `DATABASE_URL` to `swingrl_test` before invoking pytest.
- **Memory-service health gate falls back silently.** When memory service is unreachable at iteration start, the iteration silently flips back to baseline mode (`scripts/train_pipeline.py:404-410` log line `memory_service_unavailable_falling_back_to_baseline`). The training_state.json does not record that the iteration "intended to be" memory-enhanced — the iteration's results look indistinguishable from a baseline iteration.
- **Per-env results may be partially persisted on retry.** If env A succeeds and env B fails after retry-once, the iteration's full result includes A's data and a `{"error": ...}` slot for B. The iteration is marked "complete" in `completed_iterations` regardless (`:511`). On the next run, B's slot stays as the error dict — there's no automatic re-attempt of the failed env on a subsequent invocation.
- **`scripts/train.py` is dormant but undeleted.** Not imported anywhere in `src/`, `scripts/`, or `tests/`. Recommend either removing or marking as deprecated in a future cleanup pass.
- **`scripts/migrations/` lineage.** `scripts/migrations/recover_iteration_results.py` is the one-off migration referenced in the autocommit-bug comment at `train_pipeline.py:2295-2303`. `scripts/migrations/add_cps_columns.py` and `scripts/backfill_cps_history.py` exist in the repo per `ls scripts/`; the `MEMORY.md` 2026-04-07 entry notes both were also `docker cp`-injected into the running swingrl container at `/app/scripts/migrations/` and `/app/scripts/` — those container-side copies survive container restart but not recreation.
- **Pre-commit hook gotcha (per handoff).** The repo's security pre-commit hook flags certain Python serialization library names. This doc avoids naming the library directly when discussing the env-stats file produced by `VecNormalize.save()`. If anything trips the hook unexpectedly during commit, fix the phrasing rather than skip the hook.

## Source of truth

| Concern | File |
|---------|------|
| Iteration loop + per-env dispatch | `scripts/train_pipeline.py` (`run_all_iterations` at `:348`, `run_environment` at `:2057`) |
| State file (atomic write/load) | `scripts/train_pipeline.py:85-118` |
| Memory-service health gate | `scripts/train_pipeline.py:285-346` |
| CLI parser | `scripts/train_pipeline.py:2775-2848` |
| Walk-forward backtester | `src/swingrl/agents/backtest.py` |
| Fold generator | `src/swingrl/agents/backtest.py:181-237` |
| Per-env constants | `src/swingrl/agents/backtest.py:44-59` |
| Ensemble gate + recent-window slicing | `src/swingrl/training/pipeline_helpers.py` |
| Wrapper script | `scripts/run_iterations.sh` |
| Production CMD + healthcheck | `Dockerfile:59-97` |
| Compose service | `docker-compose.yml:63-96` |
| CI test-DB override | `scripts/ci-homelab.sh:48-54` |
| Test-DB guard | `tests/conftest.py:36-60` |
| Operational runbook | `docs/TRAINING_RUNBOOK.md` |

## Changelog

- **2026-05-07** — Initial version.
