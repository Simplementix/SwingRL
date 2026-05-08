# Validation & Promotion Reference

Living reference for SwingRL's three-layer gate stack and two model-promotion paths. Covers the per-fold informational gate, the ensemble training-time gate (with its soft tuning trigger and hard deployment block), the shadow→active promotion gate, and the on-disk model lifecycle. Includes a verified architectural disconnect between the shadow lifecycle and the live trader's read path.

**Last verified against code:** 2026-05-07

**Honest-gap policy:** every concrete claim is `file:line`-cited. Cross-table writers, ensemble configs, and per-algo HP details are not restated — see [`training-data-capture.md`](training-data-capture.md), [`agent-architecture.md`](agent-architecture.md), and [`training-pipeline.md`](training-pipeline.md).

## Architecture at a glance

| Layer | Type | Location | Threshold | Effect |
|-------|------|----------|-----------|--------|
| Per-fold validation gate | Informational | `src/swingrl/agents/validation.py:79–149` (`check_validation_gates`) | Sharpe>0.7, MDD<0.15, PF>1.5, OvfGap<0.20 | Logged + stamped onto `backtest_results.overfitting_class` (via `diagnose_overfitting` at `:33–76`); never blocks |
| Ensemble training-time gate (soft) | Tuning trigger | `scripts/train_pipeline.py:67` (`_TUNING_SHARPE_THRESHOLD = 0.5`) | baseline Sharpe < 0.5 | Triggers tuning rounds 1 & 2 |
| Ensemble training-time gate (hard) | Deployment block | `src/swingrl/training/pipeline_helpers.py:57–58, 205` (`_GATE_MIN_SHARPE=1.0`, `_GATE_MAX_MDD=0.15`) | Sharpe>1.0 AND \|MDD\|<0.15 | Pass → final training + active deploy. Fail AND Sharpe<0.5 → early return at `train_pipeline.py:2611–2631` |
| Shadow → active promotion gate | Replacement block | `src/swingrl/shadow/promoter.py:62–162` (`evaluate_shadow_promotion`) | shadow Sharpe>active Sharpe, shadow MDD≤1.2×active MDD, shadow PF>1.5, no CB triggers | Pass → `lifecycle.promote()`; fail → `lifecycle.archive_shadow()` |

**CPS is not a gate.** Computed post-gate at `train_pipeline.py:2332` (via `compute_and_persist_iteration_cps`), persisted to `iteration_results`, surfaced in `iteration_report.py` regression detection — never blocks anything. See [CPS — informational, not a gate](#cps--informational-not-a-gate).

**Two promotion paths exist** and they do not compose. See [Two promotion paths](#two-promotion-paths) and [Known issues](#known-issues--open-questions).

## Per-fold validation gate (informational)

`src/swingrl/agents/validation.py:79–149` — `check_validation_gates(sharpe, mdd, profit_factor, overfit_gap)` returns a `GateResult` with `.passed: bool` and `.failures: list[str]`.

Thresholds (hardcoded — not yaml-tunable):

| # | Metric | Threshold | Line |
|---|--------|-----------|------|
| 1 | Sharpe | `> 0.7` | `validation.py:106` |
| 2 | MDD | `< 0.15` | `validation.py:112` |
| 3 | Profit factor | `> 1.5` | `validation.py:118` |
| 4 | Overfit gap | `< 0.20` | `validation.py:128` |

Log emit: `log.info("validation_gates_checked", passed=..., failures=...)` at `validation.py:139–148`.

**Companion classifier** — `diagnose_overfitting(in_sample_sharpe, out_of_sample_sharpe)` at `validation.py:33–76` computes `gap = 1 - (OOS / IS)` and classifies:

- `< 0.20` → `healthy`
- `0.20 – 0.50` → `marginal`
- `> 0.50` → `reject`
- `IS ≤ 0` → `reject` (early-out at `:55–60`)

The `< 0.20` boundary coincides with the gate-4 threshold; the same number does double duty as both a soft classification and the hard-fail cutoff.

**Per-fold writes:**
- `agents/backtest.py:451` — `log.info("fold_complete", ..., gate_passed=gate.passed, failures=gate.failures)`
- `agents/backtest.py:853` — sets `fold.overfitting.get("classification")` onto the `backtest_results.overfitting_class` column. Writer details: see [`training-data-capture.md` `backtest_results`](training-data-capture.md#backtest_results).

This gate runs once per (env, algo, fold). It never blocks training and never gates promotion — it only annotates the per-fold record.

## Ensemble training-time gate

The gate that actually decides whether a freshly-trained model gets deployed to `models/active/{env}/{algo}/`. Has a **soft tier** (triggers tuning) and a **hard tier** (blocks deployment).

### Hard-gate constants

`src/swingrl/training/pipeline_helpers.py:57–58`:

```python
_GATE_MIN_SHARPE: float = 1.0
_GATE_MAX_MDD: float = 0.15
```

`src/swingrl/training/pipeline_helpers.py:205`:

```python
passed = ensemble_sharpe > _GATE_MIN_SHARPE and abs(ensemble_mdd) < _GATE_MAX_MDD
```

Returns `(passed, ensemble_sharpe, ensemble_mdd)`. Note: `ensemble_mdd` is stored as a negative float (e.g. -0.10 = 10% drawdown), so the gate uses `abs()`.

Log emit: `log.info("ensemble_gate_checked", passed=..., ensemble_sharpe=..., ensemble_mdd=..., fold_count=..., weighted=...)` at `pipeline_helpers.py:207–214`.

### Soft-gate constant

`scripts/train_pipeline.py:67`:

```python
_TUNING_SHARPE_THRESHOLD: float = 0.5
```

Used at two sites:

1. `train_pipeline.py:2437` — `if not passed and ensemble_sharpe < _TUNING_SHARPE_THRESHOLD:` triggers tuning rounds 1 & 2.
2. `train_pipeline.py:2611–2631` — same condition gates the *post-tuning* deployment block: if both rounds exhausted and still `not passed AND sharpe < 0.5`, the function writes a diagnostic JSON report and returns early — no final training, no active deploy.

**Behavioral nuance:** if `passed=False` BUT `ensemble_sharpe >= 0.5`, neither the tuning trigger nor the hard block fires. Final training proceeds and the model is deployed. The gate is a *soft floor at Sharpe 0.5*, not a hard cutoff at Sharpe 1.0.

### Call sites

`check_ensemble_gate(...)` is called at five sites in `scripts/train_pipeline.py`:

| Line | Call | Purpose |
|------|------|---------|
| `:2283` | `check_ensemble_gate(all_wf_results, ensemble_weights=...)` | Baseline ensemble after walk-forward |
| `:2473` | `check_ensemble_gate({"ppo": folds_v})` | Per-variant PPO check during tuning round 1 |
| `:2516` | `check_ensemble_gate(...)` | Re-check after tuning round 1 |
| `:2557` | `check_ensemble_gate({algo_r2: folds_r2})` | Per-algo check during tuning round 2 |
| `:2598` | `check_ensemble_gate(...)` | Final re-check after tuning round 2 |

The two per-algo helper calls (`:2473`, `:2557`) feed local Sharpe comparisons during tuning; the three full-ensemble calls (`:2283`, `:2516`, `:2598`) update the `gate_result` dict that drives `_evaluate_gate_and_decide` and the deployment block.

### Decision helper

`scripts/train_pipeline.py:718–757` — `_evaluate_gate_and_decide(gate_result, baseline_sharpe, env_name)` returns:

```python
{"deploy": bool, "sharpe": ..., "mdd": ..., "tuning_triggered": bool}
```

with `tuning_triggered = baseline_sharpe < _TUNING_SHARPE_THRESHOLD`. Log: `ensemble_gate_passed` (line 738) or `ensemble_gate_failed` (line 745).

### `iteration_results.gate_passed` writer

After the gate runs, the result is persisted to pg16 via `agents/backtest.py:888 store_iteration_results_to_duckdb(..., gate_passed: bool, ...)` (function name preserved from the duckdb era; writes pg16). The INSERT/ON-CONFLICT block is at `backtest.py:947–986`. The call site is `train_pipeline.py:2305–2317` inside the per-env driver.

**Autocommit caveat:** that connection is opened with `psycopg.connect(database_url, ..., autocommit=True)` at `train_pipeline.py:2304`. The `store_iteration_results_to_duckdb` function does not call `conn.commit()`, so without autocommit the `finally: conn_ens.close()` rolls back the INSERT silently. Bug existed iter 0–4; fixed iter 5+. See `train_pipeline.py:2295–2304` block-comment and `scripts/migrations/recover_iteration_results.py`. Cross-link: [`training-pipeline.md` "iteration_results autocommit caveat"](training-pipeline.md#iteration_results-autocommit-caveat).

### Post-deployment verification

`scripts/train_pipeline.py:657–680` — `_verify_deployment(env_name, models_dir)` checks that **both** `model.zip` AND `vec_normalize.pkl` exist for all 3 algos under `models/active/{env}/{algo}/`. Raises `ModelError` if any file is missing. Called at `train_pipeline.py:2737`. This is the canonical assertion that the trainer's per-algo layout is the production layout.

## CPS — informational, not a gate

`src/swingrl/metrics/cps.py` defines three Capital Preservation Score formulas — none are gates:

| Function | Line | Formula |
|----------|------|---------|
| `compute_cps_v1_multiplicative(per_fold)` | `cps.py:152` | Multiplicative across fold CPS |
| `compute_cps_v2_additive(per_fold, ...)` | `cps.py:185` | Additive |
| `compute_cps_v3_sortino(per_fold, ...)` | `cps.py:229` | Sortino-based |

Persistence: `src/swingrl/reporting/iteration_report.py:612 persist_iteration_cps(...)` — UPSERT to `iteration_results` columns `cps_v1_multiplicative`, `cps_v2_additive`, `cps_v3_sortino`, `cps_v1_treatment_only`, `cps_v1_control_only`, `cps_components` (JSON), `cps_formula_version`. DDL: `src/swingrl/data/postgres_schema.py:198–204`.

Trigger: `train_pipeline.py:2332` — `compute_and_persist_iteration_cps(cps_conn, env_name, iteration_number)`. Called *after* the ensemble gate has been written, in a separate connection so a CPS write failure cannot mask the (more critical) ensemble-results write.

**Regression detection** lives at read-time in `iteration_report.py:152 compute_iter_deltas(history)` — flags `regression_flag=True` when `return_delta < -REGRESSION_RETURN_THRESHOLD` (constant `0.02` at `iteration_report.py:52`) or worst-MDD widens. The flag is computed when the report is built; it is **not stored** in pg16. The markdown summary table at `iteration_report.py:312 format_iteration_summary(history)` prefixes regression rows with `⚠`.

CPS does not fire Discord alerts — promotion alerts come exclusively from `promoter.py`. See [Discord alerts](#discord-alerts).

## Two promotion paths

### Path A: Trainer-internal "best-of-iterations" deploy (production)

This is the path that actually feeds the live trader.

**Per-iteration WF training** writes to `models/iterations/iter_{N}/active/{env}/{algo}/` because `train_pipeline.py:434` constructs `iter_models_dir = models_dir / "iterations" / f"iter_{i}"` and passes it to `TrainingOrchestrator(models_dir=iter_models_dir, ...)` at `:453, :474`.

**Final post-tuning training** writes directly to `models/active/{env}/{algo}/` because `train_pipeline.py:1907–1909` constructs `TrainingOrchestrator(... models_dir=models_dir, ...)` with the *base* `models_dir`. The trainer itself uses `self._models_dir / "active" / env_name / algo_name` (`src/swingrl/training/trainer.py:502, 568`).

**Best-of-iterations selection** — `scripts/train_pipeline.py:193 deploy_best_models(winners, models_dir)` copies winners from per-iteration `iterations/iter_{idx}/active/{env}/{algo}/` to `models/active/{env}/{algo}/`. Crucially, it copies **both** files together at `:219–230`:

```python
for filename in ["model.zip", "vec_normalize.pkl"]:
    src = src_dir / filename
    dst = dst_dir / filename
    if src.exists():
        shutil.copy2(str(src), str(dst))
```

Triggered at `train_pipeline.py:592` after multi-iteration completion.

**Live trader read** — `src/swingrl/execution/pipeline.py:362–366`:

```python
model_path = self._models_dir / "active" / env_name / algo_name / "model.zip"
vec_path = self._models_dir / "active" / env_name / algo_name / "vec_normalize.pkl"
```

Confirms the canonical production layout is **per-algo**.

### Path B: Shadow → Active lifecycle (wired but disconnected — see Known Issues)

Standalone subsystem at `src/swingrl/shadow/` that runs daily but does not match Path A's directory layout.

**Daily promotion check** — `src/swingrl/scheduler/jobs.py:391–428 shadow_promotion_check_job()`:

1. `:399–401` — short-circuit if `is_halted(ctx.db)`. Halt = no promotion.
2. `:405–409` — `ModelLifecycle(Path(ctx.config.paths.models_dir))` — instantiated with the base models root.
3. `:411–419` — for each env in `("equity", "crypto")`, calls `evaluate_shadow_promotion(...)`.
4. `:420–423` — logs `shadow_promotion_check_complete` per env.

Registration: `scripts/main.py:170–177` schedules this job. Per the docstring at `jobs.py:391–392`, it runs "daily at 7 PM ET."

**Shadow inference is continuous, not standalone.** `src/swingrl/scheduler/jobs.py:127, ~178` — `run_shadow_inference(ctx, env)` is called after every `equity_cycle()` and `crypto_cycle()`. Shadow trades accumulate continuously into `shadow_trades`. Source: `src/swingrl/shadow/shadow_runner.py:53 run_shadow_inference`; reads single `models/shadow/{env}/*.zip` (no algo subdir) at `:65–73`; writes to `shadow_trades` table at `:282 _record_shadow_trades`.

## Shadow promotion criteria

`src/swingrl/shadow/promoter.py:39–162 evaluate_shadow_promotion(config, db, env_name, lifecycle, alerter)`.

### Min eval threshold (gates the comparison itself)

`promoter.py:62–85`:

- **Equity:** `config.shadow.equity_eval_days` shadow trades required (default 10).
- **Crypto:** `config.shadow.crypto_eval_cycles` shadow trades required (default 30).

Determined by counting rows in `shadow_trades WHERE environment = %s` (`promoter.py:71–77`). If insufficient, log `shadow_eval_insufficient_data` and return `False`. No promotion attempted.

### Returns sources

- **Shadow returns:** `_get_portfolio_returns(db, env_name, source="shadow")` — reads from `shadow_trades` (synthetic returns from paired buy/sell trades, `promoter.py:215–270`).
- **Active returns:** `_get_portfolio_returns(db, env_name, source="active")` — reads from `portfolio_snapshots` (`promoter.py:185–212`).

Annualization periods: `promoter.py:30–34` — `equity: 252.0`, `crypto: 6 * 365 = 2190.0` (4H bars/year).

### Four criteria

`promoter.py:116–122`:

| # | Criterion | Source |
|---|-----------|--------|
| 1 | `shadow_sharpe > active_sharpe` | annualized OOS Sharpe |
| 2 | `shadow_mdd <= mdd_tolerance_ratio * active_mdd` (yaml; default 1.2) | `_safe_max_drawdown` |
| 3 | `shadow_pf > _MIN_PROFIT_FACTOR` (= 1.5, hardcoded at `promoter.py:36`) | `_compute_profit_factor` over `shadow_trades` |
| 4 | `not cb_triggered` — no circuit-breaker events during shadow window | `_check_cb_during_shadow` |

If `active_mdd <= 0`, criterion 2 degenerates to `shadow_mdd <= 0` (`promoter.py:118`). Edge case worth noting.

### Decision branches

`promoter.py:122–162`:

- **All 4 pass AND `config.shadow.auto_promote` is True** → `lifecycle.promote(env_name)`, send Discord `info` alert, log `shadow_promoted`. Default `auto_promote=True` per yaml.
- **All 4 pass AND `auto_promote` is False** → fall through. Shadow stays in place. No automatic archival.
- **Any criterion fails** → `lifecycle.archive_shadow(env_name)`, send Discord `warning` alert with reason list, log `shadow_failed`.

## Lifecycle on disk

`src/swingrl/shadow/lifecycle.py:38–54 ModelLifecycle.__init__(models_dir)`:

```
models_dir/
  shadow/{env_name}/    # candidate models
  active/{env_name}/    # currently serving (FLAT — see Known Issues)
  archive/{env_name}/   # previously active, timestamp-suffixed
```

Subdirs created on init (`:53–54`). State enum at `:27–34` — `TRAINING`, `SHADOW`, `ACTIVE`, `ARCHIVE`, `DELETED`.

### Promote sequence

`lifecycle.py:80–113 promote(env_name)`:

1. Glob `shadow/{env}/*.zip`; raise `ModelError` if empty (`:91–96`).
2. If `active/{env}/*.zip` exists, archive the existing one via `_archive_model` (`:103–106`).
3. Move shadow `.zip` to `active/{env}/` via `shutil.move` (`:108–111`).
4. Log `model_promoted`.

**No transactional guard.** If the process is killed between step 2 and step 3, `active/{env}/` ends up empty while the shadow `.zip` is gone. No `.lock` file, no two-phase commit, no rollback hook.

### Archive layout

`lifecycle.py:225–243 _archive_model(model_path, env_name)`:

```python
timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
stem = model_path.stem
dest = archive_dir / f"{stem}_{timestamp}.zip"
shutil.move(str(model_path), str(dest))
```

UTC-naive name; sortable lexicographically. Used by both `promote()` (archive existing active) and `archive_shadow()` (`lifecycle.py:136`).

### Other operations

| Method | Line | Behavior |
|--------|------|----------|
| `deploy_to_shadow(model_path, env_name)` | `:56–78` | `shutil.copy2` (not move) — source remains. |
| `archive(env_name)` | `:115–134` | Move active to archive with timestamp suffix. |
| `archive_shadow(env_name)` | `:136–155` | Move shadow to archive (used on promotion failure). |
| `rollback(env_name)` | `:157–184` | Sort `archive/{env}/*.zip` lexicographically, move `[-1]` (most recent) to active. |
| `delete_archived(model_path)` | `:186–200` | `model_path.unlink()` — caller selects which file. No retention/TTL logic. |
| `get_state(env_name)` | `:202–223` | Returns `{active_model, shadow_model, archive_count}` — read-only state for monitoring. |

### Companion `vec_normalize.pkl` is NOT moved

`lifecycle.promote()` only globs `*.zip` (`:96`). The companion stats file (saved alongside by SB3 — see [`agent-architecture.md` "TrainingOrchestrator"](agent-architecture.md#trainingorchestrator)) is left behind. By contrast, `deploy_best_models` (Path A) copies both files together (`train_pipeline.py:219–230`).

This compounds the directory-layout problem documented in [Known issues](#known-issues--open-questions).

## Live-trader cache & restart requirement

`src/swingrl/execution/pipeline.py:354–357`:

```python
def _load_models(self, env_name: str) -> dict[str, tuple[Any, Any]]:
    if env_name in self._models:
        return self._models[env_name]
    ...
```

Models are lazy-loaded on first call to `execute_cycle(env)` and cached in `self._models[env_name]` indefinitely. There is **no file-watch, no signal handler, no cache invalidation**.

**Operational consequence:** any model file change — Path A `deploy_best_models`, Path B `lifecycle.promote()`, manual `scp` — is invisible to the running live trader until the container is restarted (`docker compose restart swingrl`). Not flagged as TODO in code; documented here.

## Discord alerts

Routing — `src/swingrl/monitoring/alerter.py:100–112`:

- `critical` / `warning` → `alerts_webhook_url` (fallback `webhook_url`)
- `info` → `daily_webhook_url` (fallback `webhook_url`)

Cooldown / dedup — `alerter.py:91–92`:

- `cooldown_minutes`: default 30 — minimum minutes between identical critical/warning alerts.
- `consecutive_failures_before_alert`: default 3 — number of consecutive identical warnings before sending.

Promotion alert sites:

- `promoter.py:126–134` — `info` on success: `"Shadow Model Promoted: {env_name.title()}"`. Goes to daily webhook.
- `promoter.py:152–160` — `warning` on failure: `"Shadow Model Failed: {env_name.title()}"` with reason list. Goes to alerts webhook.

Promotion is the *only* path that fires Discord alerts in the gate/promotion stack — the per-fold gate, ensemble gate, and CPS regression do not alert.

## pg16 audit surface

What's queryable when an operator asks "why was iteration N not promoted?":

| Table | Relevant columns | Schema |
|-------|------------------|--------|
| `iteration_results` | `gate_passed` (BOOLEAN), `ensemble_sharpe`, `ensemble_mdd`, `cps_v1_multiplicative`, `cps_components` (JSON), `worst_fold_mdd`, `chronic_failure_count`, `return_regression_delta`, `created_at` | `postgres_schema.py:171–217` |
| `backtest_results` | `overfitting_class` ("healthy"/"marginal"/"reject"), `sharpe`, `mdd`, `profit_factor`, `is_control_fold`, `iteration_number` | `postgres_schema.py:123–168` |
| `model_metadata` | `model_path`, `vec_normalize_path`, `ensemble_weight` | `postgres_schema.py:106` (DDL); writer `train_pipeline.py:1612 _write_model_metadata` (idempotent INSERT…ON CONFLICT, `:1640`) |
| `shadow_trades` | hypothetical inferences during shadow window | `postgres_schema.py:489`; writer `shadow_runner.py:282` |
| `alert_log` | sent Discord alerts (`level`, `title`, `message_hash`, `sent`) | `postgres_schema.py:507–514` |

Full per-table writers and readers: [`training-data-capture.md`](training-data-capture.md) — `model_metadata` (`:88`), `backtest_results` (`:95`), `iteration_results` (`:104`), `shadow_trades` (`:252`), `alert_log` (`:232`).

**Not present in pg16:**

- No `promotion_log` / `promotions` table.
- No `promoted_at` / `archived_at` / `gate_reason` columns on `iteration_results`.
- No FK from `backtest_results` to `iteration_results` (relationship is implicit via `(iteration_number, environment)`).

To answer "what was active on date X," an operator must parse archive filenames (`models/archive/{env}/{stem}_{YYYYMMDD_HHMMSS}.zip`) — there is no SQL audit trail for promotion events.

## Configurable values (yaml)

All under `config.shadow.*` — schema bounds at `src/swingrl/config/schema.py:233–236`; yaml defaults at `config/swingrl.yaml:73–76`:

| Field | Default | Bound | Purpose |
|-------|---------|-------|---------|
| `equity_eval_days` | 10 | `ge=5` | Minimum shadow trades before equity evaluation runs |
| `crypto_eval_cycles` | 30 | `ge=10` | Minimum shadow trades before crypto evaluation runs |
| `auto_promote` | true | bool | If false, criteria-pass leaves shadow in place — no auto-archive, no auto-promote |
| `mdd_tolerance_ratio` | 1.2 | `gt=1.0` | Multiplier on `active_mdd` for criterion 2 |

Discord webhooks (under `config.alerting.*`) — `alerts_webhook_url` and `daily_webhook_url` — are loaded by `Alerter`. Cooldown is constructor-configurable (default 30 min) but not exposed in the default yaml.

No yaml field controls the per-fold gate thresholds, ensemble gate thresholds, or the soft tuning threshold.

## Hardcoded values (code edit required)

| Value | Constant | Location |
|-------|----------|----------|
| Per-fold Sharpe gate | `0.7` | `validation.py:106` |
| Per-fold MDD gate | `0.15` | `validation.py:112` |
| Per-fold PF gate | `1.5` | `validation.py:118` |
| Per-fold OvfGap gate | `0.20` | `validation.py:128` |
| Overfitting healthy/marginal boundary | `0.20` | `validation.py:60` |
| Overfitting marginal/reject boundary | `0.50` | `validation.py:62` |
| Ensemble gate Sharpe min | `_GATE_MIN_SHARPE = 1.0` | `pipeline_helpers.py:57` |
| Ensemble gate MDD max | `_GATE_MAX_MDD = 0.15` | `pipeline_helpers.py:58` |
| Tuning trigger / hard-block floor | `_TUNING_SHARPE_THRESHOLD = 0.5` | `train_pipeline.py:67` |
| CPS regression delta | `REGRESSION_RETURN_THRESHOLD = 0.02` | `iteration_report.py:52` |
| Shadow PF floor | `_MIN_PROFIT_FACTOR = 1.5` | `promoter.py:36` |
| Equity periods/year | `252.0` | `promoter.py:30–34` |
| Crypto periods/year | `2190.0` (= 6 × 365) | `promoter.py:30–34` |

## Invariants

1. **Production model layout is per-algo.** Live trader (`execution/pipeline.py:362`) and post-deployment verifier (`train_pipeline.py:657`) both assume `models/active/{env}/{algo}/{model.zip,vec_normalize.pkl}`. Trainer writes to this layout (`trainer.py:502, 568`).
2. **`gate_passed=False` AND `ensemble_sharpe < 0.5` blocks final training and active deploy.** See `train_pipeline.py:2611–2631`. Failing the gate alone is *not* sufficient — Sharpe must also be below the soft floor.
3. **Shadow inference accumulates continuously.** Every cycle calls `run_shadow_inference` (`jobs.py:127, ~178`) — `shadow_trades` is append-only across the entire trading lifetime, not just during a "shadow run."
4. **Halt blocks promotion check, not promotion mechanics.** `is_halted` short-circuits the *scheduled job* (`jobs.py:399–401`). Manually invoking `lifecycle.promote()` does not check halt.
5. **Promotion archives the *outgoing* active before moving in the new one.** `lifecycle.py:103–106` archives existing active first, then moves shadow on top. Order matters: rollback uses the archive.
6. **The ensemble gate never re-runs a holdout.** It aggregates per-fold OOS metrics from the walk-forward result already in memory. There is no separate validation set.
7. **CPS never blocks anything.** It is a metric, not a gate. Computed post-deploy; surfaced in reports.

## Known issues / open questions

### 🚨 Shadow lifecycle directory layout does not match production

**Verified bug.** Shadow promotion and the live trader operate at different directory granularities:

| Reader/writer | Path | Granularity | File:line |
|---------------|------|-------------|-----------|
| Live trader (`execution.pipeline._load_models`) | `models/active/{env}/{algo}/model.zip` | per-algo (3 files per env) | `execution/pipeline.py:362` |
| Post-deploy verify (`_verify_deployment`) | `models/active/{env}/{algo}/{model.zip,vec_normalize.pkl}` | per-algo | `train_pipeline.py:670–675` |
| Trainer final write | `models/active/{env}/{algo}/{model.zip,vec_normalize.pkl}` | per-algo | `trainer.py:502, 568` |
| Path A `deploy_best_models` | `models/active/{env}/{algo}/{model.zip,vec_normalize.pkl}` | per-algo | `train_pipeline.py:214–230` |
| Shadow runner read | `models/shadow/{env}/*.zip` | flat (single file) | `shadow_runner.py:65–73` |
| Shadow lifecycle promote | moves shadow `*.zip` → `models/active/{env}/{shadow_filename}.zip` | flat (single file) | `lifecycle.py:91–111` |

Calling `ModelLifecycle.promote("equity")` deposits a single `.zip` file at `models/active/equity/` — but `execution/pipeline.py:362` reads `models/active/equity/ppo/model.zip`, `models/active/equity/a2c/model.zip`, `models/active/equity/sac/model.zip`. The shadow promotion does not replace any of those files. **The cron job runs daily, may write Discord alerts claiming "Shadow Model Promoted," and the live trader continues using the previously-deployed Path A models, unaware.**

Compounding: `lifecycle.promote()` only moves the `.zip`, leaving any companion `vec_normalize.pkl` orphaned (`lifecycle.py:96` — only `*.zip` glob, vs. `deploy_best_models` at `train_pipeline.py:219` which copies both files).

**Action required:** decide whether shadow should be retired, refactored to a per-algo bundle layout, or replaced with a different validation strategy. Until then, Path B should be considered inert with respect to the live trader. Discord promotion alerts from `promoter.py:126–134` are misleading — they announce a swap that does not reach production.

### No transactional guard during `promote()`

`lifecycle.py:103–111` performs two `shutil.move` calls back-to-back: archive existing active, then move shadow into active. If the process is killed between them, `active/{env}/` ends up empty. No `.lock` file, no two-phase commit. No automated recovery. Manual remediation: `lifecycle.rollback(env)` or operator copies from `archive/`.

### Live-trader model cache is never invalidated

`execution/pipeline.py:354–357` caches loaded models per env on first use; never re-checks the file. Container restart is required for any model change to take effect. There is no file-watch, no signal handler, no API endpoint to trigger a reload. Restart procedure: `docker compose restart swingrl`.

### Archive retention is unbounded

No auto-pruning, no TTL, no cron job. `lifecycle.delete_archived(model_path)` exists (`:186–200`) but takes a specific path — caller selects which to remove. Disk grows monotonically until manual cleanup.

### No `promotion_log` / `promoted_at` columns in pg16

Promotion events are preserved only in: structlog (`shadow_promoted` / `shadow_failed`), filesystem (timestamped archive filename), Discord (alert message). No SQL audit trail. Cannot answer "what was active on date X" via a query — must parse `archive/{env}/{stem}_{YYYYMMDD_HHMMSS}.zip` filenames.

### `iteration_results.gate_passed` carries no rationale

The column is BOOLEAN. There is no `gate_reason` / `gate_failures_json` column to record *why* a gate failed. Reconstructing the rationale requires reading per-fold `backtest_results.overfitting_class` plus comparing `ensemble_sharpe`/`ensemble_mdd` against hardcoded thresholds in `pipeline_helpers.py:57–58`.

### CPS `regression_flag` is computed at read-time, not stored

`iteration_report.py:152 compute_iter_deltas` derives the flag from `cps_v1_multiplicative` deltas at report build time. There is no `regression_flag` column in `iteration_results`. If thresholds change (`REGRESSION_RETURN_THRESHOLD = 0.02` at `:52`), historical reports change retroactively. Persisting the flag would prevent retroactive shifts.

### Shadow-vs-active comparison windows are not aligned

Shadow returns are built from `shadow_trades` over the shadow run window (since first shadow trade); active returns are built from `portfolio_snapshots` over all-time history. No date-window normalization (`promoter.py:185–212` for active, `:215–270` for shadow). If active has a long history with one bad week years ago, the comparison may favor a fresh shadow whose run doesn't overlap that period.

### `auto_promote: true` defaulted on, no human approval step

`promoter.py:122` calls `lifecycle.promote()` directly when criteria pass and `auto_promote=True` (the default). No intermediate hold/approval. Setting `auto_promote: false` keeps shadow alive without auto-archiving — operator must manually promote via Python REPL inside the container.

### No CLI subcommand for manual `promote` / `rollback`

`lifecycle.promote()` and `lifecycle.rollback()` are Python-only. `scripts/main.py:170–177` registers the cron job but exposes no CLI flag for one-shot invocation. `scripts/deploy_model.sh` ships a model into `shadow/` and prints a manual command (`docker exec ... ModelLifecycle(...).promote(...)`) at the end of its smoke test, but does not run promotion itself. Manual promotion procedure:

```bash
docker exec swingrl python -c "
from pathlib import Path
from swingrl.shadow.lifecycle import ModelLifecycle
ModelLifecycle(Path('/app/models')).promote('equity')
"
```

### Cross-iteration `iter_models_dir` "active" subdirectory is confusingly named

`train_pipeline.py:434` creates `iter_models_dir = models_dir / "iterations" / f"iter_{i}"` and the trainer (`trainer.py:502`) appends `active/` to it — producing `models/iterations/iter_{N}/active/{env}/{algo}/`. The `active/` here means "trainer output," not "currently serving." Confusingly, `deploy_best_models` then copies from this `iterations/iter_N/active/...` to `models/active/...` (the actual active). Worth a rename, but the path is now baked into iter 0–5 history.

## Source of truth

**Validation gates:**
- `src/swingrl/agents/validation.py:33–149` — `diagnose_overfitting`, `check_validation_gates`, `GateResult`.
- `src/swingrl/training/pipeline_helpers.py:57–58, 133–214` — ensemble gate constants and `check_ensemble_gate`.

**Pipeline orchestration:**
- `scripts/train_pipeline.py:67` — `_TUNING_SHARPE_THRESHOLD`.
- `scripts/train_pipeline.py:193–240` — `deploy_best_models`.
- `scripts/train_pipeline.py:434, 453, 474` — per-iteration `iter_models_dir`.
- `scripts/train_pipeline.py:592` — `deploy_best_models(...)` invocation.
- `scripts/train_pipeline.py:657–680` — `_verify_deployment`.
- `scripts/train_pipeline.py:718–757` — `_evaluate_gate_and_decide`.
- `scripts/train_pipeline.py:1612–1677` — `_write_model_metadata` (model_metadata writer).
- `scripts/train_pipeline.py:1907–1909` — final-training orchestrator instantiation (base `models_dir`).
- `scripts/train_pipeline.py:2283, 2473, 2516, 2557, 2598` — `check_ensemble_gate` call sites.
- `scripts/train_pipeline.py:2295–2317` — `iteration_results` write block (autocommit).
- `scripts/train_pipeline.py:2332` — `compute_and_persist_iteration_cps`.
- `scripts/train_pipeline.py:2437, 2611` — `_TUNING_SHARPE_THRESHOLD` use sites.
- `scripts/train_pipeline.py:2611–2631` — hard deployment block early-return.
- `scripts/train_pipeline.py:2737` — `_verify_deployment` invocation.

**Trainer:**
- `src/swingrl/training/trainer.py:5` — module-docstring write-path claim (with caveat — see Path A).
- `src/swingrl/training/trainer.py:502, 568` — actual save paths.

**Per-fold writes:**
- `src/swingrl/agents/backtest.py:451` — `fold_complete` log.
- `src/swingrl/agents/backtest.py:853` — `overfitting_class` write.
- `src/swingrl/agents/backtest.py:888–1015` — `store_iteration_results_to_duckdb` (writes pg16 despite the legacy name).

**Live trader:**
- `src/swingrl/execution/pipeline.py:354–409` — `_load_models` cache + per-algo path.

**Shadow subsystem:**
- `src/swingrl/shadow/lifecycle.py:1–243` — `ModelLifecycle` state machine.
- `src/swingrl/shadow/promoter.py:1–162` — `evaluate_shadow_promotion`.
- `src/swingrl/shadow/shadow_runner.py:1–~310` — `run_shadow_inference` + `_record_shadow_trades`.

**Scheduler:**
- `src/swingrl/scheduler/jobs.py:127, ~178` — `run_shadow_inference` per-cycle calls.
- `src/swingrl/scheduler/jobs.py:391–428` — `shadow_promotion_check_job`.
- `scripts/main.py:170–177` — job registration.

**Reporting / CPS:**
- `src/swingrl/metrics/cps.py:152, 185, 229` — three CPS formulas.
- `src/swingrl/reporting/iteration_report.py:52` — `REGRESSION_RETURN_THRESHOLD`.
- `src/swingrl/reporting/iteration_report.py:152, 312, 612` — `compute_iter_deltas`, `format_iteration_summary`, `persist_iteration_cps`.

**Observability:**
- `src/swingrl/monitoring/alerter.py:67–112` — webhook routing + cooldown.

**Schema:**
- `src/swingrl/data/postgres_schema.py:106, 123–168, 171–217, 489, 507–514` — `model_metadata`, `backtest_results`, `iteration_results`, `shadow_trades`, `alert_log` DDL.

**Config:**
- `src/swingrl/config/schema.py:233–236` — shadow settings schema.
- `config/swingrl.yaml:73–76` — shadow settings defaults.

**Tests:**
- `tests/agents/test_validation.py` — per-fold gate + overfitting classifier.
- `tests/metrics/test_cps.py` — CPS formulas.
- `tests/shadow/test_lifecycle.py` — promote, archive, rollback.
- `tests/shadow/test_promoter.py` — `evaluate_shadow_promotion` (requires DATABASE_URL).
- `tests/test_deploy_smoke.py` — 6-point post-shadow-deploy smoke test.
- `tests/reporting/test_iteration_report.py` — regression flag + summary rendering.

## Changelog

- **2026-05-07** — Initial reference card. Three-tier gate model. Documented verified directory-layout disconnect between shadow lifecycle and live trader (per-algo vs. flat). Documented soft (Sharpe<0.5) vs. hard (Sharpe>1.0 AND |MDD|<0.15) gate behavior, including the gap zone where `passed=False AND sharpe>=0.5` lets deployment proceed. Surfaced 10 known issues; flagged the shadow disconnect as action-required.
