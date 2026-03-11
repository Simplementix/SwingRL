# Pitfalls Research

**Domain:** Operational deployment of RL trading system — automated retraining, Discord alerting, production Docker, operator runbooks, paper trading validation
**Researched:** 2026-03-10
**Confidence:** HIGH (codebase inspected directly; pitfalls derived from v1.0 code + established operational patterns)

---

## Critical Pitfalls

### Pitfall 1: MPS-Trained Models Produce Wrong Outputs When Loaded on CPU

**What goes wrong:**
The v1.0 training code was designed for M1 Mac (MPS device). When SB3 models saved on MPS are loaded on the homelab (CPU-only), PyTorch silently remaps tensors but the VecNormalize statistics (serialized via `vec_normalize.pkl`) may contain device-pinned tensor references from the training context. The model loads without error but produces subtly wrong outputs — actions that pass shape and NaN checks but encode incorrect policy estimates.

**Why it happens:**
`trainer.py:_save_model()` serializes the entire VecNormalize object to disk, which may embed device references from the MPS training context. SB3's `model.load()` accepts `device="cpu"` as a parameter but loading VecNormalize from disk does not automatically remap device references. The existing smoke tests (shape, non-degenerate, inference speed) do not verify that outputs are numerically consistent between the saved and a freshly-instantiated policy on a different device.

**How to avoid:**
1. Always run initial training on the homelab itself (CPU), not the M1 Mac. Then no cross-device transfer occurs and the file is clean from the start.
2. If M1-trained weights must be transferred, save model weights only via `model.get_parameters()`, then reconstruct the model on homelab and call `model.set_parameters()`. Do not transfer the VecNormalize file from M1.
3. Add a device assertion to the retraining job immediately after model load: `assert str(next(model.policy.parameters()).device) == "cpu"`.

**Warning signs:**
- All smoke tests pass but shadow model consistently scores worse Sharpe than active after promotion
- Inference latency is anomalously fast (< 1ms) suggesting no real computation
- Shadow Sharpe drifts toward 0.0 within the first week despite variety in market conditions

**Phase to address:**
Initial training phase (first phase of v1.1). Establish the policy: all models trained on homelab CPU from day 1. Document the no-MPS-transfer rule explicitly in the runbook.

---

### Pitfall 2: Automated Retraining Runs During Active Trading Window

**What goes wrong:**
The equity retraining job (monthly) and crypto retraining job (biweekly) are scheduled via APScheduler inside the same container as the live trading scheduler. If the retraining job starts at a time that overlaps with an active trading cycle (equity at 4:15 PM ET, crypto every 4H), both compete for CPU on the homelab (Intel i5, constrained single-thread throughput). CPU-bound SB3 training saturates all cores for 30-90 minutes, causing the trading cycle to miss its scheduled fire time. APScheduler marks it as a misfire and silently drops it — no trade, no alert.

**Why it happens:**
APScheduler does not preempt running jobs. A retraining job consuming 100% CPU during training will delay the ThreadPoolExecutor from scheduling the next trading cycle. With `misfire_grace_time` at its default (1 second), missed cycles are silently dropped.

**How to avoid:**
1. Schedule retraining jobs with explicit non-overlapping time windows:
   - Equity retraining: Saturday 2 AM ET (market closed, 6H gap before any Sunday crypto cycle)
   - Crypto retraining: Sunday 3 AM ET
2. Set `misfire_grace_time=600` (10 minutes) for trading cycle jobs so brief delays are tolerated.
3. Run retraining in a subprocess (`subprocess.Popen`) or separate systemd service, not inside APScheduler's thread pool. This isolates CPU pressure.
4. Optionally: add a `is_training_active` flag to the halt system; training job sets it, trading jobs check it and log a warning (not a silent drop).

**Warning signs:**
- APScheduler logs `equity_cycle missed by X seconds` around monthly retraining dates
- `logs/` shows a gap in healthcheck pings corresponding to a training window
- Healthchecks.io fires a "missed check" alert on the same calendar day retraining runs

**Phase to address:**
Automated retraining phase. The job scheduler wiring must explicitly define non-overlapping windows and document them in the runbook before enabling automated retraining.

---

### Pitfall 3: Shadow Model Auto-Promotes at Bootstrap When Active Baseline Is Zero

**What goes wrong:**
The shadow promotion logic in `promoter.py` computes Sharpe by querying the `trades` table (active trades) and the `shadow_trades` table. At initial deployment, if the system has been running for fewer days than `shadow.equity_eval_days`, the `trades` table has no rows, so `_compute_sharpe(db, "trades", env_name)` returns `0.0`. Shadow Sharpe (even a mediocre 0.1) passes `shadow_sharpe >= active_sharpe` (0.1 >= 0.0 is True), and the shadow model auto-promotes immediately regardless of quality.

**Why it happens:**
The evaluator was built for an ongoing system, not a cold-start scenario. The bootstrap period (first N eval cycles) has no meaningful active baseline to compare against, so any shadow model passes the promotion criteria.

**How to avoid:**
1. Add a `min_active_trades` bootstrap guard to `promoter.py`: if the `trades` table has fewer rows than `min_trades`, log `shadow_eval_bootstrap_mode` and return False.
2. Document the bootstrap period in the runbook: "For the first N days after fresh deployment, shadow promotion is disabled. Operator reviews and manually promotes the first model."
3. Optionally: seed the `trades` table with synthetic baseline rows representing a conservative do-nothing policy (Sharpe = 0.0, MDD = 0.0) so the comparison has a meaningful floor.

**Warning signs:**
- `shadow_promoted` log event fires within the first week of deployment
- Active Sharpe shows 0.0 in `shadow_eval_metrics` log line at time of promotion
- Shadow was running for fewer trading days than `equity_eval_days` config value when promotion fired

**Phase to address:**
Shadow mode / retraining phase. Add bootstrap guard before enabling auto-promotion. This must be in place before the 6-month paper trading validation period begins, as spurious early promotions would corrupt the baseline comparison.

---

### Pitfall 4: Docker Bind Mount File Permission Mismatch Breaks DB Writes Silently

**What goes wrong:**
`docker-compose.prod.yml` uses bind mounts (`./db:/app/db`, `./data:/app/data`, etc.). On the homelab Linux host, these directories may be owned by `root` or a different UID, but the container runs as `trader` (UID 1000). At startup, the container can read the mounted directories but cannot write to them, so `DatabaseManager` initialization fails or DuckDB cannot open the persistent file for writing. The container starts and its `HEALTHCHECK` passes (process liveness check), but no trades or logs are written.

**Why it happens:**
Docker bind mounts inherit host filesystem permissions. Running `docker compose up -d` on a fresh homelab likely creates directories as root. The HEALTHCHECK only verifies process liveness and DB connectivity at startup, not write access to mounted volumes.

**How to avoid:**
1. In the homelab setup runbook, include mandatory `chown` steps before first launch: `sudo chown -R 1000:1000 ./db ./data ./models ./logs ./status`.
2. Add a startup write-test to the entrypoint: attempt to write a sentinel file to each mounted volume, fail fast with a descriptive error if write fails.
3. Add a post-deploy verification step: `docker exec swingrl touch /app/db/.write_test && echo "Write OK"`.

**Warning signs:**
- Container shows as "healthy" in `docker ps` but `logs/` directory is empty after 1+ hours
- `swingrl.db` file never grows in size after deployment
- DuckDB error: `IO Error: Could not open file ... Permission denied` in `docker compose logs`

**Phase to address:**
Docker deployment phase. Include a pre-flight permission checklist in the runbook. This is the most common silent failure mode for first homelab deployments.

---

### Pitfall 5: Discord Rate Limit Silently Drops Critical Alerts During Event Storms

**What goes wrong:**
Discord webhooks are rate-limited at 30 requests per minute per webhook URL. The `Alerter` uses cooldown per `(level, title)` key, which prevents duplicate alerts but does not prevent bursts of distinct alerts. If multiple distinct critical events fire simultaneously (e.g., equity cycle fails + reconciliation fails + circuit breaker trips within seconds), three separate critical alerts are sent in rapid succession. The third hits the rate limit; Discord returns HTTP 429. The current `_post_webhook` catches the exception and logs `alert_send_failed` but does not retry or queue the message. The critical alert is silently lost.

**Why it happens:**
`httpx.post(..., timeout=10.0)` does not handle 429 with `Retry-After` back-off. The exception is caught and logged at ERROR level, but the alert is not re-queued. The `alert_log` SQLite table records `sent=False`, but no watchdog monitors that table in real time.

**How to avoid:**
1. Add `Retry-After` back-off to `_post_webhook`: if response is 429, extract `retry_after` from the response JSON, sleep, and retry once.
2. Cap the burst send rate to one request per 2 seconds (30/min with a 1.5x safety margin) using a simple inter-request delay.
3. Add an alert watchdog: if `sent=False` rows for `level="critical"` accumulate in `alert_log` within a 5-minute window, write a critical event to stderr so the operator sees it in `docker logs`.

**Warning signs:**
- `alert_send_failed` appears in logs during multi-event scenarios
- HTTP 429 in the exception traceback in structlog output
- `alert_log` table shows `sent=0` rows for `level="critical"` events

**Phase to address:**
Discord alert setup phase. Before testing the full alert suite end-to-end, add the retry/back-off logic. Verify with a burst test: fire 5 distinct criticals in < 2 seconds and confirm all 5 appear in `alert_log` with `sent=1`.

---

### Pitfall 6: APScheduler CronTrigger Fires at Wrong Time Due to Implicit Timezone

**What goes wrong:**
The equity job fires at 4:15 PM ET. APScheduler's `CronTrigger` must be configured with the `America/New_York` timezone. On a fresh Debian/Ubuntu homelab, the system timezone is UTC. If the `CronTrigger` is instantiated without an explicit `timezone` parameter, it uses the system timezone. The Docker `TZ=America/New_York` environment variable sets the OS timezone for Python's `datetime.now()` calls but does not automatically propagate into APScheduler's trigger timezone object. The job fires at 4:15 PM UTC (11:15 PM ET in winter) — wrong time, missed market close window.

This is a documented APScheduler issue: `CronTrigger.from_crontab()` does not inherit the scheduler timezone in some configurations.

**How to avoid:**
1. Always instantiate `CronTrigger` with explicit `timezone` parameter:
   ```python
   CronTrigger(hour=16, minute=15, timezone="America/New_York")
   ```
2. Set the APScheduler's `timezone` parameter at scheduler creation to match.
3. In the deployment runbook: after first deploy, verify the next scheduled fire time via `scheduler.get_jobs()` and confirm it shows the correct ET wall clock time before leaving the system unattended.
4. Add a startup log: `log.info("equity_job_scheduled", next_run=str(equity_job.next_run_time))` so the operator can visually confirm the correct time.

**Warning signs:**
- Equity cycle fires at UTC time (visible in `logs/` — 20:15 UTC vs expected 21:15 UTC in winter)
- Healthcheck pings arrive at UTC time instead of the expected ET offset
- APScheduler job list shows `next_run_time` that does not correspond to 4:15 PM ET

**Phase to address:**
Docker deployment / scheduler wiring phase. Include an explicit timezone verification step in the operator runbook and post-deploy checklist.

---

### Pitfall 7: VecNormalize Statistics Not Applied at Inference Time

**What goes wrong:**
At inference time (trading cycles), the code loads the SB3 model from `models/active/{env}/{algo}/model.zip` but may not load the corresponding `vec_normalize.pkl`. VecNormalize's running mean and variance normalize raw observations to the [-5, 5] range that the network was trained on. Without them, raw feature observations (RSI=70, close=450.0) are passed directly to the network. The model produces nonsensical actions — no error, valid output shape, but the policy operates outside its training distribution.

**Why it happens:**
Training correctly saves both `model.zip` and `vec_normalize.pkl`, but these are two separate files. Any inference path that calls `model.predict(raw_obs)` without first passing observations through the saved VecNormalize statistics silently bypasses normalization. There is no runtime error — wrong outputs are indistinguishable from correct outputs without a reference comparison.

**How to avoid:**
1. Enforce a two-file contract in every model-loading path: model and VecNormalize are always loaded together. Add an assertion: `assert vec_path.exists(), f"VecNormalize missing for {env}/{algo}"`.
2. Add a normalization integration test: load model + VecNormalize, feed known raw observations (e.g., all zeros), verify the normalized observation passes through the network and produces a bounded action in the expected range.
3. Document in the runbook: never copy `model.zip` to a new location without also copying `vec_normalize.pkl`.

**Warning signs:**
- Action values cluster near 0 or at extreme boundaries regardless of market state
- Stuck agent detector fires frequently after a new model is deployed
- Observations in debug logs show raw feature magnitudes (RSI=70) rather than normalized values (~0.3)

**Phase to address:**
Initial training + inference wiring phase (first phase of v1.1). Validate the full observation-normalization-prediction pipeline end-to-end before enabling any trading cycle.

---

### Pitfall 8: Retraining Skips Walk-Forward Validation Gates

**What goes wrong:**
The automated retraining job trains a new model and writes it to `models/shadow/` for shadow evaluation. If the retraining job calls `TrainingOrchestrator.train()` and then immediately promotes to shadow without running the walk-forward backtest (the 4-gate validation in `agents/backtest.py` and `agents/validation.py`), a model that fails the Sharpe, drawdown, win rate, or profit factor gates enters the shadow pipeline. It may survive shadow promotion if market conditions during shadow testing are favorable, promoting a fundamentally flawed model to active.

**Why it happens:**
Training completion and validation are two separate steps. The `TrainingOrchestrator` was built to train — it does not automatically trigger backtesting. A retraining job that chains `train() → save_to_shadow()` without an intermediate `validate()` call skips the safety net.

**How to avoid:**
1. The retraining job must call `AgentValidator.validate()` from `agents/validation.py` after training completes. Only if all 4 gates pass should the model be written to `models/shadow/`.
2. Add a structured log line at each validation gate with pass/fail result. Discord alert on any gate failure.
3. Include in the runbook: "A retraining run that fails any validation gate archives the new model to `models/archive/failed/` and sends a warning alert. The existing active model remains unchanged."

**Warning signs:**
- Retraining logs show `training_complete` followed immediately by `shadow_model_saved` with no `validation_gate_*` log lines between them
- New shadow model produces dramatically different Sharpe distribution than previous model
- Validation gate metrics never appear in the daily digest

**Phase to address:**
Automated retraining phase. The retraining job wiring must explicitly include the validation step. This is the operational-level connection of two v1.0 modules that were built separately.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Retraining inside APScheduler thread pool | No separate process management | CPU contention starves trading cycles during training | Never — use subprocess or separate service |
| Skip VecNormalize load at inference | Simpler code path | Silent policy degradation — model trades outside training distribution | Never |
| Deploy without volume permission check | Faster startup | Container runs healthy but writes nothing; silent data loss | Never in production |
| Placeholder Sharpe ratios in ensemble weights (v1.0 known debt) | Skip backtest integration | Ensemble weights are uniform, not Sharpe-weighted; wastes ensemble benefit | Acceptable for first training run only; must be resolved before 6-month paper period starts |
| Single Discord webhook for all alert types | Simpler config | Webhook deletion or rotation silences all alerts simultaneously | Acceptable for development; never for production |
| First model deployed without smoke test | Faster first run | Silent bad model in production with no baseline for comparison | Never |
| Bootstrap period without shadow promotion guard | No code changes needed | Mediocre or random model auto-promotes based on 0.0 active baseline | Never — add guard before enabling auto-promotion |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Discord webhooks | Deleting a webhook URL invalidates it permanently; subsequent requests return 404, which the current code treats as a non-retriable error | Rotate webhook URLs with a two-stage process: add new URL to config, verify it delivers, then remove old; never delete before new is verified |
| Alpaca paper API | Paper and live endpoints are different base URLs; a config mistake sends paper orders to live | Enforce at config load: add a `@model_validator` that cross-checks `trading_mode=paper` with the Alpaca base URL containing "paper"; fail fast at startup |
| DuckDB + Docker bind mount | DuckDB acquires an exclusive file lock; if two processes simultaneously open `market_data.ddb`, DuckDB throws `IOException: Could not set lock on file` | Dashboard container already uses `db:ro` (read-only bind mount); never open `market_data.ddb` on the host while the container is running |
| Healthchecks.io dead man's switch | If the healthcheck ping URL changes or the account lapses, the dead man's fires a false alarm (alarm fires when system is fine) | Validate ping URL at startup; add a startup log showing which URL is being pinged; test by deliberately missing a ping to confirm the alert fires to Discord |
| Binance.US paper mode | `binance_sim.py` (paper) still reads API key from config at startup — a missing key causes `KeyError` at startup, not at trade time | Validate API key presence at startup regardless of trading mode; log a clear error with remediation instructions |
| APScheduler + Docker TZ | `TZ=America/New_York` in docker-compose sets the OS timezone but APScheduler CronTrigger requires explicit `timezone=` parameter | Always pass `timezone="America/New_York"` to every `CronTrigger` that must fire in ET; verify at deployment with `scheduler.get_jobs()` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| SAC replay buffer (1M steps) held in RAM during CPU training | Training OOMs or causes homelab swap usage; 1M steps at 45 dims = ~180MB per SAC buffer — manageable alone, but three sequential trainings (PPO+A2C+SAC) plus OS baseline will use ~30-40GB during SAC training on 64GB homelab | Run one algorithm at a time sequentially; never parallelize across algorithms; monitor `htop` during first retraining run | May become a problem if observation vector dimensions grow significantly in future milestones |
| DuckDB OLAP query competing with active trading cycle | Brief file lock contention causing latency spike in execution pipeline | All OLAP queries run from the dashboard container (read-only mount); ensure no trading cycle code queries DuckDB directly | Breaks when any future development inadvertently queries DuckDB from the trading path |
| HMM regime refit with insufficient live data | `hmmlearn` diverges with < 60 observations, producing NaN in regime probabilities that propagate into the feature vector | Assert minimum observation count before HMM fit; log and skip regime feature if insufficient; fall back to regime=0 (unknown) | Breaks on the first retraining cycle if only 1-2 months of live data have accumulated |
| Full feature recomputation on every retrain | Retraining recomputes 8+ years of features on each run, adding 10-30 minutes before training starts | Cache the computed feature matrix; recompute only features for new bars since the last training run; store last-retrain timestamp | Noticeable at first retrain cycle; grows worse as historical depth increases |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| `.env` file committed to git | Alpaca/Binance.US/Discord keys exposed in repo history; trading account compromised | `detect-secrets` pre-commit hook is already in place; ensure `.env` is in `.gitignore`; verify with `git check-ignore -v .env` before first homelab commit |
| Secrets baked into Docker image via build-time ARG/ENV | Secrets visible in `docker history`; anyone with image access has the keys | Current `docker-compose.prod.yml` correctly uses `env_file: .env` at runtime only; never add secrets to `Dockerfile` |
| Dashboard port 8501 exposed on all interfaces (0.0.0.0) | Streamlit dashboard readable by anyone on the local network; shows portfolio data and PnL | Bind to `127.0.0.1:8501` only; access via SSH tunnel from M1 Mac; never expose to external network |
| Discord webhook URL leaked in a screenshot or shared message | Anyone with webhook URL can spam the Discord channel; disrupts real alerts | Treat webhook URLs as secrets; store only in `.env`; rotate immediately if accidentally exposed |
| SQLite `trading_ops.db` world-readable on host filesystem | Trade history, PnL, and position data visible to any process on the homelab | Set `chmod 700 db/` on the homelab so only the trader user can read; re-verify after each `docker compose up` |

---

## "Looks Done But Isn't" Checklist

- [ ] **Automated retraining:** Often missing the post-retrain validation step — verify that the retraining job runs all 4 walk-forward validation gates before writing to `models/shadow/`, not just `training_complete`
- [ ] **Discord alert suite:** Often missing end-to-end test with actual webhook — verify each alert level (critical, warning, info, daily digest) fires to the correct channel with correct Discord embed formatting before relying on it in production
- [ ] **Shadow promotion:** Often missing the bootstrap guard — verify that `evaluate_shadow_promotion` returns False when the `trades` table has zero rows, not True
- [ ] **Docker deployment:** Often missing volume permission verification — verify that the trader user (UID 1000) can write to all bind-mounted directories after `docker compose up -d` via `docker exec swingrl touch /app/db/.write_test`
- [ ] **APScheduler timezone:** Often missing post-deploy schedule verification — check `scheduler.get_jobs()` and confirm `next_run_time` shows 4:15 PM ET (not 4:15 PM UTC) before leaving system unattended
- [ ] **VecNormalize at inference:** Often missing — verify that the model-loading path in the trading cycle explicitly loads and applies the VecNormalize statistics file alongside `model.zip`
- [ ] **Healthchecks.io dead man's switch:** Often configured but never tested — deliberately miss a ping and confirm the expected alert fires to Discord before enabling automated trading
- [ ] **Emergency stop from outside container:** Often written but never tested — verify that `emergency_stop.py` can be invoked via `docker exec swingrl python scripts/emergency_stop.py --tier 1` without needing APScheduler to be running
- [ ] **Paper trading validation criteria:** Often started without defining pass/fail — define the 6-month exit criteria (minimum Sharpe, maximum MDD, minimum trade count per environment) before the validation period begins, not at month 6

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| MPS-trained model produces wrong outputs on CPU | MEDIUM | Re-run training on homelab CPU (1-4 hours per env/algo on i5); smoke tests will verify correctness on CPU; existing shadow period restarts from zero |
| Retraining job caused missed trading cycles | LOW | Check `alert_log` for missed cycles; no positions entered so no capital exposure; reschedule retraining to non-trading window; verify next cycle fires at correct time |
| Shadow model auto-promoted during bootstrap | LOW-MEDIUM | Add bootstrap guard to `promoter.py`; use `lifecycle.archive_shadow()` to roll back the spuriously promoted model; redeploy; shadow period restarts from zero |
| Docker volume permission denied at startup | LOW | Run `sudo chown -R 1000:1000 db/ data/ models/ logs/` on homelab; `docker compose restart`; verify write succeeds within 5 minutes via `docker exec` |
| Critical Discord alert dropped due to rate limit | LOW | Missed alert is logged in `alert_log` with `sent=0`; read missed alert from DB; add retry logic to prevent recurrence; no financial impact since circuit breakers and halt system still function independently |
| APScheduler timezone misconfigured — equity cycles fire at wrong time | MEDIUM | No trades placed during wrong-time cycles (market closed or activity mismatch); fix `CronTrigger(timezone=...)`, rebuild container, redeploy; verify next fire time before leaving unattended |
| Retraining promoted a model that skipped validation gates | HIGH | Check `models/active/` timestamps to identify when bad model was promoted; use `lifecycle.archive_shadow()` to remove; restore previous active model from `models/archive/`; investigate which validation gate was bypassed and add enforcement |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| MPS/CPU model device mismatch | Initial training on homelab | Post-train assertion: `device == "cpu"`; smoke test output against expected action range |
| Retraining overlaps trading window | Automated retraining phase | Inspect job schedule: confirm retraining windows have >= 4H gap from any trading cycle; confirm `misfire_grace_time` is set |
| Shadow promotion at bootstrap | Shadow/retraining phase | Unit test: `promoter.py` with empty `trades` table returns False; integration test with near-empty trades table |
| Docker bind mount permissions | Docker deployment phase | Pre-flight: `docker exec swingrl touch /app/db/.write_test && echo OK` must succeed as user trader (UID 1000) |
| Discord rate limit drops critical alerts | Discord alert setup phase | Burst test: fire 5 distinct critical alerts in < 3 seconds; verify all 5 appear in `alert_log` with `sent=1` |
| APScheduler timezone mismatch | Docker deployment phase | Post-deploy: print `equity_job.next_run_time` and confirm it corresponds to 4:15 PM ET wall clock |
| VecNormalize not loaded at inference | Initial training / inference wiring phase | Integration test: load model + VecNormalize file, feed raw observation vector, verify normalized action output is bounded in expected range |
| Retraining skips validation gates | Automated retraining phase | Verify retraining job logs show all 4 validation gate results before `shadow_model_saved` log event |

---

## Sources

- SwingRL v1.0 codebase reviewed directly: `src/swingrl/training/trainer.py`, `src/swingrl/monitoring/alerter.py`, `src/swingrl/shadow/promoter.py`, `src/swingrl/scheduler/jobs.py`, `docker-compose.prod.yml`, `src/swingrl/config/schema.py`
- [Discord Webhook Rate Limits — Official Discord Docs](https://discord.com/developers/docs/topics/rate-limits) — 30 requests/minute per webhook
- [Discord Webhooks Guide: Rate Limits](https://birdie0.github.io/discord-webhooks-guide/other/rate_limits.html)
- [APScheduler CronTrigger timezone issue #346](https://github.com/agronholm/apscheduler/issues/346) — timezone not defaulting to scheduler timezone
- [APScheduler DST skip bug #606](https://github.com/agronholm/apscheduler/issues/606) — jobs skip a day on DST transition
- [SB3: Loading GPU-trained model on CPU](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/159)
- [DuckDB persistent storage file lock behavior](https://github.com/duckdb/duckdb/discussions/11451)
- [SQLite in Docker — bind mount permission patterns](https://oneuptime.com/blog/post/2026-02-08-how-to-run-sqlite-in-docker-when-and-how/view)
- SwingRL `.planning/RETROSPECTIVE.md` — v1.0 lesson: gap closure phases caused by cross-module wiring not verified during development
- SwingRL `.planning/PROJECT.md` — known tech debt: placeholder Sharpe ratios in ensemble weights

---
*Pitfalls research for: SwingRL v1.1 operational deployment (retraining, Discord alerting, production Docker, operator runbooks, paper trading validation)*
*Researched: 2026-03-10*
