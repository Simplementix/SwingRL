# Architecture Research

**Domain:** Automated RL trading system — operational deployment
**Researched:** 2026-03-10
**Confidence:** HIGH (derived from reading existing production code, not inference)

---

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                          M1 MacBook Pro (Dev)                          │
│  ┌───────────────────┐  ┌──────────────────────────────────────────┐   │
│  │  scripts/train.py │  │  scripts/deploy_model.sh                 │   │
│  │  (MPS-accelerated │  │  SCP → homelab:/models/shadow/{env}/     │   │
│  │   for initial     │  │  SHA256 verify → remote smoke test       │   │
│  │   bootstrap only) │  └──────────────────────────────────────────┘   │
│  └───────────────────┘                                                 │
└────────────────────────────────────────────────────────────────────────┘
                              │ Tailscale (SSH/SCP)
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Intel i5 Homelab (Docker Production)                │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Container: swingrl (production target)                         │   │
│  │                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  scripts/main.py — APScheduler daemon (12 jobs)         │   │   │
│  │  │                                                         │   │   │
│  │  │  Trading Jobs             Maintenance Jobs              │   │   │
│  │  │  ─────────────            ─────────────────             │   │   │
│  │  │  equity_cycle (4:15 ET)   daily_sqlite_backup (2am)     │   │   │
│  │  │  crypto_cycle (6x/day)    weekly_duckdb_backup (sun 3am) │   │   │
│  │  │  daily_summary (6pm ET)   monthly_offsite (1st 4am)     │   │   │
│  │  │  reconciliation (5pm ET)  monthly_macro (1st 6pm)       │   │   │
│  │  │  stuck_agent (5:30pm ET)  weekly_fundamentals (sun 6pm) │   │   │
│  │  │  shadow_promo (7pm ET)    automated_trigger (every 5m)  │   │   │
│  │  │                                                         │   │   │
│  │  │  [MISSING] equity_retrain_job (monthly)                 │   │   │
│  │  │  [MISSING] crypto_retrain_job (biweekly)                │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐  │   │
│  │  │  Execution  │  │  Feature    │  │  Monitoring/Alerter   │  │   │
│  │  │  Pipeline   │  │  Pipeline   │  │  (Discord webhooks)   │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────────────────┘  │   │
│  │         │                │                                      │   │
│  │  ┌──────┴────────────────┴──────────────────────────────────┐  │   │
│  │  │  DatabaseManager                                          │  │   │
│  │  │  SQLite (trading_ops.db — OLTP)                          │  │   │
│  │  │  DuckDB (market_data.ddb — OLAP)                         │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Container: swingrl-dashboard                                   │   │
│  │  Streamlit on :8501 (read-only DB mount)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Bind mounts (persist across restarts):                                │
│  ./data  ./db  ./models  ./logs  ./config  ./status                   │
└────────────────────────────────────────────────────────────────────────┘
                              │ webhooks
                              ▼
                    ┌──────────────────┐
                    │  Discord         │
                    │  #alerts channel │
                    │  #daily channel  │
                    └──────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `scripts/main.py` | APScheduler daemon, 12 jobs, signal handling, startup wiring | EXISTS |
| `swingrl/scheduler/jobs.py` | All job functions; init_job_context pattern | EXISTS |
| `swingrl/monitoring/alerter.py` | Discord webhook POST, level routing, cooldown, embed building | EXISTS |
| `swingrl/monitoring/embeds.py` | Discord embed constructors (trade, daily summary) | EXISTS |
| `swingrl/shadow/lifecycle.py` | Model state machine: Training→Shadow→Active→Archive | EXISTS |
| `swingrl/shadow/promoter.py` | 3-criteria auto-promotion (Sharpe, MDD, CB check) | EXISTS |
| `swingrl/shadow/shadow_runner.py` | Shadow inference per trading cycle (non-blocking) | EXISTS |
| `swingrl/training/trainer.py` | TrainingOrchestrator wrapping SB3 learn() | EXISTS |
| `swingrl/training/ensemble.py` | EnsembleBlender (Sharpe-weighted softmax) | EXISTS |
| `scripts/train.py` | CLI for manual training runs | EXISTS |
| `scripts/deploy_model.sh` | SCP to shadow + SHA256 verify + remote smoke test | EXISTS |
| `Dockerfile` | Two-stage: ci (with dev deps) + production (minimal) | EXISTS |
| `docker-compose.prod.yml` | Two containers: swingrl + swingrl-dashboard | EXISTS |
| **Retraining job** | APScheduler cron calling train.py logic on homelab CPU | **MISSING** |
| **Retrain alert suite** | Discord embeds for retrain start/complete/fail/promote | **MISSING** |
| **Production .env** | Alpaca/Binance.US/Discord secrets for homelab | **MISSING** |
| **Operator runbook** | Walkthroughs: deploy, retrain, emergency stop, key rotation | **MISSING** |

---

## New Component: Automated Retraining

### Gap Analysis

The v1.0 scheduler has 12 jobs, none of which trigger retraining. Retraining is explicitly listed as an active requirement in PROJECT.md. The training machinery exists (`trainer.py`, `scripts/train.py`, `scripts/deploy_model.sh`) but nothing calls it on a schedule.

### Retraining Integration Points

The retraining job needs to fit the existing `jobs.py` pattern exactly:

1. `_get_ctx()` to obtain `JobContext` (config, db, pipeline, alerter)
2. Halt check via `is_halted(ctx.db)` — skip retrain when system is halted
3. Call `TrainingOrchestrator.train(env, algo, features, prices)` for all 3 algorithms
4. On success, call `ModelLifecycle.deploy_to_shadow(model_path, env_name)`
5. Alert via `ctx.alerter.send_alert(...)` at each stage (start, complete per algo, shadow deploy, fail)
6. Shadow evaluation runs independently on the existing `shadow_promotion_check_job` (daily 7pm ET)

The retraining job does NOT need to handle promotion — that already exists in `shadow_promotion_check_job`. It only trains and drops models into `models/shadow/{env}/`. This is the correct separation.

### New Jobs to Add to `scheduler/jobs.py`

```python
def equity_retrain_job() -> None:
    """Retrain PPO/A2C/SAC on equity environment (monthly, 1st of month 8pm ET)."""
    # halt check
    # load features from DuckDB (existing FeaturePipeline)
    # TrainingOrchestrator.train("equity", algo) for ppo, a2c, sac
    # ModelLifecycle.deploy_to_shadow(path, "equity") for each
    # alert: retrain_complete or retrain_failed

def crypto_retrain_job() -> None:
    """Retrain PPO/A2C/SAC on crypto environment (biweekly, 1st and 15th 9pm ET)."""
    # same pattern as equity_retrain_job
```

### New APScheduler Registrations in `main.py`

```python
# Equity: monthly, 1st of month at 8pm ET (after monthly_macro runs at 6pm)
scheduler.add_job(equity_retrain_job, trigger="cron",
    day=1, hour=20, minute=0, timezone="America/New_York",
    id="equity_retrain", replace_existing=True)

# Crypto: biweekly, 1st and 15th at 9pm ET
scheduler.add_job(crypto_retrain_job, trigger="cron",
    day="1,15", hour=21, minute=0, timezone="America/New_York",
    id="crypto_retrain", replace_existing=True)
```

Job count increases from 12 to 14. Update `main.py` docstring and log counts.

### Scheduling Rationale

- Equity retraining on the 1st: aligns with `monthly_macro_job` which refreshes FRED data at 6pm on the 1st. Retrain starts at 8pm, guaranteeing updated macro features are available.
- Crypto biweekly (1st and 15th): crypto markets run 24/7 with faster regime changes; 2x monthly reduces staleness. 9pm ET gives 5 hours of off-peak homelab time before US markets open.
- Both jobs use CPU training — homelab Intel i5 has no GPU. SB3 defaults to CPU when no GPU detected. No code change needed; the existing `ALGO_MAP` in `trainer.py` uses SB3 defaults.

### Config: No New Fields Required

Retraining schedule is hardcoded in APScheduler (same pattern as existing 12 jobs). If configurability is needed later, add `RetrainingConfig` to `schema.py` under `SwingRLConfig`. Do not add it speculatively.

---

## Discord Alert Integration

### What Exists

`Alerter` in `monitoring/alerter.py` is complete:
- Two-webhook routing: `alerts_webhook_url` (critical/warning) + `daily_webhook_url` (info/digest)
- Cooldown (default 30 min) for critical/warning dedup
- `send_alert(level, title, body)` and `send_embed(level, embed_dict)` methods
- Thread-safe (used by APScheduler thread pool)
- `AlertingConfig` in `schema.py` already has all four fields: `alerts_webhook_url`, `daily_webhook_url`, `healthchecks_equity_url`, `healthchecks_crypto_url`

### What Is Missing

1. **Production webhook URLs** — the config fields exist, but `config/swingrl.yaml` has empty strings. Production values must be set in homelab `.env` file via:
   ```
   SWINGRL_ALERTING__ALERTS_WEBHOOK_URL=https://discord.com/api/webhooks/...
   SWINGRL_ALERTING__DAILY_WEBHOOK_URL=https://discord.com/api/webhooks/...
   SWINGRL_ALERTING__HEALTHCHECKS_EQUITY_URL=https://hc-ping.com/...
   SWINGRL_ALERTING__HEALTHCHECKS_CRYPTO_URL=https://hc-ping.com/...
   ```
   These override the YAML at runtime via Pydantic `SWINGRL_` prefix + double-underscore nested fields (already wired in `schema.py`).

2. **Retraining alert embeds** — new embeds in `monitoring/embeds.py`:
   - `build_retrain_start_embed(env, algo_count)` — info level
   - `build_retrain_complete_embed(env, results)` — info level (Sharpe per algo)
   - `build_retrain_failed_embed(env, error)` — critical level
   - `build_shadow_deployed_embed(env, model_path)` — info level

3. **Discord channel setup** — two webhooks needed: `#alerts` (critical/warning immediate) and `#daily` (daily digest + retrain summaries). This is operator configuration, not code.

### Alert Flow for Retraining

```
equity_retrain_job starts
  → send_alert("info", "Equity Retrain Started", "Training PPO/A2C/SAC on 3yr equity data")

  PPO completes → send_embed("info", build_retrain_complete_embed("equity", "ppo", sharpe=1.23))
  A2C completes → send_embed("info", ...)
  SAC completes → send_embed("info", ...)

  shadow deploy succeeds
  → send_embed("info", build_shadow_deployed_embed("equity", model_path))

  OR: any algo fails
  → send_alert("critical", "Equity Retrain Failed", str(error))
```

---

## Production Docker Deployment

### What Exists and Is Ready

The Docker setup is production-grade:
- `Dockerfile` two-stage (ci + production), non-root trader user (UID 1000)
- `docker-compose.prod.yml` with resource limits (2.5GB/1CPU for swingrl, 512MB/0.5CPU for dashboard)
- Five bind mounts persist data: `./data`, `./db`, `./models`, `./logs`, `./config`, `./status`
- `restart: unless-stopped` for automatic recovery
- `HEALTHCHECK` in Dockerfile calls `scripts/healthcheck.py` every 60s

### What Is Missing

1. **Homelab `.env` file** — must be created on homelab (never committed). Minimum required:
   ```
   ALPACA_API_KEY=...
   ALPACA_SECRET_KEY=...
   BINANCE_API_KEY=...
   BINANCE_SECRET_KEY=...
   SWINGRL_ALERTING__ALERTS_WEBHOOK_URL=...
   SWINGRL_ALERTING__DAILY_WEBHOOK_URL=...
   SWINGRL_ALERTING__HEALTHCHECKS_EQUITY_URL=...
   SWINGRL_ALERTING__HEALTHCHECKS_CRYPTO_URL=...
   SWINGRL_TRADING_MODE=paper
   ```

2. **Initial data ingestion on homelab** — `market_data.ddb` and historical bars must exist before the daemon starts. The production container mounts `./data` from host, but bars must be seeded first. Existing `scripts/seed_production.py` likely handles this — verify before deployment.

3. **Initial trained models** — `models/active/{equity,crypto}/{ppo,a2c,sac}/` must have models loaded before inference can run. Either: (a) train on M1 Mac and deploy via `deploy_model.sh` then promote, or (b) train directly on homelab via `docker exec` before starting the daemon.

4. **DB schema initialization** — `scripts/init_db.py` must run before the daemon starts. Existing script, just needs to be wired into the deployment procedure.

### Deployment Procedure (Architecture View)

```
Homelab setup sequence:
  1. git clone / git pull → /home/user/swingrl
  2. Create .env with all secrets
  3. docker compose -f docker-compose.prod.yml build --target production
  4. docker exec swingrl python scripts/init_db.py       # create 28 tables
  5. docker exec swingrl python scripts/seed_production.py  # ingest historical bars
  6. On M1 Mac: python scripts/train.py --env equity --algo all
  7. bash scripts/deploy_model.sh models/active/equity/ppo/*.zip equity homelab
     (repeat for a2c, sac; and both envs)
  8. On homelab: promote shadow → active (lifecycle.promote or manual)
  9. docker compose -f docker-compose.prod.yml up -d
  10. Verify: docker logs swingrl | head -50
```

The training-on-M1-then-deploy pattern applies only for the initial bootstrap. After that, automated retraining runs entirely on the homelab.

---

## Architectural Patterns

### Pattern 1: Jobs-as-Functions with Module-Level Context

**What:** All scheduled jobs are plain functions in `jobs.py`. Shared state (config, db, pipeline, alerter) lives in a module-level `_ctx: JobContext`. `init_job_context()` is called once at startup.

**When to use:** For new jobs, including retraining. Follow identically — do not pass context as arguments or use a class.

**Trade-offs:** Simpler than dependency injection; testable via direct `_ctx` assignment in tests. APScheduler calls functions by reference so no extra wiring needed.

**Example for new retraining jobs:**
```python
def equity_retrain_job() -> None:
    """Retrain equity models monthly."""
    ctx = _get_ctx()
    if is_halted(ctx.db):
        log.warning("equity_retrain_skipped", reason="halt_flag_active")
        return
    try:
        # ... training logic ...
    except Exception:
        log.exception("equity_retrain_failed")
        ctx.alerter.send_alert("critical", "Equity Retrain Failed", ...)
```

### Pattern 2: Shadow Promotion Separation

**What:** Retraining and promotion are decoupled. Retraining drops models to `models/shadow/`. The existing `shadow_promotion_check_job` (daily 7pm ET) handles promotion via 3-criteria evaluation. Retraining jobs never promote directly.

**When to use:** Always. This separation is load-bearing: it ensures retrained models soak in shadow evaluation before touching active inference, even on months where retraining runs at 8pm and shadow promotion runs at 7pm (next day's evaluation will handle it).

**Trade-offs:** Adds 1+ day evaluation lag on newly trained models. This is intentional — capital preservation over speed.

### Pattern 3: Discord Alert Routing by Level

**What:** `Alerter.send_alert(level, title, body)` routes by level. "critical" and "warning" go to `alerts_webhook_url` immediately. "info" buffers to `daily_webhook_url` digest (or sends to daily immediately for embeds). Cooldown prevents duplicate critical alerts within 30 minutes.

**When to use:** Retrain jobs should use "critical" for failures (operator must investigate), "info" for starts/completes (historical record, not actionable).

**Trade-offs:** Info alerts may not appear until daily digest if not using `send_embed`. Use `send_embed("info", ...)` with `build_*_embed()` for immediate-but-info-routed messages.

### Pattern 4: Config via Environment Variables for Secrets

**What:** `SwingRLConfig` uses `pydantic_settings` with `SWINGRL_` prefix and `__` for nested fields. Secrets in homelab `.env` file override YAML defaults without modifying config files. Docker `env_file: .env` in `docker-compose.prod.yml` loads these at container start.

**When to use:** All broker API keys, Discord webhook URLs, Healthchecks.io URLs. Never put these in `config/swingrl.yaml`.

---

## Data Flow

### Retraining Data Flow

```
APScheduler triggers equity_retrain_job (monthly, 1st 8pm ET)
    ↓
halt_check (skip if halted)
    ↓
FeaturePipeline.get_features("equity")
    → reads market_data.ddb (DuckDB) via duckdb_conn
    → returns (features_df: DataFrame, prices_df: DataFrame)
    ↓
TrainingOrchestrator.train("equity", "ppo", features, prices)
    → constructs StockTradingEnv → DummyVecEnv → VecNormalize
    → SB3 PPO.learn(timesteps=1_000_000, callback=ConvergenceCallback)
    → saves: models/active/equity/ppo/{algo}_{timestamp}.zip
    ↓ (repeat for a2c, sac)
    ↓
EnsembleBlender.compute_weights([ppo_result, a2c_result, sac_result])
    → writes ensemble weights to DuckDB (model_metadata table)
    ↓
ModelLifecycle.deploy_to_shadow(model_path, "equity")
    → copies .zip to models/shadow/equity/
    ↓
Alerter.send_embed("info", build_shadow_deployed_embed(...))
    → HTTP POST to Discord daily webhook
    ↓
[Next day 7pm ET]: shadow_promotion_check_job runs
    → evaluate_shadow_promotion() → 3 criteria
    → promote OR archive
    → alert via Discord
```

### Trading Cycle Data Flow (Unchanged)

```
APScheduler triggers equity_cycle (4:15pm ET)
    ↓
ExecutionPipeline.execute_cycle("equity")
    → FeaturePipeline → observation vector (156-dim)
    → EnsembleBlender → weighted action
    → RiskVeto → CircuitBreaker check
    → AlpacaAdapter.submit_order()
    ↓
FillResult → build_trade_embed → Alerter.send_embed("info", ...)
    → Discord #daily channel
    ↓
ping_healthcheck(healthchecks_equity_url)
    → Healthchecks.io dead man's switch
    ↓
run_shadow_inference(ctx, "equity")
    → shadow model scores trade (no execution)
    → writes shadow_trades table (SQLite)
```

---

## Integration Points

### External Services

| Service | Integration Pattern | Config Location | Notes |
|---------|---------------------|-----------------|-------|
| Discord webhooks | HTTP POST via `httpx` in `Alerter` | `SWINGRL_ALERTING__*_WEBHOOK_URL` env vars | Two channels: #alerts, #daily |
| Alpaca paper API | `AlpacaAdapter` in `execution/adapters/` | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` env vars | Paper mode enforced by `SWINGRL_TRADING_MODE=paper` |
| Binance.US | `BinanceAdapter` in `execution/adapters/` | `BINANCE_API_KEY`, `BINANCE_SECRET_KEY` env vars | Crypto only |
| Healthchecks.io | HTTP GET in `scheduler/healthcheck_ping.py` | `SWINGRL_ALERTING__HEALTHCHECKS_*_URL` env vars | Dead man's switch per environment |
| FRED API | `fred.py` in `data/` | No API key for free tier | Monthly macro refresh |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `scheduler/jobs.py` → `training/trainer.py` | Direct function call within same process | New for retraining jobs |
| `scheduler/jobs.py` → `shadow/lifecycle.py` | Direct function call | Already used by shadow_promotion_check_job; retraining adds deploy_to_shadow call |
| `training/trainer.py` → `data/` (DuckDB) | Read via `FeaturePipeline` passed by caller | Trainer does not access DB directly; receives features DataFrame |
| `monitoring/alerter.py` → Discord | `httpx` HTTP POST, async-safe with threading.Lock | Thread pool executor calls alerter from multiple jobs |
| APScheduler → job functions | Function reference registered at startup | `coalesce=True, max_instances=1` prevents overlapping retrains |
| Docker container → host filesystem | Bind mounts for data, db, models, logs | Retraining writes to `./models/` which is bind-mounted; persists across restarts |

### New Internal Boundary: Retraining → Shadow

The key new boundary is: `equity_retrain_job` / `crypto_retrain_job` → `ModelLifecycle.deploy_to_shadow()`. This is the only action retraining takes on the model directory. The shadow promoter reads from there independently.

```
equity_retrain_job()
    calls ModelLifecycle.deploy_to_shadow(new_model, "equity")
    [writes to models/shadow/equity/]

shadow_promotion_check_job() (next day)
    calls evaluate_shadow_promotion(env="equity")
    [reads models/shadow/equity/, compares shadow_trades vs trades in SQLite]
    [on pass: lifecycle.promote("equity") moves to models/active/equity/]
```

---

## Build Order for v1.1

Based on the gap analysis — what exists vs what is missing — the correct build order accounts for data/model dependencies:

1. **Data ingestion pipeline** (load historical bars into DuckDB on homelab)
   - Prerequisite: everything else that reads DuckDB. Must come first.
   - Existing code: `data/alpaca.py`, `data/binance.py`, `data/fred.py`, `scripts/seed_production.py`

2. **Initial model training** (train PPO/A2C/SAC on M1 Mac, deploy to homelab)
   - Prerequisite: bars in DuckDB. FeaturePipeline.get_features() reads DuckDB.
   - Tools: `scripts/train.py` (M1), `scripts/deploy_model.sh` (transfer), `ModelLifecycle.promote()` (activate)

3. **Homelab Docker deployment** (production container up with active models + paper trading)
   - Prerequisite: models in `models/active/`, DB initialized, `.env` created.
   - This establishes the paper trading baseline before adding retraining.

4. **Discord webhook setup** (create server channels, configure `.env`)
   - Can be parallel with step 3. Alerter handles missing webhooks gracefully (logs warning, skips POST).

5. **Automated retraining jobs** (add `equity_retrain_job` + `crypto_retrain_job` to `jobs.py` and `main.py`)
   - Prerequisite: paper trading running (provides `shadow_trades` data for promotion evaluation).
   - Prerequisite: Discord webhooks (retrain alerts are the first test of the new embed types).
   - New code: two job functions in `jobs.py`, two `add_job()` calls in `main.py`, 4 embed builders in `embeds.py`.

6. **Operator runbook** (document all workflows with exact commands)
   - Last: can only be written accurately once all components are deployed and verified.

---

## Anti-Patterns

### Anti-Pattern 1: Training on M1 Mac for Automated Retraining

**What people do:** Run `scripts/train.py` on M1 Mac and SCP models to homelab for every retrain cycle.

**Why it's wrong:** Defeats the "hands-off" constraint. Requires operator present and M1 Mac online. Automated retraining must run on homelab without human intervention. The initial bootstrap is the only justified use of M1 training.

**Do this instead:** Add `equity_retrain_job` and `crypto_retrain_job` to `scheduler/jobs.py`. These call `TrainingOrchestrator` directly in the homelab Docker container. SB3 detects no GPU and uses CPU automatically. Intel i5 with 64GB RAM is sufficient for 1M timestep CPU training (expect 2-4 hours per environment per algo run).

### Anti-Pattern 2: Promoting Directly from Retraining Job

**What people do:** Train model, then immediately promote to active within the same job.

**Why it's wrong:** Skips the shadow evaluation period. A newly trained model might have worse out-of-sample Sharpe than the current active model. The 3-criteria promoter in `shadow/promoter.py` is the capital-preservation gate.

**Do this instead:** Retraining job only calls `ModelLifecycle.deploy_to_shadow()`. Let the existing `shadow_promotion_check_job` (daily 7pm ET) handle promotion after the model accumulates shadow trade data.

### Anti-Pattern 3: Hardcoding Discord Webhook URLs

**What people do:** Put webhook URLs in `config/swingrl.yaml` committed to the repo.

**Why it's wrong:** Exposes webhook URLs in git history. Anyone with the URL can POST to your Discord channel. CLAUDE.md explicitly prohibits hardcoded values.

**Do this instead:** Set `SWINGRL_ALERTING__ALERTS_WEBHOOK_URL=...` and `SWINGRL_ALERTING__DAILY_WEBHOOK_URL=...` in the homelab `.env` file. Never commit `.env`. The `AlertingConfig` fields default to empty string, which `Alerter.__init__` handles by disabling sending.

### Anti-Pattern 4: Running Retraining and Trading Cycles Concurrently

**What people do:** Allow retraining and the equity/crypto trading cycles to run in the same APScheduler thread pool slot at the same time.

**Why it's wrong:** Retraining writes to `models/active/` (before shadow handoff) while the execution pipeline reads from `models/active/` for inference. File-level race condition.

**Do this instead:** The existing APScheduler config already has `max_instances=1` per job ID and `coalesce=True`. Ensure retraining jobs write to a timestamped temp path within `models/active/{env}/{algo}/`, not overwriting in place. The `TrainingOrchestrator` already uses `{algo}_{timestamp}.zip` naming. Confirm this before deployment.

---

## Scaling Considerations

This is a single-operator system with fixed asset universe (8 ETFs + 2 crypto). Scaling is not a meaningful concern. The relevant operational stability considerations are:

| Concern | Current State | Risk | Mitigation |
|---------|--------------|------|------------|
| Retraining duration | ~2-4h CPU per env (6 algos total) | Overlaps with trading cycle if started at wrong time | Schedule retrain jobs after market close (8pm ET), market opens next day |
| DuckDB concurrent reads | Single connection in production | Training reads DuckDB same time as inference | Both are read-only during inference; conflict only if retraining also writes DuckDB (ensemble weights). Use separate connection for trainer writes. |
| Docker memory (2.5GB limit) | SB3 SAC has 1M buffer_size replay buffer | OOM during SAC training | SAC buffer_size can be reduced at training time; replay buffer of 1M float32 transitions = ~6GB for equity obs. Reduce to 200K for Docker. Flag in PITFALLS.md. |
| Model file atomicity | Trainer saves `.zip` in place | Inference reads partial file if trainer writes during cycle | Trainer writes to temp path → `shutil.move()` for atomic replace. Verify in `trainer.py` before deployment. |

---

## Sources

- `scripts/main.py` — APScheduler wiring, 12 existing jobs (read directly, HIGH confidence)
- `src/swingrl/scheduler/jobs.py` — All job function implementations (read directly, HIGH confidence)
- `src/swingrl/monitoring/alerter.py` — Discord webhook alerter, two-webhook routing (read directly, HIGH confidence)
- `src/swingrl/shadow/lifecycle.py` — ModelState machine, deploy_to_shadow, promote, archive (read directly, HIGH confidence)
- `src/swingrl/shadow/promoter.py` — 3-criteria evaluation (read directly, HIGH confidence)
- `src/swingrl/training/trainer.py` — TrainingOrchestrator, HYPERPARAMS, ALGO_MAP (read directly, HIGH confidence)
- `src/swingrl/config/schema.py` — All config models including AlertingConfig, ShadowConfig (read directly, HIGH confidence)
- `docker-compose.prod.yml` — Resource limits, bind mounts, two-container setup (read directly, HIGH confidence)
- `Dockerfile` — Two-stage build, non-root trader user, HEALTHCHECK (read directly, HIGH confidence)
- `scripts/deploy_model.sh` — SCP + SHA256 + remote smoke test workflow (read directly, HIGH confidence)
- `.planning/PROJECT.md` — Current milestone goals, active requirements, known tech debt (read directly, HIGH confidence)

---

*Architecture research for: SwingRL v1.1 Operational Deployment*
*Researched: 2026-03-10*
