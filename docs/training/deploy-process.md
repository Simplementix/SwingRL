# Deploy process — trader / trainer / collector isolation (A30)

**Owner:** Plan A 2.R Task E. **Spec:** `docs/superpowers/specs/2026-06-12-training-system-redesign-design.md`
§4.14 point 5 (A30). **Prime directive:** *a training deploy must never interrupt running
paper trading, and must never silently swap the live model mid-cycle.*

## Glossary

| Term | Meaning |
|---|---|
| **trader** | `swingrl-trader` compose service — the always-on paper-trading scheduler (`scripts/main.py`). Container name `swingrl`. |
| **trainer** | `swingrl-trainer` compose service — the SAME image, profile-gated, run ephemerally for training only. |
| **collector** | `swingrl-collector` compose service — the always-on options/data collector (`scripts/collector_main.py`). |
| **pinned tag** | An explicit `image:` tag (e.g. `swingrl:trader-2026-07-16`), never `:latest`, changed only by a human edit. |
| **market-safe window** | A time span when no trading cycle is in flight, so a trader restart cannot orphan an order. |
| **hot-reload** | The trader loader re-reads a model when its file mtime changes (Plan A Task D). This is why training must never write the live `models/active/` tree. |

## Why the split exists

Before A30 a single `swingrl` service ran the scheduler **and** hosted training, with code
baked into the image (bind mounts are data-only). Two hazards followed:

1. **Process:** a `docker compose build` + recreate to ship new *training* code stopped and
   recreated the container — killing live paper trading.
2. **Models:** training writes model artifacts; the trader hot-reloads `models/active/` on
   mtime. A training write to `models/active/` would swap the live model mid-flight, unvetted.

The fix is one image, two services, plus a `models/active/` write ban.

## Service topology

```
always-on (bare `docker compose up -d` starts these):
  swingrl-trader      scheduler / paper trading      pinned  image: swingrl:trader-<date>   container_name: swingrl
  swingrl-collector   options + data capture         pinned  image: swingrl-collector:<date>
  swingrl-memory      memory / consolidation agent   build (services/memory)
  swingrl-ollama      local LLM for epoch advice     ollama/ollama:latest
  swingrl-dashboard   Streamlit read-only UI         build (dashboard)   depends_on: swingrl-trader

profile-gated (started ONLY with `--profile training`, never by bare `up -d`):
  swingrl-trainer     training host                  image: swingrl:trainer-latest   command: sleep infinity (idle)
```

All four build-based services share **one Dockerfile `production` target** and **one build
context (`.`)**. They differ only by `image:` tag, `command`, and `profiles`.

## The two build/deploy paths

### A. Training deploy — routine, trader untouched

Ship new training code and run training **without touching the trader**:

```bash
# 1. Build ONLY the trainer image (service-scoped — never bare `docker compose build`).
docker compose --profile training build swingrl-trainer

# 2. Run training ephemerally. --rm cleans up the container when it exits.
docker compose run --rm swingrl-trainer python scripts/train_pipeline.py --config config/swingrl.yaml
```

- `docker compose build swingrl-trainer` re-tags **only** `swingrl:trainer-latest`. The
  trader's `swingrl:trader-<date>` image is a different tag and is left byte-for-byte intact.
- `docker compose run` starts an ephemeral container; it never recreates the running trader.
- Training writes `models/iterations/` and shadow slots only — **never** `models/active/`.

> ⚠️ **Never run a bare `docker compose build`** (no service argument). It rebuilds *every*
> build service, which would silently re-tag `swingrl:trader-<date>` with current code and
> change what the next trader recreate deploys. Always name the service. Same rule protects
> the collector (D9).

### B. Trading deploy — deliberate, market-safe only

Shipping new *trader* code (or a vetted model bootstrap) is the only thing that recreates the
trader. It is a hand operation, done in a market-safe window:

```bash
# 1. Build + tag the new trader image with a NEW explicit date tag.
docker compose build swingrl-trader           # builds current code
docker tag swingrl:trader-2026-07-16 swingrl:trader-<new-date>   # if bumping by hand

# 2. Edit docker-compose.yml: bump swingrl-trader `image:` to swingrl:trader-<new-date>.

# 3. In a market-safe window with no cycle in flight (see checks below):
docker compose up -d swingrl-trader           # recreates ONLY the trader (its tag moved)
```

`docker compose up -d` recreates a service only when its resolved image tag changed. Because
the trader tag moves **only by hand**, unrelated `up -d` runs never recreate it.

**Market-safe windows** (no cycle in flight):

| Env | Safe window |
|---|---|
| Equity | After the daily cycle (`equity.cycle_time_et` = **15:45 ET**) **and** fill polling complete — practically **≥ ~16:05 ET** on a trading day. |
| Crypto | Between 4H cycles (cycles fire at 00:05/04:05/08:05/12:05/16:05/20:05 UTC). |
| Shared quiet window | **15:30–16:45 ET on trading days** — recreate nothing (also protects the 15:45 equity cycle and the collector's 16:00/16:35 pulls). |

**No-in-flight-cycle check** (do all three before recreating the trader):

1. `docker exec swingrl cat status/heartbeat.json` (or `ls -l status/`) — confirm the last
   cycle finished and no cycle is mid-flight.
2. `docker logs --tail 50 swingrl` — no `execute_cycle` / order-submission activity in flight;
   look for the completed-cycle log line, not an in-progress one.
3. Clock check — you are inside a market-safe window above (outside 15:30–16:45 ET).

**Rollback:** re-pin the previous tag. Edit `swingrl-trader` `image:` back to the prior
`swingrl:trader-<old-date>` and `docker compose up -d swingrl-trader` in a safe window. The old
image is still present locally (or rebuildable from the tagged git commit).

## Restart semantics — why a trader restart is safe by design

(A30 restart addendum, Task B — verified.) A recreate/restart of the trader loses no state and,
worst case, costs one graced-late or cleanly-skipped cycle:

- **Durable state in pg16 + bind mounts.** Positions, fills, portfolio snapshots, and
  `circuit_breaker_events` live in Postgres; the breaker/halt state is **DB-derived** on boot,
  not held in memory.
- **Persistent APScheduler jobstore** (`scheduler.apscheduler_db_path`, SQLite bind mount) — a
  restart re-loads the same job schedule; it does not double-fire or drop jobs.
- **Misfire grace, per env** (`scheduler.misfire_grace_s`): **equity 720 s**, **crypto 3600 s**.
  A cycle whose fire time was missed during the restart still runs if within grace; beyond grace
  it is **skipped, never replayed** (`coalesce: true`, `max_instances: 1`).
- **Boot reconciliation.** `build_app()` runs the reconciliation job once at startup (the same
  job the 17:00 ET cron runs) to catch any fill/position drift from downtime immediately.
- **Self-correcting rebalancer.** The target-weight rebalancer converges a partially-applied
  cycle on the next run.

Net: recreate the trader freely *inside a market-safe window*; the restart itself is stateless-safe.

## Migrations while the trader runs

- **Additive-only** while the trader is up: no `ALTER`/`DROP` of anything the deployed trader
  reads, outside a trader deploy window. Trainer-side deploys may apply new additive migrations.
- **Floor semantics** (Task 3): `assert_schema_current` **refuses** to start when the DB is
  *behind* the image's expected version (missing migrations), and **warns-and-runs** when the DB
  is *ahead* (newer additive migrations applied by a trainer-side deploy). Exact-match semantics
  would brick the running trader's next restart on the first Plan B migration.
- The Plan B **cutover** (V010 REVOKE etc.) is the one gated exception — the trader is
  deliberately stopped for it.

## Collector rules (D9 / C4)

The collector is always-on and independent of trader/trainer churn:

- Its own pinned tag (`swingrl-collector:<date>`). Plan A/B image churn must never recreate it
  via a bare `up -d` — use **service-scoped** compose commands only.
- Recreate it only **outside 15:30–16:45 ET** on trading days (the shared quiet window; also
  protects the 15:45 equity cycle and the 16:00/16:35 pulls).
- Restart-safe by design (persistent jobstore + boot self-check: reconcile + lookback health).
- It keeps running through trader deploys **and** through Plan B's cutover — its tables are
  untouched by V010.

## Discord alert delivery (two latent gaps, both fixed)

Both gaps surfaced live during the T16 collector deploy (2026-07-15) and both bite once paper
trading's notifications come into play. Verify at Task 16 Step 1.

1. **Webhook env.** The Alerter reads `config.alerting.alerts_webhook_url` /
   `daily_webhook_url`, populated only by the env overrides
   `SWINGRL_ALERTING__ALERTS_WEBHOOK_URL` / `SWINGRL_ALERTING__DAILY_WEBHOOK_URL`. The legacy
   `DISCORD_WEBHOOK_URL` in `.env` is read by **nothing** (it appears only in compose comments);
   with it alone, every alert boots disabled (`alert_disabled reason=no_webhook_url`). The
   homelab `.env` sets both overrides (fixed 2026-07-15). Both `swingrl-trader` and
   `swingrl-trainer` load the shared `env_file: .env`, so both inherit the overrides — **verify,
   don't assume**, at deploy.

2. **INFO alerts.** `Alerter.send_daily_digest()` had **zero** production callers, so INFO-level
   alerts buffered in memory forever and died on restart. **Decision (this task): the trader
   keeps digest semantics** (it is a persistent scheduler with meaningful INFO volume and a
   natural end-of-day flush point, unlike the low-volume collector, which uses
   `info_immediate=True`). The digest flush is wired into the existing end-of-day job:
   `daily_summary_job` (18:00 ET) now calls `send_daily_digest()`, **before** its halt gate so
   INFO buffered on a halted day still ships. An INFO-path delivery is part of the Task 16 Step 1
   Discord live proof (not just critical/warning embeds).

## `models/active/` write ban (load-bearing)

The trader loader hot-reloads `models/active/{env}/{algo}/{model.zip,vec_normalize.pkl}` on
mtime (Plan A Task D). Therefore:

- **Training writes `models/iterations/` and shadow only.** The only sanctioned writers of the
  live `models/active/` tree are the **shadow promotion / lifecycle module**
  (`src/swingrl/shadow/`) and the **Task 5 era-0 bootstrap deploy step**
  (`scripts/migrations/bootstrap_era0_models.py`) — both gated and documented.
- Enforced by `tests/test_deploy_isolation.py`: a static scan of `src/swingrl/training/`,
  `src/swingrl/memory/training/`, and `scripts/train*` fails the suite if any of them writes the
  live active tree (function-level; reads and per-iteration `models/iterations/.../active/`
  writes are allowed).
- **Known exception, retired by Plan B:** `scripts/train_pipeline.py::deploy_best_models` still
  copies winners into `models/active/` at the end of a full pipeline run. It is recorded as the
  single allowlisted exception; the Plan B cutover (spec §4.14 point 5) removes it so promotion
  to `models/active/` happens only through the gated shadow promoter. **Until then, run full
  training pipelines only in a market-safe window** — the closing deploy step writes the live
  tree and the trader will hot-reload it.

## One-time rename cutover

Renaming the service from `swingrl` to `swingrl-trader` is itself a one-time trader recreate:

1. Do it in a market-safe window (outside 15:30–16:45 ET, no cycle in flight).
2. `docker compose build swingrl-trader` then `docker compose up -d` — compose stops the old
   `swingrl`-named service definition and creates `swingrl-trader`. Because `container_name`
   stays `swingrl`, the running container name and all `docker exec swingrl` runbooks are
   unchanged.
3. Verify: `docker ps` shows the `swingrl` container healthy; `docker logs --tail 50 swingrl`
   shows `scheduler_jobs_registered count=12` and a clean boot reconciliation.
