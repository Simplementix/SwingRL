# Options Collector — Operations Runbook

Operational guide for the `swingrl-collector` container: lifecycle, schedule, storage, and
alerting. Provider is CBOE's unauthenticated delayed-quotes endpoint (spec §17 C1) — there is
no auth/token machinery to operate.

**Last verified against code:** 2026-07-15

## Service identity

| Property | Value |
|---|---|
| Container | `swingrl-collector` |
| Image | pinned explicit tag, e.g. `swingrl-collector:2026-07-15` — never `latest` |
| Entrypoint | `python scripts/collector_main.py` |
| Own scheduler jobstore | `db/apscheduler_options.sqlite` (separate from the trader's `db/apscheduler_jobs.sqlite`) |
| Networks | `default` + `br0` (pg16 access) |
| Volumes | `./data`, `./db`, `./config`, `./logs` — same bind-mount pattern as the trader, data-only |

**A30 isolation.** `swingrl-collector` is a separate compose service with its own image, own
scheduler/jobstore, and its own tables (`options_snapshots`, `options_chains` — never the
trader's tables). Building, restarting, or recreating it **never touches** the `swingrl`
(trader/scheduler) container. It never writes to `models/active/`.

## Container lifecycle

**Service-scoped compose commands only.** Never run a bare `docker compose build`,
`docker compose up -d`, or `docker compose down` on this host — those act on every service in
`docker-compose.yml`, including the trader, and (for `up -d`) can silently recreate a pinned
container if its build context changed. Always name the service explicitly.

```bash
# Start
cd ~/swingrl && docker compose up -d swingrl-collector

# Logs (follow)
docker compose logs -f swingrl-collector
# or, tail-only:
docker logs swingrl-collector --tail 50

# Restart (re-reads config/swingrl.yaml; re-runs the boot self-check; does not change the image)
docker compose restart swingrl-collector
```

### Pinned-tag bump discipline

The `image:` line in `docker-compose.yml` is a hand-bumped, date-stamped tag
(`swingrl-collector:2026-07-15`) — deliberately **not** `latest` — so an unrelated
`docker compose build`/`up -d` on another service never recreates this container as a
side effect.

To ship new collector code:

1. Edit `image:` in `docker-compose.yml` to a new date-stamped tag.
2. Confirm you are **outside the 15:30–16:45 ET quiet window** on a trading day (see below).
3. `docker compose build --no-cache swingrl-collector`
4. `docker compose up -d swingrl-collector`
5. `docker compose logs -f swingrl-collector` — confirm `options_boot_self_check_done` then
   `options_collector_started` with the 5 job ids.

### Quiet window

**15:30–16:45 ET on trading days:** no recreation of `swingrl-collector` (image bump, `up -d`
after a build, `docker compose down`/`up`), and no homelab CI runs that span this window. The
window brackets both snapshot pulls (16:00 decision, 16:35 eod) with margin. `ci-homelab.sh`'s
cleanup is scoped to the dev compose project (`docker compose -p swingrl-ci -f
docker-compose-dev.yml down`, verified in `scripts/ci-homelab.sh`) — CI never tears down the
always-on `swingrl-collector` service.

A **restart that happens anyway** inside the window (host reboot, OOM, etc.) is not silent: the
decision snapshot's misfire grace is short (900s) so a missed cron fires and stamps `late_by_s`
+ WARNs; the eod snapshot's grace is long (18000s / 5h) so it simply fires late within the same
session; and the next boot's self-check (below) catches anything still missing by the next
`health_check_time_et` (17:15) or on the following start.

## Jobs & schedule

One job per configured snapshot (`options_collector.snapshots` in `config/swingrl.yaml`) plus
three fixed jobs, and optional calendar + candle jobs (gated on their `enabled` flags). All
times are `America/New_York`, Mon–Fri unless noted — except `candles_crypto`, which runs on a
UTC 4H grid.

| Job id | Trigger (ET) | Represents / does |
|---|---|---|
| `options_decision_snapshot` | 16:00 | Pulls the chain; label `decision`; `market_time_et=15:45` — the ~15-min-delayed feed pulled at 16:00 is assumed to show the 15:45 market state. `misfire_grace_s=900` (15 min) — beyond that, skip + WARN rather than mislabel. |
| `options_eod_snapshot` | 16:35 | Pulls the chain; label `eod`; `market_time_et=16:15` — options freeze ~16:15, so 16:35 is comfortably past the close with a 20-min buffer. `misfire_grace_s=18000` (5h) — a frozen close stays valid to capture much later in the day. |
| `options_health_check` | 17:15 | Scans the last `health_lookback_days` (3) NYSE sessions for missing/incomplete snapshots per symbol. CRITICAL if a whole snapshot is missing for any session; WARNING if only some symbols are missing. |
| `options_data_audit` | 1st of month, 18:00 | Runs `run_data_quality_audit` over the trailing 30 days (greeks range, crossed markets, OI-null, OI same-day stability). CRITICAL on any failure; INFO digest (rows/median IV per symbol) on pass. Fires on the calendar day regardless of weekday. |
| `options_offsite_backup` | 02:30, every day | `rclone sync data/options_eod/cboe → b2:swingrl-options`. Not trading-day-gated (runs weekends too, harmlessly a no-op sync). WARNING on failure. |
| `candles_equity` | `candle_jobs.equity_time_et` (16:50), Mon–Fri | Incremental daily-bar ingest via `run_equity(config, backfill=False)`, then `run_features(config)` only when new rows landed. Owns equity OHLCV freshness while training is paused (2026-07-18). `misfire_grace_time=candle_jobs.equity_misfire_grace_s` (6h). WARNING on ingestion failure — never raises (the scheduler survives). |
| `candles_crypto` | `candle_jobs.crypto_minute` (:01) past 4H UTC closes 0,4,8,12,16,20 | Incremental 4H-bar ingest via `run_crypto(config, backfill=False)`, then `detect_and_fill_crypto_gaps(config)`, then `run_features(config)` when new rows OR gaps filled. Fires ahead of the trader's :05 crypto cycles so it reads fresh bars. `misfire_grace_time=candle_jobs.crypto_misfire_grace_s` (3h). WARNING on failure — never raises. |

Both candle jobs are optional, gated on `options_collector.candle_jobs.enabled` (default on), and
drop out of the keep-set when disabled. `run_features` has no env-scoped variant, so each job
recomputes both equity and crypto features on a new-rows run. They call the EXISTING
Alpaca/Binance ingestors — CBOE stays options-only ("NEVER replace Alpaca/Binance for trader
candles", source-seams ruling 2026-07-14).

Snapshot jobs additionally skip themselves entirely on non-trading days
(`guarded_snapshot` checks `market_calendar.is_trading_day`) — no separate holiday
calendar to maintain.

## Boot self-check

Every container start (before the scheduler starts accepting cron fires) runs
`boot_self_check()`:

1. `store.reconcile()` — loads any Parquet snapshot with no matching Postgres parent row
   (see "pg16-outage self-heal" below).
2. `run_health_check(...)` — the same lookback scan as the 17:15 job, run immediately at boot.

This means a deliberate restart is always safe outside the quiet window: it re-verifies the
last 3 trading sessions and backfills Postgres from Parquet on every start, so downtime never
needs a manual "did we miss anything?" check.

## Where data lands

**Parquet (durable, written first):**

```
data/options_eod/cboe/
  SPX/                              # _SPX -> "SPX" (symbol_to_dir strips leading $/_)
    2026-07-15_decision.parquet
    2026-07-15_decision.header.json
    2026-07-15_eod.parquet
    2026-07-15_eod.header.json
  SPY/  QQQ/  VTI/  XLV/  XLI/  XLE/  XLF/  XLK/
    ... one directory per symbol (9 total: _SPX + 8 equity ETFs) ...
```

One Parquet file per `(symbol, quote_date, snapshot_label)` — the atomic write + resume unit
(`temp file → rename`, both the `.parquet` and its `.header.json` sidecar). A snapshot already
on disk is skipped on the next run for that symbol/date/label (idempotent).

**Postgres (`pg16`, database `swingrl`), synced after the Parquet write:**

| Table | Grain | Notes |
|---|---|---|
| `options_snapshots` | one row per `(underlying_symbol, quote_date, snapshot_label)` | Parent/header row: `snapshot_time_utc`, `pulled_at_utc`, `underlying_price`, `is_delayed`, `is_early_close`, `number_of_contracts`, `raw_header` (jsonb — includes `payload_timestamp`, `late_by_s`). |
| `options_chains` | one row per `(underlying_symbol, quote_date, snapshot_label, contract_symbol)` | Contract rows. `PARTITION BY RANGE (quote_date)`, monthly partitions `options_chains_YYYY_MM` auto-created on first insert into that month. `raw_json` is nullable — see below. |

Both inserts are `ON CONFLICT DO NOTHING` — safe to re-run.

## Disk growth & `postgres_store_raw_json`

`raw_json`/`raw_header` (the complete original CBOE contract objects, kept so nothing is lost
to an un-mapped field) are the bulk of the storage — spec §13 estimates tens of GB/yr across
Parquet + Postgres at ~70–100k rows/day.

- **Parquet always keeps `raw_json`** — it is the durable archive; this is never turned off.
- **Postgres's copy is gated by `options_collector.postgres_store_raw_json`** (default `true`
  in `config/swingrl.yaml`). This is the fastest-growing, least-queried column in the
  Postgres mirror. Flip it to `false` once real GB/day is known, then restart the collector
  (outside the quiet window) to pick up the change.

**Known gap:** flipping the flag back to `true` later does **not** backfill `raw_json` for
rows already synced with it `NULL` — `reconcile()` only inserts snapshots whose parent row is
entirely absent from `options_snapshots`; it does not detect or patch a `NULL`-`raw_json` row
that already has a parent. Any Parquet file (which always retains the full `raw_json`) can
still be re-read manually if a specific day ever needs backfilling — there's just no automated
job that does it.

**Monitor disk usage:**

```bash
du -sh data/options_eod/cboe
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) AS size
  FROM pg_catalog.pg_statio_user_tables
  WHERE relname LIKE 'options_%'
  ORDER BY pg_total_relation_size(relid) DESC;"
```

## Reconcile & pg16-outage self-heal

`OptionsStore.write_snapshot()` (Parquet) always runs **before**
`OptionsStore.sync_to_postgres()` for a given symbol/snapshot. So if pg16 is unreachable when a
snapshot job fires, the (un-backfillable) chain has already landed safely on disk — only the
Postgres mirror is behind.

`store.reconcile()` walks every `data/options_eod/cboe/*/*.parquet` file, reads its header
sidecar, and — for any `(symbol, quote_date, snapshot_label)` with no matching row in
`options_snapshots` — loads it into Postgres (parent + contract rows, monthly partition
created as needed). This runs automatically at every container boot (self-check trio above);
it can also be run on demand:

```bash
docker compose exec swingrl-collector python -c \
  "from scripts.collector_main import build_app; \
   n = build_app('config/swingrl.yaml')['store'].reconcile(); \
   print(f'reconciled {n} snapshot(s)')"
```

## Alert catalogue

All alerts route through the shared `Alerter` (same class/webhooks as the trader; spec §10.4,
**auth alerts retired** — CBOE needs no token, so there is no token-age/`invalid_client`
category in this collector).

| Source | Condition | Level |
|---|---|---|
| Snapshot run summary (`collector.py`) | Every symbol attempted failed | 🔴 CRITICAL |
| Snapshot run summary | Any symbol failed, OR any warning (schema drift, contract-count drop, late decision fire) | 🟡 WARNING |
| Snapshot run summary | All attempted symbols succeeded, no warnings | 🔵 INFO |
| Health check (`options_health_check`) | No snapshot for any symbol on a scanned session (missed run) | 🔴 CRITICAL |
| Health check | Snapshot present for some but not all symbols on a scanned session | 🟡 WARNING |
| Data-quality audit (`options_data_audit`) | Any hard failure (delta out of range, crossed market, OI entirely null, OI unstable intraday) | 🔴 CRITICAL |
| Data-quality audit | Pass | 🔵 INFO (rows + median IV digest per symbol) |
| Offsite backup (`options_offsite_backup`) | `rclone sync` failed | 🟡 WARNING |

Per-symbol issues (schema drift, a >50%-vs-previous contract-count drop, a late decision
fire) do **not** send their own alert — they accumulate into `SnapshotResult.warnings` and
surface once as part of that run's single summary alert, so a bad run for 9 symbols produces
one Discord message, not nine.

## Troubleshooting

### Container restart-looping

`docker logs swingrl-collector --tail 50`. Common cause: `config/swingrl.yaml` fails
`load_config()` validation, or the mounted `db/` directory is not writable (jobstore SQLite
file).

### A snapshot is missing for one symbol but others are fine

Expected transient behavior — per-symbol isolation means one fetch/parse failure doesn't abort
the run. Check the WARNING summary alert / logs for that symbol's `options_symbol_failed`
entry; it will retry automatically on the next scheduled snapshot for that label (skip-if-exists
means today's gap stays a gap until tomorrow's run, unless re-triggered manually).

### `options_data_audit` fires CRITICAL

Read the failure list in the alert (capped at 20). Cross-check with the week-1 audit runbook
(`docs/options/week1-data-quality-audit.md`) for the same SQL used to inspect the underlying
rows directly.

### Offsite backup WARNING

Confirm the `rclone` remote `b2:swingrl-options` is configured and reachable
(`docker compose exec swingrl-collector rclone lsd b2:swingrl-options`); the job broad-catches
any exception so a misconfigured remote alerts rather than crashing the container.

## Known issues / open questions

- `options_data_audit` fires on the calendar day-of-month (`audit_day_of_month=1`) with no
  weekend/holiday adjustment — if the 1st falls on a non-trading day the job still runs, just
  over whatever trailing window has data.
- `postgres_store_raw_json` has no automated backfill path when re-enabled after being off
  (see "Disk growth" above).
- Exact delay-offset (`payload_timestamp` vs wall clock, measured on a live trading day) is
  still pending the T6 probe — see `docs/options/data-caveats.md`.

## Source of truth

| Concern | File |
|---|---|
| Collector orchestration, alert routing | `src/swingrl/data/options/collector.py` |
| Scheduler, jobs, boot self-check, offsite backup | `scripts/collector_main.py` |
| Parquet + Postgres storage, reconcile | `src/swingrl/data/options/store.py` |
| Postgres DDL | `src/swingrl/data/options/schema.py` |
| Data-quality audit | `src/swingrl/data/options/audit.py` |
| Trading-day / early-close calendar | `src/swingrl/data/options/market_calendar.py` |
| Config schema | `src/swingrl/config/schema.py` (`OptionsCollectorConfig`, `OptionsSnapshotConfig`, `OptionsIntegrityConfig`, `OptionsBackupConfig`) |
| Config values | `config/swingrl.yaml` (`options_collector:` block) |
| Compose service | `docker-compose.yml` (`swingrl-collector`) |
| Design + amendments | `docs/superpowers/specs/2026-07-14-schwab-options-collector-design.md` §9, §10, §17 C1–C4 |

## Changelog

- **2026-07-15** — Initial version (Task 15).
