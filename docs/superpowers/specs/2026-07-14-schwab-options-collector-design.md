# Schwab EOD Option-Chain Data Collector — Design Spec

> **⚠ AMENDED 2026-07-14 (same day, user-approved): the primary provider is now CBOE, not
> Schwab — see §17 (amendments C1–C4) before reading.** The Schwab design below is retained
> in full as **shelved fallback #1**; §4/§7 (architecture, auth) are superseded for the
> primary path. The plan (`2026-07-14-schwab-options-collector-plan.md`) is restructured
> accordingly.

- **Date:** 2026-07-14
- **Status:** Approved — amended C1–C4 (CBOE primary)
- **Author:** vpanchal-code (with Claude)
- **Related:** `~/thetadata_pull/SPX_DATA_SOURCING_SPEC.md`, `~/thetadata_pull/SPX_PREMIUM_DATA_SPEC.md` (prior intent), `docs/training/training-data-capture.md` (options table was scaffold-only)

---

## 1. Glossary (no undefined jargon — read this first)

| Term | Plain meaning |
|---|---|
| **Option chain** | The full list of option contracts for one underlying — every strike × expiration × call/put — with their prices and analytics. |
| **Underlying** | The thing the option is on (e.g. `SPY`, `$SPX`). |
| **`$SPX`** | Schwab's request symbol for S&P 500 **index** options — European-style, cash-settled. (Confirm exact form at first run; may be `$SPX` or `$SPX.X`.) |
| **Greeks** | Risk sensitivities of an option: delta, gamma, theta, vega, rho. |
| **IV (implied volatility)** | The market's expected volatility backed out of the option price. Schwab returns it as a **percent** (12.34 = 12.34%). |
| **OI (open interest)** | Number of contracts currently outstanding. |
| **EOD** | End of day. |
| **Decision snapshot** | The chain captured at **15:45 ET** — the moment the future premium-selling agent will actually trade. |
| **EOD snapshot** | The chain captured at **16:30 ET** — the frozen post-close state, for drift + settlement. |
| **Drift** | How much a contract's price/greeks moved between the 15:45 decision and the close. |
| **Lookahead bias** | Training/testing a strategy on data it could not have seen at decision time — inflates backtests, fails live. |
| **Forward-capture** | We only collect from *now* onward; history accrues over time. We cannot backfill the past. |
| **Snapshot** | One pull of one symbol's full chain at one scheduled time on one trading day. |
| **Parquet** | A compact columnar **file format** (not a database, no locks). Here it is the durable first write. |
| **Postgres / pg16** | The project's live PostgreSQL 16 database. Concurrent-safe (MVCC) — many writers/readers at once. |
| **JSONB** | Postgres's binary JSON column type — stores raw JSON and lets you query into it. |
| **MVCC** | Multi-version concurrency control — why Postgres never has the single-writer lock problem DuckDB had. |
| **schwab-py** | The Python library we use to talk to the Schwab API (handles OAuth). |
| **Access token** | Short-lived (~30 min) credential for API calls; auto-refreshed. |
| **Refresh token** | Longer-lived credential used to mint new access tokens. Schwab's expires in **7 days** (see §7). |
| **OAuth manual flow** | Headless login: print a URL, log in in any browser, copy the redirected URL back. No local server needed. |
| **`isDelayed`** | A flag in Schwab's response saying whether the quotes are real-time or 15-min delayed. |
| **APScheduler** | The Python job scheduler already used by the live trader. |
| **`exchange_calendars` / XNYS** | Library + calendar the project uses to know NYSE trading days, holidays, early closes. |
| **A30** | The project's deploy-isolation rule: never rebuild/restart the live trader; additive-only DB migrations while it runs. |
| **NBBO** | National Best Bid and Offer — best bid/ask across exchanges. What we capture; **not** the same as your fill price. |
| **Fill / slippage** | The price you actually trade at (fill) vs the quoted mid; the gap is slippage. A premium seller's edge lives here. |
| **Market regime** | The prevailing market condition (calm, volatile, crash, rally). RL needs history spanning several. |
| **Early assignment** | An American option exercised against you before expiry (common near ex-dividend). SPX (European) can't be; the 8 ETFs can. |
| **VRP (volatility risk premium)** | Implied vol minus realized vol — the core signal premium sellers harvest; reconstructable from captured IV. |
| **`isChainTruncated`** | A top-level response flag meaning Schwab returned a *partial* chain. Must be checked or strikes are silently lost. |
| **Schema drift** | Schwab silently adding/renaming/removing response fields over time; can null out our typed columns without any error. |
| **3-2-1 backup** | The standard backup rule: 3 copies, on 2 media types, with 1 offsite. |

---

## 2. Goal & scope

**Goal.** Starting immediately, reliably capture the **full option chain** for `$SPX` + the 8 equity ETFs at **two times each trading day** (15:45 ET decision, 16:30 ET close), so we accrue a proprietary historical options dataset for future RL work — a separate SPX premium-selling environment and the existing equity system.

**Forward-capture only.** We are not backfilling history. The point is to *start collecting reliably* so history builds. The 15:45 decision snapshot in particular is **un-backfillable** — if we don't record it now, that decision-time history is gone forever.

**Dual-use by design.** The Schwab integration is built as a **reusable client library** (`src/swingrl/data/options/`). The EOD collector is its first consumer; the future SPX premium-selling trader is the second — it will call the same client intraday. The library choice is quarantined behind our own wrapper so it can be swapped without touching consumers.

### In scope
- Twice-daily forward capture of full chains for the configured symbols.
- Durable, idempotent, resumable storage (Parquet → Postgres).
- Independent scheduled runtime that never touches the live trader.
- Loud Discord alerting on any failure, and explicit handling of the 7-day auth risk.
- Docs + a re-auth CLI for registering the app and doing the OAuth login.

### Non-goals (YAGNI)
- **No trading.** This collects data only. (Schwab app is Market-Data-only.)
- **No historical backfill** from Schwab or other vendors (that's the separate multi-era sourcing plan).
- **No greek recomputation** now — we store Schwab's raw greeks/IV/OI as given; recompute later if we want cross-source consistency.
- **No live options paper-trading engine** — that's the future premium env; this spec only ensures its data exists.
- **No fill/slippage modeling** — we capture quotes, not executable fills; fill realism is the future env's concern (see §6.5).

---

## 3. Verified findings & confidence

Labelled per the project's confidence discipline.

| Claim | Basis | Confidence |
|---|---|---|
| `GET /marketdata/v1/chains` returns bid/ask+sizes, greeks, IV, OI, underlying price | Schwab docs + third-party mirrors + prior `SPX_DATA_SOURCING_SPEC` | High |
| Free with a brokerage account + registered dev app; no data-subscription fee | Schwab dev portal + community | High |
| `schwab-py` handles OAuth; `client_from_manual_flow` supports headless login | schwab-py docs (fetched) | High |
| **Refresh token: hard 7-day expiry from creation; API use does NOT extend it** | schwab-py docs (verbatim), community consensus | High — **but disputed by Lumibot docs** (see §7); resolved empirically in week 1 |
| Top-level response carries `isDelayed` entitlement flag | mirrors + search | High |
| Individual option contract field names (greeks/OI/etc.) | Stable TDA-inherited schema; schwab-py does not document JSON | Medium — **pinned against a real captured response during build (fixture step)** |
| Schwab Trader API has **no paper-trading** environment (Production only) | Schwab sandbox guide + community | High |
| `$SPX` exact request symbol | Inferred; not round-tripped live | **Low — confirm on first run** |
| Real-time vs 15-min-delayed entitlement for our app | Unknown until first live pull | **Unknown — first-run check via `isDelayed`** |

**Sources:** schwab-py auth docs (readthedocs); Schwab OAuth restart-vs-refresh-token; Lumibot Schwab broker docs (the dissenting token source); schwab-py & schwabdev PyPI/GitHub; Schwab sandbox guide.

---

## 4. Architecture

Three layers. Only Layer 1 is reused by the future premium trader.

```
Layer 1 — Reusable Schwab client library   src/swingrl/data/options/
  schwab_auth.py      token manager: load / refresh / detect-expiry / instrument age / trigger alerts
  schwab_client.py    thin wrapper over schwab-py: get_option_chain(symbol) -> raw dict; rate-limit + retry
  chain_parser.py     raw chain dict -> normalized DataFrame (typed columns + raw_json)
  store.py            Parquet writer (atomic) + Postgres sync + reconcile
  config models       OptionsCollectorConfig (in src/swingrl/config/schema.py)

Layer 2 — EOD collector job                 src/swingrl/data/options/collector.py
  orchestrates: for each symbol -> fetch -> parse -> store; per-symbol isolation; alert routing;
  idempotency (skip already-captured); missed-run + token-age health checks

Layer 3 — Container + scheduler             scripts/options_collector_main.py
  configure_logging(); build Alerter; own APScheduler + own jobstore;
  register jobs (15:45 decision, 16:30 eod, daily token reminder, 17:15 health check); run
```

CLI: `scripts/schwab_reauth.py` — runs the manual OAuth flow and writes/refreshes the token file (initial login + weekly re-auth).

---

## 5. Symbols & configuration

The 8 ETFs are **read from `config.equity.symbols`** (single source of truth — never re-listed). Index underlyings come from the collector's own config. Nothing hardcoded.

```yaml
# config/swingrl.yaml  — new section
options_collector:
  enabled: true
  provider: schwab
  index_symbols: ["$SPX"]         # confirm exact request symbol at first run
  include_equity_symbols: true    # also capture chains for config.equity.symbols (SPY…XLK)
  output_dir: "data/options_eod/schwab"
  schema_version: "v1"
  snapshots:
    - { label: decision, time_et: "15:45" }
    - { label: eod,      time_et: "16:30" }
  chain:                          # nulls = full chain (all strikes, all expirations)
    contract_type: ALL
    strike_range: ALL
    include_underlying_quote: true
    from_date: null
    to_date: null
    strike_count: null
  auth:
    token_path: "secrets/schwab_token.json"
    api_key_env: SCHWAB_API_KEY
    app_secret_env: SCHWAB_APP_SECRET
    callback_url: "https://127.0.0.1:8182"
    max_token_age_days: 6.5       # proactive re-auth reminder threshold
    reminder_days: [5, 6]         # WARNING nudges before day 7
  rate_limit_per_sec: 2
  health_check_time_et: "17:15"   # verify today's snapshots landed
  token_reminder_time_et: "09:00" # daily token-age check
  integrity:
    fail_on_truncated: false      # true = raise; false = WARN + flag + bounded re-fetch
    audit_day_of_month: 1         # monthly data-quality audit
    audit_time_et: "18:00"
  backup:
    enabled: true                 # nightly offsite copy of un-backfillable data (3-2-1)
    rclone_remote: "b2:swingrl-options"
    time_et: "02:30"
```

Implemented as a Pydantic `OptionsCollectorConfig(BaseModel)` attached to `SwingRLConfig` via `Field(default_factory=...)`, mirroring `SchedulerConfig`. Env overrides work automatically (`SWINGRL_OPTIONS_COLLECTOR__ENABLED=true`). API key/secret are read from env (`.env`), never from YAML.

**Symbol → filesystem-safe directory name:** strip `$` (`$SPX` → dir `SPX`); the raw `$SPX` is preserved in the `underlying_symbol` data column.

---

## 6. Capture: what we record

### 6.1 Two snapshots per trading day

| Snapshot | Time (ET) | Purpose |
|---|---|---|
| `decision` | 15:45 | Chain as the premium agent will see it at trade time — the no-lookahead training input. **Un-backfillable.** |
| `eod` | 16:30 | Frozen post-close chain — drift measurement + settlement/marking. Robust to delayed entitlement. |

**Timing rationale.** ETF options freeze at 16:00/16:15, SPX/SPXW at 16:15; 16:30 clears all closes *and* the 15-min-delay boundary, so the EOD snapshot is the true frozen close regardless of entitlement. The 15:45 decision snapshot is only *exactly* 15:45 prices if we have **real-time** entitlement — see §11.

**Early-close days.** On NYSE half-days the market closes 13:00 ET. The collector still fires at the configured times; it records `is_early_close` (from `exchange_calendars`) in provenance. On such days the 15:45 "decision" snapshot is post-close — captured anyway for completeness; the future premium trader will adjust its own decision time on half-days.

### 6.2 Capture everything

For **un-backfillable** data we keep the complete payload, two ways at once:
- **Typed columns** for the ~40 fields we know we want (fast to query/train on).
- **`raw_json`** — the complete original contract object (JSON in Parquet, JSONB in Postgres). If Schwab returns a field we didn't map, it's still there and extractable later **without re-capturing**.

### 6.3 Flattened contract row (the grain)

**Grain:** one row per `(underlying_symbol, quote_date, snapshot_label, contract_symbol)`. Snapshot-level context (underlying price, timestamps, `is_delayed`) is **denormalized onto each row** for self-contained training reads — cheap because those repeated constants compress to near-nothing.

| Column | Schwab source | Type | Notes |
|---|---|---|---|
| `underlying_symbol` | top `symbol` | text | `$SPX`, `SPY`, … |
| `quote_date` | derived (ET session) | date | partition key |
| `snapshot_label` | config | text | `decision` \| `eod` |
| `underlying_price` | top `underlyingPrice` | float | denormalized |
| `is_delayed` | top `isDelayed` | bool | entitlement flag (denormalized) |
| `quote_time_utc` | `quoteTimeInLong` | timestamptz | epoch-ms → UTC |
| `trade_time_utc` | `tradeTimeInLong` | timestamptz | epoch-ms → UTC |
| `pulled_at_utc` | wall clock | timestamptz | when fetched |
| `source`, `schema_version` | constants | text | `schwab`, `v1` |
| `contract_symbol` | `symbol` | text | OSI id — natural key part |
| `option_root` | `optionRoot` | text | e.g. `SPX` vs `SPXW` |
| `expiration` | `expirationDate` | date | |
| `dte` | `daysToExpiration` | int | |
| `strike` | `strikePrice` | float | |
| `option_right` | `putCall` | text | `CALL` \| `PUT` (avoids SQL `right`) |
| `expiration_type` | `expirationType` | text | W/S/Q |
| `settlement_type` | `settlementType` | text | A=AM, P=PM (matters for SPX) |
| `exercise_type` | `exerciseType` | text | A/E (SPX is European) |
| `multiplier` | `multiplier` | float | 100 |
| `in_the_money` | `inTheMoney` | bool | |
| `bid`,`ask`,`last`,`mark` | same | float | |
| `bid_size`,`ask_size`,`last_size` | `bidSize`… | int | |
| `open`,`high`,`low`,`close` | `openPrice`… | float | |
| `volume` | `totalVolume` | bigint | |
| `open_interest` | `openInterest` | bigint | |
| `net_change` | `netChange` | float | |
| `delta`,`gamma`,`theta`,`vega`,`rho` | same | float | raw Schwab greeks |
| `iv` | `volatility` | float | **percent** (12.34 = 12.34%) |
| `theoretical_value` | `theoreticalOptionValue` | float | |
| `time_value`,`intrinsic_value`,`extrinsic_value` | same | float | |
| `raw_json` | entire contract object | jsonb | complete raw payload |

**Missing-data honesty.** On illiquid contracts (VTI, deep wings) Schwab returns sentinels like `-999.0`/`NaN` for greeks/IV. The parser maps those to real `NaN` on capture (never stores `-999`), and records the mapping in provenance.

### 6.4 Snapshot-level context (parent row / metadata)

Recorded once per `(symbol, date, snapshot)`: `underlying_price`, `is_delayed`, `interest_rate`, `dividend_yield`, underlying `volatility`, `number_of_contracts`, `status`, `snapshot_time_utc`, `pulled_at_utc`, `is_early_close`, `source`, `schema_version`, and the full `raw_header` (the response minus the strike maps).

### 6.5 Known data limitations (read before training)

This dataset is the right thing to start now, but it has a specific shape. Know its boundaries before trusting a model trained on it.

- **Quotes are not fills.** We capture NBBO bid/ask, not executable prices. A premium-selling agent trained on quote-mid will be **systematically optimistic** vs live fills — the single biggest reason options backtests lie. This data is *necessary but not sufficient*: the eventual env must model slippage/fill-probability, or be re-grounded on **actual fills** captured once paper/live trading runs. Never trust a premium backtest that assumes mid-price fills.
- **Two moments, not the path.** We snapshot 15:45 and 16:30 — not the intraday path and not time-&-sales prints. "Did the underlying touch my strike intraday?" is not answerable from this data. Fine to start; know the gap.
- **Starting from zero, likely one regime.** History accrues forward only. A year of capture may span a single market regime; the premium env is not meaningfully trainable until the data covers varied conditions (calm/volatile/crash/rally) — realistically 1–2+ years. Value depends on the *variety* of conditions captured, not just elapsed time.
- **SPX (European, cash-settled) vs the 8 ETFs (American).** ETF options carry **early-assignment risk** (esp. around ex-dividend: XLV, XLF, XLE, …) that is invisible in a chain snapshot. SPX has none. Expect SPX to be the far more useful premium-selling substrate; the ETF options data may prove secondary.

---

## 7. Auth & the 7-day token

### 7.1 The two tokens

| Token | Lifetime | Extended by use? |
|---|---|---|
| Access token | ~30 min | ✅ Yes — every call auto-refreshes it. |
| **Refresh token** | **7 days, hard, from creation** | ❌ **No** (per schwab-py). |

**Disputed.** schwab-py docs say the 7 days is a hard limit from creation that use does *not* extend; Lumibot docs claim daily use "rolls the window forward." We do not resolve this in the abstract. We **design for the worse case** (hard 7-day) and **instrument the token** so week 1 reveals the truth empirically. The asymmetry: designing for hard-7-day when it actually rolls forward costs nothing; the reverse means silent weekly death — the exact failure we're preventing.

### 7.2 Strategy (chosen: manual weekly safety-net + loud alerts)

- **Initial + weekly re-auth:** `scripts/schwab_reauth.py` runs schwab-py `client_from_manual_flow` — prints the Schwab login URL, user logs in + MFA in any browser, copies the redirected `https://127.0.0.1:8182/?code=…` URL back into the terminal (that page failing to load is expected — nothing listens on 8182; see §7.4), token written to `secrets/schwab_token.json`.
- **Instrumentation every run:** log the token file's issue timestamp + age. **WARNING** Discord reminders at `reminder_days` (5, 6). **CRITICAL** the instant a refresh is rejected (`invalid_client`).
- **Week-1 finding recorded** in `metadata.json`: does the token survive past 7 days under daily use? If it rolls forward, the weekly manual step becomes a rarely-fired safety net and we can relax reminders.

### 7.3 Secrets & token file

- `SCHWAB_API_KEY`, `SCHWAB_APP_SECRET` in `.env` (gitignored). Never in YAML or git.
- Token file `secrets/schwab_token.json`: gitignored, `chmod 600`, mounted into the container (never baked into the image). It holds account-scoped access — treat as a credential.
- `.gitignore` additions: `data/options_eod/`, `secrets/schwab_token.json`, `secrets/`.

### 7.4 Callback URL — no open port needed

Registered callback: **`https://127.0.0.1:8182`** (HTTPS, exact match required). `127.0.0.1` is loopback — never network-facing, nothing to open in a firewall/router, no Docker port mapping. Because we use the **manual flow**, nothing listens on 8182 at all; the redirect lands on a dead loopback address in the browser and we copy the URL by hand.

---

## 8. Storage model

**House pattern (confirmed in recon):** Parquet written first (durable) → synced into Postgres (`ON CONFLICT DO NOTHING`). We follow it.

### 8.1 Parquet layout (durable capture + resume unit)

```
data/options_eod/schwab/
  SPX/  2026-07-14_decision.parquet   2026-07-14_eod.parquet   …
  SPY/  2026-07-14_decision.parquet   2026-07-14_eod.parquet   …
  …one dir per symbol…
  metadata.json      # directory-level provenance sidecar
  collector.log      # append-only run log (structlog to stdout is primary)
```

- One Parquet file per `(symbol, date, snapshot_label)` — the **resume unit**.
- Atomic write: temp file → `rename` (matches existing `ParquetStore`).
- Carries all typed columns + `raw_json`.
- The Parquet file **is** the durability/dead-letter layer: written *before* any DB call, so a pg16 outage at 15:45/16:30 never loses the (un-backfillable) capture.

### 8.2 Postgres tables (queryable mirror) — migration `V011_options_capture` (additive)

```sql
-- Parent: one row per (symbol, trading day, snapshot)
CREATE TABLE options_snapshots (
  underlying_symbol   text NOT NULL,
  quote_date          date NOT NULL,
  snapshot_label      text NOT NULL,          -- 'decision' | 'eod'
  snapshot_time_utc   timestamptz NOT NULL,
  pulled_at_utc       timestamptz NOT NULL,
  underlying_price    double precision,
  is_delayed          boolean,
  is_early_close      boolean,
  interest_rate       double precision,
  dividend_yield      double precision,
  underlying_volatility double precision,
  number_of_contracts integer,
  status              text,
  source              text NOT NULL DEFAULT 'schwab',
  schema_version      text NOT NULL,
  raw_header          jsonb NOT NULL,
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label)
);

-- Child: one row per contract, monthly range-partitioned on quote_date
CREATE TABLE options_chains (
  underlying_symbol text NOT NULL,
  quote_date        date NOT NULL,
  snapshot_label    text NOT NULL,
  contract_symbol   text NOT NULL,
  option_root text, expiration date, dte integer, strike double precision,
  option_right text, expiration_type text, settlement_type text, exercise_type text,
  multiplier double precision, in_the_money boolean,
  bid double precision, ask double precision, last double precision, mark double precision,
  bid_size integer, ask_size integer, last_size integer,
  open double precision, high double precision, low double precision, close double precision,
  volume bigint, open_interest bigint, net_change double precision,
  delta double precision, gamma double precision, theta double precision,
  vega double precision, rho double precision,
  iv double precision, theoretical_value double precision,
  time_value double precision, intrinsic_value double precision, extrinsic_value double precision,
  underlying_price double precision, is_delayed boolean,
  quote_time_utc timestamptz, trade_time_utc timestamptz,
  pulled_at_utc timestamptz NOT NULL,
  source text NOT NULL DEFAULT 'schwab', schema_version text NOT NULL,
  raw_json jsonb NOT NULL,
  PRIMARY KEY (underlying_symbol, quote_date, snapshot_label, contract_symbol)
) PARTITION BY RANGE (quote_date);
-- Monthly partitions (options_chains_2026_07, …) auto-created by the collector before insert.
```

- **Idempotent:** `INSERT … ON CONFLICT (PK) DO NOTHING`, plus a pre-pull "skip if this `(symbol, date, snapshot)` already exists" check (Parquet file exists **and** parent row exists).
- **Reconcile (self-heal):** at each run start, find Parquet files whose `(symbol, date, snapshot)` has no `options_snapshots` parent row → load them into Postgres. DB outages self-heal on the next successful run.
- **Volume:** ~70–100k rows/day → ~18–25M rows/yr. Monthly partitioning + the PK keep it fast. If training ever needs faster slices, export a date-range to a flat file on demand (future, only if needed).

### 8.3 `metadata.json` sidecar

Directory-level provenance (human-readable, matches the existing `spx_options` convention): `source=schwab`, endpoint, `schema_version`, symbol list, snapshot policy (15:45 + 16:30), first-run entitlement finding (`isDelayed`), week-1 token-behavior finding, `pulled_at`, per-symbol coverage summary.

---

## 9. Scheduling & runtime

### 9.1 Independent container

New `docker-compose.yml` service `swingrl-options`, **reusing the trader's image** (`target: production`) with a different entrypoint (`python scripts/options_collector_main.py`). Mounts: `./data`, `./db`, `./config`, `./logs`, `./secrets`. Network: `default` + `br0` (pg16 access). `restart: unless-stopped`, `TZ=America/New_York`. Not behind a `profiles:` gate — it runs always-on like the trader. Rebuilding/restarting it **never touches the trader** (separate service, separate lifecycle) — A30-compliant.

### 9.2 Its own scheduler

APScheduler `BackgroundScheduler` with its **own** jobstore `db/options_collector_jobs.sqlite` (never shares the trader's `db/apscheduler_jobs.sqlite`). Jobs:

| Job | Trigger (ET) | Action |
|---|---|---|
| `options_decision_snapshot` | cron 15:45, Mon–Fri | Capture `decision` snapshot for all symbols |
| `options_eod_snapshot` | cron 16:30, Mon–Fri | Capture `eod` snapshot for all symbols |
| `options_token_reminder` | cron 09:00, daily | Check token age → WARNING at day 5/6 |
| `options_health_check` | cron 17:15, Mon–Fri | Safety-net for a *silently missed* run: verify today's expected snapshots exist — whole snapshot missing → CRITICAL (missed run); some symbols missing → WARNING |
| `options_data_audit` | cron monthly (1st, 18:00) | Automated data-quality audit over the trailing month → CRITICAL on failure (see §10.6) |
| `options_offsite_backup` | cron 02:30, daily | `rclone` sync of `data/options_eod/` + Postgres dump to offsite (3-2-1) — see §13 |

`misfire_grace_time` set so a briefly-down box still fires late-but-same-session. **Trading-day guard inside each snapshot job:** `exchange_calendars.get_calendar("XNYS").is_session(today)` — skip holidays/weekends (cron already excludes Sat/Sun; the guard catches holidays). The health check exists because the per-run alerts only fire when a run *executes*; the check catches the case where a job never ran at all.

---

## 10. Idempotency, resumability & error handling

### 10.1 Idempotency / resumability
- Skip any `(symbol, date, snapshot)` already captured (Parquet exists + parent row exists).
- Per-symbol, per-snapshot resume unit — a partial run resumes cleanly.
- Atomic Parquet writes; `ON CONFLICT DO NOTHING` on DB sync; reconcile drains unsynced files.

### 10.2 Per-symbol isolation
One symbol failing (fetch/parse) never aborts the others. Results are collected, then a single summary alert is sent. Auth failure is the exception — it blocks all symbols → CRITICAL.

### 10.3 Errors
Raise/catch `DataError` (`src/swingrl/utils/exceptions.py`) for fetch/parse/empty failures. Log with structlog **before** raising. `tenacity` for transient-retry with backoff on the HTTP calls.

### 10.4 Discord alerting (reuses existing `Alerter`)

Constructed from `config.alerting.*` exactly as `scripts/main.py` does.

| Event | Level |
|---|---|
| Auth/token expired (`invalid_client`) — blocks all | 🔴 CRITICAL |
| Missed run (a snapshot absent for all symbols by health-check time) | 🔴 CRITICAL |
| All symbols failed | 🔴 CRITICAL |
| Token age day 5 / 6 | 🟡 WARNING |
| Single symbol fetch/parse failed | 🟡 WARNING |
| Empty chain (e.g. VTI unexpectedly empty) | 🟡 WARNING |
| Truncated chain (`isChainTruncated=true`) — partial data | 🟡 WARNING (+ bounded re-fetch) |
| Schema drift — an expected field missing/renamed | 🟡 WARNING |
| Data-quality audit failure (greeks/OI/spread sanity) | 🔴 CRITICAL |
| Successful run | 🔵 INFO (daily digest) |

### 10.5 Silent-corruption guards

The dangerous failure is the collector *succeeding* while storing broken data (un-backfillable). Guards on every capture:
- **Truncation:** read the top-level `isChainTruncated`. If true, attempt a bounded re-fetch (e.g. split the request by expiration ranges to stay under any size cap); if still truncated, store what we got, flag it in provenance, and WARN. Never store a partial chain as if it were complete.
- **Schema drift:** assert the expected key fields (`delta`/`gamma`/`theta`/`vega`, `volatility`, `openInterest`, `bid`/`ask`, `strikePrice`, `expirationDate`) are present in the raw payload. If a mapped field is missing/renamed, WARN with the field name — the value is still safe in `raw_json`; only the typed column is affected.
- **Row-count sanity:** compare `number_of_contracts` to the rows actually parsed; a mismatch → WARN.

### 10.6 Data-quality audit (the most important safety net)

Because the data is un-backfillable, "it didn't crash" is not "it's good." A dedicated audit runs the checks a crash never catches:
- **Week 1 (manual, in the runbook):** reconstruct an IV surface for SPX; confirm greeks are sane (delta ∈ [-1, 1], monotone across strikes), OI is populated, `bid ≤ ask` with plausible spreads, and decision→eod drift is non-trivial and plausible.
- **Monthly (scheduled job `options_data_audit`):** re-run the same automated checks over the trailing month per symbol; any failure → CRITICAL Discord alert. This is the only thing that surfaces slow quality rot before it costs a year of data.

---

## 11. First-run entitlement check (gates the decision snapshot)

On the first live pull, read the top-level **`isDelayed`** flag and record it in `metadata.json` + `options_snapshots.is_delayed`:
- `isDelayed = false` (real-time) → the 15:45 decision snapshot is genuine decision-time data. Ideal.
- `isDelayed = true` (15-min delayed) → a 15:45 pull returns ~15:30 data. Still captured (a fixed-offset intraday snapshot), but flagged as not exact decision-time until entitlement is resolved. The 16:30 EOD snapshot is unaffected (frozen close).

This is the **#1 first-run finding** — it directly determines the decision snapshot's fidelity for the future premium env.

---

## 12. Testing (TDD, tests-first)

Per project convention (`tests/test_<module>.py`, `test_<behavior>`, REQ-ID docstrings, conftest fixtures).

| Module | Key behaviors under test |
|---|---|
| `chain_parser` | raw dict fixture → DataFrame: schema, dtypes, greeks passthrough, `-999`/`NaN` → NaN, CALL/PUT split, `dte`, epoch-ms → UTC, `iv` percent preserved, `raw_json` populated |
| `store` (Parquet) | atomic write, skip-existing, per-`(symbol,date,label)` file, round-trip |
| `store` (Postgres) | `ON CONFLICT DO NOTHING` idempotency, monthly-partition auto-create, reconcile loads unsynced files |
| `schwab_auth` | token-age math, reminder thresholds, re-auth-needed detection, `invalid_client` → CRITICAL (schwab-py mocked) |
| calendar/schedule | trading-day skip (weekend/holiday), early-close detection, missed-run detection |
| `collector` | per-symbol isolation (one fails, others succeed), alert routing (`Alerter` mocked), empty-chain WARNING |

Fixtures seeded from **one sanitized real chain response** captured during the build (Phase 0/3) — this is also where the exact contract field mapping is pinned.

---

## 13. Security, operational discipline & A30 compliance

- **Least privilege:** Schwab app is Market-Data-only — the token cannot trade or read account balances.
- **Secrets:** API key/secret in `.env`; token file `chmod 600`, gitignored, mounted (never in image).
- **Re-auth single-source discipline:** a *new* Schwab login **invalidates the previous refresh token**. Always re-auth into the homelab's mounted token file and **nowhere else** — a stray login on another machine silently kills the homelab token. Called out prominently in the re-auth runbook.
- **Storage growth + offsite backup:** `raw_json`/`raw_header` JSONB is bulky (~tens of GB/yr across Parquet + Postgres). Monitor disk. Because this data is **un-backfillable**, an offsite copy is high-value: a nightly `rclone` sync of `data/options_eod/` + the Postgres dump to offsite storage (B2), via the `options_offsite_backup` job — this folds in the previously-deferred **3-2-1 backup**.
- **Alert fatigue:** the system depends on you noticing the auth-lapse alert. Keep CRITICAL rare and real; all routine/INFO goes to the daily digest so the signal isn't buried.
- **A30:** separate always-on container (rebuild/restart never touches the trader); migration `V011` is **additive** (new tables only) → safe while the trader runs; the collector writes only to its own tables + `data/options_eod/` (no writes to `models/active/`, no shared trader tables).

---

## 14. Build sequence (phased, TDD)

| Phase | Deliverable |
|---|---|
| **0** | Schwab app (✅ done, market-data-only) · `.env` keys · `scripts/schwab_reauth.py` · **first manual OAuth → token file** · **capture one sanitized chain response** · **record `isDelayed` entitlement** |
| **1** | `OptionsCollectorConfig` schema + YAML block + tests |
| **2** | `schwab_client.py` (token manager + `get_option_chain`) + tests (schwab-py mocked) |
| **3** | `chain_parser.py` (raw → DataFrame, typed + `raw_json`) + tests (recorded fixture) |
| **4** | `store.py` Parquet writer (atomic, per symbol/date/label) + tests |
| **5** | Migration `V011_options_capture` + Postgres sync/upsert + reconcile + tests |
| **6** | `collector.py` orchestration (per-symbol loop, isolation, alerting, **silent-corruption guards §10.5**) + tests |
| **7** | `scripts/options_collector_main.py` scheduler entrypoint (jobs: 2 snapshots, token reminder, health check, **monthly audit §10.6**, **offsite backup**) + `audit.py` module + tests |
| **8** | Docker service + compose wiring + `.gitignore` + `.env.example` |
| **9** | Docs: register-schwab-app · first-OAuth · **weekly-reauth runbook (single-source discipline)** · **week-1 data-quality audit runbook** · ops |
| **10** | Homelab CI → first live run → confirm entitlement + chain volumes + `isChainTruncated` → **week-1 data-quality audit** → observe week-1 token behavior → **stand up offsite backup** |

---

## 15. Open questions / risks (to resolve empirically, not guess)

1. **`$SPX` request symbol** — confirm exact form on first pull (`$SPX` vs `$SPX.X`). *Low effort, first-run.*
2. **Entitlement** (`isDelayed`) — real-time vs 15-min-delayed. Gates decision-snapshot fidelity. *First-run.*
3. **7-day token behavior** — hard expiry vs rolls-forward. Instrumented; resolved in week 1.
4. **Chain volume per symbol** — confirm full-chain row counts; tune `strike_range`/date bounds only if volume is a problem (defaults stay full).
5. **Greeks/IV presence** — confirm liquid contracts return real greeks (not sentinels).
6. **App activation lag** — a freshly created Schwab app may sit "Approved – Pending" for hours before keys work.
7. **SPX chain truncation** — does a full `$SPX` chain trip `isChainTruncated`? If so, define the expiration-split bounded re-fetch (§10.5). *First-run.*

---

## 16. Success criteria

- Both snapshots (15:45, 16:30) captured for all 9 symbols on every trading day, idempotently, with no interference to the live trader.
- Data lands in both Parquet and Postgres; DB outages self-heal via reconcile.
- Every failure mode (auth lapse, missed run, empty chain) produces a loud Discord alert.
- Re-auth is a documented ~30-second weekly action; auth lapses never pass silently.
- The Schwab client library is reusable, unchanged, by the future SPX premium trader.
- No silent data corruption: truncation and schema drift are detected and alerted; week-1 and monthly data-quality audits pass (sane greeks/OI/spreads, reconstructable IV surface, plausible decision→eod drift).
- An offsite (3-2-1) backup of the un-backfillable captured data exists.

---

## 17. Amendments (2026-07-14, user-approved — session: three-plan master-sequence reconciliation)

Numbered C1–C4, mirroring the redesign spec's amendment-log convention. Each states what
it supersedes. The Schwab sections above are **not deleted** — they are the documented,
fully-researched fallback path.

### C1 — Primary provider = CBOE delayed-quotes endpoint (supersedes §4 architecture for the primary path)

**Verified live 2026-07-14 (empirical pulls, this session):**

| Claim | Evidence | Confidence |
|---|---|---|
| `https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` returns the **full SPX chain, no auth**: 29,434 contracts (SPX + SPXW roots), 13 MB, 0.25 s | Live pull 2026-07-14 ~23:38 ET | **Verified** |
| Per-contract fields: `bid/ask` **with sizes**, `iv`, `open_interest`, `volume`, `delta/gamma/theta/vega/rho`, `theo`, `last_trade_price/time`, `prev_day_close`, OHLC | Same pull, payload inspected | **Verified** |
| Header (`data`): `current_price`, `bid/ask`, `iv30`, `seqno`, `last_trade_time`, top-level `timestamp` | Same pull | **Verified** |
| All 8 equity ETFs served by the same endpoint (`SPY.json` 13,730 rows … `VTI.json` 944 rows, real OI/IV) | Live pulls, all four spot-checked symbols 200 OK | **Verified** |
| Quotes populated even ~23:38 ET — SPX trades overnight (global trading hours) | Same pull | Verified (bonus finding) |
| Companion endpoint `…/charts/historical/{symbol}.json`: daily OHLCV, SPY→2004, **`_SPX` index →1975** (12,990 rows, current same-day) | Live pulls | **Verified** |
| Data is 15-minute delayed: **content delay = 15 m 27 s at both probes** (wall clock vs header `last_trade_time`, 15:50 + 16:05 ET). Top-level `timestamp` = UTC *generation* time (~23 s behind wall clock — not the quote time); `last_trade_time` = ET. Pull times 16:00/16:35 confirmed, unchanged. | T6 probe 2026-07-15, log `.superpowers/sdd/t6-probe-2026-07-15.log` (dev checkout) | **Verified** |
| First **scheduled** captures 2026-07-16: 9/9 symbols both labels (Parquet + Postgres `SUCCESS`); decision content 15:44:36 ET (delay 15 m 26 s, 24 s pre-label — **no-lookahead re-confirmed**); eod content 16:14:59 ET = the 16:15 freeze; full 9-symbol run ≈ 11 s; iv sentinel 0 literal zeros, NaN 3.7–14.7 % per symbol; Discord INFO path proven E2E (eod message delivered). Contract-count baselines recorded in `data/options_eod/cboe/metadata.json`. | Collector logs + pg16 + Discord, 2026-07-16 | **Verified** |
| Endpoint stability / rate tolerance | Undocumented public feed powering cboe.com; widely used by open-source tooling; **no SLA** | Assumed — mitigated (health checks catch breakage same-day; two researched fallbacks behind the provider wrapper) |

**Why CBOE wins for this collector:** zero auth (no token, no 7-day ritual, no secrets file,
no OAuth machinery — deletes the design's single biggest operational burden and its whole
failure class); SPX is CBOE's own exclusive product (primary source, not a broker re-serve);
one GET per underlying; free. Costs accepted: 15-min delay (handled by C3), undocumented
endpoint (handled by fallback ladder + guards), fewer metadata fields (strike/expiry/right
parsed from the OSI symbol; no settlement/exercise-type or rate/dividend header — FRED
covers rate/dividend if greeks are ever recomputed, per §6.5/data-caveats).

### C2 — Provider fallback ladder (supersedes §7 for the primary path; §7 stays as fallback documentation)

| Rank | Provider | Status | Why not primary |
|---|---|---|---|
| Primary | **CBOE delayed-quotes** | This spec, as amended | — |
| Fallback #1 | **Schwab** (§4/§7 design, plan's shelved tasks) | App registered; design complete; **no token created — zero standing maintenance** | 7-day token = weekly human ritual; entitlement unverified |
| Fallback #2 | **moomoo OpenAPI** | Researched 2026-07-14 (session record): free real-time OPRA LV1 w/ funded account, SPX in scope, snapshot path fits chain volumes | OpenD gateway auth churn (SMS/CAPTCHA on server-side whitelist expiry, hours–days; login-blocking forced upgrades) — unfit for unattended |
| Rejected | E*TRADE | Researched 2026-07-14 | Daily hard token expiry at midnight ET, no refresh mechanism; SPX support unverifiable; platform stagnant |
| Rejected | Headless-browser auth automation (any provider) | — | Full account credentials on disk + ToS risk + silent breakage — bad capital-preservation trade |

The provider stays quarantined behind the wrapper (§2 dual-use design) — a swap touches the
client module only, never parser/store/scheduler consumers' interfaces.

### C3 — Timing model: pull time ≠ market time (amends §6.1/§11)

- The stored record separates **`pull time`** (wall clock of the fetch) from **`quote/market
  time`** (the moment the quotes represent, from the payload's own timestamp). Provenance
  honesty replaces the entitlement question — §11's `isDelayed` first-run check is
  superseded by a **delay-convention measurement** (plan T6): pull twice around 15:50/16:05
  on a trading day, compare payload timestamps to wall clock, pin the offset.
- **Decision snapshot: pull at ~16:00 ET to capture the ~15:45 market state** (15-min delay
  assumed, verified at T6). The label's `market_time_et` stays 15:45; a late-fired pull
  records `late_by_s` and alerts — a "decision" row must never silently contain post-close
  state (lookahead-bias guard).
- **EOD snapshot: pull at 16:35 ET unchanged** — options freeze by 16:15, so the delayed
  view at 16:35 *is* the frozen close (decision D3's reasoning survives the provider swap).
- **Live real-time chains are explicitly NOT this collector's problem**: the future premium
  trader gets real-time data from its executing broker (ranked candidates: Schwab / moomoo /
  Tradier-IBKR — session research on record). **Source-seam note (standing):** training
  history = CBOE-delayed; live decisions = broker real-time; at premium go-live, run an
  overlap period capturing both to measure the offset. Features built on z-scores/ratios/
  spreads (per §6.5) wash out most level effects.

### C4 — Restart-resilience decisions (amends §9/§10; user-approved in-session)

The collector will be restarted many times while Plans A/B rework the same host. Locked:

1. **Per-label misfire grace, config-driven** (replaces the single constant): decision
   ~600–900 s (beyond that, skip + alert — never capture mislabeled state); eod ~4–6 h
   (frozen close stays valid). Late fires stamp `late_by_s`.
2. **Health check gains a lookback window** (last N trading days) and **also runs at boot** —
   a watchdog that was down at 17:15 still catches yesterday's hole on the next start.
3. **Boot-time self-check trio**: reconcile (already designed) + lookback health check
   (+ the token-age check, applicable only if the Schwab fallback is ever activated).
4. **`swingrl-collector`** (renamed from `swingrl-options` — it will absorb the FRED
   calendar ingest per Plan A Task 11's amendment, and later a scheduled OHLCV refresh)
   is **pinned to its own explicit image tag**; recreation only outside the
   **15:30–16:45 ET quiet window** on trading days.
5. **`ci-homelab.sh` cleanup must be scoped to the dev compose project** before the first
   always-on deploy (verified: today's stage 5 `docker compose down` would kill any
   always-on service on every CI run).

Invariant achieved: any restart outside the quiet window is a non-event; downtime inside it
either self-heals within grace or alerts loudly within hours — never silently.
