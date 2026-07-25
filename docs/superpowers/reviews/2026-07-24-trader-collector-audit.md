# Trader / Collector Audit — Data, Accuracy, Risk Controls, Observability

> **Status: OPEN — partially specced. See the 2026-07-25 update below.**
> **Audited:** 2026-07-24, read-only, against the live homelab database and the code at
> `e1d86df` (branch `swingrl/25-monitoring-dashboard`, cut from `swingrl/2.R-training-redesign`).
>
> **UPDATE 2026-07-25.** A follow-up value-audit session answered Q-A/Q-B/Q-C, added **8 findings**
> (A-1 … A-8 below), and produced a spec and plan covering A-2, A-3, A-6 and G1-5:
> - Spec: `docs/superpowers/specs/2026-07-25-equity-data-reingestion-design.md` (LD-2 … LD-7)
> - Plan: `docs/superpowers/plans/2026-07-25-equity-data-reingestion-plan.md`
> - Branch: `swingrl/26-equity-data-reingestion`
>
> **Method note that generalises:** everything in *Confirmed healthy* below was cleared on
> **recency** — "the newest row is today". That is the same check that passed `features_equity`
> while 32 of its columns were constant zero. Value-auditing three of those "healthy" tables found
> real defects in all three. **Recency is not health.**
> **How this was found:** writing the Phase 0 data audit for the monitoring dashboard
> (`docs/superpowers/plans/2026-07-24-monitoring-dashboard-plan.md`) surfaced defects in the live
> trading path that matter more than the dashboard. That plan is TABLED pending these fixes.
> **Companion:** the dashboard plan's Phase 0 audit (findings A-1 … A-27) is the original,
> dashboard-facing write-up of many of the same facts.

Every `file:line` reference below was read during the audit. Anything inferred rather than read is
marked **UNVERIFIED**. Nothing here has been fixed; no code was written.

---

## Glossary — read this first

Per the project's plain-English rule, every shorthand used below, defined once.

| Term | Meaning |
|---|---|
| **Carve-out** | The notional capital the trader is allowed to use ($400 equity, $47.09 crypto) inside a much larger $100,000 Alpaca paper account. The carve-out is a config value, not the broker's balance |
| **Observation vector** | The array of numbers the RL agent sees each cycle — prices, indicators, macro, regime. If one slot is stale, the agent decides on stale information without knowing |
| **Train/serve skew** | The model was trained on one shape of input and is now served a different one. It degrades decisions silently, because nothing errors |
| **Fill** | An executed trade. "Partial fill" means only part of the order executed |
| **Notional order** | An order placed in dollars ("buy $50 of SPY") rather than in shares. Alpaca supports this; it is how this system trades |
| **Regulatory fees** | SEC and FINRA TAF charges on equity **sells**. Not commission — Alpaca charges no commission — but they still reduce the ledger |
| **Account activities** | Alpaca's separate record of post-fill events including fees. Not present on the order object, so it needs its own API call |
| **Staleness fuse** | A rule that halts trading when an input has not refreshed within a window |
| **Fail-open** | A recorder that swallows its own errors so it can never block trading. Consequence: a missing row does not prove the event did not happen |
| **Audit row** | A `circuit_breaker_events` row written for the record only, never representing a real halt. Marked by `reason` starting `stop-breach-audit:` |
| **Mahalanobis distance** | The statistical measure behind the turbulence signal — how unusual today's readings are versus their historical relationship |
| **Additive migration** | A schema change that only adds (new column, new table). Required while the trader is running, per A30 |
| **A30** | The project's deploy-isolation rule: while paper trading is live, no deploy may rebuild or restart the trader outside a market-safe window |
| **P1 … P4** | Priority. P1 affects live decisions now; P2 corrupts the record or breaks on the next sell; P3 unblocks dashboard monitoring; P4 is cleanup |

---

## ⚠️ Binding constraint — read before designing anything in Group 2

The Alpaca paper account holds **$100,000**. The trader must trade a **$400 equity carve-out**
(`capital.equity_usd = 400.0`) and a **$47.09 crypto carve-out** (`capital.crypto_usd`), **not** the
broker's balance.

- **Never make the trader read broker portfolio value, account equity, or buying power.** It would
  see $100K and size orders roughly 250× too large.
- **VERIFIED:** nothing in `src/swingrl/execution/` reads `buying_power`, `get_account`, or account
  equity today — grep returns zero hits. The carve-out is already respected. Do not "fix" this.
- **`PositionTracker.get_portfolio_value()` (`execution/risk/position_tracker.py:48-73`) reading
  `portfolio_snapshots.total_value` is CORRECT BY DESIGN** — it is the carve-out's own ledger. An
  earlier draft of this audit called it a self-referential defect; **that framing was wrong and was
  retracted by the user on 2026-07-24.** Leave the portfolio-value path alone.
- The ledger is only as accurate as the transactions written into it. That, and nothing above it, is
  Group 2's scope: per-transaction quantity, price and fees.

---

## Locked decisions

Settled with the user. **Do not re-litigate these** — read the rationale before proposing anything
that depends on them.

### LD-1 · Crypto stays on Binance.US (decided 2026-07-24)

Moving crypto to Alpaca — for paper trading now, and for live later — was raised and **rejected**.

**Rationale (user's):** Binance.US has materially better data, international volume, and deeper
liquidity than Alpaca's crypto offering. No change for now.

**Supporting evidence gathered while considering it:**

- **The venues price different books, and this project already has a scar from it.** Review **H5** →
  `scheduler/stop_polling.py:1-7` and `:152-154`: remapping `BTCUSDT` → `BTCUSD` *"priced a different
  book than the position was opened on."* The fix was to price the configured symbol verbatim.
- **The instruments differ, not just the venues.** Binance.US trades `BTCUSDT` (Tether-quoted);
  Alpaca trades `BTC/USD` (dollar-quoted). USDT floats near $1 without pegging exactly.
- **Train/serve skew risk.** The crypto models are trained on **19,269 Binance 4H bars per symbol,
  2017-08-17 → 2026-07-24 (~9 years)**. Switching execution to a shorter Alpaca series without
  retraining would introduce the same silent defect class as G1-4's empty `fundamentals` table.
- **Sunk anchor.** `benchmark_baselines` was re-anchored for crypto at $48.09 against real first-fill
  prices (PR #42, 2026-07-22). Changing venue invalidates that anchor.
- **Never verified, and no longer needs to be:** Alpaca's crypto history depth, fee schedule,
  minimum order size, and available pairs. The $47.09 crypto carve-out would have made the minimum
  order size a live constraint.

**Consequences — these are BY DESIGN, not defects. Do not "fix" them:**

| Observation | Why it is correct |
|---|---|
| `reconciliation_job` is **equity-only** (`scheduler/jobs.py:702`) | Binance.US has no paper trading, so crypto fills are simulated locally and there is **no external authority** to reconcile against |
| Group 2's entire scope is equity-only | Same reason. This is deliberate, not an oversight |
| `api_errors` has one writer, `execution/adapters/binance_sim.py:345` | Those are simulated errors; there is no real crypto broker to error |
| Crypto has no pending-order lifecycle | Simulated fills are immediate, so no pending state exists. The equity-only gate at `execution/pipeline.py:472` is correct |
| Crypto commission ($0.1524 over 9 trades) is a **local model, not a measurement** | `binance_sim.py` remains load-bearing indefinitely. **There are no real crypto fees to capture — do not go looking for them** |

**Standing limitation to state plainly wherever crypto P&L is presented:** the crypto side of the
carve-out ledger is a *model*, not a measurement, and will remain so while LD-1 holds. Equity is
broker-verifiable; crypto is not.

---

## Chunk map — how this work is being sequenced

The register is large deliberately; it is worked in chunks, not all at once.

| Chunk | Scope | Items | Status |
|---|---|---|---|
| **1a** | **Equity bar data** — specced and planned 2026-07-25 | A-2, A-3, A-6, G1-5 | 📋 **planned, not executed** |
| **1b** | **Remaining feeds** — the live-decision defects | **A-1**, G1-1, G1-2, G1-3 | 🔨 **next — A-1 first** |
| **1c** | **The ledger** — what the trader writes | Group 2 | ⏸ **blocked on the Q-D ruling (still owed)** |
| 2 | Risk controls | Group 3 | ☐ queued |
| 3 | Observability — the backend work the dashboard needs | Group 4 | ☐ queued |
| 4 | Discord notifications | Group 5 | ☐ queued |
| 5 | Resume the tabled dashboard plan (restore the pending-orders panel, D-1, D-2) | — | ☐ queued |

Item IDs (`G1-1`, `G4-3`, …) are stable. Reference them from specs, plans and commit messages.

---

## Group 1 — Data feeds

**Files:** `data/fred.py` · `features/health.py` · `features/pipeline.py` · `features/assembler.py` ·
`scripts/main.py` (schedules) · the collector (`data/options/collector.py` for the 4-hourly pattern)

### G1-1 🔴 Three FRED daily series frozen since 2026-04-09; live cycles use a stale VIX

- `macro_features` newest observation per series: `VIXCLS` **2026-04-09**, `DFF` **2026-04-09**,
  `T10Y2Y` **2026-04-10**. `CPIAUCSL` and `UNRATE` are current to 2026-06-01 (monthly series, fine).
- Ingest **runs and reports success** — `data_ingestion_log` shows macro runs on 2026-07-18/19 with
  `status='success'`; the three daily series return `no_data`. The FRED key and API therefore work;
  the failure is series-specific.
- Reaches live decisions: `features/pipeline.py:443` → `vix = latest.get("VIXCLS", 0.0)`. Every
  crypto cycle on 2026-07-24 recorded `vix = 19.49` — the 2026-04-09 value.
- Six shared macro features are affected (`SHARED_MACRO = 6`, `features/assembler.py:42`).
- Consequence: the emergency auto-stop trigger "VIX > 40 **and** 24 h drawdown ≥ 13%" can never
  fire, because the number cannot move.
- **NOT DIAGNOSED — root cause.** Start at the incremental-fetch watermark logic in `data/fred.py`.
  Note that April 2026 was the Postgres migration window.

### G1-2 🔴 Macro cadence is monthly; it must become 4-hourly in the collector *(user requirement)*

- Today: `monthly_macro_job`, cron `day=1, hour=18` ET (`scripts/main.py:192`), run by the **trader**.
- Wanted: **daily data, refreshed by the collector every 4 hours alongside crypto ingestion.**
- Retire or repoint `monthly_macro_job`; mirror the collector's existing 4-hourly job pattern.

### G1-3 🟠 The staleness fuse is incompatible with the cadence, and resets on every restart

- `FeatureHealthTracker.STALENESS_SECONDS = 7 days` (`features/health.py:81`), and macro staleness
  sets `should_block = True`, which **halts trading** (`features/health.py:125-134`).
- A monthly refresh cannot satisfy a 7-day fuse — trading would block roughly 23 days in 30.
- The only reason it never fires: `FeatureHealth.last_success_ts` is
  `field(default_factory=time.time)`, and `FeatureHealthTracker()` is constructed per
  `TradingPipeline` — once per process start (`execution/pipeline.py:86`). **Every trader restart
  resets the clock.** Frequent deploys have been masking a real condition.
- Fix alongside G1-2 so the fuse matches the new cadence. Consider a durable clock rather than an
  in-memory one, since an in-memory fuse is defeated by the thing that happens most often.

### G1-4 🟡 32 equity features are constant zero — **DOWNGRADED from 🔴 by Q-A**

- **The empty `fundamentals` table is a red herring — no production code reads it.**
  `grep "FROM fundamentals" src/` returns zero hits. The values come from a hardcoded stub:
  `features/pipeline.py:173-175` writes literal `0.0` into all four columns.
- **Verified across the full history:** all 21,088 `features_equity` rows, 8 symbols,
  2016-01-03 → 2026-07-23, have `COUNT(DISTINCT) = 1` and `min = max = 0` on `pe_zscore`,
  `earnings_growth`, `debt_to_equity` and `dividend_yield`. Not one non-zero value in any year.
- **Training saw the same zeros** — `training/data_loader.py:163` reads the same columns; the stub
  landed `81243cc` (2026-03-06), the active equity models trained 2026-04-02.
- **Verdict: 32 dead constant features (~19.5% of the 164-dim observation), NOT train/serve skew.**
  Wasted capacity, not degraded decisions. Populating them for real *would* be a distribution
  change and **requires a retrain** → Stage 2.R, not a feeds fix.
- `weekly_fundamentals_job` (`scheduler/jobs.py:563`) is a **misnomer** — it instantiates
  `FREDIngestor` and runs macro ingestion. It has never touched fundamentals. See A-6b.

### G1-5 ⚪ `corporate_actions` is empty (0 rows) — **ADDRESSED by the 2026-07-25 plan**

- Referenced only by `data/corporate_actions.py`, which only *detects* via an overnight-spike
  heuristic — nothing ever ingested into it.
- **Closed by Task 8** of the re-ingestion plan: Alpaca's corporate-actions API returns
  **334 cash dividends** across all 8 symbols, 2016-03-15 → 2026-06-22, verified live.
- **Caveat: Alpaca returns zero spin-offs**, so this does not fix A-7.

---

## Findings added 2026-07-25 (value audit)

Found by checking value *distributions* rather than row recency. All measured against the live DB.

### A-1 🔴 The HMM served a 4-month-stale regime, and reported it as healthy

- `_get_hmm_probs` (`features/pipeline.py:459-476`) runs
  `WHERE environment = %s AND date <= %s ORDER BY date DESC LIMIT 1` — **no max-age bound at all.**
- `hmm_state_history` holds **14 rows total**, with a gap from **2026-03-10 to 2026-07-17**. Every
  equity and crypto cycle in those four months served the 2026-03-10 probabilities.
- **The fuse is actively defeated:** line 475 calls `record_success("hmm")` on the stale read, so
  `FeatureHealthTracker` was told the feed was fine.
- **Still intermittent** — equity has no row for 2026-07-21 even in the recent daily run.
- HMM rows are written only as a side effect of feature computation, gated on
  `symbol == hmm_proxy_symbol and len(ohlcv) >= 100` (`pipeline.py:183-192`). A dedicated job would
  be more robust.
- Live state: equity `p_bull` is **exactly `0.000000e+00`** from 2026-07-22 onward.
- **This is the last known defect actively feeding wrong live inputs. Do it next.**

### A-2 🔴 18 NYSE sessions missing; 26.7 days of 4H bars missing → *specced*

2026-03-11 → 2026-04-06 on all 8 equity symbols (2,636 held vs 2,654 XNYS sessions), and
2026-03-10 20:00 → 2026-04-06 12:00 on both crypto symbols. Same migration-window outage, and
immediately before the FRED freeze (04-09). **Root cause: `_resolve_start` (`alpaca.py:218-240`)
resumes from `max + 1 day` and cannot see interior gaps. Check `data/fred.py` for the same pattern
— it would explain G1-1.**

### A-3 🔴 The equity price series is a two-feed patchwork → *specced*

Average volume drops **13×–47× across all 8 symbols** around 2024-01-09 (SPY 53,562,323 → 1,132,544);
QQQ additionally jumps 471.25 → 401.38. **Cause: `alpaca.py:105-107` selects `DataFeed.IEX` for
incremental and `DataFeed.SIP` for backfill.** Contamination is patchier than one date — SPY
2023-06-15 already reads 1,762,275. **It arrives nightly via `candles_equity_job` at 20:15 ET.**

Also found: stored prices are **adjusted with a frozen factor** (our SPY 2016-06-15 close 177.66 vs
Alpaca's adjusted value today 176.83), so the stored series drifts from its own source and is not
reproducible. Resolved by LD-4 — store raw, never adjust.

### A-4 🟠 Fabricated crypto bars in the training set

After the 2019-08-31 → 2019-09-23 outage, 7 consecutive identical bars per symbol at exactly
9930.13 (BTC) / 209.55 (ETH), volume 0, `source` NULL. **Deferred by LD-6** — this gap straddles
`STITCH_DATE = 2019-09-01` (`binance.py:51`), the Binance.US/Global seam. Training-only data.

### A-5 🟠 The HMM's crisis state has collapsed

`p_crisis` max is **1.7e-03** across all history in both environments — the three-state model is
effectively two-state. Mitigating: only `p_bull`/`p_bear` are served (`pipeline.py:464`), so this is
a training-side defect. Fixing it changes observation dimensions → **retrain** → Stage 2.R.

### A-6 🟡 Three bad SPY bars, and the path that let them in

2018-11-01 is flat (`o=h=l=c=242.68`, volume 200); 2024-01-02 and 2024-01-04 violate
`low ≤ min(open, close)`. **`DataValidator` would have caught the 2024 pair** — with
`_TOLERANCE = 0.0001`, open 469.63 < low−tol 470.11. So they **bypassed validation entirely**,
most likely via the Postgres migration. The flat bar is a genuine rule gap (Step 7 only catches
volume *exactly* zero). Both addressed by the re-ingestion plan (Tasks 5 and 9).

### A-6b 🟡 `weekly_fundamentals_job` is a misnomer

`scheduler/jobs.py:563-585` instantiates `FREDIngestor` and calls `run_all()` — it runs **macro**
ingestion weekly under a fundamentals name, silently duplicating `monthly_macro_job`. Anyone reading
the scheduler would reasonably believe fundamentals are refreshed. Relevant to G1-2.

### A-7 🟡 XLF 2016-09-19 spin-off is unadjusted and **has no available source**

−18.3% (19.89 → 16.26) on full volume — the real XLRE spin-off. Undetected because the detector
threshold is 30% and `corporate_actions` is empty. **Alpaca's corporate-actions API returns zero
spin-off records**, so this cannot be fixed from there. Under LD-4 nothing is adjusted anyway, so
this is a documentation gap rather than a correctness regression — but it stays **open**.

### A-8 🟠 Crypto `rsi_14` and `four_h_rsi_14` are the same number, stored twice

**Byte-identical across all 52,233 rows**, both symbols, `max_abs_diff = 0`. Cause: crypto's base
timeframe *is* 4H, so `technical.py:163-164` recomputes via `stockstats` exactly what
`compute_price_action` already produced. **2 of 26 per-asset crypto slots (~7.7%) carry zero
incremental information.** Contrast: the SMA pair genuinely differs (722 coincidental matches,
max diff 2.94). Natural replacement is drawdown-from-rolling-high → **retrain** → Stage 2.R.

### A-9 🟠 Model provenance is unrecorded

`models` / `model_metadata` / `training_runs` carry `code_version = 'unknown_era0'`,
`data_fingerprint = 'unknown_era0'`, and NULL `training_window_start`/`end`. **You cannot currently
prove what data any deployed model was trained on** — Q-A had to reconstruct it from git history.
Matters now that a retrain is on the path.

### A-10 ⚪ Two more declared-but-never-written columns

`ohlcv_daily.adjusted_close` is NULL on all 21,088 rows (same class as G4-5's
`portfolio_snapshots.equity_value`/`.crypto_value`). Moot under LD-4. Separately, the turbulence
trip reason hardcodes `_exceeds_90th_pct` (`risk_manager.py:289`) while the actual gate is the
**97th** percentile (`config/swingrl.yaml:344`) — every trip in the audit log is mislabelled.

### Also confirmed healthy by value audit

All 13 crypto features carry real varying values — 37,819 distinct each, std 1.00–1.34, **zero**
exact-zeros, 1.4% NULL (warmup only). All 11 real equity features likewise. OHLC invariants hold on
21,086 of 21,088 equity bars and all 38,537 crypto bars. HMM probabilities sum to 1.0 exactly with
finite log-likelihoods.

---

## Group 2 — Transaction accuracy (the carve-out ledger)

**Files:** `execution/pipeline.py` · `execution/fill_processor.py` · `execution/reconciliation.py` ·
`execution/adapters/alpaca_adapter.py` · `scheduler/jobs.py` (confirmation + reconciliation)

**Goal:** every `trades` row carries the broker's **actual** quantity, price and fees.
Portfolio value is **out of scope** — see the binding constraint above.

**This group is equity-only, and that is deliberate — see LD-1.** Crypto fills are simulated
locally because Binance.US has no paper trading, so there is no broker to reconcile against. Do not
scope crypto reconciliation into this work, and do not treat its absence as a defect.

### Already built — do not rebuild

- `PositionReconciler` (`execution/reconciliation.py`; INSERT :164, UPDATE :212, DELETE :252).
- `reconciliation_job()` (`scheduler/jobs.py:702`), scheduled **daily 17:00 ET**
  (`scripts/main.py:272`, id `daily_reconciliation`), **equity-only** — crypto uses a virtual
  balance and Binance.US has no paper trading, so there is no external authority to compare against.
- `fill_processor.record_reconciliation_adjustment` (`execution/fill_processor.py:124`).
- **Equity quantity and price are already broker-sourced** at the ~09:35 confirmation job: it reads
  `filled_qty` and `filled_avg_price`, and refuses to stamp when the amounts are unusable, alerting
  instead (`scheduler/jobs.py:900-935`). This path is careful — extend it, do not replace it.

### G2-1 🟠 Fees are not captured for equity, and the first sell will silently drift the ledger

- `trades.commission`: crypto **9/9 populated** (total $0.1524); equity **0/10, all zero**.
- All 10 equity trades are buys, and Alpaca equity buys are commission-free — so zero is legitimate
  *so far*. The first sell incurs SEC and FINRA TAF regulatory fees, and **no code path captures
  them.**

### G2-2 🟡 `trades.slippage` is never populated for equity

- Equity 0/10, yet `fill_quality.slippage_frac` **is** computed and stored (equity 8/10, crypto 9/9).
- Crypto mirrors it into `trades`; equity does not. Back-fill and wire it forward.

### G2-3 🟡 Two of ten equity trades have no `fill_quality` row

- Coverage: equity 8/10, crypto 9/9. Two trades entered through a path that skips the sidecar.
  There are 10 equity trades but only 8 `pending_orders` rows, so the two extras likely came from
  the reset or a manual path. Identify which.

### G2-4 🟠 The Alpaca adapter has no account-activities method — this blocks G2-1

- Public surface is `submit_order`, `get_order_status`, `get_clock`, `get_positions`,
  `cancel_order`, `get_current_price` (`execution/adapters/alpaca_adapter.py`).
- Fees are **not on the order object**. Alpaca reports regulatory fees as separate
  account-activity records that post after the fill, so fee capture needs a **new adapter method**.
- **UNVERIFIED:** the exact account-activities API for the pinned `alpaca-py>=0.20,<0.44`.

### G2-5 🟡 `realized_pnl` is computed and then discarded

- Computed at `execution/fill_processor.py:440-442`, returned via `fill_processor.py:102`, carried on
  `FillResult` (`execution/types.py:76`) — and **never persisted**. `trades` has no such column.
- The dashboard design forbids *synthesizing* per-trade P&L. This is not synthesis; it is a real
  computed value being thrown away. Adding `trades.realized_pnl` unlocks the column honestly.

### G2-6 🟠 Reconciliation timing — reviewed 2026-07-24, recommendation below

- **~09:35 ET (existing confirmation job): quantity + price.** Already broker-sourced. This is the
  moment the operator can still act, so mismatches belong here.
- **17:00 ET (existing daily sweep): fees, plus a re-verify of the day's fills.** Fees cannot be read
  at 09:35 — they are not on the order object and post after the fill (G2-4). Extend
  `reconciliation_job` to pull the day's activities, attach fees to the matching `trades` rows, and
  back-fill `trades.slippage` from `fill_quality` (G2-2).
- Keep the daily sweep as the backstop for late broker corrections, cancels and amendments.
- **OPEN USER DECISION** (was dashboard spec U-5, now re-scoped): on a broker/DB mismatch in
  quantity or price — alert only, or alert **and** correct the stored `trades` row? Correcting
  rewrites recorded history; not correcting leaves a known-wrong number in the carve-out ledger.

---

## Group 3 — Risk controls

**Files:** `execution/pipeline.py` · `execution/position_sizer.py` · `scheduler/stop_polling.py` ·
`execution/adapters/alpaca_adapter.py` · `execution/risk/circuit_breaker.py`

### G3-1 🔴 No stop-loss or take-profit is ever set, so the crypto stop-monitor is a no-op

- `execution/pipeline.py:416` hardcodes `stop_loss_price=None` and `take_profit_price=None` on every
  `SizedOrder`; `scheduler/jobs.py:868` does the same.
- All 10 live `positions` rows have NULL stop and target.
- Consequence: `scheduler/stop_polling.py` wakes every 60 s, selects both columns, and returns
  immediately at `stop_polling.py:146` (`if stop_loss is None and take_profit is None: return`).
  The crypto stop-loss monitor has never checked anything.
- `execution/position_sizer.py:141` **can** compute a stop; nothing consumes it.
- **Establish intent before "fixing".** `execution/adapters/alpaca_adapter.py:110` only submits a
  bracket order when **both** are non-None, and equity fills through the opening auction — the
  `None` may be deliberate for equity and an oversight for crypto.
- **Severity nuance:** `stop_polling.py`'s own docstring says a breach is only recorded and alerted,
  never auto-sold. So this is a lost *alert*, not lost protection — the circuit breakers are the
  real downside control, and they work (one fired 2026-07-24).
- Blocks G5-2 and the dashboard's Holdings stop/target columns.

---

## Group 4 — Observability (the backend work the dashboard needs)

**Files:** `data/postgres_schema.py` + a new additive migration · `monitoring/alerter.py` ·
`execution/cycle_recorder.py` · `execution/risk/position_tracker.py` · `scheduler/jobs.py`

Every item here is why a dashboard panel had to be watered down. Fixing them upstream is what lets
the dashboard match its approved design instead of apologising for the schema.

### G4-1 🟡 `system_events` has zero writers

- The table exists (`event_id, timestamp, level, module, event_type, message, metadata_json`), is
  **empty**, and the only reference anywhere in the codebase is the migration script's table list.
- The design's **Services grid** wants real health. Without this the dashboard can only *infer*
  liveness from the age of the newest `inference_cycles` row, and can honestly say no more than
  "last seen 4 m ago" — never "healthy".
- Fix: have the trader and collector emit startup / shutdown / heartbeat rows here.

### G4-2 🟡 `api_errors` has exactly one writer, and it is crypto-only

- Only writer: `execution/adapters/binance_sim.py:345`. Read by `execution/emergency.py:456` for the
  Binance HTTP-418 emergency trigger.
- **Alpaca / equity API errors are never recorded**, so the design's broker-health row can never
  show an equity problem.
- Fix: record Alpaca errors from the adapter's `_retry` path.

### G4-3 🟡 `alert_log` has no message body

- Columns: `alert_id, timestamp, level, title, message_hash, sent`. `Alerter._compute_hash` is
  SHA-256 of `f"{title}:{message}"` (`monitoring/alerter.py:400-411`) — irreversible, so the body is
  unrecoverable. Single writer: `alerter.py:375-391`. `sent` is INTEGER 0/1, not boolean.
- 109 rows: info 71 / critical 32 / warning 6.
- The dashboard's Event feed can therefore show only time, level and title, where the design wanted a
  log stream with detail text.
- Deliverable: decide whether to add a `message` column, **and** audit what is worth capturing as an
  alert at all. Same work as G5-5 — do it once.

### G4-4 🟡 `pending_orders` lacks the columns its panel needs

- The dashboard's pending-orders panel was **removed** for this reason (user decision 2026-07-24,
  reversing spec decision D-11). Restoring it needs an additive migration plus a writer change at
  `execution/pipeline.py:676-682`.
- Current columns: `order_id, cycle_id, symbol, side, submitted_at, resolved_at, created_at,
  decision_price, disposition`.
- Add — all available in scope at the insert, **UNVERIFIED** line-by-line:

  | Column | Why | Source at insert |
  |---|---|---|
  | `quantity` | the panel's headline column | `fill.quantity` |
  | `notional_usd` | orders submit in dollars, so this is the truer intent | `sized_order.dollar_amount` |
  | `filled_quantity` | partial-fill visibility — a half-filled order is a signal | updated by the 09:35 job |
  | `environment` | avoids a nullable-`cycle_id` join | `env_name` in scope |
  | `order_type` | design column | order request |
  | `limit_price` | design column; NULL for market/notional orders | order request |
  | `broker` | consistency with `trades` | adapter |
  | `last_broker_status` | show `accepted`/`new`/`partially_filled`, not just resolved-or-not | `get_order_status` |
  | `last_checked_at` | separates "stuck" from "never polled" — would have made G4-7 obvious | 09:35 job |

- Crypto legitimately has no pending state (fills are simulated and immediate), so the equity-only
  gate at `execution/pipeline.py:472` is correct. The panel should **say** "equity only" rather than
  look broken.

### G4-5 🟡 `portfolio_snapshots.equity_value` and `.crypto_value` are never written

- Both columns exist in the DDL; **0 of 39 live rows are populated.** `record_snapshot()`
  (`execution/risk/position_tracker.py:289-323`) does not write them.
- Consequence: any "combined portfolio value" must sum two per-environment rows carrying *different*
  timestamps, so it has no single as-of instant. Related known quirk: the crypto total drifts
  $46.96–$48.22.

### G4-6 🟡 Cycle capture is fail-open with no durable failure record

- `CycleRecorder._alert_capture_failed` (`execution/cycle_recorder.py:294-306`) swallows the
  exception and returns `None`, so a cycle can run without leaving a row.
- The dashboard must therefore hedge and say "last **recorded** cycle"; it cannot distinguish "no
  cycle ran" from "a cycle ran but capture failed". Persisting the failure (G4-1 is the natural home)
  removes the hedge.

### G4-7 🟡 `pending_orders.resolved_at` lagged a real fill by 24 hours

- The 2026-07-23 09:15 batch of 8 equity orders filled at 09:35 that same morning — `trades` records
  SPY at $739.25, 2026-07-23 09:35:00 — but was not stamped until **2026-07-24 09:35**. For a full
  day the table asserted 8 orders were still working when all 8 had filled.
- Likely fixed by PR #40 ("auction-fill status-fix") — **the PR was not read, 50% confidence.**
  See Q-C.
- Defensive note for whoever rebuilds the panel: `order_id` equals `trades.trade_id`, so a LEFT JOIN
  to `trades` distinguishes "genuinely working" from "filled but not stamped". Report the latter as a
  data-integrity warning, not as a stuck order.

---

## Group 5 — Discord notifications

**Files:** `monitoring/embeds.py` · `monitoring/alerter.py` · `scheduler/jobs.py`

**Pattern:** three of these are not "wire up Discord". The embed builders already accept the data and
the **callers omit it**. Cheap fixes.

### G5-1 🟡 Circuit-breaker status missing from the daily digest

- `build_daily_summary_embed` **already accepts** `cb_status: dict[str, str] | None = None`
  (`monitoring/embeds.py:151`). The caller at `scheduler/jobs.py:494` **omits it**.
- Fix = one keyword argument plus a breaker query. **Use a read-only SQL derivation, never
  `CircuitBreaker.get_state()`** — that method writes.

### G5-2 🟡 Trade alerts never pass stop / take-profit

- `build_trade_embed` **already accepts** `stop_price` / `take_profit` (`monitoring/embeds.py:63`,
  documented at `embeds.py:9`). All three callers omit them: `scheduler/jobs.py:187`, `:241`, `:885`.
- **Blocked by G3-1** — the values do not exist to pass. Fix the source first or this is cosmetic.

### G5-3 ⚪ `build_circuit_breaker_embed` is dead code — confirmed

- `monitoring/embeds.py:321`; zero production callers (only `tests/monitoring/test_embeds.py`).
- Decide: wire it to the breaker trip path, or delete it. Do not leave it.

### G5-4 ⚪ `build_stuck_agent_embed` is dead code — confirmed

- `monitoring/embeds.py:262`; zero production callers. `stuck_agent_check_job` exists and is
  scheduled but does not use the embed.

### G5-5 🟡 Audit what is captured as an alert

- Same work as G4-3 — do it once. Current mix over 109 rows: info 71 / critical 32 / warning 6.
- Known noise: "Options decision MISSED" alerts on or before 2026-07-23 are ignorable; 2026-07-24
  onward are real.

---

## Dashboard panel → backend blocker map

Use this when deciding how much of Group 4 to do. Each row is a panel that is currently watered
down, and the single backend item that would restore it to the approved design.

| Dashboard panel | Currently | Blocker | Restores |
|---|---|---|---|
| Pending orders (Dashboard + Trade Log) | **removed entirely** | **G4-4** (+ G4-7) | The whole panel as drawn |
| System Health → Services grid | inferred "last seen 4 m ago" | **G4-1** | Real service health |
| System Health → broker row | can never show an equity problem | **G4-2** | Broker health |
| System Health → Event feed | time + level + title only | **G4-3** | Log stream with detail text |
| System Health → "last recorded cycle" | hedged wording | **G4-6** | "no cycle" vs "capture failed" |
| Portfolio → Holdings stop / target | always `—` | **G3-1** | Real stop and target columns |
| Portfolio → combined value | sums 2 rows, 2 timestamps | **G4-5** | One authoritative as-of |
| Trade Log → per-trade P&L | absent | **G2-5** | A recorded, non-synthesized P&L column |
| Trade Log → slippage (equity) | blank | **G2-2** | Populated slippage |
| Trade Log → fill-quality drill-in | 8/10 equity coverage | **G2-3** | Full coverage |
| System Health → data feeds | macro reads 5 days, data is 106 days stale | **G1-1** | A feed panel that tells the truth |

---

## Open questions — answer these first, they resize the work

**Q-A, Q-B and Q-C were all ANSWERED on 2026-07-25.** Q-D remains open and blocks Group 2.

| # | Question | Answer |
|---|---|---|
| **Q-A** | Fundamentals: zeros or NaN, and was the model trained with real values? | **Zeros, and trained on zeros too.** `pipeline.py:173-175` writes literal `0.0`; the table is never read. **32 dead constant features, NOT train/serve skew.** G1-4 downgraded 🔴→🟡 and moved to Stage 2.R |
| **Q-B** | Does turbulence read the macro block? | **VERIFIED NO.** Its sole input is 8 equity daily log-returns from `ohlcv_daily` (`pipeline.py:508-561`); macro is a sibling array, never an input. **The 2026-07-24 halt is GENUINE** — reproduced bit-exactly from price data alone (5.988613162880 vs recorded 5.988613162880422, threshold 4.988766759763). Caused by broad market −1.3% against XLI +1.75% on 2026-07-23. Auto-resume is lazy, due ~2026-07-31. Threshold is the **97th** percentile, not the 90th the label claims (A-10) |
| **Q-C** | Did PR #40 fix the `resolved_at` stamp lag? | **YES — G4-7 is CLOSED.** PR #40 (merged 2026-07-23) fixed `str(OrderStatus.FILLED).lower()` never matching `"filled"`, which left 8 filled orders stuck open. Self-heals on the next 09:35 run |
| **Q-D** | On a broker/DB mismatch: alert only, or correct the stored `trades` row? | **STILL OPEN — blocks all of Group 2.** Q-D1 (quantity) and Q-D2 (fees) have safe defaults; **Q-D3 (price) genuinely needs a ruling** |

---

## Confirmed healthy — do not re-audit

`ohlcv_daily` 2026-07-24 · `ohlcv_4h` 2026-07-24 16:00 · `features_crypto` 2026-07-24 16:00 ·
`features_equity` 2026-07-23 (**correct** — equity decides on t-1 bars by design) ·
`hmm_state_history` 2026-07-24 · `options_chains` 2026-07-24 (918 k rows) ·
`options_snapshots` 2026-07-24 · `calendar_events` populated to 2027-12-08 ·
`emergency_flags` 0 rows (nothing manually halted) ·
no code reads broker account value anywhere in `execution/`.

---

## Dashboard-side gaps — not trader work

Found while self-reviewing the tabled dashboard plan. Fix in that plan when it resumes; recorded
here only so they are not lost.

- **D-1:** The design's **Range 7D / 30D / ALL** control has no task. `snapshot_series(since=...)`
  supports it and the design header has the segmented control, but nothing builds it.
- **D-2:** The **accent swatches** are missing from the shell. Plan Task 13 builds theme and density
  toggles; the design has four accent colours and `read_prefs` already handles `accent`.

---

## Suggested sequence

1. **Q-A, Q-B, Q-C** — read-only, cheap, and they resize everything downstream.
2. **Chunk 1 · Group 1** (G1-1, G1-2, G1-3 together) — the only P1s reaching live decisions, and
   G1-2 is an explicit user requirement. One coherent change: move macro to the collector on a
   4-hourly cadence, fix the freeze, reconcile the fuse.
3. **Chunk 1 · Group 2** (G2-4 → G2-1 → G2-6, then G2-2 / G2-3 / G2-5) — the adapter method comes
   first because fee capture depends on it.
4. **Chunk 2 · G3-1** — establish intent, then either set stops or document explicitly why not.
   Unblocks G5-2.
5. **Chunk 3 · Group 4** — pick from the panel → blocker map according to how much dashboard
   fidelity is wanted. G4-1 and G4-4 give the most back per unit of work.
6. **Chunk 4 · Group 5** — G5-1 is a one-line fix; G5-2 follows G3-1; G5-3 / G5-4 are
   delete-or-wire decisions.
7. **Chunk 5** — resume the dashboard plan, restoring the pending-orders panel and D-1 / D-2.

---

## Standing rules that constrain any fix here

- **The $400 / $47.09 carve-out is inviolable.** Never read broker account value, equity or buying
  power anywhere in the trader.
- **Never branch from `main`** — it has zero migration files; the integration branch is ~297 commits
  ahead and holds all eleven (V001–V011). Branch from `swingrl/2.R-training-redesign`; PRs target it.
- **Migrations are additive-only while the trader runs** (A30). New columns, never destructive DDL.
- **Plan-first** — no file created, edited or deleted without plan mode and explicit approval.
- **A30 deploy isolation** — service-scoped builds only; never a bare `docker compose build`. Quiet
  window: no recreations or CI spanning 15:30–16:45 ET on trading days.
- **Never call `CircuitBreaker.get_state()` or `get_capacity_fraction()` from a read path** — both
  write (auto-resume plus a Discord alert when cooldown completes). Re-derive from SQL, and always
  exclude audit rows: `COALESCE(reason, '') NOT LIKE 'stop-breach-audit:%'`.
- **Never `--no-verify`.** detect-secrets false positives go in `.secrets.baseline`.
- UTC internally; ET only at the display edge.
- TDD — commit the RED test, then the GREEN implementation.
