# Macro & Regime Feed Integrity — Design Spec

> **Status: DRAFT — awaiting user review.**
> **Branch:** `swingrl/26-equity-data-reingestion` (sibling spec; may split to its own branch).
> **Closes:** G1-1, G1-2, G1-3, A-1, A-11, A-12, A-14.
> **Does not close:** A-5, G1-4, A-9 — all retrain-gated, see *Out of scope*.
> **Sibling spec:** `2026-07-25-equity-data-reingestion-design.md` — shares the root cause (LD-8).
> **Register:** `docs/superpowers/reviews/2026-07-24-trader-collector-audit.md`
> **Evidence:** measured against the live system 2026-07-25. Anything inferred is marked
> **UNVERIFIED**.

---

## Glossary

| Term | Meaning |
|---|---|
| **FRED** | The St. Louis Fed's public economic data service. Source of VIX, rates, CPI, unemployment |
| **ALFRED** | FRED's archive. Records not just a value but **when that value was first published** |
| **Observation date** | The date a number *describes* — "CPI for June 2026" |
| **Release date** | The date that number was actually *published* — CPI for June 2026 appears in mid-July |
| **Vintage** | One published version of an observation. Figures get revised, so one observation date has many vintages |
| **First release** | The **initial** publication. What you would genuinely have known at the time |
| **Lookahead bias** | Training a model on information that did not exist yet. Makes backtests look better than reality |
| **Point-in-time (PIT)** | Reconstructing exactly what was knowable on a given date. The opposite of lookahead |
| **Watermark** | The "last thing I loaded" marker an incremental job uses to resume |
| **Interior gap** | Missing data *behind* the watermark. A `max + 1 day` watermark can never see one |
| **Staleness fuse** | A rule that halts trading when an input has not refreshed within a window |
| **Fail-open** | A check that swallows its own errors, so a missing signal does not prove the event did not happen |
| **A30** | Deploy-isolation rule: no trader rebuild or restart outside a market-safe window |

---

## Problem

### G1-1 is misdiagnosed in the register. The data is not frozen — it never reached Postgres.

| Series | Parquet | Postgres | Missing |
|---|---|---|---|
| VIXCLS | 2,682 rows → **2026-07-20** | 2,604 → 2026-04-09 | **78 rows** |
| DFF | 3,854 rows → **2026-07-20** | 3,745 → 2026-04-09 | 109 rows |
| T10Y2Y | 2,637 rows → **2026-07-21** | 2,563 → 2026-04-10 | 74 rows |
| CPIAUCSL | 953 → 2026-06-01 | 953 → 2026-06-01 | ✅ none |
| UNRATE | 941 → 2026-06-01 | 941 → 2026-06-01 | ✅ none |

**Mechanism:** `_resolve_start_date` (`fred.py:102-108`) reads the watermark from **Parquet**
(`max + 1 day`) while the trader reads **Postgres**. The stores diverged in March 2026. The
watermark now says "up to date through 07-20", so the incremental fetch requests nothing and logs
`no_data` — and Postgres can never catch up. **The missing rows are scattered, not contiguous**
(VIXCLS: 78 rows spread across 2026-03-20 → 2026-07-20), so the repair must be diff-based.

**This is the same root cause as A-2 in the sibling spec**, wearing a different hat: there the
watermark was blind to a calendar gap; here it is blind to a store divergence. Both are "the
watermark is not read from the store the consumer reads."

**Consequence reaching live decisions:** `features/pipeline.py:443` reads `VIXCLS` and every crypto
cycle on 2026-07-24 recorded `vix = 19.49` — the 2026-04-09 value. The emergency auto-stop
"VIX > 40 **and** 24 h drawdown ≥ 13%" can never fire.

### A-14 — the health tracker treats "a query returned rows" as "the feed is healthy"

- `_get_macro_array` calls `record_success("macro")` (`pipeline.py:450`) after any query returning
  rows, regardless of age — including data 102 days old.
- `_get_hmm_probs` does the same at `pipeline.py:475`.
- So the 7-day macro staleness fuse (`health.py:81`, blocking at `:132`) **cannot fire**. It is fed
  a success every cycle.
- **This is one bug class across two feeds.** Fix it once.

### A-1 — the HMM served a 4-month-stale regime

- `_get_hmm_probs` (`pipeline.py:459-476`) runs `WHERE date <= %s ORDER BY date DESC LIMIT 1` —
  **no max-age bound at all.**
- `hmm_state_history` holds **14 rows**, with a gap from 2026-03-10 to 2026-07-17.
- Writes happen only as a side effect of `compute_equity`, gated on
  `symbol == hmm_proxy_symbol and len(ohlcv) >= 100` (`pipeline.py:181`), with failures swallowed by
  `except (ValueError, RuntimeError)` (`pipeline.py:196`).
- **Still intermittent** — equity has no row for 2026-07-21.
- Live: equity `p_bull` is exactly `0.000000e+00` from 2026-07-22.

### G1-3 — the staleness fuse is defeated twice over

`FeatureHealth.last_success_ts` defaults to `time.time()` and the tracker is constructed per
`TradingPipeline` — once per process start — so **every restart resets the clock**. Even with a
durable clock the fuse would still never fire, because of A-14. Both must be fixed together.

### A-11 / A-12 — release dates are fetched, then truncated away

- `_MACRO_COLUMNS = ["date", "series_id", "value"]` (`base.py:32`) and `_build_sync_df` returns
  `sync_df[_MACRO_COLUMNS]` (`base.py:226`) — a hard slice. `vintage_date` is produced by
  `_fetch_vintage`, written to Parquet, then **silently dropped going into Postgres**.
  Result: `release_date` is NULL on all **10,806** rows.
- Both read paths therefore key on **observation date**: `pipeline.py:422` and
  `training/data_loader.py:48`. Serving and training agree, so there is **no train/serve skew** —
  they are wrong in the same direction, which is why nothing flagged it.
- **The stored `vintage_date` is not usable as a release date.** `_fetch_vintage` does
  `sort_values("vintage_date").groupby(level=0).last()` (`fred.py:157`) — the **latest revision**,
  not the first release. Combined with a truncated realtime window, UNRATE's 1948 observation
  carries a "vintage" of 2026-06-02.

**Measured against ALFRED's full realtime window:** CPIAUCSL vintages begin **1972-07-21**;
for the 646 observations after that, the true first-release lag is a **median of 46 days**, max 62.
Requesting the full realtime window for a daily series fails — *"3,897 vintage dates … exceeds the
maximum number of vintage dates allowed for this file type (2000)"* — so daily series need chunking.

### G1-2 — macro cadence *(standing user requirement)*

Today: `monthly_macro_job`, cron `day=1, hour=18` ET (`scripts/main.py:192`), run by the **trader**.
Wanted: refreshed by the **collector every 4 hours**, mirroring `candles_crypto_job`
(`collector_main.py:502-509`, cron `hour="0,4,8,12,16,20"`).

---

## Locked decisions

| # | Decision | Reason |
|---|---|---|
| **LD-8** | **The watermark reads from Postgres — the store the consumer reads.** Parquet becomes a non-authoritative archive | Makes the G1-1 / A-2 divergence structurally impossible rather than merely detectable |
| **LD-9** | **True first-release vintages for all five series**, chunking around FRED's 2,000-vintage cap | User decision 2026-07-25. Chosen over the cheaper daily-series shortcut, having been told it buys ~1 day of accuracy on three of the five |
| **LD-10** | **`record_success` must mean "fresh data arrived", not "a query returned rows"** | Fixes the macro fuse and A-1's HMM in one place |
| **LD-11** | **Macro moves to the collector on a 4-hourly cadence**; `monthly_macro_job` is retired | Standing user requirement (G1-2) |
| **LD-12** | Observations older than ALFRED's vintage coverage get **NULL** `release_date` | An honest unknown beats a fabricated date. Never synthesise a release date |
| **LD-13** | Use **`.first()`**, not `.last()`, when collapsing vintages | The first release is what was knowable at the time; the latest revision is itself lookahead |

---

## Design

**One store of record. One meaning of "healthy". Release dates that reflect publication.**

```
FRED / ALFRED ──► FredIngestor ──► macro_features (date, series_id, value, release_date)
                       ▲                          │
                       │                          ▼
              watermark read from ────────── Postgres  ──► _get_macro_array ──► observation
              Postgres, gap-aware                            │
                                                             ▼
                                             record_success ONLY if the row
                                             is within the freshness window
```

### Components

**Modified**

| Component | Change |
|---|---|
| `data/base.py:32` | `_MACRO_COLUMNS` gains `release_date`; `_build_sync_df` maps `vintage_date` → `release_date` (`:222-226`) |
| `data/fred.py:102-108` | Watermark from Postgres, not Parquet (**LD-8**), with interior-gap detection |
| `data/fred.py:131-159` | `_fetch_vintage` uses `.first()` (**LD-13**) and chunks the realtime window to stay under 2,000 vintages (**LD-9**). All five series become vintage series |
| `features/pipeline.py:450` | `record_success("macro")` only when the newest row is inside the freshness window (**LD-10**) |
| `features/pipeline.py:459-476` | `_get_hmm_probs` gains a max-age bound; stale reads call `record_failure` (**A-1**) |
| `features/health.py:81` | `STALENESS_SECONDS` matched to the new 4-hourly cadence, not 7 days |
| `scripts/main.py:192` | `monthly_macro_job` retired (**LD-11**) |
| `scripts/collector_main.py` | New `macro_job`, cron `hour="0,4,8,12,16,20"`, mirroring `candles_crypto_job` |

**New**

| Component | Responsibility |
|---|---|
| `scripts/backfill_macro_release_dates.py` | One-off: chunked ALFRED pull for all five series, first-release semantics, NULL before coverage |
| Migration **V012** (additive) | Index on `(series_id, date)` if the diff-based re-sync needs it. `release_date` already exists — **no DDL required for it** |

### Repair sequence

1. `pg_dump macro_features` and `hmm_state_history`
2. **Diff-based re-sync** Parquet → Postgres for the 261 scattered missing rows
3. Chunked ALFRED backfill to populate `release_date` (LD-9, LD-12, LD-13)
4. Repoint the watermark to Postgres (LD-8)
5. Fix `record_success` semantics (LD-10) and the HMM max-age bound
6. Move macro to the collector at 4-hourly; retire `monthly_macro_job` (LD-11)
7. Backfill `hmm_state_history` for the gap, and give HMM its own job rather than a side effect

### Error handling

- A stale read is a **failure**, not a success (LD-10). That is the entire point.
- The HMM fit currently swallows `ValueError`/`RuntimeError` with a warning (`pipeline.py:196`).
  It must record a failure so the fuse can see it.
- Chunked ALFRED requests that fail mid-way must leave `release_date` NULL for the untouched range
  rather than a partial or guessed value (LD-12).
- Raising `STALENESS_SECONDS` is not a fix. If the fuse fires after LD-10, the feed is genuinely
  stale and trading **should** block.

### Testing

TDD: RED then GREEN.

- Watermark reads Postgres, and finds a hole behind the newest row
- `release_date` survives the sync (the regression test for `base.py:226`)
- Vintage collapse takes the **first** release, not the last — fixture with three vintages
- `record_success` is **not** called when the newest row is older than the window
- `_get_hmm_probs` returns neutral **and** records a failure past the max age
- Pre-coverage observations get NULL, never a synthesised date
- Fast lane, then `scripts/ci-homelab.sh` from `~/swingrl` before PR

---

## Out of scope

| Item | Why |
|---|---|
| **A-5** `p_crisis` collapsed | Changes observation dimensions → retrain → Stage 2.R |
| **G1-4** 32 dead fundamentals features | Retrain-gated → Stage 2.R |
| **A-9** model provenance (`'unknown_era0'`) | Its own item; matters before the next retrain |
| **A-8** crypto duplicate RSI feature | Retrain-gated → Stage 2.R |
| Retraining on corrected macro | Deliberate. This spec makes the data correct; using it is Stage 2.R |

---

## Risks

| Risk | Handling |
|---|---|
| Fixing A-14 may cause the fuse to fire and **halt trading** | That is correct behaviour if the feed is genuinely stale. Sequence the repair *before* the fuse fix so the data is fresh when the fuse goes live |
| Chunked ALFRED backfill is slow or rate-limited | It is a one-off, run off-hours. Partial completion leaves NULL, never a guess (LD-12) |
| Moving macro to the collector changes which process holds the FRED key | The collector already holds FRED credentials for `calendar_ingest` (`collector_main.py:464-474`) |
| Postgres-based watermark is slower than reading a local Parquet file | Five series, one indexed query each, four times a day. Immaterial |
| Backfilled `release_date` changes what training sees | Intended — it removes lookahead. No live impact; both read paths still key on `date` until Stage 2.R switches them |

## Rollback

`pg_dump` of `macro_features` and `hmm_state_history` taken at step 1. `release_date` is an existing
nullable column, so populating it is additive and reversible with `UPDATE ... SET release_date =
NULL`. No destructive DDL (A30). The trader is never stopped.

---

## Verification checklist

- [ ] `macro_features` Postgres row counts match Parquet for all five series
- [ ] VIXCLS max date in Postgres is current, not 2026-04-09
- [ ] A live crypto cycle records a `vix` that is **not** 19.49
- [ ] `release_date` is populated wherever ALFRED has coverage; NULL before it; **never fabricated**
- [ ] CPIAUCSL release dates show a ~46-day median lag against observation date
- [ ] A deliberately stale fixture causes `record_success` **not** to fire, and the fuse to block
- [ ] `_get_hmm_probs` past its max age returns neutral and records a failure
- [ ] `hmm_state_history` has a row per trading day, written by its own job
- [ ] `monthly_macro_job` is gone; the collector runs macro at `0,4,8,12,16,20`
- [ ] Fast lane green, then homelab CI `=== CI PASSED ===`
