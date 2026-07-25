# Equity Data Re-Ingestion — Design Spec

> **Status: DRAFT — awaiting user review.**
> **Branch:** `swingrl/26-equity-data-reingestion`, cut from `swingrl/2.R-training-redesign`.
> **Closes:** A-2, A-3, A-6, G1-5. **Enables (does not deliver):** `dividend_yield`.
> **Does not close:** A-1, A-4, A-5, A-7, G1-1, G1-2, G1-3 — see *Out of scope*.
> **Source register:** `docs/superpowers/reviews/2026-07-24-trader-collector-audit.md`
> **Evidence:** every number below was measured against the live database or a live API call on
> 2026-07-24/25. Anything inferred rather than measured is marked **UNVERIFIED**.

---

## Glossary — read this first

| Term | Meaning |
|---|---|
| **Bar** | One row of open/high/low/close/volume for one symbol for one period. A "daily bar" covers one trading session |
| **OHLCV** | Open, High, Low, Close, Volume — the five numbers that make up a bar |
| **Raw price** | The price actually traded that day, never restated afterwards |
| **Adjusted price** | A raw price rewritten backwards to account for dividends and splits, so a price chart shows total return. **Rewritten again every time a new dividend is paid** |
| **Consolidated tape / SIP** | The combined record of trades across all US exchanges. The full picture of volume |
| **IEX** | One single exchange, carrying roughly 2% of US volume. An IEX-only feed under-reports volume by 10–50× |
| **Corporate action** | A dividend, split, or spin-off — an event that changes a share's economics |
| **Spin-off** | A fund or company distributing part of itself as a separate security. Causes a genuine price drop that is not a loss |
| **T-1 / T-2** | "One / two trading days ago". The equity trader decides on the **T-1** bar by design |
| **Seam** | The join point where a data series switches from one source to another. Seams create artificial discontinuities |
| **Watermark** | The "last thing I loaded" marker an incremental job uses to know where to resume |
| **Interior gap** | Missing data *behind* the watermark. A `max + 1 day` watermark can never see one |
| **Upsert** | Insert a row, or overwrite it if it already exists |
| **`ON CONFLICT DO NOTHING`** | Insert a row, or **silently discard it** if it already exists. The opposite of upsert |
| **Additive migration** | A schema change that only adds. Required while the trader is running (rule A30) |
| **A30** | Deploy-isolation rule: no rebuild or restart of the trader outside a market-safe window |
| **Carve-out** | The notional capital the trader may use ($400 equity, $47.09 crypto) inside a $100,000 paper account |

---

## Problem

Four defects, all in the data the trader reads, all found by value-auditing tables that a
recency check had already cleared as healthy.

| ID | Defect | Measured evidence |
|---|---|---|
| **A-3** | The equity price series is a **patchwork of two different feeds**. Historical bars came from the consolidated tape; later bars came from IEX | SPY average volume 53,562,323 before 2024-01-09 vs **1,132,544** after — a 47× drop, mirrored across all 8 symbols (13–47×). QQQ additionally jumps 471.25 → 401.38 (−14.8%) at the boundary |
| **A-2** | **18 NYSE sessions missing**, 2026-03-11 → 2026-04-06, on every symbol | Compared against the XNYS calendar: 2,636 bars held vs 2,654 sessions |
| **A-6** | Three bad SPY bars | 2018-11-01 is flat (o=h=l=c=242.68, volume 200); 2024-01-02 and 2024-01-04 violate `low ≤ min(open, close)` |
| **G1-5** | `corporate_actions` is empty (0 rows) | Nothing ingests it — `data/corporate_actions.py` only *detects* via a price heuristic |

### Two root causes, both verified in code

1. **Feed switching is by design and undocumented in its consequences.** `alpaca.py:105-107`
   selects `DataFeed.IEX` for incremental fetches and `DataFeed.SIP` for backfill, exactly as the
   docstring at `alpaca.py:50-51` states. The historical load used SIP; every incremental load since
   has used IEX.
2. **The watermark cannot see interior gaps.** `_resolve_start` (`alpaca.py:218-240`) resumes from
   `max(existing) + 1 day`. Once the watermark passed 2026-04-06, the March hole became permanently
   unreachable. **The same pattern should be checked in `data/fred.py`** — it would explain G1-1.

### A third defect found while diagnosing

Stored prices are **adjusted with a frozen factor**. Our SPY 2016-06-15 close is 177.66; Alpaca's
adjusted value for the same bar *today* is 176.83. Adjusted prices are rewritten retroactively by
every subsequent dividend, so a stored adjusted series silently drifts from its own source. **The
same model retrained six months later sees different inputs, with nothing to flag the change.**

---

## Locked decisions

Settled with the user on 2026-07-25. Do not re-litigate without new evidence.

| # | Decision | Reason |
|---|---|---|
| **LD-2** | **Replace the full equity history**, not just the gap | Filling only the 18 dates would drop 19 bars of ~68M volume into a stretch reading ~1.1M — a sharper discontinuity than the gap. The contamination is also patchier than a single date: SPY 2023-06-15 already reads 1,762,275, well before the 2024-01-09 boundary |
| **LD-3** | **CBOE is the sole source for equity bars** | One vendor, one basis, across all 22.5 years. No seam at all |
| **LD-4** | **Store raw. Never adjust prices** | Raw never changes, so training is reproducible permanently. Every equity feature is a ratio (`price÷SMA`) or scale-invariant (RSI, MACD, BB position), and a quarterly dividend is a ~0.3% one-day artifact on SPY — smaller than a normal daily move |
| **LD-5** | **Take the full 2004→present depth** | +24,318 bars (+115%) and the 2008 crisis, which no deployed model has ever seen |
| **LD-6** | **Crypto: fill the 2026 outage only** | `detect_and_fill_crypto_gaps()` fills detected windows without re-stitching. But the 2019 gap (2019-08-31 → 2019-09-23) **straddles `STITCH_DATE = 2019-09-01`** (`binance.py:51`) — that one is the Binance.US/Global seam and stays untouched |
| **LD-7** | **No paper-trading restart** | Nothing in `execution/` reads historical bar *values* — only `MAX(date)` for a freshness warning (`execution/pipeline.py:1192`) and the latest close for pricing (`scheduler/jobs.py:374-375`). The ledger derives from fills, so a restart would discard the PR #42 benchmark re-anchor for no gain |

### Why CBOE, evidenced

| Check | Result |
|---|---|
| Price agreement with Alpaca **raw** | **Exact** on every date tested — 207.75, 442.60, 509.83, 738.18 |
| Depth | 2004-01-02 → present, ~5,676 bars per symbol |
| Completeness, 2016–2026 vs XNYS calendar | **0 missing sessions on 7 of 8 symbols.** XLI misses 4 (2019-01-07, 01-16, 01-17, 01-18) — versus **18 missing on all 8** today |
| Cost | Free. No API key, no rate limit |
| Volume basis vs Alpaca SIP | 3–8% lower. Irrelevant: `volume_sma20_ratio` is a rolling ratio, so a constant multiplier cancels exactly |

---

## Design

**Raw at rest. One vendor. Adjust nothing.**

```
CBOE charts/historical  ──►  deep history 2004 → T-2  ─┐
                                                       ├──►  ohlcv_daily  (RAW, one basis)
CBOE quotes             ──►  the just-closed session  ─┘           │
                                                                   ▼
Alpaca corporate-actions ──►  corporate_actions (dividends)    features_equity
                              [reference only — NOT used to adjust prices]
```

`corporate_actions` is populated because it closes G1-5 and because the future `dividend_yield`
feature needs it. Under LD-4 it never touches a price.

### Why two CBOE endpoints

`charts/historical` lags — on 2026-07-25 at 00:49 ET it still had no bar for Friday 2026-07-24.
`quotes` already carried that session in full (`last_trade_time: 2026-07-24T16:00:00`, o 738.47 /
h 743.72 / l 737.29 / c 738.93 / v 44,743,922). Together they cover the whole span.

**Two constraints this creates:**

1. **Capture window.** `quotes` reflects the prior session only until the 09:30 open, then switches
   to the live session. Capture must run **pre-open or post-close, never mid-session.** The existing
   09:15 ET equity cycle satisfies this.
2. **Volume settles late.** `quotes` reported 44,743,922 for Friday; the historical file typically
   lands ~4% higher once finalised. The daily bar is therefore **re-synced from `charts/historical`
   once it publishes** — the same capture-then-reconcile pattern the trade path already uses at
   09:35 and 17:00.

### Components

**New**

| Component | Responsibility |
|---|---|
| `data/cboe_bars.py` — `CboeBarsIngestor` | Fetch, validate and store CBOE daily bars. Follows the existing `BaseIngestor` fetch→validate→store→sync contract so it inherits ingestion logging |
| `data/corporate_actions_ingest.py` | Pull dividends from Alpaca's corporate-actions API into `corporate_actions` |
| Bar-quality validators | Reject OHLC-invariant violations (`high < max(open, close)`, `low > min(open, close)`, `high < low`), non-positive prices, negative volume, and flat zero-volume bars |

**Modified**

| Component | Change |
|---|---|
| `candles_equity_job` (`scripts/collector_main.py:303`) | **Repointed from Alpaca to CBOE.** The collector already owns equity candle ingestion — it runs at **20:15 ET Mon–Fri** (`collector_main.py:491-501`) and calls `run_equity(config, backfill=False)`, which is the Alpaca **IEX** path. **This job is how the contamination arrives nightly.** Under LD-3 the Alpaca equity bar path is retired, not left dormant, so it cannot silently resume |
| Watermark logic | The new CBOE ingestor resolves its start by **detecting interior gaps**, not `max + 1`. `alpaca.py:218-240`'s pattern is the bug that made A-2 permanent and must not be reproduced. **`data/fred.py` is to be checked for the same pattern** — it would explain G1-1, but fixing it belongs to that item |
| Postgres write path | An explicit **replace** path. `_sync_to_db` uses `ON CONFLICT DO NOTHING` (`base.py:167, :191`), so it can only insert new keys — it cannot correct an existing wrong row |
| `features/pipeline.py` | Recompute `features_equity` from the new bars. **No adjustment logic** (LD-4) |

### Scope covers both the one-off repair and the ongoing path

This spec delivers **two** things, and neither is complete without the other. Repairing the history
while leaving Alpaca on the daily path would let IEX bars start re-contaminating the series the very
next session.

### Data flow A — one-off replacement

1. `pg_dump` `ohlcv_daily`, `ohlcv_4h`, `features_equity`, `features_crypto`
2. Fetch all 8 symbols from CBOE `charts/historical` → validate → **replace** `ohlcv_daily`
3. Top up the latest session from CBOE `quotes`
4. Fetch dividends from Alpaca → populate `corporate_actions`
5. `detect_and_fill_crypto_gaps()` for the 2026 window only
6. Recompute `features_equity` and the affected `features_crypto` range
7. Verify: row counts, calendar completeness, invariant checks, before/after feature diff

### Data flow B — ongoing daily collection

**This reverses a documented prior ruling.** `config/swingrl.yaml:116` and
`scripts/collector_main.py:487` both state *"existing Alpaca/Binance ingestors — CBOE stays
options-only"* (2026-07-18). LD-3 supersedes that **for equity bars only**. Crypto bars stay on
Binance, and CBOE remains the options source as before.

| When | Job | Action |
|---|---|---|
| **20:15 ET, Mon–Fri** | `candles_equity_job` (existing schedule, unchanged) | Read the completed session from CBOE **`quotes`** and upsert it. 20:15 ET is 4¼ hours after the 16:00 close, so `quotes` carries the finished session and the pre-open/post-close capture constraint is satisfied |
| **T+2, same job** | `candles_equity_job` | Re-sync recent bars from CBOE **`charts/historical`** once published, correcting the provisional volume. Bounded to a short trailing window, so it also acts as a standing interior-gap repair |

The 20:15 slot is deliberately kept: it is already proven in production, it sits well clear of the
A30 quiet window (15:30–16:45 ET), and it lands the day-D bar on day-D evening so the trader's
09:15 ET cycle the next morning reads a T-1 bar exactly as it does today.

**Why the daily job cannot use `charts/historical` directly:** measured on 2026-07-25 at 00:49 ET,
that file still had no bar for Friday 2026-07-24 — roughly 8.8 hours after the close. At 20:15 ET,
only 4¼ hours after the close, it certainly will not. `quotes` is the only CBOE endpoint that
carries the session in time.

### Error handling

- Any validation failure **aborts the replacement** with the dump intact. Partial replacement is
  never an acceptable end state.
- CBOE returning fewer than the expected sessions for a symbol aborts rather than silently
  under-filling — the exact failure mode that produced A-2.
- Ingestion continues to log to `data_ingestion_log`. **`status='success'` must mean rows were
  written**; a `no_data` result must not be reported as success (the G1-1 defect).
- Alpaca corporate-actions failure is non-fatal — it blocks G1-5 only, not the bar replacement.

### Testing

TDD per project rules: RED commit, then GREEN.

- CBOE payload parsing, including a malformed and an empty payload
- Every bar-quality validator, each with a known-bad fixture drawn from the real defects found
  (the flat 2018-11-01 SPY bar; the two `low > min(open, close)` bars)
- Interior-gap detection: a series with a hole behind the watermark must be detected
- Replace semantics: an existing wrong row must be corrected, proving we are not on `DO NOTHING`
- Calendar completeness against XNYS
- Fast lane, then `scripts/ci-homelab.sh` from `~/swingrl` before PR

---

## Out of scope — deliberately

| Item | Why, and where it goes |
|---|---|
| **The four replacement features** (relative strength, drawdown, beta, dividend yield) | They change the observation vector and require a **retrain**. This spec makes the data correct so they become possible. → Stage 2.R |
| **A-7** — XLF 2016-09-19 spin-off (−18.3%) | **Alpaca returns zero spin-off records**, so it cannot be fixed from that source. Needs its own source decision. Under LD-4 no price is adjusted anyway, so this is a *documentation* gap, not a correctness regression |
| **A-1** — HMM served 4-month-stale regime with no staleness bound | A live code bug in `_get_hmm_probs`, independent of bar data. → next Group 1 item |
| **A-4** — 2019 crypto gap and its 7 fabricated bars per symbol | Sits on the Binance stitch seam (LD-6). Training-only data from 2019 |
| **A-5** — `p_crisis` collapsed | A modelling decision. Changes observation dimensions → retrain. → Stage 2.R |
| **G1-1/2/3** — macro freeze, cadence, staleness fuse | Separate Group 1 work. **G1-1 likely shares this spec's watermark root cause** — check `data/fred.py` for the `max + 1` pattern |

---

## Risks

| Risk | Handling |
|---|---|
| CBOE is an undocumented CDN with no SLA | Prices match Alpaca raw **exactly**, so Alpaca raw is a drop-in fallback. Ingestion aborts rather than writing partial data |
| Doubling history to 2004 changes the training window | No live impact. The training window stays a config choice → Stage 2.R |
| Pre-2016 bars cannot be dividend-adjusted (Alpaca actions start 2016-03-15) | Moot under LD-4 — nothing is adjusted, so the basis is uniform across all 22.5 years |
| `quotes` volume differs from the finalised value | The T+2 re-sync from `charts/historical` corrects it |
| XLI missing 4 sessions in Jan 2019 in CBOE | Known and documented. Still far better than the current 18 missing on all 8 symbols |
| Feature values shift after recompute | Expected and quantified by the before/after diff in step 7. Ratio features are largely invariant to rebasing; the real change comes from filled gaps and removed bad bars |

## Rollback

`pg_dump` taken at step 1 restores `ohlcv_daily`, `ohlcv_4h`, `features_equity` and
`features_crypto`. No schema change is destructive; any new column is additive (A30). The trader is
never stopped, and per LD-7 the trading ledger is untouched throughout.

---

## Verification checklist

- [ ] `ohlcv_daily` holds ~45,406 rows across 8 symbols, 2004-01-02 → present
- [ ] Zero missing XNYS sessions 2016→present, except XLI's 4 known January 2019 dates
- [ ] Zero OHLC-invariant violations, zero non-positive prices, zero negative volume
- [ ] No volume discontinuity: the 13–47× step at 2024-01-09 is gone
- [ ] `corporate_actions` holds 334 dividend rows from 2016-03-15
- [ ] `ohlcv_4h` has no gap between 2026-03-10 20:00 and 2026-04-06 12:00
- [ ] `features_equity` recomputed; before/after diff reviewed
- [ ] `candles_equity_job` reads CBOE, not Alpaca — confirmed by a live 20:15 ET run writing a bar
      whose volume is consolidated-scale, not IEX-scale
- [ ] The Alpaca equity bar path is retired, not merely unscheduled
- [ ] A second consecutive daily run leaves no duplicate or conflicting row (upsert, not
      `DO NOTHING`)
- [ ] Fast lane green, then homelab CI `=== CI PASSED ===`
