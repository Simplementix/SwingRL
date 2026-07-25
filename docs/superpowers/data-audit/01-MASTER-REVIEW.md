# Group A/B Data Audit — Master Review

> **Append-only.** One section per dataset, added by the session that reviews it.
> Review sessions record **findings only** — no solutions, no fixes, no recommendations.
> Every claim carries a `file:line`, a query, or command output. Inferences are marked
> **UNVERIFIED**.
>
> Process contract: [`00-PROCESS.md`](00-PROCESS.md).

---

## Glossary (terms used in this document)

| Term | Meaning |
|---|---|
| **Bar / candle** | One OHLCV record: open, high, low, close, volume for a time period |
| **OHLCV** | Open / High / Low / Close / Volume |
| **SIP** | Securities Information Processor — the US consolidated tape. Volume from *all* exchanges |
| **IEX** | A single US exchange (~2% of volume). Alpaca's free feed. Prices are real; volume is only IEX's slice |
| **Upsert** | Insert a row, or overwrite it if the key already exists |
| **`ON CONFLICT DO NOTHING`** | Insert a row, or **silently skip** it if the key already exists. The existing row is never updated |
| **Watermark** | The "how far have I got" marker an incremental fetch resumes from |
| **Vintage** | A distinct ingestion run, identified here by its `fetched_at` timestamp |
| **Stitch** | The join point where two data sources are concatenated into one series |
| **XNYS** | The NYSE trading calendar (`exchange_calendars` library), used to decide which dates *should* have an equity bar |
| **Quarantine** | Rows the validator rejected; written to `data_quarantine` and a Parquet file rather than the main table |
| **Observation slot** | One element of the fixed-length vector the RL agent sees each step |

---

## A1 — Candles

**Session 1 of 33. Reviewed 2026-07-25. Branch `swingrl/27-data-pipeline-audit`.**

**Scope:** `ohlcv_daily` (equity, 8 ETFs) + `ohlcv_4h` (crypto, BTCUSDT/ETHUSDT), both stores
(PostgreSQL and Parquet), plus the XNYS session calendar as a validation dependency.

**Headline.** The candle *values* are structurally healthy — no NULLs, no negative volume, no
`high < low`, no non-session dates, no degenerate columns. The defects fall into two families.

**Family 1 — the seams.** Two stores that disagree, and in each environment two feeds spliced
together with no normalisation: equity SIP→IEX at 2020-07-27 (30–90× volume step, 5.6 years wide),
crypto Binance Global→Binance.US at the 2019 stitch (~200×). Plus 18 missing equity sessions, a
22-day crypto hole that turns out to be self-inflicted, three invalid SPY bars and a five-bar
island of another symbol's prices.

**Family 2 — nothing is watching, and almost everything needed to watch already exists.** No
candle row has ever been quarantined. The candle layer raises **zero** alerts. The verification
gate passes vacuously — it reported "no gaps >24h" while two gaps of 22 and 26 days sat in the
table. `data_ingestion_log` recorded `rows_inserted=0` for four months and **nothing has ever
read it**. Yet 11 of 12 integrity capabilities are already built and mostly read the correct
store; only one is both correct and connected.

That second family is the more consequential result of this review: **the shortfall is wiring,
ownership and thresholds — not missing capability.** The two exceptions are equity gap *fill*,
which does not exist, and a place to record an integrity fact, which needs a schema change.

---

### A1.0 — Disposition of carried-forward assumptions

Per the process contract, every prior claim was re-verified from scratch. Result:

| Prior claim | Verdict | What is actually true |
|---|---|---|
| **A-2** 18 NYSE sessions missing 2026-03-11→2026-04-06; crypto 4H gap 2026-03-10 20:00→2026-04-06 12:00 (~160 bars/symbol) | **Confirmed in the authoritative store** | The dates are exactly right, and they are missing from **Postgres — the store the trainer and trader actually read**, so the gap is real for every consumer. Parquet happens to hold those rows, which makes it a *drift signal*, not a rescue: for equity, Parquet is Alpaca-derived and adjusted, and is superseded by the CBOE ruling → **A1-F1**, **A1-F20** |
| **A-3** Two-feed patchwork; volume drops 13–47× "around 2024-01-09"; QQQ 471.25 → 401.38; cause `alpaca.py:105-107` | **Mechanism confirmed, extent wrong** | Cause confirmed at `alpaca.py:104-107`. But the seam is **2020-07-27**, not 2024-01-09, the drop is **30–90×** (115×/189× adjacent-day), and the contaminated span is ~5.6 years → **A1-F4**. The "2024-01-09" event is a separate 5-bar artefact → **A1-F5** |
| **A-6** 3 bad SPY bars: 2018-11-01 flat (242.68, vol 200); 2024-01-02 & 2024-01-04 violate `low ≤ min(open, close)`; bypassed `DataValidator` | **Confirmed exactly** | All three present, values as stated. Bypass confirmed by elimination → **A1-F6** |
| **A-15** Crypto Parquet frozen at 2026-03-11 while Postgres is current | **Refuted — and inverted** | Crypto Parquet is **current** (last row 2026-07-25 12:00 UTC, mtime 12:01). It is **Postgres** that has the hole. Note this is the *crypto* store only; for equity the staleness runs the other way → **A1-F1**, **A1-F20** |
| **A-4** 7 *fabricated* flat bars per crypto symbol after the 2019-08-31→2019-09-23 *outage* at 9930.13 / 209.55, volume 0; sits on `STITCH_DATE` | **Confirmed as rows, wrong on all three characterisations** | 7 per symbol confirmed at those exact prices. But (i) they are **not fabricated** — the live Binance.US API returns those identical bars, they are the venue's own first-listing data → **A1-F7**; (ii) only **6 of 7** carry volume 0, the first carries 9.93013 (BTC) / 2.0955 (ETH); (iii) they sit at **2019-09-23/24**, 22 days *after* `STITCH_DATE`. A *different*, genuinely synthetic bar does sit exactly on the stitch → **A1-F8**. And the preceding 22 days are **not an outage** → **A1-F19** |
| Watermark reads Parquet, not the store the consumer reads (`alpaca.py:218-240`, `binance.py:254-258`); blind to interior gaps | **Confirmed** | Both confirmed, and the consequence is worse than stated: the two stores also diverge in *values* → **A1-F2**, **A1-F3** |
| `ohlcv_daily.adjusted_close` is 0/21,088 populated | **Confirmed** | And the column is redundant — `close` is already adjusted → **A1-F14** |
| CBOE proposed as replacement source (`charts/historical`, ~45,406 bars) | **Now a decision — USER RULING 2026-07-25** | CBOE is the going-forward equity candle source: a fuller history with **unadjusted** values. No CBOE code path touches candles today — `grep` over `src/swingrl/data/` finds CBOE only under `data/options/` — so this is a stated direction, not implemented state. It changes which store and which vintage count as authoritative → **A1-F1**, **A1-F20** |

---

### A1.1 — The data itself

#### What is stored

| | Equity `ohlcv_daily` | Crypto `ohlcv_4h` |
|---|---|---|
| Symbols | 8 (SPY, QQQ, VTI, XLE, XLF, XLI, XLK, XLV) | 2 (BTCUSDT, ETHUSDT) |
| Rows (Postgres) | **21,088** (2,636 × 8) | **38,545** (BTC 19,273 / ETH 19,272) |
| Rows (Parquet) | **21,232** (2,654 × 8) | **38,862** (19,431 × 2) |
| Range | 2016-01-04 → 2026-07-24 | 2017-08-17 00:00 → 2026-07-25 08:00 ET |
| Grain | 1 calendar day | 4 hours |
| Columns | symbol, date, o/h/l/c, volume, adjusted_close, fetched_at | symbol, datetime, o/h/l/c, volume, source, fetched_at |

Schema via `docker exec pg16 psql -U swingrl -d swingrl -c "\d ohlcv_daily"` / `"\d ohlcv_4h"`.
Primary keys: `(symbol, date)` and `(symbol, datetime)`.

Freshness is correct as of review: 2026-07-25 is a Saturday, so the newest equity bar
(2026-07-24, Friday) is the right one.

#### Value quality — clean results

Recency is not health, so these were measured rather than assumed:

| Check | Equity | Crypto |
|---|---|---|
| NULL in any of o/h/l/c | 0 | 0 |
| NULL volume | 0 | 0 |
| `high < low` | 0 | 0 |
| `high < max(open, close)` | 0 | 0 |
| Price ≤ 0 | 0 | 0 |
| Negative volume | 0 | 0 |
| Zero volume | 0 | 12 (all inside A1-F7) |
| Distinct `close` / rows | 0.617–0.964 | 0.952–0.993 |
| Distinct `volume` / rows | 2,615–2,636 of 2,636 | 19,267–19,268 of ~19,272 |
| Dates that are not XNYS sessions | **0** | n/a (24×7) |

No constant or degenerate column exists in either table. The low end of the `close`-uniqueness
range (XLE 0.617, XLF 0.648) is expected: both trade around \$10–20, so tick values repeat.
This is not the failure mode that killed the 32 equity features.

```sql
SELECT symbol, COUNT(DISTINCT close) d_close, COUNT(DISTINCT volume) d_vol, COUNT(*) n
FROM ohlcv_daily GROUP BY symbol ORDER BY symbol;
```

#### Value quality — defects

**A1-F6 — Three invalid equity bars. (Confirms A-6 exactly.) Confidence: high (direct query).**

```
 symbol |    date    |       open        |       high       |        low        |       close       |  volume  |     fetched_at
 SPY    | 2018-11-01 |            242.68 |           242.68 |            242.68 |            242.68 |      200 | 2026-03-11 08:43:17
 SPY    | 2024-01-02 | 469.6338118078721 | 471.351030207974 | 470.1549567159393 | 470.6094341595088 | 58261453 | 2026-03-06 19:22:37
 SPY    | 2024-01-04 | 469.2443040333638 | 471.318098578802 | 469.9027338264324 | 470.0303683386408 | 67726298 | 2026-03-06 19:22:37
```

- **2018-11-01** is flat (o=h=l=c) with volume 200 — SPY's real 2018-11-01 volume was ~10⁸.
- **2024-01-02** and **2024-01-04** have `low > open`, which is arithmetically impossible in a
  real bar. The irrational decimals are the signature of a price-adjustment computation applied
  per-field with inconsistent factors.

**These are two defects wearing one label, and 2018-11-01 is the concrete instance of A1-F3.**
The 2024 pair sit inside the **A1-F5** probe island (vintage 2026-03-06, SIP + unadjusted) — not a
production path. 2018-11-01 came from the ordinary backfill, and the vintage boundaries show
exactly how:

| | SPY | The other 7 symbols |
|---|---|---|
| Value on 2018-11-01 | **242.68 flat, volume 200**, vintage **08:43** (IEX) | real values, volume 3.7 M–67 M, vintage **10:31** (SIP) |
| First date of the IEX vintage | **2018-11-01** | 2020-07-27 |
| Rows in the IEX vintage before 2020-07-27 | **exactly 1** | **0** |
| Rows received from the SIP vintage | **1,147** — one short of the 1,148 sessions | complete |

The IEX backfill at 08:43 over-reached by a single row and claimed that primary key. The SIP
backfill arrived **108 minutes later** carrying the correct bar — CBOE independently confirms it as
o 271.60 / h 273.73 / l 270.38 / c 273.51, volume 99,495,037, invariants valid — and
`ON CONFLICT DO NOTHING` discarded it silently. The other seven symbols had no IEX row on that
date, so their SIP rows landed intact. That is why SPY alone is corrupt here, and why SPY alone is
exactly one row short in the SIP era.

**A1-F3 consequence #1 caught in the act, with timestamps: correct data arrived, and was thrown
away without a log line.** It is also a 1-row rehearsal of a 21,088-row event — see carry item 3.

**A1-F7 — Fourteen flat crypto bars — genuine venue data, not fabricated. (Corrects A-4 on all
three of its characterisations.) Confidence: high (direct query + live API confirmation).**

Seven consecutive bars per symbol, 2019-09-23 04:00 → 2019-09-24 04:00 ET, every field equal to
**9930.13** (BTCUSDT) / **209.55** (ETHUSDT).

Three corrections to the prior claim:

1. **They are not fabricated.** A live call to the Binance.US API for that window returns the same
   bars, beginning at 2019-09-23 08:00 UTC with `close = 9930.13000000`. These are the exchange's
   own first-listing bars, published before the pair had meaningful trade flow. Nothing in SwingRL
   synthesised them; the ingestion code contains no bar-synthesis logic.
2. Only **6 of 7** carry volume 0 — the first carries 9.93013 (BTC) / 2.0955 (ETH).
3. They sit 22 days *after* `STITCH_DATE = 2019-09-01` (`binance.py:51`), not on it.

What remains a defect is not their origin but their treatment: they are flat, economically
meaningless bars that entered the training series unflagged, because Step 7's zero-volume rule is
equity-only (`validation.py:133`) and no flat-bar check exists at all. See **A1-F10**.

They also mark the true start of Binance.US coverage, which is the evidence behind **A1-F19**.

**A1-F8 — A synthetic bar sits exactly on the crypto stitch date. NEW — in no prior document.
Confidence: high (direct query + set difference).**

```
 BTCUSDT | 2019-08-31 20:00 ET (= 2019-09-01 00:00 UTC) | o 10000 | h 10200 | l 9900 | c 10100 | vol 3030000
```

Perfectly round OHLC values and a round volume, at exactly `STITCH_DATE`. The preceding real bar
closes at 9587.47; the next real bar opens at 9930.13. This bar is **present in Postgres and
absent from Parquet** — it is the *only* row in either crypto table that exists in one store and
not the other in that direction:

```
BTCUSDT: pq_only=159, pg_only=1 → ['2019-09-01 00:00']
ETHUSDT: pq_only=159, pg_only=0
```

**UNVERIFIED:** its origin. Its shape (round numbers, single row, exactly on the stitch boundary,
Postgres-only) is consistent with test-fixture data reaching the production table, but no code
path was found that writes it.

**A1-F5 — A five-bar "probe island" in SPY and QQQ, and QQQ's bars are not QQQ. NEW.
Confidence: high for the anomaly (direct query); medium for the SPY-contamination reading.**

Ten rows — SPY and QQQ only, 2024-01-02 → 2024-01-08 — carry `fetched_at` of 2026-03-06 19:22/19:25,
a vintage that exists nowhere else in the table. Surrounding rows are all from the 2026-03-11
backfill. Within this island:

```
 QQQ | 2023-12-29 | close 405.02  | vol    819,315  | vintage 2026-03-11   <- normal
 QQQ | 2024-01-02 | close 472.35  | vol 54,672,100  | vintage 2026-03-06   <- island
 QQQ | 2024-01-08 | close 471.25  | vol 55,894,300  | vintage 2026-03-06   <- island
 QQQ | 2024-01-09 | close 401.38  | vol    754,373  | vintage 2026-03-11   <- normal
```

QQQ jumps **+16.62 %** into the island and back out again, with no split in January 2024. It is the
single largest one-day move in the entire equity table after two genuine COVID-crash bars:

```sql
WITH r AS (SELECT symbol,date,close, LAG(close) OVER (PARTITION BY symbol ORDER BY date) p FROM ohlcv_daily)
SELECT symbol,date,p,close, ROUND((100*(close/p-1))::numeric,2) pct FROM r WHERE ABS(close/p-1)>0.15 ORDER BY 5 DESC;
-- XLE 2020-03-09 -20.16 | XLF 2016-09-19 -18.25 | QQQ 2024-01-02 +16.62 | XLE 2020-03-24 +16.54 | XLF 2020-03-16 -15.61
```

QQQ's true level on both sides of the island is ~405; inside it, ~470 — SPY's level. A per-field
ratio test against SPY's island rows gives ratios of 0.9927–1.0067, i.e. QQQ's island values track
SPY's to within 0.7 % every day. Two ETFs whose real prices differ by ~15 % cannot do that.
**Verified:** the QQQ island bars are not QQQ data.

**Cross-checked against CBOE — attribution upgraded, and the island is larger than it appeared.**
The ratio test above compares our QQQ against our *own* SPY, which is circular because those SPY
rows are themselves island rows. Repeating it against an independent source removes that. Ratio of
stored `close` to CBOE `close`:

| | 2023-12-29 | **2024-01-02** | **01-03** | **01-04** | **01-05** | **01-08** | 2024-01-09 |
|---|---|---|---|---|---|---|---|
| **SPY** | 1.025 | 1.004 | 1.001 | 0.994 | 0.992 | 1.014 | 1.025 |
| **QQQ** | 1.011 | 0.852 | 0.846 | 0.844 | 0.846 | 0.859 | 1.011 |

Three results:

1. **The QQQ island bars are SPY's — now established independently.** Stored QQQ close on
   2024-01-02 is 472.35; CBOE's real QQQ is 402.59 and CBOE's real **SPY is 472.65**. Across all
   five days the island tracks CBOE SPY to within 0.7 %. Attribution moves **medium → high**.
2. **All five SPY island bars are also wrong, not just the two of A1-F6.** Outside the island SPY
   holds a stable **1.025** against CBOE — the dividend adjustment. Inside, it scatters
   **0.992–1.014**. Correctly-adjusted SPY would read 1.025 there too. The island is therefore
   **10 bad bars, not 5**.
3. **The island was fetched with a different configuration entirely.** Its volume is SIP-scale
   (53–98 M) while its immediate neighbours are IEX-scale (1.0–4.8 M), and its prices sit near
   ratio 1.000 rather than 1.025 — i.e. **unadjusted**. An unadjusted, SIP-feed probe dropped into
   an adjusted, IEX-feed series, which is why its `fetched_at` vintage isolates it so cleanly.

**UNVERIFIED:** what wrote it. The configuration signature (SIP + unadjusted, two symbols, five
days, a vintage that exists nowhere else) narrows it to a manual or exploratory fetch, but no code
path was found.

Note the two bad SPY bars of **A1-F6** live inside this same island. CBOE's own 2024-01 bars are
correctly valued for both symbols, so unlike **A1-F4** this finding is removed outright by a full
CBOE replacement, with nothing left to qualify.

**A1-F15 — An unadjusted spin-off discontinuity in XLF. Confidence: high (query) / medium
(attribution).**

XLF shows **−18.25 %** on 2016-09-19 — the date the real-estate constituents were spun out into
XLRE. `alpaca.py:126` requests `Adjustment.ALL`, which covers splits and dividends but not
spin-offs, so the discontinuity is carried into the series as a genuine-looking return. It is
below the validator's 50 % spike threshold (`validation.py:31`) and was never flagged.

**CBOE does not fix this bar.** Both sources carry the identical move, which confirms neither
adjusts for spin-offs:

| XLF | 2016-09-16 | 2016-09-19 | return |
|---|---|---|---|
| Postgres (`Adjustment.ALL`) | 19.89 | 16.26 | **−18.25 %** |
| CBOE (unadjusted) | 23.62 | 19.31 | **−18.25 %** |

**Under the unadjusted ruling this finding inverts from one bar to a systemic property.** The
cumulative gap between unadjusted CBOE and our adjusted series *is* the total false return that
unadjusted storage would inject, and it is threshold-free arithmetic:

| Cumulative adjustment, 2016-01 → 2026-07 | SPY | QQQ | VTI | XLE | XLF | XLI | XLK | XLV |
|---|---|---|---|---|---|---|---|---|
| | 16.9 % | 7.9 % | 17.9 % | **43.8 %** | 20.1 % | 18.9 % | 11.4 % | 16.4 % |

Our stored 2016 XLE close sits **43.8 % below** the price that actually traded that day, because a
decade of dividends has been adjusted out of it. Today that absorbs every ex-dividend date
invisibly. Unadjusted, it re-materialises as discrete downward steps at roughly 42 dates per
symbol — each one a plausible-looking small loss, and each one *below* the 50 % spike threshold,
exactly as the spin-off is.

**An honest gap.** A per-event count could not be established from the two series. Day-to-day noise
between the sources has σ = 0.151 %, comparable to a single SPY dividend step of ~0.26 %, so the
detected event count swings from 159 to 44 purely with the threshold. **No event count is published
here.** Enumerating them requires actual `corporate_actions` data, which holds **0 rows**
(**A1-F26**) — which is why B3 is a dependency of the equity migration rather than a parallel task.

Two consequences, recorded as consequences and not as a design:

1. **Complete corporate actions would fix this finding, not merely offset the ruling.** The
   spin-off has been wrong in our data since 2016 and no vendor adjustment column has ever
   corrected it. Raw storage plus a complete actions table is strictly better than today on every
   axis — immutable store, dividends adjusted, splits adjusted (×4.0000 verified in **A1-F3**),
   spin-offs adjusted **for the first time**, and history reproducible.
2. **A missing action becomes silent rather than visible.** Today a missing spin-off is one
   conspicuous −18.25 % bar. Under read-time adjustment, a missing action scales the entire
   pre-event history of that symbol incorrectly — no discontinuity, no visible bar, just a quietly
   wrong series. The failure mode becomes less noisy, which puts the burden of catching it on the
   cross-source check of carry item 12.

---

### A1.2 — Historical one-time ingestion, and its checks and balances

`fetched_at` acts as a vintage marker because Postgres rows are never updated (see **A1-F3**), so
it records the **first** insert of each key. This makes the ingestion history directly readable:

```sql
SELECT date_trunc('second', fetched_at) AS vintage, COUNT(*) n, COUNT(DISTINCT symbol) syms,
       MIN(date), MAX(date) FROM ohlcv_daily GROUP BY 1 ORDER BY 1;
```

| Vintage | Rows | Symbols | Date span | What it is |
|---|---|---|---|---|
| 2026-03-06 19:22 / 19:25 | 10 | SPY, QQQ | 2024-01-02 → 2024-01-08 | The probe island (**A1-F5**) |
| 2026-03-11 08:43 | 11,287 | 8 | 2018-11-01 → 2026-03-10 | **IEX** vintage — 2020-07-27 onward |
| 2026-03-11 10:31 | 9,183 | 8 | 2016-01-04 → 2020-07-24 | **SIP** vintage — the deep backfill |
| 2026-07-18 22:19 | 568 | 8 | 2026-04-07 → 2026-07-17 | Post-outage catch-up |
| 2026-07-21 → 07-24 | 40 | 8 | 2026-07-20 → 2026-07-24 | Collector, daily |

Crypto (`ohlcv_4h`, same query on `fetched_at`): one historical vintage 2026-03-11 (37,225 rows,
2017-08-17 → 2026-03-10), one catch-up 2026-07-18 (1,240 rows, 2026-04-06 → 2026-07-18), then 12
rows/day from the collector.

**A1-F4 — The equity volume seam is at 2020-07-27, not 2024-01-09, and it is 30–90× wide.
Corrects A-3. Confidence: high (direct query + code).**

Mechanism confirmed at `alpaca.py:104-107`:

```python
if since == "incremental":
    feed = DataFeed.IEX
else:
    feed = DataFeed.SIP
```

Backfill gets consolidated-tape volume; incremental gets one exchange's slice. Because Postgres
inserts are `DO NOTHING`, the IEX vintage landed first and the later SIP backfill could only fill
the dates IEX had not already claimed — which is why the boundary is a clean date, not a mixture.

Adjacent-day evidence:

```
 SPY | 2020-07-24 | vol 74,375,086      SPY | 2020-07-27 | vol   647,693   (115×)
 QQQ | 2020-07-24 | vol 54,756,516      QQQ | 2020-07-27 | vol   289,367   (189×)
```

Whole-era ratio, all 8 symbols (`AVG(volume)` before vs from 2020-07-27):

| | SPY | QQQ | VTI | XLE | XLF | XLI | XLK | XLV |
|---|---|---|---|---|---|---|---|---|
| ratio | 63.0 | 44.9 | 71.8 | 30.2 | 39.4 | 43.1 | **89.4** | 35.3 |

So **5.6 years of the 10.5-year equity history — 2020-07-27 to 2026-03-10 — carries IEX-only
volume**, not a localised patch around 2024-01. Prices are unaffected (IEX prices are real); only
the volume magnitude changes. The prior claim's cited example (SPY 2023-06-15 = 1,762,275) sits
inside the IEX era and is consistent with it, not evidence of patchiness.

**The IEX regime is not a bounded historical span — it is the standing one.** The collector's
`candles_equity_job` calls `run_equity(config, backfill=False)`, i.e. `since="incremental"`, so
every bar written from 20:15 ET each weekday takes the IEX branch. Measured on bars written after
2026-07-01: SPY averages **1,487,773** — IEX scale, against SIP's 92.8 M. The seam has no closing
date. Read correctly, the 4.5-year SIP era at the start is the anomaly and IEX is the norm.

**Conditional supersession.** Under the CBOE ruling this seam disappears — one source, one basis,
no feed branch — and CBOE's volume basis is now **verified consolidated** at 0.968× Alpaca SIP
(**A1-F27**). Three qualifications apply, and none is a reason to keep Alpaca: the crypto seam
(**A1-F9**) is untouched by an equity source change; CBOE brings its own defect set (**A1-F27**);
and a changeover that *retains* pre-CBOE Alpaca rows merely relocates the seam to the changeover
date. **USER RULING 2026-07-25 — the equity candle table is replaced in full, precisely so no
provider seam exists.** That is the one operation **A1-F3**'s guard currently forbids, which is
what couples carry items 3 and 11.

**A1-F9 — Crypto has the same class of seam, ~200× wide, at the archive/API stitch. NEW.
Confidence: high (query + code).**

The crypto history is spliced from two *different exchanges*:

- pre-`STITCH_DATE`: Binance **Global** monthly archives, `ARCHIVE_BASE_URL = https://data.binance.vision/...` (`binance.py:45`)
- from `STITCH_DATE`: Binance **US** API, `BINANCE_US_BASE_URL = https://api.binance.us` (`binance.py:37`)

Both paths normalise volume to the *quote* asset (`binance.py:218`, `binance.py:383-386`;
`gap_fill.py:265`), so the units are consistent — that part is correct. But Binance.US carries a
fraction of Binance Global's liquidity:

```sql
SELECT symbol,
 AVG(volume) FILTER (WHERE datetime < '2019-09-01 00:00+00')                                  archive_era,
 AVG(volume) FILTER (WHERE datetime >= '2019-09-24 08:00+00' AND datetime < '2020-09-01')     us_early,
 AVG(volume) FILTER (WHERE datetime >= '2026-01-01')                                          recent
FROM ohlcv_4h GROUP BY symbol;
```

| symbol | archive era | Binance.US early | ratio | recent |
|---|---|---|---|---|
| BTCUSDT | 40,373,541 | 198,968 | **203×** | 514,570 |
| ETHUSDT | 11,368,226 | 57,534 | **198×** | 253,897 |

This is a **DS-7 parity finding**: equity and crypto each carry an unnormalised venue-volume seam
of the same order of magnitude, created by the same design choice (splice two sources, keep the
raw volume), and neither is flagged anywhere.

**A1-F16 — The stitch validator checks price only, and its verdict is discarded.
Confidence: high (code read).**

`_validate_stitch` (`binance.py:390-442`) compares **close prices** across the overlap window
against `STITCH_MAX_DEVIATION = 0.005`. It never looks at volume, so a 200× volume shift passes
cleanly. Its boolean return is also thrown away at the call site:

```python
# binance.py:505
self._validate_stitch(archive_df, api_df)
```

A failing stitch therefore logs a warning and the backfill proceeds regardless.

More fundamentally, **the stitch performs no reconciliation of any kind.** A grep for scaling in
`binance.py` returns only lines 351 and 382, and both are *within-source* unit conversion (use
quote volume, else `base × close`). There is no cross-venue factor, no volume normalisation and no
price alignment. Phase 4 is a hard cut:

```python
# binance.py:509-510
archive_portion = archive_df[archive_df.index < stitch_ts]   # Binance Global
api_portion     = api_df[api_df.index >= stitch_ts]          # Binance.US
```

So the venue difference that motivates having a stitch at all is precisely what the stitch does not
handle: it *detects* a >0.5 % price divergence and ignores the answer, and it never looks at volume.

**A1-F19 — The 22-day crypto "outage" is not an outage. It is created by the stitch filter, and
the missing data was downloaded and then discarded. NEW — in no prior document.
Confidence: high (live archive fetch + live API + code read).**

`STITCH_DATE = "2019-09-01"` is justified in-code as "Binance.US launched Sep 2019"
(`binance.py:50-51`). But Binance.US's first 4H bar is **2019-09-23 08:00 UTC** — the constant is
wrong by 22 days. Because Phase 4 keeps archive rows only where `index < stitch_ts`, every archive
bar from 2019-09-01 onward is dropped, and Binance.US has nothing to put in their place.

The discarded data exists, is free, and is already being fetched. `_download_archives`
(`binance.py:481`) requests through `stitch + 30 days` = 2019-10-01, so the September 2019 archive
**is downloaded on every backfill and then thrown away**. Verified by fetching it directly:

```
GET https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-2019-09.zip
-> HTTP 200, 11,394 bytes, 180 bars, 2019-09-01 00:00 UTC .. 2019-09-30 20:00 UTC
   bars inside the 22-day hole: 134
```

The recovered bars bridge both edges cleanly, so this is not a quality trade-off:

| Boundary | Value | Continuity |
|---|---|---|
| Last stored real bar — 2019-08-31 20:00 UTC | close **9587.47** | — |
| Archive first bar in the hole — 2019-09-01 00:00 UTC | open **9588.74** | **0.01 %** |
| Archive last bar in the hole — 2019-09-23 04:00 UTC | close **9908.94** | — |
| First Binance.US bar — 2019-09-23 08:00 UTC | open **9930.13** | **0.21 %** |

Both deviations sit far inside `STITCH_MAX_DEVIATION = 0.005`.

Two properties of this gap that matter downstream, recorded as findings rather than as a remedy:

- **The gap and the volume seam are independent defects.** Archive bars carry Global-scale volume
  (~2 × 10⁷ for BTC). Closing the gap with them extends the Global volume regime to 2019-09-23 —
  it *relocates* the **A1-F9** step by 22 days rather than removing it. Fixing one does not fix the
  other.
- **`DO NOTHING` blocks the correction at exactly one timestamp.** The synthetic bar of **A1-F8**
  occupies `2019-09-01 00:00 UTC`, which is precisely where the archive's real bar
  (o 9588.74 / h 9642.58 / l 9579.34 / c 9623.02) belongs. Under `base.py:191` / `gap_fill.py:345`
  semantics, an insert at that key is silently skipped — so the fabricated bar survives and its
  real replacement is dropped without a log line.

**A1-F13 root cause — `gap_fill.py` cannot reach its source from this host.
Confidence: high (live HTTP).**

`BINANCE_GLOBAL_BASE_URL = "https://api.binance.com"` (`gap_fill.py:38`) returns, from homelab:

```json
{"code": 0, "msg": "Service unavailable from a restricted location according to 'b. Eligibility' ..."}
```

The crypto gap-filler is geo-blocked and can never have worked from this host — which is why
`ohlcv_4h.source` is 100 % NULL. This is not a subtle bug. Note the archive CDN
(`data.binance.vision`, used by `binance.py:45`) is **not** blocked and returned HTTP 200 in the
same session, so the two Binance endpoints have different reachability from this environment.

#### What validated the history at the time

**A1-F10 — Not one candle has ever been quarantined. Confidence: high (direct query).**

```sql
SELECT source, failure_reason, COUNT(*) FROM data_quarantine GROUP BY 1,2;
-- macro | Step 1: null value | 3939      (only row)
```

All 3,939 quarantine rows are macro. Corroborated independently by the ingestion log:
`SUM(errors_count) = 0` for both `equity` and `crypto` across all 287 + 84 recorded runs.
The equity Parquet quarantine directory contains only `CPIAUCSL_*.parquet` files.

Given **A1-F6**, this is not "clean data" — it is a validator that did not see the data:

- **SPY 2024-01-02 / 2024-01-04 *should* have been caught.** Step 5 (`validation.py:116-123`)
  flags `open < low − tol`. For 2024-01-02: open 469.6338, low 470.1550, tol = 471.351 × 0.0001 =
  0.0471, so `l_minus_tol` = 470.1078 and open falls below it. The check fires. Since the rows are
  in the table and the quarantine is empty, **they did not pass through `DataValidator` at all.**
  Verified by elimination; the specific bypass path is **UNVERIFIED**, though
  `sync_parquet_to_db` (`base.py:341-386`) is a validation-free Parquet→Postgres loader that
  matches the signature.
- **SPY 2018-11-01 would never have been caught.** A flat bar passes every one of steps 1–7: no
  nulls, prices > 0, volume ≥ 0, `high ≥ low`, open/close inside the range, no >50 % move, and
  volume 200 ≠ 0 so the zero-volume rule does not apply. There is **no flat-bar check and no
  implausible-volume check** in `validation.py`.
- **The crypto flat bars would never have been caught either.** Step 7's zero-volume rule is
  explicitly equity-only (`validation.py:133`: `if self._source == "equity":`), so 12 zero-volume
  crypto bars are legal by design. A second **DS-7 parity gap**.

**Tested forward against CBOE.** The above is a historical account. Because the CBOE ruling makes
this validator the gate that a new source must pass, the twelve steps were executed against the
~50 defects of **A1-F27** to establish what would happen on migration day:

| CBOE defect | Count | Current validator verdict |
|---|---|---|
| OHLC invariant violations | 36 | **30 caught** by Step 5 — **6 pass**, sitting inside the `high × _TOLERANCE` band (`validation.py:113`, `_TOLERANCE = 0.0001`) |
| Non-session bars, 2018-11-22 (Thanksgiving) | 7 | **0 caught** |
| Non-session bars, 2025-05-26 (Memorial Day) | 8 | **0 caught** |
| Corrupt 2019-01 volume (up to 1,518× under-report) | 30 | **0 caught** — no implausible-volume check exists |
| Duplicate dates | 3 | Step 8 dedups **silently** — no flag, no quarantine row |
| Flat bars (o=h=l=c) | 7 | **0 caught** — no flat-bar check exists |

**Roughly 30 of ~50 are caught; roughly 20 would enter the authoritative store.** Two mechanisms
behind that, neither previously recorded:

1. **Step 7 excuses the holiday bars by design.** `validation.py:140` reads
   `if self._nyse.is_session(normalized):` — zero volume is flagged **only when the date is a
   trading session**. Thanksgiving is not, so those rows pass deliberately. More fundamentally,
   **no step anywhere rejects a date that is not a trading session.** Step 9 detects *missing*
   sessions and never *extra* ones. The clean result in **A1.1** — zero non-session dates in
   `ohlcv_daily` — reflects what Alpaca supplied, not anything the pipeline enforces.
2. **The Memorial Day set is invisible to every check.** Unlike Thanksgiving's flat, zero-volume
   rows, the 2025-05-26 bars carry full OHLC and plausible volume: **SPY reads 75,990,006 shares
   traded on a day the market was closed.** Six of the eight simply repeat the prior session's
   close; XLE and XLK are doubled and revert the next day, which is the only reason Step 6 sees
   anything at all.

**A1-F17 — Step 12 (cross-source price check) is dead on the equity path.
Confidence: high (code read).**

`validation.py:235` requires `self._db is not None and self._config is not None`. But
`AlpacaIngestor.validate()` constructs its validator with neither:

```python
# alpaca.py:164
validator = DataValidator(source="equity")
```

So the yfinance cross-check in `cross_source.py` never runs during ingestion. Step 12 compares our
stored equity closes against **yfinance** — a genuinely independent provider — over a 7-day
lookback, warning when they disagree. It is the only pre-existing attempt anywhere in SwingRL to
answer "is our data actually right?", and it has never executed.

Three extensions established this session:

1. **The parameters are dead across the whole codebase.** `db` and `config` default to `None`
   (`validation.py:53-54`) and **no caller anywhere passes them** — `alpaca.py:78`, `alpaca.py:164`,
   `binance.py:111` and `fred.py:69` all pass `source=` alone. There is no code path that could
   enable Step 12, on any source.
2. **The skip is invisible.** The `else` branch logs `cross_source_check_skipped` at **DEBUG**
   (`validation.py:257`) while `config/swingrl.yaml:40` sets the level to **INFO**. Nothing has ever
   reported that the check is being skipped.
3. **The obvious fix would not work, and equity is the odd one out.** `alpaca.py:78` assigns
   `self._validator` and nothing reads it; `validate()` at `:164` constructs a **fresh bare
   validator** and uses that. Passing `db`/`config` at line 78 would therefore change nothing.
   `binance.py` (`:317`, `:319`) and `fred.py` (`:237`) both use their instance validator —
   **`alpaca.py` is the only ingestor that discards its own**, and it is the one Step 12 lives on.

---

### A1.3 — Ongoing ingestion, and its checks and balances

#### Owner and schedule

The **`swingrl-collector`** container owns candle freshness (`swingrl-collector:2026-07-22-1`,
up 2 days at review).

| Job | `file:line` | Trigger | Source |
|---|---|---|---|
| `candles_equity_job` | `scripts/collector_main.py:303` | cron, Mon–Fri **20:15 ET**, misfire grace 6 h | Alpaca, `since="incremental"` → **IEX** |
| `candles_crypto_job` | `scripts/collector_main.py:336` | cron, UTC hours 0/4/8/12/16/20 at **minute 1**, misfire grace 3 h | Binance.US API |

Schedules registered at `collector_main.py:491-509`; values from `config/swingrl.yaml:116-121`.
Both confirmed against observed `fetched_at`: equity rows land at 20:15:00, crypto at 12:01:00.

Both jobs swallow all exceptions and alert rather than raise, so the scheduler survives a failure.
Both send an INFO Discord alert on success carrying `rows_added`.

#### Watermark resolution — and the store split

**A1-F2 — The watermark reads Parquet; every consumer reads Postgres. Confidence: high (code read).**

| | Store read |
|---|---|
| Equity watermark | **Parquet** — `alpaca.py:218-240`, specifically `:220` `existing = self._store.read(path)` then `:228` `max_ts = existing.index.max()` |
| Crypto watermark | **Parquet** — `binance.py:253-261`, specifically `:256` and `:258` |
| Feature pipeline | **Postgres** — `features/pipeline.py:659`, `:681` |
| Trainer | **Postgres** — `training/data_loader.py:165,178,292,305`; `scripts/train.py:244,260,369,385` |
| Trader freshness guard | **Postgres** — `execution/pipeline.py:1192` |
| Shadow runner | **Postgres** — `shadow/shadow_runner.py:35` |
| Macro join | **Postgres** — `features/macro.py:43,87` |
| Benchmark baselines | **Postgres** — `scheduler/jobs.py:374-375`; `scripts/record_benchmark_baselines.py:50-51` |

A full `grep` for `ohlcv_daily|ohlcv_4h` across `src/` and `scripts/` finds **no consumer that
reads Parquet**. Parquet is read only by the two watermarks and by `scripts/repair_partial_bars.py`.

Both watermarks take `MAX()` of the index, so they are blind to interior gaps by construction — a
hole in the middle of the series does not move the maximum, so the next incremental fetch resumes
after the newest bar and never revisits it.

Combined, these two facts mean **a gap in Postgres is permanent**: the consumer's store has the
hole, and the watermark's store does not, so nothing will ever ask for those bars again.

**The watermark performs no audit function whatsoever.** It answers exactly one question — "from
which timestamp do I resume?" — and asks nothing about the integrity of what precedes it. It does
not count rows, does not compare against the trading calendar, does not look for interior gaps,
does not compare the two stores, and does not raise anything. Describing it as a check or balance
would misstate what the code does: it is a resume pointer, and the only reason it appears in this
section is that its *choice of store* is what makes **A1-F1** permanent.

#### The two stores disagree

**A1-F1 — The authoritative store is missing 18 equity sessions and 159 crypto bars per symbol.
Refutes A-15 and re-frames A-2. Confidence: high (set difference, both stores).**

Postgres is the store every consumer reads (**A1-F2**, **A1-F20**), so this is a live gap for the
trainer and the trader, not a bookkeeping curiosity. Parquet holding the rows is what makes the
divergence *measurable* — it is not a reason to call the data present.

Equity — identical for all 8 symbols:

```
SPY: parquet 2654  postgres 2636   in-parquet-not-pg = 18   in-pg-not-parquet = 0
2026-03-11, 03-12, 03-13, 03-16, 03-17, 03-18, 03-19, 03-20, 03-23, 03-24,
03-25, 03-26, 03-27, 03-30, 03-31, 04-01, 04-02, 04-06
```

Crypto:

```
BTCUSDT: parquet 19431  postgres 19273   pq_only = 159 (2026-03-11 04:00 → 2026-04-06 12:00 UTC)
ETHUSDT: parquet 19431  postgres 19272   pq_only = 159 (same span)
```

Cross-checked against the trading calendar: XNYS lists **2,654** sessions between 2016-01-04 and
2026-07-24. Parquet has exactly 2,654 rows per symbol and **zero** non-session dates; Postgres has
2,636. Parquet is calendar-complete; Postgres is 18 short.

```
XNYS sessions expected: 2654   PG has: 2636   MISSING: 18   EXTRA: 0   (exchange_calendars 4.13.1)
```

The equity gap is a single contiguous run in Postgres:

```sql
-- only gap > 4 calendar days in the whole equity table
gap_after 2026-03-10 | resumes 2026-04-07 | 28 calendar days
```

Crypto Postgres carries the same 2026 hole plus the 2019 stitch hole (**A1-F19**); all other
interior gaps are 8–12 h and pre-date 2023:

```
BTCUSDT | 2019-08-31 20:00 -> 2019-09-23 04:00 | 22 days 08:00
BTCUSDT | 2026-03-10 20:00 -> 2026-04-06 12:00 | 26 days 16:00
(+ 12 gaps of 8h–1d08h between 2017-09 and 2023-02, identical for both symbols)
```

The crypto 2019 hole is present in **both** stores, so it is not a divergence — it is a shared
defect inherited from the backfill.

**A1-F20 — Store authority: Postgres is the source of truth; Parquet is an audit/drift check, and
for equity it is stale by construction. USER RULING 2026-07-25. Confidence: high (ruling + query).**

The intended contract, stated so later passes do not have to re-derive it:

| Store | Role |
|---|---|
| **PostgreSQL** `ohlcv_daily` / `ohlcv_4h` | **Single source of truth.** Every consumer reads it and only it (**A1-F2**) |
| **Parquet** `data/{equity,crypto}/*.parquet` | **Audit / drift check only.** Not authoritative, not a fallback |

This resolves what "complete" means, and the answer differs by environment:

- **Crypto** — the two stores agree. A full row-by-row comparison of all 19,272 common bars per
  symbol shows **zero** mismatches on open/high/low/close, and volume differs on 17 BTCUSDT bars by
  ≤ 6 × 10⁻⁸ (float32 storage rounding, not a data difference). ETHUSDT is bit-identical on all
  five columns. The 203× / 198× volume seam of **A1-F9** is present in *both* stores at the same
  magnitude, confirming it lives in the source data rather than in either store.
- **Equity** — Parquet's extra 18 sessions do **not** make it the more complete store. It is
  Alpaca-derived, carries `Adjustment.ALL` prices, and inherits both the IEX volume era
  (**A1-F4**) and the probe island (**A1-F5**). Under the CBOE ruling it is superseded at source:
  the going-forward equity feed is CBOE, with **unadjusted** values and a deeper history. So the
  equity Parquet files are **stale and incomplete relative to the intended source**, and their
  18-session surplus is a *drift signal that the sync path failed*, not a reserve to restore from.

Read that way, **A1-F1** is not "Parquet is right and Postgres is wrong". It is: the two stores
disagree, nothing detects the disagreement, and neither store currently holds what the equity
series is supposed to contain.

**A1-F3 — Postgres never updates a row; Parquet always does. Confidence: high (code read).**

| Store | Write semantics | `file:line` |
|---|---|---|
| Parquet | upsert, **new data wins** | `parquet_store.py:53-59` — `concat([existing, new])` then `duplicated(keep="last")` |
| Postgres (ingest) | `INSERT … ON CONFLICT **DO NOTHING**` | `base.py:191` |
| Postgres (bulk Parquet load) | `DO NOTHING` | `base.py:383` |
| Postgres (crypto gap-fill) | `DO NOTHING` | `gap_fill.py:345` |

Consequences, all verified:

1. **A bad Postgres row can never be corrected by re-ingesting — and the guard is deliberate.**
   The three bad equity bars (**A1-F6**) survived the full 2026-03-11 backfill that covered their
   dates precisely because of this. So did the whole probe island. **USER RULING 2026-07-25:**
   `DO NOTHING` is *correct* — its purpose is to stop validated, audited history being overwritten
   by an unchecked re-fetch. The defect is therefore not the guard. It is that nothing in the
   system can adjudicate a *specific* row as bad and license its replacement → **A1-F26**.
2. **The stores diverge in values, not only in row counts.** `alpaca.py:126` requests
   `Adjustment.ALL`, so every dividend changes the historical adjusted price. A re-fetch therefore
   returns *different numbers* for old dates: Parquet takes them, Postgres discards them. The
   divergence widens on its own over time.

   **Measured this session.** Comparing our stored SPY `close` against unadjusted CBOE by date
   shows the adjustment as a clean step, not noise:

   | Window | CBOE − stored | Cause |
   |---|---|---|
   | 2026-04-07 → 2026-06-17 | flat **+25 to +28 bps** | one SPY quarterly dividend already applied to our bars |
   | 2026-06-18 → 2026-07-23 | **±2 bps**, mean ≈ 0 | past the ex-div date; nothing left to adjust |

   So a bar we stored in April is *today* worth 0.26 % less than the price that actually traded,
   and it will drift again next quarter. Verified independently against Alpaca itself on a symbol
   that split: for **AAPL 2020-08-28**, `adjustment=raw` returns close 499.23 / volume 53,137,957
   while `adjustment=all` returns close 121.06 / volume 212,551,828. The volume ratio is
   **exactly 4.0000** (`53,137,957 × 4 = 212,551,828`), the split factor; the price ratio is 4.124,
   the extra 3.1 % being six years of accumulated dividends. **Alpaca therefore adjusts volume by
   the split factor and does not apply the dividend factor to it** — correct behaviour, and a
   dependency that transfers to us under the CBOE ruling. None of the 8 ETFs split between 2016 and
   2026, so this has not yet fired; the first split will rewrite every historical price *and*
   volume, and **A1-F3** guarantees Postgres refuses the correction while Parquet accepts it.
3. `fetched_at` is a first-insert marker, not a last-updated marker — which is what made the
   vintage archaeology in **A1.2** possible.

Direction of drift is also asymmetric: normal ingestion writes **both** stores (`base.py:153-154`);
crypto gap-fill writes **Postgres only** (`gap_fill.py:309-348`, no Parquet call);
`scripts/repair_partial_bars.py` writes both.

**A1-F26 — Candle rows carry no audit state, so a gated correction is not expressible. NEW —
direction raised by the user 2026-07-25, evidence gathered here. Confidence: high (live DDL).**

Both candle tables are exactly 9 columns, read live from `pg16`:

```
ohlcv_daily | symbol, date,     open, high, low, close, volume, adjusted_close, fetched_at
ohlcv_4h    | symbol, datetime, open, high, low, close, volume, source,         fetched_at
```

Every column is a *value*. Nothing in either table says anything about a row's **standing** — how
it was obtained, whether it has been checked, or whether it has been superseded:

| Per-row state | `ohlcv_daily` | `ohlcv_4h` |
|---|---|---|
| Source / feed provenance | ❌ **absent** | ⚠️ `source` exists, **100 % NULL** (**A1-F13**) |
| When last audited | ❌ | ❌ |
| Flagged for review or update | ❌ | ❌ |
| Revision number | ❌ | ❌ |
| Ingestion time | `fetched_at` — **first-insert only** (**A1-F3**) | same |

Three consequences follow.

1. **Good and bad rows are indistinguishable in the schema.** The three invalid SPY bars of
   **A1-F6** differ from the 21,085 sound ones in no column whatsoever. There is no predicate that
   selects "rows known to be wrong", so there is no way to license a targeted overwrite while
   keeping the **A1-F3** guard for everything else. Every correction to date has had to happen by
   hand, outside the pipeline.
2. **Provenance is recoverable only by archaeology.** The SIP→IEX boundary of **A1-F4** was located
   this session by inferring it from `fetched_at` vintages plus an adjacent-day volume ratio —
   because no row records which feed produced it. The crypto table has the column purpose-built for
   this and never writes it, so the ~200× stitch of **A1-F9** is equally unqueryable. Both seams are
   properties of the data that the data itself does not state.
3. **`source` on crypto and not on equity is a fifth DS-7 divergence** — a schema asymmetry, not a
   behavioural one, and like the other four it is unexamined and unjustified.

No remedy is proposed here. Carried to the spec pass as item 11.

#### What validates each new record

Path for a normal incremental run (`base.py:120-162`): `fetch → validate → store(Parquet) →
_sync_to_db(Postgres) → _log_ingestion`. The 12-step `DataValidator` runs on this path — but see
**A1-F10** for what it actually catches, and **A1-F17** for the step that never runs at all.

**A1-F11 — Gap detection during ingestion cannot see a gap. Confidence: high (code read).**

Step 9 (`validation.py:261-296`) operates only on the **incoming batch**:

```python
# validation.py:285-289
start = df.index.min()... end = df.index.max()
expected = self._nyse.sessions_in_range(start, end)
missing = expected.difference(actual)
```

The comparison window is the fetched DataFrame's own min→max, so it can never detect a gap between
the store's last bar and the new batch. And a daily incremental fetch returns one row, which exits
at `validation.py:268-269` (`if len(df) < 2: return`) before any check runs.

**Measured: that early exit fires on every production run, in both environments.** The validator is
invoked per symbol, and each scheduled run delivers exactly one row per symbol:

| Run | Rows | Symbols | Rows per symbol |
|---|---|---|---|
| equity 2026-07-24 20:15 | 8 | 8 | **1** |
| equity 2026-07-23 20:15 | 8 | 8 | **1** |
| crypto 2026-07-25 16:01 | 2 | 2 | **1** |
| crypto 2026-07-25 12:01 | 2 | 2 | **1** |

So **Step 9 has effectively never executed on the production path.** The one larger run in the
recent window — equity 2026-07-22, 16 rows across 8 symbols — still gives 2 rows per symbol, and
`sessions_in_range` over a 2-row window cannot contain an interior hole either.

**And when it does fire, nothing follows.** `_detect_equity_gaps` terminates at
`log.warning("equity_gaps_detected", …)` (`validation.py:290-296`): no quarantine row, no return
value to the caller, no alert, no `data_ingestion_log` entry. It is a log line in a container whose
logs nothing queries.

**A1-F11 and A1-F12 fail in exactly complementary ways**, which is the clearest single illustration
of **A1-F22**:

| | `validation._detect_equity_gaps` (**A1-F11**) | `gap_fill.detect_equity_gaps` (**A1-F12**) |
|---|---|---|
| Method | ✅ **XNYS session comparison** (`:285-289`) | ❌ calendar-day arithmetic |
| Scope | ❌ incoming batch only | ✅ **the whole Postgres table** |
| Wired up | ✅ runs on every ingest | ❌ **no caller** |
| Consequence | ❌ `log.warning` only | ❌ n/a |

Every piece of correct equity gap detection exists — the right algorithm in one function, the right
scope in the other, and a live call site on a third axis. **No single function has two of the
three.**

**A1-F12 — Equity has no gap-fill at all. DS-7 parity defect. Confidence: high (code read).**

`gap_fill.py` defines `detect_equity_gaps` (`:123-168`) — which correctly reads **Postgres** and
would find the 18-session hole — but there is no equity filler and nothing calls it. The
orchestrator `detect_and_fill_crypto_gaps` (`:356`) is crypto-only. Correspondingly:

- `candles_crypto_job` calls `detect_and_fill_crypto_gaps` between ingest and features.
- `candles_equity_job` (`collector_main.py:303-330`) calls nothing of the kind.

**No production caller, confirmed by search.** The only references are `tests/data/test_gap_fill.py`
and — misleadingly — `validation.py:272,278`, which is a *different* function: the private method
`DataValidator._detect_equity_gaps`, not this module-level one.

**Measured at its default threshold** of `timedelta(days=5)` (`gap_fill.py:45`), against all
21,088 live rows:

| Calendar-day gap | Occurrences | Fires at > 5 d? |
|---|---|---|
| 1 day | 16,488 | no |
| 2 days (mid-week holidays) | 216 | no |
| 3 days (weekends) | 3,784 | no |
| 4 days (long weekends) | 584 | no |
| **28 days** | **8** | ✅ all 8 symbols, 2026-03-10 → 2026-04-07 |

So it would have found the **A1-F1** hole exactly, with zero false positives. Two limitations
qualify that, both established this session:

1. **It is not calendar-aware.** The comparison is plain date subtraction — `gap = curr_dt -
   prev_dt` against a `timedelta` — and `gap_fill.py` imports no calendar at all. It works here
   only because no legitimate closure in this dataset exceeds 4 calendar days, which is an
   empirical property of the window, not a structural guarantee.
2. **It cannot detect a short hole.** At a 5-day threshold it needs roughly **four consecutive
   missing sessions** to fire. A single failed ingest — the likeliest failure mode — leaves a 2-day
   calendar gap, indistinguishable from the 216 legitimate mid-week holidays. Lowering the
   threshold is not available either: 4,584 legitimate 2–4 day gaps would flood it. Only comparison
   against the session list separates the two, and that logic exists at `validation.py:285-289`
   but is confined to the incoming batch (**A1-F11**).

**A1-F13 — Crypto gap-fill bypasses the validator, and has never inserted a row.
Confidence: high (code read + query).**

`_insert_gap_fill_to_db` (`gap_fill.py:309-348`) goes straight from the Binance Global HTTP
response to `executemany_from_df` — no `DataValidator`, no quarantine. It tags rows
`source = 'binance_global'` (`:339`).

But `ohlcv_4h.source` is **100 % NULL** across all 38,545 rows:

```sql
SELECT symbol, source, COUNT(*) FROM ohlcv_4h GROUP BY 1,2;
-- BTCUSDT | (null) | 19273
-- ETHUSDT | (null) | 19272
```

So the crypto gap-filler has never successfully written anything, despite the 2026-03-11→04-06 gap
(26 days) and the 2019 stitch hole (22 days) both exceeding its 24 h threshold and both being
visible to `detect_crypto_gaps`, which reads the correct store. It runs every 4 hours and repairs
nothing.

**Root cause established** (see A1.2): its source host `https://api.binance.com` is **geo-blocked
from this host**, returning `"Service unavailable from a restricted location"`. The filler is
unreachable by construction, not intermittently failing. The archive CDN it does *not* use
(`data.binance.vision`) is reachable from the same machine.

#### What would detect a silent failure

**A1-F18 — Every silent-failure detector is either too loose to fire or reads a maximum.
Confidence: high (code read).**

| Detector | `file:line` | Why it cannot catch this |
|---|---|---|
| `_check_equity_rows` | `verification.py:74-105` | Threshold is `> 100` rows. Actual: 2,636/symbol |
| `_check_crypto_rows` | `verification.py:108-137` | Same `> 100` threshold. Actual: ~19,272/symbol |
| Verification gap check | `verification.py:209-230` | **Crypto only.** No equity equivalent |
| Verification freshness | `verification.py:381-383` | `MAX(date)` / `MAX(datetime)` — interior-gap blind |
| Trader freshness guard | `execution/pipeline.py:1179-1215` | `MAX(date)`, log-only (`never raises, never halts`), and **equity only** — the docstring states it |
| Ingestion log | `base.py:230-279` | Records `rows_inserted`, but nothing reads or alerts on it |
| Collector INFO alert | `collector_main.py:316-325` | Reports `rows_added=N`. `N=0` is indistinguishable from a legitimate no-new-bar day |

`run_equity` / `run_crypto` (`ingest_all.py:87-98`, `:120-132`) compute `rows_added` as a
**Postgres** row-count delta, which is the right store. But the delta is clamped
(`max(0, after − before)`) and a zero delta produces a success alert, not a failure.

**Three of the seven verification checks have no failing branch at all.** Enumerating every
`CheckResult` return in `verification.py`:

| Check | Can it return `passed=False`? |
|---|---|
| `equity_rows` (`:101`) | ✅ — but only at **≤ 100 rows** |
| `crypto_rows` (`:133`) | ✅ — same threshold |
| `macro_series` (`:157`) | ✅ |
| `obs_vector` (`:290`, `:296`) | ✅ |
| `equity_date_range` | ❌ **hardcoded `passed=True`** (`:183`, sole return) |
| `crypto_date_range` | ❌ **hardcoded `passed=True`** (`:205`, sole return) |
| `crypto_gaps` | ❌ **hardcoded `passed=True`** (`:268`, sole return) |

The docstrings say so explicitly — `:172` and `:194` both read *"CheckResult always passed=True"*.

**`_check_crypto_gaps` is the consequential one.** It is the only interior-gap detection anywhere
in the verification module, and it *works*: it walks consecutive timestamps, compares against
`_CRYPTO_GAP_THRESHOLD`, accumulates `gap_reports` and logs `crypto_gap_remaining` at warning level
(`:246-255`). Then it returns `passed=True` regardless of what it found. The 22-day and 26-day gaps
of **A1-F19** and **A1-F1** would be detected, described, logged — and passed.

This is a distinct defect from the vacuous run of **A1-F23**, and they stack: **A1-F23** shows the
check inspected zero symbols; this shows that **giving it the correct symbols would still not make
it fail.**

Two further properties of the same surface:

1. **The row thresholds require near-total destruction to trip.** `≤ 100` against live counts of
   2,636 equity and 19,272 crypto rows per symbol means **96.2 %** and **99.5 %** of the table must
   disappear before the check fires. The 18-session hole of **A1-F1** is 0.7 %.
2. **`_latest_date` fails to a hardcoded date.** `verification.py:378` sets
   `_FALLBACK = "2024-01-15"`, returned on any query exception. A total database failure therefore
   yields a plausible-looking date rather than an error, and the freshness comparison then runs
   against a fabricated value.

The historical trace shows exactly this signature. Between 2026-03-13 and 2026-04-01 every recorded
equity run reports `status = 'success'` with `rows_inserted = 0`:

```sql
SELECT date_trunc('day',timestamp) d, environment, status, COUNT(*) n, SUM(rows_inserted)
FROM data_ingestion_log WHERE environment IN ('equity','crypto') GROUP BY 1,2,3 ORDER BY 1 DESC;
-- 2026-03-13 | equity | success | 8  | 0
-- 2026-03-14 | equity | success | 18 | 0
-- 2026-03-26 | equity | success | 4  | 0
-- 2026-04-01 | equity | success | 22 | 0
-- 2026-04-06 | equity | no_data | 16 | 0
```

Caveat, stated plainly: those runs cover only **SPY, XLE and QQQ** and repeat minutes apart, with
`XLE` failing every time and durations of 6–21 ms — the signature of **development runs, not the
production collector**. The collector's candle jobs did not exist until the 2026-07-18 ruling. So
the honest reading is:

- **Verified:** Postgres has the 18-session hole; Parquet does not; the ingestion log records no
  production 8-symbol run with `rows_inserted > 0` in that window; the newest bar either side is
  fine, so no `MAX`-based detector could ever have fired.
- **UNVERIFIED:** the precise mechanism that wrote Parquet but not Postgres during 2026-03-11 →
  2026-04-06. `_sync_to_db` swallows every exception and returns 0 with only a `log.warning`
  (`base.py:195-197`), which is a sufficient mechanism, but no log from that period was available
  to confirm it fired.

One detector that *does* work: `scripts/repair_partial_bars.py` targets 4H bars whose `fetched_at`
falls inside their own window. A live check returns **0 rows**, so no partial bars are currently
stored. It is deliberately crypto-only — the equity daily path is structurally immune
(`repair_partial_bars.py:23-26`).

**A1-F21 — The candle layer raises no alerts at all. `alert_log` exists, works, and is never
written by any candle integrity check — because no such check exists. NEW.
Confidence: high (exhaustive grep + live table query).**

The alerting infrastructure is present and functioning. `alert_log` is a real table
(`postgres_schema.py:513`), `Alerter._write_alert_log` writes to it (`alerter.py:365-393`), and
the collector constructs its alerter with a live `db` handle (`collector_main.py:118`), so writes
are not being silently dropped.

But `send_alert` appears **nowhere** in the candle data layer. A grep across all nine modules that
own candles returns zero hits:

```
alpaca.py  binance.py  base.py  validation.py  gap_fill.py
parquet_store.py  ingest_all.py  verification.py  cross_source.py
          -> 0 occurrences of send_alert
```

Within `src/swingrl/data/`, only `calendar.py:372` and `data/options/*` alert at all. Every candle
alert ever sent came from the **job wrapper** (`collector_main.py:326`, `:333`, `:364`, `:374`),
which reports whether the job ran — not whether the data is sound.

The full alert history for candles, from `alert_log`:

```sql
SELECT title, COUNT(*) n, MIN(timestamp), MAX(timestamp) FROM alert_log
WHERE title ILIKE '%candle%' OR title ILIKE '%ingest%' OR title ILIKE '%gap%' OR title ILIKE '%stale%'
GROUP BY 1;
-- Crypto candles ingested | 38 | 2026-07-19 08:01 | 2026-07-25 12:01
-- Equity candles ingested |  5 | 2026-07-20 16:50 | 2026-07-24 20:15
```

43 of the 121 rows in `alert_log`. Three properties of that set:

1. **Every one is INFO-level and reports success.** Not one warning or error about candle data has
   ever been raised — no gap alert, no staleness alert, no validation alert, no store-divergence
   alert. The failure branches at `collector_main.py:333` and `:374` have never fired.
2. **The history begins 2026-07-19/20**, four months *after* the 18-session hole opened on
   2026-03-11. Candle alerting did not exist when the defect occurred, so nothing could have caught
   it even in principle. This closes the loop on **A1-F18**.
3. **The message body is not stored.** `alert_log` holds `alert_id, timestamp, level, title,
   message_hash, sent` — the body is hashed. So `rows_added=0` and `rows_added=8` produce the same
   title and are indistinguishable in the log after the fact. The one signal that would have
   exposed the sync failure is not retained.

Taken with **A1-F12**, the position is not that integrity checking is hard here — it is that
correct, Postgres-reading, interior-gap-aware detection **already exists** in `gap_fill.py:123-168`
and is connected to nothing. That pattern is general enough to be its own finding — see
**A1-F22**.

**A1-F22 — Most of the integrity machinery already exists, reads the right store, and is wired to
nothing. NEW. Confidence: high (grep + call-site trace for every row).**

This is the single most important structural observation in the A1 review. The gap between what
SwingRL *has* and what SwingRL *runs* is not a gap in capability — it is a gap in wiring. Full
inventory, every row traced to its call sites this session:

| Capability | Where | Reads the authoritative store? | Wiring status |
|---|---|---|---|
| Equity interior-gap detection — **calendar-day arithmetic, not XNYS-aware** | `gap_fill.py:123-168` | **Yes — Postgres** | ❌ **No caller in `src/` or `scripts/`** — tests only. Blind to holes shorter than ~4 sessions (**A1-F12**) |
| Crypto interior-gap detection | `gap_fill.py:75-120` | **Yes — Postgres** | ⚠️ Called, but only to feed a geo-blocked filler |
| Crypto gap fill | `gap_fill.py:269-348` | writes Postgres | ❌ Source geo-blocked; 0 rows ever written |
| Equity gap fill | — | — | ❌ **Does not exist** |
| 12-step row + batch validator | `validation.py` | in-flight batch only | ⚠️ Runs, but 0 candles ever quarantined; no flat-bar or volume-plausibility check |
| Cross-source price check (yfinance) | `cross_source.py`, gated at `validation.py:235` | Postgres | ❌ **Never runs** — validator built without `db`/`config` (**A1-F17**) |
| Row-count / date-range / gap verification | `verification.py` | **Yes — Postgres** | ❌ Collector never calls it; and it passes vacuously (**A1-F23**) |
| Stitch validation | `binance.py:390-442` | in-flight | ⚠️ Price-only; return value discarded (**A1-F16**) |
| Partial-bar detection + repair | `scripts/repair_partial_bars.py` | **Yes — both stores** | ✅ **Works.** Crypto-only by design |
| Trader freshness guard | `execution/pipeline.py:1179-1215` | **Yes — Postgres** | ⚠️ `MAX()`-based, log-only, equity-only |
| Per-run ingestion log | `base.py:230-279` → `data_ingestion_log` | **Yes — Postgres** | ⚠️ Written faithfully; **nothing ever reads it** |
| Alert log | `alerter.py:365-393` → `alert_log` | **Yes — Postgres** | ⚠️ Works; no candle integrity check writes to it (**A1-F21**) |

Exactly **one** of twelve is both correct and connected. The `data_ingestion_log` row deserves
emphasis: a `grep` for `FROM data_ingestion_log` across `src/` and `scripts/` returns **no SELECT
anywhere** — it is a write-only table. Every ingest for four months faithfully recorded
`rows_inserted=0`, and nothing was ever positioned to read it.

**A1-F23 — The verification gate passes vacuously, treats "no data" as success, and its only
artefact on the production runtime is failed test output. NEW.
Confidence: high (code read + live artefact) / medium (attribution of the run).**

Two defects, both in `verification.py`:

1. **An empty symbol list passes.** `_check_equity_rows` (`:91`) iterates `config.equity.symbols`.
   With an empty list the loop body never executes, `problems` stays empty, and the check returns
   `passed=True`.
2. **Absent data passes.** The date-range checks return `passed=True` with detail
   `"No equity data found"`.

The live artefact at `data/verification.json` on the production runtime (2026-07-24 09:18) shows
both firing at once:

```json
{"name": "equity_rows",       "passed": true,  "detail": "All 0 equity symbols present with >100 rows"}
{"name": "crypto_rows",       "passed": true,  "detail": "All 0 crypto symbols present with >100 rows"}
{"name": "equity_date_range", "passed": true,  "detail": "No equity data found"}
{"name": "crypto_date_range", "passed": true,  "detail": "No crypto 4H data found"}
{"name": "crypto_gaps",       "passed": true,  "detail": "No gaps >24h in crypto 4H data"}
```

The `crypto_gaps` line is the clearest illustration: it reports no gaps >24 h at a moment when the
table demonstrably holds two, of 22 days and 26 days (**A1-F19**, **A1-F1**). It "passed" because
it inspected zero symbols.

Two further observations on the same artefact:

- **It is test output sitting in the production data directory.** One failure reads
  `Input should be a valid string [type=string_type, input_value=<MagicMock name='mock.sen...'>]`,
  so the run used a mocked config. **UNVERIFIED:** which run wrote it — the project convention of
  using `~/swingrl` as both the production runtime and the CI bed
  (`CLAUDE.md`, git workflow step 4) is a sufficient explanation, but was not confirmed.
- **The report says `"passed": false` overall and nothing acted on it.** It has sat on disk for a
  day. Consistent with **A1-F21**: no path exists from a verification result to an alert.

Finally, the gate is unreachable from the running system regardless: `run_verification` is invoked
only at `ingest_all.py:264`, inside the full `main()` sequence, and the collector imports just
`detect_and_fill_crypto_gaps`, `run_crypto`, `run_equity`, `run_features`
(`scripts/collector_main.py:28-32`). It is not part of any scheduled job.

**A1-F24 — Metadata-table inventory: the collector *does* write `data_ingestion_log`, but the
schema has nowhere to put an integrity fact — and two purpose-built tables sit empty. NEW.
Confidence: high (live counts + writer/reader grep for every table).**

Complete inventory of the database metadata surface relevant to candles:

| Table | Rows | Written by | Read by | Fit for candle integrity |
|---|---|---|---|---|
| `data_ingestion_log` | **1,562** | `base.py:258`, `calendar.py:71` | ❌ **nothing** (no `SELECT` anywhere) | Per-run telemetry only — see below |
| `data_quarantine` | 3,939 | `base.py:314` | ❌ nothing | Row-level rejects. **0 candle rows** (**A1-F10**) |
| `alert_log` | 121 | `alerter.py:379` | ❌ nothing | 43 candle rows, all INFO success; **body hashed** (**A1-F21**) |
| `system_events` | **0** | ❌ **nothing, anywhere** | ❌ nothing | **Defined in schema and never used** |
| `api_errors` | **0** | `execution/adapters/binance_sim.py:345` only | `execution/emergency.py:518` (IP-ban detection) | Execution path only; the data layer never writes it |

Three observations, each with a consequence for the spec pass.

**1. The collector already writes `data_ingestion_log` — indirectly.** It does so through
`BaseIngestor._log_ingestion` (`base.py:230-279`), not from `collector_main.py` itself. Live proof
at the collector's own cadence:

```
 crypto | ETHUSDT | success | 1 | 2026-07-25 12:01:00.17984-04
 crypto | BTCUSDT | success | 1 | 2026-07-25 12:01:00.12553-04
```

What is **not** logged is everything the collector does around the ingest: `detect_and_fill_crypto_gaps`
writes no row, the feature recompute writes no row, and no job-level outcome is recorded. So the
table describes *fetches*, not *the collector's work*.

**2. The schema cannot hold an integrity fact.** All nine columns are run telemetry:

```
run_id | timestamp | environment | symbol | status | rows_inserted | errors_count | duration_ms | binance_weight_used
```

There is no column in which "18 sessions missing versus the XNYS calendar", "store drift = 18 rows"
or "calendar completeness 99.32 %" could be expressed. Recording integrity and quality here is
therefore a **schema change**, not a wiring change — the only such item found in the whole A1
review, and it interacts with the A30 additive-only-migrations constraint.

**3. `system_events` is exactly the right shape and is completely empty.** Its columns —
`event_id, timestamp, level, module, event_type, message, metadata_json` — are a general typed
event log with a JSON payload. A grep across `src/` and `scripts/` finds **no writer at all**; it
exists only in `postgres_schema.py`. It has never held a row.

Related, and directly relevant to **A1-F13**: `api_errors` is written only from the execution
adapter (`binance_sim.py:345`) and read only by emergency IP-ban detection
(`emergency.py:518`). The **data layer never writes it**, which is why the geo-block on
`api.binance.com` — an HTTP-level rejection returned on every gap-fill attempt for months — left no
trace in any table.

**A1-F25 — The five metadata tables are five independent designs for one concept. They accreted
per-feature rather than being modelled. NEW — observation raised by the user 2026-07-25, evidence
gathered here. Confidence: high (DDL read + migration history).**

Every one of the five records the same underlying thing — *something happened while handling data,
here is when, how bad, and what it concerned* — and no two agree on how to say it. Full DDL
comparison, from `postgres_schema.py`:

| | `data_ingestion_log` :92 | `data_quarantine` :80 | `alert_log` :513 | `system_events` :433 | `api_errors` :533 |
|---|---|---|---|---|---|
| **Primary key** | `TEXT run_id` | `INTEGER` identity | `TEXT alert_id` | `TEXT event_id` | `INTEGER` identity |
| **Time column** | `timestamp` | **`quarantined_at`** | `timestamp` | `timestamp` | `timestamp` |
| **Time nullability** | default, nullable | default, nullable | `NOT NULL`, no default | `NOT NULL`, no default | `NOT NULL`, no default |
| **Severity expressed as** | `status` | `severity` | `level` | `level` | `status_code` |
| **Subject identified by** | `environment` + `symbol` | **`source`** + `symbol` | `title` | `module` + `event_type` | `broker` + `endpoint` |
| **Payload** | *(none)* | `raw_data_json` | **`message_hash`** — lossy | `metadata_json` | `error_message` |

That is 2 primary-key strategies, 2 time-column names, 2 nullability conventions, **4 vocabularies
for severity**, 5 schemes for naming the subject, and 4 payload strategies — one of which discards
the payload.

The sharpest single instance: `data_ingestion_log.environment` and `data_quarantine.source` hold
**the same values** — `equity`, `crypto`, `macro` — under different column names. A query that
joins an ingest run to the rows it quarantined has to know that.

Two pieces of evidence that this is accretion rather than design:

1. **None of the five was ever migrated in.** `schema_migrations` holds 11 rows, all applied
   2026-07-18 or later (`registries_era0` → `pending_order_lifecycle`), and **not one** creates any
   of these tables. All five predate the migration system, arriving in the initial DDL blob during
   the DuckDB→PostgreSQL move.
2. **They are not even adjacent in the schema file.** They sit at lines 80, 92, 433, 513 and 533,
   interleaved with `corporate_actions` (:444), `inference_outcomes` (:523) and `emergency_flags`
   (:543). There is no metadata section — each was appended beside whatever feature needed it.

The functional consequences are already recorded elsewhere and are symptoms of this same root:
`alert_log` cannot carry evidence because its payload is hashed (**A1-F21**); `data_ingestion_log`
has no column for an integrity fact (**A1-F24**); `system_events` is the only one shaped for
general use and has zero writers (**A1-F24**); and an HTTP failure in the data layer has no table
that will accept it (**A1-F13**).

**No merged model is proposed here** — that is a spec-pass decision, carried as item 10. What this
review establishes is that the incoherence is real, structural, and measurable, not stylistic.

**A1-F28 — No independent cross-source validation exists in either environment. Equity's is built
and dead; crypto's has never existed, so Binance is unfalsifiable. NEW — direction raised by the
user 2026-07-25, evidence gathered here. Confidence: high (code read + live measurement).**

Nothing in the running system ever compares a stored candle against a second, independent source.

| Environment | Second source | State |
|---|---|---|
| Equity | yfinance, via `cross_source.py` → `validation.py:235` | ❌ **Built but never runs** — the validator is constructed without `db`/`config` (**A1-F17**) |
| Crypto | — | ❌ **Does not exist.** No module, no config, no second endpoint |

The crypto case is the more serious of the two, because it is not a wiring gap but an absence.
Binance is the *only* crypto source, so every crypto finding in this review had to be confirmed
against Binance itself:

- **A1-F7**'s fourteen flat bars were established as genuine by calling the Binance.US API and
  receiving the same bars back. That establishes the store faithfully reflects the vendor. It says
  nothing about whether the vendor is right — a venue's own first-listing prints cannot corroborate
  themselves.
- **A1-F9**'s ~200× volume seam is present identically in both stores, which proves it lives in the
  source data and not in our handling — and leaves no way to tell which side of the seam, if
  either, states the true traded volume.

**The equity side, by contrast, now has a demonstrated second source.** Measured this session
against CBOE:

| Test | Result |
|---|---|
| Price agreement, post-ex-div window | **±2 bps**, mean ≈ 0 (see **A1-F3**) |
| Price agreement, pre-ex-div window | offset by exactly one dividend — a *comparison artefact*, not a disagreement |
| Volume agreement, SIP era | median ratio **0.968** — same basis, 3.2 % apart |
| Volume agreement, IEX era | median **61.3×** — the **A1-F4** seam, correctly surfaced |

Two constraints on using Alpaca as the validator, both measured rather than assumed:

1. **Prices — usable, with a caveat.** Agreement is ~2 bps only when comparing like with like. Our
   `close` is `Adjustment.ALL`; CBOE is unadjusted. A naive comparison reports a false
   26 bps discrepancy that is purely the dividend adjustment.
2. **Volume — not usable as configured.** The live equity path runs `since="incremental"` → IEX
   (`alpaca.py:104-107`), which reads 12–53× below CBOE. Volume cross-validation requires Alpaca on
   the **SIP** feed, which that branch selects only on a backfill call.

No remedy is proposed here. Carried to the spec pass as item 12.

---

### A1.4 — The calculation

Candles are **raw stored data**, not a computed quantity. The transformations between the API
response and the stored row are these, and no others:

| Step | Equity `file:line` | Crypto `file:line` | What it does |
|---|---|---|---|
| Feed / venue selection | `alpaca.py:104-107` | `binance.py:37,45` (US API vs Global archive) | **Source of A1-F4 / A1-F9** |
| Price adjustment | `alpaca.py:126` `Adjustment.ALL` | none — crypto has no corporate actions | Splits + dividends only; **not spin-offs (A1-F15)** |
| Index flattening | `alpaca.py:144-145` | — | Drops Alpaca's `symbol` MultiIndex level |
| Column projection | `alpaca.py:148-149` | `binance.py:266` | Keeps o/h/l/c/volume |
| Volume normalisation | none | `binance.py:218` (API: `volume_quote`); `binance.py:383-386` (archive: `volume_quote`, else `volume_base × close`); `gap_fill.py:265` (`volume_quote`) | Units consistent across all three crypto paths — **verified correct** |
| Incomplete-bar guard | `alpaca.py:108` (`end` capped to UTC midnight) | `binance.py:272-278`, `:297-298` (`_incomplete_bar_open_ms`) | Prevents storing a still-forming bar |
| Symbol/date shaping | `base.py:212-220` | `base.py:217-220` | Adds `symbol`; `date` from index for daily, `datetime` for 4H |

**Correct.** Volume-unit handling on the crypto side is consistent across all three write paths —
this was checked specifically because a units mismatch would have looked identical to **A1-F9**,
and it is not the cause. The incomplete-bar guards are present on both sides and currently
effective (zero partial bars stored).

**Dead.** `alpaca.py:78` (`self._validator`, never read). `ohlcv_daily.adjusted_close`
(**A1-F14**). `detect_equity_gaps` (**A1-F12**). `sync_parquet_to_duckdb` alias (`base.py:390`).
`_duckdb_table` naming throughout (`alpaca.py:59`, `base.py:57`) — legacy from the pre-Postgres era.

**A1-F14 — `adjusted_close` is empty and redundant. Confidence: high (query + code).**

```sql
SELECT COUNT(*) total, COUNT(adjusted_close) adj_nonnull FROM ohlcv_daily;
-- 21088 | 0
```

Nothing writes it: `_OHLCV_DAILY_COLUMNS` (`base.py:30`) omits it entirely. Nothing reads it
either — a grep for `adjusted_close` across `src/` and `scripts/` returns hits only in
`postgres_schema.py`, where it is defined. It has been empty since the table was created.

It is also conceptually redundant — `alpaca.py:126` requests `Adjustment.ALL`, so the `close`
column already holds an adjusted price. Worse than unused: the column's presence next to `close`
implies `close` is *raw*, which is false, so the schema actively misleads about what the main price
column contains.

**USER RULING 2026-07-25 — the column is to be removed.** It holds no data and the adjusted series
will be derived at read time under carry item 13, so there is nothing for it to hold in either the
current or the intended design. Two constraints recorded for the spec pass, not resolved here:

- A `DROP COLUMN` is **not additive**, so it sits outside the A30 additive-only rule that applies
  while the trader runs. The practical risk is minimal — zero writers, zero readers, zero values —
  but the rule is a standing project constraint and the exception should be explicit rather than
  assumed.
- The natural moment is the same migration that adds the audit-state columns of carry item 11:
  one change to the candle table rather than two.

**Duplication.** The Postgres-loading logic exists twice in near-identical form:
`BaseIngestor._build_sync_df` (`base.py:199-228`) and the module-level `sync_parquet_to_db`
(`base.py:341-386`). The second bypasses validation; the first does not.

---

### A1.5 — The pipeline and its wiring

```
                    Alpaca SIP ──┐                    Binance Global archive ──┐
                    Alpaca IEX ──┤                    Binance.US API ──────────┤
                                 v                                             v
                        AlpacaIngestor                              BinanceIngestor
                       (alpaca.py:80-181)                          (binance.py:240-345)
                                 │                                             │
                                 └──────────► BaseIngestor.run (base.py:120) ◄─┘
                                                       │
                          ┌────────────────────────────┼───────────────────────────┐
                          v                            v                           v
                  DataValidator                   ParquetStore              _sync_to_db
                 (validation.py)                 upsert keep-last         DO NOTHING
                 0 candles ever                  data/{equity,crypto}/    ohlcv_daily
                 quarantined                            │                 ohlcv_4h
                       │                                │                      │
                       v                                v                      │
                data_quarantine                   ◄── WATERMARK ──►            │
                 (macro only)                     alpaca.py:220                │
                                                  binance.py:256               │
                                                                               │
   crypto gap-fill ──► Binance Global ──► DO NOTHING, no validation ──────────►┤
   (gap_fill.py, crypto only, source column proves 0 rows written)             │
                                                                               v
                                            ┌──────────────────────────────────┴─────────┐
                                            v                                            v
                                  FeaturePipeline                              Trader / scheduler
                              (pipeline.py:659, :681)                    execution/pipeline.py:1192
                                            │                            scheduler/jobs.py:374-375
                                            v                            shadow/shadow_runner.py:35
                                  features_equity / features_crypto
                                            │
                                            v
                                  Trainer (data_loader.py:165)
```

**The divergence that matters:** the watermark and the consumers read different stores
(**A1-F2**), and those stores have different write semantics (**A1-F3**). Nothing in the system
compares them. There is no reconciliation job, no row-count assertion across stores, and no
checksum — verified by the absence of any Parquet read outside the two watermarks and the partial-bar
repairer.

Against the intended contract of **A1-F20** — Postgres authoritative, Parquet an audit/drift check
— the wiring is inverted in one specific place: the *only* production reads of Parquet are the two
watermarks, i.e. the audit store is steering ingestion while the source of truth is never consulted
about how far it has got. That is the mechanism by which **A1-F1** became permanent.

**Trainer vs trader divergence:** none at the candle layer. Both read the same two Postgres tables
through the same feature pipeline. The only asymmetries are in *guards*, not data:
the trader has a freshness warning that the trainer lacks, and it is equity-only.

---

### A1.6 — Use, current and planned

#### Observation-space accounting

Candles occupy **zero direct observation slots**. Layout from `features/assembler.py:38-51`:

| Block | Equity | Crypto |
|---|---|---|
| Per-asset features | 15 × 8 = **120** | 13 × 2 = **26** |
| Macro | 6 | 6 |
| HMM regime | 2 | 2 |
| Turbulence | 1 | 1 |
| Overnight context | — | 1 |
| Portfolio state | 35 | 11 |
| **Total** | **164** | **47** |

Candles are the **upstream input** to three of those blocks:

1. **Per-asset features (A2).** All 15 equity and 13 crypto per-asset names
   (`assembler.py:117-149`) are derived from OHLCV — `price_sma50_ratio`, `rsi_14`, `macd_line`,
   `bb_position`, `atr_14_pct`, `adx_14`, and so on.
2. **Turbulence (A5).** Computed from OHLCV-derived log-returns
   (`features/pipeline.py:610-644`), explicitly *not* stored in `features_*`.
3. **Portfolio valuation.** `close` drives the price pivot the simulator marks positions against
   (`data_loader.py:177-198`).

**Where the seams land.** `volume_sma20_ratio` is the one per-asset feature that reads volume
directly. It therefore carries **A1-F4** (equity, 30–90× step at 2020-07-27) and **A1-F9** (crypto,
~200× step at 2019-09-01) straight into the observation: **8 of 164** equity dimensions and
**2 of 47** crypto dimensions. Prices are unaffected by both seams; the affected quantity is volume
magnitude only. Whether the downstream normalisation neutralises the step is an **A2** question and
is explicitly not answered here.

#### How the trainer consumes candles today

`training/data_loader.py:157-183` — features are gated on candle dates by an inner join:

```sql
FROM features_equity f
INNER JOIN (SELECT DISTINCT date FROM ohlcv_daily) o ON f.date = o.date
```

so the 18 Postgres-missing sessions are excluded from training **even if `features_equity` holds
rows for them**. A second query pulls `close` per symbol for the price pivot
(`data_loader.py:175-183`). Crypto follows the same shape at `:292` and `:305`.
`scripts/train.py:244,260,369,385` and `scripts/train_pipeline.py:2104-2108` repeat the pattern.

#### How the trader consumes candles today

- **Features** — same `FeaturePipeline` reads (`pipeline.py:659`, `:681`); no separate trader path.
- **Freshness guard** — `_warn_if_stale_ohlcv` (`execution/pipeline.py:1179-1215`) compares
  `MAX(date)` against the previous XNYS session. Log-only, never halts, equity-only.
- **Benchmark baselines** — latest `close` per symbol (`scheduler/jobs.py:374-375`;
  `scripts/record_benchmark_baselines.py:50-51`).
- **Shadow runner** — `shadow/shadow_runner.py:35` selects the table by environment.
- **Backup guard** — `backup/duckdb_backup.py:28` treats both tables as required.

#### Planned use

The training redesign moves Group B datasets into the observation space, which raises the number of
consumers downstream of candles but does not change the candle contract itself.

**USER RULING 2026-07-25 — equity candles move to CBOE.** CBOE becomes the going-forward equity
source: a more complete history with **unadjusted** values. Recorded here as direction, not as
implemented state — no candle code path references CBOE today, which appears in
`src/swingrl/data/options/` exclusively. Three consequences follow for later passes, stated as
consequences and not as a design:

1. It supersedes both current equity vintages. The SIP era, the IEX era (**A1-F4**) and the probe
   island (**A1-F5**) are all Alpaca artefacts that a CBOE-sourced series would not carry.
2. Unadjusted values change what `close` means. Today `close` is split- and dividend-adjusted
   (`alpaca.py:126`), which is why `adjusted_close` is redundant (**A1-F14**) and why re-fetches
   return different numbers for old dates (**A1-F3**). Unadjusted values invert both properties.
3. It settles **A1-F20** for equity: Parquet is not a restore path, because it holds the superseded
   feed.

**A1-F27 — CBOE's volume basis is confirmed consolidated, but the source carries ~50
uncharacterised defects of its own. NEW. Confidence: high (live payloads, all 8 symbols).**

All 8 symbols were fetched this session from
`https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json` (HTTP 200,
~5,676 bars each, 2004-01-02 → 2026-07-23) and compared bar-by-bar against Postgres.

**What the ruling gets right, now measured rather than assumed:**

| Property | Result |
|---|---|
| Volume basis vs Alpaca **SIP** | median ratio **0.968** — consolidated-tape scale, 3.2 % lower |
| Volume basis vs Alpaca **IEX** | median **61.3×** — consistent with the **A1-F4** seam |
| Adjustment state | **Unadjusted**, confirmed — see the dividend step in **A1-F3** |
| History depth | **2004-01-02**, ~5,676 bars/symbol vs our 2,636 — 2.15× deeper |
| SPY 2018-11-01 (**A1-F6**'s flat bar) | CBOE reads **99,495,037** — the source **corrects** that defect |

**What the ruling does not yet account for.** "One source, no stitching, no seams" is true of the
*seam*. It is not true of *data quality*:

| Defect | Count | Detail |
|---|---|---|
| OHLC invariant violations | **36**, all 8 symbols | Same class as **A1-F6**. Clustered on shared dates — 2015-03-31 hits 6 symbols, 2022-02-22 hits 4, 2023-06-05 hits 3. One is recent: QQQ **2026-05-20**. **6 of the 36 fall inside the validator's tolerance band** and would pass (**A1-F10**) |
| Non-session bars — **2018-11-22**, Thanksgiving | **7** | Flat, zero-volume bars on 7 of 8 symbols. Five repeat the prior session's close exactly; **XLE and XLK are doubled** (2.069× and 2.000×) and revert the next day |
| Non-session bars — **2025-05-26**, Memorial Day | **8** | The dangerous set: **full OHLC and plausible volume** — SPY reads **75,990,006 shares traded on a day the market was closed**. Six repeat the prior close; XLE and XLK are again doubled. Nothing in the validator inspects it (**A1-F10**) |
| Corrupt 2019-01 window | **30 bars** | SPY 2019-01-07 reads **69,304** against a real 105,208,591 — a **1,518×** under-report. CBOE's own localised version of the **A1-F4** failure mode |
| Duplicate rows | 3 symbols | 2024-12-31 appears twice in SPY, VTI and XLV, with identical values |
| Missing sessions | **4**, XLI only | Checked against the XNYS calendar (`exchange_calendars`, 5,674 sessions 2004-01-02 → 2026-07-23). Every other symbol is calendar-complete. Confirms the prior spec's note |

Calendar verdict in full: **15 non-session bars across 2 dates, and 4 missing sessions in one
symbol.** Postgres today has **zero** non-session dates (**A1.1**) — a property of what Alpaca
supplied, not of anything the pipeline enforces (**A1-F10**).

Stated neutrally: the migration trades **one characterised boundary defect** for **~50 scattered
defects spread over 22 years that nothing has yet characterised**. That is not an argument against
the ruling — the deeper history and the correct volume basis are real gains, and CBOE fixes
**A1-F6** — but two consequences follow for later passes:

1. Ingesting CBOE **unvalidated** would import 36 invariant violations and 7 holiday bars directly
   into the authoritative store. The prior spec's acceptance criterion of "zero OHLC-invariant
   violations" is **not achievable by raw ingest**; it requires rejection or repair.
2. It is the concrete case for per-row audit state (**A1-F26**) and for cross-source validation
   (**A1-F28**) — a second source is what distinguishes CBOE's 2019-01 corruption from real data.

**USER RULING (standing) — LD-1 holds:** crypto stays on Binance.US, not re-litigated. Recorded
only because LD-1 governs the *execution venue*, while **A1-F9** and **A1-F19** are about the
*historical data source* — the two are separable, and no decision has been taken on the second.

---

### A1.7 — Findings index

| ID | Severity | Env | Finding | Status vs prior docs |
|---|---|---|---|---|
| **A1-F1** | High | both | **Authoritative store** missing 18 equity sessions + 159 crypto bars/symbol; nothing detects it | Re-frames A-2; **refutes A-15** |
| **A1-F2** | High | both | Watermark reads Parquet, every consumer reads Postgres; both `MAX()`-based | Confirms prior |
| **A1-F3** | High | both | Postgres `DO NOTHING` vs Parquet keep-last → stores diverge in values; bad rows uncorrectable | **New** |
| **A1-F4** | High | equity | Volume seam at **2020-07-27**, 30–90×, spanning 5.6 years | **Corrects A-3** |
| **A1-F5** | High | equity | 5-bar SPY/QQQ probe island 2024-01-02→08; QQQ bars are not QQQ (+16.62 % artefact) | **New** |
| **A1-F6** | Medium | equity | 3 invalid SPY bars | Confirms A-6 |
| **A1-F7** | Medium | crypto | 14 flat bars (7/symbol) — **genuine Binance.US first-listing data, not fabricated**; entered training unflagged | **Corrects A-4 on all 3 claims** |
| **A1-F8** | Medium | crypto | Synthetic BTCUSDT bar exactly on `STITCH_DATE`, Postgres-only | **New** |
| **A1-F9** | High | crypto | ~200× volume seam at Binance Global → Binance.US stitch | **New — DS-7 parity with A1-F4** |
| **A1-F10** | High | both | Zero candles ever quarantined; validator has no flat-bar or implausible-volume check | Confirms + explains A-6 |
| **A1-F11** | Medium | both | Ingest-time gap detection sees only the incoming batch; exits early on 1-row fetches | **New** |
| **A1-F12** | Medium | equity | No equity gap-fill exists; `detect_equity_gaps` has no caller | **New — DS-7 defect** |
| **A1-F13** | High | crypto | Gap-fill bypasses the validator, writes Postgres only, and has **never inserted a row** — root cause: `api.binance.com` is **geo-blocked from this host** | **New** |
| **A1-F14** | Low | equity | `adjusted_close` 0/21,088, no writer, no reader, and redundant — `close` is already adjusted, so the column misleads about what `close` contains. **USER RULING: remove it**; note a `DROP COLUMN` is not A30-additive | Confirms prior, adds redundancy + ruling |
| **A1-F15** | Medium | equity | XLF 2016-09-19 −18.25 % unadjusted XLRE spin-off | **New** |
| **A1-F16** | Medium | crypto | Stitch performs **no reconciliation** — checks close price only, discards its own verdict, never looks at volume | **New** |
| **A1-F17** | Medium | equity | Step 12 cross-source check never runs (validator built without `db`/`config`) | **New** |
| **A1-F18** | High | both | Every silent-failure detector is `MAX()`-based, `>100`-row loose, crypto-only, or log-only — and **3 of 7 verification checks are hardcoded `passed=True`** with no failing branch, including the only interior-gap check, which finds gaps, logs them, and passes anyway. Thresholds need 96–99 % of the table destroyed to trip; `_latest_date` falls back to a hardcoded `2024-01-15` | **New** |
| **A1-F19** | High | crypto | The 22-day 2019 "outage" is **self-inflicted** — `STITCH_DATE` is 22 days early, and the 134 archive bars that fill it are downloaded on every backfill and discarded | **New** |
| **A1-F20** | High | both | Store authority: Postgres is the source of truth, Parquet an audit/drift check. Crypto stores agree bit-for-bit; **equity Parquet is stale and superseded** by the CBOE ruling | **New — USER RULING** |
| **A1-F21** | High | both | Candle layer raises **zero** alerts; `alert_log` works but no candle integrity check writes to it. All 43 candle alerts are INFO successes, and they post-date the defect by 4 months | **New** |
| **A1-F22** | High | both | **11 of 12 integrity capabilities exist, mostly read the right store, and are wired to nothing.** `data_ingestion_log` is write-only — no SELECT anywhere | **New** |
| **A1-F23** | High | both | Verification gate passes **vacuously** — 0 symbols = pass, "no data found" = pass; live artefact reports "no gaps >24h" while two gaps of 22 and 26 days exist. Never called by the collector | **New** |
| **A1-F24** | Medium | both | Collector **does** write `data_ingestion_log` (1,562 rows) but its 9 columns are run telemetry with nowhere to record an integrity fact; `system_events` is purpose-shaped and has **0 writers anywhere**; `api_errors` is execution-only, so the geo-block left no trace | **New** |
| **A1-F25** | Medium | both | The five metadata tables are **five independent designs for one concept** — 4 severity vocabularies, 2 PK strategies, `environment` vs `source` for identical values. None was ever migrated in; they accreted per-feature | **New — user observation, evidenced here** |
| **A1-F26** | Medium | both | Candle rows carry **no audit state** — no provenance, no last-audited, no review flag, no revision. Good and bad rows are indistinguishable, so no gated correction is expressible and both volume seams are knowable only by archaeology | **New — user direction, evidenced here** |
| **A1-F27** | High | equity | CBOE's volume basis **verified consolidated** (0.968× SIP) and unadjusted, with 2.15× deeper history — but the source carries **36 OHLC-invariant violations, 7 Thanksgiving bars, 30 corrupt 2019-01 bars, 3 duplicate rows** and symbol-varying row counts. Raw ingest would import all of them | **New** |
| **A1-F28** | High | both | **No independent cross-source validation runs anywhere.** Equity's is built and dead (**A1-F17**); crypto's **does not exist**, so Binance is unfalsifiable — every crypto finding was confirmed against Binance itself. Alpaca is now a demonstrated equity second source at **±2 bps** | **New — user direction, evidenced here** |

**DS-7 parity summary.** Five unexamined equity/crypto divergences found, none previously
justified or recorded: zero-volume validation is equity-only (**A1-F10**); gap-fill is crypto-only
(**A1-F12**); the verification gap check is crypto-only (**A1-F18**); the trader freshness guard is
equity-only (**A1-F18**); and the `source` provenance column exists on crypto only (**A1-F26**).
By the process contract these are defects until justified.

### Disposition under the CBOE ruling

The question "does a CBOE migration make this finding moot?" was asked repeatedly during the
walkthrough, so it is answered once here for all 28. **This is a disposition map, not a plan** — it
states which findings a source change would and would not remove, and nothing about how or whether
to sequence one.

| Outcome | Count | Findings | Why |
|---|---|---|---|
| **Removed outright** | 4 + part of **A1-F1** | **A1-F4**, **A1-F5**, **A1-F6**, **A1-F14**, **A1-F1** (equity half) | Pure Alpaca artefacts — wrong feed, bad rows, missing sessions. CBOE carries all of it correctly back to 2004, and independently corrects **A1-F6**'s flat SPY bar (99,495,037 vs 200) |
| **Untouched — crypto** | 6 + part of **A1-F1** | **A1-F7**, **A1-F8**, **A1-F9**, **A1-F13**, **A1-F16**, **A1-F19**, **A1-F1** (crypto half) | An equity source change does nothing for Binance. LD-1 keeps the execution venue; **A1-F9**/**A1-F19** concern the historical data source, which is undecided |
| **Untouched — structural** | 15 | **A1-F2**, **A1-F3**, **A1-F10**, **A1-F11**, **A1-F12**, **A1-F17**, **A1-F18**, **A1-F20**, **A1-F21**, **A1-F22**, **A1-F23**, **A1-F24**, **A1-F25**, **A1-F26**, **A1-F28** | Properties of the **pipeline**, not the vendor: nothing watches, nothing validates, nothing can be corrected, nothing is recorded. A new source inherits every one of them |
| **Made worse** | 1 | **A1-F15** | See below |
| **Is the migration risk** | 1 | **A1-F27** | CBOE's own ~50 defects |

**A1-F15 inverts, and this is the sharpest consequence of the ruling.** Today it is narrow: one XLF
spin-off in 2016 carries a false −18.25 % because `Adjustment.ALL` does not cover spin-offs. Under
**unadjusted** CBOE nothing is adjusted at all, so **every dividend ex-date and every split becomes
a false negative return** — all 8 symbols, 22 years, ~90 SPY dividends alone. What is currently one
bad bar becomes a systematic property of the entire series.

This is not an argument against the ruling. It is why carry item 13 exists: read-time adjustment
from `corporate_actions` is what turns raw storage from a liability into the correct design. But it
has a scheduling consequence worth stating plainly — **B3 Corporate actions is a dependency of the
equity migration, not a parallel workstream**, and `corporate_actions` currently holds **0 rows**.

Two caveats on this map:

- "Removed outright" requires a **full replacement** of `ohlcv_daily`. A splice that retains
  pre-CBOE Alpaca rows preserves every artefact on its own side of the changeover and adds a new
  seam there. **USER RULING 2026-07-25 settles this: the migration replaces the equity candle table
  in full, so that no seam or stitch exists between providers** (carry item 3). What remains open
  is *how* that replacement is executed against the **A1-F3** guard, which today would silently
  skip all 21,088 colliding keys — carry item 11.
- Removal is conditional on CBOE data being **validated on the way in**. Ingested raw, it replaces
  Alpaca's defects with its own (**A1-F27**) rather than removing them.

**Carried into the spec pass.** Recorded here so the A1 spec session does not have to re-derive
them. No remedy is proposed or implied by their presence in this list.

> **Note on item 14.** Every other item on this list is wiring, ownership or a decision. Item 14 is
> the one place where the validator itself is short of checks — it is the exception to carry item
> 8's "predominantly wiring, not new capability", alongside equity gap fill and the schema change.

| # | Item | Anchored by |
|---|---|---|
| 1 | Restoring the 22-day crypto hole from the already-downloaded archive — including that `DO NOTHING` will silently reject the one bar that collides with the synthetic row | **A1-F19**, **A1-F8**, **A1-F3** |
| 2 | The two venue-volume seams, which are **independent** of the gaps and are not fixed by closing them | **A1-F4**, **A1-F9** |
| 3 | Migrating equity candles to CBOE with unadjusted values, and what that does to `close`, `adjusted_close` and re-fetch determinism. **USER RULING 2026-07-25 — the migration must replace the equity candle table in full, so that no seam or stitch exists between providers.** The mechanism this rules out is verified: a full CBOE load writes ~45,000 rows into a table where **21,088 keys already hold Alpaca rows**, and under `ON CONFLICT DO NOTHING` (**A1-F3**) every one of those collisions is silently skipped. The result would be CBOE's 2004–2016 deep history grafted onto Alpaca's contaminated 2016–2026 present — a **new provider seam at 2016-01-04** — while `rows_inserted` reports a large positive number and nothing alerts (**A1-F21**). **A1-F6**'s SPY 2018-11-01 bar is that exact failure already observed at 1-row scale. Settling *how* the replacement is performed without violating the **A1-F3** guard is carry item 11's problem, and A30 additive-only applies while the trader runs | **USER RULING**, **A1-F3**, **A1-F6**, **A1-F4**, **A1-F14**, **A1-F21** |
| 4 | Store authority — making Postgres the watermark's store, and defining what the Parquet drift check asserts | **A1-F20**, **A1-F2** |
| 5 | The four DS-7 parity divergences | **A1-F10**, **A1-F12**, **A1-F18** |
| 6 | **USER DIRECTION 2026-07-25** — the watermark stops being a bare resume pointer and becomes the candle integrity audit, reading Postgres and raising through `alert_log()`. Scope to settle at spec: what it asserts (calendar completeness, interior gaps, store drift, value invariants), what severity each raises, and whether `alert_log` must retain the message body to be evidential | **A1-F2**, **A1-F11**, **A1-F12**, **A1-F18**, **A1-F21** |
| 7 | **USER DIRECTION 2026-07-25 — the collector owns candles end to end:** (a) ongoing collection, (b) **auditing historical data**, (c) maintaining data quality and integrity. Today it owns only (a); (b) has no owner at all, and (c) is spread across unwired modules. Note the collector currently imports four symbols from `ingest_all` and none of the verification surface (`collector_main.py:28-32`) | **A1-F22**, **A1-F23**, **A1-F12**, **A1-F21** |
| 8 | Most of this machinery **already exists and reads the correct store** — the spec question is predominantly wiring, ownership and thresholds, not new capability. Two exceptions: equity gap *fill* does not exist, and recording integrity facts needs a **schema change** | **A1-F22**, **A1-F24** |
| 9 | Where integrity/quality results should live. `data_ingestion_log` is the natural home but has no column for them; `system_events` is purpose-shaped and unused (0 writers); `alert_log` hashes its message body so it cannot carry evidence. Any choice is additive-only while the trader runs (A30) | **A1-F24**, **A1-F21** |
| 10 | **USER OBSERVATION 2026-07-25** — whether the five metadata tables collapse into a coherent model rather than five per-feature designs. Note the constraints this sits under: A30 additive-only while the trader runs; `api_errors` and `alert_log` have live readers in the execution path (`emergency.py:518`) so they are **not** free to reshape; `system_events` has zero writers and zero readers, making it the only one with no migration cost. Group C owns the execution-side tables, so any merged model crosses this audit's scope boundary | **A1-F25**, **A1-F24**, **A1-F21** |
| 11 | **USER DIRECTION 2026-07-25** — candle rows must carry audit state: **source of the data** (feed/venue provenance), **when last audited**, a **flag for update/review**, and a **revision number**. `DO NOTHING` (**A1-F3**) stays as the default guard — it exists to protect audited history — and a correction is permitted only for rows whose audit state licenses it. To settle at spec: this is a **schema change** on both candle tables (nullable adds are A30-additive, so permitted while the trader runs, but the *choice* between in-table columns and a side table is open); it is **per-row** state, distinct from the **per-run** integrity fact of item 9 which also has no home; it should carry the **`adjusted_close` drop** ruled in **A1-F14** in the same migration, noting that a `DROP COLUMN` is *not* A30-additive and needs an explicit exception; it interacts with item 1, where **A1-F8**'s synthetic bar wants flagging rather than silent replacement; and — **UNVERIFIED inference** — provenance is what would distinguish a retained CBOE series from the superseded Alpaca one under item 3 | **A1-F26**, **A1-F3**, **A1-F4**, **A1-F9**, **A1-F13** |
| 12 | **USER DIRECTION 2026-07-25** — cross-source validation becomes a standing gate in both environments, so a candle is corroborated by a source independent of the one that produced it. Measured constraints: Alpaca is a demonstrated equity second source at **±2 bps**, but only against *unadjusted* values (a naive comparison reports a false 26 bps dividend artefact), and its volume is usable only on the **SIP** feed, which `alpaca.py:104-107` selects on backfill alone. **Crypto has no candidate source at all** — that is the open question, and it must be genuinely independent of Binance, unlike the confirmations used for **A1-F7** and **A1-F9**. **Do not rebuild this from scratch — most of it exists.** Reuse inventory, traced this session, so the spec pass starts from what is built: **`CrossSourceValidator`** (`cross_source.py:41`) with `validate_prices(symbols, lookback_days=7, as_of_date)` → `list[CrossSourceResult]` (`symbol, date, alpaca_close, yfinance_close, diff, status`), a live call site at `validation.py:234-256`, and a test file at `tests/data/test_cross_source.py`. **Five things must change for it to be usable, all measured:** (i) it is never constructed with `db`/`config`, on any source (**A1-F17**); (ii) `_TOLERANCE_USD = 0.05` is an **absolute dollar** threshold — it cannot express the ±2 bps agreement actually measured, and does not scale across a \$59–\$740 price range; (iii) it compares yfinance **`Adj Close`** (`auto_adjust=False`, `cross_source.py:91`) against our `Adjustment.ALL` close — **adjusted vs adjusted, which is consistent today but breaks the moment the CBOE unadjusted ruling lands**, producing a false discrepancy on every historical bar equal to the cumulative drag of **A1-F15** (16.9 % SPY, 43.8 % XLE); (iv) results are **log-only** — no alert, no quarantine, no return to the caller; (v) the 7-day lookback makes it a recency check, not a history audit | **A1-F28**, **A1-F17**, **A1-F27**, **A1-F15**, **A1-F7**, **A1-F9** |
| 13 | **USER DIRECTION 2026-07-25** — settle the price basis: **one consistent basis for all bars**, with raw and adjusted both obtainable. Verified inputs: Alpaca `Adjustment.ALL` adjusts volume by the split factor **exactly** (AAPL 2020-08-31, ×4.0000) and leaves volume untouched by dividends; CBOE is unadjusted; a materialised adjusted column must be **rewritten across all history** on every split *and* every dividend (32+ events/year across 8 symbols), which **A1-F3**'s guard forbids, whereas a derived-at-read-time series needs only **one new `corporate_actions` row** and relocates the churn to `features_*`, which is regenerable by definition. **Verified dependency:** `corporate_actions` exists with the right shape (`symbol, action_type, effective_date, ratio, amount, processed`) and holds **0 rows**. Four event types are needed — split/reverse-split (one type, direction carried by `ratio`), cash dividend, **spin-off**, and special/return-of-capital distribution. **This fixes A1-F15 rather than merely offsetting the ruling** — the XLF spin-off has been wrong since 2016 and no vendor column has ever corrected it; both Alpaca and CBOE carry the identical −18.25 %. **USER RULING 2026-07-25 — proceed on the assumption that corporate actions are obtainable back to 2004; any residual unaccounted events are researched and entered manually.** That makes a manual-entry path part of the design, which in turn means `corporate_actions` rows need provenance of their own (entered by whom, sourced from where) — the same argument as **A1-F26**, applied to a second table. Note the failure mode also changes shape: a missing action no longer shows as one conspicuous bar but silently mis-scales a symbol's entire pre-event history, so detection falls to carry item 12. **UNVERIFIED:** whether `features_*` exposes a full recompute path — an A2 question | **A1-F3**, **A1-F14**, **A1-F15**, **A1-F26**, **B3** |
| 14 | **USER DIRECTION 2026-07-25 — the validator needs new checks, not just wiring.** Measured against the ~50 CBOE defects, the current 12 steps catch ~30 and let ~20 through (**A1-F10**). The enumerated gaps: **(a) no non-session-date rejection** — Step 9 finds missing sessions, nothing finds extra ones, and Step 7 explicitly *excuses* zero volume off-session (`validation.py:140`), which is why 15 holiday bars pass; **(b) no flat-bar check** — o=h=l=c passes every step, which is how **A1-F6**'s SPY bar and **A1-F7**'s 14 crypto bars entered; **(c) no implausible-volume check** — a 1,518× under-report is invisible, as is volume 200 on SPY; **(d) Step 5's `_TOLERANCE = 0.0001` band lets 6 of 36 real invariant violations pass** (`validation.py:113`); **(e) Step 8 dedups duplicate timestamps silently** with no quarantine row, so a source shipping duplicates leaves no trace; **(f) the DS-7 split** — Step 7 is equity-only (`validation.py:133`), so zero-volume crypto bars are legal by design. Note this is the gate the CBOE migration must pass, so it is a **prerequisite** of carry item 3, not a follow-up | **A1-F10**, **A1-F27**, **A1-F6**, **A1-F7**, **A1-F15** |

**Forward note to the B3 review — Corporate actions.** Raised during the A1 walkthrough
2026-07-25 and recorded here so the B3 session can correlate back to this discussion rather than
re-deriving it. A1 established the following, all measured:

| Established in A1 | Consequence for B3 |
|---|---|
| `corporate_actions` exists with a workable shape and holds **0 rows** | The table is not the gap; a **source** is |
| Alpaca `Adjustment.ALL` adjusts price for splits **and** dividends, and volume for splits only (×4.0000 exactly on AAPL 2020-08-31) | Whatever B3 chooses must reproduce this, since CBOE is unadjusted (**A1-F27**) |
| `Adjustment.ALL` does **not** cover spin-offs — XLF 2016-09-19 carries a **−18.25 %** discontinuity as a genuine-looking return (**A1-F15**) | The one event type already known to be wrong in live data |
| The process contract records **Alpaca returning zero spin-offs** | The incumbent vendor cannot supply the event type that is already broken |
| Four event types are needed: split/reverse-split (one type, `ratio` carries direction), cash dividend, spin-off, special/return-of-capital distribution | B3's scope question is source coverage per type, not schema |
| Crypto has **no** corporate actions — none of the four apply to BTC/ETH on Binance.US | A **justified** DS-7 divergence, recorded as justified so it is not counted among the five unexamined ones |
| **USER RULING 2026-07-25** — assume the actions are obtainable back to 2004; research and **enter any residual gaps manually** | B3's job is coverage per event type per symbol, plus a **manual-entry path with its own provenance** — who entered a row and from which source. Same argument as **A1-F26**, applied to `corporate_actions` |
| A missing action is **silent** under read-time adjustment — it mis-scales a symbol's whole pre-event history with no visible bar | Completeness is not a nice-to-have; it is the correctness condition. Detection falls to carry item 12's cross-source check |

Carry item 13 depends on this dataset: read-time adjustment is only as complete as
`corporate_actions`, and that table is currently empty.

**Evidence base.** Everything in this review was measured this session, not recalled:

- Live SQL against `pg16` — row counts, ranges, distributions, distinct counts, invariant checks,
  interior-gap scans, vintage archaeology via `fetched_at`, and the contents of
  `data_ingestion_log`, `data_quarantine`, `alert_log`, `system_events`, `api_errors` and
  `schema_migrations`.
- The live Parquet files, compared row-by-row against Postgres for both environments.
- The XNYS calendar via `exchange_calendars` 4.13.1.
- Live HTTP against Binance.US, Binance Global and `data.binance.vision`.
- The live `data/verification.json` artefact on the production runtime.
- DDL read directly from `postgres_schema.py` for all five metadata tables.
- Every code claim carries a `file:line` read this session, with call sites traced for each of the
  twelve capabilities in **A1-F22**.

**Confidence.** High on every measured claim. Three items are weaker and are flagged as such in
the text rather than averaged away:

| Item | Status |
|---|---|
| Mechanism behind the 2026-03-11→04-06 Postgres write failure | **UNVERIFIED** — `_sync_to_db` swallowing exceptions (`base.py:195-197`) is sufficient but was not confirmed from logs |
| Origin of the **A1-F8** synthetic bar | Unattributed. Its slot is now known to displace a real archive bar |
| Which run wrote the mocked `verification.json` (**A1-F23**) | **UNVERIFIED** — the `~/swingrl` CI-bed convention is a sufficient explanation, unconfirmed |

The **A1-F5** reading that QQQ's island bars are SPY-derived is medium confidence; that they are
*not QQQ* is high.

**Review status: A1 complete.** 28 findings, 14 items carried to the spec pass. Per session rule
#6 this review ends here — no spec, and A2 does not begin.

---
