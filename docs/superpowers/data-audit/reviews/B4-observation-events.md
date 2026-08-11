# B4 — Observation events — REVIEW

> **Dataset 3 of 12.** Review pass. Process contract: [`../00-PROCESS.md`](../00-PROCESS.md).
> Spine: [`../01-MASTER-REVIEW.md`](../01-MASTER-REVIEW.md).
> **Reviewed 2026-08-04 → 2026-08-05.** Moved ahead of A2 by **DS-11**.
>
> **This review contains no solutions.** Findings, evidence, gaps and use — nothing else.

---

## What this dataset is, and why it exists separately

B3 established that "corporate actions" was silently carrying three jobs: what the **instrument**
did (adjust), what the **venue or market** did (flag), and what **we** did (replay). Only the
first is B3. **DS-10** created B4 for the second.

The distinction that defines the dataset:

| | Instrument event (**B3**) | Observation event (**B4**) |
|---|---|---|
| What changed | The security itself | Only our *view* of it |
| Examples | Split, dividend, spin-off | Market closed, exchange outage, vendor gap, our own collector failure |
| Correct handling | **Retroactive adjustment** of the price series | **Flagging or exclusion** — never adjustment |
| Gets it wrong by | Applying the wrong factor | Treating an absence as a fact, or a fact as an absence |

B4 owns two halves that cannot be assessed apart, which is why DS-10 kept them in one dataset:

1. **The baseline** — what observations *should* exist. For equity that is the XNYS session
   calendar; for crypto it is 24/7 uptime with exceptions.
2. **The incidents** — what made an observation untrustworthy without changing the instrument.

You cannot judge an incident without the baseline (a missing bar on a closed day is not an
incident), and you cannot trust the baseline without the incidents (a present bar on a closed day
is not a session).

**The dataset does not exist.** There is no calendar table, no incident table, no incident column,
no status value and no vocabulary. What exists is a **library call made five times at runtime**,
and a set of gaps whose causes were never recorded. This review is therefore built the way B3's
rebuild was: forward from *what should exist* → *what is collected* → *how it is checked* → *who
consumes it*, with vendor evidence used inside that chain rather than as its starting point.

**Two things this review proves rather than asserts.** First, the calendar bound is *rolling* —
it was measured on two consecutive days and moved. Second, gap attribution is not academic — of
the crypto gaps that could be tested, some are the venue's and some are ours, and nothing in the
system can tell them apart.

---

## B4.0 — Disposition of carried-forward assumptions

Every prior claim about this dataset, and its verdict. This is where the contract's overriding
rule is discharged.

| Prior claim | Source | Verdict | What is actually true |
|---|---|---|---|
| "`get_calendar("XNYS")` built with no `start` yields a ~20-year rolling window — first session 2006-07-31 when measured, 2006-07-28 the day before" | B3 forward note to B4 | **Confirmed, and re-measured twice** | First session **2006-08-04** on 2026-08-04 and **2006-08-07** on 2026-08-05. The roll is now demonstrated across a day boundary, not inferred → **B4-F1** |
| "**647 sessions** of CBOE history are outside the bound today, growing yearly" | B3 forward note to B4 | **Corrected twice over** | **652** on 2026-08-04, **653** on 2026-08-05. And it grows **daily**, not yearly — the two measurements differ by exactly one session → **B4-F1** |
| "Three call sites: `:140` raises and is caught locally; `:287` raises unguarded; `:220` does not wrap it" | B3 forward note to B4 | **Confirmed on all three, and the count is wrong** | There are **five** calendar construction sites, not three. The three named are all inside `validation.py`; two more exist in the execution path → **B4-F4**. And "`:140` is caught locally" understates it: the catch **rewrites the failure as a different defect** → **B4-F3** |
| "`start="1990-01-01"` resolves fine — a missing argument, not a data limit" | B3 forward note to B4 | **Confirmed** | `first_session` becomes 1990-01-02. Verified live |
| "This blocks A1-C3 today" | B3 forward note to B4 | **Confirmed, and sharpened** | `validation.py:287` is reached from `validate_batch` with no `try` anywhere on the path. The migration does not degrade — it **raises** → **B4-F2** |
| "Venue events are crypto's equivalent and are unowned" (**B3-F8**, **B3-F32**) | B3 | **Confirmed, and now quantified** | Unowned confirmed. Newly measured: **7 venue-outage runs actually exist in our stored history** and are unrecorded → **B4-F8**, **B4-F12** |
| "Compounded by `ohlcv_4h.source` being 100 % NULL (**A1-F13**)" | A1 via B3 | **Confirmed as a compounding factor** | A1's finding, cited not re-found. Its B4 consequence: **no bar can be attributed to a venue**, so venue-specific incidents cannot be scoped to the segment they affected → **B4-F10** |
| "Phantom bars on closed sessions — 2018-11-22 and 2025-05-26, 20 across the 11-symbol snapshot" (**A1-F10**) | A1 | **Confirmed on the 8-symbol basis; A1's finding, not re-found** | Re-measured on SPY: exactly **2** non-session dates in 5,676, and **0** missing sessions. Used here only as evidence that the calendar is their **sole possible detector** |
| "Our crypto history is spliced — Binance Global archive for history, Binance.US ongoing" | B3 forward note to B4 | **Confirmed, with a consequence the note did not reach** | The Global venue is **HTTP 451 from here**. The historical half of our crypto series can never be re-verified against the venue that produced it → **B4-F10** |
| "The 22-day crypto hole is an outage" | pre-audit register | **Already refuted by A1-F19** — recorded here because it is B4's cautionary precedent | It is a `STITCH_DATE` filter artefact. A hole was attributed to a venue that was innocent. That is exactly the misclassification B4 exists to prevent, and it has already cost one finding |
| "The 2023 Binance.US regulatory episode detached our **prices** from the global market" | **This review's own hypothesis**, raised in session 2026-08-05 | **REFUTED by measurement** | Over 1,282 4h bars (2023-03 → 2023-09) our BTCUSDT close vs Binance Global's diverges by a mean of **+0.006 %** and a maximum of **1.36 %**; the June maximum is **0.60 %**. Our pairs are **USDT-denominated**, and the USD-rail disruption did not propagate to them. **What did happen is a volume collapse** → **B4-F20**. Recorded so the price theory is not chased again |
| "The 12 small crypto gap runs were never examined" | **This review's own first draft** | **Corrected** | **A1 enumerated them** (`A1-candles.md:742`). A1 owns their existence; **B4 owns only their classification** (§B4.1(g), §B4.3) |
| "An in-progress bar could be stored and then frozen forever by `ON CONFLICT DO NOTHING`" | **This review's completeness pass**, 2026-08-10 | **CLEAN — the guard exists in both environments, deliberately** | Crypto: `_incomplete_bar_open_ms` (`binance.py:79-92`) sets an exclusive `endTime`, and `_drop_incomplete_bars` (`:221-238`) repeats the check belt-and-braces on both the incremental (`:272-298`) and backfill (`:486-502`) paths. Equity: `alpaca.py:98-111` caps `end` to start-of-today UTC. The one known instance — a 2h19m partial stored 2026-07-18 — **was repaired**. Its residue is a *different* finding: the repair itself is unrecorded → **B4-F25** |
| "Vendors silently revise bars we already hold, and `DO NOTHING` would discard the correction" | **This review's completeness pass**, 2026-08-10 | **CLEAN for crypto — none found** | **3,409 bars re-fetched from Binance.US** across 7 windows 2020→2026 and compared on all five OHLCV fields: **0 mismatches** at 1e-6 relative tolerance. This is evidence of *no harm so far*, **not** of a working control — the exposure is real and remains → **B4-F21** |
| "A symbol or pair may have been delisted, suspended or reused under us" | **This review's completeness pass**, 2026-08-10 | **CLEAN today** | Both crypto pairs report `status=TRADING` on Binance.US `exchangeInfo`; the equity symbol list is **unchanged since the first commit** (`720e03e`). A **latent** gap, not a live one → **B4-C13** |
| *(note to B3)* "`api.nasdaq.com` connection failure — DNS resolves, TCP/TLS does not complete. **Unexplained**" | **B3 §Confidence**, 2026-07-30 | **STALE, not wrong** | It returns **HTTP 200** from this host on 2026-08-11. B3's claim was true when measured; live state moved, exactly as the contract's third exception anticipates. Recorded as a **note to B3, not re-found as a B4 finding** — and it matters because it means a Nasdaq-side source should be re-probed rather than assumed dead |
| *(method)* "Our crypto volume disagrees with the venue on 100 % of bars" | **This review's own first measurement**, 2026-08-10 | **REFUTED — my error, not a defect** | The first check compared our **quote** volume against the venue's **base** volume. `_parse_klines` documents the choice at `binance.py:200` and applies it at `:218`; `_parse_archive_csv` uses the same field at `:383`. Re-run correctly: **0 mismatches**. Recorded because the same trap is waiting for the spec pass — and because the opposite result would have overturned **A1-F9** |

---

## B4.1 — The data itself

### (a) The equity baseline: what a session is, and where that knowledge lives

There is **no stored calendar**. The full table list is 71 tables; none holds sessions, holidays,
early closes or venue status. The baseline exists only as a call into the
`exchange_calendars` library, made at runtime, five times, in five places.

```
$ docker exec pg16 psql -U swingrl -d swingrl -c "\dt"
71 rows   -- alert_log … weakness_profiles. No calendar table, no incident table.
```

The library version is **4.13.1**, declared as `"exchange-calendars>=4.13"`
(`pyproject.toml:43`) — a floor, not a pin. `uv.lock` pins 4.13.1 today, so the running version
is deterministic; the **declared dependency is not**.

**Why a stored baseline matters and a library call does not substitute.** A library upgrade can
change which dates are sessions. `exchange_calendars` ships corrections to historical holiday
data between releases. Because nothing stores what the calendar said when a bar was validated,
**a past validation cannot be reproduced**, and a changed answer is indistinguishable from no
change at all. → **B4-F5**

### (b) The rolling bound — measured on two consecutive days

`xcals.get_calendar("XNYS")` with no `start` argument does not return the full NYSE history. It
returns a window anchored to *now*.

| Measured on | `first_session` | `last_session` | CBOE-era sessions **outside** the bound |
|---|---|---|---|
| **2026-08-04** | **2006-08-04** | 2027-08-04 | **652** |
| **2026-08-05** | **2006-08-07** | 2027-08-05 | **653** |

The bound is **today − 20 years**, and it moved between the two measurements. 2006-08-05 and
2006-08-06 are a Saturday and Sunday, so the first session jumped to Monday 2006-08-07 and the
excluded count rose by exactly one. **The blocker grows by one session per trading day.**

Passing `start` fixes it outright:

```
xcals.get_calendar("XNYS")                      -> first_session 2006-08-07
xcals.get_calendar("XNYS", start="1990-01-01")  -> first_session 1990-01-02
```

It is a **missing argument, not a data limit**. → **B4-F1**

**What falls outside.** The CBOE vendor snapshot — the source A1-C3 migrates to — runs
2004-01-02 → 2026-07-23:

| Measure | Value |
|---|---|
| Distinct dates in the snapshot (SPY) | **5,676** |
| XNYS sessions in the same range | 5,674 |
| Snapshot dates that are **not** sessions (phantoms) | **2** — 2018-11-22, 2025-05-26 → **A1-F10**, not re-found here |
| Sessions with **no** snapshot bar | **0** |
| Snapshot dates **before** the rolling bound | **653** = **11.5 %** of the snapshot |

Two things follow. The snapshot is **session-complete** — zero missing sessions across 5,674 —
so the calendar is the *only* instrument that can catch its two phantom bars. And **11.5 % of the
range the migration must load cannot be validated at all today**, because the calendar refuses to
answer for those dates.

### (c) The exception the calendar raises, and where it lands

```
>>> xcals.get_calendar("XNYS").sessions_in_range("2004-01-02", "2004-12-31")
DateOutOfBounds: Parameter `start` receieved as '2004-01-02 00:00:00' although cannot be
earlier than the first session of calendar 'XNYS' ('2006-08-04 00:00:00').

>>> xcals.get_calendar("XNYS").is_session(pd.Timestamp("2004-06-01"))
DateOutOfBounds: Parameter `date` receieved as '2004-06-01 00:00:00' although cannot be
earlier than the first session of calendar 'XNYS'
```

Both raise. They land in two different places, with two different outcomes — §B4.3 covers both.

### (d) Is XNYS even the right calendar for our instruments?

The 8 ETFs are not all NYSE-listed — QQQ is Nasdaq, the sector SPDRs and SPY are NYSE Arca. So
"is XNYS the correct baseline" is a fair question. **Measured, it makes no difference:**

| Comparison, 2004-01-02 → 2026-08-03 | Result |
|---|---|
| XNYS sessions | 5,681 |
| XNAS sessions | 5,681 |
| In XNYS but not XNAS | **0** |
| In XNAS but not XNYS | **0** |
| Early closes, XNYS vs XNAS | 46 vs 46, **symmetric difference 0** |

`ARCX` and `BATS` are also available in the library and carry the **same rolling bound**. So the
calendar *choice* is not a defect — US equity venues close together. The bound is the defect, and
it affects every one of them identically. **No finding is raised for the calendar choice.**

### (e) The shape of a normal year — and the two gaps with no margin

Distribution of the interval between consecutive XNYS sessions, 2004-01-02 → 2026-08-03:

| Calendar days between sessions | Count |
|---|---|
| 1 | 4,448 |
| 2 | 53 |
| 3 (ordinary weekend) | 1,023 |
| 4 | 154 |
| **5** | **2** |

The maximum legitimate gap is **exactly 5 days**, and it occurs twice:

| Gap | Cause |
|---|---|
| 2006-12-29 → 2007-01-03 | New Year, a holiday combination |
| **2012-10-26 → 2012-10-31** | **Hurricane Sandy** — an unscheduled two-day venue closure, not a holiday |

`verification.py:42-44` sets `_EQUITY_GAP_THRESHOLD = timedelta(days=5)` with the comment
*"5 calendar days covers all holiday combos (max normal ~4 days)"*. The constant survives, but
the reasoning behind it is wrong on both counts: the max is 5 and not ~4, leaving **zero margin**,
and one of the two instances is **not a holiday combo at all** — it is precisely the class of
unscheduled venue closure this dataset exists to represent. → **B4-F16**

Also measured: **211 weekdays** in the range are not sessions, and **46 early closes** occur
(1–3 per year, none in 2026 to date; the next are 2026-11-27 and 2026-12-24).

### (f) The crypto baseline: an assumption, not a dataset

For crypto there is no calendar object of any kind. The baseline is a **constant**:

```python
_CRYPTO_FREQ = timedelta(hours=4)        # validation.py:38
gaps = time_diffs[time_diffs > expected_diff * 1.5]   # validation.py:308
```

That encodes "a bar exists every 4 hours, forever". There is no maintenance calendar, no venue
status source, and no representation of a bar that is legitimately absent. §B4.1(g) shows the
assumption is **factually false for our own stored history**. → **B4-F12**

### (g) What is actually stored, and what is missing from it

| Table | Rows | Symbols | Range |
|---|---|---|---|
| `ohlcv_daily` | 21,144 | 8 | 2016-01-04 → 2026-08-04 |
| `ohlcv_4h` | 38,669 | 2 | 2017-08-17 → 2026-08-04 |

**Equity.** Against the XNYS calendar, every one of the 8 symbols is missing **the same 18
sessions**, and none has a phantom bar:

```
QQQ  2016-01-04..2026-08-04  n=2643  missing=18  phantom=0
SPY  2016-01-04..2026-08-04  n=2643  missing=18  phantom=0
...  (identical for VTI, XLE, XLF, XLI, XLK, XLV)
```

The 18 are contiguous: **2026-03-11 → 2026-04-06**. Identical across all 8 symbols means this is
a *pipeline* event, not a symbol event. The gap itself is **A1-F1 / A1-F20** — not re-found here.

**Crypto.** Expected bars on a strict 4-hour grid vs stored:

| Symbol | Stored | Expected | Missing | Gap runs |
|---|---|---|---|---|
| BTCUSDT | 19,335 | 19,649 | **314** | 14 |
| ETHUSDT | 19,334 | 19,649 | **315** | 14 |

The 314 decompose cleanly into three populations:

| Population | Bars | Owner |
|---|---|---|
| 2019-09-01 → 2019-09-23 — the "22-day outage" | 133 (BTC) / 134 (ETH) | **A1-F19** — a `STITCH_DATE` artefact, *not* an outage |
| 2026-03-11 → 2026-04-06 | **159** | **A1-F1 / A1-F20** — its *attribution* is B4's, see §B4.3 |
| **12 small runs, 2017–2023** | **22** | **Enumerated by A1** (`A1-candles.md:742`), **never classified** — the classification is B4's |

**Attribution, stated precisely.** A1 §A1.3 already records these: *"+ 12 gaps of 8h–1d08h between
2017-09 and 2023-02, identical for both symbols"*. Their **existence is A1's**. What no document
has ever established is **what caused them**, and that is what §B4.3 measures. Every one of them
is **identical across BTCUSDT and ETHUSDT** — the same missing hours on both symbols, every time.
A defect affecting both symbols at the same instant is the signature of a **source-side** event,
not a per-symbol data-quality problem. §B4.3 tests that inference against the venue.

### (h) The other half of the dataset: observations that are **present but distorted**

Everything above concerns **absence**. There is a second class, and it is the harder one: the bar
**exists**, is arithmetically valid, and is *wrong to use as if it were an ordinary observation* —
because the venue's price discovery was impaired while it was formed. Nothing goes missing, so
**no absence-shaped detector can ever apply**.

#### Equity — market-wide circuit breakers

Trading was halted market-wide on four days in March 2020. From the CBOE snapshot (SPY):

| Date | Daily return | Intraday range | Volume | Validator verdict |
|---|---|---|---|---|
| 2020-03-09 | −7.81 % | 3.92 % | 294,947,173 | **passes every step** |
| 2020-03-12 | −9.57 % | 7.65 % | 366,863,758 | **passes every step** |
| 2020-03-16 | −10.94 % | 8.15 % | 282,719,246 | **passes every step** |
| 2020-03-18 | −5.06 % | 8.48 % | 316,009,853 | **passes every step** |

Step 6 flags a bar-to-bar move above **50 %** (`_SPIKE_THRESHOLD`, `validation.py:31`). The worst
of these four days moved **10.94 %** — a fifth of the threshold. Steps 1–5 and 7 are satisfied
because the bars are genuinely well-formed. **All four are in `ohlcv_daily` today**, indexed and
consumed as ordinary sessions. → **B4-F18**

#### Equity — 2015-08-24, the ETF dislocation

Ranked by intraday range `(high − low) / close` across all 8 ETFs over the full 2004 → 2026 CBOE
history, **the top three days are all 2015-08-24**:

| Rank | Symbol | Date | Range |
|---|---|---|---|
| 1 | XLK | **2015-08-24** | **21.71 %** |
| 2 | XLF | **2015-08-24** | **21.68 %** |
| 3 | XLV | **2015-08-24** | **21.28 %** |
| 4 | XLE | 2008-10-09 | 20.40 % |
| 6 | QQQ | **2015-08-24** | 18.07 % |

What the bars say for that day:

| Symbol | Prev close | Open | Low | Close | Low vs prev close | Close vs prev close |
|---|---|---|---|---|---|---|
| XLF | 23.64 | 22.21 | 18.52 | 22.65 | **−21.66 %** | −4.19 % |
| XLK | 19.78 | 18.35 | 15.66 | 19.07 | **−20.83 %** | −3.59 % |
| XLV | 70.90 | 66.59 | 55.83 | 67.85 | **−21.26 %** | −4.30 % |
| QQQ | 102.40 | 94.23 | 84.74 | 98.46 | −17.25 % | −3.85 % |
| SPY | 197.63 | 187.49 | 182.40 | 189.55 | −7.71 % | −4.09 % |

The pattern is the signature: **a low 17–22 % below the prior close, and a close down only 3–4 %**.
The bar is a truthful record of prints that occurred while LULD halts had broken the link between
the ETFs and their underlying baskets. Treating that low as a price at which value could be
transacted is the error, and **nothing marks it**.

Note where this sits: **2015-08-24 is outside `ohlcv_daily`** (which starts 2016-01-04) and
**inside the CBOE snapshot**. So it is not a defect in today's data — it is a defect **A1-C3 will
import**, along with the 2008 cluster below it in the ranking. → **B4-F19**

#### Crypto — the June 2023 Binance.US break

Our post-2019 crypto bars are Binance.US; the public archive is Binance **Global**. Comparing
them measures venue divergence directly. Over 1,282 4h bars, 2023-03 → 2023-09:

| Measure | Value |
|---|---|
| Mean close divergence (ours vs Global) | **+0.006 %** |
| Max absolute divergence | **1.356 %** (2023-03-24 08:00) |
| Max within June 2023 | **0.598 %** |

**Prices did not detach.** The hypothesis that they did is refuted (§B4.0). What did happen is a
liquidity collapse. Mean 4h BTCUSDT volume in our own table:

| Month | Mean 4h volume | Median |
|---|---|---|
| 2023-03 | 11,599,016 | 9,139,636 |
| 2023-04 | 11,046,117 | 8,397,825 |
| 2023-05 | 4,217,266 | 2,873,118 |
| **2023-06** | **1,326,617** | **844,850** |
| **2023-07** | **854,463** | **668,814** |
| 2023-08 | 858,834 | 625,438 |
| 2023-09 | 737,635 | 562,050 |
| 2023-12 | 1,567,795 | 1,186,857 |
| 2024-02 | 840,270 | 524,020 |

**~13× on medians** (9.14 M → 0.67 M), and it **never recovers** — still 0.8–1.5 M through
2024-03. Daily totals date the break to a two-day step:

```
2023-06-06 | 21,030,051
2023-06-07 |  9,957,155      <- step begins
2023-06-08 |  3,655,435      <- step completes
2023-06-11 |  3,991,077      ... and stays there
```

**How much is venue-specific.** Our volume ÷ Global volume, monthly mean: **559 (May) → 180 (Jun)
→ 184 (Jul) → 172 (Aug) → 171 (Sep)**. The ratio itself is meaningless in absolute terms (the two
series are on different units — that is **A1-F9**'s territory), but its *change* is not: ours fell
**~3.1× relative to Global**. So roughly a third of the 13× is a market-wide mid-2023 volume
decline and **the remainder is specific to this venue**.

Every price-based check agrees across this break, because the prices agree. Every gap check is
silent, because no bar is missing. The series simply changes scale by an order of magnitude,
permanently, mid-history. → **B4-F20**

**Why this is not A1-F9.** A1-F9 and A1-C2 cover **source-change seams** — the Binance
Global→Binance.US stitch (~200×) and the Alpaca IEX/SIP seam. Here the source is **unchanged on
both sides**: Binance.US before, Binance.US after. This is a real market event recorded faithfully,
which is precisely why it needs an event record rather than a source fix.

### (i) Completeness pass — nine candidate blind spots, measured

Added 2026-08-10 after the question *"what haven't we considered?"*. Nine classes the first draft
did not cover were enumerated and each was measured. **Six produced findings, three produced clean
negatives** — the negatives are recorded in §B4.0 so they are not re-raised.

| # | Class | Outcome |
|---|---|---|
| 1 | In-progress bars | **Clean** — guarded in both environments → §B4.0 |
| 2 | Vendor bar revisions | **Clean for crypto**, exposure remains → **B4-F21** |
| 3 | Duplicate bars | **Finding** → **B4-F22** |
| 4 | Vendor vs venue identity | **Finding** → **B4-F23** |
| 5 | Symbol / pair lifecycle | **Clean today**, latent → **B4-C13** |
| 6 | DST and timezone | **Finding** → **B4-F24** |
| 7 | Our own interventions | **Finding** → **B4-F25** |
| 8 | Incidents spanning datasets | **Finding** → **B4-F26** |
| 9 | Knowledge date | **Carry** → **B4-C14** |

#### The revision test, and the unit trap inside it

Our post-2019 crypto bars come from Binance.US, so re-fetching the same bars from Binance.US today
detects any revision directly.

| Window | Ours | Venue | Compared | Field mismatches |
|---|---|---|---|---|
| 2020-01 → 04 | 545 | 546 | 544 | **0** |
| 2021-06 → 09 | 551 | 551 | 550 | **0** |
| 2022-06 → 09 | 552 | 552 | 551 | **0** |
| 2023-06 → 09 | 552 | 552 | 551 | **0** |
| 2024-01 → 04 | 545 | 546 | 544 | **0** |
| 2025-01 → 04 | 539 | 540 | 538 | **0** |
| 2026-07-10 → 08-01 | 132 | 132 | 131 | **0** |
| **Total** | | | **3,409** | **0** |

**The first run of this test reported a mismatch on 100 % of bars.** It compared our `volume`
against the venue's **base-asset** volume, when we deliberately store **quote-asset volume in USD**
(`binance.py:200` documents it; `:218` applies it). The apparent discrepancy was the BTC price.
Corrected, every field agrees. Two things follow, and the second matters more:

- No revision has occurred in the sampled range. The exposure is untouched, because
  `ON CONFLICT DO NOTHING` would discard one silently → **B4-F21**.
- `_parse_archive_csv` uses the **same** `volume_quote` field (`binance.py:383`, falling back to
  `volume_base × close`). So the two sides of the 2019 stitch are in the **same unit**, and
  **A1-F9's ~200× seam is genuine liquidity, not a unit bug**. A1's finding is confirmed, not
  re-found — recorded so the spec pass does not re-open it.

#### The timezone finding, demonstrated on this review's own query

```
SHOW timezone;                       ->  America/New_York
SELECT '2020-01-01'::timestamptz;    ->  2020-01-01 00:00:00-05   (= 05:00 UTC)
```

The database session runs in **`America/New_York`**, so a bare date literal against a
`timestamptz` column silently means 05:00 (EST) or 04:00 (EDT) UTC. The table itself spans both:

| Rendered offset | Rows |
|---|---|
| EDT | 25,059 |
| EST | 13,674 |

The audit query in the revision test above lost exactly the 00:00 and 04:00 UTC bars of each
window's first day — **10 bars** — for this reason, before it was caught. The project rule is
**UTC internally** (`CLAUDE.md`); the store honours it, the *session* does not, and the gap falls
precisely where observation completeness is judged. → **B4-F24**

#### What the outage window actually shows

| Environment | `success` | `failed` | `no_data` |
|---|---|---|---|
| equity | **108** | 30 | 16 |
| macro | **436** | 114 | 165 |
| crypto | — | — | 4 |

Equity logged **108 successful runs** over 2026-03-11 → 04-01 and **still lost 18 sessions**.
Macro logged **436** and retained **48 rows** for the window. The status field is not merely
missing a term for "market closed" (**B4-F14**) — it reports success while data is being lost.

The window also hit **three datasets**: equity, crypto *and* macro. Options are unaffected only
because capture began 2026-07-15. So an incident record has to be **dataset-wide**; a
per-environment one would have split this single event into three unrelated ones — which is
exactly how it has been carried until now. → **B4-F26**

#### Duplicates, identity, interventions

| Question | Measured |
|---|---|
| Can a duplicate bar exist in store? | **No** — PK `(symbol, date)` / `(symbol, datetime)`. Step 8 dedups **pre-insert**, `keep="last"`, one `log.warning` (`validation.py:210-217`), persisted nowhere. The discarded row is unrecoverable → **B4-F22** |
| Can a row name its vendor? | **No.** `ohlcv_daily` has **no source column at all**; `ohlcv_4h.source` is 100 % NULL (**A1-F13**) → **B4-F23** |
| Can a row name its venue? | **No** — and vendor and venue are different questions with the same symptom |
| Is any manual data intervention recorded? | **No.** `operator_actions` = **0 rows**, yet the 2026-07-18 partial was demonstrably repaired → **B4-F25** |
| Does anything carry a knowledge date? | **Partly.** `fetched_at` gives 138 (daily) / 270 (4h) vintages — when a **bar** arrived, never when something **became known about** it → **B4-C14** |

---

## B4.2 — Historical one-time ingestion, and its checks and balances

**There was no historical ingestion of this dataset, because the dataset was never conceived.**
No calendar was ever loaded, no incident was ever recorded, and no backfill ever attempted either.

What *did* happen historically is that observation events **occurred and were absorbed silently
into the candle tables** — which is why they surface as A1 findings rather than B4 records. The
history of this dataset is therefore the history of its absence:

| Era | What happened | What recorded it |
|---|---|---|
| 2004 → 2016 | CBOE equity history exists at the vendor, un-ingested | Nothing — outside `ohlcv_daily`'s range entirely |
| 2017-08 → 2019-09 | Binance **Global** archive supplies crypto history, including **7 venue-outage runs** | Nothing. `ohlcv_4h.source` is 100 % NULL (**A1-F13**), so not even the venue is recorded |
| 2019-09 | The Global→US splice. `STITCH_DATE = "2019-09-01"` filters 22 days that were never missing | Nothing — and the absence was later **misread as an outage** (**A1-F19**) |
| 2019-09 → 2026-03 | Binance.US ongoing; **5 of our own gap runs** open in this era | Nothing |
| 2026-03-11 → 04-06 | A ~4-week loss across **both** environments | `data_ingestion_log` exists but cannot say what happened — §B4.3 |

**Checks and balances at ingestion time: none applied to this dataset.** The 12-step
`DataValidator` has no step for "was this date a session", no step for "is this absence
expected", and no step that writes anything durable about an absence — §B4.3.

**A note on what the history proves.** The 2019 case is the important one. A hole was present in
the data; a prior document attributed it to a venue outage; A1 established it was self-inflicted.
The system had no way to distinguish those two explanations, so the wrong one was recorded and
believed. That is not a hypothetical failure mode for this dataset — it is its **only recorded
instance of anyone trying**.

---

## B4.3 — Ongoing ingestion, and its checks and balances

### Owner and schedule

**Nothing owns observation events.** No collector job fetches a calendar, a holiday list, a venue
status feed or an incident report. The collector (`scripts/collector_main.py`) owns candles,
macro, and options chains; the calendar appears only as a **gate** on the options job
(`collector_main.py:144`), never as data to be captured.

### Is a gap detectable? — the equity detector's blind spot

`_detect_equity_gaps` is the only equity gap check:

```python
def _detect_equity_gaps(self, df, symbol):                     # validation.py:278
    start = df.index.min().normalize().tz_localize(None)       # :285
    end   = df.index.max().normalize().tz_localize(None)       # :286
    expected = self._nyse.sessions_in_range(start, end)        # :287
    actual   = ...
    missing  = expected.difference(actual)
    if len(missing) > 0:
        log.warning("equity_gaps_detected", ...)               # :291
```

Two independent defects:

1. **`start` and `end` come from the fetched batch's own data.** The expected-session range can
   never extend past the newest row the batch contains. So the detector can only ever find
   **interior** gaps. If the most recent sessions are missing — the exact shape of a live
   ingestion failure, and the exact shape of the 2026-03-11 → 04-06 event as it was unfolding —
   the range shrinks with the data and the gap is **invisible by construction**. → **B4-F11**
2. **`:287` raises `DateOutOfBounds` for any pre-2006 date, unguarded.** The call path is
   `validate_batch` (`:220`) → `_detect_gaps` (`:272`) → `_detect_equity_gaps` (`:287`), and
   there is **no `try` anywhere on it**. `validate_batch`'s docstring promises it "Raises
   `DataError` only for stale data (step 10)" (`:203-204`) — it also raises `DateOutOfBounds`,
   undocumented. The CBOE migration hits this on its first pre-2006 bar. → **B4-F2**

And when it does fire, the outcome is a `log.warning`. **Nothing durable is written** — no row,
no alert, no flag. `alert_log` holds 286 rows and begins **2026-07-15**, months after the
2026-03-11 event, so nothing could have alerted even in principle.

### The second calendar call, and the failure it disguises

Step 7 checks zero volume on a trading day:

```python
for ts in zero_vol[zero_vol].index:
    try:
        normalized = ts.normalize().tz_localize(None)
        if self._nyse.is_session(normalized):                       # :140
            reasons[...].append("Step 7: zero volume on trading day")
    except Exception:                                               # :146
        reasons[...].append("Step 7: zero volume (calendar check failed)")   # :151
```

For any pre-2006 bar, `:140` raises `DateOutOfBounds`, `:146` swallows it, and the row is
quarantined as *"zero volume (calendar check failed)"*. The record that survives says the
**volume** was suspect. The truth is that **the calendar could not answer**. A bar with
legitimately zero volume on a genuine 2004 session and a bar the calendar simply could not judge
receive the identical, misleading verdict — and it is written to `data_quarantine` as evidence.
→ **B4-F3**

This is the same failure mode as the 2019 hole: an absence of knowledge recorded as a fact about
the data.

### Is a crypto gap detectable, and can it be attributed?

```python
gaps = time_diffs[time_diffs > expected_diff * 1.5].dropna()   # validation.py:308
if len(gaps) > 0:
    log.warning("crypto_gaps_detected", ...)                   # :309
```

It fires on any gap over 6 hours and, like the equity detector, produces only a log line. It
cannot distinguish a venue outage from a collector failure because it has nothing to compare
against.

**So this review made the comparison the system cannot.** Every historical crypto gap was tested
bar-for-bar against Binance's own public archive (`data.binance.vision`, daily 4h zips —
reachable, unauthenticated, no geo-block):

| Date | Venue archive has | We have | Verdict |
|---|---|---|---|
| 2017-09-06 | 0,4,8,12,16,20 | 0,4,8,12,20 | **OURS** — venue has 16 |
| 2018-02-08 | **0** | 0 | **VENUE OUTAGE** |
| 2018-02-09 | **8,12,16,20** | 8,12,16,20 | **VENUE OUTAGE** |
| 2018-06-26 | **0,12,16,20** | 0,12,16,20 | **VENUE OUTAGE** |
| 2018-07-04 | **0,8,12,16,20** | 0,8,12,16,20 | **VENUE OUTAGE** |
| 2018-11-14 | **0,8,12,16,20** | 0,8,12,16,20 | **VENUE OUTAGE** |
| 2019-03-12 | **0,8,12,16,20** | 0,8,12,16,20 | **VENUE OUTAGE** |
| 2019-05-15 | **0,12,16,20** | 0,12,16,20 | **VENUE OUTAGE** |
| 2019-08-15 | **0,8,12,16,20** | 0,8,12,16,20 | **VENUE OUTAGE** |
| 2020-04-28 | 0,4,8,12,16,20 | 0,12,16,20 | **OURS** — venue has 4, 8 |
| 2020-07-09 | 0,4,8,12,16,20 | 0,12,16,20 | **OURS** — venue has 4, 8 |
| 2021-06-22 | 0,4,8,12,16,20 | 0,8,12,16,20 | **OURS** — venue has 4 |
| 2023-02-06 | 0,4,8,12,16,20 | 0,4,12,16,20 | **OURS** — venue has 8 |

**13 days tested: 8 match the venue exactly, 5 do not.** Grouped into gap runs: **7 runs are
genuine Binance venue outages (15 bars); 5 runs are our own losses (7 bars).**

Both populations sit in `ohlcv_4h` as **the same thing — an absent row**. One is a true fact
about the world that an agent arguably should know (the venue was down; no price was
discoverable). The other is a defect that is **still repairable from the archive today**. Nothing
in the system separates them, and nothing ever has. → **B4-F8**

### The 2026 hole, attributed

The largest gap — 159 bars per symbol, 2026-03-11 04:00 → 2026-04-06 12:00 UTC — was tested
against the live venue:

```
api.binance.us  2026-03-11  BTCUSDT -> ['00:00','04:00','08:00','12:00','16:00','20:00']
api.binance.us  2026-04-06  BTCUSDT -> ['00:00','04:00','08:00','12:00','16:00','20:00']
```

**The venue serves all six bars for both days, right now.** The gap is ours. It has been open
roughly four months, the data to close it is a single API call away, and **nothing anywhere
records that the venue was healthy throughout.** The gap itself belongs to A1-F1 / A1-F20; what
is B4's is that its *cause* was never established and, absent this measurement, is
indistinguishable from a venue failure. → **B4-F9**

The same window is the equity 18-session gap. **One incident, both environments** — and nothing
connects them, because there is no incident record to connect them in.

### The historical venue is unreachable

```
api.binance.com  ->  HTTP 451  {"code":0,"msg":"Service unavailable from a restricted location"}
```

Binance Global — the venue that produced everything before 2019-09 — cannot be reached from here.
The public archive is the **only** route to that history, and `grep` finds **zero** references to
`data.binance.vision` anywhere in `src/` or `scripts/`. Combined with `ohlcv_4h.source` being
100 % NULL (**A1-F13**), no historical crypto observation can be attributed to a venue *or*
re-verified against one through any path the system currently has. → **B4-F10**

### What the ingestion log can and cannot say

`data_ingestion_log` (1,759 rows, 2026-03-06 → 2026-08-04) is the closest thing to an incident
record. Its `status` vocabulary is `success` / `failed` / `no_data`. Joined against the XNYS
calendar:

| Environment | Status | Market **open**? | Runs |
|---|---|---|---|
| equity | success | no | **63** |
| equity | failed | no | **19** |
| equity | no_data | **yes** | **24** |
| equity | success | yes | 202 |
| equity | failed | yes | 41 |
| crypto | no_data | — | 6 |

The vocabulary **has no term for "the market was closed, so nothing was expected"**. A closed
Saturday is logged `success` 63 times and `failed` 19 times — the identical situation producing
opposite verdicts, neither of which is correct. Meanwhile `no_data` appears **24 times on real
sessions**, where it is a genuine problem, and is indistinguishable from the benign case.
→ **B4-F14**

### Rejects: where an observation defect would go, and whether it ever has

`data_quarantine` is the rejection store. Its entire contents:

| source | severity | reason | rows | range |
|---|---|---|---|---|
| macro | warning | Step 1: null value | **3,945** | 2026-03-11 → 2026-08-02 |

**3,945 rows, 100 % macro.** Not one equity or crypto row has ever been quarantined. Across a
10.5-year equity history and a 9-year crypto history — including 18 missing sessions, 314 missing
crypto bars, two phantom bars on closed days and three invalid SPY bars (**A1-F6**) — the
row-level quarantine path for candles has **never fired once**. → **B4-F17**

Supporting: `api_errors`, `system_events`, `operator_actions` and `emergency_flags` are all
**0 rows**. `circuit_breaker_events` holds 10 rows, all ours, 2026-07-19 → 2026-07-27.

### Source availability — what exists, and what covers nothing

Added 2026-08-11. The contract's §3 asks *which source*; B3 set the precedent (**B3-F22**) of
recording a source's reachability and coverage without selecting it. Every row below is a live
probe from this host.

| Need | Candidate source | Status |
|---|---|---|
| Session calendar | `exchange_calendars`, explicit `start` | **Demonstrated** — `start="1990-01-01"` resolves |
| Early closes | `exchange_calendars.early_closes` | **Demonstrated** — 46 in range |
| **Equity halts, 2019+** | **NYSE `trade-halts/historical/download`** | **Demonstrated — 71,718 records, free, unauthenticated** |
| **Equity halts, pre-2019** | — | **BLANK** |
| Crypto gap classification | `data.binance.vision` archive | **Demonstrated** — answered 13 of 13 dates |
| Crypto recent repair | `api.binance.us` klines | **Demonstrated** — still serving the missing bars |
| **Crypto venue incidents** | — | **BLANK** — four candidates tested, all failed |
| Pre-2019 crypto venue truth | `api.binance.com` | **Blocked — HTTP 451** (**B4-F10**) |
| Our own interventions | self-generated | No source needed; nothing writes it (**B4-F25**) |

**The asymmetry is the finding.** The **baseline** half of this dataset is well-sourced and cheap —
the calendar and its early closes are one library call away. The **incident** half is sourced for
**equity only, and only after 2019-02-22**. Everything before that, and all of crypto, has no
identified source at all.

#### The equity halt source, measured

```
GET https://www.nyse.com/api/trade-halts/historical/download   ->  HTTP 200, 7.6 MB CSV
columns: Halt Date, Halt Time, Symbol, Name, Exchange, Reason, Resume Date, NYSE Resume Time
rows: 71,718        range: 2019-02-22 -> 2026-08-10
```

| Year | Halts | | Reason code | Count |
|---|---|---|---|---|
| 2019 | 1,214 | | LULD pause | 56,042 + 4,320 |
| **2020** | **14,928** | | News pending | 9,259 + 676 |
| 2021 | 5,633 | | Corporate Action | 526 |
| 2022 | 6,892 | | News Released | 233 |
| 2023 | 9,384 | | Regulatory Concern | 221 |
| 2024 | 11,218 | | New Security Offering | 11 |
| 2025 | 13,095 | | **ETF Component Prices Not Available** | **3** |
| 2026 | 9,354 | | | |

Three things follow.

1. **It confirms B4-F18 from a venue record**, not from price behaviour: **522 / 787 / 825 /
   1,481** halt records on 2020-03-09, 03-12, 03-16 and 03-18 respectively.
2. **It names a halt on one of our own instruments** — see **B4-F28**.
3. **Its reason vocabulary already contains the 2015-08-24 failure mode** as a coded field,
   *"ETF Component Prices Not Available"* — the exact condition B4-F19 had to infer from price.
   The vendor considers it a nameable event; we have no field to put it in.

**And it stops at 2019-02-22.** 2015-08-24, the 2008 cluster, and every other pre-2019 event inside
the range **A1-C3** will import are outside it. → **B4-F27**

#### The crypto incident hunt, and its four failures

| Candidate | Result |
|---|---|
| `status.binance.us` | **DNS does not resolve** — no such host |
| `binance.statuspage.io/api/v2/incidents.json` | **HTTP 401** — *"Your page is inactive"* |
| `api.binance.com/sapi/v1/system/status` | **HTTP 451** — consistent with **B4-F10** |
| `api.binance.us/sapi/v1/system/status` | Requires an API key, and returns **current state only** — no history |

No retrospective crypto incident source was found. **B4-F8**'s archive-vs-stored comparison is
therefore not one method among several — it is the **only** one, and it *infers* an incident from a
missing bar rather than reading a record of it. A crypto observation event can consequently carry
an absence and nothing else: no reason, no duration, no resume time, no severity. → **B4-F29**

### What would detect a silent failure — and can it fire?

| Detector | Exists? | Can it fire on an observation event? |
|---|---|---|
| `_detect_equity_gaps` | yes | **Only for interior gaps**, and only as a log line. Raises on pre-2006 |
| `_detect_crypto_gaps` | yes | Fires on any >6 h gap, as a log line. **Cannot attribute** |
| `_check_staleness` | yes | Raises `DataError` at 4 days (equity) / 8 h (crypto) — the closest thing to a working trailing-gap detector, but it is a *freshness* check, not a completeness one |
| `data_quarantine` | yes | **Never fired for candles in the table's life** |
| `alert_log` | yes | Begins 2026-07-15 — after every event in this review |
| Venue status monitoring | **no** | — |
| Calendar-vs-data reconciliation | **no** | — |

---

## B4.4 — The calculation

B4 has no arithmetic. Its "calculation" is a **classification** — deciding, for a given
(symbol, timestamp), which of four states applies:

| State | Meaning | Correct handling |
|---|---|---|
| **Expected and present** | A session with a bar | Use it |
| **Legitimately absent** | Market closed / venue not required to trade | Not a gap. Exclude from any completeness count |
| **Venue outage** | Venue was open but produced nothing | A fact about the world. **Flag**, never fill |
| **Our loss** | Venue produced it; we do not hold it | A defect. Repairable |

**No code performs this classification.** The four states collapse into two in every existing
code path — "row present" or "row absent" — and every consumer reads the collapsed version.

The only transformation that touches the question at all is `resolve_crypto_gaps`:

```python
def resolve_crypto_gaps(df, symbol):                    # ingest_all.py:167
    filled = df.ffill(limit=2)                          # :184
    remaining = int(filled.isna().any(axis=1).sum())
    if remaining > 0:
        raise DataError(f"Crypto gap in {symbol} exceeds 2 bars ...")   # :192
```

Two facts about it:

1. **It is dead.** `grep` across `src/`, `scripts/` and `tests/` returns the definition, and
   references only from `tests/data/test_ingest_all.py`. **No production caller exists.** It is
   tested, and never runs.
2. **If it ran, it would be wrong for this dataset.** `ffill(limit=2)` converts a 1–2 bar gap
   into a copy of the previous bar with no marker of any kind — turning a venue outage into a
   fabricated observation. Of the 12 small crypto gap runs, **9 are 1–2 bars** and would be
   silently filled this way, including 6 of the 7 genuine venue outages. A1-F7 and A1-F8 record
   that fabricated crypto bars have already caused a misreading in this project. → **B4-F13**

Elsewhere, forward-fill is used deliberately for *alignment* rather than gap repair —
`technical.py:126` (weekly → daily) and `technical.py:174` (daily → 4H). Those are A2's to assess;
noted here only because they mean an observation event **propagates** into derived features
without any marker travelling with it.

---

## B4.5 — The pipeline and its wiring

### The five calendar construction sites

| # | Site | Purpose | `start` passed? | Historical dates? |
|---|---|---|---|---|
| 1 | `validation.py:61` | Bar existence — steps 7 and 9 | **no** | **Yes — this is the only one that looks backwards** |
| 2 | `execution/emergency.py:108` | Liquidation strategy — is the market open now | **no** | no |
| 3 | `execution/risk/circuit_breaker.py:344` → `:353` | Cooldown ramp — count sessions since trigger | **no** | no |
| 4 | `execution/pipeline.py:1202` → `:1204` | Staleness warning — find the previous session | **no** | no |
| 5 | `data/options/market_calendar.py:19` | Options capture gate + early-close flag | **no** | no |

**None passes `start`.** Four are safe only because they ask about recent dates; site 1 is the
one that walks history, which is why the bound bites there and nowhere else — and it is precisely
the site the CBOE migration will drive.

Two structural consequences:

- **No shared helper.** Five sites construct the calendar independently, three of them via a
  local `import exchange_calendars` inside a function body (`circuit_breaker.py:342`,
  `pipeline.py:1200`). A fix applied at one site does not reach the others, and nothing makes the
  omission visible.
- **The library caches globally.** `xcals.get_calendar("XNYS") is xcals.get_calendar("XNYS")`
  → `True`. So within a long-running process the bound is **frozen at first construction**, and
  two processes started on different days hold **different bounds** while sharing one database.
  The rolling window does not roll inside a process; it jumps at restart. → **B4-F4**

### Two authorities for one question

For "may we trade today", the execution pipeline does **not** use the calendar at all:

```python
if env_name == "equity" and self._config.equity.market_calendar_gate:   # pipeline.py:233
    if not self._equity_cycle_allowed():                                 # :1112
        return []
        ...
        clock = adapter.get_clock()                                      # :1128  <- Alpaca broker clock
```

So the system holds **two independent answers** to "is today a session": the **Alpaca broker
clock** decides whether to trade, and **xcals** decides whether a bar should exist. Nothing
compares them, and no code path would notice if they disagreed. A broker clock outage and a
calendar error produce different failures that look identical downstream. → **B4-F6**

### The early-close capability that exists, and reaches one dataset

`is_early_close()` (`options/market_calendar.py:28`) correctly derives half-days from
`session_close`. Its **only** caller is `options/collector.py:119`, which persists the result as
`options_chains.is_early_close` (`options/schema.py:24`).

So the system *can* identify a half-day, *does* record it — for **one dataset of twelve**. The
46 early closes in the CBOE equity range enter `ohlcv_daily` and the feature pipeline carrying no
marker at all, with roughly half a normal session's volume, and are treated as ordinary
observations by every consumer. → **B4-F15**

This is **B3-F20**'s cross-cutting shape appearing exactly where B3's forward note predicted it:
a capability wired to one dataset, with nothing announcing the omission elsewhere.

### Full path, source to consumer

```
  EQUITY                                            CRYPTO
  ──────                                            ──────
  vendor bars                                       Binance Global archive (pre-2019, HTTP 451 today)
      │                                             Binance.US (2019-09 →)
      ▼                                                 │
  DataValidator                                         ▼
   step 7  is_session ──► raises pre-2006 ──► caught ──► WRONG quarantine reason   [B4-F3]
   step 9  sessions_in_range ──► raises pre-2006, UNGUARDED                        [B4-F2]
           │  interior gaps only ──► log.warning                                   [B4-F11]
           ▼                                                 _detect_crypto_gaps
      ohlcv_daily                                            │ >6h ──► log.warning
           │                                                 ▼  (cannot attribute)  [B4-F8]
           │                                            ohlcv_4h   (source 100% NULL, A1-F13)
           ▼                                                 │
      FeaturePipeline ◄──────────────────────────────────────┘
           │   an absent bar is simply an absent row — no state, no flag
           ▼
      features_equity / features_crypto ──► observation vector ──► agent
```

**Nothing on this path writes a durable record of an observation event.** Every detector
terminates in a log line; the tables that could hold a verdict are empty or macro-only.

---

## B4.6 — Use, current and planned

### Current use — trainer

The trainer reads candles and features. It has **no concept of an observation event**. A missing
bar is an absent row: the series is shorter, indicator windows slide across the hole as if the
adjacent bars were adjacent in time, and nothing marks the discontinuity.

Concretely, for the 22 bars of small crypto gaps and the 159-bar hole, an agent trained on this
series learns from a 4-hour grid that silently contains **3.7-day** jumps presented as single
steps. The 46 equity early closes appear as ordinary sessions with anomalously low volume — an
input to turbulence (A5) and to volume-derived features (A2) that is real but **not comparable**
to the sessions around it.

### Current use — trader

The trader's only interaction with the calendar is the **gate** — and it asks the broker, not the
calendar (**B4-F6**). Once past the gate, the observation path is identical to the trainer's and
equally blind.

The staleness check (`_check_staleness`, `validation.py:317`) is the single mechanism that would
notice a *trailing* gap, and it is a freshness threshold (4 days equity / 8 hours crypto), not a
completeness check. During the 2026-03-11 → 04-06 outage it should have raised `DataError` for
equity within 4 days. **A1 §A1.3 establishes why alerting did not surface it; that mechanism is
A1's, not re-found here.** What is B4's: even a firing staleness check would have said *"data is
old"*, never *"27 observations that should exist do not"*, and would have left nothing behind.

### Observation slots occupied

**Zero.** No observation event reaches the agent's input vector in either environment — not as a
flag, a count, a recency or a mask. The agent cannot distinguish a market that was closed from a
market that produced nothing from a fetch that failed.

### Planned use — the training redesign

Group B exists in this audit because these datasets are **entering the observation space** in the
Stage 2.R redesign. Two consequences specific to B4, both **UNVERIFIED** against the redesign
documents (not read this session — they are assumptions under the contract's overriding rule):

- If observation events become an input, they need a per-bar representation that does not exist
  in either candle table today, and the point-in-time question of **A1-C17** applies to them —
  "was this venue outage known at bar time" is a knowledge-axis question of exactly the kind
  **B3-C12** raised.
- If they do not become an input, the classification is still required **upstream**, because
  §B3.12 ruling 2 moved the raw→adjusted build inside our system, and that build is where
  non-session bars are dropped.

### The DS-7 parity position

| Dimension | Equity | Crypto | Divergence justified? |
|---|---|---|---|
| Baseline exists | XNYS via library, rolling bound | **A constant, `4h`** | **No — a defect.** Crypto's baseline has never been stated, and is measurably false (**B4-F12**) |
| Baseline is stored | **No** | **No** | Symmetric absence — not a divergence |
| Incident record | **None** | **None** | Symmetric absence |
| Venue attribution possible | Single venue, implicit | **No** — `source` 100 % NULL, history venue HTTP 451 | **No — crypto is strictly worse** (**B4-F10**) |
| Half-day / partial-session concept | 46 early closes, unflagged | n/a — no venue analogue | **Justified divergence** |
| Gaps classified | not attempted | **not attempted** — though 7 venue outages provably exist | **No — a defect in both** |

**Crypto is the weaker half on every dimension where the two differ**, and the weakness compounds:
it has more real venue events, no venue recorded per bar, and no reachable authority for the
history in which most of those events occurred.

---

## B4.7 — Findings index

### Grouping test — recorded so it is not silently retried

The contract asks for grouping **by disposition** against a governing ruling, and requires the
test to be recorded.

**Tested: disposition against A1-C3 (the CBOE migration) + §B3.12 ruling 2 (the stored adjusted
table). It fails.** The migration is **equity-only** — LD-1 holds and crypto stays on Binance.US
— so all five crypto findings land in "untouched" for a mechanical reason that carries no
information. The resulting grouping reproduces the environment column, which already exists.

**Fallback applied, per the contract: grouped by severity.** B4 has no governing ruling of its
own; DS-10 created the dataset but rules nothing about its contents.

*(A second grouping — by half, calendar vs incident — was also considered and rejected: B4-F8,
B4-F9, B4-F11 and B4-F14 belong to both halves, and forcing them into one loses the point that
the two halves are inseparable, which is DS-10's entire premise.)*

### High — 21

| ID | Env | Finding | § | Feeds |
|---|---|---|---|---|
| **B4-F1** | equity | The session baseline is a **rolling ~20-year window** — no `start` at `validation.py:61`. First session **2006-08-04** (2026-08-04) → **2006-08-07** (2026-08-05); **653** CBOE dates, **11.5 %** of the migration's range, outside it, **growing daily** | B4.1(b) | **B4-C1**, **B4-C2**, **A1-C3** |
| **B4-F2** | equity | `validation.py:287` raises `DateOutOfBounds` **unguarded**; no `try` on the `:220` → `:272` → `:287` path, and `validate_batch`'s docstring does not admit it. **The CBOE migration raises on its first pre-2006 bar** | B4.3 | **B4-C2**, **A1-C3** |
| **B4-F3** | equity | `:140`'s `is_session` also raises pre-2006; `:146`'s bare `except Exception` records it as *"Step 7: zero volume (calendar check failed)"* — **a defect in the data, written to `data_quarantine`, when the truth is the calendar could not answer** | B4.3 | **B4-C2**, **B4-C7** |
| **B4-F4** | both | **Five** independent calendar constructions, **none** passing `start`, no shared helper, three via function-local imports. `get_calendar` caches globally, so a process **freezes its bound at start** and two processes on one database can disagree | B4.5 | **B4-C1**, **B4-C6** |
| **B4-F5** | both | **No stored calendar.** 71 tables, none holds sessions. The baseline is a runtime library call, so a past validation is **not reproducible** and a library upgrade **silently rewrites history**; `pyproject.toml:43` declares a floor, not a pin | B4.1(a) | **B4-C1** |
| **B4-F7** | both | **No venue or vendor incident record of any kind** — no table, no column, no status value, no vocabulary. *"Halt"* is entirely taken by our own circuit breaker, so the system **cannot name a venue halt** | B4.1(a), B4.3 | **B4-C4**, **B4-C8** |
| **B4-F8** | crypto | **Gap attribution is real and absent.** 13 tested days: **8 match Binance's archive exactly (7 venue-outage runs, 15 bars), 5 do not (5 runs, 7 bars — ours)**. Both populations are the same absent row in `ohlcv_4h` | B4.3 | **B4-C3**, **B4-C10** |
| **B4-F9** | both | The 2026-03-11 → 04-06 hole (**159 crypto bars + 18 equity sessions**, A1-F1/A1-F20) is **provably ours** — Binance.US serves every bar today. **One incident spanning both environments**, ~4 months open, nothing recording that the venue was healthy and nothing connecting the two halves | B4.3, B4.1(g) | **B4-C3**, **B4-C10** |
| **B4-F10** | crypto | **The venue that produced our pre-2019 history is unreachable** — `api.binance.com` → **HTTP 451**. With `ohlcv_4h.source` 100 % NULL (**A1-F13**), no historical crypto observation can be attributed *or* re-verified through any path the system has. The public archive works and **nothing uses it** | B4.3 | **B4-C4**, **B4-C5**, **A1-C18** |
| **B4-F11** | equity | **The gap detector cannot see the gap that matters.** `_detect_equity_gaps` takes `start`/`end` from the fetched batch's own min/max, so only **interior** gaps are findable — a **trailing** missing session, the shape of every live ingestion failure, is invisible by construction | B4.3 | **B4-C7** |
| **B4-F12** | crypto | **Crypto has no uptime baseline.** `_CRYPTO_FREQ = 4h` encodes "a bar every 4 hours forever" — no maintenance calendar, no venue status, no concept of a legitimately absent bar. **7 measured venue-outage runs prove the assumption false** | B4.1(f) | **B4-C5** |
| **B4-F17** | both | The candle quarantine path **has never fired**. `data_quarantine` = **3,945 rows, 100 % macro** — not one equity or crypto row in the table's life, across 18 missing sessions, 314 missing crypto bars, 2 phantom bars and 3 invalid SPY bars | B4.3 | **B4-C7**, **B4-C8** |
| **B4-F18** | equity | **A market-wide halt leaves no trace.** The **4 MWCB days of March 2020** sit in `ohlcv_daily` today as ordinary bars and pass all 12 validation steps — the spike check is set at **50 %**, the worst day moved **10.94 %**. No existing detector is even the right *shape*: nothing is missing, so no gap or staleness check can apply. *(Amended 2026-08-11: the halts are now **confirmed from a venue record** — **522 / 787 / 825 / 1,481** halt entries on those four days — no longer inferred from price. See **B4-F27**, **B4-F28**.)* | B4.1(h), B4.3 | **B4-C11**, **B4-C4** |
| **B4-F27** | equity | **A free retrospective halt source exists, is reachable, and nothing uses it.** NYSE `trade-halts/historical/download` — **71,718 records, 2019-02-22 → 2026-08-10**, unauthenticated, carrying symbol, exchange, **coded reason** and resume time. Its vocabulary already contains *"ETF Component Prices Not Available"*, the exact condition **B4-F19** had to infer. **It does not cover pre-2019**, so 2015-08-24 and the 2008 cluster — inside the range **A1-C3** imports — stay unsourced | B4.3 | **B4-C4**, **A1-C3** |
| **B4-F28** | equity | **One of our own instruments was halted and the bar does not say so.** **XLI — 2020-03-12 09:55:59 ET, NYSE Arca, `LULD pause`, resumed same day.** This is the **first observation event in this review confirmed from a venue record rather than from price behaviour**, and it moves **B4-F18** from market context to a symbol-level fact about a series we train and trade on | B4.3 | **B4-C11**, **B4-C8** |
| **B4-F29** | crypto | **Crypto has no venue-incident source at all.** Four candidates tested and all failed: `status.binance.us` does not resolve, `binance.statuspage.io` is inactive (401), `api.binance.com` is **451**, and Binance.US's authenticated status endpoint reports **current state only**. So **B4-F8**'s archive comparison is the *only* retrospective method, and it **infers** an incident from an absent bar — meaning a crypto observation event can carry **no reason, no duration, no resume time, no severity**. The DS-7 divergence from equity is now structural, not incidental | B4.3 | **B4-C4**, **B4-C5** |
| **B4-F19** | equity | **The most distorted observations in the whole equity history are unflagged, and A1-C3 will import them.** **2015-08-24** holds the **top 3 intraday ranges across 22 years and 8 ETFs** (21.71 %, 21.68 %, 21.28 %), with lows **17–22 % below the prior close** against closes down only 3–4 %. Outside `ohlcv_daily` (2016+); **inside the CBOE snapshot** | B4.1(h) | **B4-C11**, **A1-C3** |
| **B4-F20** | crypto | **A venue event permanently rescaled the volume series while leaving prices intact.** BTCUSDT 4h volume fell **~13× on medians** across **2023-06-06 → 06-08** and never recovered; **~3.1× of it is venue-specific** against Binance Global. Price divergence over the same span is **≤ 0.60 %**, so every price-based check agrees and sees nothing. **Not A1-F9/A1-C2** — those are *source-change* seams; here the source is unchanged on both sides. *(Unit caveat: our crypto `volume` is **quote volume in USD** (`binance.py:200`, `:218`) while the Global comparison used **base** volume, so the ours÷Global ratios are mixed-unit — only their **change** is claimed. Our own 13× series is internally consistent.)* | B4.1(h), B4.1(i) | **B4-C11**, **B4-C12** |
| **B4-F23** | both | **Neither candle table can name its vendor or its venue.** `ohlcv_daily` has **no source column at all**; `ohlcv_4h.source` is **100 % NULL** (**A1-F13**). A venue-scoped incident therefore cannot be applied to the rows it affected — and a **vendor** outage cannot be told from a **venue** outage, which are different truths about the market with an identical symptom | B4.1(i) | **B4-C3**, **B4-C4**, **A1-C3** |
| **B4-F24** | both | **The database session timezone is `America/New_York`, not UTC.** A bare date literal against a `timestamptz` column silently resolves **4–5 hours off**; the table spans **25,059 EDT / 13,674 EST** rows. **This review's own audit query dropped 10 bars this way before it was caught.** Contradicts the project's *UTC internally* rule at exactly the edge where observation completeness is judged | B4.1(i) | **B4-C15**, **B4-C7** |
| **B4-F26** | both | **`success` is logged while data is being lost.** Over 2026-03-11 → 04-01 the log records **108 equity `success` runs** and 18 sessions went missing anyway; macro records **436 successes** against **48** retained rows. The window hit **equity + crypto + macro**, so an incident record must be **dataset-wide** — a per-environment one splits a single event into three. Escalates **B4-F14** from *wrong vocabulary* to *actively misleading* | B4.1(i) | **B4-C3**, **B4-C4**, **B4-C7** |

### Medium — 8

| ID | Env | Finding | § | Feeds |
|---|---|---|---|---|
| **B4-F6** | equity | **Two calendar authorities, never reconciled** — the trading gate asks the **Alpaca broker clock** (`pipeline.py:1128`), bar existence asks **xcals** (`validation.py:61`). Nothing compares them; a disagreement is undetectable | B4.5 | **B4-C6** |
| **B4-F13** | crypto | `resolve_crypto_gaps` — the **only** gap-resolution logic in the codebase — is **dead** (tested, zero production callers), and if live would `ffill(limit=2)` **9 of the 12 small gap runs, including 6 of the 7 real venue outages**, into unmarked fabricated bars | B4.4 | **B4-C3** |
| **B4-F14** | both | `data_ingestion_log` **cannot express "the market was closed"**. Equity logs `success` **63×** and `failed` **19×** on non-sessions — the same closed day, opposite verdicts — while `no_data` appears **24× on real sessions**, where it is a genuine problem | B4.3 | **B4-C3**, **B4-C4** |
| **B4-F15** | equity | **46 early closes** enter `ohlcv_daily` and the features unflagged, at roughly half normal volume. `is_early_close()` exists and works — wired to **options only** (`options/collector.py:119`). **B3-F20**'s shape, exactly where B3 predicted it | B4.5 | **B4-C9** |
| **B4-F16** | equity | `_EQUITY_GAP_THRESHOLD = 5 days` is right by luck. Measured max legitimate gap **is exactly 5 days — zero margin** — and one of its two instances is **Hurricane Sandy 2012-10-26→31**, an unscheduled venue closure, not the "holiday combo" the comment claims | B4.1(e) | **B4-C2**, **B4-C9** |
| **B4-F21** | both | **A vendor bar revision is undetectable by construction.** Every candle write is `ON CONFLICT DO NOTHING`, so a corrected bar is discarded in silence. **Measured clean for crypto** — 3,409 bars × 5 fields, **0 mismatches** — which is evidence of *no harm so far*, not of a working control. **Untested for equity**, and **A1-C3 introduces a new vendor** whose revision behaviour is unknown. **B3-F27** established that this vendor class does restate | B4.1(i) | **B4-C3**, **A1-C3** |
| **B4-F22** | both | **A duplicate bar is resolved silently and recorded nowhere.** Step 8 keeps the **last** row and emits one `log.warning` (`validation.py:210-217`); the primary key then makes the discarded row unrecoverable. Two conflicting bars for one timestamp is an observation event — the system picks one and forgets the conflict happened | B4.1(i) | **B4-C7**, **B4-C8** |
| **B4-F25** | both | **Our own interventions in the data are unrecorded.** `operator_actions` holds **0 rows**, yet the 2026-07-18 partial bar was repaired by replacing a row that `ON CONFLICT DO NOTHING` cannot replace — a manual intervention with no trace. This is **B3-C5**'s remediation ledger, now with a concrete instance in candles rather than corporate actions | B4.1(i), B4.0 | **B4-C8**, **B3-C5** |

### Low — 0

No finding was judged Low. Every item either blocks a ruled decision, corrupts a stored record,
or removes the ability to tell a defect from a fact.

### Candidates rejected — the scope test, applied at numbering time

*"Is this a fact about observation events?"* — not *"did this session find it"* (§B3.11).

| Candidate | In B4? | Reasoning |
|---|---|---|
| The 2026-03-11 → 04-06 hole **exists** | **No** | Candle completeness — **A1-F1**, **A1-F20**. Cited, never re-found |
| That hole's **attribution** | **Yes** — B4-F9 | Nothing records *what kind* of event produced it. That is the dataset's subject |
| The 22-day 2019 crypto hole | **No** | **A1-F19**, already established as a stitch artefact. Used as B4's cautionary precedent (§B4.2) |
| The 2 phantom CBOE bars | **No** | **A1-F10**. Used as evidence the calendar is their sole detector |
| `ohlcv_4h.source` is 100 % NULL | **No** | **A1-F13**. Cited where it blocks attribution |
| The 3 invalid SPY bars | **No** | **A1-F6**. Cited only as quarantine-never-fired evidence |
| Alpaca IEX-vs-SIP volume seam | **No** | A1's. Untouched |
| `technical.py` weekly→daily forward-fill | **No** | A derived-feature transformation — **A2's**. Flagged forward, not numbered |
| Whether XNYS is the right calendar for Arca/Nasdaq ETFs | **No** | Tested and found to be a **non-issue** — 0 session differences vs XNAS. Recorded in §B4.1(d) so it is not re-tested |
| The `exchange-calendars` version floor | **Folded into B4-F5** | A packaging fact on its own; load-bearing only as part of the unreproducible-baseline finding |
| The 2015-08-24 and March-2020 distorted bars | **Yes** — B4-F18, B4-F19 | The **instrument** did not change; our **view** did. Correct handling is a flag, never an adjustment — the exact B3/B4 dividing line |
| The June 2023 crypto volume step | **Yes** — B4-F20 | Checked against A1 before numbering: **A1-F9 / A1-C2 are the Global→US *stitch* seam**, a source change. Same venue on both sides here, so it is a different phenomenon |
| The extreme daily-return census at `A1-candles.md:204` | **No** | A1's, and it was hunting malformed bars, not venue events. Not re-found |
| That the halt-day bars are arithmetically valid | **No** | Not a defect at all — it is the *evidence* for B4-F18. Passing validation is the finding |
| That crypto `volume` is **quote**-denominated USD | **No** | A candle **field definition** — A1's. Recorded only as a unit caveat on B4-F20 and as the correction to my own measurement (§B4.0) |
| That **A1-F9**'s ~200× seam is genuine liquidity, not a unit bug | **No** | **A1's finding, confirmed not re-found.** Both parsers use `volume_quote` (`binance.py:218`, `:383`). Stated explicitly so the spec pass does not re-open a settled finding on the strength of a mis-run comparison |
| The in-progress-bar guards | **No** | Working as designed, in both environments. A tested-clean hypothesis in §B4.0, not a finding |
| The 2026-07-18 partial *repair* | **Yes** — B4-F25 | The guard is not the finding; the **unrecorded manual intervention** is |

---

## B4.8 — What the disposition means

**The severity grouping is not a fallback of convenience.** B4's findings do not sort against a
ruling because B4 is *upstream* of every ruling that touches it. A1-C3 cannot proceed through
B4-F1/F2, and §B3.12 ruling 2's adjusted-table build cannot know which bars to drop without the
baseline. A disposition column would be measuring the wrong direction of dependency.

**The two halves are inseparable, and the findings prove DS-10 right.** Four findings —
**B4-F8**, **B4-F9**, **B4-F11**, **B4-F14** — cannot be assigned to "calendar" or "incidents"
because each is about the *boundary*: whether an absence is expected. Had the two been split into
separate datasets, each of those four would have fallen between them, which is precisely how the
rolling bound and the phantom bars came to have no owner in the first place.

**"High" here mostly means "cannot distinguish", not "is wrong".** Very little in this dataset is
incorrect — the XNYS calendar is accurate, CBOE is session-complete, and the 4-hour crypto grid is
right 98.4 % of the time. The severity comes from the system's **inability to tell two situations
apart**: a closed market from a failed fetch, a venue outage from our own loss, an unanswerable
calendar from a suspect bar. Each of those confusions has already produced a wrong record —
A1-F19's misattributed outage, B4-F3's misfiled quarantine reason, B4-F14's contradictory
statuses.

**One finding is conditional. B4-F2 is latent, not active.** Nothing raises `DateOutOfBounds`
today, because `ohlcv_daily` starts 2016-01-04 and the four non-validation calendar sites only
ask about recent dates. It becomes an active failure **the moment A1-C3 runs**, and only then.
Its High severity is a statement about the migration, not about the running system.

**Two findings are worsened by the passage of time alone.** B4-F1's excluded-session count grows
by one per trading day with no code change — measured at 652 and 653 on consecutive days. B4-F9's
repairability is the mirror image: the archive and the live API still serve the missing bars
today, but that is a vendor retention policy nobody controls.

**Three findings invert the dataset's own framing, and they are the most dangerous ones.** B4-F18,
B4-F19 and B4-F20 are not absences. The bar is there, it is well-formed, it passes all twelve
validation steps — and it is still not a usable observation, because the venue that produced it
was not functioning normally. This class defeats **every** mechanism in the system by
construction: gap detection needs something to be missing, staleness needs the data to be old,
cross-source comparison agrees (B4-F20's prices match Binance Global to 0.006 %), and no threshold
on the bar itself can help, because a genuine 21 % range and a broken one look identical. The only
thing that separates them is **knowing an event occurred** — which is the record B4-F7 says does
not exist.

This is also why the two halves of DS-10 had to stay together. An absence-only dataset would have
found none of these three, and a "data quality" framing would have rejected all three, because
nothing about the data is wrong.

**Three clean negatives are load-bearing, not filler.** The completeness pass tested nine classes
and cleared three: in-progress bars are guarded in both environments, no vendor revision has
occurred in 3,409 sampled crypto bars, and no symbol or pair has changed under us. Each is recorded
in §B4.0 with its evidence because an untested risk and a tested-clean risk are different objects,
and only the second can be safely deprioritised. **None of the three is a working control** — the
first is a real guard, but the other two are absences of harm, and B4-F21 says so explicitly.

**The pass also caught one of this review's own errors**, and the failure mode is worth naming:
the first revision test compared quote volume against base volume and reported a 100 % mismatch.
Had it been believed, it would have manufactured a finding *and* appeared to overturn **A1-F9**.
The correction is in §B4.0 for the same reason the other negatives are — the trap is still there
for the spec pass.

**One finding is good news, and is recorded as such.** B4-F8's classification was *possible* —
Binance's public archive answered every question asked of it, for free, without authentication.
The 5 provably-ours gap runs are repairable, and the 7 venue outages are documentable. The
capability gap is not evidentiary; it is that nothing has ever looked.

**What crypto's weakness actually consists of.** The DS-7 table in §B4.6 shows crypto worse on
every dimension where the two environments differ. That is not a resourcing observation — it is
structural. Equity's baseline is wrong in a *bounded, fixable* way (a missing argument). Crypto's
baseline has **never been articulated at all**, and the venue that would arbitrate most of its
history is unreachable. The equity half has a defect; the crypto half has an absence.

---

## B4.9 — Carry register

Ten items. **No remedies** — each records a question or a requirement, not a design.

| ID | Kind | Statement | Anchored in |
|---|---|---|---|
| **B4-C1** | Scoping | **The session baseline must be sourced, owned, stored and reproducible for a past date.** Today it is a runtime library call made in five places, with no store, so "what should have existed on date D" cannot be answered as of any moment other than now | **B4-F1**, **B4-F4**, **B4-F5** |
| **B4-C2** | Correctness | **Settle what span the baseline must cover**, given the bound is rolling and the shortfall grows by one session per trading day. The span question is not "how far back is nice" — it is set by the earliest bar any consumer will validate, which the CBOE migration puts at **2004-01-02** | **B4-F1**, **B4-F2**, **B4-F3**, **B4-F16** |
| **B4-C3** | Correctness | **Settle how an absent observation is classified** — legitimately absent / venue outage / our loss / filter artefact — and where the classification is recorded. All four already coexist in the stored data, and **B4-F8** proves they are currently indistinguishable | **B4-F8**, **B4-F9**, **B4-F13**, **B4-F14** |
| **B4-C4** | Scoping | **An observation-event record must be sourced and owned for both environments.** *(Restated 2026-08-11 after the source hunt — the question has narrowed, not closed.)* **Equity 2019+ is solved**: NYSE's halt history is free, unauthenticated and coded (**B4-F27**). **Two blanks remain and neither has a candidate**: equity **pre-2019-02-22** — which is most of what **A1-C3** imports — and **all of crypto**, where four sources were tested and all failed (**B4-F29**). The vocabulary problem stands: *"halt"* is already taken by our own circuit breaker, and the vendor's coded reasons have nowhere to land | **B4-F7**, **B4-F10**, **B4-F14**, **B4-F27**, **B4-F29** |
| **B4-C5** | Parity | **Settle the crypto uptime baseline.** What "should exist" means for a 24/7 venue has never been stated, and the implicit answer (`4h` forever) is measurably false. Includes whether the Global-archive era and the Binance.US era can share one baseline at all | **B4-F10**, **B4-F12** |
| **B4-C6** | Wiring | **Reconcile the two calendar authorities** — the Alpaca broker clock and xcals — or record the divergence as justified under DS-7. Today nothing would notice them disagreeing | **B4-F4**, **B4-F6** |
| **B4-C7** | Correctness | **Settle what a detector must be able to see.** The current one cannot see a trailing gap, cannot write anything durable, and has never fired for a candle. A detector that cannot fire on the newest missing bar cannot detect a live ingestion failure | **B4-F3**, **B4-F11**, **B4-F17** |
| **B4-C8** | Schema | **Settle whether an observation flag lives in the candle row or a side table.** Directly coupled to **A1-C11** (candle rows carry audit state) and to A1-C3's full-table replacement, which in-row state does not survive | **B4-F7**, **B4-F17** |
| **B4-C9** | Correctness | **Settle whether a half-day session is a comparable observation.** 46 early closes carry roughly half a session's volume and are currently indistinguishable from full sessions everywhere except options | **B4-F15**, **B4-F16** |
| **B4-C10** | Data repair | **The provably-ours crypto losses are still recoverable** — 5 gap runs (7 bars) from the public archive and the 159-bar hole from the live Binance.US API, both verified serving today. Scope and sequencing sit with **A1-C1** | **B4-F8**, **B4-F9** |
| **B4-C11** | Correctness | **Settle how a "present but distorted" observation is handled** — a bar that exists, is arithmetically valid, and passes every check, but whose price discovery was impaired: a market-wide halt, an LULD cascade, a venue dislocation. **Distinct from B4-C3**, which classifies *absences*. Nothing is missing, so no absence-shaped detector can ever apply, and no threshold on the bar itself separates a real 21 % range from a broken one | **B4-F18**, **B4-F19**, **B4-F20** |
| **B4-C12** | Correctness | **Settle whether crypto volume is comparable across 2023-06-07.** A **~13×** permanent level change sits mid-series with no marker, roughly a third market-wide and the rest venue-specific. Adjacent to **A1-C2** (the two *source* seams) but a **different cause**, so fixing those does not fix this | **B4-F20** |
| **B4-C13** | Scoping | **Symbol and pair lifecycle must be represented** — listing, delisting, suspension, ticker reuse. Measured **clean today** (both crypto pairs `TRADING`; the equity symbol list unchanged since `720e03e`), which makes this a **latent** gap rather than a live one: nothing in the system would tell us if it changed, and a reused ticker would splice two different instruments into one series | *(clean negative, §B4.0)* |
| **B4-C14** | Correctness | **Observation events need a knowledge date.** `fetched_at` records when a **bar** arrived, never when something **became known about** it. This session classified 2017–2023 gaps in 2026 — a past effect-date with a present knowledge-date, and nowhere to record it. The same axis **B3-C12** raised for corporate actions, applied to observations | **B4-F8**, **B4-F9**, **B4-F25** |
| **B4-C15** | Wiring | **Settle the storage and query timezone convention.** A `timestamptz` column under an `America/New_York` session makes every date-bounded query silently wrong by 4–5 hours unless it casts explicitly — including the queries that judge completeness | **B4-F24** |

---

## B4.10 — Dependency map

### Within B4

| Blocked | Blocked by | Why |
|---|---|---|
| **B4-C2** (span) | **B4-C1** (the baseline exists at all) | A span is meaningless without a store to hold it |
| **B4-C3** (classification) | **B4-C1**, **B4-C4** | Classifying an absence needs both the expected-set and an incident source; without either, only "row absent" is knowable |
| **B4-C7** (detector) | **B4-C1**, **B4-C3** | A detector compares observed against expected and must name what it found — it needs both inputs |
| **B4-C8** (where the flag lives) | **B4-C3** | The shape of the record follows from what is being recorded |
| **B4-C9** (half-days) | **B4-C1** | An early close is an attribute of a session, so it presupposes a stored session |
| **B4-C5** (crypto baseline) | **B4-C4** *(partially)* | The uptime baseline for the pre-2019 era depends on whether any source can speak for a venue that is HTTP 451 |
| **B4-C10** (repair) | **B4-C3** | Repairing without classifying would overwrite 7 genuine venue outages with fabricated continuity |

```
   B4-C1 (baseline exists, stored) ──┬──► B4-C2 (span) ──► A1-C3   [653 sessions, +1/day]
                                     ├──► B4-C9 (half-days)
                                     └──► B4-C3 (classification) ──┬──► B4-C7 (detector) ──► A1-C12
                                                ▲                  ├──► B4-C8 (where) ◄──► A1-C11
                                                │                  └──► B4-C10 (repair) ──► A1-C1
   B4-C4 (incident source) ─────────────────────┘
        └──► B4-C5 (crypto baseline) ──► A1-C18   [Global is HTTP 451]

   B4-F1 / B4-F2  ──►  A1-C3   BLOCKS TODAY, and worsens daily
   B4-F8 / B4-F9  ──►  the classification is not theoretical: 4 states already coexist
```

### Crossing dataset boundaries

| Blocked | Blocked by | Why |
|---|---|---|
| **A1-C3** — the CBOE migration | **B4-F1**, **B4-F2** | `validation.py:287` raises `DateOutOfBounds` unguarded on any pre-2006 date and `:220` does not wrap it, so the migration fails on its first pre-2006 bar. **653 sessions affected today, +1 per trading day.** This *sharpens* the edge the spine already carried from B3 — the count was 647 and the growth rate was recorded as yearly |
| **A1-C3** — the CBOE migration | ← *also informed by* **B4-F3** | The pre-2006 quarantine path does not crash but **mislabels**: bars the calendar could not judge are recorded as zero-volume defects |
| **A1-C1** — restore the crypto hole | ← *extended by* **B4-C10**, **B4-F8** | A1-C1 scopes the 2019 hole. B4 adds **5 further gap runs that are provably ours** and the **159-bar 2026 hole**, all still served by the venue today — and warns that 7 *genuine* venue outages must not be filled alongside them |
| **A1-C11** — candle rows carry audit state | ↔ **B4-C8** | Observation flags and audit state are the same schema question asked twice; both must survive A1-C3's full-table replacement |
| **A1-C12** — cross-source validation gate | ← *strengthened by* **B4-F8**, **B4-F11** | B3-F26 made cross-source the only completeness signal for events. B4 shows the same for observations: the internal detector cannot see a trailing gap, so an **external** comparison is the only thing that can. B4-F8 demonstrates a working one that costs nothing |
| **A1-C18** — a free independent crypto source | ← *sharpened by* **B4-F10** | Independence is not the only requirement — **reachability of the historical venue** is a separate, unmet one. `data.binance.vision` is demonstrated free, unauthenticated and complete for the tested dates |
| **A1-F13** — `ohlcv_4h.source` 100 % NULL | ← *consequence recorded by* **B4-F10** | A1 recorded the NULL; B4 records what it costs — no bar can be attributed to a venue, so venue-scoped incidents cannot be applied to the segment they affected |
| **A1-C3** — the CBOE migration | ← *also carries* **B4-F19** | The migration imports **2015-08-24**, which holds the 3 largest intraday ranges in 22 years, plus the 2008-10 cluster beneath it. These are not in today's data — the migration **adds** them, unflagged |
| **A1-C2** — the two venue-volume seams | ↔ **B4-C12** | Adjacent but **distinct causes**: A1-C2 is *source change* (Global→US stitch, IEX/SIP); B4-C12 is a *venue event* with the source unchanged on both sides. A fix for one does not address the other, and a normalisation that assumed a single seam per series would be wrong |
| **A2 Derived features** | ← *warned by* **B4-F15**, **B4-F13**, **B4-F18**, **B4-F19**, **B4-F20** | 46 half-day sessions and 22 gap bars enter the feature pipeline unmarked, and `technical.py:126/174` forward-fills across them. Added: halt-day bars and a **13× permanent volume rescale** at 2023-06-07 — any volume-derived or volatility feature spans that break with no marker |
| **A5 Turbulence**, **A4 Regime** | ← *warned by* **B4-F18**, **B4-F19**, **B4-F20** | Turbulence percentiles and HMM fits are computed over series containing the four MWCB days, the 2015-08-24 dislocation (post-migration) and the crypto volume break. Each is a genuine extreme in the data and a **non-comparable** observation in reality |
| **B2 Calendar events** | ← *boundary with* **B4-C4** | `calendar_events` (553 rows, 2015-01-09 → 2027-12-08) holds *scheduled economic* events. B4's calendar is *market structure*. The two are different datasets sharing a word — the boundary must be stated, not assumed |
| **B3-C5** — the remediation ledger | ← *evidenced by* **B4-F25** | B3-C5 was raised for corporate actions. B4 supplies a **concrete candle instance**: the 2026-07-18 partial bar was repaired by replacing a row that `ON CONFLICT DO NOTHING` cannot replace, and nothing records it. The ledger is not hypothetical |
| **B3-C12** — the knowledge axis | ↔ **B4-C14** | The same axis on the other dataset. B3-C12 asks it of events; B4-C14 asks it of observations, and this session created the instance — 2017–2023 gaps classified in 2026 |
| **A1-C3** — the CBOE migration | ← *also blocked by* **B4-F21**, **B4-F23** | The migration brings a **new vendor** whose revision behaviour is unknown, into tables that **cannot name a vendor**. `ON CONFLICT DO NOTHING` would discard any correction it issued, silently |
| **A1-C3** — the CBOE migration | ← *coverage gap from* **B4-F27** | The halt source starts **2019-02-22**. The migration imports **2004-01-02 onward**, so roughly **15 of the 22 years it adds have no halt record available at any price** — including 2015-08-24 (**B4-F19**) and the 2008 cluster |
| **A1-C18** — free independent crypto source | ← *also informed by* **B4-F29** | The crypto blank is not only about *prices*. Four venue-incident candidates failed, so whatever source is chosen must be judged on whether it can speak to **venue state**, not just quotes |
| **Every remaining dataset** | ← *warned by* **B3-F20** | Confirmed for B4: `is_early_close` is wired to **options only**, exactly the one-dataset-of-twelve shape B3 predicted |
| **Every remaining dataset** | ← *warned by* **B4-F24**, **B4-F26** | Two cross-cutting hazards, neither specific to B4: the session timezone silently shifts date-bounded queries in **any** dataset's audit, and `data_ingestion_log`'s status is unreliable for **all** of them — it logged 436 macro successes for 48 retained rows |

---

## Forward notes

**To A2 (Derived features).** Two observation-event populations enter the feature pipeline
unmarked and must be assessed there, not re-found: **46 early-close sessions** at roughly half
normal volume (**B4-F15**), and **22 gap bars** whose absence shortens indicator windows without
any discontinuity marker. `technical.py:126` (weekly → daily) and `technical.py:174` (daily → 4H)
both reindex with `method="ffill"`, so an observation event **propagates** into derived features
with nothing travelling with it. A2 should also settle whether a feature computed across a venue
outage is recomputable at all — which compounds **A1-F32** (UNVERIFIED).

**To B1 (Options chains).** B1 is the **only** dataset that already treats observation events as
first-class: `options_chains.is_early_close` is populated, and the capture job gates on
`is_trading_day` (`collector_main.py:144`). B1 should record *why* it got this right when eleven
other datasets did not — that reasoning is the closest thing to a design precedent B4 has. Note
also that `market_calendar.py:19` caches a module-level singleton on top of the library's own
cache, so B1 inherits **B4-F4**'s frozen-bound behaviour.

**To B2 (Calendar events).** B2 and B4 both hold something called a calendar and they are not the
same thing: B2's `calendar_events` holds **scheduled economic releases** (FOMC, FRED), B4's holds
**market structure** (is the venue open). B2 should state the boundary explicitly rather than
inherit the collision. Worth checking whether B2's events are themselves keyed to sessions — an
FOMC release on a non-session date would be a defect only B4's baseline can catch.

**To A5 (Turbulence) and A4 (Regime).** Both consume return series computed across the gaps in
this review. A 3.7-day jump presented as one 4-hour step, and a half-day session's volume, feed
directly into turbulence percentiles and HMM fits. Neither dataset can see the discontinuity.

---

## Evidence base

- **Live Postgres (pg16)** — full 71-table list; `data_quarantine` (3,945 rows, full
  `GROUP BY source, severity, failure_reason`); `data_ingestion_log` (1,759 rows, joined against
  XNYS sessions); `alert_log`, `api_errors`, `system_events`, `operator_actions`,
  `emergency_flags`, `circuit_breaker_events`, `calendar_events` row counts and ranges;
  `ohlcv_daily` and `ohlcv_4h` full timestamp extracts and DDL.
- **`exchange_calendars` 4.13.1, run live on two consecutive days** — `XNYS` bound measured
  2026-08-04 and 2026-08-05; `XNAS`, `ARCX`, `BATS` bounds; XNYS-vs-XNAS session and early-close
  set comparison over 5,681 sessions; session-interval distribution; early-close census;
  `DateOutOfBounds` reproduced for both `sessions_in_range` and `is_session`;
  `start="1990-01-01"` verified; `get_calendar` identity check for the global cache.
- **Live Binance.US API** (`api.binance.us/api/v3/klines`) — 4h bars for 2026-03-11 and
  2026-04-06, both symbols.
- **Binance public archive** (`data.binance.vision`) — **daily** 4h zips for 13 dates spanning
  2017-09-06 → 2023-02-06, compared bar-for-bar against the stored series; **monthly** 4h archives
  for 2023-03 → 2023-09 (**1,282 bars**) for the close-divergence and volume-ratio study.
- **Live Postgres, volume series** — BTCUSDT monthly mean/median 4h volume 2022-09 → 2024-03, and
  daily totals 2023-05-25 → 2023-06-20 to date the break.
- **CBOE vendor snapshot, all 8 ETFs** — full intraday-range ranking over 2004 → 2026; the four
  March 2020 MWCB sessions; the 2015-08-24 open/low/close census.
- **Binance.US re-fetch for revision detection** — **3,409 4h bars** across 7 windows 2020→2026,
  compared to the stored series on all five OHLCV fields (run twice: once with the wrong volume
  unit, once corrected).
- **Binance.US `exchangeInfo`** — listing status for BTCUSDT and ETHUSDT.
- **Postgres metadata** — `SHOW timezone`; `'2020-01-01'::timestamptz`; PK/constraint definitions
  for both candle tables; EDT/EST row split; `fetched_at` vintage counts; `data_ingestion_log`
  status census restricted to 2026-03-11 → 04-08; `macro_features` and `options_chains` coverage.
- **Git** — `git log -L` on the `config/swingrl.yaml` equity symbol line.
- **Source hunt, 2026-08-11** — live probes of `nyse.com/api/trade-halts/{current,historical}`,
  `nasdaqtrader.com` (halt RSS + `TradeHalts` + `TradeHaltHistory`), `api.nasdaq.com`,
  `sec.gov`, `status.binance.us`, `binance.statuspage.io`, and both Binance
  `sapi/v1/system/status` endpoints. The NYSE historical CSV (**71,718 rows**) was parsed in full:
  per-year counts, reason-code census, per-symbol filter against our 8 ETFs, and per-date counts
  for the four MWCB dates and 2015-08-24.
- **Code, read at line level (completeness pass)** — `binance.py` (`_incomplete_bar_open_ms`,
  `_drop_incomplete_bars`, `_parse_klines`, `_parse_archive_csv`, `_KLINES_COLUMNS`),
  `alpaca.py:98-124`, `base.py`, `pg_helpers.py`, `gap_fill.py` (conflict policy).
- **Binance Global** (`api.binance.com`) — HTTP 451 confirmed from this host.
- **CBOE vendor snapshot** `~/swingrl/data/vendor_snapshots/cboe/2026-07-25/cboe_SPY.json.gz` —
  5,676 distinct dates 2004-01-02 → 2026-07-23, compared against the XNYS session set.
- **Code, read at line level** — `data/validation.py` (all 354 lines), `data/ingest_all.py`
  (gap path), `data/verification.py` (thresholds), `data/options/market_calendar.py` (all 42
  lines), `execution/emergency.py`, `execution/risk/circuit_breaker.py`, `execution/pipeline.py`
  (calendar gate + staleness), `scripts/collector_main.py`, `pyproject.toml`, `uv.lock`,
  `config/swingrl.yaml`.
- **Repo-wide greps** — calendar/`xcals` construction sites; `halt|outage|incident|suspend|delist|maintenance`;
  `is_early_close` / `recent_sessions` / `market_calendar` callers; `resolve_crypto_gaps` callers;
  `binance.vision`; `ffill|reindex|interpolate` across `features/` and `data/`.

---

## Confidence

Weaker than the rest, explicitly:

| Claim | Status |
|---|---|
| The 7 venue-outage runs are genuine Binance outages | **Measured as agreement between two Binance surfaces** (our stored bars and Binance's own archive), not confirmed against an incident announcement. Agreement proves the archive does not hold the bars; it does not independently prove the venue was down |
| The 5 "ours" gap runs are collector-side losses | **Measured** — the archive holds bars we do not. The *mechanism* of each loss is **UNVERIFIED**; no log survives from those dates |
| The pre-2019 crypto history came from the Binance Global archive | **Inherited from A1** (`STITCH_DATE`, `binance.py:50-51`), not independently re-verified this session |
| The 2026-03-11 → 04-06 equity and crypto gaps are **one** incident | **Inference from coincident dates**, marked **UNVERIFIED**. The windows match to the day and the equity gap is identical across all 8 symbols, but no log or record ties them |
| `resolve_crypto_gaps` would fill 9 of 12 runs if live | **Arithmetic on measured run lengths**, not executed. The function is dead, so this is a statement about what its code would do |
| The redesign's intent for observation events | **UNVERIFIED** — the Stage 2.R documents were **not read** this session. §B4.6's "planned use" is reasoning from the audit's own premise that Group B enters the observation space |
| A library upgrade would change historical sessions | **Reasoned from the absence of a store**, not demonstrated by diffing two `exchange_calendars` releases |
| `_check_staleness` should have fired during the 2026 outage | **Reasoned from the 4-day threshold and the 27-day gap.** Why it did not surface is **A1's** territory and was not re-derived |
| Half-day volume is "roughly half" a session | **Not measured this session.** Stated as the reason the flag matters, not as a quantity. The 46 early-close dates are measured; their volumes are not |
| The XNYS-vs-XNAS equivalence | **Measured over 2004–2026 only.** Says nothing about pre-2004 or about intraday differences |
| ~~That trading was **halted** on the four March 2020 dates — recalled, not verified~~ | **SUPERSEDED 2026-08-11 — now VERIFIED from a venue record.** NYSE's halt history returns **522 / 787 / 825 / 1,481** entries on those four dates, plus a named `LULD pause` on **XLI** (**B4-F28**). Replaced rather than deleted: this entry existed for six days and the correction is the point |
| That **LULD halts** caused the 2015-08-24 lows | **Still inferred — the halt source does not reach that far back** (it starts 2019-02-22). The **price behaviour is measured**; the mechanism is not. Weight added, though: the vendor's own reason vocabulary includes *"ETF Component Prices Not Available"*, so the failure mode is one the exchange names — just not for a date we can query |
| The June 2023 volume break was caused by the regulatory action | **UNVERIFIED — deliberately.** The **break is measured** (13×, dated to 2023-06-06→08, 3.1× venue-specific). Attributing it to a specific external cause is exactly the leap **A1-F19** shows going wrong. The finding claims a venue event, not a named one |
| The market-wide share of the crypto volume decline | **Estimated from the ours÷Global ratio**, which is computed across two series on **different units** (**A1-F9**). The *ratio change* is meaningful; the ~⅓ / ~⅔ split is approximate |
| That 2015-08-24's bars are "correct" | **Measured as internally consistent** (OHLC ordering, bounds). Not verified against a second vendor for that date — the snapshot is single-source pre-2016 |
| "No vendor revisions have occurred" | **Measured on 3,409 crypto bars in 7 sampled windows** — not the full 19,335-bar series, and **not at all for equity**. A revision outside the sample would not have been seen. The finding (**B4-F21**) claims *exposure*, not absence |
| That the 2026-07-18 partial was **repaired** rather than never committed | **Inferred** from `fetched_at` on the 2026-07-19 00:00 UTC bar being a later vintage than the backfill that would have written the partial. The code comment (`binance.py:487-489`) states the partial *was* stored. No operator record exists to confirm — which is **B4-F25** itself |
| Who performed that repair, and how | **UNVERIFIED and unknowable from the data.** `operator_actions` is empty |
| The equity in-progress guard is sufficient | **Read, not tested.** `alpaca.py:98-111` caps `end` to start-of-today UTC, which is conservative for a 16:00 ET close, but no equity re-fetch comparison was run |
| That no symbol has ever changed | **Measured for the current config line only** (`git log -L 9,9`). A symbol added and later removed *between* commits to that line would not appear |
| "No crypto venue-incident source exists" | **Four candidates tested and failed.** That is *"I did not find one"*, **not** *"none exists"* — a paid feed, an archived status history, or a third-party aggregator was not searched for. **B4-F29** claims the tested set, not the universe |
| The NYSE halt history is complete for 2019+ | **Its range and row count are measured**; its *completeness within* that range is not verified against a second source. It is also **NYSE-published** — whether it covers every venue's halts equally is untested, though rows carry `Exchange = Nasdaq`, `NYSE American`, `NYSE Arca` |
| That the endpoint is stable or intended for programmatic use | **UNVERIFIED.** It ignored the `startDate`/`endDate` parameters I passed and returned the full file, which suggests an undocumented surface. No terms-of-use check was done |

Everything else carries a `file:line`, a query, or command output produced this session.

---

**Status: COMPLETE — 29 findings (21 High / 8 Medium / 0 Low), 15 carry items.**
**Third dataset reviewed; moved ahead of A2 by DS-11. No finding re-states an A1 or B3 fact — the rejected-candidate table records the test.**
**Amended 2026-08-05 after a user question — §B4.1(h) added, measuring the "present but distorted" class (B4-F18, B4-F19, B4-F20; B4-C11, B4-C12). One of this review's own hypotheses was REFUTED by that measurement (Binance.US price dislocation) and one of its own claims corrected (the 12 small crypto gaps were enumerated by A1) — both recorded in §B4.0.**
**Amended again 2026-08-10 after "what haven't we considered?" — §B4.1(i) records a nine-class completeness pass: 6 findings (B4-F21…F26), 3 carry items (B4-C13…C15), and 3 clean negatives logged in §B4.0 with their evidence. One of this review's own measurements was wrong and was corrected before it was written down.**
**Source hunt 2026-08-11 (§B4.3) — equity halts have a free retrospective source (71,718 records, 2019-02-22→); equity pre-2019 and all of crypto have none. B4-F27/F28/F29 added, B4-C4 restated, B4-F18 upgraded from inferred to venue-confirmed, and one §Confidence entry superseded in place.**
**Not yet walked with the user.**
