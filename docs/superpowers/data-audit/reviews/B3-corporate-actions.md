# B3 — Corporate actions — REVIEW

> **Pass:** Review (session 2 of 36) · **Date:** 2026-07-28 → 30 · **Status:** ☑ **COMPLETE**
>
> Process contract: [`../00-PROCESS.md`](../00-PROCESS.md) · Spine: [`../01-MASTER-REVIEW.md`](../01-MASTER-REVIEW.md)
>
> **This document was rebuilt from first principles on 2026-07-30.** The prior version derived its
> coverage from a CBOE vendor snapshot — the instrument used to answer one question (*is CBOE
> adjusted?*) — and recorded whatever that instrument happened to reveal. The cost was measurable:
> of 20 findings, four were relocated out and three split, and what survived was largely *absence*
> ("no rows", "no producer", "no source") while the sections that should be thickest — what must be
> collected, how it is checked, and who consumes it — were the thinnest. This version works forward
> from **what a corporate event is for the instruments we actually hold**, through collection →
> validation → storage → consumption. See [§B3.11](#b311--how-this-document-was-rebuilt) for the
> rebuild's own audit trail.
>
> **Every ID is preserved.** Findings that survived the rebuild keep their numbers; new ones start
> at **B3-F21**, new carry items at **B3-C8**. **B3-F10, B3-F14, B3-F15 and B3-F16 stay retired**
> and are never reused.
>
> **This dataset has no governing ruling** equivalent to A1's CBOE decision, so §B3.7 is grouped
> **by severity**, as the contract requires. No decision was invented to sort against.

---

## B3.0 — Disposition of carried-forward assumptions

Every prior claim about this dataset, and what is actually true. Claims 1–8 were discharged in the
first pass; **9–14 are new to the rebuild** and four of them correct a claim this document itself
previously asserted.

| # | Prior claim | Source | Verdict |
|---|---|---|---|
| 1 | `corporate_actions` exists with 7 columns and holds **0 rows** | A1 forward note | **Confirmed.** Re-measured on pg16 |
| 2 | Alpaca returns **zero spin-offs** | `00-PROCESS.md` §Scope | **Confirmed for our universe, reason corrected.** Alpaca *supports* `spin_off` and returns it from ~2023 (6/6 probes). It returns none for the 8 ETFs — but not because our only spin-off predates coverage. It types that event as something else entirely (claim 12) |
| 3 | Four event types are needed | A1 forward note | **Corrected twice.** Splits must not be *applied* to CBOE bars (**B3-F11**); and four types is far short of the universe our instruments can produce (**B3-F21**) |
| 4 | CBOE is **unadjusted**, so raw storage requires corporate actions | A1, 2026-07-25 ruling | **Refuted.** CBOE is **split-adjusted in price and volume**, dividend- and spin-off-raw (**B3-F11**) |
| 5 | Crypto has no corporate actions — a justified DS-7 divergence | A1 forward note | **Confirmed, now measured twice** — against our own range, and independently against Yahoo (**B3-F32**) |
| 6 | Alpaca `Adjustment.ALL` adjusts price for splits and dividends, volume for splits only | A1 | **Confirmed**, and its consequence measured (**B3-F12**) |
| 7 | `ohlcv_4h.source` is 100 % NULL | **A1-F13** | **Confirmed.** Belongs to A1; cited, not re-raised |
| 8 | The XLF 2016-09-19 spin-off carries −18.25 % | **A1-F15** | **Confirmed** at −18.2 % from the CBOE snapshot |
| 9 | *"yfinance is unreachable — the outage is total; the front-runner source is untestable"* | **B3-F5**, this document, 2026-07-29 | **Mis-framed — corrected in place.** Only `fc.yahoo.com` is DNS-blocked. `query1`/`query2.finance.yahoo.com` resolve normally and return HTTP 200 from the host **and from inside the collector container**. The endpoint serves the contested window (**B3-F22**) |
| 10 | *"Per-symbol-per-year counts confirm 2017–2025 is unbroken at 4/yr"* | **B3-F4**, this document | **Wrong — corrected in place.** SPY 2018 returns **three** dividends; 2018-06-15 is absent. Four other symbol-years also deviate. The 2016 floor is likewise **per-symbol** (VTI 2016-03-15, QQQ/XLF 2016-03-18, the other five 2016-06-17), not a single global floor (**B3-F23**) |
| 11 | *"Alpaca returns 13 typed event keys"* | **B3-F6 (h)**, this document | **Corrected in place — the vocabulary is 16**, enumerated from the API's own rejection message (**B3-F21**) |
| 12 | *"Alpaca does not type the XLF 2016 spin-off; manual entry is demonstrated necessary"* | **B3-F17** / **B3-C3**, this document | **Both corrected.** The $0.139146 distribution **is** Alpaca's representation of the event — Yahoo returns the *same date* as a `1231:1000` ratio, i.e. a **−18.8 %** reduction matching the measured −18.2 %. So the event is typed usably by a source, and the "demonstrated necessary" framing was an escalation of what the user actually said (claim 13) |
| 13 | *"USER RULING: manual entry is demonstrated necessary, not merely provisioned"* | **B3-C3**, this document, 2026-07-29 | **Not a ruling — corrected 2026-07-30 at the user's instruction.** What was actually stated: obtain a **free, usable source for historical events back to 2004**; **Alpaca serves ongoing**; manual entry is a **last-resort, one-time** path for whatever gaps remain after that source. It was never a decision to hand-enter 400+ events. **B3-C3 is restated accordingly** |
| 14 | *"Alpaca is reliable for ongoing events"* | User, 2026-07-30 | **Not confirmed — flagged, not contested.** Alpaca's corporate-actions API is currently missing **SPY 2026-06-18 ($1.904)**, a live current-year event that its *own bar adjustment* applies (**B3-F23**, **B3-F24**). Recorded so the assumption is chosen with evidence rather than inherited. The source decision remains the user's, at spec time |

**New this session, in no prior document:** `src/swingrl/data/corporate_actions.py` exists as a
176-line `CorporateActionDetector` with **zero production callers** — the same "capability wired to
nothing" pattern as A1-F22.

---

## B3.1 — The data itself

For a dataset that holds nothing, "the data itself" has two halves: **what must exist**, derived
from the instruments we hold, and **the precise shape of the nothing that does**.

### (a) The event universe — derived from our instruments, not from a vendor's catalogue

**We hold ten instruments. None of them is an operating company.**

| Environment | Instruments | Legal form |
|---|---|---|
| Equity | SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK (`config/swingrl.yaml:9`) | **Exchange-traded funds** — 3 sponsors: State Street (SPY + 5 sector SPDRs), Invesco (QQQ), Vanguard (VTI) |
| Crypto | BTCUSDT, ETHUSDT (`config/swingrl.yaml:22`) | **Trading pairs** — no issuer, no corporate form |

This matters because **the vendor's event vocabulary is an operating-company vocabulary.** Alpaca's
accepted `types` values, enumerated by submitting an invalid type and reading the API's own
rejection (`400`, 2026-07-30):

```
forward_split      reverse_split       stock_dividend      spin_off
cash_merger        stock_merger        stock_and_cash_merger
unit_split         cash_dividend       redemption          name_change
worthless_removal  rights_distribution contract_adjustment partial_call
reorganization
```

**Sixteen keys** — correcting this document's earlier "13". Of the sixteen, several are structurally
impossible for a fund we hold (`worthless_removal`, `partial_call`, `cash_merger` of the fund
itself), while the classes that genuinely apply to **funds** have no key at all:

| Fund event class | Can it happen to our holdings? | Typed by Alpaca? | Consequence if uncollected |
|---|---|---|---|
| Regular income distribution | Yes — quarterly, all 8 | ✅ `cash_dividend` | Price history drifts down; measured at 0.71–3.78 %/yr (**B3-F18**) |
| **Capital-gains distribution** | Yes — year-end, irregular | ❌ folded into `cash_dividend` | Indistinguishable from income; adjusts identically, but cannot be audited or explained |
| **Return of capital** | Yes — sector funds | ❌ no key | Same |
| **Index/sector reconstitution distribution** | **Yes — happened.** XLF → XLRE, 2016-09-19 | ❌ typed as a $0.139 `cash_dividend` | **−18.2 % mis-stated as −0.3 %** |
| Share split | Yes — XLE, XLK 2025-12-05 | ✅ `forward_split` | Under CBOE, not an adjustment input (**B3-F11**) |
| Fund merger / liquidation / closure | Possible; none in range | ⚠️ operating-company keys only | Series would terminate with no marker |
| Ticker or name change | Possible; none in range | ✅ `name_change` | Series would break at the rename with no link |
| Sponsor / index change | Yes, silently | ❌ no key anywhere | Not a price event; a *meaning* event, invisible to any schema |

**Neither the 7-column table nor any vendor vocabulary was derived from this list.** The schema was
written for "an action", the vendor typing was written for equities, and the one fund-specific
event our 22 years of history actually contains is the one both get wrong.

→ **B3-F21**

### (b) Per event type: what is needed to apply, to audit, and to replay

Three distinct field sets, routinely conflated. Derived from the arithmetic, then checked against
what the vendors supply.

| Purpose | Question it answers | Fields required |
|---|---|---|
| **Apply** | "What number do I multiply the bar by?" | event type · **ex-date** · the magnitude in its stated convention · the share-basis that convention is expressed in · for spin-offs, the child instrument and its rate |
| **Audit** | "Is this record right, and where did it come from?" | source · fetch timestamp · vendor record id · the raw payload · instrument identity that survives a rename (CUSIP) · event-nature flags |
| **Replay** | "What did we believe on date D, and what did we change?" | **knowledge date** (when the record became known, distinct from ex-date) · revision number or supersession link · what *we* altered and why |

Two consequences fall straight out of this table and are the subject of §B3.4 and §B3.6:

- **"Apply" needs a field no vendor supplies** — the share-basis convention of the magnitude. Both
  candidate sources omit it and **they use opposite conventions** (**B3-F25**).
- **"Replay" needs a knowledge date.** A1-C17 rules that read-time adjustment filters actions to
  `ex_date ≤ bar date`. That reproduces *today's* view of history, not what was knowable at the
  time, because a revision carries a past ex-date and a present knowledge date (**B3-F31**).

**B3-F6 enumerated 10 schema gaps (a)–(j) against live vendor payloads.** That enumeration is the
seed for this section, not its conclusion: it measures the table against *what Alpaca returns*,
whereas the table above measures it against *what the arithmetic needs*. Gaps (f) CUSIP, (g)
event-nature flags and (h) vocabulary are the three that appear in both.

### (c) The shape of the nothing

**`corporate_actions` has held zero rows since creation.**

```
docker exec pg16 psql -U swingrl -d swingrl -c "SELECT count(*) FROM corporate_actions;"
 count
-------
     0
```

Schema, `src/swingrl/data/postgres_schema.py:444-453`:

| Column | Type | Note |
|---|---|---|
| `action_id` | TEXT | PK — populated with a locally generated `uuid4` |
| `symbol` | TEXT NOT NULL | |
| `action_type` | TEXT NOT NULL | free text, no enum, no CHECK |
| `effective_date` | DATE NOT NULL | **not** an ex-date; which of four dates it means is unstated |
| `ratio` | DOUBLE PRECISION | direction convention undefined |
| `amount` | DOUBLE PRECISION | share-basis convention undefined |
| `processed` | INTEGER DEFAULT 0 | written, never read |

The nothing is precise, and it is fivefold:

| Dimension | State | Evidence |
|---|---|---|
| Rows | **0** | query above |
| Producer | **none** — the only INSERT has no caller | `corporate_actions.py:107`; no reference in `src/` or `scripts/` |
| Source | **none** — no module fetches, no config field names a provider | `config/swingrl.yaml`, full-tree grep |
| Schedule | **none** — no job in the collector | `scripts/collector_main.py`, all jobs enumerated |
| Consumer | **none** — one dead read path | `is_known_action()` → `check_and_suppress()`, never imported by `validation.py` |

→ **B3-F1**, **B3-F2**, **B3-F3**, **B3-F6**

### (d) The 10 representational gaps, with the candidate schema measured against them

**B3-F6 in full.** The DDL was read; the vendor payloads were fetched live; the candidate's field
lists were read from the research file. The *closes?* column is reasoned from those field lists,
not tested.

What the vendor actually supplies (live, 2026-07-30):

```
cash_dividend   symbol cusip ex_date record_date payable_date process_date rate special foreign id
forward_split   symbol cusip ex_date record_date payable_date process_date new_rate old_rate
                due_bill_redemption_date id
spin_off        source_symbol source_cusip new_symbol new_cusip source_rate new_rate
                ex_date process_date id
```

| Gap | What the vendor supplies (measured) | Why the 7-column table cannot hold it | Closed by **B3-C4**'s candidate? |
|---|---|---|---|
| **(a)** No ex-date | **Four distinct dates** — `ex_date`, `record_date`, `payable_date`, `process_date`. The XLE split is ex **2025-12-05**, record **12-02**, payable **12-04** — *payable before ex* | One `effective_date`, and nothing states which of the four it is. Adjustment keys on **ex_date** | ✅ carries `ex_date` **and** `effective_date`; record/payable survive only in `payload_json` |
| **(b)** No child symbol | Spin-off carries `new_symbol` + `new_cusip` | One `symbol` column. A spin-off is a relationship between two instruments | ✅ `child_symbol` |
| **(c)** Ratio direction undefined | Splits carry a **pair** (`new_rate: 2`, `old_rate: 1`); spin-offs carry `source_rate: 1`, `new_rate: 0.333333333` | One `ratio` with no CHECK and no convention recorded. `2.0` cannot distinguish 2:1 forward from 1:2 reverse | ✅ `ratio` + `normalized_factor` + `normalization_method` |
| **(d)** No vendor event identity | Every payload carries a vendor `id` (UUID) | PK is a **locally generated** `uuid4`, so the same event ingested twice yields two unlinked rows | ✅ `source_record_id` |
| **(e)** No uniqueness constraint | — | Only the PK exists. **And the obvious natural key is wrong:** QQQ 2022-09-19 legitimately returns **two** rows (**B3-F27**), so a constraint on `(symbol, action_type, effective_date)` would reject real vendor data | ❌ the candidate declares no constraint either — and now must solve a harder problem than "add one" |
| **(f)** No CUSIP | `cusip` on dividends and splits; `source_cusip`/`new_cusip` on spin-offs | Symbol alone does not identify an instrument across a rename | ❌ only inside `payload_json`, so not queryable |
| **(g)** No event-nature flags | `special: bool`, `foreign: bool` on every dividend | No column, and no way to express it but by inventing an `action_type` string. **Note (B3-F29): the flag is unreliable even when carried** | ❌ not in the candidate's field list |
| **(h)** `action_type` is free text | **16 typed keys** *(corrected from 13)* | `TEXT NOT NULL`, no enum, no vocabulary defined anywhere in the codebase | ❌ `event_type` is equally untyped; and the right vocabulary is **not** the vendor's (**B3-F21**) |
| **(i)** No provenance | — | No source, no fetched-at, no entered-by — the same argument **A1-F26** makes for candles | ✅ `source`, `source_ts`, `source_record_id`, `confidence_score`, `status` |
| **(j)** `processed` is write-only and type-ambiguous | — | Written at insert, **read nowhere**. Declared `INTEGER`, but its own test hedges — `processed is False or == 0` (`tests/data/test_corporate_actions.py:194`) | ✅ dropped in favour of `status` + `factor_version` |

**The candidate closes 6 of 10** — (a), (b), (c), (d), (i), (j) — leaving **(e)** uniqueness,
**(f)** CUSIP, **(g)** event-nature flags, **(h)** vocabulary. The rebuild did not change that
count, but it **hardened (e)**: what was "no constraint exists" is now "the natural key is
demonstrably not unique in real vendor data."

→ **B3-F6**

### (e) CBOE's adjustment state — the finding that redefines the dataset's purpose

| Symbol | Split | Median CBOE ÷ Alpaca `RAW`, pre-2025-12-05 (n=2,496) | Post (n=156) |
|---|---|---|---|
| XLE | 2:1 on 2025-12-05 | **0.499941** | 1.000000 |
| XLK | 2:1 on 2025-12-05 | **0.500000** | 1.000000 |
| SPY | none | 1.000000 | 1.000000 |
| IBM | none | 1.000000 | 1.000000 |
| XLF | none | 1.000000 | 1.000000 |

Volume is split-adjusted too (≈1.98× Alpaca raw pre-split). Dividends are **not** adjusted — CBOE
equals Alpaca `RAW` at median exactly 1.000000 across 2,652 bars on all three controls. Spin-offs
are **not** adjusted — XLF still carries −18.2 % on 2016-09-19.

**Stated plainly:** CBOE restates history for splits and leaves dividends alone. A1's "CBOE is
unadjusted" was checked only against dividends, which is why it passed for the wrong reason.

**Pre-2016 remains an inference, now with one candidate test.** Yahoo reports a **VTI 2:1 split on
2008-06-18** that Alpaca does not carry. CBOE's VTI closes run continuously across that date
(2008-06-17 = 68.00, 06-18 = 67.50, 06-19 = 67.70) — no discontinuity, which is what a
split-adjusted series looks like. **This is corroboration, not proof:** continuity is equally
consistent with the split never having occurred, and the split is single-sourced from Yahoo and
contradicted by Alpaca (**B3-F23**). Recorded as a conditional upgrade.

→ **B3-F11**

---

## B3.2 — Historical one-time ingestion, and its checks and balances

**There has never been one.**

- No backfill script exists for corporate actions anywhere in `src/` or `scripts/`.
- The table appears in `scripts/migrate_to_postgres.py:58` only as a name in a migration list.
- `tests/data/test_db.py:216` references it only in an existence check.
- Legacy DuckDB provenance could not be checked — `~/swingrl/db/swingrl.db` is no longer a valid
  DuckDB file. **Recorded as an unclosed gap**, not asserted either way.

Nothing validated it at the time because nothing ever ran. → **B3-F1**, **B3-F2**

**The history that would have to be built is now measured, not estimated.** Across the 8 ETFs,
before Alpaca's earliest event (2016-03-15), Yahoo returns **374 cash dividends**, earliest
2004-03-19 — against the previous document's extrapolated "~380". Per symbol: SPY 49, QQQ 46,
VTI 48, XLV 48, XLI 48, XLE 48, XLF 48, XLK 39. → **B3-F22**

---

## B3.3 — Ongoing ingestion, and its checks and balances

### Owner and schedule

**None.** Every scheduled job in `scripts/collector_main.py` was enumerated — snapshots, health
check, data audit, offsite backup, calendar ingest, calendar staleness, equity candles, crypto
candles. **No corporate-actions job exists.**

**USER DIRECTION 2026-07-29:** the collector will own collection *and* validation for this dataset,
mirroring A1-C7 for candles. → **B3-C2**

### Source coverage, per event type and per time band

**USER REQUIREMENT, restated verbatim 2026-07-30** — this replaces the "manual entry is
demonstrated necessary" framing that this document previously carried:

| Band | What the user requires |
|---|---|
| **Historical, 2004 →** | A **free, usable** source for historical corporate events |
| **Ongoing** | **Alpaca**, treated as reliable *(see **B3-F23**/**B3-F24** — flagged, not contested)* |
| **Residual gaps only** | Manual entry, **one-time, last resort**. Never a plan for 400+ events |

**Alpaca**, measured live for the 8 ETFs, 2004-01-01 → 2026-07-28 — **334 cash dividends, 2 forward
splits, 0 spin-offs, 0 reverse splits, 0 distributions**:

| Window | Dividends | Splits | Spin-offs |
|---|---|---|---|
| **2023 → ongoing** | ✅ | ✅ | ✅ — 6/6 probes (DHR→VLTO, K→KLG, GE→GEHC, MMM→SOLV, GE→GEV, WDC→SNDK) |
| **2016 – 2023** | ⚠️ near-complete, with holes | ✅ | ❌ — 0/2 probes (IBM→Kyndryl, T→WBD) over multi-year windows |
| **2004 – 2016** | ❌ | ❌ | ❌ |

**Correction to this document's earlier claim.** It stated "per-symbol-per-year counts confirm
2017–2025 is unbroken at 4/yr". That is wrong:

| Deviation | Detail |
|---|---|
| **SPY 2018 — three dividends** | 2018-06-15 is **absent**. Independently confirmed present by Yahoo at **$1.246** |
| SPY 2026 | **2026-06-18 absent** ($1.904 per Yahoo) — and applied by Alpaca's *own bars* (**B3-F24**) |
| QQQ 2022, QQQ 2023, XLE 2019, XLV 2019 | five events in the year — genuine extra distributions, not defects |
| The 2016 floor is **per-symbol** | VTI 2016-03-15 · QQQ, XLF 2016-03-18 · SPY, XLE, XLI, XLK, XLV 2016-06-17 |

→ **B3-F4** *(corrected)*, **B3-F23**

**Yahoo** — and the correction to **B3-F5**. The previous conclusion was "the outage is total; the
front-runner source is untestable". That is mis-framed:

| Host | Local resolution | Public (`@1.1.1.1`) | Reachable? |
|---|---|---|---|
| `fc.yahoo.com` | **127.0.0.1** | 69.147.92.11 | ❌ blocked |
| `query1.finance.yahoo.com` | 69.147.92.11 | 69.147.92.12 | ✅ **HTTP 200** |
| `query2.finance.yahoo.com` | — | — | ✅ **HTTP 200** |

`fc.yahoo.com` is the **cookie-bootstrap host the yfinance *library* calls first**. Blocking it kills
the library; it does not block Yahoo's data endpoint. Verified **inside the collector container** —
where any such job would run — that `query1.finance.yahoo.com` resolves to the real Yahoo edge while
`fc.yahoo.com` still maps to 127.0.0.1.

What that endpoint returns for the 8 ETFs, 2004 → today: **713 dividends** (374 of them before
Alpaca's floor) and **4 splits** — including **XLF 2016-09-19 as a `1231:1000` ratio**, i.e. a
−18.8 % reduction that matches the measured −18.2 % drop.

*Per the user's instruction, this is recorded as measurement only. No recommendation to adopt Yahoo
is made here; source choice stays open in **B3-C3** for the spec.*

→ **B3-F5** *(corrected)*, **B3-F22**

**Untested candidates remain untested**, with reachability now probed: SEC EDGAR (HTTP 200;
reservation recorded — its assumed forms are operating-company forms and ETF distributions may not
appear), `api.nasdaq.com` (DNS resolves, **connection fails**, cause unidentified), the three fund
sponsors (SSGA returns 301 on the probed path), CBOE DataShop (paid, unprobed).

### The checks and balances

Four distinct questions. **All four are currently unanswered**, and one of them turns out to be
much harder than it looks.

#### (i) What validates an event record?

**Nothing.** No validator, no constraint beyond the PK, no cross-source comparison. What such a
check would face, measured on the 331 events where Alpaca and Yahoo overlap:

| Outcome | Count | Meaning |
|---|---|---|
| Agree on amount (≤0.5 %) | **251** | The check would pass |
| **Differ on amount** | **80** | **Convention, not error** — see below |
| Alpaca-only | 2 | SPY 2018-06-15, SPY 2026-06-18 |
| Yahoo-only | 2 | QQQ 2020-09-21 · plus XLF 2016-09-19 typed as a split rather than a dividend |

**The 80 mismatches are a convention divergence, and it is systematic:**

| Symbol | n | Yahoo ÷ Alpaca | Cause |
|---|---|---|---|
| XLE | 39 | 0.499506 – 0.500587 | Yahoo restates pre-split dividends onto the **post-split** share basis; Alpaca reports **as paid** |
| XLK | 38 | 0.499149 – 0.501094 | same, XLK 2:1 2025-12-05 |
| XLF | 3 | 0.660474, 0.809610, 0.811696 | 0.8117/0.8096 ≈ **1/1.231** — the XLRE spin-off factor. **0.6605 on 2016-03-18 is unexplained** |

A naive cross-source check would report **80 false discrepancies** and miss the **4 real ones**.
→ **B3-F25**, **B3-F26**

#### (ii) What detects a *missed* event? — the hard one

An absent event has no row to fail a check, so the only signal is the price series itself. **That
signal is not strong enough.** Measured over 2017+ on CBOE bars (split-adjusted, dividend-raw, so
ex-date drops are present and undisturbed):

| Symbol | Known ex-dates | Median drop | Daily return σ | Signal ÷ noise |
|---|---|---|---|---|
| SPY | 36 | 40.3 bp | 115.3 bp | 0.349 |
| QQQ | 39 | 16.1 bp | 143.9 bp | **0.112** |
| VTI | 38 | 39.9 bp | 117.3 bp | 0.340 |
| XLV | 39 | 40.5 bp | 106.1 bp | 0.382 |
| XLI | 38 | 40.9 bp | 128.2 bp | 0.319 |
| XLE | 39 | 183.9 bp | 297.0 bp | 0.619 |
| XLF | 38 | 44.8 bp | 141.2 bp | 0.317 |
| XLK | 38 | 44.0 bp | 277.3 bp | 0.159 |

**The event is always smaller than a normal day's noise.** Tuning a detector to catch half the known
events yields:

| Detector | True positives | Missed | **False positives** | Precision |
|---|---|---|---|---|
| Raw return magnitude | 159 | 146 | **6,402** | **2.4 %** |
| Return minus SPY (residual) | 140 | 129 | **4,838** | **2.8 %** |

And the obvious refinement — compare each ETF against its peers — is **structurally defeated**:
all eight share one quarterly ex-date calendar, so the peer median drops on the same day and
cancels the signal. On SPY's three known 2018 ex-dates the peer residual reads −1.5, −28.5 and
−12.7 bp against actual dividends of ~40 bp.

**Conclusion, measured:** completeness cannot be established from price. It requires a second
independent source. → **B3-F26**, **B3-C9**

This also settles **B3-F7**'s calibration argument at full scale rather than on one example:

| Threshold | Purpose | Catches, of 336 real events |
|---|---|---|
| `validation.py:31` — 50 % | quarantine as bad data | **0** |
| `corporate_actions.py:21` — 30 % equity | flag possible unrecorded action | **0** |
| `corporate_actions.py:22` — 40 % crypto | same, crypto | **0** — and guards an environment with no corporate actions |

The **largest single dividend in our entire universe** is XLE 2019-12-30 at **6.001 %** of price.
No threshold in the codebase is within a factor of five of it.

#### (iii) What detects a vendor *revision* of an event already stored?

**Nothing — and revisions are real.** Measured instance, Alpaca, QQQ:

```
ex_date 2022-09-19  rate 0.51856  record 2022-09-20  cusip 46090E103
  id d0409c14-…  payable 2022-09-23   process 2022-09-23
  id 1966d92d-…  payable 2022-10-31   process 2022-10-31
```

Identical event, identical amount, **two vendor ids and two payable dates**. This is a restatement
shipped as an *additional row*, not an update. Both failure modes are live:

- Ingest keyed on vendor `id` → **two dividends**, and the adjustment is applied twice.
- Ingest keyed on `(symbol, ex_date)` → the restatement is **silently discarded**, and the stored
  copy never learns it was corrected.

Nothing re-fetches a trailing window, nothing hashes payloads, nothing compares a stored row to a
fresh one. → **B3-F27**, **B3-C10**

#### (iv) What happens to rejects, who is alerted, where is a human verdict recorded?

| Question | Answer |
|---|---|
| Rejects | `data_quarantine` holds `raw_data_json`, `failure_reason`, `severity` — a bin for what was **thrown away at ingest**, with no action-taken field, no resolution state, and an identity key that cannot be joined back |
| Alerting | **None.** The detector's only output is `log.warning("overnight_spike_detected")` at `corporate_actions.py:78`. It never imports `Alerter`, so no `alert_log` row and no Discord message. `alert_log` is written by exactly one module — `monitoring/alerter.py:379` — and this isn't it |
| Human verdict | **Nowhere.** No review state, no adjudication table |
| Audit | The monthly `data_audit_job` is **options-only** (`data/options/audit.py:105`). A job named for the whole system covers **1 dataset of 12** — B1, B2 and B4 will each meet this gap unannounced |

→ **B3-F7**, **B3-F20**

### A further defect in the vendor record itself

Two field-level defects that any schema requiring these fields must survive:

| Defect | Measurement |
|---|---|
| **Date completeness is banded** | `record_date` and `payable_date` are **NULL in 120 of 334** Alpaca dividends — 26/28 (2016), 32/32 (2017), 31/31 (2018), 31/34 (2019), **0 from 2020 onward**. `ex_date` and `process_date` are always present |
| **`special` does not identify irregular distributions** | Set on exactly **1 of 334** (QQQ 2023-12-27, $0.21584) — while XLE 2019-12-30, a **fifth** distribution in its year at **$1.791209** (3× the largest regular XLE dividend, and the largest event in our universe at 6.001 % of price), carries `special=False`. `foreign` is False on all 334 |

→ **B3-F28**, **B3-F29**

### The vendor is not one thing

**Alpaca's two endpoints disagree with each other, in both directions.** Recovered by comparing its
`dividend`-adjusted bars against its `raw` bars and reading off every re-basing step, then matching
those against its corporate-actions API (2016 → today, 8 symbols):

| Direction | Instance | Meaning |
|---|---|---|
| Bars know, API doesn't | **SPY 2026-06-18**, step 25.7 bp ⇒ implied **$1.9197** (Yahoo: $1.904) | The event API is missing a **current-year** event its own price data applies |
| API knows, bars don't | **QQQ 2023-12-27**, $0.21584, the only `special=True` in the set | The **one** flagged special dividend is the one the bar adjustment ignores |

Every other symbol matched exactly (implied steps = API events, 0 discrepancies). So the
disagreement is narrow and specific — which makes it worse, not better: it cannot be dismissed as
noise. → **B3-F24**

### The cost of the gap, quantified

Per missing dividend, as a fraction of unadjusted price (334 events, 2016+):

| Symbol | Median distortion | Max | Annual drag |
|---|---|---|---|
| XLE | **88 bp** | 300 bp | **3.78 %/yr** |
| XLK | 23 bp | 53 bp | 1.00 %/yr |
| QQQ | 18 bp | 30 bp | 0.71 %/yr |
| SPY / VTI / XLV / XLI / XLF | 41–46 bp | 61–90 bp | 1.45–1.86 %/yr |

Over the 12 unsourced years a dividend-blind deep history would drift roughly **8–19 % in level**
for most symbols and **~45 % for XLE**, across **374 measured ex-dates** *(was: ~380 estimated)*.
Per-event figures are measured; the 12-year level drift is an estimate. → **B3-F18**

### The stored equity history is already stranded

`ohlcv_daily` was backfilled 2026-03-06 (SPY, QQQ) / 2026-03-11 (rest) and appended daily.
Re-fetching the same dates from the same vendor today returns different values, because
`Adjustment.ALL` re-bases the whole history at every ex-date:

| Symbol | Mean drift vs fresh fetch | Bars affected |
|---|---|---|
| XLE | **−1.34 %** | 2,556 / 2,556 |
| XLF | −0.90 % | 2,556 |
| SPY | −0.51 % | 2,556 |
| XLK | −0.25 % | 2,556 |

The drift tracks each ETF's dividend yield, which identifies the cause. `base.py:191` inserts with
`ON CONFLICT DO NOTHING`, so re-ingesting cannot correct it. **And the vendor's re-basing inherits
the vendor's holes** — SPY's stored history is adjusted without its 2018-06-15 dividend, so every
SPY bar before that date is wrong by ~40 bp for a reason no re-fetch will fix.

*(The unexplained 2024-12-23 return cluster was split off to **A1-F35** on 2026-07-29 — it has no
corporate-action explanation.)* → **B3-F12**

### No record of our own interventions

Only two candidate tables exist — `data_quarantine` and `llm_audit_log`, neither with an
action-taken field or resolution state. Because CBOE re-bases silently on a split (**B3-F11**) and
`DO NOTHING` forbids in-place correction, recovery is a **full replace** — which destroys every
manual fix applied since the last load: the 20 phantom bars, the VTI 2006 segment, XLI's missing
sessions, and the crypto 22-day hole repair of **A1-C1**. → **B3-F13**

---

## B3.4 — The calculation

**There is no adjustment calculation anywhere in the codebase.** Adjustment is performed
**server-side by Alpaca**, selected at `alpaca.py:126` with `adjustment=Adjustment.ALL`. Nothing in
`src/` computes a factor, a cumulative factor, or an adjusted price.

### What the calculation would have to be — and the trap in it

A cash-dividend adjustment factor is:

```
factor = 1 − rate / close(day before ex-date)
```

**Two properties of that formula are load-bearing and neither is captured by any event schema
reviewed here.**

**(1) The factor is not a property of the event.** It requires a *candle*. So the adjusted series
depends on the raw series, and any candle defect propagates into every factor at or before it —
the phantom closed-session bars (**A1-F10**), the VTI 2006 corrupt window (**A1-F34**), the
2024-12-23 basis island (**A1-F35**). A corporate-action ledger cannot be validated in isolation
from the candles it will be applied to.

**(2) The rate and the close must be on the same share basis — and on the planned architecture
they are not.** Measured, XLE 2025-06-23:

| Input | Value | Basis |
|---|---|---|
| Alpaca `cash_dividend.rate` | **$0.718258** | **as paid** — pre-split |
| Alpaca `raw` close, 2025-06-20 | **$88.98** | as traded — pre-split |
| **CBOE** close, 2025-06-20 | **$44.49** | **split-adjusted** — post-split |

```
1 − 0.718258 / 88.98  =  0.991928   →   −80.7 bp    ✅ correct
1 − 0.718258 / 44.49  =  0.983856   →  −161.4 bp    ❌ exactly 2.000× wrong
```

The planned architecture is **CBOE candles plus a separate event source**. Combining an as-paid rate
with a split-adjusted close therefore **doubles every pre-split dividend factor by default** — not
as a corner case, but on the main path. Yahoo's rates are on the *opposite* convention
(**B3-F25**), so pairing Yahoo rates with CBOE closes happens to be correct for exactly the same
reason that pairing Alpaca rates with CBOE closes is wrong. Neither source states its convention.

→ **B3-F30**, **B3-C8**

### The heuristic detector that does exist

`src/swingrl/data/corporate_actions.py`, 176 lines, four methods:

| Method | Line | What it does |
|---|---|---|
| `detect_overnight_spike` | `:36` | `abs((curr−prev)/prev)` vs threshold; True = **alarm** |
| `record_action` | `:88` | INSERT with a fresh `uuid4` |
| `is_known_action` | `:123` | `SELECT 1 … WHERE symbol=%s AND effective_date=%s` — exact date equality, so it cannot distinguish the two QQQ 2022-09-19 rows (**B3-F27**) |
| `check_and_suppress` | `:140` | same threshold logic re-derived; True = **all clear** |

`check_and_suppress()` exists solely to stop `validation.py` Step 6 quarantining a genuine action.
**`validation.py` never imports it.** And per §B3.3(ii), wiring them together would change nothing:
**0 of 336** real events reach any threshold.

The crypto branch is wrong by construction — it guards an environment with no corporate actions, so
every `True` it returns is a false positive, and a genuine 45 % BTC drawdown would be reported as a
suspected corporate action. → **B3-F7**, **B3-F8**

---

## B3.5 — The pipeline and its wiring

```
   ┌─ SOURCE ─────────────┐
   │  (nothing)           │   no module fetches corporate actions          B3-F3
   └──────────┬───────────┘
              ✗ no producer
   ┌──────────▼───────────┐
   │  corporate_actions   │   0 rows                                        B3-F1
   │  (7 columns)         │   10 representational gaps, (a)–(j)             B3-F6
   └──────────┬───────────┘
              │ read by exactly one function
   ┌──────────▼───────────┐
   │ is_known_action()    │──► check_and_suppress() ──✗── validation.py     B3-F7
   │ CorporateActionDetector│  (never imported; 30/40 % vs 50 %; 0/336)
   └──────────────────────┘
              ✗ no Alerter, no alert_log, no review state                    B3-F7
              ✗ no completeness check — infeasible from price                B3-F26
              ✗ no revision check — restatements measured, undetected        B3-F27

   ADJUSTMENT lives outside the system entirely:
      Alpaca server-side  ──►  Adjustment.ALL  ──►  ohlcv_daily             B3-F12
      CBOE server-side    ──►  splits only     ──►  (not yet ingested)      B3-F11
      basis mismatch between the two ──► 2.000× factor error                B3-F30
```

**Every reference to the table in the entire codebase:** `corporate_actions.py` (its own module),
`postgres_schema.py:445` (DDL), `migrate_to_postgres.py:58` (a name in a list),
`tests/data/test_db.py:216` (existence check), `tests/data/test_corporate_actions.py` (12 tests).
No production caller exists. → **B3-F2**

**Trainer/trader divergence:** none, because neither path reads this dataset. Both consume
already-adjusted candles from `ohlcv_daily`, so the adjustment basis is inherited from the vendor
and identical on both sides — which is why **B3-F12**'s staleness reaches trainer and trader
equally.

**The test suite gives false assurance.** 12 tests exercise every method of the detector and pass.
But `pytestmark = skipif(not DATABASE_URL)` at `test_corporate_actions.py:25` means the module is
skipped in the fast lane, which runs `env -u DATABASE_URL`. And **no test asserts that anything
calls the detector** — a green suite proves the methods behave, not that the capability exists.
*(Tests were read, not executed.)* → **B3-F19**

---

## B3.6 — Use, current and planned

### Current use

| Consumer | Uses this dataset? |
|---|---|
| Trainer | **No.** Consumes `features_equity` / `features_crypto`, built from already-adjusted candles |
| Trader | **No.** Same path |
| Validator | **No** — the one integration point was never wired (**B3-F7**) |
| Observation space | **Zero slots.** No observation dimension derives from corporate actions |

The dataset's influence today is entirely **indirect and invisible**: every price the agent sees has
been adjusted by a vendor, using event data we neither hold nor can inspect — and which is
measurably incomplete (**B3-F23**).

### Planned use — and what a consumer needs that an event row does not carry

Three A1 carry items are blocked here, which is why B3 was reviewed second (DS-8):

| Blocked item | Why |
|---|---|
| **A1-C13** — price basis (store raw, derive adjusted at read time) | Read-time adjustment is exactly as complete as this table |
| **A1-C17** — point-in-time adjustment | Needs actions to filter against |
| **A1-C3** — the CBOE migration | Blocked transitively via A1-C13 |

**B3-F11 changes what "blocked" means.** Since CBOE already applies splits, the migration does not
need split rows to produce a usable series — it needs **dividends and spin-offs**, and split
*records* for a different reason: to know when the vendor has silently re-based history under a
stored copy.

**Four things a consumer needs that a raw event row does not carry.** Each is a live gap:

| # | What the consumer needs | Why the event row is insufficient | Finding |
|---|---|---|---|
| 1 | A **factor**, not an event | Requires the prior close, so the adjusted series depends on the raw series; candle defects propagate into all earlier factors | **B3-F30** |
| 2 | The **share basis** of the magnitude | Unstated by both sources, and they use opposite conventions — mixing them is a silent **2.000×** error | **B3-F25**, **B3-F30** |
| 3 | A deterministic **order and identity** when a date carries more than one row | QQQ 2022-09-19 carries two rows that are one event; there is no rule distinguishing "two events" from "one restated event" | **B3-F27** |
| 4 | A **knowledge date**, distinct from the ex-date | **A1-C17 rules that read-time adjustment filters to `ex_date ≤ bar date`.** That reproduces *today's* view of history. A revision has a past ex-date and a present knowledge date, so ex-date filtering alone cannot reproduce what was knowable at the time — the exact lookahead A1-C17 exists to prevent, re-entering by a different door. Alpaca's `process_date` is the only candidate knowledge field (present on all 334); Yahoo supplies none | **B3-F31** |

Item 4 is **not a challenge to A1-C17** — the ex-date filter is necessary and correct. It is
incomplete: two axes are needed where the ruling names one. → **B3-C12**

### Crypto — the DS-7 parity position

No corporate actions exist for BTCUSDT/ETHUSDT over our data range (2017-08-17 → today, 19,295 bars
each). No token split, no redenomination, no in-range fork airdrop — Bitcoin Cash forked
2017-08-01, sixteen days before our history begins.

**Now measured against a second, independent source** rather than by our own reasoning alone: Yahoo
returns **BTC-USD 3,270 daily bars, 0 dividends, 0 splits** and **ETH-USD 3,186 bars, 0 dividends,
0 splits**. The divergence is justified and doubly measured. *(Incidental: that endpoint is a
reachable free crypto candidate, which is **A1-C18**'s open question — noted, not pursued.)*

→ **B3-F32**

A distinct class does exist and is unowned: **venue events** — exchange insolvency, delisting,
halts, stablecoin de-pegs, venue flash crashes:

| | Corporate action | Venue event |
|---|---|---|
| What changes | The **instrument** | The **observation** |
| Correct handling | Retroactive **adjustment** | **Flag or exclude** the window |

Our history is the Binance Global archive while ongoing collection is Binance.US, so a
venue-specific event would distort the live segment while being absent from history — and
`ohlcv_4h.source` is 100 % NULL (**A1-F13**), so no bar can be attributed to a venue. Owned by
**B4** under DS-10. → **B3-F8**, **B3-F9**

---

## B3.7 — Findings index

**28 findings — 20 High, 8 Medium, 0 Low.** *(16 carried from the first pass; **B3-F21 … B3-F32**
added by the 2026-07-30 rebuild. **B3-F10, B3-F14, B3-F15, B3-F16** are retired — see below.)*

**Grouped by severity.** B3 has **no governing ruling** equivalent to A1's CBOE decision, so per the
contract no decision was invented to sort against. The **Q** column records which of the six review
questions each finding answers — that is coverage information, carried as a column rather than as
an ordering.

*Q key:* **1** what should be collected · **2** what is collected · **3** how/where/when ·
**4** checks and balances · **5** consumption · **6** parity

### High — 20

| ID | Q | Env | § | Feeds | Finding |
|---|---|---|---|---|---|
| **B3-F1** | 2 | equity | B3.1 | C1, C3 | `corporate_actions` holds **0 rows** — the dataset has never existed |
| **B3-F2** | 2 | equity | B3.2 | C1, C2 | The table has **no producer**; the only INSERT (`corporate_actions.py:107`) has no caller |
| **B3-F3** | 2 | equity | B3.1 | C1, C3 | **No source** is fetched or configured anywhere in the system |
| **B3-F4** | 3 | equity | B3.3 | C3 | Alpaca has three coverage bands; **nothing before 2016**, spin-offs only from ~2023. *(Corrected 2026-07-30: the 2016 floor is **per-symbol**, and 2017–2025 is **not** unbroken — see **B3-F23**)* |
| **B3-F5** | 3 | both | B3.3 | C3 | **`fc.yahoo.com` is DNS-blocked** — 127.0.0.1 from host and both containers, killing the yfinance *library* at its cookie handshake. *(Corrected 2026-07-30: the block is confined to that host; Yahoo's data endpoint is reachable — **B3-F22**. Swallowed-failure half split to **A1-F36** on 2026-07-29)* |
| **B3-F6** | 1 | equity | B3.1 | C1, C3, C4 | The schema **cannot represent the events** — **10 gaps, (a)–(j)** against live payloads. *(Amended 2026-07-30: **(h)** is **16** typed keys, not 13; **(e)** hardened — the natural key is demonstrably **not unique** in real vendor data)* |
| **B3-F7** | 4 | both | B3.4 | C1, C9 | Three thresholds, none wired; the loop is open at **detection, alerting and adjudication**. *(Sharpened 2026-07-30: **0 of 336** real events reach any threshold; the largest is **6.001 %** vs a 30 % floor)* |
| **B3-F11** | 1 | equity | B3.1 | C3, C7, C8 | **CBOE is split-adjusted**, not raw — dividends and spin-offs unadjusted. Redefines the dataset's purpose. *(2026-07-30: pre-2016 gains one corroborating test case, VTI 2008-06-18 — conditional, see §B3.1(e))* |
| **B3-F12** | 4 | equity | B3.3 | C1, C5 | **Vendor dividend re-basing strands a stored copy** — all 2,556 bars/symbol drift 0.25–1.34 %, and `DO NOTHING` forbids correcting it. *(2026-07-30: the re-basing also **inherits the vendor's holes** — SPY's stored history is adjusted without its 2018-06-15 dividend)* |
| **B3-F13** | 4 | both | B3.3 | C5 | **No remediation ledger** — a re-ingest destroys every manual fix with no way to replay it |
| **B3-F18** | 3 | equity | B3.3 | C3 | The gap costs **~40 bp per missing dividend** (XLE 88 bp), 0.7–3.8 %/yr of drag, across **374 measured** pre-2016 ex-dates |
| **B3-F21** | 1 | equity | B3.1 | C11, C3 | **The event universe was never derived from our instruments.** All 8 equity holdings are **funds**; the vendor's **16-key** vocabulary is an operating-company vocabulary. Fund classes that genuinely apply — capital-gains distribution, return of capital, reconstitution distribution, fund merger/liquidation, sponsor change — have **no typed representation anywhere**, and the one fund event in our 22-year history is typed correctly by neither vendor |
| **B3-F22** | 3 | equity | B3.3 | C3 | **The contested 2004–2016 window has a reachable free source.** Yahoo's chart endpoint returns **374 dividends** before Alpaca's floor (earliest 2004-03-19) plus 4 splits, including **XLF 2016-09-19 as `1231:1000` = −18.8 %**, matching the measured −18.2 %. Reachable from host **and the collector container**. *Recorded as measurement only; no adoption recommended* |
| **B3-F23** | 3, 4 | equity | B3.3 | C3, C9 | **Neither candidate source is complete, and the gaps are current.** Alpaca omits **SPY 2018-06-15** ($1.246) and **SPY 2026-06-18** ($1.904), both confirmed by Yahoo; Yahoo omits **QQQ 2020-09-21** ($0.38824), confirmed by Alpaca. Bears directly on "Alpaca is reliable for ongoing" |
| **B3-F24** | 4 | equity | B3.3 | C3, C9 | **Alpaca contradicts itself across endpoints.** Its bars apply **SPY 2026-06-18** (25.7 bp ⇒ $1.9197) which its event API omits; its event API returns **QQQ 2023-12-27** ($0.21584 — the only `special=True`) which its bars do not apply. All other symbols match exactly, so this is specific, not noise |
| **B3-F25** | 4, 5 | equity | B3.3 | C8, C3 | **Dividend amounts carry an unstated share-basis convention, and the two sources use opposite ones.** Yahoo restates onto the current basis (XLE 39, XLK 38 events at ×0.4995–0.5006); Alpaca reports as paid. XLF's pre-spin-off dividends differ by ≈1/1.231. A naive cross-source check reports **80 false discrepancies** and misses the **4 real** ones |
| **B3-F26** | 4 | both | B3.3 | C9 | **A missed event cannot be detected from price.** Median ex-date drop 16–184 bp vs daily σ 106–297 bp. At 50 % recall: **2.4 %** precision / **6,402** false positives (raw), **2.8 %** / **4,838** (SPY-residual). Peer-relative detection is **structurally defeated** — all 8 ETFs share one quarterly ex-date calendar. Completeness requires a second source, not a threshold |
| **B3-F27** | 4 | equity | B3.3 | C10, C9 | **Vendor revisions are real and undetectable today.** QQQ 2022-09-19 returns **twice** from Alpaca — same rate/ex/record/cusip, different `payable_date` and vendor `id`. A restatement shipped as a new row. Keyed on `id` it double-counts the dividend; keyed on `(symbol, ex_date)` it is silently discarded. Nothing re-fetches, hashes or compares |
| **B3-F30** | 5 | equity | B3.4 | C8, C4 | **The adjustment factor is not a property of the event.** It needs the prior close, so the adjusted series depends on the raw series and candle defects propagate backwards. And the bases must match: XLE 2025-06-23 reads **−80.7 bp** against the as-traded close and **−161.4 bp** against CBOE's split-adjusted close — **exactly 2.000× wrong**, on the planned architecture's default path |
| **B3-F31** | 5 | both | B3.6 | C12 | **Point-in-time replay needs a knowledge date, which A1-C17 does not name.** Filtering to `ex_date ≤ bar date` reproduces *today's* view; a revision (**B3-F27**) has a past ex-date and a present knowledge date. `process_date` is the only candidate field (present on all 334 Alpaca rows); Yahoo supplies none. Not a challenge to A1-C17 — it is incomplete, not wrong |

### Medium — 8

| ID | Q | Env | § | Feeds | Finding |
|---|---|---|---|---|---|
| **B3-F8** | 6 | crypto | B3.6 | C6 | No corporate actions in range (justified DS-7 divergence); **venue events** are an unexamined analogue |
| **B3-F9** | 6 | both | B3.6 | C6 | Observation events had **no dataset** in the audit's roster — resolved by DS-10 |
| **B3-F17** | 3 | equity | B3.3 | C3 | Alpaca types the XLF 2016 spin-off as a **$0.139146 cash dividend** — a fifth 2016 distribution where every other symbol-year shows four. *(**Resolved 2026-07-30:** Yahoo returns the same date as a `1231:1000` ratio = −18.8 %, matching the measured −18.2 %. The row **is** Alpaca's representation, and it is unusable — but the event **is** typed usably by a source)* |
| **B3-F19** | 4 | both | B3.5 | C1 | The test suite gives **false assurance** — 12 tests over dead code, skipped in the fast lane, none asserting that anything calls the detector |
| **B3-F20** | 4 | both | B3.3 | C1 | **Nothing audits corporate actions** — and `data_audit_job` covers **1 dataset of 12**, so every remaining dataset will meet the same gap unannounced. *(Candle half split to **A1-F37** on 2026-07-29)* |
| **B3-F28** | 1, 3 | equity | B3.3 | C3, C4 | **Vendor field completeness is banded.** `record_date` and `payable_date` are **NULL in 120 of 334** — 26/28 (2016), 32/32 (2017), 31/31 (2018), 31/34 (2019), **0 from 2020**. A schema requiring four dates has a four-year hole, and any check keyed on them silently no-ops for the early band |
| **B3-F29** | 1, 4 | equity | B3.3 | C3, C11 | **`special` does not identify irregular distributions.** Set on **1 of 334** (QQQ 2023-12-27) while XLE 2019-12-30 — a fifth distribution at **$1.791209**, 3× the largest regular XLE dividend and the biggest event in our universe at **6.001 %** of price — carries `special=False` |
| **B3-F32** | 6 | crypto | B3.6 | C6 | **Crypto's DS-7 divergence is now measured against an independent source** — Yahoo BTC-USD (3,270 bars) and ETH-USD (3,186 bars) return **0 dividends, 0 splits**. Incidentally a reachable free crypto candidate for **A1-C18** |

### Low — 0

None. **B3-F16**, the only Low, was relocated to A1.

### Retired IDs — never reused

**USER RULING 2026-07-29.** Four findings were **not corporate-action facts**; three more were
**split**, the corporate-action half staying here. Listed so any existing reference resolves.

| Retired / split ID | Now lives as | Why it was ever in B3 |
|---|---|---|
| **B3-F14** — phantom bars on closed sessions | **A1-F10 / A1-C14 (a)**; count amended there 15 → 20 | Its only B3-relevant property is that the doubled bars escaped CBOE's split-adjustment pass — evidence for **B3-F11**, kept in §B3.1(e) |
| **B3-F15** — CBOE VTI 2006 corrupt window | **A1-F34** | Half-value looks like a mis-applied split; B3 work **ruled that out**. Once ruled out it is plain vendor corruption |
| **B3-F16** — CBOE session completeness *(positive)* | A1's missing-sessions row | No corporate-action content whatsoever |
| **B3-F10** — XNYS rolling calendar bound | **B4** (DS-10); measurement preserved in the forward note | The trading calendar is B4's dataset. **The blocking edge survives: A1-C3 is blocked by B4** |
| **B3-F12**, in part — the 2024-12-23 return cluster | **A1-F35** | Same `fetched_at`, no event on the date — no corporate-action explanation |
| **B3-F5**, in part — swallowed cross-source failure | **A1-F36** | A candle-validator defect; composes with **A1-F17** |
| **B3-F20**, in part — candles are not audited | **A1-F37** | Recasts **A1-C6** from a build into an extension |

### Grouping tests — recorded so they are not silently retried

| Grouping tried | Outcome |
|---|---|
| **By disposition** ("what does the CBOE migration do to this finding") | **Failed** — 14 of the then-20 findings were unchanged by it, because they describe a dataset that does not exist and therefore cannot be improved or worsened by a change to a different table |
| **By severity, with a coverage column** | **Adopted.** The contract's fallback when no governing ruling exists. The **Q** column carries coverage without inventing an ordering |
| **By the six review questions** *(new, 2026-07-30)* | **Rejected as an ordering, kept as a column.** 6 of 28 findings answer two questions at once (**B3-F23**, **B3-F25**, **B3-F28**, **B3-F29**, plus **B3-F4**/**B3-F7** partially), so an exclusive grouping would have to duplicate rows or arbitrarily pick one. A column expresses it losslessly |

---

## B3.8 — What the disposition means

**This dataset cannot be "fixed" — it has to be created.** Most findings describe absence rather
than defect. That inverts the usual review consequence: there is no legacy behaviour to preserve,
no migration risk to the dataset itself, no rollback concern. The risk lives entirely in what
depends on it.

**The rebuild moved the centre of gravity from *absence* to *fidelity*.** The first pass established
that nothing exists. The rebuild establishes something harder: **the things that would fill it are
themselves defective, and defective in ways that are silent.** Neither candidate source is
complete; one contradicts itself between its own endpoints; they use opposite and unstated
conventions for the same number; a revision arrives as a duplicate; a flag meant to mark irregular
distributions marks the wrong one. Building the dataset is not a matter of pointing a fetcher at an
API.

**The premise of the migration moved, and stayed moved.** A1 and the 2026-07-25 ruling both rest on
"CBOE is unadjusted". **B3-F11** refutes that. The CBOE decision is not wrong — the completeness
result supports it strongly — but the *work* changes shape: splits leave the adjustment problem and
enter the change-detection problem, and dividends become the whole of the reconstruction burden.

**The 2004–2016 window is no longer the crux.** The first pass called it "contested" — the band
where no source had been demonstrated. A source has now been demonstrated to *exist and be
reachable*, covering 374 measured events (**B3-F22**). What replaces it as the crux is **conventions
and completeness**: the deep history is obtainable, but obtaining it correctly requires knowing
which share basis each amount is quoted on (**B3-F25**, **B3-F30**) and having some way to know the
set is complete when price cannot tell you (**B3-F26**).

**Absence detection is the finding with the longest reach.** Because a missing event cannot be seen
in price, *every* downstream guarantee about the adjusted series rests on the completeness of the
event ledger, and completeness can only come from source agreement. This makes **A1-C12** (the
cross-source gate) load-bearing for **A1-C13** in a way neither item previously stated — and it
means a single-source event ledger is unverifiable by construction, whichever source is chosen.

**Three ledgers, not one.** "Corporate actions" was carrying three distinct jobs: what the
instrument did (adjust), what the venue did (flag), and what we did (replay). Conflating them makes
`ratio` mean "multiply history by this" for some rows and nothing for others.

**Two conclusions stay conditional.** **B3-F11** is verified only from 2016 onward — the VTI
2008-06-18 test corroborates but does not prove, since price continuity is equally consistent with
the split never having happened, and the split itself is single-sourced. And **B3-F22**'s source is
an unauthenticated free endpoint with no SLA — the same class of dependency **A1-F30** flags for
CBOE, and it carries the same risk.

---

## B3.9 — Carry register

**No remedies.** Each item states what must be settled, not how.

| ID | Item |
|---|---|
| **B3-C1** | **The dataset must be sourced and owned.** Covers what fills 2004→today, what keeps it current, and what makes a failure visible. Anchored by **B3-F1**, **B3-F2**, **B3-F3**, **B3-F7**, **B3-F19**, **B3-F20** |
| **B3-C2** | **USER DIRECTION 2026-07-29** — the **collector owns** corporate-action collection *and* validation, end to end, mirroring **A1-C7** for candles. Anchored by **B3-F2** |
| **B3-C3** | **Settle the source, per event type and per time band.** **RESTATED 2026-07-30 at the user's correction** — the previous wording ("manual entry is *demonstrated necessary*") escalated what was actually said and is withdrawn. The requirement as stated: obtain a **free, usable source for historical corporate events back to 2004**; **Alpaca serves ongoing**; **manual entry is a last-resort, one-time path for residual gaps only** — never a plan for 400+ events. Measured constraints: Alpaca covers dividends+splits from a **per-symbol** 2016 floor and spin-offs from ~2023, nothing before (**B3-F4**); **a free source covering 2004–2016 has been demonstrated reachable** and returns 374 pre-floor dividends plus the XLF spin-off typed usably (**B3-F22**, **B3-F17**); **neither candidate is complete** — each is missing events the other holds, including a current-year one (**B3-F23**); **Alpaca disagrees with itself across endpoints** (**B3-F24**); and the two sources quote amounts on **opposite, unstated share bases** (**B3-F25**). Any shortlist must be judged on **completeness, self-consistency, convention disclosure and spin-off fidelity**, not merely on whether rows are returned. **Untested candidates:** SEC EDGAR (reachable; reservation — its assumed forms are operating-company forms), the three fund sponsors, `api.nasdaq.com` (resolves, connection fails), CBOE DataShop (paid). Anchored by **B3-F3**, **B3-F4**, **B3-F5**, **B3-F17**, **B3-F18**, **B3-F22**, **B3-F23**, **B3-F24**, **B3-F29** |
| **B3-C4** | **A candidate schema exists and should be evaluated.** The research file proposes `corporate_events_raw` → `corporate_events_norm` → `adjustment_factors_daily` → `adjusted_bars`. Against **B3-F6**'s enumerated gaps it closes **6 of 10** — (a) ex-date, (b) child symbol, (c) ratio convention, (d) vendor event id, (i) provenance, (j) `processed` — and leaves **(e)** uniqueness, **(f)** CUSIP, **(g)** event-nature flags, **(h)** vocabulary. **Not adopted — USER DIRECTION 2026-07-29 records it as the leading candidate**, which the user is inclined toward "unless we uncover something that will cause issues with it". **Three things the rebuild uncovered that bear on that test:** gap **(e)** is harder than "add a constraint" — the natural key is **not unique in real vendor data** (**B3-F27**); the candidate has **no field for the share basis** of an amount, which is a silent 2× error (**B3-F25**, **B3-F30**); and it carries `source_ts` but no **knowledge date** distinct from ex-date (**B3-F31**). Two questions stay open for the spec: whether the adjusted series is **materialised or derived at read time**, and whether these tables fold into **A1-C10**'s metadata consolidation. **A consequence the user stated 2026-07-29:** if raw CBOE bars are stored verbatim and an adjusted table is built from them, the raw→adjusted step is also where non-session bars are dropped (**A1-F10**), making the adjusted build dependent on **B4** as well as on corporate actions. Source: `../research by varun/Raw Candles and Adjustments and Validation.md` — **ideas, not fact**; currently untracked in git |
| **B3-C5** | **A remediation ledger is required** — recording what *we* changed, keyed so corrections replay deterministically onto a freshly re-ingested series. Forced by **B3-F11** (CBOE re-bases silently, so recovery is a full replace) plus `DO NOTHING`. **Settles an argument A1 left open:** in-table audit state does not survive the replacement operation — evidence feeding **A1-C11**. Anchored by **B3-F12**, **B3-F13**, and by **A1-F10** and **A1-F34**, the manual cleanups a full replace would destroy |
| **B3-C6** | **USER RULING 2026-07-29 (DS-10) — observation events become a 12th dataset**, covering the **trading calendar** and **venue events**. Kept together because neither is assessable without the other; crypto's half is 24/7 uptime with maintenance exceptions. Anchored by **B3-F8**, **B3-F9**, **B3-F32**; by **A1-F10**; and by the rolling XNYS bound relocated to B4 |
| **B3-C7** | **The CBOE adjustment convention must be encoded wherever series are compared.** Split-adjusted but dividend-raw matches no standard vendor convention, so any cross-source check assuming otherwise reports false discrepancies — the trap **A1-C12** identified. Anchored by **B3-F11** |
| **B3-C8** | **NEW 2026-07-30 — the share basis of every amount and every price must be explicit, per source and per row.** A dividend rate is meaningless without knowing which share basis it is quoted on, and the two candidate sources use **opposite** conventions while stating neither: Yahoo restates onto the current basis, Alpaca reports as paid (**B3-F25**). Combining an as-paid rate with a split-adjusted CBOE close is **exactly 2.000× wrong** and is the *default* pairing on the planned architecture (**B3-F30**). Covers: which basis is stored, which is canonical, how a re-basing event (a new split) restates already-stored amounts, and how a comparison declares its basis before comparing. Anchored by **B3-F25**, **B3-F30**, **B3-F11** |
| **B3-C9** | **NEW 2026-07-30 — settle what proves the event set complete, given that price cannot.** Absence has no row to fail a check, and the price signal is measurably too weak: at 50 % recall a price detector runs at **2.4 % precision / 6,402 false positives**, and peer-relative detection is structurally defeated by a shared ex-date calendar (**B3-F26**). Both candidate sources have holes, including current-year ones, and each holds events the other lacks (**B3-F23**); one contradicts itself across its own endpoints (**B3-F24**). Covers: what a completeness assertion is made against, at what cadence, what happens when two sources disagree, and what is alerted. **Makes A1-C12 load-bearing for A1-C13** — a single-source ledger is unverifiable by construction. Anchored by **B3-F23**, **B3-F24**, **B3-F26**, **B3-F27**, **B3-F7** |
| **B3-C10** | **NEW 2026-07-30 — event identity and revision semantics.** A vendor restates an event by shipping an **additional row**, not an update: QQQ 2022-09-19 returns twice with identical amount and ex-date but different `payable_date` and vendor `id` (**B3-F27**). Keyed on vendor id this double-counts the dividend; keyed on `(symbol, ex_date)` the correction is silently discarded — and that natural key is therefore **not** available as a uniqueness constraint (**B3-F6 (e)**). Covers: what identifies an event, what distinguishes "two events" from "one restated event", whether superseded rows are retained, and what re-fetch or hashing would surface a revision at all. Anchored by **B3-F27**, **B3-F24**, **B3-F6** |
| **B3-C11** | **NEW 2026-07-30 — the event-type vocabulary must be derived from the instruments we hold, not adopted from a vendor.** All eight equity holdings are **funds**; the vendor's 16-key vocabulary is an operating-company vocabulary, and the fund classes that genuinely apply — capital-gains distribution, return of capital, reconstitution distribution, fund merger/liquidation, sponsor or index change — have no typed representation anywhere (**B3-F21**). The one fund event in our history is typed as a $0.139 cash dividend by one source and a 1231:1000 split by the other. Vendor-supplied nature flags do not substitute: `special` is set on 1 of 334 and misses the largest irregular distribution in the set (**B3-F29**). Covers: the closed vocabulary, how a vendor's typing maps onto it, and what happens to an event that maps to nothing. Anchored by **B3-F21**, **B3-F29**, **B3-F6 (g)(h)**, **B3-F17** |
| **B3-C12** | **NEW 2026-07-30 — point-in-time needs a knowledge axis as well as an event axis.** **A1-C17** rules that read-time adjustment filters actions to `ex_date ≤ bar date`. That is necessary and correct, and **incomplete**: a revision carries a past ex-date and a present knowledge date, so ex-date filtering alone reproduces *today's* view of history rather than what was knowable at the time (**B3-F31**). Alpaca's `process_date` is the only candidate knowledge field (present on all 334 rows, unlike `record_date`/`payable_date` — **B3-F28**); Yahoo supplies none, so a knowledge date for the deep history may have to be **assigned by us at ingest** rather than sourced. Covers: which field carries knowledge time, what it means for manually entered rows, and whether the trainer and trader need different as-of views. **Extends A1-C17; does not contradict it.** Anchored by **B3-F31**, **B3-F27**, **B3-F28** |

---

## B3.10 — Dependency map

### Within B3

| Blocked | Blocked by | Why |
|---|---|---|
| **B3-C1** (source and own) | **B3-C3** | Cannot own a feed until its source is chosen |
| **B3-C4** (schema) | **B3-C3**, **B3-C8**, **B3-C10**, **B3-C11**, **B3-C12** | Field set depends on what the sources supply *and* on four questions the candidate does not currently answer: share basis, identity/revision, vocabulary, knowledge time |
| **B3-C9** (completeness) | **B3-C3** | Completeness is asserted *between sources*, so it cannot be defined before they are |
| **B3-C10** (identity/revision) | — | Independent; forced by measured vendor behaviour regardless of source choice |
| **B3-C11** (vocabulary) | — | Independent; derived from our instruments, not from any vendor |
| **B3-C5** (remediation ledger) | — | Independent; forced by **B3-F11** regardless of source choice |
| **B3-C7** (convention) | — | Independent; a documentation and comparison constraint |
| **B3-C8** (share basis) | — | Independent; forced by **B3-F11** plus measured source divergence |
| **B3-C12** (knowledge axis) | — | Independent; extends **A1-C17** |

### Crossing dataset boundaries

| Blocked | Blocked by | Why |
|---|---|---|
| **A1-C13** — price basis | **B3-C3**, **B3-C8** | Read-time adjustment is exactly as complete as the actions ledger — and the factor is **2.000× wrong** if the share bases are not reconciled |
| **A1-C17** — point-in-time adjustment | **B3-C3**, **B3-C12**, **A2** | Needs actions to filter against, and needs a knowledge axis the ruling does not name |
| **A1-C3** — the CBOE migration | **B3-C3**, **B3-C5**, **B3-C7**, **B3-C8**, **B4** | Needs dividends and spin-offs (not splits — **B3-F11**); replayable cleanup (**B3-C5**); a reconciled share basis (**B3-C8**); and the validator currently **throws** on pre-2006 dates (**B4**) |
| **A1-C12** — cross-source gate | ← *informed by* **B3-C7**; **now load-bearing** via **B3-C9** | The convention must be encoded, or 80 of 331 comparisons are false discrepancies. And because absence is undetectable from price, cross-source agreement is the **only** completeness signal there is |
| **A1-C11** — candle audit state | ← informed by **B3-C5** | In-table state does not survive a full replacement |
| **A1-C18** — free independent crypto source | ← *informed by* **B3-F32** | Yahoo serves BTC-USD / ETH-USD (3,270 / 3,186 bars) and is reachable. *(Supersedes the earlier "constrained by B3-F5 — untestable" edge)* |
| **B4** — the 12th dataset | ← created by **B3-C6** | Observation events and the trading calendar |

```
   B3-C3 (source) ──┬──► B3-C1 ──► collector ownership (B3-C2)
                    ├──► B3-C4 (schema) ◄── B3-C8, B3-C10, B3-C11, B3-C12
                    ├──► B3-C9 (completeness) ──► A1-C12  [now load-bearing]
                    ├──► A1-C13 ──► A1-C3   (the equity migration)
                    └──► A1-C17 ◄── B3-C12, A2

   B3-F11 (CBOE split-adjusted) ──┬─► B3-C5 (replay) ──► A1-C11
                                  ├─► B3-C7 (convention) ──► A1-C12
                                  └─► B3-C8 (share basis) ──► A1-C13   [2.000× error]

   B3-F26 (absence undetectable) ──► B3-C9 ──► A1-C12   [only completeness signal]
   B3-F8 / F9 / F32 ──► B3-C6 ──► B4  (12th dataset)
```

---

## B3.11 — How this document was rebuilt

Recorded so the method can be reused and its cost is not paid twice.

**The diagnosis.** The first pass used the CBOE vendor snapshot as its instrument, because the
question it needed to answer was *is CBOE adjusted?*. Everything that instrument revealed got
written down as a B3 finding. Four were relocated out and three split, because they were candle
facts, not corporate-action facts.

**The test applied to every candidate finding in the rebuild:** *is this a fact about corporate
events?* — not *did this session find it*. Applied prospectively rather than as a cleanup.

**The method.** Work forward from the instruments we hold → the events they can produce → what each
event needs in order to be applied, audited and replayed → what is collected → how it would be
collected → how it would be checked → who consumes it. Findings fall out of that chain. The vendor
snapshot is evidence within it, not its starting point.

**What the change bought,** counted honestly:

| Section | First pass | Rebuild |
|---|---|---|
| What should be collected | 1 finding (**B3-F6**, schema gaps) | +**B3-F21** — the universe itself, plus the apply/audit/replay field split |
| How collection would work | Alpaca bands + "yfinance unreachable" | +**B3-F22**, **B3-F23** — a reachable source for the contested window, and holes in both candidates |
| Checks and balances | "nothing exists" | +**B3-F24**, **B3-F25**, **B3-F26**, **B3-F27**, **B3-F28**, **B3-F29** — measured infeasibility of price-based detection, a real revision, opposite conventions, banded fields, an unreliable flag |
| Consumption | "nothing reads it" | +**B3-F30**, **B3-F31** — the 2.000× basis error on the default path, and the missing knowledge axis |
| Parity | measured for our range | +**B3-F32** — measured against an independent source |

**Four claims the first pass asserted were corrected** — B3-F4, B3-F5, B3-F6 (h), B3-F17 — and one
"USER RULING" was withdrawn as an escalation of what the user actually said (**B3-C3**).

**Borderline calls, recorded because the test was close:**

| Candidate | Kept in B3? | Reasoning |
|---|---|---|
| Yahoo's chart endpoint is reachable | **Yes** | Its relevance here is that it is the only demonstrated source of *corporate events* for 2004–2016. The DNS filter itself remains infrastructure, out of scope |
| Yahoo serves BTC-USD / ETH-USD | **Yes, narrowly** | Recorded as **B3-F32** because it measures the **parity divergence** against a second source, which DS-7 requires. The A1-C18 relevance is noted and **not** pursued — that is A1's item |
| The factor depends on the prior close | **Yes** | It is a fact about how an *event* becomes usable. The candle defects it would propagate are cited (**A1-F10**, **A1-F34**, **A1-F35**), never re-found |
| SPY's stored history is adjusted without its 2018 dividend | **Yes** | The defect is in `ohlcv_daily`, but its *cause* is a corporate-action gap. Folded into **B3-F12** rather than raised as a candle finding |
| `ohlcv_daily.adjusted_close` is unused | **No** | A candle-schema fact. Already **A1-F14** |
| The 2-decimal price rounding in stored bars | **No** | A candle-precision fact; surfaced only as noise in this session's method |

---

## Forward notes

**To A2 (Derived features).** Everything the agent sees is computed from candles whose adjustment
basis is vendor-controlled and, today, **stale by 0.25–1.34 %** (**B3-F12**) *and* missing at least
one real dividend (**B3-F23**). A2 should establish whether `features_*` can be fully recomputed
from candles — A1 marked that **UNVERIFIED** (**A1-F32**) — because if it cannot, the adjustment
basis is baked irreversibly into stored features. **B3-C12** adds a second question: if adjustment
becomes point-in-time on two axes, every derived feature inherits both.

**To B4 (observation events, DS-10).** Three things are measured and should not be re-derived:

| Seed | What is established |
|---|---|
| **The rolling XNYS calendar bound** *(written up as B3-F10, relocated)* | `exchange_calendars.get_calendar("XNYS")` built with no `start` at `validation.py:61` yields a **~20-year rolling window** — first session 2006-07-31 when measured, 2006-07-28 the day before. Three call sites: `:140` raises and is caught locally; `:287` raises `DateOutOfBounds` **unguarded**; `:220` does not wrap it. **647 sessions** of CBOE history are outside the bound today, growing yearly. `start="1990-01-01"` resolves fine — a missing argument, not a data limit. **This blocks A1-C3 today** |
| **Phantom bars on closed sessions** — **A1-F10** | 2018-11-22 and 2025-05-26; 15 bars across the 8 ETFs, 20 across the 11-symbol snapshot. SPY reads 75,990,006 shares on a closed day |
| **The crypto analogue** — **B3-F8**, **B3-F32** | Venue events are crypto's equivalent and are unowned. Compounded by `ohlcv_4h.source` being 100 % NULL (**A1-F13**). Yahoo confirms there is no *corporate* event class to conflate them with |

**To B1 (Options chains).** Options are struck on the *unadjusted* underlying. If the equity series
becomes split-adjusted-but-dividend-raw (**B3-F11**), strike comparisons across a split date need
the same convention encoded (**B3-C7**), and the share-basis question of **B3-C8** applies to strikes
exactly as it does to dividend rates. Not examined this session.

**To every remaining dataset.** **B3-F20**'s cross-cutting form: `data_audit_job` audits **one
dataset of twelve**. B1, B2 and B4 will each meet this gap, and nothing announces the omission.

---

## Evidence base

- **Live Postgres (pg16)** — `corporate_actions` (0 rows, DDL), `ohlcv_daily` (vintages, DDL),
  `ohlcv_4h`, `data_quarantine`, `alert_log`, full table list.
- **Live Alpaca corporate-actions API** — 8 ETFs 2004–2026 (334 dividends, 2 splits, full field
  dumps); the **16-type vocabulary** enumerated via the API's own 400 response; per-symbol-per-year
  completeness pivot; `special`/`foreign`/date-completeness census; 8 spin-off probes 2021–2025.
- **Live Alpaca bars API** — `RAW`, `ALL`, `SPLIT`, `DIVIDEND` modes; the full 2016–2026 `dividend ÷
  raw` step decomposition across 8 symbols used to recover Alpaca's *implied* event calendar.
- **Live Yahoo chart API** (`query1`/`query2.finance.yahoo.com`, v8 chart, `events=div|split`) —
  8 ETFs 2004→today (713 dividends, 4 splits) and BTC-USD / ETH-USD.
- **CBOE vendor snapshot** `~/swingrl/data/vendor_snapshots/cboe/2026-07-25/` — 11 symbols,
  5,671 common trading dates 2004-01-02 → 2026-07-23; used for the ex-date signal/noise study,
  the detector precision test and the VTI 2008 continuity check.
- **Network** — `getent`/`dig` against local and public resolvers, from host and the collector
  container; HTTP reachability probes of Nasdaq, SSGA, stockanalysis.com and SEC EDGAR.
- **Code, read at line level** — `data/corporate_actions.py` (all 176 lines), `postgres_schema.py`,
  `validation.py`, `cross_source.py`, `alpaca.py:100-140`, `monitoring/alerter.py`,
  `scripts/collector_main.py`, `data/options/audit.py`, `config/swingrl.yaml`,
  `tests/data/test_corporate_actions.py` (read, **not executed**).

## Confidence

Weaker than the rest, explicitly:

| Claim | Status |
|---|---|
| CBOE is split-adjusted **before 2016** | **Inference with one corroborating case.** VTI 2008-06-18 shows the expected continuity, but continuity is equally consistent with no split, and the split is single-sourced from Yahoo and contradicted by Alpaca |
| SPY genuinely paid a dividend on 2018-06-15 | **Two independent sources agree it exists** (Yahoo returns $1.246; Alpaca's cadence has a single 6-month gap in an otherwise unbroken quarterly series). Not confirmed against an issuer document |
| The 8–19 % / ~45 % level-drift figures for 2004–2016 | **Estimates** extrapolated from the measured per-event rate. The **374 event count is measured** |
| XLF 2016-03-18's Yahoo ÷ Alpaca ratio of 0.6605 | **Measured but unexplained** — the other two pre-spin-off ratios are ≈1/1.231 |
| Yahoo's data quality beyond the overlap window | **UNVERIFIED.** Agreement was tested only where Alpaca also has data (2016+). The 374 pre-2016 events are **unvalidated against any second source** |
| Which device applies the DNS filter (**B3-F5**) | **UNVERIFIED** — infrastructure, out of scope |
| `api.nasdaq.com` connection failure | **Unexplained** — DNS resolves, TCP/TLS does not complete |
| Legacy DuckDB provenance of the empty table | **Unclosed** — the file is no longer valid DuckDB |
| The **B3-C4** candidate's gap closures | **Reasoned from its field lists**, not tested against an implementation |
| Why the detector and validator were never integrated | **Inference** from code structure and a DATA-11 docstring |
| The test suite passes | **Read, not executed** |

Everything else carries a `file:line`, a query, or command output produced this session.

---

**Status: COMPLETE — 28 findings (20 High / 8 Medium / 0 Low), 12 carry items.**
**Rebuilt from first principles 2026-07-30; all prior IDs preserved, B3-F10/F14/F15/F16 retired.**
**Four first-pass claims corrected (B3-F4, B3-F5, B3-F6 (h), B3-F17) and one mis-recorded "ruling" withdrawn (B3-C3).**
