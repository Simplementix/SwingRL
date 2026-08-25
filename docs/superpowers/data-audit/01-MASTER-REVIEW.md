# Group A/B Data Audit — Review pass, master spine

> **This file is an index, not a container.** Each dataset's review lives in its own file under
> [`reviews/`](reviews/). What lives here is the material that **crosses dataset boundaries**:
> the carry register, the dependency map, and the shared glossary.
>
> Process contract: [`00-PROCESS.md`](00-PROCESS.md).

---

## How this document is organised

Read this once; it is written so no future session has to re-derive it.

| Where | What it holds | Who adds to it |
|---|---|---|
| `00-PROCESS.md` | The contract — scope, session rules, ID conventions, the review-section template | Amended only by a recorded scope decision (DS-*) |
| **this file** | Dataset roster, master carry register, cross-dataset dependency map, shared glossary | Every review session appends its dataset's rows |
| `reviews/<ID>-<slug>.md` | One dataset's full review — evidence, findings, that dataset's own index | Written once, by the session that reviews the dataset |

**Why the split.** A1 alone runs to ~1,900 lines. Eleven datasets in one file would reach roughly
10,000, and every session would open all of it to append to the end. Splitting was done at one
dataset, when it cost nothing (DS-9).

### ID conventions

| Kind | Form | Rule |
|---|---|---|
| Finding | `<DS>-F<n>` — e.g. `A1-F17` | Dataset-scoped, assigned in discovery order. **Never renumbered, never reused.** Discovery order carries no meaning; severity, environment and disposition are columns in each dataset's index |
| Carry item | `<DS>-C<n>` — e.g. `A1-C13` | Dataset-scoped. Carried into the spec pass. **Never renumbered, never reused** |

Both are stable identifiers, not sort keys. A finding is cross-referenced dozens of times across
the corpus; renumbering to make an ordering prettier would break every one of those references and
buy nothing that a column cannot express.

### What each register is for

- **Carry register** — a carry item is the unit of work that reaches the spec pass, and items
  routinely depend on *other datasets*. That is why they are indexed centrally and findings are not:
  findings belong to one dataset, items do not.
- **Dependency map** — answers "what blocks X?" without reading a 1,900-line review.
- **Findings are not duplicated here.** Each dataset's own §*.7 index is the authority. Copying
  ~300 finding rows into a second place would guarantee drift.

---

## Dataset roster

| # | Dataset | Review | Findings | Carry items | File |
|---|---|---|---|---|---|
| **A1** | Candles | ☑ 2026-07-25 | 37 — 22 H / 14 M / 1 L *(A1-F34…F37 amended 2026-07-29; the residual "CBOE is unadjusted" wording swept 2026-07-30 — 9 passages incl. 3 carry items)* | 18 | [`reviews/A1-candles.md`](reviews/A1-candles.md) |
| **B3** | Corporate actions | ☑ 2026-07-30 · **walked 2026-08-03** | 28 — 20 H / 8 M / 0 L *(**rebuilt from first principles**; B3-F14/F15/F16 relocated to A1, B3-F10 to B4; IDs retired. **All 28 walked with the user 2026-08-03 — every one stands unamended**)* | **14** *(B3-C13, B3-C14 added at the walkthrough)* | [`reviews/B3-corporate-actions.md`](reviews/B3-corporate-actions.md) |
| **B4** | Observation events *(trading calendar + venue events)* | ☑ 2026-08-05 · **extended 2026-08-10/11** · **walked 2026-08-25** | **29 — 21 H / 8 M / 0 L** *(new dataset, DS-10; reviewed third by **DS-11**. **§B4.1(h)** "present but distorted" (F18–F20); **§B4.1(i)** nine-class completeness pass (F21–F26); **§B4.3** source hunt (F27–F29). **Three clean negatives**, **two of the review's own errors**, and **one stale B3 claim** recorded in §B4.0. **All 29 walked with the user 2026-08-25 — 27 stand unamended; B4-F20 and B4-F26 amended after re-measurement, B4-F13/F17/F24 corrected; §B4.11)**)* | **15** | [`reviews/B4-observation-events.md`](reviews/B4-observation-events.md) |
| **A2** | Derived features | ☐ **next** | — | — | — |
| **A3** | Macro | ☐ | — | — | — |
| **A4** | Regime (HMM) | ☐ | — | — | — |
| **A5** | Turbulence | ☐ | — | — | — |
| **A6** | Portfolio state | ☐ | — | — | — |
| **A7** | Fundamentals | ☐ | — | — | — |
| **A8** | Sentiment | ☐ | — | — | — |
| **B1** | Options chains | ☐ | — | — | — |
| **B2** | Calendar events | ☐ | — | — | — |

---

## Master carry register

One line per item. **The full statement of each item lives in its dataset's review** — this table
exists to make cross-dataset dependencies visible, not to restate them.

| ID | Item, in one line | Depends on | Kind |
|---|---|---|---|
| **A1-C1** | Restore the 22-day crypto hole from the already-downloaded archive | — | Data repair |
| **A1-C2** | The two venue-volume seams, independent of the gaps | — | Data quality |
| **A1-C3** | Migrate equity candles to CBOE, replacing the table **in full** | **A1-C14**, **A1-C15**, **A1-C11**, **B3** | USER RULING |
| **A1-C4** | Store authority — move the watermark to Postgres; define what the Parquet drift check asserts | — | Wiring |
| **A1-C5** | The five DS-7 parity divergences | **A1-C11** *(fifth only)* | Parity |
| **A1-C6** | The watermark becomes the candle integrity audit, raising through `alert_log` | — | USER DIRECTION |
| **A1-C7** | The collector owns candles end to end — collection, historical auditing, data quality | — | USER DIRECTION |
| **A1-C8** | Most machinery already exists and reads the right store — this is wiring, not new capability | — | Scoping |
| **A1-C9** | Where integrity and quality results should live (needs a schema change) | — | Schema |
| **A1-C10** | Whether the five metadata tables collapse into one coherent model | — | USER OBSERVATION |
| **A1-C11** | Candle rows must carry audit state — provenance, last-audited, review flag, revision | ↔ **A1-C3** | USER DIRECTION |
| **A1-C12** | Cross-source validation becomes a standing gate in both environments | **A1-C18** *(crypto half)* | USER DIRECTION |
| **A1-C13** | Settle the price basis — store raw; the adjusted series lives in a **separate stored, versioned table**, rebuilt from the actions ledger *(**USER RULING 2026-08-03**, superseding "derive adjusted at read time" — B3 §B3.12)* | **B3** | USER DIRECTION |
| **A1-C14** | The validator needs new checks, not just wiring — six enumerated gaps | — | USER DIRECTION |
| **A1-C15** | Backup and rollback **before** the destructive equity replacement | — | USER RULING |
| **A1-C16** | A reference store for independent vendor payloads (yfinance, Alpaca, CBOE) | — | USER QUESTION |
| **A1-C17** | Point-in-time adjustment — filter actions to ≤ bar date. *(Amended 2026-08-03: the filter applies when the **stored adjusted table is built**; indicators key to the same `adjustment_version`; the unassessed cost moves from the hot path to a **rebuild** cost — relocated, not removed)* | **B3**, **A2** | USER RULING |
| **A1-C18** | Find a free, genuinely independent crypto source | — | USER RULING |
| **B3-C1** | The corporate-actions dataset must be sourced and owned | **B3-C3** | Scoping |
| **B3-C2** | The collector owns corporate-action collection **and** validation, end to end | — | USER DIRECTION |
| **B3-C3** | Settle the source per event type and per time band. **Restated 2026-07-30:** a free usable source back to 2004; Alpaca ongoing; manual entry is a **last-resort, one-time** path for residual gaps only | — | USER REQUIREMENT |
| **B3-C4** | Evaluate the candidate 4-table schema (raw → norm → factors → adjusted) | **B3-C3**, **B3-C8**, **B3-C10**, **B3-C11**, **B3-C12** | Schema |
| **B3-C5** | A remediation ledger — record what *we* changed, so fixes replay after a re-ingest | — | USER OBSERVATION |
| **B3-C6** | **DS-10** — observation events become a 12th dataset: trading calendar + venue events | — | USER RULING |
| **B3-C7** | Encode CBOE's split-adjusted / dividend-raw convention wherever series are compared | — | Wiring |
| **B3-C8** | **Share basis must be explicit** per source and per row — the two candidate sources use **opposite, unstated** conventions, and mixing them is a silent **2.000×** factor error | — | Correctness |
| **B3-C9** | **Settle what proves the event set complete**, given price cannot — a price detector runs at 2.4 % precision. Makes **A1-C12** load-bearing | **B3-C3** | Correctness |
| **B3-C10** | **Event identity and revision semantics** — vendors restate by shipping an extra row; the natural key is not unique | — | Schema |
| **B3-C11** | **Derive the event vocabulary from our instruments** (all funds), not from a vendor's operating-company catalogue | — | Scoping |
| **B3-C12** | **Point-in-time needs a knowledge axis** as well as an event axis — extends **A1-C17**, does not contradict it | — | Correctness |
| **B3-C14** | **Dividend cash must be credited** — the trader receives real dividends, so `payable_date` becomes a **required** field and the dataset gains a fourth purpose (apply · audit · replay · **credit**). Blocked historically: payable date is NULL for 2016–2019 and absent from Yahoo | **B3-C3** | USER REQUIREMENT |
| **B3-C13** | **Validating our computed adjustments against an independent adjusted series (Alpaca) is a required gap, distinct from completeness (B3-C9)** — completeness asks whether every event is held, validation asks whether the held events were applied correctly | — | USER DIRECTION |
| **B4-C1** | **The session baseline must be sourced, owned, stored and reproducible for a past date** — today it is a runtime library call made in five places, with no store | — | Scoping |
| **B4-C2** | **Settle what span the baseline must cover** — the bound is rolling and the shortfall grows by one session per trading day; the CBOE migration sets the floor at 2004-01-02 | **B4-C1** | Correctness |
| **B4-C3** | **Settle how an absent observation is classified** — legitimately absent / venue outage / our loss / filter artefact — and where that is recorded. All four already coexist in the stored data | **B4-C1**, **B4-C4** | Correctness |
| **B4-C4** | **An observation-event record must be sourced and owned** for both environments. *(Restated 2026-08-11 — narrowed, not closed.)* **Equity 2019+ is solved** — NYSE's halt history is free, unauthenticated, coded, 71,718 records (**B4-F27**). **Two blanks, no candidate for either**: equity **pre-2019-02-22** (most of what A1-C3 imports) and **all of crypto** (4 sources tested, all failed — **B4-F29**). *"Halt"* is still taken by our circuit breaker | — | Scoping |
| **B4-C5** | **Settle the crypto uptime baseline** — what "should exist" means for a 24/7 venue has never been stated, and the implicit `4h`-forever answer is measurably false | **B4-C4** *(pre-2019 era)* | Parity |
| **B4-C6** | **Reconcile the two calendar authorities** — Alpaca broker clock vs xcals — or record the divergence as justified under DS-7 | — | Wiring |
| **B4-C7** | **Settle what a detector must be able to see** — today's cannot see a trailing gap, writes nothing durable, and has never fired for a candle | **B4-C1**, **B4-C3** | Correctness |
| **B4-C8** | **Settle whether an observation flag lives in the candle row or a side table** — the same schema question as **A1-C11**, and in-row state does not survive A1-C3's full-table replacement | **B4-C3** | Schema |
| **B4-C9** | **Settle whether a half-day session is a comparable observation** — 46 early closes are indistinguishable from full sessions everywhere except options | **B4-C1** | Correctness |
| **B4-C10** | **The provably-ours crypto losses are still recoverable** — 5 gap runs (7 bars) from the public archive plus the 159-bar 2026 hole from the live API, both verified serving today. Scope sits with **A1-C1** | **B4-C3** | Data repair |
| **B4-C11** | **Settle how a "present but distorted" observation is handled** — a bar that exists, is valid and passes every check, but whose price discovery was impaired (halt, LULD cascade, venue dislocation). **Distinct from B4-C3**, which classifies *absences*: nothing is missing, so no absence-shaped detector applies and no threshold on the bar separates a real 21 % range from a broken one | **B4-C4** | Correctness |
| **B4-C12** | **Settle whether crypto volume is comparable across 2023-06-07** — a **4.30×** permanent level change sits mid-series with no marker, **1.37×** market-wide and **3.15×** venue-specific. Adjacent to **A1-C2** but a **different cause** | — | Correctness |
| **B4-C13** | **Symbol and pair lifecycle must be represented** — listing, delisting, suspension, ticker reuse. Measured **clean today**, so a **latent** gap: nothing would tell us if it changed, and a reused ticker would splice two instruments into one series | **B4-C4** | Scoping |
| **B4-C14** | **Observation events need a knowledge date** — `fetched_at` records when a *bar* arrived, never when something *became known about* it. This session classified 2017–2023 gaps in 2026 with nowhere to record that. Same axis as **B3-C12**, applied to observations | ↔ **B3-C12** | Correctness |
| **B4-C15** | **Settle the storage and query timezone convention** — a `timestamptz` column under an `America/New_York` session makes every date-bounded query silently wrong by 4–5 hours, including the queries that judge completeness | — | Wiring |

---

## Cross-dataset dependency map

Only the edges that **cross a dataset boundary** appear here. Within-dataset chains live in that
dataset's own review — for A1, in [§A1.10](reviews/A1-candles.md).

| Blocked | Blocked by | Why |
|---|---|---|
| **A1-C3** — the CBOE migration | **B3 Corporate actions** | CBOE is **split-adjusted but dividend- and spin-off-raw** (**B3-F11**, 2026-07-29 — this corrects A1's "unadjusted", which was inferred from a dividend-only check). The conclusion is unchanged but the requirement is narrower: the migration needs **dividends and spin-offs**, plus split *records* for change detection, not split factors. `corporate_actions` holds **0 rows** and has no producer |
| **A1-C13** — price basis | **B3-C3**, **B3-C8** | Read-time adjustment is exactly as complete as that table — and the factor is **exactly 2.000× wrong** if the share bases of rate and close are not reconciled (**B3-F30**) |
| **A1-C17** — point-in-time adjustment | **B3-C3**, **B3-C12**, **A2 Derived features** | Needs the actions table to filter against; needs a **knowledge axis** the ruling does not name (**B3-F31**); and needs to know whether `features_*` can be fully recomputed (**A1-F32**, UNVERIFIED) |
| **A1-C11** — candle audit state | ← *informed by* **B3-C5** | In-table audit state does not survive the full-table replacement that **A1-C3** requires — evidence for the in-table-vs-side-table question |
| **A1-C12** — cross-source gate | ← *informed by* **B3-C7**; **now load-bearing** via **B3-C9** | The CBOE convention must be encoded or the comparator reports **80 of 331** false discrepancies (**B3-F25**). And because a missing event is **undetectable from price** (**B3-F26**), cross-source agreement is the *only* completeness signal that exists — so a single-source event ledger is unverifiable by construction |
| **A1-C18** — free independent crypto source | ← *informed by* **B3-F32** | **Supersedes the earlier "blocked by B3-F5" edge.** Only `fc.yahoo.com` is DNS-filtered; Yahoo's data endpoint is reachable and serves BTC-USD (3,270 bars) and ETH-USD (3,186 bars) |
| **B4** — the 12th dataset | ← *created by* **B3-C6** | Trading calendar + venue events (DS-10). Seeded with the rolling-XNYS-bound measurement relocated from B3 on 2026-07-29 |
| **A1-C3** — the CBOE migration | **B4-F1**, **B4-F2** | `validation.py:287` raises `DateOutOfBounds` **unguarded** on pre-2006 dates and `:220` does not wrap it, so the migration crashes on its first pre-2006 bar. **653 sessions** affected on 2026-08-05 — **and the count grows by one per trading day**, measured at 652 the day before. *(Supersedes the earlier "647, growing yearly" — the growth rate was wrong.)* Also mislabels rather than crashes on the quarantine path (**B4-F3**) |
| **A1-C1** — restore the crypto hole | ← *extended by* **B4-C10**, **B4-F8** | A1-C1 scopes the 2019 hole. B4 adds **5 further gap runs that are provably ours** and the **159-bar 2026 hole**, all still served by the venue today — and warns that **7 genuine venue outages** sit in the same table and must not be filled alongside them |
| **A1-C11** — candle audit state | ↔ **B4-C8** | Observation flags and audit state are the same schema question asked twice; both must survive A1-C3's full-table replacement |
| **A1-C12** — cross-source gate | ← *strengthened by* **B4-F8**, **B4-F11** | B3-F26 made cross-source the only completeness signal for *events*. B4 shows the same for *observations*: the internal detector cannot see a trailing gap by construction, so an external comparison is the only thing that can. **B4-F8 demonstrates a working one that costs nothing** (`data.binance.vision`) |
| **A1-C18** — free independent crypto source | ← *sharpened by* **B4-F10** | Independence is not the only requirement — **reachability of the historical venue** is a separate, unmet one: `api.binance.com` is **HTTP 451** from here. The public archive is demonstrated free, unauthenticated and complete for 13 tested dates |
| **A1-C2** — the two venue-volume seams | ↔ **B4-C12** | Adjacent, **different causes**: A1-C2 is *source change* (Global→US stitch, IEX/SIP); B4-C12 is a *venue event* with the source unchanged on both sides. A normalisation assuming one seam per series would be wrong |
| **A2 Derived features** | ← *warned by* **B4-F15**, **B4-F13**, **B4-F18**, **B4-F19**, **B4-F20** | 46 unflagged half-day sessions and 22 gap bars enter the feature pipeline, `technical.py:126/174` forward-fill across them — plus halt-day bars and a **4.30× permanent volume rescale** at 2023-06-07 |
| **A5 Turbulence**, **A4 Regime** | ← *warned by* **B4-F18**, **B4-F19**, **B4-F20** | Turbulence percentiles and HMM fits span the 4 MWCB days, the 2015-08-24 dislocation (imported by A1-C3) and the crypto volume break — each a genuine extreme in the data and a non-comparable observation in reality |
| **A1-C3** — the CBOE migration | ← *also blocked by* **B4-F21**, **B4-F23** | It brings a **new vendor**, whose revision behaviour is unknown, into tables that **cannot name a vendor**. `ON CONFLICT DO NOTHING` would discard any correction that vendor issued, silently |
| **A1-C3** — the CBOE migration | ← *coverage gap from* **B4-F27** | The only halt source starts **2019-02-22**; the migration imports from **2004-01-02**, so ~15 of the 22 years it adds have **no halt record obtainable at any price** — including 2015-08-24 (**B4-F19**) and the 2008 cluster |
| **B3-C5** — the remediation ledger | ← *evidenced by* **B4-F25** | Raised for corporate actions; B4 supplies a concrete **candle** instance — the 2026-07-18 partial bar was repaired by replacing a row `DO NOTHING` cannot replace, with no record. `operator_actions` = 0 rows |
| **B3-C12** — the knowledge axis | ↔ **B4-C14** | The same axis on the other dataset: B3-C12 asks it of events, B4-C14 of observations. This session created the instance by classifying 2017–2023 gaps in 2026 |
| **Every remaining dataset** | ← *warned by* **B4-F24**, **B4-F26** | Two cross-cutting hazards: the DB session timezone is **`America/New_York`**, so date-bounded audit queries silently shift 4–5 h in *any* dataset; and `data_ingestion_log` status is unreliable everywhere — **436 macro `success` runs for 48 retained rows** over 2026-03-11 → 2026-04-06 |
| **B2 Calendar events** | ← *boundary with* **B4-C4** | `calendar_events` holds *scheduled economic* events; B4's calendar is *market structure*. Two datasets sharing a word — the boundary must be stated, not assumed |
| **B3-C1**, **B3-C4** | **B3-C3** | Neither ownership nor schema can be settled before the source is |
| **B3-C4** — the candidate schema | **B3-C8**, **B3-C10**, **B3-C11**, **B3-C12** | The rebuild found four questions the candidate does not answer: share basis, identity/revision, vocabulary, knowledge time |
| **B1 Options chains** | ← *constrained by* **B3-C7**, **B3-C8** | Options are struck on the *unadjusted* underlying, so strike comparisons across a split date need the same convention and share basis encoded |
| **Every remaining dataset** | ← *warned by* **B3-F20** | `data_audit_job` audits **1 dataset of 12**. B1, B2 and B4 will each meet this gap, unannounced |

```
   B3-C3 (source) ──┬──► B3-C1 ──► collector ownership (B3-C2)
                    ├──► B3-C4 (schema) ◄── B3-C8, B3-C10, B3-C11, B3-C12
                    ├──► B3-C9 (completeness) ──► A1-C12   [now load-bearing]
                    ├──► A1-C13 ──► A1-C3   (the equity migration)
                    └──► A1-C17 ◄── B3-C12, A2 Derived features

   B3-F11 (CBOE is split-adjusted) ──┬─► B3-C5 (replay) ──► A1-C11
                                     ├─► B3-C7 (convention) ──► A1-C12
                                     └─► B3-C8 (share basis) ──► A1-C13   [2.000× error]

   B3-F26 (absence undetectable from price) ──► B3-C9 ──► A1-C12
                                                  [cross-source is the ONLY completeness signal]

   B4 (rolling calendar bound) ──► A1-C3   [validation throws on pre-2006 bars today]
   B3-F8 / F9 / F32 ──► B3-C6 ──► B4  (12th dataset)

   B4-C1 (baseline stored) ──┬──► B4-C2 (span) ──► A1-C3      [653 sessions, +1 per trading day]
                             ├──► B4-C9 (half-days) ──► A2
                             └──► B4-C3 (classification) ──┬──► B4-C7 (detector) ──► A1-C12
                                        ▲                  ├──► B4-C8 (where) ◄──► A1-C11
                                        │                  └──► B4-C10 (repair) ──► A1-C1
   B4-C4 (incident source) ─────────────┘
        └──► B4-C5 (crypto baseline) ──► A1-C18             [Binance Global is HTTP 451]

   B4-F8 (8 of 13 gaps are the venue's, 5 are ours) ──► B4-C3
        [4 kinds of absence already coexist in ohlcv_4h and are indistinguishable]
```

**This is why B3 was moved ahead of A2 in the review order (DS-8).** Three of A1's items are
blocked on a table that is empty, and one of them is the migration the whole equity plan rests on.

**What B3 changed about that picture.** The blocker is *not* the empty table alone — it is the
**source** (**B3-C3**), because the table has no producer either. And **B3-F11** refuted the premise
that CBOE is unadjusted: it is **split-adjusted**, so the migration needs dividends and spin-offs,
not splits.

**What the 2026-07-30 rebuild changed on top of that.** The first B3 pass established *absence*;
the rebuild established *fidelity*, which is harder. A free source covering the contested 2004–2016
window is now demonstrated reachable (**B3-F22**, 374 measured events), so the deep history is
obtainable — but **neither candidate source is complete**, one **contradicts itself across its own
endpoints**, they quote the same dividend on **opposite unstated share bases**, a vendor **revision
arrives as a duplicate row**, and a **missing event cannot be seen in price at all**. The blocker
moved from "where do we get the data" to "how do we know the data is right".

---

## Shared glossary

Terms used across the audit. Each review may add dataset-specific terms in its own file.

| Term | Meaning |
|---|---|
| **Bar / candle** | One OHLCV record: open, high, low, close, volume for a time period |
| **OHLCV** | Open / High / Low / Close / Volume |
| **SIP** | Securities Information Processor — the US consolidated tape. Volume from *all* exchanges |
| **IEX** | A single US exchange (~2% of volume). Alpaca's free feed. Prices are real; volume is only IEX's slice |
| **Upsert** | Insert a row, or overwrite it if the key already exists |
| **`ON CONFLICT DO NOTHING`** | Insert a row, or **silently skip** it if the key already exists. The existing row is never updated |
| **Watermark** | The "how far have I got" marker an incremental fetch resumes from |
| **Vintage** | A distinct ingestion run, identified by its `fetched_at` timestamp |
| **Stitch** | The join point where two data sources are concatenated into one series |
| **XNYS** | The NYSE trading calendar (`exchange_calendars` library), used to decide which dates *should* have an equity bar |
| **Quarantine** | Rows the validator rejected; written to `data_quarantine` and a Parquet file rather than the main table |
| **Observation slot** | One element of the fixed-length vector the RL agent sees each step |
| **Disposition** | For a finding: what a known, ruled-on change does to it — removed, untouched, made worse, or created |
| **Carry item** | A question or direction recorded by a review and handed to the spec pass. **Never a solution** |
| **Ex-date** | The first day a buyer is *not* entitled to a pending distribution. Price adjustment keys off this date, not the record or payable date |
| **Cumulative adjustment factor** | The product of every event factor after a given bar. Multiplying a raw bar by it produces the adjusted bar |
| **Split-adjusted** | Prices and volumes restated onto the current share basis. **CBOE is split-adjusted but dividend-raw** (B3-F11) — a hybrid matching no standard vendor convention |
| **Instrument event** | Something that changed the security itself — split, dividend, spin-off. Handled by **retroactive adjustment** |
| **Observation event** | Something that changed only our *view* — exchange outage, halt, delisting, de-peg, vendor defect. Handled by **flagging or exclusion**, never adjustment |
| **Remediation ledger** | A record of corrections *we* applied, keyed so they can be replayed onto a freshly re-ingested series (B3-C5) |
| **Spin-off** | A distribution of a subsidiary's shares. Removes value from the parent without changing its share count — so price adjusts, volume does not |
| **Share basis** | Which share count an amount is quoted against. A dividend paid before a 2:1 split can be stated **as paid** (the amount a holder actually received) or **restated** onto today's post-split basis (half of it). Both are "correct"; neither vendor says which it uses. Mixing bases is a silent 2× error (B3-C8) |
| **As-paid / restated** | The two share-basis conventions. **Alpaca reports as-paid; Yahoo reports restated** — measured across 77 events (B3-F25) |
| **Knowledge date** | When a record became known *to us*, as distinct from when the event took effect (ex-date). Needed to reproduce what was knowable at a past moment, because a revision has a past ex-date and a present knowledge date (B3-C12). **Applies to observations too** — B4 classified 2017–2023 gaps in 2026, a past effect-date with a present knowledge-date, and `fetched_at` cannot express it (B4-C14) |
| **Halt** | In this codebase the word is **already taken** by our own circuit breaker (`CBState.HALTED`). A **venue** halt — trading suspended by the exchange — has no name, no field and no record (B4-F7). NYSE's feed codes them as `LULD pause`, `News pending`, `Corporate Action`, `ETF Component Prices Not Available` and others |
| **LULD pause** | Limit Up-Limit Down: an automatic exchange halt when a security moves outside a price band. The dominant halt reason — **60,362 of 71,718** NYSE records. **XLI took one on 2020-03-12** and our bar does not say so (B4-F28) |
| **Vendor vs venue** | Two different sources of the same symptom. The **venue** is where trading happened (NYSE, Binance.US); the **vendor** is who sold us the record (CBOE, Alpaca, the archive). A vendor outage and a venue outage produce an identical absent bar and mean opposite things about the market. Neither candle table can name either (B4-F23) |
| **Restatement / revision** | A vendor re-issuing an event it already published. Measured behaviour: it arrives as an **additional row** with a new vendor id, not as an update (B3-F27) |
| **Implied event calendar** | The event dates recovered by dividing a vendor's *adjusted* price series by its *raw* series and reading off the re-basing steps. Used to check a vendor's price data against its own event API (B3-F24) |
| **Instrument identity** | What still identifies a security after a ticker change — CUSIP, not symbol. Absent from the current schema (B3-F6 (f)) |
| **Session baseline** | The set of (symbol, timestamp) slots an observation *should* occupy. For equity, the XNYS session calendar; for crypto, 24/7 uptime with exceptions. **Not stored anywhere today** (B4-F5) |
| **Rolling calendar bound** | `exchange_calendars.get_calendar(name)` built with no `start` returns a window anchored to *now* — first session = today − 20 years — so the answerable range **moves every day**. Measured 2006-08-04 on 2026-08-04 and 2006-08-07 on 2026-08-05 (B4-F1) |
| **Venue outage vs collector outage** | Two causes of an identical symptom — an absent bar. The first is a fact about the world (flag it); the second is our defect (repair it). Measured to coexist in `ohlcv_4h`: 7 runs venue, 5 runs ours (B4-F8) |
| **Legitimately absent** | A slot the baseline says should be empty — a closed market. **Not a gap**, and must be excluded from any completeness count |
| **Trailing gap** | Missing observations at the *newest* end of a series. Invisible to any detector that derives its expected range from the fetched data's own max — the shape of every live ingestion failure (B4-F11) |
| **Early close** | A half-day session (typically 13:00 ET). 46 in the CBOE equity range, carrying roughly half a session's volume. Flagged for options, unflagged for candles (B4-F15) |
| **Present but distorted** | An observation that **exists, is arithmetically valid and passes every check**, yet is not comparable to its neighbours because the venue's price discovery was impaired — a market-wide halt, an LULD cascade, a venue liquidity break. Defeats gap, staleness and cross-source checks by construction, because nothing is missing and the prices agree. Only an **event record** separates it from a genuine extreme (B4-F18, B4-F19, B4-F20; B4-C11) |
