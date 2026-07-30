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
| **B3** | Corporate actions | ☑ 2026-07-30 | 28 — 20 H / 8 M / 0 L *(**rebuilt from first principles**; B3-F14/F15/F16 relocated to A1, B3-F10 to B4; IDs retired)* | 12 | [`reviews/B3-corporate-actions.md`](reviews/B3-corporate-actions.md) |
| **A2** | Derived features | ☐ **next** | — | — | — |
| **A3** | Macro | ☐ | — | — | — |
| **A4** | Regime (HMM) | ☐ | — | — | — |
| **A5** | Turbulence | ☐ | — | — | — |
| **A6** | Portfolio state | ☐ | — | — | — |
| **A7** | Fundamentals | ☐ | — | — | — |
| **A8** | Sentiment | ☐ | — | — | — |
| **B1** | Options chains | ☐ | — | — | — |
| **B2** | Calendar events | ☐ | — | — | — |
| **B4** | Observation events *(trading calendar + venue events)* | ☐ — **new, DS-10** | — | — | — |

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
| **A1-C13** | Settle the price basis — store raw, derive adjusted at read time | **B3** | USER DIRECTION |
| **A1-C14** | The validator needs new checks, not just wiring — six enumerated gaps | — | USER DIRECTION |
| **A1-C15** | Backup and rollback **before** the destructive equity replacement | — | USER RULING |
| **A1-C16** | A reference store for independent vendor payloads (yfinance, Alpaca, CBOE) | — | USER QUESTION |
| **A1-C17** | Point-in-time adjustment — filter actions to ≤ bar date; **read-time cost unassessed** | **B3**, **A2** | USER RULING |
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
| **A1-C3** — the CBOE migration | **B4** | `validation.py:287` raises `DateOutOfBounds` **unguarded** on pre-2006 dates and `:220` does not wrap it, so the migration crashes on its first pre-2006 bar. **647 sessions** affected today, growing yearly |
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
| **Knowledge date** | When a record became known *to us*, as distinct from when the event took effect (ex-date). Needed to reproduce what was knowable at a past moment, because a revision has a past ex-date and a present knowledge date (B3-C12) |
| **Restatement / revision** | A vendor re-issuing an event it already published. Measured behaviour: it arrives as an **additional row** with a new vendor id, not as an update (B3-F27) |
| **Implied event calendar** | The event dates recovered by dividing a vendor's *adjusted* price series by its *raw* series and reading off the re-basing steps. Used to check a vendor's price data against its own event API (B3-F24) |
| **Instrument identity** | What still identifies a security after a ticker change — CUSIP, not symbol. Absent from the current schema (B3-F6 (f)) |
