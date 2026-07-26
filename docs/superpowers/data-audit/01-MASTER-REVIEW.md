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
| **A1** | Candles | ☑ 2026-07-25 | 33 — 19 H / 13 M / 1 L | 18 | [`reviews/A1-candles.md`](reviews/A1-candles.md) |
| **B3** | Corporate actions | ☐ **next** | — | — | — |
| **A2** | Derived features | ☐ | — | — | — |
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
| **A1-C13** | Settle the price basis — store raw, derive adjusted at read time | **B3** | USER DIRECTION |
| **A1-C14** | The validator needs new checks, not just wiring — six enumerated gaps | — | USER DIRECTION |
| **A1-C15** | Backup and rollback **before** the destructive equity replacement | — | USER RULING |
| **A1-C16** | A reference store for independent vendor payloads (yfinance, Alpaca, CBOE) | — | USER QUESTION |
| **A1-C17** | Point-in-time adjustment — filter actions to ≤ bar date; **read-time cost unassessed** | **B3**, **A2** | USER RULING |
| **A1-C18** | Find a free, genuinely independent crypto source | — | USER RULING |

---

## Cross-dataset dependency map

Only the edges that **cross a dataset boundary** appear here. Within-dataset chains live in that
dataset's own review — for A1, in [§A1.10](reviews/A1-candles.md).

| Blocked | Blocked by | Why |
|---|---|---|
| **A1-C3** — the CBOE migration | **B3 Corporate actions** | CBOE is unadjusted, so raw storage is only correct once corporate actions exist. `corporate_actions` holds **0 rows** |
| **A1-C13** — price basis | **B3 Corporate actions** | Read-time adjustment is exactly as complete as that table |
| **A1-C17** — point-in-time adjustment | **B3**, **A2 Derived features** | Needs the actions table to filter against, and needs to know whether `features_*` can be fully recomputed (**A1-F32**, UNVERIFIED) |

```
   B3 Corporate actions  ──►  A1-C13  ──►  A1-C3   (the equity migration)
        (0 rows)          └─►  A1-C17  ◄──  A2 Derived features
```

**This is why B3 was moved ahead of A2 in the review order (DS-8).** Three of A1's items are
blocked on a table that is empty, and one of them is the migration the whole equity plan rests on.

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
