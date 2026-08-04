# Group A/B Data Audit — Process

> **This document is the contract for the whole project. Read it in full at the start of every
> session, before touching anything else.**
>
> **Branch:** `swingrl/27-data-pipeline-audit`, cut from `swingrl/2.R-training-redesign` @ `977d446`.
> **Established:** 2026-07-25.
> **Why:** the data is the IP. It deserves deliberate, per-dataset attention rather than a broad
> sweep that reads plausible and misses things.

---

## The rule that overrides everything else

**Every prior document is an ASSUMPTION, not a finding.**

The audit register (`docs/superpowers/reviews/2026-07-24-trader-collector-audit.md`), the two
2026-07-25 specs, and the equity re-ingestion plan were written before this project began. Their
claims are *starting hypotheses*. Each one must be **re-verified against live data, the live
pipeline and the current code** before it may be restated as fact.

This is not pedantry. During the sessions that produced those documents:

- G1-1 was described as "three FRED series frozen at source". It was actually a Parquet/Postgres
  divergence, and probably a validator quarantine on top of that. **The register is wrong.**
- A-12 was presented at 95% confidence as "every model trained with ~2 weeks of macro lookahead".
  It affects 2 of 6 macro slots, the correct data was already being fetched, and the real CPI lag is
  46 days. **Overstated in both directions.**
- G4-5 was framed as "populate these two columns". The rows are per-environment, so the columns are
  probably redundant and populating them fixes nothing. **Mis-framed.**

Reading a query correctly says nothing about whether the thing it implies is true.
**Do not take anything for granted from docs alone.**

**What this rule does *not* cover.** A completed review in `reviews/` is a different class of
document: its claims were measured, each carries a `file:line`, a query or command output, and its
`<DS>.0` section records what it corrected. Those may be cited as established. Three exceptions
stay live:

- Anything the review itself marks **UNVERIFIED** is still an assumption, however confident the
  surrounding prose sounds.
- Anything **measured against live state** — row counts, table contents, freshness — was true on
  the review date and may have moved since. Re-measure before restating.
- Anything a **later ruling supersedes**.

---

## Scope

**In scope: Groups A and B, for BOTH equity and crypto, for BOTH trainer and trader.**

### Group A — observation inputs

| ID | Dataset | Equity source | Crypto source | Notes |
|---|---|---|---|---|
| **A1** | Candles | **CBOE — USER RULING 2026-07-25**, replacing Alpaca *(current)* in full | Binance.US + Binance Global archive | ☑ **Reviewed** — [`reviews/A1-candles.md`](reviews/A1-candles.md). Includes the XNYS session calendar as a validation dependency |
| **A2** | Derived features | `features_equity` | `features_crypto` | indicators, normalization, **and overnight context** |
| **A3** | Macro | FRED | FRED — the same 6, forward-filled | |
| **A4** | Regime (HMM) | fitted on SPY closes | fitted on BTCUSDT closes | |
| **A5** | Turbulence | 8 ETF daily log-returns | crypto 4H log-returns | computed live, not stored |
| **A6** | Portfolio state | own ledger | own ledger | **read path only** — see exclusions |
| **A7** | Fundamentals | source TBD | n/a — record as a justified divergence | currently a `0.0` stub |
| **A8** | Sentiment | Finnhub / Alpaca news | same | currently disabled by config |

### Group B — decision-adjacent; entering the observation space in the training redesign

| ID | Dataset | Equity source | Crypto source | Notes |
|---|---|---|---|---|
| **B1** | Options chains | CBOE — 8 ETFs + `_SPX` | **OPEN — flag in review, decide at spec** | ongoing capture is a **must**, with its own checks and balances |
| **B2** | Calendar events | FRED releases + FOMC | shared | includes `event_outcomes` (FK child of `calendar_events`) |
| **B3** | Corporate actions | **source OPEN** — Alpaca covers dividends/splits from 2016 and spin-offs only from ~2023 | n/a — justified divergence, **measured** | ⚠️ **In progress** — [`reviews/B3-corporate-actions.md`](reviews/B3-corporate-actions.md). **CBOE is split-adjusted, not raw (B3-F11)** — so this dataset is about dividends and spin-offs, plus split *records* for change detection |
| **B4** | Observation events | trading calendar (XNYS) + venue/vendor incidents | 24/7 uptime + venue incidents | **New — DS-10.** Not yet reviewed. Distinct from B3: instrument events drive *adjustment*, observation events drive *flagging* |

**12 datasets.**

### Out of scope — do not drift into these

| Group | Contents | Why |
|---|---|---|
| **C — execution** | fills, fees, commissions, slippage, reconciliation, pending orders, risk decisions, shadow trades, wash sales, circuit-breaker actions | Trader execution. **Its own separate audit, later.** Explicitly excluded 2026-07-25 |
| **D — trainer internals** | `memories` (4.9 M rows), `patterns*`, `consolidations*`, `training_epochs`, `training_runs`, `backtest_results`, `fold_results`, LLM logs, `eras`, `gate_versions`, harness tables | Handled by the trainer redesign |

**A6 boundary, stated precisely:** audit *what the 35 equity / 11 crypto observation numbers are*,
how they are calculated, and how each consumer reads them. **Do not** audit fill accuracy, fees or
reconciliation — the moment a session reaches for `trades` or `fill_quality`, it has left scope.

---

## The three passes — globally ordered

| Sessions | Pass | Output |
|---|---|---|
| 1 → 12 | **Review**, one dataset per session | `01-MASTER-REVIEW.md` (spine) + `reviews/<ID>-<slug>.md` |
| — | 🚧 **gate: all 12 reviews complete** | |
| 13 → 24 | **Spec**, one dataset per session | `02-MASTER-SPEC.md` (spine) + `specs/<ID>-<slug>.md` |
| — | 🚧 **gate: all 12 specs complete** | |
| 25 → 36 | **Plan**, one dataset per session | `03-MASTER-PLAN.md` (spine) + `plans/<ID>-<slug>.md` |

**The passes are globally ordered, not per-dataset.** A1 does not get a spec until A8, B1, B2 and B3
have all been reviewed. There is never a session that reviews *and* specs.

Each pass's **spine** is append-only: one roster row and one block of carry-register rows per
dataset. The dataset's own document is written once, by the session that covers it (DS-9).

---

## Session rules

1. **One dataset. One pass. One session.** No exceptions.
2. **Both environments, every time.** Equity and crypto are covered together in the same session
   (parity principle). A divergence is only acceptable if it is *justified and recorded*, the way
   LD-1 records the Binance decision. An unexamined divergence is a defect.
3. **Review sessions contain no solutions.** Findings, evidence, gaps, and how the data is used —
   nothing else. Proposing a fix is scope creep.
4. **Verify, don't recall.** Every claim carries a `file:line`, a query, or a command output.
   Anything inferred is marked **UNVERIFIED** in the text.
5. **Confidence tracks evidence, not tone.** State it per claim where claims differ in strength.
   Do not average a verified mechanism together with an unverified consequence.
6. **The session is not done until the spine is updated.** Writing `reviews/<ID>-<slug>.md` is
   most of the work, but the roster row, the carry-register rows, any cross-dataset dependency
   edges and any new glossary terms go into `01-MASTER-REVIEW.md` in the same session. A dataset
   document that nothing indexes is invisible.
7. **Stop at the gate.** Once that is done, the session ends. Do not begin the next dataset and do
   not begin its spec.

---

## What every REVIEW session must cover

For the dataset in question, for **equity and crypto**:

### 1. The data itself
- What is actually stored, where, how much, over what range. Measured, not assumed.
- Value quality — not just recency. **Recency is not health:** a table whose newest row is today can
  still be entirely constant, which is exactly how 32 dead equity features survived.
- Distributions, distinct counts, degenerate or constant columns, NULLs, invariant violations.
- Known-bad or fabricated records.

### 2. Historical one-time ingestion, and its checks and balances
- Where the history came from, over what span, in how many vintages or sources.
- Any seams, source changes or backfill artifacts.
- What validated it at the time, what was quarantined, and what bypassed validation entirely.

### 3. Ongoing ingestion, and its checks and balances
- Which process owns it, on what schedule, from which source.
- How the watermark resolves, and whether it can see interior gaps.
- Which store is authoritative, and whether the stores can diverge.
- What validates each new record; what happens to rejects.
- What would detect a silent failure — and whether that detector can actually fire.

### 4. The calculation
- Every transformation between source and observation slot, with `file:line`.
- Whether the calculation is correct, duplicated, or dead.

### 5. The pipeline and its wiring
- The full path from source to consumer. Every reader, including dead ones.
- Where the trainer path and the trader path diverge — and whether that divergence is intended.

### 6. Use — current and planned
- How the trainer consumes it today. How the trader consumes it today.
- What the training redesign intends for it, per the user or the redesign documents.
- Which observation slots it occupies, and how many dimensions.

---

## ID conventions — binding

| Kind | Form | Rule |
|---|---|---|
| Finding | `<DS>-F<n>` — e.g. `A1-F17` | Dataset-scoped, assigned in discovery order |
| Carry item | `<DS>-C<n>` — e.g. `A1-C13` | Dataset-scoped, assigned in the order raised |

**Never renumber and never reuse an ID.** They are stable identifiers, not sort keys — a finding is
cross-referenced dozens of times, and renumbering to improve an ordering breaks every reference
while buying nothing a column cannot express. Severity, environment, disposition and the items a
finding feeds are all **columns**, so no ordering has to encode them.

Bare numbers (`item 5`) are forbidden: they collide the moment a second dataset is reviewed.

---

## The shape every review document must take

One opening section, the six content sections above, then **four** closing sections in this order:

| § | Section | Must contain |
|---|---|---|
| `<DS>.0` | **Disposition of carried-forward assumptions** | Every prior claim about this dataset, and its verdict: confirmed / corrected / refuted, with what is actually true. This is where the overriding rule is discharged — a review that skips it has not done the work |
| `<DS>.7` | **Findings index** | Every finding, with columns for ID, severity, environment, the section holding the evidence, and the carry items it feeds. **Grouped by disposition** — what a known, ruled-on change does to each: removed / untouched / made worse / created / is-the-risk. **If the dataset has no governing ruling, say so explicitly and group by severity instead** — do not invent a decision to sort against. Any grouping is **tested against the document** before adoption, and the test is recorded so it is not silently retried |
| `<DS>.8` | **What the disposition means** | The prose consequences the table cannot carry — inversions, caveats, what a disposition is conditional on |
| `<DS>.9` | **Carry register** | Every item, full statement, with its anchoring findings. **No remedies** |
| `<DS>.10` | **Dependency map** | Which items block which, including edges to other datasets. Prose chains are not sufficient — they must be tabulated |

Then: any forward notes to later datasets, the **evidence base**, a **confidence** statement listing
what is weaker than the rest, and the closing status line with finding and item counts.

[`reviews/A1-candles.md`](reviews/A1-candles.md) is the reference implementation. Copy its shape.

**After writing the dataset document**, append to the spine ([`01-MASTER-REVIEW.md`](01-MASTER-REVIEW.md)):
its roster row, its carry-register rows, any new cross-dataset dependency edges, and any new
glossary terms.

---

## Progress tracker

Update this table at the end of every session.

Order follows DS-8, not the A/B numbering.

| Order | # | Dataset | Review | Spec | Plan |
|---|---|---|---|---|---|
| 1 | A1 | Candles | ☑ 2026-07-25 — 37 findings, 18 carry items *(A1-F34…F37 amended 2026-07-29; "CBOE is unadjusted" wording swept 2026-07-30)* | ☐ | ☐ |
| 2 | B3 | Corporate actions | ☑ 2026-07-30 — **28 findings, 14 carry items** *(rebuilt from first principles; **all 28 walked with the user 2026-08-03 — all stand unamended**, B3-C13 and B3-C14 added)* | ☐ | ☐ |
| **3** | **A2** | **Derived features** | ☐ **next** | ☐ | ☐ |
| 4 | A3 | Macro | ☐ | ☐ | ☐ |
| 5 | A4 | Regime (HMM) | ☐ | ☐ | ☐ |
| 6 | A5 | Turbulence | ☐ | ☐ | ☐ |
| 7 | A6 | Portfolio state | ☐ | ☐ | ☐ |
| 8 | A7 | Fundamentals | ☐ | ☐ | ☐ |
| 9 | A8 | Sentiment | ☐ | ☐ | ☐ |
| 10 | B1 | Options chains | ☐ | ☐ | ☐ |
| 11 | B2 | Calendar events | ☐ | ☐ | ☐ |
| 12 | B4 | Observation events *(new, DS-10)* | ☐ | ☐ | ☐ |

**B3 is complete.** It was **re-run from first principles on 2026-07-30** because the first pass had
derived its coverage from a CBOE vendor snapshot rather than from the dataset itself — a scope
failure that cost four relocations and three splits. The rebuild kept every ID, added
**B3-F21 … B3-F32** and **B3-C8 … B3-C12**, corrected four first-pass claims (**B3-F4**, **B3-F5**,
**B3-F6 (h)**, **B3-F17**) and withdrew one mis-recorded "USER RULING" (**B3-C3**).
**B3-F10**, **B3-F14**, **B3-F15**, **B3-F16** stay retired; their IDs are never reused.

**Lesson recorded for every remaining review — see §B3.11.** The test for a finding is
***"is this a fact about this dataset?"*** — not *"did this session find it"*. Apply it when the
finding is numbered, not as a cleanup afterwards.

**Next session: A2 Derived features — review.**

---

## Scope decisions made 2026-07-25

| # | Decision |
|---|---|
| **DS-1** | Portfolio state is **IN**, as an observation input only. Fills, fees and reconciliation stay in Group C |
| **DS-2** | Candles and derived features are **separate datasets** — they fail in different ways |
| **DS-3** | Fundamentals and sentiment are **each their own dataset** (they have real external sources); overnight context folds into derived features (it is a hardcoded scalar with no source) |
| **DS-4** | Crypto options are **OPEN** — the B1 review documents equity capture and flags crypto as an open question; the decision happens at spec time |
| **DS-5** | Three passes × 11 datasets = **33 sessions**, globally ordered |
| **DS-6** | First review session is **A1 Candles** — everything else derives from it |
| **DS-7** | **Parity by default.** Every audit covers both environments. Divergence must be justified and recorded, like LD-1, or it is a defect |
| **DS-8** | **Review order changed: B3 Corporate actions is reviewed second, ahead of A2.** A1 established that `corporate_actions` holds **0 rows** and that three A1 carry items — the CBOE migration (**A1-C3**), the price basis (**A1-C13**) and point-in-time adjustment (**A1-C17**) — are blocked on it. Reviewing datasets in an order that leaves the hardest blocker until last would be arbitrary. Does **not** touch the gate: all 11 reviews still precede any spec |
| **DS-10** | **A 12th dataset — B4 Observation events — USER RULING 2026-07-29.** B3 established that "corporate actions" was silently carrying three jobs: what the **instrument** did (adjust), what the **venue or market** did (flag), and what **we** did (replay). Only the first is B3. Observation events had no home in the roster at all, which is why the 20 phantom CBOE bars on closed NYSE sessions (**A1-F10**) and the rolling XNYS calendar bound (written up as B3-F10 and relocated to B4 the same day) had no owner. B4 covers the **trading calendar** — the baseline of what should exist — together with **venue incidents**, kept in one dataset because neither is assessable without the other. Amends **DS-5**: 12 datasets × 3 passes = **36 sessions** |
| **DS-9** | **One document per dataset per pass.** Each pass has a **spine** (`0N-MASTER-*.md`) holding the roster, the carry register, the cross-dataset dependency map and the shared glossary; each dataset's full document lives beside it in `reviews/`, `specs/` or `plans/`. Reason: A1 alone is ~1,900 lines, so a single review file would reach ~10,000. Split at one dataset, when it cost nothing. Amends the earlier "one file, append-only" wording |

---

## Standing project rules that still bind

- **Never branch from `main`.** Branch from `swingrl/2.R-training-redesign`; PRs target it.
- **Plan-first** — no file created, edited or deleted without plan mode and explicit approval.
- **Additive-only migrations while the trader runs** (A30).
- **A30 deploy isolation** — service-scoped builds only; quiet window 15:30–16:45 ET on trading days.
- **Never call `CircuitBreaker.get_state()` or `get_capacity_fraction()` from a read path** — both
  write. Re-derive from SQL, and exclude audit rows:
  `COALESCE(reason, '') NOT LIKE 'stop-breach-audit:%'`.
- **DB access is read-only** unless explicitly approved:
  `docker exec pg16 psql -U swingrl -d swingrl -c "SELECT ..."`
- **The $400 equity / $47.09 crypto carve-out is inviolable.** Never read broker account value.
- **LD-1 holds: crypto stays on Binance.US.** Do not re-litigate.
- UTC internally; ET only at the display edge.
- Never `--no-verify`.
