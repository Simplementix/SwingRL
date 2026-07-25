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

---

## Scope

**In scope: Groups A and B, for BOTH equity and crypto, for BOTH trainer and trader.**

### Group A — observation inputs

| ID | Dataset | Equity source | Crypto source | Notes |
|---|---|---|---|---|
| **A1** | Candles | CBOE *(proposed)* / Alpaca *(current)* | Binance.US + Binance Global archive | includes the XNYS session calendar as a validation dependency |
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
| **B3** | Corporate actions | source TBD — Alpaca returns **zero** spin-offs | n/a — record as a justified divergence | |

**11 datasets.**

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

| Sessions | Pass | Output document |
|---|---|---|
| 1 → 11 | **Review**, one dataset per session | `01-MASTER-REVIEW.md` |
| — | 🚧 **gate: all 11 reviews complete** | |
| 12 → 22 | **Spec**, one dataset per session | `02-MASTER-SPEC.md` |
| — | 🚧 **gate: all 11 specs complete** | |
| 23 → 33 | **Plan**, one dataset per session | `03-MASTER-PLAN.md` |

**The passes are globally ordered, not per-dataset.** A1 does not get a spec until A8, B1, B2 and B3
have all been reviewed. There is never a session that reviews *and* specs.

Each master document is **append-only** across its pass — one section per dataset, added by the
session that covers it.

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
6. **Stop at the gate.** When the dataset's review is written, the session ends. Do not begin the
   next dataset and do not begin its spec.

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

## Progress tracker

Update this table at the end of every session.

| # | Dataset | Review | Spec | Plan |
|---|---|---|---|---|
| A1 | Candles | ☐ | ☐ | ☐ |
| A2 | Derived features | ☐ | ☐ | ☐ |
| A3 | Macro | ☐ | ☐ | ☐ |
| A4 | Regime (HMM) | ☐ | ☐ | ☐ |
| A5 | Turbulence | ☐ | ☐ | ☐ |
| A6 | Portfolio state | ☐ | ☐ | ☐ |
| A7 | Fundamentals | ☐ | ☐ | ☐ |
| A8 | Sentiment | ☐ | ☐ | ☐ |
| B1 | Options chains | ☐ | ☐ | ☐ |
| B2 | Calendar events | ☐ | ☐ | ☐ |
| B3 | Corporate actions | ☐ | ☐ | ☐ |

**Next session: A1 Candles — review.**

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
