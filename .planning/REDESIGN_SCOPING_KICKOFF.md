# SwingRL Training-System Redesign — Scoping Session Kickoff (for Fable)

> ✅ SCOPING COMPLETE — Stage 2.R scoping is done (G1 signed 2026-07-07; Plan A + Plan B written & walkthrough-signed). Live status: .planning/V1.1_EXECUTION_PLAN.md → Stage 2.R. Retained for history.

> **Read this first, then drive a guided scoping conversation with me. This session
> produces a DESIGN DOCUMENT, not code.**

Created 2026-06-11 ET. Paste this as the opening prompt of the new Fable session (or point
Fable at this file). Companion tracker: [`V1.1_EXECUTION_PLAN.md`](V1.1_EXECUTION_PLAN.md)
▶ Stage 2.R.

---

## Your role this session

You are helping me redesign SwingRL's "meta-trainer" and its memory system *before* we
retrain. This is a **guided, conversational scoping session** — go **one topic at a time**,
ask me questions, give recommendations (not neutral menus), and **confirm alignment at the
end of each topic before moving to the next**. Do **not** write or edit code. The deliverable
is a **design document (spec + plan)**.

Style: plain English, lists/tables over prose, label **verified vs assumed** and give
**confidence levels** (high/medium/low), and check every proposal against (a) the Goal we set
in Topic 1 and (b) the already-merged Group C code (don't silently contradict or duplicate it).
Use the `superpowers:brainstorming` approach to run the conversation, and `writing-plans` for
the deliverable.

---

## Read before we start (canonical, in this order)

1. `.planning/V1.1_EXECUTION_PLAN.md` — esp. **Stage 2.R** and **▶ RESUME HERE**
2. `.planning/research/reward-shaping-vs-hyperparameters.md` — attribution / one-lever-per-iteration
   (see lines 128–132, 237, 256: changing reward-weights *and* HPs in the same iteration makes
   attribution **mathematically impossible**; it recommends **alternating one lever per iteration**)
3. `.planning/research/phase-19.1-prompt-baseline.md` (C0) — harmful patterns + data bugs
   (`max_single_loss` units, `outcome_sharpe`)
4. `.planning/research/phase-19.1-prompt-refocus.md` (C6) — current diagnosis taxonomy, levers, prompt blocks
5. `.planning/PHASE_19.1_HANDOFF.md` — empirical harm table + Groups A–E
6. `docs/superpowers/specs/2026-06-11-stage2-training-refocus-design.md` — current lever decisions
   (C4 dropped, fold protection, CPS goal)
7. `docs/training/*` — current capture + pipeline reference
8. Inspect the **live pg16 schema (read-only)** for the memory/epoch/consolidation/reward tables —
   note that `memories` and `training_epochs` have **no iteration column** today.

---

## Essential context

- **SwingRL** trains a PPO/A2C/SAC ensemble for swing trading on two environments (equity daily,
  crypto 4H). Capital preservation is the prime constraint.
- **The meta-trainer** is an LLM "coach" that influences training via three levers: (1) **mid-fold
  reward-weight adjustments**, (2) **between-iteration hyperparameter tuning**, (3) **consolidated
  patterns** fed into the next iteration's prompts.
- **The empirical failure:** across iterations 3–4, **control folds (no LLM advice) beat treatment
  folds by 2.7–5.1× on CPS** (Capital Preservation Score). The memory/advice system was actively
  *hurting* training.
- **What's already merged (Group C, PR #19):** the coach's *guardrails* were refined — a
  deterministic diagnosis→correction menu, fold-role protection (don't touch winning folds), an
  explicit CPS-v1 goal block, attribution columns. It did **not** change the lever set, did **not**
  verify the levers actually work, and the existing memory corpus is **pre-CPS and unreliable**
  (capture bugs + wrong objective).
- **Why we're redesigning** instead of re-running the old runbook: the runbook assumed
  re-consolidating the old corpus + re-running with better prompts would fix it. Investigation
  (2026-06-11) showed the corpus is pre-CPS, "iter-5" isn't cleanly selectable, and the levers were
  never verified. We fix the foundation first.
- **The governing metaphor (mine):** *the meta-trainer is the coach — but if he can't see the game
  properly, he can't give proper advice.* Information sufficiency is central. A coach with the wrong
  stats or a blurry view will coach badly no matter how good his judgment.

---

## Conversation agenda — three topics, in order

### Topic 1 — The detailed GOAL of the redesign (do this first; everything aligns to it)

Establish, in detail and in writing, what this training system is *for* and what "good" means —
before any mechanism talk.

- What is the meta-trainer ultimately optimizing, given capital preservation is the prime directive?
  Is **CPS v1** the right *single* objective — confirm it or revise it (and if revised, why, and what
  replaces it)?
- What does **success** look like, measurably? (e.g., treatment ≥ control on CPS; lever effects
  attributable; no tail blowups.)
- What is explicitly **out of scope** for this redesign?
- **Output:** a crisp, detailed goal statement we check every later decision against.

### Topic 2 — The meta-trainer as coach: decisions (levers) + observability (what it can see)

Two sides of one coin — scope them together. The coach's *decisions* determine what he needs to *see*.

- **Decisions / levers:** What impacts *should* the coach have? Review the current three levers and
  the candidates: adopt **one-lever-per-iteration** attribution (so we can tell which lever moved CPS);
  **restrict or remove the mid-fold reward-weight lever** (the harm source); **reopen C4** (base-weight
  rebalance); **add new levers** (position sizing, trade-frequency caps, regime vetoes, conviction
  thresholds). For each lever we keep: **how do we PROVE it moves training in the intended direction**,
  cheaply, *before* an expensive full re-run?
- **Observability / "can the coach see the game?":** Inventory exactly what information the meta-trainer
  receives today (epoch payloads, run-config payloads, consolidated patterns, the diagnosis). Then do a
  **gap analysis** — what's **missing or mis-measured**? Known issues to fold in: `max_single_loss`
  stored in dollars; `outcome_sharpe` was written wrong; trade-activity rate only added recently;
  attribution impossible when reward-weights and HPs move together; regime/context coverage; per-fold vs
  per-iteration visibility; the missing iteration key.
- **Output:** the coach's target **decision-set** + the **complete information requirements** to make
  each decision well, with every gap named.

### Topic 3 — Durable, reusable memory capture: types, timing, and the exact data model

Now turn the information requirements from Topic 2 into a durable capture design. The past failure was
that captured data was **incomplete/unusable** when it came time to re-consolidate — we must not repeat it.

- For **each memory type** we capture: *what* it is, *when* it's captured, and the **exact data points /
  schema** (fields, types, **units** — dollars vs fraction — keys, provenance). Cover today's types
  (epoch memories, reward-adjustment outcomes, trading patterns, walk-forward results, run/iteration
  summaries, consolidations) **and any new types** the goal/levers require.
- **Hard requirement:** every record must be unambiguously attributable to **iteration / run / fold /
  env / algo** (fixing the "no iteration column" problem) so future re-consolidation never lacks a field.
- Frame it as **durable/reusable capture**: capture enough, at the right granularity, that a future coach
  can reconstruct the game from the record alone.
- **Output:** a concrete data-model spec for memory capture.

---

## Deliverable

A **design document (spec + plan)** under `docs/superpowers/specs/` (+ a plan in
`docs/superpowers/plans/`) covering: the Goal, the meta-trainer decision-set + information model, and
the memory-capture data model. Write it **concretely enough to be verified against the existing code** —
name real files/functions/tables where known.

## The step immediately after this session

A **targeted code review** that verifies this design against the existing code with **high confidence** —
pinpointing every change-site and cataloguing existing bugs (e.g., F2 SAC epoch-cadence root cause,
`max_single_loss` units, `_SAFE_DEFAULTS` vs `DEFAULT_WEIGHTS` divergence). Design the doc so that review
can *confirm* it, not rewrite it.

---

## How to start

Read the files above, give me a 5–10 line **"here's my understanding"** back so we confirm we're on the
same page, then open **Topic 1**.
