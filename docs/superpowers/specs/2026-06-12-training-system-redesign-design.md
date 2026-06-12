# Training-System Redesign (Stage 2.R) — Design Spec

> **Status: IN PROGRESS** — written incrementally; each topic section is locked during the
> Fable scoping sessions and committed as it closes. Pending sections are scope checklists,
> not approved content.
> **§1 (Goal): LOCKED 2026-06-12.** §2 (Topic 2) and §3 (Topic 3) pending.
> **Kickoff:** `.planning/REDESIGN_SCOPING_KICKOFF.md` ·
> **Tracker:** `.planning/V1.1_EXECUTION_PLAN.md` ▶ Stage 2.R
> **Gates:** topic-level approval in conversation → full-spec G1 sign-off when all topics close
> → implementation plan (G2) → targeted code-verification review → only then run/replace the
> Stage 2 runbook (A/B/D/E).

---

## §1 — Goal (LOCKED 2026-06-12)

### §1.1 What the training system is for

Produce deployable trading policies that grow capital without unrecoverable loss — scored per
iteration by **CPS v1 (multiplicative)**:

```
CPS v1 = median_return × (1 − max_mdd)² × tanh(mean_winner_sharpe / 2) × pass_ratio
```

(`max_mdd` = the single worst fold's drawdown; `pass_ratio` = fraction of folds passing the
per-fold validation gate. Implementation: `src/swingrl/metrics/cps.py`.)

### §1.2 The redesign's goal — five pillars

1. **Scoring is trustworthy.** CPS v1 is the single training objective; v2/v3 computed as
   cross-checks (v2 trusted only after the `max_single_loss` dollars→fraction fix). The score is
   **regime-blind** — a crisis drawdown costs full price. This is deliberate: under the
   multiplicative math, going defensive in a single crisis fold is cheap (median_return is robust
   to one ~0 fold; `(1−max_mdd)²` is protected) while systemic timidity is catastrophic — so the
   score already rewards the desired crisis behavior. **Validation gates are re-derived to align
   with CPS**: the current per-fold gate (`sharpe > 0.7, mdd < 0.15, profit_factor > 1.5,
   overfitting_gap < 0.20`) is pre-CPS, has no return or activity floor, and provably rewarded
   trade-shyness (iter 4–5: A2C returns fell 6.0% → 4.9% while pass rate *rose*); it leaks into
   CPS via `pass_ratio`. The ensemble gate (`sharpe > 1.0 AND |mdd| < 0.15`) gets the same review.
2. **The coach's authority is explicit and bounded.** The redesign enumerates the complete lever
   set — how many levers, which ones, what bounds, what cadence, what scope (per-epoch /
   per-fold / per-iteration). Every path by which the coach influences training is a named lever;
   any unenumerated influence path (e.g. the `_SAFE_DEFAULTS` cold-start weight shift,
   `services/memory/memory_agents/query.py:122–131`) is a bug. Each lever's blast radius is
   stated: what it can move, by how much, how fast.
3. **The coach earns its keep.** The meta-trainer is judged by **lever-attributable uplift over
   control**: one lever per iteration (per
   `.planning/research/reward-shaping-vs-hyperparameters.md` §5/§9 — simultaneous reward-weight +
   HP changes make attribution mathematically impossible); every retained lever cheaply proven to
   move training in the intended direction *before* any expensive re-run; **benching rule** — no
   demonstrated uplift → the coach is removed and control becomes production. The uplift judgment
   is **regime-aware**: control folds are fixed calendar windows (equity `[0,5,10,15,20]`, crypto
   `[0,4,9,13]`), so a disaster fold with no control-arm regime counterpart is flagged and
   excluded from the benching verdict, not silently counted (per-fold regime context exists:
   `backtest_results.hmm_p_bull/hmm_p_bear/vix_mean`).
4. **The coach sees the game — including the consequences of its own calls.** Every decision in
   the lever set has named information requirements, gaps identified and filled; no decision is
   asked of the coach without the data to make it. Every lever pull records a **falsifiable
   intent** (target metric + expected direction + horizon) at pull time; the realized effect is
   measured *against that intent* and the verdict — per-pull and as an aggregated per-lever track
   record — is fed back into the coach's context. No silent pulls, no intent-free verdicts. (The
   current `effective` flag — `sharpe_delta > 0 OR mdd_delta > 0` — is not intent-aware: a
   risk-reduction pull that increased risk but bumped Sharpe is marked "effective". This is the
   failure being designed out.)
5. **The record outlives the run.** Every captured record is structurally attributable to
   **iteration / run / fold / env / algo** via schema keys (never text parsing); units explicit;
   payloads machine-readable; volume bounded by fixed cadence/thresholds. The bar: a future coach
   can reconstruct the game — re-consolidation and trend analysis of any past iteration — from
   the record alone. The 4.9M-row unusable-corpus failure (zero structural keys; 83% of rows
   tagged `*:historical` with env/algo unrecoverable; undeclared units inside pattern text) must
   be structurally impossible to repeat.

### §1.3 Success criteria

| # | Criterion | Test |
|---|---|---|
| S1 | Treatment/control CPS v1 ratio ≥ 1.0 on the next full run | Extend the harm table (handoff §"empirical case"); regime-aware comparison |
| S2 | Treatment worst-fold MDD ≤ control's (+ agreed margin) | Per-iteration check against `backtest_results` |
| S3 | Every lever pull has recorded before/after attribution | No advice row without identity + outcome columns |
| S4 | Re-consolidation of a past iteration succeeds from captured data alone | Dry-run produces patterns with correct iteration/fold/env/algo |
| S5 | Gates re-derived and documented | Pass rate can no longer rise while returns fall |
| S6 | Each retained lever passes a cheap directional verification before the re-run | Lever-verification harness results on record |
| S7 | The lever inventory is exhaustive | Audit finds no training-influence path outside it |
| S8 | Every pull carries an intent and gets an intent-matched verdict; prompts include the per-lever track record | A risk-reduction pull cannot be marked effective by a Sharpe improvement alone |

### §1.4 Out of scope

- Live-trading bugs: F1 turbulence column (`execution/pipeline.py:537`); shadow-promotion
  directory mismatch (`models/active/{env}/` flat vs `{env}/{algo}/` expected) — recorded in §4,
  not fixed here.
- Stage 3 repository refactor (incl. the `bounds.py` / `query.py` `_FALLBACK_REWARD_BOUNDS`
  duplication — Stage 3.4).
- GHA coverage check (issue #18).
- CPS formula redesign; new RL algorithms.
- `DEFAULT_WEIGHTS` changes — unless Topic 2 explicitly reopens it as a single-variable
  controlled experiment.

### §1.5 Topic 1 decisions log

| # | Decision | Rationale |
|---|---|---|
| D-T1.1 | CPS v1 confirmed single objective (v2/v3 cross-checks; v2 post-units-fix) | Multiplicative form structurally punishes trade-shyness; spec D3 rationale survived re-examination; v2 currently corrupted by the dollar bug (equity v2 negative at iters 3–4 in live data) |
| D-T1.2 | "Maintain risk while increasing profit" encoded as a **guardrail/acceptance criterion** (S2), not a formula change | CPS prices risk steeply (~+27% return needed to pay for 10%→20% MDD) but has no ceiling; a hard ceiling in the score risks Goodhart cliff-edge games |
| D-T1.3 | Crisis handling: score regime-blind, coach evaluation regime-aware | Capital doesn't care why it was lost; but the coach must not be benched for crisis folds control never faced (fixed control fold indices = different calendar windows) |
| D-T1.4 | Gates misaligned with CPS; re-derivation **in scope** (criteria decided in Topic 2) | Empirical proof: pass rate rose while returns fell (iter 4–5); `pass_ratio` is a CPS factor, so gate misalignment corrupts the objective |
| D-T1.5 | Intent→outcome closure mandatory for every lever pull | Current `effective` flag is intent-blind; live rows show decision/action mismatch (see §1.6) |

### §1.6 Evidence base (verified during scoping, 2026-06-11/12)

- **Harm table:** control beat treatment 2.7–5.1× CPS across iters 3–4 (+ wiped iter 5);
  equity treatment max-MDD 0.36–0.38 vs control 0.067–0.069. (`PHASE_19.1_HANDOFF.md`, C0 §6.)
- **Gate rewarding harm:** iter 4–5 A2C returns 6.0% → 4.9% while pass rate rose (C6
  anti-pattern block).
- **Group C merged ≠ deployed:** the 6 `reward_adjustments` attribution columns do **not** exist
  in live pg16 (verified 2026-06-12); migration rides the container rebuild. The design must
  treat "merged" and "deployed/migrated" as separate states.
- **Lever bookkeeping contradicts action:** live `reward_adjustments` rows (ids 2, 3) have
  `trigger_reason` = "keeping baseline — insufficient evidence" yet `weight_after ≠
  weight_before`.
- **Provenance vacuum:** `memories` (4,958,918 rows) has only
  `id/text/source/created_at/archived` — no iteration/run/fold/env/algo columns; 83% of rows are
  `*:historical` with env/algo unrecoverable. `training_epochs` (850,430 rows) and
  `reward_adjustments` carry no iteration/fold columns; `run_id` is free text in ≥3 formats
  (`{env}_{algo}_fold{N}`, `…_fold{N}_CTRL`, `{env}_{algo}_{timestamp}Z`).
- **Iter-5 vestiges outside structured tables:** 1,835 `pattern_presentations` rows + 2
  `pattern_outcomes` rows tagged iteration 5 survive the Plan-A landscape.
- **Observability mismatch:** wrapper rolling metrics use a 500-step window; SAC's adjustment
  cooldown is 20,000 steps — the coach's view is 40× shorter than its minimum action interval.
- **Restriction precedent:** mid-fold reward lever already hard-disabled for PPO-crypto
  (`_MAX_REWARD_DELTA["ppo"]["crypto"] = 0.0`, `bounds.py:110`) after iter-4 analysis (~29.5%
  treatment underperformance).
- **Units chaos:** `max_single_loss` confirmed dollars (to −19,871) where CPS v2 expects a
  fraction; `reward_adjustments.trigger_value` mixes fraction-scale and multiple-scale values;
  JSON stored as `text` not `jsonb`; several date columns stored as `text`.

---

## §2 — Meta-trainer decision-set + observability (Topic 2 — PENDING)

Scope checklist derived from pillars 1–4. Topic 2 is not closed until every item has a decision:

- [ ] **Exhaustive lever enumeration** incl. currently unlisted influence paths:
      `_SAFE_DEFAULTS` cold-start weights, the epoch-advice `stop_training` flag, any fallback
      defaults applied on service failure. Close or formalize each. (S7)
- [ ] **Keep / restrict / remove / add** decision per lever, with blast radius (bounds, cadence,
      scope) per retained lever. (Pillar 2)
- [ ] **One-lever-per-iteration** attribution discipline at runtime — design the alternation.
      (Pillar 3)
- [ ] **Lever verification harness** — how each retained lever is cheaply proven directional
      *before* the iter-5 re-run. (S6)
- [ ] **Benching rule** operationalized — what evidence, over how many iterations, triggers
      removal. (Pillar 3)
- [ ] **Gate re-derivation** — new per-fold gate criteria (return/activity floor so silence
      can't pass) + ensemble gate review. (D-T1.4, S5)
- [ ] **Observability gap analysis** — inventory what the coach receives today (epoch payload,
      run-config payload, patterns, diagnosis) vs what each retained decision requires; fix the
      500-step-window vs cooldown mismatch class of problems. (Pillar 4)
- [ ] **Intent→outcome closure design** — pull-time intent record (metric + direction +
      horizon), intent-matched verdicts, per-lever track record fed back into prompts. (S8,
      D-T1.5)
- [ ] **Regime-aware uplift evaluation** — how disaster folds without control counterparts are
      flagged/excluded. (D-T1.3)

## §3 — Durable memory-capture data model (Topic 3 — PENDING)

Scope checklist derived from pillar 5 (+ Topic 2's information requirements):

- [ ] Per memory/record type: *what* it is, *when* captured, **exact schema** (fields, types,
      **units**, keys, provenance). Cover today's types (epoch memories, reward-adjustment
      outcomes, trading patterns, walk-forward results, run/iteration summaries, consolidations)
      and any new types Topic 2's decisions require.
- [ ] **Hard requirement:** every record structurally attributable to
      iteration / run / fold / env / algo (the "no iteration column" fix; replace free-text
      `run_id` parsing with keys).
- [ ] Corpus disposition: salvage vs discard the existing 4.9M rows (lean per tracker: discard;
      keep iter 0–4 *results* as evidence + diagnosis baselines).
- [ ] Volume bounds: capture cadence/thresholds set so the corpus stays consolidatable
      (root-cause F2 SAC cadence first — symptom was fixed, cause never found).
- [ ] Machine-readable payloads (jsonb not text; declared units; no narrative-only records).
- [ ] **Dry-run re-consolidation test** as the acceptance gate. (S4)

## §4 — Bug & finding catalogue (running; fix scope varies)

| Finding | Evidence | Fix scope |
|---|---|---|
| `max_single_loss` stored in dollars; breaks CPS v2 | live pg16 values to −19,871; equity v2 negative iters 3–4 | This redesign (Topic 3 units) |
| Group C attribution migration not applied to live pg16 | live `reward_adjustments` has 18 columns | Deployment step (rides container rebuild) |
| `reward_adjustments` decision/action mismatch ("keeping baseline" yet weights changed, ids 2–3) | live pg16 sample | This redesign (Topic 2 intent→outcome) |
| F2: SAC wrote 688k epoch memories (~350 expected); root cause unknown | `memories` source counts | Root-cause investigation feeds Topic 3 |
| Epoch logger used PPO-only SB3 keys | project memory (`project_epoch_logger_bug.md`) | This redesign (Topic 3 capture) |
| `_SAFE_DEFAULTS` ≠ `DEFAULT_WEIGHTS` (cold-start silently shifts weights) | C0 §4.2 | Topic 2 (unenumerated lever) |
| Rolling window 500 steps vs SAC cooldown 20,000 | docs/training inventory | Topic 2 (observability) |
| Risk penalty silently discarded when reward wrapper activates | `reward-shaping.md` known issue | Topic 2 (lever blast-radius) |
| `pattern_outcomes` missing UNIQUE(iteration, env_name); Phase B duplicate-pattern risk on retry | docs/training inventory | Topic 3 |
| `pattern_presentations.iteration` NULL for 4,933/9,575 rows | live pg16 | Topic 3 (provenance) |
| `training_epochs` has only a PK index (850k-row seq scans) | live pg16 | Topic 3 |
| JSON-as-text columns; `last_confirmed_at` + date columns as text; mixed timestamp/timestamptz | live pg16 | Topic 3 |
| F1 turbulence column queried but doesn't exist (silent 0.0) | handoff Group F | **Out of scope** (Stage 4; blocks live trading) |
| Shadow promotion writes flat `models/active/{env}/`; live trader reads `{env}/{algo}/` | `validation-promotion.md` | **Out of scope** (record only) |
| Live-trader model cache never invalidated (restart required) | `validation-promotion.md` | **Out of scope** (record only) |
