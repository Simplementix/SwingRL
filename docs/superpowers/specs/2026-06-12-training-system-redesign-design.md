# Training-System Redesign (Stage 2.R) — Design Spec

> **Status: IN PROGRESS** — written incrementally; each topic section is locked during the
> Fable scoping sessions and committed as it closes. Pending sections are scope checklists,
> not approved content.
> **§1 (Goal): LOCKED 2026-06-12.** **§2 (Topic 2): LOCKED 2026-07-06.** **§3 (Topic 2.5,
> Meta-Trader): LOCKED 2026-07-06.** §4 (Topic 3) pending.
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
| S1 | Treatment/control CPS v1 ratio ≥ 1.0 on the next full run † | Extend the harm table (handoff §"empirical case"); regime-aware comparison |
| S2 | Treatment worst-fold MDD ≤ control's (+ agreed margin) | Per-iteration check against `backtest_results` |
| S3 | Every lever pull has recorded before/after attribution | No advice row without identity + outcome columns |
| S4 | Re-consolidation of a past iteration succeeds from captured data alone | Dry-run produces patterns with correct iteration/fold/env/algo |
| S5 | Gates re-derived and documented | Pass rate can no longer rise while returns fall |
| S6 | Each retained lever passes a cheap directional verification before the re-run | Lever-verification harness results on record |
| S7 | The lever inventory is exhaustive | Audit finds no training-influence path outside it |
| S8 | Every pull carries an intent and gets an intent-matched verdict; prompts include the per-lever track record | A risk-reduction pull cannot be marked effective by a Sharpe improvement alone |

† **Approved amendment (D-T2.5, 2026-07-06):** "control" in S1 means the **coach-free reference
season** (iteration 5 under baseline HPs + `DEFAULT_WEIGHTS`), compared season-over-season on
identical folds — not within-iteration control folds. Rationale in §2.5; the within-iteration
control-fold mechanism is retained dormant for L1 live re-earn only.

### §1.4 Out of scope

- Live-trading bugs: F1 turbulence column (`execution/pipeline.py:537`); shadow-promotion
  directory mismatch (`models/active/{env}/` flat vs `{env}/{algo}/` expected) — recorded in §5,
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

## §2 — Meta-trainer decision-set + observability (Topic 2 — LOCKED 2026-07-06)

Every §2 scope-checklist item is resolved by decisions D-T2.1–D-T2.11 (log: §2.9). The coach's
governing posture: **iteration-scale coach only** — no live mid-fold influence path — until
levers individually re-earn scope via §2.3 + §2.4.

### §2.1 Lever posture (D-T2.1)

| Lever | Decision | Blast radius when live |
|---|---|---|
| **L1** mid-fold reward-weight adjustment | **BENCHED** for all (env, algo) pairs: `_MAX_REWARD_DELTA` = 0.0 everywhere (config posture via the existing PPO-crypto mechanism, `bounds.py:110` — no code deletion). Runs in **shadow** (§2.4). Re-earn per (env, algo): §2.3 harness pass + §2.4 judgment track record | Per-epoch; component clamps + per-(env,algo) max-delta; step-based cooldowns (PPO 24,576 / A2C 500 / SAC 20,000) |
| **L2** between-iteration HP tuning | **RETAINED** — the coach's primary live lever, staged in per the §2.5 staircase | Once per (env, algo, iteration); double-clamped (service + trainer); every pull carries an intent record (§2.4) |
| **L3** consolidated patterns → prompts | **RETAINED** — corpus discarded/regenerated (disposition formalized in Topic 3). Structural note: with L1 benched, L3 has **no independent actuator** — its live channel is the L2 run-config prompt. Evaluated as the marginal value of patterns on L2 advice (§2.5) | Per iteration; top-3 per category, min confidence 0.4; patterns must pass C6 §7 QA (absorbed into §2.3 Stage 2) |
| New levers (position sizing, trade-frequency caps, regime vetoes, conviction thresholds) | **DEFERRED** → §2.10 future-levers appendix with pre-written verification requirements | — |

### §2.2 Unenumerated influence paths closed (D-T2.2, S7)

| Path | Decision |
|---|---|
| **U1** `stop_training` flag (`query.py:996–999`; actuated `epoch_callback.py:694–713`) | **Advice-only.** Runtime ignores the flag — folds always run to completion. A stop request is logged with full context and graded post-fold like any bet ("was the fold, in fact, wasted?"). Evidence: **0 stop requests in 850,430 live epochs** (pg16, verified 2026-06-15) — never-used power, no case for keeping actuation. Promotion to a real lever requires the §2.10 appendix path |
| **U2** `_SAFE_DEFAULTS` cold-start (`query.py:122–131`) | Cold-start/fallback weight sets **must equal `DEFAULT_WEIGHTS`** — the silent 4-weight shift is a bug, fixed in the implementation plan |
| **U3** service-failure fallback behaviors | **Audit item** for the targeted code-verification review: enumerate every fallback default; each must be identity with baseline (no influence) or become a named lever |

After these closures the lever inventory of §2.1 is exhaustive (S7); any further influence path
found in the code review is a defect against this section.

### §2.3 Lever-verification harness (D-T2.3, S6)

Two stages; a lever goes live only after passing both. Harness runs are standalone (reuse
trainer + data + folds; models discarded); every harness record carries a run-type quarantine
tag and is **never consolidated** into patterns/memories (schema in Topic 3).

- **Stage 1 — mechanics, no LLM.** Scripted pull of known direction/magnitude on one fold,
  shortened run: **3 seeds with the pull vs 3 same-seed controls; majority of seed-pairs must
  agree with the intended direction** without collapsing trade activity. Minimum run length per
  (algo, lever): ≥ cooldown + one trend-window measurement period (table of exact lengths in the
  implementation plan). The 6 runs are independent → run in parallel.
- **Stage 2 — judgment, no training.** Replay recorded fold situations to the coach with
  **production-identical prompts** (including regenerated patterns); grade against the
  deterministic diagnosis→correction menu (`cps_diagnosis.py`) and fold-protection rules.
  Absorbs the C6 §7 Group-D pattern-QA criteria — a Stage-2 failure indicts prompt or pattern.
- **Fold selection:** a **neutral fold is the gate**. The chronic-failure tryout is additionally
  required when (a) the lever is fold-targeted and its permitted scope includes
  `chronic_failure` folds (true for any L1 re-earn — the fold-protection blocks steer reward
  shaping there), or (b) at runtime, a live lever's failed bets cluster on chronic-failure folds
  (retroactive trigger → pass it or lose that scope).

### §2.4 Shadow mode + intent records (D-T2.4, D-T2.8; S8, D-T1.5)

Epoch advice keeps running with production prompts but is **log-only**: nothing the coach says
mid-fold changes the run. Every lever call — shadow L1, U1 stop requests, and **live L2 pulls
alike** — writes a five-block **intent record**:

1. **Identity** — iteration / run / fold / env / algo / epoch / timestep + %-complete /
   advice_id / mode (`shadow`|`live`) / lever.
2. **Evidence snapshot** — both observation windows' metrics (§2.6), diagnosis + confidence,
   fold role, current weights/HPs. Self-contained: graders never join other tables.
3. **Proposal** — the change (or explicit no-change) + rationale.
4. **Falsifiable bet** — target metric + expected direction + horizon. **Horizons are
   system-fixed, never coach-chosen** (aggregability; no gaming): mid-fold = one trend window
   after the call; L2 = one season, verdict via same-fold season-over-season comparison.
5. **Verdict** (deterministic, post-horizon) — actual value, direction-match, plus an immediate
   menu-consistency check. Replaces the intent-blind `effective` flag (D-T1.5).

**Grading semantics (honest limits):** shadow verdicts grade **judgment** — was the diagnosis
right, did the prognosis materialize absent intervention — not treatment effect (the action was
never applied). Treatment effect is Stage 1's job. **Re-earn = medicine proven (harness) +
judgment proven (shadow track record).** Prognosis grading needs no control arm: the bet is
about the fold's own future. Aggregated per-lever/per-scope track records are fed into the
coach's prompts each iteration (S8).

### §2.5 Attribution model + staircase schedule (D-T2.5, D-T2.6)

**Verified finding that reshaped this section:** control folds were only ever shielded from
epoch advice (`backtest.py:384`); **advised HPs reached all folds including controls**
(`backtest.py:393`, unconditional). Consequences: (a) the 2.7–5.1× harm indicts the **mid-fold
lever alone** — both arms shared HPs; (b) with L1 benched, within-iteration arms would differ
by nothing live.

- **Active control arm dropped.** Its question ("does mid-fold advice help?") is answered.
  The mechanism (yaml `control_folds_*`) is **retained dormant** solely for L1 live re-earn.
- **Reference season:** iteration 5 runs coach-free — baseline HPs, `DEFAULT_WEIGHTS`, shadow
  coach mic'd up with full production prompts. This is the bar every lever season must beat.
- **Season-over-season comparison on identical folds:** fold boundaries are deterministic and
  calendar-stable (`backtest.py:202` — new data only appends folds), so iter-N fold-k vs
  iter-M fold-k is a same-regime comparison; the D-T1.3 regime confound dissolves for lever
  evaluation. **Per-fold seeds pinned across iterations** (feasibility → code review) so the
  lever is the only variable.
- **Staircase (one change per season):** iter 5 reference → iter 6 **L2 bare** (HP advice,
  patterns withheld from the prompt) → iter 7 **L2 + regenerated patterns** (measures L3's
  marginal value) → iter 8+ decided by the track record and the §2.7 ladder.
- **S1 amendment** recorded (see §1.3 footnote).

### §2.6 Observability (D-T2.7, D-T2.11; pillar 4)

- **Two observation windows per algo, defined in percent-of-fold** (measurable today:
  `num_timesteps / _total_timesteps`, cf. `epoch_callback.py:698`): a **short window**
  (~0.5–1%) as acute-change detector (keeps the trade-shy alarm live mid-cooldown) and a
  **trend window** (~12–15%, sized to cover the longest cooldown at current fold budgets) as
  the decision basis. Fixes the 500-step-window vs 20,000-step-cooldown mismatch (40×).
- **Cooldowns stay step-based** — they protect learner adaptation (SAC replay-buffer turnover,
  PPO rollout cycles), which runs on absolute steps, not fold fractions.
- **Startup guard:** at config load, assert trend-window-steps ≥ longest cooldown; refuse to
  start otherwise. Makes the eyesight-mismatch bug class structurally unrepeatable.
- **Dual-unit capture:** every windowed value recorded with both percent and absolute steps,
  units labeled (the undeclared-units corpus failure must be impossible to repeat).
- **New capture requirements surfaced by the decision audit** (→ Topic 3): L2's own
  settings-history-with-outcomes (today the run-config coach picks HPs blind to his past
  picks); **gate version + era tag + pinned seed on every season and fold record**.
- **Regime residues:** newly-appended folds are excluded from lever verdicts until played in
  two eras (counted in standings, silent in attribution); the D-T1.3 disaster-fold rule applies
  as written if control folds ever reactivate.

### §2.7 Benching ladder (D-T2.9; pillar 3)

| Level | Trigger | Response |
|---|---|---|
| 0 Noise | Single wrong bet | Record only — no reaction to single calls |
| 1 Weak spot | One scope (diagnosis label, or (env, algo)) below threshold over the minimum sample | **Scoped demotion to shadow** (the PPO-crypto precedent as a standing rule); rest of the coach unaffected |
| 2 Broad failure | A lever below threshold across scopes | Lever benched → shadow; re-earn path required |
| 3 Coach removed | No lever shows uplift over the reference season across the agreed season count | **No-coach baseline becomes production** (pillar 3's guarantee) |

Principles: **deterministic grading** — verdicts, aggregation and ladder decisions are computed
by scripts from intent records; the LLM never grades itself. **Minimum sample:** ≥10 graded
bets per scope before any ladder verdict. **Demotion threshold:** directional accuracy not
credibly better than coin flip (~<60% — exact numbers are spec *proposals*, finalized against
per-season bet volumes in the implementation plan). **Demotions are recoverable** via the same
re-earn path as L1.

### §2.8 Gate re-derivation (D-T2.10; S5, D-T1.4)

The per-fold gate (`sharpe > 0.7, mdd < 0.15, profit_factor > 1.5, overfitting_gap < 0.20`) is
replaced by a win condition that cannot mark an unplayed game a win (iter 4–5 proof: A2C
returns 6.0%→4.9% while pass rate rose):

| Requirement | Basis |
|---|---|
| **Profit floor** (NEW) | Fold return above a small positive floor |
| **Activity floor** (NEW) | Trade count above a per-(env, algo) minimum from `TRADE_BASELINES` (`cps_diagnosis.py`) |
| Risk cap | Per-env MDD ceiling (equity ≠ crypto weather) |
| Quality bar | Sharpe / profit-factor minimums, re-checked |
| Overfit cap | In-sample vs OOS gap ceiling, kept |

**Derivation + acceptance:** thresholds derived from the 564 iter-0–4 `backtest_results` rows
and validated by **offline replay** before season 5. Acceptance: (a) the fake-win anti-pattern
disappears (returns fall ⇒ pass rate falls), (b) historical pass rate lands in a sane band
(~40–70% proposed), (c) passing correlates with fold-level CPS contribution. Ensemble gate
(`sharpe > 1.0 AND |mdd| < 0.15`): same method.

**Loop safety:** the gate is a **floor, not a target** — sanity minimums; excellence is priced
by CPS's continuous factors; the coach's prompts state CPS as the sole objective and never
present the gate as a target. **No self-tuning:** live pass rate is monitored against an alarm
band; out-of-band triggers a *human-approved* review only. Every gate change is
**version-stamped and starts a new era; cross-era CPS values are never compared** (this also
covers old-era iters 0–4 vs the new gate regime).

### §2.9 Topic 2 decisions log

| # | Decision | Rationale |
|---|---|---|
| D-T2.1 | L1 benched everywhere (config max-delta 0.0); coach iteration-scale only; L2+L3 retained; new levers deferred | Only lever with direct harm evidence (2.7–5.1×, tail MDD 0.36–0.38 vs 0.067–0.069); eyesight 40× too short for its action interval; bookkeeping contradicted action (ids 2–3); pillar 3 = prove first |
| D-T2.2 | U1 advice-only; U2 must return `DEFAULT_WEIGHTS`; U3 audited | Mid-fold halt is a mid-fold lever (posture consistency); 0 uses in 850k epochs; logging preserves the signal at zero risk |
| D-T2.3 | Two-stage harness (mechanics w/o coach; judgment w/o training); 3 seeds majority; neutral gate + two chronic triggers; quarantined | Separates "tool broken" from "operator wrong" — the old system could never tell; RL noise demands seed replication |
| D-T2.4 | Shadow mode for mid-fold advice | Free judgment evidence for re-earn; trade-shy detector keeps an audience; zero influence |
| D-T2.5 | Reference-season attribution; same-fold cross-iteration comparison; pinned seeds; control mechanism dormant; S1 amended | Control arm's question is answered; HP leak (`backtest.py:393`) made arms meaningless post-benching; stable fold boundaries make same-fold comparison regime-clean |
| D-T2.6 | Staircase: iter 5 reference → 6 L2-bare → 7 L2+patterns | One variable per step; L3 has no actuator without L2; bare-first finds a harmful L2 one season sooner |
| D-T2.7 | Percent windows (short + trend), step cooldowns, startup guard, dual-unit capture | Windows serve the coach (proportions); cooldowns serve learner physics (absolute steps); guard makes the mismatch class unrepeatable |
| D-T2.8 | Five-block intent record for every pull; system-fixed horizons | Falsifiable bets or no call; fixed horizons keep track records aggregable and ungameable; kills the intent-blind `effective` flag |
| D-T2.9 | Four-level scoped ladder; deterministic grading; ≥10-bet minimum; recoverable | Punishment lands where evidence is; no self-grading; no verdicts on noise; ladder runs both directions |
| D-T2.10 | Gates: profit+activity floors, per-env caps, replay-derived, floor-not-target, no self-tuning, versioned eras | Gate leaks into CPS via pass_ratio; floors kill the fake-win channel; self-tuning gates destroy season comparability |
| D-T2.11 | Observability closeouts: L2 history, version/era/seed stamps; regime residue rules | Last gaps from the per-decision information audit (pillar 4) |

### §2.10 Future-levers appendix (deferred; D-T2.1)

Each candidate is **not designed** here; listed with the verification it must pass before any
design work (all: §2.3 two-stage harness + §2.4 intent records + §2.7 ladder from day one):

| Candidate | Pre-registered directional test (Stage 1 sketch) |
|---|---|
| Position sizing | Scripted size cap on one fold: MDD falls without return collapse beyond CPS-neutral |
| Trade-frequency caps | Scripted cap on a `churning` fold: profit factor rises, trade count falls to band |
| Regime vetoes | Scripted veto on a disaster fold: worst-fold MDD falls; healthy folds untouched |
| Conviction thresholds | Scripted threshold raise on a `poor_selection` fold: win rate rises, activity floor still met |

### §2.11 Hand-offs

- **→ Topic 3 (data model):** intent-record schema (5 blocks); dual-unit fields; gate
  version/era/seed stamps on season+fold records; harness run-type quarantine tag; L2
  settings-history capture; structural identity keys on every record type.
- **→ Code-verification review:** U3 fallback enumeration; per-fold seed-pinning feasibility;
  startup-guard placement; gate-replay SQL against the 564 rows; `_MAX_REWARD_DELTA` config
  surface for the all-pairs bench; exact minimum harness run lengths per (algo, lever).

## §3 — Meta-Trader: mission, boundaries, and data requirements (Topic 2.5 — LOCKED 2026-07-06)

A **second LLM coach for trade time** (paper + live), distinct from the §2 meta-trainer: the
meta-trainer develops players between seasons; the Meta-Trader manages them during real games.
This section is a **bounded charter** — mission, boundaries, candidate levers (listed, not
authorized), the shared weakness-profile asset, and capture requirements. The full design
(lever mechanics, authority thresholds, cadence, prompts) is **deferred to its own spec** (§3.8).

### §3.1 Mission (D-MT.1)

> The Meta-Trader is a game-day defender: it watches live and paper trading, maps sensor
> readings and scheduled events onto each player's documented weaknesses, and — only with
> earned authority — reduces a player's influence or the team's exposure before damage,
> never adding risk. It detects nothing itself, trades nothing itself, and can never
> override the referee.

Direction is **reduce-only**: every lever it could ever hold can only shrink exposure or
influence, so its worst wrong call costs upside, never unrecoverable capital. Symmetric/boosting
management is excluded from the mission; any future upgrade requires a §2.10-style appendix
path with pre-registered verification.

The structural gap it fills (true regardless of doc/code drift): between trainings, nothing
judges live form — ensemble weights are set once per training from WF Sharpe
(`pipeline_helpers.py:219–256`) and read per cycle (`execution/pipeline.py:410–445`); the only
in-game protections are deterministic tripwires that fire *after* damage. The Meta-Trader is
the judgment layer on the middle timescale: faster than retraining, earlier than the breakers.

### §3.2 Division of labor (D-MT.2)

| Condition | Detected by | Response |
|---|---|---|
| Crash (vol spike, loss breach) | Circuit breakers (deterministic, existing) | Breakers halt — the Meta-Trader is never in this loop |
| Regime shift | Statistical sensors (HMM p_bull/p_bear, VIX features — existing) | Meta-Trader *interprets* against weakness profiles |
| Scheduled events (FOMC, CPI, earnings) | Script + calendar feed (new plumbing, no LLM) | Meta-Trader *interprets significance* |
| Rule violations | Risk layer + broker middleware | Deterministic veto — the Meta-Trader always sits behind it |

The Meta-Trader's **only unique job is interpretation**: does the current picture — sensor
readings + upcoming events + recent per-player behavior — match a documented weakness of a
specific (algo, env), and is reducing that player's influence warranted *before* the tripwires
fire? It never detects (sensors are faster and better at that) and never trades (the players
are the traders).

### §3.3 Day-one duties — no authority (D-MT.3)

Active from paper trading onward; all outputs advisory and graded:

1. **Commentator** — periodic structured judgment on live/paper telemetry: diagnosis, matched
   weakness signature + confidence, proposal, falsifiable bet. Log-only; written as §2.4
   five-block intent records, mode=`shadow`, system-fixed horizons, deterministic grading.
2. **Alarm-raiser** — escalate-to-human (Discord) when a weakness signature fires or an
   unscheduled event warrants a look. Advisory only; escalations are themselves graded calls.
3. **Event-significance interpretation** — weighs upcoming *scheduled* events in context;
   output feeds its own commentary/alarms only.

### §3.4 Candidate lever set (LISTED, NOT AUTHORIZED; D-MT.4)

All reduce-only. Each goes live only after its own §2.3-style harness pass + shadow track
record + §2.7 ladder standing — pre-registered in the future Meta-Trader spec:

| Candidate | Action | Natural surface |
|---|---|---|
| Ensemble tilt (down) | Reduce a flagged player's blend weight, bounded | `model_metadata` ensemble weights read per cycle (`execution/pipeline.py:410–445`) |
| Position-size scaling (down) | Shrink sizes when a flagged player drove the decision, or env-wide | Sizing path in `execution/` |
| Live benching | Player weight → 0, time-boxed, recoverable | Same weight surface |
| Per-algo trade veto | Block trades where the flagged player was the driver | Pre-middleware check |
| Pre-event de-risk | Reduce/veto new entries in a defined window around scheduled high-impact events | Calendar-triggered; highly gradeable (each CPI/FOMC print is a repeated natural experiment) |

Flagged unknown (honest gap): whether pre-event de-risking is net-positive for a swing
strategy at all — plenty of setups profit from post-event moves. Answered by its shadow track
record before any authority question arises.

### §3.5 Prohibitions / hard caps (D-MT.5)

- **No play-calling** — per-order approve/modify is being a trader, not a coach.
- **No posture-switching** — team-wide defensive/cash calls duplicate the circuit breakers in
  the most expensive seat; the escalation path covers the genuine cases.
- **No autonomous action on unscheduled/breaking news, ever** — news text is untrusted input
  to a system with trade authority (hallucination + injection risk); escalate-only, permanently.
- **No training-side influence** — not a side door for new training levers or unbenching L1
  (§2.10 remains the only path).
- **No LLM-to-LLM channel** with the meta-trainer (§3.6).

### §3.6 Shared weakness profiles (D-MT.6)

One **script-maintained** asset per (algo, env): failure mode → data signature → early
detectability → evidence rows. Seeded from existing training-time knowledge
(`.planning/research/hp-tuning-reference.md` per-algo diagnostics; iter-1 forensics and
per-algo reward sensitivity in `reward-shaping-vs-hyperparameters.md` §6/§10); enriched by
structured live records as they accumulate. **Both coaches read it; neither writes it
directly** — scripts and consolidation maintain it from graded records. This replaces the
earlier "Meta-Trader feeds Meta-Trainer" sketch: there is no coach-to-coach advice channel,
only a shared, attributable evidence base.

### §3.7 Capture requirements → Topic 3 (D-MT.7)

Start accumulating at paper trading (cold-start avoidance — the time-critical part of this topic):

1. **Per-cycle per-player record**: each algo's proposed action vs the blended action vs
   actual fills — today only the blend is visible, so live per-algo behavior is unattributable.
2. **Regime context stamped on trade records** (HMM probabilities, VIX at decision time).
3. **Event-calendar ingestion + event-stamping** on trades, verdicts, and grading windows —
   event shocks must not silently contaminate weakness attribution (the D-T1.3 disaster-fold
   exclusion applied at trade time: don't grade a player on a game played in a hurricane).
4. **Meta-Trader intent records** — §2.4 five-block format, system-fixed horizons,
   mode=`shadow` from day one.
5. **Slippage / fill-quality vs backtest expectation per algo** — the train-vs-live transfer
   signal (e.g. does a model's backtest trade-rate survive contact with live markets).

### §3.8 Governance preconditions + deferral (D-MT.8)

The §2 template applies wholesale as a precondition: shadow-first; deterministic grading (the
LLM never grades itself); ≥10 graded bets per scope before any ladder verdict; the §2.7
benching ladder in both directions; always behind broker middleware + the risk-veto layer
(CLAUDE.md critical rule). **Full lever design is deferred to a dedicated Meta-Trader spec**,
gated on (a) the §3.7 capture data existing and (b) the §2 machinery operating in code —
designing levers before the signatures exist would repeat the original meta-trainer's sin
(levers never verified).

### §3.9 Topic 2.5 decisions log

| # | Decision | Rationale |
|---|---|---|
| D-MT.1 | Mission = reduce-only game-day defender; boosting excluded | Worst wrong call = missed upside (recoverable) — matches capital preservation + the 2.7–5.1× humility prior; upgrade only via appendix path |
| D-MT.2 | Detect/interpret split: breakers own crashes, sensors own regimes, calendar is plumbing; the LLM interprets only | Deterministic layers are faster and more reliable at detection; interpretation against weakness profiles is the one judgment no threshold expresses |
| D-MT.3 | Day-one duties advisory-only (commentator / alarm-raiser / event significance), intent-recorded from day one | Free evidence at zero risk; builds the graded track record any authority must be earned from |
| D-MT.4 | Five candidate levers listed, none authorized; all reduce-only | Player-level levers match the weakness-compensation intent; each must individually earn scope (§2 pattern) |
| D-MT.5 | Hard caps: no play-calling, no posture-switching, no autonomous news action, no training-side influence, no LLM-to-LLM channel | Keeps the Meta-Trader a coach not a trader; keeps untrusted input away from trade authority; keeps §2's governance closed |
| D-MT.6 | Weakness profiles = one shared script-maintained asset consumed by both coaches | Replaces an ungradeable coach-to-coach channel with an attributable evidence base |
| D-MT.7 | Five capture requirements start at paper trading | The signatures the whole design depends on don't exist yet; capture is the only time-critical piece |
| D-MT.8 | Full design deferred to its own spec, gated on capture data + §2 machinery operating | Designing levers without signatures = the original unverified-lever mistake |

### §3.10 Hand-offs

- **→ Topic 3 (data model):** the five §3.7 capture requirements as first-class record types
  (per-player cycle records, regime/event stamps, trade-time intent records, fill-quality
  records); the event calendar as a new ingested data source; all under the same
  structural-identity-key regime as §2.11.
- **→ Future Meta-Trader spec:** lever mechanics + bounds, authority-ladder thresholds,
  cadence, prompt design, harness adaptation for trade time (what "3 seeds" becomes when the
  fold is a live window).

## §4 — Durable memory-capture data model (Topic 3 — PENDING)

Scope checklist derived from pillar 5 (+ Topic 2's §2.11 hand-offs + Topic 2.5's §3.7
capture requirements):

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

## §5 — Bug & finding catalogue (running; fix scope varies)

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
