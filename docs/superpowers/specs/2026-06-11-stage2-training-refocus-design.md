# Stage 2 — Training Refocus (Phase 19.1 Groups A–E) — Design Spec

> **Gate:** G1 (spec approval). Approved-in-conversation 2026-06-11 ET; this document is the
> written record for review.
> **Branch:** `swingrl/19.1-training-refocus` (cut from `main` post-PR-#17).
> **Tracker:** `.planning/V1.1_EXECUTION_PLAN.md` Stage 2.
> **Runbook detail:** `.planning/PHASE_19.1_HANDOFF.md` Groups A–E (this spec amends Group C;
> A/B/D/E execute as written there, with gates restated in §8).

---

## 1. Problem statement

Across iterations 3–4 (and the wiped iter 5), **control folds (no LLM advice) beat treatment
folds by 2.7–5.1× CPS v1**. The memory system — built to improve training — has been actively
damaging it.

**Observed failure mode (user-confirmed):** treatment folds went *trade-shy*. The LLM's
adjustments reduced trading activity, which suppressed profits and risk together, instead of
increasing profits while keeping drawdowns low.

**Root cause (verified in code, 2026-06-11):** the advice path has **no objective function**.
The string `cps` appears **zero** times in `epoch_callback.py`, `meta_orchestrator.py`, or
`query.py` (the 1,896-line prompt builder). The LLM is asked for advice with no definition of
success, no visibility into the score it is judged by, and no causal breakdown of past
performance — so it defaults to the locally "safe" action: damp risk. Under CPS v1's
multiplicative form, damping profit alongside risk collapses the score.

| Env | Iter | Treatment CPS v1 | Control CPS v1 | Control / Treatment |
|---|---|---|---|---|
| equity | 3 | 0.01233 | 0.03427 | 2.78× |
| equity | 4 | 0.01485 | 0.04071 | 2.74× |
| crypto | 3 | 0.08161 | 0.28397 | 3.48× |
| crypto | 4 | 0.08318 | 0.42063 | 5.06× |

---

## 2. Design decisions (locked at G1)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Spec scope = Group C code only; A/B/D/E are a gated runbook** (§8) | Only C is designable/TDD-able code. A=destructive DB ops, B=image rebuilds, D=operational QA loops, E=a training run. One clean spec→plan→TDD→PR cycle for the code; ops keep their runtime gates. |
| D2 | **C4 (base reward-weight rebalance) is DROPPED.** `DEFAULT_WEIGHTS` stays `{profit 0.50, sharpe 0.25, drawdown 0.15, turnover 0.10}` | The control fold — the empirical *winner* — runs on these exact weights, untouched (control's early-return at `epoch_callback.py:602` never calls `update_weights`). The harm is the LLM *moving* weights/HPs, not the base values. Changing the base would also change the control baseline, turning Group E into a two-variable experiment. `CHRONIC_FAILURE_WEIGHTS` is **not** added as a code constant; chronic-failure guidance lives in the C3 prompt (§5.3) so the LLM steers within existing clamps. |
| D3 | **CPS v1 multiplicative is the single objective.** No second objective; no CPS v4 | v1 already punishes the observed failure: in the formula, a return collapse (8%→2%) costs ~4× while the paired drawdown improvement (15%→5% MDD) earns only ~1.25× back. Trade-shyness cannot win under v1. Multi-objective framing would recreate the original "no clear driving force" failure. v2/v3 remain computed side-by-side as cross-checks (existing behavior). |
| D4 | **Three-layer metric hierarchy** (one scorecard, three time scales) — §4 | CPS is iteration-scale (post-fold); advice is epoch-scale (mid-fold). Without leading indicators the LLM is told to optimize a number it never sees while acting. |
| D5 | **Deterministic CPS diagnosis layer** (`diagnose_cps()`, pure function) — §5.2 | A low CPS has ≥4 distinct causes requiring *opposite* corrections (§4.1). Code derives the labeled diagnosis from per-fold stats we already store; the LLM only acts on it. Removes the hardest inference step from the least reliable component; fully TDD-able. |
| D6 | **Trade-activity rate added as a leading indicator — NOT a CPS term** | The one missing signal: CPS catches trade-shyness only indirectly (via return collapse) and late (post-fold). A rolling trades-per-window rate detects it mid-fold. Kept out of the objective to avoid Goodhart gaming (LLM pushing overtrading to pump a trade-count term; turnover costs). |

---

## 3. What already exists (verified) vs. what is new

**Exists — build on, don't reinvent:**

- `src/swingrl/metrics/cps.py` — three pure CPS formulas + `compute_all_cps()` returning a
  per-component breakdown; persisted per iteration.
- `backtest_results` per-fold columns: `total_trades`, `win_rate`, `profit_factor`,
  `max_single_loss`, `sharpe`, `sortino`, `mdd`, `total_return`, `overfitting_class` — the raw
  inputs for diagnosis (§5.2). The film exists; nobody shows it to the coach.
- `MemoryVecRewardWrapper` rolling metrics: `rolling_sharpe()`, `rolling_mdd()`,
  `rolling_win_rate()`, `rolling_mean_reward()` (`reward_wrapper.py:175–223`).
- Weight-adjustment safety clamps: `clamp_reward_weights` + max-delta scaling
  (`epoch_callback.py:680–705`, `src/swingrl/memory/training/bounds.py`). The over-steer
  limiter is already built; this spec does not change clamp values.
- Control/treatment split: `advice_enabled` / `is_control_fold` flags (`epoch_callback.py:109–135`).
- `detect_chronic_failures` / `detect_protected_winners` in `iteration_report.py` (reused by C2).

**New in this stage:**

- `src/swingrl/memory/training/fold_context.py` — fold role classification + fold history loader (C2).
- `src/swingrl/memory/training/cps_diagnosis.py` — the diagnosis pure function (C2, §5.2).
- Trade-activity tracking in `MemoryVecRewardWrapper` (C2, §5.2).
- Payload enrichment in `epoch_callback.py` / `meta_orchestrator.py` (C1, §5.1).
- Three prompt blocks in `query.py::_build_system_prompt` (C3, §5.3).
- `reward_adjustments` attribution columns (C5, §5.4).
- C0/C6 documentation bookends (§6).

**Discrepancy resolved:** the handoff's C4 cites `services/memory/memory_agents/bounds.py`
— that file no longer exists. Bounds live in `src/swingrl/memory/training/bounds.py`, with
`_FALLBACK_REWARD_BOUNDS` **duplicated** in `services/memory/memory_agents/query.py:85`.
The duplication is recorded here for Stage 3.4 (memory refactor) to resolve; this stage does
not consolidate it (out of scope, and `services/` is the deployed memory-service copy).

---

## 4. The metric architecture — one scorecard, three time scales

| Layer | Time scale | Metrics | Role |
|---|---|---|---|
| **North Star** | per iteration | **CPS v1 multiplicative** (v2/v3 computed alongside as cross-checks) | The single objective. The only "goal" the LLM is given. |
| **Leading indicators** | per epoch (mid-fold) | rolling Sharpe, rolling MDD, rolling win-rate *(existing)* + **trade-activity rate** *(new)* | The LLM's live dashboard; prompt maps each to its CPS consequence ("falling trade rate + falling reward trend = trade-shy collapse → your CPS is dying even though MDD looks good"). |
| **Attribution** | per advice event | `fold_cps_v1_before/after`, `advice_was_effective` (C5) | Measures the *advisor*. By iter 6 we know empirically which advice categories help. |

### 4.1 Diagnosis taxonomy

A low/falling CPS gets a **deterministic labeled cause** before the LLM sees it. Initial
taxonomy (detection thresholds finalized at plan time against iter 0–4 data):

| Label | Signature (per-fold stats) | Paired correction (C3 prompt) |
|---|---|---|
| `trade_shy` | trade count well below fold baseline; small losses *and* small profits | increase participation; do NOT damp risk further |
| `poor_selection` | trade count normal; win_rate / profit_factor degraded | tighten entry quality; reduce frequency, not size |
| `single_disaster` | aggregate stats healthy; `max_single_loss` breach dominates MDD | cap per-trade risk; leave frequency alone |
| `churning` | trade count elevated; turnover costs erode positive gross | reduce frequency; raise conviction threshold |
| `healthy` | no degradation signature | no adjustment warranted (distinct from the *fold-role* `protected_winner`, which is cross-iteration — §5.3 block 3 handles that) |

The same labels apply at two scopes: **post-fold** (from `backtest_results`, feeds
run-config advice and consolidation) and **mid-fold** (approximated from rolling indicators,
feeds epoch advice). The pure function is the single source of truth for both.

---

## 5. Component design (Group C, revised)

### 5.1 C1 — Payload enrichment

`epoch_callback.py::_query_epoch_advice` payload adds:
`fold_number`, `fold_role`, `fold_history` (last 6 iters, this fold), `hmm_regime`,
`vix_mean`, `chronic_failure_folds`, `protected_winner_folds`, `prev_iter_cps_v1`
*(per handoff C1)* — **plus** `leading_indicators` (the four §4 values incl. trade-activity
rate) and `cps_diagnosis` (label + supporting numbers from §5.2).

`meta_orchestrator.py::_query_run_config` payload adds:
`fold_role`, `prev_iterations` (last 3), `chronic_failure_folds`, `protected_winner_folds`,
`target_metric="cps_v1_multiplicative"`, explicit goal text *(per handoff C1)* — **plus**
per-fold `cps_diagnosis` labels from the previous iteration.

### 5.2 C2 — New helper modules

**`src/swingrl/memory/training/fold_context.py`** *(per handoff)*:
- `classify_fold_role(env, fold_number) -> Literal["chronic_failure", "protected_winner", "neutral"]`
- `load_fold_history(env, fold_number, n_iters=6)`
- Reuses `detect_chronic_failures` / `detect_protected_winners` from `iteration_report.py`.

**`src/swingrl/memory/training/cps_diagnosis.py`** *(new — D5)*:
- `diagnose_fold(fold: FoldMetrics, baseline: FoldBaseline) -> CpsDiagnosis` — post-fold scope.
- `diagnose_rolling(indicators: RollingIndicators, baseline: FoldBaseline) -> CpsDiagnosis` — mid-fold scope.
- `CpsDiagnosis` = TypedDict: `label` (§4.1), `evidence` (the numbers that fired),
  `confidence` ("clear" | "mixed" — mixed when signatures overlap).
- Pure functions: no DB access, no side effects, fully unit-testable. Baselines are passed in
  by callers (epoch_callback / meta_orchestrator own the I/O).

**Trade-activity tracking** *(new — D6)*: `MemoryVecRewardWrapper` gains a rolling
trades-per-window counter alongside the existing rolling metrics, exposed as
`rolling_trade_rate()`. Baseline definition (what "normal" trade rate is per env/fold) is an
**open item finalized at plan time** — candidate: the fold's own first-N-epochs rate, or the
env's iter 0–4 historical median (decided against real data during planning).

### 5.3 C3 — Prompt blocks (`query.py::_build_system_prompt`)

1. **Goal block** — "Your single objective is CPS v1 = median_return × (1−max_mdd)² ×
   tanh(mean_winner_sharpe/2) × pass_ratio. It is multiplicative: collapsing any factor —
   including profit — kills your score. Reducing trading to reduce risk LOWERS your score.
   Pass rate is NOT your goal."
2. **Anti-pattern block** — cites the actual iter 3–4 harm numbers (table in §1) and the
   trade-shy collapse mechanism; includes the **diagnosis→correction map** from §4.1 so advice
   is matched to labeled cause, not pattern-matched to "looks risky".
3. **Fold-protection block** — `protected_winner` → return baseline (no adjustments);
   `chronic_failure` → regime-conditional shaping may lean into drawdown *within existing
   clamp bounds* (this replaces the dropped `CHRONIC_FAILURE_WEIGHTS` constant, per D2).

### 5.4 C5 — Attribution schema

`reward_adjustments` gains: `fold_number`, `iteration_number`, `advice_id`,
`fold_cps_v1_before`, `fold_cps_v1_after`, `advice_was_effective` (nullable boolean —
populated post-fold). Wired in `epoch_callback` at advice time (identity columns) and
post-fold (outcome columns). Additive migration script under `scripts/migrations/`
(idempotent, same pattern as `add_cps_columns.py`).

### 5.5 C0 / C6 / C7 — unchanged from handoff

C0 baseline doc and C6 post-change doc per §6; C7 rebuilds `swingrl` + `swingrl-memory`
images after C lands (second rebuild after Group B's initial pass).

---

## 6. Documentation bookends

- **C0 (before any code):** `.planning/research/phase-19.1-prompt-baseline.md` — current
  prompt text, payload schemas, reward weights (all locations incl. the `query.py` /
  `bounds.py` duplication), and 3–5 representative iter 0–4 patterns annotated with
  harm hypotheses **now expressed in the §4.1 taxonomy** (e.g. "this pattern induces
  `trade_shy`"). These annotations become Group D's QA criteria.
- **C6 (after):** `.planning/research/phase-19.1-prompt-refocus.md` — mirror of C0 + diff
  summary with expected CPS impact per change.

---

## 7. Testing strategy

- **TDD throughout** (RED commit → GREEN commit, per CLAUDE.md):
  - `cps_diagnosis.py`: table-driven unit tests — one per taxonomy label per scope, plus
    `mixed`-confidence overlap cases and empty/None-field edge cases. Fixtures derived from
    real iter 0–4 fold rows (anonymized into conftest fixtures, no live DB).
  - `fold_context.py`: classification + history-loading tests against fixture data.
  - `rolling_trade_rate()`: wrapper unit tests alongside existing rolling-metric tests.
  - C1: payload schema tests (keys present, types, serializable).
  - C3: prompt assembly tests (blocks present, fold-role conditionality, no template holes).
  - C5: migration idempotency + column tests (DB-gated, skip without `DATABASE_URL`).
- Full suite green locally and on homelab CI (Stage 1 safety floor active) before PR.

---

## 8. Execution runbook (A → B → C → D → E)

Sequence and gates; task detail lives in the handoff.

| Step | What | Gate |
|---|---|---|
| **A** | Wipe iter-5 artifacts (~850k rows across 7 tables + model files); verify iter 0–4 untouched (10 `iteration_results` rows, 564 `backtest_results`, harm table reproduces) | 🛑 **Backup gate**: separate plan-mode approval + verified backup immediately before the DELETEs |
| **B** | Rebuild `swingrl`, `swingrl-memory`, `swingrl-dashboard` images; verify with the handoff's grep checks | ⚠ Not during a training run (container recreation kills scheduler + in-flight training) |
| **C** | This spec's code via TDD → homelab CI green (G3) → PR (G4) | Standard G2→G4 |
| **D** | Delete iter 0–4 consolidations (D1), then regenerate patterns one iteration at a time with QA per iteration. QA criteria now include: patterns reflect correct §4.1 diagnoses (not blanket risk-damping); iter 3–4 patterns must surface the harm evidence | 🛑 **Backup gate on D1**; halt→revise-C→restart loop if QA fails (2–3 trips budgeted) |
| **E** | Fresh iter-5 run from iter-4 baseline; verify pg16 writes, model auto-deploy; capture treatment-vs-control CPS | Success criterion below |

**Stage 2 acceptance:** Group E's treatment-vs-control CPS ratio collapses toward 1.0 (or
inverts) from the 2.7–5.1× baseline. If the harm persists, C+D get a revision pass — that
outcome is informative, not a process failure (§9).

---

## 9. Risks and known caveats

1. **Control is a strong baseline.** Even well-aimed advice may not beat *no advice*. Group E
   is designed to answer exactly this; budget allows a C→D revision loop. If treatment still
   loses after revision, the honest conclusion is that epoch-scale LLM advice doesn't add
   value for this system — and the attribution data (C5) will say so with evidence.
2. **CPS v1 negative-return quirk (verified):** when `median_return < 0`, a *larger* MDD
   shrinks the multiplier and makes the score *less negative* (ranks worse outcomes higher);
   an iteration with zero winner folds scores exactly 0.0 regardless of severity. Accepted:
   v2 cross-checks it, historical iterations are positive-territory. Documented so a disaster
   iteration scoring "0.0" surprises nobody.
3. **Diagnosis threshold risk:** mis-calibrated §4.1 signatures could mislabel causes and
   steer advice wrong. Mitigated: thresholds set against real iter 0–4 data at plan time;
   `mixed` confidence is a first-class output (the prompt tells the LLM to be conservative on
   `mixed`); Group D QA checks labels against the C0 annotations.
4. **Stage 3.4 overlap (accepted at re-sequencing):** 3.4 refactors the DB I/O boundary of
   the same memory files; C touches prompt text + payloads. Mostly disjoint hunks; the
   `bounds` duplication (§3) is explicitly left for 3.4.
5. **`services/` vs `src/` duplication:** prompt changes land in the deployed memory-service
   copy (`services/memory/memory_agents/query.py`); C7/B rebuilds make them live. Verification
   greps in the runbook confirm the running images carry the changes.

---

## 10. Out of scope (tracked elsewhere)

- F1 turbulence-column bug (Stage 4; blocks live trading, not training)
- F2 SAC epoch-cadence root cause; F3 `llm_audit_log` NULL backfill (Stage 4)
- `pg-test` server + SAVEPOINT rollback (Stage 3.0); server-side DB hardening (Stage 3.5)
- GHA Postgres service for the coverage check (issue #18)
- Base reward-weight experimentation (explicitly dropped per D2; if ever revisited, it is a
  separate single-variable experiment after Group E)

---

## 11. Open items for plan time (G2)

1. Trade-activity **baseline definition** (fold's own early-epoch rate vs. env historical
   median) — decide against iter 0–4 data.
2. §4.1 detection **thresholds** — derive from iter 0–4 `backtest_results` distributions.
3. Exact `fold_history` / `prev_iterations` payload shapes (field lists, token budget for the
   epoch-advice call — Cerebras/Groq context limits).
4. Whether `meta_orchestrator` consumes post-fold diagnoses from the DB or recomputes via the
   pure function (leaning: recompute — single source of truth, no schema addition).
