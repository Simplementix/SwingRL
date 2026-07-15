# Plan B — Training-System Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

> **Status: WALKTHROUGH-REVIEWED (2026-07-12) — all 9 sections + N-table signed off by
> the user. In-walkthrough amendments: P-B1 verified (equity 1.0M / crypto 500k fold
> budgets; N5 split per env); Task 5 MDD records both worst + mean sub-env (worst drives
> triggers); approval-gates table G-1–G-9 added; Task 0 Step 5 pre-flight read of the two
> unread large files; Discord routing pinned (all Plan B alarms from swingrl-container
> scripts) + training-side delivery proof (`--test-alert`, Task 29). AMENDED 2026-07-12
> (A30): deploy-isolation constraint added — all Plan B deploys via the trainer service,
> additive-only migrations while the trader runs (Plan A Task E owns the compose split).**
> Companion: Plan A (`2026-07-07-capture-foundation-plan.md`) — the paper-trading capture
> foundation; this plan owns everything training-side.
> Spec: `docs/superpowers/specs/2026-06-12-training-system-redesign-design.md`
> (G1 signed off; amendments A1–A29).
> Review inputs: `docs/superpowers/reviews/2026-07-07-execution-path-code-review.md` §6
> (training-side verdicts) — all file:line references below were verified by that review
> at `2d5b6f7` and re-confirmed by direct reads on 2026-07-11 at `661a2ad` (docs-only
> commits in between; code unchanged).

**Goal:** Rebuild the training side of SwingRL so that iteration 5 (the coach-free
reference season) can run on the new schema with honest rewards, bounded capture,
deterministic grading, derived gates, and a verified cutover — every coach call a
falsifiable bet, every result reconstructable from the record alone (S4).

**Architecture:** Phase 1 fixes the training path's live defects (risk penalty discarded,
cold-start weight drift, stop-flag actuation, hardcoded lever limits, broken observation
windows, F2 volume bomb, unpinned seeds) — these are preconditions, not features. Phase 2
ships the remaining §4 tables as V005–V008 migrations on Plan A's runner. Phase 3 rewires
every training-side writer onto the new schema through the identity spine. Phases 4–6
build the evidence engine (derived gates, script graders, benching ladder, two-stage
harness, consolidation v2). Phase 7 defines the era-1 training environment (A28). Phase 8
proves it (S4 CI tier), protects it (dumps + drills), and cuts over (REVOKE + gated
archive-and-drop).

**Tech Stack:** Python 3.11, psycopg/psycopg_pool (Postgres 16), Stable Baselines 3
2.7.1, structlog, pydantic v2 config, pytest. Training runs only inside the `swingrl`
container on homelab (standing rule).

## Glossary — read this first

Every shorthand used in this plan, defined once (per the project's plain-English rule).
Coach/players/seasons analogy terms are tied to their concrete table or action.

| Term | Meaning |
|---|---|
| **Coach / meta-trainer** | The between-seasons LLM that proposes hyperparameters (its calls land in `llm_calls`, its proposals in `intent_records`) |
| **Players** | The three RL algorithms (PPO / A2C / SAC); **games** = folds; **seasons** = iterations; **leagues** = the two envs (equity, crypto) |
| **L1 / L2 / L3** | The coach's levers: L1 = mid-fold reward-weight nudges (BENCHED — config max-delta 0.0 everywhere), L2 = between-season hyperparameter picks (live), L3 = consolidated patterns fed into L2's prompt |
| **U1 / U2 / U3** | Unenumerated influence paths closed by §2.2: U1 = the `stop_training` flag (becomes advice-only), U2 = cold-start weight defaults (must equal `DEFAULT_WEIGHTS`), U3 = service-failure fallbacks (must be identity with baseline) |
| **Intent record ("bet slip")** | The five-block form every coach call writes (identity, evidence, proposal, falsifiable bet, verdict) — tables `intent_records` / `intent_verdicts` / `intent_applications` (DDL ships in Plan A's V004; the *writers* ship here) |
| **Verdict / grader** | A script (never an LLM) that checks a bet after its horizon and appends an `intent_verdicts` row |
| **Benching ladder** | §2.7's four escalation levels: record-only → scoped demotion to shadow → lever benched → coach removed. Computed by script from graded bets |
| **Pooled verdict** | A26: bets counted across all six (env, algo) scopes together — powers ladder levels 2–3 only, and can only *reduce* the coach's authority |
| **Harness Stage 1 / Stage 2** | §2.3's two-part lever tryout: Stage 1 = scripted pulls, 3 seed-pairs, no LLM ("does the tool work"); Stage 2 = replay recorded situations to the coach with production prompts ("does the operator judge well") |
| **Reference season** | Iteration 5: coach-free (baseline HPs, `DEFAULT_WEIGHTS`), shadow coach mic'd up. The bar every later season must beat |
| **Staircase** | One change per season: iter 5 reference → 6 L2-bare → 7 L2+patterns (stamped in `season_results.coach_config`) |
| **Era / gate version** | Comparability period / rulebook edition (`eras`, `gate_versions` — created by Plan A's V001). Cross-era CPS values are never compared |
| **CPS** | Composite Performance Score — the training system's single objective (v1 primary) |
| **Spine / `run_pk`** | The `training_runs` identity table (Plan A V002); every training-scoped row points at it |
| **Migration V{NNN}** | Numbered one-time SQL file in `src/swingrl/data/migrations/`, applied by Plan A Task 1's runner, recorded in `schema_migrations`. Plan A owns V001–V004; **Plan B starts at V005** |
| **Rollout-end / "epoch"** | The SB3 callback tick. PPO: every 12,288 steps (~82/fold). SAC with `train_freq=1`: **every vec-step** (~167k/fold) — the F2 root cause |
| **Cadence path / event path** | §4.10's two write reasons: every-Nth-epoch heartbeat (never capped) vs notable-event rows (rate-capped + hard-capped) |
| **Trend window / short window** | §2.6's two observation windows, sized in percent-of-fold: short (~1%) = acute detector, trend (~15%) = decision basis |
| **Equity-fraction MDD** | Max drawdown measured on portfolio value ((peak − value) / peak, a 0–1 fraction) — replaces the old cumsum-of-shaped-rewards quantity behind the broken −25.0 threshold |
| **FIFO round-trip** | How backtest trades are matched (first-in-first-out buy/sell pairing) to compute win_rate / profit_factor (`agents/backtest.py`) |
| **Canonical run** | For any (iteration, env, algo, fold, run_type): the highest `attempt` with `status='completed'`. All aggregation binds to canonical runs (A6) |
| **Corpus view** | `v_consolidation_corpus` — the *only* surface consolidation may read (§4.6); harness rows are structurally excluded |
| **S1–S8** | Spec §1.3 success criteria (S2 = worst-fold MDD guardrail, S4 = re-consolidation from record alone, S6 = harness before re-run) |
| **VecNormalize** | SB3's observation/reward normalizer, saved beside each model artifact |
| **Fail-open (counted)** | On LLM/service failure: training continues on baseline, but the failure is *recorded* (`llm_calls.success=false`) — never silently swallowed |
| **🛑 Backup gate** | Standing rule: destructive live-DB operations need plan-mode approval + a *verified* (restored + row-counted) backup immediately before |
| **DDL** | Data Definition Language — `CREATE TABLE`-class SQL, as opposed to row reads/writes |

## Global Constraints

- **Plan A dependency:** Phase 2+ tasks (Task 8 onward) require Plan A Tasks 1–4 (migration
  runner, fingerprint assertion, V001 registries, V002 spine) merged into the integration
  branch. **Phase 1 (Tasks 1–7) is independent and can start immediately.** Task 17
  additionally requires Plan A's V004 (`llm_calls` + intent DDL).
- **Branch strategy:** integration branch `swingrl/2.R-training-redesign` (merges to `main`
  only when the whole redesign is done). Plan B executes on
  **`swingrl/2.R-B-training-engine`**, branched from the integration branch. PRs target the
  integration branch, never `main`. (PROPOSED — confirm branch name at walkthrough.)
- **Migration numbering:** Plan B uses V005–V010. If Plan A adds files beyond V004 before a
  Plan B task lands, renumber at rebase — the ledger is ordered, numbers are not sacred.
- **No season mid-transition** (A25 cutover rule): no training iteration runs between the
  first Phase 3 writer landing and Task 28's cutover completing. Iteration 5 starts only
  after Task 29's readiness checklist passes.
- **Deploy isolation (A30, user-approved 2026-07-12):** once paper trading is live, every
  Plan B homelab deploy goes through the **trainer service** (Plan A Task E's compose
  split) — the trader container is never rebuilt or recreated by Plan B work. Corollary
  rules that bind every Plan B task: migrations are **additive-only while the trader
  runs** (never ALTER/DROP anything the deployed trader reads; V010's REVOKE executes
  inside Task 28's cutover window, where the trader is deliberately stopped — the one
  gated exception); training code **never writes `models/active/`** (Task E's tested
  ban; era-1 deployment goes through gated promotion); the schema assertion is
  floor-semantics (Plan A Task 1/3), so trainer-applied V005–V009 never brick the
  running trader.
- **Ops-jobs seam accepted (2026-07-14, master-sequence reconciliation D-4):** Tasks 19/27
  register the grader / freshness / nightly-dump crons in the **trader's** scheduler as
  written — meaning post-go-live grader tweaks ride trader rebuilds in market-safe windows.
  Accepted (graders stabilize after cutover); documented escape hatch if rebuild churn
  proves painful: re-home the training-ops crons to the `swingrl-collector` container's
  scheduler (same pattern as Plan A Task 11's amended calendar jobs). Related Task 28 note:
  **`swingrl-collector` keeps running through the cutover window** — its tables are not in
  V010's REVOKE list and its lifecycle is independent (A30 3-service topology, Plan A
  Task E as amended).
- **Locked decisions bind throughout** (never re-litigate): D-T1.*/D-T2.*/D-T3.*/D-MT.*,
  amendments A1–A29, A26 stricter numbers (**pooled ≥12 bets AND ≥3 seasons; ladder level 3
  = 4 seasons**), turbulence memo adopted in full, P-A1 sentinels.
- CLAUDE.md rules bind: no hardcoded symbols/paths/amounts (use `SwingRLConfig`);
  `load_config()` only; UTC; typed `SwingRLError` subclasses; structlog kwargs; TDD — RED
  commit before GREEN; never `--no-verify`; line length 100; `from __future__ import
  annotations`; training runs in the `swingrl` container; `docs/training/*.md` updated in
  the same commit as the code they describe; full test suite (background, 10-min timeout,
  0 failures) before any push.
- **DB write discipline during training:** never open a write connection while
  `model.learn()` is running (standing rule from the DuckDB era, kept under Postgres for
  the same reason — the epoch callback buffers and `flush_telemetry()` runs after learn).
- Tests requiring Postgres follow the repo pattern (`skipif` no `DATABASE_URL`); dev/CI
  use `swingrl_test`. GHA coverage stays red (issue #18); homelab CI is the gate.
- 🛑 **Live-DB gate:** migrations against live pg16 run only at deployment under the
  backup gate. Task 28's archive-and-drop is additionally gated on a *verified* restore.

## Approval gates — every point where execution STOPS for the user

Two kinds. **In-session gates** are checklist steps execution cannot pass (the next step
consumes the approved artifact, so there is nothing to run until approval). **Operational
gates** arrive via Discord / season reports once seasons run; nothing actuates without a
human, and every actuation is an `operator_actions` row.

| # | Gate | Kind | Mechanism |
|---|---|---|---|
| G-1 | This walkthrough (sections + N-table) | in-session | Plan status flips to WALKTHROUGH-REVIEWED only on full sign-off |
| G-2 | Every commit / push / PR | in-session | Standing plan-first + commit-approval rules |
| G-3 | Task 18 Step 4 — gate-derivation report | in-session | V009 is written only after approval; recorded in `gate_versions.approved_by/approved_at` |
| G-4 | Task 28 — cutover runbook execution | in-session | Plan-mode approval per step; REVOKE and archive-and-drop approved separately |
| G-5 | Task 28 — archive-and-drop 🛑 | in-session | Additionally requires the verified-restore row-count proof presented before DROP; script prints DROP statements, never executes them |
| G-6 | Task 29 Step 2 — homelab CI / deployment | in-session | Standing no-deploy-without-approval rule |
| G-7 | Ladder actions (demotion / bench / level-3) | operational | Season report + Discord recommend; human executes the documented config actuation + `operator_actions` row |
| G-8 | Outlier / freshness / hard-cap alarms | operational | Discord escalation; human review only, no automatic action |
| G-9 | S4 real gate on iteration 5 + iteration-6 hold | operational | Iter 6 does not start until S4 passes on an iter-5 attempt (A23); user calls it |

## Verified change-site register (2026-07-11, read from code at `661a2ad`)

| Site | Fact | Status |
|---|---|---|
| `src/swingrl/memory/training/reward_wrapper.py:154–174` | `_shape_rewards` replaces the env reward with a weighted component sum; **no risk-penalty term** (A3, live defect) | VERIFIED (direct read) |
| `src/swingrl/envs/base.py` step() §7 | `reward = sharpe_reward − risk_penalty`; `reward_components` = profit/sharpe/drawdown/turnover only; `risk_penalty` not exposed in info | VERIFIED (direct read) |
| `reward_wrapper.py:36,76–78,208–220` | Fixed `_ROLLING_WINDOW = 500`; `rolling_mdd()` = min of cumsum-of-shaped-rewards drawdown (not an equity fraction) | VERIFIED (direct read) |
| `services/memory/memory_agents/query.py:122–139` | `_SAFE_DEFAULTS` / `_SAFE_EPOCH_DEFAULTS` reward_weights = {0.4, 0.35, 0.20, 0.05} ≠ `DEFAULT_WEIGHTS` {0.50, 0.25, 0.15, 0.10} (U2, live) | VERIFIED (direct read) |
| `src/swingrl/memory/training/bounds.py:109–126` | `_MAX_REWARD_DELTA` + `_ADJUSTMENT_COOLDOWN_STEPS` hardcoded — no config surface (PPO 24,576 / A2C 500 / SAC 20,000) | VERIFIED (direct read) |
| `src/swingrl/memory/training/epoch_callback.py:325–377` | `_on_rollout_end` + `_should_store`: pure thresholding, no rate limiting (rate-cap greenfield); SAC docstring claims *less* frequent — opposite is true | VERIFIED (direct read) |
| `epoch_callback.py:41–46,77–78` | Per-algo cadence PPO 60 / A2C 8000 / SAC 40000; `NOTABLE_MDD_THRESHOLD = −25.0` against the cumsum quantity | VERIFIED (direct read) |
| `epoch_callback.py:608–619,694–713` | Advice gated on `epoch % cadence == 0`; **no startup guard on advice** (only `stop_training` has `MIN_TRAINING_PROGRESS = 0.20`); `model.stop_training = True` actuation at :712 | VERIFIED (direct read) |
| `epoch_callback.py:50–72` | `_ALGO_LOGGER_KEYS` per-algo SB3 key map — correct for SB3 2.7.1 per review; SAC `ent_coef` (the live entropy coefficient) not captured | VERIFIED (review §6 + direct read) |
| `src/swingrl/training/trainer.py:71` | `SEED_MAP = {"ppo": 42, "a2c": 43, "sac": 44}` — per-algo constants, identical across folds/iterations | VERIFIED (direct read) |
| `trainer.py:450–487` | `_create_eval_env` builds DummyVecEnv + VecNormalize with **no seed** (M9 — nondeterministic early-stop) | VERIFIED (direct read) |
| `scripts/train_pipeline.py:1997,2495,2570` | The three rich `backtest_results` write sites (single-writer collapse target); plain 22-col writer lives only in `scripts/backtest.py` (dead in the training path) | VERIFIED (review §6) |
| `services/memory/db.py:88–210` vs `postgres_schema.py` | Duplicate memory-table DDL, **types diverge** (`last_confirmed_at` TEXT vs TIMESTAMPTZ) — M8; startup order decides a fresh DB's schema | VERIFIED (review) |
| `agents/backtest.py:114–199` + `agents/metrics.py:218–272` | Backtest trades = FIFO round-trips; open lots at fold end excluded from win_rate/profit_factor; a buy-and-hold fold reports `total_trades=0` | VERIFIED (review §6) |
| `src/swingrl/memory/training/cps_diagnosis.py` | `TRADE_BASELINES` per-(env, algo) — the activity floor's source; diagnosis taxonomy = the shared category vocabulary | VERIFIED (review) |
| Live pg16 | 564 `backtest_results` + 10 `iteration_results` rows (iters 0–4) — the gate-derivation evidence base; era-0 back-stamp = Plan A Task 2 | VERIFIED (read-only checks 2026-07-06/11) |
| `config/schema.py:443–444` | `training.bounds` already carries `hyperparam_bounds`/`reward_bounds` pydantic models — the natural home for the new lever-limit fields | VERIFIED (direct read) |

## Assumptions register — each with its concrete verification method

| # | Assumption | Confidence | How it gets verified (task-wired) |
|---|---|---|---|
| P-B1 | ~~Fold budget ≈ 1.0M for all~~ **VERIFIED 2026-07-11**: `DEFAULT_TIMESTEPS` = equity 1,000,000 / crypto 500,000 (`pipeline_helpers.py:45–48`; `ESCALATED_TIMESTEPS` can raise these on non-convergence). 15% trend window = 150k (equity) / 75k (crypto) steps — both exceed the longest cooldown (SAC 20k) by 7.5× / 3.75× | Verified | Windows are computed at `_on_training_start` from the run's *actual* `total_timesteps` (so escalated runs resize correctly); the startup guard enforces trend ≥ cooldown at every run regardless |
| P-B2 | SB3 2.7.1 logs `train/ent_coef` for SAC (the live entropy coefficient, distinct from `ent_coef_loss`) | High | Task 13 test asserts the key appears in `logger.name_to_value` after a short SAC `learn()` on a toy env |
| P-B3 | Seeding model + train env + eval env makes `ConvergenceCallback` early-stop deterministic | Medium-High | Task 7 Step 4: two short same-seed runs must stop at the identical step; if they don't, the residual nondeterminism source is hunted before the task closes (fallback: seed-pair replication is already the A25 pre-statement for advice-enabled folds) |
| P-B4 | Both containers (`swingrl`, `swingrl-memory`) talk to the same pg16 and can share the migration-managed schema | High | Task 22 removes the second DDL copy; Task 28 cutover step verifies both containers boot against the migrated `swingrl_test` clone |
| P-B5 | The 564 era-0 rows are enough to derive per-(env, algo) gate floors that land in the 40–70% historical pass band | Medium | Task 18's replay script prints the band per candidate threshold set; if per-(env, algo) floors are unsatisfiable, floors fall back to per-env (documented decision in the derivation report) |
| P-B6 | Turbulence percentile and HMM p_crisis are complementary (ρ < ~0.9) — the memo's condition for keeping turbulence as an era-1 obs feature | Medium | Task 24 Step 1 computes correlation + MI on our history *before* the feature freeze; ρ ≥ 0.9 → obs feature dropped (halt keeps it regardless), decision recorded |
| P-B7 | Era-1 decomposed turbulence = **2 obs slots per env** replacing the 1 raw slot (net +1); layout constants + `CRYPTO_OBS_DIM` updated; era-0 models never see the new layout (they stay on `zero_turbulence_obs=true` until retired) | High | Task 24 shape tests per env × sentiment on/off; Plan A Task 7's `turbulence_obs_index` helper is extended, not bypassed |

## Proposed numbers — finalized at the walkthrough (single sign-off table)

Everything below is a **spec proposal made concrete**. Each row is wired into exactly one
task; signing off this table locks them.

| # | Parameter | Proposed value | Where used |
|---|---|---|---|
| N1 | Short window | 1% of fold (dual-unit recorded) | Task 5 |
| N2 | Trend window | 15% of fold (~150k steps at P-B1; must be ≥ longest cooldown — guarded) | Tasks 5, 6, 19, 21 |
| N3 | Notable-event triggers | `kl_spike`: approx_kl > 0.10 · `mdd_breach`: short-window `mdd_frac_worst` > 0.10 (equity) / 0.12 (crypto) · `trade_shy`: rate < 0.5× locked baseline · `churning`: rate > 3× baseline · `numeric_anomaly`: NaN/inf | Task 6 |
| N4 | Event-path hard cap | 50 event rows per run (rate cap already bounds ≤ ~40; cap = expected×10 class), Discord alarm on hit, cadence path never capped | Task 6 |
| N5 | Harness minimum run lengths | cooldown + one trend window, **per (env, algo)** (P-B1 verified budgets): equity — PPO ≥ 175k, A2C ≥ 151k, SAC ≥ 170k; crypto — PPO ≥ 100k, A2C ≥ 76k, SAC ≥ 95k steps | Task 21 |
| N6 | S2 margin | worst-fold MDD ≤ reference season's + **0.02** (2 points of MDD fraction), same fold set | Task 29 checklist; §1.3 |
| N7 | K (L2 settings-history digest depth) | 5 seasons, yaml `training.l2_digest_seasons` | Task 11 view + Task 17 prompt wiring |
| N8 | Gate replay acceptance band | historical pass rate 40–70%; fake-win anti-pattern must vanish; pass ⇒ fold CPS-contribution correlation positive | Task 18 |
| N9 | Per-fold seed formula | `seed = SEED_MAP[algo] * 1000 + fold_number` (42000+f / 43000+f / 44000+f — disjoint ranges) | Task 7 |
| N10 | Pooled-verdict thresholds (A26 — already locked, restated) | ≥12 pooled bets AND ≥3 seasons for ladder levels 2–3; level 3 = 4 seasons without uplift; per-scope minimum stays ≥10 | Task 20 |
| N11 | Per-scope outlier alarm | ≤1 correct of first 5 graded bets in a scope → Discord escalation, human review only | Task 20 |
| N12 | Season-close fail-open error band | > 5% of expected advice calls errored/timed out in a season → season report FAILS (loud) | Task 27 |
| N13 | Grader freshness alarm | any intent > 3 days past horizon ungraded → Discord | Task 19 |
| N14 | `operator_actions` table | INCLUDE in V008 (append-only: actor, action_type, target_table/target_id, reason, payload, ts) | Task 11 |
| N15 | Provider/tier per call_type | `epoch_advice` → Cerebras (existing free tier) · `run_config`, `consolidate_stage1/2` → OpenRouter nemotron-120b primary / kimi-k2.5 backup (existing) · `harness_replay` → **the production model of the call type it replays** (§2.3 production-identical) · `trade_*` → deferred to Meta-Trader spec (cap ≤1 intent/cycle stands) | Task 17, 21 |
| N16 | Plan B branch name | `swingrl/2.R-B-training-engine` | Task 0 |

---

## Phase 0 — Gate

### Task 0: Preconditions gate (no code)

**Files:** none.

- [ ] **Step 1:** Confirm Plan A Tasks 1–4 are merged into `swingrl/2.R-training-redesign`
  (migration runner + `assert_schema_current` + V001 + V002 exist). Phase 1 (Tasks 1–7)
  may start before this; Phase 2+ may not.
- [ ] **Step 2:** Confirm the proposed-numbers table (N1–N16) is signed off.
- [ ] **Step 3:** Confirm no training iteration is scheduled to run until Task 29 passes
  (the no-season-mid-transition rule).
- [ ] **Step 4:** Create branch `swingrl/2.R-B-training-engine` from
  `swingrl/2.R-training-redesign`.
- [ ] **Step 5: Pre-flight verification read (user-added 2026-07-12).** Read-only pass
  over the two large files this plan modifies but did not read during planning:
  `scripts/train_pipeline.py` (the fold loop + the three write sites — confirms Task
  12/14/16/17 insertion points) and `services/memory/memory_agents/consolidate.py`
  (confirms Task 22's rewire scope; split Task 22 in two if the read shows the rewrite
  exceeds one task). Findings recorded as amendments to the affected tasks **before**
  Phase 3 starts — the same review-before-execution discipline Plan A got. Phase 1 does
  not wait on this step (its sites were read directly during planning).

---

## Phase 1 — Training-path correctness (Tasks 1–7; no schema dependency)

### Task 1: A3 — the risk penalty survives reward shaping

The env computes `reward = sharpe_reward − risk_penalty`, but when shaping is active the
wrapper replaces that with a weighted component sum containing no penalty term — every
shaped run has trained penalty-free. Fix: the env exposes the penalty; the wrapper
subtracts it *outside* the weighted sum (the safety term is never reweightable). This is
a **precondition of any L1 harness run** (§2.11/A3) and of the reference season.

**Files:**
- Modify: `src/swingrl/envs/base.py` (step() §7b + `_build_info`)
- Modify: `src/swingrl/memory/training/reward_wrapper.py:137–174`
- Test: `tests/memory/test_reward_wrapper.py` (extend), `tests/test_envs.py` (extend)

**Interfaces:**
- Produces: `info["risk_penalty"]: float` on every env step (both envs, via `base.py`);
  wrapper constant `RISK_PENALTY_INFO_KEY = "risk_penalty"`; shaped reward =
  `Σ(weight_k × component_k) − risk_penalty`.
- Consumes: existing `reward_components` info contract.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_envs.py
def test_step_info_exposes_risk_penalty(equity_env):
    """A3: every step's info dict must carry the risk penalty the reward already uses."""
    equity_env.reset()
    _, reward, _, _, info = equity_env.step(equity_env.action_space.sample())
    assert "risk_penalty" in info
    assert info["risk_penalty"] >= 0.0
    # reward identity: sharpe component minus penalty
    assert reward == pytest.approx(
        info["reward_components"]["sharpe"] - info["risk_penalty"], abs=1e-9
    )

# tests/memory/test_reward_wrapper.py
def test_shaping_subtracts_risk_penalty():
    """A3: shaped reward = weighted component sum MINUS the unweighted risk penalty."""
    wrapper = _make_wrapper()  # existing test helper
    infos = [{
        "reward_components": {"profit": 1.0, "sharpe": 1.0, "drawdown": 0.0, "turnover": 0.0},
        "risk_penalty": 0.5,
    }]
    shaped = wrapper._shape_rewards(np.array([0.7]), infos)
    weighted = 0.50 * 1.0 + 0.25 * 1.0  # DEFAULT_WEIGHTS: profit .50, sharpe .25
    assert shaped[0] == pytest.approx(weighted - 0.5)

def test_shaping_without_penalty_key_is_unchanged_behavior():
    """Missing risk_penalty key (old envs, unit fixtures) → penalty treated as 0.0."""
    wrapper = _make_wrapper()
    infos = [{"reward_components": {"profit": 1.0, "sharpe": 0.0, "drawdown": 0.0,
                                    "turnover": 0.0}}]
    shaped = wrapper._shape_rewards(np.array([0.3]), infos)
    assert shaped[0] == pytest.approx(0.50)
```

- [ ] **Step 2:** Run: `uv run pytest tests/test_envs.py::test_step_info_exposes_risk_penalty tests/memory/test_reward_wrapper.py -v` — expect FAIL (no `risk_penalty` key). Commit RED.
- [ ] **Step 3: Implement.** In `envs/base.py` step(): pass `risk_penalty=risk_penalty`
  into `_build_info(...)` and add it to the returned dict. In `reward_wrapper.py`
  `_shape_rewards`, after the weighted sum:

```python
            # A3: the risk penalty is a safety term, never reweightable — it is
            # subtracted outside the weighted component sum (spec §2.11, amendment A3).
            weighted_reward -= float(info.get(RISK_PENALTY_INFO_KEY, 0.0))
            shaped[i] = weighted_reward
```

- [ ] **Step 4:** Run the same tests — PASS. Run the full wrapper + env suites:
  `uv run pytest tests/memory/ tests/test_envs.py tests/envs/ -v` — 0 failures.
- [ ] **Step 5:** Update `docs/training/reward-shaping.md` known-issue entry (bug → fixed,
  reference this task) in the same commit. Commit GREEN:
  `fix(training): A3 — risk penalty survives reward shaping (spec §2.11 precondition)`.

### Task 2: U2 — cold-start defaults equal `DEFAULT_WEIGHTS`

**Files:**
- Modify: `services/memory/memory_agents/query.py:122–139`
- Test: `tests/memory/test_query_safe_defaults.py` (new)

**Interfaces:**
- Produces: `_SAFE_DEFAULTS["reward_weights"]` and `_SAFE_EPOCH_DEFAULTS["reward_weights"]`
  both equal `{"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}`.
- Consumes: `DEFAULT_WEIGHTS` (`reward_wrapper.py:28–33`) as the canonical values. The
  memory service cannot import `swingrl` (separate container) — the test pins both copies
  to the same literals so drift fails a test instead of shifting weights silently.

- [ ] **Step 1: Failing test**

```python
# tests/memory/test_query_safe_defaults.py
"""U2 (spec §2.2): cold-start / fallback weights must be identity with DEFAULT_WEIGHTS."""
import importlib.util
from pathlib import Path

from swingrl.memory.training.reward_wrapper import DEFAULT_WEIGHTS

def _load_query_module():
    path = Path(__file__).parents[2] / "services" / "memory" / "memory_agents" / "query.py"
    spec = importlib.util.spec_from_file_location("memory_agents_query", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_safe_defaults_match_default_weights():
    q = _load_query_module()
    assert q._SAFE_DEFAULTS["reward_weights"] == DEFAULT_WEIGHTS
    assert q._SAFE_EPOCH_DEFAULTS["reward_weights"] == DEFAULT_WEIGHTS
```

  (If `query.py` cannot be imported standalone because of service-local imports, fall back
  to parsing the literal with `ast` — same assertion, no service dependencies; note which
  variant was used in the test docstring.)

- [ ] **Step 2:** Run: `uv run pytest tests/memory/test_query_safe_defaults.py -v` — FAIL
  (0.4 ≠ 0.5). Commit RED.
- [ ] **Step 3:** Edit both dicts in `query.py` to
  `{"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}` with an inline
  comment: `# U2: must equal reward_wrapper.DEFAULT_WEIGHTS — cross-checked by test`.
- [ ] **Step 4:** PASS + full memory suite green. Commit:
  `fix(memory): U2 — cold-start weights = DEFAULT_WEIGHTS (spec §2.2)`.

### Task 3: U1 — `stop_training` becomes advice-only

Evidence: 0 stop requests in 850,430 live epochs. The runtime stops honoring the flag;
the request is logged with full context and (from Task 17 on) written as a bet slip and
graded like any other call.

**Files:**
- Modify: `src/swingrl/memory/training/epoch_callback.py:317–323,694–713`
- Modify: `src/swingrl/training/trainer.py:358–370`
- Test: `tests/memory/test_epoch_callback_extended.py` (extend)

**Interfaces:**
- Produces: `_on_step()` always returns True; a stop advice logs
  `log.warning("llm_stop_request_advice_only", ...)` and increments
  `self._stop_requests: list[dict]` (epoch, timestep, pct_complete, reason) — consumed by
  Task 17's intent writer. `model.stop_training` is never set.
- Consumes: nothing new.

- [ ] **Step 1: Failing test**

```python
def test_stop_training_advice_is_never_actuated(callback_with_mock_client):
    """U1 (spec §2.2): a stop_training=true response must not stop the run."""
    cb = callback_with_mock_client  # existing fixture pattern in this file
    cb._client.epoch_advice.return_value = {
        "reward_weights": {}, "stop_training": True, "rationale": "test stop",
    }
    cb._epoch = cb._cadence - 1  # next rollout end is an advice epoch
    cb._on_rollout_end()
    assert getattr(cb.model, "stop_training", False) is False
    assert cb._on_step() is True
    assert len(cb._stop_requests) == 1
    assert cb._stop_requests[0]["reason"] == "test stop"
```

- [ ] **Step 2:** RED run + commit.
- [ ] **Step 3: Implement.** In `_query_epoch_advice`: replace the actuation branch —
  keep the `MIN_TRAINING_PROGRESS` context in the log line, drop the
  `self.model.stop_training = True` assignment, append to `self._stop_requests`
  (initialized `[]` in `__init__`). In `_on_step`: `return True` unconditionally (delete
  the getattr). In `trainer.py`: delete the `elif getattr(model, "stop_training", ...)`
  branch (converged_at now means convergence only).
- [ ] **Step 4:** PASS + full suite. Update `docs/training/memory_meta_trainer.md`
  stop-semantics section in the same commit. Commit:
  `feat(training): U1 — stop_training advice-only, folds always run to completion`.

### Task 4: L1 bench — lever limits move to config, default all-zero

The all-pairs bench (D-T2.1) must be a *config posture*, not a code edit. Today
`_MAX_REWARD_DELTA` and `_ADJUSTMENT_COOLDOWN_STEPS` are hardcoded (`bounds.py:109–126`).

**Files:**
- Modify: `src/swingrl/config/schema.py` (extend `BoundsConfig` at :443)
- Modify: `src/swingrl/memory/training/bounds.py:109–152`
- Modify: `config/swingrl.yaml`, `config/swingrl.prod.yaml.example`
- Test: `tests/memory/test_bounds.py` (extend), `tests/test_config.py` (extend)

**Interfaces:**
- Produces: config fields `training.bounds.max_reward_delta: dict[str, dict[str, float]]`
  (algo → env → max abs delta; **shipped default: 0.0 for all six pairs** — the bench)
  and `training.bounds.adjustment_cooldown_steps: dict[str, int]` (defaults: ppo 24576,
  a2c 500, sac 20000). `get_max_reward_delta(algo, env)` / `get_adjustment_cooldown(algo)`
  read config first, falling back to the current constants only on config-load failure
  (fail-open *counted*: warning log).
- Consumes: `load_config()`.

- [ ] **Step 1: Failing tests** — (a) `load_config(tmp_config)` exposes the new fields
  with all-zero delta defaults; (b) `get_max_reward_delta("a2c", "crypto") == 0.0` under
  default config (today returns 0.05); (c) yaml override `{"ppo": {"equity": 0.03}}`
  flows through.

```python
def test_max_reward_delta_benched_by_default(loaded_config):
    """D-T2.1: shipped posture is L1 benched everywhere — max delta 0.0 for all pairs."""
    for algo in ("ppo", "a2c", "sac"):
        for env in ("equity", "crypto"):
            assert loaded_config.training.bounds.max_reward_delta[algo][env] == 0.0
```

- [ ] **Step 2:** RED + commit.
- [ ] **Step 3:** Add pydantic fields with `Field(default_factory=...)` producing the
  zero/cooldown dicts; rewire the two getters through `_load_bounds()`'s pattern; yaml
  gains a commented `max_reward_delta` block (all zeros, with the re-earn note pointing
  at spec §2.3).
- [ ] **Step 4:** PASS + full suite. Update
  `.planning/research/algo-reward-shaping.md` pointer comment in `bounds.py` (values now
  config-owned). Commit: `feat(training): L1 bench as config posture (D-T2.1) — lever
  limits move to SwingRLConfig`.

### Task 5: Percent-of-fold windows, equity-fraction MDD, startup guard, dual-unit

Replaces the fixed 500-step deque with the §2.6 design: a short window (acute detector)
and a trend window (decision basis), both sized as fractions of the fold; window MDD
becomes an equity fraction from `info["portfolio_value"]`; a startup guard refuses to run
if the trend window can't cover the longest cooldown.

**Files:**
- Modify: `src/swingrl/memory/training/reward_wrapper.py` (window plumbing + MDD basis)
- Modify: `src/swingrl/memory/training/epoch_callback.py` (`_on_training_start`, metric
  collection call sites)
- Modify: `src/swingrl/config/schema.py` (+ `training.windows` model), `config/swingrl.yaml`
- Test: `tests/memory/test_reward_wrapper.py`, `tests/memory/test_epoch_callback_extended.py`

**Interfaces:**
- Produces:
  - Config `training.windows.short_pct_of_fold: float = 0.01`,
    `training.windows.trend_pct_of_fold: float = 0.15` (N1/N2).
  - `MemoryVecRewardWrapper.configure_windows(short_steps: int, trend_steps: int) -> None`
    — called once by the callback at `_on_training_start` (it knows
    `model._total_timesteps`); resizes deques; both sizes retained for dual-unit output.
  - `wrapper.window_metrics(window: Literal["short","trend"]) -> dict` returning
    `{"pct_of_fold": float, "steps": int, "sharpe_annualized": float,
    "mdd_frac_worst": float, "mdd_frac_mean": float, "win_rate": float,
    "trade_rate": float}` — MDD computed per sub-env from portfolio-value curves
    (peak-to-trough fraction), then **both** the worst sub-env and the mean recorded
    (user decision 2026-07-12, resolving the review's per-env-vs-pooled question):
    triggers and coach evidence read `mdd_frac_worst` (safety-first, conservative);
    `mdd_frac_mean` rides along in the JSONB for analysis and threshold recalibration —
    never the alarm basis. The two bases are never mixed (units-in-names).
  - Startup guard: in `_on_training_start`, `trend_steps < get_adjustment_cooldown(algo)`
    → raise `ConfigError` (refuse to start — the §2.6 guard).
  - `rolling_mdd()` retained temporarily as deprecated alias returning
    `-window_metrics("trend")["mdd_frac"]` for the diagnosis call sites, removed when
    Task 6 rewires callers.
- Consumes: `info["portfolio_value"]` (present per step — review-verified
  `envs/base.py:399–401`), `get_adjustment_cooldown` (Task 4's config-backed version).

- [ ] **Step 0:** Read the actual fold budget (`total_timesteps` passed to `learn()` in
  `train_pipeline.py`) and record it in the test file header (P-B1 verification).
- [ ] **Step 1: Failing tests** — (a) `configure_windows` sizes deques and
  `window_metrics` returns both units; (b) mdd_frac: feed a synthetic portfolio-value
  sequence 100→110→99 through infos → `mdd_frac == pytest.approx(0.1)`; (c) guard: config
  with `trend_pct_of_fold` small enough that trend_steps < SAC cooldown at a 100k-step
  fold → `ConfigError` at `_on_training_start`.
- [ ] **Step 2:** RED + commit.
- [ ] **Step 3:** Implement wrapper window storage (deques of per-step dicts:
  shaped reward, portfolio values per sub-env, trades) + metric computation; callback
  `_on_training_start` computes sizes, calls `configure_windows`, runs the guard, and
  logs `window_config` with both units (dual-unit capture, D-T2.7).
- [ ] **Step 4:** PASS + full suite green.
- [ ] **Step 5:** Update `docs/training/memory_meta_trainer.md` observability section
  (same commit). Commit: `feat(training): §2.6 percent-of-fold windows + equity-fraction
  MDD + startup guard`.

### Task 6: F2 — redesigned trigger set, rate cap, hard cap, honest docstring

`_should_store` is pure thresholding today; the −25.0 threshold against a cumsum quantity
is quasi-permanently true for crypto SAC, and SAC's rollout-end fires every vec-step.
This task makes the F2 *class* impossible (three-layer bounding, D-T3.19).

**Files:**
- Modify: `src/swingrl/memory/training/epoch_callback.py:325–377` (+ trigger config)
- Modify: `src/swingrl/config/schema.py` (+ `training.notable_events` model), `config/swingrl.yaml`
- Test: `tests/memory/test_epoch_callback_extended.py` (extend)

**Interfaces:**
- Produces:
  - Config `training.notable_events` with per-trigger thresholds (N3), all units in field
    names: `kl_spike_threshold: 0.10`, `mdd_breach_frac: {"equity": 0.10, "crypto": 0.12}`,
    `trade_shy_ratio: 0.5`, `churning_ratio: 3.0`, `hard_cap_per_run: 50` (N4).
  - `_should_store(...) -> tuple[bool, str | None]` evaluating the five triggers on
    **short-window metrics** (Task 5), with: rate cap = max one event row per trigger type
    per trend window; hard cap = `hard_cap_per_run` total event rows, past which event
    rows drop, a Discord alarm fires once (`alerter` available? training container has no
    alerter today — the alarm goes through the memory client's existing ingest path as a
    `capture_alarm` log row + structlog error; Discord wiring lands in Task 19 with the
    grader alarms — noted, not silent).
  - **Cadence path untouched and uncapped** — `epoch % cadence == 0` rows always flow.
  - Corrected `_on_rollout_end` docstring (SAC fires per vec-step, ~167k/fold — the
    review-pinned fact) + `log.info("rollout_cadence_observed", ...)` once per run at
    fold end with the actual rollout-end count (the F2 instrumentation, trivially from
    `self._epoch`).
- Consumes: Task 5 `window_metrics("short")`, locked `baseline_trade_rate`.

- [ ] **Step 1: Failing tests** — (a) same trigger type twice inside one trend window →
  second is suppressed (rate cap); (b) 51 distinct events → row 51 dropped, alarm flag
  set, cadence row on the next cadence epoch still stored (fail-safe direction: lose
  telemetry, never the heartbeat); (c) `mdd_breach` fires on `mdd_frac_worst` 0.11
  (equity) even when `mdd_frac_mean` is 0.03 — the worst basis is load-bearing,
  not on the old cumsum scale; (d) `numeric_anomaly` fires on NaN reward.
- [ ] **Step 2:** RED + commit.
- [ ] **Step 3:** Implement trigger evaluation + `_events_this_window: dict[str, int]`
  (reset at each trend-window boundary) + `_event_rows_this_run: int`; docstring +
  instrumentation.
- [ ] **Step 4:** PASS + full suite. Update `docs/training/memory_meta_trainer.md`
  notable-events table (same commit). Commit: `feat(training): F2 class-fix — §4.10
  trigger set + rate cap + hard cap; SAC cadence documented honestly`.

### Task 7: Per-fold seed pinning (incl. the eval env — M9)

D-T2.5's same-fold season-over-season comparison needs the lever to be the only variable.
Seeds today are per-algo constants; the eval env is entirely unseeded, so early-stop is
nondeterministic (M9) and would defeat pinning.

**Files:**
- Modify: `src/swingrl/training/trainer.py` (seed threading: model, train env, eval env)
- Modify: `scripts/train_pipeline.py` (pass `fold_number` into the trainer call path)
- Test: `tests/training/test_seed_pinning.py` (new)

**Interfaces:**
- Produces: `fold_seed(algo: str, fold_number: int) -> int` =
  `SEED_MAP[algo] * 1000 + fold_number` (N9) in `trainer.py`; `Trainer.train(...,
  fold_number: int | None = None)` — when provided, the fold seed is applied to (a)
  `model = ALGO(..., seed=fold_seed)`, (b) train VecEnv via `vec_env.seed(fold_seed)` +
  `env.action_space.seed(fold_seed)`, (c) **eval env** likewise with `fold_seed + 1`
  (distinct stream, still pinned). Seed recorded in `TrainingResult.seed` for Task 12's
  spine row. `fold_number=None` (ad-hoc runs) keeps today's constants — behavior change
  is opt-in per call site, and `train_pipeline.py` opts in for every fold run.
- Consumes: `SEED_MAP` (:71), `_create_env`/`_create_eval_env`.

- [ ] **Step 1: Failing tests** — (a) `fold_seed("ppo", 7) == 42007`; algo ranges
  disjoint; (b) determinism: two 2,000-step PPO runs on the synthetic fixture env with
  the same `fold_number` produce identical `num_timesteps` at convergence-callback stop
  and bitwise-identical final `predict()` outputs on a fixed obs batch; (c) different
  folds → different seeds. (Test (b) is the P-B3 verification — marked slow, runs in CI.)
- [ ] **Step 2:** RED + commit.
- [ ] **Step 3:** Implement threading; log `seed_pinned` (algo, fold, seed, eval_seed).
- [ ] **Step 4:** PASS + full suite. Document in `docs/training/training-pipeline.md`:
  pinning holds for coach-free folds; **advice-enabled folds are inherently
  irreproducible → seed-pair replication is the fallback** (A25 pre-statement, verbatim).
  Commit: `feat(training): per-fold seed pinning incl. eval env (D-T2.5, M9)`.

---

## Phase 2 — Remaining §4 schema (Tasks 8–11; requires Plan A Tasks 1–4)

All DDL follows spec §4 field lists **verbatim** (they were locked at G1 and are not
re-designed here); each migration test applies `apply_migrations` to `swingrl_test` and
asserts the load-bearing constraints. Every task bumps `EXPECTED_SCHEMA_VERSION`.

### Task 8: V005 — training records (`epoch_snapshots`, `fold_results`, `season_results`, `backtest_trades`)

**Files:**
- Create: `src/swingrl/data/migrations/V005__training_records.sql`
- Modify: `src/swingrl/data/migration_runner.py` (bump `EXPECTED_SCHEMA_VERSION` to 5)
- Test: `tests/data/test_migrations_content.py` (extend)

**Interfaces:**
- Produces: the four §4.3 tables. Load-bearing constraints, exactly as spec'd:
  `fold_results.run_pk UNIQUE` (read-time dedup dies — the `backtest_results`
  9-duplicate class); `fold_results` explicit `era_id` + `gate_version_id` FKs (A29
  surrogate) + `seed` denorm (A11) + `turbulence_mean` (A27) +
  `max_single_loss_frac` / `initial_capital_usd` (the CPS-v2 units fix);
  `season_results` UNIQUE(iteration_number, environment, scope, result_version) with
  scope CHECK (`ppo`,`a2c`,`sac`,`ensemble`) + NOT NULL `coach_config` JSONB;
  `backtest_trades` UNIQUE(run_pk, bar_ts, symbol) (the physical ceiling);
  `epoch_snapshots.learner_metrics` JSONB NOT NULL.
  Indexes (per §4.14 misc): spine FKs on all four; `fold_results(fold_start_ts)`;
  `season_results(iteration_number, environment)`.
- Consumes: Task 1 runner, V001 registries, V002 spine (all Plan A).

- [ ] **Step 1: Failing test** — after `apply_migrations`: (a) inserting two
  `fold_results` rows with the same `run_pk` raises; (b) a `season_results` insert
  without `coach_config` raises; (c) duplicate `backtest_trades` (run_pk, bar_ts, symbol)
  bounces; (d) `SELECT max(version) FROM schema_migrations` == 5.
- [ ] **Step 2:** RED (V005 absent) + commit.
- [ ] **Step 3:** Write V005 DDL — transcribe §4.3 field lists column-for-column with
  the units-in-names convention and column comments for sign conventions; every JSONB
  payload documented to carry `schema_version` (A8 — enforced by writers, commented in
  DDL).
- [ ] **Step 4:** PASS. Commit: `feat(schema): V005 §4.3 training records`.

### Task 9: V006 — patterns family (`patterns`, `pattern_sources`, `pattern_links`, `pattern_presentations`)

**Files:**
- Create: `src/swingrl/data/migrations/V006__patterns.sql`
- Modify: `migration_runner.py` (version → 6)
- Test: `tests/data/test_migrations_content.py` (extend)

**Interfaces:**
- Produces: §4.5 tables verbatim: `patterns.claim` JSONB NOT NULL + `qa_passed` +
  status CHECK + script-maintained counters; `pattern_sources(source_table CHECK,
  source_id)` polymorphic; `pattern_links` PK(parent, child) + link_type CHECK
  (`merged_into`,`split_into`,`refined_into`); `pattern_presentations` with **mandatory**
  pattern_id + llm_call_id FKs (the NULL-iteration class dies).
- Consumes: V004 `llm_calls` (Plan A Task 12 DDL), V001 `eras`.

- [ ] **Step 1: Failing test** — presentations insert with NULL `llm_call_id` raises;
  `pattern_links` self-link (parent == child) rejected by CHECK; source_table outside the
  allowlist rejected.
- [ ] **Step 2:** RED + commit. **Step 3:** DDL. **Step 4:** PASS + commit:
  `feat(schema): V006 §4.5 patterns + lineage DAG`.

### Task 10: V007 — harness records (`harness_experiments`, `harness_experiment_runs`, `harness_replays`)

**Files:**
- Create: `src/swingrl/data/migrations/V007__harness.sql`
- Modify: `migration_runner.py` (version → 7)
- Test: `tests/data/test_migrations_content.py` (extend)

**Interfaces:**
- Produces: §4.6 tables verbatim: `harness_experiments.pull_spec` JSONB NOT NULL
  (pre-registration — written before any run); `harness_experiment_runs(experiment_id,
  run_pk, arm CHECK (pull|control), seed_pair SMALLINT)`; `harness_replays` linking
  experiment → llm_call.
- Consumes: V002 spine, V004 `llm_calls`.

- [ ] **Steps 1–4:** same RED→GREEN pattern; the content test asserts `pull_spec` NOT
  NULL and the arm CHECK. Commit: `feat(schema): V007 §4.6 harness records`.

### Task 11: V008 — weakness profiles, `operator_actions`, and the six views

**Files:**
- Create: `src/swingrl/data/migrations/V008__weakness_operator_views.sql`
- Modify: `migration_runner.py` (version → 8)
- Test: `tests/data/test_migrations_content.py`, `tests/data/test_views.py` (new)

**Interfaces:**
- Produces:
  - `weakness_profiles` (§4.8 verbatim: UNIQUE(environment, algorithm, failure_mode,
    version); append-only versioning) + `weakness_evidence` polymorphic.
  - `operator_actions` (N14): `(id BIGINT identity PK, actor TEXT NOT NULL, action_type
    TEXT NOT NULL, target_table TEXT, target_id BIGINT, reason TEXT NOT NULL, payload
    JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now())` — append-only record of human
    interventions outside pre-built slots (§4.14 misc decision → INCLUDE).
  - The six derived views (never store the derivable):
    `v_consolidation_corpus` (per-record-type definition, A19 — canonical
    season/reference runs; non-run-scoped allowlist; harness excluded),
    `v_l2_settings_history` (D-T2.11), `v_lever_track_record` (S8 aggregation over
    intents ⋈ verdicts by coach/lever/scope + `coach_config`), `v_consolidator_quality`
    (A18), `v_pattern_effectiveness` (presentations → calls → results),
    `v_live_transfer` (§4.7).
- Consumes: V005–V007 + Plan A V003/V004 tables (views reference trade-time tables).

- [ ] **Step 1: Failing tests** — migration content test (constraints) + view smoke
  tests: seed one canonical run + one harness run with epoch/fold rows; assert
  `v_consolidation_corpus` returns the canonical season rows and **zero** harness rows
  (S4 criterion 5's foundation); assert `v_l2_settings_history` returns one row per
  (env, algo, iteration) with NULL `source_intent_id` for a reference season.
- [ ] **Step 2:** RED + commit. **Step 3:** DDL + view SQL (each view's SELECT written
  out in the migration; corpus-view predicates: `run_type IN ('season','reference')`,
  canonical-attempt subquery per A6, `call_type <> 'harness_replay'`).
- [ ] **Step 4:** PASS. Commit: `feat(schema): V008 weakness profiles + operator_actions
  + derived views (incl. consolidation corpus boundary)`.

---

## Phase 3 — Writer rewiring (Tasks 12–17)

The change-site inventory (§4.14 hand-off "every §4 table vs today's writers") is
resolved as: `training_epochs` writer → Task 13; `backtest_results` writers (rich ×3
sites, plain ×1) → Task 14; `iteration_results` writer → Task 16; `reward_adjustments`
two-pass core → Task 17 (intent writers); `memories`/`meta_decisions`/
`pattern_outcomes` writers → retired (Task 22 consolidation v2 + Task 28 REVOKE).
Old-table writes continue in parallel until Task 28's cutover **flips them off in one
gated operation** — no dual-write period is skipped silently (each new writer lands
alongside, not instead of, its legacy counterpart until cutover).

### Task 12: Run registration — `training_runs` rows with provenance

**Files:**
- Create: `src/swingrl/training/run_registry.py`
- Modify: `scripts/train_pipeline.py` (fold-run call path: register at start, close at end)
- Test: `tests/training/test_run_registry.py` (new)

**Interfaces:**
- Produces:
  - `register_run(db, *, iteration: int, environment: str, algorithm: str, fold_number:
    int, run_type: str, seed: int, config: SwingRLConfig, fold_slice: pd.DataFrame) ->
    int` — resolves `era_id` (`max(era_id) WHERE first_iteration <= iteration`, the A7
    rule), computes `attempt` (`max(attempt)+1` for the identity tuple), stamps
    provenance: `code_version` (git SHA via `SWINGRL_CODE_VERSION` env var set at image
    build; fallback `git rev-parse HEAD`; never 'unknown' silently — raises `ConfigError`
    if both fail, A12), `config_hash` = sha256 of the canonical JSON dump of the loaded
    config, `config_snapshot` JSONB, `data_fingerprint` = sha256 over (row_count,
    min/max bar timestamps, sum of close prices) of the fold's OHLCV slice. Inserts
    `status='running'`, returns `run_pk`.
  - `close_run(db, run_pk: int, status: Literal["completed","failed","aborted"]) -> None`.
  - Crash semantics: a run left `running` is never canonical (A6); the pipeline's next
    attempt gets a fresh row.
- Consumes: V002 `training_runs`, V001 `eras`; Task 7 `fold_seed`.

- [ ] **Step 1: Failing tests** — (a) registering the same identity twice yields
  attempt 1 then 2; (b) era resolution picks era 0 for iteration ≤ 4 and the max
  qualifying era otherwise; (c) missing code version raises `ConfigError`; (d)
  `data_fingerprint` changes when one close price changes (gap-fill revision detection —
  the A12 purpose).
- [ ] **Step 2:** RED + commit. **Step 3:** implement; wire `train_pipeline.py` fold loop:
  register before `Trainer.train`, close after (status from result/exception). Log
  `run_registered` with all identity kwargs.
- [ ] **Step 4:** PASS + full suite. Commit: `feat(training): training_runs registration
  + A12 provenance (identity spine live)`.

### Task 13: `epoch_snapshots` writer + `learner_metrics` per-algo contract

**Files:**
- Create: `src/swingrl/memory/training/epoch_writer.py`
- Modify: `src/swingrl/memory/training/epoch_callback.py` (`_collect_metrics`,
  `_ingest_epoch_snapshot`, `flush_telemetry`)
- Test: `tests/memory/test_epoch_writer.py` (new)

**Interfaces:**
- Produces:
  - `LEARNER_METRIC_CONTRACT: dict[str, tuple[str, ...]]` — per-algo required JSONB keys:
    ppo `("policy_gradient_loss","value_loss","entropy_loss","approx_kl","clip_fraction")`,
    a2c `("policy_loss","value_loss","entropy_loss")`,
    sac `("actor_loss","critic_loss","ent_coef_loss","ent_coef")` (P-B2; the missing
    health signal added).
  - **Absent-key rule (kills the PPO-only-keys class):** a metric SB3 hasn't logged yet
    (SAC during `learning_starts`) is **omitted** from the JSONB and listed under
    `"missing": [...]` — never written as fake `0.0`. Contract validated at write time;
    a key outside the contract for that algo raises in tests (drift alarm).
  - `EpochSnapshotBuffer.append(row: dict) -> None` + `.flush(db) -> int` — buffered
    during `learn()`, flushed by `flush_telemetry()` after (the no-writes-during-learn
    rule). Row shape = §4.3 columns: `run_pk`, epoch, timestep, `pct_complete`,
    `mean_reward`, `learner_metrics` (with `schema_version: 1` per A8), `window_short` /
    `window_trend` (Task 5 `window_metrics`, dual-unit), `reward_weights`,
    `notable_event`.
- Consumes: Task 12 `run_pk` (callback gains `run_pk` init arg), Task 5 windows, Task 6
  trigger labels, V005 DDL.

- [ ] **Step 1: Failing tests** — (a) a PPO row missing `approx_kl` from SB3 logs →
  JSONB has no `approx_kl` key and `"missing": ["approx_kl"]`; (b) SAC contract includes
  `ent_coef` and a short SAC `learn()` on the toy env yields it (P-B2 — slow marker);
  (c) flush inserts rows with the correct `run_pk` and `schema_version`; (d) buffer
  survives a flush failure without raising into the training loop (fail-open counted:
  error logged, rows retained for retry at next flush).
- [ ] **Step 2:** RED + commit. **Step 3:** implement; `_collect_metrics` rewritten to
  contract form (legacy `training_epochs` ingest path kept in parallel until Task 28).
- [ ] **Step 4:** PASS + full suite. Update `docs/training/memory_meta_trainer.md`
  learner-metrics table (same commit). Commit: `feat(training): epoch_snapshots writer —
  per-algo learner_metrics contract, absent ≠ 0.0`.

### Task 14: `fold_results` single writer

**Files:**
- Create: `src/swingrl/training/fold_results_writer.py`
- Modify: `scripts/train_pipeline.py:1997,2495,2570` (three call sites → one function)
- Modify: `scripts/backtest.py` (plain writer: replaced by the same function)
- Test: `tests/training/test_fold_results_writer.py` (new)

**Interfaces:**
- Produces: `write_fold_result(db, *, run_pk: int, gate: GateVerdict, metrics:
  FoldMetrics) -> int` — the **only** code path that inserts `fold_results`.
  `FoldMetrics` frozen dataclass mirrors §4.3 columns (all `_frac`/`_annualized` units in
  field names; `max_single_loss_frac` computed from `backtest_trades`-bound data, never
  dollars; `initial_capital_usd` alongside; `turbulence_mean` from the fold window per
  A27). `GateVerdict` = `(gate_version_id: int, passed: bool, components: dict)` —
  produced by Task 18's evaluator; until era 1 exists, tests use gate v0 rows from Plan
  A's V001 bootstrap. Era/gate stamps + `seed` denormalized from the spine row
  (single SELECT, belt-and-suspenders per §4.1).
- Consumes: Task 12 `run_pk`, V005, Task 18 `evaluate_gate` (interface defined there;
  this task takes the verdict as input — no circular dependency).

- [ ] **Step 1: Failing tests** — (a) second write for the same `run_pk` raises
  (UNIQUE); (b) a dollars-scale `max_single_loss` (≤ −1.0) is rejected by validation
  (`abs(frac) ≤ 1.0` guard — the CPS-v2 poison can't recur); (c) `gate_components` JSONB
  carries threshold + actual + pass per requirement and `schema_version`.
- [ ] **Step 2:** RED + commit. **Step 3:** implement; rewire the three
  `train_pipeline.py` sites and `scripts/backtest.py` through it (legacy
  `backtest_results` write kept in parallel until cutover).
- [ ] **Step 4:** PASS + full suite. Commit: `feat(training): fold_results single writer
  (D-T3.3) — two-writer split collapsed`.

### Task 15: `backtest_trades` writer (evaluation episodes only)

**Files:**
- Modify: `src/swingrl/agents/backtest.py` (emit per-trade records from the FIFO pass)
- Create: `src/swingrl/training/backtest_trades_writer.py`
- Test: `tests/agents/test_backtest_trades.py` (new)

**Interfaces:**
- Produces: `write_backtest_trades(db, run_pk: int, trades: list[BacktestTrade]) -> int`;
  `BacktestTrade` frozen dataclass per §4.3 (`bar_ts`, symbol, side, `weight_delta_frac`,
  `price_usd`, `cost_frac`, `position_after_frac`, `realized_pnl_frac | None`).
  **Hard rule enforced structurally:** the writer takes `run_pk` and asserts the run's
  `run_type != 'season_learning'` — more precisely, it is called only from the OOS
  evaluation pass in `agents/backtest.py`; the training loop has no import of this
  module (test asserts by grep — same pattern Plan A uses for layout constants).
  Open-lot honesty: lots still open at fold end are emitted with `realized_pnl_frac =
  NULL` and counted in a returned `open_lots: int` — the win_rate/profit_factor
  exclusion is now *visible* in data instead of silent (the review's
  open-position-metric-gap input; gate design consumes `trade_count` = all trades, so a
  buy-and-hold fold no longer reports 0 activity while holding).
- Consumes: Task 12 `run_pk`, V005 UNIQUE(run_pk, bar_ts, symbol).

- [ ] **Step 1: Failing tests** — (a) FIFO round-trip on a 3-trade fixture produces
  matched `realized_pnl_frac` and one NULL open lot; (b) duplicate (run_pk, bar_ts,
  symbol) bounces off the schema; (c) `trade_count` in Task 14's metrics equals rows
  written (consistency).
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Commit: `feat(training):
  backtest_trades capture (D-T3.16) — eval episodes only, open lots visible`.

### Task 16: `season_results` writer

**Files:**
- Create: `src/swingrl/training/season_writer.py`
- Modify: `scripts/train_pipeline.py` (season-close path)
- Test: `tests/training/test_season_writer.py` (new)

**Interfaces:**
- Produces: `write_season_results(db, *, iteration: int, environment: str, coach_config:
  dict, gate_version_ids: tuple[int, int]) -> list[int]` — computes the four scope rows
  (`ppo`,`a2c`,`sac`,`ensemble`) **from canonical runs only** (A6 subquery), assigns
  `result_version` = max+1 per key (recompute = new row, never UPDATE, A10), stamps
  era + both gate versions + `coach_config` (NOT NULL — the staircase stamp),
  CPS v1/v2/v3 + components JSONB, `worst_fold_number`/`worst_fold_mdd_frac` (S2's read
  surface), `hyperparams_used` (algo scopes) / `ensemble_weights` (ensemble scope),
  ensemble-scope `gate_passed` + `gate_components` (A10 — the ensemble gate's home).
- Consumes: Tasks 12/14 rows, Task 18 ensemble gate evaluator, V005.

- [ ] **Step 1: Failing tests** — (a) re-running the writer produces `result_version` 2,
  version-1 rows untouched; (b) a non-canonical (failed) attempt's fold rows do not move
  CPS; (c) missing `coach_config` raises before any insert.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Commit: `feat(training):
  season_results writer — per-scope rows, result_version, coach_config stamp`.

### Task 17: Coach-call capture — `llm_calls` + intent writers (training side)

Every training-side LLM conversation becomes an `llm_calls` row; every lever call
becomes a five-block bet slip. Trade-time call types stay Plan A Task 12's.

**Files:**
- Create: `src/swingrl/memory/training/intent_writer.py`
- Modify: `src/swingrl/memory/training/epoch_callback.py` (advice path), `meta_orchestrator.py`
  (run-config path), `src/swingrl/memory/client.py` (surface provider/model/latency/tokens)
- Modify: `scripts/train_pipeline.py` (L2 application → `intent_applications`)
- Test: `tests/memory/test_intent_writer.py` (new)

**Interfaces:**
- Produces:
  - `record_llm_call(db, *, coach, call_type, run_pk=None, iteration=None, environment=
    None, algorithm=None, provider, model, prompt_version, prompt_text, response_text,
    response_parsed, success, error, latency_ms, tokens_in, tokens_out) -> int` —
    honoring V004's identity CHECK matrix (A15); **failures are rows too**
    (`success=false`, fail-open counted — the U3 closure: every fallback event is now
    visible; enumerated U3 sites from the review each gain a `record_llm_call` failure
    write: transport `{}`, callback timeout, validation-ignored, orchestrator
    cold-start/exception).
  - `write_intent(db, *, llm_call_id, coach, lever, mode, identity: IntentIdentity,
    evidence: dict, proposal: dict, bet: Bet) -> int` — blocks 1–4;
    `Bet = (metric: str, direction: Literal["up","down"], baseline_value: float)`;
    `horizon_spec` is **system-written** from the lever (D-T2.8): L1/U1 →
    `{"type": "trend_window", "steps": N2_steps}`, L2 → `{"type": "season_same_fold"}`.
    Bet-metric menu = a module registry `BET_METRICS: dict[str, str]` (metric → unit,
    A9).
  - `write_application(db, intent_id, applied: dict) -> None` → `intent_applications`
    (A13) — called by `train_pipeline.py` where clamped HPs actually land.
  - Wire-in: **L2** — every season, every (env, algo): one intent (explicit no-change
    included; reference seasons write `mode='shadow'`, proposal = no-change — the
    `v_l2_settings_history` invariant). **Shadow L1** — each epoch-advice response with a
    weight proposal writes `mode='shadow'` intent (weights never applied while benched).
    **U1** — each stop request (Task 3's `_stop_requests`) writes a `U1_stop` intent.
    ≤1 intent per llm_call enforced by writer.
  - Prompt addition: run-config prompt gains the `v_l2_settings_history` digest (last
    N7=5 seasons) + per-lever track record (S8) — rendered by a pure function
    `render_l2_digest(rows) -> str` with tests.
- Consumes: V004 DDL (Plan A Task 12), Task 12 `run_pk`, N7/N15.

- [ ] **Step 1: Failing tests** — (a) `epoch_advice` call without `run_pk` violates the
  CHECK (test documents the matrix); (b) an advice timeout produces a `success=false`
  row (fail-open counted); (c) L2 no-change intent has `proposal={"change": null,
  "rationale": ...}` and a `season_same_fold` horizon; (d) second intent for one
  llm_call raises; (e) `render_l2_digest` on a 2-season fixture contains both CPS deltas.
- [ ] **Step 2:** RED + commit. **Step 3:** implement + wire the three call paths.
- [ ] **Step 4:** PASS + full suite. Update `docs/training/memory_meta_trainer.md`
  (intent lifecycle diagram) same commit. Commit: `feat(training): five-block intent
  records live for L2/L1-shadow/U1 (D-T2.4, D-T2.8, S3, S8)`.

---

## Phase 4 — Evidence engine (Tasks 18–20)

### Task 18: Gate re-derivation — replay against the 564 rows, gate v1, era 1

Replaces the fake-win-capable gate (`sharpe > 0.7, mdd < 0.15, profit_factor > 1.5,
overfitting_gap < 0.20`) with §2.8's win condition. The thresholds come from data, the
rows go in as a **human-approved migration** (gates are never written by training code).

**Files:**
- Create: `scripts/derive_gates.py` (the replay/derivation script — runs offline, prints
  a report, writes nothing to live)
- Create: `src/swingrl/training/gate_eval.py`
- Create: `src/swingrl/data/migrations/V009__gate_v1_era_1.sql` (values filled from the
  approved derivation report)
- Test: `tests/training/test_gate_eval.py`, `tests/training/test_derive_gates.py`

**Interfaces:**
- Produces:
  - `evaluate_gate(definition: dict, metrics: FoldMetrics) -> GateVerdict` — pure
    function; `definition` is the `gate_versions.definition` JSONB (each §2.8
    requirement with per-env / per-(env, algo) thresholds); returns passed +
    per-requirement components (threshold, actual, pass). Same shape for the ensemble
    gate over season metrics.
  - `scripts/derive_gates.py`: loads the 564 era-0 `backtest_results` rows (read-only),
    proposes thresholds — profit floor (small positive, per env), activity floor from
    `TRADE_BASELINES` (`cps_diagnosis.py`), per-env MDD ceiling, Sharpe/profit-factor
    minimums, overfit-gap ceiling — then **replays**: applies candidate definitions to
    all 564 rows and prints (a) historical pass rate per (env, algo) vs the N8 40–70%
    band, (b) the fake-win check: iterations where returns fell must show pass rate
    falling too (the iter 4–5 A2C proof case must flip), (c) rank correlation between
    passing and fold CPS contribution. Report saved to
    `docs/training/gate-derivation-report.md`.
  - V009: inserts `gate_versions` v1 rows (per_fold + ensemble, `approved_by` = user,
    `derivation_evidence` = report path) + the `eras` era-1 row (`first_iteration = 5`,
    reason = 'gate re-derivation + schema cutover', both gate version FKs).
    **V009 is written only after the user approves the derivation report** — the
    walkthrough of that report is a scheduled stop-point in this plan's execution.
- Consumes: live-pg16 read-only access (or the archived dump) for derivation; V001
  `gate_versions`/`eras`; `TRADE_BASELINES`.

- [ ] **Step 1: Failing tests** — `evaluate_gate` fixtures: (a) a fold with positive
  return but zero trades FAILS (activity floor — the unplayed-game fix); (b) a fold
  passing all floors with mdd_frac above the per-env cap FAILS; (c) components JSONB
  lists every requirement with threshold/actual/pass; (d) derivation script unit test on
  a 20-row fixture: report contains pass-rate table + fake-win verdict line.
- [ ] **Step 2:** RED + commit. **Step 3:** implement evaluator + script; run the script
  against the real 564 rows on homelab (read-only); save the report.
- [ ] **Step 4: STOP — user approval of the derivation report** (thresholds + band
  evidence). Only then write V009 with the approved numbers; migration content test
  asserts era-1 `first_iteration = 5` and both v1 rows exist.
- [ ] **Step 5:** PASS + full suite. Wire `evaluate_gate` into Task 14/16 writers
  (replacing their placeholder gate-v0 verdicts for era-1 runs). Commit:
  `feat(training): §2.8 derived gates — replay-validated v1 + era 1 (S5, D-T2.10)`.

### Task 19: Graders + freshness alarm — the verdict engine

Every grader class gets a named owner and a schedule; a dead grader gets loud within
days (A25). Graders are scripts; the LLM never grades itself (D-T2.9).

**Files:**
- Create: `src/swingrl/training/graders.py`
- Create: `scripts/run_graders.py` (CLI entry — cron/scheduler-owned)
- Modify: `src/swingrl/scheduler/jobs.py` (register the grader job + alarm job)
- Test: `tests/training/test_graders.py`

**Interfaces:**
- Produces (each appends `intent_verdicts` rows, `grader_version = 1`, idempotent via
  UNIQUE(intent_id, grader_version)):
  - `grade_midfold_intents(db) -> int` — L1-shadow/U1 bets: reads the bet's
    `trend_window` horizon, pulls the target metric from `epoch_snapshots` after the
    horizon epoch, writes actual_value + direction_match + menu_consistent (diagnosis →
    correction menu re-check via `cps_diagnosis` mapping).
  - `grade_l2_intents(db) -> int` — season horizon: same-fold season-over-season CPS
    comparison (canonical runs, same env/algo/fold set, era-guarded — cross-era bets
    excluded with `excluded_reason='new_fold_residue'`-class rows per §2.6).
  - `sweep_unreachable(db) -> int` — A16 terminal-verdict guarantee: any intent whose
    horizon can no longer arrive (run aborted, season cancelled, fold set changed) gets
    `excluded=true, excluded_reason='horizon_unreachable'`.
  - `grade_pattern_claims(db) -> int` — season-close mechanical re-check of active
    pattern claims against new graded fold rows → confirmation/contradiction counter
    updates + confidence formula (Task 22 wires the formula; owner lives here).
  - `freshness_alarm(db, alerter) -> None` — N13: any intent > 3 days past horizon
    without a verdict → Discord alert with count + oldest intent id.
  - `run_graders.py --test-alert` — sends one real Discord alert through the same route
    the graders use (fire-and-forget, timeout-bounded) and exits; the training-side
    delivery proof consumed by Task 29's checklist (user-added 2026-07-12). **This task also
    lands the Discord route from the training/memory side**: `graders.py` receives the
    existing `monitoring/alerter.py` instance via `run_graders.py` (scheduler context) —
    the A25 "neither is wired today" gap closes here; Task 6's capture alarm switches to
    it in the same commit.
  - Trade-time verdicts (MT commentary bets) reuse `grade_midfold_intents`' machinery
    with wall-clock horizons — enabled when Plan A Task 12's writers go live; the grader
    is horizon-type-driven, not coach-driven.
- Consumes: Tasks 13/14/16/17 rows; `scheduler/jobs.py` patterns; N13.

- [ ] **Step 1: Failing tests** — (a) an L1-shadow intent with a reachable horizon and
  a fixture metric path gets direction_match computed correctly for both bet directions;
  (b) an intent past horizon with no data → `horizon_unreachable` from the sweep, never
  silence; (c) grading twice under one grader_version inserts once; (d) freshness alarm
  fires on a 4-day-old ungraded intent fixture and not on a graded one.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Scheduler registration:
  graders nightly (03:30 ET, after ingestion windows), freshness alarm daily. Commit:
  `feat(training): script graders + horizon_unreachable sweep + freshness alarm (A16/A25)`.

### Task 20: Benching ladder + track record — A26 layered counting

**Files:**
- Create: `src/swingrl/training/ladder.py`
- Create: `scripts/season_close.py` (season-close orchestration CLI: graders → ladder →
  season report)
- Test: `tests/training/test_ladder.py`

**Interfaces:**
- Produces:
  - `compute_ladder(db, *, as_of_iteration: int) -> LadderReport` — deterministic, pure
    SQL + arithmetic over `v_lever_track_record`:
    - **Per-scope tallies** (scope = diagnosis label or (env, algo)): ≥10 graded bets
      and directional accuracy < 0.60 → level-1 **scoped demotion to shadow**
      (recorded as a `LadderAction`; per-scope evidence is also the *only* basis for
      authority expansion/re-earn — asymmetric burden per A26).
    - **Pooled lever-wide** (stratified by `coach_config`, A26): ≥12 pooled bets AND
      ≥3 seasons (N10, locked) below threshold → level-2 lever bench (reduce-only).
    - **Level 3**: no lever shows uplift over the reference season across **4 seasons**
      (N10, locked) → no-coach baseline becomes production (pillar 3) — emitted as a
      report recommendation requiring human sign-off, never auto-executed.
    - **Outlier alarm** (N11): any scope at ≤1 correct of its first 5 graded bets →
      Discord escalation, human review only, no automatic action.
  - `LadderAction` rows are written to `operator_actions` when a human approves them
    (actor = user) — the ladder computes, humans actuate, the record shows both.
  - Excluded verdicts (`excluded=true`) counted in denominators explicitly per A16.
- Consumes: V008 views, Task 19 verdicts, `operator_actions` (Task 11), N10/N11.

- [ ] **Step 1: Failing tests** — fixture track records: (a) 10 bets at 50% in one scope
  → level-1 demotion for that scope only; (b) 12 pooled bets across 3 seasons at 45% →
  level-2 flag; 12 bets in 2 seasons → no flag (both conditions required); (c) 1-of-5
  scope → alarm action, no demotion; (d) pooled evidence never produces an expansion
  action (asymmetry test); (e) determinism: same fixture → identical report dict.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Commit: `feat(training): §2.7
  benching ladder — A26 layered counting, script-graded, human-actuated`.

---

## Phase 5 — Lever-verification harness (Task 21)

### Task 21: Two-stage harness runner

**Files:**
- Create: `scripts/harness.py` (CLI: `stage1 --lever L1_reward_weights --env crypto
  --algo sac --fold N` / `stage2 --experiment-id E`)
- Create: `src/swingrl/training/harness.py`
- Test: `tests/training/test_harness.py`

**Interfaces:**
- Produces:
  - `preregister_experiment(db, *, lever, stage, environment, algorithm, fold_number,
    fold_role, pull_spec: dict, min_run_length_steps: int) -> int` — pull_spec (scripted
    direction/magnitude + expected metric/direction) is written **before any run starts**
    (§4.6); min lengths from the N5 table (module constant
    `HARNESS_MIN_STEPS: dict[algo, int]`, recomputed from real fold budgets at Step 0).
  - **Stage 1** — `run_stage1(experiment_id)`: 3 pull + 3 control runs, seed pairs via
    Task 7 (`fold_seed(algo, fold) + 100*pair_index`), `run_type='harness_stage1'`
    (quarantined by the corpus view), shortened runs at `min_run_length_steps`, models
    discarded; runs are independent → launched in parallel (process pool, container
    resources permitting — sequential fallback flag). Verdict: majority of seed pairs
    agree with `pull_spec.expected` direction AND no trade-activity collapse
    (trade_rate ≥ 0.5× the control pair's) → `passed`, with per-pair working in
    `verdict_detail`.
  - **A3 precondition enforced in code:** `run_stage1` for any L1 lever asserts the
    wrapper exposes the risk-penalty passthrough (Task 1's `RISK_PENALTY_INFO_KEY`
    integration — checked via a 1-step probe env run) — refuses to start otherwise.
  - **Stage 2** — `run_stage2(experiment_id)`: replays recorded fold situations
    (sampled from `epoch_snapshots` + intents of real seasons) to the coach with
    **production-identical prompts** (`prompt_version` + N15 production model equality
    asserted against `llm_calls` of the replayed call type); grades responses against
    the `cps_diagnosis.py` menu + fold-protection rules → `harness_replays` rows;
    `call_type='harness_replay'` keeps them out of the corpus view.
  - Fold selection per §2.3: neutral fold = the gate; chronic-failure tryout required
    when the lever's scope includes `chronic_failure` folds (true for L1 re-earn).
- Consumes: V007, Tasks 7/12/13/17, N5/N15, `run_type` quarantine tags (V002 CHECK).

- [ ] **Step 0:** Recompute N5 from the actual fold budget (P-B1 numbers) and write the
  table into `harness.py` docstring + this plan (walkthrough sign-off note).
- [ ] **Step 1: Failing tests** — (a) starting Stage 1 without a preregistered pull_spec
  raises; (b) verdict logic on fixture seed-pair outcomes (2-of-3 agree + activity holds
  → pass; activity collapse → fail regardless of direction); (c) L1 Stage 1 against a
  wrapper without the A3 passthrough refuses to run; (d) Stage-2 prompt-equality check
  rejects a mismatched prompt_version.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Update
  `docs/training/memory_meta_trainer.md` harness section. Commit: `feat(training): §2.3
  two-stage lever harness — preregistered, seed-paired, quarantined (S6)`.

---

## Phase 6 — Consolidation v2 + weakness profiles (Tasks 22–23)

### Task 22: Consolidation v2 — corpus-view input, structured claims, lifecycle scripts

Also closes M8: the duplicate memory-table DDL dies; migrations become the only DDL
source for both containers.

**Files:**
- Modify: `services/memory/memory_agents/consolidate.py` (input source + output shape)
- Create: `services/memory/memory_agents/pattern_lifecycle.py` (season-close script)
- Modify: `services/memory/db.py:88–210` (delete the DDL copy; startup asserts schema
  via the ledger instead — the fingerprint assertion pattern from Plan A Task 3)
- Test: `tests/memory/test_consolidation_v2.py`, `tests/memory/test_pattern_lifecycle.py`

**Interfaces:**
- Produces:
  - Consolidator input = `SELECT ... FROM v_consolidation_corpus` **only** (module-level
    constant `CORPUS_VIEW = "v_consolidation_corpus"`; a grep test asserts no other
    table name appears in consolidation SQL — the S4 input contract, criterion-5
    enforcement at the source).
  - Output = `patterns` rows: structured `claim` JSONB (`{"scope": {...}, "condition":
    {...}, "effect": {"metric", "direction", "magnitude_frac"}, "schema_version": 1}`,
    units per key from the A9 registry), `prompt_text` rendering alongside, category
    from the `cps_diagnosis` taxonomy, C6 §7 QA gate → `qa_passed`/`qa_checks` (QA
    criteria transcribed from the C6 review doc into a scriptable checklist; a pattern
    is prompt-eligible only when `qa_passed AND confidence ≥ 0.4 AND status='active'`).
  - `pattern_sources` written per pattern (structured records only; write-time check
    rejects quarantined sources — the §4.6 application-layer guard).
  - `pattern_lifecycle.py` (script-only writers, D-T3.7): mechanical confirmation
    (key-comparison on claims vs new graded rows → counters + confidence formula
    `confidence = confirmations / (confirmations + contradictions + 1)`); mechanical
    conflict detection (same scope+condition, opposite direction → both `conflicted` +
    shared `conflict_group_id`, excluded from prompts); resolution paths: evidence
    dominance (score ≥ 2:1 support → loser `retired`), scope split (LLM-synthesized
    QA-gated children via `split_into` edges), unresolvable (stays quarantined). Merge =
    LLM child + `merged_into` edges + union of sources, atomically with status changes.
  - Contradiction dominance (A18): if `v_consolidator_quality` shows contradictions >
    confirmations over a season → patterns withheld from prompts (the L2-bare mechanism)
    + Discord alert, pending human review. **Alert routing (user decision 2026-07-12):
    all Plan B alarms originate from `swingrl`-container scripts — this check runs in
    `scripts/season_close.py` (which has `monitoring/alerter.py` access), NOT inside the
    memory container. The memory container never sends Discord directly; the A25
    "memory container not wired" gap closes by routing, not by duplicating wiring.**
  - M8 fix: `services/memory/db.py` DDL block deleted; container start asserts
    `schema_migrations` current (shared helper — the memory container mounts the same
    check; on mismatch it refuses to serve).
- Consumes: V006/V008, Task 19's `grade_pattern_claims` owner slot, `llm_calls` writer
  (Task 17) for `consolidate_stage1/2` call types.

- [ ] **Step 1: Failing tests** — (a) corpus-only grep test; (b) a claim without a
  declared unit fails validation (A9); (c) conflicting fixture patterns both flip to
  `conflicted` and disappear from the prompt-eligible query; (d) evidence-dominance
  resolution retires the loser and writes no UPDATE to history tables (append-only
  audit); (e) memory container boot against a stale-ledger DB fails loudly.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite (both containers' tests).
  Commit: `feat(memory): consolidation v2 — corpus-view input, structured claims,
  script-graded lifecycle; M8 single-source DDL`.

### Task 23: Weakness profiles — maintenance script + doc seeding

**Files:**
- Create: `scripts/maintain_weakness_profiles.py`
- Test: `tests/training/test_weakness_profiles.py`

**Interfaces:**
- Produces: script (cron-owned, season-close; registered in `scripts/season_close.py`)
  that (a) seeds initial profiles from `.planning/research/hp-tuning-reference.md`
  content (`seed_provenance` stamped per §4.8); (b) graduates confirmed patterns
  (status active, confidence ≥ 0.6, ≥2 seasons of confirmations) into
  `weakness_profiles` versions with `weakness_evidence` rows pointing at the pattern +
  its underlying records (lineage unbroken); (c) retires trained-out weaknesses (no
  supporting evidence in 2 consecutive seasons → `status='retired'`, new version row).
  **No code path lets an LLM response mutate a profile** — the script's only LLM-free
  inputs are graded records + patterns (ownership test: `services/memory/routers/` has
  no write path to these tables; grep + negative API test).
- Consumes: V008, Task 22 patterns.

- [ ] **Steps 1–4:** RED (versioning: revision = new row, UNIQUE enforced; ownership
  negative test) → implement → GREEN + full suite. Commit: `feat(training): weakness
  profiles — script-maintained career files (D-T3.17, D-MT.6)`.

---

## Phase 7 — Era-1 training environment (Tasks 24–25; A28)

### Task 24: Decomposed turbulence observation features

Era-1 models train on what they will face live (A28). Per the adopted memo: the raw
composite never enters the observation — decomposed features do.

**Files:**
- Modify: `src/swingrl/features/turbulence.py` (component outputs), `src/swingrl/features/assembler.py`
  (layout: 1 slot → 2 per env), `src/swingrl/training/data_loader.py:237,350` (feed real
  values — the F1b training-side fix), `src/swingrl/config/schema.py`
  (`environment.era1_turbulence_features: bool`, default false until era-1 training)
- Create: `scripts/check_turbulence_redundancy.py` (the P-B6 MI/correlation check)
- Test: `tests/features/test_turbulence_components.py`, `tests/features/test_assembler_era1.py`

**Interfaces:**
- Produces:
  - Equity: `turbulence_components(returns) -> (magnitude_pctile: float,
    correlation_surprise_pctile: float)` — Kinlaw-Turkington decomposition as
    percentile-ranks over the trailing 3y window (memo row: equity obs).
  - Crypto: `(vol_zscore_signed: float, corr_change_signed: float)` — signed, never the
    multiplicative composite (memo row: crypto obs).
  - Assembler layout: the single turbulence slot becomes two per env; layout constants +
    `CRYPTO_OBS_DIM` updated; Plan A Task 7's `turbulence_obs_index` helper extended to
    `turbulence_obs_slice(env, n_symbols, sentiment_enabled) -> slice`; era-0 models
    keep the old layout + `zero_turbulence_obs=true` — **two layouts coexist keyed by
    the model's era** (from `models` → spine), and the loader picks per model. The flag
    retires automatically when no active model is era-0 (query at load, Plan A Task 7's
    noted automation).
  - `check_turbulence_redundancy.py`: prints ρ and MI between turbulence percentile and
    HMM p_crisis over our history; **ρ ≥ 0.9 → the obs feature is dropped** (halt
    unaffected) — decision recorded in the script's saved report before feature freeze
    (P-B6).
- Consumes: Plan A Task 6's rebuilt calculators (equity hygiene fixes + crypto
  2-asset Mahalanobis replacement land there; this task consumes their component
  outputs), assembler layout constants.

- [ ] **Step 0:** Run the redundancy check on homelab history; save report; STOP if
  ρ ≥ 0.9 (feature dropped, task reduces to the flag-retirement automation).
- [ ] **Step 1: Failing tests** — (a) component functions return finite values with
  correct signs on synthetic calm/spike/decoupling fixtures (the memo's two crypto
  defect cases become regression tests: dead-calm must NOT score as turbulence; a
  +0.8→−0.8 correlation flip MUST register); (b) era-1 obs shape = era-0 shape + 1 per
  env, sentiment on/off × both envs; (c) era-0 model loading still zeroes its (old
  layout) slot.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Update
  `docs/training/feature-engineering.md` (same commit). Commit: `feat(features): era-1
  decomposed turbulence observations (A28, memo-adopted)`.

### Task 25: Live-parity circuit breakers inside the training env

Era-1 policies must experience halts in training exactly as they will in production
(A28b) — a policy that never saw a halt treats the post-halt world as noise.

**Files:**
- Modify: `src/swingrl/envs/base.py` (halt state machine), `src/swingrl/config/schema.py`
  (`environment.training_breakers: bool`, default false; era-1 run configs enable)
- Test: `tests/envs/test_training_breakers.py`

**Interfaces:**
- Produces: when enabled, the env mirrors the production breaker rules from the same
  config the live risk layer reads (`risk.max_drawdown_frac`, turbulence halt at the
  Task 18-derived percentile — **one config source, no duplicated thresholds**): on
  breach, actions are forced to cash (weights → 0) for the configured cooldown bars,
  `info["breaker_active"] = True`, and the episode continues (halt ≠ termination —
  matching live). Reward flows through unchanged (the policy feels the flat period).
- Consumes: Task 24 turbulence values in-env; config risk fields; Plan A Task A's
  production semantics as the parity reference (documented side-by-side table in the
  test file header).

- [ ] **Step 1: Failing tests** — (a) synthetic price path breaching the drawdown cap →
  weights forced flat for exactly `cooldown_bars`, then released; (b) breaker disabled →
  behavior identical to today (era-0 reproducibility untouched); (c) `info` carries
  `breaker_active` both ways.
- [ ] **Step 2–4:** RED → implement → GREEN + full suite. Commit: `feat(envs):
  live-parity training breakers (A28) — era-1 policies train through halts`.

---

## Phase 8 — Acceptance, protection, cutover (Tasks 26–29)

### Task 26: S4 acceptance gate — CI tier

**Files:**
- Create: `scripts/s4_check.py` (the criteria 1–8 checker — pure script)
- Create: `tests/fixtures/synthetic_season.py` (fixture-season generator)
- Modify: `scripts/ci-homelab.sh` (new stage: ephemeral pg16 + fixture season + checker)
- Test: `tests/training/test_s4_check.py`

**Interfaces:**
- Produces:
  - `synthetic_season(db, *, iteration: int) -> None`: generates a compact but complete
    season — 2 envs × 3 algos × 3 folds of canonical runs (+1 failed attempt, +1 harness
    run for the quarantine check), epoch rows at cadence, fold/season results, L2
    intents with verdicts, one pattern with sources, calendar/trade-time rows if Plan A
    tables exist (skipped gracefully otherwise).
  - `s4_check.py`: runs criteria 1–8 (§4.11) against a given DB + iteration, exits
    non-zero on any failure, prints per-criterion working: (1) identity/era/category
    resolution, zero dangling sources; (2) source coverage of every (env, algo, fold)
    that ran; (3) sampled claim-scope vs evidence keys; (4) QA gate passed on all
    patterns (canned-consolidator mode in CI — `--canned` flag substitutes a
    deterministic consolidator stub; the QA gate on real LLM output is exercised only at
    the real gate, per A23); (5) zero harness sources; (6) declared units on every claim
    numeric; (7) trend surfaces return complete results (`v_l2_settings_history`,
    track record, season-over-season CPS); (8) grading completeness — zero intents past
    horizon unverdicted + per-fold capture-completeness assertions (epoch rows ≥
    cadence expectation, fold_results present, intent count matches advice cadence).
  - CI stage: spin ephemeral pg16 (Docker, random port, torn down always), apply all
    migrations, generate fixture season, run `s4_check.py --canned` — every commit.
  - **The real gate** (iteration 5 against real data) is an operational step *after*
    this plan: documented in Task 28's runbook — iteration 6 does not start until S4
    passes on some iter-5 attempt (A23).
- Consumes: everything from Phases 2–6.

- [ ] **Step 1: Failing tests** — checker unit tests: each criterion has one fixture
  that passes and one mutation that fails it (e.g. delete a fold's sources → criterion
  2 fails; point a source at a harness run → criterion 5 fails; strip a unit key →
  criterion 6 fails).
- [ ] **Step 2–4:** RED → implement → GREEN; run the new CI stage locally, then in
  homelab CI. Commit: `feat(training): S4 CI tier — synthetic season + criteria 1–8
  checker (D-T3.20)`.

### Task 27: Corpus protection — nightly dumps, restore drill, season-close error band

**Files:**
- Create: `scripts/backup_new_schema.sh` (pg_dump of the new-schema tables, dated,
  rotated), `scripts/restore_drill.sh` (restore to ephemeral instance + row-count
  comparison, exits non-zero on mismatch)
- Modify: `src/swingrl/scheduler/jobs.py` (nightly dump job registration),
  `scripts/season_close.py` (error-band check)
- Test: `tests/training/test_season_close_error_band.py`

**Interfaces:**
- Produces: nightly `pg_dump` (02:30 ET) of all §4 tables to the existing backup
  volume; restore drill script (run once per era, manually — documented in the Task 28
  runbook, 🛑-gated only in that it touches an ephemeral instance, not live); season
  report **fails loudly** when `llm_calls` error/timeout rate for the season exceeds
  N12 = 5% of expected advice calls (fail-open is allowed per-call; a season quietly
  advised at 60% coverage is not — the A25 band).
- Consumes: Task 17 `llm_calls` failure rows (the U3 counting), `scripts/season_close.py`.

- [ ] **Steps 1–4:** RED (error-band unit test on fixture call rows: 4% passes, 6%
  fails the report) → implement → GREEN + full suite. Commit: `feat(ops): corpus
  protection — nightly dumps + restore drill + season fail-open error band (A25)`.

### Task 28: Cutover runbook + REVOKE + gated archive-and-drop

The one place where the old world ends. Everything here is 🛑-gated and human-executed;
this task ships the *runbook and scripts*, not an autonomous cutover.

**Files:**
- Create: `docs/training/cutover-runbook.md`
- Create: `src/swingrl/data/migrations/V010__revoke_legacy_writes.sql`
- Create: `scripts/archive_and_drop_legacy.sh` (dump → verify-restore → row-count →
  DROP, each step confirmable)
- Test: `tests/data/test_migrations_content.py` (V010 content)

**Interfaces:**
- Produces:
  - **Runbook** (ordered, each step with verification): (1) no season in flight
    (no-season-mid-transition assert: zero `training_runs.status='running'`); (2) both
    containers rebuilt in lockstep (`--no-cache`, standing rule) against the migrated
    schema; fingerprint assertion proves "merged = deployed" on both; (3)
    write-verification: one synthetic row through every Phase 3 writer on live (tagged
    + deleted via documented cleanup — or written to `swingrl_test` clone when the 🛑
    gate prefers zero live writes: **default = the test-clone path**); (4) V010 REVOKE;
    (5) archive-and-drop under the full backup gate; (6) iter-5 readiness checklist
    (Task 29); (7) the S4 real gate + iteration-6 hold rule (A23); (8) rollback path:
    V010 has a documented `GRANT`-restoring counterpart script.
  - **V010**: `REVOKE INSERT, UPDATE, DELETE` from the application role on the legacy
    training tables — `memories`, `training_epochs`, `reward_adjustments`,
    `meta_decisions`, `pattern_outcomes`, `pattern_presentations` (legacy writer),
    `consolidation_sources`, `consolidation_quality`, `llm_audit_log`,
    `backtest_results`, `iteration_results`, `consolidations` — stragglers fail loudly
    (A25). `model_metadata` is **not** revoked (execution-side; Plan A owns its
    retirement). The same migration removes the Phase 3 dual-writes (legacy write calls
    deleted in the same commit — the code and the REVOKE land together).
  - **Archive-and-drop** (§4.9): `archive_and_drop_legacy.sh` dumps the ARCHIVE list
    (`memories` 4.96M, `training_epochs` 850k, `reward_adjustments`,
    `pattern_presentations`, `pattern_outcomes`, `meta_decisions`,
    `consolidation_sources`, `consolidation_quality`, `llm_audit_log`) to cold storage,
    restores into an ephemeral instance, compares row counts per table, and only then
    prints the DROP statements for **manual** execution under plan-mode approval. KEEP
    list untouched: `iteration_results`, `backtest_results` (era-0 evidence),
    `consolidations` (84 retired rows).
  - **Stage-3.5 hand-off section** in the runbook (documented, NOT implemented here):
    `REVOKE UPDATE/DELETE` on all append-only tables + separate DB roles per the §4
    writer matrix (trainer / grader / consolidator / ingest / execution) — grants
    replace conventions at Stage 3.5.
- Consumes: all prior tasks; 🛑 backup gate (standing).

- [ ] **Step 1: Failing test** — V010 content test: after apply, an INSERT into
  `training_epochs` as the app role raises; an INSERT into `epoch_snapshots` still
  succeeds; `model_metadata` unaffected.
- [ ] **Step 2–4:** RED → write V010 + scripts + runbook → GREEN. Runbook reviewed at
  the walkthrough (it is itself a 🛑 document). Commit: `feat(ops): cutover runbook +
  V010 legacy-write revoke + gated archive-and-drop (§4.9, A25)`.

### Task 29: Full verification + homelab CI + iteration-5 readiness

**Files:**
- Modify: `.planning/V1.1_EXECUTION_PLAN.md` (tracker: Plan B execution record)
- Create: `docs/training/iter5-readiness-checklist.md`

- [ ] **Step 1:** Full test suite natively (background, 10-min timeout): 0 failures.
- [ ] **Step 2:** Push branch; homelab CI per CLAUDE.md:
  `cd ~/swingrl && git fetch origin && git checkout swingrl/2.R-B-training-engine &&
  git pull && bash scripts/ci-homelab.sh --no-cache` — includes the new S4 CI stage.
  (Deploy/CI runs remain user-approved per standing rules.)
- [ ] **Step 3:** Readiness checklist (all must hold before iteration 5):
  - [ ] Phase 1 fixes verified in container (A3 probe run shows penalty in shaped
    reward; U1 stop never actuates; bench posture = all-zero deltas from config)
  - [ ] Gate v1 + era 1 approved and migrated (Task 18 report signed)
  - [ ] Graders + freshness alarm scheduled and firing on fixture data
  - [ ] Training-side Discord path proven: `run_graders.py --test-alert` delivered a
        real message, seen on homelab (execution-path Discord proof = Plan A Task 16)
  - [ ] S4 CI tier green on homelab
  - [ ] Cutover runbook executed through its REVOKE step (Task 28, 🛑-gated)
  - [ ] Reference-season run config prepared: coach-free (advice shadow-only), baseline
    HPs, `DEFAULT_WEIGHTS`, pinned per-fold seeds, era-1 env flags per approved scope,
    `run_type='reference'`
  - [ ] S2 margin (N6) recorded in the checklist as the season-close acceptance test
  - [ ] Iteration-6 hold rule acknowledged: does not start until S4 real gate passes on
    an iter-5 attempt (A23)
- [ ] **Step 4:** PR from `swingrl/2.R-B-training-engine` → `swingrl/2.R-training-redesign`
  (never `main`), with phase summary. User merges.

---

## Coverage — where every hand-off item landed

### Spec §2.11 hand-offs

| Item | Where |
|---|---|
| U3 fallback enumeration → identity-or-named-lever | Task 17 (every fallback = counted `llm_calls` row; baselines unchanged — review §6 verified all fail-open) |
| Per-fold seed-pinning feasibility | Task 7 (FEASIBLE per review; eval env seeded per M9; seed-pair replication fallback documented) |
| Startup-guard placement | Task 5 (`_on_training_start`, refuses to run) |
| Gate-replay SQL against the 564 rows | Task 18 |
| `_MAX_REWARD_DELTA` config surface (all-pairs bench) | Task 4 |
| Harness minimum run lengths per (algo, lever) | Task 21 Step 0 + N5 |
| **A3 risk-penalty fix — precondition of any L1 harness run** | Task 1 (fix) + Task 21 (enforced in code before any L1 Stage 1) |

### Spec §4.14 hand-offs (Plan B share; Plan A's share in its own coverage table)

| Item | Where |
|---|---|
| Migration + writer change-site inventory (training share) | Phase 3 preamble + Tasks 12–17 |
| Final cap values + trigger thresholds | N3/N4 + Task 6 |
| Harness min-run-length table | N5 + Task 21 |
| Archive-and-drop runbook under 🛑 | Task 28 |
| K (settings-history digest depth) yaml key | N7 + Tasks 11/17 |
| Grader orchestration + freshness alarm (A25) | Task 19 |
| Cutover runbook (REVOKE, lockstep rebuild, fingerprint, no-season rule) | Task 28 (+ Plan A Tasks 1/3 machinery) |
| Corpus protection: nightly dumps + restore drill; Stage-3.5 grants hand-off | Task 27 + Task 28 runbook section |
| Alerting routes (epoch callback + memory container Discord; season error band; calendar staleness) | Tasks 6/19/27 (calendar staleness = Plan A Task 11) |
| **L2 evidence-accrual (A26 layered counting + locked numbers)** | Task 20 (N10/N11) |
| S2 margin value (A1) | N6 + Task 29 checklist |
| Per-table index plan | Tasks 8–11 DDL (+ Plan A's trade-time subset) |
| Era-0/gate-v0 bootstrap | Plan A Task 2 (already covered there) |
| S4 isolated instance + dump restore-verification instance in CI | Tasks 26/27 |
| Key rotation precondition | DONE 2026-07-07 (Plan A Task 0) |
| `operator_actions` decision | N14 + Task 11 (INCLUDE) |
| Provider/tier + cadence per call_type | N15 + Tasks 17/21 |
| F1 re-triage (capture-quality blocker) | Plan A Tasks 6–7 (out of Plan B scope, noted) |

### Review doc §6 training-side verdicts

| Verdict | Where |
|---|---|
| F2 instrumentation + root cause | Task 6 (rate cap + docstring + rollout-cadence log) |
| Backtest trade semantics (FIFO; open-lot gap) | Task 15 (+ gate design in Task 18 uses trade_count honestly) |
| `fold_results` single-writer collapse | Task 14 |
| Wrapper MDD → equity-fraction (consequence map) | Task 5 (basis) + Task 6 (threshold) + prompt fields via Task 17 |
| Trend-window rate-cap (greenfield) | Task 6 |
| `learner_metrics` per-algo contract + SAC ent_coef | Task 13 |
| Seed pinning + M9 eval-env seeding | Task 7 |
| U3 fallbacks (fail-open, counted) | Task 17 |
| Startup guard (advice has none today) | Task 5 |
| `_MAX_REWARD_DELTA` no config surface | Task 4 |
| M8 DDL divergence → cutover input | Task 22 (single source) + Task 28 (runbook check) |
| `backtest_results` no natural key | Task 8 (`fold_results` UNIQUE run_pk) + Task 14 |
| SAC docstring wrong | Task 6 |

### Plan A "Deferred to Plan B" list

| Item | Where |
|---|---|
| Remaining §4 tables | Tasks 8–11 |
| Caps/triggers | Task 6 |
| Gate re-derivation | Task 18 |
| Harness | Task 21 |
| Graders + freshness alarms | Task 19 |
| A26 numbers | Task 20 (N10/N11 — locked values encoded) |
| Cutover runbook REVOKE step | Task 28 (V010) |
| Archive-and-drop 🛑 | Task 28 |
| Nightly dumps + restore drill | Task 27 |
| Stage-3.5 grants | Task 28 (documented hand-off) |
| S4 gate | Task 26 (CI tier) + Task 28/29 (real-gate scheduling) |
| K digest depth | N7 |
| S2 margin | N6 |
| `operator_actions` | Task 11 |
| Provider/tier table | N15 |
| Era-1 env definition (A28) incl. train-with-real-turbulence | Tasks 24–25 |
| Minimal MT grader | Task 19 (horizon-type-driven grader covers trade-time bets when Plan A Task 12 goes live) |

### Deliberately NOT in Plan B

- Full Meta-Trader design (own spec, §3.8 — gated on capture data existing).
- Stage-3.5 role/grant implementation (documented hand-off only).
- Running iteration 5 itself and the S4 real gate (operational, post-plan; runbook'd).
- Crypto stop auto-sell, crypto reconciliation (Plan A documented deferrals).
- CPS formula redesign, new algorithms, `DEFAULT_WEIGHTS` changes (§1.4 out of scope).



