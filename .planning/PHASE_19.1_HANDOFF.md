# Phase 19.1 — Memory Agent Refocus: Active Handoff

**Status** — Plan A, Plan B, Task C, Task D complete. Empirically
confirmed memory is **hurting** training (control folds outperform
treatment folds by 2.7–5.1× CPS across iter 3-4). Decision 2026-04-13:
wipe iter 5 entirely, work from the recovered iter 0-4 baseline, and
re-run iter 5 fresh after Phase 1 refocus is in place.

**Branch** — `gsd/phase-19.1-memory-agent-infrastructure-and-training`
(clean tree)

**Tests** — 975 passed / 391 skipped / 0 failed (391 are live-DB tests
that skip without `DATABASE_URL`; structural guard in
`tests/conftest.py` catches misuse before any fixture runs)

**Historical record** — see `.planning/PHASE_19.1_ARCHIVE.md`

**Original design** — `~/.claude/plans/clever-meandering-pinwheel.md`
(Phase 0 locked; Phase 1 locked; informs Group C below)

---

## Why we're doing this — empirical case

After six training iterations with memory-guided LLM advice, the memory
system has been **actively damaging training**. Treatment-vs-control
CPS comparison (iter 3-4 from restored `iteration_results`; iter 5
before the wipe):

| Env | Iter | Treatment v1 | Control v1 | Control / Treatment |
|---|---|---|---|---|
| equity | 3 | 0.01233 | 0.03427 | **2.78×** |
| equity | 4 | 0.01485 | 0.04071 | **2.74×** |
| equity | 5 (wiped) | 0.01325 | 0.04354 | **3.29×** |
| crypto | 3 | 0.08161 | 0.28397 | **3.48×** |
| crypto | 4 | 0.08318 | 0.42063 | **5.06×** |
| crypto | 5 (wiped) | 0.08211 | 0.39922 | **4.86×** |

Three iterations of evidence that control folds (no LLM advice)
outperform treatment folds by 2.7–5.1×. The Phase 1 prompt and reward
refocus is empirically mandatory.

---

## Task list

### Group A — Wipe iter 5 artifacts from pg16 and disk

Do first, while no training is running. iter 0-4 data is authoritative
after Plan A recovery; iter 5 is being abandoned and re-run fresh.

- [ ] **A1** Delete `memories` rows from iter 5 (~850k rows:
  688k crypto SAC + ~160k others). Filter by iteration tag in the source
  prefix, not by `created_at` alone, to avoid clipping legitimate iter
  0-4 rows that happened to be written late.
- [ ] **A2** Delete `training_epochs` rows from iter 5 (~850,430 rows)
- [ ] **A3** Delete `consolidations` ids 170-174 (the 5 Stage-1 patterns
  from iter 5)
- [ ] **A4** Delete `pattern_outcomes` rows 13, 14 (iter 5 replay)
- [ ] **A5** Delete `pattern_presentations` iter 5 rows (~1,835 rows)
- [ ] **A6** Delete `meta_decisions` iter 5 rows (6 rows)
- [ ] **A7** Delete `reward_adjustments` iter 5 rows (subset of 149)
- [ ] **A8** Delete model files:
  - `/app/models/iterations/iter_5/` (6 model.zip + 6 vec_normalize.pkl)
  - `/app/models/active/*` — the manual iter 5 copies placed during recovery
- [ ] **A9** Verify clean state — iter 0-4 untouched:
  - `iteration_results`: 10 rows, all CPS columns populated
  - `backtest_results`: 564 rows
  - Harm table above still reproduces for iter 3-4

Group A is a **task description**. The actual DELETE statements need
their own plan-mode approval when it's time to execute.

### Group B — Container rebuilds (data safety)

Three containers diverge from git: Phase 0 fixes (autocommit, deploy
path, CPS wiring), Task D env-prefix fix, and the dashboard Iteration
History page are all in git but NOT in the running images. Rebuilds
prevent another silent rollback during the iter 5 re-run.

- [ ] **B1** Rebuild `swingrl` image — carries the autocommit fix,
  `deploy_best_models` path fix, Phase 0.5 lifecycle logs, Phase 0.6
  Discord embed wiring, and CPS persistence wiring
- [ ] **B2** Rebuild `swingrl-memory` image — carries the Task D
  env-qualified Phase B prefix fix (equity consolidation fails without
  it)
- [ ] **B3** Rebuild `swingrl-dashboard` image — promotes the `docker
  cp`-deployed Iteration History page into the image layer
- [ ] **B4** Verify all three images are live after rebuild:
  - `swingrl` contains both `autocommit=True` and
    `compute_and_persist_iteration_cps` strings in `train_pipeline.py`
  - `swingrl-memory` `consolidate.py` contains `training_epoch:` (env-qualified)
  - `swingrl-dashboard` has `dashboard/pages/5_Iteration_History.py`
    present in the image, not just the live container

### Group C — Phase 1 prompt + reward refocus with baseline and post-change documentation

This is the original Phase 1 design from
`~/.claude/plans/clever-meandering-pinwheel.md`. The documentation
bookends (C0, C6) produce two archival artifacts used by Group D to
validate that the prompt changes actually shifted pattern quality.

Implemented on `swingrl/19.1-training-refocus` (PR #19); see spec `docs/superpowers/specs/2026-06-11-stage2-training-refocus-design.md` for amendments (C4 dropped; diagnosis layer + trade-activity indicator added beyond original scope).

- [x] **C0** Baseline documentation — **before any code changes**.
  Write `.planning/research/phase-19.1-prompt-baseline.md` capturing:
  - Current `services/memory/memory_agents/query.py::_build_system_prompt` full text
  - Current consolidation prompts (Phase A, Phase B, Stage 2 — system + few-shot)
  - Current `epoch_callback.py::_query_epoch_advice` payload schema
  - Current `meta_orchestrator.py::_query_run_config` payload schema
  - Current reward weights across all 4 files that hold them
  - 3-5 representative iter 0-4 pattern examples (raw
    `consolidations.pattern_text`) annotated with a hypothesis for how
    each may have contributed to the 2.7–5.1× CPS harm — cross-referenced
    with treat-vs-control evidence. This replaces the earlier
    "retire-harmful" design; the analysis becomes documentation used
    as QA criteria in Group D, rather than a database mutation.
- [x] **C1** LLM context enrichment:
  - `epoch_callback.py::_query_epoch_advice` — added `fold_number`,
    `fold_role`, `prev_iter_cps_v1`, `target_metric`, `leading_indicators`
    (rolling_sharpe, rolling_mdd, rolling_win_rate, trade_rate,
    baseline_trade_rate), `diagnosis` to the context JSON. Old bare
    f-string fields `rolling_sharpe=` / `rolling_mdd=` removed.
  - `meta_orchestrator.py::_query_run_config` — added context JSON with
    `target_metric`, `chronic_failure_folds`, `protected_winner_folds`,
    `prev_iter_cps_v1`, `prev_iter_diagnoses`.
- [x] **C2** New helper module
  `src/swingrl/memory/training/fold_context.py`:
  - `load_fold_context(database_url, env, fold_number)` — returns fold role +
    chronic/protected lists + prev_iter_cps_v1 from pg16
  - `record_fold_attribution(conn, run_id, fold)` — writes attribution
    closure to `reward_adjustments` 6 new columns post-fold
  - Reuses `detect_chronic_failures` / `detect_protected_winners` from
    `iteration_report.py`
- [x] **C3** Prompt updates in
  `services/memory/memory_agents/query.py` — `_build_epoch_system_prompt`
  and `_build_algo_system_prompt` now include:
  - **Goal block**: explicit "your single objective metric is CPS v1;
    pass rate is NOT your goal"
  - **Anti-pattern block**: cite iter 4-5 trade-shy collapse and
    conviction-trading regression with empirical numbers
  - **Fold-protection block**: payload-shape-aware variants for epoch
    vs run-config contexts
- [x] **C4** DROPPED — superseded by spec D2. Control wins on current
  `DEFAULT_WEIGHTS = {profit: 0.50, sharpe: 0.25, drawdown: 0.15, turnover: 0.10}`;
  base rebalance would contaminate Group E comparison. Chronic-failure
  guidance moved to the fold-protection prompt block. No `CHRONIC_FAILURE_WEIGHTS`
  constant exists. See spec §2 for full rationale.
- [x] **C5** Per-fold attribution — extended `reward_adjustments` table
  with `fold_number`, `iteration_number`, `advice_id`,
  `fold_cps_v1_before`, `fold_cps_v1_after`, `advice_was_effective`.
  Migration: `scripts/migrations/add_attribution_columns.py`.
  `outcome_sharpe` bug fixed (now stores current sharpe at resolution).
  Attribution closure via `fold_context.py::record_fold_attribution`.
- [x] **C6** Post-change documentation — wrote
  `.planning/research/phase-19.1-prompt-refocus.md` symmetric to C0:
  - New prompt full text
  - New payload schemas
  - New reward weights (unchanged from baseline per C4 drop) + rationale
  - Diff summary: what changed, why, expected CPS impact per change
- [ ] **C7** Rebuild `swingrl` + `swingrl-memory` images with Phase 1
  code (second rebuild after Group B's initial pass) — pending Group B.

### Group D — Iteration-by-iteration pattern regeneration with QA gates

Per-iteration regeneration catches prompt issues on iter 0 before
re-doing four more consolidations. Budget for 2-3 trips around the
C → D loop if early QA reveals prompt weaknesses.

- [ ] **D1** Delete all remaining iter 0-4 consolidations (clean slate
  after Group A already wiped iter 5 consolidations). Clean-slate
  regeneration is preferred over preserving and mixing old/new patterns.
- [ ] **D2** Verify per-iteration consolidation scoping mechanism
  before starting. Check whether `/consolidate` accepts an iteration
  filter; if not, fall back to time-gating by `memories.created_at`
  range or sequentially pruning later-iter memories between runs.
  Document the chosen mechanism in
  `.planning/research/phase-19.1-pattern-regeneration-qa.md` so D3-D7
  are reproducible.
- [ ] **D3** Consolidate **iter 0** memories → QA → document findings.
  QA criteria:
  1. Patterns reflect the new **goal block** (CPS v1 as objective, not
     pass rate)
  2. Patterns reflect the new **anti-pattern block** (iter 4-5
     trade-shy collapse, conviction-trading regression)
  3. Patterns reflect the new **fold-protection block** (chronic
     failures vs protected winners)
  4. Patterns do **not** reproduce any harmful behavior documented in C0.

  **If QA fails**: halt → revise Group C prompts → restart from D1.
- [ ] **D4** Consolidate **iter 1** → QA → append to doc. Additional
  criterion: patterns build coherently on iter 0 patterns.
- [ ] **D5** Consolidate **iter 2** → QA → append.
- [ ] **D6** Consolidate **iter 3** → QA → append. Additional
  criterion: patterns **must** encode the treatment-vs-control harm
  evidence (equity 2.78×, crypto 3.48×). If the patterns don't
  surface this harm, prompts failed to make it visible to the LLM —
  halt and revise.
- [ ] **D7** Consolidate **iter 4** → QA → append. Same harm-evidence
  criterion (equity 2.74×, crypto 5.06×).
- [ ] **D8** Cross-iteration final QA — verify pattern evolution iter
  0 → 4 is coherent; verify no harmful behavior from C0 reproduced
  anywhere in the set; sign off in the regeneration QA doc.

### Group E — Iter 5 fresh re-run

After Groups A-D complete, iter 5 starts from a clean baseline with the
refocused prompts, new reward weights, and regenerated iter 0-4 pattern
library.

- [ ] **E1** Kick off iter 5 from iter 4's model/state baseline with
  all Phase 1 code live
- [ ] **E2** Verify writes land in pg16 — new iter 5 rows in
  `iteration_results` (with CPS columns), `backtest_results`,
  `training_epochs`, `memories`. Autocommit fix from Group B must be
  in effect.
- [ ] **E3** Verify iter 5 models auto-deploy to `/app/models/active/`
  via the fixed `deploy_best_models` path (no manual copying)
- [ ] **E4** Capture treatment-vs-control CPS for iter 5. Extend the
  harm table above with the new row. Expectation: the harm ratio
  drops toward 1.0 (or inverts) — if it persists, Phase 1 failed to
  reverse the effect and Groups C-D need another pass.

### Group F — Tracked deferred (non-blocking)

- [ ] **F1** Turbulence column bug —
  `src/swingrl/execution/pipeline.py:537` queries a non-existent
  `turbulence` column on `features_equity` / `features_crypto`
  (verified: no `turbulence` in `postgres_schema.py`). Silent fallback
  to 0.0 means turbulence crash protection has never worked. Blocks
  live trading; not training.
- [ ] **F2** SAC epoch cadence root cause — Task C fixed the OOM
  symptom (stream-parse), not the reason SAC wrote 688k memories vs
  expected ~350. Probably SAC's off-policy callback pattern bypasses
  the cadence guard.
- [ ] **F3** `llm_audit_log.iteration_number` NULL — 801 historical
  rows lack iteration context. Out of scope for Phase 1.
- [ ] **F4** Full test suite against live postgres — spin up a local
  docker postgres, set `DATABASE_URL`, run the 391 skipped live-DB
  tests before Phase 1 lands in production.

### Group G — Open question for user

- [ ] **G1** After Group E completes, re-run `select_best_per_algo_env`
  across iter 0-5 and redeploy the actual winners? Or accept iter 5's
  fresh output as the production deployment? Defer until after the
  `llm_audit_log.iteration_number` cleanup so the selector has a clean
  training-history view.

---

## Phase 0 reference — what's been built

Commit `972567e` (Phase 0 framework). Full file inventory in
`.planning/PHASE_19.1_ARCHIVE.md`.

Key modules:
- `src/swingrl/metrics/cps.py` — three CPS formulas + FoldMetrics
  TypedDict
- `src/swingrl/reporting/iteration_report.py` — loaders, pure helpers,
  persistence orchestrator for `iteration_results`
- `scripts/backfill_cps_history.py` — idempotent CPS backfill
- `scripts/migrations/add_cps_columns.py` — additive 16-column
  migration
- `scripts/migrations/recover_iteration_results.py` — reconstructs
  iteration_results rows from backtest_results using live-pipeline math
- `scripts/migrations/restore_iter_0_4_from_duckdb.py` — Plan A
  recovery (iter 0-4 from duckdb backup)
- `dashboard/pages/5_Iteration_History.py` — CPS trend + per-fold
  heatmap + treatment-vs-control harm banner

Pre-existing bugs fixed in Phase 0 (not yet live in running containers;
see Group B):
1. Silent rollback in `train_pipeline.py` (missing `autocommit=True`)
2. Path mismatch in `deploy_best_models` — production `/app/models/active/`
   was empty the entire project before the manual iter 5 copy during recovery
3. Dashboard cached connection rot (`dashboard/app.py`)

---

## Phase 1 design spec (detail for Group C)

From `~/.claude/plans/clever-meandering-pinwheel.md`. Summaries only;
full spec in that file.

### 1.1 — LLM context enrichment (Group C1)

Expand the payload the memory service sees at epoch-advice and run-config
query time. New fields let the LLM distinguish chronic failures from
protected winners, see fold-level CPS history, and anchor on the actual
objective metric.

### 1.2 — `fold_context.py` helper (Group C2)

Centralize the fold-classification logic. `classify_fold_role` and
`load_fold_history` become the single source of truth; callers stop
reimplementing the queries in multiple places.

### 1.3 — Prompt updates (Group C3)

Three new system-prompt blocks: goal, anti-pattern, fold-protection.
The anti-pattern block cites empirical numbers from iter 4-5 so the
model is anchored on real harm cases, not abstract warnings.

### 1.4 — Reward weight rebalance (Group C4)

Unified weights — profit/sharpe/drawdown weighted equally at 0.30,
turnover at 0.10, with a dedicated `CHRONIC_FAILURE_WEIGHTS` constant
that leans harder into drawdown penalty (0.40) for chronic-failure folds.

### 1.5 — Per-fold attribution (Group C5)

`reward_adjustments` gets `fold_number`, `iteration_number`, `advice_id`,
`fold_cps_v1_before/after`, and `advice_was_effective`. This closes the
observability gap that made it hard to tell which advice helped and
which hurt.

---

## Reference

### Quick context-restoration commands (run after any context clear)

```bash
# 1. Branch and recent commits
cd /Users/varunpanchal/Documents/Projects/Simplementix/SwingRL
git branch --show-current
# Expect: gsd/phase-19.1-memory-agent-infrastructure-and-training
git log --oneline -5

# 2. Working tree
git status
# Expect: clean

# 3. Test suite (safe — conftest guard in place)
source .venv/bin/activate && uv run pytest -q
# Expect: 975 passed, 391 skipped, 0 failed

# 4. pg16 iter 0-4 baseline still intact
ssh homelab "docker exec swingrl python3 -c '
import os, psycopg
conn = psycopg.connect(os.environ[\"DATABASE_URL\"], autocommit=True)
cur = conn.cursor()
cur.execute(\"SELECT iteration_number, environment, ROUND(cps_v1_multiplicative::numeric, 5) FROM iteration_results ORDER BY iteration_number, environment\")
for r in cur.fetchall(): print(r)
cur.execute(\"SELECT count(*) FROM backtest_results\")
print(\"backtest_results:\", cur.fetchone()[0])
'"
# Expect: 10 iteration_results rows (iter 0-4 × {equity, crypto}) + 564 backtest_results

# 5. Container deployment state (before Group B rebuilds)
ssh homelab "
docker exec swingrl grep -c 'autocommit=True' /app/scripts/train_pipeline.py
docker exec swingrl-memory grep -c 'training_epoch:' /app/memory_agents/consolidate.py
docker exec swingrl-dashboard ls /app/dashboard/pages/5_Iteration_History.py
"
# Before Group B: swingrl count=0, swingrl-memory count=0, dashboard file present via docker-cp
# After Group B: swingrl count>=1, swingrl-memory count>=1, dashboard file present via image
```

### How to access pg16

pg16 is not exposed on the host. Go through the `swingrl` container:

```bash
ssh homelab "docker exec swingrl python3 -c '
import os, psycopg
conn = psycopg.connect(os.environ[\"DATABASE_URL\"], autocommit=True)
cur = conn.cursor()
cur.execute(\"<your query here>\")
for r in cur.fetchall(): print(r)
conn.close()
'"
```

For larger queries, write the script locally → `scp` to homelab → `docker cp` into the container → `docker exec python3`.

### Memory service API

Health: `GET /health` (no z). API key in `X-API-Key` header, value
from `MEMORY_API_KEY` or `SWINGRL_MEMORY_AGENT__API_KEY` env vars
inside the container.

### Dashboard

`http://172.184.1.5:8501` (homelab IP, not exposed externally). New
page at `/Iteration_History`. Auto-refreshes every 5 minutes; hard
refresh (Cmd+Shift+R) when verifying changes.

---

## Key learnings / notes

1. **Activate the venv before git operations**. Pre-commit hooks need
   ruff / mypy / bandit on PATH.
2. **Pre-commit hooks catch**: bandit asserts (B101), ruff format
   drift, mypy errors, detect-secrets (strict — even docstring URLs with
   credential-shaped substrings get flagged; describe in prose instead).
3. **`store_iteration_results_to_duckdb` is misnamed** — it writes to
   PostgreSQL. Leftover from migration. Don't rename without a dedicated
   cleanup commit.
4. **Dashboard container has no torch/SB3** — only what's in
   `dashboard/requirements.txt`. Don't add torch-dependent imports to
   dashboard pages.
5. **pg16 isn't exposed to the host** — only the `swingrl` container
   reaches it on the docker network. SSH + docker exec is the only path.
6. **Memory service health is `/health`, not `/healthz`.**
7. **Memory service API key in `X-API-Key` header** — 401 without it.
8. **iter 1 has 9 duplicate fold rows** from a mid-iteration
   restart-with-fixes. Dedup with `DISTINCT ON (...) ORDER BY ...,
   created_at DESC` — later row is the keeper.
9. **Recovery script produces byte-identical output** to the original
   training pipeline (crypto iter 5 sharpe=4.8128, mdd=0.1412 from both
   sources). Trust it.
10. **Plan files**:
    - `~/.claude/plans/clever-meandering-pinwheel.md` — Phase 0/1 design
    - `~/.claude/plans/delegated-skipping-umbrella.md` — Plan A recovery
11. **CLAUDE.md plan-first workflow is in effect** — never edit files
    without explicit approval. New design work needs plan mode.
12. **User prefers verbose, evidence-rich responses**, especially when
    delivering regressions. Don't sugarcoat. Show the numbers.
13. **Always present times in ET** (per user auto-memory feedback).
14. **Streamlit file watcher does not reliably pick up `docker cp`
    changes** — restart the container to force a clean reload. Survives
    restart, not recreation.
15. **Dashboard auto-refreshes every 5 min** — hard-refresh the browser
    to verify changes immediately.
16. **Historical training context** lives in `memory/`:
    - `project_iter2_training_session.md` — iter 2 setup
    - `project_iter3_analysis.md` — iter 3 deep analysis (gamma=0.999
      A2C regression, 6 poisoned patterns retired)
    - `feedback_*.md` — user preferences
    Read these before doing pattern regeneration (Group D) — prior art
    on what was retired and why.
17. **`gsd:` tooling exists** — see `.planning/` and `gsd:*` slash
    commands. Do not run `gsd:complete-milestone` or other gsd commands
    without user approval; they're destructive of planning state.
18. **Memory service consolidation timeout is 1800s (30 min)** per
    config. The OOM before Task C's fix happened during Stage-1
    aggregation, well before timeout. OOM kills the process; service
    auto-restarts.
19. **The `swingrl` container runs the scheduler** (apscheduler) every
    5 min for `automated_trigger_check_job`. Recreating the container
    kills both the scheduler and any in-flight training — do not
    recreate during a training run.
20. **iter 5 had been started as iteration 5 of a 6-iteration budget**
    (`total=6` in the `iteration_complete` event). After the wipe, iter
    5 re-run (Group E) resumes the same budget — still one iteration
    short of plan.

---

## See also

- `.planning/PHASE_19.1_ARCHIVE.md` — full historical record (STEP 0,
  incident narrative, Plan A R1-R7, Plan B/C/D postmortems, original
  23-file fixture audit, Phase 0 validation gate results)
- `.planning/research/` — per-topic research docs; Groups C0, C6, and
  D will add `phase-19.1-prompt-baseline.md`,
  `phase-19.1-prompt-refocus.md`, and
  `phase-19.1-pattern-regeneration-qa.md`
- `~/.claude/plans/clever-meandering-pinwheel.md` — locked Phase 0/1
  design
- `~/.claude/plans/delegated-skipping-umbrella.md` — Plan A recovery

---

## End of active handoff
