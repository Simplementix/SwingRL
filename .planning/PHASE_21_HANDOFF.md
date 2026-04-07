# Phase 21 — Memory Agent Refocus: Context Handoff

**Created:** 2026-04-07 ~08:00 ET (snapshot before context clear)
**Branch:** `gsd/phase-20-production-deployment`
**Last commit:** `c7943bc feat(21): Phase 0 — CPS framework + measurement infrastructure`
**Plan file:** `/Users/varunpanchal/.claude/plans/clever-meandering-pinwheel.md`

This document is a complete handoff for resuming Phase 21 work after a
context clear. **Read this top to bottom before doing anything.**

---

## TL;DR — Where we are right now

We are mid-execution of **Phase 21 (memory agent refocus)**. Phase 0
(measurement infrastructure) is **DONE and committed** as
`c7943bc`. Iter 5 has finished training, and we discovered the memory
system has been actively *hurting* training (control folds outperform
treatment folds by 2.7-5.1× CPS across iter 3, 4, 5). We have **3
pre-existing bugs fixed**, **iter 5 fully recovered**, and **15
roadmap-style tasks completed**.

Remaining queue (in order, set by user):
1. ~~A — Commit Phase 0 work~~ ✅ done (`c7943bc`)
2. **C — Fix crypto SAC epoch memory volume bug** ← next
3. D — Investigate equity `consolidation_skipped_no_memories`
4. iter5 QA — review memories/patterns, cleanup harmful patterns
5. B — Phase 1 (prompt + reward weight refocus)

---

## Why we're doing this — the empirical case

After 6 training iterations with memory-guided LLM advice, the memory
system has been **actively damaging training**. Treatment-vs-control
CPS comparison (iter 3-5, all numbers from `iteration_results` after
Phase 0.4 backfill):

| Env    | Iter | Treatment v1 | Control v1 | Control / Treatment |
|--------|------|--------------|-----------|--------------------|
| equity | 3    | 0.01233      | 0.03427   | **2.78×**          |
| equity | 4    | 0.01485      | 0.04071   | **2.74×**          |
| equity | 5    | 0.01325      | 0.04354   | **3.29×**          |
| crypto | 3    | 0.08161      | 0.28397   | **3.48×**          |
| crypto | 4    | 0.08318      | 0.42063   | **5.06×**          |
| crypto | 5    | 0.08211      | 0.39922   | **4.86×**          |

**Three iterations of empirical evidence that control folds (no LLM
advice) outperform treatment folds by 2.7-5.1×.** The Phase 1 prompt
and reward refocus is now empirically mandatory.

Iter 5 specifically tripped the regression flag for both envs:

| Env    | CPS v1 (iter 4 → 5) | Median return delta | Status      |
|--------|---------------------|---------------------|-------------|
| equity | 0.01526 → 0.01401 (-8%)  | 6.62% → 5.85% (-0.77pp) | ⚠️ REGRESSION |
| crypto | 0.12361 → 0.08077 (-35%) | 60.16% → 37.79% (-22.4pp) | ⚠️ REGRESSION |

---

## What's been built (Phase 0 complete)

Commit `c7943bc` lands 24 files / 4,828 insertions / 39 deletions.

### New modules
| File | Purpose |
|------|---------|
| `src/swingrl/metrics/cps.py` | Three pure CPS formulas (multiplicative v1, additive v2, sortino-anchored v3). FoldMetrics TypedDict maps to backtest_results columns. Unit-converts max_single_loss from dollars to fraction at the boundary (`_BACKTEST_INITIAL_CAPITAL = 100_000.0`). |
| `src/swingrl/reporting/iteration_report.py` | First reader of `iteration_results` in production code. Loaders + pure helpers + persistence orchestrator. Key functions: `load_iteration_history`, `load_fold_history` (DISTINCT ON dedup for iter 1), `compute_iter_deltas`, `detect_chronic_failures`, `detect_protected_winners`, `compute_iteration_cps`, `persist_iteration_cps`, `compute_and_persist_iteration_cps`. |
| `scripts/backfill_cps_history.py` | Idempotent CPS backfill for any iteration range. Run via `--max-iter N`. Currently iter 0-5 are populated. |
| `scripts/migrations/add_cps_columns.py` | Additive 16-column migration for iteration_results. Already applied to pg16. |
| `scripts/migrations/recover_iteration_results.py` | One-shot recovery script that reconstructs iteration_results rows from backtest_results using the same softmax/gate math as the live pipeline. Used to recover iter 5. |
| `dashboard/pages/5_Iteration_History.py` | Streamlit page with iteration table, CPS trend chart, per-fold heatmap, treatment-vs-control panel with smoking-gun harm banner, regression panel. |
| `tests/metrics/test_cps.py` | 21 CPS unit tests including the empirical iter 4 → iter 5 A2C regression case. |
| `tests/reporting/test_iteration_report.py` | 38 iteration_report tests (9 are live DB and skipped without DATABASE_URL). |
| `tests/monitoring/test_iteration_embed.py` | 16 Discord embed tests covering color logic + treatment/control display + edge cases. |
| `tests/data/test_iteration_results_extension.py` | 19 schema migration tests including parametrized column presence + idempotent migration. |

### Modified files
| File | What changed |
|------|--------------|
| `src/swingrl/data/postgres_schema.py` | Added 16 nullable columns to `iteration_results` DDL. |
| `src/swingrl/monitoring/embeds.py` | New `build_iteration_completion_embed` function with green/yellow/red color logic. |
| `scripts/train_pipeline.py` | (1) Added Phase 0.5 lifecycle log events + Phase 0.6 Discord embed wiring after `store_iteration_results_to_duckdb`. (2) **Bug fix:** added `autocommit=True` to the iteration_results connection at line ~2295. (3) **Bug fix:** `deploy_best_models` now reads from `models/iterations/iter_N/active/{env}/{algo}/` instead of the legacy path. |
| `dashboard/app.py` | **Bug fix:** `get_pg_conn` now self-heals when the cached singleton was closed by another page (pages 1-4 all close it). |
| `dashboard/Dockerfile.dashboard` | Copies `src/swingrl/{reporting,metrics}` into the dashboard image. |
| `dashboard/requirements.txt` | Added `structlog>=24.1`. |
| `tests/dashboard/test_pages.py` | Added `TestIterationHistoryHelpers` (5 tests) + `TestGetPgConnSelfHeal` (2 tests) + parse test for the new page. |
| `tests/agents/test_backtest.py` | Added 3 regression tests for the iteration_results rollback bug (2 behavioral against pg16 + 1 static source check). |
| `tests/training/test_train_pipeline.py` | Fixed `_create_iter_models` fixture to match canonical path layout. |

### Pre-existing bugs surfaced and fixed

1. **Silent rollback bug** (`scripts/train_pipeline.py` ~L2295): The
   iteration_results connection was opened without `autocommit=True`,
   `store_iteration_results_to_duckdb` never called `commit()`, and
   `finally: conn.close()` rolled back the INSERT. **Live since the
   postgres migration.** Iter 0-4 rows in pg16 came from the one-time
   migration script, NOT from training. **Fixed.**

2. **Path mismatch in `deploy_best_models`** (`scripts/train_pipeline.py`
   ~L208): Selector looked at `models/iterations/iter_N/{env}/{algo}/`
   but the trainer writes to `models/iterations/iter_N/active/{env}/{algo}/`.
   **Live since iter 0.** Production active path was empty the entire
   project, causing `crypto_cycle_failed: No actions to blend`. **Fixed
   + iter 5 manually deployed to `models/active/`.** Test fixture was
   matching the buggy path, so the test wasn't catching it. Fixture
   corrected.

3. **Dashboard cached connection rot** (`dashboard/app.py`): Pages 1-4
   all call `conn.close()` at the end, closing the `@st.cache_resource`
   singleton. New Iteration History page hit it after Risk Metrics and
   crashed. **Fixed** with self-heal — `get_pg_conn` now detects a
   closed cached connection, clears the cache, and reopens.

---

## Iter 5 recovery — what was lost, what was recovered

### What we have (verified against pg16)
- ✅ `backtest_results` for iter 5: 111/111 rows (equity 23×3 + crypto 14×3)
- ✅ `training_epochs`: 850,430 rows for Apr 6-7
- ✅ `reward_adjustments`: 149 rows
- ✅ `meta_decisions`: 6 rows (one per env×algo)
- ✅ `memories`: 850,748 rows (raw text snapshots)
- ✅ `pattern_presentations`: 1,835 rows
- ✅ All 6 model.zip files in `/app/models/iterations/iter_5/active/{env}/{algo}/`
- ✅ JSON reports: `/app/data/{training_report,training_comparison}.json`
- ✅ Stage-1 consolidations: 5 patterns (172, 173, 174 equity; 173-174 crypto). One pattern explicitly says **"Control folds for PPO show a mean Sharpe of 3.8606, while treatment folds regress..."** — the LLM independently caught the smoking gun.
- ✅ `iteration_results` for iter 5: **RECOVERED** via `recover_iteration_results.py` (equity sharpe=2.34 gate_passed; crypto sharpe=4.81 gate_passed — byte-identical to original log)
- ✅ CPS columns for iter 5: **POPULATED** via backfill (regression flag set for both envs)
- ✅ `pattern_outcomes` for iter 5: **REPLAYED** via `MemoryClient.record_outcome()` — rows 13 (crypto) + 14 (equity)

### What's still missing (and why)
- ❌ **Stage-2 cross-env consolidations for iter 5**: blocked on the
  upstream crypto SAC memory volume bug — memory service OOMs trying
  to aggregate 831,907 crypto epoch memories. **This is the next task (C).**
- ❌ **`llm_audit_log.iteration_number`**: pre-existing bug (not iter 5
  specific). All 801 historical rows have iteration_number=NULL. The
  audit logger doesn't pass iteration through. Out of scope for now.
- ❌ Equity `pipeline_env_complete` log line: rotated out of retained
  log files. Cosmetic only.

---

## Database state (verified 2026-04-07)

**Postgres = pg16** (NOT duckdb anymore — `store_iteration_results_to_duckdb`
is a misnomer left over from migration; it writes to PostgreSQL).

```
iteration_results (12 rows after Phase 0 backfill):
  iter 0  equity   sharpe=1.896  cps_v1=0.01167
  iter 0  crypto   sharpe=4.196  cps_v1=0.15313
  iter 1  equity   sharpe=1.899  cps_v1=0.01325  dedup_dropped=9 (restart-with-fixes)
  iter 1  crypto   sharpe=3.547  cps_v1=0.09674
  iter 2  equity   sharpe=2.010  cps_v1=0.01133  ⚠ regression
  iter 2  crypto   sharpe=3.366  cps_v1=0.06683  ⚠ regression
  iter 3  equity   sharpe=1.895  cps_v1=0.01338
  iter 3  crypto   sharpe=3.098  cps_v1=0.09589
  iter 4  equity   sharpe=2.065  cps_v1=0.01526
  iter 4  crypto   sharpe=3.887  cps_v1=0.12361
  iter 5  equity   sharpe=2.340  cps_v1=0.01401  ⚠ regression  ← recovered + backfilled
  iter 5  crypto   sharpe=4.813  cps_v1=0.08077  ⚠ regression  ← recovered + backfilled
```

---

## How to access pg16

**The DATABASE_URL is set inside the swingrl container**. Don't try to
connect from your Mac directly — pg16 isn't exposed. Use SSH + docker exec:

```bash
ssh homelab "docker exec swingrl python3 -c '
import os, psycopg
conn = psycopg.connect(os.environ[\"DATABASE_URL\"], autocommit=True)
cur = conn.cursor()
cur.execute(\"SELECT iteration_number, environment, cps_v1_multiplicative FROM iteration_results ORDER BY iteration_number\")
for r in cur.fetchall():
    print(r)
conn.close()
'"
```

For larger queries, write to `/tmp/script.py` locally, scp to homelab,
docker cp into the swingrl container, then docker exec python3.

The memory service API key (for /consolidate, /training/record_outcome):
- `SWINGRL_MEMORY_AGENT__API_KEY` env var inside the container
- Or `MEMORY_API_KEY` env var

---

## How to access dashboard

Dashboard is at **http://172.184.1.5:8501** (homelab IP — NOT exposed
externally without VPN). Iteration History is at `/Iteration_History`.

The dashboard container has Phase 0 changes installed via `docker cp`
into /app — they survive restart but are NOT in the production image
yet. Next `docker compose build swingrl-dashboard` will rebuild from
the Dockerfile changes that ARE in git.

---

## Validation gate — all 8 criteria PASSED

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Iter 5 A2C shows regression flag | ✅ flagged on equity |
| 2 | Iter 5 PPO `cps_v1_delta` lateral (±0.003) | ✅ -0.00126 |
| 3 | Chronic failures = `[2, 4, 7, 13, 15]` | ✅ confirmed |
| 4 | Protected winners non-empty | ✅ `[1, 8, 10, 16, 20, 22]` |
| 5 | CPS v1 trend across iter 0-5 | ✅ 0.01167 → 0.01526 → 0.01401 |
| 6 | Discord embed renders correctly | ✅ verified |
| 7 | `dedup_rows_dropped = 9` for iter 1 equity | ✅ confirmed |
| 8 | Treatment/control split renders for iter 3+ | ✅ confirmed (3.29-5.06× harm) |

---

## Test status

- **Full suite collected: 1365 tests**
- **Last full run: 975 passed, 390 skipped, 0 failed** (Apr 7, 49 sec wall time)
- The 390 skipped are mostly live-DB tests that need `DATABASE_URL` set
  pointing at a running postgres
- I did NOT spin up a docker postgres for the live integration tests —
  this is a gap. **TODO before Phase 1 deploy: run live tests with a
  real postgres.**

To run the full suite:
```bash
source .venv/bin/activate
uv run pytest -q
```

To run live DB tests, either:
- Start a docker postgres locally and export `DATABASE_URL` pointing at it (postgres connection string)
- Or run tests inside the swingrl container via `docker exec swingrl python3 -m pytest /tmp/test_X.py`

---

# REMAINING WORK QUEUE (in order)

## Task C — Fix crypto SAC epoch memory volume bug ⏳ NEXT

### Symptom
Crypto SAC iter 5 produced **688,899** epoch memories (out of 850,748
total memories for Apr 6-7). The configured cadence is 40000 steps.
With 1M timesteps per fold and 14 folds, expected ~350 snapshots, not
~50,000. The volume causes the memory service to OOM during Stage-1
consolidation aggregation, blocking Stage-2 cross-env consolidation.

### Where the bug lives
- `src/swingrl/memory/training/epoch_callback.py` (the `MemoryEpochCallback`)
- Cadence config in `config/swingrl.yaml` and per-algo defaults in epoch_callback.py
- Per the project memory: PPO=60, A2C=8000, SAC=40000 are the configured cadences

### Investigation plan (do these in order)
1. Read `src/swingrl/memory/training/epoch_callback.py` `_on_rollout_end` and the epoch counter logic. Look for whether the cadence is checked correctly for SAC (which uses `_n_calls` from BaseCallback rather than rollout-end events).
2. Query pg16 to characterize the crypto SAC memory volume per fold:
   ```sql
   SELECT source, count(*), MIN(created_at), MAX(created_at)
   FROM memories
   WHERE source LIKE 'training_epoch:crypto:sac'
     AND created_at >= '2026-04-06'
   GROUP BY source;
   ```
3. Check `training_epochs` table for the actual epoch values stored —
   are there 50k rows for crypto SAC iter 5? If yes, the callback is
   firing per-step. If no, the discrepancy is in the memories table only.
4. Check if SAC's `_on_rollout_end` is even called (SAC is off-policy
   and may use a different SB3 callback hook).
5. Diff the cadence handling between PPO/A2C (working) and SAC (broken).

### Likely root cause hypotheses (untested)
- **H1:** SAC doesn't fire `_on_rollout_end` the same way as PPO/A2C, so the cadence guard never trips. The callback falls back to per-step memory writes.
- **H2:** Cadence is loaded from yaml but the SAC value isn't being applied correctly (silent default fallback).
- **H3:** A separate code path (e.g., `update_locals` or `_on_step`) writes memories outside the cadence.

### Fix gates
- All `tests/memory/test_epoch_callback*.py` must pass
- Add a regression test that proves SAC respects its cadence
- Run a synthetic SAC training inside a tmp env with cadence=10 and
  verify exactly N memories are written
- After fix: re-run iter 5 backfill (no, just clean up the old crypto
  SAC memories — they're already in pg16 and we don't want to retrain)

### What needs to happen to recover
1. Fix the bug
2. Decide what to do with the existing 688k crypto SAC memories:
   a. Delete them (cleanest — they're junk per-step snapshots)
   b. Retain but mark as `archived=1`
   c. Sample down to a reasonable count
3. Re-run `/consolidate` on the memory service — should now succeed
   because crypto memory volume is sane
4. Verify Stage-2 cross-env consolidations land in `consolidations` table

---

## Task D — Investigate equity `consolidation_skipped_no_memories`

### Symptom
When the memory service tried to consolidate iter 5 equity at 03:07:50
(and again on the manual replay), it logged `consolidation_skipped_no_memories
env_name=equity`. But pg16 contains ~18,617 equity epoch memories from
Apr 6-7.

### Question
Why does the consolidator think there are no equity memories? Possible:
- **H1:** Already-consolidated marker — the service tracks which memory
  IDs have been consolidated and skips them. Iter 4 may have processed
  all equity memories, leaving none unprocessed for iter 5.
- **H2:** Date filter excludes them — the consolidator may filter by
  `created_at` and the iter 5 equity memories are outside the window.
- **H3:** Source filter is wrong — the consolidator looks for source
  patterns like `training_epoch:equity:*` and isn't matching.

### Investigation plan
1. Read the memory service consolidation code at
   `services/memory/memory_agents/consolidate.py` (or similar).
2. Look for "no memories" log emit and trace back to the query.
3. Run that exact query against pg16 to see what it returns for equity.
4. Compare against the crypto query path which DID find 831k memories
   (and crashed). What's different about equity?

### Connection to Task C
Tasks C and D are likely linked. The crypto path finds **too many**
memories and the equity path finds **zero**. Either:
- The consolidator's source filter is broken (matches crypto sac
  but not equity)
- OR the iter 5 equity memories were marked processed by an earlier
  consolidation

---

## Task — Iter 5 QA: review memories, patterns, cleanup harmful patterns

### Why this matters
Three iterations of empirical evidence say memory is HURTING training.
Before Phase 1 (which makes the LLM see fold roles + protected winners),
we need to clean out actively harmful patterns that would otherwise
keep influencing the LLM.

### What to review
1. **All 5 Stage-1 consolidations from iter 5** (ids 170, 171, 172,
   173, 174). Read each `pattern_text`. For each, decide:
   - Is this empirically supported by iter 5 data?
   - Could this pattern reinforce the treatment-hurts-training behavior?
   - Should it be marked `status='archived'` or `status='retired'`?
2. **All consolidations referenced in iter 4-5 `pattern_presentations`**
   (the patterns the LLM actually saw during training). Same review.
3. **Cross-iteration pattern history** — query `consolidations`
   ordered by `created_at` to see the full lineage. Any pattern that
   was active across multiple iterations during which CPS regressed is
   a candidate for retirement.

### Specific patterns to investigate
- **id=170 (equity, stage 1)**: "Control folds for PPO show a mean Sharpe of 3.8606, while treatment folds regress..." — this is the LLM correctly identifying the smoking gun. **KEEP** — this is evidence the model can see the truth when given the right data.
- **id=171 (equity, stage 1)**: "A2C's learning_rate=0.00015 is at the lower bound of the safe range [1e-4, 5e-4]..." — likely an HP tuning suggestion. Review whether it's the kind of advice that pushed A2C into the trade-shy collapse.
- **id=172 (equity, stage 1)**: "SAC's high_vix trades (>1.5σ) show a win_rate=0.761 and avg_pnl=12.3577..." — SAC-specific. Probably benign but verify.
- **id=173 (crypto, stage 1)**: "SAC algorithm exhibits extreme performance degradation in negative yield spread..." — review for potentially dangerous "avoid trading in X regime" type guidance.
- **id=174 (crypto, stage 1)**: "PPO control folds ([CTRL]) consistently outperform treatment folds ([TREATMENT])..." — similar to 170. **KEEP** — the model is seeing the truth.

### Tools to use
- The dashboard's Iteration History page (now live) for visual review
- Direct SQL against the `consolidations` and `pattern_presentations` tables
- Look for patterns that reference HP changes, reward weight changes,
  or behavioral nudges that align temporally with the regression flags

### Output
A list of pattern IDs to retire/archive, plus a justification doc
saved to `.planning/research/iter5-pattern-cleanup.md`.

---

## Task B — Phase 1: Prompt + reward weight refocus

This is the original Phase 1 from the planning doc at
`/Users/varunpanchal/.claude/plans/clever-meandering-pinwheel.md`.
Read that file for the full design. Summary:

### 1.1 — LLM context enrichment
- `epoch_callback.py::_query_epoch_advice()`: add `fold_number`, `fold_role`, `fold_history` (last 6 iters this fold), `hmm_regime`, `vix_mean`, `chronic_failure_folds`, `protected_winner_folds`, `prev_iter_cps_v1` to the payload
- `meta_orchestrator.py::_query_run_config()`: add `fold_role`, `prev_iterations` (last 3), `chronic_failure_folds`, `protected_winner_folds`, `target_metric=cps_v1_multiplicative`, explicit goal text

### 1.2 — New `fold_context.py` helper
- `src/swingrl/memory/training/fold_context.py`
- `classify_fold_role(env, fold_number) -> "chronic_failure" | "protected_winner" | "neutral"`
- `load_fold_history(env, fold_number, n_iters=6)`
- Reuses `detect_chronic_failures` and `detect_protected_winners` from `iteration_report.py`

### 1.3 — Prompt updates (`services/memory/memory_agents/query.py`)
Three new blocks in `_build_system_prompt`:
- **Goal block:** explicit "your single objective metric is CPS v1, pass rate is NOT your goal"
- **Anti-pattern block:** cite iter 4-5 trade-shy collapse + conviction-trading regression with empirical numbers
- **Fold-protection block:** if `fold_role == protected_winner` return baseline; if `chronic_failure` recommend regime-conditional shaping

### 1.4 — Reward weight rebalance
Four files, all unified to `{profit:0.30, sharpe:0.30, drawdown:0.30, turnover:0.10}`:
- `services/memory/memory_agents/query.py` (`_SAFE_DEFAULTS`)
- `services/memory/memory_agents/bounds.py` (`_FALLBACK_REWARD_BOUNDS`)
- `src/swingrl/memory/training/reward_wrapper.py` (`DEFAULT_WEIGHTS`)
- `config/swingrl.yaml` (`training.bounds.reward_bounds`)

Plus a new `CHRONIC_FAILURE_WEIGHTS = {profit:0.20, sharpe:0.30, drawdown:0.40, turnover:0.10}` constant.

### 1.5 — Per-fold attribution
Extend `reward_adjustments` table with `fold_number`, `iteration_number`, `advice_id`, `fold_cps_v1_before/after`, `advice_was_effective`. Wire into epoch_callback at advice time + post-fold.

---

# RESUMING AFTER CONTEXT CLEAR

When you clear context, the next session needs to know:

## Where to find this plan
**`.planning/PHASE_21_HANDOFF.md`** — this file. Read it top to bottom first.

## Where to find the original Phase 0/1 design
`/Users/varunpanchal/.claude/plans/clever-meandering-pinwheel.md` —
the locked-in design that Phase 0 implemented.

## Auto-memory pointer
The new session should also load the project's auto-memory at:
`/Users/varunpanchal/.claude/projects/-Users-varunpanchal-Documents-Projects-Simplementix-SwingRL/memory/MEMORY.md`

That file should be updated to point at this handoff doc. **TODO when
saving this file: also update MEMORY.md to add a pointer.**

## Quick context restoration commands

After clearing, run these to verify state:

```bash
# 1. Check git is at the right commit
cd /Users/varunpanchal/Documents/Projects/Simplementix/SwingRL
git log --oneline -3
# Expect: c7943bc feat(21): Phase 0 — CPS framework + measurement infrastructure

# 2. Check working tree is clean
git status

# 3. Verify the test suite still passes
source .venv/bin/activate
uv run pytest -q
# Expect: 975 passed, 390 skipped, 0 failed

# 4. Verify pg16 has the recovered iter 5 rows
ssh homelab "docker exec swingrl python3 -c '
import os, psycopg
conn = psycopg.connect(os.environ[\"DATABASE_URL\"], autocommit=True)
cur = conn.cursor()
cur.execute(\"SELECT iteration_number, environment, cps_v1_multiplicative, ensemble_sharpe FROM iteration_results WHERE iteration_number IN (4, 5) ORDER BY iteration_number, environment\")
for r in cur.fetchall():
    print(r)
'"
# Expect 4 rows: iter 4-5 × equity+crypto, all with non-NULL CPS values

# 5. Verify dashboard is healthy
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://172.184.1.5:8501/Iteration_History
# Expect: HTTP 200

# 6. Verify production active models exist (manually deployed during recovery)
ssh homelab "docker exec swingrl bash -c 'find /app/models/active -name model.zip | wc -l'"
# Expect: 6
```

## Working state
- Branch: `gsd/phase-20-production-deployment` (despite the name; Phase 21 work is on this branch)
- Working tree: should be clean after this commit
- TaskList: see the snapshot below — the tasks I created in the last
  session won't survive context clear, so re-create them from this list

## Tasks to recreate in next session
The TaskCreate state doesn't persist across context clears. Re-create
these in priority order:

1. **Task C** — Fix crypto SAC epoch memory volume bug (status: pending)
2. **Task D** — Investigate equity consolidation_skipped_no_memories (status: pending)
3. **Iter 5 QA** — Review memories and patterns, cleanup harmful patterns (status: pending)
4. **Task B / Phase 1.1** — LLM context enrichment in epoch_callback + meta_orchestrator (status: pending)
5. **Phase 1.2** — New fold_context.py helper module (status: pending)
6. **Phase 1.3** — Prompt updates in query.py (status: pending)
7. **Phase 1.4** — Reward weight rebalance across 4 files (status: pending)
8. **Phase 1.5** — Per-fold attribution in reward_adjustments (status: pending)

---

# KEY LEARNINGS / NOTES FOR THE NEXT SESSION

1. **Always activate the venv before git operations**: pre-commit hooks
   need ruff/mypy/bandit on PATH. Run
   `source /Users/varunpanchal/Documents/Projects/Simplementix/SwingRL/.venv/bin/activate`
   first or inline it in commit commands.

2. **Pre-commit hooks WILL catch**: bandit asserts (B101), ruff format
   drift, mypy errors, secret detection. The detect-secrets hook is
   strict — even text that LOOKS like a connection string in a
   docstring will be flagged as basic-auth credentials. When you need
   to document database URLs, write them with the auth section omitted
   entirely (e.g., describe in prose: "the postgres URL with user,
   password, host, and database") rather than embedding any
   credential-shaped literal.

3. **`store_iteration_results_to_duckdb` is misnamed** — it writes to
   PostgreSQL, not DuckDB. The name is a leftover from migration. Don't
   rename it without a separate cleanup commit (touches the live write
   path of the running training pipeline).

4. **The dashboard container doesn't have torch/SB3** — only what's in
   `dashboard/requirements.txt`. The `Dockerfile.dashboard` now copies
   `src/swingrl/{reporting,metrics}` so the new page can import them
   without pulling the full src tree. Don't add torch-dependent imports
   to the dashboard page.

5. **pg16 isn't exposed to the host** — only the swingrl container can
   reach it on the docker network. SSH + docker exec is the only path.

6. **Memory service health check is `/health` (no z)** — not `/healthz`.

7. **The memory service API key needs to be in the X-API-Key header** —
   401 if missing. Read from `MEMORY_API_KEY` or
   `SWINGRL_MEMORY_AGENT__API_KEY` env vars.

8. **iter 1 has 9 duplicate fold rows** from a restart-with-fixes
   (post-fix run replaces pre-fix run). Dedup with `DISTINCT ON (...)
   ORDER BY ..., created_at DESC` — the LATER row is the keeper.

9. **Recovery script has been verified to produce byte-identical
   output** to the original training pipeline (crypto iter 5
   sharpe=4.8128 mdd=0.1412 from both sources). When using
   `recover_iteration_results.py`, you can trust it.

10. **The plan file location** for the original Phase 0/1 design is
    `~/.claude/plans/clever-meandering-pinwheel.md`. It's been approved
    and Phase 0 is locked in. Phase 1 design is in there too.

11. **CLAUDE.md plan-first workflow is in effect**: never edit files
    without entering plan mode first and getting explicit approval.
    The user is strict about this. Recovery work was implicitly
    pre-approved by the "proceed" after the 7-step recovery plan was
    presented, but new design work needs plan mode.

12. **The user prefers verbose, evidence-rich responses**, especially
    when delivering bad news (regressions). Don't sugarcoat. Show the
    numbers.

13. **Always present times in ET** (per the user's explicit feedback
    in their auto-memory).

---

# OPEN QUESTIONS FOR THE USER

These are things I would have asked if we weren't clearing context:

1. **Iter 5 model deployment**: I copied iter 5 models to
   `/app/models/active/` as the recovery step, but these may not be
   the "best" by Sortino. Should the next session re-run
   `select_best_per_algo_env` against all iterations and redeploy the
   ACTUAL winners? Or accept iter 5 as the production deployment
   (matches what training would have done if the path bug hadn't
   silently failed)?

2. **Test suite gap**: I should have run the full test suite with a
   docker postgres for live integration tests before committing. We
   have 390 skipped tests that may be hiding regressions. **Should
   the next session spin up a docker postgres locally and run the
   live tests as a backstop before doing more invasive work?**

3. **Pre-existing `llm_audit_log.iteration_number` NULL issue**: 801
   historical rows have NULL iteration_number. Out of scope for now,
   but worth flagging to the user. Should we file it as a tracked
   issue or leave it as a known degradation?

4. **Crypto SAC memory volume bug — fix vs. delete-and-fix**: When we
   fix the bug in Task C, do we delete the existing 688k bad memories
   from pg16? They're junk per-step snapshots that bloat the table
   and (more importantly) keep causing the memory service OOM if
   anything tries to consolidate them. Recommended: yes, DELETE them
   with `DELETE FROM memories WHERE source LIKE 'training_epoch:crypto:sac' AND created_at >= '2026-04-06'`.

---

# END OF HANDOFF

Last updated: 2026-04-07 ~08:00 ET, before context clear.
Author: Claude (current session, opus-4-6-1m).
Next session: read this top to bottom, then check the
[Resuming After Context Clear](#resuming-after-context-clear) section.
