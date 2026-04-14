# Phase 19.1 — Historical Archive

Historical record of completed Phase 19.1 work. Active tasks live in
`.planning/PHASE_19.1_HANDOFF.md`. Commit SHAs reference
`gsd/phase-19.1-memory-agent-infrastructure-and-training`.

---

## Timeline

| Date | Event | Commit |
|---|---|---|
| 2026-04-06 | Phase 0 framework landed (CPS + measurement infrastructure) | `972567e` |
| 2026-04-07 ~07:55 ET | Production data wipe (test fixture DROP-TABLE against production pg16) | — |
| 2026-04-07 | Emergency test fixture disables committed | `1197c0b` |
| 2026-04-07 22:04 ET | Plan A recovery complete (iter 0-4 restored) | `1d3390a` |
| 2026-04-10 | Plan B complete (branch consolidation + conftest guard + fixture restoration) | `c7b6eef` |
| 2026-04-11 | Task C complete (consolidation OOM fix) | `3177ce0` |
| 2026-04-12 | Task D complete (Phase B env-prefix fix) | `d4d718f` |
| 2026-04-13 | Decision: wipe iter 5 entirely, work from iter 0-4 baseline | handoff rewrite |

---

## STEP 0 — Branch consolidation (complete, `c7b6eef`)

Phase 19.1 work originally landed on `gsd/phase-20-production-deployment`
because the session that started the work misidentified the phase as
"Phase 21." Plan B moved the commits to the correct branch:

1. Cherry-picked the full range `98f9151..gsd/phase-20-production-deployment`
   onto `gsd/phase-19.1-memory-agent-infrastructure-and-training` (range
   syntax, to avoid off-by-1 from literal SHA lists).
2. Reworded the 3 mislabelled `(21):` commit subjects to `(19.1):` via
   interactive rebase; newer commits already had correct prefixes.
3. Reset `gsd/phase-20-production-deployment` back to its legitimate tip
   (`98f9151`) and deleted the branch.

Commands preserved in commit `c7b6eef` body.

---

## Incident 2026-04-07 — Production data wipe

### What happened

While verifying Phase 0.5 regression tests, Claude copied a test file into
the `swingrl` container and ran `python3 -m pytest /tmp/test_backtest.py`
via `docker exec`. The container's `DATABASE_URL` pointed at production
pg16. The test fixture `_create_backtest_schema` in
`tests/agents/test_backtest.py:201` executed
`DROP TABLE IF EXISTS backtest_results` and
`DROP TABLE IF EXISTS iteration_results`, then recreated both tables
with the pre-Phase-0.2 schema (no CPS columns).

Result in pg16:
- `backtest_results`: emptied (was 700+ rows)
- `iteration_results`: 1 fixture leftover row (`iter=42 equity baseline`)
- 16 CPS columns gone (Phase 0.2 migration reverted)

### Why the existing safeguard failed

Commit `d144e1d fix(20): isolate CI tests from production DB`
(Apr 6) had added DB isolation at the orchestration layer —
`scripts/ci-homelab.sh` created a temporary `swingrl_test` database and
overrode `DATABASE_URL` before invoking pytest. Running pytest directly
via `docker exec` bypassed that entirely. The test fixtures had no
database-name guard at the test-code level.

The incident was the exact pattern called out in `d144e1d`'s commit
message as the original reason for the isolation fix — reproduced one
day later by a different code path.

### What was intact

- `/app/data/db_backup_pre_postgres/market_data.ddb` — 103 MB snapshot
  from Apr 4 containing full iter 0-4 data
- `training_epochs` (850,430 rows)
- `memories` (4,958,918 rows)
- `meta_decisions` (30 rows)
- `reward_adjustments` (149 rows)
- `consolidations` (84 rows)
- All model.zip files in `/app/models/iterations/`
- All git commits including Phase 0 work

---

## Plan A — Iter 0-4 recovery (complete, `1d3390a`)

Approved plan: `/Users/varunpanchal/.claude/plans/delegated-skipping-umbrella.md`

### Steps executed

- **R1** ✅ Verified duckdb backup integrity (10 iteration_results, 564
  backtest_results, 43 cols) and snapshotted pg16 pre-recovery state
- **R2a** ✅ Surgical DELETE of the lone `iter=42` baseline test fixture leftover
- **R2b** ✅ Re-applied `add_cps_columns.py` migration (idempotent
  `ADD COLUMN IF NOT EXISTS`) — 16 columns added, `iteration_results`
  now 40 cols
- **R3** ✅ New committed script
  `scripts/migrations/restore_iter_0_4_from_duckdb.py` (213 lines,
  `ON CONFLICT DO NOTHING`, `--source` / `--max-iter` / `--dry-run`
  flags) restored 564 `backtest_results` + 10 `iteration_results` from
  the duckdb backup via column intersection. Dry-run + real run both clean.
- **R4** ⏭ SKIPPED — iter 5 rebuild from training logs. Sources too
  lossy: `training_iter5.log*` mostly rotated (12 of 111 `fold_complete`
  events on disk); `training_comparison.json` missing `max_single_loss`
  and `is_control_fold`. User accepted loss.
- **R5** ⏭ SKIPPED — iter 5 `iteration_results` recovery (depends on R4)
- **R6** ✅ Ran `backfill_cps_history.py --max-iter 4` — all 10
  `iteration_results` rows now have CPS columns populated
- **R7** ✅ Final verification: every CPS value matches the handoff
  baseline byte-for-byte

### Post-recovery pg16 state (iter 0-4 only)

| Table | Rows | Notes |
|---|---|---|
| `iteration_results` | 10 | iter 0-4 × {equity, crypto}; 40 columns; all CPS populated |
| `backtest_results` | 564 | iter 1 equity has 78 rows incl. 9 restart-with-fixes dupes |

### Container side effect

`add_cps_columns.py` and `backfill_cps_history.py` were `docker cp`'d
into the `swingrl` container at `/app/scripts/migrations/` and
`/app/scripts/` during Plan A so the recovery could run. They survive
container restart but NOT container recreation. Tracked in active
handoff Group B (container rebuilds).

### Smoking-gun harm chart (iter 3-4)

Treatment = fold with LLM memory advice; control = fold without.

| Env | Iter | Treatment v1 | Control v1 | Harm ratio |
|---|---|---|---|---|
| equity | 3 | 0.01233 | 0.03427 | 2.78× |
| equity | 4 | 0.01485 | 0.04071 | 2.74× |
| crypto | 3 | 0.08161 | 0.28397 | 3.48× |
| crypto | 4 | 0.08318 | 0.42063 | 5.06× |

iter 5 row preserved in the main handoff as it drove the original
Phase 19.1 decision, even though the iter 5 data was later wiped.

---

## Plan B — Test-fixture safety net (complete, `c7b6eef`)

### What landed

1. **Branch consolidation** (see STEP 0 section above)
2. **`pytest_configure` guard** added to `tests/conftest.py` (~25 lines).
   Refuses to run pytest unless `DATABASE_URL` ends with `_test` or is
   in the allowlist `{swingrl_test}`. Fires before any fixture runs.
   Catches the failure mode that caused the 2026-04-07 incident.
3. **All 23 fixture disables removed** — the 18 originally disabled in
   `1197c0b` plus the 5 previously-pending files
   (`test_shadow_runner`, `test_promoter`, `test_wash_sale`,
   `test_stuck_agent`, `test_alerter`). The structural guard replaces
   all per-file disables.

### Test suite state after Plan B

975 passed / 391 skipped / 0 failed. The 391 skipped tests are live-DB
tests that skip without `DATABASE_URL` set.

### Original 23-file fixture audit table

All entries below are now covered by the `pytest_configure` guard. No
per-fixture disables remain as of `c7b6eef`.

| # | File | Pattern |
|---|------|---------|
| 1 | `tests/agents/test_backtest.py` | `_create_backtest_schema`: DROP TABLE backtest_results, iteration_results |
| 2 | `tests/agents/test_validation.py` | 3 test methods: DROP TABLE model_metadata, backtest_results CASCADE |
| 3 | `tests/memory/test_meta_orchestrator.py` | `_create_hmm_db` + `test_current_regime_vector_handles_null_columns`: DROP TABLE hmm_state_history CASCADE |
| 4 | `tests/features/test_fundamentals.py` | DROP TABLE fundamentals |
| 5 | `tests/data/test_db.py` | `db_manager`: TRUNCATE-all-public-tables CASCADE |
| 6 | `tests/data/test_gap_fill.py` | `_make_pg_conn` / `_cleanup_pg_conn`: TRUNCATE ohlcv_4h, ohlcv_daily |
| 7 | `tests/data/test_cross_source.py` | `cs_db`: TRUNCATE-all-public-tables |
| 8 | `tests/data/test_corporate_actions.py` | `ca_db`: TRUNCATE-all-public-tables |
| 9 | `tests/data/test_ingestion_logging.py` | `db_manager`: DELETE + TRUNCATE-all |
| 10 | `tests/data/test_parquet_to_duckdb.py` | `db_manager`: TRUNCATE-all-public-tables |
| 11 | `tests/data/test_verification.py` | `_make_pg_conn` / `_cleanup_pg_conn`: DELETE FROM ohlcv_daily/ohlcv_4h/macro_features |
| 12 | `tests/test_memory_service.py` | 5× `_clean_memory_tables` + `test_telemetry_tables_insertable`: TRUNCATE memory tables + training_epochs / meta_decisions / reward_adjustments |
| 13 | `tests/execution/conftest.py` | `mock_db`: TRUNCATE-all-public-tables |
| 14 | `tests/scheduler/test_halt_check.py` | `_clean_emergency_flags`: DELETE FROM emergency_flags |
| 15 | `tests/scheduler/test_jobs.py` | `_clean_test_tables`: DELETE FROM emergency_flags, portfolio_snapshots |
| 16 | `tests/features/test_pipeline.py` | `seeded_duckdb`: DELETE FROM features_equity/crypto, hmm_state_history, ohlcv_daily/4h, macro_features |
| 17 | `tests/test_phase15.py` | `test_empty_*_table_raises`: DELETE FROM features_equity, ohlcv_daily, features_crypto |
| 18 | `tests/dashboard/test_pages.py` | `trade_db`: DELETE FROM trade_log |
| 19 | `tests/shadow/test_shadow_runner.py` | `_create_pg_with_shadow_trades`: DELETE FROM shadow_trades |
| 20 | `tests/shadow/test_promoter.py` | DELETE FROM shadow_trades, trades, portfolio_snapshots, circuit_breaker_events |
| 21 | `tests/monitoring/test_wash_sale.py` | DELETE FROM wash_sale_tracker |
| 22 | `tests/monitoring/test_stuck_agent.py` | DELETE FROM portfolio_snapshots |
| 23 | `tests/monitoring/test_alerter.py` | DELETE FROM alert_log |

---

## Task C — Consolidation OOM fix (complete, `3177ce0`)

### Root cause

Phase B consolidation loaded all epoch memories into a single Python list
before parsing. With 688k crypto SAC memories at ~5 KB text each, peak
usage hit ~3.4 GB vs the 1 GB container limit, causing OOM and killing
the memory service mid-consolidation.

### Fix

Stream-parse each 10k-row chunk via `_parse_epoch_memory()`, keeping only
the extracted lightweight metrics (~80 B per memory) instead of the raw
text. Peak memory: ~3.4 GB → ~160 MB. The 688k memories are legitimate
iter 5 data per `feedback_crypto_memories_not_junk.md` — fix handles
volume, does not delete data.

### Not fixed

The root cause of why crypto SAC wrote 688k memories instead of the
expected ~350 is not investigated. SAC's configured cadence (40,000
steps) may not be enforced correctly under SAC's off-policy callback
pattern. Tracked in active handoff Group F (deferred).

---

## Task D — Phase B env-prefix fix (complete, `d4d718f`)

### Root cause

Phase B of `run_stage1()` used bare source prefixes
(`"training_epoch"`, `"reward_adjustment"`, `"cross_iteration"`) in its
SQL `LIKE` query, returning all environments globally. A client-side
text filter (`f"env={env_name}" not in text`) then tried to discard
wrong-env rows. With 688k crypto SAC + 18k equity rows ordered by
`created_at DESC`, equity pagination was starved — most pages contained
crypto rows that were discarded, and the loop often exited with
`total_phase_b_count=0`, logging `consolidation_skipped_no_memories`
against an env that actually had ~18k memories available.

### Fix

Changed to env-qualified prefixes
(`f"training_epoch:{env_name}"`, `f"reward_adjustment:{env_name}"`)
matching Phase A's existing pattern. All three production writers
already embed env in the source tag. Dropped `cross_iteration` from
Phase B entirely — Phase A already handles it, and Phase B was
accidentally archiving preserved-on-failure rows. Removed the
now-redundant client-side text filter.

Added `TestPhaseBEnvIsolation` regression test proving equity
consolidation does not consume or archive crypto memories.

### Confirmed hypothesis

H3 from Task C's hypothesis list (source filter is wrong). The
connection to Task C was exactly as predicted: crypto found too many
(global scan), equity found zero (starved by crypto's volume in the
same global scan).

---

## Iter 5 recovery status (historical — superseded by wipe decision 2026-04-13)

At the time of Plan A, iter 5 was a partial story:

| Table | Iter 5 state at 2026-04-12 |
|---|---|
| `backtest_results` | 0 rows (wiped by test fixture, declared lost) |
| `iteration_results` | absent |
| `training_epochs` | 850,430 rows intact |
| `memories` | ~850k rows intact (688k crypto SAC + ~160k others) |
| `consolidations` | 5 Stage-1 patterns intact (ids 170-174) |
| `pattern_outcomes` | 2 rows from replay (ids 13, 14) |
| `pattern_presentations` | 1,835 rows intact |
| `meta_decisions` | 6 rows intact |
| model.zip files | 6 files in `/app/models/iterations/iter_5/` |

### Iter 5 recovered ensemble weights (from `recover_iteration_results.py`)

Preserved in case it informs the fresh iter 5 re-run.

| Env | PPO | A2C | SAC |
|---|---|---|---|
| equity | 0.6277 | 0.2234 | 0.1489 |
| crypto | 0.8028 | 0.1850 | 0.0123 |

PPO dominates both ensembles. SAC near-zero in crypto, consistent with
SAC's known crypto fragility.

### Decision 2026-04-13

Wipe all remaining iter 5 artifacts and work from the clean iter 0-4
baseline. iter 5 becomes a fresh re-run after Phase 1 (prompt + reward
refocus) is complete. See active handoff Group A for the wipe procedure
and Group E for the re-run.

---

## Pre-existing bugs surfaced and fixed during Phase 0

Fixed in commit `972567e` (Phase 0 framework):

1. **Silent rollback** (`scripts/train_pipeline.py` ~L2295). The
   `iteration_results` connection was opened without `autocommit=True`,
   `store_iteration_results_to_duckdb` never called `commit()`, and
   `finally: conn.close()` rolled back the INSERT. Live since the
   postgres migration. Iter 0-4 rows in pg16 came from the one-time
   migration script, not from training.
2. **Path mismatch in `deploy_best_models`** (`scripts/train_pipeline.py`
   ~L208). Selector looked at `models/iterations/iter_N/{env}/{algo}/`
   but the trainer writes to
   `models/iterations/iter_N/active/{env}/{algo}/`. Live since iter 0 —
   production active path was empty the entire project, causing
   `crypto_cycle_failed: No actions to blend`. Iter 5 was manually
   deployed to `/app/models/active/` via `docker exec` during recovery;
   that manual copy will be wiped in Group A.
3. **Dashboard cached connection rot** (`dashboard/app.py`). Pages 1-4
   close the `@st.cache_resource` singleton; the new Iteration History
   page hit the closed connection. Fixed with self-heal in
   `get_pg_conn`.

These three fixes are in git but NOT in the running `swingrl` /
`swingrl-dashboard` container images as of 2026-04-13. Tracked in
active handoff Group B.

---

## Phase 0 artifacts (reference)

Commit `972567e` landed 24 files / 4,828 insertions / 39 deletions.

### New modules

| File | Purpose |
|---|---|
| `src/swingrl/metrics/cps.py` | Three CPS formulas (multiplicative v1, additive v2, sortino-anchored v3); FoldMetrics TypedDict |
| `src/swingrl/reporting/iteration_report.py` | Loaders + pure helpers + persistence orchestrator for iteration_results |
| `scripts/backfill_cps_history.py` | Idempotent CPS backfill |
| `scripts/migrations/add_cps_columns.py` | Additive 16-column migration for iteration_results |
| `scripts/migrations/recover_iteration_results.py` | Reconstructs iteration_results rows from backtest_results |
| `dashboard/pages/5_Iteration_History.py` | Streamlit page with CPS trend, per-fold heatmap, treatment-vs-control harm banner |
| `tests/metrics/test_cps.py` | 21 CPS unit tests |
| `tests/reporting/test_iteration_report.py` | 39 iteration_report tests (9 live DB) |
| `tests/monitoring/test_iteration_embed.py` | 16 Discord embed tests |
| `tests/data/test_iteration_results_extension.py` | 19 schema migration tests |

### Modified files

| File | Change |
|---|---|
| `src/swingrl/data/postgres_schema.py` | Added 16 nullable CPS columns to `iteration_results` DDL |
| `src/swingrl/monitoring/embeds.py` | New `build_iteration_completion_embed` with green/yellow/red color logic |
| `scripts/train_pipeline.py` | Phase 0.5 lifecycle log events + Phase 0.6 Discord embed wiring; autocommit fix; `deploy_best_models` path fix |
| `dashboard/app.py` | `get_pg_conn` self-heal for closed cached connections |
| `dashboard/Dockerfile.dashboard` | Copies `src/swingrl/{reporting,metrics}` into the dashboard image |
| `dashboard/requirements.txt` | Added `structlog>=24.1` |

---

## Validation gate — 8/8 criteria passed at end of Phase 0

| # | Criterion | Result |
|---|---|---|
| 1 | Iter 5 regression flag set on at least one CPS formula | ✅ flagged on both equity AND crypto (cps_v1 ensemble level) |
| 2 | Iter 5 PPO `cps_v1_delta` lateral (±0.003) | ⚠️ Equity ensemble cps_v1_delta = -0.00126 within magnitude |
| 3 | Chronic failures = `[2, 4, 7, 13, 15]` | ✅ confirmed live for equity |
| 4 | Protected winners non-empty | ✅ `[1, 8, 10, 16, 20, 22]` for equity |
| 5 | CPS v1 trend across iter 0-5 (equity) | ✅ 0.01167 → 0.01325 → 0.01133 → 0.01338 → 0.01526 → 0.01401 |
| 6 | Discord embed renders correctly | ✅ verified with synthetic inputs (color logic, regression marker) |
| 7 | `dedup_rows_dropped = 9` for iter 1 equity | ✅ confirmed (6 A2C + 3 PPO re-runs from restart-with-fixes) |
| 8 | Treatment/control split renders for iter 3+ | ✅ confirmed (equity 2.74-3.29×, crypto 3.48-5.06× across iter 3-5) |

---

## Open questions resolved during the 2026-04-13 replan

- **Iter 5 recovery strategy** — decided: wipe entirely, re-run fresh after Phase 1
- **Crypto SAC 688k memories keep/archive/sample** — moot after the wipe decision
- **Test suite gap (live DB coverage)** — Plan B added structural guard; 391 live tests still need a docker postgres (tracked in active handoff Group F)

Remaining open questions moved to active handoff Group G.

---

## End of archive
