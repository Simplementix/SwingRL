# Code Review Fix Plan

**Based on:** CODE-REVIEW-FINDINGS.md (35 issues: 5 CRITICAL, 16 IMPORTANT, 14 MINOR)
**Branch:** gsd/phase-19.1-memory-agent-infrastructure-and-training

---

## Approach

Fixes grouped into 4 waves by dependency and risk. Each wave is one commit.
**Estimated effort:** 2-3 hours total.

---

## Wave 1: Critical Fixes (training reliability + Docker safety)

**Commit message:** `fix(19.1): close SubprocVecEnv, remove unused DuckDB locks, fix Docker build`

| ID | Fix | File(s) |
|----|-----|---------|
| C1 | Add `vec_env.close()` + `eval_vec_env.close()` in finally block after training | `trainer.py` |
| C2 | Remove unused `duckdb.connect()` calls in tuning rounds | `train_pipeline.py` |
| C3 | Wrap `_load_consolidation_config()` in try/except with env var fallback | `consolidate.py` |
| C4 | Create `services/memory/.dockerignore` | `services/memory/.dockerignore` (new) |
| C5 | Reorder Dockerfile: mkdir before chown, use `--chown` on COPY | `services/memory/Dockerfile` |

---

## Wave 2: Auth + Data Integrity Fixes (memory pipeline correctness)

**Commit message:** `fix(19.1): fix memory auth, archival loop, body limits, and consolidation source batching`

| ID | Fix | File(s) |
|----|-----|---------|
| I3 | Pass `api_key` to MemoryClient in `_train_final_algo` | `train_pipeline.py` |
| I4 | Add API key header to epoch_callback's direct urllib calls | `epoch_callback.py` |
| I7 | Always archive memories after consolidation (remove `created > 0` gate) | `consolidate.py` |
| I8 | Add `max_length=50000` to `IngestRequest.text` | `routers/core.py` |
| I10 | Batch `insert_consolidation_source` into single connection | `db.py`, `consolidate.py` |
| I11 | Remove dead merge/conflict prompt code | `consolidate.py` |
| I14 | Bind memory service port to localhost: `127.0.0.1:8889:8889` | `docker-compose.prod.yml` |

---

## Wave 3: Training Pipeline Correctness (metrics + timestep logic)

**Commit message:** `fix(19.1): fix crypto Sharpe annualization, epoch advice change detection, curriculum end_bar`

| ID | Fix | File(s) |
|----|-----|---------|
| I1 | Track `best_ppo_folds` in tuning round 1, pass to ensemble gate | `train_pipeline.py` |
| I2 | Populate `converged_at_step` from WF fold results for `decide_final_timesteps` | `train_pipeline.py` |
| I5 | Add `periods_per_year` param to MemoryVecRewardWrapper (252 equity, 2191 crypto) | `reward_wrapper.py`, `trainer.py` |
| I6 | Add change-detection + resolve-before-overwrite in epoch advice | `epoch_callback.py` |
| I13 | Rewrite `from_date_ranges` end_bar with forward iteration | `curriculum.py` |

---

## Wave 4: Minor Cleanup (quality-of-life)

**Commit message:** `chore(19.1): minor cleanup from code review (logging, deps, docker)`

| ID | Fix | File(s) |
|----|-----|---------|
| M1 | Remove `/no_think` prefix for qwen2.5 | `consolidate.py` |
| M3 | Upgrade `_track_presentations` logging to warning | `query.py` |
| M4 | Read `LOG_LEVEL` from env var for structlog | `app.py` |
| M5 | Add HEALTHCHECK to memory service Dockerfile | `services/memory/Dockerfile` |
| M6 | Use `httpx.Timeout(connect=10, read=120)` | `consolidate.py`, `query.py` |
| M7 | Clamp negative weights to 0 in `_normalize_weights` | `reward_wrapper.py` |
| M9 | Use context manager for SQLite in `_load_fills` | `seed_memory_from_backtest.py` |
| M10 | Use `with` for `urlopen` in `check_memory_service_health` | `train_pipeline.py` |
| M13 | Remove `black` from dev deps | `pyproject.toml` |
| I15 | Pin memory service requirements | `services/memory/requirements.txt` |

---

## Deferred (not blocking merge, tracked for later)

| ID | Reason |
|----|--------|
| I9 | httpx connection pooling — optimization, not correctness. Defer to Phase 20. |
| I12 | Conflict detection heuristic — needs design work for LLM-based approach. Defer. |
| I16 | `PathsConfig` str→Path — touches many consumers, risk of breakage. Defer to next refactor. |
| M8 | Dynamic import refactor — low risk, low urgency. |
| M11 | SQL table name allowlist — already uses hardcoded constants. |
| M12 | CI bandit/detect-secrets — additive, doesn't block merge. |
| M14 | TZ=UTC in Docker — needs audit of all datetime usage first. |

---

## Execution Order

1. **Wave 1** first — unblocks training reliability (SubprocVecEnv leak is actively burning resources)
2. **Wave 2** next — fixes auth so memory pipeline actually works when training completes
3. **Wave 3** then — corrects metrics and logic for next training run
4. **Wave 4** last — polish

**Note:** Waves 1-3 should be done before pushing the fixed code to homelab for the next training run. Wave 4 can follow.
