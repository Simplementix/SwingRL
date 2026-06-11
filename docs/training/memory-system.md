# Memory System Reference

Living reference for SwingRL's memory subsystem — the LLM-backed pattern store that ingests training events, consolidates them into reusable patterns, and feeds them back as run-config and epoch-level advice. The subsystem lives in its own FastAPI service (`services/memory/`, container `swingrl-memory`); the trainer talks to it over HTTP via `src/swingrl/memory/client.py` only — never SQL.

**Last verified against code:** 2026-05-05

**Honest-gap policy:** every concrete claim is `file:line`-cited. Where a behavior or writer is referenced from project memory but cannot be located in current code, the gap is flagged inline and aggregated in [Known issues](#known-issues--open-questions). Discrepancies between code and `MEMORY.md` are surfaced rather than silently corrected.

**Schema cross-link:** the 7 memory-system tables (`memories, consolidations, consolidation_quality, consolidation_sources, pattern_presentations, pattern_outcomes, llm_audit_log`) are defined in [`training-data-capture.md`](training-data-capture.md) cluster 3 (writer file:line, readers, indexes, idempotency). This doc covers the *flow* and *lifecycle* — what calls what, when, and why — not the schema.

## Architecture at a glance

The memory service is a single FastAPI app (`services/memory/app.py`) running in the `swingrl-memory` container. It exposes three logical agents and seven HTTP endpoints. Trainer-side code (`src/swingrl/memory/`) is a thin async HTTP client; all writes to memory tables happen inside the memory container.

| Logical agent | File | Role |
|---------------|------|------|
| `IngestAgent` | `services/memory/memory_agents/ingest.py` | Wrap text in `<memory>` envelope and persist via `insert_memory()` |
| `ConsolidateAgent` | `services/memory/memory_agents/consolidate.py` | Phase A / Phase B / Stage 2 LLM-driven pattern synthesis |
| `QueryAgent` | `services/memory/memory_agents/query.py` | Pattern selection + LLM advice for `run_config` / `epoch_advice` |

| Endpoint | Method | Caller |
|----------|--------|--------|
| `/ingest` | POST | `client.ingest_training()` from `epoch_callback.py`, `curriculum.py`, `scripts/train_pipeline.py` |
| `/consolidate` | POST | `train_pipeline.py:567` after each iteration |
| `/training/run_config` | POST | `meta_orchestrator.py:159` at iteration start |
| `/training/epoch_advice` | POST | `epoch_callback.py` per cadence + on notable events |
| `/training/record_outcome` | POST | `train_pipeline.py:576` after gate evaluation |
| `/training/pattern_effectiveness` | GET | external dashboards / human review |
| `/debug/consolidations` | GET | `meta_orchestrator._get_pattern_count()` (cold-start gate) |

All endpoints require `X-API-Key` (HMAC compare against `MEMORY_API_KEY`, `services/memory/auth.py:21-40`). The memory service has no scheduler — every consolidation and advice call is event-driven from the trainer.

## Memory types & lifecycle

### Source tags currently emitted from training

The `memories.source` column is a free-form tag string. Eight prefixes are emitted by current trainer code:

| Source prefix | Emitter | When |
|---------------|---------|------|
| `training_epoch:{env}:{algo}` | `epoch_callback.py:436` | Each captured epoch (cadence + notable events; cross-link to [`reward-shaping.md`](reward-shaping.md)) |
| `reward_adjustment:{env}:{algo}` | `epoch_callback.py:507-508` (trigger pass), `:562-563` (outcome pass) | LLM-approved reward-weight adjustments — two-pass write per adjustment |
| `curriculum_performance:historical` | `curriculum.py:200` | Per-window performance at fold end (not env/algo qualified) |
| `walk_forward:{env}:{algo}` | `scripts/train_pipeline.py:1283` | Per-algo walk-forward results, ingested at the end of each algo's WF run |
| `walk_forward:{env}:ensemble` | `scripts/train_pipeline.py:1376` | Ensemble-level WF summary, ingested after the per-algo block |
| `trading_pattern:{env}:{algo}` | `scripts/train_pipeline.py:1583` | Per-algo trading-pattern observation (entry/exit, regime, win rate) |
| `cross_iteration:{env}` | `scripts/train_pipeline.py:977` | Cross-iteration regression / improvement narrative |
| `training_run:{env}` | `scripts/train_pipeline.py:1859` | End-of-run summary, ensemble-level per env |

Phase A consolidation reads the matching prefixes (`walk_forward:`, `trading_pattern:`, `cross_iteration:`, `training_run:`) at `consolidate.py:1672-1691`. Phase B reads `training_epoch:` and `reward_adjustment:`. `curriculum_performance:` is currently not consumed by either consolidation phase — it appears to feed `manual_cross_algo_consolidation.py:101` and external dashboards only.

### XML wrapping

Stored text is wrapped at ingest time:

- Wrapper: `<memory>...</memory>` (`ingest.py:17-18, 57-58, 73-74`).
- Sanitization: only C0 control characters (U+0000-U+0008, U+000B-U+000C, U+000E-U+001F) are stripped (`ingest.py:28-41`). Ampersands and angle brackets are preserved deliberately so natural metric syntax (`sharpe=0.42 & mdd=-0.08`) survives unchanged.
- The wrapping happens caller-side (inside `IngestAgent.store()` / `store_async()`), not at the DB layer.

### Archive flag (the only soft-delete)

`memories.archived` is `INTEGER NOT NULL DEFAULT 0` (`db.py:87-93`). Lifecycle:

- `0` (default) on insert.
- `→ 1` only by `archive_memories()` (`db.py:445-465`) after a Phase A or Phase B LLM call returns a valid response. Batched 10k rows per UPDATE (`_BATCH = 10_000`, `db.py:455`).
- `1 → 0` only via `unarchive_memories()` (`db.py:468-479`) — manual restore for re-consolidation. Not invoked by the consolidation pipeline.

**No hard delete.** No `DELETE FROM memories` exists in `db.py` or `cleanup_patterns.py`. **No TTL.** Memories accumulate indefinitely; archived ones are skipped by readers but kept on disk. Per `MEMORY.md`, ~688K crypto SAC memories were observed historically — that is an observation, not an extrapolation.

## Ingestion flow

### `POST /ingest`

Schema (`routers/core.py:27-32`):

- Request: `{text: str (max 50KB), source: str}`
- Response: `{id: int, status: "ok"}`

Auth (`routers/core.py:66-77`, `auth.py:21-40`): `X-API-Key` header verified by HMAC constant-time compare against the `MEMORY_API_KEY` env var. Returns 401 on mismatch.

Handler creates an `IngestAgent`, calls `await agent.store_async(text, source)`, returns the row id (`routers/core.py:66-77`).

### `insert_memory()` mechanics

- Sync (`db.py:309-328`): `INSERT INTO memories (text, source) VALUES (%s, %s) RETURNING id`. Parameterized. Explicit `conn.commit()`. Returns `int(row_id)`.
- Async (`db.py:902-904`): `_run_live()` thread-pool wrapper (2 threads).
- Each insert is its own transaction — no batching, no buffer.
- PK `id` is `GENERATED BY DEFAULT AS IDENTITY` (`db.py:88-89`) — monotonic but not strictly sequential under concurrent inserts.

### Fail-open client

The trainer client `MemoryClient.ingest()` (`client.py:57-99`) catches every exception, logs a warning, and returns `False`. Training never blocks on memory-service downtime. Failed ingests are silently dropped — no retry, no in-memory queue, no on-disk buffer. If the memory service is unhealthy, those memories are lost.

Training-side call sites for ingest (all use `client.ingest_training(text, source)` at `client.py:101-122`):

| Call site | Source prefix | Trigger |
|-----------|---------------|---------|
| `epoch_callback.py:436` | `training_epoch:{env}:{algo}` | Captured epoch (cadence + notable event) |
| `epoch_callback.py:507-508` | `reward_adjustment:{env}:{algo}` | Reward-weight adjustment proposed (pass 1) |
| `epoch_callback.py:562-563` | `reward_adjustment:{env}:{algo}` | 10 epochs after trigger (pass 2 outcome) |
| `curriculum.py:200` | `curriculum_performance:historical` | End of training fold |
| `scripts/train_pipeline.py:977` | `cross_iteration:{env}` | Cross-iteration narrative emitted between iterations |
| `scripts/train_pipeline.py:1283` | `walk_forward:{env}:{algo}` | After per-algo WF block completes |
| `scripts/train_pipeline.py:1376` | `walk_forward:{env}:ensemble` | After ensemble WF aggregation |
| `scripts/train_pipeline.py:1583` | `trading_pattern:{env}:{algo}` | Per-algo trading-pattern observation |
| `scripts/train_pipeline.py:1859` | `training_run:{env}` | End-of-run summary per env |

## Consolidation pipeline

Consolidation is **event-driven only** (`consolidate.py:1-29`). The single trigger is `POST /consolidate` (`routers/core.py:80-94`), invoked from `train_pipeline.py:567` after each walk-forward iteration. The endpoint accepts an optional `env_name`; when omitted, all three sub-runs (Stage 1 equity, Stage 1 crypto, Stage 2 cross-env) execute back-to-back.

The agent acquires a global `_run_lock` (`consolidate.py:1605-1623`) to serialize concurrent calls. The trainer-side timeout is `cfg.memory_agent.consolidation.timeout_sec` (default 1800s).

Consolidation runs in three discrete sub-runs per `/consolidate` call:

```
            ┌─ Stage 1 Phase A ─ WF + trading + cross_iter + training_run ─→ LLM → patterns (env=equity, stage=1)
Stage 1 ──┤                                                                                ↓ archive on success
(per env) ├─ Stage 1 Phase B ─ epoch + reward_adjustment (locally aggregated) ──→ LLM → patterns (env=equity, stage=1)
            └─ (same for env=crypto)
                                                                                            ↓
Stage 2 ─── Cross-env ─ Phase-A patterns from BOTH envs ──→ LLM → patterns (env=NULL, stage=2)
            (only fires if both envs produced ≥1 active pattern)
```

### Stage 1 Phase A — WF + trading patterns

**Inputs (4 source prefixes per env, batched at 200/call):** `walk_forward:{env}`, `trading_pattern:{env}`, `cross_iteration:{env}`, `training_run:{env}` (`consolidate.py:1671-1691`). Backward-compat fallback: raw `{env_name}` prefix (`consolidate.py:1701-1707`). Filter: `archived=False`. Batch size: `_MEMORY_BATCH_SIZE = 200` (`consolidate.py:208`).

**Prompt shape (sections — not verbatim text)** at `consolidate.py:398-478`:

1. Preamble + role + data format
2. Confidence calibration guide (0.4-0.7 typical for financial data)
3. Control-fold analysis: distinguish HP impact vs reward-shaping impact via `[CTRL]` / `[TREATMENT]` tags
4. HP bounds violations (per-algo PPO/A2C/SAC ranges)
5. Category list (13 categories: `regime_performance`, `macro_transition`, `trade_quality`, `overfit_diagnosis`, `drawdown_recovery`, `data_size_impact`, `iteration_regression`, `hp_effectiveness`, `cross_env`, `live_cycle_gate`, `live_blend_weights`, `live_risk_thresholds`, `live_position`, `live_trade_veto`, `cross_env_correlation` — full set in `_VALID_CATEGORIES` at `consolidate.py:264-282`)
6. Common failure modes (excessive penalties, turnover penalty, high entropy coeff)
7. JSON return schema with `patterns` array

**Few-shot:** 7 example patterns (fictional metrics) at `consolidate.py:647-728`.

**LLM call** (`_call_llm_with_retry()` at `consolidate.py:2171-2212` → `_call_llm()` at `:2214-2269`):

- Primary: configured provider (default `mistral`, `mistral-large-latest`).
- Backup: `openrouter` `nvidia/nemotron-3-super-120b-a12b:free` (hard-coded fallback at `:120`).
- Temperature: 0 (deterministic).
- Retry on malformed JSON: 2 attempts, same provider (`:2196-2204`).
- Max tokens: from config (default 32768).
- Structured output: provider-specific (Cerebras/Gemini → `response_format.json_schema`; NVIDIA → `json_object` + `guided_json`; others → `json_object`).

**Output validation** (`_validate_consolidation()` at `:2460-2495` → `_validate_single_pattern()` at `:2497-2535`):

- Required fields: `pattern_text, category, affected_algos, affected_envs, actionable_implication, confidence, evidence`.
- `confidence` clamped to `[0.0, 1.0]`; defaults to `0.5` on parse error.
- `category` must appear in `_VALID_CATEGORIES` (18 options); defaults to `regime_performance`.
- Patterns missing required fields are discarded; single-pattern responses are auto-wrapped into `{"patterns": [...]}`.

**Persistence (per accepted pattern):**

1. `insert_consolidation_async()` → `consolidations` row, `status='active'` default (`db.py:542-564`).
2. `insert_consolidation_sources_async(row_id, memory_ids)` → `consolidation_sources` join (`db.py:602-605`).
3. If conflict detected: `update_consolidation_status_async(conflict_id, 'superseded', superseded_by=row_id, conflict_group_id=group_id)` (`consolidate.py:2156-2161` → `db.py:703-730`).
4. `insert_audit_log()` synchronous, fire-and-forget (`consolidate.py:1725` → `db.py:331-376`).
5. `log_consolidation_quality_async()` fire-and-forget (`db.py:1098-1109`).

Each `insert_*` commits independently; there is no distributed transaction across the four tables. Re-running consolidation after a partial DB write may create duplicate patterns (no idempotency).

**Archive timing** — only on LLM success (`consolidate.py:1747`): `archive_memories_async(memory_ids)` flips `archived=1` for all source memories. On LLM failure, memories are deliberately preserved for retry on the next consolidation run, with log line `consolidation_memories_preserved` (`:1749-1754`).

### Stage 1 Phase B — epoch + reward dynamics

Phase B does **not** send raw memory batches to the LLM. Instead it streams `training_epoch:{env}` and `reward_adjustment:{env}` memories in `_FETCH_CHUNK = 10_000` row chunks (`consolidate.py:1798, 1802`), parses each via `_parse_epoch_memory()` (`:1820`), and aggregates locally into fold summaries (cross-fold IQM, trajectory shape, outlier rates, weight-adjustment summaries) before a single LLM call per env (`:1886-1892`). This was the architectural move that replaced the prior 500-batch approach.

**Prompt shape (sections)** at `consolidate.py:480-591`: training-dynamics focus — convergence shape, reward-shaping effectiveness, KL instability (PPO only), drawdown recovery; 6 categories (`iteration_progression, overfit_diagnosis, drawdown_recovery, reward_shaping, hp_effectiveness, iteration_regression`).

**Few-shot:** 7 example patterns at `consolidate.py:730-799`.

LLM call mechanics, validation, persistence, and archive timing are identical to Phase A.

### Stage 2 — cross-environment

**Trigger condition:** Stage 2 only runs if **both** equity and crypto Phase A produced ≥1 active pattern in this consolidation run (`consolidate.py:1643-1649`). If only one env has patterns, Stage 2 is skipped.

**Inputs:** Stage 1 active patterns for each env, formatted as `[EQUITY] pattern_text (category=..., confidence=...)` and `[CRYPTO] ...` lines (`consolidate.py:1996-2009`).

**LLM call:** Single attempt — no retry-on-malformed (`consolidate.py:2024`). Yields 0-3 patterns. Two valid categories: `cross_env`, `cross_env_correlation`.

**Persistence:** Same dedup-and-insert path. `stage=2`, `env_name=NULL`, `memory_ids=[]`. **Stage 2 patterns reference Stage 1 patterns by inclusion in the prompt only — there is no DB-level link back to source Stage 1 rows.**

## State machine: active → superseded → retired

```
                  conflict detected
   active ─────────────────────────────→ superseded
   ▲│                                       │
   ││ manual restore                        │ manual cleanup
   │└─────────────  (cleanup_patterns.py)──┐│
   │                                        ▼│
   └──────── retired ←──────────── (cleanup_patterns.py)
   (manual cleanup)
```

| From → To | Trigger | Writer | Citation |
|-----------|---------|--------|----------|
| (new) → active | `insert_consolidation()` default | `insert_consolidation()` | `db.py:542-564` |
| active → superseded | Conflict detected during dedup | `update_consolidation_status()` | `consolidate.py:2156-2161`, `db.py:703-730` |
| active → retired | Manual cleanup | `cleanup_patterns.py:55-70` | hardcoded `RETIRE_IDS` dict |
| superseded → retired | Manual cleanup | `cleanup_patterns.py:55-70` | hardcoded `RETIRE_IDS` dict |
| superseded → active | Manual restore (false-positive conflict) | `cleanup_patterns.py:72-89` | hardcoded `RESTORE_IDS` dict |

**Conflict detection** (`consolidate.py:2554-2624`) — sentiment-metric adjacency. Three gates required:

1. Same `category`.
2. Shared `affected_algo` OR shared `affected_env`.
3. Opposite sentiment on the same metric (e.g., one pattern says "higher sharpe", another says "lower sharpe", same category, shared algo).

Sentiment-metric pairs are extracted with a 3-token adjacency window, skipping common filler words (`consolidate.py:311-343`). Both patterns are stamped with the same `conflict_group_id` (uuid4). The new pattern is inserted as `active`; the older one is flipped to `superseded` with `superseded_by` pointing at the new row.

**Dedup confirmation (NOT a state change)** — exact `(category, affected_algos)` match: `increment_confirmation()` is called on the first matching active row (`consolidate.py:2102` → `db.py:733-745`), and the new pattern is **not** inserted. `confirmation_count++` and `last_confirmed_at = NOW()`.

**No automated retire.** `cleanup_patterns.py` is a manual CLI invoked as `docker exec swingrl-memory python3 cleanup_patterns.py`. It iterates two hardcoded dicts (`RETIRE_IDS`, `RESTORE_IDS` at lines 26-45), idempotent per row. The consolidation pipeline never retires patterns on its own.

**Superseded patterns are not requeried.** `get_active_consolidations()` filters `status='active'` (`db.py:646`); only manual restore can bring a superseded pattern back into the advice path.

**No semantic merge.** Despite docstring language at `consolidate.py:16` referencing "merge", no LLM merge step exists — conflicts produce two co-existing rows (one active, one superseded) sharing a `conflict_group_id`. There is no third "merged" row.

## Retrieval & advice

### `/training/run_config`

**Call site:** `meta_orchestrator.py:159` (`self._query_run_config()` at iteration start). Query string format:

```
TRAINING RUN CONFIG ADVICE: env=equity algo=ppo iteration=2 current_regime={"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
```

The regime vector is sourced from the latest `hmm_state_history` row.

**Pattern selection** (`query.py:1393-1425`):

1. Category filter: `_RELEVANT_CATEGORIES["run_config"]` (9 categories at `query.py:749-759`: `regime_performance`, `overfit_diagnosis`, `iteration_progression`, `data_size_impact`, `macro_transition`, `cross_env`, `cross_env_correlation`, `hp_effectiveness`, `iteration_regression`).
2. Confidence filter: `min_confidence ≥ _MIN_CONFIDENCE` (default `0.4`, configurable from yaml at `query.py:806-818`).
3. Composite scoring (see [Composite scoring](#composite-scoring)).
4. `limit_per_category=3` — top 3 per category (`query.py:1416`).
5. Stage 2 patterns prepended, then Stage 1 — LLM sees cross-env patterns first.

**LLM call:** Primary `gemini-2.5-flash` → backup OpenRouter nemotron (`query.py:146-196, 304-329`). Algo-specific system prompt with HP bounds, per-algo guide, hallucination guards (`query.py:650-695`). Response uses provider-specific structured-output formats.

**Output schema** (`routers/training.py:39-54`):

```python
{
  "learning_rate":   float | None,
  "entropy_coeff":   float | None,
  "clip_range":      float | None,
  "n_epochs":        int   | None,
  "batch_size":      int   | None,
  "gamma":           float | None,
  "reward_weights":  {"profit": float, "sharpe": float, "drawdown": float, "turnover": float},
  "rationale":       str  # required
}
```

Fields are nullable so the LLM can choose to leave a knob alone. `rationale` is required and propagates to `pattern_presentations.advice_response` (truncated to 200 chars at `query.py:1498`).

**Side effects:**

- `pattern_presentations` insert per pattern shown (`query.py:1502-1508`): `request_type='run_config'`.
- `llm_audit_log` insert (`call_type='run_config'`, `query.py:1077-1086`).

### `/training/epoch_advice`

**Call site:** `epoch_callback.py` per cadence (PPO 20, A2C 8000, SAC 40000 epochs) plus notable events (KL > 0.10, MDD < -25.0). Cadence detail in [`reward-shaping.md`](reward-shaping.md).

**Inputs:** env, algo, iteration, current epoch metrics — **plus within-fold adjustment history** and a compact per-fold context block. When the `run_id` is present, `query.py:1123-1172` fetches the most recent 5 `REWARD_ADJUSTMENT_OUTCOME` memories from the same fold and embeds extracted fields (`epoch_triggered`, `post_adjustment_sharpe_delta`, `post_adjustment_mdd_delta`, `adjustment_effective`, `weights_before/after`) into the user message. This prevents the LLM from re-recommending an adjustment that just failed.

**Payload query string format** (assembled in `epoch_callback.py::_query_epoch_advice`):

```
EPOCH ADVICE: run_id=<run_id> algo=<algo> env=<env> epoch=<N> [iteration=<N>]
current_weights={"profit": 0.50, ...}
context={"fold_number": 3, "fold_role": "neutral", "prev_iter_cps_v1": 0.034,
         "target_metric": "cps_v1_multiplicative",
         "leading_indicators": {"rolling_sharpe": 1.2, "rolling_mdd": -0.05,
             "rolling_win_rate": 0.55, "trade_rate": 0.12, "baseline_trade_rate": 0.10},
         "diagnosis": {"label": "healthy", "fired": [], "confidence": "clear", "evidence": {}}}
```

`context` keys:

| Key | Type | Description |
|---|---|---|
| `fold_number` | `int \| null` | Walk-forward fold index (0-based); null when not wired. |
| `fold_role` | `str` | `"chronic_failure"` \| `"protected_winner"` \| `"neutral"` — from `fold_context.load_fold_context()`. |
| `prev_iter_cps_v1` | `float \| null` | Most recent `cps_v1_multiplicative` from `iteration_results` for this env. Null on iter 0 cold start. |
| `target_metric` | `str` | Always `"cps_v1_multiplicative"` — reminds the LLM what to optimise for. |
| `leading_indicators` | `dict` | Five rolling scalars from the wrapper: `rolling_sharpe`, `rolling_mdd`, `rolling_win_rate`, `trade_rate`, `baseline_trade_rate`. Moved here from bare f-string fields. |
| `diagnosis` | `CpsDiagnosis` | Output of `cps_diagnosis.diagnose_rolling()` — `label`, `fired`, `confidence`, `evidence`. Falls back to `{"label": "healthy", ...}` on `DataError` (unknown algo). |

The fold context is lazy-loaded once per fold from PostgreSQL with a 5-second timeout (fails open to neutral defaults). `rolling_sharpe` and `rolling_mdd` are no longer bare f-string fields in the query; they live exclusively inside `context.leading_indicators`.

**Pattern selection:** Same composite-score path as `run_config`, with a different category filter — `_RELEVANT_CATEGORIES["epoch_advice"]` (7 categories at `query.py:760-768`: `drawdown_recovery`, `trade_quality`, `reward_shaping`, `overfit_diagnosis`, `iteration_progression`, `hp_effectiveness`, `iteration_regression`).

**LLM chain:** Primary Cerebras `qwen-3-235b-a22b-instruct-2507` (`query.py:288`) → Groq `meta-llama/llama-4-scout-17b-16e-instruct` (`:296`) → Ollama `qwen2.5:14b` if reachable (`:242, 1800-1860`).

**Output schema** (`routers/training.py:62-69`):

```python
{
  "reward_weights": {"profit": float, "sharpe": float, "drawdown": float, "turnover": float},
  "stop_training":  bool,
  "rationale":      str,   # required
  "provider":       str,   # echo for audit
  "model":          str    # echo for audit
}
```

The `stop_training` flag, if `True`, lets the LLM signal the trainer to abort the current fold's training loop.

**Side effects:** Same `pattern_presentations` write per pattern shown (`query.py:1237`); `llm_audit_log` with `call_type='epoch_advice'`. Audit row has `fold_number` and `is_control_fold` parsed from the run_id (`query.py:1186-1189, 1220-1231`). When advice is accepted and weights are updated, a row is appended to `reward_adjustments` including 4 attribution columns: `fold_number`, `iteration_number`, `advice_id` (UUID v4, unique per accepted advice call), and `fold_cps_v1_before` (the most recent `cps_v1_multiplicative` from `iteration_results` at the time of the call — null on iter 0 cold start).

### Cold-start gate

Source: `meta_orchestrator.py:48-52, 283-293`.

- Constant: `_COLD_START_MIN_PATTERNS = 1` (`:52`).
- Pre-flight check: `_get_pattern_count(env)` queries `GET /debug/consolidations?limit=100`, sums rows where `env_name == env` OR `env in affected_envs` (`:354-390`).
- Below threshold: `_query_run_config()` returns `{}`, logs `meta_cold_start_guard` with `action="using_baseline_params"`. Training proceeds with baseline HPs only.

**Caveat — `limit=100` cap.** If a single env ever accumulates more than 100 active patterns, the count returned to the cold-start gate would be undercounted (the underlying query truncates). Below 100 active patterns per env this is moot; above, the gate would still pass since 100 ≫ 1, but any future use of this counter for non-binary thresholds would be wrong.

### Fallback when LLM fails

Two distinct fallback shapes — both are dicts, never raise.

**`advise_run_config()`** returns `dict(_SAFE_DEFAULTS)` on any unrecoverable failure (`query.py:1089`):

```python
{
  "learning_rate": 3e-4, "entropy_coeff": 0.01, "clip_range": 0.2,
  "n_epochs": 10, "batch_size": 64, "gamma": 0.99,
  "reward_weights": {"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
  "rationale": "cold_start_defaults"
}
```

`meta_orchestrator.py:160` clamps the result through `clamp_run_config(...)` against per-algo bounds before merging into the actual run config.

**`advise_epoch()`** returns `dict(_SAFE_EPOCH_DEFAULTS)` (`query.py:1234`): same weights, `stop_training=False`, `provider="none"`, `model="none"`.

The trainer never blocks, never raises, and never aborts the iteration on memory-service unavailability. Failed advice means baseline HPs / unchanged weights, and the iteration proceeds.

## Composite scoring

Source: `db.py:666-697` (the ORDER BY clause inside `get_active_consolidations()`).

Formula:

```
score = 0.5 * confidence
      + 0.3 * (confirmation_count / max_confirmation)
      + 0.2 * recency
recency = max(0.0, 1.0 - EXTRACT(EPOCH FROM (NOW() - created_at::TIMESTAMPTZ)) / (90.0 * 86400))
```

Half-life is the implicit 90-day window — a brand-new pattern at `confidence=1.0, confirmation_count=0` ranks above an old pattern at `confidence=1.0, confirmation_count=max` after ~75 days.

The composite score is applied only when `limit_per_category` is set (advice path), via a `ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC)` window function (`db.py:685-693`). When `limit_per_category` is unset (e.g., debug listing), the rank degrades to `ORDER BY confidence DESC, created_at DESC` (`:697`).

## Pattern effectiveness tracking

Two append-only tables let humans (and dashboards) measure whether the LLM advice is actually helping.

### `pattern_presentations` writer

Inserted once per pattern shown to the LLM during `run_config` or `epoch_advice` (`db.py:748-777`). Fields:

| Field | Source |
|-------|--------|
| `consolidation_id` | The pattern shown |
| `iteration` | Training iteration from query string |
| `env_name` | Environment from query string |
| `request_type` | `'run_config'` or `'epoch_advice'` |
| `advice_response` | First 200 chars of LLM rationale (`query.py:1498`) |
| `presented_at` | `NOW()` default |

Call sites: `query.py:1502-1508` (run_config), `query.py:1237` (epoch_advice).

### `pattern_outcomes` writer

Inserted twice per iteration (one per env) by `train_pipeline.py:576-583` after the gate evaluation, via POST `/training/record_outcome` (`routers/training.py:148-174` → `db.py:780-824`). Fields: `iteration, env_name, gate_passed, sharpe, mdd, sortino, pnl, patterns_presented (JSON), recorded_at`.

The `patterns_presented` field is a JSON array of consolidation IDs — a forward-looking per-pattern attribution surface. No reader exists yet; `/training/pattern_effectiveness` joins by `(iteration, env_name)` for now. The column is plumbed end-to-end so a future per-pattern attribution view can backfill without a migration.

### `/training/pattern_effectiveness` GET

Endpoint: `routers/training.py:177-186` → `db.py:827-842`.

```sql
SELECT pp.consolidation_id, pp.iteration, pp.env_name, pp.request_type,
       pp.advice_response, po.gate_passed, po.sharpe, po.mdd, po.sortino, po.pnl
  FROM pattern_presentations pp
  LEFT JOIN pattern_outcomes po
    ON pp.iteration = po.iteration AND pp.env_name = po.env_name
 ORDER BY pp.presented_at DESC
```

LEFT JOIN — presentations are returned even when no outcome row exists yet (e.g., advice was given mid-iteration but the gate hasn't run). Returns a flat list of dicts for human review.

## LLM provider chain

Seven providers exist in code; routing is per `call_type`. There is **no single unified provider abstraction class** — `ConsolidateAgent` and `QueryAgent` each call `httpx.AsyncClient` directly with provider-specific request shaping (`consolidate.py:2271-2305, 2389-2458` and `query.py:1512-1645, 1696-1798, 1800-1860`).

### Provider inventory

| Provider | Base URL | Env var | Default model | Used for |
|----------|----------|---------|---------------|----------|
| Mistral | `api.mistral.ai/v1` | `MISTRAL_API_KEY` | `mistral-large-latest` | consolidation primary |
| OpenRouter | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-super-120b-a12b:free` | consolidation backup, run_config backup |
| Gemini | `generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `gemini-2.5-flash` | run_config primary |
| Cerebras | `api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `qwen-3-235b-a22b-instruct-2507` | epoch_advice primary |
| Groq | `api.groq.com/openai/v1` | `GROQ_API_KEY` | `meta-llama/llama-4-scout-17b-16e-instruct` | epoch_advice fallback |
| NVIDIA | `integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | `moonshotai/kimi-k2.5` | configured but NOT wired to consolidation |
| Ollama | local URL from config | n/a | `qwen2.5:14b` | epoch_advice safety net |

Provider-key resolution: `_build_provider_entry()` reads `{NAME}_API_KEY` from `os.environ` first, then falls back to `config.api_key` (`consolidate.py:78-96`, `query.py:256-265`).

### Routing per `call_type`

| `call_type` | Primary | Fallback chain |
|-------------|---------|----------------|
| `consolidation_phase_a` | Mistral | OpenRouter |
| `consolidation_phase_b` | Mistral | OpenRouter |
| `consolidation_stage_2` | Mistral | OpenRouter |
| `run_config` | Gemini | OpenRouter |
| `epoch_advice` | Cerebras | Groq → Ollama |

The consolidation/run_config backup (`openrouter`) is hard-coded at `consolidate.py:120` and `query.py:190`. The epoch-advice fallback chain is hard-coded at `query.py:273-279`. Config can override the **primary** for each via `memory_agent.consolidation.provider`, `memory_agent.query_provider`, and `memory_agent.epoch_advice_provider`, but the fallback identity is not exposed as a yaml knob.

**Honest gap — discrepancy with `MEMORY.md`.** Project memory says the epoch-advice chain is "Cerebras → Groq llama → Cerebras llama → Ollama". Code does not have a "Cerebras llama" tier; the actual chain is Cerebras qwen-3-235b → Groq llama-4-scout → Ollama. Two providers, not three. If `MEMORY.md` is referenced as the source-of-truth elsewhere, expect this drift.

### Retry & backoff

| Layer | Behavior | Citation |
|-------|----------|----------|
| Malformed JSON (consolidation only) | 2 attempts, same provider | `consolidate.py:2196-2204` |
| 429 rate-limit (consolidation) | Exponential 30s → 60s → 120s, max 3 retries | `consolidate.py:2338-2387` |
| 429 calendar-day blocking (query) | 5min → 15min → 60min → rest-of-UTC-day; resets at midnight UTC | `query.py:53-57, 381-410` |
| Connect timeout | 10s | `consolidate.py:2405`, `query.py:1547` |
| Write timeout | 30s | same |
| Pool timeout | 10s | same |
| Read timeout | Per-provider (60s Gemini → 1800s OpenRouter) | config + `:2405` |
| Ollama concurrency | Semaphore = 1 (serialized) | `query.py:60, 1753` |

Provider switch is triggered by HTTP 5xx, connection error, or rate-limit exhaustion. Non-429 4xx and 5xx that are not retried fall straight to the next provider in the chain.

### Audit logging — `llm_audit_log`

Writer: `insert_audit_log()` (`db.py:331-376`), fire-and-forget — exceptions swallowed. Schema (`db.py:170-190`):

```
id, timestamp, call_type, algo, env, fold_number, iteration_number,
is_control_fold, provider, model_name, prompt_text, response_text,
response_parsed, latency_ms, success, error_text
```

| Concern | Behavior |
|---------|----------|
| `call_type` enum (observed) | `consolidation_phase_a`, `consolidation_phase_b`, `consolidation_stage_2`, `run_config`, `epoch_advice` |
| Latency | `int((time.monotonic() - t0) * 1000)` measured around the `_call_llm_with_retry()` boundary (`consolidate.py:1732, 2074-2076`; `query.py:1074-1076, 1219-1220`) — includes 429 backoff sleep time |
| `prompt_text` | Truncated to 50K chars (`query.py:1020`) |
| Training-context propagation | `algo, env, fold_number, iteration_number, is_control_fold` parsed from the query string by `_parse_query_context()` (`query.py:780`) and forwarded into the audit row |
| `error_text` on LLM failure | **Not captured.** When all providers fail, the row is written with `success=False` but `error_text` is `NULL` — no exception message is propagated. |

**Note — call_type taxonomy.** The canonical naming is Stage 1 / Phase A & B / Stage 2; emitted call_types are `consolidation_phase_a`, `consolidation_phase_b`, `consolidation_stage_2`. Prior notes that pre-date the Phase-A/B split used `consolidate_stage1_*` / `consolidate_stage2`; audit-log queries against the old names return zero rows.

## Configurable values (yaml)

Knobs live under `memory_agent.*` in `config/swingrl.yaml`. Roughly:

- `consolidation.provider` — primary provider name (default `"mistral"`)
- `consolidation.timeout_sec` — wall-clock cap per LLM call (default 1800)
- `consolidation.max_retries` — 429 retries per provider (default 3)
- `consolidation.backoff_base_sec` — exponential base (default 30)
- `consolidation.inter_phase_delay_sec` — pause between Phase A and Phase B (default 60)
- `query_provider` — primary for `run_config` (default `"gemini"`)
- `epoch_advice_provider` — primary for `epoch_advice` (default `"cerebras"`)
- `min_confidence` — pattern filter threshold (default 0.4)
- `cloud_block_on_429` — enable calendar-day blocking (default `true`)
- Per-provider `model_name`, `timeout_sec`, `max_tokens` (sub-blocks under each provider key)
- Per-algo epoch-advice cadence — see [`reward-shaping.md`](reward-shaping.md) and [`agent-architecture.md`](agent-architecture.md)

**Honest gap:** the exact `memory_agent.*` block was not exhaustively re-audited against `src/swingrl/config/schema.py` for this doc — re-verify field names and validators before changing any production value.

## Hardcoded values (not yaml-tunable — code edit required)

| Value | Location |
|-------|----------|
| `_MEMORY_BATCH_SIZE = 200` (Phase A per-prefix batch) | `consolidate.py:208` |
| `_FETCH_CHUNK = 10_000` (Phase B chunked read) | `consolidate.py:1802` |
| `_BATCH = 10_000` (archive UPDATE batch) | `db.py:455` |
| `_COLD_START_MIN_PATTERNS = 1` | `meta_orchestrator.py:52` |
| `_VALID_CATEGORIES` (18 options) | `consolidate.py:264-282` |
| `_RELEVANT_CATEGORIES["run_config"]` (9) / `["epoch_advice"]` (7) | `query.py:749-768` |
| `_SAFE_DEFAULTS`, `_SAFE_EPOCH_DEFAULTS` | `query.py:122-131` and adjacent |
| Composite-score weights (0.5 / 0.3 / 0.2) and 90-day recency decay | `db.py:666-697` |
| Conflict-detection adjacency window + filler-word list | `consolidate.py:311-343` |
| Backup providers (hard-coded fallback identity) | `consolidate.py:120`, `query.py:190` |
| Epoch-advice fallback chain (Cerebras → Groq → Ollama) | `query.py:273-279` |
| `cleanup_patterns.py` `RETIRE_IDS` / `RESTORE_IDS` dicts | `cleanup_patterns.py:26-45` |
| `<memory>` XML wrapper | `ingest.py:17-18, 57-58, 73-74` |
| Audit-log `prompt_text` truncation (50K) | `query.py:1020` |
| Auth env var name `MEMORY_API_KEY` | `auth.py:21-40` |

## Invariants

- Memory-system tables are written exclusively by the `swingrl-memory` container. The trainer talks HTTP, never SQL.
- All timestamps (`created_at`, `recorded_at`, `presented_at`, `last_confirmed_at`, `timestamp`) are `TIMESTAMPTZ` and stored in UTC. ET only at presentation (Discord, dashboards).
- Archive flag is the only soft-delete; hard delete never happens. Memory volume grows monotonically.
- Consolidation is event-driven (no scheduler). A single global `_run_lock` serializes overlapping `/consolidate` calls.
- Stage 2 fires only when both equity and crypto Phase A produced ≥1 active pattern in the same `/consolidate` call.
- No DB-level foreign keys: `consolidation_sources` joins `consolidations.id` and `memories.id` by convention only.
- The trainer-side memory client is fail-open. `client.ingest()`, `client.consolidate()`, `client.advise_run_config()`, and `client.advise_epoch()` always return — they never raise, never block training.
- Audit writes (`llm_audit_log`, `consolidation_quality`) are fire-and-forget by design — caller is never aborted on audit failure.
- LLM fallback is always to a dict — `_SAFE_DEFAULTS` / `_SAFE_EPOCH_DEFAULTS` — never to `None` and never to an exception.
- Conflict resolution is permanent until manual restore; superseded patterns are not requeried by the advice path.

## Known issues / open questions

- **Empirical: memory hurts training (iter 0-4).** Per `MEMORY.md` Active Session State, control folds outperformed treatment folds by 2.7-5.1× across iter 3-4 (treatment Sharpe 0.012-0.083 vs. control 0.034-0.42). The empirical signal contradicts the design intent. Cross-link: [`reward-shaping.md`](reward-shaping.md) Known issues; `MEMORY.md` Active Session State.
- **`MEMORY.md` epoch-advice chain wrong.** Project memory claims "Cerebras → Groq llama → Cerebras llama → Ollama"; code has only Cerebras qwen-3-235b → Groq llama-4-scout → Ollama. No "Cerebras llama" tier exists.
- **`call_type` naming history.** Code uses the canonical Stage 1 / Phase A & B / Stage 2 taxonomy: `consolidation_phase_a`, `consolidation_phase_b`, `consolidation_stage_2`. Prior notes that pre-date the Phase-A/B split used `consolidate_stage1_*` / `consolidate_stage2`; queries against the old names return zero rows.
- **Plan C (open per `MEMORY.md`):** crypto SAC memory volume bug — ~688K memories observed historically. Cited as observation, not extrapolation.
- **Plan D (open per `MEMORY.md`):** equity `consolidation_skipped_no_memories` — the equity side is sometimes skipping consolidation because it has no unarchived memories. Cause not yet diagnosed.
- **`patterns_presented` JSON has no reader yet.** `pattern_outcomes.patterns_presented` is plumbed end-to-end (DDL, client kwarg, router body, INSERT) as a forward-looking per-pattern attribution surface. The current `/training/pattern_effectiveness` endpoint joins by `(iteration, env_name)` only; the per-pattern attribution view is a future consumer.
- **Audit `error_text` is `NULL` on LLM failure.** When all providers exhaust, the audit row is written with `success=False` and `error_text` left empty — exception messages are not captured. Loses debuggability for hard failures.
- **Superseded patterns are never re-activated by the pipeline** — only manual `cleanup_patterns.py` can flip them back. There is no automated "supersession decayed, restore" path.
- **`/debug/consolidations?limit=100` cap on cold-start counter.** The cold-start gate uses this endpoint; if a single env ever has >100 active patterns, the count is undercounted. Below 100, this is moot.
- **`pattern_outcomes` has no UNIQUE constraint on `(iteration, env_name)`.** Already noted in [`training-data-capture.md`](training-data-capture.md#pattern_outcomes); accidental double-write would silently duplicate.
- **Ingest fail-open is silent.** Memory loss during memory-service downtime has no buffer, no retry, and no alert. Memory volume during outages is permanently lower than during healthy operation.
- **Stage 2 patterns have no DB-level link to Stage 1 sources.** The cross-env pattern references the per-env patterns through prompt inclusion only. Traceability stops at the Stage 1 layer.
- **Pre-commit hook gotcha (per handoff).** The repo's security hook flags certain Python serialization library names. This doc does not need to mention them; if anything trips the hook unexpectedly, fix the phrasing rather than skip the hook.

## Source of truth

| Concern | File |
|---------|------|
| Memory-service entrypoint + agents | `services/memory/app.py`, `services/memory/memory_agents/{ingest,consolidate,query}.py` |
| DB layer | `services/memory/db.py` |
| HTTP routes | `services/memory/routers/{core,training,debug}.py` |
| Auth | `services/memory/auth.py` |
| Pattern cleanup CLI | `services/memory/cleanup_patterns.py` |
| Trainer-side HTTP client | `src/swingrl/memory/client.py` |
| Cold-start gate + advice integration | `src/swingrl/memory/training/meta_orchestrator.py` |
| Per-epoch ingest + advice trigger | `src/swingrl/memory/training/epoch_callback.py` |
| Per-iteration `/consolidate` + `/training/record_outcome` call sites | `scripts/train_pipeline.py` |
| YAML knobs | `config/swingrl.yaml` `memory_agent.*` block |
| Schema (DDL + indexes + idempotency patterns) | [`training-data-capture.md`](training-data-capture.md) cluster 3, ultimately `src/swingrl/data/postgres_schema.py` |

## Changelog

- **2026-05-05** — Initial version.
