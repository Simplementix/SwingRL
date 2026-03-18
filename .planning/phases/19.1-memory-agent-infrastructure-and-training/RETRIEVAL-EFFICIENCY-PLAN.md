# Memory Agent Configurability & Retrieval Efficiency Analysis

## Context
Three design questions raised during Phase 19.1:
1. Should query and consolidation agent models be operator-configurable?
2. Should Ollama container memory auto-size to match the configured model?
3. Is the current retrieval pattern (SQL query → text dump → Ollama prompt) efficient?

---

## Q1: Model Configurability

### Current State
| Agent | Model source | Configurable? |
|-------|-------------|---------------|
| **Consolidation** | `ConsolidationConfig.providers[provider].default_model` | Yes — full provider map with per-provider models, env var overrides |
| **Query** | Hardcoded `_QUERY_MODEL = "qwen2.5:3b"` in `query.py:77` | **No** — ignores `MemoryAgentConfig.ollama_smart_model` |

`MemoryAgentConfig` already has `ollama_smart_model` and `ollama_fast_model` fields (`schema.py:311-312`) but the memory service container **cannot import** `swingrl.config.schema` (cross-container boundary, Pitfall 4). Instead, `query.py` reads `/app/config/swingrl.yaml` via `yaml.safe_load` but only extracts `min_confidence_for_advice` — it never reads the model name.

### Recommendation: Make query model configurable via YAML

**Changes needed:**
1. `query.py:_load_min_confidence()` → rename to `_load_query_config()`, also extract `memory_agent.ollama_smart_model` from YAML, default `"qwen2.5:3b"`
2. Replace `_QUERY_MODEL = "qwen2.5:3b"` with the loaded value
3. No schema changes needed — `MemoryAgentConfig.ollama_smart_model` already exists

**Consolidation**: Already fully configurable. No changes needed.

**Cost**: ~15 lines changed in `query.py`. No new dependencies.

---

## Q2: Ollama Container Memory Sizing

### Current State
`docker-compose.prod.yml:23`: `mem_limit: 4g` — hardcoded. Sized for qwen2.5:3b (~2.0-2.4 GB runtime).

### Recommendation: Environment variable with documented model-to-memory mapping

Docker Compose supports `${VAR:-default}` substitution natively.

**Changes:**
1. `docker-compose.prod.yml`: `mem_limit: ${OLLAMA_MEM_LIMIT:-4g}`
2. Add comment table in docker-compose and `config/swingrl.prod.yaml.example`:

```
# Model memory requirements (set OLLAMA_MEM_LIMIT in .env):
#   qwen2.5:3b   → 4g  (model ~2.0 GB + inference overhead)
#   qwen2.5:7b   → 8g  (model ~4.7 GB + overhead)
#   qwen3:8b     → 10g (model ~5.5 GB + overhead)
#   qwen3:14b    → 12g (model ~9.0 GB + overhead)
```

**Why not auto-sizing?** Docker Compose runs before the application starts — it can't read `swingrl.yaml` to determine the model. The operator sets both `ollama_smart_model` in YAML and `OLLAMA_MEM_LIMIT` in `.env`. This is the standard Docker pattern and avoids a templating layer.

**Cost**: 3-4 lines in docker-compose + documentation.

---

## Q3: Memory Retrieval Efficiency Analysis

### Current Flow (per `run_config` or `epoch_advice` call)

```
MetaTrainingOrchestrator._query_run_config()
  → HTTP POST /training/run_config  (meta_orchestrator.py:238)
    → QueryAgent.advise_run_config()  (query.py:236)
      → _build_context()  (query.py:311)
        → get_active_consolidations(stage=2, min_confidence=0.4)  — ALL cross-env patterns
        → get_active_consolidations(stage=1, min_confidence=0.4)  — ALL per-env patterns
        → get_memories(archived=False, limit=20)                  — 20 raw memories
      → Concatenate ALL into one text blob
      → POST to Ollama qwen2.5:3b with full blob as context
      → Parse JSON, clamp, return
```

### Problems Identified

| Issue | Impact | Severity |
|-------|--------|----------|
| **No env/algo filtering on retrieval** | Sends equity patterns when advising crypto PPO. Wastes context tokens, confuses 3B model. | High |
| **Unbounded pattern count** | All active patterns sent. As system matures (50+ patterns), exceeds useful context for a 3B model. | High |
| **Raw memories always included** | 20 pre-consolidation memories add noise once consolidations exist. | Medium |
| **No caching** | Every call re-queries DB + re-calls Ollama even if patterns unchanged since last call 5 seconds ago. | Medium |
| **Full LLM call for every epoch** | `epoch_advice` called every N epochs — same patterns, slightly different metrics each time. | Low (cadence=5 epochs) |

### Recommendations (ordered by impact/effort ratio)

#### R1: Filter consolidations by env + algo (HIGH impact, LOW effort)
`_build_context()` already receives no env/algo context. The query string contains `env=equity algo=PPO` but isn't parsed until `_track_presentations()`.

**Change**: Parse `env_name` and `algo_name` from query string at top of `advise_run_config()`/`advise_epoch()`. Pass to `_build_context()`. Use `get_active_consolidations(env_name=env_name, ...)` which already supports env filtering AND already filters `status='active'` (hardcoded in `db.py:447`). Post-filter by `affected_algos` containing the current algo (JSON array in consolidation row).

**Note on status='active'**: Already enforced — `get_active_consolidations()` has `status = 'active'` as its first WHERE clause. Superseded/retired patterns are never returned.

**Effect**: Drops irrelevant patterns from context. A crypto-only pattern won't pollute an equity PPO query.

**Files**: `query.py` — modify `_build_context()` signature, parse env/algo in callers.

#### R2: Category-aware pattern cap (HIGH impact, LOW effort)
Instead of a flat `LIMIT 10` (which could miss old-but-relevant patterns in underrepresented categories), use **per-category limits** with composite ranking.

**Approach**: Define relevant categories per request type:
- `run_config`: `regime_performance`, `overfit_diagnosis`, `iteration_progression`, `data_size_impact`, `macro_transition`, `cross_env`
- `epoch_advice`: `drawdown_recovery`, `trade_quality`, `overfit_diagnosis`, `iteration_progression`

Fetch top-3 per relevant category, ordered by composite score:
```
score = (confidence * 0.5) + (confirmation_count / max_confirmation * 0.3) + (recency * 0.2)
```
where `recency = 1.0 - (days_since_created / 90)` clamped to [0, 1].

**Effect**: ~12-18 patterns max, spread across relevant categories. Old high-confidence patterns with many confirmations rank above new weak ones. No relevant pattern is systematically excluded.

**Files**: `db.py` — add `limit` param + `ORDER BY` composite score. `query.py` — pass request_type to `_build_context()` for category selection.

#### R3: Skip raw memories when consolidations exist (MEDIUM impact, LOW effort)
**Change**: In `_build_context()`, only fetch raw memories if `len(consolidations) < 3`. Consolidations are strictly higher quality (LLM-synthesized, deduplicated, confidence-scored).

**Effect**: Cleaner context, less noise for the 3B model to parse.

**Files**: `query.py` — conditional in `_build_context()`.

#### R4: Cache advisory results (MEDIUM impact, MEDIUM effort)

**Cache key**: `f"{env_name}:{algo_name}:{regime_hash}"` — each algo+env+regime combo gets its own entry. With 3 algos × 2 envs = 6 distinct keys, parallel algo training never collides.

**Invalidation**: NOT time-based (consolidation is event-driven, not on a 30-min schedule). Instead:
- Before using cache, query `MAX(created_at)` from consolidations table (cheap SQL, no full scan needed with index)
- If `max_created_at > cache_timestamp` → bust ALL cache entries (new pattern = new advice needed)
- Regime changes → automatic cache miss (regime is part of the key)
- Different algos → automatic cache miss (algo is part of the key)

**Parallel training**: 3 algos train concurrently for the same env. Each has a different cache key (`equity:PPO:regime_x`, `equity:A2C:regime_x`, `equity:SAC:regime_x`). No cross-contamination. Cache helps **across iterations** — if iteration N+1 has the same regime and no new consolidations since iteration N, cache hits avoid 6 redundant Ollama calls.

**Implementation**: Module-level `_advisory_cache: dict[str, tuple[float, str, dict]]` (timestamp, consolidation_max_ts, result). `QueryAgent` is instantiated per-request (stateless), so cache must be module-level. Thread-safe via GIL (FastAPI async + single worker).

**Files**: `query.py` — add cache dict + check/invalidate logic in `advise_run_config()` and `advise_epoch()`.

#### R5: Embedding-based retrieval (NOT recommended — reassess at 500+ patterns)
Replace "dump all patterns" with semantic similarity search using vector embeddings.

**Not recommended now**: With R1+R2, the query pipeline selects ~12-18 patterns via structured SQL filters (env, algo, category, composite score). This is precise and deterministic. Pattern count will stay under 200 for many months of training.

**When it becomes necessary**: If the system reaches 500+ active patterns across many categories, SQL filtering alone may miss semantically relevant patterns that don't match the predefined category→request_type mapping. At that point, embedding similarity (via Ollama's `/api/embeddings` endpoint — no new dependency) would find patterns by meaning rather than metadata. But that's a Phase 20+ concern at earliest.

**Why it's a future problem specifically**: Each training iteration can produce 0-5 new consolidation patterns. At ~2 iterations/week × 2 envs × ~3 patterns = ~12 patterns/week. Reaching 500 patterns takes ~10 months. The category-aware SQL approach handles this entire growth period.

### Expected Improvement from R1-R4

| Metric | Before | After R1-R4 |
|--------|--------|-------------|
| Patterns in context | All active (unbounded) | 5-10 relevant to env+algo |
| Raw memories in context | Always 20 | 0 when consolidations exist |
| Ollama calls per iteration | 6 (3 algos × 2 envs) | 2-3 (cache hits for shared regime) |
| Context token count | ~2000-5000+ | ~500-1500 |
| qwen2.5:3b response quality | Diluted by irrelevant patterns | Focused on relevant patterns |

---

## Implementation Plan

### Phase A: Model Configurability (Q1 + Q2)
1. `services/memory/memory_agents/query.py` — rename `_load_min_confidence()` → `_load_query_config()`, also extract `memory_agent.ollama_smart_model` from YAML, use as `_QUERY_MODEL`
2. `docker-compose.prod.yml` — `mem_limit: ${OLLAMA_MEM_LIMIT:-4g}` + model-memory table comment
3. `config/swingrl.prod.yaml.example` — add model-memory guidance comment

### Phase B: Retrieval Efficiency (R1-R3)
1. `services/memory/memory_agents/query.py`:
   - Parse `env_name`/`algo_name` from query string at top of `advise_run_config()`/`advise_epoch()` (reuse existing parsing logic from `_track_presentations()`)
   - Add `request_type` param to `_build_context()` for category selection
   - Pass env/algo/request_type to `_build_context()`, filter consolidations by env + post-filter by algo
   - Define `_RELEVANT_CATEGORIES` map: `run_config` → [...], `epoch_advice` → [...]
   - Fetch top-3 per relevant category using composite score ranking
   - Skip raw memories when `len(consolidations) >= 3`
2. `services/memory/db.py`:
   - Add `limit` and `categories` params to `get_active_consolidations()`
   - Add composite score `ORDER BY`: `confidence * 0.5 + confirmation_count * 0.3 + recency * 0.2`

### Phase C: Caching (R4)
1. `services/memory/memory_agents/query.py`:
   - Module-level `_advisory_cache: dict[str, tuple[float, str, dict]]`
   - Cache key: `f"{env_name}:{algo_name}:{regime_hash}"`
   - Invalidation: check `MAX(created_at)` from consolidations before cache lookup
   - Separate caches for `run_config` and `epoch_advice`

### Tests
- New: `_build_context()` with env/algo filtering returns only matching patterns
- New: category-aware retrieval returns top-3 per category, respects composite score
- New: cache hit returns previous result without Ollama call
- New: cache invalidated when consolidation `MAX(created_at)` advances
- `uv run pytest tests/ -x -q` — all pass

### Verification
- Manual: change `ollama_smart_model` in YAML, verify query.py picks it up
- Manual: set `OLLAMA_MEM_LIMIT=8g` in `.env`, verify docker-compose uses it
- Log inspection: verify filtered pattern count in `run_config_advised` log line
