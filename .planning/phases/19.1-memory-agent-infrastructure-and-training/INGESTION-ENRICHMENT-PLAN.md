# Memory Ingestion Enrichment + Consolidation Pipeline Overhaul

**Date:** 2026-03-17
**Status:** All decisions resolved, ready for implementation

---

## Part 1: Ingestion Data Enrichment (ALL DECIDED)

### Category 1: Per-Fold OOS/IS Metrics — Add 8 (both OOS and IS)
- `calmar`, `rachev`, `avg_drawdown`, `max_dd_duration`
- `avg_win`, `avg_loss`, `trade_frequency_per_week`, `total_return`
- Skip `final_portfolio_value` (derivable)

### Category 2: Per-Fold FoldResult Fields — Add 3
- `train_range` dates, `train_range` bar count, `gate_result.failures`
- Skip `gate_result.details` (derivable)

### Category 3: Per-Fold Features Array — Add all
- HMM p_bull/p_bear min/max (currently only mean)
- All 6 macro min/max: VIX, yield_spread, rate_direction, fed_funds, cpi_yoy, unemployment
- Turbulence max (peak stress)

### Category 4: Per-Algo Header — Add 2
- Iteration number (0=baseline, 1-5=memory), Total timesteps

### Category 5: Per-Env Ensemble — Add all 7
- Per-algo calmar + rachev, Fold Sharpe variance, best/worst fold Sharpe
- Bear/Bull conditional Sharpe, WF parameters, env config

### Category 6: Per-Trade — 2 fields
- `max_single_loss`, `best_single_trade`
- `avg_holding_period` deferred (needs backtest.py change)

---

## Part 2: Consolidation Pipeline Overhaul

### Issue 1: System Prompt Update — DECIDED

Apply research-backed prompt engineering techniques (see LLM-PROMPT-RESEARCH.md):

**System prompt:**
- Role: quantitative analyst reviewing algorithmic trading results
- Grounding rule: only reference metrics from input data, cite at least 2 specific values per pattern
- Skeptical framing: ask "could this be random variance?" for each candidate
- Confidence calibration: anchored scale (0.4-0.7 typical for noisy financial data, >0.85 requires exception-free evidence)
- Category enum with definitions (14 categories)
- No CoT — direct JSON output (CoT increases hallucination for analytical tasks)
- Paired positive/negative constraints

**User prompt:**
- 14 few-shot examples with PLACEHOLDER values (labeled "FORMAT EXAMPLE ONLY — fictional data")
- Data delimiter: `<training_data>...</training_data>`
- Varied confidence values in examples (0.42, 0.67, 0.55) to avoid anchoring

**LLM call configuration:**
- NVIDIA NIM: `guided_json` via `nvext` (token-level schema enforcement, fastest path)
- Ollama Qwen3:14b: full JSON schema in `format` param, disable thinking mode (`/no_think`)
- Temperature: 0, max_tokens: 4096 (explicit to prevent truncation)
- Client-side JSON validation always (neither NIM nor Ollama guarantees valid JSON on truncation)

**14 Few-Shot Examples:**

Training Patterns (8):
1. `regime_performance` — algo Sharpe divergence in bull vs bear folds
2. `macro_transition` — all algos fail when fed_funds transitions sharply
3. `trade_quality` — algo trade frequency vs avg_win/avg_loss ratio
4. `iteration_progression` — mean_sharpe improvement/degradation across iterations
5. `overfit_diagnosis` — IS/OOS gap patterns across regimes
6. `drawdown_recovery` — max_dd_duration and avg_drawdown comparison
7. `data_size_impact` — gate pass rate vs train_bars
8. `cross_env` — algo preference differences between equity and crypto

Live Trading Patterns (5, one per Phase 20 endpoint):
9. `live_cycle_gate` — skip crypto during high turbulence + bear
10. `live_blend_weights` — favor A2C in bear regimes
11. `live_risk_thresholds` — tighten drawdown limits during VIX spikes
12. `live_position` — halve SAC positions during volatility
13. `live_trade_veto` — block SAC trades during rate hikes

Cross-Env Pattern (1, for Stage 2):
14. `cross_env_correlation` — equity bear onset predicts crypto drawdown

### Issue 2: Two-Stage Consolidation — DECIDED (Event-Driven)

**Stage 1: Per-Env (Cross-Algo)**
- Input: 3 algo memories + 1 ensemble for one env (~14K tokens)
- All 14 examples in prompt
- Output: N patterns with categories (unbounded, single LLM call)
- Finds: per-algo AND cross-algo patterns
- Dedup/conflict check before inserting each pattern

**Stage 2: Per-Iteration (Cross-Env)**
- Input: Stage 1 patterns from equity + crypto (~3K tokens)
- Output: Cross-env patterns with categories
- Only runs if BOTH envs produced Stage 1 patterns
- Skipped for single-env training

**Context window math (Kimi K2.5 = 128K tokens):**
- Stage 1: ~16K tokens (12.5% of limit)
- Stage 2: ~5K tokens (3.9% of limit)
- Dedup/conflict merge: ~14K tokens (10.6% of limit, with raw memories)

**Event-driven model (APScheduler removed):**
| Event | Trigger | Action |
|-------|---------|--------|
| `env_wf_complete` | After each env's walk-forward | Ingest raw memories |
| `iteration_complete` | After both envs done | Stage 1 (equity) + Stage 1 (crypto) + Stage 2 |
| `iteration_complete_single_env` | Only one env done | Stage 1 (that env), skip Stage 2 |
| `live_cycle_complete` | Phase 20+ | Ingest live results |
| `live_session_end` | Phase 20+ | Consolidate live memories |
| Manual | `POST /consolidate` | Full pipeline (debugging) |

### Issue 2b: Pattern Output Format — DECIDED

- Single LLM call returns unbounded array of patterns
- Each pattern has: `pattern_text`, `category` (enum), `affected_algos`, `affected_envs`, `actionable_implication`, `confidence`, `evidence`
- Category enum enforced at schema level (token-level via guided_json/format)
- Empty array acceptable — "no clear patterns" is a valid result
- Each pattern stored as separate consolidation row

### Issue 3: Meta-Trainer Consuming Consolidations — DECIDED (Option A)

`/training/run_config` endpoint queries consolidations table internally:
- Filter by `min_confidence_for_advice` (configurable, default 0.4)
- Only `status='active'` patterns
- Prefer Stage 2 over Stage 1 when both available
- Include category, confidence, confirmation count, evidence in context

### Issue 4: Feedback Loop — DECIDED (Tracking Infrastructure)

**No automatic confidence adjustment.** Track everything, human reviews.

**Presentation tracking:**
- `pattern_presentations` table logs when query agent includes a pattern in context
- Fields: consolidation_id, presented_at, iteration, env_name, request_type, full advice response

**Outcome recording:**
- `pattern_outcomes` table stores iteration results alongside presented pattern IDs
- Fields: iteration, env_name, gate_passed, sharpe, mdd, sortino, pnl (Phase 20+), patterns_presented

**New endpoints:**
- `POST /training/record_outcome` — called from train_pipeline.py after each iteration
- `GET /training/pattern_effectiveness` — human review of pattern impact data

### Issue 5: Pattern Lifecycle Management — DECIDED (NEW)

**Pattern status lifecycle:** `active` → `superseded` → `retired`

**Duplicate detection + LLM merge:**
1. Before inserting each new pattern, check existing active patterns with same category + env_name + affected_algos
2. Keyword similarity check (reuse conflict detection heuristic, looking for agreement)
3. If similar: LLM merge call — feed both patterns + raw source memories (via `consolidation_sources` join table)
4. Old pattern: `status='superseded'`, `superseded_by=new_id`
5. New merged pattern: `confirmation_count = old.confirmation_count + 1`
6. **Pairwise limit:** Max 2 patterns per merge. 3+ similar → sequential pairs.

**Conflict detection + LLM resolve:**
1. Enhanced keyword heuristic (positive vs negative sentiment words)
2. If contradiction: both patterns get same `conflict_group_id`
3. LLM resolve call — feed both patterns + raw source memories with explicit conflict context
4. Resolution pattern supersedes both. Old patterns: `status='superseded'`
5. **Pairwise limit:** Max 2 patterns per resolve.

**Source tracing:**
- `consolidation_sources` join table links patterns to their source raw memories
- Enables fetching raw memories for dedup merges and conflict resolution
- Populated during consolidation when archiving source memories

### Issue 6: Event-Driven Model — DECIDED (NEW)

Remove 30-min APScheduler entirely from `services/memory/app.py`.
All consolidation is triggered explicitly:
- From `train_pipeline.py` at `iteration_complete` event
- From `/consolidate` endpoint for manual debugging
- From future Phase 20 live trading events

### Issue 7: Critical Bug — `:historical` Assertion — DISCOVERED

`MemoryClient.ingest_training()` (line 113 of `src/swingrl/memory/client.py`) contains:
```python
assert source.endswith(":historical")
```
But `train_pipeline.py` calls it with `source="walk_forward:ppo"`.
The AssertionError is caught by the outer `try/except` on line 786, silently dropping ALL walk-forward memories.
**Impact:** Zero WF memories have been stored since Phase 19.1 began.
**Fix:** Remove the assertion.

---

## Pending Decisions

**NONE** — All decisions resolved.

---

## DB Schema Changes

### Modified: `consolidations` table (new columns)
```sql
category, affected_algos, affected_envs, actionable_implication, confidence,
evidence, stage, env_name, confirmation_count, last_confirmed_at,
superseded_by, status, conflict_group_id
```

### New: `consolidation_sources` (join table)
```sql
consolidation_id, memory_id — links patterns to source raw memories
```

### New: `pattern_presentations` (tracking)
```sql
id, consolidation_id, presented_at, iteration, env_name, request_type, advice_response
```

### New: `pattern_outcomes` (tracking)
```sql
id, iteration, env_name, gate_passed, sharpe, mdd, sortino, pnl, recorded_at, patterns_presented
```

---

## Config Additions

```yaml
memory_agent:
  consolidation:
    min_confidence_for_advice: 0.4    # patterns below this excluded from query context
    max_patterns_per_merge: 2         # pairwise comparison limit for dedup/conflict
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/swingrl/memory/client.py` | Remove `:historical` assertion (critical bug fix) |
| `scripts/train_pipeline.py` | Enriched ingestion (40+ fields), iteration threading, env source tags, outcome recording |
| `services/memory/db.py` | Schema migration, 3 new tables, new insert/query/update functions |
| `services/memory/memory_agents/consolidate.py` | System prompt, schema, two-stage, dedup/conflict lifecycle, validation |
| `services/memory/memory_agents/query.py` | Enriched context building, presentation tracking |
| `services/memory/routers/training.py` | New `/training/record_outcome`, `/training/pattern_effectiveness` endpoints |
| `services/memory/routers/core.py` | `/consolidate` unchanged (still manual trigger) |
| `services/memory/app.py` | Remove APScheduler |
| `config/swingrl.yaml` | Add `min_confidence_for_advice`, `max_patterns_per_merge` |
| `src/swingrl/config/schema.py` | Add config fields for new consolidation settings |
| `tests/test_memory_service.py` | 23 new/updated tests |

## Verification

1. `uv run pytest tests/ -v` — full suite green
2. `/project:lint` + `/project:typecheck` — clean
3. Push, pull on homelab, rebuild (`--no-cache`), restart training
4. After first equity WF: verify enriched memories + correct source tags + assertion bug fixed
5. After both envs complete: verify Stage 1 patterns per env with categories + evidence
6. Verify Stage 2 cross-env patterns (only if both envs have Stage 1 output)
7. Verify dedup check ran, no duplicate patterns
8. After second iteration: verify presentation tracking + outcome recording
9. Check consolidation quality: patterns cite specific numbers from enriched data
10. Verify APScheduler removed — no background consolidation
