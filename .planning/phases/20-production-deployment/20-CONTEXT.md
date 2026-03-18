# Phase 20: Production Deployment - Context

**Gathered:** 2026-03-15 (original), 2026-03-18 (updated with 19.1 impacts)
**Status:** Ready for re-planning

<domain>
## Phase Boundary

Deploy the homelab Docker stack with paper trading firing on schedule (equity 4:15 PM ET, crypto every 4H). Both containers healthy, startup validation passes, and end-to-end paper trade cycle completes. Additionally, implement the memory live enrichment pipeline from `memory_live_enrichment.md` — memory agent controls the trading pipeline (cycle gate, blend weights, risk thresholds, position advice, trade veto) without modifying RL agent obs space. Fix missing production dependencies. Add MacroWatcher for economic calendar events.

**Removed from original scope:** Alpaca FillIngestor and Binance FillIngestor (websocket fill threads). Post-cycle memory ingestion covers all data needed for pattern finding; `trades` table via FillProcessor handles execution audit logging.

</domain>

<decisions>
## Implementation Decisions

### Startup Validation
- Standalone `scripts/startup_checks.py` called by `main.py` before scheduler init — also runnable via `docker exec` for debugging
- **Hard fail: exit(1)** on any validation failure — Docker restart policy retries
- Checks performed:
  - Broker credentials: ping Alpaca + Binance.US APIs to confirm auth
  - DB tables exist: verify SQLite (trading_ops.db) + DuckDB (market_data.ddb) have expected tables, create if absent
  - Model files present: check `models/active/{equity,crypto}/{ppo,a2c,sac}/model.zip` + `vec_normalize.pkl` — warn but don't block
  - Memory service reachable: check `swingrl-memory:8889/health` — warn if down (fail-open)
  - Memory seeding status: query memory.db row count, warn if empty (operator needs to run `seed_memory_from_backtest.py`)
  - Bind-mount writability: touch temp file in each mount dir (data/, db/, models/, logs/, status/)
- Structlog output only (no JSON report file)
- Full production dependency audit (not just SQLAlchemy) — researcher does build-and-trace to find all ImportErrors

### Deployment Method
- **Git pull + docker compose build** on homelab (consistent with Phase 18/19 CI pattern)
- `scripts/deploy.sh`: git pull → docker compose build → docker compose down → docker compose up -d → wait for healthy → run `scripts/setup_ollama.sh` (model pull)
- .env is **manual, never touched** by deploy.sh — only checks it exists and has required keys (no empty values)
- deploy.sh handles **stack only** — data ingestion, training, seeding are separate commands
- No rollback mechanism — git revert + rebuild is sufficient
- Smoke test is **separate** from deploy.sh (operator runs independently)
- Runbook (Phase 24) will include both deploy.sh usage AND manual commands

### Memory Live Enrichment
- **Included in Phase 20** (not deferred) — memories are seeded from training, so no cold start concern
- **All endpoints enabled immediately** — seeded memories provide warm start context
- **No obs space expansion** — RL agents stay at 156 equity / 45 crypto dims, unaware of memory
- Memory agent controls the **execution pipeline from the outside**: cycle gate, blend weights, risk thresholds, position advice, trade veto, post-trade ingest
- **Integration point:** hooks inside `ExecutionPipeline.execute_cycle()` — both scheduler and manual `run_cycle.py` get memory automatically
- **Per-endpoint config toggles:** `memory_agent.live_endpoints.{cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto}` — all default to `true` when `memory_agent.enabled=true`
- **Skip /live/obs_enrichment** endpoint entirely — no obs space changes needed
- 5 live endpoints implemented in existing `services/memory/` FastAPI service (new `/live/` router)
- **Reuse query pipeline:** All live endpoints call `_build_context(request_type='live_trading', env_name=...)` for filtered patterns. Same R1-R3 filtering, composite scoring, and category-aware caps as training endpoints.
- **Client passes market state:** LiveMemoryClient passes current_regime, hmm_probs, vix, positions as fields in the /live/ request body. Memory service does NOT query swingrl's DuckDB.
- **Model:** All live endpoints use `ollama_fast_model` from config (default qwen2.5:3b). Ollama only, no cloud API fallback. If Ollama down, fail-open to safe defaults.
- **No caching for live endpoints:** Market state changes each cycle, cache would rarely hit. R4 caching applies to training run_config/epoch_advice only.
- **Dedicated system prompts per endpoint** — each prompt tailored to that decision's domain. Apply same research-backed techniques as consolidation (LLM-PROMPT-RESEARCH.md): role definition, grounding rule, paired constraints, no CoT.
- **Schema enforcement via Ollama `format` param** — live endpoints define response schemas passed to Ollama. Client-side JSON validation always, fallback to safe defaults on parse failure.
- **HTTP client: stdlib urllib** (consistent with existing MemoryClient in `src/swingrl/memory/client.py`, no new dependency)

### Post-Trade Memory Ingestion (UPDATED)
- **6 memories per cycle** (5 per-endpoint + 1 cycle summary)
- **Per-endpoint memories** tagged `live:{env}:{endpoint}` (e.g., `live:equity:trade_veto`):
  - Endpoint decision (allow/block, weights, thresholds) + rationale
  - Market state snapshot: regime, HMM probs, VIX, turbulence, current positions
  - Whether decision was clamped (e.g., blend weights hit ±0.20 bound)
- **Cycle summary memory** tagged `live:{env}:cycle_summary`:
  - All endpoint decisions summarized
  - Orders submitted this cycle (symbol, side, quantity, fill_price)
  - Realized P&L for positions closed this cycle
  - Which decisions were clamped
  - NO unrealized P&L or portfolio value (noise, not useful for pattern finding)
- **Individual POSTs via background thread** — 6 calls to existing /ingest endpoint (~10ms total on Docker network). No batch endpoint needed.
- **Sequential background task:** ingest all 6 memories → check consolidation cadence → consolidate if due. All sequential within one background thread. Trading pipeline returns immediately.
- **Ingestion completes before consolidation check** — ensures newly ingested memories are included in consolidation.

### Live Consolidation (NEW — replaces APScheduler)
- **Cadence:** Weekly for equity, every 3 days for crypto. Configurable via YAML: `memory_agent.live_consolidation.equity_interval_days=7`, `memory_agent.live_consolidation.crypto_interval_days=3`
- **Trigger:** Post-cycle check. After ingesting live memories, check `last_consolidated_at` for this env. If cadence elapsed, fire consolidation async.
- **Same env only:** Equity post-cycle only checks equity cadence. Crypto only checks crypto. Independent.
- **Uses existing consolidation pipeline:** Two-stage (per-env Stage 1, cross-env Stage 2 if both envs have new patterns). Same dedup, conflict resolution, pattern lifecycle.
- **Shared consolidation:** Training and live memories consolidated together. Live outcomes can confirm training patterns (increment confirmation_count). Dedup checks ALL active patterns regardless of origin.
- **All active patterns visible to live endpoints:** Query pipeline returns all active patterns (training-origin + live-origin) filtered by category + env + algo. No origin-based filtering.
- **No new categories needed:** Existing 14 categories (8 training + 5 live + 1 cross-env) cover all cross-concern patterns.
- **Confidence via confirmation_count:** When live confirms a training pattern, existing dedup pipeline merges and increments count. No special live_confirmed flag (add later if needed after observing real patterns).

### Blend Weights Integration
- Memory **overrides with bounds**: each algo weight clamped to **±0.20** from Sharpe-softmax baseline
- Bound is **configurable via YAML**: `memory_agent.live_bounds.blend_weight_delta=0.20` — memory can tighten but not widen beyond config
- Weights **reset each cycle** — each cycle queries `/live/blend_weights` fresh, no state carryover

### Circuit Breaker Overrides
- **Save + restore pattern**: `apply_overrides()` saves current values to `_saved_*` fields, sets new values; `reset_overrides()` restores originals
- ExecutionPipeline calls `reset_overrides()` at cycle start, `apply_overrides()` after memory risk thresholds received
- **Always log** original and override thresholds (audit trail)
- Halt events in `circuit_breaker_events` table record **override thresholds** (not config defaults) + `override_active` boolean column
- Live bounds **configurable via YAML** as maximum envelope — memory can tighten but not widen beyond config values

### Memory Seeding Workflow
- Seeding runs **after training, before first deploy** — operator runs `seed_memory_from_backtest.py` manually once
- **Seed + consolidate**: add `POST /consolidate` call at end of seeding so memory agent has synthesized patterns for first live cycle
- **Seed only once** — no re-seeding after Phase 22 automated retraining; organic live memories accumulate naturally
- startup_checks.py warns if memory is empty

### MacroWatcher
- Researcher finds best **free economic calendar API** for high-impact events (FOMC, CPI, NFP, PCE, GDP, PPI, JOLTS)
- **Fallback: hardcode known schedule** (FOMC meets 8x/year on known dates, CPI/NFP have predictable release patterns)
- **Poll interval configurable** via YAML: `memory_agent.macro_watcher.poll_interval_seconds` (default 300)
- Config fields for macro watcher: Claude's discretion during implementation

### Schedule Verification
- `scripts/show_schedule.py`: CLI that prints all 12 jobs with next fire times
- **Reconstruct from config** (doesn't require running scheduler) — imports `create_scheduler_and_register_jobs`, creates temporary scheduler, reads next_run_times
- Display times in **both ET and UTC**
- TZ=America/New_York kept in docker-compose + APScheduler timezone params — verify no double-conversion in testing

### End-to-End Smoke Test
- `scripts/smoke_test.py`: triggers one equity cycle + one crypto cycle immediately via `docker exec`
- **Real paper mode**: Alpaca paper orders (equity), simulated fills (crypto)
- **Full pipeline with memory**: exercises all memory touchpoints (cycle gate, risk thresholds, blend weights, position advice, trade veto, post-trade ingest)
- **Pass/fail checklist to stdout**: features assembled, model loaded, risk check passed, order submitted, fill logged, memory ingested — exit 0 all pass, exit 1 any fail

### Docker Resource Limits (ALREADY APPLIED in Phase 19.1)
- swingrl-ollama: **4GB / 4 CPU, cpuset=0-3** (pinned to P-cores 0-1, 4 HT threads at 4.8 GHz — dedicated for LLM inference. Only qwen2.5:3b local ~2.4GB)
- swingrl-memory: **1GB / 1 CPU** (unpinned, shares remaining cores)
- swingrl: **no mem_limit / 8 CPU** (unpinned, shares remaining cores)
- swingrl-dashboard: **512MB / 0.5 CPU** (unpinned)
- **Already applied** in `docker-compose.prod.yml` — no Phase 20 work needed

### Config Schema Updates
- `memory_agent.live_endpoints.*`: per-endpoint toggles (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto) — all default true
- `memory_agent.live_bounds.*`: blend_weight_delta, circuit_breaker_deltas — configurable maximum envelope, memory can tighten only
- `memory_agent.live_consolidation.equity_interval_days`: default 7 (weekly)
- `memory_agent.live_consolidation.crypto_interval_days`: default 3
- `memory_agent.macro_watcher.*`: poll_interval_seconds, API config — Claude's discretion on structure

### Dependencies (UPDATED)
- **SQLAlchemy>=2.0,<3**: missing — needed by APScheduler SQLAlchemyJobStore
- **requests**: used in 4 modules but not declared as explicit dep — verify/add
- **python-binance**: REMOVED — no longer needed (fill ingestors dropped, BinanceSimAdapter uses raw requests)
- **Full audit**: researcher does production Docker build to trace all ImportErrors

### Testing Strategy (UPDATED)
- **Mock Ollama** in all unit tests — fast, deterministic, no Docker needed
- **Explicit fail-open tests**: each client module tests connection refused → safe default, timeout → safe default, HTTP 500 → safe default, malformed JSON → safe default
- Test `startup_checks.py` with mocked broker APIs and DB connections
- Test post-cycle memory ingestion: verify 6 memories ingested with correct tags and content
- Test consolidation cadence check: verify consolidation triggered when cadence elapsed, skipped when not
- Skip `deploy.sh` tests — verified during homelab deployment
- Integration tested during homelab deployment checkpoint
- **Removed:** Fill ingestor websocket mock tests (Alpaca TradingStream, python-binance BinanceSocketManager)

### Healthcheck Improvements
- Add memory service reachability check (HTTP to swingrl-memory:8889/health)
- Add Ollama reachability check (HTTP to swingrl-ollama:11434/api/tags)
- Fail = UNHEALTHY with degraded flag (trading continues fail-open)

### Error Recovery
- Track **consecutive failures per env** in SQLite
- After **3 consecutive failures**: fire Discord warning (if configured) + structlog error
- No auto-retry within same cycle — wait for next scheduled fire

### Status File
- Write `status/status.json` after each cycle: last_cycle_time, positions, memory_health, scheduler_state, next_fire_times
- Mounted via `./status:/app/status` bind mount (already in docker-compose.prod.yml)

### Logging and Observability
- **Structured log with rationale** for each memory call: endpoint, env, decision, rationale (from LLM), latency_ms, clamped (bool)
- **INFO for decisions, DEBUG for details** — key decisions always visible in production logs
- **DuckDB table `memory_live_decisions`**: timestamp, env, endpoint, decision_json, rationale, latency_ms, clamped — ops audit trail (separate from pattern_presentations which tracks pattern effectiveness)
- **Memory latency alert**: if avg response time > 3s for 3 consecutive calls → structlog warning + Discord alert (if webhook configured)
- Dashboard display of memory decisions **deferred to future phase**

### Homelab CI Updates
- CI tests **both containers**: swingrl (existing) + services/memory/ (new live endpoints)
- **Docker compose build only** (no start): `docker compose -f docker-compose.prod.yml build` validates Dockerfiles and compose config
- No full compose up/health check in CI — verified during homelab deployment

### Claude's Discretion
- Exact system prompts for each live endpoint (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto) — following LLM-PROMPT-RESEARCH.md techniques
- MacroWatcher config field structure
- Whether Alpaca streaming needs separate auth vs same keys
- Background ingest queue implementation details (queue depth, batch size)
- Exact status.json schema
- Exact response schemas for each live endpoint (Ollama format param)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Memory Architecture
- `docs/memory_live_enrichment.md` — Live enrichment architecture, endpoint definitions, integration patterns. Exclude Section 5 (obs space expansion — deferred)
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/INGESTION-ENRICHMENT-PLAN.md` — Enriched ingestion fields, two-stage consolidation pipeline, event-driven model, pattern lifecycle management, system prompt design
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/RETRIEVAL-EFFICIENCY-PLAN.md` — R1-R3 retrieval filtering (env/algo, category-aware caps, cold-start fallback), R4 caching, model configurability

### Prompt Engineering
- `docs/LLM-PROMPT-RESEARCH.md` — Research-backed prompt techniques for memory agents (role, grounding, skeptical framing, confidence calibration, paired constraints)

### Specification Summary
- `docs/12-specification-summary.md` — Canonical reference card for all SwingRL specs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 19.1 already built)
- `services/memory/memory_agents/consolidate.py`: Two-stage consolidation with cloud/Ollama dispatch, 14 categories, dedup/conflict resolution, pattern lifecycle — extend with live consolidation cadence check
- `services/memory/memory_agents/query.py`: `_build_context()` with R1-R3 filtering (env, algo, category-aware caps, composite scoring, cold-start fallback) — reuse for live endpoints via `request_type='live_trading'`
- `services/memory/db.py`: Full schema with `consolidations`, `consolidation_sources`, `pattern_presentations`, `pattern_outcomes` tables — add `memory_live_decisions` table and `last_consolidated_at` tracking
- `services/memory/routers/training.py`: Training advisor endpoints — pattern for live endpoint router
- `src/swingrl/memory/client.py`: MemoryClient with fail-open urllib pattern — extend with LiveMemoryClient for /live/ endpoints
- `src/swingrl/execution/fill_processor.py`: FillProcessor._record_trade() — already logs all fills to trades table (no separate fill ingestors needed)

### Established Patterns
- Event-driven consolidation: consolidation triggered by events (iteration_complete, manual POST), not timers
- Fail-open memory: all memory calls wrapped in try/except → safe defaults
- Config via YAML + env vars: memory service reads mounted swingrl.yaml for config values
- Composite scoring: confidence (50%) + confirmation_count (30%) + recency (20%) for pattern ranking
- Schema enforcement: Ollama `format` param for JSON schema, client-side validation always

### Integration Points
- `ExecutionPipeline.execute_cycle()` — where memory hooks integrate (5 live endpoint calls + post-cycle ingestion)
- `CircuitBreaker.__init__()` → `_max_dd`, `_daily_limit` — save/restore targets for memory overrides
- `EnsembleBlender.sharpe_softmax_weights()` — blend weight adjustment target
- `scripts/main.py` `build_app()` → add startup_checks, memory thread startup
- `docker-compose.prod.yml` → verify bind mounts, add any new env vars
- `pyproject.toml` → add missing deps (SQLAlchemy, verify requests)
- `services/memory/app.py` → add live router
- `config/swingrl.yaml` → add live_endpoints, live_bounds, live_consolidation, macro_watcher config

</code_context>

<specifics>
## Specific Ideas

- Memory agent controls trading from the OUTSIDE — RL agents are unaware of memory, memory controls the pipeline
- "No cold start" — backtest seeding gives memory agent warm context from day one
- Per-endpoint toggles give granular control even though all start enabled
- Both ET and UTC in schedule display — crypto jobs are scheduled in UTC
- Latency alerting via both structlog + Discord (graceful if Discord not configured yet)
- Reference spec: `memory_live_enrichment.md` (architecture, code examples, endpoint definitions) — exclude Section 5 (obs space expansion)
- Resource limits already applied in Phase 19.1 — no changes needed
- Live consolidation cadence chosen for data density: ~25-30 raw memories/week (equity) gives meaningful patterns; per-cycle would be too thin
- Post-cycle ingestion captures ALL memory agent decisions (not just trades), providing rich data for consolidation even when no trades execute

</specifics>

<deferred>
## Deferred Ideas

- Obs space expansion (memory dims 156→164, 45→53) — requires retraining, defer to Phase 22 or later
- Memory dashboard tab in Streamlit — deferred from Phase 19 context, DuckDB table ready for it
- Progressive ramp-up of live endpoints — decided against (all enabled immediately with seeded memories)
- Inbox file watcher for memory ingestion — deferred from Phase 19
- Multi-pass consistency for consolidation patterns — review after seeing initial pattern quality
- Websocket fill ingestors (Alpaca TradingStream, Binance.US BinanceSocketManager) — dropped from Phase 20, post-cycle ingestion covers pattern needs
- `live_confirmed` flag on consolidation patterns — add if needed after observing real pattern quality
- Cloud API fallback for live endpoints — Ollama-only with fail-open is sufficient for paper trading
- R5 embedding-based retrieval — reassess at 500+ patterns (months away per growth projection)

</deferred>

---

*Phase: 20-production-deployment*
*Context gathered: 2026-03-15 (original), 2026-03-18 (updated)*
