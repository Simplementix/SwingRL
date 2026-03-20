# Phase 20: Production Deployment - Context

**Gathered:** 2026-03-15 (original), 2026-03-18 (19.1 impacts), 2026-03-20 (full reset with 19.1 alignment audit)
**Status:** Ready for re-planning

<domain>
## Phase Boundary

Deploy the homelab Docker stack (4 containers: swingrl, swingrl-memory, swingrl-ollama, swingrl-dashboard) with paper trading firing on schedule. Equity at **3:45 PM ET via Market-on-Close (MOC) orders**, crypto every **4H via limit orders**. Memory agents actively query consolidated patterns and influence trading decisions. Full verification that the entire trading pipeline works end-to-end for both paper and eventual live trading.

**Includes:**
- Memory live enrichment pipeline (5 /live/ endpoints controlling trading from the outside)
- Post-trade memory ingestion (6 memories/cycle) + cadence-based consolidation
- Price deviation gate (3% equity, 8% crypto)
- Startup validation, deploy script, smoke test
- Feature pipeline verification against training-time dimensions
- MacroWatcher for economic calendar events
- Performance tracking (SQLite table + status.json)
- PositionSizer removal (dead code after weight-based rebalancing)
- Config schema extensions for all live endpoint toggles and bounds
- Hardcoded values audit (automated grep + manual review)

**Removed from original scope (already done in Phase 19.1):**
- MemoryClient epoch_advice() and public properties (base_url, api_key) — DONE
- SQLAlchemy + APScheduler dependency additions — DONE
- DuckDB short-lived connections in FeaturePipeline — DONE
- Emergency crypto liquidation (_tier2_liquidate_crypto) — DONE via BinanceSimAdapter.emergency_sell()
- Shadow runner hardcoded fallback prices — DONE (skips on missing price)
- Reward component decomposition in env step() — DONE
- ExecutionPipeline public accessors — DONE
- Circuit breaker high-water mark — DONE
- Turbulence split into equity/crypto calculators — DONE
- Websocket fill ingestors (Alpaca TradingStream, Binance.US BinanceSocketManager) — dropped, post-cycle ingestion covers all data needs

</domain>

<decisions>
## Implementation Decisions

### Equity Execution Timing (CHANGED from 4:15 PM to 3:45 PM)
- **Schedule: 3:45 PM ET** — submit Market-on-Close (MOC) orders via Alpaca `time_in_force="cls"` before 3:50 PM cutoff
- Fills execute in the closing auction at the official close price — this is the exact price models were trained on
- Full order type support: bracket orders (SL/TP), fractional shares, market orders
- **Fallback chain:** MOC fails → limit order at 4:05 PM post-market at last trade price → market-on-open (`time_in_force="opg"`) next day
- Log each fallback step with reason for MOC failure
- Update APScheduler cron: equity cycle fires at `hour=15, minute=45` ET (was `hour=16, minute=15`)

### Crypto Execution Timing (UPDATED to limit orders)
- **Schedule: 5 minutes past 4H boundary** (00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC) — unchanged
- **Switch from market orders to limit orders**: set limit price = current best ask + 0.05% (buys) or best bid - 0.05% (sells)
- 30-second fill timeout: if not filled, cancel and retry at market
- Crypto is 24/7, no gap risk from market close

### Price Deviation Gate (UPDATED thresholds and actions)
- **Pre-execution check**: compare real-time price (`adapter.get_current_price()`) against observation price (last bar close from features)
- **Equity >3%**: switch to limit orders at prior_close ±1% instead of market/MOC. Alert Discord.
- **Crypto >8%**: skip the cycle entirely. Alert Discord.
- Config-driven: `environment.equity_price_deviation_pct=3.0` and `environment.crypto_price_deviation_pct=8.0`
- Log warning with both prices and deviation percentage when gate triggers

### Memory Live Enrichment
- **All endpoints enabled immediately** — seeded memories provide warm start context
- **No obs space expansion** — RL agents stay at 164 equity / 47 crypto dims, unaware of memory
- Memory agent controls the **execution pipeline from the outside**: cycle gate, blend weights, risk thresholds, position advice, trade veto, post-trade ingest
- **Integration point:** hooks inside `ExecutionPipeline.execute_cycle()` — both scheduler and manual `run_cycle.py` get memory automatically
- **Per-endpoint config toggles:** `memory_agent.live_endpoints.{cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto}` — all default to `true` when `memory_agent.enabled=true`
- **Skip /live/obs_enrichment** endpoint entirely — no obs space changes
- 5 live endpoints implemented in existing `services/memory/` FastAPI service (new `/live/` router)
- **Reuse query pipeline:** All live endpoints call `_build_context(request_type='live_trading', env_name=...)` for filtered patterns. Same R1-R3 filtering, composite scoring, and category-aware caps as training endpoints.
- **Model:** All live endpoints use `ollama_fast_model` from config (default qwen2.5:3b). Ollama only, no cloud API fallback. If Ollama down, fail-open to safe defaults.
- **Dedicated system prompts per endpoint** — each prompt tailored to that decision's domain. Apply same research-backed techniques as consolidation (LLM-PROMPT-RESEARCH.md): role definition, grounding rule, paired constraints, no CoT.
- **Schema enforcement via Ollama `format` param** — live endpoints define response schemas passed to Ollama. Client-side JSON validation always, fallback to safe defaults on parse failure.
- **HTTP client: stdlib urllib** (consistent with existing MemoryClient pattern)

### Memory Agent Decision Context (NEW)
- **Client passes market state + key feature signals** in the /live/ request body:
  - Core: current_regime, hmm_probs (4-dim), vix, current positions
  - Feature signals: turbulence score, ATR for relevant symbols, momentum signals
  - Macro: event proximity from MacroWatcher (e.g., "FOMC in 2 hours")
- **Memory service does NOT query swingrl's DuckDB** — all context passed by client
- **Consolidated patterns only** — query pipeline returns active consolidated patterns filtered by category + env + algo. No raw memories in live context (trust consolidation pipeline).
- **Pattern confidence threshold:** `min_confidence >= 0.5` for live endpoints (higher than training's 0.4 — live decisions are more time-sensitive)
- **No caching for live endpoints:** market state changes each cycle, cache would rarely hit

### Post-Trade Memory Ingestion
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
- **Individual POSTs via background thread** — 6 calls to existing /ingest endpoint. Trading pipeline returns immediately.
- **Sequential background task:** ingest all 6 memories → check consolidation cadence → consolidate if due.
- **Post-cycle timing:** ingestion happens after all orders submitted and fills confirmed

### Live Consolidation
- **Cadence:** Weekly for equity, every 3 days for crypto. Configurable via YAML: `memory_agent.live_consolidation.equity_interval_days=7`, `memory_agent.live_consolidation.crypto_interval_days=3`
- **Trigger:** Post-cycle check after memory ingestion. Check `last_consolidated_at` for this env. If cadence elapsed, fire consolidation.
- **Same env only:** Equity post-cycle only checks equity cadence. Crypto only checks crypto. Independent.
- **Uses existing consolidation pipeline:** Two-stage (per-env Stage 1, cross-env Stage 2 if both envs have new patterns). Same dedup, conflict resolution, pattern lifecycle.
- **Shared consolidation:** Training and live memories consolidated together. Live outcomes can confirm training patterns.

### Blend Weights Integration
- Memory **overrides with bounds**: each algo weight clamped to **±0.20** from Sharpe-softmax baseline
- Bound is **configurable via YAML**: `memory_agent.live_bounds.blend_weight_delta=0.20`
- Weights **reset each cycle** — each cycle queries `/live/blend_weights` fresh

### Circuit Breaker Overrides
- **Save + restore pattern**: `apply_overrides()` saves current values, sets new; `reset_overrides()` restores originals
- ExecutionPipeline calls `reset_overrides()` at cycle start, `apply_overrides()` after memory risk thresholds
- **Always log** original and override thresholds (audit trail)
- Live bounds **configurable via YAML** as maximum envelope — memory can tighten but not widen

### MemoryClient Extension (NOT a new class)
- **Extend existing MemoryClient** with 5 new methods for /live/ endpoints: `cycle_gate()`, `blend_weights()`, `risk_thresholds()`, `position_advice()`, `trade_veto()`
- Same fail-open urllib pattern, same base_url, same API key auth
- No separate LiveMemoryClient class — one client, all endpoints
- **epoch_advice() and public properties already exist** from Phase 19.1 — no changes needed there

### Startup Validation
- Standalone `scripts/startup_checks.py` called by `main.py` before scheduler init — also runnable via `docker exec`
- **Hard fail: exit(1)** on any validation failure — Docker restart policy retries
- Checks:
  - Broker credentials: ping Alpaca + Binance.US APIs
  - DB tables: verify SQLite + DuckDB have expected tables
  - Model files: check `models/active/{equity,crypto}/{ppo,a2c,sac}/model.zip` + `vec_normalize.pkl` — warn but don't block
  - Memory service reachable: check `swingrl-memory:8889/health` — warn if down (fail-open)
  - Memory seeding status: query memory.db row count, warn if empty
  - Bind-mount writability: touch temp file in each mount dir
  - **Feature pipeline dimensions**: verify observation vector dimensions (164 equity, 47 crypto) and column names match assembler constants. Verify macro data is non-NaN in recent window.
  - **Ollama reachable**: check `swingrl-ollama:11434/api/tags` — warn if down (fail-open)
- Structlog output only

### Deployment Method
- **Git pull + docker compose build** on homelab
- `scripts/deploy.sh`: git pull → docker compose build → docker compose down → docker compose up -d → wait for healthy → run `scripts/setup_ollama.sh`
- **Optional smoke test flag**: `deploy.sh --smoke-test` runs full smoke test after containers healthy
- .env is manual, never touched by deploy.sh — only checks it exists and has required keys
- No rollback mechanism — git revert + rebuild is sufficient

### End-to-End Smoke Test (EXPANDED)
- `scripts/smoke_test.py`: full pipeline verification
- Triggers one equity cycle + one crypto cycle immediately via `docker exec`
- **Full pipeline with memory**: exercises all memory touchpoints (cycle gate, risk thresholds, blend weights, position advice, trade veto, post-trade ingest)
- **Memory verification**: after cycle, queries /debug/memories and /debug/consolidations to verify row counts, source tags, and pattern categories match expectations
- **Impact verification**: checks DuckDB `memory_live_decisions` table for at least 1 non-default memory decision (structured decision logging)
- **Pass/fail checklist to stdout**: features assembled, model loaded, memory queried (5 endpoints), risk check passed, order submitted, fill logged, memories ingested, consolidation cadence checked — exit 0 all pass, exit 1 any fail
- Real paper mode: Alpaca paper orders (equity), simulated fills (crypto)
- **Operator manual review checklist** included in runbook (Phase 24): review memory decisions, pattern quality, ingestion content

### Memory Verification (Automated + Manual)
- **Automated (smoke test)**: verify row counts, source tags, pattern categories after cycle
- **Manual (operator)**: review memory DB via debug endpoints after first few cycles. Checklist in runbook.
- **Structured decision logging**: every memory decision logged to DuckDB `memory_live_decisions` table with: timestamp, env, endpoint, decision_json, rationale, latency_ms, clamped (bool), differed_from_default (bool)

### Performance Tracking
- **SQLite `cycle_performance` table**: per-cycle metrics — P&L, cumulative P&L, rolling Sharpe (30-day), current drawdown, fills count, memory decisions summary
- **status.json**: written after each cycle to `status/status.json` — last_cycle_time, positions, memory_health, scheduler_state, next_fire_times, rolling metrics snapshot
- Mounted via `./status:/app/status` bind mount (already in docker-compose.prod.yml)

### PositionSizer Removal (Dead Code Cleanup)
- Remove `PositionSizer` class — dead code after weight-based rebalancing replaced it in Phase 19.1
- Update shadow runner to use weight-based rebalancing matching the paper trading pipeline
- Verify no other callers reference PositionSizer

### Shadow Fill Realism
- **Add slippage estimates**: 5 bps for equity, 10 bps slippage + 10 bps commission for crypto (matching Binance.US 0.10% maker/taker)
- Shadow runner already skips signals on missing price (19.1 fix) — no fallback price issue
- Log when slippage is applied

### MacroWatcher
- Researcher finds best **free economic calendar API** for high-impact events (FOMC, CPI, NFP, PCE, GDP, PPI, JOLTS)
- **Fallback: hardcode known schedule** (FOMC meets 8x/year, CPI/NFP have predictable patterns)
- **Poll interval configurable** via YAML: `memory_agent.macro_watcher.poll_interval_seconds` (default 300)
- **Event proximity passed to memory endpoints** — enables macro-aware decisions

### Schedule Verification
- `scripts/show_schedule.py`: CLI that prints all jobs with next fire times
- Display times in **both ET and UTC**
- Reconstruct from config (doesn't require running scheduler)

### Docker Resource Limits
- **swingrl cpus: 10.0 → revert to 8.0** — temporarily bumped for Phase 19.1 training run
- All other container limits already applied in Phase 19.1 — no changes needed

### Config Schema Extensions
- `memory_agent.live_endpoints.*`: per-endpoint toggles — all default true
- `memory_agent.live_bounds.*`: blend_weight_delta, circuit_breaker_deltas
- `memory_agent.live_consolidation.equity_interval_days`: default 7
- `memory_agent.live_consolidation.crypto_interval_days`: default 3
- `memory_agent.macro_watcher.*`: poll_interval_seconds, API config
- `environment.equity_price_deviation_pct`: default 3.0
- `environment.crypto_price_deviation_pct`: default 8.0

### Dependencies
- **Full audit**: researcher does production Docker build to trace all ImportErrors
- **Verify `requests`**: used in 4 modules but possibly not declared as explicit dep
- **Remove `python-binance`**: no longer needed (fill ingestors dropped)
- SQLAlchemy and APScheduler already in deps (Phase 19.1)

### Healthcheck Improvements
- Add memory service reachability check (HTTP to swingrl-memory:8889/health)
- Add Ollama reachability check (HTTP to swingrl-ollama:11434/api/tags)
- Fail = UNHEALTHY with degraded flag (trading continues fail-open)

### Error Recovery
- Track **consecutive failures per env** in SQLite
- After **3 consecutive failures**: fire Discord warning (if configured) + structlog error
- No auto-retry within same cycle — wait for next scheduled fire

### Memory Seeding Workflow
- Seeding runs **after training, before first deploy** — operator runs `seed_memory_from_backtest.py` manually once
- **Seed + consolidate**: POST /consolidate at end of seeding
- startup_checks.py warns if memory is empty

### Logging and Observability
- **Structured log with rationale** for each memory call: endpoint, env, decision, rationale, latency_ms, clamped
- **INFO for decisions, DEBUG for details**
- **DuckDB table `memory_live_decisions`**: timestamp, env, endpoint, decision_json, rationale, latency_ms, clamped, differed_from_default
- **Memory latency alert**: avg response time > 3s for 3 consecutive calls → structlog warning + Discord alert

### Homelab CI Updates
- CI tests **both containers**: swingrl + services/memory/ (new live endpoints)
- Docker compose build validates Dockerfiles and compose config
- No full compose up in CI — verified during homelab deployment

### Testing Strategy
- **Mock Ollama** in all unit tests — fast, deterministic
- **Explicit fail-open tests**: connection refused → safe default, timeout → safe default, HTTP 500 → safe default, malformed JSON → safe default
- Test startup_checks.py with mocked broker APIs and DB connections
- Test post-cycle memory ingestion: verify 6 memories ingested with correct tags
- Test consolidation cadence check
- Test price deviation gate: verify limit order fallback (equity) and cycle skip (crypto)
- Test MOC order submission + fallback chain
- **Hardcoded values audit**: automated grep for IPs, URLs, dollar amounts, ticker symbols in src/

### Claude's Discretion
- Exact system prompts for each live endpoint (following LLM-PROMPT-RESEARCH.md techniques)
- MacroWatcher config field structure and free API selection
- Background ingest queue implementation details
- Exact status.json schema
- Exact response schemas for each live endpoint (Ollama format param)
- Which key features to include in market state context (turbulence, ATR, momentum specifics)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Memory Architecture
- `docs/memory_live_enrichment.md` — Live enrichment architecture, endpoint definitions, integration patterns. Exclude Section 5 (obs space expansion — deferred)
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/INGESTION-ENRICHMENT-PLAN.md` — Enriched ingestion fields, two-stage consolidation pipeline, event-driven model, pattern lifecycle
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/RETRIEVAL-EFFICIENCY-PLAN.md` — R1-R3 retrieval filtering, R4 caching, model configurability
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/PHASE-19.1-CHANGES.md` — Complete change log of 19.1 fixes (86 files, +5,251/-1,435 lines). MUST READ to understand what's already implemented.

### Prompt Engineering
- `docs/LLM-PROMPT-RESEARCH.md` — Research-backed prompt techniques for memory agents

### Specification Summary
- `docs/12-specification-summary.md` — Canonical reference card for all SwingRL specs

### Alpaca Order Types
- Alpaca documentation on MOC orders (time_in_force="cls"), extended hours restrictions, fractional shares

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 19.1 already built)
- `services/memory/memory_agents/consolidate.py`: Two-stage consolidation with cloud/Ollama dispatch, 14 categories, dedup/conflict resolution — extend with live consolidation cadence check
- `services/memory/memory_agents/query.py`: `_build_context()` with R1-R3 filtering, `request_type='live_trading'` already defined with live categories — reuse for all live endpoints
- `services/memory/db.py`: Full schema with consolidations, pattern_presentations, pattern_outcomes — add `memory_live_decisions` table and `last_consolidated_at` tracking
- `services/memory/routers/training.py`: Training advisor endpoints — pattern for live endpoint router
- `src/swingrl/memory/client.py`: MemoryClient with fail-open urllib, `epoch_advice()`, `base_url`/`api_key` properties — extend with 5 live endpoint methods
- `src/swingrl/execution/fill_processor.py`: FillProcessor._record_trade() — logs all fills to trades table
- `src/swingrl/execution/emergency.py`: Fully implemented crypto liquidation via `emergency_sell()` — no stub, works for paper trading
- `src/swingrl/features/pipeline.py`: FeaturePipeline with short-lived DuckDB connections (`duckdb_path` + `_db()` context manager) — no connection contention
- `src/swingrl/features/turbulence.py`: Split into EquityTurbulenceCalculator + CryptoTurbulenceCalculator — production-ready
- `src/swingrl/execution/risk/circuit_breaker.py`: Uses high-water mark for drawdown/daily loss — correct for memory override integration

### Established Patterns
- Event-driven consolidation: triggered by events, not timers
- Fail-open memory: all memory calls → try/except → safe defaults
- Config via YAML + env vars: memory service reads mounted swingrl.yaml
- Composite scoring: confidence (50%) + confirmation_count (30%) + recency (20%)
- Schema enforcement: Ollama `format` param for JSON schema
- Weight-based rebalancing: `process_actions()` replaces old Kelly/signal pipeline

### Integration Points
- `ExecutionPipeline.execute_cycle()` — where memory hooks integrate (5 live endpoint calls + post-cycle ingestion)
- `CircuitBreaker` — save/restore targets for memory overrides
- `EnsembleBlender.sharpe_softmax_weights()` — blend weight adjustment target
- `scripts/main.py` `build_app()` → add startup_checks, memory thread startup, update equity schedule to 3:45 PM
- `docker-compose.prod.yml` → verify bind mounts, revert CPU limit 10→8
- `pyproject.toml` → audit and fix missing deps
- `services/memory/app.py` → add live router
- `config/swingrl.yaml` → add live_endpoints, live_bounds, live_consolidation, macro_watcher, price_deviation config
- `src/swingrl/execution/adapters/alpaca_adapter.py` → add MOC order support (time_in_force="cls")

</code_context>

<specifics>
## Specific Ideas

- Memory agent controls trading from the OUTSIDE — RL agents are unaware, memory controls the pipeline
- "No cold start" — backtest seeding gives memory agent warm context from day one
- MOC orders at 3:45 PM give the exact training price with deepest liquidity — research-backed decision
- Crypto limit orders reduce slippage risk on occasionally thin Binance.US order books
- Price deviation gate with tiered response: equity gets limit order fallback, crypto skips entirely
- Both ET and UTC in schedule display — crypto jobs scheduled in UTC
- Structured decision logging (DuckDB table) proves memory agent impact beyond logs
- Full smoke test verifies not just trading but the entire memory pipeline end-to-end
- Post-cycle ingestion captures ALL memory decisions (not just trades) — rich consolidation input even on no-trade cycles

</specifics>

<deferred>
## Deferred Ideas

- **Training execution timing shift** (next-bar open price) — deferred to Phase 22. Mild look-ahead bias acceptable for swing trading. Fix when retraining infrastructure exists.
- **Shadow agent verification** (gates, promotion, rejection process) — deferred to Phase 22. Shadow promotion logic is Phase 22 scope.
- **BinanceAdapter for live crypto orders** — BinanceSimAdapter works for paper trading. Build real adapter for live trading (Phase 25+).
- Obs space expansion for memory dims — deferred to Phase 22+
- Memory dashboard tab in Streamlit — deferred, DuckDB table ready for Phase 25
- Cloud API fallback for live endpoints — Ollama-only with fail-open sufficient for paper trading
- R5 embedding-based retrieval — reassess at 500+ patterns
- `live_confirmed` flag on patterns — add after observing real pattern quality
- **Data pipeline overhaul** (Tiingo, raw prices) — Phase 23+ scope
- `bars_since_trade` vs `days_since_trade` semantic mismatch — document during live pipeline prep

</deferred>

---

*Phase: 20-production-deployment*
*Context gathered: 2026-03-15 (original), 2026-03-18 (updated), 2026-03-20 (full reset)*
