# Phase 20: Production Deployment - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Deploy the homelab Docker stack with paper trading firing on schedule (equity 4:15 PM ET, crypto every 4H). Both containers healthy, startup validation passes, and end-to-end paper trade cycle completes. Additionally, implement the memory live enrichment pipeline from `memory_live_enrichment.md` — memory agent controls the trading pipeline (cycle gate, blend weights, risk thresholds, position advice, trade veto) without modifying RL agent obs space. Fix missing production dependencies. Add Alpaca + Binance.US websocket fill ingestion threads and a MacroWatcher for economic calendar events.

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
- All live endpoints use **qwen2.5:3b** (fast model, consistent with QueryAgent pattern — latency-sensitive, frequent calls)
- **Dedicated system prompts per endpoint** — each prompt tailored to that decision's domain (trade veto knows about position risk, cycle gate knows about market hours)
- **Post-trade ingestion via background thread queue** — zero latency impact on trading cycle, swallow all exceptions
- **HTTP client: stdlib urllib** (consistent with existing MemoryClient in `src/swingrl/memory/client.py`, no new dependency)

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

### Alpaca FillIngestor
- Full websocket thread using **Alpaca's trade_updates** stream
- Mock TradingStream in unit tests — test FillMemory construction, client_order_id parsing, ingest calls
- Daemon thread (exits with main process)

### Binance FillIngestor
- Build for live, **stub for paper** — connects in live mode, no-op in paper mode
- Use **python-binance** library (BinanceSocketManager, `tld='us'`)
- **Auto-reconnect with exponential backoff** (1s→2s→4s→8s→16s, max 60s)
- Starts when `trading_mode != 'paper'`, **regardless of memory_agent.enabled** — fills are logged for all modes
- Mock stream in unit tests

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
- **Full pipeline with memory**: exercises all memory touchpoints (cycle gate, HMM watcher, risk thresholds, blend weights, position advice, trade veto, post-trade ingest)
- **Pass/fail checklist to stdout**: features assembled, model loaded, risk check passed, order submitted, fill logged, memory ingested — exit 0 all pass, exit 1 any fail

### Docker Resource Limits
- swingrl-ollama: **24GB / 8 CPU** (unchanged)
- swingrl-memory: **1GB / 1 CPU** (up from 512MB/0.5)
- swingrl: **16GB / 8 CPU** (up from 2.5GB/1 — sized for concurrent live trading + retraining in Phase 22)
- swingrl-dashboard: **512MB / 0.5 CPU** (unchanged)
- Stack total: ~41.5GB / 17.5 CPU of 64GB / 20T — ~22GB headroom

### Config Schema Updates
- `memory_agent.live_endpoints.*`: per-endpoint toggles (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto) — all default true
- `memory_agent.live_bounds.*`: blend_weight_delta, circuit_breaker_deltas — configurable maximum envelope, memory can tighten only
- `memory_agent.macro_watcher.*`: poll_interval_seconds, API config — Claude's discretion on structure

### Dependencies
- **SQLAlchemy>=2.0,<3**: missing — needed by APScheduler SQLAlchemyJobStore
- **requests**: used in 4 modules but not declared as explicit dep — add
- **python-binance**: new dep for Binance.US websocket fill ingestion
- **Full audit**: researcher does production Docker build to trace all ImportErrors

### Testing Strategy
- **Mock Ollama** in all unit tests — fast, deterministic, no Docker needed
- **Explicit fail-open tests**: each client module tests connection refused → safe default, timeout → safe default, HTTP 500 → safe default, malformed JSON → safe default
- **Mock both websocket streams** (Alpaca TradingStream, python-binance BinanceSocketManager) in unit tests
- Test `startup_checks.py` with mocked broker APIs and DB connections
- Skip `deploy.sh` tests — verified during homelab deployment
- Integration tested during homelab deployment checkpoint

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
- **DuckDB table `memory_live_decisions`**: timestamp, env, endpoint, decision_json, rationale, latency_ms, clamped — enables post-hoc analysis of memory agent effectiveness
- **Memory latency alert**: if avg response time > 3s for 3 consecutive calls → structlog warning + Discord alert (if webhook configured)
- Dashboard display of memory decisions **deferred to future phase**

### Homelab CI Updates
- CI tests **both containers**: swingrl (existing) + services/memory/ (new live endpoints)
- **Docker compose build only** (no start): `docker compose -f docker-compose.prod.yml build` validates Dockerfiles and compose config
- No full compose up/health check in CI — verified during homelab deployment

### Claude's Discretion
- Exact system prompts for each live endpoint (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto)
- MacroWatcher config field structure
- Whether Alpaca streaming needs separate auth vs same keys
- Background ingest queue implementation details (queue depth, batch size)
- Exact status.json schema
- How to handle edge cases in client_order_id parsing for signal confidence threading

</decisions>

<specifics>
## Specific Ideas

- Memory agent controls trading from the OUTSIDE — RL agents are unaware of memory, memory controls the pipeline
- "No cold start" — backtest seeding gives memory agent warm context from day one
- Resource limits sized for concurrent live trading + Phase 22 retraining (8GB swingrl container)
- deploy.sh is one-command deployment but runbook also covers manual steps
- Per-endpoint toggles give granular control even though all start enabled
- Both ET and UTC in schedule display — crypto jobs are scheduled in UTC
- Latency alerting via both structlog + Discord (graceful if Discord not configured yet)
- Reference spec: `memory_live_enrichment.md` (architecture, code examples, endpoint definitions) — exclude Section 5 (obs space expansion)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/main.py`: APScheduler entrypoint with 12 jobs, signal handling, stop polling — add memory thread startup
- `scripts/run_cycle.py`: Manual CLI for trading cycles — benefits from ExecutionPipeline memory integration automatically
- `src/swingrl/execution/pipeline.py`: ExecutionPipeline.execute_cycle() — primary integration point for memory hooks
- `src/swingrl/execution/risk/circuit_breaker.py`: CircuitBreaker with _max_dd, _daily_limit — add apply_overrides()/reset_overrides()
- `src/swingrl/training/ensemble.py`: EnsembleBlender.sharpe_softmax_weights — blend weight adjustment target
- `src/swingrl/memory/client.py`: Existing MemoryClient using stdlib urllib — pattern for live enrichment clients
- `services/memory/memory_agents/query.py`: QueryAgent with Ollama tool calls — pattern for live endpoint agents
- `scripts/healthcheck.py`: Docker health probe checking SQLite + DuckDB — extend with memory/Ollama checks
- `docker-compose.prod.yml`: 4-service stack already defined (ollama, memory, swingrl, dashboard)
- `Dockerfile`: Multi-stage (ci + production) with non-root trader user
- `.env.example`: 5 required keys defined

### Established Patterns
- Docker exec for operator commands: `docker exec swingrl python scripts/...`
- Config via `load_config()` → `SwingRLConfig` with env var overrides
- structlog with keyword args for all logging
- Typed exceptions: `SwingRLError` hierarchy
- Memory calls fail-open with try/except → safe defaults
- APScheduler with SQLAlchemy jobstore + ThreadPoolExecutor
- Pydantic v2 nested config models

### Integration Points
- `ExecutionPipeline.execute_cycle()` — where memory hooks integrate
- `CircuitBreaker.__init__()` → `_max_dd`, `_daily_limit` — save/restore targets
- `EnsembleBlender.sharpe_softmax_weights()` — blend weight adjustment target
- `scripts/main.py` `build_app()` → add startup_checks, memory threads
- `docker-compose.prod.yml` → update resource limits, verify bind mounts
- `pyproject.toml` → add missing deps (SQLAlchemy, requests, python-binance)
- `services/memory/app.py` → add live router
- `config/swingrl.yaml` → add live_endpoints, live_bounds, macro_watcher config

</code_context>

<deferred>
## Deferred Ideas

- Obs space expansion (memory dims 156→164, 45→53) — requires retraining, defer to Phase 22 or later
- Memory dashboard tab in Streamlit — deferred from Phase 19 context, DuckDB table ready for it
- Progressive ramp-up of live endpoints — decided against (all enabled immediately with seeded memories)
- Inbox file watcher for memory ingestion — deferred from Phase 19

</deferred>

---

*Phase: 20-production-deployment*
*Context gathered: 2026-03-15*
