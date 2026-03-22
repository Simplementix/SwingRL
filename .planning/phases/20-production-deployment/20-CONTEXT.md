# Phase 20: Production Deployment - Context

**Gathered:** 2026-03-15 (original), 2026-03-18 (19.1 impacts), 2026-03-20 (full reset), 2026-03-22 (deep-dive: metrics, timing, schemas, macro)
**Status:** Ready for re-planning

<domain>
## Phase Boundary

Deploy the homelab Docker stack (3 containers: swingrl, swingrl-memory, swingrl-dashboard) with paper trading firing on schedule. Equity at **4:15 PM ET via fractional limit orders** (extended hours), crypto every **4H via limit orders**. Memory agents actively query consolidated patterns and influence trading decisions via OpenRouter + NVIDIA cloud LLM providers. Full verification that the entire trading pipeline works end-to-end for both paper and eventual live trading.

**Includes:**
- Automated OHLCV data ingest inside execute_cycle() (Alpaca Snapshot API for equity, Binance API for crypto)
- Memory live enrichment pipeline (5 /live/ endpoints controlling trading from the outside)
- Post-trade memory ingestion (6 memories/cycle) + cadence-based consolidation
- Price deviation gate (3% equity, 8% crypto) with two-price-check flow
- Comprehensive performance tracking (SQLite tables, status.json, WFE comparison, regime-conditioned analysis)
- Startup validation, deploy script with optional smoke test, full E2E smoke test
- Feature pipeline dimension verification against training-time constants
- MacroWatcher using FRED API release dates (7 events: FOMC, CPI, NFP, GDP, PPI, PCE, JOLTS)
- PositionSizer removal + shadow runner rewrite to weight-based rebalancing
- Shadow fill realism (slippage estimates)
- Config schema extensions for live endpoint toggles, bounds, consolidation, macro, price deviation
- Hardcoded values audit (automated grep + manual review)
- Progressive drawdown alerts aligned to circuit breaker thresholds

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
- Websocket fill ingestors — dropped, post-cycle ingestion covers all data needs
- Ollama container — REMOVED (replaced by OpenRouter + NVIDIA cloud providers)

</domain>

<decisions>
## Implementation Decisions

### Equity Execution Timing (CHANGED: 4:15 PM fractional limit orders)
- **Schedule: 4:15 PM ET** (keep current timing, market closed at 4:00 PM)
- **Auto-ingest OHLCV inside execute_cycle()**: first step downloads today's completed bar via Alpaca **Snapshot API** (`GET /v2/stocks/snapshots` returns `dailyBar` for all 8 ETFs in one call)
- Compute features from today's completed bar — matches training semantics exactly (see bar-t, trade at ~bar-t close)
- **No retraining needed** — today's features with post-market execution closely matches training
- Submit **fractional limit orders** with `extended_hours=true`, `time_in_force="day"`
- Limit price = current_price ±0.1% (see two-price-check flow below)
- **Fill timeout**: 30s → if unfilled, widen tolerance to ±0.3% and retry → 60s total, skip symbol
- **After fill**: submit GTC stop order for next regular session (brackets don't work in extended hours)
- **Unfilled orders auto-cancel at 8:00 PM ET** (Alpaca behavior)
- Update APScheduler cron: equity cycle stays at `hour=16, minute=15` ET

### Two-Price-Check Flow (NEW)
- **Price 1 (features)**: Today's close price from Snapshot API `dailyBar` → used for feature computation
- **Price 2 (orders)**: Real-time price via `adapter.get_current_price()` right before order submission
- **Deviation check**: `abs(price2 - price1) / price1`
  - If ≤ 3%: use `price2` (current price) for limit order — most accurate
  - If > 3%: price deviation gate triggers → switch to limit at `price1 ±1%`, alert Discord
- This catches post-market moves between feature computation and order submission

### Crypto Execution Timing (limit orders, auto-ingest)
- **Schedule: 5 minutes past 4H boundary** (00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC) — unchanged
- **Auto-ingest inside execute_cycle()**: first step downloads just-closed 4H bar from Binance API
- **Limit orders**: limit price = current best ask + 0.05% (buys) or best bid - 0.05% (sells)
- 30-second fill timeout: if not filled, cancel and retry at market
- **Price deviation gate**: if current price > 8% from 4H close → skip cycle entirely, alert Discord

### Cloud LLM Provider (CHANGED: OpenRouter replaces Ollama)
- **Ollama container removed** from docker-compose.prod.yml (commit e15643c)
- **3 containers total**: swingrl, swingrl-memory, swingrl-dashboard
- **Primary**: OpenRouter nemotron-120b (free tier, ~0.4s/call)
- **Backup**: NVIDIA kimi-k2.5 (~140s/call)
- Same dual-provider `_call_lm()` pattern as consolidation
- All live endpoints, consolidation, and training queries use this stack
- **Fail-open**: if both providers fail → safe defaults (no trading blocked by LLM unavailability)

### Memory Live Enrichment
- **All endpoints enabled immediately** — seeded memories provide warm start context
- **No obs space expansion** — RL agents stay at 164 equity / 47 crypto dims, unaware of memory
- Memory agent controls the **execution pipeline from the outside**: cycle gate, blend weights, risk thresholds, position advice, trade veto, post-trade ingest
- **Integration point:** hooks inside `ExecutionPipeline.execute_cycle()`
- **Per-endpoint config toggles:** `memory_agent.live_endpoints.{cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto}` — all default to `true` when `memory_agent.enabled=true`
- **Skip /live/obs_enrichment** endpoint entirely
- 5 live endpoints in `services/memory/` FastAPI service (new `/live/` router)
- **Reuse query pipeline:** All endpoints call `_build_context(request_type='live_trading', env_name=...)` for filtered patterns
- **Dedicated system prompts per endpoint** — following LLM-PROMPT-RESEARCH.md techniques
- **Schema enforcement via OpenRouter `response_format` param** with client-side JSON validation, fallback to safe defaults

### Live Endpoint Response Schemas (DECIDED)

**All responses include:** `confidence: float (0.0-1.0)`, `rationale: str`, `latency_ms: float` (server-side)
**Confidence < 0.3 → use safe default instead of LLM response**

| Endpoint | Response Fields | Safe Default | Clamp |
|----------|----------------|--------------|-------|
| `cycle_gate` | `approved: bool` | `approved=True` | confidence < 0.3 → approve |
| `blend_weights` | `deltas: {PPO: float, A2C: float, SAC: float}` | `deltas={}` (no change) | ±0.20 per algo, re-normalize to sum 1.0 |
| `risk_thresholds` | `max_dd: float\|null`, `daily_limit: float\|null` | `null` (no tightening) | tighten-only, never widen beyond config |
| `position_advice` | `kelly_scale_factor: float`, `hold_bias: bool`, `max_simultaneous_positions: int\|null` | `scale=1.0, hold=false, max=null` | scale bounded 0.5-1.5 |
| `trade_veto` | `vetoed: bool` | `vetoed=False` | confidence < 0.3 → don't veto |

**Blend weights format**: LLM outputs **deltas** (e.g., `{PPO: +0.05, A2C: -0.10, SAC: +0.05}`). Client adds to Sharpe-softmax baseline, clamps ±0.20, re-normalizes.

### Memory Agent Decision Context (DECIDED)
- **Mirror training format** — pass same fields as walk-forward fold results so consolidated patterns apply directly
- **Same context for all 5 endpoints** — build once, each endpoint's system prompt guides focus
- **Key fields in query string format:**
  - Regime: `p_bull`, `p_bear`, `vix_zscore`, `turbulence`, `yield_spread`
  - Portfolio: `portfolio_sharpe`, `portfolio_mdd`, `position_exposure`, `daily_return_pct`, `drawdown_pct`
  - Per-symbol: `weight`, `unrealized_pnl_pct`, `rsi_14`, `atr_14_pct`, `bars_since_trade`
  - Macro: `nearest_macro_event`, `hours_until_event` (from MacroWatcher)
- **Memory service does NOT query swingrl's DuckDB** — all context passed by client
- **Consolidated patterns only** — `min_confidence >= 0.5` for live endpoints (higher than training's 0.4)
- **No caching** — market state changes each cycle

### Post-Trade Memory Ingestion
- **6 memories per cycle** (5 per-endpoint + 1 cycle summary)
- **Per-endpoint memories** tagged `live:{env}:{endpoint}`:
  - Endpoint decision + rationale + market state snapshot + whether clamped
- **Cycle summary memory** tagged `live:{env}:cycle_summary`:
  - All endpoint decisions, orders submitted, realized P&L, which decisions clamped
  - NO unrealized P&L or portfolio value (noise)
- **Individual POSTs via background thread** — trading pipeline returns immediately
- **Sequential background task:** ingest 6 → check consolidation cadence → consolidate if due

### Live Consolidation
- **Cadence:** Weekly equity (`equity_interval_days=7`), every 3 days crypto (`crypto_interval_days=3`)
- **Trigger:** Post-cycle check after memory ingestion
- **Uses existing two-stage consolidation pipeline** (per-env Stage 1, cross-env Stage 2)
- **Shared consolidation:** Training and live memories consolidated together

### Blend Weights Integration
- Memory overrides with bounds: ±0.20 from Sharpe-softmax baseline (`blend_weight_delta=0.20`)
- Weights reset each cycle — fresh `/live/blend_weights` query per cycle

### Circuit Breaker Overrides
- Save/restore pattern for memory risk threshold overrides
- Memory can tighten but never widen beyond config values
- Always log original and override thresholds

### MemoryClient Extension
- Extend existing MemoryClient with 5 new methods: `cycle_gate()`, `blend_weights()`, `risk_thresholds()`, `position_advice()`, `trade_veto()`
- Same fail-open urllib pattern, same base_url, same API key auth

### Performance Tracking (DECIDED — comprehensive)

**SQLite `cycle_performance` table (per-cycle):**
- Portfolio: `portfolio_value`, `daily_pnl`, `daily_return_pct`, `cumulative_pnl`
- Rolling metrics: `rolling_sharpe_30d`, `rolling_sharpe_60d`, `rolling_sortino_30d`, `rolling_sortino_60d`
- Risk: `current_drawdown_pct`, `high_water_mark`
- Trade: `fills_count`, `symbols_traded`, `win_rate_trailing_30d`, `avg_hold_duration`
- Context: `exposure_ratio`, `cash_ratio`, `turbulence_at_trade`, `regime_at_trade`
- Memory: `memory_decisions_count`, `memory_decisions_differed_from_default`

**SQLite `symbol_performance` table (per-symbol per-cycle):**
- `symbol`, `env`, `action` (buy/sell/hold), `pnl`, `return_pct`

**SQLite `deployment_baseline` table (frozen at deployment):**
- WF OOS metrics per env: Sharpe, Sortino, MDD, win_rate, profit_factor
- Captured when models deployed to models/active/

**Performance comparison (full framework):**
- **Walk-Forward Efficiency (WFE)**: `live_sharpe / backtest_sharpe` — primary degradation metric
- **Dual benchmark**: equal-weight buy-and-hold of traded assets (8 ETFs equity, 50/50 BTC/ETH crypto) AND haircut-adjusted backtest (50% of WF OOS Sharpe)
- **Regime-conditioned comparison**: match current HMM regime to backtest folds, compare only against similar-regime performance
- Alert if live Sharpe < 50% of backtest Sharpe on 63-day rolling window

**Memory decision outcome tagging:**
- After each cycle, tag memory decisions with outcome (profit/loss)
- Stored in `memory_live_decisions` table with `differed_from_default` and `outcome_pnl`

**status.json (comprehensive snapshot, updated after every cycle):**
- `last_cycle_time`, `portfolio_value` (per-env), `positions` (symbol, qty, unrealized_pnl)
- `rolling_sharpe_30d`, `current_drawdown_pct`, `next_fire_times`
- `memory_health` (reachable, latency), `last_error`

### Performance Alerts (DECIDED — progressive tiers aligned to circuit breakers)

**Drawdown alerts (equity, CB at 10%):**
- Watch: 5% DD (50% of CB)
- Warning: 7% DD (70% of CB)
- Critical: 9% DD (90% of CB)
- Halt: 10% DD (CB triggers)

**Drawdown alerts (crypto, CB at 12%):**
- Watch: 6% DD
- Warning: 9% DD
- Critical: 11% DD
- Halt: 12% DD (CB triggers)

**Daily loss alerts:**
- Watch: 50% of daily_loss_limit_pct
- Warning: 75% of daily_loss_limit_pct

**Sharpe alerts (conservative tiers):**
- Watch: 63d rolling Sharpe < 0.5
- Warning: 63d rolling Sharpe < 0.3
- Critical: 63d rolling Sharpe < 0.0

**Consecutive losing days:** warn at 5, critical at 7

### Reporting Cadence (DECIDED)
- **Daily**: Discord embed (exists) — portfolio snapshot, P&L, trade count
- **Weekly**: Automated aggregate — Sharpe/Sortino/drawdown/win rate, best-worst trades, benchmark comparison
- **Monthly**: Full tearsheet — backtest-to-live ratio, regime analysis, feature drift, model degradation assessment

### Startup Validation
- Standalone `scripts/startup_checks.py` called by `main.py` before scheduler init
- **Hard fail: exit(1)** on any validation failure
- Checks: broker credentials, DB tables, model files (warn), memory service (warn, fail-open), memory seeding status (warn), bind-mount writability, feature pipeline dimensions (164 equity, 47 crypto), macro data non-NaN
- Structlog output only

### Deployment Method
- `scripts/deploy.sh`: git pull → docker compose build → down → up -d → wait healthy
- **Optional smoke test flag**: `deploy.sh --smoke-test`
- .env is manual, never touched by deploy.sh

### End-to-End Smoke Test (comprehensive)
- Full pipeline: equity + crypto cycles with all memory touchpoints
- **Memory verification**: query /debug/memories and /debug/consolidations for row counts and tags
- **Impact verification**: check memory_live_decisions for at least 1 non-default decision
- **Operator manual review checklist** for runbook (Phase 24)

### PositionSizer Removal + Shadow Runner Rewrite (DECIDED)
- **Remove PositionSizer** class entirely (dead code)
- **Rewrite shadow runner** to use `process_actions()` + weight-based rebalancing matching the live pipeline
- **Add slippage**: 5 bps equity, 10 bps slippage + 10 bps commission crypto
- Shadow trades should match what live would actually do

### MacroWatcher (DECIDED)
- **Data source: FRED API** `fred/release/dates` endpoint (already have fredapi + API key)
- **7 release IDs**: FOMC (101), CPI (10), NFP (50), GDP (53), PPI (46), PCE (54), JOLTS (192)
- **Cache weekly** — release schedules rarely change
- **Integration**: check at cycle start inside execute_cycle()
- **Context format**: `nearest_macro_event: "CPI"`, `hours_until_event: 4.5` (null if none within 48h)
- **No fallback needed** — FRED is government-backed, free, reliable

### Schedule Verification
- `scripts/show_schedule.py`: prints all jobs with next fire times in both ET and UTC

### Docker Resource Limits
- **swingrl cpus: 10.0 → revert to 8.0** (was temporarily bumped for training)
- **swingrl-ollama removed** — no longer needed
- Remaining: swingrl (no mem_limit, 8 CPU), swingrl-memory (1GB, 1 CPU), swingrl-dashboard (512MB, 0.5 CPU)

### Config Schema Extensions
- `memory_agent.live_endpoints.*`: per-endpoint toggles — all default true
- `memory_agent.live_bounds.blend_weight_delta`: default 0.20
- `memory_agent.live_bounds.kelly_scale_min/max`: 0.5 / 1.5
- `memory_agent.live_consolidation.equity_interval_days`: default 7
- `memory_agent.live_consolidation.crypto_interval_days`: default 3
- `memory_agent.macro_watcher.release_ids`: dict of event→FRED ID
- `memory_agent.macro_watcher.cache_days`: default 7
- `environment.equity_price_deviation_pct`: default 3.0
- `environment.crypto_price_deviation_pct`: default 8.0

### Dependencies
- **Full audit**: researcher does production Docker build to trace all ImportErrors
- **Verify `requests`**: used in modules but possibly not declared
- **Remove `python-binance`**: no longer needed
- **Remove Ollama-related config** from docker-compose and schema

### Error Recovery
- Track consecutive failures per env in SQLite
- After 3 consecutive failures: Discord warning + structlog error
- No auto-retry within same cycle

### Logging and Observability
- Structured log with rationale for each memory call
- INFO for decisions, DEBUG for details
- DuckDB `memory_live_decisions` table with outcome tagging
- Memory latency alert: avg > 3s for 3 consecutive calls → warning + Discord

### Testing Strategy
- **Mock cloud LLM** in all unit tests (mock httpx responses)
- **Explicit fail-open tests**: connection refused, timeout, HTTP 500, malformed JSON → safe defaults
- Test price deviation gate: verify limit order fallback (equity) and cycle skip (crypto)
- Test two-price-check flow
- Test OHLCV auto-ingest inside execute_cycle
- **Hardcoded values audit**: automated grep for IPs, URLs, dollar amounts, ticker symbols

### Claude's Discretion
- Exact system prompts for each live endpoint (following LLM-PROMPT-RESEARCH.md)
- Background ingest queue implementation details
- Exact status.json schema field names
- FRED API query implementation details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Memory Architecture
- `docs/memory_live_enrichment.md` — Live enrichment architecture, endpoint definitions. Exclude Section 5 (obs space expansion — deferred)
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/INGESTION-ENRICHMENT-PLAN.md` — Enriched ingestion, two-stage consolidation, pattern lifecycle
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/RETRIEVAL-EFFICIENCY-PLAN.md` — R1-R3 retrieval filtering, R4 caching
- `.planning/phases/19.1-memory-agent-infrastructure-and-training/PHASE-19.1-CHANGES.md` — Complete 19.1 change log. MUST READ.

### Prompt Engineering
- `docs/LLM-PROMPT-RESEARCH.md` — Research-backed prompt techniques

### Cloud LLM Migration
- Commit `e15643c` — OpenRouter migration. Removed Ollama container. Dual-provider pattern in consolidate.py and query.py.

### Alpaca API
- Fractional limit orders work in extended hours with `extended_hours=true`, `time_in_force="day"`
- MOC orders (`time_in_force="cls"`) do NOT support fractional/notional orders
- Market orders NOT allowed in extended hours — only limit orders
- Snapshot API returns `dailyBar` (today's completed/in-progress bar)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets (Phase 19.1 already built)
- `services/memory/memory_agents/consolidate.py`: Two-stage consolidation with OpenRouter/NVIDIA dual-provider `_call_lm()` — reuse pattern for live endpoint agent
- `services/memory/memory_agents/query.py`: `_build_context()` with R1-R3 filtering, `request_type='live_trading'` already defined — reuse for all live endpoints
- `services/memory/db.py`: Full schema — add `memory_live_decisions` table and `last_consolidated_at` tracking
- `services/memory/routers/training.py`: Training advisor endpoints — pattern for live router
- `src/swingrl/memory/client.py`: MemoryClient with fail-open — extend with 5 live methods
- `src/swingrl/execution/fill_processor.py`: FillProcessor._record_trade() — logs all fills
- `src/swingrl/agents/metrics.py`: Sharpe, Sortino, Calmar, max_drawdown, win_rate, profit_factor — reuse for performance tracking
- `src/swingrl/execution/risk/position_tracker.py`: Portfolio value, HWM, daily P&L, exposure — data source for performance metrics

### Established Patterns
- Dual-provider LLM: OpenRouter primary, NVIDIA backup, fail-open
- Event-driven consolidation
- Config via YAML + env vars
- Composite scoring: confidence (50%) + confirmation_count (30%) + recency (20%)
- Weight-based rebalancing via `process_actions()`

### Integration Points
- `ExecutionPipeline.execute_cycle()` — add OHLCV ingest as first step, memory hooks, two-price-check, performance recording
- `CircuitBreaker` — save/restore for memory overrides
- `EnsembleBlender.sharpe_softmax_weights()` — blend weight adjustment target
- `scripts/main.py` — add startup_checks, memory thread startup
- `docker-compose.prod.yml` — remove Ollama references, revert CPU limit 10→8
- `pyproject.toml` — audit deps
- `services/memory/app.py` — add live router
- `config/swingrl.yaml` — add live_endpoints, live_bounds, live_consolidation, macro_watcher, price_deviation config
- `src/swingrl/execution/adapters/alpaca_adapter.py` — add fractional limit order support with `extended_hours=true`

</code_context>

<specifics>
## Specific Ideas

- Memory agent controls trading from the OUTSIDE — RL agents unaware
- "No cold start" — backtest seeding gives warm context from day one
- Two-price-check catches post-market drift between feature computation and order submission
- FRED API already integrated via fredapi — zero new dependencies for MacroWatcher
- Mirror training context format for live queries — consolidated patterns apply directly
- WFE (live/backtest Sharpe ratio) is the primary degradation metric with 50% haircut baseline
- Progressive DD alerts at 50%/70%/90% of circuit breaker thresholds give operator time to intervene
- Structured decision logging (DuckDB) with outcome tagging proves/disproves memory agent value
- Shadow runner rewrite to weight-based rebalancing ensures shadow evaluation matches live pipeline
- OpenRouter nemotron-120b is fast enough (~0.4s) for live trading decisions with 3s timeout

</specifics>

<deferred>
## Deferred Ideas

- **Training execution timing shift** (next-bar open price) — deferred to Phase 22. Fix when retraining.
- **Shadow agent verification** (gates, promotion, rejection) — deferred to Phase 22.
- **BinanceAdapter for live crypto orders** — BinanceSimAdapter works for paper trading. Phase 25+.
- Obs space expansion for memory dims — Phase 22+
- Memory dashboard tab in Streamlit — Phase 25
- Cloud API fallback differentiation for live endpoints — single stack sufficient
- R5 embedding-based retrieval — reassess at 500+ patterns
- `live_confirmed` flag on patterns — add after observing real pattern quality
- `bars_since_trade` vs `days_since_trade` semantic mismatch — document during live pipeline prep

</deferred>

---

*Phase: 20-production-deployment*
*Context gathered: 2026-03-15 (original), 2026-03-18 (updated), 2026-03-20 (full reset), 2026-03-22 (deep-dive)*
