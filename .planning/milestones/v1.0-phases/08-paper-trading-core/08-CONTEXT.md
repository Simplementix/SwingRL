# Phase 8: Paper Trading Core - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire up the full execution pipeline: ensemble inference signals flow through 5-stage middleware (Signal Interpreter → Position Sizer → Order Validator → Exchange Adapter → Fill Processor), with a two-tier risk management veto layer and circuit breakers, to Alpaca paper API (equity) and simulated Binance.US fills (crypto). Deploy to homelab Docker in paper mode with production-ready multi-stage image. Covers PAPER-01, PAPER-02, PAPER-03, PAPER-04, PAPER-05, PAPER-06, PAPER-07, PAPER-08, PAPER-09, PAPER-10, PAPER-11, PAPER-18, PAPER-19, PAPER-20.

Phase 9 handles APScheduler automation, Discord alerting (trade-level), stuck agent detection, Streamlit dashboard, and Healthchecks.io.

</domain>

<decisions>
## Implementation Decisions

### Crypto paper fill simulation (PAPER-02)
- **Mid-price with slippage model**: fetch bid/ask from Binance.US REST API order book, take midpoint, add configurable slippage (0.01-0.05%) — matches Doc 04 §3 cost model (0.22% round-trip)
- **Instant fill**: no simulated exchange latency — swing trading on 4H bars makes latency simulation pointless
- **Virtual balance**: track $47 starting balance locally in SQLite (portfolio_snapshots + positions tables). No Binance.US account needed for paper mode. Switch to real balance only when trading_mode=live
- **Stop-loss/take-profit monitoring**: poll-based — check BTC/ETH price against open stop/TP levels every 60s via REST. Trigger simulated fill when price crosses threshold. Matches Binance.US's two-step OCO pattern from PAPER-10

### Execution retry & error handling
- **Basic retries in Phase 8**: 3 attempts with exponential backoff for order submission. Phase 10's HARD-02 adds comprehensive error handling (data feed gaps, circuit-aware recovery, structured error categories)
- **Fail action**: when all retries fail, skip the trade — log failure to risk_decisions table, send Discord critical alert, skip trade for this cycle. Next cycle re-evaluates from scratch. No trade queue
- **Independent pipelines**: each environment runs its own middleware pipeline independently. Separate risk budgets, separate circuit breakers, separate broker connections. Only shared check is the Tier 2 global portfolio aggregator (-15% DD, -3% daily combined). Matches Doc 04 two-tier architecture

### Position reconciliation
- **Startup + on-demand**: auto-reconcile equity positions against Alpaca on container start (before first trade cycle). Also available via CLI (`scripts/reconcile.py`). Crypto uses virtual balance so no broker-side reconciliation needed in paper mode
- **Auto-correct DB + alert on mismatch**: trust broker as source of truth. Update DB positions to match broker state. Insert adjustment trade rows. Send Discord warning alert. Automates the procedure from Doc 15 §14

### Docker production image (PAPER-19)
- **Multi-stage build**: production stage excludes dev deps (pytest, ruff, mypy, bandit). CI stage keeps dev deps. Phase 1 explicitly deferred this to Phase 8+
- **Docker HEALTHCHECK**: lightweight Python probe that verifies main process alive + DB connectivity. Docker marks container unhealthy if probe fails 3x, triggering restart policy
- **Single process entrypoint**: one Python process with APScheduler (Phase 9) managing scheduled jobs. Stop-price polling loop runs as background daemon thread. Docker-native — one container = one process. Sufficient for swing trading volume (~12 crypto cycles/day + 1 equity/day)

### Manual cycle trigger
- **scripts/run_cycle.py**: `--env equity|crypto [--dry-run]`. Normal mode: fetches observation, runs ensemble predict, sends through middleware, executes. Dry-run: does everything except submit to broker (logs what it WOULD do). Essential for development testing and post-deployment smoke tests. Follows existing scripts/ pattern (train.py, backtest.py)

### Logging
- **Same logging in all modes, config-driven level**: use identical structlog calls in paper and live mode. Set log level to DEBUG in paper mode config (default) for full detail (observations, raw actions, blend weights, risk checks). In live mode, change level to INFO via config. No code difference between modes — Phase 2's structlog pattern handles this

### Claude's Discretion
- 5-stage middleware internal architecture (class design, interfaces between stages)
- Exchange adapter abstraction pattern (base class vs protocol)
- Position sizing calculator implementation (Kelly formula + ATR stop calculation)
- Circuit breaker state machine internals (cooldown tracking, ramp-up progression)
- Turbulence crash protection integration with Phase 5's TurbulenceCalculator
- DB seeding script implementation details (scripts/seed_production.py)
- Order ID generation strategy for simulated crypto fills
- Reconciliation diff algorithm
- Docker healthcheck probe implementation
- Production docker-compose.yml structure

</decisions>

<specifics>
## Specific Ideas

- Doc 04 §4 two-tier risk architecture: evaluation order is (1) per-environment position sizing check, (2) per-environment exposure check, (3) per-environment drawdown check, (4) per-environment daily loss check, (5) global aggregator
- Doc 04 position size guardrails: equity min $1.00 / max 25% per position, crypto min $10.00 (Binance.US floor) / max 50% per position
- Circuit breaker cooldowns from Doc 04/12: equity 5 business days, crypto 3 calendar days, with graduated ramp-up 25% → 50% → 75% → 100%
- Bracket orders from PAPER-10: OTO (one-triggers-other) for Alpaca with ATR(2x) stop-loss and R:R take-profit; two-step OCO for Binance.US (simulated in paper mode)
- Cost gate from PAPER-11: reject orders where estimated round-trip transaction costs exceed 2.0% of order value
- Doc 14 §10.1 production DB seeding: 7-step M1 Mac → homelab transfer procedure
- Doc 15 scripts referenced: reconcile.py (Doc 05 §2.5), reset_cb.py (Doc 04 §4), emergency_stop.py (Doc 09 §15)
- Observation assembly from Doc 08 §13.1: runs identically during training and inference — use same ObservationAssembler from Phase 5
- Phase 7 ensemble inference: load 3 agents per env, call .predict(observation) on each, blend via EnsembleBlender
- Config already has trading_mode, capital, risk thresholds, min_order_usd=10.0 from Phase 2

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `EnsembleBlender` (training/ensemble.py): Sharpe-weighted softmax blending with adaptive validation windows — Phase 8 calls this for inference
- `ObservationAssembler` (features/assembler.py): produces (156,) equity / (45,) crypto vectors — same assembler for training and inference
- `FeaturePipeline` (features/pipeline.py): `get_observation(env, date_or_datetime)` returns complete observation from DuckDB
- `PortfolioSimulator` (envs/portfolio.py): tracks portfolio value, positions, trade log — can estimate cost/impact before broker submission
- `TurbulenceCalculator` (features/turbulence.py): Mahalanobis distance for crash protection (PAPER-20)
- `Alerter` (monitoring/alerter.py): thread-safe Discord webhook with cooldown, deduplication, severity routing — ready for trade alerts
- `DatabaseManager` (data/db.py): singleton with DuckDB/SQLite context managers, SQLite tables for trades/positions/circuit_breaker_events/portfolio_snapshots/risk_decisions already defined
- `SwingRLConfig` (config/schema.py): trading_mode, EquityConfig (max_position_size, max_drawdown_pct, daily_loss_limit_pct), CryptoConfig (same + min_order_usd=10.0), CapitalConfig (equity_usd=400, crypto_usd=47)
- Exception hierarchy (utils/exceptions.py): BrokerError, RiskVetoError, CircuitBreakerError ready for execution layer

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- Absolute imports only (from swingrl.execution.alpaca import ...)
- Pydantic config access via load_config()
- TDD: failing test first, then implementation
- structlog with keyword args for logging
- Scripts in scripts/ directory with CLI argument parsing (train.py, backtest.py, compute_features.py)

### Integration Points
- Phase 7 models: `models/active/{env}/{algo}/model.zip` + `vec_normalize.pkl` — load for inference
- Phase 5 features: `features_equity`, `features_crypto` DuckDB tables — observation source
- Phase 5 turbulence: TurbulenceCalculator for PAPER-20 crash protection
- Phase 4 databases: SQLite operational tables ready for trade logging
- Phase 4 alerter: Discord webhooks for critical/warning/info alerts
- Phase 2 config: SwingRLConfig with all risk parameters
- Empty `src/swingrl/execution/__init__.py` — target module for broker middleware

### New Dependencies (Phase 8)
- `alpaca-py`: Alpaca SDK for paper trading API (already installed from Phase 3 data ingestion)
- `python-binance` or `requests`: Binance.US REST API for order book + simulated fills (requests already installed)
- No other new dependencies expected

</code_context>

<deferred>
## Deferred Ideas

- APScheduler automation (equity 4:15 PM ET, crypto every 4H) — Phase 9 (PAPER-12)
- Discord alerting for trade executions, daily summary — Phase 9 (PAPER-13)
- Stuck agent detection (10 equity days / 30 crypto cycles all-cash) — Phase 9 (PAPER-14)
- Streamlit dashboard for system health monitoring — Phase 9 (PAPER-15)
- Healthchecks.io dead man's switch — Phase 9 (PAPER-16)
- Wash sale tracker — Phase 9 (PAPER-17)
- Comprehensive error handling with exponential backoff for all API calls — Phase 10 (HARD-02)
- Structured JSON logging to bind-mounted logs/ volume — Phase 10 (HARD-05)
- Shadow mode for new models — Phase 10 (PROD-03/04)
- deploy_model.sh (M1 Mac to homelab SCP + smoke test) — Phase 10 (PROD-02)

</deferred>

---

*Phase: 08-paper-trading-core*
*Context gathered: 2026-03-08*
