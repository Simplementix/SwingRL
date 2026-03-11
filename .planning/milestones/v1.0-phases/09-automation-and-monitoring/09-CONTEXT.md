# Phase 9: Automation and Monitoring - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Scheduled autonomous execution via APScheduler, Discord alerting for all significant events (trades, circuit breakers, daily summary, stuck agents), Streamlit dashboard for system health monitoring, Healthchecks.io dead man's switch, and wash sale tracking. Covers PAPER-12, PAPER-13, PAPER-14, PAPER-15, PAPER-16, PAPER-17.

Phase 10 handles comprehensive error handling (HARD-02), structured JSON logging (HARD-05), backup automation (PROD-01), model deployment pipeline (PROD-02), shadow mode (PROD-03/04), security review (PROD-06), emergency_stop.py four-tier protocol enhancements (PROD-07), and disaster recovery testing (PROD-08/09).

</domain>

<decisions>
## Implementation Decisions

### Scheduler architecture
- **Single main.py entrypoint**: one Python process initializes config, DB, APScheduler (BackgroundScheduler with SQLAlchemy jobstore), registers all jobs, and blocks. Replaces Phase 8's sleep-loop placeholder CMD in Dockerfile
- **Pre-cycle halt check**: every job checks emergency_flags table in SQLite before executing. If halt flag is active, skip the cycle and log it (PAPER-12 requirement)
- **Job failure handling**: log the error, send Discord critical alert, skip this cycle. Next scheduled cycle starts fresh — no retry queue, no immediate retry. Matches Phase 8's "skip the trade" pattern
- **Graceful shutdown**: Claude's discretion on SIGTERM handling (wait for active job vs immediate). Bracket orders live at broker, APScheduler state persists in SQLite — crash mid-cycle is safe
- **Register what's needed**: don't force-match Doc 05's estimate of 13 jobs. Register all necessary jobs; HC pings and status page regen are post-cycle callbacks, not separate scheduled jobs

### Scheduled jobs
- **Equity trading cycle**: 1 cron job, daily at 4:15 PM ET. Includes data fetch, validate, compute features, inference, execute trades, post-cycle callbacks (HC ping, status page regen)
- **Crypto trading cycle**: 1 cron job firing 6x/day at 5 min past each 4H bar close (00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC). Same pipeline as equity but with crypto-specific parameters
- **Daily summary**: 1 cron job at 6:00 PM ET
- **Stuck agent check**: 1 cron job (daily). Thresholds: 10 consecutive equity trading days or 30 consecutive crypto cycles all-cash
- **Wash sale scan**: runs after each equity cycle (callback, not separate job). Equities only
- **Weekly fundamentals refresh**: 1 cron job, Sunday ~6 PM ET
- **Monthly macro refresh**: 1 cron job, 1st of month ~6 PM ET
- **Data pipeline integrated**: data fetch + validate + store runs as part of each trading cycle, not as a separate job

### Discord alert formatting
- **Trade alerts**: detailed per-trade — separate embed per fill with symbol, side, quantity, fill price, stop/TP levels, risk check results, observation summary
- **Daily summary (6 PM ET)**: portfolio + P&L focus — per-environment current value, daily P&L ($ and %), positions held, trades today. Combined: total portfolio value, combined daily P&L, circuit breaker status
- **Stuck agent alerts**: include diagnostics — consecutive all-cash cycle count, last non-trivial action date, current regime state, turbulence level
- **Channel split by severity**: critical/warning alerts → #swingrl-alerts channel, daily summary/info → #swingrl-daily channel. Two webhook URLs in config. Lets operator mute daily without missing critical alerts
- **Existing Alerter**: Phase 4's Alerter (cooldown, dedup, daily digest, thread-safety) is the foundation. Extend with trade-specific and summary-specific embed builders

### Streamlit dashboard
- **Streamlit only**: no static HTML status page layer. Streamlit serves as both quick-glance status and analytical dashboard. Skip Doc 09 Layer 2 (static HTML)
- **Separate Docker container**: swingrl-dashboard service in docker-compose. Reads from same DB bind mounts (read-only). Isolates web surface from trading bot. Can restart independently
- **Full multi-page dashboard**:
  - Page 1: Portfolio overview (equity curves, P&L per environment and combined)
  - Page 2: Trade log with filters
  - Page 3: Risk metrics (drawdown tracking, circuit breaker status)
  - Page 4: System health (traffic-light status per environment, last heartbeat times, last 5 trades)
- **Auto-refresh every 5 minutes** via st.auto_rerun. No manual refresh needed (PAPER-15 requirement)

### Healthchecks.io
- **Ping after each successful cycle**: crypto check pinged after each crypto cycle completion, equity check pinged after each equity cycle completion. If a cycle fails, no ping → dead man's switch fires within the alert window (70 min crypto, 25 hr equity)
- **Alerts via Discord webhook**: configure Healthchecks.io to alert via Discord webhook integration. All alerts in one place. No email fallback needed
- **Post-cycle callback**: HC ping is a callback at the end of each cycle, not a separate scheduled job

### Wash sale tracking
- **Warn only, don't block**: send Discord warning alert when buying a symbol within its 30-day wash window. Log triggered=true in wash_sale_tracker table. Don't prevent the trade — tax implications are informational
- **Equities only**: crypto exempt under current IRS rules. No crypto wash sale tracking
- **Runs after each equity cycle**: scan for new realized losses and update wash windows as a post-cycle callback

### Emergency stop integration
- **DB flag + docker exec**: emergency_stop.py sets a halt flag in SQLite (emergency_flags table), then calls broker APIs to cancel orders/liquidate. Invoked via `docker exec swingrl-bot python scripts/emergency_stop.py` or SSH
- **DB flag only (no scheduler.shutdown())**: the flag prevents all future cycles. APScheduler keeps running but every job is a no-op due to pre-cycle halt check. To resume: clear the flag via a reset script. Simpler and survives container restart

### Stop-price polling
- **Daemon thread started from main.py**: main.py starts APScheduler AND launches the crypto stop-price polling daemon thread. Thread runs independently, polls every 60s via Binance.US REST API, checks emergency flag each iteration. Dies when main process exits. Matches Phase 8 CONTEXT decision

### Claude's Discretion
- APScheduler configuration details (jobstore path, executor type, max_instances)
- Streamlit page layout and styling (chart libraries, color schemes)
- Discord embed field layout and formatting details
- SIGTERM graceful shutdown implementation specifics
- Status page regen callback implementation
- Healthchecks.io ping implementation (HTTP GET to check URL)
- Emergency flags table schema
- Wash sale scanner implementation details

</decisions>

<specifics>
## Specific Ideas

- Doc 09 §7: Status page regenerated after each cycle — now handled by Streamlit dashboard refresh instead of static HTML
- Doc 09 §9: Multi-page Streamlit dashboard layout with Portfolio Overview, Trade Log, Risk Metrics, System Health pages
- Doc 05 §2.8: APScheduler BackgroundScheduler with SQLAlchemy jobstore for crash-resilient job state
- Doc 09 §15: Emergency stop four-tier protocol — Phase 9 builds the emergency_stop.py script and DB flag mechanism
- Doc 08 Part A §3: Separate cadences for data pipelines — equity 5 PM ET, crypto 4H, weekly fundamentals Sunday, monthly macro 1st
- Doc 07 §wash_sale: Detection flow — after equity sell at loss, record symbol + 30-day window, flag on repurchase within window
- Doc 15 INFRA-1: APScheduler job state persists in SQLite — system self-heals after container restart
- Phase 8 CONTEXT: "Single process entrypoint: one Python process with APScheduler managing scheduled jobs. Stop-price polling loop runs as background daemon thread"

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Alerter` (monitoring/alerter.py): thread-safe Discord webhook with cooldown, dedup, daily digest — extend with trade-specific and summary embed builders
- `run_cycle.py` (scripts/): full trading cycle CLI — refactor core logic into a callable function for APScheduler jobs
- `ExecutionPipeline` (execution/pipeline.py): execute_cycle() for a given environment — called by scheduled jobs
- `FeaturePipeline` (features/pipeline.py): get_observation() for inference — part of each trading cycle
- `DatabaseManager` (data/db.py): singleton with DuckDB/SQLite context managers — foundation for all DB operations
- `healthcheck.py` (scripts/): Docker HEALTHCHECK probe — already validates DB connectivity
- `SwingRLConfig` (config/schema.py): alerting config (cooldown_minutes) — extend with webhook URLs, HC check IDs

### Established Patterns
- Type hints on all signatures, absolute imports, structlog logging
- Scripts in scripts/ with CLI argument parsing (train.py, backtest.py, run_cycle.py)
- TDD: failing test first, then implementation
- Docker single-container with bind-mounted volumes (db/, models/, logs/)
- Phase 8 placeholder CMD in Dockerfile — Phase 9 replaces with `CMD ["python", "scripts/main.py"]`

### Integration Points
- Phase 8 execution pipeline: ExecutionPipeline.execute_cycle() — called by scheduler for each environment
- Phase 8 risk layer: circuit breaker state in SQLite — checked during pre-cycle halt check
- Phase 4 Alerter: existing alert infrastructure — extend for trade-specific messages
- Phase 3 data ingestors: AlpacaIngestor, BinanceIngestor, FredIngestor — called within each trading cycle
- Phase 5 feature pipeline: FeaturePipeline — data fetch + feature compute within each cycle
- Phase 8 stop-price polling: background daemon thread for crypto stop/TP monitoring

### New Dependencies (Phase 9)
- `apscheduler`: APScheduler 3.x with SQLAlchemy jobstore
- `streamlit`: Streamlit for dashboard (separate container)
- `plotly` or `altair`: charting library for dashboard visualizations
- `httpx`: already installed — used for Healthchecks.io pings

</code_context>

<deferred>
## Deferred Ideas

- Comprehensive error handling with exponential backoff for all API calls — Phase 10 (HARD-02)
- Structured JSON logging to bind-mounted logs/ volume — Phase 10 (HARD-05)
- Backup automation (daily SQLite, weekly DuckDB, monthly off-site rsync) — Phase 10 (PROD-01)
- deploy_model.sh (M1 Mac to homelab SCP + smoke test) — Phase 10 (PROD-02)
- Shadow mode for new models — Phase 10 (PROD-03/04)
- Security review (non-root containers, IP allowlisting, key rotation) — Phase 10 (PROD-06)
- emergency_stop.py four-tier protocol with Tier 2-4 (liquidation, extended hours) — Phase 10 (PROD-07)
- Disaster recovery testing — Phase 10 (PROD-08/09)
- Jupyter analysis notebooks for weekly performance review — Phase 10 (HARD-01)

</deferred>

---

*Phase: 09-automation-and-monitoring*
*Context gathered: 2026-03-09*
