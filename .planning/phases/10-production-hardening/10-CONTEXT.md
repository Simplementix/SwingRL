# Phase 10: Production Hardening - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Harden the system for sustained autonomous operation: backup automation (daily SQLite, weekly DuckDB, monthly off-site rsync), model deployment pipeline (deploy_model.sh with SCP + smoke test), shadow mode validation (parallel hypothetical trades with auto-promotion), FinBERT sentiment pipeline with A/B experiment, comprehensive error handling with retry logic, structured JSON logging, security review, full four-tier emergency stop protocol with automated triggers, and disaster recovery testing. Covers HARD-01 through HARD-05 and PROD-01 through PROD-09.

</domain>

<decisions>
## Implementation Decisions

### Backup retention & scheduling (PROD-01)
- Daily SQLite backups: retain 14 days, rotate oldest
- DuckDB backups: **never rotate** — market data is a permanent training asset, keep all weekly backups forever
- Config and secrets (.env, swingrl.yaml) copied alongside trading_ops.db during daily backup (per Doc 14)
- Each backup job verifies integrity before considering successful (PRAGMA integrity_check for SQLite, table count + row count for DuckDB)
- Monthly off-site rsync via Tailscale to a **separate NAS/drive** (not M1 Mac)
- Discord alerts on **every backup** (success and failure) — routed to #swingrl-daily channel

### Shadow mode mechanics (PROD-03, PROD-04, PROD-05)
- Minimum evaluation period: 10 equity trading days / 30 crypto 4H cycles (roadmap success criteria, consistent with Doc 07's 2-4 weeks / 1-2 weeks range)
- Evaluation period configurable via swingrl.yaml (default 10/30)
- **Fully automatic promotion** when all 3 criteria met (per roadmap spec): Shadow Sharpe >= Active Sharpe, Shadow MDD <= 120% Active MDD, no circuit breaker triggers during shadow period
- Discord alert sent after auto-promotion with comparison stats
- Shadow hypothetical trades stored in **SQLite shadow_trades table** in trading_ops.db (same schema as trades table + model_version column)
- Failed shadow models: move to models/archive/, send Discord alert with comparison stats, active model continues unchanged
- Model lifecycle: Training → Shadow → Active → Archive → Deletion (PROD-05)
- Rollback available via CLI if auto-promotion needs reversal

### FinBERT sentiment experiment (HARD-03, HARD-04)
- **Feature toggle in config**: `sentiment.enabled` flag in swingrl.yaml, train two identical agents with/without sentiment for A/B comparison
- Sharpe improvement threshold: **+0.05** (matches existing FEAT-08 feature addition protocol from Doc 08 §16)
- FinBERT runs **within each trading cycle** (not a separate scheduled job): fetch headlines → score → store in sentiment table → observation assembler reads scores
- Resource cost: ~10-30 seconds CPU for ~100 headlines per cycle, ~500-800MB RAM during inference (fits within 2GB Docker mem_limit)
- Headline sources: **Alpaca News API** (primary, free with trading account, 200 calls/min) + **Finnhub** (secondary, 60 calls/min free)
- ~2 features per asset added to observation space (sentiment score + confidence)
- ProsusAI/finbert from Hugging Face transformers library
- If A/B test shows < 0.05 Sharpe improvement, disable sentiment permanently (discard the feature)

### Emergency stop four-tier protocol (PROD-07)
- **All 4 tiers run automatically in sequence** — no tier selection flags, no confirmation prompts. This is the panic button
- Tier 1 (<1s): Set halt flag in DB + cancel ALL open orders (Alpaca + Binance.US)
- Tier 2 (<30s): Market-sell all crypto positions on Binance.US (24/7 market)
- Tier 3 (<30s): **Auto time-aware** equity liquidation — check market hours via exchange_calendars; if extended hours open: submit limit sell at current bid; if closed: queue for market open via Alpaca time_in_force='opg'
- Tier 4 (<1min): Verify all positions closed/queued, send Discord critical alert with full status report
- **3 automated triggers** implemented (per Doc 15): (1) VIX > 40 AND global CB within 2% of firing (combined DD > -13%), (2) 2+ consecutive NaN inference outputs from any environment within 24 hours, (3) Binance.US IP ban (418) while crypto positions are open
- Manual invocation: `docker exec swingrl-bot python scripts/emergency_stop.py --reason "description"` or via SSH

### Jupyter analysis notebooks (HARD-01)
- Weekly performance review notebooks in notebooks/ directory
- Portfolio curves, trade logs, risk metrics per Doc 09 dashboard spec
- Complements Streamlit dashboard — notebooks for deep analysis, dashboard for quick-glance monitoring

### Error handling & retry logic (HARD-02)
- Comprehensive retry with exponential backoff for all external API calls (Alpaca, Binance.US, FRED, Finnhub, Alpaca News)
- Extends Phase 8's basic 3-retry pattern to all data pipeline and execution API calls
- Structured error categories mapping to SwingRLError hierarchy

### Structured JSON logging (HARD-05)
- Extends Phase 2's structlog JSON/console switching to comprehensive audit trails
- JSON logs to bind-mounted logs/ volume in Docker
- All trading decisions, risk checks, and system events logged with structured context

### Security review (PROD-06)
- Verify non-root containers (trader user UID 1000 from Phase 1)
- env_file permissions (chmod 600)
- Binance.US API key: withdrawals OFF, IP allowlisting, Reading + Spot Trading permissions only
- 90-day key rotation schedule (staggered: Alpaca month 1, Binance.US month 2, per Doc 07)
- Rotation documented as runbook in scripts/ or docs/

### Disaster recovery test (PROD-08, PROD-09)
- Full 9-step quarterly recovery checklist per Doc 14/15
- Test: stop container, delete volumes, restore from backup, verify system resumes paper trading correctly
- Verification includes: DB integrity, row counts, model loading, feature computation, scheduler starts, first cycle executes

### Model deployment pipeline (PROD-02)
- deploy_model.sh: SCP model from M1 Mac to homelab via Tailscale
- 6-point smoke test on homelab: model deserialization, valid output shape, non-degenerate actions, inference < 100ms, VecNormalize loads, no NaN outputs (per Doc 15)
- Model goes to models/shadow/ first (not directly to active) — shadow mode evaluates before promotion

### Claude's Discretion
- Backup cron schedule times (exact hours for daily/weekly/monthly jobs)
- Shadow mode inference integration with existing scheduler jobs (how to run both active + shadow in same cycle)
- Jupyter notebook layout and visualization library choices
- Retry backoff parameters (base delay, max retries, jitter)
- Security review checklist script implementation
- Disaster recovery test automation level (fully scripted vs manual checklist)
- deploy_model.sh shell script implementation details

</decisions>

<specifics>
## Specific Ideas

- Doc 14 backup verification: SQLite uses `PRAGMA integrity_check`, DuckDB verifies table count + row counts for ohlcv_daily, ohlcv_4h
- Doc 07 §3: Shadow comparison period 2-4 weeks equity / 1-2 weeks crypto (implementation minimum = 10 trading days / 30 4H cycles)
- Doc 07 §5: FinBERT pipeline runs per-cycle, not as separate job. 10 assets x ~10 headlines = ~100 headlines per batch
- Doc 09 §15: Emergency stop "designed for mixed-market-hours operation" — Tier 3 must handle regular hours, extended hours, overnight, and weekend scenarios
- Doc 15: Three automatic emergency triggers defined alongside manual triggers
- Doc 08 §16: Feature addition protocol — +0.05 Sharpe threshold applies uniformly to all features including FinBERT
- Doc 15: deploy_model.sh smoke test has 6 specific checks (deserialization, output shape, non-degenerate actions, inference speed, VecNormalize, no NaN)
- Doc 14 §10.1: 7-step production DB seeding procedure already implemented in Phase 8 — disaster recovery reuses this
- Doc 07: Staggered key rotation — Alpaca and Binance.US rotated in different months to avoid simultaneous credential changes

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `emergency_stop.py` (scripts/): Phase 9's Tier 1 implementation (halt flag only) — extend with Tiers 2-4
- `halt_check.py` (scheduler/): set_halt() and check_halt() functions — pre-cycle halt check pattern
- `reset_halt.py` (scripts/): Clears halt flag for resuming after emergency stop
- `Alerter` (monitoring/alerter.py): Discord webhook with severity routing — use for backup alerts, shadow promotion alerts, emergency alerts
- `ExecutionPipeline` (execution/pipeline.py): execute_cycle() — shadow mode runs a second pipeline with shadow model
- `EnsembleBlender` (training/ensemble.py): Sharpe-weighted blending — shadow model uses same blender
- `DatabaseManager` (data/db.py): SQLite/DuckDB context managers — add shadow_trades table, backup job uses DB paths
- `AlpacaAdapter` + `BinanceSimAdapter` (execution/adapters/): broker connections for order cancellation and liquidation in emergency stop
- `healthcheck.py` (scripts/): Docker HEALTHCHECK probe — disaster recovery verifies this works after restore
- `seed_production.py` (scripts/): DB seeding — disaster recovery reuses this procedure
- `SwingRLConfig` (config/schema.py): extend with backup, shadow, sentiment, and security config sections
- `configure_logging()` (utils/logging.py): JSON/console switching — extend for file output to logs/ volume
- `FeaturePipeline` (features/pipeline.py): observation assembly — FinBERT features integrate here

### Established Patterns
- Type hints on all signatures, absolute imports, structlog logging
- Scripts in scripts/ with CLI argument parsing
- TDD: failing test first, then implementation
- Docker single-container with bind-mounted volumes (db/, models/, logs/)
- SwingRLError hierarchy for typed exceptions
- Pydantic config access via load_config()

### Integration Points
- Phase 9 scheduler: APScheduler jobs — shadow inference added to existing cycle jobs
- Phase 9 Streamlit dashboard: backup health and shadow mode status could be added to System Health page
- Phase 8 execution pipeline: shadow mode runs parallel pipeline per cycle
- Phase 7 model loading: shadow model loaded from models/shadow/ using same agent loading code
- Phase 5 feature pipeline: FinBERT sentiment features added to observation assembler
- Phase 4 Alerter: backup, shadow, and emergency alerts via existing Discord infrastructure
- Phase 3 data ingestors: retry/backoff patterns extended across all ingestors

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 10-production-hardening*
*Context gathered: 2026-03-09*
