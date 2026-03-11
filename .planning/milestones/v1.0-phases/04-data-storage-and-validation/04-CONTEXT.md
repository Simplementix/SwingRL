# Phase 4: Data Storage and Validation - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

DuckDB (market_data.ddb) and SQLite (trading_ops.db) databases operational with validated schema, Parquet-to-DuckDB data loading, cross-source validation (Alpaca vs yfinance), ingestion logging, quarantine migration from Parquet to DuckDB, corporate actions detection and handling, Discord alerter module with level-based routing, and aggregation views for "store lowest, aggregate up" strategy. Covers DATA-06 through DATA-13.

</domain>

<decisions>
## Implementation Decisions

### Table scope and incremental strategy
- **DuckDB (market_data.ddb):** 5 data pipeline tables only — ohlcv_daily, ohlcv_4h, macro_features, data_quarantine, data_ingestion_log
- **SQLite (trading_ops.db):** All 10 tables upfront per Doc 14 §12 — trades, positions, risk_decisions, portfolio_snapshots, system_events, corporate_actions, wash_sale_tracker, circuit_breaker_events, options_positions, alert_log
- Rationale: SQLite tables are small, DDL is defined in Doc 14, avoids migration complexity. circuit_breaker_events needed as runtime halt flag even before trading. DuckDB feature/training tables deferred to Phases 5-7 (CREATE TABLE IF NOT EXISTS is trivial)

### DuckDB DDL (market_data.ddb)
- **ohlcv_daily:** symbol (TEXT), date (DATE), open (DOUBLE), high (DOUBLE), low (DOUBLE), close (DOUBLE), volume (BIGINT), adjusted_close (DOUBLE), fetched_at (TIMESTAMP). Sort: symbol, date
- **ohlcv_4h:** symbol (TEXT), datetime (TIMESTAMP), open (DOUBLE), high (DOUBLE), low (DOUBLE), close (DOUBLE), volume (DOUBLE), source (TEXT), fetched_at (TIMESTAMP). Sort: symbol, datetime
- **macro_features:** date (DATE), series_id (TEXT), value (DOUBLE), release_date (DATE). Sort: date
- **data_quarantine:** quarantined_at (TIMESTAMP), source (TEXT), symbol (TEXT), raw_data_json (JSON/TEXT), failure_reason (TEXT), severity (TEXT). Sort: quarantined_at. Retention: 1 year
- **data_ingestion_log:** run_id (UUID PK), timestamp (TIMESTAMP), environment (TEXT), symbol (TEXT), status (TEXT), rows_inserted (INTEGER), errors_count (INTEGER), duration_ms (INTEGER), binance_weight_used (INTEGER). Sort: timestamp. Granularity: per-symbol per run

### SQLite DDL (trading_ops.db)
- All dates as TEXT (ISO-8601), booleans as INTEGER (0/1)
- **trades:** trade_id (PK TEXT/UUID), timestamp, symbol, side, quantity (REAL), price (REAL), commission (REAL), slippage (REAL), environment, broker, order_type, trade_type
- **positions:** symbol + environment (composite PK), quantity (REAL), cost_basis (REAL), last_price (REAL), unrealized_pnl (REAL), updated_at
- **risk_decisions:** decision_id (PK), timestamp, environment, symbol, proposed_action, final_action, risk_rule_triggered, reason
- **portfolio_snapshots:** timestamp (PK), total_value (REAL), equity_value (REAL), crypto_value (REAL), cash_balance (REAL), high_water_mark (REAL), daily_pnl (REAL), drawdown_pct (REAL)
- **system_events:** event_id (PK), timestamp, level, module, event_type, message, metadata_json
- **corporate_actions:** action_id (PK), symbol, action_type, effective_date, ratio (REAL), amount (REAL), processed (INTEGER)
- **wash_sale_tracker:** symbol, sale_date, loss_amount (REAL), wash_window_end, triggered (INTEGER)
- **circuit_breaker_events:** event_id (PK), environment, triggered_at, resumed_at (NULLable), trigger_value (REAL), threshold (REAL), reason
- **options_positions:** spread_id (PK), underlying, strategy, expiration, short_strike (REAL), long_strike (REAL), premium_received (REAL), current_value (REAL), delta (REAL), theta (REAL)
- **alert_log:** alert_id (PK), timestamp, level, title, message_hash, sent (INTEGER)
- **Indexes:** circuit_breaker_events (environment, resumed_at), trades (symbol, environment), positions (symbol, environment)

### DatabaseManager class
- Full DatabaseManager class (not just utility functions) — singleton pattern
- Separate methods for DuckDB and SQLite (different type systems, different OLAP/OLTP purposes)
- Context managers: `with db.duckdb() as conn:`, `with db.sqlite() as conn:`
- SQLite WAL mode enforced for concurrent read/write
- DuckDB write lock (threading.Lock) — DuckDB is single-writer, APScheduler ThreadPoolExecutor(3) may trigger concurrent writes
- Auto-install and load sqlite_scanner extension during initialization
- Cross-DB join support: `attach_sqlite()` method for sqlite_scanner interop
- Handles schema creation via `init_schema()` method (CREATE TABLE IF NOT EXISTS)
- Connection init at startup, graceful close at shutdown
- Location: `src/swingrl/data/db.py`

### Parquet-to-DB migration path
- Parquet remains as intermediate/archival format — ingestors continue writing to Parquet
- Loader/sync uses DuckDB's native `read_parquet()` for bulk loading (no pandas overhead)
- One-time migration script for existing Phase 3 Parquet files → DuckDB tables
- Parquet = immutable archive / backup, DuckDB = analytical engine
- BaseIngestor.run() modified to: fetch → validate → Parquet → auto-sync to DuckDB → ingestion log (all in same run)
- BaseIngestor calls DatabaseManager directly (no separate pipeline orchestrator)

### Store lowest, aggregate up (DATA-09)
- Only store daily (equity) and 4H (crypto) base bars
- Implement SQL views in Phase 4: CREATE VIEW ohlcv_weekly AS ... (first open, max high, min low, last close, sum volume, last adjusted_close)
- No pre-stored aggregates — DuckDB aggregates 10 years in <100ms at this scale
- Views available for Phase 5 feature engineering and status page

### Cross-source validation (DATA-11)
- Weekly scheduled job (separate from daily ingestion)
- Compare Alpaca vs yfinance daily closing prices for a sample of symbols
- Tolerance: $0.05 for liquid stocks
- Minor discrepancies → log with "warning" severity, flag as "review"
- Critical failures only → quarantine + alert
- Results included in weekly digest alert

### Corporate actions handling (DATA-10)
- Two-layer detection: Alpaca corporate actions API pre-market check (M-F 8:00 AM ET) + >30% overnight price change heuristic
- Secondary sources: yfinance calendar, Alpha Vantage
- On split detection: automated refetch of full price history + recompute technical indicators
- Both raw close and adjusted_close stored in ohlcv_daily (Alpaca provides adjusted prices)
- Crypto: 40% threshold (wider due to volatility), no traditional corporate actions — hard forks not handled in v1
- corporate_actions table cross-referenced during validation to avoid false-positive quarantine on known splits

### Discord alerter (DATA-13)
- General-purpose module at `src/swingrl/monitoring/alerter.py` — used by all future phases
- Interface: `send_alert(level, title, message)` where level is "critical" | "warning" | "info"
- `send_daily_digest()` — flush info buffer, scheduled at 6:00 PM ET
- Three alert levels: Critical (immediate, Discord), Warning (immediate, Discord), Info (batched into daily digest)
- Discord-only notifications (PROJECT.md decision — no email backup despite spec mentioning it)
- Discord webhook URL via DISCORD_WEBHOOK_URL env var (secret, not in YAML)
- Discord embeds with color-coded sidebar: Critical=Red, Warning=Yellow/Orange, Info=Blue/Green
- Include: timestamp, environment (Equity/Crypto/Global), title, markdown details
- Rate limiting: consecutive_failures_before_alert=3, alert_cooldown_minutes=30
- Thread-safe (APScheduler ThreadPoolExecutor(3) may call from different threads)
- Synchronous webhook POST via httpx (not discord.py bot framework)
- alert_log table in SQLite for deduplication and audit trail

### Adjusted close handling
- Alpaca provides adjusted close prices via API
- Phase 3 AlpacaIngestor needs modification to capture and store adjusted_close
- Used for all historical training data and indicator computation

### FRED vintage mapping
- Phase 3 Parquet vintage_date → macro_features.release_date column
- Forward-fill via ASOF JOIN handled in Phase 5 feature engineering

### Quarantine migration
- Phase 3 Parquet quarantine files → migrate into DuckDB data_quarantine table (JSON format)
- Future quarantine goes directly to DuckDB table
- Quarantine retention: 1 year

### Ingestion logging
- Automatic on every ingestor run (not opt-in)
- Per-symbol per run granularity (one row per symbol per ingestor run)
- Tracks: run_id, timestamp, environment, symbol, status, rows_inserted, errors_count, duration_ms, binance_weight_used

### Config schema additions
- DB paths → `system` section in swingrl.yaml (defaults: data/db/market_data.ddb, data/db/trading_ops.db)
- Discord webhook URL → .env file (DISCORD_WEBHOOK_URL)
- Alert cooldown settings → `alerting` section (alert_cooldown_minutes, consecutive_failures_before_alert)

### Schema versioning
- Additive approach — CREATE TABLE IF NOT EXISTS + manual ALTER TABLE
- No migration framework (no alembic) — DuckDB doesn't support traditional migration tools
- DuckDB recomputes features from raw OHLCV when schema changes

### Scripts and CLI
- `scripts/init_db.py` for DDL initialization and schema creation
- Schema creation is manual via script (not auto-on-first-run)
- DB health check included: PRAGMA integrity_check for SQLite, table existence checks for DuckDB
- Cross-source validator integrated into weekly pipeline (not standalone CLI)

### Dependencies
- duckdb — already in pyproject.toml
- yfinance — already in pyproject.toml (cross-source validation + fundamentals)
- httpx — for Discord webhook POST (lightweight, already available)
- No new heavy dependencies needed

### Testing strategy
- File-based temp DuckDB databases (not in-memory) — matches production behavior
- Discord webhook testing via pytest-mock — verify JSON payloads without hitting Discord API
- Schema validation tests: confirm all tables created correctly
- Cross-DB join tests: verify sqlite_scanner interop works
- Ingestion logging tests: verify per-symbol logging and Binance weight tracking
- Corporate actions tests: split detection heuristic with fixture data

### Claude's Discretion
- Exact DatabaseManager API beyond the decisions above (helper methods, error handling patterns)
- DuckDB read_parquet bulk loading details (batch size, error handling for malformed Parquet)
- Exact Discord embed field layout and formatting
- init_db.py CLI argument design
- Test fixture data values for corporate actions and cross-source validation
- Aggregation view SQL syntax details
- DuckDB write lock implementation specifics (threading.Lock vs queue-based)

</decisions>

<specifics>
## Specific Ideas

- DatabaseManager as singleton ensures consistent connection management across APScheduler's ThreadPoolExecutor(3)
- DuckDB's native read_parquet avoids pandas serialization overhead for bulk loads
- Parquet kept as immutable archive alongside DuckDB — resilient if DuckDB file corrupts
- Per-symbol ingestion logging enables the retraining pipeline to abort if any symbol has >5% missing bars
- Aggregation views in Phase 4 mean Phase 5 feature engineering can consume weekly/monthly bars immediately
- corporate_actions cross-reference prevents false quarantine on known split days — critical for automated pipeline reliability
- Thread-safe alerter with cooldown prevents "alert storms" from concurrent crypto cycles

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaseIngestor` (data/base.py): ABC with fetch/validate/store/run — needs modification to add DuckDB sync and ingestion logging
- `DataValidator` (data/validation.py): 12-step checklist with row-level and batch-level validation — Step 12 (cross-source) currently deferred, implement in Phase 4
- `ParquetStore` (data/parquet_store.py): Read/upsert Parquet files with atomic write — continues as intermediate storage
- `SwingRLConfig` (config/schema.py): Pydantic v2 config — needs new system.duckdb_path, system.sqlite_path, alerting section
- `DataError` (utils/exceptions.py): Ready for DB and alerter errors
- structlog (utils/logging.py): configure_logging() established — DatabaseManager and alerter use structlog

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- pathlib.Path for all file operations
- Absolute imports only
- Pydantic config access via load_config()
- TDD: failing test first, then implementation
- API credentials via environment variables only
- structlog with keyword args (no f-strings)

### Integration Points
- Phase 3 ingestors write Parquet to data/equity/, data/crypto/, data/macro/ — Phase 4 loads these into DuckDB
- Phase 3 quarantine writes to data/quarantine/ — Phase 4 migrates to DuckDB data_quarantine table
- Phase 5 reads DuckDB ohlcv_daily/ohlcv_4h + macro_features for feature engineering
- Phase 5 uses ASOF JOIN on macro_features.release_date for forward-fill alignment
- Phase 8 uses circuit_breaker_events as runtime halt flag
- Phase 9 APScheduler triggers ingestors + cross-source validation + corporate actions check
- Config: new system and alerting sections added to swingrl.yaml

</code_context>

<deferred>
## Deferred Ideas

- Retraining pipeline CLI (scripts/retrain.py) — Phase 7
- Emergency stop script (scripts/emergency_stop.py) — Phase 10
- Circuit breaker reset script (scripts/reset_cb.py) — Phase 8
- Production DB seeding procedure (7-step M1 Mac → homelab transfer) — Phase 8
- APScheduler job scheduling — Phase 9
- Stuck agent detection — Phase 9
- Multiple Discord channels (#trades, #risk-alerts, #daily-digest) — Phase 9 (single webhook sufficient for Phase 4)
- Chart image attachments in Discord alerts — Phase 9
- Crypto hard fork handling — future version

</deferred>

---

*Phase: 04-data-storage-and-validation*
*Context gathered: 2026-03-06*
