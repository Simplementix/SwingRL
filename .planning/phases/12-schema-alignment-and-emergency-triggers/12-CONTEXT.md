# Phase 12: Schema Alignment and Emergency Triggers - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix stop_polling.py to query the correct table with correct columns, and fix all 3 automated emergency trigger queries in emergency.py to work against real data sources. Pure bug-fix phase — no new capabilities.

</domain>

<decisions>
## Implementation Decisions

### INT-03: stop_polling.py table and column fixes
- Change table name from `position_tracker` to `positions`
- ALTER TABLE `positions` to add `stop_loss_price REAL` and `take_profit_price REAL` columns (nullable)
- Add `side TEXT` column to `positions` as well (stop_polling queries it)
- Schema migration goes in `DatabaseManager._init_sqlite_schema()` using `ALTER TABLE IF NOT EXISTS` pattern or try/except for idempotency
- Update the SELECT query to use the correct table and column names

### INT-04: Emergency trigger query rewrites
- Rewrite queries against existing data sources rather than creating 3 new tables (avoids adding population code with no existing callers)
- **Trigger 1 (VIX + CB):** Query VIX from DuckDB `vix_term_structure.vix_close` (cross-db via sqlite_scanner attachment is already set up). Query drawdown from SQLite `portfolio_snapshots.combined_drawdown` (already exists)
- **Trigger 2 (NaN inference):** Track NaN count via execution pipeline logging. Add a lightweight `inference_outcomes` table to SQLite (environment, timestamp, had_nan) populated by ExecutionPipeline during execute_cycle — minimal new code since the pipeline already catches NaN observations
- **Trigger 3 (IP ban):** Query from `data_ingestion_log` for recent Binance.US 418 status codes, or add a simple `api_errors` table populated by the exchange adapter's error handling path

### Claude's Discretion
- Exact DDL for any new lightweight tables (inference_outcomes, api_errors)
- Whether to use DuckDB cross-db query or a simpler approach for VIX trigger (e.g., read VIX from DuckDB directly via DatabaseManager.duckdb())
- Test structure and fixture design

</decisions>

<specifics>
## Specific Ideas

No specific requirements — fixes are well-defined by the audit findings and the existing schema in Doc 14 and db.py.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DatabaseManager.duckdb()` context manager — can query `vix_term_structure` directly for Trigger 1
- `DatabaseManager.sqlite()` context manager — for all SQLite queries
- `portfolio_snapshots` table already has `combined_drawdown` column (used by Trigger 1)
- `data_ingestion_log` table exists with status/error info from ingestion runs
- `_init_sqlite_schema()` in db.py — where ALTER TABLE migrations belong

### Established Patterns
- Idempotent DDL: `CREATE TABLE IF NOT EXISTS` throughout db.py
- Try/except wrapped trigger checks in emergency.py:449-479 — each trigger independently fails gracefully
- `check_automated_triggers()` returns `list[str]` of trigger reasons

### Integration Points
- `stop_polling.py:80-84` — SELECT query to fix (table name + columns)
- `emergency.py:433-435` — Trigger 1 VIX query (currently queries non-existent `market_indicators`)
- `emergency.py:454-457` — Trigger 2 NaN query (currently queries non-existent `inference_log`)
- `emergency.py:470-474` — Trigger 3 IP ban query (currently queries non-existent `api_status_log`)
- `db.py:321-330` — `positions` table DDL (needs new columns)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-schema-alignment-and-emergency-triggers*
*Context gathered: 2026-03-10*
