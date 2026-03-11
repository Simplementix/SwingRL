# Phase 12: Schema Alignment and Emergency Triggers - Research

**Researched:** 2026-03-10
**Domain:** SQLite/DuckDB schema alignment, SQL query bug fixes
**Confidence:** HIGH

## Summary

Phase 12 is a pure bug-fix phase addressing two integration gaps (INT-03, INT-04) found during the v1.0 milestone audit. The work involves fixing incorrect table/column references in `stop_polling.py` and rewriting three non-functional emergency trigger queries in `emergency.py` to use actual data sources.

The codebase analysis reveals precise, well-bounded fixes. stop_polling.py references a non-existent `position_tracker` table and missing columns (`stop_loss_price`, `take_profit_price`, `side`). emergency.py references three non-existent tables (`market_indicators`, `inference_log`, `api_status_log`). All fixes have been scoped by the CONTEXT.md decisions with clear data source mappings.

**Primary recommendation:** Fix queries against existing schema, add 3 new columns to `positions` table, create 2 lightweight SQLite tables (`inference_outcomes`, `api_errors`), and query VIX from DuckDB `macro_features` directly via `DatabaseManager.duckdb()`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **INT-03 fix:** Change table name from `position_tracker` to `positions`, add `stop_loss_price REAL`, `take_profit_price REAL`, `side TEXT` columns to `positions` via idempotent ALTER TABLE in `_init_sqlite_schema()`
- **INT-04 Trigger 1:** Query VIX from DuckDB `vix_term_structure.vix_close` or `macro_features`, query drawdown from SQLite `portfolio_snapshots.combined_drawdown`
- **INT-04 Trigger 2:** Track NaN via `inference_outcomes` table in SQLite, populated by ExecutionPipeline
- **INT-04 Trigger 3:** Query from `data_ingestion_log` for 418 status codes, or add `api_errors` table

### Claude's Discretion
- Exact DDL for new lightweight tables (inference_outcomes, api_errors)
- Whether to use DuckDB cross-db query or direct `DatabaseManager.duckdb()` for VIX trigger
- Test structure and fixture design

### Deferred Ideas (OUT OF SCOPE)
None

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-10 | Bracket orders: OTO for Alpaca (ATR stop-loss, R:R take-profit), two-step OCO for Binance.US | Stop polling must query correct table with stop/TP prices -- INT-03 fix enables PAPER-10 |
| PROD-07 | emergency_stop.py: four-tier kill switch | Automated triggers must query real data to fire emergency stop -- INT-04 fix enables PROD-07 |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlite3 | stdlib | SQLite operations | Already used throughout codebase via DatabaseManager |
| duckdb | existing | DuckDB analytical queries | Already used for macro_features table |
| structlog | existing | Logging | Project convention |

No new dependencies required. All fixes use existing libraries.

## Architecture Patterns

### Recommended Changes Structure
```
src/swingrl/
  data/db.py                    # Add columns to positions, add 2 new tables
  scheduler/stop_polling.py     # Fix table name + column names in SELECT
  execution/emergency.py        # Rewrite 3 trigger queries
  execution/pipeline.py         # Add NaN tracking to execute_cycle (inference_outcomes insert)
tests/
  test_emergency_stop.py        # Update trigger tests to use real schemas
  scheduler/test_stop_polling.py # Add test for correct query execution
```

### Pattern 1: Idempotent ALTER TABLE for SQLite
**What:** SQLite lacks `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Use try/except around `ALTER TABLE` to handle the case where column already exists.
**When to use:** Adding columns to existing tables in `_init_sqlite_schema()`.
**Example:**
```python
# Source: Existing project pattern (db.py uses CREATE TABLE IF NOT EXISTS)
# SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN
for col_sql in [
    "ALTER TABLE positions ADD COLUMN stop_loss_price REAL",
    "ALTER TABLE positions ADD COLUMN take_profit_price REAL",
    "ALTER TABLE positions ADD COLUMN side TEXT",
]:
    try:
        conn.execute(col_sql)
    except sqlite3.OperationalError:
        pass  # Column already exists
```

### Pattern 2: Direct DuckDB Query for VIX (Recommended)
**What:** Use `DatabaseManager.duckdb()` context manager to query VIX directly from `macro_features` table, instead of cross-db SQLite scanner attachment.
**When to use:** Trigger 1 VIX check.
**Why preferred:** Simpler than cross-db attachment, avoids potential locking issues, and the VIX query is independent from the drawdown query anyway (separate DB calls are fine for a 5-minute trigger check).
**Example:**
```python
# Query VIX from DuckDB macro_features
with db.duckdb() as cursor:
    vix_row = cursor.execute(
        "SELECT value FROM macro_features "
        "WHERE series_id = 'VIXCLS' "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()

# Then query drawdown from SQLite
with db.sqlite() as conn:
    dd_row = conn.execute(
        "SELECT drawdown_pct FROM portfolio_snapshots "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
```

### Anti-Patterns to Avoid
- **Querying non-existent tables:** The root cause of both INT-03 and INT-04. Always verify table/column names against `db.py` DDL.
- **Creating tables that have no population path:** CONTEXT.md explicitly chose to rewrite queries against existing data rather than creating 3 new unpopulated tables. Only `inference_outcomes` and `api_errors` are new, and both have clear population paths.
- **Using `combined_drawdown` column name:** The actual column is `drawdown_pct` in `portfolio_snapshots`. The emergency trigger queries this wrong column name.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| VIX data access | New `vix_term_structure` or `market_indicators` table | Existing `macro_features` in DuckDB with `series_id='VIXCLS'` | Data already there from FRED ingestion |
| Drawdown monitoring | New combined_drawdown column | Existing `drawdown_pct` in `portfolio_snapshots` | Already populated by PositionTracker.record_snapshot() |
| IP ban detection | Complex HTTP interceptor | Simple `api_errors` SQLite table + exchange adapter error path logging | Lightweight, matches existing error handling pattern |

## Common Pitfalls

### Pitfall 1: Wrong Column Name for Drawdown
**What goes wrong:** `emergency.py:439` queries `combined_drawdown` but the actual column is `drawdown_pct`.
**Why it happens:** Column name was assumed from circuit breaker terminology, not verified against DDL.
**How to avoid:** The fix must change `combined_drawdown` to `drawdown_pct` AND adjust the comparison logic. Note: `drawdown_pct` is stored as a positive fraction (e.g., 0.13 = 13% drawdown) by `position_tracker.py:167`, while the current code compares against `_CB_DRAWDOWN_TRIGGER = -13.0` (a negative percentage). The comparison logic needs adjustment.
**Warning signs:** Query returns None even when data exists.

### Pitfall 2: DuckDB fetchone() Returns Tuple, Not Dict
**What goes wrong:** DuckDB cursor `fetchone()` returns a tuple, not a `sqlite3.Row` dict-like object.
**Why it happens:** SQLite connections have `row_factory = sqlite3.Row` set in `DatabaseManager.sqlite()`, but DuckDB cursors return plain tuples.
**How to avoid:** Access VIX value by index (`row[0]`) not by name (`row["value"]`), or use `cursor.description` to map columns.
**Warning signs:** `TypeError: tuple indices must be integers or slices, not str`.

### Pitfall 3: data_ingestion_log Is in DuckDB, Not SQLite
**What goes wrong:** CONTEXT.md mentions querying `data_ingestion_log` for 418 status codes, but this table is in DuckDB (see `db.py:211`), not SQLite. Also, it has no `status_code` column -- it has `status TEXT` and `errors_count INTEGER`.
**Why it happens:** The table stores ingestion run summaries, not individual HTTP request errors.
**How to avoid:** Create a new lightweight `api_errors` SQLite table for HTTP error tracking instead.
**Warning signs:** Query fails silently in the try/except block.

### Pitfall 4: Drawdown Comparison Direction
**What goes wrong:** Current code uses `dd_row["combined_drawdown"] < _CB_DRAWDOWN_TRIGGER` where `_CB_DRAWDOWN_TRIGGER = -13.0`. But `drawdown_pct` is stored as a positive fraction (0.0 to 1.0), not a negative percentage.
**Why it happens:** Original code assumed drawdown would be stored as negative percentage.
**How to avoid:** Convert: `drawdown_pct >= 0.13` means drawdown exceeds 13% threshold.
**Warning signs:** Trigger never fires despite large drawdowns.

### Pitfall 5: portfolio_snapshots Has Per-Environment Rows
**What goes wrong:** Querying `ORDER BY timestamp DESC LIMIT 1` returns the latest snapshot for ANY environment, not necessarily a "combined" value.
**Why it happens:** `portfolio_snapshots` has composite PK `(timestamp, environment)`.
**How to avoid:** For the global CB trigger, either sum across environments or query with appropriate filtering. The simplest approach: query without environment filter and take the worst drawdown.
**Warning signs:** Trigger inconsistency depending on which environment snapshot was recorded last.

## Code Examples

### Fix 1: stop_polling.py SELECT Query
```python
# Source: Direct code analysis of stop_polling.py:79-84 and db.py:326-335
# BEFORE (broken):
"SELECT symbol, side, quantity, stop_loss_price, take_profit_price "
"FROM position_tracker "
"WHERE environment = 'crypto' AND quantity > 0"

# AFTER (fixed):
"SELECT symbol, side, quantity, stop_loss_price, take_profit_price "
"FROM positions "
"WHERE environment = 'crypto' AND quantity > 0"
```

### Fix 2: Trigger 1 VIX + CB Query
```python
# Source: Direct code analysis of emergency.py:433-448, db.py:190-196, position_tracker.py:167
# VIX from DuckDB macro_features (series_id='VIXCLS', value column, date column)
with db.duckdb() as cursor:
    vix_row = cursor.execute(
        "SELECT value FROM macro_features "
        "WHERE series_id = 'VIXCLS' ORDER BY date DESC LIMIT 1"
    ).fetchone()

if vix_row is not None and vix_row[0] > _VIX_TRIGGER:
    with db.sqlite() as conn:
        dd_row = conn.execute(
            "SELECT drawdown_pct FROM portfolio_snapshots "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    # drawdown_pct is positive fraction (0.13 = 13%)
    # CB fires at -15% combined; trigger when within 2% (>= 13%)
    if dd_row is not None and dd_row["drawdown_pct"] >= 0.13:
        triggers.append(...)
```

### DDL: inference_outcomes Table
```sql
-- Lightweight table for NaN inference tracking
CREATE TABLE IF NOT EXISTS inference_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    environment TEXT NOT NULL,
    had_nan INTEGER NOT NULL DEFAULT 0
)
```

### DDL: api_errors Table
```sql
-- Lightweight table for API error tracking (IP bans, rate limits)
CREATE TABLE IF NOT EXISTS api_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    broker TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    endpoint TEXT,
    error_message TEXT
)
```

## State of the Art

| Old Approach (Current Broken) | Current Approach (Fix) | Impact |
|-------------------------------|------------------------|--------|
| Query `market_indicators` table | Query `macro_features` in DuckDB for VIX | VIX trigger becomes functional |
| Query `combined_drawdown` column | Query `drawdown_pct` column with corrected comparison | Drawdown trigger becomes functional |
| Query `inference_log` table | New `inference_outcomes` SQLite table | NaN trigger becomes functional |
| Query `api_status_log` table | New `api_errors` SQLite table | IP ban trigger becomes functional |
| Query `position_tracker` table | Query `positions` table | Stop polling becomes functional |

## Open Questions

1. **Who populates api_errors?**
   - What we know: Exchange adapters catch HTTP errors. BinanceSimAdapter and AlpacaAdapter have error handling paths.
   - What's unclear: The exact error handling path in BinanceSimAdapter that would catch 418 responses.
   - Recommendation: Add a simple INSERT in the exchange adapter's error handling `except` block. For paper mode (BinanceSimAdapter), this is a simulated adapter so 418s won't occur naturally -- but the table and query should still work correctly with test data. The api_errors table exists as infrastructure for when live mode is enabled.

2. **NaN tracking population point**
   - What we know: ExecutionPipeline catches NaN observations in execute_cycle. CONTEXT.md says to populate from there.
   - What's unclear: Exact location in pipeline.py where NaN detection occurs.
   - Recommendation: Check `execute_cycle` for observation NaN checks and add INSERT there.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_emergency_stop.py tests/scheduler/test_stop_polling.py -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-10 | stop_polling queries positions table with correct columns | unit | `uv run pytest tests/scheduler/test_stop_polling.py -v -x` | Exists (needs update) |
| PROD-07 | VIX+CB trigger queries real tables and returns triggers | unit | `uv run pytest tests/test_emergency_stop.py::TestAutomatedTriggers -v -x` | Exists (needs update) |
| PROD-07 | NaN inference trigger queries inference_outcomes | unit | `uv run pytest tests/test_emergency_stop.py::TestAutomatedTriggers::test_nan_inference_trigger -v -x` | Exists (needs update) |
| PROD-07 | IP ban trigger queries api_errors table | unit | `uv run pytest tests/test_emergency_stop.py::TestAutomatedTriggers::test_ip_ban_trigger -v -x` | Exists (needs update) |
| PROD-07 | Automated triggers fire execute_emergency_stop | unit | `uv run pytest tests/test_emergency_stop.py -v -x` | Exists (mock-based) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_emergency_stop.py tests/scheduler/test_stop_polling.py -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Update `tests/test_emergency_stop.py` -- current tests use MagicMock for all DB access; need to update trigger tests to reflect new query structure (DuckDB for VIX, corrected column names)
- [ ] Update `tests/scheduler/test_stop_polling.py` -- add test that verifies correct table name `positions` in query
- [ ] Add integration-style test with real SQLite DB verifying `inference_outcomes` and `api_errors` DDL + query

## Sources

### Primary (HIGH confidence)
- `src/swingrl/data/db.py` -- Authoritative DDL for all tables. `positions` table at line 326, `portfolio_snapshots` at line 352, `data_ingestion_log` at line 211 (DuckDB), `macro_features` at line 190 (DuckDB)
- `src/swingrl/scheduler/stop_polling.py` -- Bug at line 82 (`position_tracker` instead of `positions`)
- `src/swingrl/execution/emergency.py` -- Bugs at lines 434, 454, 470 (three non-existent tables)
- `src/swingrl/execution/risk/position_tracker.py` -- Shows `drawdown_pct` is stored as positive fraction (line 167)
- `.planning/v1.0-MILESTONE-AUDIT.md` -- INT-03 and INT-04 gap definitions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new libraries needed, all fixes use existing codebase patterns
- Architecture: HIGH - All data sources verified against actual DDL in db.py
- Pitfalls: HIGH - Every pitfall identified from direct code comparison between emergency.py queries and db.py DDL

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable -- internal bug fixes, no external dependencies)
