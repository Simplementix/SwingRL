# Phase 4: Data Storage and Validation - Research

**Researched:** 2026-03-06
**Domain:** DuckDB analytical storage, SQLite operational storage, cross-source validation, Discord alerting
**Confidence:** HIGH

## Summary

Phase 4 builds the dual-database storage layer (DuckDB for market data analytics, SQLite for trading operations) with cross-database join capability via DuckDB's sqlite_scanner extension. It also implements cross-source validation (Alpaca vs yfinance), ingestion logging, corporate actions detection, quarantine migration from Parquet to DuckDB, and a general-purpose Discord alerter module.

The core challenge is creating a DatabaseManager singleton that safely manages connections to both databases under APScheduler's ThreadPoolExecutor(3) concurrency. DuckDB supports concurrent appends within a single process (MVCC-based), but the Python connection object is not thread-safe -- each thread must call `.cursor()` to get a thread-local handle. SQLite in WAL mode supports concurrent readers with a single writer. The alerter is a straightforward httpx POST to Discord webhook URLs with JSON embed payloads.

**Primary recommendation:** Use DuckDB's native `.cursor()` per-thread pattern for concurrent access (no external threading.Lock needed for appends), SQLite WAL mode for concurrent read/write, and httpx synchronous client with retry transport for Discord webhooks.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **DuckDB (market_data.ddb):** 5 data pipeline tables only -- ohlcv_daily, ohlcv_4h, macro_features, data_quarantine, data_ingestion_log
- **SQLite (trading_ops.db):** All 10 tables upfront per Doc 14 -- trades, positions, risk_decisions, portfolio_snapshots, system_events, corporate_actions, wash_sale_tracker, circuit_breaker_events, options_positions, alert_log
- **DatabaseManager class** as singleton with separate DuckDB/SQLite context managers, auto-install sqlite_scanner, DuckDB write lock, init_schema() method
- **Parquet remains as intermediate/archival format** -- ingestors continue writing Parquet, then auto-sync to DuckDB
- **BaseIngestor.run() modified** to: fetch -> validate -> Parquet -> auto-sync to DuckDB -> ingestion log
- **Store lowest, aggregate up** -- only daily/4H bars stored, SQL views for weekly aggregation
- **Cross-source validation** -- weekly scheduled job, Alpaca vs yfinance, $0.05 tolerance for liquid stocks
- **Corporate actions** -- Alpaca API pre-market check + >30% overnight price change heuristic, crypto 40% threshold
- **Discord alerter** at `src/swingrl/monitoring/alerter.py` -- httpx synchronous POST, single webhook URL via DISCORD_WEBHOOK_URL env var, three levels (critical/warning/info), daily digest batching, thread-safe, rate limiting
- **Config additions** -- system.duckdb_path, system.sqlite_path, alerting section
- **Schema versioning** -- additive approach (CREATE TABLE IF NOT EXISTS), no migration framework
- **init_db.py** script for manual DDL initialization
- **File-based temp DuckDB** for testing (not in-memory)

### Claude's Discretion
- Exact DatabaseManager API beyond locked decisions (helper methods, error handling patterns)
- DuckDB read_parquet bulk loading details (batch size, error handling for malformed Parquet)
- Exact Discord embed field layout and formatting
- init_db.py CLI argument design
- Test fixture data values for corporate actions and cross-source validation
- Aggregation view SQL syntax details
- DuckDB write lock implementation specifics (threading.Lock vs queue-based)

### Deferred Ideas (OUT OF SCOPE)
- Retraining pipeline CLI (scripts/retrain.py) -- Phase 7
- Emergency stop script (scripts/emergency_stop.py) -- Phase 10
- Circuit breaker reset script (scripts/reset_cb.py) -- Phase 8
- Production DB seeding procedure -- Phase 8
- APScheduler job scheduling -- Phase 9
- Stuck agent detection -- Phase 9
- Multiple Discord channels (#trades, #risk-alerts, #daily-digest) -- Phase 9
- Chart image attachments in Discord alerts -- Phase 9
- Crypto hard fork handling -- future version
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-06 | DuckDB analytical database (market_data.ddb) with tables created incrementally per milestone | DuckDB Python API, CREATE TABLE IF NOT EXISTS, read_parquet for bulk loading, cursor() for thread safety |
| DATA-07 | SQLite operational database (trading_ops.db) with tables created incrementally per milestone | sqlite3 stdlib, WAL mode, ISO-8601 text dates, INTEGER booleans |
| DATA-08 | Cross-database joins via DuckDB sqlite_scanner extension | INSTALL/LOAD sqlite, ATTACH syntax, cross-DB SELECT queries |
| DATA-09 | "Store lowest, aggregate up" strategy -- daily/4H bars only, weekly/monthly views | CREATE VIEW with GROUP BY date_trunc, first/last/max/min/sum aggregations |
| DATA-10 | Corporate action handling with corporate_actions table | yfinance download for validation, Alpaca corporate actions API, overnight spike heuristic |
| DATA-11 | Cross-source validation (Alpaca vs yfinance closing prices, weekly) | yfinance.download() for comparison data, percentage tolerance check, DataValidator Step 12 |
| DATA-12 | Data ingestion logging to data_ingestion_log table | UUID run_id, timing via time.perf_counter, per-symbol row counts, binance_weight_used tracking |
| DATA-13 | Alerter module with Discord webhook integration, level-based routing, daily digest | httpx POST with JSON embeds, color-coded sidebar, thread-safe buffer, cooldown logic |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| duckdb | >=1.0,<2 | Analytical database for market data | Native Parquet reader, columnar OLAP, sqlite_scanner extension, Python API |
| sqlite3 | stdlib | Operational database for trading ops | Zero-config, WAL mode, production-proven for low-write OLTP |
| httpx | >=0.27 | Discord webhook HTTP client | Sync + async, transport-level retries, timeout config, lightweight |
| yfinance | >=0.2.36 | Cross-source validation data | Free Yahoo Finance data, ticker.history() for closing prices |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-mock | >=3.12 | Discord webhook mocking | Test alerter without hitting Discord API |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| httpx | requests | requests already transitive dep, but httpx has built-in transport retries and is more modern |
| threading.Lock for DuckDB | DuckDB .cursor() | cursor() is the official pattern -- DuckDB handles internal MVCC for concurrent appends |

### Dependencies to Add

**IMPORTANT:** CONTEXT.md states duckdb, yfinance, and httpx are "already in pyproject.toml" but they are NOT present. These must be added:

```toml
# In [project] dependencies:
"duckdb>=1.0,<2",
"httpx>=0.27",
"yfinance>=0.2.36",

# In [dependency-groups] dev:
"pytest-mock>=3.12",
```

Also add to mypy overrides:
```toml
module = ["...", "duckdb.*", "yfinance.*", "httpx.*"]
```

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/
  data/
    db.py              # DatabaseManager singleton
    base.py            # BaseIngestor (modified for DuckDB sync + logging)
    validation.py      # DataValidator (Step 12 implemented)
    parquet_store.py   # Unchanged -- intermediate storage
    alpaca.py          # Unchanged
    binance.py         # Unchanged
    fred.py            # Unchanged
  config/
    schema.py          # New SystemConfig, AlertingConfig sections
  monitoring/
    __init__.py        # Exists
    alerter.py         # Discord alerter module (NEW)
  utils/
    exceptions.py      # Unchanged
scripts/
  init_db.py           # DB initialization script (NEW)
tests/
  data/
    test_db.py         # DatabaseManager tests (NEW)
    test_ingestion_logging.py  # Ingestion log tests (NEW)
    test_cross_source.py       # Cross-source validation tests (NEW)
    test_corporate_actions.py  # Corporate actions tests (NEW)
  monitoring/
    test_alerter.py    # Discord alerter tests (NEW)
```

### Pattern 1: DatabaseManager Singleton
**What:** Single class managing both DuckDB and SQLite connections with context managers
**When to use:** All database access throughout the application
**Example:**
```python
# Source: DuckDB official docs + project conventions
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
import structlog

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

class DatabaseManager:
    """Singleton manager for DuckDB and SQLite connections."""

    _instance: DatabaseManager | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> DatabaseManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, config: SwingRLConfig) -> None:
        if hasattr(self, "_initialized"):
            return
        self._duckdb_path = Path(config.paths.data_dir) / "db" / "market_data.ddb"
        self._sqlite_path = Path(config.paths.data_dir) / "db" / "trading_ops.db"
        self._duckdb_conn: duckdb.DuckDBPyConnection | None = None
        self._write_lock = threading.Lock()
        self._initialized = True

    @contextmanager
    def duckdb(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Thread-safe DuckDB connection via cursor()."""
        if self._duckdb_conn is None:
            self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
            self._duckdb_conn = duckdb.connect(str(self._duckdb_path))
        cursor = self._duckdb_conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    @contextmanager
    def sqlite(self) -> Generator[sqlite3.Connection, None, None]:
        """SQLite connection with WAL mode."""
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
```

### Pattern 2: DuckDB Write Lock for Non-Append Operations
**What:** threading.Lock for DuckDB writes that could conflict (UPDATE/DELETE), not needed for INSERT/append
**When to use:** Only when multiple threads might UPDATE or DELETE same rows simultaneously
**Key insight from DuckDB docs:** "Appends will never conflict, even on the same table." Since ingestion is append-only, the write lock is only needed for schema changes (CREATE TABLE) during init, not for normal operation.
```python
def init_schema(self) -> None:
    """Create all tables if not exist. Thread-safe via write lock."""
    with self._write_lock:
        with self.duckdb() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS ohlcv_daily (...)")
            # ... other tables
```

### Pattern 3: Parquet-to-DuckDB Sync via read_parquet
**What:** Use DuckDB's native read_parquet() for bulk loading from Parquet files
**When to use:** Initial migration and ongoing sync after ingestor runs
```python
# Source: DuckDB official docs
def sync_parquet_to_duckdb(
    self, conn: duckdb.DuckDBPyConnection, parquet_path: Path, table: str
) -> int:
    """Load Parquet data into DuckDB table with dedup on natural key."""
    conn.execute(f"""
        INSERT INTO {table}
        SELECT * FROM read_parquet('{parquet_path}')
        WHERE (symbol, date) NOT IN (SELECT symbol, date FROM {table})
    """)
    return conn.fetchone()[0] if conn.description else 0
```

### Pattern 4: Discord Webhook Embed
**What:** Color-coded Discord embeds via httpx POST
**When to use:** All alerting throughout the application
```python
# Source: Discord webhook API docs
import httpx

COLORS = {"critical": 0xFF0000, "warning": 0xFFA500, "info": 0x3498DB}

def _build_embed(level: str, title: str, message: str) -> dict:
    return {
        "embeds": [{
            "title": title,
            "description": message,
            "color": COLORS[level],
            "timestamp": datetime.now(UTC).isoformat(),
            "footer": {"text": f"SwingRL | {level.upper()}"},
        }]
    }

def send(webhook_url: str, payload: dict) -> None:
    response = httpx.post(webhook_url, json=payload, timeout=10.0)
    response.raise_for_status()
```

### Anti-Patterns to Avoid
- **Opening new DuckDB connections per query:** Use a single persistent connection with `.cursor()` per thread. Opening/closing connections is expensive and defeats DuckDB's buffer pool.
- **Using pandas to load Parquet into DuckDB:** DuckDB's native `read_parquet()` is faster and avoids pandas serialization overhead.
- **Using in-memory DuckDB for tests:** File-based temp databases match production behavior. In-memory databases hide file-locking and persistence issues.
- **Using discord.py bot framework:** Overkill for webhook-only usage. httpx POST with JSON is sufficient and avoids the async event loop dependency.
- **Storing aggregated weekly/monthly bars:** DuckDB aggregates 10 years of daily bars in <100ms. Views are the correct approach.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet reading in DuckDB | pandas.read_parquet + INSERT | DuckDB read_parquet() | 10x faster, no serialization, native pushdown |
| Cross-DB joins | Manual ETL between DBs | sqlite_scanner ATTACH | DuckDB handles transparently |
| HTTP retries for Discord | Manual retry loops | httpx.HTTPTransport(retries=3) | Built-in exponential backoff |
| Thread-safe singletons | Manual double-checked locking | threading.Lock in __new__ | Standard Python pattern |
| UUID generation | Custom ID schemes | uuid.uuid4() | Standard, no collisions |
| SQLite WAL mode | Manual journal config | PRAGMA journal_mode=WAL | One-line, massive concurrency improvement |

**Key insight:** DuckDB's native Parquet reader bypasses Python entirely -- the Parquet file is read directly by DuckDB's C++ engine with predicate pushdown and column pruning. Never route Parquet data through pandas when the destination is DuckDB.

## Common Pitfalls

### Pitfall 1: DuckDB Connection Not Thread-Safe
**What goes wrong:** Sharing a single DuckDB connection across threads without cursor() causes crashes or data corruption.
**Why it happens:** DuckDB's Python binding locks the connection for the duration of a query. Concurrent access deadlocks or raises internal errors.
**How to avoid:** Always call `conn.cursor()` to get a thread-local handle. The cursor shares the same underlying database but has independent state.
**Warning signs:** `RuntimeError: DuckDB objects created in a thread can only be used in that same thread` errors in logs.

### Pitfall 2: SQLite TEXT Dates Not Sorted Correctly
**What goes wrong:** ISO-8601 text dates sort lexicographically, which works for YYYY-MM-DD but breaks with timezone offsets.
**Why it happens:** SQLite has no native DATE type. TEXT comparison is byte-by-byte.
**How to avoid:** Always store dates as YYYY-MM-DD (no timezone) in SQLite. All timestamps as YYYY-MM-DDTHH:MM:SS (UTC, no Z suffix) for consistent sorting.
**Warning signs:** Query results in unexpected order, date range queries returning wrong rows.

### Pitfall 3: DuckDB sqlite_scanner Extension Not Auto-Loading
**What goes wrong:** ATTACH with TYPE sqlite fails if extension not installed.
**Why it happens:** sqlite_scanner is a loadable extension that must be installed first.
**How to avoid:** Call `INSTALL sqlite; LOAD sqlite;` during DatabaseManager initialization. DuckDB caches extensions after first install.
**Warning signs:** `Catalog Error: Type with name "sqlite" does not exist` error.

### Pitfall 4: Discord Webhook Rate Limiting
**What goes wrong:** Discord returns 429 Too Many Requests when sending alerts too fast.
**Why it happens:** Discord webhooks are rate-limited to ~30 requests per minute per webhook.
**How to avoid:** Implement alert cooldown (30 min per CONTEXT.md), batch info-level alerts into daily digest, deduplicate via alert_log table message_hash.
**Warning signs:** httpx.HTTPStatusError with status_code=429 in logs.

### Pitfall 5: Parquet Schema Mismatch During DuckDB Load
**What goes wrong:** read_parquet fails when Parquet file has different column names/types than DuckDB table.
**Why it happens:** Phase 3 Parquet files may not have the exact same schema as DuckDB tables (e.g., missing adjusted_close, different column names).
**How to avoid:** Use explicit column mapping in the INSERT ... SELECT statement. Validate Parquet schema before loading.
**Warning signs:** `Binder Error: Column count mismatch` or `Conversion Error` in logs.

### Pitfall 6: Singleton State Leaking Between Tests
**What goes wrong:** DatabaseManager singleton persists across tests, using stale connections or wrong paths.
**Why it happens:** Singleton pattern retains instance across test functions.
**How to avoid:** Add a `reset()` classmethod that clears `_instance`. Call it in test fixtures (function-scoped).
**Warning signs:** Tests pass individually but fail when run together.

### Pitfall 7: yfinance Adjusted Close vs Unadjusted
**What goes wrong:** Cross-source comparison fails because yfinance returns adjusted prices by default while Alpaca may return unadjusted.
**Why it happens:** yfinance's `auto_adjust=True` (default since v0.2.x) modifies OHLC columns.
**How to avoid:** Use `auto_adjust=False` in yfinance and compare the `Adj Close` column against Alpaca's adjusted_close. Or compare with tolerance.
**Warning signs:** Systematic bias in cross-source validation (always one direction).

## Code Examples

### DuckDB Table Creation (ohlcv_daily)
```sql
-- Source: CONTEXT.md DDL + DuckDB docs
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol      TEXT NOT NULL,
    date        DATE NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    volume      BIGINT NOT NULL,
    adjusted_close DOUBLE,
    fetched_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);
```

### DuckDB Table Creation (data_ingestion_log)
```sql
CREATE TABLE IF NOT EXISTS data_ingestion_log (
    run_id          TEXT PRIMARY KEY,  -- UUID as TEXT
    timestamp       TIMESTAMP DEFAULT current_timestamp,
    environment     TEXT NOT NULL,     -- 'equity' | 'crypto' | 'macro'
    symbol          TEXT NOT NULL,
    status          TEXT NOT NULL,     -- 'success' | 'partial' | 'failed'
    rows_inserted   INTEGER DEFAULT 0,
    errors_count    INTEGER DEFAULT 0,
    duration_ms     INTEGER,
    binance_weight_used INTEGER       -- NULL for non-Binance sources
);
```

### SQLite Table Creation (circuit_breaker_events)
```sql
-- Source: CONTEXT.md DDL
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    event_id     TEXT PRIMARY KEY,
    environment  TEXT NOT NULL,
    triggered_at TEXT NOT NULL,        -- ISO-8601
    resumed_at   TEXT,                 -- NULL while active
    trigger_value REAL NOT NULL,
    threshold    REAL NOT NULL,
    reason       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cb_env_resumed
    ON circuit_breaker_events (environment, resumed_at);
```

### Cross-Database Join via sqlite_scanner
```python
# Source: DuckDB sqlite_scanner docs
def attach_sqlite(self, conn: duckdb.DuckDBPyConnection) -> None:
    """Attach SQLite database for cross-DB queries."""
    conn.execute(f"ATTACH '{self._sqlite_path}' AS ops (TYPE sqlite)")

# Usage: join market data with trading ops
with db.duckdb() as conn:
    db.attach_sqlite(conn)
    result = conn.execute("""
        SELECT m.symbol, m.date, m.close, t.side, t.quantity
        FROM ohlcv_daily m
        JOIN ops.trades t ON m.symbol = t.symbol
            AND m.date = CAST(t.timestamp AS DATE)
        WHERE m.symbol = 'SPY'
    """).fetchdf()
```

### Weekly Aggregation View (DATA-09)
```sql
-- Source: DuckDB SQL syntax
CREATE VIEW IF NOT EXISTS ohlcv_weekly AS
SELECT
    symbol,
    date_trunc('week', date) AS week_start,
    FIRST(open ORDER BY date)    AS open,
    MAX(high)                    AS high,
    MIN(low)                     AS low,
    LAST(close ORDER BY date)    AS close,
    SUM(volume)                  AS volume,
    LAST(adjusted_close ORDER BY date) AS adjusted_close
FROM ohlcv_daily
GROUP BY symbol, date_trunc('week', date);
```

### Parquet Bulk Load
```python
# Source: DuckDB official docs
def load_parquet(self, conn: duckdb.DuckDBPyConnection, path: Path, table: str) -> int:
    """Bulk load Parquet into DuckDB table, skipping existing rows."""
    result = conn.execute(f"""
        INSERT OR IGNORE INTO {table}
        SELECT * FROM read_parquet('{path}')
    """)
    count = result.fetchone()
    return count[0] if count else 0
```

### Discord Alerter Thread-Safe Buffer
```python
# Source: Project design decisions
import threading
from collections import deque
from datetime import datetime, UTC

class Alerter:
    def __init__(self, webhook_url: str, cooldown_minutes: int = 30) -> None:
        self._webhook_url = webhook_url
        self._cooldown_minutes = cooldown_minutes
        self._info_buffer: deque[dict] = deque()
        self._lock = threading.Lock()
        self._last_alert_times: dict[str, datetime] = {}

    def send_alert(self, level: str, title: str, message: str) -> None:
        """Send alert with level-based routing and cooldown."""
        if level == "info":
            with self._lock:
                self._info_buffer.append({"title": title, "message": message, "ts": datetime.now(UTC)})
            return

        # Check cooldown for critical/warning
        key = f"{level}:{title}"
        now = datetime.now(UTC)
        with self._lock:
            last = self._last_alert_times.get(key)
            if last and (now - last).total_seconds() < self._cooldown_minutes * 60:
                return
            self._last_alert_times[key] = now

        self._post_webhook(level, title, message)
```

### Modified BaseIngestor.run() with Logging
```python
def run(self, symbol: str, since: str | None = None) -> None:
    """Orchestrate fetch -> validate -> Parquet -> DuckDB sync -> log."""
    import time
    import uuid

    run_id = str(uuid.uuid4())
    start = time.perf_counter()
    rows_inserted = 0
    errors_count = 0
    status = "success"

    try:
        raw = self.fetch(symbol, since)
        if raw.empty:
            status = "no_data"
            return
        clean, quarantine = self.validate(raw, symbol)
        errors_count = len(quarantine)
        if not quarantine.empty:
            self._store_quarantine(quarantine, symbol)
        if not clean.empty:
            parquet_path = self.store(clean, symbol)
            rows_inserted = self._sync_to_duckdb(clean, symbol)
    except Exception as e:
        status = "failed"
        log.error("ingestor_run_failed", symbol=symbol, error=str(e))
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        self._log_ingestion(run_id, symbol, status, rows_inserted, errors_count, duration_ms)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pandas.read_parquet + INSERT | DuckDB read_parquet() native | DuckDB 0.8+ (2023) | 5-10x faster bulk loads |
| sqlite_scanner as separate install | Bundled core extension | DuckDB 0.9+ (2023) | Auto-install via INSTALL sqlite |
| Single-threaded DuckDB | MVCC concurrent appends | DuckDB 0.8+ (2023) | Safe multi-thread INSERTs without locking |
| discord.py bot framework | httpx direct webhook POST | - | No async event loop, lighter dependency |
| requests library | httpx | - | Transport-level retries, modern API |

**Deprecated/outdated:**
- `sqlite_scanner` extension name: Still works but the canonical name is now `sqlite`. Both resolve to the same extension.
- DuckDB `INSERT OR IGNORE`: DuckDB uses `INSERT OR IGNORE INTO` syntax (similar to SQLite), but for tables with PRIMARY KEY constraints only.

## Open Questions

1. **DuckDB INSERT OR IGNORE behavior**
   - What we know: DuckDB supports PRIMARY KEY constraints and `INSERT OR IGNORE INTO` syntax
   - What's unclear: Whether INSERT OR IGNORE works efficiently with composite primary keys (symbol, date) for dedup during Parquet loads
   - Recommendation: Test during implementation. If performance is poor, use NOT EXISTS subquery pattern instead.

2. **yfinance auto_adjust default behavior**
   - What we know: yfinance changed default `auto_adjust` behavior across versions
   - What's unclear: Exact current default in yfinance >=0.2.36
   - Recommendation: Always pass `auto_adjust=False` explicitly and use `Adj Close` column for comparison. This eliminates version-dependent behavior.

3. **Alpaca adjusted_close availability in Phase 3 Parquet files**
   - What we know: CONTEXT.md mentions Phase 3 AlpacaIngestor needs modification to capture adjusted_close
   - What's unclear: Whether existing Phase 3 Parquet files already have this column
   - Recommendation: Check during implementation. If missing, the Phase 4 migration handles adding adjusted_close as NULL and backfilling via re-fetch.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 (installed) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/data/test_db.py tests/monitoring/test_alerter.py -x -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-06 | DuckDB tables created and queryable | unit | `uv run pytest tests/data/test_db.py::test_duckdb_schema_creation -x` | Wave 0 |
| DATA-07 | SQLite tables created and queryable | unit | `uv run pytest tests/data/test_db.py::test_sqlite_schema_creation -x` | Wave 0 |
| DATA-08 | Cross-DB join returns correct results | integration | `uv run pytest tests/data/test_db.py::test_cross_db_join -x` | Wave 0 |
| DATA-09 | Weekly aggregation view returns correct OHLCV | unit | `uv run pytest tests/data/test_db.py::test_weekly_aggregation_view -x` | Wave 0 |
| DATA-10 | Corporate action detection (split heuristic) | unit | `uv run pytest tests/data/test_corporate_actions.py -x` | Wave 0 |
| DATA-11 | Cross-source validation within 0.1% tolerance | unit | `uv run pytest tests/data/test_cross_source.py -x` | Wave 0 |
| DATA-12 | Ingestion log row created with correct fields | unit | `uv run pytest tests/data/test_ingestion_logging.py -x` | Wave 0 |
| DATA-13 | Discord alerter sends correct embed payload | unit | `uv run pytest tests/monitoring/test_alerter.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/data/ tests/monitoring/ -x -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/data/test_db.py` -- covers DATA-06, DATA-07, DATA-08, DATA-09
- [ ] `tests/data/test_ingestion_logging.py` -- covers DATA-12
- [ ] `tests/data/test_cross_source.py` -- covers DATA-11
- [ ] `tests/data/test_corporate_actions.py` -- covers DATA-10
- [ ] `tests/monitoring/test_alerter.py` -- covers DATA-13
- [ ] `tests/monitoring/__init__.py` -- package init for monitoring tests
- [ ] Dependencies: `duckdb>=1.0,<2`, `httpx>=0.27`, `yfinance>=0.2.36`, `pytest-mock>=3.12`

## Sources

### Primary (HIGH confidence)
- [DuckDB Python API overview](https://duckdb.org/docs/stable/clients/python/overview) -- connection management, context managers, cursor()
- [DuckDB Concurrency](https://duckdb.org/docs/stable/connect/concurrency) -- MVCC, single-writer, append safety
- [DuckDB Multiple Python Threads](https://duckdb.org/docs/stable/guides/python/multiple_threads) -- cursor() per thread pattern with full code example
- [DuckDB SQLite Extension](https://duckdb.org/docs/stable/core_extensions/sqlite) -- ATTACH syntax, cross-DB joins, type caveats
- [DuckDB Parquet](https://duckdb.org/docs/stable/data/parquet/overview) -- read_parquet() function, pushdown optimization
- [Discord Webhook API](https://gist.github.com/Bilka2/5dd2ca2b6e9f3573e0c2defe5d3031b2) -- embed JSON structure, color codes, fields

### Secondary (MEDIUM confidence)
- [httpx Documentation](https://www.python-httpx.org/) -- synchronous client, timeout, transport retries
- [yfinance PyPI](https://pypi.org/project/yfinance/) -- download API, auto_adjust parameter
- [DuckDB PyPI](https://pypi.org/project/duckdb/) -- latest version 1.4.4 (Jan 2026)

### Tertiary (LOW confidence)
- DuckDB INSERT OR IGNORE with composite keys -- needs implementation-time verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- DuckDB, SQLite, httpx all well-documented with official sources verified
- Architecture: HIGH -- DatabaseManager patterns verified against DuckDB threading docs, singleton is standard Python
- Pitfalls: HIGH -- Thread safety, schema mismatch, rate limiting all documented in official sources
- Cross-source validation: MEDIUM -- yfinance auto_adjust behavior varies by version, needs explicit parameter
- Corporate actions: MEDIUM -- Alpaca corporate actions API not deeply verified, heuristic approach is straightforward

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (30 days -- stable technologies)
