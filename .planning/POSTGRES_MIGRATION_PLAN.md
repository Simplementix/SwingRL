# PostgreSQL Migration Plan: Replace DuckDB + Main SQLite

## Context
DuckDB's single-writer exclusive-access model blocks the dashboard and memory service during training (hours-long windows). SQLite works fine but consolidating into one database simplifies operations: one backup, one connection pool, one monitoring point, and enables cross-table JOINs between OLAP (market data) and OLTP (trades).

**Scope:** Migrate `market_data.ddb` (16 DuckDB tables) and `trading_ops.db` (13 SQLite tables) into one PostgreSQL 17 database. Memory service stays on its own SQLite (`memory.db`).

**Timeline:** After all 6 training iterations complete. This is a dedicated migration phase.

**Postgres:** `postgres:17-alpine` in a new Docker container.

---

## Phase 1: Infrastructure Setup

### 1.1 Docker Compose — Add PostgreSQL container

**File:** `docker-compose.yml`

Add `swingrl-postgres` service:
- Image: `postgres:17-alpine`
- Named volume: `postgres_data`
- Port: 5432 (internal only, no host bind needed)
- Healthcheck: `pg_isready -U swingrl`
- Environment: `POSTGRES_USER=swingrl`, `POSTGRES_PASSWORD` from `.env`, `POSTGRES_DB=swingrl`
- Resource limits: `mem_limit: 1g`, `cpus: 2.0`
- Restart: `unless-stopped`

Update `swingrl`, `swingrl-dashboard` services to `depends_on: swingrl-postgres`

### 1.2 Dependencies

**File:** `pyproject.toml`

Add: `"psycopg[binary]>=3.2"` (psycopg3 — modern, supports sync + async, connection pooling built-in)

Remove after migration verified: `"duckdb>=1.0,<2"`

### 1.3 Environment Variables

**File:** `.env.example`, `.env`

Add: `DATABASE_URL=postgresql://swingrl:changeme@swingrl-postgres:5432/swingrl`  # pragma: allowlist secret

---

## Phase 2: Schema Translation (29 tables)

### 2.1 Type Mapping

| DuckDB/SQLite Type | PostgreSQL Type |
|---------------------|-----------------|
| TEXT | TEXT |
| REAL | DOUBLE PRECISION |
| DOUBLE | DOUBLE PRECISION |
| INTEGER | INTEGER |
| BIGINT | BIGINT |
| BOOLEAN | BOOLEAN |
| DATE | DATE |
| TIMESTAMP | TIMESTAMPTZ |
| TEXT (ISO datetime) | TIMESTAMPTZ |
| INTEGER (0/1 boolean) | BOOLEAN |

### 2.2 Syntax Translation

| Pattern | DuckDB/SQLite | PostgreSQL |
|---------|---------------|------------|
| Upsert (ignore) | `INSERT OR IGNORE` | `INSERT ... ON CONFLICT DO NOTHING` |
| Upsert (replace) | `INSERT OR REPLACE` | `INSERT ... ON CONFLICT (...) DO UPDATE SET ...` |
| DataFrame insert | `INSERT INTO t SELECT * FROM df` (replacement scan) | `cursor.executemany()` or `COPY FROM` via psycopg |
| ASOF JOIN | `ASOF JOIN ... ON a.date >= b.date` | `LATERAL JOIN (SELECT ... ORDER BY date DESC LIMIT 1)` |
| FIRST/LAST agg | `FIRST(col)`, `LAST(col)` | Subquery with `ORDER BY ... LIMIT 1` or window function |
| Array param | `UNNEST(?::TEXT[])` | `= ANY(%s::text[])` with Python list |
| date_trunc | `date_trunc('week', date)` | Same (compatible) |
| ROW_NUMBER | `ROW_NUMBER() OVER(...)` | Same (compatible) |
| .fetchdf() | DuckDB-specific | `pd.read_sql(query, conn)` or `cursor.fetchall()` + DataFrame |
| read_parquet() | DuckDB function | Python: `pd.read_parquet()` + bulk insert |
| SHOW TABLES | DuckDB-specific | `SELECT tablename FROM pg_tables WHERE schemaname = 'public'` |
| PRAGMA journal_mode=WAL | SQLite-specific | Not needed (Postgres MVCC) |
| PRAGMA foreign_keys=ON | SQLite-specific | Always on in Postgres |
| datetime('now') | SQLite function | `NOW()` or `CURRENT_TIMESTAMP` |
| date('now') | SQLite function | `CURRENT_DATE` |
| julianday('now') | SQLite function | `EXTRACT(EPOCH FROM NOW())` |
| AUTOINCREMENT | SQLite | `GENERATED ALWAYS AS IDENTITY` |
| `?` placeholder | SQLite/DuckDB | `%s` (psycopg) |

### 2.3 Schema Files to Create

**New file:** `src/swingrl/data/postgres_schema.py`
- All 29 tables as CREATE TABLE IF NOT EXISTS statements
- Indexes (existing + new for common query patterns)
- Migration function: `init_postgres_schema(conn)`

### 2.4 Tables by Group (all 29)

**From DuckDB (16):**
- `ohlcv_daily` (symbol, date) PK
- `ohlcv_4h` (symbol, datetime) PK
- `macro_features` (date, series_id) PK
- `features_equity` (symbol, date) PK
- `features_crypto` (symbol, datetime) PK
- `fundamentals` (symbol, date) PK
- `hmm_state_history` (environment, date) PK
- `backtest_results` (result_id) PK
- `iteration_results` (result_id) PK + UNIQUE(iteration_number, environment, run_type)
- `model_metadata` (model_id) PK
- `training_epochs` (id IDENTITY) PK
- `meta_decisions` (id IDENTITY) PK
- `reward_adjustments` (id IDENTITY) PK
- `data_ingestion_log` (run_id) PK
- `data_quarantine` (no PK, add IDENTITY)
- 2 views: `ohlcv_weekly`, `ohlcv_monthly`

**From SQLite (13):**
- `trades` (trade_id) PK
- `positions` (symbol, environment) PK
- `portfolio_snapshots` (timestamp, environment) PK
- `circuit_breaker_events` (event_id) PK
- `risk_decisions` (decision_id) PK
- `system_events` (event_id) PK
- `corporate_actions` (action_id) PK
- `wash_sale_tracker` (symbol, sale_date) PK
- `shadow_trades` (trade_id) PK
- `alert_log` (alert_id) PK
- `options_positions` (spread_id) PK
- `inference_outcomes` (id IDENTITY) PK
- `api_errors` (id IDENTITY) PK

---

## Phase 3: DatabaseManager Rewrite

### 3.1 Connection Pool

**File:** `src/swingrl/data/db.py` — Major rewrite

Replace dual-DB architecture with single PostgreSQL connection pool:

```python
# New pattern
from psycopg_pool import ConnectionPool

class DatabaseManager:
    _pool: ConnectionPool | None = None

    def connection(self) -> Generator[psycopg.Connection, None, None]:
        """Yield a connection from the pool. Auto-commits on success."""
        with self._pool.connection() as conn:
            yield conn

    # Remove: duckdb() context manager
    # Remove: sqlite() context manager
    # Remove: attach_sqlite()
    # Remove: _init_duckdb_schema()
    # Remove: _init_sqlite_schema()
    # Add: init_schema() — runs postgres_schema.py DDL
```

### 3.2 Backward Compatibility During Migration

**Temporary approach:** Keep `db.duckdb()` and `db.sqlite()` as deprecated aliases that log warnings, pointing to `db.connection()`. This allows incremental file-by-file migration instead of a big-bang rewrite.

### 3.3 Files to Modify (all callers of db.duckdb() and db.sqlite())

**DuckDB callers (replace `db.duckdb()` → `db.connection()`):**
- `src/swingrl/data/base.py` — 4 call sites (sync, log, quarantine, parquet)
- `src/swingrl/data/ingest_all.py` — 2 call sites (count, features)
- `src/swingrl/data/gap_fill.py` — 3 call sites (detect equity, detect crypto, store)
- `src/swingrl/data/verification.py` — 1 call site
- `src/swingrl/data/cross_source.py` — 1 call site
- `src/swingrl/training/data_loader.py` — docstring only (callers pass conn)
- `src/swingrl/features/pipeline.py` — called with passed conn
- `src/swingrl/features/fundamentals.py` — 1 call site
- `src/swingrl/features/hmm_regime.py` — 1 call site (via pipeline)
- `src/swingrl/features/macro.py` — called with passed conn
- `src/swingrl/agents/backtest.py` — 1 call site
- `src/swingrl/execution/pipeline.py` — 2 call sites (ensemble weights, turbulence)
- `src/swingrl/execution/emergency.py` — 1 call site
- `src/swingrl/shadow/shadow_runner.py` — 1 call site
- `src/swingrl/backup/duckdb_backup.py` — 2 call sites (rewrite for pg_dump)

**SQLite callers (replace `db.sqlite()` → `db.connection()`):**
- `src/swingrl/execution/fill_processor.py` — 1 call site (positions)
- `src/swingrl/execution/reconciliation.py` — 3 call sites
- `src/swingrl/execution/risk/position_tracker.py` — 6 call sites
- `src/swingrl/execution/risk/risk_manager.py` — 1 call site
- `src/swingrl/execution/risk/circuit_breaker.py` — 3 call sites
- `src/swingrl/execution/pipeline.py` — 1 call site (inference_outcomes)
- `src/swingrl/execution/adapters/binance_sim.py` — 2 call sites
- `src/swingrl/monitoring/alerter.py` — 1 call site
- `src/swingrl/monitoring/wash_sale.py` — 2 call sites
- `src/swingrl/monitoring/health.py` — 1 call site
- `src/swingrl/scheduler/halt_check.py` — 2 call sites
- `src/swingrl/shadow/shadow_runner.py` — 1 call site (shadow trades)
- `src/swingrl/data/corporate_actions.py` — 2 call sites

**Scripts (direct duckdb.connect() → use DatabaseManager or psycopg):**
- `scripts/train_pipeline.py` — 6+ direct connections
- `scripts/compute_features.py` — 1 direct connection
- `scripts/init_db.py` — via DatabaseManager
- `scripts/main.py` — via DatabaseManager
- `scripts/disaster_recovery.py` — backup/restore rewrite
- `scripts/seed_memory_from_backtest.py` — 1 direct connection
- `scripts/validate_memory.py` — 1 direct connection
- `scripts/backtest.py` — 1 direct connection
- `scripts/seed_production.py` — 1 direct connection
- `scripts/healthcheck.py` — 1 direct connection

**Dashboard:**
- `dashboard/app.py` — replace DuckDB + SQLite connections with psycopg

### 3.4 SQL Rewrites (DuckDB-specific → PostgreSQL)

**ASOF JOINs in macro.py (highest complexity):**
- `src/swingrl/features/macro.py` lines 30-84: 5 simultaneous ASOF JOINs
- Rewrite as LATERAL JOINs:
```sql
-- DuckDB:
ASOF JOIN (SELECT ... FROM macro_features WHERE series_id='VIXCLS') AS vix
  ON ohlcv.date >= vix.release_date

-- PostgreSQL:
LEFT JOIN LATERAL (
    SELECT value FROM macro_features
    WHERE series_id = 'VIXCLS' AND release_date <= ohlcv.date
    ORDER BY release_date DESC LIMIT 1
) AS vix ON true
```

**DataFrame insertion pattern (multiple files):**
- DuckDB: `INSERT INTO t SELECT * FROM pandas_df` (replacement scan)
- PostgreSQL: `cursor.executemany(INSERT ..., df.to_records())` or `cursor.copy()` for bulk

**Views (db.py lines 364-393):**
- `ohlcv_weekly`: `FIRST(open)` / `LAST(close)` → subquery with window functions
- `ohlcv_monthly`: Same treatment

---

## Phase 4: Data Migration Script

### 4.1 One-Time Migration Script

**New file:** `scripts/migrate_to_postgres.py`

Steps:
1. Connect to existing DuckDB + SQLite files
2. Connect to new PostgreSQL
3. Run `init_postgres_schema()` to create all tables
4. For each DuckDB table: `SELECT * → INSERT batch into Postgres` (chunked, 1000 rows/batch)
5. For each SQLite table: same pattern
6. Verify row counts match
7. Run integrity checks (PK uniqueness, FK constraints)

### 4.2 Data Volume Estimates

| Source | Table | Est. Rows | Migration Time |
|--------|-------|-----------|---------------|
| DuckDB | ohlcv_daily | ~20,480 | <1s |
| DuckDB | ohlcv_4h | ~37,225 | <1s |
| DuckDB | macro_features | ~10,700 | <1s |
| DuckDB | features_equity | ~20,000 | <1s |
| DuckDB | features_crypto | ~37,000 | <1s |
| DuckDB | backtest_results | ~500 | <1s |
| SQLite | trades | ~100 | <1s |
| SQLite | positions | ~10 | <1s |
| SQLite | portfolio_snapshots | ~200 | <1s |
| **Total** | **~130K rows** | **<10 seconds** |

---

## Phase 5: Test Migration

### 5.1 Test Infrastructure

**File:** `tests/conftest.py` — Update fixtures

- Replace `tmp_config` fixture to use PostgreSQL test database
- Use `pytest-postgresql` or spin up test Postgres via testcontainers
- Option: Use SQLite as test backend with compatibility layer (simpler but less faithful)
- **Recommended:** Use `psycopg` with in-memory test database or ephemeral container per test session

### 5.2 Test Updates

Every test that uses `mock_db` fixture needs to work with PostgreSQL:
- `tests/execution/conftest.py` — `mock_db` fixture creates DatabaseManager
- `tests/data/test_db.py` — directly tests schema creation
- `tests/features/test_*.py` — use in-memory DuckDB, need Postgres equivalent
- `tests/agents/test_backtest.py` — uses in-memory DuckDB for backtest_results

---

## Phase 6: Cleanup

### 6.1 Remove DuckDB

- Remove `duckdb` from `pyproject.toml` dependencies
- Remove `duckdb.*` from mypy ignore list
- Delete `src/swingrl/data/db.py` DuckDB methods
- Delete `src/swingrl/features/schema.py` (merged into postgres_schema.py)
- Delete `src/swingrl/backup/duckdb_backup.py` (replace with pg_dump wrapper)
- Remove `attach_sqlite()` hack

### 6.2 Remove SQLite (main app only)

- Remove SQLite schema from `db.py`
- Remove `sqlite3` imports from all src/ files
- Remove PRAGMA statements
- Keep SQLite in memory service (unchanged)

### 6.3 Update Config

- Remove `system.duckdb_path` and `system.sqlite_path` from schema.py
- Add `system.database_url` field
- Update `config/swingrl.yaml` and `config/swingrl.prod.yaml.example`

---

## Execution Order (Recommended Waves)

**Wave 1 — Infrastructure (no code behavior change):**
1. Add postgres container to docker-compose.yml
2. Add psycopg dependency
3. Create postgres_schema.py with all 29 tables
4. Create migration script
5. Add DATABASE_URL to .env

**Wave 2 — DatabaseManager rewrite:**
6. Rewrite DatabaseManager with connection pool
7. Keep deprecated db.duckdb() / db.sqlite() aliases during transition
8. Update init_schema() to use Postgres

**Wave 3 — Migrate DuckDB callers (OLAP):**
9. Migrate data ingestion (base.py, ingest_all.py)
10. Migrate feature pipeline (pipeline.py, macro.py, fundamentals.py, hmm_regime.py)
11. Migrate backtest/training (backtest.py, data_loader.py, trainer.py)
12. Migrate scripts (train_pipeline.py and others)
13. Rewrite ASOF JOINs in macro.py → LATERAL JOINs
14. Replace DataFrame replacement scan with executemany/COPY

**Wave 4 — Migrate SQLite callers (OLTP):**
15. Migrate execution pipeline (fill_processor, reconciliation, pipeline)
16. Migrate risk layer (position_tracker, risk_manager, circuit_breaker)
17. Migrate monitoring (alerter, wash_sale, health)
18. Migrate scheduler (halt_check, jobs)
19. Update dashboard to use psycopg

**Wave 5 — Tests & Cleanup:**
20. Update test fixtures for Postgres
21. Run full suite, fix failures
22. Remove DuckDB + SQLite dependencies and dead code
23. Update config schema
24. Update documentation

---

## Verification

1. **Data integrity:** Row counts match pre/post migration for all 29 tables
2. **Feature parity:** Training pipeline produces identical features from Postgres as from DuckDB
3. **OLTP correctness:** Paper trading cycle works end-to-end (submit order → fill → position update → portfolio snapshot → risk check)
4. **Concurrency:** Dashboard reads work during training writes (the whole point)
5. **Full test suite:** 1248+ tests pass
6. **CI:** `scripts/ci-homelab.sh` passes with Postgres container

---

## Risk Mitigation

- **Rollback:** Keep DuckDB + SQLite files intact until Postgres is verified in production for 1+ iteration
- **Dual-write period:** Optionally write to both old + new during first iteration on Postgres
- **Backup:** `pg_dump` cron job replaces DuckDB backup script
- **Performance:** At 130K total rows, Postgres is more than fast enough — no optimization needed
