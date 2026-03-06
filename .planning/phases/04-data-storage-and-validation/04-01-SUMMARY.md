---
phase: 04-data-storage-and-validation
plan: 01
subsystem: database
tags: [duckdb, sqlite, singleton, context-manager, wal, sqlite-scanner, aggregation-views]

# Dependency graph
requires:
  - phase: 02-developer-experience
    provides: SwingRLConfig Pydantic schema, load_config(), structlog logging
provides:
  - DatabaseManager singleton with DuckDB/SQLite context managers
  - 5 DuckDB tables (ohlcv_daily, ohlcv_4h, macro_features, data_quarantine, data_ingestion_log)
  - 10 SQLite tables (trades, positions, risk_decisions, portfolio_snapshots, system_events, corporate_actions, wash_sale_tracker, circuit_breaker_events, options_positions, alert_log)
  - Weekly and monthly OHLCV aggregation views
  - Cross-DB join support via sqlite_scanner
  - init_db.py CLI script
  - SystemConfig and AlertingConfig in SwingRLConfig
affects: [04-02, 04-03, 04-04, 05-feature-engineering, 08-paper-trading, 09-automation]

# Tech tracking
tech-stack:
  added: [duckdb>=1.0]
  patterns: [singleton-with-reset, duckdb-cursor-context-manager, sqlite-wal-context-manager, cross-db-sqlite-scanner]

key-files:
  created:
    - src/swingrl/data/db.py
    - scripts/init_db.py
    - tests/data/test_db.py
  modified:
    - src/swingrl/config/schema.py
    - config/swingrl.yaml
    - config/swingrl.prod.yaml.example
    - tests/conftest.py
    - pyproject.toml

key-decisions:
  - "DuckDB connection type annotated as Any due to missing type stubs (duckdb.* in mypy ignore list)"
  - "DatabaseManager uses class-level _initialized bool for mypy compatibility with singleton __new__"

patterns-established:
  - "DatabaseManager singleton: thread-safe via threading.Lock in __new__, reset() for test isolation"
  - "DuckDB context manager: persistent connection with per-call cursor(), cursor closed on exit"
  - "SQLite context manager: new connection per call, WAL mode, row_factory=Row, auto-commit/rollback"
  - "init_schema() idempotent via CREATE TABLE/VIEW IF NOT EXISTS"

requirements-completed: [DATA-06, DATA-07, DATA-08, DATA-09]

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 4 Plan 1: Database Storage Layer Summary

**DatabaseManager singleton with DuckDB/SQLite dual-database layer, 15 tables, cross-DB joins via sqlite_scanner, and weekly/monthly aggregation views**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T23:14:43Z
- **Completed:** 2026-03-06T23:21:52Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- DatabaseManager singleton providing thread-safe DuckDB cursor and SQLite WAL context managers
- 5 DuckDB tables + 10 SQLite tables + 3 indexes + 2 aggregation views created via init_schema()
- Cross-DB join support via sqlite_scanner ATTACH for querying market data alongside trading ops
- Config schema extended with SystemConfig (DB paths) and AlertingConfig (rate limiting)
- init_db.py CLI creates both databases with integrity verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Config schema additions and DatabaseManager class** - `4b37cff` (feat)
2. **Task 2: DDL schemas, aggregation views, cross-DB join, and init_db.py** - `9fb5039` (feat)

## Files Created/Modified
- `src/swingrl/data/db.py` - DatabaseManager singleton with DuckDB/SQLite context managers, init_schema(), attach_sqlite()
- `scripts/init_db.py` - CLI script for database initialization with PRAGMA integrity_check
- `tests/data/test_db.py` - 20 tests covering singleton, context managers, schema, cross-DB join, aggregation views
- `src/swingrl/config/schema.py` - Added SystemConfig and AlertingConfig sub-models
- `config/swingrl.yaml` - Added system and alerting sections
- `config/swingrl.prod.yaml.example` - Added system and alerting sections with production paths
- `tests/conftest.py` - Updated valid_config_yaml with system and alerting sections
- `pyproject.toml` - Added duckdb>=1.0 dependency, duckdb.* to mypy ignore

## Decisions Made
- Used `Any` type annotation for DuckDB connection/cursor types since duckdb has no type stubs and is in mypy ignore list
- Class-level `_initialized: bool = False` for mypy compatibility with singleton pattern's `__new__` setting instance attributes
- DuckDB views (ohlcv_weekly, ohlcv_monthly) show as tables in SHOW TABLES output (7 items = 5 tables + 2 views)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-commit hooks caught: ruff B017 (blind Exception in pytest.raises), mypy `_initialized` type inference, bandit B110 (try/except/pass) -- all resolved before commit.
- DuckDB cursor raises `ConnectionException` (not `InvalidInputException`) when used after close -- test updated accordingly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- DatabaseManager ready for Plan 02 (Parquet-to-DuckDB data loading)
- init_schema() provides all tables needed for ingestion logging and quarantine migration
- Aggregation views ready for Phase 5 feature engineering consumption
- Config system and alerting sections ready for Plan 04 (Discord alerter)

---
*Phase: 04-data-storage-and-validation*
*Completed: 2026-03-06*
