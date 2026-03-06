# Phase 3: Data Ingestion - Context

**Gathered:** 2026-03-06
**Updated:** 2026-03-06 (gaps & gray areas pass)
**Status:** Ready for planning

<domain>
## Phase Boundary

Raw OHLCV bars and macro indicators flowing reliably from Alpaca (8 equity ETFs, daily), Binance.US (BTC/ETH, 4H), and FRED (5 Tier 1 macro series) into Parquet files. Includes 8+ year crypto historical backfill stitching international Binance archives with Binance.US API data, and a 12-step data validation checklist with row-level quarantine. Storage is Phase 4; this phase delivers ingestion and intermediate Parquet output.

</domain>

<decisions>
## Implementation Decisions

### Ingestor architecture
- Abstract base class (BaseIngestor) with fetch(), validate(), store() contract
- Three concrete implementations: AlpacaIngestor, BinanceIngestor, FREDIngestor
- Each ingestor callable as CLI (python -m swingrl.data.alpaca) AND importable as library
- CLI supports manual backfills with flags (--symbols, --days, --backfill); library API for Phase 9 automation
- Both full-fetch and incremental (since-last-timestamp) modes from day one
- API credentials via environment variables only (ALPACA_API_KEY, ALPACA_SECRET_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, FRED_API_KEY) — never in YAML config

### Crypto backfill strategy
- Source: Binance public data archives (data.binance.vision) for 2017-2019, Binance.US API for 2019+
- 30-day overlap validation window at the stitch point — fetch both sources, validate prices match within <0.5% deviation, log discrepancies
- Use Binance.US data from the overlap period onward
- Volume normalization: use quote_asset_volume field (already in USDT) where available, fall back to volume_base * close_price where not
- Backfill integrated into BinanceIngestor as .backfill(start_date, end_date) method, invoked via CLI --backfill flag

### Validation & quarantine
- Row-level quarantine — bad rows go to quarantine, good rows proceed (one bad bar doesn't block 7 good ETFs)
- Mix of row-level and batch-level checks:
  - Row-level: null checks, price > 0, volume >= 0, OHLC ordering (H >= L, O/C between H/L)
  - Batch-level: gap detection, duplicate detection, stale data check, row count threshold
- Quarantine storage: Parquet files in data/quarantine/{source}_{date}.parquet (Phase 4 migrates to DB table)
- Alerting: structlog warnings per failure + summary at end of run ("3 rows quarantined: 2 missing volume, 1 price spike"). Discord alerting deferred to Phase 4 (DATA-13)

### Output format & storage
- Parquet files organized as:
  - data/equity/{SYMBOL}_daily.parquet (e.g., SPY_daily.parquet)
  - data/crypto/{SYMBOL}_4h.parquet (e.g., BTCUSDT_4h.parquet)
  - data/macro/{SERIES}.parquet (e.g., VIX.parquet, T10Y2Y.parquet)
  - data/quarantine/{source}_{date}.parquet
- Single file per symbol/series (no year partitioning — ~17.5K rows for 8yr crypto, ~2.5K for 10yr equity)
- FRED Parquet files include vintage_date alongside observation_date (ALFRED vintage data for look-ahead bias prevention)
- Equity backfill: 10 years (2016-present) — covers 2018 correction, 2020 COVID crash, 2022 bear market
- Phase 4 reads these Parquets directly into DuckDB tables

### FRED macro pipeline
- 5 Tier 1 series: VIXCLS (VIX via FRED), T10Y2Y, DFF, CPIAUCSL (CPI), UNRATE (unemployment)
- VIX sourced from FRED VIXCLS series — keeps all macro data on one API, avoids adding a 4th data source
- Store each series at native frequency (VIX/T10Y2Y/DFF = daily, CPI/UNRATE = monthly); Phase 5 ASOF JOIN handles alignment
- ALFRED vintage data back to 2016 (matches 10-year equity training window) — prevents look-ahead bias from revised CPI/unemployment
- Refresh cadence: daily for daily series, weekly check for monthly series (CPI/UNRATE update monthly — daily fetch is wasteful)

### Error handling & retry policy
- 3 retries with exponential backoff (2s, 4s, 8s) on API failures (timeout, rate limit, server error)
- After 3 failures: log error with structlog, move to next symbol/series — partial success is better than total failure
- Run exits with non-zero code if ANY symbol/series failed (so CI and Phase 9 scheduler can detect failures)
- Binance.US rate limits: proactive throttling via X-MBX-USED-WEIGHT header — if >80% of 1200/min limit, sleep until window resets (prevents 429s during backfill)
- All API errors raise DataError with context (source, symbol, HTTP status, attempt count) — no new exception subclasses needed

### Incremental update semantics
- Upsert on matching timestamps — if a bar for the same symbol+timestamp exists, replace it (handles corrected/partial bars)
- "Last fetched" timestamp derived from max(timestamp) in existing Parquet file — the data IS the state, no separate tracking file
- Source-aware gap detection:
  - Equity: NYSE trading calendar via exchange_calendars library — only flag gaps on actual trading days (not weekends/holidays)
  - Crypto: continuous 4H bars expected 24/7 — any gap is a real gap
  - FRED: each series has its own cadence (daily or monthly) — gap detection per-series
- exchange_calendars dependency added for proper NYSE holiday handling (~9 holidays/year)

### Testing strategy
- Unit tests: mocked HTTP responses (responses or pytest-httpx library) with fixture JSON/CSV payloads per source
- Integration tests: marked with @pytest.mark.integration, skipped in CI (pytest -m 'not integration'), require real API keys — for manual validation of real API connectivity
- Backfill tests: small fixture CSV (~100 rows) mimicking Binance archive format — tests parsing, volume normalization, stitch validation without real download
- Validation test fixtures: one fixture per defect type (null price, negative volume, OHLC ordering violation, duplicate timestamp, price spike >50% jump, stale data, gap, zero volume) — each triggers a specific validation check

### Claude's Discretion
- Exact 12-step validation checklist mapping (specific checks beyond the ones enumerated above)
- Alpaca IEX feed pagination approach
- Parquet compression and detailed column schema (types, metadata)
- Specific mocking library choice (responses vs pytest-httpx)
- exchange_calendars integration details
- CLI argument parser implementation (argparse vs click)

</decisions>

<specifics>
## Specific Ideas

- Binance public archive CSVs from data.binance.vision are well-documented and deterministic — good for reproducible backfills
- Quote asset volume field preferred over close*volume for accuracy
- 30-day overlap at stitch point gives confidence in data continuity
- 10-year equity history ensures training sees multiple market regimes (essential for robust walk-forward validation)
- FRED VIXCLS avoids adding Yahoo Finance as a 4th data source just for VIX
- Proactive Binance throttling critical for backfill (many sequential requests)
- One-fixture-per-defect testing pattern makes it obvious which validation check caught which problem

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SwingRLConfig` (schema.py): Has equity.symbols, crypto.symbols, paths.data_dir — ingestors read these
- `DataError` (exceptions.py): Ready for pipeline error raising with rich context
- structlog (logging.py): configure_logging() established — ingestors use `structlog.get_logger(__name__)`
- `src/swingrl/data/`: Package scaffolded with __init__.py — ingestor modules go here

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- pathlib.Path for all file operations
- Absolute imports only (from swingrl.data.alpaca import AlpacaIngestor)
- Pydantic config access via load_config() — never raw YAML
- TDD: failing test first, then implementation
- Test naming: `test_<what>` with docstring referencing requirement ID

### Integration Points
- Config: load_config() provides symbols, data_dir path, and all settings
- Credentials: Docker env_file (.env) provides API keys as environment variables
- Output: data/ directory (already in repo structure with .gitkeep)
- Phase 4 reads Parquet files from data/ into DuckDB/SQLite tables
- Phase 5 reads FRED Parquet files and applies ASOF JOIN for forward-fill alignment
- Phase 9 scheduler calls ingestor library API for automated runs

### New Dependencies (Phase 3)
- exchange_calendars: NYSE trading calendar for source-aware gap detection
- HTTP mocking library (responses or pytest-httpx): for unit testing API code
- pyarrow: Parquet read/write (likely already transitive via pandas)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-data-ingestion*
*Context gathered: 2026-03-06*
*Updated: 2026-03-06 (gaps & gray areas)*
