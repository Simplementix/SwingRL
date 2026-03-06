# Phase 3: Data Ingestion - Context

**Gathered:** 2026-03-06
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

### Claude's Discretion
- Exact 12-step validation checklist mapping (which checks are row-level vs batch-level)
- Rate limit handling strategy for Binance.US (X-MBX-USED-WEIGHT header monitoring)
- Alpaca IEX feed pagination and rate limiting approach
- FRED API retry logic and backoff strategy
- Parquet compression and schema details (column types, metadata)
- Incremental fetch timestamp tracking mechanism (file-based vs config-based)

</decisions>

<specifics>
## Specific Ideas

- Binance public archive CSVs from data.binance.vision are well-documented and deterministic — good for reproducible backfills
- Quote asset volume field preferred over close*volume for accuracy
- 30-day overlap at stitch point gives confidence in data continuity
- 10-year equity history ensures training sees multiple market regimes (essential for robust walk-forward validation)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SwingRLConfig` (schema.py): Has equity.symbols, crypto.symbols, paths.data_dir — ingestors read these
- `DataError` (exceptions.py): Ready for pipeline error raising
- structlog (logging.py): configure_logging() established — ingestors use `structlog.get_logger(__name__)`
- `src/swingrl/data/`: Package scaffolded with __init__.py — ingestor modules go here

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- pathlib.Path for all file operations
- Absolute imports only (from swingrl.data.alpaca import AlpacaIngestor)
- Pydantic config access via load_config() — never raw YAML
- TDD: failing test first, then implementation

### Integration Points
- Config: load_config() provides symbols, data_dir path, and all settings
- Credentials: Docker env_file (.env) provides API keys as environment variables
- Output: data/ directory (already in repo structure with .gitkeep)
- Phase 4 reads Parquet files from data/ into DuckDB/SQLite tables
- Phase 9 scheduler calls ingestor library API for automated runs

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-data-ingestion*
*Context gathered: 2026-03-06*
