# Phase 18: Data Ingestion - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Populate homelab DuckDB with maximum available historical OHLCV (equity + crypto), FRED macro data, and computed feature vectors — all observation vector dimensions non-NaN and ready for training. Runs entirely inside the Docker container on the homelab. Training, deployment, and alerting are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Historical Depth
- Ingest maximum available history from all sources — don't artificially limit
- Equity: pull everything Alpaca free tier provides (typically 5-7 years) for all 8 ETFs
- Crypto: maximum depth with pre-2019 Binance global archive stitching (back to 2017) + Binance.US API (2019+)
- FRED macro: match equity OHLCV date range — pull aligned, trim edges where no overlap
- Stay on Alpaca free tier — no paid upgrade. Work with whatever depth is available

### Ingestion Orchestration
- Single Python CLI module: `swingrl.data.ingest_all` with argparse
- Runs the full pipeline in order: equity OHLCV → crypto OHLCV → macro → feature computation → verification
- Fail fast on errors — stop immediately, log which source/symbol failed, non-zero exit
- Existing CLIs support incremental mode so re-runs pick up where they left off
- Feature pipeline runs automatically as the final step (no separate command)

### NaN/Gap Resolution
- Trim to shared date window across all sources — only use dates where ALL sources have data
- Weekend/holiday gaps in equity OHLCV are expected and silently skipped (not flagged)
- Crypto 4H gaps ≤2 bars (8 hours): interpolate (forward-fill). Gaps >2 bars: flag as errors
- Trim warmup rows automatically for indicators needing lookback (e.g., 200 bars for SMA-200)

### Verification Workflow
- Automatic verification suite runs after feature computation
- Output: console summary (pass/fail per check) AND detailed JSON report to `data/verification.json` (fixed path, overwritten each run)
- Quality gate: zero NaN in all observation vectors + no unexpected date gaps
- Checks include: row counts per symbol, date range coverage, NaN counts per dimension, observation vector shape (156-dim equity, 45-dim crypto)
- Non-zero exit code on any verification failure

### Docker Execution Model
- Ingestion runs inside the Docker container (consistent with production env, satisfies DATA-05)
- Operator triggers via: `docker exec swingrl python -m swingrl.data.ingest_all --backfill`
- DuckDB lives on a bind-mounted volume — persists across container restarts

### Idempotency & Re-runs
- Incremental by default — check DuckDB for latest date per symbol, only fetch new data
- Feature recomputation only runs if new OHLCV rows were added (skip on no-op re-runs)
- No --force flag needed for initial implementation (can add later if needed)

### Rate Limiting
- Rely on existing CLI retry/backoff logic in alpaca.py, binance.py (via swingrl.utils.retry)
- No wrapper-level rate coordination — each ingestor handles its own API limits

### Credential Handling
- API keys (Alpaca, Binance.US, FRED) provided via Docker .env file
- Container reads credentials from environment variables with SWINGRL_ prefix
- Consistent with DEPLOY-02 requirement — same .env file used for trading

### Claude's Discretion
- Exact implementation of the ingest_all module internals
- How to detect "new rows added" for conditional feature recomputation
- Interpolation method for small crypto gaps (forward-fill vs linear)
- Console summary formatting (table style, colors)

</decisions>

<specifics>
## Specific Ideas

- Operator experience: one command does everything — `docker exec swingrl python -m swingrl.data.ingest_all --backfill`
- Verification report at `data/verification.json` should be machine-readable for potential CI integration later
- Fail fast philosophy — don't continue with partial data, make the operator fix and re-run

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/swingrl/data/alpaca.py`: CLI ingestor with argparse, incremental fetch, 10-year backfill mode
- `src/swingrl/data/binance.py`: CLI ingestor with archive stitching (pre-2019), backfill method
- `src/swingrl/data/fred.py`: CLI ingestor for 5 macro series, --backfill flag
- `src/swingrl/data/db.py`: DatabaseManager singleton with DuckDB context manager
- `src/swingrl/data/validation.py`: 12-step data validation with quarantine
- `src/swingrl/features/pipeline.py`: FeaturePipeline orchestrator (compute_equity, compute_crypto)
- `src/swingrl/utils/retry.py`: Retry/backoff decorator used by all ingestors

### Established Patterns
- All data CLIs use argparse with `--backfill` and `--since` flags
- Config loaded via `load_config()` from `SwingRLConfig`
- Logging via structlog with keyword args
- Typed exceptions: `DataError` for data issues, `BrokerError` for API failures

### Integration Points
- New `ingest_all` module imports and calls existing ingestor classes directly (not subprocess)
- DatabaseManager provides DuckDB cursor for feature pipeline
- Config schema already has broker credentials and FRED API key fields
- Docker compose bind-mounts `db/` directory for DuckDB persistence

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-data-ingestion*
*Context gathered: 2026-03-11*
