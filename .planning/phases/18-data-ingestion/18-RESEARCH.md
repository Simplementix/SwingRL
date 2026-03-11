# Phase 18: Data Ingestion - Research

**Researched:** 2026-03-11
**Domain:** Python data pipeline orchestration — DuckDB, Alpaca, Binance.US, FRED, feature pipeline wiring
**Confidence:** HIGH (primary evidence is the existing codebase — no external unknowns)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Historical Depth**
- Ingest maximum available history from all sources — don't artificially limit
- Equity: pull everything Alpaca free tier provides (typically 5-7 years) for all 8 ETFs
- Crypto: maximum depth with pre-2019 Binance global archive stitching (back to 2017) + Binance.US API (2019+)
- FRED macro: match equity OHLCV date range — pull aligned, trim edges where no overlap
- Stay on Alpaca free tier — no paid upgrade. Work with whatever depth is available

**Ingestion Orchestration**
- Single Python CLI module: `swingrl.data.ingest_all` with argparse
- Runs the full pipeline in order: equity OHLCV → crypto OHLCV → macro → feature computation → verification
- Fail fast on errors — stop immediately, log which source/symbol failed, non-zero exit
- Existing CLIs support incremental mode so re-runs pick up where they left off
- Feature pipeline runs automatically as the final step (no separate command)

**NaN/Gap Resolution**
- Trim to shared date window across all sources — only use dates where ALL sources have data
- Weekend/holiday gaps in equity OHLCV are expected and silently skipped (not flagged)
- Crypto 4H gaps ≤2 bars (8 hours): interpolate (forward-fill). Gaps >2 bars: flag as errors
- Trim warmup rows automatically for indicators needing lookback (e.g., 200 bars for SMA-200)

**Verification Workflow**
- Automatic verification suite runs after feature computation
- Output: console summary (pass/fail per check) AND detailed JSON report to `data/verification.json` (fixed path, overwritten each run)
- Quality gate: zero NaN in all observation vectors + no unexpected date gaps
- Checks include: row counts per symbol, date range coverage, NaN counts per dimension, observation vector shape (156-dim equity, 45-dim crypto)
- Non-zero exit code on any verification failure

**Docker Execution Model**
- Ingestion runs inside the Docker container (consistent with production env, satisfies DATA-05)
- Operator triggers via: `docker exec swingrl python -m swingrl.data.ingest_all --backfill`
- DuckDB lives on a bind-mounted volume — persists across container restarts

**Idempotency & Re-runs**
- Incremental by default — check DuckDB for latest date per symbol, only fetch new data
- Feature recomputation only runs if new OHLCV rows were added (skip on no-op re-runs)
- No --force flag needed for initial implementation

**Rate Limiting**
- Rely on existing CLI retry/backoff logic in alpaca.py, binance.py (via swingrl.utils.retry)
- No wrapper-level rate coordination — each ingestor handles its own API limits

**Credential Handling**
- API keys (Alpaca, Binance.US, FRED) provided via Docker .env file
- Container reads credentials from environment variables with SWINGRL_ prefix
- Consistent with DEPLOY-02 requirement — same .env file used for trading

### Claude's Discretion
- Exact implementation of the ingest_all module internals
- How to detect "new rows added" for conditional feature recomputation
- Interpolation method for small crypto gaps (forward-fill vs linear)
- Console summary formatting (table style, colors)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | System ingests maximum available equity OHLCV history from Alpaca (all 8 ETFs) into homelab DuckDB | AlpacaIngestor.backfill() already implemented; ingest_all must call it with `since=None` for each symbol in config.equity.symbols |
| DATA-02 | System ingests maximum available crypto 4H OHLCV history from Binance.US (BTC/ETH) into homelab DuckDB | BinanceIngestor.backfill() already implemented with archive stitching back to 2017; ingest_all calls it |
| DATA-03 | System ingests FRED macro data (VIX, T10Y2Y, DFF, CPI, UNRATE) aligned to OHLCV date ranges | FREDIngestor.run_all() already implemented; date-window alignment is the new work |
| DATA-04 | All observation vector dimensions populated without NaN (156-dim equity, 45-dim crypto) | FeaturePipeline.compute_equity/compute_crypto already implemented; verification module is the new work |
| DATA-05 | Data ingestion runs directly on the homelab Docker container (no M1 Mac dependency) | docker-compose.yml bind-mounts data/db/models/logs; ingest_all is the missing entrypoint |
</phase_requirements>

---

## Summary

Phase 18 is a wiring and orchestration phase, not a build-from-scratch phase. The v1.0 codebase already contains fully working implementations of `AlpacaIngestor`, `BinanceIngestor`, `FREDIngestor`, `DatabaseManager`, and `FeaturePipeline`. What is missing is:

1. `src/swingrl/data/ingest_all.py` — the single CLI module that calls existing ingestors in sequence, detects whether new data was added, triggers feature recomputation conditionally, then runs verification.
2. A verification module (`src/swingrl/data/verification.py`) that queries DuckDB and produces the JSON report + console summary.
3. A thin date-window alignment step that trims FRED data to the equity OHLCV date range.

The fail-fast philosophy means pipeline execution is linear and stateless: each stage either succeeds fully or the whole run stops with non-zero exit. This is the safest pattern for a first-run backfill that could take 10-30 minutes.

**Primary recommendation:** Build `ingest_all.py` as a thin orchestrator that imports and calls existing ingestor classes directly (no subprocess). Count DuckDB rows before/after each stage to detect "new data added". Run verification as the final gate.

---

## Standard Stack

### Core (all already in pyproject.toml and tested in v1.0)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| `duckdb` | >=1.0,<2 | Primary analytical store; `DatabaseManager` singleton wraps it | Installed, schema initialized |
| `alpaca-py` | >=0.20 | StockHistoricalDataClient for equity OHLCV | Installed, `AlpacaIngestor` complete |
| `requests` | (transitive) | Binance.US REST API + archive download | Installed, `BinanceIngestor` complete |
| `fredapi` | >=0.5 | FRED macro series fetch | Installed, `FREDIngestor` complete |
| `pandas` | >=2.0,<3 | DataFrame transform throughout | Installed |
| `structlog` | >=24.0 | Structured logging (keyword args pattern enforced) | Installed |
| `pyarrow` | >=14.0 | Parquet read/write via `ParquetStore` | Installed |

### No new dependencies required

All libraries needed for Phase 18 are already declared in `pyproject.toml`. The new module `ingest_all.py` uses only stdlib (`argparse`, `json`, `sys`, `pathlib`) plus the existing `swingrl.*` imports.

---

## Architecture Patterns

### Recommended Project Structure (new files only)

```
src/swingrl/data/
├── ingest_all.py          # NEW — CLI orchestrator, the only entry point for Phase 18
└── verification.py        # NEW — DuckDB verification queries + JSON report writer
tests/data/
└── test_ingest_all.py     # NEW — unit tests for orchestration logic
```

### Pattern 1: Thin Orchestrator (no subprocess)

**What:** `ingest_all.py` imports ingestor classes and calls their `run_all()` / `backfill()` methods directly. No `subprocess.run("python -m swingrl.data.alpaca")`.

**Why direct imports (not subprocess):**
- Easier to detect "new rows added" — compare DuckDB row counts before/after the call in the same process
- Shared logging context (structlog binds run-level context once)
- Exception propagation is clean — no need to parse exit codes
- Consistent with `## Integration Points` in CONTEXT.md: "New `ingest_all` module imports and calls existing ingestor classes directly (not subprocess)"

**Pattern:**

```python
# Source: CONTEXT.md + existing ingestor pattern
def _count_rows(cursor: Any, table: str) -> int:
    return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608

def run_equity(config: SwingRLConfig, backfill: bool) -> int:
    """Returns rows added."""
    db = DatabaseManager(config)
    with db.duckdb() as cursor:
        before = _count_rows(cursor, "ohlcv_daily")
    ingestor = AlpacaIngestor(config)
    since = None if backfill else "incremental"
    failed = ingestor.run_all(config.equity.symbols, since=since)
    if failed:
        raise DataError(f"Equity ingestion failed for: {failed}")
    with db.duckdb() as cursor:
        after = _count_rows(cursor, "ohlcv_daily")
    return after - before
```

### Pattern 2: Conditional Feature Recomputation

**What:** Feature pipeline only runs if at least one ingestor added new rows.

**Detection method (Claude's discretion — recommendation):** Sum row deltas from all three stages. If `equity_added + crypto_added + macro_added > 0`, run features. This is the simplest correct approach. There is no timestamp-based approach needed; DuckDB's `INSERT OR IGNORE` guarantees idempotency.

```python
# Recommended implementation
rows_added = run_equity(config, backfill) + run_crypto(config, backfill) + run_macro(config, backfill)
if rows_added > 0:
    log.info("new_data_added", rows=rows_added)
    run_features(config)
else:
    log.info("no_new_data_skipping_features")
```

### Pattern 3: Verification as Final Gate

**What:** `verification.py` queries DuckDB after the pipeline and checks all quality dimensions.

**Structure:**

```python
# src/swingrl/data/verification.py
@dataclass
class VerificationResult:
    passed: bool
    checks: list[CheckResult]

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

def run_verification(config: SwingRLConfig) -> VerificationResult:
    """Run all checks, return result."""
    ...

def write_report(result: VerificationResult, path: Path) -> None:
    """Write JSON report to data/verification.json."""
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2, default=str))
```

**Checks to implement:**

| Check | Implementation |
|-------|---------------|
| Equity row count per symbol | `SELECT symbol, COUNT(*) FROM ohlcv_daily GROUP BY symbol` |
| Equity date continuity | Compare NYSE trading days to actual dates in DuckDB |
| Crypto row count per symbol | `SELECT symbol, COUNT(*) FROM ohlcv_4h GROUP BY symbol` |
| Crypto 4H gap detection | `SELECT datetime FROM ohlcv_4h ORDER BY datetime` then diff |
| Macro series present | `SELECT DISTINCT series_id FROM macro_features` |
| Macro NaN check | `SELECT series_id, COUNT(*) FILTER (WHERE value IS NULL) FROM macro_features GROUP BY series_id` |
| Equity obs vector shape | `FeaturePipeline.get_observation("equity", last_date)` → assert shape == (156,) |
| Crypto obs vector shape | `FeaturePipeline.get_observation("crypto", last_datetime)` → assert shape == (45,) |
| Zero NaN in obs vectors | `np.isnan(obs).sum() == 0` |

### Pattern 4: DataValidator staleness exemption during backfill

**Critical pitfall (see Pitfalls section):** `DataValidator._check_staleness()` raises `DataError` if the most recent bar is older than 4 days (equity) or 8 hours (crypto). During a **backfill** of historical data (where we download years of old data and process it chronologically), the validator would reject the data as stale on every symbol except the final batch.

The existing `run()` method in `BaseIngestor` calls `validate()` which calls `validate_batch()` which calls `_check_staleness()`. For `AlpacaIngestor.run_all()` called with `since=None`, the data returned already includes the most recent bars (Alpaca returns data up to `datetime.now(UTC)`), so staleness should NOT trigger. However, for `BinanceIngestor.backfill()`, the archive download processes old monthly CSVs first. The `backfill()` method calls `validate()` on the combined final DataFrame (not per-month), so the combined result includes current bars and staleness check should pass.

**Verification:** Confirmed by reading `binance.py` lines 452-456: `self.validate(combined, symbol)` is called after the full combine step, not per archive batch. No code change needed.

### Pattern 5: FRED date-window alignment

**What:** After FRED data is fetched (from 2016-01-01), trim it to the equity OHLCV date window so macro aligner has no NaN gaps in the shared training window.

**How:** Query DuckDB for `MIN(date)` and `MAX(date)` from `ohlcv_daily`, then delete macro rows outside that window — or simply filter during feature computation (MacroFeatureAligner already does this via JOIN).

**Recommendation (Claude's discretion):** Do NOT delete FRED rows. The `MacroFeatureAligner` already uses a forward-fill join strategy that handles sparse macro series. The "align" requirement is met by ensuring FRED data covers the same date range as OHLCV, which it does since both start from 2016-01-01. No additional alignment step needed in `ingest_all`. The verification check should confirm: equity date range ⊆ FRED date range for all 5 series.

### Recommended argparse Interface

```python
# src/swingrl/data/ingest_all.py CLI contract
python -m swingrl.data.ingest_all --backfill
# --backfill: full historical fetch from each source's max depth
# (no flag): incremental — only new data since last bar in DuckDB

# Internal pipeline order (always):
# 1. equity OHLCV (AlpacaIngestor)
# 2. crypto OHLCV (BinanceIngestor.backfill or run_all)
# 3. macro (FREDIngestor)
# 4. feature pipeline (if new rows added)
# 5. verification (always)
```

### Anti-Patterns to Avoid

- **Subprocess calls to other ingestors:** Use direct class imports. The CONTEXT.md is explicit.
- **Running verification before features:** Features must be stored in DuckDB first; verification reads from feature tables.
- **Catching all exceptions in the orchestrator:** Fail fast. Only catch exceptions from individual symbol fetches if partial failure is acceptable (it is not for `ingest_all` — any failure stops the run).
- **Using `--force` to bypass DuckDB incremental check:** Not needed; `INSERT OR IGNORE` handles duplicates.
- **Storing observation vectors in DuckDB:** `FeaturePipeline.get_observation()` assembles them on-the-fly from stored feature tables. Verification calls `get_observation()` once per environment, does not persist the vector.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rate limiting for Alpaca | Custom sleep/throttle | `AlpacaIngestor._fetch_with_retry()` — already implemented with exponential backoff | Built and tested in v1.0 |
| Rate limiting for Binance | Custom sleep/throttle | `BinanceIngestor._fetch_klines()` — reads X-MBX-USED-WEIGHT-1M header and sleeps at 80% | Built and tested in v1.0 |
| Archive download for pre-2019 crypto | Custom HTTP download | `BinanceIngestor.backfill()` — downloads monthly ZIPs from data.binance.vision, stitches, validates | Built and tested in v1.0 |
| Parquet upsert logic | Custom file merge | `ParquetStore.upsert()` | Built and tested in v1.0 |
| DuckDB schema init | Manual DDL | `DatabaseManager.init_schema()` — idempotent, handles all tables | Built and tested in v1.0 |
| Feature computation | Custom indicator math | `FeaturePipeline.compute_equity()` / `compute_crypto()` | Built and tested in v1.0 |
| OHLCV gap detection | Custom NYSE calendar logic | `DataValidator._detect_equity_gaps()` uses `exchange_calendars` | Built and tested in v1.0 |
| Observation vector assembly | Manual array indexing | `FeaturePipeline.get_observation()` | Built and tested in v1.0; 156-dim and 45-dim shapes verified |

**Key insight:** Phase 18 is a wiring phase. Nearly every hard problem (API pagination, archive stitching, Parquet upsert, DuckDB schema, feature computation) is already solved. The new code surface is ~300-500 lines across two new files.

---

## Common Pitfalls

### Pitfall 1: DatabaseManager singleton not reset between test runs

**What goes wrong:** `DatabaseManager` is a singleton (class-level `_instance`). Tests that construct it with different configs or temp paths will reuse the first instance. Subsequent tests see wrong paths.

**Why it happens:** Singleton pattern in `db.py` lines 52-59 — `__new__` always returns the existing `_instance`.

**How to avoid:** Call `DatabaseManager.reset()` in test teardown. The `conftest.py` already uses this for data tests. Any test for `ingest_all` that touches DuckDB must also call `reset()`.

**Warning signs:** `duckdb_path` in logs points to a different path than the test's tmp_dir.

### Pitfall 2: DataValidator staleness check fails on first-run backfill

**What goes wrong:** `_check_staleness()` raises `DataError` because the most recent bar in a freshly fetched month is "old".

**Why it happens:** The stale check compares `df.index.max()` to `datetime.now(UTC)`. Monthly archive CSVs for 2017-2018 have max timestamps years in the past.

**How to avoid:** This is NOT a problem for Phase 18 — `BinanceIngestor.backfill()` calls `self.validate(combined, symbol)` on the fully assembled DataFrame (archive + API), whose max timestamp is the most recent 4H bar (hours ago). Do NOT call `validate()` per monthly archive chunk.

**Warning signs:** `DataError: Stale data for BTCUSDT` during backfill. If this surfaces, it means the code is calling validate per-chunk rather than on the combined result.

### Pitfall 3: Feature pipeline warmup rows produce NaN for early dates

**What goes wrong:** SMA-200 requires 200 rows of history. For equity data starting 2016-01-01, the first 200 trading days (roughly through late 2016) will have NaN for `price_sma200_ratio`. Verification then fails zero-NaN check.

**Why it happens:** Rolling indicators always produce NaN in the warmup window.

**How to avoid:** The CONTEXT.md decision is "trim warmup rows automatically for indicators needing lookback." After feature computation, determine the maximum warmup period (200 bars for SMA-200) and trim the feature tables accordingly. OR: ensure verification only checks observation vectors built from dates AFTER the warmup period.

**Implementation recommendation:** The verification check for zero-NaN in obs vectors should use the LAST date in the feature table (i.e., today's data), not the first. `FeaturePipeline.get_observation("equity", last_date)` will use fully warmed-up features.

### Pitfall 4: Crypto forward-fill for gaps >2 bars contaminates training data

**What goes wrong:** A network outage causes a 24-hour gap (6 missing 4H bars). Forward-filling propagates stale prices, creating artificial flat periods in the training data. The model learns that "no movement" is normal.

**Why it happens:** Simple `ffill()` without a gap limit.

**How to avoid:** The CONTEXT.md decision is clear: gaps ≤2 bars (8 hours) → forward-fill; gaps >2 bars → flag as errors and fail. Use `pd.DataFrame.ffill(limit=2)`. After filling, verify no remaining NaN exists from the fill operation; if any remain, raise `DataError`.

**Where to implement:** In the `ingest_all.py` gap resolution step after crypto ingestion, before feature computation.

### Pitfall 5: DuckDB connection held open across Python module re-imports

**What goes wrong:** During testing, if `DatabaseManager` singleton holds an open DuckDB connection and a test tries to open a new connection to the same file, DuckDB raises a "database is locked" error (DuckDB only allows one open connection per file by default).

**Why it happens:** DuckDB's single-writer model. The singleton's persistent connection blocks.

**How to avoid:** Always use `DatabaseManager.reset()` in test teardown. In production, `ingest_all.py` is run in a fresh process each time (`docker exec` spawns a new process), so this is test-only.

### Pitfall 6: AlpacaIngestor uses Parquet as incremental checkpoint, not DuckDB

**What goes wrong:** `AlpacaIngestor._resolve_start()` reads the latest bar from the Parquet file (line 185-199 in alpaca.py), NOT from DuckDB. If the Parquet file exists but DuckDB does not have the data (e.g., first run on homelab with existing Parquet from dev), the ingestor will think it's up to date and skip backfill.

**Why it happens:** The original design used Parquet as the source of truth for incremental tracking.

**How to avoid:** For the `--backfill` flag, always pass `since=None` to `AlpacaIngestor.run_all()`. This bypasses the Parquet check entirely. The `ingest_all --backfill` command must always use `since=None`.

**Warning sign:** DuckDB `ohlcv_daily` is empty but no data is fetched.

---

## Code Examples

Verified patterns from existing source:

### ingest_all skeleton (the new module)

```python
# src/swingrl/data/ingest_all.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import structlog

from swingrl.config.schema import load_config
from swingrl.data.alpaca import AlpacaIngestor
from swingrl.data.binance import BinanceIngestor
from swingrl.data.db import DatabaseManager
from swingrl.data.fred import FREDIngestor
from swingrl.data.verification import run_verification, write_report
from swingrl.features.pipeline import FeaturePipeline
from swingrl.utils.exceptions import DataError
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)

_VERIFICATION_PATH = Path("data/verification.json")


def _count_rows(db: DatabaseManager, table: str) -> int:
    """Return current row count for a DuckDB table."""
    with db.duckdb() as cursor:
        return cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608


def run_equity(config: ..., backfill: bool) -> int:
    db = DatabaseManager(config)
    before = _count_rows(db, "ohlcv_daily")
    ingestor = AlpacaIngestor(config)
    since = None if backfill else "incremental"
    failed = ingestor.run_all(config.equity.symbols, since=since)
    if failed:
        raise DataError(f"Equity ingestion failed for symbols: {failed}")
    return _count_rows(db, "ohlcv_daily") - before


def run_crypto(config: ..., backfill: bool) -> int:
    db = DatabaseManager(config)
    before = _count_rows(db, "ohlcv_4h")
    ingestor = BinanceIngestor(config)
    if backfill:
        for symbol in config.crypto.symbols:
            ingestor.backfill(symbol)
    else:
        failed = ingestor.run_all(config.crypto.symbols, since=None)
        if failed:
            raise DataError(f"Crypto ingestion failed for symbols: {failed}")
    return _count_rows(db, "ohlcv_4h") - before


def run_macro(config: ..., backfill: bool) -> int:
    db = DatabaseManager(config)
    before = _count_rows(db, "macro_features")
    ingestor = FREDIngestor(config)
    failed = ingestor.run_all(backfill=backfill)
    if failed:
        raise DataError(f"Macro ingestion failed for series: {failed}")
    return _count_rows(db, "macro_features") - before


def run_features(config: ...) -> None:
    db = DatabaseManager(config)
    with db.duckdb() as conn:
        pipeline = FeaturePipeline(config, conn)
        pipeline.compute_equity()
        pipeline.compute_crypto()
```

### Verification query pattern

```python
# src/swingrl/data/verification.py — example check
def _check_equity_rows(cursor: Any, config: SwingRLConfig) -> CheckResult:
    """Verify each equity symbol has at least N rows."""
    rows = cursor.execute(
        "SELECT symbol, COUNT(*) AS cnt FROM ohlcv_daily GROUP BY symbol"
    ).fetchdf()
    missing = [s for s in config.equity.symbols if s not in rows["symbol"].values]
    if missing:
        return CheckResult(name="equity_rows", passed=False, detail=f"Missing symbols: {missing}")
    low = rows[rows["cnt"] < 100]
    if not low.empty:
        return CheckResult(name="equity_rows", passed=False, detail=f"Low row count: {low.to_dict('records')}")
    return CheckResult(name="equity_rows", passed=True, detail=f"All {len(config.equity.symbols)} symbols present")
```

### Forward-fill with gap limit

```python
# Claude's discretion: use pandas ffill with limit=2 for crypto gaps
def resolve_crypto_gaps(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Forward-fill gaps of ≤2 bars; raise DataError for larger gaps."""
    filled = df.ffill(limit=2)
    remaining_nan = filled.isna().any(axis=1)
    if remaining_nan.any():
        gap_count = remaining_nan.sum()
        msg = f"Crypto gap >2 bars for {symbol}: {gap_count} unfillable NaN rows"
        log.error("crypto_gap_unfillable", symbol=symbol, gap_count=gap_count)
        raise DataError(msg)
    return filled
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Run ingestors manually one by one | Single `ingest_all --backfill` command | DATA-05 compliance — operator runs one command |
| Feature pipeline run separately | Feature pipeline auto-triggered by ingest_all | No manual steps; pipeline is atomic |
| No post-ingestion verification | verification.py produces JSON + console summary | Catches NaN issues before training begins |
| Parquet only | Parquet + DuckDB (dual-write via BaseIngestor.run()) | DuckDB enables fast SQL verification queries |

---

## Open Questions

1. **HMM warmup requirement during feature computation**
   - What we know: `FeaturePipeline.compute_equity()` calls `self._hmm_equity.initial_fit(spy_ohlcv["close"])` and requires `len(spy_ohlcv) >= 100`
   - What's unclear: If SPY data starts 2016-01-01, the first 100 trading days (~5 months) will not produce HMM state. Verification must not query HMM state for dates before late 2016.
   - Recommendation: Verification uses LAST date in feature tables, not FIRST. HMM state is stored in `hmm_state_history` for the last fitted date. This is fine.

2. **Alpaca free tier historical depth**
   - What we know: CONTEXT.md says "typically 5-7 years". `_BACKFILL_START = datetime(2016, 1, 1)` is hardcoded in alpaca.py.
   - What's unclear: Exact depth available as of 2026; Alpaca may limit IEX feed to fewer years.
   - Recommendation: Accept whatever Alpaca returns. The verification check should log date range coverage per symbol rather than enforcing a minimum year.

3. **FeaturePipeline.compute_equity() reads from DuckDB, not Parquet**
   - What we know: `_read_equity_ohlcv()` queries `ohlcv_daily` in DuckDB. BaseIngestor._sync_to_duckdb() syncs Parquet data to DuckDB after store.
   - What's unclear: If DuckDB sync fails silently (log warning, return 0), feature pipeline will compute on empty data.
   - Recommendation: Verification check should explicitly confirm `ohlcv_daily` row counts match across DuckDB before running features.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (>=7.x, configured via pyproject.toml) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/data/test_ingest_all.py -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | `run_equity()` calls AlpacaIngestor and returns row delta | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_equity_backfill -x` | Wave 0 |
| DATA-02 | `run_crypto()` calls BinanceIngestor.backfill() for each symbol | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_crypto_backfill -x` | Wave 0 |
| DATA-03 | `run_macro()` calls FREDIngestor.run_all(backfill=True) | unit (mock) | `uv run pytest tests/data/test_ingest_all.py::test_run_macro_backfill -x` | Wave 0 |
| DATA-04 | `run_verification()` detects NaN in obs vectors | unit (tmp DuckDB) | `uv run pytest tests/data/test_ingest_all.py::test_verification_nan_detection -x` | Wave 0 |
| DATA-04 | `run_verification()` confirms 156-dim equity and 45-dim crypto obs shapes | unit (tmp DuckDB) | `uv run pytest tests/data/test_ingest_all.py::test_verification_obs_shapes -x` | Wave 0 |
| DATA-05 | `python -m swingrl.data.ingest_all --help` exits 0 (importable as module) | smoke | `uv run python -m swingrl.data.ingest_all --help` | Wave 0 |
| DATA-01–03 | fail-fast: any ingestor failure raises and propagates | unit | `uv run pytest tests/data/test_ingest_all.py::test_fail_fast_on_equity_error -x` | Wave 0 |
| DATA-01–03 | feature pipeline skipped when no new rows added | unit | `uv run pytest tests/data/test_ingest_all.py::test_skip_features_on_no_new_data -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/data/test_ingest_all.py -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/data/test_ingest_all.py` — covers DATA-01 through DATA-05 (full file, does not exist yet)
- [ ] `src/swingrl/data/ingest_all.py` — the orchestrator module (does not exist yet)
- [ ] `src/swingrl/data/verification.py` — DuckDB verification queries (does not exist yet)

*(No new framework install required — pytest and all supporting libraries already installed)*

---

## Sources

### Primary (HIGH confidence)

- Existing source code: `src/swingrl/data/alpaca.py`, `binance.py`, `fred.py`, `base.py`, `db.py`, `validation.py` — read directly during research
- Existing source code: `src/swingrl/features/pipeline.py` — read directly; `compute_equity()` and `compute_crypto()` methods verified
- `.planning/phases/18-data-ingestion/18-CONTEXT.md` — locked decisions, code context, integration points
- `.planning/REQUIREMENTS.md` — DATA-01 through DATA-05 definitions
- `docker-compose.yml` — bind mount configuration confirmed; `./data:/app/data`, `./db:/app/db`
- `pyproject.toml` — all required libraries confirmed present

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — project history; DummyVecEnv/SAC buffer decisions from v1.0 (relevant background)
- `CLAUDE.md` — project conventions enforced throughout

### Tertiary (LOW confidence — not needed)

None — all research grounded in the actual codebase.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed installed; no new dependencies
- Architecture: HIGH — existing patterns fully read and understood; new module design follows established patterns
- Pitfalls: HIGH — identified by reading actual implementation (DataValidator staleness, AlpacaIngestor Parquet checkpoint, DuckDB singleton)
- Verification design: MEDIUM — verification module design is new; the DuckDB query patterns are straightforward but untested

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable domain; Alpaca/Binance API changes unlikely within 30 days)
