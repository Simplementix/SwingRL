# Phase 3: Data Ingestion - Research

**Researched:** 2026-03-06
**Domain:** Multi-source financial data ingestion (Alpaca, Binance.US, FRED), Parquet storage, data validation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Ingestor architecture:**
- Abstract base class (BaseIngestor) with fetch(), validate(), store() contract
- Three concrete implementations: AlpacaIngestor, BinanceIngestor, FREDIngestor
- Each ingestor callable as CLI (`python -m swingrl.data.alpaca`) AND importable as library
- CLI supports manual backfills with flags (`--symbols`, `--days`, `--backfill`); library API for Phase 9 automation
- Both full-fetch and incremental (since-last-timestamp) modes from day one
- API credentials via environment variables only (ALPACA_API_KEY, ALPACA_SECRET_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, FRED_API_KEY) — never in YAML config

**Crypto backfill strategy:**
- Source: Binance public data archives (data.binance.vision) for 2017-2019, Binance.US API for 2019+
- 30-day overlap validation window at the stitch point — fetch both sources, validate prices match within <0.5% deviation, log discrepancies
- Use Binance.US data from the overlap period onward
- Volume normalization: use quote_asset_volume field (already in USDT) where available, fall back to `volume_base * close_price` where not
- Backfill integrated into BinanceIngestor as `.backfill(start_date, end_date)` method, invoked via CLI `--backfill` flag

**Validation and quarantine:**
- Row-level quarantine — bad rows go to quarantine, good rows proceed
- Row-level checks: null checks, price > 0, volume >= 0, OHLC ordering (H >= L, O/C between H/L)
- Batch-level checks: gap detection, duplicate detection, stale data check, row count threshold
- Quarantine storage: Parquet files in `data/quarantine/{source}_{date}.parquet` (Phase 4 migrates to DB)
- Alerting: structlog warnings per failure + summary at end of run; Discord deferred to Phase 4

**Output format:**
- `data/equity/{SYMBOL}_daily.parquet`
- `data/crypto/{SYMBOL}_4h.parquet`
- `data/macro/{SERIES}.parquet`
- `data/quarantine/{source}_{date}.parquet`
- Single file per symbol (no year partitioning)
- FRED Parquet includes vintage_date alongside observation_date
- Equity backfill: 10 years (2016-present)

**FRED macro pipeline:**
- 5 Tier 1 series: VIXCLS, T10Y2Y, DFF, CPIAUCSL, UNRATE
- ALFRED vintage data back to 2016
- Store at native frequency; Phase 5 ASOF JOIN handles alignment
- Daily series fetched daily; monthly series checked weekly

**Error handling:**
- 3 retries with exponential backoff (2s, 4s, 8s)
- After 3 failures: log + move to next symbol — partial success over total failure
- Run exits with non-zero code if ANY symbol/series failed
- Binance.US: proactive throttling via X-MBX-USED-WEIGHT — if >80% of 1200/min limit, sleep until window resets
- All API errors raise DataError with context

**Incremental update semantics:**
- Upsert on matching timestamps (read + merge + dedup + write back)
- Last-fetched timestamp derived from max(timestamp) in existing Parquet
- exchange_calendars for NYSE gap detection
- Crypto: continuous 4H bars expected 24/7

**Testing strategy:**
- Unit tests: mocked HTTP responses
- Integration tests: @pytest.mark.integration, skipped in CI
- Backfill tests: small fixture CSV (~100 rows) mimicking Binance archive format
- Validation fixtures: one fixture per defect type

### Claude's Discretion
- Exact 12-step validation checklist mapping
- Alpaca IEX feed pagination approach
- Parquet compression and detailed column schema (types, metadata)
- Specific mocking library choice (responses vs pytest-httpx)
- exchange_calendars integration details
- CLI argument parser implementation (argparse vs click)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Alpaca OHLCV ingestion for 8 equity ETFs via IEX feed | alpaca-py StockHistoricalDataClient + StockBarsRequest; feed="iex"; pagination via next_page_token |
| DATA-02 | Binance.US 4H bar ingestion for BTC/USD and ETH/USD with rate limit monitoring | GET /api/v3/klines; weight=1 per request; 1200 weight/min limit; X-MBX-USED-WEIGHT header |
| DATA-03 | 8+ year crypto historical backfill stitching Binance archives with Binance.US API | data.binance.vision CSV archives; IMPORTANT: timestamps in microseconds from 2025-01-01 onwards; quote_asset_volume for USD normalization |
| DATA-04 | FRED macro pipeline using ALFRED vintage data | fredapi get_series_all_releases(); realtime_start/realtime_end parameters; stores vintage_date |
| DATA-05 | 12-step data validation checklist with quarantine | Row-level + batch-level checks; quarantine Parquet; structlog warnings; DataError exception |
</phase_requirements>

---

## Summary

Phase 3 builds the three data ingestors that feed all downstream phases. The design is well-specified in CONTEXT.md with clear architectural decisions already locked. Research confirms the primary libraries (alpaca-py, fredapi, exchange_calendars, pyarrow) are stable, well-maintained, and fit the locked design without friction.

The most important technical findings are: (1) Binance public archive timestamps switched from milliseconds to microseconds on 2025-01-01 for SPOT data — this is a critical parsing gotcha for the backfill; (2) Binance.US klines weight is 1 per request (not 2 as stated in generic Binance docs), meaning the 1200/min ceiling allows ~1200 requests/min during backfill; (3) fredapi's `get_series_all_releases()` returns the exact ALFRED vintage structure needed, with `realtime_start` as the vintage date column to store; (4) alpaca-py's multi-symbol bar endpoint paginates by symbol first — all 8 ETFs can be requested together but multiple pages may be needed; (5) Parquet upsert must be implemented as read-merge-dedup-write (no in-place update support).

**Primary recommendation:** Use `responses` (not pytest-httpx) for HTTP mocking since alpaca-py and fredapi both use `requests` under the hood, not `httpx`. Use `argparse` over `click` to avoid adding a new dependency.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| alpaca-py | >=0.20 (already in pyproject.toml) | Alpaca REST API client | Official Alpaca SDK; StockHistoricalDataClient handles auth, pagination, rate limits |
| fredapi | >=0.5.2 | FRED/ALFRED API client | Official FRED wrapper; only library with get_series_all_releases() for vintage data |
| exchange_calendars | >=4.13.1 | NYSE trading calendar | Maintained; Python 3.10+; is_session() + sessions_in_range() cover all gap detection needs |
| pyarrow | >=14.0 (transitive via pandas) | Parquet read/write | pandas.read_parquet() and DataFrame.to_parquet() use pyarrow engine by default |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| responses | >=0.25.8 | Mock HTTP for unit tests | alpaca-py and fredapi both use `requests`; matches test fixtures pattern established in Phase 2 |
| requests | (transitive) | HTTP for FRED/Binance | Binance.US raw API calls (no official Python SDK for Binance.US) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| responses | pytest-httpx | pytest-httpx is for httpx clients — alpaca-py uses requests internally, making responses the correct choice |
| argparse | click | click adds a new dependency; argparse is stdlib and sufficient for the 3 flags needed |
| fredapi | requests directly to FRED API | fredapi encapsulates auth, pagination, and ALFRED vintage methods — significant code reduction |
| exchange_calendars | pandas_market_calendars | Both valid; exchange_calendars is the more actively maintained fork (4.13.1 vs older releases) |

**Installation (new Phase 3 deps):**
```bash
uv add "fredapi>=0.5" "exchange_calendars>=4.13" "responses>=0.25"
```

Note: `pyarrow` is already a transitive dependency via `pandas`. `alpaca-py` is already in pyproject.toml.

---

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/data/
├── __init__.py               # already exists
├── base.py                   # BaseIngestor ABC
├── alpaca.py                 # AlpacaIngestor + CLI entrypoint
├── binance.py                # BinanceIngestor + CLI entrypoint (includes backfill)
├── fred.py                   # FREDIngestor + CLI entrypoint
├── validation.py             # DataValidator with 12-step checklist
└── parquet_store.py          # ParquetStore: read/upsert/write helpers

tests/data/
├── __init__.py
├── fixtures/
│   ├── alpaca_bars_spy.json       # Mocked Alpaca API response
│   ├── binance_klines_btc.json    # Mocked Binance API response
│   ├── binance_archive_btc.csv    # ~100 rows mimicking data.binance.vision format
│   └── fred_cpiaucsl_releases.json  # Mocked FRED get_series_all_releases response
├── test_alpaca_ingestor.py
├── test_binance_ingestor.py
├── test_fred_ingestor.py
├── test_validation.py
└── test_parquet_store.py
```

### Pattern 1: BaseIngestor Abstract Contract
**What:** ABC defining the three-method protocol all ingestors implement
**When to use:** All three ingestor implementations inherit from this

```python
# src/swingrl/data/base.py
from __future__ import annotations

import abc
from pathlib import Path

import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class BaseIngestor(abc.ABC):
    """Abstract base for all data ingestors."""

    def __init__(self, config: SwingRLConfig) -> None:
        self._config = config
        self._data_dir = Path(config.paths.data_dir)

    @abc.abstractmethod
    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch OHLCV bars. Since is ISO timestamp for incremental mode."""
        ...

    @abc.abstractmethod
    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (clean_df, quarantine_df). Never raises — quarantines bad rows."""
        ...

    @abc.abstractmethod
    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert df into Parquet. Returns path written."""
        ...

    def run(self, symbol: str, since: str | None = None) -> None:
        """Orchestrate fetch -> validate -> store for one symbol."""
        log.info("ingestor_run_start", symbol=symbol, since=since)
        raw = self.fetch(symbol, since)
        clean, quarantine = self.validate(raw, symbol)
        if not quarantine.empty:
            self._store_quarantine(quarantine, symbol)
            log.warning(
                "rows_quarantined",
                symbol=symbol,
                count=len(quarantine),
            )
        self.store(clean, symbol)
        log.info("ingestor_run_complete", symbol=symbol, rows=len(clean))

    def _store_quarantine(self, df: pd.DataFrame, symbol: str) -> None:
        """Write quarantined rows to data/quarantine/."""
        from datetime import date

        q_dir = self._data_dir / "quarantine"
        q_dir.mkdir(parents=True, exist_ok=True)
        path = q_dir / f"{symbol}_{date.today().isoformat()}.parquet"
        df.to_parquet(path, index=True)
```

### Pattern 2: Alpaca Multi-Symbol Bars with Pagination
**What:** Fetch all 8 ETFs in a single request, handle next_page_token loop
**When to use:** DATA-01 implementation

```python
# Source: alpaca-py SDK + Alpaca docs https://docs.alpaca.markets/reference/stockbars
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

request = StockBarsRequest(
    symbol_or_symbols=["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"],
    timeframe=TimeFrame.Day,
    start="2016-01-01",
    feed=DataFeed.IEX,          # free tier — IEX exchange data
    adjustment="all",           # split + dividend adjusted
)
bars = client.get_stock_bars(request)
df = bars.df   # MultiIndex: (symbol, timestamp)
```

Key: The SDK handles pagination automatically via `get_stock_bars()` — it keeps requesting with `next_page_token` until exhausted. The `.df` property returns a combined multi-symbol DataFrame.

Incremental mode: derive `start` from `max(timestamp)` in existing Parquet before constructing request.

### Pattern 3: Binance.US Klines with Rate Limit Monitoring
**What:** Paginate klines 1000 bars at a time, check X-MBX-USED-WEIGHT on each response
**When to use:** DATA-02 incremental fetch and DATA-03 backfill

```python
# Source: Binance.US API docs https://docs.binance.us/
import time
import requests

BASE_URL = "https://api.binance.us"
KLINES_ENDPOINT = "/api/v3/klines"
WEIGHT_LIMIT = 1200      # per minute
WEIGHT_THRESHOLD = 0.80  # pause at 80%

def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    headers: dict[str, str],
) -> list[list]:
    """Fetch up to 1000 klines; returns raw list."""
    resp = requests.get(
        BASE_URL + KLINES_ENDPOINT,
        params={
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        },
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()

    # Monitor rate limit
    used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0))
    if used > WEIGHT_LIMIT * WEIGHT_THRESHOLD:
        # Sleep until the 1-minute window likely resets
        time.sleep(60)

    return resp.json()
```

For backfill: loop with `startTime += 1000 * 4 * 3600 * 1000` (1000 bars × 4h × 3600s × 1000ms) until `startTime > end_ms`.

**CRITICAL timestamp note:** Binance public archive CSVs use millisecond timestamps for data before 2025-01-01 and microsecond timestamps for 2025-01-01 onward. The live API always returns milliseconds. Parse accordingly:
```python
# Archive CSV parsing — detect unit from value magnitude
ts_raw = int(row[0])
if ts_raw > 2_000_000_000_000:  # > year 2033 in ms => must be microseconds
    ts = pd.Timestamp(ts_raw, unit="us", tz="UTC")
else:
    ts = pd.Timestamp(ts_raw, unit="ms", tz="UTC")
```

Klines response column order (positions, all strings):
```
[0]  open_time (ms)
[1]  open
[2]  high
[3]  low
[4]  close
[5]  volume (base asset)
[6]  close_time (ms)
[7]  quote_asset_volume      <- use THIS for USD volume
[8]  number_of_trades
[9]  taker_buy_base_volume
[10] taker_buy_quote_volume
[11] ignore
```

### Pattern 4: FRED ALFRED Vintage Data
**What:** Fetch full release history for a series, store with vintage_date column
**When to use:** DATA-04 — prevents look-ahead bias in CPI and unemployment

```python
# Source: fredapi README + FRED API docs https://fred.stlouisfed.org/docs/api/fred/series_observations.html
from fredapi import Fred

fred = Fred(api_key=api_key)

# get_series_all_releases returns DataFrame with columns: date, realtime_start, value
releases = fred.get_series_all_releases("CPIAUCSL")
# Rename to project convention
releases = releases.rename(columns={"realtime_start": "vintage_date"})
# Store: observation_date (index), vintage_date, value
```

For daily series (VIXCLS, T10Y2Y, DFF) where revisions are rare, `get_series()` is sufficient.
For revised series (CPIAUCSL, UNRATE) use `get_series_all_releases()` to capture each vintage.

The CONTEXT.md decision to store `vintage_date` alongside `observation_date` enables Phase 5 to do an ASOF JOIN: for any training bar at date T, join to the CPI value known as of T (no future revisions).

### Pattern 5: Parquet Upsert (Read-Merge-Dedup-Write)
**What:** Parquet files are immutable — upsert requires reading, merging, deduplicating, writing
**When to use:** All three ingestors, incremental mode

```python
# Source: pyarrow docs https://arrow.apache.org/docs/python/parquet.html
import pandas as pd
from pathlib import Path


def upsert_parquet(path: Path, new_df: pd.DataFrame, key_col: str = "timestamp") -> None:
    """Upsert new_df into existing Parquet at path. Creates file if missing."""
    if path.exists():
        existing = pd.read_parquet(path)
        # Concatenate and keep most recent value for each key
        combined = pd.concat([existing, new_df])
        # Sort descending so new values win on dedup
        combined = combined.sort_index(ascending=False)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index(ascending=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.to_parquet(path, index=True, compression="snappy")
```

Parquet column schema (equity and crypto, all UTC timestamps as index):
```
Index: DatetimeTZDtype(tz=UTC)  — open_time
open:  float64
high:  float64
low:   float64
close: float64
volume: float64                 — quote asset volume (USD) for crypto; share volume for equity
```

FRED Parquet schema:
```
Index: DatetimeTZDtype(tz=UTC)  — observation_date
value: float64
vintage_date: datetime64[ns, UTC]
```

Compression: `snappy` — fast decompression (read-heavy workload for training), small enough for single-symbol files.

### Pattern 6: exchange_calendars for Equity Gap Detection
**What:** Check only trading days when detecting gaps in equity data
**When to use:** Batch-level gap check in DataValidator for equity sources

```python
# Source: exchange_calendars README https://github.com/gerrymanoim/exchange_calendars
import exchange_calendars as xcals

nyse = xcals.get_calendar("XNYS")

def find_equity_gaps(df: pd.DataFrame, symbol: str) -> list[pd.Timestamp]:
    """Return trading days with missing bars."""
    if df.empty:
        return []
    start = df.index.min()
    end = df.index.max()
    expected_sessions = nyse.sessions_in_range(start, end)
    actual_dates = pd.DatetimeIndex(df.index.normalize())
    missing = expected_sessions.difference(actual_dates)
    return missing.tolist()

# Single-date check:
nyse.is_session("2024-01-15")  # True/False
```

### Pattern 7: CLI Entry Points via `__main__.py`
**What:** Each ingestor module has a `__main__.py` for `python -m swingrl.data.alpaca` invocation
**When to use:** Manual backfills, testing, Phase 9 scheduler calls

```python
# src/swingrl/data/alpaca/__main__.py  (if using package style)
# OR src/swingrl/data/alpaca.py with if __name__ == "__main__": guard

import argparse
import sys

from swingrl.config.schema import load_config
from swingrl.data.alpaca import AlpacaIngestor


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpaca equity data ingestor")
    parser.add_argument("--symbols", nargs="+", help="Override symbols from config")
    parser.add_argument("--days", type=int, default=None, help="Number of days to backfill")
    parser.add_argument("--backfill", action="store_true", help="Full historical backfill")
    args = parser.parse_args()

    config = load_config()
    ingestor = AlpacaIngestor(config)
    symbols = args.symbols or config.equity.symbols
    failed: list[str] = []
    for symbol in symbols:
        try:
            ingestor.run(symbol, since=None if args.backfill else "incremental")
        except Exception as e:
            log.error("symbol_failed", symbol=symbol, error=str(e))
            failed.append(symbol)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

### Anti-Patterns to Avoid
- **Calling broker APIs from ingestor modules:** Binance.US is used here purely as a market data source, not an order broker. Raw `requests` calls to the market data endpoint are appropriate — do NOT use the execution middleware (that's Phase 8).
- **Storing API keys in config YAML:** All credentials via `os.environ` only, as locked in CONTEXT.md.
- **Partitioned Parquet (year/month subdirs):** The dataset sizes are small (≤17,500 rows for 8yr crypto, ≤2,500 rows for 10yr equity). Single-file Parquet outperforms partitioned at this scale.
- **Raising new exception subclasses:** `DataError` from Phase 2 covers all data pipeline errors. Log before raising.
- **Using pandas `datetime.now()` without UTC:** Always `pd.Timestamp.utcnow()` or `datetime.now(timezone.utc)`.
- **Parsing Binance archive timestamps without checking unit:** From 2025-01-01, archive CSVs switched to microseconds. Must detect unit dynamically (see Pattern 3).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NYSE holiday calendar | Custom holiday list | exchange_calendars XNYS | exchange_calendars tracks 50+ years of NYSE schedule changes, early closes, ad-hoc closures (9/11, COVID) |
| FRED API client | requests + manual pagination | fredapi | fredapi handles auth, pagination, and ALFRED vintage methods (get_series_all_releases) |
| Alpaca auth + pagination | Manual HTTP | alpaca-py StockHistoricalDataClient | SDK auto-paginates, handles next_page_token loop, manages auth headers |
| Parquet deduplication | Custom merge logic | pandas concat + drop_duplicates on index | Standard pattern; no lock-in |
| Backoff/retry | sleep() loops | tenacity or stdlib time.sleep with explicit counter | 3-attempt exponential backoff is simple enough for stdlib; tenacity if needs grow |
| Binance timestamp unit detection | Magic constants | Threshold check (>2e12 = microseconds) | Binance changed units in 2025-01-01 — any hard-coded "ms" assumption will break on recent archives |

**Key insight:** The data sources (Alpaca, FRED) have well-maintained official Python SDKs that handle all the boilerplate. Binance.US does NOT have an official Python SDK — use raw `requests`. This asymmetry is important: don't assume a binance-python or python-binance SDK works correctly against Binance.US (different endpoint base URLs, different rate limits).

---

## Common Pitfalls

### Pitfall 1: Binance Archive Timestamp Unit Change (2025-01-01)
**What goes wrong:** CSV rows from 2025-01-01 onward in data.binance.vision have timestamps in microseconds. Code that unconditionally applies `pd.to_datetime(ts, unit="ms")` will produce timestamps 1000x in the future (year ~35000).
**Why it happens:** Binance silently changed the timestamp unit for SPOT data starting 2025. Historical archives use milliseconds; 2025+ archives use microseconds.
**How to avoid:** Detect the unit from value magnitude before parsing (see Pattern 3 threshold check). Write a test fixture with a 2025 CSV row to confirm correct parsing.
**Warning signs:** Timestamps in resulting DataFrame after 2050.

### Pitfall 2: Alpaca IEX Feed Data Gaps
**What goes wrong:** IEX only captures trades that execute on the IEX exchange (~1-3% of US equity volume). Some ETFs on some days may have zero IEX trades, producing missing bars.
**Why it happens:** IEX is a single exchange. SIP aggregates all exchanges but requires a paid Alpaca subscription.
**How to avoid:** After fetch, run the gap detection validator. Log missing days as warnings (don't quarantine — they're not bad data, just absent). Consider forward-filling from the last known close for training data. Document this limitation.
**Warning signs:** ETFs like XLE or XLF (lower volume) showing more gaps than SPY.

### Pitfall 3: FRED CPI Monthly Revisions Causing Look-Ahead Bias
**What goes wrong:** `fred.get_series("CPIAUCSL")` returns the most recently revised values. If used in backtesting, the model "knows" the revised CPI number from 2016 that was only published in 2020.
**Why it happens:** FRED serves current data by default; ALFRED requires explicit vintage parameters.
**How to avoid:** Use `get_series_all_releases()` for CPIAUCSL and UNRATE. Store `vintage_date` column. Phase 5 ASOF JOIN filters to only use data known as of each training bar's date.
**Warning signs:** CPI values in 2016 Parquet file exactly matching current BLS data (not the preliminary release).

### Pitfall 4: Binance.US vs International Binance API URLs
**What goes wrong:** Code written for Binance.com (the international exchange) uses `api.binance.com` as the base URL. Binance.US is a separate exchange at `api.binance.us`. Many Python binance libraries target the international exchange.
**Why it happens:** The two exchanges share API design but are separate entities with separate endpoints.
**How to avoid:** Hardcode `BASE_URL = "https://api.binance.us"` as a named constant in `src/swingrl/config/schema.py` or as a module-level constant. Never use a third-party "binance" library that defaults to `.com`.
**Warning signs:** 403 Forbidden or "service unavailable in your region" errors.

### Pitfall 5: Parquet Write Errors on Existing Locked File
**What goes wrong:** On macOS, writing a Parquet file while it's also open by another process (e.g., DuckDB query) raises a PermissionError.
**Why it happens:** Parquet write in pyarrow creates a new file then replaces the old one atomically on most platforms, but file locks can interfere.
**How to avoid:** Write to a temp file in the same directory, then `Path.replace()` (atomic on POSIX). Or simply avoid running DuckDB and ingestors concurrently in Phase 3 (storage is isolated).
**Warning signs:** PermissionError or "file is being used by another process."

### Pitfall 6: exchange_calendars Calendar Cache on First Import
**What goes wrong:** `xcals.get_calendar("XNYS")` on first call downloads and caches holiday data. In CI without internet access, this fails silently or raises.
**Why it happens:** exchange_calendars fetches holiday data from a bundled JSON on import; it does NOT require internet after installation. The download only happens when installing the package.
**How to avoid:** This is actually not a problem post-installation. Document that `exchange_calendars` must be in `uv.lock` for CI. No special handling needed.
**Warning signs:** ImportError or FileNotFoundError on calendar load in Docker.

### Pitfall 7: mypy Overrides Needed for New Libraries
**What goes wrong:** `fredapi` and `exchange_calendars` have limited or no type stubs. mypy strict mode fails CI.
**Why it happens:** Both libraries predate widespread type annotation adoption.
**How to avoid:** Add to `[[tool.mypy.overrides]]` in pyproject.toml:
```toml
[[tool.mypy.overrides]]
module = ["fredapi.*", "exchange_calendars.*"]
ignore_missing_imports = true
```
**Warning signs:** `mypy` errors on `import fredapi` or `import exchange_calendars`.

---

## Code Examples

### Verified: alpaca-py Multi-Symbol Bar Fetch
```python
# Source: alpaca-py SDK docs https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

client = StockHistoricalDataClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
)

request = StockBarsRequest(
    symbol_or_symbols=["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"],
    timeframe=TimeFrame.Day,
    start="2016-01-01",
    feed=DataFeed.IEX,
    adjustment="all",
)
bars = client.get_stock_bars(request)
# Returns BarSet; .df gives MultiIndex (symbol, timestamp) DataFrame
df = bars.df
```

### Verified: exchange_calendars NYSE Gap Check
```python
# Source: exchange_calendars README https://github.com/gerrymanoim/exchange_calendars
import exchange_calendars as xcals

nyse = xcals.get_calendar("XNYS")
sessions = nyse.sessions_in_range("2024-01-01", "2024-01-15")
# DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
#                '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11',
#                '2024-01-12'])  — excludes New Year's Day holiday

nyse.is_session("2024-01-01")   # False — New Year's Day
nyse.is_session("2024-01-02")   # True
```

### Verified: fredapi ALFRED Vintage Fetch
```python
# Source: fredapi README https://github.com/mortada/fredapi
import os
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])

# All historical releases for CPI — use this for CPIAUCSL and UNRATE
releases_df = fred.get_series_all_releases("CPIAUCSL")
# Columns: date (observation date), realtime_start (vintage date), value

# Simple series for daily data that rarely revises (VIXCLS, T10Y2Y, DFF)
vix_series = fred.get_series("VIXCLS", observation_start="2016-01-01")
# Returns pd.Series with DatetimeIndex
```

### Verified: Binance.US Klines Column Parsing
```python
# Source: Binance.US API docs https://docs.binance.us/ + Binance spot API docs
import pandas as pd

KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close",
    "volume_base",     # base asset volume (BTC for BTCUSDT)
    "close_time",
    "volume_quote",    # quote asset volume (USDT) — USE THIS for USD normalization
    "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

def parse_klines(raw: list[list]) -> pd.DataFrame:
    """Parse raw Binance klines response into clean DataFrame."""
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    # Live API: timestamps always in milliseconds
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume_quote"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume_quote"]   # project convention: volume = USD value
    return df[["open", "high", "low", "close", "volume"]]
```

### Verified: responses Library Mocking Pattern
```python
# Source: responses README https://github.com/getsentry/responses
import responses as responses_lib
import pytest

@responses_lib.activate
def test_fetch_bars_calls_api():
    """DATA-01: AlpacaIngestor.fetch() calls bars endpoint with IEX feed."""
    responses_lib.add(
        responses_lib.GET,
        "https://data.alpaca.markets/v2/stocks/bars",
        json={"bars": {"SPY": [{"t": "2024-01-02T00:00:00Z", "o": 470.0, "h": 472.0,
                                "l": 469.0, "c": 471.0, "v": 80000000}]},
              "next_page_token": None},
        status=200,
    )
    # ... test body
```

---

## 12-Step Validation Checklist

Claude's Discretion applies here — the following is a recommended mapping that covers the success criteria for DATA-05:

| Step | Type | Check | Action on Failure |
|------|------|-------|-------------------|
| 1 | Row | Null check: open/high/low/close/volume are not NaN | Quarantine row |
| 2 | Row | Price sanity: open, high, low, close > 0 | Quarantine row |
| 3 | Row | Volume sanity: volume >= 0 | Quarantine row |
| 4 | Row | OHLC ordering: high >= low | Quarantine row |
| 5 | Row | OHLC bounds: open and close between low and high (with 0.01% tolerance for float imprecision) | Quarantine row |
| 6 | Row | Price spike: close/prev_close > 1.5 or < 0.5 (>50% move in one bar) | Quarantine row + warn |
| 7 | Row | Zero volume: volume == 0 AND it is not a known market holiday (equity only) | Quarantine row |
| 8 | Batch | Duplicate timestamps: same symbol + timestamp appears more than once | Keep latest, log warning |
| 9 | Batch | Gap detection: missing expected bars (NYSE calendar for equity; continuous for crypto) | Log warning, do NOT quarantine (absence is different from corruption) |
| 10 | Batch | Stale data: max(timestamp) older than expected freshness threshold (>2 trading days equity, >8H crypto) | Log warning, raise DataError after retries exhausted |
| 11 | Batch | Row count threshold: fewer than N rows returned when N expected (e.g., <90% of expected for backfill) | Log warning; partial success acceptable |
| 12 | Batch | Cross-source consistency check (DATA-11, Phase 4 only): this step is deferred; placeholder in checklist | Skip in Phase 3 |

**Implementation note:** Steps 1-7 are row-level — apply as a vectorized pandas filter, collect failing rows into quarantine_df, return clean rows. Steps 8-11 operate on the full batch after row-level filtering.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| alpaca-trade-api-python (deprecated) | alpaca-py | 2022 | New SDK; different class names and import paths |
| direct FRED REST calls | fredapi | — | fredapi encapsulates ALFRED vintage methods not easily replicated manually |
| pandas_market_calendars | exchange_calendars | Ongoing | exchange_calendars is the actively maintained fork with Python 3.10+ support |
| Binance archive timestamp in milliseconds | Binance archive timestamp in MICROSECONDS for 2025+ | 2025-01-01 | Critical parsing change for backfill of recent data |

**Deprecated/outdated:**
- `alpaca-trade-api` (the old SDK): Do not use. `alpaca-py` is the official successor with different import paths.
- `python-binance` library: Targets international Binance.com, not Binance.US. Avoid.
- `pandas.io.data` or `pandas_datareader` for FRED: Use `fredapi` directly for ALFRED support.

---

## Open Questions

1. **alpaca-py feed parameter enum vs string**
   - What we know: The locked decision says `feed="iex"`. The SDK may accept `DataFeed.IEX` enum or the string `"iex"`.
   - What's unclear: Whether `StockBarsRequest(feed="iex")` works or requires `feed=DataFeed.IEX`.
   - Recommendation: Use `DataFeed.IEX` (the enum) from `alpaca.data.enums` to be type-safe. Test both forms in the integration test.

2. **Binance.US archive availability for 4H interval**
   - What we know: data.binance.vision has spot klines for many intervals. The 4H interval (`4h`) should be available.
   - What's unclear: Whether `BTCUSDT` 4H data on data.binance.vision goes back to 2017 or only to a later date.
   - Recommendation: Verify URL `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/4h/` during Wave 0 or first implementation task. May need to fall back to aggregating 1H bars.

3. **fredapi version and maintenance status**
   - What we know: fredapi 0.5.2 was the latest as of August 2025 (knowledge cutoff). It wraps the FRED REST API.
   - What's unclear: Whether there are breaking changes or a more recently maintained fork that should be preferred.
   - Recommendation: `uv add "fredapi>=0.5"` and verify `get_series_all_releases()` works in the integration test against a test API key.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 (already installed) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/data/ -v -m "not integration"` |
| Full suite command | `uv run pytest tests/ -v -m "not integration"` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | AlpacaIngestor.fetch() returns DataFrame with expected columns for SPY | unit | `uv run pytest tests/data/test_alpaca_ingestor.py -x` | ❌ Wave 0 |
| DATA-01 | Incremental mode: start derived from max(timestamp) in existing Parquet | unit | `uv run pytest tests/data/test_alpaca_ingestor.py::test_incremental_start -x` | ❌ Wave 0 |
| DATA-01 | CLI --backfill flag triggers full historical fetch | unit | `uv run pytest tests/data/test_alpaca_ingestor.py::test_cli_backfill -x` | ❌ Wave 0 |
| DATA-02 | BinanceIngestor.fetch() returns 4H DataFrame; X-MBX-USED-WEIGHT logged | unit | `uv run pytest tests/data/test_binance_ingestor.py -x` | ❌ Wave 0 |
| DATA-02 | Rate limit throttle: sleep called when used_weight > 80% of 1200 | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_rate_limit_throttle -x` | ❌ Wave 0 |
| DATA-03 | Backfill: archive CSV parsed correctly with millisecond timestamps | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_archive_parse_ms -x` | ❌ Wave 0 |
| DATA-03 | Backfill: archive CSV with microsecond timestamps (2025+ format) parsed correctly | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_archive_parse_us -x` | ❌ Wave 0 |
| DATA-03 | Stitch point validation: price mismatch >0.5% logged as discrepancy | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_stitch_validation -x` | ❌ Wave 0 |
| DATA-03 | Volume normalization: quote_asset_volume used when available | unit | `uv run pytest tests/data/test_binance_ingestor.py::test_volume_normalization -x` | ❌ Wave 0 |
| DATA-04 | FREDIngestor stores vintage_date alongside observation_date for CPI | unit | `uv run pytest tests/data/test_fred_ingestor.py::test_vintage_date_stored -x` | ❌ Wave 0 |
| DATA-04 | All 5 Tier 1 series fetched and written to expected Parquet paths | unit | `uv run pytest tests/data/test_fred_ingestor.py::test_all_series_paths -x` | ❌ Wave 0 |
| DATA-05 | Row with null price is quarantined, clean rows proceed | unit | `uv run pytest tests/data/test_validation.py::test_null_price_quarantined -x` | ❌ Wave 0 |
| DATA-05 | Row with negative volume is quarantined | unit | `uv run pytest tests/data/test_validation.py::test_negative_volume_quarantined -x` | ❌ Wave 0 |
| DATA-05 | Row with OHLC ordering violation quarantined | unit | `uv run pytest tests/data/test_validation.py::test_ohlc_ordering_quarantined -x` | ❌ Wave 0 |
| DATA-05 | Row with >50% price spike quarantined | unit | `uv run pytest tests/data/test_validation.py::test_price_spike_quarantined -x` | ❌ Wave 0 |
| DATA-05 | Duplicate timestamps deduplicated (batch-level check) | unit | `uv run pytest tests/data/test_validation.py::test_duplicate_dedup -x` | ❌ Wave 0 |
| DATA-05 | Gap on NYSE trading day logged as warning (not quarantined) | unit | `uv run pytest tests/data/test_validation.py::test_equity_gap_logged -x` | ❌ Wave 0 |
| DATA-05 | Stale data (max timestamp >2 trading days old) raises DataError | unit | `uv run pytest tests/data/test_validation.py::test_stale_data_error -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/data/ -v -m "not integration" -x`
- **Per wave merge:** `uv run pytest tests/ -v -m "not integration"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/data/__init__.py` — package init
- [ ] `tests/data/test_alpaca_ingestor.py` — covers DATA-01
- [ ] `tests/data/test_binance_ingestor.py` — covers DATA-02, DATA-03
- [ ] `tests/data/test_fred_ingestor.py` — covers DATA-04
- [ ] `tests/data/test_validation.py` — covers DATA-05
- [ ] `tests/data/test_parquet_store.py` — covers upsert logic (shared by all ingestors)
- [ ] `tests/data/fixtures/alpaca_bars_spy.json` — Alpaca API mock response
- [ ] `tests/data/fixtures/binance_klines_btcusdt.json` — Binance live API mock response
- [ ] `tests/data/fixtures/binance_archive_btcusdt_4h.csv` — ~100 rows, 2018 (ms) + 2025 (us) rows for timestamp unit tests
- [ ] `tests/data/fixtures/fred_cpiaucsl_releases.json` — FRED all-releases mock response
- [ ] Framework install: `uv add "fredapi>=0.5" "exchange_calendars>=4.13" "responses>=0.25"` — not yet in pyproject.toml

---

## Sources

### Primary (HIGH confidence)
- alpaca-py official docs `https://docs.alpaca.markets/reference/stockbars` — StockBarsRequest parameters, feed options, next_page_token pagination
- Binance.US REST API docs `https://docs.binance.us/` — klines endpoint weight=1, rate limit 1200/min, X-MBX-USED-WEIGHT header
- Binance Spot API klines `https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints` — response column order, max 1000 per request
- FRED series/observations `https://fred.stlouisfed.org/docs/api/fred/series_observations.html` — vintage_dates, realtime_start/end parameters
- fredapi README `https://github.com/mortada/fredapi` — get_series_all_releases(), get_series_as_of_date() API
- exchange_calendars README `https://github.com/gerrymanoim/exchange_calendars` — is_session(), sessions_in_range() API, version 4.13.1
- pyarrow Parquet docs `https://arrow.apache.org/docs/python/parquet.html` — write_table, read_table, compression options

### Secondary (MEDIUM confidence)
- Binance public data announcement (via WebSearch) — confirmed timestamp unit change to microseconds from 2025-01-01 for SPOT archive data
- responses library `https://github.com/getsentry/responses` — version 0.25.8, @responses.activate decorator, add() API
- alpaca-py SDK forum posts — confirmed multi-symbol pagination behavior (sorted by symbol then timestamp, requires multiple next_page_token requests)

### Tertiary (LOW confidence)
- fredapi 0.5.2 version: PyPI listing verified but maintenance status post-August 2025 not confirmed. Treat as current unless integration test reveals issues.
- Binance.US 4H archive availability on data.binance.vision: confirmed the URL pattern exists for spot klines but specific 4H interval availability back to 2017 for BTCUSDT not directly verified against a live URL.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — alpaca-py already in pyproject.toml; fredapi and exchange_calendars are well-established; responses matches the requests-based client pattern
- Architecture: HIGH — locked decisions in CONTEXT.md are concrete; patterns directly map to the locked decisions with no ambiguity
- Pitfalls: HIGH for critical items (Binance timestamp unit change, Binance.US base URL, FRED look-ahead bias); MEDIUM for minor operational issues
- 12-step validation checklist: MEDIUM — specific check ordering is Claude's Discretion; the enumerated checks in CONTEXT.md plus reasonable additions cover DATA-05 success criteria

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable APIs; Binance.US rate limits and exchange_calendars holiday data update more frequently)
