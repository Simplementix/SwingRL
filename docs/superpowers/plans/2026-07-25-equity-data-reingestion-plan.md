# Equity Data Re-Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CBOE the single source of equity daily bars, stored raw and never adjusted, replacing a two-feed patchwork that is missing 18 sessions and under-reports volume by 13–47×.

**Architecture:** A new `CboeBarsIngestor` follows the existing `BaseIngestor` fetch→validate→store→sync contract, reading deep history from CBOE's `charts/historical` endpoint and the just-closed session from `quotes`. Prices are stored raw; `corporate_actions` is populated separately for reference only and never adjusts a price. The collector's existing `candles_equity_job` is repointed from Alpaca to CBOE so contamination stops arriving nightly.

**Tech Stack:** Python 3.11 · pandas · psycopg 3 · pydantic v2 · structlog · APScheduler · pytest · PostgreSQL 16

**Spec:** `docs/superpowers/specs/2026-07-25-equity-data-reingestion-design.md`
**Branch:** `swingrl/26-equity-data-reingestion` (already cut from `swingrl/2.R-training-redesign`)

## Global Constraints

- Python 3.11 only. `from __future__ import annotations` at the top of every module.
- Type hints on **all** function signatures (`disallow_untyped_defs = true`).
- Line length 100 (`ruff` + `black`). Lockfile ruff is CI authority: `uv run ruff check src/ tests/`.
- Absolute first-party imports only — never `from .x import` inside `src/swingrl/`.
- `pathlib.Path` for all file operations. Never `os.path`.
- structlog with **keyword args**, never f-strings: `log.info("event", symbol=sym, rows=n)`.
- Raise typed `SwingRLError` subclasses (`DataError` here). Never bare `Exception`/`ValueError`.
- **UTC internally.** ET only at the display edge.
- No hardcoded symbols, paths, URLs or amounts — everything through `SwingRLConfig`.
- **Migrations are additive only** while the trader runs (A30). No destructive DDL.
- **Never `--no-verify`.** detect-secrets false positives go in `.secrets.baseline`.
- Test naming: `tests/<pkg>/test_<module>.py`, `test_<behavior>`, docstring `"""REQ-ID: what."""`.
- Fast lane before every commit: `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
- **LD-4 is absolute: no code in this plan may adjust a price for dividends or splits.**

---

## File Structure

| File | Responsibility |
|---|---|
| `src/swingrl/data/cboe_bars.py` **(create)** | `CboeBarsIngestor` — fetch/validate/store equity daily bars from CBOE |
| `src/swingrl/data/corporate_actions_ingest.py` **(create)** | Pull dividends from Alpaca into `corporate_actions`. Reference only |
| `scripts/reingest_equity_history.py` **(create)** | One-off full replacement, with `--dry-run` and `--apply` |
| `src/swingrl/data/base.py` **(modify)** | `_sync_to_db` gains an overridable conflict clause so a wrong row can be corrected |
| `src/swingrl/data/validation.py` **(modify)** | Add the flat-bar rule (`o==h==l==c`) that let 2018-11-01 through |
| `src/swingrl/config/schema.py` **(modify)** | `EquityBarsConfig` block |
| `config/swingrl.yaml` **(modify)** | CBOE equity-bar settings; correct the stale "options-only" comment |
| `scripts/collector_main.py` **(modify)** | Repoint `candles_equity_job`; correct the stale comment |
| `src/swingrl/data/ingest_all.py` **(modify)** | `run_equity` uses the CBOE ingestor |

**Test files mirror each:** `tests/data/test_cboe_bars.py`, `tests/data/test_corporate_actions_ingest.py`, plus additions to `tests/data/test_validation.py` and `tests/data/test_base_ingestor.py`.

---

## Key facts the implementer needs

Measured on 2026-07-24/25 against the live system. Do not re-derive these.

- **CBOE historical:** `https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json`
  → `{"timestamp": ..., "symbol": ..., "data": [{"date": "2004-01-02", "open": 111.74, "high": 112.19, "low": 110.73, "close": 111.23, "volume": 38072300}, ...]}`
  ~5,676 rows per symbol, 2004-01-02 → **T-2**. No auth.
- **CBOE quotes:** `https://cdn.cboe.com/api/global/delayed_quotes/quotes/{symbol}.json`
  → `data` contains `open`, `high`, `low`, `close`, `volume`, `last_trade_time` (e.g. `"2026-07-24T16:00:00"`). This is the **only** endpoint carrying the just-closed session. No auth.
- **`BaseIngestor` contract** (`src/swingrl/data/base.py`): subclasses set `_environment` and `_duckdb_table` class attrs and implement `fetch(symbol, since) -> DataFrame`, `validate(df, symbol) -> tuple[DataFrame, DataFrame]`, `store(df, symbol) -> Path`. `run()` orchestrates fetch→validate→store→`_sync_to_db`→`_log_ingestion`.
- **`_sync_to_db` currently hardcodes `on_conflict="DO NOTHING"`** (`base.py:191`) — it cannot correct an existing wrong row. Task 3 fixes this.
- **`executemany_from_df(conn, table, df, columns, *, on_conflict="DO NOTHING") -> int`** (`pg_helpers.py:52`). The conflict clause is appended verbatim after `ON CONFLICT`, e.g. `"(symbol, date) DO UPDATE SET open=EXCLUDED.open"`.
- **`_build_sync_df`** (`base.py:199`) already maps `ohlcv_daily` → `["symbol","date","open","high","low","close","volume"]` from a DatetimeIndex. Reuse it.
- **`DataValidator(source="equity").validate_rows(df, symbol)`** already checks price ≤ 0, negative volume, `high < low`, open/close outside bounds (`_TOLERANCE = 0.0001`), and zero volume on trading days. **It does not check flat bars** — that is why SPY 2018-11-01 (o=h=l=c=242.68, volume 200) survived. Task 5 adds it.
- **The bad 2024 SPY bars would have been caught** by the existing Step 5 check, so they bypassed validation entirely — most likely via the Postgres migration. Re-ingesting through `run()` fixes them by construction.
- **Fixtures** in `tests/conftest.py`: `tmp_config`, `loaded_config`, `equity_ohlcv`, `crypto_ohlcv`.

---

### Task 1: Config block for CBOE equity bars

**Files:**
- Modify: `src/swingrl/config/schema.py`
- Modify: `config/swingrl.yaml`
- Test: `tests/config/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.equity_bars` with fields `enabled: bool`, `historical_url_template: str`, `quotes_url_template: str`, `resync_lookback_days: int`, `request_timeout_s: float`.

- [ ] **Step 1: Write the failing test**

```python
def test_equity_bars_config_defaults(loaded_config: SwingRLConfig) -> None:
    """LD-3: CBOE equity-bar settings load with usable defaults."""
    eb = loaded_config.equity_bars
    assert eb.enabled is True
    assert "{symbol}" in eb.historical_url_template
    assert "{symbol}" in eb.quotes_url_template
    assert eb.resync_lookback_days >= 3
    assert eb.request_timeout_s > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/config/test_schema.py::test_equity_bars_config_defaults -v`
Expected: FAIL — `AttributeError: 'SwingRLConfig' object has no attribute 'equity_bars'`

- [ ] **Step 3: Add the model to `schema.py`**

```python
class EquityBarsConfig(BaseModel):
    """CBOE equity daily-bar ingestion settings (LD-3: CBOE is the sole bar source)."""

    enabled: bool = Field(default=True)
    historical_url_template: str = Field(
        default="https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json"
    )
    quotes_url_template: str = Field(
        default="https://cdn.cboe.com/api/global/delayed_quotes/quotes/{symbol}.json"
    )
    resync_lookback_days: int = Field(default=5, ge=3)
    request_timeout_s: float = Field(default=30.0, gt=0)
```

Then add to `SwingRLConfig`: `equity_bars: EquityBarsConfig = Field(default_factory=EquityBarsConfig)`

- [ ] **Step 4: Add the YAML block to `config/swingrl.yaml`**

```yaml
equity_bars:                        # LD-3 (2026-07-25): CBOE is the sole equity bar source.
  enabled: true                     # charts/historical for depth, quotes for the just-closed session.
  historical_url_template: "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json"
  quotes_url_template: "https://cdn.cboe.com/api/global/delayed_quotes/quotes/{symbol}.json"
  resync_lookback_days: 5           # re-sync recent bars once charts/historical publishes (T+2)
  request_timeout_s: 30.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `env -u DATABASE_URL uv run pytest tests/config/test_schema.py::test_equity_bars_config_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/config/schema.py config/swingrl.yaml tests/config/test_schema.py
git commit -m "feat(config): CBOE equity-bar settings (LD-3)"
```

---

### Task 2: CBOE payload parsing

**Files:**
- Create: `src/swingrl/data/cboe_bars.py`
- Test: `tests/data/test_cboe_bars.py`

**Interfaces:**
- Consumes: `config.equity_bars` from Task 1.
- Produces: `parse_historical(payload: dict) -> pd.DataFrame` and `parse_quote(payload: dict) -> pd.DataFrame`. Both return OHLCV columns `["open","high","low","close","volume"]` on a **UTC-normalised `DatetimeIndex`**, matching what `_build_sync_df` expects.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import pandas as pd
import pytest

from swingrl.data.cboe_bars import parse_historical, parse_quote
from swingrl.utils.exceptions import DataError


def test_parse_historical_returns_ohlcv_on_utc_index() -> None:
    """LD-3: CBOE historical payload parses to an OHLCV frame on a UTC DatetimeIndex."""
    payload = {
        "symbol": "SPY",
        "data": [
            {"date": "2004-01-02", "open": 111.74, "high": 112.19,
             "low": 110.73, "close": 111.23, "volume": 38072300},
            {"date": "2004-01-05", "open": 111.5, "high": 113.0,
             "low": 111.4, "close": 112.9, "volume": 40000000},
        ],
    }
    df = parse_historical(payload)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2
    assert df.iloc[0]["close"] == pytest.approx(111.23)


def test_parse_historical_rejects_empty_payload() -> None:
    """LD-3: an empty CBOE payload raises rather than silently returning nothing."""
    with pytest.raises(DataError, match="no data"):
        parse_historical({"symbol": "SPY", "data": []})


def test_parse_quote_uses_last_trade_time_as_the_bar_date() -> None:
    """LD-3: the quotes endpoint bar is dated by last_trade_time, not by wall clock."""
    payload = {
        "symbol": "SPY",
        "data": {
            "open": 738.47, "high": 743.72, "low": 737.29, "close": 738.93,
            "volume": 44743922, "last_trade_time": "2026-07-24T16:00:00",
        },
    }
    df = parse_quote(payload)
    assert len(df) == 1
    assert df.index[0].date().isoformat() == "2026-07-24"
    assert df.iloc[0]["volume"] == 44743922


def test_parse_quote_rejects_missing_last_trade_time() -> None:
    """LD-3: without last_trade_time the bar cannot be dated, so it must not be stored."""
    with pytest.raises(DataError, match="last_trade_time"):
        parse_quote({"symbol": "SPY", "data": {"open": 1.0, "high": 1.0,
                                               "low": 1.0, "close": 1.0, "volume": 1}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swingrl.data.cboe_bars'`

- [ ] **Step 3: Write the parsers**

```python
"""CBOE equity daily-bar parsing and ingestion (LD-3).

Prices are stored RAW and are never adjusted for dividends or splits (LD-4).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def parse_historical(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a CBOE charts/historical payload into an OHLCV frame.

    Args:
        payload: Decoded JSON from the charts/historical endpoint.

    Returns:
        OHLCV DataFrame indexed by UTC-normalised bar date.

    Raises:
        DataError: If the payload carries no rows.
    """
    rows = payload.get("data") or []
    if not rows:
        raise DataError(f"CBOE historical payload has no data for {payload.get('symbol')}")

    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame["date"], utc=True)
    frame.index.name = None
    result: pd.DataFrame = frame[_OHLCV_COLUMNS].astype(float)
    return result


def parse_quote(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a CBOE quotes payload into a single-row OHLCV frame.

    The bar is dated by ``last_trade_time`` so a stale payload can never be
    written under today's date.

    Args:
        payload: Decoded JSON from the quotes endpoint.

    Returns:
        Single-row OHLCV DataFrame indexed by the session date (UTC).

    Raises:
        DataError: If ``last_trade_time`` is absent.
    """
    data = payload.get("data") or {}
    last_trade = data.get("last_trade_time")
    if not last_trade:
        raise DataError(f"CBOE quote for {payload.get('symbol')} has no last_trade_time")

    bar_date = pd.to_datetime(last_trade).normalize().tz_localize("UTC")
    result = pd.DataFrame(
        [[float(data[col]) for col in _OHLCV_COLUMNS]],
        columns=_OHLCV_COLUMNS,
        index=pd.DatetimeIndex([bar_date]),
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/cboe_bars.py tests/data/test_cboe_bars.py
git commit -m "feat(data): CBOE bar payload parsing (LD-3)"
```

---

### Task 3: Upsert support in `_sync_to_db`

**Files:**
- Modify: `src/swingrl/data/base.py:164-197`
- Test: `tests/data/test_base_ingestor.py`

**Interfaces:**
- Consumes: `executemany_from_df(..., on_conflict=...)`.
- Produces: class attribute `_sync_conflict_clause: str = "DO NOTHING"` on `BaseIngestor`, overridable by subclasses. `_sync_to_db` passes it through.

**Why:** `_sync_to_db` hardcodes `"DO NOTHING"` (`base.py:191`), so re-ingesting a corrected bar over a wrong one **silently discards the correction**. Without this task the entire re-ingestion is a no-op for every existing row.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.db
def test_sync_to_db_upserts_when_conflict_clause_overridden(db_conn) -> None:
    """LD-2: a subclass opting into upsert corrects an existing wrong row."""
    from swingrl.data.base import BaseIngestor

    assert BaseIngestor._sync_conflict_clause == "DO NOTHING"

    clause = (
        "(symbol, date) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, "
        "low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume"
    )
    db_conn.execute(
        "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
        "VALUES ('TEST', '2024-01-02', 1, 1, 1, 1, 1)"
    )
    df = pd.DataFrame(
        [["TEST", date(2024, 1, 2), 10.0, 11.0, 9.0, 10.5, 999]],
        columns=["symbol", "date", "open", "high", "low", "close", "volume"],
    )
    executemany_from_df(
        db_conn, "ohlcv_daily", df,
        ["symbol", "date", "open", "high", "low", "close", "volume"],
        on_conflict=clause,
    )
    row = db_conn.execute(
        "SELECT close, volume FROM ohlcv_daily WHERE symbol='TEST' AND date='2024-01-02'"
    ).fetchone()
    assert row[0] == pytest.approx(10.5)
    assert row[1] == 999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_base_ingestor.py::test_sync_to_db_upserts_when_conflict_clause_overridden -v -m db`
Expected: FAIL — `AttributeError: type object 'BaseIngestor' has no attribute '_sync_conflict_clause'`

- [ ] **Step 3: Add the overridable clause**

In `base.py`, alongside the other class attributes:

```python
    _sync_conflict_clause: str = "DO NOTHING"
```

Then in `_sync_to_db`, replace the hardcoded argument:

```python
                inserted = executemany_from_df(
                    conn, table, sync_df, columns, on_conflict=self._sync_conflict_clause
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_base_ingestor.py -v -m db`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/base.py tests/data/test_base_ingestor.py
git commit -m "feat(data): overridable ON CONFLICT clause so corrections can land"
```

---

### Task 4: Interior-gap-aware start resolution

**Files:**
- Modify: `src/swingrl/data/cboe_bars.py`
- Test: `tests/data/test_cboe_bars.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `find_missing_sessions(have: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex`.

**Why:** `alpaca.py:218-240` resumes from `max(existing) + 1 day`, which cannot see a hole behind the watermark. That is precisely why 18 sessions stayed missing for four months. **Do not reproduce that pattern.**

- [ ] **Step 1: Write the failing test**

```python
def test_find_missing_sessions_detects_an_interior_gap() -> None:
    """A-2: a hole BEHIND the newest bar must be detected, not just the trailing edge."""
    from swingrl.data.cboe_bars import find_missing_sessions

    have = pd.DatetimeIndex(
        ["2026-03-09", "2026-03-10", "2026-04-07", "2026-04-08"], tz="UTC"
    ).normalize()
    missing = find_missing_sessions(have, "2026-03-09", "2026-04-08")
    assert pd.Timestamp("2026-03-11", tz="UTC") in missing
    assert pd.Timestamp("2026-04-06", tz="UTC") in missing
    assert pd.Timestamp("2026-03-14", tz="UTC") not in missing  # Saturday
    assert pd.Timestamp("2026-03-10", tz="UTC") not in missing  # already held
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py::test_find_missing_sessions_detects_an_interior_gap -v`
Expected: FAIL — `ImportError: cannot import name 'find_missing_sessions'`

- [ ] **Step 3: Implement against the exchange calendar**

```python
def find_missing_sessions(
    have: pd.DatetimeIndex, start: str, end: str
) -> pd.DatetimeIndex:
    """Return NYSE sessions in [start, end] that are absent from ``have``.

    Compares against the full session list rather than resuming from the newest
    bar, so gaps behind the watermark are found (A-2).

    Args:
        have: Bar dates currently held, any timezone.
        start: ISO start date, inclusive.
        end: ISO end date, inclusive.

    Returns:
        UTC-normalised DatetimeIndex of missing sessions, ascending.
    """
    import exchange_calendars as xcals  # noqa: PLC0415

    sessions = pd.DatetimeIndex(
        xcals.get_calendar("XNYS").sessions_in_range(start, end)
    ).normalize().tz_localize("UTC")
    held = have.normalize()
    if held.tz is None:
        held = held.tz_localize("UTC")
    else:
        held = held.tz_convert("UTC")
    return sessions.difference(held)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/cboe_bars.py tests/data/test_cboe_bars.py
git commit -m "feat(data): interior-gap detection for bar coverage (A-2)"
```

---

### Task 5: Flat-bar validation rule

**Files:**
- Modify: `src/swingrl/data/validation.py` (inside `validate_rows`, after Step 7)
- Test: `tests/data/test_validation.py`

**Interfaces:**
- Consumes: existing `DataValidator.validate_rows`.
- Produces: no new public symbol — one additional quarantine reason, `"Step 8: flat bar"`.

**Why:** SPY 2018-11-01 (`o=h=l=c=242.68`, volume 200) passed every existing check. Step 7 only catches volume **exactly** zero.

- [ ] **Step 1: Write the failing test**

```python
def test_validate_rows_quarantines_a_flat_bar() -> None:
    """A-6: a bar with open==high==low==close is not a real session."""
    df = pd.DataFrame(
        {"open": [242.68, 100.0], "high": [242.68, 101.0], "low": [242.68, 99.0],
         "close": [242.68, 100.5], "volume": [200, 5_000_000]},
        index=pd.to_datetime(["2018-11-01", "2018-11-02"], utc=True),
    )
    clean, quarantine = DataValidator(source="equity").validate_rows(df, "SPY")
    assert len(clean) == 1
    assert len(quarantine) == 1
    assert "flat bar" in quarantine.iloc[0]["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_validation.py::test_validate_rows_quarantines_a_flat_bar -v`
Expected: FAIL — `assert 2 == 1` (both rows currently pass)

- [ ] **Step 3: Add the rule**

In `validate_rows`, after the existing Step 7 block:

```python
        # Step 8: flat bar — open == high == low == close is not a real session (A-6)
        flat = (
            df["open"].eq(df["high"])
            & df["high"].eq(df["low"])
            & df["low"].eq(df["close"])
            & df[price_cols].notna().all(axis=1)
        )
        _flag(flat, "Step 8: flat bar")
```

- [ ] **Step 4: Run the validation suite**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_validation.py -v`
Expected: PASS. If a pre-existing test breaks, the fixture legitimately contains a flat bar — fix the fixture, not the rule.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/validation.py tests/data/test_validation.py
git commit -m "feat(data): quarantine flat bars (A-6)"
```

---

### Task 6: `CboeBarsIngestor`

**Files:**
- Modify: `src/swingrl/data/cboe_bars.py`
- Test: `tests/data/test_cboe_bars.py`

**Interfaces:**
- Consumes: `parse_historical`, `parse_quote`, `find_missing_sessions` (Tasks 2 and 4); `_sync_conflict_clause` (Task 3).
- Produces: `CboeBarsIngestor(config)` with `_environment = "equity"`, `_duckdb_table = "ohlcv_daily"`, and `fetch(symbol, since)`, `validate(df, symbol)`, `store(df, symbol)`. `since="quote"` reads the just-closed session; anything else reads full history.

- [ ] **Step 1: Write the failing test**

```python
def test_ingestor_fetches_history_and_upserts(loaded_config, monkeypatch) -> None:
    """LD-2/LD-3: the CBOE ingestor reads history and opts into upsert semantics."""
    from swingrl.data.cboe_bars import CboeBarsIngestor

    payload = {"symbol": "SPY", "data": [
        {"date": "2004-01-02", "open": 111.74, "high": 112.19,
         "low": 110.73, "close": 111.23, "volume": 38072300}]}
    monkeypatch.setattr(
        "swingrl.data.cboe_bars.CboeBarsIngestor._get_json", lambda self, url: payload
    )
    ing = CboeBarsIngestor(loaded_config)
    assert ing._environment == "equity"
    assert ing._duckdb_table == "ohlcv_daily"
    assert "DO UPDATE SET" in ing._sync_conflict_clause  # corrections must land
    df = ing.fetch("SPY", since=None)
    assert len(df) == 1
    assert df.iloc[0]["close"] == pytest.approx(111.23)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py::test_ingestor_fetches_history_and_upserts -v`
Expected: FAIL — `ImportError: cannot import name 'CboeBarsIngestor'`

- [ ] **Step 3: Implement the ingestor**

```python
class CboeBarsIngestor(BaseIngestor):
    """Ingest equity daily bars from CBOE. Prices are stored RAW (LD-4)."""

    _environment = "equity"
    _duckdb_table = "ohlcv_daily"
    _sync_conflict_clause = (
        "(symbol, date) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, "
        "low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume"
    )

    def __init__(self, config: SwingRLConfig) -> None:
        super().__init__(config)
        self._settings = config.equity_bars
        self._store_ = ParquetStore()

    def _get_json(self, url: str) -> dict[str, Any]:
        """Fetch and decode a CBOE JSON payload.

        Args:
            url: Fully-formed endpoint URL.

        Returns:
            Decoded payload.

        Raises:
            DataError: On any transport or decode failure.
        """
        try:
            resp = httpx.get(url, timeout=self._settings.request_timeout_s)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
            return payload
        except Exception as exc:
            raise DataError(f"CBOE request failed for {url}: {exc}") from exc

    def fetch(self, symbol: str, since: str | None = None) -> pd.DataFrame:
        """Fetch bars for a symbol.

        Args:
            symbol: Ticker symbol.
            since: ``"quote"`` for the just-closed session; anything else for
                full history.

        Returns:
            OHLCV DataFrame on a UTC DatetimeIndex.
        """
        if since == "quote":
            url = self._settings.quotes_url_template.format(symbol=symbol)
            return parse_quote(self._get_json(url))
        url = self._settings.historical_url_template.format(symbol=symbol)
        return parse_historical(self._get_json(url))

    def validate(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate via the shared equity validator.

        Args:
            df: Raw OHLCV frame.
            symbol: Ticker symbol for logging context.

        Returns:
            Tuple of (clean, quarantine).
        """
        validator = DataValidator(source="equity")
        clean, quarantine = validator.validate_rows(df, symbol)
        clean = validator.validate_batch(clean, symbol)
        return clean, quarantine

    def store(self, df: pd.DataFrame, symbol: str) -> Path:
        """Upsert into the equity Parquet file.

        Args:
            df: Validated OHLCV frame.
            symbol: Ticker symbol.

        Returns:
            Path written.
        """
        path = self._data_dir / "equity" / f"{symbol}_daily.parquet"
        self._store_.upsert(path, df)
        return path
```

Add the imports this needs at the top of the module: `httpx`, `Path`, `Any`, `BaseIngestor`, `ParquetStore`, `DataValidator`, `SwingRLConfig`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_cboe_bars.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/cboe_bars.py tests/data/test_cboe_bars.py
git commit -m "feat(data): CboeBarsIngestor — raw equity bars, upsert semantics (LD-3/LD-4)"
```

---

### Task 7: Repoint `run_equity` and the collector job

**Files:**
- Modify: `src/swingrl/data/ingest_all.py:70-102`
- Modify: `scripts/collector_main.py:303-333` and the comment at `:487`
- Modify: `config/swingrl.yaml:116` (comment only)
- Test: `tests/data/test_ingest_all.py`

**Interfaces:**
- Consumes: `CboeBarsIngestor` (Task 6).
- Produces: `run_equity(config, backfill)` unchanged in signature, now backed by CBOE. `backfill=False` → `since="quote"`; `backfill=True` → full history.

**Why this task matters most:** `candles_equity_job` runs at **20:15 ET Mon–Fri** and currently calls the Alpaca IEX path. It is how the contamination arrives nightly. Until this lands, every repair is undone the next evening.

- [ ] **Step 1: Write the failing test**

```python
def test_run_equity_uses_cboe_not_alpaca(loaded_config, monkeypatch) -> None:
    """LD-3: equity ingestion no longer touches Alpaca."""
    import swingrl.data.ingest_all as ia

    used: list[str] = []

    class _Spy:
        def __init__(self, config): ...
        def run_all(self, symbols, since=None):
            used.append(type(self).__name__)
            return []

    monkeypatch.setattr(ia, "CboeBarsIngestor", _Spy)
    monkeypatch.setattr(ia, "_count_rows", lambda db, table: 0)
    monkeypatch.setattr(ia, "DatabaseManager", lambda config: object())
    ia.run_equity(loaded_config, backfill=False)
    assert used == ["_Spy"]
    assert not hasattr(ia, "AlpacaIngestor")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_ingest_all.py::test_run_equity_uses_cboe_not_alpaca -v`
Expected: FAIL — `AlpacaIngestor` still imported and used.

- [ ] **Step 3: Swap the ingestor in `ingest_all.py`**

Replace the `AlpacaIngestor` import with `from swingrl.data.cboe_bars import CboeBarsIngestor`, and inside `run_equity`:

```python
    ingestor = CboeBarsIngestor(config)
    since: str | None = None if backfill else "quote"
    failed = ingestor.run_all(config.equity.symbols, since=since)
```

- [ ] **Step 4: Correct the two stale comments**

`config/swingrl.yaml:116` — replace `existing Alpaca/Binance ingestors, CBOE stays options-only` with `equity bars from CBOE (LD-3 2026-07-25); crypto stays on Binance`.
`scripts/collector_main.py:487` — replace `Existing Alpaca/Binance ingestors — CBOE stays options-only.` with `Equity bars from CBOE (LD-3 2026-07-25); crypto stays on Binance; options unchanged.`

Also update the `candles_equity_job` docstring at `:303` to say the source is CBOE.

- [ ] **Step 5: Run tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_ingest_all.py tests/data/test_cboe_bars.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/data/ingest_all.py scripts/collector_main.py config/swingrl.yaml tests/data/test_ingest_all.py
git commit -m "feat(collector): equity candles from CBOE, retiring the Alpaca IEX path (LD-3)"
```

---

### Task 8: Corporate-actions ingestion

**Files:**
- Create: `src/swingrl/data/corporate_actions_ingest.py`
- Test: `tests/data/test_corporate_actions_ingest.py`

**Interfaces:**
- Consumes: Alpaca credentials already present in the environment.
- Produces: `fetch_cash_dividends(config, start, end) -> pd.DataFrame` with columns `["action_id","symbol","action_type","effective_date","ratio","amount","processed"]`, and `store_actions(db, df) -> int`.

**Reference only — LD-4 forbids using these to adjust any price.** Verified live: 334 dividends across all 8 symbols, 2016-03-15 → 2026-06-22. Alpaca returns **zero** spin-offs, so A-7 stays open.

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_cash_dividends_maps_to_corporate_actions_schema(monkeypatch, loaded_config) -> None:
    """G1-5: Alpaca dividends map onto the corporate_actions columns."""
    from swingrl.data import corporate_actions_ingest as cai

    payload = {"corporate_actions": {"cash_dividends": [
        {"id": "abc", "symbol": "SPY", "ex_date": "2026-03-20",
         "rate": 1.40, "special": False}]}}
    monkeypatch.setattr(cai, "_get_json", lambda url, key, secret, timeout: payload)
    df = cai.fetch_cash_dividends(loaded_config, "2016-01-01", "2026-07-24")
    assert list(df.columns) == [
        "action_id", "symbol", "action_type", "effective_date", "ratio", "amount", "processed"]
    assert df.iloc[0]["action_type"] == "dividend"
    assert df.iloc[0]["amount"] == pytest.approx(1.40)
    assert df.iloc[0]["effective_date"] == "2026-03-20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_corporate_actions_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Map `id`→`action_id`, `symbol`→`symbol`, literal `"dividend"`→`action_type`, `ex_date`→`effective_date`, `None`→`ratio`, `rate`→`amount`, `0`→`processed`. Paginate on `next_page_token`. Store via `executemany_from_df(conn, "corporate_actions", df, columns, on_conflict="(action_id) DO NOTHING")`.

Endpoint: `https://data.alpaca.markets/v1/corporate-actions?symbols={csv}&types=cash_dividend&start={start}&end={end}&limit=1000`, headers `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`.

**Note:** `corporate_actions.action_id` is the PRIMARY KEY (`postgres_schema.py:446`), so the conflict target is `(action_id)`, not `(symbol, date)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_corporate_actions_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/corporate_actions_ingest.py tests/data/test_corporate_actions_ingest.py
git commit -m "feat(data): ingest Alpaca cash dividends into corporate_actions (G1-5)"
```

---

### Task 9: One-off re-ingestion script

**Files:**
- Create: `scripts/reingest_equity_history.py`
- Test: `tests/scripts/test_reingest_equity_history.py`

**Interfaces:**
- Consumes: `CboeBarsIngestor`, `find_missing_sessions`, `fetch_cash_dividends`.
- Produces: CLI `--dry-run` (default) / `--apply`, `--symbols`, `--start`.

**Follow the established pattern in `scripts/reanchor_benchmark_baselines.py`:** dry-run prints before/after rows and exits without writing; `--apply` is required to write.

- [ ] **Step 1: Write the failing test**

```python
def test_dry_run_writes_nothing(monkeypatch, loaded_config, capsys) -> None:
    """LD-2: dry-run is the default and must not write."""
    from scripts import reingest_equity_history as r

    wrote: list[str] = []
    monkeypatch.setattr(r, "_apply", lambda *a, **k: wrote.append("wrote"))
    r.main(["--config", "config/swingrl.yaml"])
    assert wrote == []
    assert "DRY RUN" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u DATABASE_URL uv run pytest tests/scripts/test_reingest_equity_history.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

The `--apply` path, in order: `pg_dump` guard check (refuse to run if `--backup-taken` is absent), then per symbol `ingestor.run(symbol, since=None)`, then `fetch_cash_dividends` → `store_actions`, then report `find_missing_sessions` for each symbol as a post-check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/scripts/test_reingest_equity_history.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/reingest_equity_history.py tests/scripts/test_reingest_equity_history.py
git commit -m "feat(scripts): one-off equity history re-ingestion (LD-2)"
```

---

### Task 10: Execute the replacement (gated — requires user approval)

**No code. This is the operational run. STOP and get explicit approval before any step.**

- [ ] **Step 1: Back up**

```bash
docker exec pg16 pg_dump -U swingrl -d swingrl \
  -t ohlcv_daily -t ohlcv_4h -t features_equity -t features_crypto \
  > /tmp/claude-1000/.../swingrl-pre-reingest-$(date +%Y%m%d-%H%M).sql
```

- [ ] **Step 2: Dry run and read the output**

```bash
docker exec swingrl python -m scripts.reingest_equity_history --config config/swingrl.yaml
```

- [ ] **Step 3: Apply**

```bash
docker exec swingrl python -m scripts.reingest_equity_history --config config/swingrl.yaml --apply --backup-taken
```

- [ ] **Step 4: Fill the crypto 2026 gap only (LD-6)**

```bash
docker exec swingrl python -c "
from swingrl.config.schema import load_config
from swingrl.data.gap_fill import detect_crypto_gaps
for g in detect_crypto_gaps(load_config('config/swingrl.yaml')):
    print(g.symbol, g.gap_start, g.gap_end, g.gap_hours)"
```

Confirm the 2026-03-10 → 2026-04-06 window appears and that **no gap near 2019-09-01 is filled**, then fill only that window. The 2019 seam is out of scope by LD-6.

- [ ] **Step 5: Recompute features**

```bash
docker exec swingrl python scripts/compute_features.py --env equity
```

- [ ] **Step 6: Run the spec's verification checklist**

Every box in the spec's *Verification checklist* section. Report the measured numbers, not a summary.

---

### Task 11: Full CI and PR

- [ ] **Step 1: Fast lane**

Run: `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
Expected: 0 failures.

- [ ] **Step 2: Lockfile lint (CI authority)**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/`
Expected: clean.

- [ ] **Step 3: Homelab CI** — launch harness-tracked, announce once, never poll

```bash
cd ~/swingrl && git fetch origin && git checkout swingrl/26-equity-data-reingestion \
  && git pull origin swingrl/26-equity-data-reingestion && bash scripts/ci-homelab.sh --no-cache
```

Expected: `=== CI PASSED ===`

- [ ] **Step 4: Open the PR against `swingrl/2.R-training-redesign`** — never `main`.

---

## Self-Review

**Spec coverage**

| Spec item | Task |
|---|---|
| LD-2 full replacement | 3, 6, 9, 10 |
| LD-3 CBOE sole source | 1, 2, 6, 7 |
| LD-4 raw, never adjust | 2, 6 (no adjustment code exists anywhere in this plan) |
| LD-5 2004→present depth | 2, 9 |
| LD-6 crypto 2026 only | 10 Step 4 |
| LD-7 no restart | N/A — nothing in this plan stops the trader |
| Data flow A one-off | 9, 10 |
| Data flow B ongoing daily | 7 |
| T+2 re-sync | 1 (`resync_lookback_days`), 7 |
| Interior-gap watermark | 4 |
| Replace semantics | 3 |
| Validation extensions | 5 |
| G1-5 corporate actions | 8 |
| Stale comments | 7 Step 4 |
| Verification checklist | 10 Step 6, 11 |

**Gap found and closed during review:** the spec's error-handling clause *"`status='success'` must mean rows were written"* had no task. It belongs with G1-1's watermark work rather than here, and is recorded as out of scope in the spec — **no task added, deliberately.**

**Placeholder scan:** none. Task 8 Step 3 and Task 9 Step 3 describe field mappings and ordering in prose rather than full code — acceptable because the exact payload shape, column list, conflict target and CLI pattern are all given, and the reference implementation (`scripts/reanchor_benchmark_baselines.py`) is named.

**Type consistency:** `parse_historical` / `parse_quote` return `pd.DataFrame` with `["open","high","low","close","volume"]` on a UTC `DatetimeIndex` — consumed unchanged by `CboeBarsIngestor.fetch`, then by `_build_sync_df`, which requires exactly that shape. `_sync_conflict_clause` is defined in Task 3 and set in Task 6 with the same name. `find_missing_sessions` has one signature, used in Tasks 4, 9 and 10.
