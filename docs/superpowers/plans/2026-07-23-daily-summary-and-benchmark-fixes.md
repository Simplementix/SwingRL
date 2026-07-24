# Daily Summary & Benchmark Accuracy Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the daily digest (drops equity / shows stale crypto), the same-symbol partial-alert cooldown, re-anchor the Buy & Hold benchmark to the agent's real first-fill prices, and clear a batch of deferred test/tooling/docstring minors.

**Architecture:** Six workstreams on one branch. Group A (production alert/report code in `scheduler/jobs.py` + `monitoring/embeds.py`) drives a trader deploy. Group C is a data-only benchmark re-record delivered as a tested, gated maintenance script (no code path changes, no deploy). Groups B & D are test/tooling/docstring cleanups (merge only). Spec: `docs/superpowers/specs/2026-07-23-daily-summary-and-benchmark-fixes-design.md`.

**Tech Stack:** Python 3.11, PostgreSQL via psycopg, pytest (`-n` xdist, `db` marker), structlog, ruff/black/mypy.

## Global Constraints

- **Python 3.11 only**; `from __future__ import annotations` at the top of every module; type hints on all defs (`disallow_untyped_defs`).
- **No hardcoded values** in business logic — symbols/capital come from `SwingRLConfig`; one-time migration dates are CLI args with documented defaults.
- **UTC internally, ET only for display.** ET = `ZoneInfo("America/New_York")`.
- **Never call broker APIs outside `execution/`.** These tasks don't; they read the DB.
- **TDD:** commit the RED test, then the GREEN implementation. Never `--no-verify`.
- **`DISTINCT ON` is the repo idiom** for latest-per-group (see `V008`).
- **`db`-marked tests hit real PostgreSQL** and `pytest.skip` (visibly) under `env -u DATABASE_URL`; run them against a scratch DB derived from `~/swingrl/.env` (host `pg16` → container IP; never print the password).
- **Lockfile ruff is CI authority:** `uv run ruff check src/ tests/` before push. Line length 100.
- **Branch:** `swingrl/2.R-H-daily-summary-and-benchmark` off `origin/swingrl/2.R-training-redesign`. PR targets `swingrl/2.R-training-redesign` (never `main`).

---

## Task 1: Daily digest — latest-per-env query (#1, D1)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:452-456` (query) and `:474-486` (build loop)
- Test: `tests/scheduler/test_jobs.py` (`TestDailySummaryJob`)

**Interfaces:**
- Produces: `daily_summary_job()` now selects one newest snapshot per env; each `rows` element gains a `timestamp` key (consumed by Task 2).

- [ ] **Step 1: Write failing tests** — append to `class TestDailySummaryJob` in `tests/scheduler/test_jobs.py`:

```python
    def test_latest_per_env_keeps_equity_and_newest_crypto(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: with an OLDER equity row and TWO crypto rows, the digest keeps the
        equity section AND the NEWEST crypto snapshot (not the older one)."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
            rows = [
                ("2026-07-23T13:15:00Z", "equity", 402.0, 300.0, 0.0, 100.0, 402.0, 2.0, 0.0),
                ("2026-07-23T16:05:00Z", "crypto", 48.50, 0.0, 48.5, 0.0, 48.5, 0.5, 0.0),
                ("2026-07-23T20:05:00Z", "crypto", 47.42, 0.0, 47.42, 0.0, 48.5, -1.08, 0.0),
            ]
            for r in rows:
                conn.execute(
                    "INSERT INTO portfolio_snapshots VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT DO NOTHING", r,
                )
        daily_summary_job()
        embed = mock_alerter.send_embed.call_args.args[1]
        names = [f["name"] for f in embed["embeds"][0]["fields"]]
        values = {f["name"]: f["value"] for f in embed["embeds"][0]["fields"]}
        assert any(n.startswith("Equity Value") for n in names)          # equity NOT dropped
        assert "$47.42" in values[next(n for n in names if n.startswith("Crypto Value"))]

    def test_crypto_only_db_omits_equity_without_crashing(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: a DB with only crypto snapshots renders crypto, omits equity, no crash."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
            conn.execute(
                "INSERT INTO portfolio_snapshots VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT DO NOTHING",
                ("2026-07-23T16:05:00Z", "crypto", 48.5, 0.0, 48.5, 0.0, 48.5, 0.5, 0.0),
            )
        daily_summary_job()
        embed = mock_alerter.send_embed.call_args.args[1]
        names = [f["name"] for f in embed["embeds"][0]["fields"]]
        assert not any(n.startswith("Equity Value") for n in names)
        assert any(n.startswith("Crypto Value") for n in names)

    def test_empty_snapshots_returns_early(
        self, job_ctx: JobContext, mock_db: MagicMock, mock_alerter: MagicMock
    ) -> None:
        """DIGEST-D1: an empty portfolio_snapshots table hits the `if not rows` early return."""
        init_emergency_flags(mock_db)
        with mock_db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TIMESTAMPTZ NOT NULL, environment TEXT NOT NULL,
                    total_value DOUBLE PRECISION NOT NULL,
                    equity_value DOUBLE PRECISION, crypto_value DOUBLE PRECISION,
                    cash_balance DOUBLE PRECISION, high_water_mark DOUBLE PRECISION,
                    daily_pnl DOUBLE PRECISION, drawdown_pct DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, environment)
                )
            """)
            conn.execute("DELETE FROM portfolio_snapshots")
        daily_summary_job()
        mock_alerter.send_embed.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestDailySummaryJob::test_latest_per_env_keeps_equity_and_newest_crypto" -v`
Expected: FAIL — equity section absent / crypto shows the older `$48.50`.

- [ ] **Step 3: Fix the query** — replace `jobs.py:452-456`:

```python
            rows = conn.execute(
                "SELECT DISTINCT ON (environment) "
                "environment, total_value, cash_balance, daily_pnl, drawdown_pct, timestamp "
                "FROM portfolio_snapshots "
                "ORDER BY environment, timestamp DESC"
            ).fetchall()
```

- [ ] **Step 4: Capture the per-env timestamp in the build loop** — replace `jobs.py:474-486` (the `# Build snapshots per environment` block):

```python
        # Build snapshots per environment (DISTINCT ON already gives ≤1 newest row per env).
        equity_snap = None
        crypto_snap = None
        equity_as_of = None
        crypto_as_of = None
        for row in rows:
            snap = {
                "total_value": row["total_value"],
                "daily_pnl": row["daily_pnl"],
                "cash_balance": row["cash_balance"],
            }
            if row["environment"] == "equity":
                equity_snap = snap
                equity_as_of = row["timestamp"]
            elif row["environment"] == "crypto":
                crypto_snap = snap
                crypto_as_of = row["timestamp"]
```

- [ ] **Step 5: Run to verify pass**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestDailySummaryJob" -v`
Expected: PASS (new tests + existing digest tests).

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(digest): latest snapshot per environment (DISTINCT ON) — restores equity section"
```

---

## Task 2: Daily digest — staleness marker (#19, D19)

**Files:**
- Modify: `src/swingrl/monitoring/embeds.py` (`build_daily_summary_embed` + new helper)
- Modify: `src/swingrl/scheduler/jobs.py:488-496` (pass `equity_as_of`/`crypto_as_of`)
- Test: `tests/monitoring/test_embeds.py`

**Interfaces:**
- Consumes: `equity_as_of` / `crypto_as_of` (`datetime | None`) from Task 1.
- Produces: `build_daily_summary_embed(..., equity_as_of=None, crypto_as_of=None)`; a section whose snapshot's ET date ≠ today ET gets an "(as of YYYY-MM-DD)" label suffix.

- [ ] **Step 1: Write the failing test** — append to `tests/monitoring/test_embeds.py`:

```python
def test_daily_summary_marks_stale_section() -> None:
    """DIGEST-D19: a section whose snapshot is from a prior ET day gets an '(as of …)' suffix;
    a same-day section does not."""
    from datetime import UTC, datetime, timedelta
    from swingrl.monitoring.embeds import build_daily_summary_embed

    stale = datetime.now(UTC) - timedelta(days=2)
    fresh = datetime.now(UTC)
    embed = build_daily_summary_embed(
        equity_snapshot={"total_value": 400.0, "daily_pnl": 0.0, "cash_balance": 400.0},
        crypto_snapshot={"total_value": 48.0, "daily_pnl": 0.0, "cash_balance": 48.0},
        equity_trades_today=0, crypto_trades_today=0,
        equity_as_of=stale, crypto_as_of=fresh,
    )
    names = [f["name"] for f in embed["embeds"][0]["fields"]]
    assert any(n.startswith("Equity Value (as of ") for n in names)
    assert any(n == "Crypto Value" for n in names)  # fresh: no suffix
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/monitoring/test_embeds.py::test_daily_summary_marks_stale_section -v`
Expected: FAIL — `build_daily_summary_embed()` got an unexpected keyword argument `equity_as_of`.

- [ ] **Step 3: Add the helper** — in `src/swingrl/monitoring/embeds.py`, add `from zoneinfo import ZoneInfo` to the import block, then add near the top (after imports):

```python
_ET = ZoneInfo("America/New_York")


def _as_of_suffix(as_of: datetime | None) -> str:
    """Return ' (as of YYYY-MM-DD)' when the snapshot is from a prior ET day, else ''.

    Guards the D1 latest-per-env query, which resurrects an env's newest row regardless of
    age: a missed env-day would otherwise present stale numbers as today's.
    """
    if as_of is None:
        return ""
    et_date = as_of.astimezone(_ET).date()
    if et_date == datetime.now(_ET).date():
        return ""
    return f" (as of {et_date.isoformat()})"
```

- [ ] **Step 4: Thread it through `build_daily_summary_embed`** — change the signature to add the two params (after `crypto_snapshot`):

```python
def build_daily_summary_embed(
    equity_snapshot: dict[str, float] | None,
    crypto_snapshot: dict[str, float] | None,
    equity_trades_today: int,
    crypto_trades_today: int,
    cb_status: dict[str, str] | None = None,
    equity_benchmark: float | None = None,
    crypto_benchmark: float | None = None,
    equity_as_of: datetime | None = None,
    crypto_as_of: datetime | None = None,
) -> dict[str, list[dict[str, object]]]:
```

Then change only the first field-name of each section (leave P&L/Trades untouched):
- equity: `{"name": "Equity Value", ...}` → `{"name": f"Equity Value{_as_of_suffix(equity_as_of)}", ...}`
- crypto: `{"name": "Crypto Value", ...}` → `{"name": f"Crypto Value{_as_of_suffix(crypto_as_of)}", ...}`

- [ ] **Step 5: Pass the timestamps from the job** — in `jobs.py`, add the two kwargs to the `build_daily_summary_embed(...)` call (`jobs.py:488-496`):

```python
            embed = build_daily_summary_embed(
                equity_snapshot=equity_snap,
                crypto_snapshot=crypto_snap,
                equity_trades_today=counts["equity"],
                crypto_trades_today=counts["crypto"],
                equity_benchmark=benchmarks["equity"],
                crypto_benchmark=benchmarks["crypto"],
                equity_as_of=equity_as_of,
                crypto_as_of=crypto_as_of,
            )
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/monitoring/test_embeds.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/swingrl/monitoring/embeds.py src/swingrl/scheduler/jobs.py tests/monitoring/test_embeds.py
git commit -m "feat(digest): mark a section '(as of DATE)' when its snapshot is from a prior ET day"
```

---

## Task 3: Partial/close alert cooldown de-collision (#2, D2)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:1001` (partial title) and `:969` (close title)
- Test: `tests/scheduler/test_jobs.py` (`TestEquityFillConfirmationJob`)

- [ ] **Step 1: Write the failing test** — append to `class TestEquityFillConfirmationJob`:

```python
    def test_partial_fill_alert_titles_unique_per_order(self, mock_ctx: _MockCtx) -> None:
        """DIGEST-D2: two partials on the SAME symbol get DISTINCT titles (order_id in title)
        so the alerter's per-title 30-min cooldown cannot swallow the second (found 2026-07-23)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="sp1", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.set_pending_order(order_id="sp2", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "sp1", status=OrderStatus.PARTIALLY_FILLED, filled_avg_price=600.0, filled_qty=0.02, qty=0.05
        )
        mock_ctx.alpaca.order_status(
            "sp2", status=OrderStatus.PARTIALLY_FILLED, filled_avg_price=601.0, filled_qty=0.03, qty=0.05
        )
        equity_fill_confirmation_job()
        titles = [
            c.kwargs["title"]
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        ]
        assert len(titles) == 2
        assert len(set(titles)) == 2, titles          # RED today: both identical
        assert any("sp1" in t for t in titles) and any("sp2" in t for t in titles)
```

- [ ] **Step 2: Run to verify it fails**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_partial_fill_alert_titles_unique_per_order" -v`
Expected: FAIL — `len(set(titles)) == 2` (both titles identical).

- [ ] **Step 3: Add `order_id` to both titles** — `jobs.py:1001`:

```python
            title=f"Equity auction order PARTIALLY filled — {row['symbol']} recorded (order {order_id})",
```

`jobs.py:969`:

```python
            title=f"Equity auction order {disposition} — {row['symbol']} closed (order {order_id})",
```

- [ ] **Step 4: Run to verify pass**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob" -v`
Expected: PASS (new test + existing partial/close tests, which use substring `"PARTIALLY filled"`/`"closed"` matches that still hold).

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(alerts): order_id in partial/close titles so same-symbol alerts aren't cooldown-suppressed"
```

---

## Task 4: `$None notional` guard + 2dp format (#3a/#3b, D3/D6)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:993-998`
- Test: `tests/scheduler/test_jobs.py:1231` (update) + one new case

- [ ] **Step 1: Update the coupled assertion + add the both-None case** — in `test_partial_fill_alert_notional_order_text` change line 1231:

```python
        assert "$61.10 notional" in msg, msg
```

Then append a new test to `class TestEquityFillConfirmationJob`:

```python
    def test_partial_fill_alert_unknown_amount_when_qty_and_notional_none(
        self, mock_ctx: _MockCtx
    ) -> None:
        """DIGEST-D3: qty AND notional both None renders a neutral phrase, never '$None'."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="opn", cycle_id=42, symbol="VTI", side="buy")
        mock_ctx.alpaca.order_status(
            "opn", status=OrderStatus.PARTIALLY_FILLED, filled_avg_price=100.0,
            filled_qty=0.05, qty=None, notional=None,
        )
        equity_fill_confirmation_job()
        msg = next(
            c.kwargs["message"]
            for c in mock_ctx.alerter.send_alert.call_args_list
            if "PARTIALLY filled" in (c.kwargs.get("title") or "")
        )
        assert "$None" not in msg
        assert "an unknown amount" in msg, msg
```

- [ ] **Step 2: Run to verify fail**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_partial_fill_alert_unknown_amount_when_qty_and_notional_none" "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_partial_fill_alert_notional_order_text" -v`
Expected: FAIL — `$61.1` vs `$61.10`, and `$None notional` present.

- [ ] **Step 3: Guard + format** — replace `jobs.py:993-998`:

```python
        requested_qty = getattr(order, "qty", None)
        if requested_qty is not None:
            requested = requested_qty
        else:
            notional = _safe_float(getattr(order, "notional", None))
            requested = f"${notional:,.2f} notional" if notional is not None else "an unknown amount"
```

- [ ] **Step 4: Run to verify pass**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(alerts): format partial-fill notional to 2dp; neutral text when qty+notional both None"
```

---

## Task 5: Row-4 "still working" title + else-branch coverage (#4d/#4b, D4/D9)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:1018`
- Test: `tests/scheduler/test_jobs.py` (`TestEquityFillConfirmationJob`)

**Note for implementer:** the else branch (`jobs.py:1015-1028`) fires when no new slice was recorded — either a genuinely-unfilled order (`new`/`accepted`) or a partial with no *new* shares since the last run. Read `_confirm_one_pending_order` (`jobs.py:801-1029`) to confirm the `slice_recorded` precondition before writing the row-4 test.

- [ ] **Step 1: Write the failing test** (unfilled `new` → still-working title):

```python
    def test_unfilled_order_title_says_still_working(self, mock_ctx: _MockCtx) -> None:
        """DIGEST-D4: an order with no fill yet is titled 'still working …', not 'unfilled'."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="onew", cycle_id=42, symbol="IWM", side="buy")
        mock_ctx.alpaca.order_status("onew", status=OrderStatus.NEW, filled_avg_price=None, filled_qty=0.0)
        equity_fill_confirmation_job()
        titles = [c.kwargs.get("title") or "" for c in mock_ctx.alerter.send_alert.call_args_list]
        assert any(t.startswith("Equity auction order still working") and "IWM" in t for t in titles), titles
        assert not any("unfilled" in t for t in titles), titles
```

- [ ] **Step 2: Run to verify it fails**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_unfilled_order_title_says_still_working" -v`
Expected: FAIL — title is `"Equity auction order unfilled"`.

- [ ] **Step 3: Reword the else-branch title** — `jobs.py:1018`:

```python
            title=f"Equity auction order still working — {row['symbol']}",
```

- [ ] **Step 4: Run to verify pass**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_unfilled_order_title_says_still_working" -v`
Expected: PASS.

- [ ] **Step 5: Add the row-4 (partial, no new shares) coverage test.** Read the function, then add a test that seeds an order whose full slice was already recorded on a prior run so `slice_recorded` is False, asserting it routes to the same "still working" title. Run it, confirm pass.

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(alerts): 'still working' title for no-new-shares/unfilled auction orders + coverage"
```

---

## Task 6: `trade_id LIKE` ESCAPE hardening (#4g, D5)

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:1040-1045` (`_recorded_for_order`)
- Test: `tests/scheduler/test_jobs.py`

**Note:** only the `order_id` *value* is escaped; the trailing `#%` wildcard stays live to match slice suffixes.

- [ ] **Step 1: Write the failing test** — append to `TestEquityFillConfirmationJob` (or a `_recorded_for_order` unit test class):

```python
    def test_recorded_for_order_escapes_wildcards_in_order_id(self, mock_ctx: _MockCtx) -> None:
        """DIGEST-D5: an order_id containing SQL LIKE wildcards ('%','_') matches only its own
        slices, not siblings, via ESCAPE. Guards a future non-UUID id format."""
        from swingrl.scheduler.jobs import _recorded_for_order

        with mock_ctx.db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL,
                    side TEXT NOT NULL, quantity DOUBLE PRECISION NOT NULL, price DOUBLE PRECISION NOT NULL,
                    commission DOUBLE PRECISION DEFAULT 0.0, slippage DOUBLE PRECISION DEFAULT 0.0,
                    environment TEXT NOT NULL, broker TEXT, order_type TEXT, trade_type TEXT
                )
            """)
            conn.execute("DELETE FROM trades")
            # 'a%b' is the order; 'aXb#1' is a DIFFERENT order's slice. Without ESCAPE the '%'
            # in the pattern 'a%b#%' is a wildcard that wrongly swallows 'aXb#1' (n==2).
            for tid in ("a%b", "aXb#1"):
                conn.execute(
                    "INSERT INTO trades (trade_id, timestamp, symbol, side, quantity, price, "
                    "environment, trade_type) VALUES (%s, now(), 'SPY', 'buy', 1.0, 10.0, 'equity', 'signal')",
                    (tid,),
                )
        qty, dollars, n = _recorded_for_order(mock_ctx, "a%b")
        assert n == 1  # only 'a%b' itself; sibling slice 'aXb#1' excluded by ESCAPE
```

- [ ] **Step 2: Run to verify it fails**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_recorded_for_order_escapes_wildcards_in_order_id" -v`
Expected: FAIL — without ESCAPE, `LIKE 'a%b#%'` treats `%` as a wildcard and also matches the sibling slice `aXb#1`, so `n == 2`.

- [ ] **Step 3: Add ESCAPE + escape the id value** — replace `jobs.py:1040-1045`:

```python
        row = conn.execute(
            "SELECT COALESCE(SUM(quantity), 0) AS q, "
            "COALESCE(SUM(quantity * price), 0) AS d, COUNT(*) AS n "
            "FROM trades WHERE trade_id = %s OR trade_id LIKE %s ESCAPE '\\'",
            (order_id, _like_escape(order_id) + "#%"),
        ).fetchone()
```

Add the helper above `_recorded_for_order`:

```python
def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters in a literal so only the value itself matches (ESCAPE '\\')."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
```

- [ ] **Step 4: Run to verify pass**

Run: `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_recorded_for_order_escapes_wildcards_in_order_id" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py
git commit -m "fix(sql): ESCAPE the trade_id LIKE slice match; escape wildcards in order_id"
```

---

## Task 7: B&H benchmark re-anchor script (#5, D13–D17)

**Files:**
- Create: `scripts/reanchor_benchmark_baselines.py`
- Create: `tests/data/test_reanchor_benchmark_baselines.py`

**Interfaces:**
- Reuses `BaselineRow` and `PostgresBaselineGateway._UPSERT` from `scripts/record_benchmark_baselines.py`.
- Produces a CLI: dry-run (default) prints a current-vs-proposed diff; `--apply` writes a timestamped restore-SQL backup file, upserts, then asserts the per-env baseline symbol-set == the origin-fill symbol-set.

- [ ] **Step 1: Write failing tests** — `tests/data/test_reanchor_benchmark_baselines.py`:

```python
"""REQ D13-D17: re-anchor benchmark baselines to the agent's real first-fill prices."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.reanchor_benchmark_baselines import build_reanchor_rows, reanchor
from swingrl.config.schema import SwingRLConfig
from swingrl.utils.exceptions import DataError

_ORIGINS = {"equity": date(2026, 7, 23), "crypto": date(2026, 7, 22)}


class _FakeGateway:
    """In-memory ReanchorGateway (no psycopg), mirroring test_record_benchmark_baselines.py."""

    def __init__(
        self,
        fills: dict[tuple[str, str], tuple[float, str]],
        capitals: dict[str, tuple[float, float]],
        existing_symbols: dict[str, set[str]] | None = None,
    ) -> None:
        self.fills = fills
        self.capitals = capitals
        self._existing = existing_symbols or {}
        self.upserted: list = []

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        hit = self.fills.get((environment, symbol))
        return None if hit is None else (hit[0], origin_date, hit[1])

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        return self.capitals.get(environment)

    def current_baselines(self, environment: str) -> list:
        return []

    def baseline_symbols(self, environment: str) -> set[str]:
        return set(self._existing.get(environment, set())) | {
            r.symbol for r in self.upserted if r.environment == environment
        }

    def upsert(self, rows) -> None:
        self.upserted.extend(rows)


def _all_fills(cfg: SwingRLConfig) -> dict[tuple[str, str], tuple[float, str]]:
    fills = {("equity", s): (100.0, "signal") for s in cfg.equity.symbols}
    fills.update({("crypto", s): (200.0, "signal") for s in cfg.crypto.symbols})
    return fills


def test_rows_use_first_fill_price_origin_date_and_origin_capital(
    loaded_config: SwingRLConfig,
) -> None:
    """D14/D15/D16: baseline_price = first fill, baseline_date = origin, capital = origin
    total_value — NOT cash_balance (capitals differ here to pin the D16 refinement)."""
    gw = _FakeGateway(_all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)})
    rows = build_reanchor_rows(loaded_config, gw, _ORIGINS)
    crypto = [r for r in rows if r.environment == "crypto"]
    assert all(r.baseline_price == 200.0 for r in crypto)
    assert all(r.baseline_date == date(2026, 7, 22) for r in crypto)
    assert all(r.capital_usd == 48.09 for r in crypto)   # total_value, not cash 40.0


def test_missing_fill_aborts(loaded_config: SwingRLConfig) -> None:
    """D14: a symbol with no origin-day buy fill aborts rather than guessing."""
    fills = _all_fills(loaded_config)
    del fills[("crypto", loaded_config.crypto.symbols[-1])]   # drop one crypto symbol's fill
    gw = _FakeGateway(fills, {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)})
    with pytest.raises(DataError, match="first buy-fill"):
        build_reanchor_rows(loaded_config, gw, _ORIGINS)


def test_dry_run_writes_nothing(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    gw = _FakeGateway(_all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)})
    reanchor(loaded_config, gw, apply=False, origins=_ORIGINS, backup_dir=tmp_path)
    assert gw.upserted == []


def test_apply_writes_backup_and_upserts(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    gw = _FakeGateway(_all_fills(loaded_config), {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)})
    reanchor(loaded_config, gw, apply=True, origins=_ORIGINS, backup_dir=tmp_path)
    n = len(loaded_config.equity.symbols) + len(loaded_config.crypto.symbols)
    assert len(gw.upserted) == n
    assert list(tmp_path.glob("reanchor_backup_*.sql"))


def test_apply_aborts_on_stale_extra_row(loaded_config: SwingRLConfig, tmp_path: Path) -> None:
    """D17: a stale baseline row not in the origin-fill set corrupts the divisor → abort."""
    gw = _FakeGateway(
        _all_fills(loaded_config),
        {"equity": (400.0, 350.0), "crypto": (48.09, 40.0)},
        existing_symbols={"crypto": {"GHOSTUSDT"}},
    )
    with pytest.raises(DataError, match="row-set"):
        reanchor(loaded_config, gw, apply=True, origins=_ORIGINS, backup_dir=tmp_path)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_reanchor_benchmark_baselines.py -v`
Expected: FAIL — module `scripts.reanchor_benchmark_baselines` does not exist.

- [ ] **Step 3: Write the script** — `scripts/reanchor_benchmark_baselines.py`:

```python
"""Re-anchor B&H benchmark baselines to the agent's real first-fill prices (spec D13-D17).

Model A (equal-weight passive index): for every instrument the agent traded at its env's
origin, set ``baseline_price`` = the agent's EARLIEST buy-fill price, ``baseline_date`` = the
env origin (crypto 2026-07-22, equity 2026-07-23), and ``capital_usd`` = the env's total
portfolio value at origin. The digest math is unchanged — this only corrects the rows.

``--dry-run`` is the DEFAULT (prints a diff, writes nothing). ``--apply`` first writes the
current rows as restore-SQL to a timestamped backup file, upserts, then asserts each env's
baseline symbol-set equals exactly the origin-fill instrument set (a wrong count corrupts the
equal-weight divisor). LIVE-DB write — run only with explicit approval.

Usage:
    python scripts/reanchor_benchmark_baselines.py                 # dry-run
    python scripts/reanchor_benchmark_baselines.py --apply         # write (gated)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from scripts.record_benchmark_baselines import BaselineRow, PostgresBaselineGateway
from swingrl.config.schema import load_config
from swingrl.utils.exceptions import DataError
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

_ENVIRONMENTS: tuple[str, ...] = ("equity", "crypto")
_DEFAULT_ORIGINS: dict[str, date] = {"equity": date(2026, 7, 23), "crypto": date(2026, 7, 22)}


class ReanchorGateway(Protocol):
    """DB access for the re-anchor (keeps psycopg out of unit tests)."""

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        """(price, origin_date, trade_type) of the earliest origin-day buy fill, or None."""
        ...

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        """(total_value, cash_balance) of the earliest origin-day snapshot, or None."""
        ...

    def current_baselines(self, environment: str) -> list[BaselineRow]:
        """Current benchmark_baselines rows for an env (for the diff + backup)."""
        ...

    def baseline_symbols(self, environment: str) -> set[str]:
        """Symbols currently in benchmark_baselines for an env (post-write invariant check)."""
        ...

    def upsert(self, rows: Sequence[BaselineRow]) -> None:
        """Idempotently upsert baseline rows on (environment, symbol)."""
        ...


def build_reanchor_rows(
    config: SwingRLConfig, gateway: ReanchorGateway, origins: dict[str, date]
) -> list[BaselineRow]:
    """Compute one corrected BaselineRow per env symbol from first fills + origin capital.

    Raises:
        DataError: an env has no origin-day snapshot, or a symbol has no origin-day buy fill.
    """
    rows: list[BaselineRow] = []
    for env in _ENVIRONMENTS:
        origin = origins[env]
        capital = gateway.origin_capital(env, origin)
        if capital is None:
            raise DataError(f"No {env} portfolio_snapshots on origin {origin}; cannot set capital.")
        total_value, _cash = capital
        for symbol in getattr(config, env).symbols:
            fill = gateway.first_buy_fill(env, symbol, origin)
            if fill is None:
                raise DataError(
                    f"No first buy-fill for {env}/{symbol} on origin {origin} — refusing to guess."
                )
            price, _dt, _ttype = fill
            rows.append(
                BaselineRow(
                    environment=env,
                    symbol=symbol,
                    baseline_date=origin,
                    baseline_price=float(price),
                    capital_usd=float(total_value),
                )
            )
    return rows


def _write_backup(gateway: ReanchorGateway, backup_dir: Path, stamp: str) -> Path:
    """Write current rows as restore-SQL (idempotent UPDATEs) to a timestamped file."""
    path = backup_dir / f"reanchor_backup_{stamp}.sql"
    lines = ["-- benchmark_baselines restore point (re-anchor)", "BEGIN;"]
    for env in _ENVIRONMENTS:
        for r in gateway.current_baselines(env):
            stmt = (  # nosec B608 — restore-SQL text from our own dataclass, written to a file, never executed
                "UPDATE benchmark_baselines SET "
                f"baseline_date = '{r.baseline_date}', baseline_price = {r.baseline_price}, "
                f"capital_usd = {r.capital_usd} "
                f"WHERE environment = '{r.environment}' AND symbol = '{r.symbol}';"
            )
            lines.append(stmt)
    lines.append("COMMIT;")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reanchor(
    config: SwingRLConfig,
    gateway: ReanchorGateway,
    *,
    apply: bool,
    origins: dict[str, date],
    backup_dir: Path,
    stamp: str | None = None,
) -> list[BaselineRow]:
    """Compute corrected rows and, when ``apply``, back up + upsert + verify the row-set.

    Raises:
        DataError: missing data (via build_reanchor_rows), or a post-write env symbol-set that
            does not equal the origin-fill set (a gap/extra corrupts the equal-weight divisor).
    """
    rows = build_reanchor_rows(config, gateway, origins)
    if not apply:
        return rows
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    _write_backup(gateway, backup_dir, stamp)
    gateway.upsert(rows)
    for env in _ENVIRONMENTS:
        want = {r.symbol for r in rows if r.environment == env}
        have = gateway.baseline_symbols(env)
        if have != want:
            raise DataError(
                f"{env} baseline row-set {sorted(have)} != origin-fill set {sorted(want)}; "
                "extras/gaps corrupt the equal-weight divisor. Restore from the backup file."
            )
    log.info("benchmark_baselines_reanchored", rows=len(rows))
    return rows


class PostgresReanchorGateway(PostgresBaselineGateway):
    """Concrete gateway: first fills + origin snapshots + baselines from production PostgreSQL."""

    _FIRST_FILL = (
        "SELECT price, timestamp, trade_type FROM trades "
        "WHERE environment = %s AND symbol = %s AND side = 'buy' "
        "AND (timestamp AT TIME ZONE 'America/New_York')::date = %s "
        "ORDER BY timestamp ASC LIMIT 1"
    )
    _ORIGIN_CAP = (
        "SELECT total_value, cash_balance FROM portfolio_snapshots "
        "WHERE environment = %s AND (timestamp AT TIME ZONE 'America/New_York')::date = %s "
        "ORDER BY timestamp ASC LIMIT 1"
    )
    _CURRENT = (
        "SELECT symbol, baseline_date, baseline_price, capital_usd "
        "FROM benchmark_baselines WHERE environment = %s ORDER BY symbol"
    )
    _SYMBOLS = "SELECT symbol FROM benchmark_baselines WHERE environment = %s"

    def first_buy_fill(
        self, environment: str, symbol: str, origin_date: date
    ) -> tuple[float, date, str] | None:
        with self._db.connection() as conn:
            row = conn.execute(self._FIRST_FILL, (environment, symbol, origin_date)).fetchone()
        return None if row is None else (float(row["price"]), origin_date, row["trade_type"])

    def origin_capital(self, environment: str, origin_date: date) -> tuple[float, float] | None:
        with self._db.connection() as conn:
            row = conn.execute(self._ORIGIN_CAP, (environment, origin_date)).fetchone()
        return None if row is None else (float(row["total_value"]), float(row["cash_balance"]))

    def current_baselines(self, environment: str) -> list[BaselineRow]:
        with self._db.connection() as conn:
            rows = conn.execute(self._CURRENT, (environment,)).fetchall()
        return [
            BaselineRow(environment, r["symbol"], r["baseline_date"], float(r["baseline_price"]),
                        float(r["capital_usd"]))
            for r in rows
        ]

    def baseline_symbols(self, environment: str) -> set[str]:
        with self._db.connection() as conn:
            rows = conn.execute(self._SYMBOLS, (environment,)).fetchall()
        return {r["symbol"] for r in rows}


def _print_diff(gateway: ReanchorGateway, proposed: list[BaselineRow], *, apply: bool) -> None:
    """Print current-vs-proposed rows so the operator can verify before/after any write."""
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n=== Re-anchor benchmark baselines ({mode}) ===")
    for env in _ENVIRONMENTS:
        current = {r.symbol: r for r in gateway.current_baselines(env)}
        print(f"\n{env}:")
        for r in [p for p in proposed if p.environment == env]:
            cur = current.get(r.symbol)
            was = f"was ${cur.baseline_price:,.4f}/cap ${cur.capital_usd:,.2f}" if cur else "was (none)"
            print(
                f"  {r.symbol:<10} price ${r.baseline_price:,.4f}  cap ${r.capital_usd:,.2f}  "
                f"date {r.baseline_date}   [{was}]"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-anchor B&H benchmark baselines (D13-D17).")
    parser.add_argument("--apply", action="store_true", help="Write rows (default: dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Report only (default; wins over --apply).")
    parser.add_argument("--config", default="config/swingrl.yaml")
    parser.add_argument("--equity-origin", default=_DEFAULT_ORIGINS["equity"].isoformat())
    parser.add_argument("--crypto-origin", default=_DEFAULT_ORIGINS["crypto"].isoformat())
    parser.add_argument("--backup-dir", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (0 = ok; 1 = refused/aborted)."""
    from pathlib import Path

    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    origins = {
        "equity": date.fromisoformat(args.equity_origin),
        "crypto": date.fromisoformat(args.crypto_origin),
    }
    apply = bool(args.apply) and not bool(args.dry_run)

    from swingrl.data.db import DatabaseManager  # noqa: PLC0415

    gateway = PostgresReanchorGateway(DatabaseManager(config))
    try:
        rows = reanchor(config, gateway, apply=apply, origins=origins, backup_dir=Path(args.backup_dir))
    except DataError as exc:
        log.error("benchmark_reanchor_refused", error=str(exc))
        print(f"REFUSED: {exc}")
        return 1
    _print_diff(gateway, rows, apply=apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u DATABASE_URL uv run pytest tests/data/test_reanchor_benchmark_baselines.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/reanchor_benchmark_baselines.py tests/data/test_reanchor_benchmark_baselines.py
git commit -m "feat(benchmark): re-anchor script (first-fill prices, origin capital, gated apply)"
```

> **Operational (post-merge, gated — NOT part of this plan's code):** run `python scripts/reanchor_benchmark_baselines.py` (dry-run) against the live DB, eyeball the diff with the user (crypto capital ≈ $48.09; each price = the real first fill), then `--apply` with the backup file captured. Requires explicit user approval.

---

## Task 8: `slippage_frac` approx pin (#4a, D8)

**Files:** Modify `tests/scheduler/test_jobs.py:1281`.

- [ ] **Step 1:** Replace line 1281:

```python
        assert fq["slippage_frac"] == pytest.approx(0.001)  # 0.60/600 — measured, sign+math pinned
```

- [ ] **Step 2: Run** `<scratch-db-env> uv run pytest "tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_fill_confirmation_passes_decision_price" -v` → PASS.
- [ ] **Step 3: Commit** `git commit -am "test(fill): pin slippage_frac to approx(0.001)"`

---

## Task 9: Silence `websockets.legacy` warning (#3c, D7)

**Files:** Modify `pyproject.toml` `[tool.pytest.ini_options]`.

- [ ] **Step 1:** After the `markers = [...]` block (before `timeout = 600`), add:

```toml
filterwarnings = [
    "ignore:.*websockets.legacy is deprecated.*:DeprecationWarning",
    "ignore::DeprecationWarning:websockets.legacy",
]
```

- [ ] **Step 2: Run** `uv run pytest tests/monitoring/test_embeds.py -q -W error::DeprecationWarning` and confirm no `websockets.legacy` deprecation surfaces (other warnings out of scope).
- [ ] **Step 3: Commit** `git commit -am "chore(pytest): filter the websockets.legacy DeprecationWarning (dependency noise)"`

---

## Task 10: Docstring `%%` + `test_v010` rename (#4e/#4f, D10/D11)

**Files:** Modify `tests/execution/test_risk_manager.py:343`; `tests/data/test_migrations_content.py:1487`.

- [ ] **Step 1:** In `test_risk_manager.py:343`, change `max_position_size%%` → `max_position_size%`.
- [ ] **Step 2:** In `test_migrations_content.py:1487`, rename `def test_v010_schema_version_is_10` → `def test_v010_in_migration_ledger` (keep the body/docstring). Also fix the now-stale reference to `test_v010_schema_version_is_10` in the `test_v009_…` docstring (~line 1407) to the new name.
- [ ] **Step 3: Run** `uv run pytest tests/execution/test_risk_manager.py -q` and `<scratch-db-env> uv run pytest "tests/data/test_migrations_content.py::test_v010_in_migration_ledger" -q` → PASS.
- [ ] **Step 4: Commit** `git commit -am "test: fix %% docstring artifact; rename test_v010 to superseded convention"`

---

## Task 11: Guard the `technical.py:76` numpy RuntimeWarning (#6, D18.11)

**Files:** Modify `src/swingrl/features/technical.py:72-78`. (Prod src, behaviorally inert — rides the deploy.)

- [ ] **Step 1:** Wrap the `np.where` volume-ratio block in an errstate guard (np.where evaluates both branches, so the divide-by-zero fires even though it's masked):

```python
        # 8. Volume/SMA-20 ratio (default 1.0 when SMA is zero). np.where evaluates BOTH
        # branches, so silence the masked divide-by-zero rather than let it print a warning.
        vol_sma = sdf["volume_20_sma"]
        with np.errstate(divide="ignore", invalid="ignore"):
            result["volume_sma20_ratio"] = np.where(
                vol_sma > 0,
                ohlcv["volume"].values / vol_sma.values,
                1.0,
            )
```

- [ ] **Step 2: Run** `uv run pytest tests/features/ -q -W error::RuntimeWarning -k technical` → PASS with no RuntimeWarning. Confirm existing `technical` tests still pass (values unchanged).
- [ ] **Step 3: Commit** `git commit -am "chore(features): silence masked divide-by-zero RuntimeWarning in volume ratio"`

---

## Task 12: conftest/preflight hardening (#6, D18.1/2/5/6)

**Files:** Modify `tests/conftest.py:52-54` and `:73-79`; `tests/db_marker.py:39-43` + `tests/conftest.py:97-101`; Test: `tests/test_db_marker_derivation.py` (existing).

- [ ] **Step 1:** Broaden the preflight's reach error (`tests/conftest.py:73-79`) to also catch `ProgrammingError`:

```python
    try:
        errors = schema_integrity_errors(db_url)
    except (psycopg.OperationalError, psycopg.ProgrammingError) as exc:
        pytest.exit(
            f"Cannot run the schema preflight against {db_name!r}: {exc}",
            returncode=2,
        )
```

- [ ] **Step 2:** Catch psycopg errors from activation (`tests/conftest.py:52-54`) so a DB failure exits cleanly, not as a raw traceback:

```python
    try:
        activate_isolated_db()
    except (RuntimeError, psycopg.Error) as exc:
        pytest.exit(str(exc), returncode=2)
```

- [ ] **Step 3:** DRY the marker hook — change `is_db_test` (`tests/db_marker.py:39-43`) to take the module path (using the cached reader), then call it from the hook:

```python
def is_db_test(fixturenames: Iterable[str], module_path: str) -> bool:
    """Return True when a test can reach the real database (see module docstring)."""
    if DB_FIXTURE_NAMES.intersection(fixturenames):
        return True
    return module_mentions_database_url(module_path)
```

Then `tests/conftest.py:97-101` becomes:

```python
        if is_db_test(getattr(item, "fixturenames", []), str(item.path)):
            item.add_marker(pytest.mark.db)
```

Ensure `conftest.py` imports `is_db_test`. NOTE: `tests/test_db_marker_derivation.py` already calls `is_db_test` with source *text* — those are the only other callers, and Step 4 updates them (a source string passed to the new path-based signature would be read as a file path → `OSError` → wrong result).

- [ ] **Step 4:** Update the two existing callers in `tests/test_db_marker_derivation.py` that pass SOURCE TEXT (now interpreted as a path) and add a `@cache` hit/miss test. Replace lines 21-29:

```python
def test_module_mention_triggers_db(tmp_path: Path) -> None:
    """A module that mentions DATABASE_URL makes all its tests db tests."""
    mod = tmp_path / "test_mentions.py"
    mod.write_text('db_url = os.environ.get("DATABASE_URL", "")\n')
    assert is_db_test(["tmp_path"], str(mod)) is True


def test_plain_test_is_not_db(tmp_path: Path) -> None:
    """No DB fixture + no DATABASE_URL mention -> fast lane."""
    clean = tmp_path / "test_plain.py"
    clean.write_text("import pandas as pd\n")
    assert is_db_test(["tmp_path", "loaded_config"], str(clean)) is False


def test_module_mentions_cache_hit_returns_stale(tmp_path: Path) -> None:
    """@cache pins the first read per path (D18.2)."""
    f = tmp_path / "test_cache.py"
    f.write_text("x = 1\n")
    assert module_mentions_database_url(str(f)) is False
    f.write_text('y = "DATABASE_URL"\n')          # changed on disk...
    assert module_mentions_database_url(str(f)) is False   # ...but cached False
```

(`test_real_db_fixture_triggers_db` at line 18 still passes — the fixture name short-circuits before the path is read.)

- [ ] **Step 5: Run** `env -u DATABASE_URL uv run pytest tests/test_db_marker_derivation.py -v` → PASS; then a full collection smoke `uv run pytest --collect-only -q >/dev/null` → no errors.
- [ ] **Step 6: Commit** (explicit `git add` — all three files are tracked)

```bash
git add tests/db_marker.py tests/conftest.py tests/test_db_marker_derivation.py
git commit -m "test(infra): call is_db_test from hook; catch ProgrammingError/psycopg in preflight; cache test"
```

---

## Task 13: fixture finalizer, PK-parse, named INSERT, drift-guard (#6, D18.3/4/7/9)

**Files:** `tests/fixtures/db_cleanup.py` + `tests/conftest.py:104-106`; `tests/fixtures/schema_preflight.py:25-35`; `tests/agents/test_validation.py`; `tests/test_wipe_conditionality.py`.

- [ ] **Step 1: Session finalizer** — in `tests/fixtures/db_cleanup.py` add:

```python
def close_cleanup_conn() -> None:
    """Close the session-persistent cleanup connection at session end (best-effort)."""
    global _cleanup_conn
    if _cleanup_conn is not None and not _cleanup_conn.closed:
        _cleanup_conn.close()
    _cleanup_conn = None
```

Then call it in `tests/conftest.py` `pytest_unconfigure` (alongside `drop_isolated_db()`), importing it from the fixtures module.

- [ ] **Step 2: Harden PK detection** — `tests/fixtures/schema_preflight.py:33` substring → word-boundary match tolerant of stripped line comments:

```python
        if match and re.search(r"\bPRIMARY\s+KEY\b", ddl, re.IGNORECASE):
```

- [ ] **Step 3: Named-column INSERT** — read `tests/agents/test_validation.py::test_model_metadata_table_created`, and rewrite its positional `INSERT INTO model_metadata VALUES (...)` as `INSERT INTO model_metadata (col1, col2, …) VALUES (%s, …)` with the columns named explicitly. Run the test.

- [ ] **Step 4: Drift-guard comment** — in `tests/test_wipe_conditionality.py`, above the inlined inner-conftest marker glue, add a comment: `# NOTE: mirrors tests/conftest.py pytest_collection_modifyitems — keep in sync (db-marker glue).`

- [ ] **Step 5: Run** `env -u DATABASE_URL uv run pytest tests/test_db_marker.py tests/execution/test_risk_manager.py -q` and a `db`-lane spot-check of `tests/agents/test_validation.py` against the scratch DB → PASS.
- [ ] **Step 6: Commit** `git commit -am "test(infra): cleanup-conn finalizer, PK-parse hardening, named INSERT, drift-guard note"`

---

## Full gate before push (production `src/` changed)

- [ ] FAST lane: `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q` (<2 min, 0 failures)
- [ ] Lockfile ruff (CI authority): `uv run ruff check src/ tests/`
- [ ] Types: `uv run mypy src/`
- [ ] Full suite `-n 4` against the scratch DB (preflight a targeted subset first; after any failure `--lf` first): 0 failures
- [ ] Homelab CI (detached `setsid nohup … > log 2>&1`): read the log; the literal `=== CI PASSED ===` is the ONLY verdict
- [ ] Push branch; open PR to `swingrl/2.R-training-redesign` (never `main`)

## Deploy (after merge, own explicit approval)

- [ ] Trader rebuild, path B pin-first (preserve `trader-2026-07-23-2` rollback), `--no-cache`, in a market-safe window (outside 15:30–16:45 ET; between crypto cycles). Driven by Group A (Tasks 1–6) + the inert Task 11. Groups B/D need no deploy.
- [ ] **#5 live apply** (Task 7 operational note): dry-run → eyeball with user → `--apply` with backup captured. Separate gated step.

## Self-review notes (author)

- **Spec coverage:** D1→T1, D19→T2, D2→T3, D3/D6→T4, D4/D9→T5, D5→T6, D13–D17→T7, D8→T8, D7→T9, D10/D11→T10, D18.11→T11, D18.1/2/5/6→T12, D18.3/4/7/9→T13. D12 (#6) = T11–T13. D15/D16 realized inside T7's `build_reanchor_rows`. **D18.8 and D18.10 deferred — see below.**
- **D16 refinement (supersedes the spec's strict equality-abort):** `capital_usd` = `total_value` of the earliest origin-day snapshot (correct whether pre- or post-first-buy); the dry-run prints `cash_balance` for eyeball verification instead of hard-aborting on inequality. Flag to user.
- **Deferred (listed in spec D18, deliberately not tasked):** the force-OUT marker asymmetry and the "stale safe URL exits 2" behavior note (documentation-only, no code target); **D18.8** create-from-absent DDL test (the ledger itself notes it is "moot on migrated DBs"); **D18.10** `_yaml_fallback_url` don't-cache-`""` (`tests/db_guard.py:34-45`) — the current cache-of-empty is the *safe* direction, and dropping `@lru_cache` would break the autouse `.cache_clear()` contract added in d0a9a0d, so it is not worth the risk for a safe-direction nicety. Both flagged to the user.
