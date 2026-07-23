# Post-Merge Rulings Implementation Plan (Rulings 1–5, 2026-07-22)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four code changes from the five user rulings of 2026-07-22 — record partial auction fills as real trades, give dead pending orders a terminal disposition, persist `decision_price` for auction-slippage measurement, and make the risk manager's exposure/size checks side-aware so sells are never vetoed for "adding" exposure. (Ruling 4 — cycle pings stay digest-bundled — is a deliberate NO-CHANGE and appears here only so nobody "fixes" it.)

**Architecture:** One additive migration (V011) extends `pending_orders` with `decision_price` and `disposition`. The 09:35 fill-confirmation job (`jobs.py`) becomes slice-based: each run records the *increment* of shares filled since the last run (at the increment's derived price), and terminal broker states stamp `resolved_at` + `disposition` so rows stop nagging. `RiskManager.evaluate` gets side-aware exposure math. Everything reuses the existing shared post-fill path (`pipeline.record_fill`) — no new recording code.

**Tech Stack:** Python 3.11, psycopg/PostgreSQL, pytest (DB-backed via `DATABASE_URL`), structlog.

## Global Constraints

- Branch: create `swingrl/2.R-E-rulings` from `origin/swingrl/2.R-training-redesign` (tip f70842b or later). PRs target `swingrl/2.R-training-redesign`, never `main`.
- Python 3.11; `from __future__ import annotations` first line; type hints on all defs; absolute imports only; 100-char lines.
- structlog only, context as kwargs, never f-strings in log calls.
- Migration is ADDITIVE ONLY (live trader runs while it may be applied).
- TDD: commit RED test before GREEN implementation. Never `--no-verify`.
- DB-backed tests skip when `DATABASE_URL` is unset — run them with it set (see `tests/scheduler/test_jobs.py::mock_ctx` for the pattern).
- Model routing (user ruling): coding subagents run `model: opus`; all code review by fable.
- Monetary/quantity float comparisons in tests use `pytest.approx`.

---

### Task 1: Migration V011 — `decision_price` + `disposition` on pending_orders

**Files:**
- Create: `src/swingrl/data/migrations/V011__pending_order_lifecycle.sql`
- Modify: `src/swingrl/data/migration_runner.py:34` (EXPECTED_SCHEMA_VERSION 10 → 11)
- Test: `tests/data/test_migration_v011.py`

**Interfaces:**
- Produces: `pending_orders.decision_price DOUBLE PRECISION NULL` and `pending_orders.disposition TEXT NULL CHECK IN ('filled','canceled','expired')` — consumed by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

```python
"""RULING-2/3: V011 adds decision_price + disposition to pending_orders."""

from __future__ import annotations

import os

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import (
    EXPECTED_SCHEMA_VERSION,
    apply_migrations,
    current_schema_version,
)


@pytest.fixture
def db(valid_config_yaml: str, tmp_path):  # type: ignore[no-untyped-def]
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available for testing")
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(valid_config_yaml)
    config = load_config(config_file)
    config.system.database_url = db_url
    DatabaseManager.reset()
    database = DatabaseManager(config)
    database.init_schema()
    yield database
    DatabaseManager.reset()


def test_v011_adds_lifecycle_columns(db: DatabaseManager) -> None:
    """RULING-2/3: after apply_migrations, pending_orders has decision_price + disposition."""
    apply_migrations(db)
    assert current_schema_version(db) >= 11
    assert EXPECTED_SCHEMA_VERSION == 11
    with db.connection() as conn:
        cols = {
            r["column_name"]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'pending_orders'"
            ).fetchall()
        }
    assert "decision_price" in cols
    assert "disposition" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/data/test_migration_v011.py -v`
Expected: FAIL (`decision_price` not in cols; EXPECTED_SCHEMA_VERSION is 10)

- [ ] **Step 3: Write the migration + version bump**

`src/swingrl/data/migrations/V011__pending_order_lifecycle.sql`:

```sql
-- Rulings 2026-07-22 #2/#3: pending-order lifecycle columns.
-- decision_price: the 09:15 sizing price (pipeline Step 9's get_current_price value),
--   persisted at submission so the 09:35 confirmation can compute auction slippage vs the
--   decision (fill_quality.decision_price_usd was NULL for every auction fill before this).
-- disposition: terminal state stamped together with resolved_at ('filled' | 'canceled' |
--   'expired'). A dead order is closed once — one final alert, then silence — instead of
--   re-warning daily forever. NULL while the row is still an open worklist item.
ALTER TABLE pending_orders ADD COLUMN decision_price DOUBLE PRECISION;
ALTER TABLE pending_orders ADD COLUMN disposition TEXT
    CHECK (disposition IN ('filled', 'canceled', 'expired'));
```

`src/swingrl/data/migration_runner.py` line 34:

```python
EXPECTED_SCHEMA_VERSION = 11  # Rulings 2026-07-22: V011 pending-order lifecycle (decision_price + disposition)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/data/test_migration_v011.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/migrations/V011__pending_order_lifecycle.sql \
        src/swingrl/data/migration_runner.py tests/data/test_migration_v011.py
git commit -m "feat(data): V011 pending-order lifecycle — decision_price + disposition (rulings 2/3)"
```

---

### Task 2: Thread decision_price submission → confirmation → fill_quality

**Files:**
- Modify: `src/swingrl/execution/pipeline.py:473` (call site) and `pipeline.py:657-691` (`_record_pending_order`)
- Modify: `src/swingrl/scheduler/jobs.py` (`equity_fill_confirmation_job` SELECT, `_confirm_one_pending_order` record_fill call)
- Test: `tests/scheduler/test_jobs.py` (extend `_MockCtx.set_pending_order` + new test), `tests/execution/test_pipeline.py` (pending insert carries price)

**Interfaces:**
- Consumes: Task 1's `pending_orders.decision_price` column.
- Produces: `_record_pending_order(self, fill: FillResult, cycle_id: int | None, decision_price: float | None) -> None` (new signature). Confirmation rows expose `row["decision_price"]`. Task 3's rewritten `_confirm_one_pending_order` keeps passing `decision_price=row.get("decision_price")` into `record_fill`.

- [ ] **Step 1: Write the failing tests**

In `tests/scheduler/test_jobs.py`, extend the `_MockCtx.set_pending_order` helper with a keyword arg (default `None`) that writes the new column:

```python
    def set_pending_order(
        self,
        order_id: str,
        cycle_id: int | None,
        symbol: str,
        side: str,
        decision_price: float | None = None,
    ) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO pending_orders "
                "(order_id, cycle_id, symbol, side, submitted_at, decision_price) "
                "VALUES (%s, %s, %s, %s, now(), %s) ON CONFLICT (order_id) DO NOTHING",
                (order_id, cycle_id, symbol, side, decision_price),
            )
```

New test in `TestEquityFillConfirmationJob`:

```python
    def test_fill_confirmation_passes_decision_price(self, mock_ctx: _MockCtx) -> None:
        """RULING-3: the confirmation job forwards the stored 09:15 decision_price into
        record_fill so fill_quality computes real auction slippage (was always NULL)."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(
            order_id="odp1", cycle_id=42, symbol="SPY", side="buy", decision_price=600.00
        )
        mock_ctx.alpaca.order_status(
            "odp1", status="filled", filled_avg_price=600.60, filled_qty=0.05
        )

        equity_fill_confirmation_job()

        with mock_ctx.db.connection() as conn:
            fq = conn.execute(
                "SELECT decision_price_usd, slippage_frac FROM fill_quality "
                "WHERE trade_id = %s",
                ("odp1",),
            ).fetchone()
        assert fq is not None
        assert float(fq["decision_price_usd"]) == pytest.approx(600.00)
        assert fq["slippage_frac"] is not None  # 0.60/600 — measured, not NULL
```

In `tests/execution/test_pipeline.py`, add (following that file's existing mock-DB conventions for pipeline construction):

```python
def test_record_pending_order_persists_decision_price(pipeline_with_mock_db) -> None:
    """RULING-3: _record_pending_order stores the sizing-time price on the worklist row."""
    fill = FillResult(
        trade_id="op-1", symbol="SPY", side="buy", quantity=0.0, fill_price=0.0,
        commission=0.0, slippage=0.0, environment="equity", broker="alpaca",
        status="pending", submitted_at="2026-07-22T13:15:00+00:00",
    )
    pipeline_with_mock_db._record_pending_order(fill, cycle_id=7, decision_price=600.25)
    insert_sql, params = last_execute_call(pipeline_with_mock_db)  # per file's mock pattern
    assert "decision_price" in insert_sql
    assert 600.25 in params
```

(Adapt the two helper names in the second test to the file's actual mock-capture idiom — the
assertion targets are the INSERT containing `decision_price` and the value 600.25.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob::test_fill_confirmation_passes_decision_price tests/execution/test_pipeline.py -k decision_price -v`
Expected: FAIL (unexpected keyword `decision_price` / column absent from INSERT)

- [ ] **Step 3: Implement**

`pipeline.py` — signature, docstring Args line, and INSERT:

```python
    def _record_pending_order(
        self, fill: FillResult, cycle_id: int | None, decision_price: float | None
    ) -> None:
        ...
                conn.execute(
                    "INSERT INTO pending_orders "
                    "(order_id, cycle_id, symbol, side, submitted_at, decision_price) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (order_id) DO NOTHING",
                    (fill.trade_id, cycle_id, fill.symbol, fill.side, submitted_at,
                     decision_price),
                )
```

Call site (pipeline.py:473, `current_price` is in scope — it is the Step 9 sizing price):

```python
                    if fill.status == "pending" and env_name == "equity":
                        self._record_pending_order(fill, cycle_id, current_price)
```

`jobs.py` — SELECT gains the column:

```python
            rows = conn.execute(
                "SELECT order_id, cycle_id, symbol, side, submitted_at, decision_price "
                "FROM pending_orders WHERE resolved_at IS NULL ORDER BY submitted_at"
            ).fetchall()
```

and the `record_fill` call in `_confirm_one_pending_order` changes `decision_price=None` →
`decision_price=row.get("decision_price")` (delete the now-stale "not stored on the pending
row" comment above it).

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/execution/pipeline.py src/swingrl/scheduler/jobs.py \
        tests/scheduler/test_jobs.py tests/execution/test_pipeline.py
git commit -m "feat(execution): persist + thread decision_price for auction slippage (ruling 3)"
```

---

### Task 3: Slice-based partial-fill recording + terminal disposition

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py` (`_confirm_one_pending_order` rewrite, `_stamp_pending_resolved`, new helper `_recorded_for_order`)
- Modify: `tests/scheduler/test_jobs.py` (update `test_fill_confirmation_partial_fill_surfaced`, `test_fill_confirmation_warns_on_unfilled`, crash-idempotency seed; add new tests)
- Modify (docs, same commit): `docs/execution/paper-readiness-runbook.md` (partial-fill policy paragraph), `docs/training/deploy-process.md` if it references "row left unresolved" semantics

**Interfaces:**
- Consumes: Task 1's `disposition` column; Task 2's `decision_price` threading.
- Produces: `_recorded_for_order(ctx: JobContext, order_id: str) -> tuple[float, float, int]` (recorded qty, recorded dollars, slice count); `_stamp_pending_resolved(ctx: JobContext, order_id: str, disposition: str) -> None` (new required arg). Slice trade ids: first slice = `order_id`, later slices = `f"{order_id}#<n>"`.

**Behavior spec (the contract the tests pin):**

| Broker state at 09:35 run | Action |
|---|---|
| `filled`, nothing recorded yet | record full fill (slice = all), embed, stamp `resolved_at` + `disposition='filled'` |
| `filled`, already fully recorded (crash re-run) | no new trade, stamp quietly (`'filled'`) |
| `partially_filled` (live), new shares since last look | record the increment as a trade at the increment's derived price, embed, WARNING "PARTIALLY filled — recorded", row stays open |
| `partially_filled` (live), no new shares | warning "still working", row stays open |
| `canceled`/`expired`/`rejected`/`done_for_day`/`replaced` | record any final unrecorded increment first, then stamp `resolved_at` + `disposition` (`'canceled'` for canceled/rejected/replaced, `'expired'` otherwise) + ONE final warning; never alerts again |
| `new`/`accepted` (nothing filled) | existing "unfilled" warning, row stays open |

Increment price derivation (broker reports cumulative avg only):
`slice_dollars = filled_avg_price * filled_qty − already_recorded_dollars`;
`slice_price = slice_dollars / slice_qty`, guarded: if ≤ 0 (degenerate), fall back to
`filled_avg_price` and log `pending_order_slice_price_fallback`.

- [ ] **Step 1: Update the two behavior-changing existing tests (RED)**

`test_fill_confirmation_partial_fill_surfaced` becomes `test_fill_confirmation_partial_fill_recorded`:

```python
    def test_fill_confirmation_partial_fill_recorded(self, mock_ctx: _MockCtx) -> None:
        """RULING-1: a partial auction fill IS recorded as a trade for the filled quantity
        at the broker's average price, embed fired, row left open for the remainder."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o5", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "o5", status="partially_filled", filled_avg_price=600.10, filled_qty=0.02, qty=0.05
        )

        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None
        assert float(trade["quantity"]) == pytest.approx(0.02)
        assert float(trade["price"]) == pytest.approx(600.10)
        assert trade["cycle_id"] == 42
        assert mock_ctx.alerter.send_embed.called
        kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
        assert "partially" in kwargs["title"].lower()
        assert "recorded" in kwargs["title"].lower()
        # Remainder still working: row stays open, no disposition yet.
        assert mock_ctx.pending_row("o5")["resolved_at"] is None
        assert mock_ctx.pending_row("o5")["disposition"] is None
```

`test_fill_confirmation_warns_on_unfilled` (canceled is now TERMINAL — update the assertions):

```python
    def test_fill_confirmation_closes_canceled_order(self, mock_ctx: _MockCtx) -> None:
        """RULING-2: a canceled never-filled order gets ONE final warning and a terminal
        disposition — it must not stay open and re-warn daily forever."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o2", cycle_id=42, symbol="QQQ", side="buy")
        mock_ctx.alpaca.order_status("o2", status="canceled")

        equity_fill_confirmation_job()

        assert mock_ctx.inserted_trade() is None
        row = mock_ctx.pending_row("o2")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "canceled"
        assert mock_ctx.alerter.send_alert.called

        # Second run: the resolved row is no longer in the worklist — no repeat warning.
        mock_ctx.alerter.send_alert.reset_mock()
        equity_fill_confirmation_job()
        assert not mock_ctx.alerter.send_alert.called
```

Crash-idempotency test: make the seeded quantity explicit and equal to the broker cumulative
(`seed_trade(..., quantity=0.0416)` — add the kwarg to the seed call and, if `seed_trade`
has no quantity parameter, add one with default 1.0) so the delta is zero; add
`assert mock_ctx.pending_row("o6")["disposition"] == "filled"` to its assertions.

- [ ] **Step 2: Add the new-behavior tests (RED)**

```python
    def test_fill_confirmation_second_slice_at_derived_price(self, mock_ctx: _MockCtx) -> None:
        """RULING-1: a later run records only the increment, priced so that recorded dollars
        match the broker's cumulative average exactly."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o7", cycle_id=42, symbol="SPY", side="buy")
        mock_ctx.alpaca.order_status(
            "o7", status="partially_filled", filled_avg_price=600.00, filled_qty=0.02, qty=0.05
        )
        equity_fill_confirmation_job()  # records slice 1: 0.02 @ 600.00

        # By the next run the order finished: cumulative 0.05 @ avg 600.60, then expired.
        mock_ctx.alpaca.order_status(
            "o7", status="expired", filled_avg_price=600.60, filled_qty=0.05, qty=0.05
        )
        equity_fill_confirmation_job()

        with mock_ctx.db.connection() as conn:
            slices = conn.execute(
                "SELECT trade_id, quantity, price FROM trades "
                "WHERE trade_id = %s OR trade_id LIKE %s ORDER BY trade_id",
                ("o7", "o7#%"),
            ).fetchall()
        assert len(slices) == 2
        assert slices[1]["trade_id"] == "o7#2"
        assert float(slices[1]["quantity"]) == pytest.approx(0.03)
        # Derived slice price: (600.60*0.05 - 600.00*0.02) / 0.03 = 601.00
        assert float(slices[1]["price"]) == pytest.approx(601.00)
        row = mock_ctx.pending_row("o7")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "expired"

    def test_fill_confirmation_expired_after_partial_stamps_terminal(
        self, mock_ctx: _MockCtx
    ) -> None:
        """RULING-2: terminal-with-partial closes the row ('expired'), keeps the recorded
        slice, and the final warning mentions both the recorded and unfilled parts."""
        from swingrl.scheduler.jobs import equity_fill_confirmation_job

        mock_ctx.set_pending_order(order_id="o8", cycle_id=42, symbol="XLK", side="buy")
        mock_ctx.alpaca.order_status(
            "o8", status="expired", filled_avg_price=175.50, filled_qty=0.01, qty=0.04
        )

        equity_fill_confirmation_job()

        trade = mock_ctx.inserted_trade()
        assert trade is not None and float(trade["quantity"]) == pytest.approx(0.01)
        row = mock_ctx.pending_row("o8")
        assert row["resolved_at"] is not None
        assert row["disposition"] == "expired"
```

- [ ] **Step 3: Run tests to verify the new/updated ones fail**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/scheduler/test_jobs.py::TestEquityFillConfirmationJob -v`
Expected: the four updated/new tests FAIL (partials not recorded; canceled row left open; no `#2` slice; no disposition); the fully-filled and decision_price tests still PASS.

- [ ] **Step 4: Implement in `jobs.py`**

New helper (place next to `_trade_already_recorded`, which it replaces — delete that function
and its call site):

```python
_TERMINAL_DEAD_STATUSES = {"canceled", "expired", "rejected", "done_for_day", "replaced"}


def _recorded_for_order(ctx: JobContext, order_id: str) -> tuple[float, float, int]:
    """Return (recorded qty, recorded dollars, slice count) for a broker order's trades.

    Slice trade ids are the broker order id for the first slice and ``{order_id}#<n>`` for
    later slices, so one broker order maps to one-or-more trades rows without violating the
    trades TEXT PK.
    """
    with ctx.db.connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(quantity), 0) AS q, "
            "COALESCE(SUM(quantity * price), 0) AS d, COUNT(*) AS n "
            "FROM trades WHERE trade_id = %s OR trade_id LIKE %s",
            (order_id, order_id + "#%"),
        ).fetchone()
    return float(row["q"]), float(row["d"]), int(row["n"])
```

`_stamp_pending_resolved` gains the disposition (update its one other caller accordingly):

```python
def _stamp_pending_resolved(ctx: JobContext, order_id: str, disposition: str) -> None:
    """Stamp a pending order terminal: resolved_at + disposition ('filled'|'canceled'|'expired')."""
    with ctx.db.connection() as conn:
        conn.execute(
            "UPDATE pending_orders SET resolved_at = now(), disposition = %s "
            "WHERE order_id = %s",
            (disposition, order_id),
        )
```

Rewritten `_confirm_one_pending_order` (full replacement — keep the existing docstring shape,
document the slice model):

```python
def _confirm_one_pending_order(ctx: JobContext, adapter: Any, row: dict[str, Any]) -> bool:
    """Confirm one pending pre-open order; return True if any fill slice was recorded.

    Slice model (ruling 2026-07-22 #1): each run records the INCREMENT of shares filled
    since the last look, at the increment's derived price, through the shared post-fill
    path — the books match the broker the same day, and the risk sweeps mark real
    positions. Terminal broker states (ruling #2) stamp resolved_at + disposition
    ('filled'/'canceled'/'expired') with one final alert, then the row leaves the worklist.
    """
    from swingrl.execution.types import FillResult, SizedOrder  # noqa: PLC0415

    order_id = row["order_id"]
    order = adapter.get_order_status(order_id)
    status = str(getattr(order, "status", "") or "").lower()
    cum_qty = _safe_float(getattr(order, "filled_qty", None)) or 0.0
    cum_avg = _safe_float(getattr(order, "filled_avg_price", None))

    prev_qty, prev_dollars, prev_slices = _recorded_for_order(ctx, order_id)
    delta_qty = cum_qty - prev_qty
    slice_recorded = False

    if delta_qty > 1e-9 and cum_avg is not None:
        slice_dollars = cum_avg * cum_qty - prev_dollars
        slice_price = slice_dollars / delta_qty if slice_dollars > 0 else cum_avg
        if slice_price <= 0:
            slice_price = cum_avg
            log.warning(
                "pending_order_slice_price_fallback",
                order_id=order_id,
                cum_qty=cum_qty,
                cum_avg=cum_avg,
                prev_dollars=prev_dollars,
            )
        trade_id = order_id if prev_slices == 0 else f"{order_id}#{prev_slices + 1}"
        submitted_at = _to_iso(row.get("submitted_at"))
        filled_at = _to_iso(getattr(order, "filled_at", None)) or datetime.now(UTC).isoformat()
        fill = FillResult(
            trade_id=trade_id,
            symbol=row["symbol"],
            side=row["side"],
            quantity=delta_qty,
            fill_price=slice_price,
            commission=0.0,
            slippage=0.0,
            environment="equity",
            broker="alpaca",
            status="filled",
            submitted_at=submitted_at,
            filled_at=filled_at,
        )
        sized_order = SizedOrder(
            symbol=row["symbol"],
            side=row["side"],
            quantity=delta_qty,
            dollar_amount=slice_price * delta_qty,
            stop_loss_price=None,
            take_profit_price=None,
            environment="equity",
        )
        recorded = ctx.pipeline.record_fill(
            fill,
            sized_order=sized_order,
            cycle_id=row.get("cycle_id"),
            decision_price=row.get("decision_price"),
            env_name="equity",
        )
        if recorded is None:
            # record_fill alerted critical; leave the row open so the next run retries.
            return False
        slice_recorded = True
        if build_trade_embed is not None:
            try:
                embed = build_trade_embed(recorded)
                ctx.alerter.send_embed("info", embed)
            except Exception:
                log.exception("equity_fill_confirmation_embed_failed", order_id=order_id)
        log.info(
            "equity_auction_fill_slice_recorded",
            order_id=order_id,
            trade_id=trade_id,
            symbol=row["symbol"],
            quantity=delta_qty,
            fill_price=slice_price,
            cumulative_qty=cum_qty,
            cycle_id=row.get("cycle_id"),
        )

    if status == "filled":
        _stamp_pending_resolved(ctx, order_id, "filled")
        if not slice_recorded:
            log.info(
                "equity_auction_fill_already_recorded", order_id=order_id, symbol=row["symbol"]
            )
        return True

    if status in _TERMINAL_DEAD_STATUSES:
        disposition = "canceled" if status in {"canceled", "rejected", "replaced"} else "expired"
        _stamp_pending_resolved(ctx, order_id, disposition)
        total_qty = prev_qty + (delta_qty if slice_recorded else 0.0)
        ctx.alerter.send_alert(
            level="warning",
            title=f"Equity auction order {disposition} — closed",
            message=(
                f"{row['symbol']} {row['side']} (order {order_id}) is {status}: "
                f"{total_qty} filled+recorded of {getattr(order, 'qty', None)} requested. "
                "Row closed — no further warnings."
            ),
            environment="equity",
        )
        log.warning(
            "pending_order_closed_terminal",
            order_id=order_id,
            symbol=row["symbol"],
            status=status,
            disposition=disposition,
            recorded_qty=total_qty,
        )
        return slice_recorded

    # Still live (partially_filled / new / accepted): warn per state, keep the row open.
    if slice_recorded:
        ctx.alerter.send_alert(
            level="warning",
            title="Equity auction order PARTIALLY filled — recorded",
            message=(
                f"{row['symbol']} {row['side']} (order {order_id}) partially filled: "
                f"{delta_qty} recorded now ({cum_qty} cumulative of "
                f"{getattr(order, 'qty', None)} requested) at {cum_avg} avg — remainder "
                "still working, row stays open."
            ),
            environment="equity",
        )
    else:
        ctx.alerter.send_alert(
            level="warning",
            title="Equity auction order unfilled",
            message=(
                f"{row['symbol']} {row['side']} (order {order_id}) is still "
                f"{status or 'unknown'} after the opening auction — nothing new to record, "
                "left pending for the next confirmation run."
            ),
            environment="equity",
        )
        log.warning(
            "equity_auction_order_unfilled", order_id=order_id, symbol=row["symbol"], status=status
        )
    return slice_recorded
```

Docs in the same commit: in `docs/execution/paper-readiness-runbook.md`, replace the
partial-fill caveat ("partials warn loudly but are unrecorded") with the slice model above +
disposition semantics; fix any "row left unresolved forever" wording in
`docs/training/deploy-process.md` if present.

- [ ] **Step 5: Run the whole class**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/scheduler/test_jobs.py -v`
Expected: ALL PASS (including untouched cycle/digest tests in the file)

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/scheduler/jobs.py tests/scheduler/test_jobs.py \
        docs/execution/paper-readiness-runbook.md docs/training/deploy-process.md
git commit -m "feat(scheduler): slice-recorded partial fills + terminal dispositions (rulings 1/2)"
```

---

### Task 4: Side-aware risk math — sells subtract exposure

**Files:**
- Modify: `src/swingrl/execution/risk/risk_manager.py:112-132` (checks 2 and 3 in `evaluate`)
- Test: `tests/execution/test_risk_manager.py`

**Interfaces:**
- Consumes: existing `RiskManager.evaluate(order, portfolio_value=None)`; `_make_order(...)` test helper (side/dollar_amount kwargs).
- Produces: no signature change — behavior only. Checks 1 (CB), 4 (drawdown), 5 (daily loss), 6 (global) are untouched.

- [ ] **Step 1: Write the failing tests**

Add to `tests/execution/test_risk_manager.py` (patterns follow the file's existing veto tests —
mock the tracker methods the same way neighboring tests do):

```python
class TestSellSideRiskMath:
    """RULING-5 (2026-07-22): sells reduce exposure — the guard must never veto risk
    reduction. D14 evidence: 85% exposed + full-close sell (43%) computed 128% and vetoed."""

    def test_full_position_sell_passes_at_high_exposure(
        self, risk_manager: RiskManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full-position close passes even when the book is almost fully invested."""
        monkeypatch.setattr(risk_manager._tracker, "get_portfolio_value", lambda env: 47.0)
        monkeypatch.setattr(risk_manager._tracker, "get_exposure", lambda env: 0.85)
        monkeypatch.setattr(risk_manager._tracker, "get_daily_pnl", lambda env: 0.0)
        monkeypatch.setattr(risk_manager._tracker, "get_high_water_mark", lambda env: 47.0)

        order = _make_order(
            symbol="BTCUSDT", side="sell", dollar_amount=20.0, environment="crypto"
        )
        decision = risk_manager.evaluate(order)
        assert decision.final_action == "sell"

    def test_sell_skips_position_size_cap(
        self, risk_manager: RiskManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sell larger than max_position_size%% of the book is NOT vetoed on size."""
        monkeypatch.setattr(risk_manager._tracker, "get_portfolio_value", lambda env: 100.0)
        monkeypatch.setattr(risk_manager._tracker, "get_exposure", lambda env: 0.90)
        monkeypatch.setattr(risk_manager._tracker, "get_daily_pnl", lambda env: 0.0)
        monkeypatch.setattr(risk_manager._tracker, "get_high_water_mark", lambda env: 100.0)

        order = _make_order(side="sell", dollar_amount=80.0)  # 80% > any size cap
        decision = risk_manager.evaluate(order)
        assert decision.final_action == "sell"

    def test_buy_still_vetoed_on_exposure(
        self, risk_manager: RiskManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Buys keep the additive math — the 1.0 exposure cap still vetoes them."""
        from swingrl.utils.exceptions import RiskVetoError

        monkeypatch.setattr(risk_manager._tracker, "get_portfolio_value", lambda env: 100.0)
        monkeypatch.setattr(risk_manager._tracker, "get_exposure", lambda env: 0.90)
        monkeypatch.setattr(risk_manager._tracker, "get_daily_pnl", lambda env: 0.0)
        monkeypatch.setattr(risk_manager._tracker, "get_high_water_mark", lambda env: 100.0)

        order = _make_order(side="buy", dollar_amount=15.0)  # 0.90 + 0.15 > 1.0
        with pytest.raises(RiskVetoError):
            risk_manager.evaluate(order)
```

(If the fixture's tracker is a plain object whose methods the file's existing tests replace
differently — e.g. `MagicMock` attributes — mirror that idiom instead of `monkeypatch`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/execution/test_risk_manager.py::TestSellSideRiskMath -v`
Expected: first two FAIL with `RiskVetoError` (exposure 128% / position_size 80%), third PASSES already (keep it — it pins the buy path against regression).

- [ ] **Step 3: Implement in `risk_manager.py`**

Replace checks 2 and 3 (current lines 112-132):

```python
        # 2. Position size check — buys only (ruling 2026-07-22 #5): a sell shrinks or
        # closes an existing position; vetoing it on size would block risk reduction.
        if order.side == "buy" and portfolio_value > 0:
            position_pct = order.dollar_amount / portfolio_value
            if position_pct > env_config.max_position_size:
                self._veto(
                    order,
                    "position_size",
                    f"position_size {position_pct:.4f} exceeds max {env_config.max_position_size}",
                )

        # 3. Exposure check — side-aware (ruling 2026-07-22 #5): a buy adds its dollar
        # amount to exposure, a sell subtracts it. A sell can therefore never breach the
        # 1.0 cap (D14 2026-07-22: the additive form computed 0.85 + 0.43 = 1.28 for a
        # full-position close and vetoed the exit — the guard pointed the wrong way).
        current_exposure = self._tracker.get_exposure(env)
        signed_amount = order.dollar_amount if order.side == "buy" else -order.dollar_amount
        new_exposure = current_exposure + (
            signed_amount / portfolio_value if portfolio_value > 0 else 0.0
        )
        if new_exposure > 1.0:
            self._veto(
                order,
                "exposure",
                f"total exposure {new_exposure:.4f} would exceed 1.0",
            )
```

- [ ] **Step 4: Run the whole risk-manager suite**

Run: `uv run pytest tests/execution/test_risk_manager.py tests/execution/test_order_validator.py -v`
Expected: ALL PASS (existing buy-side veto tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/execution/risk/risk_manager.py tests/execution/test_risk_manager.py
git commit -m "fix(risk): side-aware exposure/size checks — sells subtract, never vetoed (ruling 5)"
```

---

### Task 5: Full-suite gate + ruling 4 no-change record

**Files:**
- Modify: none (verification only; ruling 4 is already recorded in `.superpowers/sdd/progress-exec-alignment.md`)

- [ ] **Step 1: Run the full suite (DB-backed), background, 10-min-plus budget**

Run: `DATABASE_URL=<test-db-url> uv run pytest tests/ -x -q > /tmp/rulings_suite.log 2>&1; grep -E "passed|failed" /tmp/rulings_suite.log | tail -1`
Expected: `N passed, 0 failed` (N ≥ 1931: 1927 baseline + the 4+ new tests)

- [ ] **Step 2: Confirm ruling 4 left no accidental diff**

Run: `git diff origin/swingrl/2.R-training-redesign --stat -- src/swingrl/monitoring/`
Expected: empty (no alerter/embeds changes — cycle pings stay digest-bundled by ruling)

- [ ] **Step 3: Push branch; homelab CI per closeout checklist before any PR**

```bash
git push -u origin swingrl/2.R-E-rulings
# then: cd ~/swingrl is NOT used for CI while it serves as the live runtime — follow the
# session's CI procedure (scripts/ci-homelab.sh on the branch, literal "=== CI PASSED ===").
```

---

## Deploy note (outside this plan's scope — user-gated)

These changes reach the running trader only via a new image deploy (V011 applies with it, or
explicitly beforehand — additive, safe while the old trader runs). Deploy decision comes after
the runbook evidence window, before GO/NO-GO, per the 2026-07-22 sequencing discussion.
