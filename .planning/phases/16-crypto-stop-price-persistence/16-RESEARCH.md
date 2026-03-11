# Phase 16: Crypto Stop Price Persistence - Research

**Researched:** 2026-03-10
**Domain:** Python dataclass extension, SQLite parameterized queries, execution pipeline data flow
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Stop/TP prices must flow from `SizedOrder` through to `FillProcessor._update_position()` and into the `positions` table
- Both INSERT (new position) and UPDATE (add to existing) paths in `_update_position()` must include stop/TP values
- On sell, stop/TP should be cleared if position reaches zero — row deletion already handles it (no change needed)
- After a crypto buy fill, `SELECT stop_loss_price, take_profit_price FROM positions WHERE environment='crypto'` must return non-NULL values
- `stop_polling._check_stop_levels()` must receive real stop/TP values and be able to trigger

### Claude's Discretion
- Whether to add stop/TP to `FillResult` or pass `SizedOrder` context separately to `FillProcessor`
- Whether to also persist `side` column (already in schema, also currently NULL)
- Error handling if stop/TP prices are None for a fill (equity bracket orders handled at exchange level)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-07 | Position sizing: modified Kelly criterion (quarter-Kelly Phase 1), 2% max risk per trade, ATR(2x) stop-losses | PositionSizer already computes stop_loss_price and take_profit_price on SizedOrder; this phase closes the gap where those values are never written to the DB |
| PAPER-10 | Bracket orders: OTO for Alpaca (ATR stop-loss, R:R take-profit), two-step OCO for Binance.US | The stop/TP prices computed by PositionSizer must persist to the positions table so stop_polling can enforce them for crypto; FillProcessor._update_position() is the write point |
</phase_requirements>

---

## Summary

Phase 16 is a targeted data-flow fix, not a new feature. The execution pipeline already computes stop-loss and take-profit prices in `PositionSizer.size()` — they live on `SizedOrder.stop_loss_price` and `SizedOrder.take_profit_price`. The `positions` table already has `stop_loss_price REAL`, `take_profit_price REAL`, and `side TEXT` columns (added in Phase 12 via `ALTER TABLE`). `stop_polling._check_stop_levels()` already reads those columns and acts on non-NULL values. The only missing piece is the write: `FillProcessor._update_position()` receives only a `FillResult`, which has no stop/TP fields, so it writes nothing to those columns. The positions table rows come out with all three columns NULL.

The fix requires: (1) choosing how to get stop/TP context into `FillProcessor`, (2) updating `_update_position()` to write the three columns in both INSERT and UPDATE-via-replace paths, and (3) updating tests to verify the columns are populated. No schema changes are needed. No adapter changes are needed. The stop_polling code already works correctly given non-NULL data.

**Primary recommendation:** Pass `SizedOrder` as an optional second argument to `FillProcessor.process()` rather than adding fields to `FillResult`. This avoids touching the frozen `FillResult` dataclass and keeps the `FillProcessor` interface explicit about what optional context it can consume. The `side` column should also be written since it is already in the schema and NULL today.

---

## Standard Stack

### Core — all already in the project, no new dependencies

| Component | Location | Role |
|-----------|----------|------|
| `FillResult` | `src/swingrl/execution/types.py:50` | Stage 5 DTO — currently missing stop/TP fields |
| `SizedOrder` | `src/swingrl/execution/types.py:29` | Stage 2 DTO — already carries `stop_loss_price`, `take_profit_price`, `side` |
| `FillProcessor` | `src/swingrl/execution/fill_processor.py` | Writes to `trades` and `positions` tables |
| `ExecutionPipeline._execute_single()` | `src/swingrl/execution/pipeline.py:209-236` | Call site with both `sized_order` and `fill_result` in scope |
| `stop_polling._check_stop_levels()` | `src/swingrl/scheduler/stop_polling.py:102` | Reads stop/TP from position row; already correct, needs non-NULL data |
| `positions` table | `src/swingrl/data/db.py:325-466` | Has `stop_loss_price REAL`, `take_profit_price REAL`, `side TEXT` (nullable) |

**Installation:** No new packages required.

---

## Architecture Patterns

### Existing Data Flow (Gap State)

```
PositionSizer.size()
  -> SizedOrder(stop_loss_price=X, take_profit_price=Y, side="buy")
     -> OrderValidator.validate()
        -> ValidatedOrder(order=sized_order)
           -> adapter.submit_order(validated_order)
              -> FillResult(NO stop/TP fields)
                 -> FillProcessor.process(fill)        # <-- gap here
                    -> _update_position(fill)           # writes NULL to stop/TP
                       positions table: stop_loss_price=NULL, take_profit_price=NULL
                          -> stop_polling: skips row (both None)  # silently broken
```

### Target Data Flow (Fixed State)

```
PositionSizer.size()
  -> SizedOrder(stop_loss_price=X, take_profit_price=Y, side="buy")
     -> [kept in pipeline scope as sized_order]
        -> adapter.submit_order(validated_order)
           -> FillResult(no stop/TP needed)
              -> FillProcessor.process(fill, sized_order=sized_order)
                 -> _update_position(fill, sized_order)
                    -> positions: stop_loss_price=X, take_profit_price=Y, side="buy"
                       -> stop_polling: real prices, can trigger  # works
```

### Pattern 1: Optional SizedOrder Context on FillProcessor.process()

**What:** Add `sized_order: SizedOrder | None = None` parameter to `FillProcessor.process()` and thread it through to `_update_position()`.

**When to use:** Preferred approach. Does not touch the frozen `FillResult` dataclass. Backward compatible — existing callers that pass only `fill` continue to work (stop/TP will be NULL, which is acceptable for equity fills where Alpaca handles stops at the exchange level).

**Why not add to FillResult:** `FillResult` is a frozen dataclass representing broker confirmation. The broker does not return stop/TP values — those are our locally computed risk parameters. Adding them to `FillResult` would misrepresent what the broker provides and creates a semantic mismatch. Passing `SizedOrder` as optional context keeps the separation of concerns clean.

**Call site change in `pipeline.py`:**
```python
# Before (pipeline.py:236)
self._fill_processor.process(fill)

# After
self._fill_processor.process(fill, sized_order=sized_order)
```

**`FillProcessor.process()` signature change:**
```python
# Before
def process(self, fill: FillResult) -> None:

# After
def process(self, fill: FillResult, sized_order: SizedOrder | None = None) -> None:
```

**`_update_position()` INSERT path (new position):**
```python
# Source: src/swingrl/execution/fill_processor.py:165-179 (current)
conn.execute(
    "INSERT INTO positions "
    "(symbol, environment, quantity, cost_basis, last_price, "
    "unrealized_pnl, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)",
    (fill.symbol, fill.environment, fill.quantity, fill.fill_price,
     fill.fill_price, unrealized_pnl, now),
)

# After — add stop_loss_price, take_profit_price, side
stop_loss = sized_order.stop_loss_price if sized_order is not None else None
take_profit = sized_order.take_profit_price if sized_order is not None else None
side = sized_order.side if sized_order is not None else fill.side

conn.execute(
    "INSERT INTO positions "
    "(symbol, environment, quantity, cost_basis, last_price, "
    "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (fill.symbol, fill.environment, fill.quantity, fill.fill_price,
     fill.fill_price, unrealized_pnl, now, stop_loss, take_profit, side),
)
```

**`_update_position()` UPDATE path (existing position, `INSERT OR REPLACE`):**

The current `INSERT OR REPLACE` for existing buy positions replaces the entire row. Since `INSERT OR REPLACE` deletes the old row and inserts a new one, stop/TP must be included in the column list or they will be reset to NULL. Same approach: include all three new columns in the INSERT OR REPLACE statement.

```python
# Partial sell path also uses INSERT OR REPLACE — same fix applies
conn.execute(
    "INSERT OR REPLACE INTO positions "
    "(symbol, environment, quantity, cost_basis, last_price, "
    "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (fill.symbol, fill.environment, new_qty, new_cost, fill.fill_price,
     unrealized_pnl, now, stop_loss, take_profit, side),
)
```

**Critical subtlety for the UPDATE/partial-sell path:** When adding to an existing position, we should update stop/TP to the most recent order's values (overwrite). When doing a partial sell, we should preserve the existing stop/TP from the position row (carry them forward), since we are not resizing the position trigger levels. This requires reading `stop_loss_price` and `take_profit_price` from `existing` in the SELECT query and falling back to them when `sized_order` is None.

**Updated SELECT in `_update_position()`:**
```python
# Before
existing = conn.execute(
    "SELECT quantity, cost_basis FROM positions WHERE symbol = ? AND environment = ?",
    (fill.symbol, fill.environment),
).fetchone()

# After — fetch stop/TP to carry forward on partial sells
existing = conn.execute(
    "SELECT quantity, cost_basis, stop_loss_price, take_profit_price, side "
    "FROM positions WHERE symbol = ? AND environment = ?",
    (fill.symbol, fill.environment),
).fetchone()
```

**Carry-forward logic for partial sells:**
```python
# In the partial sell branch — sized_order is None (sells come through a separate signal)
# Preserve existing stop/TP; do not clear them unless position goes to zero
existing_sl = existing["stop_loss_price"] if existing else None
existing_tp = existing["take_profit_price"] if existing else None
existing_side = existing["side"] if existing else fill.side
```

### Pattern 2: Adding stop/TP to FillResult (Alternative — NOT Recommended)

Modifying `FillResult` would require: changing the frozen dataclass definition, updating all 4 exchange adapters to include the fields, and updating every test fixture that constructs `FillResult`. This is a larger change with higher blast radius for a fix that only benefits the crypto path. Reject this approach.

### Anti-Patterns to Avoid

- **Clearing stop/TP on partial sells:** When a position is partially reduced, the remaining quantity still needs stop enforcement. Do not set stop/TP to NULL on partial sells — carry existing values forward.
- **Skipping the `side` column:** `side` is already in the schema and already NULL. Write it at the same time to avoid a second pass.
- **Using `UPDATE ... SET` instead of `INSERT OR REPLACE`:** The current code uses `INSERT OR REPLACE` consistently. Do not introduce a hybrid `UPDATE` statement for just the stop/TP columns — it would create inconsistency in how the positions table is managed.
- **Requiring stop/TP to be non-None:** For equity fills, Alpaca handles stop/TP at the exchange level. The `sized_order` parameter is optional (`None` default) precisely to allow equity fills to continue working unchanged with NULL stop/TP values in the positions table. The stop_polling daemon only polls `WHERE environment='crypto'`, so NULL stop/TP on equity positions is harmless.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema migration | Custom migration script | Idempotent `ALTER TABLE` pattern already in `db.py:457-466` | Pattern already established; `positions` columns already exist |
| Stop enforcement logic | New stop-check code | Existing `_check_stop_levels()` in `stop_polling.py` | Already fully implemented; just needs non-NULL data |
| Atomic position updates | Manual transaction management | SQLite context manager `with self._db.sqlite() as conn:` | Already used throughout `FillProcessor` — same connection, same implicit transaction |

**Key insight:** This phase has zero new infrastructure. Every component already exists. The fix is purely adding columns to existing SQL statements.

---

## Common Pitfalls

### Pitfall 1: INSERT OR REPLACE Silently Drops Unspecified Columns

**What goes wrong:** SQLite `INSERT OR REPLACE` deletes the existing row and inserts a new one. Any column not listed in the INSERT statement receives its DEFAULT value (NULL for nullable columns). If `stop_loss_price` is not included in the UPDATE-path `INSERT OR REPLACE`, it will be reset to NULL even if the original row had a valid value.

**Why it happens:** The current `_update_position()` INSERT OR REPLACE statements omit `stop_loss_price`, `take_profit_price`, and `side`. This is currently harmless because those columns are always NULL, but once we start writing them, the omission would immediately erase them on the first cost-basis update.

**How to avoid:** Include all three columns in every `INSERT OR REPLACE` statement — both the new-position INSERT and the existing-position INSERT OR REPLACE.

**Warning signs:** Test `test_second_buy_adjusts_cost_basis` passes but stop/TP columns are NULL after the second buy.

### Pitfall 2: Partial Sell Loses Stop/TP Values

**What goes wrong:** On a partial sell, `sized_order` will be `None` (because the sell path in `pipeline.py` uses the sell signal's `sized_order`, not the original buy's). If the partial sell path unconditionally writes `None` to stop/TP, it erases the stop levels that were set at buy time.

**Why it happens:** The sell path in `pipeline.py` also goes through `PositionSizer.size()`, which computes reverse stop/TP for the sell side. These are not the same as the stored position's buy-side stop/TP. Writing the sell-side sized_order's stop/TP to the position row would corrupt the levels.

**How to avoid:** On the partial-sell branch, carry forward `existing["stop_loss_price"]` and `existing["take_profit_price"]` from the SELECT. Pass `None` for `sized_order` on sells, or handle the sell branch explicitly.

**Simpler approach:** In `_update_position()`, extract stop/TP values from `sized_order` only for buy fills. For sell fills (partial), read the existing stop/TP from the `existing` row and preserve them. This requires the SELECT to include those columns.

### Pitfall 3: Test Fixtures Use FillResult Without Stop/TP Context

**What goes wrong:** Existing test fixtures in `tests/execution/test_fill_processor.py` call `processor.process(buy_fill)` with no `sized_order` argument. After the fix, the default behavior (no `sized_order`) should still work — writing NULL for stop/TP. But new tests verifying stop/TP persistence must construct a `SizedOrder` and pass it.

**How to avoid:** Keep `sized_order: SizedOrder | None = None` as a default parameter. All existing tests continue to pass. New tests explicitly pass a `SizedOrder` fixture with known stop/TP values.

### Pitfall 4: `INSERT OR REPLACE` Primary Key Behavior

**What goes wrong:** The `positions` table has `PRIMARY KEY (symbol, environment)`. `INSERT OR REPLACE` on a matching PK deletes the old row and inserts a new one — this is correct behavior, but it means the entire row must be specified. Missing any column will result in NULL or DEFAULT.

**How to avoid:** Always list all columns explicitly in the INSERT OR REPLACE column list. Never rely on unspecified columns defaulting to their previous values.

---

## Code Examples

### What `_update_position()` needs to look like (new position path)

```python
# Source: derived from src/swingrl/execution/fill_processor.py:162-179
def _update_position(
    self,
    fill: FillResult,
    sized_order: SizedOrder | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()

    # Extract stop/TP from sized_order (buy context) or default to None
    new_stop = sized_order.stop_loss_price if sized_order is not None else None
    new_tp = sized_order.take_profit_price if sized_order is not None else None
    new_side = sized_order.side if sized_order is not None else fill.side

    with self._db.sqlite() as conn:
        existing = conn.execute(
            "SELECT quantity, cost_basis, stop_loss_price, take_profit_price, side "
            "FROM positions WHERE symbol = ? AND environment = ?",
            (fill.symbol, fill.environment),
        ).fetchone()

        if fill.side == "buy":
            if existing is None:
                # New position — write stop/TP from sized_order
                conn.execute(
                    "INSERT INTO positions "
                    "(symbol, environment, quantity, cost_basis, last_price, "
                    "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fill.symbol, fill.environment, fill.quantity, fill.fill_price,
                     fill.fill_price, 0.0, now, new_stop, new_tp, new_side),
                )
            else:
                # Add to existing — update cost basis and stop/TP
                old_qty = existing["quantity"]
                old_cost = existing["cost_basis"]
                new_qty = old_qty + fill.quantity
                new_cost = (old_qty * old_cost + fill.quantity * fill.fill_price) / new_qty
                unrealized_pnl = (fill.fill_price - new_cost) * new_qty
                conn.execute(
                    "INSERT OR REPLACE INTO positions "
                    "(symbol, environment, quantity, cost_basis, last_price, "
                    "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fill.symbol, fill.environment, new_qty, new_cost,
                     fill.fill_price, unrealized_pnl, now, new_stop, new_tp, new_side),
                )
        else:
            # Sell — carry forward existing stop/TP (do not clear them on partial sells)
            if existing is not None:
                existing_stop = existing["stop_loss_price"]
                existing_tp = existing["take_profit_price"]
                existing_side = existing["side"]
                new_qty = existing["quantity"] - fill.quantity
                if new_qty <= 0:
                    conn.execute(
                        "DELETE FROM positions WHERE symbol = ? AND environment = ?",
                        (fill.symbol, fill.environment),
                    )
                else:
                    old_cost = existing["cost_basis"]
                    unrealized_pnl = (fill.fill_price - old_cost) * new_qty
                    conn.execute(
                        "INSERT OR REPLACE INTO positions "
                        "(symbol, environment, quantity, cost_basis, last_price, "
                        "unrealized_pnl, updated_at, stop_loss_price, take_profit_price, side) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (fill.symbol, fill.environment, new_qty, old_cost,
                         fill.fill_price, unrealized_pnl, now,
                         existing_stop, existing_tp, existing_side),
                    )
```

### Pipeline call site change

```python
# Source: src/swingrl/execution/pipeline.py:232-236
# Before:
fill = adapter.submit_order(validated_order)
self._fill_processor.process(fill)

# After:
fill = adapter.submit_order(validated_order)
self._fill_processor.process(fill, sized_order=sized_order)
```

### Test pattern for stop/TP persistence

```python
# New test — verifies stop/TP written to positions table
def test_buy_persists_stop_tp_to_positions(
    processor: FillProcessor,
    mock_db: DatabaseManager,
) -> None:
    """PAPER-07: buy fill persists stop_loss_price and take_profit_price to positions."""
    from swingrl.execution.types import FillResult, SizedOrder

    fill = FillResult(
        trade_id="fill-crypto-001",
        symbol="BTCUSDT",
        side="buy",
        quantity=0.001,
        fill_price=60000.0,
        commission=0.06,
        slippage=0.0,
        environment="crypto",
        broker="binance_us",
    )
    sized = SizedOrder(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.001,
        dollar_amount=60.0,
        stop_loss_price=58800.0,   # 2x ATR below entry
        take_profit_price=62400.0, # 2:1 R:R above entry
        environment="crypto",
    )

    processor.process(fill, sized_order=sized)

    with mock_db.sqlite() as conn:
        pos = conn.execute(
            "SELECT stop_loss_price, take_profit_price, side "
            "FROM positions WHERE symbol = 'BTCUSDT' AND environment = 'crypto'"
        ).fetchone()

    assert pos is not None
    assert pos["stop_loss_price"] == pytest.approx(58800.0)
    assert pos["take_profit_price"] == pytest.approx(62400.0)
    assert pos["side"] == "buy"
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `FillResult` carries all context | `FillResult` = broker confirmation only; `SizedOrder` = our risk parameters passed separately | Cleaner separation of concerns; no broker DTO mutation needed |
| `INSERT OR REPLACE` without all columns | `INSERT OR REPLACE` with all columns listed explicitly | Prevents silent NULL resets on column omission |

**Deprecated/outdated:** None. All existing patterns remain.

---

## Open Questions

1. **Should stop/TP be updated on a second buy into the same position?**
   - What we know: `PositionSizer.size()` computes fresh stop/TP for each order based on the current price and ATR at order time.
   - What's unclear: Should the stored stop/TP for an existing position update to the latest order's levels, or stay at the original entry's levels?
   - Recommendation: Update to the latest order's levels. This is consistent with weighted-average cost basis (the position's effective entry point is updated). The new stop/TP reflects the current risk profile of the combined position. Simple and correct.

2. **What if `sized_order.stop_loss_price` is `None`?**
   - What we know: `PositionSizer.size()` always computes stop/TP as long as it returns a non-None `SizedOrder`. There is no code path where `stop_loss_price` is None on a returned `SizedOrder`.
   - What's unclear: Could future callers of `FillProcessor.process()` pass a `SizedOrder` with None stop/TP?
   - Recommendation: Write whatever value is on `sized_order` (None or float). No validation needed — NULL in the DB is fine and stop_polling already handles it gracefully (skips rows with both None).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (see pyproject.toml) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/execution/test_fill_processor.py tests/scheduler/test_stop_polling.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-07 | `FillProcessor._update_position()` writes `stop_loss_price` and `take_profit_price` on buy fills | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k stop` | Wave 0 gap — test must be added |
| PAPER-07 | Stop/TP preserved on partial sell (carry forward) | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k partial_sell` | Wave 0 gap — test must be added |
| PAPER-07 | `side` column populated on new position | unit | `uv run pytest tests/execution/test_fill_processor.py -v -k side` | Wave 0 gap — test must be added |
| PAPER-10 | `stop_polling._check_stop_levels()` receives non-NULL stop/TP after crypto fill | integration | `uv run pytest tests/scheduler/test_stop_polling.py -v` | Partial — existing tests cover NULL-skip behavior; new test needed for non-NULL case with real stop_polling trigger |
| PAPER-10 | Pipeline passes `sized_order` to `fill_processor.process()` | unit | `uv run pytest tests/execution/test_pipeline.py -v` | Partial — pipeline tests exist; need assertions that `sized_order` is forwarded |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/execution/test_fill_processor.py tests/scheduler/test_stop_polling.py -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] New test class `TestStopTPPersistence` in `tests/execution/test_fill_processor.py` — covers PAPER-07 (stop/TP write on buy, carry-forward on partial sell, side column)
- [ ] New test in `tests/scheduler/test_stop_polling.py` — end-to-end: fill processor writes stop/TP, stop_polling reads and triggers
- [ ] No new framework or fixture files needed — `mock_db` fixture in `tests/execution/conftest.py` already provides a real SQLite DB with schema initialized

---

## Sources

### Primary (HIGH confidence)

- Direct source inspection: `src/swingrl/execution/fill_processor.py` — full `_update_position()` implementation, confirmed gap
- Direct source inspection: `src/swingrl/execution/types.py` — `FillResult` (no stop/TP), `SizedOrder` (has stop/TP)
- Direct source inspection: `src/swingrl/execution/pipeline.py:209-236` — `sized_order` and `fill_result` both in scope at the call site
- Direct source inspection: `src/swingrl/scheduler/stop_polling.py:80-91` — SELECT already reads `stop_loss_price, take_profit_price`
- Direct source inspection: `src/swingrl/data/db.py:457-466` — Phase 12 ALTER TABLE confirms columns exist
- Direct source inspection: `tests/execution/test_fill_processor.py` — existing test coverage baseline
- Direct source inspection: `tests/execution/conftest.py` — `mock_db` fixture pattern
- SQLite documentation: `INSERT OR REPLACE` behavior (deletes + re-inserts on PK conflict, omitted columns get DEFAULT/NULL)

### Secondary (MEDIUM confidence)

- `tests/scheduler/test_stop_polling.py` — confirms `_check_stop_levels()` already handles non-NULL stop/TP correctly (test at line 104)

---

## Metadata

**Confidence breakdown:**
- Gap identification: HIGH — confirmed by direct code inspection of all four files in the data flow
- Fix approach: HIGH — `SizedOrder | None` optional parameter is a standard Python pattern, no external dependencies
- SQL behavior: HIGH — `INSERT OR REPLACE` semantics are well-defined and tested throughout the project
- Pitfalls: HIGH — derived from reading the actual SQL statements in `_update_position()` and cross-referencing with SQLite `INSERT OR REPLACE` semantics

**Research date:** 2026-03-10
**Valid until:** Stable — pure internal implementation change, no external API dependencies
