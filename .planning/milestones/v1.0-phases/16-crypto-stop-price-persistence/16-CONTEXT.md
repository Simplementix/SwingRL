# Phase 16: Crypto Stop Price Persistence - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the data flow gap where stop-loss and take-profit prices are computed by PositionSizer (on `SizedOrder`) but never reach the `positions` table. `FillResult` doesn't carry stop/TP fields, so `FillProcessor._update_position()` has no prices to persist. Result: `stop_polling` reads NULL — crypto stop enforcement silently disabled.

</domain>

<decisions>
## Implementation Decisions

### Data flow fix
- Stop/TP prices must flow from `SizedOrder` through to `FillProcessor._update_position()` and into the `positions` table
- Two viable approaches (Claude's discretion which): add fields to `FillResult` or pass `SizedOrder` context to FillProcessor alongside the fill
- The `positions` table already has `stop_loss_price`, `take_profit_price`, and `side` columns (added in Phase 12)

### Write scope
- Both INSERT (new position) and UPDATE (add to existing) paths in `_update_position()` must include stop/TP values
- On sell (position reduction), stop/TP should be cleared if position reaches zero (row deleted) — no change needed there since row deletion handles it

### Verification scope
- After a crypto buy fill, `SELECT stop_loss_price, take_profit_price FROM positions WHERE environment='crypto'` must return non-NULL values
- `stop_polling._check_stop_levels()` must receive real stop/TP values and be able to trigger

### Claude's Discretion
- Whether to add stop/TP to FillResult or pass SizedOrder context separately to FillProcessor
- Whether to also persist `side` column (already in schema, also currently NULL)
- Error handling if stop/TP prices are None for a fill (equity bracket orders handled at exchange level)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — fix is fully constrained by existing data types and table schema.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SizedOrder` (`types.py:29`): Already carries `stop_loss_price` and `take_profit_price`
- `positions` table: Already has `stop_loss_price REAL`, `take_profit_price REAL`, `side TEXT` columns (Phase 12 ALTER TABLE)
- `stop_polling._check_stop_levels()`: Already reads and handles stop/TP from position rows

### Established Patterns
- `FillProcessor.process(fill)` calls `_record_trade(fill)` then `_update_position(fill)` — both receive only `FillResult`
- `ExecutionPipeline` calls `fill_processor.process(fill_result)` after exchange adapter returns — `SizedOrder` is in scope at the call site (`pipeline.py:227-228` logs stop/TP)
- Pipeline has access to both `sized_order` and `fill_result` at the point where it calls `fill_processor.process()`

### Integration Points
- `ExecutionPipeline._execute_single()` → `fill_processor.process()` — call site where SizedOrder context can be passed
- `FillProcessor._update_position()` → SQLite INSERT/UPDATE — where stop/TP must be written
- `stop_polling._check_stop_levels()` → reads from positions — already correct, just needs non-NULL data

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-crypto-stop-price-persistence*
*Context gathered: 2026-03-10*
