# Execution Alignment & Reporting Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the live equity schedule with the training fill convention (candles 20:15
ET, pre-open cycle 09:15 ET with opening-auction fills) and fix the reporting/risk
defects found 2026-07-21 (digest zeros,
deadzone valuation freeze, missing sell P&L).

**Architecture:** Config-only schedule changes first (Tasks 1–2), then three code fixes
on the trader (Tasks 3–5) and one embed enhancement (Task 6). Era-1 training-convention
work is NOT here — it lives in `2026-07-07-training-redesign-plan.md` Task 25b.

**Tech Stack:** Python 3.11, APScheduler, psycopg (pg16), pytest.

**Spec:** `docs/superpowers/specs/2026-07-21-execution-alignment-design.md`

## Global Constraints

- TDD: commit RED test before GREEN implementation, per task.
- No hardcoded times/symbols in code — config fields only.
- All timestamps UTC internally; ET only for cron fields (documented as ET) and display.
- structlog with kwargs, never f-strings; typed `SwingRLError` subclasses only.
- Full suite green before push (background run, 10-min timeout); never `--no-verify`.
- Branch: `swingrl/2.R-C-exec-alignment` off integration head. PR → integration branch,
  never `main`. Deploys are separately gated by the user.
- Docs update in the SAME commit as the code they reference.

---

### Task 1: Equity candle job → 20:15 ET

**Files:**
- Modify: `src/swingrl/config/schema.py:384` (`equity_time_et` default `"16:50"` → `"20:15"`)
- Modify: `config/swingrl.yaml:111`, matching line in `config/swingrl.prod.yaml.example`
- Modify: `docs/options/ops.md` (candle job time references)
- Test: `tests/test_candle_jobs.py`

**Interfaces:**
- Consumes: `CandleJobsConfig.equity_time_et` (existing HH:MM string field).
- Produces: no new interfaces — schedule value change only. Task 2 relies on the bar
  landing evening of day D (its 09:15 cycle reads day t−1 bars next morning).

- [ ] **Step 1: Write the failing test** (append to `tests/test_candle_jobs.py`)

```python
def test_equity_candle_default_lands_same_evening() -> None:
    """CANDLE-D1: default equity candle time is past 00:00 UTC year-round (20:15 ET),
    so day-D bars land day-D evening (Alpaca fetch end pins to 00:00 UTC)."""
    from swingrl.config.schema import CandleJobsConfig

    assert CandleJobsConfig().equity_time_et == "20:15"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_candle_jobs.py::test_equity_candle_default_lands_same_evening -v`
Expected: FAIL — `assert '16:50' == '20:15'`

- [ ] **Step 3: Commit RED** — `git commit -m "test(data): RED — equity candle job at 20:15 ET (same-evening bar)"`

- [ ] **Step 4: Implementation** — in `schema.py:384`:

```python
    equity_time_et: str = Field(default="20:15")  # HH:MM, Mon–Fri; past 00:00 UTC year-round
```

In `config/swingrl.yaml` (and prod example, same field):

```yaml
    equity_time_et: "20:15"         # Mon–Fri; past 00:00 UTC (EDT+EST) so day-D bar lands day-D evening
```

In `docs/options/ops.md`: update the candle-job schedule mention(s) from 16:50 to 20:15
with the same one-line rationale.

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_candle_jobs.py tests/test_options_config.py -v` → all PASS.

- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(data): GREEN — equity candle job 20:15 ET; day-D bar lands same evening"`

### Task 2: Equity cycle → 09:15 ET (pre-open) + decision snapshot follows + quiet-window docs

**Files:**
- Modify: `src/swingrl/config/schema.py:60` (`cycle_time_et` default), `config/swingrl.yaml:15`
  + `:98` (decision snapshot row), `config/swingrl.prod.yaml.example` (both fields)
- Modify: `docs/options/ops.md` (quiet window: two windows per spec D4)
- Test: `tests/test_options_config.py`, `tests/scheduler/test_main.py`

**Interfaces:**
- Consumes: `EquityConfig.cycle_time_et` (validated HH:MM), `OptionsSnapshot` rows
  (`label`, `market_time_et`, `pull_time_et`, `misfire_grace_s`) — all existing.
- Produces: no new interfaces. The runbook's morning capture evidence (spec coupling
  note) depends on this task's deploy.

- [ ] **Step 1: Write the failing tests**

In `tests/test_options_config.py`, update `OPT-CFG-2` (line ~35) to the new expectation
(this IS the red — the yaml still says 15:45/16:00):

```python
    assert rows == [("decision", "09:30", "09:46", 900), ("eod", "16:15", "16:35", 18000)]
```

Append to `tests/scheduler/test_main.py`:

```python
def test_equity_cycle_default_is_preopen() -> None:
    """SCHED-D2: default equity cycle is 09:15 ET pre-open — decide on t-1 bars, submit
    before the 09:28 auction cutoff, fill at the open (spec D2/D11)."""
    from swingrl.config.schema import EquityConfig

    assert EquityConfig().cycle_time_et == "09:15"
```

- [ ] **Step 2: Run to verify both fail**

Run: `uv run pytest tests/test_options_config.py tests/scheduler/test_main.py -v`
Expected: the two changed/new tests FAIL on 15:45/16:00 values.

- [ ] **Step 3: Commit RED** — `git commit -m "test(execution): RED — pre-open equity cycle 09:15 + decision snapshot 09:30/09:46"`

- [ ] **Step 4: Implementation**

`schema.py:60`:

```python
    cycle_time_et: str = Field(default="09:15")
```

`config/swingrl.yaml:15` (+ prod example):

```yaml
  cycle_time_et: "09:15"        # pre-open decision on t-1 bars; orders submitted before the 09:28 auction cutoff (spec D2/D11)
```

`config/swingrl.yaml:98` (+ prod example):

```yaml
    - { label: decision, market_time_et: "09:30", pull_time_et: "09:46", misfire_grace_s: 900 }
```

`docs/options/ops.md`: replace the single 15:30–16:45 quiet window with the two windows
(09:00–10:15 and 16:00–16:45 ET) and one line of why (cycle, auction fills,
fill-confirmation, and decision pull all live in the morning window; eod chain
unchanged).

NOTE: until Task 8 lands, a 09:15 cycle would submit synchronous market orders into a
closed market. **Task 2's config flip and Task 8 deploy together** — keep both on this
branch, one PR, one trader rebuild (the closeout section already treats the branch as
one deployable unit).

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_options_config.py tests/scheduler/test_main.py tests/test_candle_jobs.py -v` → PASS.

- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(execution): GREEN — pre-open equity cycle 09:15 ET; decision snapshot follows (spec D2/D3/D4)"`

### Task 3: Daily Summary — real trade counts

**Files:**
- Modify: `src/swingrl/scheduler/jobs.py:264-334` (`daily_summary_job`)
- Test: `tests/scheduler/test_jobs.py` (existing digest tests live here; follow file conventions)

**Interfaces:**
- Consumes: `trades` table (`timestamp TIMESTAMPTZ`, `environment`, `trade_type`).
- Produces: `_count_trades_today(conn, env: str, now: datetime) -> int` (module-private
  helper in `jobs.py`) — Task 3 only; no other task consumes it.

- [ ] **Step 1: Write the failing test**

```python
def test_daily_summary_counts_signal_trades_today(mock_ctx: Any) -> None:
    """DIGEST-D5: digest counts today's (ET) signal trades per env from the trades
    table — never the hardcoded zeros (found live 2026-07-21)."""
    # mock_ctx fixture: ctx.db returns rows for portfolio_snapshots and the new count
    # query; capture the embed handed to ctx.alerter.send_embed.
    mock_ctx.set_query_result(
        "FROM trades", [{"environment": "crypto", "n": 2}, {"environment": "equity", "n": 0}]
    )
    daily_summary_job()
    embed = mock_ctx.alerter.send_embed.call_args.args[1]
    fields = {f["name"]: f["value"] for f in embed["embeds"][0]["fields"]}
    assert fields["Crypto Trades"] == "2"
    assert fields["Equity Trades"] == "0"
```

(Adapt the fixture plumbing to the existing mock pattern in the file — the assertion
block is the requirement.)

- [ ] **Step 2: Run to verify it fails** — count comes out "0" for crypto → FAIL.
- [ ] **Step 3: Commit RED** — `git commit -m "test(monitoring): RED — digest real trade counts"`
- [ ] **Step 4: Implementation** — in `daily_summary_job`, replace the literals:

```python
        counts = {"equity": 0, "crypto": 0}
        with ctx.db.connection() as conn:
            rows_c = conn.execute(
                "SELECT environment, count(*) AS n FROM trades "
                "WHERE trade_type = 'signal' "
                "AND (timestamp AT TIME ZONE 'America/New_York')::date = "
                "(now() AT TIME ZONE 'America/New_York')::date "
                "GROUP BY environment"
            ).fetchall()
        for r in rows_c:
            counts[r["environment"]] = int(r["n"])
        ...
                equity_trades_today=counts["equity"],
                crypto_trades_today=counts["crypto"],
```

(ET-date comparison on both sides — same convention as `position_tracker.get_daily_pnl`.)

- [ ] **Step 5: Run tests** — digest tests PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "fix(monitoring): GREEN — digest counts real signal trades (was hardcoded 0)"`

### Task 4: Mark all held positions to market every cycle

**Files:**
- Modify: `src/swingrl/execution/pipeline.py` (Step 10 block, ~line 482) — fetch missing
  prices before the snapshot; `src/swingrl/execution/risk/position_tracker.py` — persist
  `last_price`/`unrealized_pnl` for marked symbols.
- Test: `tests/execution/test_pipeline.py`, `tests/execution/test_position_tracker.py`

**Interfaces:**
- Consumes: `adapter.get_current_price(symbol) -> float` (existing, both adapters);
  `PositionTracker.get_positions(env) -> list[dict]`; `compute_portfolio_value(env, prices)`.
- Produces: `PositionTracker.mark_positions(env: str, prices: dict[str, float]) -> None`
  (updates `positions.last_price` + `unrealized_pnl = (last_price - cost_basis) * quantity`
  for symbols present in `prices`). Task 5 relies on the snapshot value being fresh.

- [ ] **Step 1: Write the failing tests**

```python
def test_deadzone_cycle_marks_held_positions(pipeline_fixture: Any) -> None:
    """MTM-D6: a cycle with zero orders still fetches prices for held symbols, so the
    snapshot moves with the market (live defect: value frozen since 07-19)."""
    fx = pipeline_fixture(positions={"BTCUSDT": (0.5, 60000.0)}, orders_generated=0)
    fx.adapter.get_current_price.return_value = 50000.0  # crashed
    fx.run_cycle("crypto")
    snap = fx.recorded_snapshot()
    assert snap.total_value == pytest.approx(0.5 * 50000.0 + fx.cash)
    fx.adapter.get_current_price.assert_called_with("BTCUSDT")


def test_mark_failure_falls_back_and_warns(pipeline_fixture: Any) -> None:
    """MTM-D6 fail-open: a price-fetch failure falls back to stored last_price, warns,
    and never blocks the cycle."""
    fx = pipeline_fixture(positions={"BTCUSDT": (0.5, 60000.0)}, orders_generated=0)
    fx.adapter.get_current_price.side_effect = BrokerError("down")
    fx.run_cycle("crypto")  # must not raise
    assert fx.recorded_snapshot().total_value == pytest.approx(0.5 * 60000.0 + fx.cash)
```

(Adapt fixture names to the file's existing pipeline-test scaffolding; assertions are
the requirement. Add a `mark_positions` unit test in `test_position_tracker.py`
asserting the UPDATE writes `last_price` and recomputed `unrealized_pnl`.)

- [ ] **Step 2: Run to verify they fail** (snapshot shows 60000-based value; no fetch call).
- [ ] **Step 3: Commit RED** — `git commit -m "test(execution): RED — deadzone cycles mark to market"`
- [ ] **Step 4: Implementation** — in `pipeline.py` Step 10, before computing the snapshot:

```python
        if not dry_run:
            for pos in self._position_tracker.get_positions(env_name):
                sym = pos["symbol"]
                if pos["quantity"] and sym not in cycle_prices:
                    try:
                        cycle_prices[sym] = adapter.get_current_price(sym)
                    except Exception:
                        log.warning("mark_price_fetch_failed", symbol=sym, exc_info=True)
            self._position_tracker.mark_positions(env_name, cycle_prices)
            cash = self._position_tracker.compute_cash(env_name)
            ...  # existing snapshot lines unchanged
```

`mark_positions` in `position_tracker.py`:

```python
    def mark_positions(self, env: str, prices: dict[str, float]) -> None:
        """Persist fresh marks: last_price + unrealized_pnl for the priced symbols."""
        if not prices:
            return
        with self._db.connection() as conn:
            for symbol, price in prices.items():
                conn.execute(
                    "UPDATE positions SET last_price = %s, "
                    "unrealized_pnl = (%s - cost_basis) * quantity, "
                    "updated_at = %s "
                    "WHERE environment = %s AND symbol = %s",
                    (price, price, datetime.now(tz=UTC).isoformat(), env, symbol),
                )
```

- [ ] **Step 5: Run tests** — targeted + `uv run pytest tests/execution/ -v` → PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "fix(execution): GREEN — mark held positions to market every cycle (deadzone freeze)"`

### Task 5: Breaker evaluation on zero-order cycles

**Files:**
- Modify: `src/swingrl/execution/pipeline.py` (Step 10, after Task 4's fresh snapshot values)
- Test: `tests/execution/test_pipeline.py`

**Interfaces:**
- Consumes: Task 4's fresh `new_portfolio_value` / `daily_pnl`;
  `CircuitBreaker.check_and_update(portfolio_value, high_water_mark, daily_pnl) -> CBState`;
  `PositionTracker.get_high_water_mark(env) -> float` (all existing).
- Produces: nothing new — a halt row/alert via the existing `_trigger` path when breached.

- [ ] **Step 1: Write the failing test**

```python
def test_zero_order_cycle_trips_breaker_on_crash(pipeline_fixture: Any) -> None:
    """CB-D7: a deadzone cycle with crashed marks still evaluates the drawdown breaker
    (drill finding #1: zero-order cycles previously skipped breaker evaluation)."""
    fx = pipeline_fixture(positions={"BTCUSDT": (0.5, 60000.0)}, orders_generated=0,
                          high_water_mark=35000.0, max_drawdown_pct=0.10)
    fx.adapter.get_current_price.return_value = 30000.0  # ~50% below entry
    fx.run_cycle("crypto")
    assert fx.circuit_breaker.check_and_update.called
```

- [ ] **Step 2: Run to verify it fails** (never called on zero-order cycles today).
- [ ] **Step 3: Commit RED** — `git commit -m "test(risk): RED — zero-order cycles evaluate breakers"`
- [ ] **Step 4: Implementation** — in Step 10, when no orders were attempted this cycle
  (`not fills and not orders_attempted` — reuse/introduce the local flag the loop already
  implies), after `daily_pnl` is computed:

```python
            if not orders_attempted:
                cb = self._circuit_breakers[env_name]
                hwm = self._position_tracker.get_high_water_mark(env_name)
                cb.check_and_update(new_portfolio_value, max(hwm, new_portfolio_value), daily_pnl)
```

(The trade path already calls `check_and_update` pre-trade; this closes the no-trade
path only — no double-evaluation.)

- [ ] **Step 5: Run tests** — targeted + `tests/execution/` PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "fix(risk): GREEN — breakers evaluate on zero-order cycles (finding #1)"`

### Task 6: Realized P&L on sell embeds

**Files:**
- Modify: `src/swingrl/execution/types.py` (FillResult: `realized_pnl: float | None = None`),
  `src/swingrl/execution/fill_processor.py` (compute on sells, before position update),
  `src/swingrl/execution/pipeline.py` (attach via `dataclasses.replace`),
  `src/swingrl/monitoring/embeds.py` (`build_trade_embed`: add field when present)
- Test: `tests/execution/test_fill_processor.py`, `tests/monitoring/test_embeds.py`
  (match actual embed-test file location)

**Interfaces:**
- Consumes: `positions.cost_basis` (read already happens in `fill_processor.py:358`).
- Produces: `FillResult.realized_pnl: float | None` — populated only for sell fills;
  `process_fill(...)` returns it so the pipeline can attach it. Embed shows
  `Realized P&L` field only when not None.

- [ ] **Step 1: Write the failing tests**

```python
def test_sell_fill_computes_realized_pnl(processor_fixture: Any) -> None:
    """PNL-D8: realized = (fill - cost_basis) * qty - commission, computed from the
    position row as it was BEFORE this fill updates it."""
    fx = processor_fixture(existing_position=dict(quantity=2.0, cost_basis=100.0))
    realized = fx.process_sell(quantity=2.0, fill_price=110.0, commission=0.44)
    assert realized == pytest.approx((110.0 - 100.0) * 2.0 - 0.44)


def test_sell_embed_shows_realized_pnl() -> None:
    """PNL-D8: sell embeds carry the Realized P&L field; buys never do."""
    sell = make_fill(side="sell", realized_pnl=19.56)
    fields = {f["name"] for f in build_trade_embed(sell)["embeds"][0]["fields"]}
    assert "Realized P&L" in fields
    buy = make_fill(side="buy")
    fields = {f["name"] for f in build_trade_embed(buy)["embeds"][0]["fields"]}
    assert "Realized P&L" not in fields
```

- [ ] **Step 2: Run to verify both fail.**
- [ ] **Step 3: Commit RED** — `git commit -m "test(execution): RED — realized P&L on sell fills + embeds"`
- [ ] **Step 4: Implementation** — `fill_processor`: on the sell branch (position row
  already loaded at `:358`), compute
  `realized = (fill.fill_price - existing["cost_basis"]) * fill.quantity - fill.commission`
  and return it from `process_fill`; pipeline:
  `fill = dataclasses.replace(fill, realized_pnl=realized)` before appending to `fills`;
  `embeds.py`: after the Commission field,

```python
    if fill.realized_pnl is not None:
        fields.append(
            {"name": "Realized P&L", "value": f"${fill.realized_pnl:+,.2f}", "inline": True}
        )
```

- [ ] **Step 5: Run tests** — targeted + full `tests/execution/` + embed tests PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(monitoring): GREEN — sell embeds show realized P&L"`

### Task 7: Risk-sweep job (D10) — mark + breaker check between cycles, no trading

**Files:**
- Modify: `src/swingrl/config/schema.py` (`risk.sweep_interval_minutes: int = 30`, ge=5),
  `src/swingrl/scheduler/jobs.py` (new `risk_sweep_job`), `scripts/main.py` (register
  IntervalTrigger job, id `risk_sweep`), `config/swingrl.yaml` + prod example
- Test: `tests/scheduler/test_jobs.py`, `tests/scheduler/test_main.py`

**Interfaces:**
- Consumes: Task 4's `PositionTracker.mark_positions(env, prices)` +
  `compute_portfolio_value` / `compute_daily_pnl`; Task 5's breaker-evaluation pattern
  (`CircuitBreaker.check_and_update`); `adapter.get_current_price` (both adapters).
- Produces: `risk_sweep_job() -> None` — for each env with held positions: fetch prices
  (fail-open per symbol), `mark_positions`, evaluate per-env + global breakers. Writes
  NO snapshot rows (snapshots stay cycle-cadence append-only) and places NO orders.

- [ ] **Step 1: Write the failing tests**

```python
def test_risk_sweep_trips_breaker_between_cycles(mock_ctx: Any) -> None:
    """SWEEP-D10: a crash between cycles is caught by the sweep — marks refresh and the
    drawdown breaker evaluates without any trading."""
    mock_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
    mock_ctx.adapter.get_current_price.return_value = 30000.0
    risk_sweep_job()
    assert mock_ctx.circuit_breakers["crypto"].check_and_update.called
    assert mock_ctx.adapter.submit_order.call_count == 0


def test_risk_sweep_writes_no_snapshots(mock_ctx: Any) -> None:
    """SWEEP-D10: sweeps never write portfolio_snapshots (cycle-cadence stays clean for
    daily-P&L baselines)."""
    mock_ctx.set_positions("crypto", {"BTCUSDT": (0.5, 60000.0)})
    risk_sweep_job()
    assert not mock_ctx.executed_sql_matching("INSERT INTO portfolio_snapshots")
```

- [ ] **Step 2: Run to verify both fail** (job doesn't exist → ImportError).
- [ ] **Step 3: Commit RED** — `git commit -m "test(risk): RED — risk-sweep job marks + evaluates breakers between cycles"`
- [ ] **Step 4: Implementation** — `risk_sweep_job` in `jobs.py` following the module's
  `_get_ctx()` pattern: skip when `is_halted(ctx.db)`; for each env in
  `ctx.circuit_breakers`: positions = tracker.get_positions(env); if none, continue;
  fetch prices per symbol (try/except → `log.warning("sweep_price_fetch_failed", ...)`),
  `mark_positions`, `value = compute_portfolio_value(env, prices)`,
  `daily_pnl = compute_daily_pnl(env, value)`,
  `hwm = max(tracker.get_high_water_mark(env), value)`,
  `ctx.circuit_breakers[env].check_and_update(value, hwm, daily_pnl)`; finish with the
  global breaker (`ctx.global_cb.check_combined(...)`) using per-env values. Register in
  `main.py` with `IntervalTrigger(minutes=config.risk.sweep_interval_minutes)`.
- [ ] **Step 5: Run tests** — targeted + `tests/scheduler/` PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(risk): GREEN — risk-sweep job (D10); between-cycle blind window -> sweep interval"`

### Task 8: Pre-open equity execution — auction fills via pre-09:28 DAY market orders (D11)

The 09:15 cycle must not fill synchronously into a closed market. Equity orders become:
submit before 09:28 (buys `notional=|delta_value|`, sells `qty`), Alpaca fills them at
the opening print, and a fill-confirmation job at 09:35 turns confirmed fills into
trades/capture/embeds. Crypto's synchronous path is untouched.

**Files:**
- Modify: `src/swingrl/execution/adapters/alpaca_adapter.py` (`submit_order`: notional
  support + return `status="pending"` when the market is closed at submission),
  `src/swingrl/execution/pipeline.py` (equity: skip synchronous fill processing for
  pending results; persist pending order ids with `cycle_id`),
  `src/swingrl/scheduler/jobs.py` (new `equity_fill_confirmation_job`),
  `scripts/main.py` (register it, cron 09:35 ET Mon–Fri),
  `src/swingrl/config/schema.py` (`equity.fill_confirmation_time_et: str = "09:35"`)
- Test: `tests/execution/test_alpaca_adapter.py`, `tests/execution/test_pipeline.py`,
  `tests/scheduler/test_jobs.py`

**Interfaces:**
- Consumes: `FillResult.status` `"pending"` semantics (existing C2 honest-fill
  lifecycle: pending results are never recorded as trades); `fill_processor` +
  capture/embed paths (existing — the confirmation job replays them for real fills);
  Task 6's `realized_pnl` attach point.
- Produces: `equity_fill_confirmation_job() -> None` — queries Alpaca for the cycle's
  submitted orders by client order id, converts `filled` ones into `FillResult`s with
  real fill price/timestamps, runs the SAME fill_processor + capture + embed path the
  synchronous route uses (fills carry the originating `cycle_id`), alerts a warning for
  orders still unfilled/canceled after the auction. Pending order ids persist in a new
  `pending_orders` table (order_id PK, cycle_id FK, symbol, side, submitted_at,
  resolved_at NULL until confirmed) — restart-safe.

- [ ] **Step 1: Write the failing tests**

```python
def test_preopen_market_order_submits_notional_and_pends(adapter_fixture: Any) -> None:
    """EXEC-D11: pre-open equity buys submit notional DAY market orders and return
    status='pending' — never a synthetic synchronous fill."""
    fx = adapter_fixture(market_open=False)
    result = fx.adapter.submit_order(make_validated_order(side="buy", dollar_amount=25.0))
    submitted = fx.api.submit_order.call_args.kwargs
    assert submitted["notional"] == 25.0 and "qty" not in submitted
    assert submitted["time_in_force"] == "day" and submitted["type"] == "market"
    assert result.status == "pending"


def test_fill_confirmation_records_auction_fill(mock_ctx: Any) -> None:
    """EXEC-D11: the 09:35 job converts a filled auction order into a trade row with the
    originating cycle_id, capture rows, and a trade embed."""
    mock_ctx.set_pending_order(order_id="o1", cycle_id=42, symbol="SPY", side="buy")
    mock_ctx.alpaca.order_status("o1", status="filled", filled_avg_price=600.10, filled_qty=0.0416)
    equity_fill_confirmation_job()
    trade = mock_ctx.inserted_trade()
    assert trade["cycle_id"] == 42 and trade["price"] == pytest.approx(600.10)
    assert mock_ctx.alerter.send_embed.called  # trade embed fired


def test_cycle_orders_info_ping_both_envs_incl_sells(mock_ctx: Any) -> None:
    """EXEC-D12: EVERY cycle (equity AND crypto) ends with one INFO listing each order
    — buys as notional, SELLS as qty + approx value — or 'no orders — deadzone held'.
    Fires both ways (candle-alert full-parity precedent, user ruling)."""
    mock_ctx.run_equity_cycle(orders=[("SPY", "buy", 25.0), ("QQQ", "sell", 0.0416, 25.0)])
    kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
    assert kwargs["level"] == "info" and "cycle orders" in kwargs["title"].lower()
    assert "BUY SPY $25.00" in kwargs["message"]
    assert "SELL QQQ 0.0416" in kwargs["message"]  # sells never omitted

    mock_ctx.run_crypto_cycle(orders=[("BTCUSDT", "buy", 20.06)])
    kwargs = mock_ctx.alerter.send_alert.call_args.kwargs
    assert "crypto" in kwargs["title"].lower() and "BUY BTCUSDT $20.06" in kwargs["message"]

    mock_ctx.run_crypto_cycle(orders=[])
    assert "deadzone" in mock_ctx.alerter.send_alert.call_args.kwargs["message"].lower()


def test_fill_confirmation_warns_on_unfilled(mock_ctx: Any) -> None:
    """EXEC-D11: an order still unfilled after the auction alerts a warning and stays
    unresolved for the next run — never silently dropped."""
    mock_ctx.set_pending_order(order_id="o2", cycle_id=42, symbol="QQQ", side="buy")
    mock_ctx.alpaca.order_status("o2", status="canceled")
    equity_fill_confirmation_job()
    assert "unfilled" in mock_ctx.alerter.send_alert.call_args.kwargs["title"].lower()
```

(Adapt fixture plumbing to the existing adapter/jobs test scaffolding; assertion
contracts are the requirement.)

- [ ] **Step 2: Run to verify all three fail.**
- [ ] **Step 3: Commit RED** — `git commit -m "test(execution): RED — pre-open notional submission + fill confirmation (D11)"`
- [ ] **Step 4: Implementation** — adapter: when `order.side == "buy"` and the clock
  reports the market closed, submit `notional=round(order.dollar_amount, 2)`
  (`time_in_force="day"`, `type="market"`), else the existing qty path; closed-market
  submissions return `FillResult(status="pending", quantity=0, fill_price=0, ...)` with
  the broker order id in `trade_id`. Pipeline: pending equity results insert a
  `pending_orders` row (new migration V009 — follow the `schema_migrations` runner
  pattern from Plan A Task 1) and skip fill processing. Confirmation job: load
  unresolved `pending_orders`, poll order status, build real `FillResult`s from
  `filled_avg_price`/`filled_qty`/timestamps, run fill_processor + capture + embeds
  (reuse the pipeline's existing post-fill helpers — extract to a shared function if
  needed), stamp `resolved_at`; unfilled/canceled → warning alert
  `"Equity auction order unfilled"`, row left unresolved. INFO ping (D12, BOTH envs):
  emit from the pipeline's cycle-completion point (one shared code path, env-agnostic):
  `alerter.send_alert(level="info", title=f"Cycle orders submitted — {env}",
  message=<one line per order: buys "BUY SPY $25.00" (notional), sells
  "SELL QQQ 0.0416 (~$25.00)" (qty + qty×reference price); empty → "no orders —
  deadzone held">)` — always fires (equity on trading days at 09:15; crypto every
  cycle, fills remain synchronous with embeds as today).
- [ ] **Step 5: Run tests** — targeted + full `tests/execution/` + `tests/scheduler/` PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(execution): GREEN — opening-auction execution via pre-open DAY orders + 09:35 fill confirmation (D11)"`

### Task 9: Buy-and-hold benchmark in the daily digest (D13)

**Files:**
- Create: migration `V010_benchmark_baselines` (follow the `schema_migrations` runner
  pattern; table: `benchmark_baselines(environment TEXT, symbol TEXT, baseline_date DATE,
  baseline_price DOUBLE PRECISION, capital_usd DOUBLE PRECISION,
  PRIMARY KEY (environment, symbol))`)
- Create: `scripts/record_benchmark_baselines.py` (one-shot, run at epoch reset: writes
  one row per env symbol — equal-weight capital split, latest stored close as
  baseline_price, today as baseline_date; `--dry-run` default, prints rows)
- Modify: `src/swingrl/scheduler/jobs.py` (`daily_summary_job`),
  `src/swingrl/monitoring/embeds.py` (`build_daily_summary_embed`: two new optional
  fields per env)
- Test: `tests/data/test_migrations_content.py` (V010), `tests/scheduler/test_jobs.py`,
  embed tests

**Interfaces:**
- Consumes: `ohlcv_daily` (equity closes) / `ohlcv_4h` (crypto closes) for current
  prices; Task 3's digest wiring.
- Produces: `_benchmark_value(conn, env: str) -> float | None` (module-private in
  `jobs.py`): `Σ over symbols (capital_usd/n_symbols) × latest_close/baseline_price`;
  None when no baselines exist (digest omits the fields — pre-reset behavior
  unchanged). Embed builder accepts `equity_benchmark: float | None = None,
  crypto_benchmark: float | None = None` and renders "Buy & Hold" +
  "vs B&H" (`agent_value − benchmark`, signed $ and %) per env when not None.

- [ ] **Step 1: Write the failing tests**

```python
def test_benchmark_value_equal_weight(mock_ctx: Any) -> None:
    """BENCH-D13: benchmark = equal-weight capital split grown by close/baseline per
    symbol. 47 capital, 2 symbols; BTC +10%, ETH -10% -> exactly 47.0."""
    mock_ctx.set_baselines("crypto", capital=47.0, rows={"BTCUSDT": 60000.0, "ETHUSDT": 2000.0})
    mock_ctx.set_latest_closes({"BTCUSDT": 66000.0, "ETHUSDT": 1800.0})
    assert _benchmark_value(mock_ctx.conn, "crypto") == pytest.approx(47.0)


def test_digest_shows_agent_vs_buy_and_hold(mock_ctx: Any) -> None:
    """BENCH-D13: digest embeds carry Buy & Hold + vs B&H fields when baselines exist,
    and omit them (unchanged shape) when none do."""
    embed = build_daily_summary_embed(
        equity_snapshot=None, crypto_snapshot={"total_value": 50.0, "daily_pnl": 1.0},
        equity_trades_today=0, crypto_trades_today=0, crypto_benchmark=47.0,
    )
    names = [f["name"] for f in embed["embeds"][0]["fields"]]
    assert "Crypto Buy & Hold" in names and "Crypto vs B&H" in names
```

- [ ] **Step 2: Run to verify both fail** (helper/params don't exist).
- [ ] **Step 3: Commit RED** — `git commit -m "test(monitoring): RED — buy-and-hold benchmark in digest (D13)"`
- [ ] **Step 4: Implementation** — migration V010 + baseline recorder script (idempotent
  upsert on (environment, symbol); refuses to overwrite without `--force`); digest job
  computes `_benchmark_value` per env inside the existing connection block and passes
  the new embed params; embed renders
  `{"name": "Crypto vs B&H", "value": f"${delta:+,.2f} ({delta_pct:+.2f}%)"}`.
- [ ] **Step 5: Run tests** — migration + digest + embed tests PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(monitoring): GREEN — digest tracks agent vs buy-and-hold (D13)"`

### Task 10: Discord alert styling — category colors + formatting (user request 2026-07-21)

Every alert category gets its own sidebar color and emoji title prefix so the phone
notification is scannable at a glance; order lists render in monospace code blocks.
Central style map — one place to retheme, no scattered hex literals. No images
(hosted-URL maintenance burden; emoji do the job — user-approved).

**Color scheme (user-approved 2026-07-21):**

| Category | Color | Emoji |
|---|---|---|
| Data ingests (candles) | `0x3498DB` blue | 📥 |
| Daily Summary digest | `0xF1C40F` gold | 📊 |
| Trade fill — buy | `0x2ECC71` green (keep) | 🟢 |
| Trade fill — sell | `0xE74C3C` red (keep — sell≠buy at a glance) | 🔴 |
| Cycle-orders ping + ops heartbeats | `0x9B59B6` purple | 🔄 |
| Warning / critical | keep orange `0xFFA500` / red `0xFF0000` | ⚠️ / 🚨 |

**Files:**
- Modify: `src/swingrl/monitoring/embeds.py` (`_CATEGORY_STYLE` dict: category →
  (color, emoji); digest embed → gold + ▲/▼ arrows on P&L + ✅/❌ on the vs-B&H delta
  from Task 9; trade embeds → emoji title prefix, colors unchanged)
- Modify: `src/swingrl/monitoring/alerter.py` (`send_alert` gains
  `category: str | None = None` — when set, overrides the level-default color and
  prefixes the title emoji; level still drives routing/footer/cooldown)
- Modify callers: `src/swingrl/scheduler/jobs.py` (candle-ingest INFOs →
  `category="ingest"`; digest embed already styled in builder),
  `src/swingrl/execution/pipeline.py` (Task 8's D12 cycle ping → `category="cycle"`,
  order lines in a ``` code block, aligned columns)
- Test: `tests/monitoring/test_embeds.py`, `tests/monitoring/test_alerter.py`
  (match actual embed/alerter test file locations)

**Interfaces:**
- Consumes: Task 8's cycle-ping message assembly; Task 9's vs-B&H delta value;
  existing `send_alert` / embed builders.
- Produces: `_CATEGORY_STYLE: dict[str, tuple[int, str]]` (module constant, embeds.py);
  `send_alert(..., category=...)` optional kwarg — omitted → behavior identical to
  today (level-color, no emoji), so untouched call sites need no changes.

- [ ] **Step 1: Write the failing tests**

```python
def test_alert_category_sets_color_and_emoji() -> None:
    """STYLE-D15: category selects sidebar color + emoji title prefix; ingest=blue 📥,
    cycle=purple 🔄; omitted category keeps today's level-default behavior."""
    a = make_alerter()
    a.send_alert(level="info", title="Equity candles ingested", message="rows=8",
                 category="ingest")
    embed = a.last_payload()["embeds"][0]
    assert embed["color"] == 0x3498DB
    assert embed["title"].startswith("📥")

    a.send_alert(level="info", title="Cycle orders submitted — crypto", message="…",
                 category="cycle")
    embed = a.last_payload()["embeds"][0]
    assert embed["color"] == 0x9B59B6 and embed["title"].startswith("🔄")

    a.send_alert(level="info", title="plain", message="no category")
    assert a.last_payload()["embeds"][0]["color"] == 0x3498DB  # unchanged legacy path


def test_digest_embed_gold_with_arrows() -> None:
    """STYLE-D15: Daily Summary is gold with ▲/▼ P&L arrows (▼ for negative)."""
    embed = build_daily_summary_embed(
        equity_snapshot=None, crypto_snapshot={"total_value": 50.0, "daily_pnl": -1.0},
        equity_trades_today=0, crypto_trades_today=0,
    )["embeds"][0]
    assert embed["color"] == 0xF1C40F
    assert any("▼" in str(f["value"]) for f in embed["fields"])


def test_cycle_ping_orders_render_in_code_block(mock_ctx: Any) -> None:
    """STYLE-D15: D12 cycle-ping order lists are monospace code blocks, aligned."""
    mock_ctx.run_equity_cycle(orders=[("SPY", "buy", 25.0), ("QQQ", "sell", 0.0416, 25.0)])
    msg = mock_ctx.alerter.send_alert.call_args.kwargs["message"]
    assert "```" in msg and "BUY  SPY" in msg
```

(Adapt fixture plumbing to the existing test scaffolding; assertion contracts are the
requirement.)

- [ ] **Step 2: Run to verify all fail** (no `category` kwarg; digest is blue today).
- [ ] **Step 3: Commit RED** — `git commit -m "test(monitoring): RED — category colors + emoji + monospace order lists"`
- [ ] **Step 4: Implementation** — `_CATEGORY_STYLE` in `embeds.py` (single source; alerter
  imports it); `send_alert`: `if category: color, emoji = _CATEGORY_STYLE[category];
  title = f"{emoji} {title}"` else today's `_COLORS[level]` path untouched; digest builder
  sets gold + formats P&L values `f"{'▲' if pnl >= 0 else '▼'} ${pnl:+,.2f} …"` and the
  vs-B&H field `f"{'✅' if delta >= 0 else '❌'} ${delta:+,.2f} …"`; trade embed titles gain
  🟢/🔴 prefix; Task 8's ping assembles orders as fixed-width rows inside triple-backticks.
- [ ] **Step 5: Run tests** — embed + alerter + pipeline ping tests PASS.
- [ ] **Step 6: Commit GREEN** — `git commit -m "feat(monitoring): GREEN — category-colored Discord alerts + formatted embeds"`

---

## Closeout (per-phase workflow)

- [ ] Full suite background (10-min timeout), 0 failures → push branch.
- [ ] Homelab CI detached (stash `~/swingrl` runtime edits first — tests read the tree),
  check the `=== CI PASSED ===` result line.
- [ ] PR → integration; user merges.
- [ ] **Paper epoch reset (D14 — gated, BEFORE deploy, each step user-approved):**
  1. Close BTC/ETH via real sim sells (first live sell-path proof; embeds/capture fire).
  2. USER resets the Alpaca paper account in the dashboard (clears phantom ETHUSD).
  3. Restart trader → boot reconciliation trues DB to broker; verify 0 open positions,
     sane obs (`cash_ratio` ≈ 1.0, `exposure` ≈ 0.0), migrations V009+V010 applied.
  4. Run `scripts/record_benchmark_baselines.py --dry-run` → present → `--apply`.
- [ ] Deploys (collector: Tasks 1–2 snapshot half; trader: everything else — one unit
  with the 09:15 config flip) — separate, explicitly gated asks.
- [ ] Runbook: record ≥2 fresh cycles/env under the new schedule as capture evidence
  before go/no-go; first morning cycle also verifies the two flagged assumptions
  (feature freshness at 09:15; fractional fills at the opening-print basis).

## Self-review notes

- Spec D1–D15 each map to exactly one home (D1→1, D2/D3/D4→2, D5→3, D6→4, D7→5, D8→6,
  D10→7, D11/D12→8, D13→9, D14→closeout step, D15→10); D9 lives in the training-redesign
  plan (Task 25b) by design.
- Task ordering constraint: Tasks 4–5 land before 7 (sweep reuses their interfaces);
  Task 2's config flip deploys WITH Task 8 (pre-open cycle needs the async path) —
  single branch/PR/rebuild enforces this; Task 10 lands LAST (touches the same
  embed/alerter files as 3, 6, 8, 9 — avoids merge friction).
- Test snippets that depend on existing fixtures say so explicitly and pin the assertion
  contract; implementers adapt plumbing, not expectations.
- Types: `mark_positions(env, prices)` (Task 4) matches Task 5's assumption that marks
  precede breaker evaluation; `FillResult.realized_pnl` name is consistent across
  Task 6's three files.
