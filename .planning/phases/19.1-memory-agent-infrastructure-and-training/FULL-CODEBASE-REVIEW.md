# SwingRL Comprehensive Code Review Findings

**Date:** 2026-03-18
**Scope:** ~60K LOC, 92 Python files across 16 packages + memory service
**Reviewers:** 6 parallel agents (security, architecture, trading logic, performance, test coverage, memory service)

---

## CRITICAL — Fix Before Going Live (8 items)

### 1. Train/Live Observation Layout Mismatch
**Files:** `envs/base.py:273-308` vs `execution/risk/position_tracker.py:186-242`

Training env uses **interleaved** per-asset layout: `[w0, pnl0, bars0, w1, pnl1, bars1, ...]`
Live PositionTracker uses **grouped** layout: `[all_weights, all_pnls, all_bars]`

The model will misinterpret every portfolio feature during live trading.

**Fix:** Align both to interleaved layout AND expand from 3 to 4 per-asset features (see Fix #2). Change `PositionTracker.get_portfolio_state_array()` to match training env's interleaved layout. Update constants in `features/assembler.py`.

### 2. "Unrealized PnL" Feature — Expand to Both Weight Deviation + Real PnL
**Files:** `envs/base.py:303-304`, `execution/risk/position_tracker.py:230-236`

Training computes `weight_i - 1/n_assets` (weight deviation). Live computes `(price - cost_basis) / cost_basis` (real unrealized PnL). These are different signals in the same observation slot.

**Decision:** Add BOTH as separate features (4 per-asset instead of 3). Per-asset layout becomes: `[weight, weight_deviation, unrealized_pnl, bars_since_trade]`.

**Obs space impact:**
- Equity: 3 + 4×8 = 35 portfolio dims → total obs 156→**164** (+8)
- Crypto: 3 + 4×2 = 11 portfolio dims → total obs 45→**47** (+2)
- No impact on episodes, folds, or training structure — only width of observation vector changes
- Existing models incompatible (shape mismatch) — retraining required before paper trading

**Changes required:**
- `features/assembler.py` — `EQUITY_PORTFOLIO = 35`, `CRYPTO_PORTFOLIO = 11`, `EQUITY_OBS_DIM = 164`, `CRYPTO_OBS_DIM = 47`
- `envs/base.py` `_get_portfolio_state()` — Add cost basis tracking to `PortfolioSimulator`, expand to 4 per-asset fields
- `envs/portfolio.py` — Add `self.cost_basis` array, update in `rebalance()` with weighted-average tracking
- `execution/risk/position_tracker.py` `get_portfolio_state_array()` — Match 4-field interleaved layout
- `data/verification.py` — Update `_CRYPTO_OBS_DIM = 47` and equity equivalent

**Note:** Phase 20 context (line 68) says "NO unrealized P&L or portfolio value" for **memory ingestion** — that decision is about memory agent pattern finding, not the RL observation space. These are separate systems.

### 3. Live Pipeline Should Use Weight-Based Rebalancing (Match Training)
**Files:** `execution/pipeline.py`, `execution/position_sizer.py`, `execution/signal_interpreter.py`

**Problem:** Training env uses weight-based rebalancing (softmax → target weights → rebalance). Live pipeline uses a completely different paradigm: signal interpreter → Kelly position sizer → dollar-amount orders with ATR stop-losses. The model learned to output target allocations, but live translates them through a Kelly/stop-loss layer the model never trained against. Additionally, the old position sizer produced $8 orders on a $400 portfolio (2% Kelly used as dollar amount, not risk budget).

**Decision:** Make live match training (Option B). Remove Kelly/stop-loss translation layer.

**New live flow:**
```
model output → process_actions (softmax + deadzone, same as training) → target weights → rebalance via broker
```

**What changes:**
- `ExecutionPipeline.execute_cycle()` — use `process_actions()` from `envs/portfolio.py` to convert model output to target weights, then submit rebalancing orders to match those weights
- Position sizer (`position_sizer.py`) — no longer used for order sizing; target weight × portfolio value = dollar amount per asset
- Signal interpreter — simplified or removed; the weights ARE the signal
- Stop-losses removed from order submission (no more bracket orders)
- Risk manager still applies as a guardrail (max position size, max exposure, circuit breakers)

**What stays the same:**
- Training env — unchanged
- Circuit breakers — still protect against catastrophic drawdown
- Max position size limits — still enforced
- Risk manager veto — still active

### ~~4. Pipeline Cash Calculation After Fills is Wrong~~ (Resolved by #3)
**File:** `execution/pipeline.py:275`

~~`cash = new_portfolio_value - sum(abs(f.quantity * f.fill_price) for f in fills)` subtracts only current cycle's fills, ignoring pre-existing positions.~~

**Status:** Resolved by #3 redesign. With weight-based rebalancing, cash = `1.0 - sum(weights)` × portfolio value. No separate cash calculation needed.

### 5. Global Circuit Breaker Uses Initial Capital Instead of High-Water Mark ✅ APPROVED
**File:** `execution/risk/circuit_breaker.py:315`

Per-env CB correctly uses high-water mark for drawdown, but global CB uses `_total_initial`. After portfolio growth from $447 to $600, the global CB is far less protective — portfolio could drop 36% before tripping.

**Fix:** Track a global high-water mark, matching the per-env CB pattern:
```python
# In GlobalCircuitBreaker.__init__:
self._total_hwm: float = total_initial

# In check():
total_value = sum(portfolio_values.values())
self._total_hwm = max(self._total_hwm, total_value)
combined_dd = 1.0 - total_value / self._total_hwm if self._total_hwm > 0 else 0.0
```

### 6. DuckDB Persistent Connection → Short-Lived Connections (Option A) ✅ APPROVED
**File:** `data/db.py:91-110`
**Related:** `.planning/phases/22-automated-retraining/22-DUCKDB-CONTENTION.md` (Option A)

**Two problems solved at once:**
1. **Race condition (review finding):** Two threads can both enter `_get_duckdb_conn()` before the connection is set, causing dual `duckdb.connect()` calls.
2. **Training contention (Phase 22):** Persistent connection holds a single-writer lock for hours, blocking the training container or retraining subprocess from accessing DuckDB.

**Current architecture:** `DatabaseManager` holds one persistent `_duckdb_conn`. The `duckdb()` context manager creates cursors on that shared connection. `FeaturePipeline` also receives the raw connection directly via `db._get_duckdb_conn()` (private access, 2 callers).

**Fix:** Switch `duckdb()` context manager to open/close a connection per call (same pattern as `sqlite()` already uses). Remove `_get_duckdb_conn()`, `_duckdb_conn`, and `close()`.

```python
@contextlib.contextmanager
def duckdb(self, read_only: bool = False) -> Generator[Any, None, None]:
    """Yield a short-lived DuckDB connection.

    Opens a fresh connection per call, closes on exit.
    Lock held for milliseconds, not hours.
    """
    self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(self._duckdb_path), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()
```

**Callers that need changes:**

| Caller | Current pattern | New pattern |
|--------|----------------|-------------|
| `data/ingest_all.py:209-210` | `conn = db._get_duckdb_conn()` then `FeaturePipeline(config, conn)` | `with db.duckdb() as conn:` then `FeaturePipeline(config, conn)` |
| `data/verification.py:349-350` | Same private access pattern | Same fix |
| `features/pipeline.py` | Stores `self._conn` from constructor, uses throughout | No change — still receives connection, but caller passes one from `db.duckdb()` context |
| `execution/pipeline.py` | Uses `self._feature_pipeline` (already receives conn externally) | No change to pipeline itself, but whoever constructs `ExecutionPipeline` + `FeaturePipeline` must wrap in `db.duckdb()` |
| All `with db.duckdb() as cursor:` callers (23 occurrences across 10 files) | Receives cursor from persistent conn | Now receives a full connection (not cursor). Use `conn.execute()` directly or `conn.cursor()` if needed. |

**Key decision:** The context manager now yields a **connection** (not a cursor), matching how callers already use `cursor.execute()`. DuckDB connections support `.execute()` directly, so most callers need no change beyond the variable name.

**`read_only` parameter:** Added for Phase 22 — training reads can use `read_only=True` to allow concurrent readers while live trading writes.

**Performance:** ~5ms overhead per open/close. Trading cycles run every 4 hours, ingestion runs daily. Completely negligible.

**`init_schema()` still uses `_write_lock`** to prevent concurrent DDL. Each DDL call opens its own short-lived connection inside the lock.

**`FeaturePipeline` long-lived conn:** `FeaturePipeline` holds its connection for the duration of `compute_equity()`/`compute_crypto()` (seconds, not hours). This is acceptable — the connection is released when the calling `with db.duckdb()` block exits.

**`close()` method:** Simplified — no persistent connection to close. Only resets singleton state for test isolation.

### 7. Missing Auth Header in Meta-Orchestrator ✅ APPROVED
**File:** `memory/training/meta_orchestrator.py:240-246`

`_query_run_config()` posts to `/training/run_config` without `X-API-Key` header. Every LLM-guided hyperparameter request silently fails with 401, making the meta-training feature completely non-functional since it was built.

**Fix:** Add the auth header (one line):
```python
headers = {"Content-Type": "application/json"}
if self._client._api_key:
    headers["X-API-Key"] = self._client._api_key
```

### 8. Blocking Sync SQLite in Async FastAPI Handlers — Priority Thread Pools ✅ APPROVED
**Files:** `services/memory/db.py`, `routers/core.py`, `routers/training.py`, `routers/debug.py`, `memory_agents/*.py`

All `async def` FastAPI handlers call synchronous `sqlite3` functions, blocking the event loop. With Phase 20 adding `/live/*` endpoints (time-sensitive trading cycle queries), live requests must never wait behind slow consolidation work.

**Fix:** Two-tier priority thread pool system using `anyio.to_thread.run_sync()` (already in requirements).

**Pool design:**
- **Live pool (2 threads):** For time-sensitive operations — `/ingest`, `/training/record_outcome`, `/health`, and Phase 20's `/live/*` endpoints. <5ms response for simple DB ops.
- **Background pool (3 threads):** For consolidation work — `/consolidate` (does many sequential DB reads/writes + LLM calls taking 60-240s), `/debug/*`, `/training/run_config`, `/training/epoch_advice` pattern reads.

**Implementation approach:**
1. Add `_run_live()` and `_run_background()` async helpers in `db.py` using `anyio.CapacityLimiter` (2 and 3 respectively)
2. Add async wrapper for every DB function (e.g., `insert_memory_async()`, `get_active_consolidations_async()`)
3. Routers call async wrappers instead of blocking functions
4. Memory agents (`consolidate.py`, `query.py`, `ingest.py`) refactored to call async DB wrappers internally
5. Sync DB functions remain for backward compatibility
6. Graceful shutdown of both pools in FastAPI lifespan hook

**Router → pool mapping:**
| Router/Endpoint | Pool | Reason |
|-----------------|------|--------|
| `POST /ingest` | Live | Called frequently during training, <5ms |
| `POST /training/record_outcome` | Live | Once per training iteration, critical path |
| `GET /health` | Live | Docker healthcheck, must never block |
| `POST /live/*` (Phase 20) | Live | Trading cycle, time-sensitive |
| `POST /consolidate` | Background | Long-running (60-240s), many sequential DB ops |
| `POST /training/run_config` | Background | Pattern reads + LLM call |
| `POST /training/epoch_advice` | Background | Pattern reads + LLM call |
| `GET /debug/*` | Background | Operator can wait |
| `GET /pattern_effectiveness` | Background | Report generation |

**Files requiring changes (9 files):**
1. `services/memory/db.py` — Add async wrappers + pool helpers
2. `services/memory/app.py` — Graceful shutdown in lifespan
3. `services/memory/memory_agents/ingest.py` — `store_async()`
4. `services/memory/memory_agents/query.py` — Async `_build_context()`, `_track_presentations()`
5. `services/memory/memory_agents/consolidate.py` — Async all DB calls in `run_stage1/2()`, `_dedup_and_insert()`
6. `services/memory/routers/core.py` — Call async DB wrappers
7. `services/memory/routers/training.py` — Call async DB wrappers
8. `services/memory/routers/debug.py` — Call async DB wrappers
9. `services/memory/Dockerfile` — Consider `--workers 4` for uvicorn

---

## HIGH — Fix Before Paper Trading (14 items)

### Trading Logic

#### 9. Agent Cannot Hold Cash (Softmax forces 100% invested) ✅ APPROVED
**Files:** `envs/base.py:97-102`, `envs/portfolio.py:136-165`

`process_actions()` uses softmax which always produces weights summing to 1.0. The agent has no ability to sit in cash during a crash. For a capital-preservation system, this is a significant gap.

**Fix:** Add a "cash" dimension to the action space. Apply softmax to all `n_assets + 1` elements. Last element = cash weight. First `n_assets` elements = asset weights (sum < 1.0, remainder = cash).

**Action space change:**
- Equity: `(8,)` → `(9,)` (8 assets + 1 cash)
- Crypto: `(2,)` → `(3,)` (2 assets + 1 cash)

**`process_actions()` new logic:**
```python
def process_actions(raw_actions, current_weights, deadzone):
    # raw_actions shape: (n_assets + 1,) — last element = cash preference
    shifted = raw_actions - np.max(raw_actions)
    exp_vals = np.exp(shifted)
    all_weights = exp_vals / np.sum(exp_vals)  # sums to 1.0 across n_assets + 1
    # First n_assets are investment weights, last is cash
    target_weights = all_weights[:-1]  # shape (n_assets,), sum < 1.0
    # Apply deadzone on asset weights only
    diff = np.abs(target_weights - current_weights)
    return np.where(diff >= deadzone, target_weights, current_weights)
```

**Files requiring changes (only 2 core files + tests):**

| File | Change | Complexity |
|------|--------|-----------|
| `envs/base.py:100` | `shape=(self._n_assets + 1,)` | One line |
| `envs/portfolio.py:156-165` | New `process_actions()` with cash extraction | Core logic |
| `tests/test_envs.py` | Add cash dimension tests | Validation |

**Files that need NO changes (verified):**
- `trainer.py` — SB3 auto-infers action space from env ✓
- `ensemble.py` — `blend_actions()` is element-wise, works on any shape ✓
- `pipeline.py` — passes actions through, shape-agnostic ✓
- `signal_interpreter.py` — loops over `enumerate(symbols)` (0 to n_assets-1), ignores cash element ✓
- `shadow_runner.py` — passes actions to signal interpreter, same as above ✓
- `assembler.py` — observation assembly, unrelated to action space ✓
- VecNormalize — only normalizes obs/rewards, never actions ✓

**Deadzone:** Applied to asset weights only, not cash. Cash preference always takes effect.

**Rebalance compatibility:** `PortfolioSimulator.rebalance()` already handles `target_weights` summing to < 1.0 — remainder stays as cash. No change needed.

**Requires retraining** (already needed for Fix #1+2). Old models incompatible (shape mismatch).

**Both training and live aligned:** Training env defines the action space → SB3 learns it → live `model.predict()` outputs same shape → `process_actions()` in live pipeline uses identical function → weights go to broker rebalancing (Fix #3).

#### 10. VecNormalize Shared Across Algos in Live Inference ✅ APPROVED
**Files:** `execution/pipeline.py:162-169, 390-410`, `shadow/shadow_runner.py:215-234`, `shadow/lifecycle.py:274-286`

**Three related bugs found during research:**

**Bug A (pipeline.py):** `_normalize_observation()` iterates models dict and returns on first VecNormalize found (PPO, since it's first in `_ALGO_NAMES`). All three models receive PPO-normalized observations. A2C and SAC get out-of-distribution inputs.

**Bug B (shadow_runner.py):** `_maybe_normalize_obs()` tries `model.get_vec_normalize_env()` which doesn't work for loaded models. Never loads the .pkl files at all. Shadow inference runs on completely unnormalized observations.

**Bug C (lifecycle.py):** `_load_vec_normalize()` calls `VecNormalize.load(path)` without the required `venv` parameter. Will crash with TypeError at runtime.

**Fix A — Pipeline per-model normalization:**
Change `_normalize_observation()` to return a dict of per-algo normalized observations. Then in `execute_cycle()`, each model gets its own normalized obs before `predict()`.

**Fix B — Shadow runner loads .pkl files:**
Change `_maybe_normalize_obs()` to accept a `vec_normalize_path` parameter, load the saved stats using DummyVecEnv wrapper (matching the pattern in `backtest.py` which does this correctly).

**Fix C — Lifecycle adds missing venv:**
Add `DummyVecEnv([lambda: None])` as the required venv parameter to `VecNormalize.load()`.

**SB3 technical note:** `normalize_obs()` works on single observations `(obs_dim,)` — no batching required. Returns a deepcopy, doesn't modify input. Safe to call per-model. SB3 requires the VecNormalize .pkl format (accepted risk, model files are internal only).

**Files requiring changes:**

| File | Change | Complexity |
|------|--------|-----------|
| `execution/pipeline.py` | `_normalize_observation()` returns dict, `execute_cycle()` uses per-model obs | Core fix |
| `shadow/shadow_runner.py` | Load .pkl files, pass to normalize function | Moderate |
| `shadow/lifecycle.py` | Add DummyVecEnv to `_load_vec_normalize()` | One line |

#### 11. Shadow Promotion Overhaul — Align with Training Gates ✅ APPROVED
**File:** `shadow/promoter.py` (full rewrite of `_compute_sharpe`, `_compute_mdd`, `_check_cb_during_shadow`, and `evaluate_shadow_promotion`)

**5 problems found during research:**
1. Sharpe/MDD computed from sequential cross-asset trade prices (meaningless data)
2. Sharpe is un-annualized (training gates use annualized)
3. Active model's Sharpe recomputed with the same broken method (line 77)
4. No profit factor gate (training requires > 1.5)
5. CB event query counts all-time events, not shadow period only

**Fix — Align shadow promotion with training gate framework:**

**A. Data source:** Use `portfolio_snapshots` table (records total portfolio value after each cycle) instead of `trades` table prices. Compute returns from portfolio value series. Both shadow and active use the same table, same method.

**B. Annualized Sharpe:** Use `agents/metrics.py` functions (same as training) with `periods_per_year` (252 equity, 2191.5 crypto).

**C. Active model comparison:** Recalculate active model's Sharpe from `portfolio_snapshots` using the same annualized method. Shadow Sharpe must be strictly **higher** than active Sharpe (not just >=).

**D. Add profit factor gate:** Compute from `trades`/`shadow_trades` table (buy/sell PnL). Threshold: > 1.5 (matching training gates).

**E. CB date filter:** Add shadow period date range to CB query:
```python
"SELECT COUNT(*) FROM circuit_breaker_events WHERE environment = ? AND triggered_at >= ?",
(env_name, shadow_start_date),
```

**Updated gate criteria (4 gates, all must pass):**

| Gate | Criteria | Threshold |
|------|----------|-----------|
| Sharpe | Shadow annualized Sharpe > Active annualized Sharpe | Strictly greater |
| MDD | Shadow MDD <= tolerance × Active MDD | Configurable tolerance |
| Profit Factor | Shadow PF > minimum | > 1.5 (matches training) |
| Circuit Breaker | No CB events during shadow period only | Must be zero |

**Reuse opportunity:** Import metric functions from `agents/metrics.py` (annualized Sharpe, MDD from returns) rather than reimplementing. Single source of truth for metric definitions.

#### 12. Shadow Runner Hardcoded Estimated Prices ✅ APPROVED
**File:** `shadow/shadow_runner.py:167`

`estimated_price = 100.0 if env_name == "equity" else 50000.0` — never refreshed. BTC at $90K would be estimated at $50K, producing incorrectly sized shadow trades that corrupt shadow evaluation metrics.

**Fix:** Batch-fetch latest close prices from DuckDB at the start of each shadow cycle, then use per-symbol prices for each signal. `ctx.db` (DatabaseManager) is already available in the shadow runner.

**Implementation:**
```python
def _fetch_latest_prices(db: Any, env_name: str, symbols: list[str]) -> dict[str, float]:
    """Fetch latest close price per symbol from DuckDB."""
    table = "ohlcv_daily" if env_name == "equity" else "ohlcv_4h"
    date_col = "date" if env_name == "equity" else "datetime"
    prices: dict[str, float] = {}
    try:
        with db.duckdb() as conn:
            for symbol in symbols:
                row = conn.execute(
                    f"SELECT close FROM {table} WHERE symbol = ? ORDER BY {date_col} DESC LIMIT 1",
                    [symbol],
                ).fetchone()
                if row and row[0] is not None:
                    prices[symbol] = float(row[0])
    except Exception:
        log.warning("shadow_price_fetch_failed", env=env_name)
    return prices

# In _generate_hypothetical_trades(), at top:
symbols = config.equity.symbols if env_name == "equity" else config.crypto.symbols
latest_prices = _fetch_latest_prices(ctx.db, env_name, symbols)

# Per signal:
estimated_price = latest_prices.get(signal.symbol, 100.0 if env_name == "equity" else 50000.0)
```

**Key details:**
- Single DB round-trip per shadow cycle (not per signal)
- DuckDB primary key on `(symbol, date/datetime)` makes queries efficient
- Hardcoded values remain as last-resort fallback only (e.g., empty DB)
- Pattern matches `ExecutionPipeline._get_latest_atr()` style
- Shadow runner has no broker API access (post-cycle, no adapters) — DB-only is correct

#### 13. Alpaca Fill Assumes Immediate Fill ✅ APPROVED
**File:** `execution/adapters/alpaca_adapter.py:119`

`filled_avg_price` could be `None` if order is accepted but not yet filled (pre-market queue, API hiccup). `float(None)` crashes with TypeError.

**Context from research:**
- **Training:** Fills are instant (PortfolioSimulator is pure accounting, no broker)
- **Equity paper:** Real Alpaca paper API. Market orders fill instantly during market hours, but can be queued pre-market.
- **Crypto paper:** Simulated fills with real mid-prices from Binance.US order book. Always instant.
- With #3 redesign (weight-based rebalancing), bracket orders go away → simple market orders, but None check still needed as defensive code.

**Fix:** Check for None before casting. Return zero-fill result if not yet filled:
```python
if response.filled_avg_price is None:
    log.warning("order_not_immediately_filled",
        order_id=str(response.id), status=str(response.status))
    return FillResult(
        symbol=sized.symbol, side=sized.side,
        quantity=0.0, fill_price=0.0,
        order_id=str(response.id), status="pending",
    )
```
Pipeline handles zero-fill gracefully (no position update, no snapshot write). Order will either fill later (Alpaca manages it) or expire.

#### 14. EST Hardcoded in Emergency Stop (Ignores EDT) ✅ APPROVED
**File:** `execution/emergency.py:83-95`

`_is_extended_hours()` uses `et_hour = (now.hour - 5) % 24` which hardcodes EST (UTC-5) and ignores EDT (UTC-4). Off by 1 hour for ~7 months/year (March-November). This is on the emergency liquidation path — wrong market state detection could use the wrong liquidation strategy.

**Only occurrence in codebase.** The same file already uses `exchange_calendars` correctly for regular market hours (line 107: `nyse.is_open_on_minute(now)`), but extended hours aren't covered by exchange_calendars.

**Fix:** Use `zoneinfo.ZoneInfo` (Python 3.11 stdlib, no new dependency). Automatically handles DST transitions:
```python
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

def _is_extended_hours() -> bool:
    et_now = datetime.now(_ET)
    return (4 <= et_now.hour < 9) or (16 <= et_now.hour < 20)
```

One-file change, 3 lines. No impact on other modules.

#### 15. Equity Env Always Starts at Step 0 ✅ APPROVED
**File:** `envs/equity.py:50-52`

Every equity episode starts at bar 0 (always the same ~2016 start). Model memorizes specific trajectory instead of learning generalizable patterns. Crypto correctly uses random starts via `self.np_random.integers()`.

**Data sufficiency:** 20,480 equity bars / 252 bars per episode = ~81 possible non-overlapping episodes, 20,228 possible random starting points. More than enough.

**Fix:** Copy crypto's pattern exactly. One method change in `equity.py`:
```python
def _select_start_step(self) -> int:
    """Select random start step within the training window."""
    max_start = len(self._features) - self._episode_bars
    if max_start <= 0:
        return 0
    return int(self.np_random.integers(0, max_start))
```

**How it works with `reset()`:** `base.py:138` calls `self._select_start_step()` during reset. `super().reset(seed=seed)` initializes `self.np_random` (Gymnasium's seeded RNG) before the subclass method is called, ensuring reproducibility when a seed is provided.

**No other files need changes.** The `_max_step` calculation in `base.py:142-145` already handles variable start positions correctly (caps at `len(features) - 1`).

**Impact on training:** Model will initially learn slower (harder problem with diverse starting conditions), but should generalize much better across market regimes. Retraining already required for #1+2+9.

### Architecture

#### 16. BinanceIngestor/GapFill Sessions Never Closed (TCP/Socket Leak) ✅ APPROVED
**Files:** `data/binance.py:97`, `data/gap_fill.py:193`

Two leak patterns found. Alpaca and FRED ingestors are fine (use SDK clients that manage sessions internally).

**Leak A — BinanceIngestor (instance-level):** `self._session = requests.Session()` created in `__init__`, used in `_fetch_klines()` and `_download_archives()`, never closed. Holds TCP pool for entire ingestor lifetime.

**Leak B — gap_fill.py (per-function):** New `session = requests.Session()` created inside `_fetch_binance_global_klines()` on every call, never closed. Leaks one session per gap fill attempt.

**Fix A — BinanceIngestor:**
Add `close()` + context manager protocol:
```python
def close(self) -> None:
    if hasattr(self, '_session') and self._session:
        self._session.close()

def __enter__(self): return self
def __exit__(self, *args): self.close()
```
Update callers in `ingest_all.py:120` and `binance.py` CLI main to use `with BinanceIngestor(config) as ingestor:`.

**Fix B — gap_fill.py:**
Use context manager on the session:
```python
with requests.Session() as session:
    resp = session.get(url, params=params, timeout=30)
```

**Files requiring changes:**

| File | Change |
|------|--------|
| `data/binance.py` | Add `close()`, `__enter__`, `__exit__` |
| `data/gap_fill.py:193` | Wrap session in `with` block |
| `data/ingest_all.py:120` | Use `with BinanceIngestor(config) as ingestor:` |

#### 17. Execution Pipeline Creates New Adapter Per Cycle (Connection Leak) ✅ APPROVED
**File:** `execution/pipeline.py:372-388`

`_get_adapter()` creates a new AlpacaAdapter (with two SDK clients: TradingClient + StockHistoricalDataClient) or BinanceSimAdapter on every call. Over days/weeks of scheduled cycles, hundreds of abandoned adapters leak HTTP connections.

**Fix:** Cache adapters. Add `self._adapters: dict[str, Any] = {}` alongside existing lazy-init dict `self._models` in `__init__` (line 79 area). Return cached adapter on subsequent calls:

```python
# In __init__ (line 79 area, alongside self._models):
self._adapters: dict[str, Any] = {}

# Replace _get_adapter():
def _get_adapter(self, env_name: str) -> Any:
    if env_name not in self._adapters:
        if env_name == "equity":
            from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter
            self._adapters[env_name] = AlpacaAdapter(config=self._config, alerter=self._alerter)
        else:
            from swingrl.execution.adapters.binance_sim import BinanceSimAdapter
            self._adapters[env_name] = BinanceSimAdapter(config=self._config, db=self._db, alerter=self._alerter)
    return self._adapters[env_name]
```

One-file change. Follows the same lazy-cache pattern already used for `self._models`.

#### 18. Bare ValueError Raises (17 occurrences) — Violates CLAUDE.md ✅ APPROVED
**17 ValueErrors across 7 files. 4 existing tests need updating, 14 new tests needed.**

**Migration map:**

| File | Count | Current | Replacement | Existing Tests | New Tests Needed |
|------|-------|---------|-------------|----------------|-----------------|
| `config/schema.py` | 4 | `ValueError` in Pydantic validators | `ConfigError` | 3 (catch `ValidationError`, no change needed) | 0 |
| `agents/backtest.py` | 1 | Insufficient folds | `DataError` | 2 (update `pytest.raises`) | 0 |
| `training/ensemble.py` | 1 | Empty actions dict | `ModelError` | 0 | 1 (empty actions) |
| `memory/training/curriculum.py` | 1 | Crisis pct exceeded | `DataError` | 2 (update `pytest.raises`) | 0 |
| `envs/equity.py` | 4 | Array shape validation in `from_arrays()` | `DataError` | 0 | 4 (1D features, 1D prices, step mismatch, asset count) |
| `envs/crypto.py` | 4 | Array shape validation in `from_arrays()` | `DataError` | 0 | 4 (same as equity) |
| `features/hmm_regime.py` | 3 | Model not fitted / all inits failed | `ModelError` | 0 | 3 (all-init-fail, warm_start before fit, predict before fit) |

**Test changes:**
- **Update 4 existing tests:** Change `pytest.raises(ValueError)` → `pytest.raises(DataError)` in `test_backtest.py` (2) and `test_curriculum.py` (2)
- **Add 12 new tests:** Env array validation (8), ensemble empty actions (1), HMM lifecycle (3)
- **Config tests unchanged:** Pydantic wraps any exception in `ValidationError`, callers already catch that

**Production code changes:** 7 files, mechanical replacement. Each raise site already has descriptive error message — just change the exception class and add the import.

#### 19. retry(reraise=False) Wraps Exceptions in RetryError ✅ APPROVED
**File:** `utils/retry.py:64`

`swingrl_retry` decorator uses `reraise=False`, so after exhausting retries, tenacity raises `RetryError` wrapping the original exception. Callers catching specific types (e.g., `except ConnectionError`) won't catch it.

**Current usage:** Not yet used in production code (defined + tested only). But will be used in Phase 20+ for broker API retries. Fixing now prevents future bugs.

**Fix:** Change `reraise=False` → `reraise=True` on line 64. One character.

**Test update:** `tests/test_retry.py:40` currently expects `RetryError`. Change to expect `ConnectionError` (the original exception):
```python
# BEFORE:
with pytest.raises(RetryError):
    always_fails()

# AFTER:
with pytest.raises(ConnectionError):
    always_fails()
```

Remove the `from tenacity import RetryError` import from the test file (no longer needed).

**Files:** `utils/retry.py` (1 line), `tests/test_retry.py` (update 1 test + remove 1 import).

### Memory Service

#### 20. Duplicated Bounds Constants → Move to Config YAML ✅ APPROVED
**Files:** `services/memory/memory_agents/query.py:42-56` vs `src/swingrl/memory/training/bounds.py:28-42`

10 bounds (6 hyperparam + 4 reward weight) manually duplicated across two Docker containers. Cannot share Python imports (cross-container boundary). Server-side also lacks `batch_size` power-of-2 rounding — potential divergence.

**Fix:** Move bounds to `config/swingrl.yaml` (already mounted in both containers at `/app/config`). Single source of truth.

**Config addition:**
```yaml
training:
  bounds:
    hyperparam_bounds:
      learning_rate: [1e-5, 1e-3]
      entropy_coeff: [0.0, 0.05]
      clip_range: [0.1, 0.4]
      n_epochs: [3, 20]
      batch_size: [32, 512]
      gamma: [0.90, 0.9999]
    reward_bounds:
      profit: [0.10, 0.70]
      sharpe: [0.10, 0.60]
      drawdown: [0.05, 0.50]
      turnover: [0.00, 0.20]
```

**How each side reads it:**
- **Client** (`bounds.py`): Loads via `load_config()` → `config.training.bounds.hyperparam_bounds`. Already has `load_config()` pattern.
- **Server** (`query.py`): Reads mounted YAML directly via `yaml.safe_load()` — same pattern already used by `consolidate.py` for consolidation config (lines 60-84).

**Files requiring changes:**

| File | Change |
|------|--------|
| `config/schema.py` | Add `TrainingBoundsConfig` Pydantic model nested in `TrainingConfig` |
| `config/swingrl.yaml` | Add `training.bounds` section |
| `memory/training/bounds.py` | Read from config instead of hardcoding. Keep `clamp_run_config()` and `clamp_reward_weights()` functions. |
| `services/memory/memory_agents/query.py` | Read from mounted YAML instead of inline `_HYPERPARAM_BOUNDS`/`_REWARD_BOUNDS`. Add `batch_size` power-of-2 rounding (currently missing). |

#### 21. IngestAgent XML Not Sanitized (Prompt Injection Surface) ✅ APPROVED
**File:** `services/memory/memory_agents/ingest.py:32`

`wrapped = f"{_XML_OPEN}{text}{_XML_CLOSE}"` — if ingested text contains `</memory>` literally, it breaks the XML boundary. Consolidation agent (`consolidate.py:672`) uses `<training_data>` tags to ground LLM output; broken XML delimiters could let content escape into prompt space.

**Fix:** One line — escape before wrapping:
```python
import html
safe_text = html.escape(text)
wrapped = f"{_XML_OPEN}{safe_text}{_XML_CLOSE}"
```

`html.escape()` converts `<` → `&lt;`, `>` → `&gt;`, `&` → `&amp;`. The stored text preserves meaning for human reading and LLM interpretation, but can't break XML structure.

**One file, one line.** No impact on query/consolidation agents — they read stored text as-is, and `&lt;memory&gt;` in content won't match the real `<memory>` delimiter.

#### 22. stop_training Signal Ignored by Epoch Callback ✅ APPROVED
**Files:** `services/memory/routers/training.py:59-64`, `src/swingrl/memory/training/epoch_callback.py:349-351`

Server `EpochAdviceResponse` defines `stop_training: bool = False` (line 63). Callback reads `body.get("reward_weights")` and `body.get("rationale")` but ignores `stop_training` entirely. If the LLM advises stopping (model plateaued, diverging), training continues wasting compute.

**Fix:** Add 4 lines after line 350 in `epoch_callback.py`, before the reward_weights processing:
```python
new_weights = body.get("reward_weights")
reason = body.get("rationale", "")

# Check stop_training signal
stop_training = body.get("stop_training", False)
if stop_training:
    log.warning("llm_advises_stop_training", epoch=self._epoch, reason=reason)
    self.model.stop_training = True  # SB3 graceful stop at next rollout boundary
    return  # Skip reward weight adjustment — stopping anyway
```

`self.model` is the SB3 algorithm instance accessible via `BaseCallback.model`. Setting `stop_training = True` causes `model.learn()` to exit cleanly after the current rollout completes — no data loss, model is saveable.

**One file, 4 lines.** No server-side changes needed — the field is already returned.

---

## MEDIUM — Fix Before Live Trading (17 items)

### Performance (highest-impact)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 23 | ~~Turbulence loads entire OHLCV table every step (20K+ rows)~~ ✅ FIXED | `features/pipeline.py:443-462` | Added lower-bound date filter (2520 days equity, 250 days crypto) |
| 24 | Macro query uses Python iterrows over 10K rows to pick 6 values ✅ APPROVED | `features/pipeline.py:382-421` | Replace with `ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC)` window query — returns exactly 6 rows. Also remove dead identical if/else branches. Low urgency, trivial fix. |
| 25 | Gap detection + data pipeline overhaul ✅ APPROVED | `data/gap_fill.py`, `data/base.py`, `data/db.py`, new `data/tiingo.py` | Full overhaul — see "Data Pipeline Overhaul" section below for details. |
| 26 | O(n^2) turbulence series — recomputes covariance at every index ✅ APPROVED | `features/turbulence.py:114-129` | Incremental EWMA update in `compute_series()` — see details below |
| 27 | Sequential symbol ingestion — no parallelism ❌ SKIPPED | `data/alpaca.py:178`, `data/binance.py:524` | Not worth the complexity. DuckDB is single-writer so parallel fetches would still serialize on write. ~20s savings on a daily unattended job. Revisit if symbol count grows significantly. |
| 28 | New `requests.Session` per gap-fill API call — RESOLVED BY #25 | `data/gap_fill.py:193` | File gets rewritten in data pipeline overhaul (#25). New code will handle sessions properly. Also overlaps with #16B (session leak). |
| 29 | Feature pipeline reads SPY/BTC OHLCV twice ✅ APPROVED | `features/pipeline.py:138-183` | Move HMM fitting inside the symbol loop — when `symbol == "SPY"`, fit HMM right there using the already-fetched OHLCV. Same for BTC in crypto path. Eliminates second DB read entirely. No caching needed. |

### Turbulence Quality (from deep research)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 40 | ~~Expanding equity covariance dilutes regime detection — switch to EWMA~~ ✅ FIXED | `features/turbulence.py` | Replaced `np.cov()` with EWMA covariance (half-life 126 days). EquityTurbulenceCalculator class. |
| 41 | ~~2-asset crypto Mahalanobis is overkill — simplify to rolling vol + correlation~~ ✅ FIXED | `features/turbulence.py` | New CryptoTurbulenceCalculator: `max(0, vol_zscore) * (1 + corr_spike)`. TurbulenceCalculator is now a factory function. |

### Correctness

| # | Issue | File | Fix |
|---|-------|------|-----|
| 30 | Transaction costs not deducted from share allocation (optimistic bias) ✅ APPROVED | `envs/portfolio.py:86-100` | Estimate cost first, allocate from `total_value - est_cost`. Applies to both training and live (Fix #3 makes live use same PortfolioSimulator). See details below. |
| 31 | Calmar ratio divides by `cum[0]` instead of 1.0 ✅ APPROVED | `agents/metrics.py:99` | `cum[0]` is `1 + returns[0]`, not `1.0`. Fix: `total_return = float(cum[-1] - 1.0)`. One-line fix. |
| 31b | Sortino downside deviation uses wrong denominator (NEW from ratio audit) ✅ APPROVED | `agents/metrics.py:68-73` | Divides by count of negatives instead of total count — understates Sortino by ~sqrt(count_neg/N). Fix: `downside = np.minimum(excess, 0.0)` then `np.mean(downside**2)` over all periods. Edge cases: all-wins → 999.0, all-zeros → 0.0 (avoids inf/nan in DuckDB and SQLite). |
| 31c | Reward wrapper rolling Sharpe uses ddof=0 (NEW from ratio audit) ✅ APPROVED | `memory/training/reward_wrapper.py:185` | Change `np.std(arr)` to `np.std(arr, ddof=1)`. One-character fix. |
| 32 | ~~Turbulence 90th pct uses ATR but compared against Mahalanobis distance~~ ✅ FIXED | `execution/pipeline.py:452-453` | Added `turbulence` column to feature tables schema, query now uses `QUANTILE(turbulence, 0.9)` |
| 33 | CB event query counts all-time events, not shadow-period only — DUPLICATE OF #11E | `shadow/promoter.py:227-232` | Already covered by Fix #11E (shadow promoter overhaul). Same file, same query. |
| 34 | EnsembleBlender computes adaptive window but never uses it (dead code) ✅ APPROVED | `training/ensemble.py:115-136` | Remove dead code: `VALIDATION_WINDOWS` dict, `DEFAULT_TURBULENCE_THRESHOLD`, `turbulence_threshold` from `__init__`, `turbulence` param + window logic from `compute_weights()`. Keep `sharpe_softmax_weights()` and `blend_actions()`. Research confirmed static weights are correct for 3 correlated agents (DeMiguel 1/N argument). Future path: memory agent `live_blend_weights` category already scaffolded in consolidation/query agents — Phase 20+ can add live endpoint to let memory agent nudge weights per regime. |
| 35 | PositionSizer uses static Kelly defaults — DEPENDENT ON #3 + #11 | `execution/position_sizer.py:71-73` | Currently used in `execution/pipeline.py:24,86` (live) and `shadow/shadow_runner.py:111,155` (shadow). Fix #3 replaces it in the execution pipeline, Fix #11 reworks the shadow runner. After both are implemented: grep codebase for remaining `PositionSizer` references, update/remove tests (`test_position_sizer.py`, `test_shadow_runner.py`), then delete `position_sizer.py`. |
| 36 | FeaturePipeline silently returns zero arrays on macro/HMM errors ✅ APPROVED | `features/pipeline.py:419-438` | Full `FeatureHealthTracker` — see details below. Agent receives out-of-distribution observations when features fail (all-zeros macro never seen in training). Block trading after 3 consecutive failures. |

### Security

| # | Issue | File | Fix |
|---|-------|------|-----|
| 37 | Dashboard port exposed to all interfaces ❌ SKIPPED | `docker-compose.prod.yml:111` | Binding to `127.0.0.1` inside Docker makes the port unreachable from the host. Dashboard needs to be accessible from the homelab network. Homelab is behind firewall. |
| 38 | Data ingestors don't validate API keys on startup ✅ APPROVED | `data/alpaca.py:63`, `data/fred.py:62` | Add early validation in `__init__` for `AlpacaIngestor` (ALPACA_API_KEY + SECRET) and `FredIngestor` (FRED_API_KEY). Raise `ConfigError` if key is empty/None. Mirror the pattern in `AlpacaAdapter` which already validates correctly. Tiingo ingestor (Fix #25) must include validation from day one. |
| 39 | API key config fields use `str` not `SecretStr` ❌ SKIPPED | `config/schema.py:245` | Not worth the `.get_secret_value()` friction. Risk mitigated by: keys in `.env` not YAML (#38 validates on startup), structlog doesn't auto-log config objects, `.env.example` updated with all keys documented. |

---

## Data Pipeline Overhaul (Fix #25, expanded scope)

Original finding was a minor SQL optimization (gap detection in Python loop). During review, expanded into a full data pipeline overhaul addressing multiple issues across ingestion, gap-fill, source tracking, and historical depth.

### 1. Tiingo as equity historical source

- **New dependency:** `tiingo` Python package (free tier, official REST API)
- **Backfill:** Raw OHLCV for all 8 ETFs from **2002-01-01** to present (clean start, all ETFs well past inception — VTI newest at 2001-05-24)
- **Daily response includes 14 fields:** raw OHLCV + adjusted OHLCV + `divCash` + `splitFactor` per row
- **Rate limits:** 50 req/hour, 1,000/day, 500 symbols/month — 8 ETFs is trivial
- **Verification step:** Register free Tiingo API key, hit meta endpoint for VTI to confirm `startDate` ≤ 2002-01-01
- **API key validation:** `TiingoIngestor.__init__()` must validate `TIINGO_API_KEY` on startup — raise `ConfigError` if empty/None. Match pattern from Fix #38 (fail fast, not cryptic errors minutes into ingestion).

### 2. Store raw prices, adjust at compute time

- **Problem:** Current Alpaca data uses `Adjustment.ALL` — adjusted prices are retroactively recalculated every dividend/split. Stored data becomes stale. Training and live see different feature scales.
- **Solution:** Store raw (unadjusted) OHLCV only. Apply adjustment factors during feature computation.
- **New table:** `corporate_actions (symbol TEXT, date DATE, event_type TEXT, value DOUBLE, PRIMARY KEY (symbol, date, event_type))` — populated from Tiingo's `divCash`/`splitFactor` fields (included free in every daily response).
- **Re-ingest Alpaca as RAW:** Change `Adjustment.ALL` → `Adjustment.RAW` in `alpaca.py`. One-time re-backfill.
- **Feature pipeline change:** Apply cumulative adjustment factors before computing technical indicators.

### 3. Source mapping

| Environment | Primary (live/incremental) | Historical backfill | Gap-fill fallback |
|---|---|---|---|
| Crypto | Binance.US | Binance Global archives (pre-2019) | Binance Global API |
| Equity | Alpaca (RAW) | Tiingo (2002-present, RAW) | Tiingo |

### 4. Unified gap-fill pipeline

Replace separate detect/fill functions with a single pipeline:
```
detect_gaps(env) → try_primary(gaps) → try_fallback(remaining) → ffill_residual(≤2 bars) → flag_unfillable(remaining)
```

- **Gap detection:** SQL `LEAD()` window function — returns only gap boundaries, not all timestamps
- **No arbitrary thresholds:** Detect all gaps > 1 expected interval (4h crypto, 1 day equity). Fill regardless of size.
- **Volume normalization:** Fix gap-fill parser to use `volume_quote` with `base * close` fallback (match `binance.py:323-327` archive parser — currently missing in `gap_fill.py:264`)
- **Session management:** All HTTP sessions (`requests.Session`) must be properly closed. Use context managers (`with requests.Session() as session:`) for gap-fill API calls. Resolves #28 and #16B for this file.

### 5. Source tracking

- **`ohlcv_daily`:** Add `source TEXT` column to DDL (currently missing)
- **`ohlcv_4h`:** Already has `source TEXT` but `BaseIngestor` doesn't set it — all Binance.US rows are NULL
- **Fix `BaseIngestor._build_sync_df()`:** Add `source` to `_OHLCV_4H_COLUMNS` and `_OHLCV_DAILY_COLUMNS`
- **Set source values:** `'binance_us'` for crypto, `'alpaca'` for equity, `'tiingo'` for backfill, `'binance_global'` for gap-fill (already set)
- **Backfill NULLs:** Migration `UPDATE` to set correct source on existing rows

### 6. Known gaps table

```sql
CREATE TABLE IF NOT EXISTS known_gaps (
    symbol TEXT NOT NULL,
    environment TEXT NOT NULL,
    gap_start TIMESTAMP NOT NULL,
    gap_end TIMESTAMP NOT NULL,
    gap_hours DOUBLE,
    status TEXT DEFAULT 'unfillable',  -- 'unfillable', 'filled', 'accepted'
    reason TEXT,  -- 'exchange_down', 'migration', etc.
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, gap_start)
)
```

- Populated when fill attempts fail
- Gap detection query excludes rows matching `known_gaps` with `status = 'unfillable'`
- Pre-seed with 4 known unfillable crypto gaps: 2018-02-07→09 (exchange down), 2019-08-31→09-23 (Binance.US migration)

### 7. Overlap validation

- Tiingo vs Alpaca for 2016-present (~8 years overlap)
- Close price tolerance: ±0.01% (raw prices should match exactly, only rounding differences)
- Volume tolerance: ±10% (different aggregation methods)
- NYSE trading calendar as ground truth for missing days
- Run validation once during initial backfill, flag discrepancies

### Files affected

| File | Change |
|------|--------|
| `data/tiingo.py` | **New** — TiingoIngestor for daily OHLCV + corporate actions |
| `data/gap_fill.py` | Unified pipeline, SQL `LEAD()` detection, `known_gaps` integration, volume normalization fix |
| `data/base.py` | Add `source` to column maps and `_build_sync_df()` |
| `data/db.py` | Add `source TEXT` to `ohlcv_daily` DDL, add `known_gaps` + `corporate_actions` tables |
| `data/alpaca.py` | Change `Adjustment.ALL` → `Adjustment.RAW` |
| `data/binance.py` | No changes (already correct) |
| `features/pipeline.py` | Apply adjustment factors before computing technical indicators |
| `config/schema.py` | Add Tiingo API key config field |
| `config/swingrl.yaml` | Add Tiingo API key placeholder |

---

## Implementation Notes (fixes applied during review)

### Turbulence Overhaul (Fixes #23, #32, #40, #41)

**Decision:** After deep research into turbulence calculation methods (Mahalanobis, GARCH, realized vol, VIX proxy, absorption ratio, SRISK/CoVaR, EWMA), we confirmed our Mahalanobis approach is sound for equity but identified improvements for both environments.

**What changed (6 files, 20 new tests, all passing):**

| File | Change |
|------|--------|
| `features/turbulence.py` | Full rewrite. `EquityTurbulenceCalculator` (EWMA Mahalanobis, 126-day half-life) + `CryptoTurbulenceCalculator` (vol z-score × correlation spike composite). `TurbulenceCalculator` is now a backward-compatible factory function. |
| `features/pipeline.py` | Added lower-bound date filters on turbulence SQL queries (2520 days equity, 250 days crypto). Fixed INSERT statements to use explicit column lists for new `turbulence` column. |
| `features/schema.py` | Added `turbulence DOUBLE` column to `features_equity` and `features_crypto` DDL. |
| `execution/pipeline.py` | Fixed `_get_turbulence_90th_pct()` — queries `QUANTILE(turbulence, 0.9)` instead of `QUANTILE(atr_14_pct, 0.9)`. Added `WHERE turbulence IS NOT NULL` for graceful handling before first batch run. |
| `tests/features/test_turbulence.py` | 20 tests: factory (2), equity EWMA (6), crypto composite (6), compute_series (4), EWMA internals (2). |
| `tests/features/test_config_and_schema.py` | Updated INSERT tests to include `turbulence` column value. |

**Why EWMA for equity:** Expanding window dilutes regime detection after years of calm data. A shock extreme relative to the last 6 months but mild relative to 10 years produces low turbulence. EWMA (half-life 126 days, JP Morgan RiskMetrics standard) exponentially downweights old data, keeping sensitivity to recent regime shifts.

**Why composite for crypto:** With only 2 assets (~90% correlated), the 2x2 covariance matrix is nearly singular. Matrix inversion amplifies estimation noise. The composite `max(0, vol_zscore) * (1 + corr_spike)` directly measures the two signals (unusual vol + correlation breakdown) without matrix inversion — same information, less noise. Produces a cleaner training signal for the RL agent.

**Not changed:** VIX (already in observation space as macro feature), absorption ratio (needs more assets), GARCH (adds complexity with marginal benefit over EWMA), SRISK/CoVaR (banking regulation tools, irrelevant for portfolio management).

**Requires:** Batch feature pipeline re-run to populate `turbulence` column. Until then, `_get_turbulence_90th_pct()` returns 0.0 gracefully. Retraining already required for observation space changes (#1, #2, #9).

### Fix #26 — Incremental EWMA in compute_series()

**Problem:** `compute_series()` loops over all bars and calls `compute()` per bar. Each `compute()` call runs `_ewma_covariance()` over `returns[:current_idx]` — O(n) work per step, O(n^2) total. For 20K equity bars × 8 assets, this is ~200M weighted operations during batch feature computation.

**Fix:** Add an incremental EWMA update path in `compute_series()`. The EWMA mean and covariance can be updated in O(n_assets^2) per step instead of recomputing from scratch:

```python
def compute_series(self, returns: np.ndarray) -> np.ndarray:
    n_periods, n_assets = returns.shape
    result = np.full(n_periods, np.nan)

    # Bootstrap: compute initial EWMA from warmup period
    if n_periods <= self.min_warmup:
        return result

    historical = returns[:self.min_warmup]
    ewma_mean, cov_matrix = _ewma_covariance(historical, _EQUITY_EWMA_DECAY)

    for idx in range(self.min_warmup, n_periods):
        current = returns[idx]

        # Incremental EWMA update: O(n_assets^2) instead of O(idx * n_assets^2)
        ewma_mean = _EQUITY_EWMA_DECAY * ewma_mean + (1 - _EQUITY_EWMA_DECAY) * current
        diff = current - ewma_mean
        cov_matrix = _EQUITY_EWMA_DECAY * cov_matrix + (1 - _EQUITY_EWMA_DECAY) * np.outer(diff, diff)

        inv_cov = np.linalg.pinv(cov_matrix)
        diff_vec = np.atleast_1d(current - ewma_mean)
        result[idx] = float(np.sqrt(np.abs(diff_vec @ inv_cov @ diff_vec)))

    return result
```

**Complexity:** O(n × d^2) where d=8 assets, down from O(n^2 × d^2). For 20K bars: ~1.3M ops vs ~200M ops.

**Single-step `compute()` unchanged** — still recomputes from scratch (needed for live inference where only one bar is computed per cycle). The optimization only applies to batch `compute_series()`.

**Crypto `compute_series()` is already O(n)** — rolling windows are fixed-size, no expanding computation. No change needed.

### Fix #30 — Transaction Costs in Share Allocation

**Problem:** `PortfolioSimulator.rebalance()` computes `target_values = target_weights * total_value` then deducts costs from cash afterward. The agent gets allocated 100% of portfolio value to positions, then pays costs from cash — potentially driving cash negative. Over many rebalancing steps, cumulative cost leakage gives training an optimistic bias.

**Current code (lines 86-100):**
```python
current_values = self.shares * prices
deltas = target_values - current_values  # target_values based on full total_value
cost = float(np.sum(np.abs(deltas))) * self.transaction_cost_pct
self.shares = target_values / safe_prices
self.cash = total_value - float(np.sum(target_values)) - cost  # cash can go negative
```

**Fix:** Estimate cost before allocation, then allocate from reduced total:
```python
# Estimate cost from current vs target (pre-allocation)
current_values = self.shares * prices
est_deltas = target_weights * total_value - current_values
est_cost = float(np.sum(np.abs(est_deltas))) * self.transaction_cost_pct

# Allocate from cost-adjusted total
adjusted_total = total_value - est_cost
target_values = target_weights * adjusted_total
deltas = target_values - current_values

# Actual cost (slightly different due to adjusted allocation)
cost = float(np.sum(np.abs(deltas))) * self.transaction_cost_pct

safe_prices = np.where(prices > 0.0, prices, 1.0)
self.shares = target_values / safe_prices
self.cash = total_value - float(np.sum(target_values)) - cost
```

**Impact:** Cash stays non-negative. Training reward reflects realistic post-cost returns. The estimation is slightly off (circular dependency: cost depends on allocation which depends on cost), but the error is second-order (cost-on-cost-difference) and negligible for typical 0.10% transaction costs.

**Files:** `envs/portfolio.py` only. One method change, ~5 lines added.

### Fix #36 — FeatureHealthTracker (Block Trading on Degraded Features)

**Problem:** When macro, HMM, or turbulence queries fail during live inference, the feature pipeline silently returns default values (zeros for macro, [0.5, 0.5] for HMM, 0.0 for turbulence). The agent receives out-of-distribution observations it never saw during training:
- All-zero macro (VIX=0, spread=0, fed=0, CPI=0, unemployment=0) is not a natural market state
- Turbulence=0.0 disables the 90th percentile crash protection (signals "calm market")
- HMM [0.5, 0.5] is possible in training but combined with all-zero macro is not

The agent makes decisions on inputs it was never trained on, with no alerting.

**Design: `FeatureHealthTracker`**

Lightweight tracker integrated into feature pipeline and execution pipeline:

| Event | Action |
|-------|--------|
| Single failure | Log warning, return defaults (unchanged behavior) |
| 3 consecutive failures (macro or HMM) | **Block trading** for that environment + Discord warning via `Alerter` |
| 5 consecutive failures | Critical Discord alert |
| Turbulence failure | Warning only (0.0 = "calm market", suboptimal but not catastrophic) |
| Recovery (success after failures) | Reset counter, resume trading automatically |

**Key decisions:**
- **Do NOT add feature quality to observation** — changes obs shape, breaks all models, agent has no training signal for it
- **Fail-safe not fail-fast** — feature pipeline keeps returning defaults (training unaffected), execution pipeline decides whether to trade
- **Macro staleness check** — if last successful macro fetch is >7 days old at cycle start, treat as degraded
- **Use existing `Alerter`** — already wired into `ExecutionPipeline`, supports Discord webhooks, cooldowns, consecutive failure suppression

**Files:**

| File | Action |
|------|--------|
| `features/health.py` | **Create** — `FeatureHealthTracker`, `FeatureHealth`, `ObservationHealth` dataclasses (~80 lines) |
| `features/pipeline.py` | **Modify** — Accept optional `FeatureHealthTracker`, call `record_success()`/`record_failure()` in `_get_macro_array()`, `_get_hmm_probs()`, `_compute_turbulence_*()`. Fix bare `except` in HMM to log warning. |
| `execution/pipeline.py` | **Modify** — After observation assembly, check `health.should_block_trading(env_name)`. If blocked, log error, send Discord alert via `self._alerter`, return empty (skip cycle). |
| `tests/features/test_health.py` | **Create** — Unit tests for tracker logic (consecutive failures, blocking, recovery, staleness) |

---

## Test Coverage Gaps

| Priority | Gap | Location |
|----------|-----|----------|
| **CRITICAL** | No integration test for `DuckDB → FeaturePipeline → Env → Trainer` flow | Missing entirely |
| **HIGH** | `sentiment/news_fetcher.py` (168 LOC) has zero tests | `tests/` — no test file |
| **HIGH** | `data/base.py` BaseIngestor core methods untested | `_sync_to_duckdb()`, `_log_ingestion_run()`, `run()` |
| **HIGH** | No `PortfolioSimulator` edge cases (NaN prices, zero prices, bankruptcy) | `envs/portfolio.py` |
| **HIGH** | No execution pipeline integration test with real (not mocked) stages | `test_pipeline.py` — all 5 stages mocked |
| **MEDIUM** | `test_emergency_stop.py` has 90 mocks — tests verify mock interactions, not behavior | Reduce mock density |
| **MEDIUM** | `trainer.py` (537 LOC) has only 11 tests | Add NaN features, GPU OOM recovery tests |

---

## Security Review Summary

| Severity | Count | Key Areas |
|----------|-------|-----------|
| CRITICAL | 0 | None found |
| HIGH | 0 | None found |
| MEDIUM | 3 | Data ingestor credential validation (#38); API keys as plain strings (#39); Dashboard port (#37) |
| LOW | 9 | No timeouts on yfinance; memory service requirements not hash-pinned; path validation gaps |

**Positive findings:** No hardcoded secrets, proper TLS verification everywhere, timing-safe auth comparison in memory service, memory service port bound to localhost, all containers run as non-root users, `yaml.safe_load()` used throughout.

---

## Implementation Waves

### Wave 1 — Capital Safety (Fixes #1-2, #5-6, #10, #14)
Observation mismatch, circuit breaker, DuckDB race, VecNormalize, timezone. These are "model trades incorrectly or loses money" bugs.

### Wave 2 — Execution Correctness (Fixes #3-4, #9, #11-13, #15, #17)
Position sizing, cash calculation, cash-holding ability, shadow metrics, fills, random starts, adapter caching.

### Wave 3 — Infrastructure Robustness (Fixes #7-8, #16, #18-22, #37-39)
Auth headers, async DB, session leaks, exception types, retry, bounds sync, XML sanitization, security hardening.

### Wave 4 — Performance (Fixes #23-29)
Database queries, turbulence O(n^2), parallelization, connection reuse.

### Wave 5 — Test Coverage
Integration tests, edge cases, untested modules, mock reduction.

---

## Overall Totals

| Severity | Count |
|----------|-------|
| CRITICAL | 8 |
| HIGH | 14 |
| MEDIUM | 19 |
| LOW | ~24 (omitted — see individual agent reports) |
