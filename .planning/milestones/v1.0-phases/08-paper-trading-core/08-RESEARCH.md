# Phase 8: Paper Trading Core - Research

**Researched:** 2026-03-08
**Domain:** Execution middleware, broker integration, risk management, Docker production
**Confidence:** HIGH

## Summary

Phase 8 wires up the full execution pipeline from ensemble inference to broker order submission. The codebase already has all upstream components: `EnsembleBlender` for action blending, `ObservationAssembler` and `FeaturePipeline.get_observation()` for inference-time observation construction, `DatabaseManager` with SQLite tables for trades/positions/circuit_breaker_events/risk_decisions/portfolio_snapshots, and `Alerter` for Discord notifications. The `alpaca-py` SDK is already installed and used for data ingestion; it also provides `TradingClient` for paper trading with `paper=True`. Crypto fills are simulated locally using Binance.US REST API order book (already used for klines in Phase 3).

The execution module (`src/swingrl/execution/`) is currently empty (just `__init__.py`). Phase 8 builds the 5-stage middleware pipeline here: Signal Interpreter (ensemble action to trade signals), Position Sizer (modified quarter-Kelly with ATR stops), Order Validator (risk checks + cost gate), Exchange Adapter (Alpaca paper + Binance.US simulated), and Fill Processor (DB recording + position update). Two-tier risk management and circuit breakers with halt-state persistence round out the core trading infrastructure.

**Primary recommendation:** Build the 5-stage middleware as a linear pipeline of composable stage classes in `src/swingrl/execution/`, with a `Protocol`-based exchange adapter abstraction to cleanly separate Alpaca (real API) from Binance.US (simulated fills). Use the existing `DatabaseManager.sqlite()` context manager for all operational writes.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Crypto paper fill simulation (PAPER-02)**: Mid-price with slippage model from Binance.US order book. Instant fill. Virtual balance in SQLite ($47 start). Stop-loss/take-profit monitoring via 60s REST polling.
- **Execution retry & error handling**: Basic 3-attempt exponential backoff for Phase 8. Fail action: skip trade, log to risk_decisions, Discord critical alert. Independent per-environment pipelines with separate risk budgets and circuit breakers. Global aggregator is the only shared check.
- **Position reconciliation**: Startup + on-demand auto-reconcile equity vs Alpaca. Trust broker as source of truth. Auto-correct DB + Discord warning alert. CLI via `scripts/reconcile.py`.
- **Docker production image (PAPER-19)**: Multi-stage build excluding dev deps in production. Docker HEALTHCHECK probe. Single process entrypoint.
- **Manual cycle trigger**: `scripts/run_cycle.py --env equity|crypto [--dry-run]`. Full pipeline execution or dry-run logging.
- **Logging**: Same structlog calls in paper and live mode, config-driven level (DEBUG for paper default).

### Claude's Discretion
- 5-stage middleware internal architecture (class design, interfaces between stages)
- Exchange adapter abstraction pattern (base class vs protocol)
- Position sizing calculator implementation (Kelly formula + ATR stop calculation)
- Circuit breaker state machine internals (cooldown tracking, ramp-up progression)
- Turbulence crash protection integration with Phase 5's TurbulenceCalculator
- DB seeding script implementation details (scripts/seed_production.py)
- Order ID generation strategy for simulated crypto fills
- Reconciliation diff algorithm
- Docker healthcheck probe implementation
- Production docker-compose.yml structure

### Deferred Ideas (OUT OF SCOPE)
- APScheduler automation (equity 4:15 PM ET, crypto every 4H) -- Phase 9 (PAPER-12)
- Discord alerting for trade executions, daily summary -- Phase 9 (PAPER-13)
- Stuck agent detection -- Phase 9 (PAPER-14)
- Streamlit dashboard -- Phase 9 (PAPER-15)
- Healthchecks.io dead man's switch -- Phase 9 (PAPER-16)
- Wash sale tracker -- Phase 9 (PAPER-17)
- Comprehensive error handling with exponential backoff for all API calls -- Phase 10 (HARD-02)
- Structured JSON logging to bind-mounted logs/ volume -- Phase 10 (HARD-05)
- Shadow mode for new models -- Phase 10 (PROD-03/04)
- deploy_model.sh -- Phase 10 (PROD-02)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-01 | Alpaca paper trading connection for equity environment | alpaca-py `TradingClient(key, secret, paper=True)` already installed; bracket orders via `OrderClass.BRACKET` |
| PAPER-02 | Binance.US simulated fills for crypto (real-time prices, local recording) | Use existing Binance.US REST `GET /api/v3/depth` for order book mid-price; simulate fills locally in SQLite |
| PAPER-03 | Two-tier risk management veto layer | Tier 1: per-env position size, exposure, drawdown, daily loss checks. Tier 2: global portfolio aggregator (-15% DD, -3% daily combined) |
| PAPER-04 | Circuit breakers: equity -10% DD / -2% daily, crypto -12% DD / -3% daily, global -15% DD / -3% combined | Config values already in `EquityConfig.max_drawdown_pct` and `CryptoConfig.max_drawdown_pct` |
| PAPER-05 | Circuit breaker cooldown: 5 biz days equity, 3 cal days crypto, 25/50/75/100% ramp-up | State machine with graduated capacity restoration |
| PAPER-06 | Halt-state persistence in circuit_breaker_events table | Table already exists in SQLite schema with `environment`, `triggered_at`, `resumed_at` columns |
| PAPER-07 | Position sizing: modified Kelly (quarter-Kelly Phase 1), 2% max risk per trade, ATR(2x) stops | Kelly formula: `K = W - (1-W)/R`, then quarter: `K/4`, capped at 2% of portfolio per trade |
| PAPER-08 | Binance.US $10 minimum order floor | `config.crypto.min_order_usd` already defaults to 10.0 in schema; apply `max(kelly_sized, 10.0)` |
| PAPER-09 | 5-stage execution middleware | Signal Interpreter -> Position Sizer -> Order Validator -> Exchange Adapter -> Fill Processor |
| PAPER-10 | Bracket orders: OTO for Alpaca, two-step OCO for Binance.US | Alpaca: `OrderClass.BRACKET` with `TakeProfitRequest` + `StopLossRequest`. Crypto: simulated locally |
| PAPER-11 | Cost gate: reject if round-trip costs > 2.0% of order value | Pre-submission check in Order Validator stage; equity ~0.06%, crypto ~0.22% RT costs |
| PAPER-18 | Production database seeding from M1 Mac historical archive | 7-step procedure: SCP databases + verify integrity |
| PAPER-19 | Full Docker production deployment with TRADING_MODE=paper | Multi-stage Dockerfile, production docker-compose.yml, HEALTHCHECK |
| PAPER-20 | Turbulence index crash protection: liquidate and halt if turbulence > 90th percentile | Integrate Phase 5 `TurbulenceCalculator`; compare against historical 90th pctile |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| alpaca-py | >=0.20 | Equity paper trading (TradingClient) | Already installed; official SDK with paper=True flag |
| requests | (installed) | Binance.US REST API for order book depth | Already used by Phase 3 BinanceIngestor; no new dependency |
| sqlite3 | stdlib | Trading ops: trades, positions, circuit breakers, risk decisions | Already managed via DatabaseManager.sqlite() |
| structlog | >=24.0 | Structured logging throughout execution pipeline | Project standard from Phase 2 |
| httpx | >=0.27 | Discord webhook alerts for critical failures | Already used by Phase 4 Alerter |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| stable-baselines3 | >=2.0 | Model loading for inference (.predict()) | Loading trained models from models/active/{env}/{algo}/ |
| numpy | >=1.26 | Action array manipulation, portfolio math | Throughout pipeline stages |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Protocol-based adapter | ABC base class | Protocol is lighter, no inheritance coupling; preferred for 2 concrete adapters |
| requests for Binance.US | python-binance library | requests already installed, simpler for just order book fetch; no new dep |

**Installation:**
```bash
# No new dependencies required -- all already installed
```

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/execution/
    __init__.py
    pipeline.py           # ExecutionPipeline orchestrator
    signal_interpreter.py # Stage 1: ensemble action -> TradeSignal
    position_sizer.py     # Stage 2: Kelly sizing + ATR stops
    order_validator.py    # Stage 3: risk checks + cost gate
    adapters/
        __init__.py
        base.py           # ExchangeAdapter Protocol
        alpaca_adapter.py # Stage 4a: Alpaca paper trading
        binance_sim.py    # Stage 4b: Binance.US simulated fills
    fill_processor.py     # Stage 5: DB recording + position update
    risk/
        __init__.py
        risk_manager.py   # Two-tier risk veto layer
        circuit_breaker.py # CB state machine with persistence
        position_tracker.py # Portfolio state tracking
    reconciliation.py     # Position reconciliation logic
scripts/
    run_cycle.py          # Manual cycle trigger CLI
    reconcile.py          # Position reconciliation CLI
    seed_production.py    # DB seeding for homelab deployment
```

### Pattern 1: Linear Pipeline with Named Stages
**What:** Each middleware stage is a class with a well-defined input/output contract. The pipeline orchestrator calls stages sequentially; any stage can halt the pipeline (e.g., risk veto).
**When to use:** All order flow from ensemble inference to fill recording.
**Example:**
```python
# src/swingrl/execution/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass
class TradeSignal:
    """Output of Stage 1: what the ensemble wants to do."""
    environment: str  # "equity" | "crypto"
    symbol: str
    action: str  # "buy" | "sell" | "hold"
    raw_weight: float  # target portfolio weight from ensemble

@dataclass
class SizedOrder:
    """Output of Stage 2: sized order with stop/TP levels."""
    symbol: str
    side: str
    quantity: float
    dollar_amount: float
    stop_loss_price: float
    take_profit_price: float
    environment: str

@dataclass
class ValidatedOrder:
    """Output of Stage 3: risk-approved order."""
    sized_order: SizedOrder
    risk_checks_passed: list[str]

@dataclass
class FillResult:
    """Output of Stage 4: exchange fill information."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    environment: str
    broker: str

class ExecutionPipeline:
    """Orchestrates the 5-stage execution middleware."""

    def execute_cycle(self, env_name: str, observation: np.ndarray) -> list[FillResult]:
        # 1. Get ensemble prediction
        # 2. Signal Interpreter: actions -> TradeSignals
        # 3. Position Sizer: signals -> SizedOrders
        # 4. Order Validator: risk checks -> ValidatedOrders (or veto)
        # 5. Exchange Adapter: submit -> FillResults
        # 6. Fill Processor: record to DB
        ...
```

### Pattern 2: Protocol-Based Exchange Adapter
**What:** Define an `ExchangeAdapter` Protocol so Alpaca and Binance.US simulation share the same interface. The pipeline doesn't know which broker it's talking to.
**When to use:** Stage 4 -- separating broker-specific logic from pipeline logic.
**Example:**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ExchangeAdapter(Protocol):
    """Interface for exchange interaction."""

    def submit_order(self, order: ValidatedOrder) -> FillResult:
        """Submit order and return fill result."""
        ...

    def get_positions(self) -> list[dict[str, object]]:
        """Get current positions from exchange."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        ...
```

### Pattern 3: Circuit Breaker State Machine
**What:** Three states: ACTIVE (trading allowed), HALTED (no trading), RAMPING (graduated capacity). Persisted in SQLite `circuit_breaker_events` table. Survives container restarts.
**When to use:** Pre-trade check in Order Validator stage.
**Example:**
```python
from enum import Enum

class CBState(Enum):
    ACTIVE = "active"
    HALTED = "halted"
    RAMPING = "ramping"

class CircuitBreaker:
    """Per-environment circuit breaker with persistent state."""

    RAMP_SCHEDULE = [0.25, 0.50, 0.75, 1.00]

    def __init__(self, environment: str, db: DatabaseManager, config: SwingRLConfig):
        self._env = environment
        self._db = db
        # Load cooldown from config
        if environment == "equity":
            self._cooldown_days = 5  # business days
        else:
            self._cooldown_days = 3  # calendar days

    def check_and_update(self, portfolio_value: float, high_water_mark: float,
                          daily_pnl: float) -> CBState:
        """Check drawdown/daily loss against thresholds, update state."""
        ...

    def get_state(self) -> CBState:
        """Load current state from SQLite (survives restarts)."""
        ...

    def get_capacity_fraction(self) -> float:
        """During RAMPING, returns 0.25/0.50/0.75/1.00 based on elapsed time."""
        ...
```

### Anti-Patterns to Avoid
- **Direct broker API calls outside adapters:** All broker interaction MUST go through Exchange Adapter classes. Never call `trading_client.submit_order()` from risk management or signal interpretation code.
- **Mutable global state for portfolio tracking:** Use DB as source of truth for positions/cash, not in-memory-only objects. Container restarts must not lose position state.
- **Mixing paper and live logic:** The pipeline should be mode-agnostic. The only difference is config values and which exchange adapter is instantiated. Never `if trading_mode == "paper"` in business logic.
- **Synchronous blocking on price polls:** The 60s stop-loss/TP polling loop for crypto should be a daemon thread, not blocking the main execution flow.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Alpaca order submission | Raw REST API calls | `alpaca-py TradingClient` | Handles auth, rate limits, order types, paper/live switching |
| Alpaca bracket orders | Manual OTO logic | `OrderClass.BRACKET` with `TakeProfitRequest`/`StopLossRequest` | SDK handles the 3-leg order chain |
| UUID generation for trade IDs | Custom ID schemes | `uuid.uuid4()` | Standard, collision-free, already used in alerter |
| SB3 model loading | Custom deserialization | `PPO.load()`, `A2C.load()`, `SAC.load()` with `VecNormalize.load()` | Handles architecture reconstruction, device mapping |
| Portfolio state persistence | Custom file-based storage | SQLite `positions` + `portfolio_snapshots` tables | Already defined in DatabaseManager schema, ACID transactions |

**Key insight:** The execution pipeline is mostly orchestration glue between existing components. The novel code is in risk management logic (two-tier veto, circuit breakers, cost gate) and the Binance.US fill simulation. Everything else should delegate to established libraries and existing codebase modules.

## Common Pitfalls

### Pitfall 1: VecNormalize Train/Serve Skew
**What goes wrong:** Loading a model for inference without setting VecNormalize `training=False`, causing observation statistics to drift.
**Why it happens:** SB3's VecNormalize defaults to `training=True` which updates running statistics on each observation.
**How to avoid:** After `VecNormalize.load()`, explicitly set `vec_normalize.training = False` and `vec_normalize.norm_reward = False`.
**Warning signs:** Model predictions become erratic after many inference calls; observation means drift from training values.

### Pitfall 2: Alpaca Paper vs Live Key Confusion
**What goes wrong:** Using live API keys with `paper=True` or vice versa. Alpaca paper keys are separate from live keys.
**Why it happens:** Both are environment variables with similar names.
**How to avoid:** Use distinct env var names: `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET_KEY` for paper mode. Validate at startup that `config.trading_mode` matches the key type being used.
**Warning signs:** 401 Unauthorized errors, orders appearing in wrong account.

### Pitfall 3: SQLite Locking with Multiple Writers
**What goes wrong:** Concurrent writes from main thread (trade recording) and daemon thread (stop-loss polling) cause `database is locked` errors.
**Why it happens:** SQLite WAL mode allows concurrent reads but writes are serialized.
**How to avoid:** Use a threading.Lock() for SQLite writes (already pattern in DatabaseManager), or use separate connections with WAL mode (current pattern). Keep write transactions short.
**Warning signs:** `sqlite3.OperationalError: database is locked` in logs.

### Pitfall 4: Crypto Min Order Floor Creating Outsized Positions
**What goes wrong:** Kelly sizes a $3 order, floor adjusts to $10, but $10 is 21% of $47 capital -- far exceeding the intended risk.
**Why it happens:** The $10 floor can dominate when capital is very small.
**How to avoid:** After floor adjustment, re-check against `max_position_size` (50%) and per-trade risk limit (2%). If floor-adjusted amount exceeds risk limits, skip the trade entirely.
**Warning signs:** Crypto positions consuming 20%+ of capital on what should be a small signal.

### Pitfall 5: Circuit Breaker Cooldown Timezone Issues
**What goes wrong:** Equity cooldown is "5 business days" but code counts calendar days. Crypto is "3 calendar days" but code uses business days.
**Why it happens:** Mixing equity (business day) and crypto (calendar day) conventions.
**How to avoid:** Use `exchange_calendars` (already installed) for equity business day counting. Use simple `timedelta(days=3)` for crypto calendar days. Make the distinction explicit in the CircuitBreaker class.
**Warning signs:** Equity resuming on weekends, crypto halted longer than expected.

### Pitfall 6: Binance.US Order Book Stale Data
**What goes wrong:** Fetching order book but the spread is abnormally wide (low liquidity hour), resulting in simulated fills at unrealistic mid-prices.
**Why it happens:** BTC/ETH can have thin order books on Binance.US during off-hours.
**How to avoid:** Add a spread width sanity check. If bid-ask spread exceeds a threshold (e.g., 0.5%), log a warning and use the last known good mid-price or skip the fill.
**Warning signs:** Simulated fill prices significantly different from reference close price.

## Code Examples

### Alpaca Paper Trading Client Setup
```python
# Source: Alpaca official docs
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# paper=True routes to paper-api.alpaca.markets
client = TradingClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

# Bracket order with stop-loss and take-profit
bracket = MarketOrderRequest(
    symbol="SPY",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    order_class=OrderClass.BRACKET,
    stop_loss=StopLossRequest(stop_price=295.0),
    take_profit=TakeProfitRequest(limit_price=310.0),
)
order = client.submit_order(order_data=bracket)

# Get account and positions
account = client.get_account()
positions = client.get_all_positions()
```

### Binance.US Order Book Fetch (Simulated Fill)
```python
# Source: Binance.US REST API docs
import requests

BINANCE_US_BASE = "https://api.binance.us"

def get_mid_price(symbol: str, depth: int = 5) -> tuple[float, float, float]:
    """Fetch order book and compute mid-price.

    Returns:
        Tuple of (mid_price, best_bid, best_ask).
    """
    resp = requests.get(
        f"{BINANCE_US_BASE}/api/v3/depth",
        params={"symbol": symbol, "limit": depth},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    best_bid = float(data["bids"][0][0])
    best_ask = float(data["asks"][0][0])
    mid_price = (best_bid + best_ask) / 2.0

    return mid_price, best_bid, best_ask


def simulate_fill(
    symbol: str,
    side: str,
    quantity: float,
    slippage_pct: float = 0.0003,
) -> float:
    """Simulate a fill at mid-price with configurable slippage."""
    mid, bid, ask = get_mid_price(symbol)

    if side == "buy":
        fill_price = mid * (1 + slippage_pct)
    else:
        fill_price = mid * (1 - slippage_pct)

    return fill_price
```

### Quarter-Kelly Position Sizing
```python
def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    portfolio_value: float,
    kelly_fraction: float = 0.25,  # quarter-Kelly
    max_risk_pct: float = 0.02,    # 2% max risk per trade
) -> float:
    """Compute position size using modified Kelly criterion.

    Args:
        win_rate: Historical win rate (0.0-1.0).
        avg_win: Average winning trade return.
        avg_loss: Average losing trade return (positive number).
        portfolio_value: Current portfolio value.
        kelly_fraction: Kelly multiplier (0.25 = quarter-Kelly).
        max_risk_pct: Maximum portfolio risk per trade.

    Returns:
        Dollar amount to allocate to the trade.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    reward_risk = avg_win / avg_loss
    kelly_pct = win_rate - (1 - win_rate) / reward_risk

    if kelly_pct <= 0:
        return 0.0

    sized_pct = kelly_pct * kelly_fraction
    risk_capped_pct = min(sized_pct, max_risk_pct)

    return portfolio_value * risk_capped_pct
```

### Model Loading for Inference
```python
# Based on SB3 patterns + Phase 7 trainer conventions
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import VecNormalize

ALGO_MAP = {"ppo": PPO, "a2c": A2C, "sac": SAC}

def load_agent(env_name: str, algo_name: str, models_dir: Path) -> tuple:
    """Load a trained agent and its VecNormalize stats.

    Returns:
        Tuple of (model, vec_normalize).
    """
    algo_cls = ALGO_MAP[algo_name]
    model_path = models_dir / "active" / env_name / algo_name / "model.zip"
    vec_path = models_dir / "active" / env_name / algo_name / "vec_normalize.pkl"

    model = algo_cls.load(str(model_path))

    # CRITICAL: set training=False to freeze normalization stats
    vec_normalize = VecNormalize.load(str(vec_path), venv=DummyVecEnv([lambda: None]))
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    return model, vec_normalize
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| alpaca-trade-api-python | alpaca-py | 2023 | New SDK; different import paths, class names |
| Manual paper API URL | `TradingClient(paper=True)` | alpaca-py 0.x | Single flag switches environments |
| Custom order book parsing | requests + Binance.US REST | Stable | No SDK needed; simple GET endpoint |
| Full Kelly sizing | Quarter Kelly (Phase 1) | Project decision | Reduces volatility at cost of slower growth |

**Deprecated/outdated:**
- `alpaca-trade-api-python` (old SDK): replaced by `alpaca-py`. Do not use old import paths.
- Binance.com endpoints: project uses Binance.US exclusively (`api.binance.us`).

## Open Questions

1. **Ensemble weight freshness during inference**
   - What we know: Weights are computed during training/backtesting and stored in DuckDB model_metadata.
   - What's unclear: Should weights be recomputed at inference time based on recent performance, or use stored weights? Phase 7 stores ensemble_weight in model_metadata.
   - Recommendation: Use stored weights from model_metadata for Phase 8. Adaptive weight recomputation is a Phase 10 enhancement.

2. **Alpaca fractional shares for small positions**
   - What we know: Alpaca supports fractional shares for equity ETFs. With $400 capital and 25% max position, a single position could be $100 -- fractional shares are needed for high-priced ETFs like SPY (~$500).
   - What's unclear: Whether `qty` in `MarketOrderRequest` accepts floats for fractional shares on all ETFs.
   - Recommendation: Use `notional` parameter (dollar amount) instead of `qty` for equity orders. Alpaca supports `notional` for market orders on eligible securities. Verify with paper trading.

3. **Portfolio state for observation assembly during live inference**
   - What we know: `ObservationAssembler._default_portfolio_state()` returns 100% cash. During training, RL envs provide portfolio state via `PortfolioSimulator`. During inference, real portfolio state must be constructed from DB positions.
   - What's unclear: How to map DB positions to the exact (27,) equity / (9,) crypto portfolio state array format.
   - Recommendation: Build a `PortfolioStateBuilder` that reads `positions` and `portfolio_snapshots` tables and constructs the array matching assembler format: [cash_ratio, exposure, daily_return, per_asset_weight, per_asset_unrealized_pnl_pct, per_asset_days_since_trade].

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/execution/ -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-01 | Alpaca paper order submission returns fill | unit (mock SDK) | `uv run pytest tests/execution/test_alpaca_adapter.py -x` | Wave 0 |
| PAPER-02 | Binance.US simulated fill at mid-price with slippage | unit (mock requests) | `uv run pytest tests/execution/test_binance_sim.py -x` | Wave 0 |
| PAPER-03 | Tier 1 per-env risk checks veto out-of-policy trades | unit | `uv run pytest tests/execution/test_risk_manager.py -x` | Wave 0 |
| PAPER-04 | Circuit breaker triggers at threshold | unit | `uv run pytest tests/execution/test_circuit_breaker.py -x` | Wave 0 |
| PAPER-05 | Cooldown timing (5 biz/3 cal) and ramp-up progression | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_cooldown -x` | Wave 0 |
| PAPER-06 | Halt state survives DB close/reopen | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_persistence -x` | Wave 0 |
| PAPER-07 | Quarter-Kelly sizing with ATR stop produces correct values | unit | `uv run pytest tests/execution/test_position_sizer.py -x` | Wave 0 |
| PAPER-08 | $10 floor applied to crypto orders | unit | `uv run pytest tests/execution/test_position_sizer.py::test_crypto_floor -x` | Wave 0 |
| PAPER-09 | Full pipeline: signal -> fill (mocked adapters) | integration | `uv run pytest tests/execution/test_pipeline.py -x` | Wave 0 |
| PAPER-10 | Bracket order params set correctly for Alpaca | unit | `uv run pytest tests/execution/test_alpaca_adapter.py::test_bracket -x` | Wave 0 |
| PAPER-11 | Cost gate rejects order with >2% RT costs | unit | `uv run pytest tests/execution/test_order_validator.py::test_cost_gate -x` | Wave 0 |
| PAPER-18 | DB seeding script runs without error | smoke | `uv run pytest tests/execution/test_seed_production.py -x` | Wave 0 |
| PAPER-19 | Docker multi-stage build succeeds | smoke | `docker build --target production -t swingrl:test .` | manual |
| PAPER-20 | Turbulence > 90pct triggers liquidation | unit | `uv run pytest tests/execution/test_risk_manager.py::test_turbulence -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/execution/ -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/execution/__init__.py` -- package init
- [ ] `tests/execution/test_signal_interpreter.py` -- covers signal interpretation
- [ ] `tests/execution/test_position_sizer.py` -- covers PAPER-07, PAPER-08
- [ ] `tests/execution/test_order_validator.py` -- covers PAPER-03, PAPER-11
- [ ] `tests/execution/test_alpaca_adapter.py` -- covers PAPER-01, PAPER-10
- [ ] `tests/execution/test_binance_sim.py` -- covers PAPER-02
- [ ] `tests/execution/test_fill_processor.py` -- covers DB recording
- [ ] `tests/execution/test_circuit_breaker.py` -- covers PAPER-04, PAPER-05, PAPER-06
- [ ] `tests/execution/test_risk_manager.py` -- covers PAPER-03, PAPER-20
- [ ] `tests/execution/test_pipeline.py` -- covers PAPER-09
- [ ] `tests/execution/conftest.py` -- shared fixtures (mock broker, mock DB)

## Sources

### Primary (HIGH confidence)
- [Alpaca Paper Trading docs](https://docs.alpaca.markets/docs/paper-trading) -- TradingClient paper=True setup
- [Alpaca Placing Orders docs](https://docs.alpaca.markets/docs/orders-at-alpaca) -- bracket orders, OTO/OCO, time-in-force
- [Alpaca Working with Orders](https://docs.alpaca.markets/docs/working-with-orders) -- Python code examples for submit_order
- [Alpaca Community Forum](https://forum.alpaca.markets/t/bracket-order-code-example-with-alpaca-py-library/12110) -- Bracket order code with alpaca-py
- Existing codebase: `src/swingrl/data/alpaca.py`, `src/swingrl/data/binance.py` (REST API patterns)
- Existing codebase: `src/swingrl/data/db.py` (SQLite schema for circuit_breaker_events, trades, positions)
- Existing codebase: `src/swingrl/config/schema.py` (risk parameters already in config)
- Existing codebase: `src/swingrl/training/ensemble.py` (EnsembleBlender for inference)

### Secondary (MEDIUM confidence)
- [Binance.US API docs](https://docs.binance.us/) -- REST API order book endpoint
- [Binance.US API GitHub](https://github.com/binance-us/binance-us-api-docs/blob/master/rest-api.md) -- GET /api/v3/depth parameters
- [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion) -- Formula verification

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and used in prior phases
- Architecture: HIGH -- 5-stage pipeline is well-defined in CONTEXT.md; patterns are standard middleware design
- Pitfalls: HIGH -- based on known issues with SQLite concurrency, VecNormalize skew, and Alpaca paper/live key separation
- Broker integration: HIGH -- Alpaca alpaca-py SDK docs verified; Binance.US REST API already used for klines
- Risk management: HIGH -- all thresholds and rules defined in project docs and config schema
- Docker: MEDIUM -- multi-stage build is standard Docker pattern but specific HEALTHCHECK design is discretionary

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable domain -- broker SDKs and patterns well-established)
