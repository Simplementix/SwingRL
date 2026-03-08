# Phase 6: RL Environments - Research

**Researched:** 2026-03-07
**Domain:** Gymnasium-compatible trading environments for Stable Baselines3
**Confidence:** HIGH

## Summary

Phase 6 builds two Gymnasium-compatible trading environments (StockTradingEnv-v0 and CryptoTradingEnv-v0) that consume observations from the Phase 5 feature pipeline and simulate portfolio allocation with transaction costs. The environments use continuous action spaces (portfolio target weights), rolling 20-day Sharpe ratio rewards, and soft risk-violation penalties. Phase 7 wraps these with VecNormalize for training.

The project already has Gymnasium 1.2.3 and Stable Baselines3 2.7.1 installed. The Phase 5 ObservationAssembler produces (156,) equity and (45,) crypto vectors with deterministic ordering. The environments must honor the Gymnasium step/reset contract (5-tuple returns, terminated/truncated split, seed support) and pass SB3's `check_env()` validation.

**Primary recommendation:** Build a shared `BaseTradingEnv(gymnasium.Env)` base class with portfolio simulation, reward computation, and trade logging, then specialize `StockTradingEnv` and `CryptoTradingEnv` with per-environment config (symbols, observation dims, episode lengths, transaction costs). Pre-load feature data as NumPy arrays in `reset()` rather than querying DuckDB per step.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Continuous portfolio weights via softmax conversion -- raw agent output transformed into positive percentages summing to 100%
- Cash as remainder -- agent outputs weights for assets only (8 ETFs equity, 2 cryptos). Unallocated portion becomes cash automatically
- Action space shape: Box(8,) for equity, Box(2,) for crypto
- Signal deadzone: actions within +/-0.02 of zero mapped to "hold" (TRAIN-09) -- no trade executed
- Rolling 20-day Sharpe ratio with expanding-window warmup for first 19 bars (TRAIN-07)
- Transaction costs deducted from portfolio value before Sharpe calculation (equity ~0.06%, crypto ~0.22% round-trip)
- Soft risk-violation penalties subtracted from reward when agent exceeds position limits or drawdown thresholds
- No additional turnover penalty -- transaction cost deductions naturally discourage excessive trading
- Soft penalties (not hard enforcement) -- agent CAN exceed position limits (25% equity, 50% crypto) and drawdown thresholds, but receives reward penalty
- Full-length episodes always -- no early termination on drawdown or bankruptcy
- Equity: 252-day segments; Crypto: 540 4H bars (~3 months) with random start within training window
- $100,000 normalized training capital for both environments
- VecNormalize applied in Phase 7, not Phase 6
- Adaptive validation windows (shrink 50% during high turbulence) -- environment exposes turbulence, Phase 7 uses it

### Claude's Discretion
- Gymnasium registration pattern (StockTradingEnv-v0, CryptoTradingEnv-v0)
- Data loading strategy (pre-load features into memory vs DuckDB per step)
- Portfolio simulation internals (dollar-based tracking with percentage weights)
- Environment info dict contents (portfolio value, trade log, metrics for Phase 7)
- Render mode support for debugging
- Soft penalty magnitude and scaling for risk violations
- Random seed / reproducibility patterns
- Config schema additions for environment parameters (initial_amount, episode_length, etc.)

### Deferred Ideas (OUT OF SCOPE)
- pandas_ta re-evaluation (deferred from Phase 5)
- Walk-forward backtesting framework -- Phase 7 (VAL-01, VAL-02)
- Sharpe-weighted softmax ensemble blending -- Phase 7 (TRAIN-06)
- ConvergenceCallback early stopping -- Phase 7 (VAL-05)
- Production risk veto layer (hard enforcement) -- Phase 8 (PAPER-03)
- Circuit breaker halt-state persistence -- Phase 8 (PAPER-06)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | StockTradingEnv -- Gymnasium-compatible, daily bars, 8 ETFs, 156-dim observation, continuous action space | Base class pattern + equity specialization; ObservationAssembler integration; Box(8,) action space with softmax conversion |
| TRAIN-02 | CryptoTradingEnv -- Gymnasium-compatible, 4H bars, BTC/ETH, 45-dim observation, continuous action space | Same base class + crypto specialization; Box(2,) action space; random start within training window |
| TRAIN-07 | Rolling 20-day Sharpe ratio reward with expanding-window warmup for first 19 bars | Reward function implementation using deque of daily returns; expanding-window Sharpe for warmup period |
| TRAIN-08 | VecNormalize statistics handling -- frozen during inference | Phase 6 scope: environments expose raw observations; Phase 7 wraps with VecNormalize. Environment must NOT pre-normalize observations |
| TRAIN-09 | Signal deadzone: actions within +/-0.02 of zero mapped to "hold" | Applied in step() after softmax conversion -- per-asset weight changes below 0.02 threshold treated as no-op |
| TRAIN-10 | Adaptive validation windows: shrink by 50% during high turbulence | Environment exposes current turbulence value in info dict; Phase 7 training loop reads it to adjust validation windows |
| TRAIN-11 | Episode structure: equity 252-day segments, crypto 540 4H bars with random start | Configurable episode lengths; crypto reset() selects random start index; equity uses sequential segments |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gymnasium | 1.2.3 | Environment interface (Env base class, spaces, registration) | Already installed; SB3 2.7.1 requires it |
| stable-baselines3 | 2.7.1 | check_env() validation, DummyVecEnv wrapping in Phase 7 | Already installed; provides env_checker |
| numpy | (installed) | Observation/action arrays, portfolio math, Sharpe calculation | Core numerical dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog | (installed) | Environment logging (trades, resets, episodes) | All environment events |
| pydantic | (installed) | Config schema extensions for env parameters | New EnvironmentConfig model |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pre-loaded NumPy arrays | DuckDB queries per step | DuckDB per step adds ~1ms/step latency; 252 steps/episode = 0.25s overhead; pre-loading eliminates this entirely |
| Custom base class | FinRL StockTradingEnv | FinRL's env uses old gym API, hardcodes state structure, mixes concerns; custom is cleaner for our assembler integration |

**Installation:**
No new packages needed. All dependencies already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/envs/
    __init__.py          # Gymnasium registration + public exports
    base.py              # BaseTradingEnv(gymnasium.Env) -- shared portfolio simulation
    equity.py            # StockTradingEnv(BaseTradingEnv) -- 8 ETFs, daily bars
    crypto.py            # CryptoTradingEnv(BaseTradingEnv) -- BTC/ETH, 4H bars
    rewards.py           # RollingSharpReward -- 20-day rolling Sharpe with warmup
    portfolio.py         # PortfolioSimulator -- dollar-based tracking, trade execution
```

### Pattern 1: BaseTradingEnv with Shared Portfolio Logic

**What:** Abstract base environment handling portfolio simulation, action processing (softmax + deadzone), reward calculation, and info dict construction. Subclasses provide observation dims, action dims, episode length, symbols, and transaction cost.

**When to use:** Both equity and crypto environments share 80%+ of logic.

**Example:**
```python
# Source: Gymnasium 1.2.3 API + SB3 2.7.1 custom env docs
class BaseTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        features: np.ndarray,       # (n_steps, n_obs_dims) pre-loaded
        prices: np.ndarray,          # (n_steps, n_assets) close prices
        config: SwingRLConfig,
        environment: Literal["equity", "crypto"],
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        # Define spaces
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_assets,), dtype=np.float32,
        )
```

### Pattern 2: Pre-loaded Feature Arrays

**What:** Load all feature data into NumPy arrays during `__init__` or a factory method, indexed by step number. No DuckDB queries during step/reset.

**When to use:** Always for training. DuckDB per-step queries add unnecessary latency.

**Example:**
```python
# Factory method loads data once, creates environment
@staticmethod
def from_features(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: SwingRLConfig,
) -> StockTradingEnv:
    features = features_df.values.astype(np.float32)
    prices = prices_df.values.astype(np.float32)
    return StockTradingEnv(features=features, prices=prices, config=config)
```

### Pattern 3: Softmax Action Conversion + Deadzone

**What:** Raw agent output from Box(-1, 1) is converted via softmax to positive weights summing to <= 1.0 (remainder is cash). Then per-asset weight changes below 0.02 are treated as holds.

**Example:**
```python
def _process_actions(self, raw_actions: np.ndarray) -> np.ndarray:
    """Convert raw agent output to target portfolio weights."""
    # Softmax to get positive weights
    exp_actions = np.exp(raw_actions - np.max(raw_actions))
    weights = exp_actions / exp_actions.sum()

    # Apply deadzone: changes < 0.02 from current weight -> keep current
    current_weights = self._portfolio.asset_weights
    delta = np.abs(weights - current_weights)
    mask = delta < 0.02
    weights[mask] = current_weights[mask]

    return weights
```

### Pattern 4: Gymnasium Registration via __init__.py

**What:** Register environments in the envs package __init__.py so `gym.make('StockTradingEnv-v0')` works.

**Example:**
```python
# src/swingrl/envs/__init__.py
from gymnasium.envs.registration import register

register(
    id="StockTradingEnv-v0",
    entry_point="swingrl.envs.equity:StockTradingEnv",
)

register(
    id="CryptoTradingEnv-v0",
    entry_point="swingrl.envs.crypto:CryptoTradingEnv",
)
```

### Pattern 5: Rolling Sharpe Reward with Warmup

**What:** Maintain a deque of portfolio returns. For first 19 steps, use expanding-window Sharpe (mean/std of available returns). From step 20+, use rolling 20-day window.

**Example:**
```python
class RollingSharpeReward:
    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._returns: deque[float] = deque(maxlen=window)

    def compute(self, daily_return: float) -> float:
        self._returns.append(daily_return)
        n = len(self._returns)
        if n < 2:
            return 0.0
        returns_arr = np.array(self._returns)
        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std(ddof=1)
        if std_ret < 1e-8:
            return 0.0
        # Annualization not needed -- relative comparison within training
        return float(mean_ret / std_ret)
```

### Pattern 6: Portfolio Simulator (Dollar-based)

**What:** Track portfolio in dollars internally. Convert target weights to trade orders, deduct transaction costs, update positions. Expose current weights for observation assembly.

**Example:**
```python
class PortfolioSimulator:
    def __init__(self, initial_cash: float, n_assets: int, transaction_cost_pct: float):
        self.cash = initial_cash
        self.shares = np.zeros(n_assets)  # fractional shares allowed
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_log: list[dict] = []

    def rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> float:
        """Execute trades to reach target weights. Returns total transaction cost."""
        total_value = self.portfolio_value(prices)
        target_values = target_weights * total_value
        current_values = self.shares * prices
        delta_values = target_values - current_values

        cost = np.sum(np.abs(delta_values)) * self.transaction_cost_pct
        # Execute trades
        self.shares = target_values / np.where(prices > 0, prices, 1.0)
        self.cash = total_value - np.sum(target_values) - cost
        return cost
```

### Anti-Patterns to Avoid
- **Querying DuckDB in step():** Adds per-step latency; defeats vectorized training. Pre-load all data.
- **Hardcoding observation dimensions:** Use constants from assembler.py (EQUITY_OBS_DIM=156, CRYPTO_OBS_DIM=45).
- **Hard enforcement of position limits in env:** User decision says soft penalties only. Hard limits go in Phase 8 production veto layer.
- **Normalizing observations in env:** VecNormalize (Phase 7) handles this. Environment returns raw values.
- **Early episode termination:** User decision says full-length episodes always, no bankruptcy termination.
- **Using FinRL's env directly:** It uses old gym API, mixes state construction with the environment, and doesn't support the assembler pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Observation assembly | Custom concatenation in env | `ObservationAssembler.assemble_equity/crypto()` | Already validated, deterministic ordering, NaN checks |
| Environment validation | Manual testing of spaces/returns | `stable_baselines3.common.env_checker.check_env()` | Catches space mismatches, dtype issues, return format bugs |
| Random number generation | `np.random.seed()` or `random.seed()` | `self.np_random` from Gymnasium (set via `super().reset(seed=seed)`) | Proper reproducibility, per-env RNG isolation |
| Turbulence calculation | New calculator | `TurbulenceCalculator` from Phase 5 | Already handles expanding/rolling modes, pinv for stability |

**Key insight:** Phase 5 did the heavy lifting for observation assembly and feature computation. Phase 6 environments are thin wrappers around portfolio simulation logic, consuming pre-computed features.

## Common Pitfalls

### Pitfall 1: Gymnasium 5-tuple vs old 4-tuple step() return
**What goes wrong:** Returning `(obs, reward, done, info)` instead of `(obs, reward, terminated, truncated, info)`.
**Why it happens:** Old gym (pre-v26) used 4-tuple. Gymnasium uses 5-tuple. Many tutorials show old API.
**How to avoid:** Always return 5 values. `terminated` = episode naturally ended (ran out of data). `truncated` = False always (no early termination per user decision).
**Warning signs:** SB3 `check_env()` will catch this.

### Pitfall 2: Observation dtype mismatch
**What goes wrong:** Returning float64 observations when observation_space specifies float32.
**Why it happens:** NumPy defaults to float64. ObservationAssembler returns float64.
**How to avoid:** Cast observations to `np.float32` before returning from step() and reset().
**Warning signs:** SB3 `check_env()` warns about dtype mismatch.

### Pitfall 3: Portfolio value going negative
**What goes wrong:** Transaction costs can push cash balance negative if not checked.
**Why it happens:** Deducting costs after rebalancing without checking remaining cash.
**How to avoid:** Clip target allocation so total allocation + estimated costs <= portfolio value. Or deduct costs first, then allocate remaining.
**Warning signs:** NaN in observations (division by zero on portfolio value), negative cash ratios.

### Pitfall 4: Softmax overflow
**What goes wrong:** `np.exp(raw_actions)` overflows for large positive values.
**Why it happens:** Action space Box(-1, 1) is clipped but agent training may produce values outside range before clipping.
**How to avoid:** Subtract max before exp: `np.exp(x - np.max(x))`. This is numerically stable softmax.
**Warning signs:** Inf or NaN in weights after softmax.

### Pitfall 5: Look-ahead bias in episode data
**What goes wrong:** Environment uses future prices to compute current-step features.
**Why it happens:** Indexing error when pre-loading data (e.g., using prices[t+1] for step t).
**How to avoid:** At step t, observation uses features[t] and prices[t]. Portfolio value updates use prices[t] (beginning of period). Returns computed from prices[t-1] to prices[t].
**Warning signs:** Unrealistically high Sharpe ratios in training (too-good performance is always suspicious).

### Pitfall 6: Non-deterministic reset with same seed
**What goes wrong:** Different observations from reset(seed=42) on different runs.
**Why it happens:** Not using `self.np_random` for all randomness (crypto random start).
**How to avoid:** Call `super().reset(seed=seed)` first, then use `self.np_random.integers()` for random start index. Never use module-level `np.random`.
**Warning signs:** Tests fail intermittently.

### Pitfall 7: Portfolio state stale in observation
**What goes wrong:** Observation includes portfolio state from BEFORE the current step's trades, not after.
**Why it happens:** Assembling observation before executing trades.
**How to avoid:** Step sequence: (1) execute trades based on action, (2) advance to next time step, (3) update portfolio values at new prices, (4) assemble observation with current portfolio state, (5) compute reward.
**Warning signs:** Portfolio weights in observation don't match actual positions.

## Code Examples

### Complete step() Flow
```python
# Source: Gymnasium 1.2.3 API contract
def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Execute one trading step.

    Args:
        action: Raw agent output, shape (n_assets,).

    Returns:
        observation, reward, terminated, truncated, info
    """
    # 1. Process actions: softmax + deadzone
    target_weights = self._process_actions(action)

    # 2. Execute trades at current prices
    prices = self._prices[self._current_step]
    cost = self._portfolio.rebalance(target_weights, prices)

    # 3. Advance time
    self._current_step += 1

    # 4. Update portfolio value at new prices
    new_prices = self._prices[self._current_step]
    new_value = self._portfolio.portfolio_value(new_prices)
    daily_return = (new_value - self._prev_value) / self._prev_value
    self._prev_value = new_value

    # 5. Compute reward (rolling Sharpe + soft penalties)
    sharpe_reward = self._reward_fn.compute(daily_return)
    penalty = self._compute_risk_penalty(target_weights, new_value)
    reward = sharpe_reward - penalty

    # 6. Build observation with current portfolio state
    portfolio_state = self._get_portfolio_state(new_prices)
    obs = self._build_observation(portfolio_state)

    # 7. Check termination
    terminated = self._current_step >= self._max_steps - 1
    truncated = False

    # 8. Build info dict
    info = {
        "portfolio_value": new_value,
        "daily_return": daily_return,
        "transaction_cost": cost,
        "turbulence": self._get_turbulence(),
        "step": self._current_step,
    }

    return obs.astype(np.float32), float(reward), terminated, truncated, info
```

### Complete reset() Flow
```python
# Source: Gymnasium 1.2.3 API contract
def reset(
    self,
    seed: int | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Reset environment for new episode.

    Args:
        seed: Random seed for reproducibility.
        options: Additional reset options.

    Returns:
        observation, info
    """
    super().reset(seed=seed)

    # Select episode start (random for crypto, sequential for equity)
    self._current_step = self._select_start_step()

    # Reset portfolio to initial state
    self._portfolio.reset(self._initial_cash)
    self._reward_fn.reset()

    # Initial observation (100% cash portfolio state)
    prices = self._prices[self._current_step]
    self._prev_value = self._initial_cash
    portfolio_state = self._get_portfolio_state(prices)
    obs = self._build_observation(portfolio_state)

    info = {
        "portfolio_value": self._initial_cash,
        "start_step": self._current_step,
    }

    return obs.astype(np.float32), info
```

### Soft Risk Penalty
```python
def _compute_risk_penalty(
    self, weights: np.ndarray, portfolio_value: float
) -> float:
    """Compute soft penalty for risk limit violations."""
    penalty = 0.0

    # Position concentration penalty
    max_weight = self._max_position_size  # 0.25 equity, 0.50 crypto
    over_limit = np.maximum(weights - max_weight, 0.0)
    # Quadratic penalty scales with violation magnitude
    penalty += float(np.sum(over_limit ** 2)) * 10.0

    # Drawdown penalty
    drawdown = 1.0 - (portfolio_value / self._peak_value)
    if drawdown > self._max_drawdown_pct:
        excess = drawdown - self._max_drawdown_pct
        penalty += excess * 5.0

    return penalty
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gym.Env` (OpenAI Gym) | `gymnasium.Env` (Farama Gymnasium) | 2023 (Gym deprecated) | 5-tuple step(), seed in reset(), metadata dict |
| `env.reset()` returns obs | `env.reset()` returns (obs, info) | Gymnasium 0.26+ | Must unpack 2-tuple |
| `done` boolean | `terminated` + `truncated` | Gymnasium 0.26+ | Separate natural end from timeout |
| FinRL env_stocktrading | Custom Gymnasium env | Current best practice | FinRL's env is outdated, mixes concerns |

**Deprecated/outdated:**
- OpenAI `gym` package: Use `gymnasium` (Farama Foundation fork). SB3 2.x requires it.
- 4-tuple step() return: Always use 5-tuple `(obs, reward, terminated, truncated, info)`.
- `env.seed()` method: Pass seed via `env.reset(seed=42)`.

## Open Questions

1. **Soft penalty magnitude calibration**
   - What we know: Penalties subtract from Sharpe reward (~range -2 to +3 typically). Need penalties large enough to discourage violations but not so large they dominate learning signal.
   - What's unclear: Optimal coefficient values (10.0 for position, 5.0 for drawdown are starting points).
   - Recommendation: Start with suggested values, tune in Phase 7 during hyperparameter search. Log penalty magnitudes to info dict for monitoring.

2. **gym.make() kwargs for data loading**
   - What we know: `gym.make('StockTradingEnv-v0')` needs to receive feature data somehow.
   - What's unclear: Whether to pass data via `gym.make(kwargs=...)` or use factory methods.
   - Recommendation: Support both. Register with `gymnasium.register()` for SB3 compatibility (test criterion 1-2). Provide `StockTradingEnv.from_dataframe()` factory for practical use in training pipelines. The `gym.make()` path can use `kwargs` dict for passing features/prices/config.

3. **Portfolio state feature computation during step**
   - What we know: Observation includes 27 equity / 9 crypto portfolio state features (cash_ratio, exposure, daily_return, per-asset weight/pnl/days_since_trade).
   - What's unclear: How to compute "days_since_trade" efficiently within the environment.
   - Recommendation: Track last-trade-step per asset in PortfolioSimulator. Compute days_since_trade = current_step - last_trade_step[asset]. Initialize to 0 at reset.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured in pyproject.toml) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_envs.py -x -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | StockTradingEnv init + reset returns (156,) finite obs | unit | `uv run pytest tests/test_envs.py::test_equity_env_reset_observation_shape -x` | No - Wave 0 |
| TRAIN-01 | StockTradingEnv step returns valid 5-tuple | unit | `uv run pytest tests/test_envs.py::test_equity_env_step_contract -x` | No - Wave 0 |
| TRAIN-01 | StockTradingEnv passes check_env | integration | `uv run pytest tests/test_envs.py::test_equity_env_check_env -x` | No - Wave 0 |
| TRAIN-02 | CryptoTradingEnv init + reset returns (45,) finite obs | unit | `uv run pytest tests/test_envs.py::test_crypto_env_reset_observation_shape -x` | No - Wave 0 |
| TRAIN-02 | CryptoTradingEnv step returns valid 5-tuple | unit | `uv run pytest tests/test_envs.py::test_crypto_env_step_contract -x` | No - Wave 0 |
| TRAIN-02 | CryptoTradingEnv passes check_env | integration | `uv run pytest tests/test_envs.py::test_crypto_env_check_env -x` | No - Wave 0 |
| TRAIN-07 | Rolling Sharpe returns warmup values for first 19 bars | unit | `uv run pytest tests/test_envs.py::test_rolling_sharpe_warmup -x` | No - Wave 0 |
| TRAIN-07 | Rolling Sharpe returns proper ratio after 20 bars | unit | `uv run pytest tests/test_envs.py::test_rolling_sharpe_full_window -x` | No - Wave 0 |
| TRAIN-08 | Environment returns raw (non-normalized) observations | unit | `uv run pytest tests/test_envs.py::test_env_raw_observations -x` | No - Wave 0 |
| TRAIN-09 | Actions within +/-0.02 result in hold (no trade) | unit | `uv run pytest tests/test_envs.py::test_signal_deadzone_hold -x` | No - Wave 0 |
| TRAIN-10 | Info dict exposes turbulence value | unit | `uv run pytest tests/test_envs.py::test_info_dict_turbulence -x` | No - Wave 0 |
| TRAIN-11 | Equity episodes run 252 steps | unit | `uv run pytest tests/test_envs.py::test_equity_episode_length -x` | No - Wave 0 |
| TRAIN-11 | Crypto episodes run 540 steps with random start | unit | `uv run pytest tests/test_envs.py::test_crypto_episode_length_random_start -x` | No - Wave 0 |
| TRAIN-01 | gym.make('StockTradingEnv-v0') works | integration | `uv run pytest tests/test_envs.py::test_gym_make_equity -x` | No - Wave 0 |
| TRAIN-02 | gym.make('CryptoTradingEnv-v0') works | integration | `uv run pytest tests/test_envs.py::test_gym_make_crypto -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_envs.py -x -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_envs.py` -- covers all TRAIN-01, TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11
- [ ] Test fixtures in conftest.py: synthetic feature arrays (252+ rows equity, 540+ rows crypto) and price arrays for environment construction
- [ ] No new framework install needed -- pytest already configured

## Sources

### Primary (HIGH confidence)
- Gymnasium 1.2.3 installed and verified -- custom env API (reset 2-tuple, step 5-tuple, spaces.Box)
- Stable Baselines3 2.7.1 installed and verified -- check_env(), DummyVecEnv compatibility
- Phase 5 ObservationAssembler source code -- (156,) equity, (45,) crypto with deterministic ordering
- Phase 5 FeaturePipeline source code -- get_observation() pattern, DuckDB queries
- SwingRLConfig schema source code -- EquityConfig, CryptoConfig with position limits, drawdown thresholds

### Secondary (MEDIUM confidence)
- [SB3 Custom Environment Docs](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html) -- check_env usage, registration
- [Gymnasium Custom Environment Tutorial](https://gymnasium.farama.org/introduction/create_custom_env/) -- reset/step signatures, metadata, registration pattern
- [FinRL StockTradingEnv](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py) -- reference for portfolio tracking patterns (not used directly due to outdated API)

### Tertiary (LOW confidence)
- Soft penalty magnitude (10.0 position, 5.0 drawdown) -- derived from general RL reward shaping literature; needs empirical tuning in Phase 7

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and version-verified
- Architecture: HIGH -- Gymnasium API is well-documented, ObservationAssembler integration is clear from source
- Pitfalls: HIGH -- common issues well-known from SB3 docs and FinRL community
- Reward function: MEDIUM -- rolling Sharpe is straightforward math but penalty magnitudes need tuning
- Soft penalty coefficients: LOW -- starting values are reasonable estimates, need empirical validation

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable -- Gymnasium and SB3 APIs are mature)
