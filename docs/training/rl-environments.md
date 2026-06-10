# RL Environments Reference

Living reference for the two RL training environments. Source-of-truth for observation dims, action shapes, episode lengths, and yaml knobs. Update this doc when any of the referenced code changes.

**Last verified against code:** 2026-04-14

## Environments at a glance

| Env | Class | Bar | Assets | Obs dim | Action dim | Episode bars |
|-----|-------|-----|--------|---------|-----------:|-------------:|
| Equity | `StockTradingEnv` | 1D | 8 ETFs (SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK) | **164** (180 w/ sentiment) | 9 | 252 (~1 yr) |
| Crypto | `CryptoTradingEnv` | 4H | 2 (BTC, ETH) | **47** | 3 | 540 (~3 mo) |

**Shared base:** `BaseTradingEnv` in `src/swingrl/envs/base.py` — both envs inherit portfolio simulation, reward, risk penalties.
Subclasses override only `_select_start_step()` (both currently random).

## Observation space

Box, `low=-inf, high=+inf, dtype=float32, shape=(obs_dim,)`. Raw values (not normalized) — `VecNormalize` wraps the env during training (TRAIN-08).

Assembly order is **deterministic**:
`[per-asset alpha-sorted] + [macro] + [HMM] + [turbulence] + [overnight (crypto only)] + [portfolio state]`

Authoritative assembly: `src/swingrl/features/assembler.py::ObservationAssembler`.
Never hardcode `164` / `180` — call `equity_obs_dim(sentiment_enabled, n_equity_symbols)`.

### Equity (164 dims, default)

| Block | Dims | Per-unit | Composition |
|-------|------|----------|-------------|
| Per-asset (× 8 symbols, alpha-sorted) | **120** | 15 | 9 price-action + 2 weekly + 4 fundamentals (see below) |
| Macro (shared) | 6 | — | vix_z, yield_spread, yield_dir, fed_funds_90d_Δ, cpi_yoy, unemployment_3m_dir |
| HMM regime | 2 | — | P(bull), P(bear) |
| Turbulence | 1 | — | turbulence_index |
| Portfolio state | **35** | — | 3 fixed + 4 × 8 assets |

**Per-asset 15 features (order matters):**
`price_sma50_ratio, price_sma200_ratio, rsi_14, macd_line, macd_histogram, bb_position, atr_14_pct, volume_sma20_ratio, adx_14, weekly_trend_dir, weekly_rsi_14, pe_zscore, earnings_growth, debt_to_equity, dividend_yield`

**Sentiment variant (180 dims):** if `config.sentiment.enabled=true`, each asset gets 2 extra features appended (`sentiment_score, sentiment_confidence`) → 17 × 8 = 136 per-asset block, total 180.

### Crypto (47 dims)

| Block | Dims | Per-unit | Composition |
|-------|------|----------|-------------|
| Per-asset (× 2 symbols, alpha-sorted) | **26** | 13 | 9 price-action + 4 multi-timeframe |
| Macro (shared) | 6 | — | same as equity |
| HMM regime | 2 | — | P(bull), P(bear) |
| Turbulence | 1 | — | turbulence_index |
| Overnight context | 1 | — | hours_since_equity_close |
| Portfolio state | **11** | — | 3 fixed + 4 × 2 assets |

**Per-asset 13 features:**
`price_sma50_ratio, price_sma200_ratio, rsi_14, macd_line, macd_histogram, bb_position, atr_14_pct, volume_sma20_ratio, adx_14, daily_trend_dir, daily_rsi_14, four_h_rsi_14, four_h_price_sma20_ratio`

### Portfolio state sub-block (both envs)

3 fixed + 4 per-asset, interleaved after the fixed block:

| Index | Name | Range | Notes |
|-------|------|-------|-------|
| 0 | portfolio_cash_ratio | [0, 1] | cash / total_value |
| 1 | portfolio_exposure | [0, 1] | Σ asset weights |
| 2 | portfolio_daily_return | real | (new − prev) / prev |
| 3 + 4i | weight | [0, 1] | asset i weight |
| 4 + 4i | weight_deviation | real | weight − 1/n_assets (0 if total_exposure=0) |
| 5 + 4i | unrealized_pnl_pct | real | (price − cost_basis) / cost_basis, 0 if no position |
| 6 + 4i | bars_since_trade | ≥0 | step_count − last_trade_step[i] |

The env overwrites the last `portfolio_dim` elements of the pre-computed feature row with live portfolio state at each step (`BaseTradingEnv._build_observation`).

## Action space

Box, `low=-1.0, high=+1.0, dtype=float32, shape=(n_assets + 1,)`. Last element is **cash preference**.

| Env | Action shape |
|-----|-------------:|
| Equity | 9 |
| Crypto | 3 |

**Pipeline (`process_actions` in `portfolio.py`):**

1. Numerically stable **softmax** across all `n_assets + 1` dims → weights that sum to 1.0.
2. Drop the cash dim → asset weights sum to ≤ 1.0 (remainder is implicit cash).
3. **Deadzone filter:** if `|new − current| < deadzone`, keep current weight (no trade).
4. Clamp Σ ≤ 1.0 by rescaling if deadzone preservation pushes the sum over.

## Episode mechanics

| Aspect | Equity | Crypto |
|--------|--------|--------|
| Episode bars | `config.environment.equity_episode_bars` (252) | `config.environment.crypto_episode_bars` (540) |
| Start step | Random via `np_random.integers(0, len(features) − episode_bars)` | Same — random |
| Truncation | Capped so `current_step` never indexes past `len(features) − 1` | Same |
| Termination | `current_step >= max_step` → `terminated=True` | Same |
| Truncation signal | Always `False` | Always `False` |

**Reset returns** `(obs, info)` where obs is built with 100% cash portfolio state at `features[start_step]`.

**Step sequence (`BaseTradingEnv.step`):**

1. Read `current_weights` from portfolio at current prices.
2. `process_actions` → `target_weights`.
3. `portfolio.rebalance` → executes trades, deducts transaction cost, returns `cost`.
4. Advance `current_step += 1`, `step_count += 1`.
5. Compute `new_value` at new prices, `daily_return`, update `peak_value`.
6. `reward = RollingSharpeReward(window=20).compute(daily_return) − risk_penalty`.
7. Build obs with live portfolio state, info dict with `reward_components`.

**On terminal step:** `info["trade_log"] = list(portfolio.trade_log)` is attached before `DummyVecEnv` auto-resets.

## Reward (summary)

Full reward shaping doc: [`docs/training/reward-shaping.md`](reward-shaping.md).

Short form:

```
reward = rolling_sharpe_20(daily_return) − risk_penalty
risk_penalty = Σ position_penalty_coeff × max(0, w − max_position_size)²
             + drawdown_penalty_coeff × max(0, drawdown − max_drawdown_pct)
```

The env also attaches `info["reward_components"]` (profit/sharpe/drawdown/turnover) — consumed by the memory system for reward-shaping guidance; these are *observations*, not reward components proper.

## Configurable values (yaml)

All yaml paths, defaults, and validators live in `src/swingrl/config/schema.py`.

**Shared (applies to both envs):** `environment.*`

| Key | Default | Notes |
|-----|---------|-------|
| `initial_amount` | 100_000.0 | Starting cash in dollars |
| `equity_episode_bars` | 252 | ≥ 50 |
| `crypto_episode_bars` | 540 | ≥ 50 |
| `equity_transaction_cost_pct` | 0.0006 | Round-trip cost fraction |
| `crypto_transaction_cost_pct` | 0.0022 | Binance.US maker/taker-ish |
| `signal_deadzone` | 0.02 | [0, 0.1] |
| `position_penalty_coeff` | 10.0 | Quadratic on excess weight |
| `drawdown_penalty_coeff` | 5.0 | Linear on excess drawdown |

**Per-env:** `equity.*` / `crypto.*`

| Key | Equity default | Crypto default |
|-----|---------------:|---------------:|
| `max_position_size` | 0.25 | 0.50 |
| `max_drawdown_pct` | 0.10 | 0.12 |
| `symbols` | 8 ETFs | [BTC, ETH] |

## Hardcoded values (not yaml-tunable — code edit required)

These are constants or in-code choices. Changing them requires editing source + tests, not yaml.

### Env spaces & shape

| Value | Where | Current | Notes |
|-------|-------|---------|-------|
| Observation `low` / `high` | `base.py::__init__` | `-inf` / `+inf` | `VecNormalize` handles scaling |
| Observation `dtype` | `base.py::__init__` | `float32` | |
| Action `low` / `high` | `base.py::__init__` | `-1.0` / `+1.0` | Softmax re-normalizes |
| Action `dtype` | `base.py::__init__` | `float32` | |
| Action layout | `base.py::__init__` | `(n_assets + 1,)` with cash as last element | |
| Initial portfolio state | `assembler.py::_default_portfolio_state` | 100% cash, zero positions | |
| Symbol sort order in obs | `assembler.py::__init__` | alphabetical | Rebuilds obs index if re-sorted |

### Feature block sizes

| Constant | Value | File |
|----------|------:|------|
| `EQUITY_PER_ASSET_BASE` | 15 | `features/assembler.py` |
| `SENTIMENT_FEATURES_PER_ASSET` | 2 | `features/assembler.py` |
| `CRYPTO_PER_ASSET` | 13 | `features/assembler.py` |
| `SHARED_MACRO` | 6 | `features/assembler.py` |
| `HMM_REGIME` | 2 | `features/assembler.py` |
| `TURBULENCE` | 1 | `features/assembler.py` |
| `OVERNIGHT_CONTEXT` | 1 (crypto only) | `features/assembler.py` |
| Portfolio block fixed | 3 | `base.py::_get_portfolio_state` |
| Portfolio block per-asset | 4 (interleaved) | `base.py::_get_portfolio_state` |

**Derived (changes with config):** `EQUITY_PORTFOLIO = 3 + 4 × len(config.equity.symbols)` (currently 35). `CRYPTO_PORTFOLIO = 3 + 4 × len(config.crypto.symbols)` (currently 11). Full `obs_dim` via `equity_obs_dim(sentiment_enabled, n_equity_symbols)`.

### Reward & action math

| Value | Where | Current | Notes |
|-------|-------|---------|-------|
| Rolling Sharpe window | `rewards.py::RollingSharpeReward` | 20 bars | Wired via `RollingSharpeReward(window=20)` in `base.py` |
| Sharpe `ddof` | `rewards.py::compute` | 1 | Sample std |
| Near-zero std guard | `rewards.py::compute` | `1e-8` | Returns 0.0 below |
| Risk penalty on position | `base.py::_compute_risk_penalty` | quadratic `coeff × max(0, w − max_pos)²` | Coefficient is yaml-tunable |
| Risk penalty on drawdown | `base.py::_compute_risk_penalty` | linear `coeff × max(0, dd − max_dd)` | Coefficient is yaml-tunable |
| Action activation | `portfolio.py::process_actions` | numerically-stable softmax | No alternative selectable |
| Cash dim handling | `portfolio.py::process_actions` | dropped after softmax; weights sum ≤ 1.0 | |

### Episode start strategy

| Value | Where | Current | Notes |
|-------|-------|---------|-------|
| Equity start | `equity.py::_select_start_step` | `np_random.integers(0, n − episode_bars)` | Uniform; no regime stratification |
| Crypto start | `crypto.py::_select_start_step` | same | Uniform |

### Thresholds / tolerances

| Value | Where | Current | Purpose |
|-------|-------|---------|---------|
| Trade-detection weight delta | `base.py::step` | `1e-8` | Marks an asset as "traded" this step |
| Trade-log share delta | `portfolio.py::rebalance` | `1e-10` | Skip writing micro-trades to log |
| Safe-price floor | `portfolio.py::rebalance` | substitute `1.0` when `price <= 0` | Division guard |

### Cost-basis accounting

| Behavior | Where | Notes |
|----------|-------|-------|
| On buy | `portfolio.py::rebalance` | Weighted average of old basis + new purchase price |
| On full exit | `portfolio.py::rebalance` | Reset to 0.0 |
| On partial sell | `portfolio.py::rebalance` | Basis unchanged |

If any value above needs to become yaml-tunable, promote it in `config/schema.py` and update this table.

## Invariants

- Features array shape must be `(n_steps, obs_dim)`; prices shape `(n_steps, n_assets)`; rows aligned on step index.
- Observation is returned **raw**; `VecNormalize` handles running-mean/std.
- `NaN` in any assembled observation raises `DataError` (`ObservationAssembler` checks).
- `np_random` is set by `super().reset(seed=seed)` — honor it for reproducible episode starts.
- Trade log is truncated/reset on `portfolio.reset()`; always consume via terminal-step `info["trade_log"]`.

## Known issues / open questions

- Sentiment features (180-dim variant) are implemented end-to-end but not yet enabled in production yaml.
- Crypto env uses the same macro block as equity; no fed-funds / CPI weighting differs per env.
- Both envs select start step randomly; no seasonal / regime-stratified sampling.
- Weight deviation feature zeroes out when `total_exposure == 0` (fully cash) — check if that's the intended neutral value for policy input.

## Source of truth

| Concern | File |
|---------|------|
| Env base | `src/swingrl/envs/base.py` |
| Equity env | `src/swingrl/envs/equity.py` |
| Crypto env | `src/swingrl/envs/crypto.py` |
| Obs assembly | `src/swingrl/features/assembler.py` |
| Portfolio sim + action processing | `src/swingrl/envs/portfolio.py` |
| Rolling-Sharpe reward | `src/swingrl/envs/rewards.py` |
| Config schema | `src/swingrl/config/schema.py` (classes `EnvironmentConfig`, `EquityConfig`, `CryptoConfig`) |

## Changelog

- **2026-04-14** — Initial version.
