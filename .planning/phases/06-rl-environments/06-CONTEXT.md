# Phase 6: RL Environments - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Build Gymnasium-compatible trading environments for both equity (StockTradingEnv) and crypto (CryptoTradingEnv) that pass the step/reset contract, produce valid observations from the Phase 5 feature pipeline, and compute rolling 20-day Sharpe rewards. Covers TRAIN-01, TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11. Agent training (PPO/A2C/SAC), ensemble blending, and walk-forward backtesting are Phase 7. Execution middleware and risk veto layer are Phase 8.

</domain>

<decisions>
## Implementation Decisions

### Action space design
- Continuous portfolio weights via softmax conversion — raw agent output transformed into positive percentages summing to 100%
- Cash as remainder — agent outputs weights for assets only (8 ETFs equity, 2 cryptos). Unallocated portion becomes cash automatically
- Action space shape: Box(8,) for equity, Box(2,) for crypto
- Signal deadzone: actions within +/-0.02 of zero mapped to "hold" (TRAIN-09) — no trade executed

### Reward function
- Rolling 20-day Sharpe ratio with expanding-window warmup for first 19 bars (TRAIN-07)
- Transaction costs deducted from portfolio value before Sharpe calculation:
  - Equity: ~0.06% round-trip (commission-free + 0.02% spread + 0.01% slippage)
  - Crypto: ~0.22% round-trip (0.20% commission + 0.01-0.05% spread + 0.01% slippage)
- Soft risk-violation penalties subtracted from reward when agent exceeds position limits or drawdown thresholds
- No additional turnover penalty — transaction cost deductions naturally discourage excessive trading

### Risk enforcement during training
- Soft penalties (not hard enforcement) — agent CAN exceed position limits (25% equity, 50% crypto per asset) and drawdown thresholds, but receives a reward penalty when it does
- This teaches risk awareness through experience rather than hitting invisible walls
- Phase 8's production veto layer provides hard enforcement as a backstop

### Episode structure
- Full-length episodes always — no early termination on drawdown or bankruptcy
- Equity: 252-day segments (TRAIN-11)
- Crypto: 540 4H bars (~3 months) with random start within training window (TRAIN-11)
- Combined with soft risk penalties, agent learns both risk awareness and recovery strategies

### Training capital
- $100,000 normalized amount for both environments — agent learns allocation percentages, not dollar amounts
- Learned strategy transfers to any capital level ($447 live or $100K paper)
- Matches Doc 04 §5 paper trading capital specification
- Configurable via SwingRLConfig (initial_amount field)

### VecNormalize handling (TRAIN-08)
- VecNormalize statistics frozen (training=False) during inference to prevent train/serve skew
- Applied in Phase 7 on top of Phase 5's rolling z-score normalization
- Phase 6 builds the raw environment; Phase 7 wraps it with VecNormalize

### Adaptive validation windows (TRAIN-10)
- Shrink ensemble validation windows by 50% during high turbulence periods
- Expand during calm periods
- Turbulence threshold per environment (90th percentile, Phase 5 calculator)

### Claude's Discretion
- Gymnasium registration pattern (StockTradingEnv-v0, CryptoTradingEnv-v0)
- Data loading strategy (pre-load features into memory vs DuckDB per step)
- Portfolio simulation internals (dollar-based tracking with percentage weights)
- Environment info dict contents (portfolio value, trade log, metrics for Phase 7)
- Render mode support for debugging
- Soft penalty magnitude and scaling for risk violations
- Random seed / reproducibility patterns
- Config schema additions for environment parameters (initial_amount, episode_length, etc.)

</decisions>

<specifics>
## Specific Ideas

- Observation assembly must be identical between training and inference (Doc 08 §13.1) — use the same ObservationAssembler from Phase 5
- Look-ahead bias already prevented by ALFRED vintage data + ASOF JOIN on release_date (Phase 3/4) — environments inherit this automatically
- Environment benchmarks for comparison: equity vs SPY buy-and-hold, crypto vs 50/50 BTC/ETH buy-and-hold (Doc 09 Dashboard spec)
- The "agent never trades" problem (Doc 03) is mitigated by: cash-as-remainder (no explicit cash hoarding lever), entropy coefficient in training (Phase 7), and signal deadzone keeping the agent from micro-churning
- Portfolio state features (27 equity / 9 crypto dims) use defaults of 100% cash in Phase 5 — Phase 6 environments provide real position data to the assembler

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ObservationAssembler` (features/assembler.py): Ready-made assembler producing (156,) equity and (45,) crypto vectors — environments call this directly
- `FeaturePipeline` (features/pipeline.py): `get_observation(environment, date_or_datetime)` returns complete observation vector from DuckDB
- `TurbulenceCalculator` (features/turbulence.py): Mahalanobis distance calculator for turbulence crash protection
- `SwingRLConfig` (config/schema.py): EquityConfig, CryptoConfig, FeaturesConfig with position limits, drawdown thresholds, normalization windows
- `DatabaseManager` (data/db.py): Singleton with DuckDB/SQLite context managers
- Exception hierarchy (utils/exceptions.py): DataError, ConfigError ready for environment errors

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- Absolute imports only (from swingrl.envs.equity import EquityEnv)
- Pydantic config access via load_config()
- TDD: failing test first, then implementation
- structlog with keyword args for logging
- DuckDB feature tables: features_equity (symbol, date), features_crypto (symbol, datetime)

### Integration Points
- Phase 5 feature pipeline: environments consume observations via ObservationAssembler
- Phase 5 DuckDB tables: features_equity, features_crypto, hmm_state_history, fundamentals
- Phase 7 training: wraps environments with DummyVecEnv + VecNormalize, trains PPO/A2C/SAC
- Phase 8 execution: production inference uses frozen VecNormalize + environment predict()
- Config: new environment section needed in swingrl.yaml (initial_amount, episode_length_bars, transaction_cost_pct, etc.)
- Target module: src/swingrl/envs/ (currently empty __init__.py)

</code_context>

<deferred>
## Deferred Ideas

- pandas_ta re-evaluation (deferred from Phase 5) — evaluate if Python 3.12+ becomes viable or pandas_ta fixes 3.11 support
- Walk-forward backtesting framework — Phase 7 (VAL-01, VAL-02)
- Sharpe-weighted softmax ensemble blending — Phase 7 (TRAIN-06)
- ConvergenceCallback early stopping — Phase 7 (VAL-05)
- Production risk veto layer (hard enforcement) — Phase 8 (PAPER-03)
- Circuit breaker halt-state persistence — Phase 8 (PAPER-06)

</deferred>

---

*Phase: 06-rl-environments*
*Context gathered: 2026-03-07*
