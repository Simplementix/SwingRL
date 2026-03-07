# Phase 5: Feature Engineering - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Compute the full 156-dimension equity and 45-dimension crypto observation vectors from raw OHLCV bars in DuckDB, with technical indicators, fundamental features (equities only), macro regime alignment, HMM regime probabilities, rolling z-score normalization, correlation pruning, and an observation vector assembler. Covers FEAT-01 through FEAT-11. Portfolio state features (27 equity / 9 crypto dims) use defaults until Phase 6 RL environments provide real position data. Turbulence Index calculator is built here but consumed by Phase 6 environments.

</domain>

<decisions>
## Implementation Decisions

### Observation vector dimensions (FEAT-07)
- **Equity (156 dimensions):**
  - Per-asset features (15 x 8 ETFs = 120): Price Action (9) + Multi-Timeframe (2) + Fundamentals (4)
  - Price Action per asset (9): Price/SMA-50 ratio, Price/SMA-200 ratio, RSI-14, MACD line, MACD histogram, Bollinger Band position (0-1), ATR-14 (% of price), Volume/SMA-20 ratio, ADX-14
  - Multi-Timeframe per asset (2): Weekly trend direction (binary), Weekly RSI-14
  - Fundamentals per asset (4): P/E ratio (sector-relative z-score), Earnings growth rate, Debt-to-equity ratio, Dividend yield
  - Shared Macro (6): VIX z-score (1yr), yield curve spread (10Y-2Y), yield curve direction, Fed Funds 90-day change, CPI YoY, unemployment 3-month direction
  - HMM Regime (2): P(bull), P(bear) from 2-state Gaussian HMM on SPY
  - Turbulence Index (1): Mahalanobis distance (computed on-the-fly, not stored)
  - Portfolio State (27): Cash ratio + portfolio exposure + daily return (3 fixed) + per-asset weight/unrealized PnL %/days since trade (3 x 8 = 24)
- **Crypto (45 dimensions):**
  - Per-asset features (13 x 2 assets = 26): Price Action (9) + Multi-Timeframe (4)
  - Price Action per asset (9): Same as equity
  - Multi-Timeframe per asset (4): Daily trend direction (binary), Daily RSI-14, 4H RSI-14, 4H Price/SMA-20 ratio
  - Shared Macro (6): Same as equity
  - HMM Regime (2): P(bull), P(bear) from 2-state Gaussian HMM on BTC
  - Turbulence Index (1): Mahalanobis distance (calibrated independently for crypto)
  - Overnight Context (1): hours_since_equity_close (macro staleness indicator)
  - Portfolio State (9): Cash ratio + exposure + daily return (3) + per-asset weight/PnL/days (3 x 2 = 6)
- **Assembly order:** [Per-Asset Features (alpha-sorted)] + [Shared Macro] + [HMM] + [Turbulence] + [Portfolio State]
- **Portfolio state in Phase 5:** Use default values (100% cash, zero positions) — real position data comes from Phase 6 RL environments

### Technical indicators (FEAT-01, FEAT-02, FEAT-10)
- **Library: stockstats** (not pandas_ta) — Phase 1 decision due to pandas_ta requiring Python >= 3.12. stockstats 0.6.8 installed. Re-evaluation scheduled for Phase 6
- **Gap to verify:** Researcher must confirm stockstats computes SMA, RSI-14, MACD (line + histogram), Bollinger Band position, ATR-14, ADX-14, Volume/SMA-20 ratio with correct semantics. If any indicator is missing or differs significantly, consider TA-Lib as alternative
- Derived features (FEAT-02): Log returns (1d, 5d, 20d) and Bollinger Band width are intermediate/diagnostic features — the authoritative 156/45 observation vector does NOT include them in the 9 per-asset price action slots. They are used as HMM inputs (log-returns + 20-day realized volatility) and for correlation analysis
- Weekly-derived features (FEAT-10): Computed from DuckDB aggregation views (Phase 4 "store lowest, aggregate up") — weekly SMA trend direction and weekly RSI-14

### Fundamental data (FEAT-03)
- **Source: yfinance** primary, **Alpha Vantage** fallback (free tier, 25 req/day)
- Alpha Vantage API key via `ALPHA_VANTAGE_API_KEY` env var
- **Tiered refresh cadence:**
  - Quarterly: full fundamental refresh for all 8 equity ETFs
  - Weekly (Sundays): pre-check for equities with earnings in next 7 days — fetch updated fundamentals immediately
  - Daily (8:00 AM ET): log upcoming earnings dates for awareness (corporate actions pipeline from Phase 4)
- **Validation:** P/E positive (or N/A for loss-making), D/E non-negative, no >100% quarter-over-quarter change
- **Normalization:** Sector-relative z-scores (cross-sectional, not time-series)
- **Storage:** DuckDB `fundamentals` table (raw ratios), then processed into `features_equity` per-asset columns

### Macro regime features (FEAT-04)
- Forward-fill via DuckDB ASOF JOIN on `macro_features.release_date` — prevents look-ahead bias
- 6 shared features: VIX z-score (1yr rolling), yield curve spread, yield curve direction (binary), Fed Funds 90-day change, CPI YoY, unemployment 3-month direction (binary)
- Same 6 features shared across both equity and crypto environments
- Crypto forward-fills from last available equity close (macro updates on business days only)

### HMM regime detection (FEAT-05)
- **Equity HMM:** 2-state Gaussian on SPY, rolling 1,260-day (5-year) window, refit weekly (Monday after close)
- **Crypto HMM:** 2-state Gaussian on BTC (4H bars aggregated to daily returns), rolling 2,000 4H-bar (~1-year) window, refit daily at 00:05 UTC
- **Inputs:** Log-returns + 20-day realized volatility (2 features)
- **Fitting:** hmmlearn, n_iter=200, 5-10 random initializations, keep best log-likelihood
- **Covariance:** Equity = "full" (captures negative return-volatility leverage effect), Crypto = "diag" (empirically better for BTC)
- **Label consistency:** Warm-start from previous parameters + mean-return ordering (higher mean = "bull")
- **Cold-start:** Informed prior initialization (positive mean/low volatility for bull) until ~500 bars available
- **Storage:** DuckDB `hmm_state_history` table for P(bull), P(bear) over time

### Rolling z-score normalization (FEAT-06)
- Per-environment windows: 252 bars (equity), 360 bars (crypto) — locked from prior phases
- Applied to stored technical features before observation vector assembly
- VecNormalize (SB3 wrapper) applied separately during training (Phase 7) — Phase 5 does rolling z-scores only

### Correlation pruning (FEAT-09)
- Pairwise Pearson r > 0.85 threshold
- Applied at pipeline build time (pre-training), not dynamically
- **Domain-driven drop rules:**
  - Momentum: favor RSI-14, drop Stochastic %K / Williams %R / CCI (already excluded per Doc 08 §12)
  - MACD: use histogram, not raw line or signal
  - Volatility: use VIX z-score, not raw VIX
  - SMA exception: keep both SMA-50 and SMA-200 ratios unless r > 0.90
- Run correlation matrix on full training dataset, finalize observation space before training
- Pruning results documented (which pairs flagged, which dropped, rationale)

### Feature addition protocol (FEAT-08)
- **Threshold:** New feature kept only if validation Sharpe improves by >= 0.05
- **Overfitting guard:** Discard if training Sharpe improves but validation Sharpe decreases
- **Statistical significance:** Major experiments (e.g., FinBERT) run across 3+ random seeds
- **Process:** Hybrid manual/automated
  - Automated: Retrain pipeline (`scripts/retrain.py`) generates metrics for baseline vs candidate
  - Manual: Jupyter notebook comparison of Sharpe, equity curves, drawdown
  - Manual promotion: Major observation space changes flagged for operator review
- **Shadow validation:** Passing features go to shadow mode 2-4 weeks (equity) or 1-2 weeks (crypto) before promotion
- **Phase 5 scope:** Build the A/B comparison infrastructure and document the protocol. Full shadow validation is Phase 7+ (requires trained agents)

### Pipeline execution model
- **Core logic:** `src/swingrl/features/` module (new package)
- **CLI:** `scripts/compute_features.py` for manual recomputation and initial seeding
- **Production trigger:** Auto-triggered after ingestion passes validation (daily equity, every 4H crypto)
- **DuckDB interaction:**
  - Read: `ohlcv_daily`, `ohlcv_4h`, `macro_features` from DuckDB
  - Write: `features_equity`, `features_crypto`, `hmm_state_history`, `fundamentals` to DuckDB
- **Observation assembler:** Separate class that reads stored features + computes on-the-fly metrics (turbulence) + pulls portfolio state (SQLite positions) into flat NumPy array
- **Deterministic ordering:** Assets alpha-sorted, feature groups concatenated in fixed order

### DuckDB tables (Phase 5 additions)
- **features_equity:** symbol (TEXT), date (DATE), + 15 feature columns per the price action / multi-timeframe / fundamentals spec. Sort: (symbol, date)
- **features_crypto:** symbol (TEXT), datetime (TIMESTAMP), + 13 feature columns per the price action / multi-timeframe spec. Sort: (symbol, datetime)
- **fundamentals:** symbol (TEXT), date (DATE), pe_ratio (DOUBLE), earnings_growth (DOUBLE), debt_to_equity (DOUBLE), dividend_yield (DOUBLE), sector (TEXT), fetched_at (TIMESTAMP). Sort: (symbol, date)
- **hmm_state_history:** environment (TEXT), date (DATE), p_bull (DOUBLE), p_bear (DOUBLE), log_likelihood (DOUBLE), fitted_at (TIMESTAMP). Sort: (environment, date)

### Claude's Discretion
- Exact stockstats API calls and any wrapper needed for missing indicators
- Turbulence Index implementation details (Mahalanobis distance, covariance window)
- Observation assembler class design and interface
- Feature pipeline module organization (per-indicator vs grouped)
- Exact DuckDB DDL column types and constraints beyond what's specified
- CLI argument design for compute_features.py
- Test fixture data for indicator validation
- Warmup period handling (how many bars before NaN-free output)

</decisions>

<specifics>
## Specific Ideas

- stockstats is FinRL's native TA library — output should be compatible with FinRL patterns if we later integrate FinRL components
- Turbulence Index computed on-the-fly via Mahalanobis distance (not stored) — matches FinRL's built-in approach
- HMM warm-start from previous parameters prevents state-flipping between refits — critical for stable regime signals
- Equity HMM uses "full" covariance to capture the well-known negative correlation between returns and volatility (leverage effect)
- ASOF JOIN on macro_features.release_date ensures FRED data alignment without look-ahead bias — this is already set up from Phase 3/4
- Observation vector assembly must be identical between training and inference to prevent train/serve skew
- Doc 14 notes that crypto multi-timeframe DDL column names may need adjustment to match Doc 08 feature spec during implementation

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `DatabaseManager` (data/db.py): Singleton with `duckdb()` and `sqlite()` context managers — feature pipeline reads/writes through this
- `BaseIngestor` (data/base.py): Pattern for fetch/validate/store + DuckDB sync — feature pipeline follows similar pattern
- `DataValidator` (data/validation.py): 12-step checklist — extend or create parallel FeatureValidator for FEAT-08 reasonableness checks
- `SwingRLConfig` (config/schema.py): Needs new `features` section for normalization windows, HMM params, correlation threshold
- `DataError` (utils/exceptions.py): Ready for feature computation errors
- structlog (utils/logging.py): Established — feature modules use `structlog.get_logger(__name__)`
- DuckDB aggregation views (Phase 4): `ohlcv_weekly` view for weekly-derived features
- stockstats 0.6.8: Installed, provides SMA, RSI, MACD, Bollinger, ATR, ADX via StockDataFrame wrapper

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- pathlib.Path for all file operations
- Absolute imports only (from swingrl.features.technical import ...)
- Pydantic config access via load_config()
- TDD: failing test first, then implementation
- API credentials via environment variables only
- structlog with keyword args (no f-strings)
- DuckDB replacement scan for DataFrame-to-table sync (Phase 4 pattern)

### Integration Points
- Phase 4 DuckDB tables: ohlcv_daily, ohlcv_4h, macro_features — input data
- Phase 4 aggregation views: ohlcv_weekly — for weekly-derived features
- Phase 6 RL environments: consume observation vectors from the assembler
- Phase 7 training: VecNormalize wrapper applied on top of Phase 5 z-score normalization
- Phase 9 scheduler: triggers feature pipeline after ingestion
- Config: new features section needed in swingrl.yaml

### New Dependencies (Phase 5)
- alpha-vantage or alpha_vantage: Alpha Vantage API client (fundamental data fallback)
- No other new heavy dependencies — hmmlearn and stockstats already installed

</code_context>

<deferred>
## Deferred Ideas

- FinBERT sentiment pipeline (HARD-03) — Phase 10, includes sentiment_scores DuckDB table and Alpha Vantage NEWS_SENTIMENT endpoint
- Alpha Vantage Earnings Call Transcript Sentiment — post-Milestone 7
- Options-derived features (IV rank, put/call OI, VRP, GEX, skew) — Phase 3 / OPT-05
- pandas_ta re-evaluation — Phase 6 (if Python 3.12+ becomes viable or pandas_ta fixes 3.11 support)
- training_datasets, model_metadata, ensemble_weights, backtest_results DuckDB tables — Phase 7
- On-chain crypto metrics — v2+ (out of scope per PROJECT.md)

</deferred>

---

*Phase: 05-feature-engineering*
*Context gathered: 2026-03-06*
