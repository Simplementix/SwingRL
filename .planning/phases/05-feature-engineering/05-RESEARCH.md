# Phase 5: Feature Engineering - Research

**Researched:** 2026-03-07
**Domain:** Financial feature engineering (technical indicators, fundamentals, HMM regime detection, normalization, observation vector assembly)
**Confidence:** HIGH

## Summary

Phase 5 transforms raw OHLCV bars (stored in DuckDB from Phase 4) into the 156-dimension equity and 45-dimension crypto observation vectors consumed by RL environments in Phase 6. The pipeline spans five major subsystems: technical indicator computation (stockstats), fundamental data retrieval (yfinance + Alpha Vantage fallback), HMM regime detection (hmmlearn), rolling z-score normalization, and observation vector assembly.

All core libraries are already installed and verified working: stockstats computes all 9 required price-action indicators (SMA-50/200, RSI-14, MACD line/histogram, Bollinger bands, ATR-14, ADX-14, Volume/SMA-20). hmmlearn 0.3.3 provides GaussianHMM with full/diag covariance, though warm-start requires a ridge regularization workaround for near-singular covariance matrices. yfinance 1.2.0 is installed for fundamental data. alpha-vantage needs to be added as a dependency for the fallback path.

**Primary recommendation:** Build the feature pipeline as a `src/swingrl/features/` package with separate modules for technical indicators, fundamentals, HMM regime, macro alignment, normalization, correlation pruning, and observation assembly. Use the established DatabaseManager pattern for all DuckDB reads/writes.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Observation vector dimensions (FEAT-07):** Equity 156 dims (120 per-asset + 6 macro + 2 HMM + 1 turbulence + 27 portfolio state), Crypto 45 dims (26 per-asset + 6 macro + 2 HMM + 1 turbulence + 1 overnight context + 9 portfolio state). Assembly order: [Per-Asset Features (alpha-sorted)] + [Shared Macro] + [HMM] + [Turbulence] + [Portfolio State]
- **Technical indicators (FEAT-01, FEAT-02, FEAT-10):** Use stockstats (not pandas_ta) -- Phase 1 decision. Re-evaluation Phase 6. Per-asset price action (9): Price/SMA-50, Price/SMA-200, RSI-14, MACD line, MACD histogram, Bollinger Band position (0-1), ATR-14 (% of price), Volume/SMA-20, ADX-14. Derived features (log returns 1d/5d/20d, Bollinger width) are intermediate only (HMM inputs, correlation analysis)
- **Fundamental data (FEAT-03):** yfinance primary, Alpha Vantage fallback. Quarterly refresh + weekly pre-check for upcoming earnings. Sector-relative z-scores
- **Macro regime (FEAT-04):** Forward-fill via DuckDB ASOF JOIN on macro_features.release_date. 6 shared features
- **HMM regime (FEAT-05):** 2-state Gaussian HMM. Equity: SPY, full covariance, 1260-day window, refit weekly. Crypto: BTC (4H->daily aggregation), diag covariance, 2000 4H-bar window, refit daily. 5-10 random inits, best log-likelihood. Warm-start + mean-return ordering for label consistency
- **Normalization (FEAT-06):** Rolling z-score with 252-bar equity / 360-bar crypto windows
- **Correlation pruning (FEAT-09):** Pearson r > 0.85 threshold. Domain-driven drop rules specified. Permutation importance as complement
- **Feature addition protocol (FEAT-08):** Sharpe improvement >= 0.05 threshold. Overfitting guard. Phase 5 builds A/B infrastructure only
- **DuckDB tables:** features_equity, features_crypto, fundamentals, hmm_state_history
- **Portfolio state in Phase 5:** Use defaults (100% cash, zero positions) until Phase 6
- **Warmup:** Equity 252 bars, Crypto 360 bars. Post-warmup NaN = pipeline bug
- **Turbulence Index:** Equity expanding lookback (post-252), Crypto rolling 1080-bar. Use np.linalg.pinv for crypto. Computed on-the-fly, not stored

### Claude's Discretion
- Exact stockstats API calls and any wrapper needed for missing indicators
- Turbulence Index implementation details (Mahalanobis distance, covariance window)
- Observation assembler class design and interface
- Feature pipeline module organization (per-indicator vs grouped)
- Exact DuckDB DDL column types and constraints beyond what is specified
- CLI argument design for compute_features.py
- Test fixture data for indicator validation
- Warmup period handling (how many bars before NaN-free output)

### Deferred Ideas (OUT OF SCOPE)
- FinBERT sentiment pipeline (HARD-03) -- Phase 10
- Alpha Vantage Earnings Call Transcript Sentiment -- post-Milestone 7
- Options-derived features (IV rank, put/call OI, VRP, GEX, skew) -- Phase 3 / OPT-05
- pandas_ta re-evaluation -- Phase 6
- training_datasets, model_metadata, ensemble_weights, backtest_results DuckDB tables -- Phase 7
- On-chain crypto metrics -- v2+ (out of scope per PROJECT.md)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FEAT-01 | Technical indicators via stockstats: SMA(50,200) ratios, RSI(14), MACD line+histogram, BB position, ATR(14)%, Volume/SMA(20), ADX-14 | All 9 indicators verified working in stockstats. BB position and ATR% require manual derivation from raw stockstats output |
| FEAT-02 | Derived features: log returns (1d, 5d, 20d), Bollinger Band width | Intermediate features for HMM input and correlation analysis. Computed via pandas/numpy, not stored in obs vector |
| FEAT-03 | Fundamental features (equities): P/E z-score, earnings growth, D/E, dividend yield | yfinance 1.2.0 `Ticker.info` provides these. Alpha Vantage `alpha-vantage` pip package needed as fallback dep |
| FEAT-04 | Macro regime features: VIX z-score, yield curve, Fed Funds, CPI, unemployment direction | DuckDB ASOF JOIN verified working. macro_features table from Phase 4 is the source |
| FEAT-05 | HMM regime detection: 2-state Gaussian per environment | hmmlearn 0.3.3 GaussianHMM verified. Warm-start requires ridge regularization on covars (eigenvalue=0 issue found) |
| FEAT-06 | Rolling z-score normalization (252 equity / 360 crypto) | Standard pandas rolling().mean()/std() computation. Bounds smoke test [-3, 3] |
| FEAT-07 | Observation space assembly: 156 equity / 45 crypto | NumPy array concatenation in deterministic order. Class design is Claude's discretion |
| FEAT-08 | Feature addition protocol: Sharpe >= 0.05 threshold | Phase 5 builds infrastructure only. Comparison framework + documentation |
| FEAT-09 | Correlation pruning: Pearson r > 0.85 | pandas DataFrame.corr() + domain-driven rules. One-time pre-training analysis |
| FEAT-10 | Weekly-derived features (SMA trend, weekly RSI) | DuckDB ohlcv_weekly view exists from Phase 4. stockstats computes RSI on resampled data |
| FEAT-11 | Per-environment feature tables in DuckDB | DDL for features_equity (DATE key), features_crypto (TIMESTAMP key), fundamentals, hmm_state_history |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| stockstats | 0.6.8 (installed) | Technical indicators (SMA, RSI, MACD, BB, ATR, ADX) | FinRL's native TA library; Python 3.11 compatible; already installed |
| hmmlearn | 0.3.3 (installed) | 2-state Gaussian HMM regime detection | Standard HMM implementation; already installed |
| yfinance | 1.2.0 (installed) | Primary fundamental data source (P/E, earnings, D/E, dividends) | Already installed for cross-source validation (Phase 4) |
| numpy | <2.0 (installed) | Mahalanobis distance, covariance, array assembly | Core numerical library |
| pandas | <3.0 (installed) | Rolling statistics, resampling, DataFrame operations | Core data library |
| duckdb | >=1.0 (installed) | Feature table storage, ASOF JOIN for macro alignment | Established in Phase 4 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| alpha-vantage | latest | Fundamental data fallback (25 req/day free tier) | When yfinance fails or is rate-limited |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| stockstats | pandas_ta | pandas_ta requires Python >= 3.12; deferred to Phase 6 |
| stockstats | TA-Lib (C library) | Better performance but requires C compilation; stockstats covers all needed indicators |
| yfinance | Alpha Vantage only | AV free tier limited to 25 req/day; yfinance is unlimited but less reliable |

**Installation:**
```bash
uv add alpha-vantage
```

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/features/
    __init__.py
    technical.py       # TechnicalIndicatorCalculator: stockstats wrapper for all 9 price-action indicators
    fundamentals.py    # FundamentalFetcher: yfinance + Alpha Vantage fallback
    hmm_regime.py      # HMMRegimeDetector: 2-state Gaussian HMM with warm-start
    macro.py           # MacroFeatureAligner: ASOF JOIN-based forward-fill from DuckDB
    normalization.py   # RollingZScoreNormalizer: per-environment windowed z-scores
    correlation.py     # CorrelationPruner: pairwise Pearson + domain rules
    turbulence.py      # TurbulenceCalculator: Mahalanobis distance (on-the-fly)
    assembler.py       # ObservationAssembler: concatenates all feature groups into flat array
    pipeline.py        # FeaturePipeline: orchestrates compute -> normalize -> store
    schema.py          # DuckDB DDL for Phase 5 tables (features_equity, features_crypto, etc.)
scripts/
    compute_features.py  # CLI entry point for manual/initial feature computation
tests/
    features/
        __init__.py
        test_technical.py
        test_fundamentals.py
        test_hmm_regime.py
        test_macro.py
        test_normalization.py
        test_correlation.py
        test_turbulence.py
        test_assembler.py
        test_pipeline.py
```

### Pattern 1: StockDataFrame Wrapper
**What:** stockstats uses a `StockDataFrame.retype()` pattern that mutates a DataFrame in-place. Wrap it to provide clean indicator extraction.
**When to use:** Every technical indicator computation.
**Example:**
```python
# Source: Verified via local stockstats 0.6.8 testing
import stockstats
import pandas as pd

def compute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all 9 price-action indicators from OHLCV data."""
    sdf = stockstats.StockDataFrame.retype(ohlcv.copy())

    result = pd.DataFrame(index=ohlcv.index)
    result["price_sma50_ratio"] = ohlcv["close"] / sdf["close_50_sma"]
    result["price_sma200_ratio"] = ohlcv["close"] / sdf["close_200_sma"]
    result["rsi_14"] = sdf["rsi_14"]
    result["macd_line"] = sdf["macd"]
    result["macd_histogram"] = sdf["macdh"]
    result["bb_position"] = (ohlcv["close"] - sdf["boll_lb"]) / (sdf["boll_ub"] - sdf["boll_lb"])
    result["atr_14_pct"] = sdf["atr_14"] / ohlcv["close"]
    result["volume_sma20_ratio"] = ohlcv["volume"] / sdf["volume_20_sma"]
    result["adx_14"] = sdf["adx"]
    return result
```

### Pattern 2: HMM Warm-Start with Ridge Regularization
**What:** hmmlearn's covariance setter raises ValueError for near-singular matrices. Apply ridge regularization before warm-starting.
**When to use:** Every HMM refit after the initial fit.
**Example:**
```python
# Source: Verified via local hmmlearn 0.3.3 testing
import numpy as np
from hmmlearn.hmm import GaussianHMM

def warm_start_hmm(
    prev_model: GaussianHMM,
    data: np.ndarray,
    n_iter: int = 200,
    ridge: float = 1e-6,
) -> GaussianHMM:
    """Refit HMM with warm-start from previous parameters."""
    model = GaussianHMM(
        n_components=2,
        covariance_type=prev_model.covariance_type,
        n_iter=n_iter,
        init_params="",  # skip random init
    )
    model.startprob_ = prev_model.startprob_.copy()
    model.transmat_ = prev_model.transmat_.copy()
    model.means_ = prev_model.means_.copy()

    # Ridge regularization for numerical stability
    covars = prev_model.covars_.copy()
    n_features = covars.shape[-1]
    if prev_model.covariance_type == "full":
        for i in range(covars.shape[0]):
            covars[i] = (covars[i] + covars[i].T) / 2  # enforce symmetry
            covars[i] += np.eye(n_features) * ridge
    elif prev_model.covariance_type == "diag":
        covars = np.maximum(covars, ridge)
    model.covars_ = covars

    model.fit(data)
    return model
```

### Pattern 3: ASOF JOIN for Look-Ahead-Free Macro Alignment
**What:** DuckDB's ASOF JOIN ensures macro features are forward-filled using only data available at the time of each bar.
**When to use:** Aligning macro_features to ohlcv_daily or ohlcv_4h.
**Example:**
```python
# Source: Verified via local DuckDB testing
MACRO_ASOF_QUERY = """
    SELECT
        p.symbol,
        p.date,
        vix.value AS vix_value,
        t10y2y.value AS yield_curve_spread,
        dff.value AS fed_funds_rate,
        cpi.value AS cpi_yoy,
        unemp.value AS unemployment_rate
    FROM ohlcv_daily p
    ASOF JOIN (SELECT * FROM macro_features WHERE series_id = 'VIXCLS')
        AS vix ON p.date >= vix.release_date
    ASOF JOIN (SELECT * FROM macro_features WHERE series_id = 'T10Y2Y')
        AS t10y2y ON p.date >= t10y2y.release_date
    ASOF JOIN (SELECT * FROM macro_features WHERE series_id = 'DFF')
        AS dff ON p.date >= dff.release_date
    ASOF JOIN (SELECT * FROM macro_features WHERE series_id = 'CPIAUCSL')
        AS cpi ON p.date >= cpi.release_date
    ASOF JOIN (SELECT * FROM macro_features WHERE series_id = 'UNRATE')
        AS unemp ON p.date >= unemp.release_date
    WHERE p.symbol = ?
    ORDER BY p.date
"""
```

### Pattern 4: Observation Vector Assembly
**What:** Concatenate feature groups in deterministic order into a flat NumPy array.
**When to use:** At every decision step (training and inference).
**Example:**
```python
# Source: Designed per CONTEXT.md assembly order specification
import numpy as np

def assemble_equity_observation(
    per_asset_features: dict[str, np.ndarray],  # symbol -> (15,) array
    macro_features: np.ndarray,                  # (6,) array
    hmm_probs: np.ndarray,                       # (2,) array [P(bull), P(bear)]
    turbulence: float,                           # scalar
    portfolio_state: np.ndarray,                 # (27,) array
    symbols: list[str],                          # alpha-sorted
) -> np.ndarray:
    """Assemble 156-dim equity observation vector."""
    parts = []
    for sym in sorted(symbols):
        parts.append(per_asset_features[sym])  # 15 dims each
    parts.append(macro_features)    # 6 dims
    parts.append(hmm_probs)         # 2 dims
    parts.append(np.array([turbulence]))  # 1 dim
    parts.append(portfolio_state)   # 27 dims
    obs = np.concatenate(parts)
    assert obs.shape == (156,), f"Expected (156,), got {obs.shape}"
    return obs
```

### Anti-Patterns to Avoid
- **Look-ahead bias in macro features:** Never use the raw `date` column from macro_features for joining. Always use `release_date` via ASOF JOIN. The `date` column is the observation date; `release_date` is when the data was actually published.
- **Fitting HMM on normalized features:** HMM should be fit on raw log-returns + realized volatility, NOT z-scored features. Z-scoring distorts the Gaussian emission distributions the HMM learns.
- **Storing turbulence in DuckDB:** Turbulence index is computed on-the-fly from the full return history at each step. Storing it would create a mismatch between training (expanding window) and inference (current state).
- **Using stockstats SMA directly as features:** The spec requires Price/SMA ratios (dimensionless), not raw SMA values. Raw SMAs scale with price level and would dominate other features.
- **Mutating the original DataFrame with stockstats:** `StockDataFrame.retype()` modifies the DataFrame in-place. Always pass `.copy()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Technical indicators | Custom SMA/RSI/MACD/ATR/ADX | stockstats 0.6.8 | Edge cases in EMA vs SMA, Wilder's smoothing for RSI/ATR, ADX directional movement logic |
| HMM fitting | Custom EM algorithm | hmmlearn GaussianHMM | Numerical stability, convergence monitoring, Viterbi decoding, forward-backward algorithm |
| Fundamental data | Custom Yahoo/AV scraping | yfinance Ticker.info + alpha-vantage FundamentalData | Rate limiting, authentication, data normalization, error handling |
| Covariance pseudo-inverse | Custom matrix inversion | np.linalg.pinv | Handles singular/near-singular matrices automatically (SVD-based) |
| Rolling statistics | Manual loop | pandas.DataFrame.rolling() | Vectorized, handles NaN, supports expanding windows |
| ASOF JOIN | Custom forward-fill logic | DuckDB ASOF JOIN | SQL-native, handles multiple series, prevents look-ahead bias |
| Bollinger Band position | Manual (close-lb)/(ub-lb) with zero-division | stockstats boll_ub/boll_lb + explicit handling | stockstats handles the EMA/SMA band computation; zero-width bands need explicit NaN |

**Key insight:** Financial indicator computation has decades of subtle conventions (Wilder's smoothing vs EMA, centered vs trailing windows, volume-weighted variants). Hand-rolling introduces silent bugs that only manifest as degraded model performance, making them nearly impossible to debug.

## Common Pitfalls

### Pitfall 1: HMM State Label Flipping
**What goes wrong:** HMM states are arbitrary (state 0 could be bull or bear). Between refits, labels can flip, causing regime signals to invert.
**Why it happens:** EM algorithm converges to equivalent solutions with permuted states.
**How to avoid:** After each fit, sort states by mean return (higher mean = bull). Warm-start from previous parameters with `init_params=""`.
**Warning signs:** P(bull) suddenly jumps from ~0.8 to ~0.2 without market change.

### Pitfall 2: Near-Singular Covariance in HMM
**What goes wrong:** hmmlearn raises `ValueError: covars must be symmetric, positive-definite` when warm-starting.
**Why it happens:** One HMM state can collapse to very low variance (eigenvalue = 0), especially with 2-feature input (log-returns + realized vol).
**How to avoid:** Add ridge regularization (`+ np.eye(n) * 1e-6`) to covariance matrices before setting as warm-start parameters. Verified this issue occurs with hmmlearn 0.3.3.
**Warning signs:** ValueError during HMM refit. Monitor condition number of covariance matrices.

### Pitfall 3: stockstats Column Name Mutation
**What goes wrong:** `StockDataFrame.retype()` adds computed columns to the DataFrame in-place and may rename existing columns.
**Why it happens:** stockstats uses lazy evaluation -- accessing `sdf["rsi_14"]` computes and stores the result as a new column on the DataFrame.
**How to avoid:** Always pass `.copy()` to `retype()`. Extract needed values immediately into a separate result DataFrame.
**Warning signs:** Unexpected columns in DataFrames downstream of indicator computation.

### Pitfall 4: Bollinger Band Position Outside [0, 1]
**What goes wrong:** BB position = (close - lower) / (upper - lower) can exceed [0, 1] when price breaks above/below bands.
**Why it happens:** Bollinger bands are statistical bounds, not hard limits. Verified: stockstats produces values from -0.20 to 1.34.
**How to avoid:** This is expected and correct -- do NOT clip to [0, 1]. The >1 and <0 values carry signal (breakout). Document this clearly.
**Warning signs:** None -- this is correct behavior.

### Pitfall 5: ETF Fundamental Data Availability
**What goes wrong:** yfinance returns different/missing fields for ETFs vs individual stocks. ETFs like SPY report fund-level P/E, not individual holdings.
**Why it happens:** ETFs are funds, not companies. Fundamental metrics are aggregated differently.
**How to avoid:** Use `Ticker.info` dict fields: `trailingPE`, `earningsQuarterlyGrowth`, `debtToEquity`, `dividendYield`. For ETFs that lack these, use the fallback to Alpha Vantage or mark as N/A per CONTEXT.md.
**Warning signs:** KeyError or None values in yfinance .info dict for ETF-specific tickers.

### Pitfall 6: Z-Score Division by Zero
**What goes wrong:** Rolling z-score = (x - mean) / std. When std = 0 (flat prices during market holidays or data gaps), produces NaN or inf.
**Why it happens:** 252/360-bar windows can contain stretches of identical values (e.g., overnight periods for crypto copied from close).
**How to avoid:** Replace zero std with a small epsilon or NaN, then forward-fill. Test with synthetic flat-price scenarios.
**Warning signs:** NaN or inf values in normalized features post-warmup.

### Pitfall 7: Crypto Multi-Timeframe Column Name Mismatch
**What goes wrong:** Doc 14 crypto multi-timeframe DDL column names may not match Doc 08 feature spec.
**Why it happens:** Documented in CONTEXT.md as a known issue discovered during design review.
**How to avoid:** Use the observation vector spec from CONTEXT.md as authoritative. Adjust DDL column names to match during implementation.
**Warning signs:** Column name mismatches between feature computation and DuckDB storage.

## Code Examples

### Computing All Technical Indicators for One Symbol
```python
# Source: Verified via local stockstats testing
import stockstats
import numpy as np
import pandas as pd

def compute_price_action_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 price-action features from OHLCV bars.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume].

    Returns:
        DataFrame with 9 feature columns, same index as input.
    """
    sdf = stockstats.StockDataFrame.retype(ohlcv.copy())

    result = pd.DataFrame(index=ohlcv.index)

    # 1. Price/SMA ratios (dimensionless)
    result["price_sma50_ratio"] = ohlcv["close"] / sdf["close_50_sma"]
    result["price_sma200_ratio"] = ohlcv["close"] / sdf["close_200_sma"]

    # 2. RSI-14 (0-100)
    result["rsi_14"] = sdf["rsi_14"]

    # 3. MACD (raw values, will be z-scored later)
    result["macd_line"] = sdf["macd"]
    result["macd_histogram"] = sdf["macdh"]

    # 4. Bollinger Band position: (close - lb) / (ub - lb)
    band_width = sdf["boll_ub"] - sdf["boll_lb"]
    result["bb_position"] = np.where(
        band_width > 0,
        (ohlcv["close"] - sdf["boll_lb"]) / band_width,
        0.5,  # default when bands collapse
    )

    # 5. ATR-14 as percentage of price
    result["atr_14_pct"] = sdf["atr_14"] / ohlcv["close"]

    # 6. Volume/SMA-20 ratio
    vol_sma = sdf["volume_20_sma"]
    result["volume_sma20_ratio"] = np.where(
        vol_sma > 0,
        ohlcv["volume"] / vol_sma,
        1.0,  # default when SMA is zero
    )

    # 7. ADX-14
    result["adx_14"] = sdf["adx"]

    return result
```

### Computing Derived Features for HMM Input
```python
# Source: Standard financial computation
import numpy as np
import pandas as pd

def compute_hmm_inputs(close: pd.Series) -> pd.DataFrame:
    """Compute log-returns + 20-day realized volatility for HMM fitting.

    Args:
        close: Close price series.

    Returns:
        DataFrame with columns [log_return, realized_vol_20d], NaN-free rows only.
    """
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(20).std()
    result = pd.DataFrame({
        "log_return": log_ret,
        "realized_vol_20d": realized_vol,
    })
    return result.dropna()
```

### Rolling Z-Score Normalization
```python
# Source: Standard normalization pattern
import pandas as pd

def rolling_zscore(
    features: pd.DataFrame,
    window: int,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Apply rolling z-score normalization to all feature columns.

    Args:
        features: Feature DataFrame (each column is a feature).
        window: Rolling window size (252 equity, 360 crypto).
        epsilon: Floor for standard deviation to prevent div-by-zero.

    Returns:
        Z-scored DataFrame with same shape and index.
    """
    rolling_mean = features.rolling(window).mean()
    rolling_std = features.rolling(window).std().clip(lower=epsilon)
    return (features - rolling_mean) / rolling_std
```

### Turbulence Index Calculation
```python
# Source: FinRL-inspired Mahalanobis distance approach
import numpy as np

def compute_turbulence(
    current_returns: np.ndarray,
    historical_returns: np.ndarray,
) -> float:
    """Compute turbulence index via Mahalanobis distance.

    Args:
        current_returns: (n_assets,) array of current period returns.
        historical_returns: (n_periods, n_assets) array of historical returns.

    Returns:
        Turbulence score (scalar).
    """
    mean_returns = historical_returns.mean(axis=0)
    cov_matrix = np.cov(historical_returns.T)
    diff = current_returns - mean_returns
    inv_cov = np.linalg.pinv(cov_matrix)  # pseudo-inverse for numerical stability
    turbulence = float(np.sqrt(diff @ inv_cov @ diff))
    return turbulence
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pandas_ta for indicators | stockstats (Phase 1 decision) | 2026-03-04 | pandas_ta requires Python >= 3.12; stockstats is Python 3.11 compatible |
| Store all features including turbulence | Turbulence on-the-fly, stored features in DuckDB | Design phase | Prevents train/serve skew for expanding-window metrics |
| Raw indicator values | Ratio-based features (Price/SMA, ATR/Price) | Design phase | Scale-invariant features work better with RL normalization |
| Global normalization | Rolling z-score + VecNormalize (Phase 7) | Design phase | Two-stage normalization prevents look-ahead bias |

**Deprecated/outdated:**
- pandas_ta: Cannot be used until Python 3.12+ migration (Phase 6 re-evaluation)
- Raw VIX values: Use VIX z-score (1-year rolling) instead

## Open Questions

1. **yfinance ETF fundamental data completeness**
   - What we know: yfinance `Ticker.info` provides `trailingPE`, `earningsQuarterlyGrowth`, `debtToEquity`, `dividendYield` for stocks. ETFs may have different field availability.
   - What is unclear: Exactly which of the 8 ETFs (SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK) return all 4 fundamental metrics via yfinance. Some sector ETFs may lack certain fields.
   - Recommendation: During implementation, test all 8 ETFs and document which fields are available. Use the CONTEXT.md fallback plan (reduce to P/E + earnings growth only if full set causes issues).

2. **alpha-vantage package API for ETF fundamentals**
   - What we know: The `alpha-vantage` pip package provides `FundamentalData.get_company_overview()`. Alpha Vantage free tier allows 25 requests/day.
   - What is unclear: Whether Alpha Vantage's company overview endpoint returns meaningful data for ETFs (vs individual stocks).
   - Recommendation: Add `alpha-vantage` as dependency. During implementation, test with one ETF ticker. If AV also lacks ETF fundamentals, consider using the underlying holdings' aggregated fundamentals from yfinance.

3. **DuckDB ASOF JOIN with multiple series**
   - What we know: Single-series ASOF JOIN works (verified). The macro alignment query needs 5 separate ASOF JOINs (one per FRED series).
   - What is unclear: Performance of chained ASOF JOINs with large datasets. May need to pivot macro_features first.
   - Recommendation: Test with realistic data volume. If chained ASOF JOINs are slow, pre-pivot the macro data into wide format.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/features/ -x -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-01 | 9 technical indicators computed correctly | unit | `uv run pytest tests/features/test_technical.py -x` | Wave 0 |
| FEAT-02 | Derived features (log returns, BB width) computed | unit | `uv run pytest tests/features/test_technical.py::test_derived_features -x` | Wave 0 |
| FEAT-03 | Fundamental data fetched and validated | unit | `uv run pytest tests/features/test_fundamentals.py -x` | Wave 0 |
| FEAT-04 | Macro features aligned via ASOF JOIN | unit+integration | `uv run pytest tests/features/test_macro.py -x` | Wave 0 |
| FEAT-05 | HMM regime produces P(bull)+P(bear)=1.0 | unit | `uv run pytest tests/features/test_hmm_regime.py -x` | Wave 0 |
| FEAT-06 | Rolling z-score correct with per-env windows | unit | `uv run pytest tests/features/test_normalization.py -x` | Wave 0 |
| FEAT-07 | Observation vector shape (156, equity) and (45, crypto) | unit | `uv run pytest tests/features/test_assembler.py -x` | Wave 0 |
| FEAT-08 | Feature A/B comparison infrastructure | unit | `uv run pytest tests/features/test_pipeline.py::test_feature_ab -x` | Wave 0 |
| FEAT-09 | Correlation pruning flags r > 0.85 pairs | unit | `uv run pytest tests/features/test_correlation.py -x` | Wave 0 |
| FEAT-10 | Weekly features from aggregated bars | unit | `uv run pytest tests/features/test_technical.py::test_weekly_features -x` | Wave 0 |
| FEAT-11 | DuckDB feature tables created and populated | integration | `uv run pytest tests/features/test_pipeline.py::test_duckdb_storage -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/features/ -x -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/features/__init__.py` -- package init
- [ ] `tests/features/test_technical.py` -- covers FEAT-01, FEAT-02, FEAT-10
- [ ] `tests/features/test_fundamentals.py` -- covers FEAT-03
- [ ] `tests/features/test_hmm_regime.py` -- covers FEAT-05
- [ ] `tests/features/test_macro.py` -- covers FEAT-04
- [ ] `tests/features/test_normalization.py` -- covers FEAT-06
- [ ] `tests/features/test_correlation.py` -- covers FEAT-09
- [ ] `tests/features/test_turbulence.py` -- covers turbulence calculator
- [ ] `tests/features/test_assembler.py` -- covers FEAT-07
- [ ] `tests/features/test_pipeline.py` -- covers FEAT-08, FEAT-11
- [ ] Config schema additions: `FeaturesConfig` with normalization windows, HMM params, correlation threshold
- [ ] Dependency: `uv add alpha-vantage`

## Sources

### Primary (HIGH confidence)
- stockstats 0.6.8: All 9 indicators verified locally -- SMA, RSI, MACD (line + histogram), Bollinger bands (upper/lower/middle), ATR, ADX, Volume SMA. Column naming: `close_50_sma`, `rsi_14`, `macd`, `macdh`, `boll_ub`, `boll_lb`, `atr_14`, `adx`, `volume_20_sma`
- hmmlearn 0.3.3: GaussianHMM verified locally. `predict_proba()` returns (n_samples, n_components) with rows summing to 1.0. Warm-start via `init_params=""` + setting `startprob_`, `transmat_`, `means_`, `covars_`. Ridge regularization needed for near-singular covariance (eigenvalue=0 found in testing)
- DuckDB ASOF JOIN: Verified locally with macro_features forward-fill pattern
- numpy `linalg.pinv`: Verified for Mahalanobis distance calculation

### Secondary (MEDIUM confidence)
- [yfinance PyPI](https://pypi.org/project/yfinance/) - v1.2.0, `Ticker.info` for fundamental data
- [alpha-vantage PyPI](https://pypi.org/project/alpha-vantage/) - `FundamentalData.get_company_overview()` for fallback
- [Alpha Vantage API docs](https://www.alphavantage.co/documentation/) - free tier 25 req/day

### Tertiary (LOW confidence)
- ETF-specific fundamental data availability via yfinance -- needs runtime validation during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified locally, versions confirmed
- Architecture: HIGH - follows established project patterns (DatabaseManager, BaseIngestor, structlog)
- Pitfalls: HIGH - key issues (HMM covariance, stockstats mutation, BB position range) verified through testing
- Fundamentals data: MEDIUM - yfinance API exists but ETF-specific field availability unverified
- Macro ASOF JOIN chaining: MEDIUM - single JOIN verified, multi-series performance untested

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable libraries, no fast-moving dependencies)
