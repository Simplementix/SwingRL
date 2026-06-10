# Feature Catalog

Every feature that enters the RL observation vector, plus the modules that produce them. Source-of-truth for data origin, computation, normalization, and env consumption. Update when any referenced module changes.

**Last verified against code:** 2026-04-15

## Feature modules at a glance

| Module | Role | Runtime / Offline | Consumed by |
|--------|------|-------------------|-------------|
| `features/technical.py` | Price-action indicators via stockstats | Runtime | Equity + Crypto |
| `features/fundamentals.py` | P/E, earnings, D/E, dividend yield | Runtime (fetched offline, stored in pg16) | Equity only |
| `features/macro.py` | FRED macro series → 6 derived features | Runtime (pg16 LATERAL JOIN) | Equity + Crypto |
| `features/hmm_regime.py` | 3-state Gaussian HMM regime probabilities | Runtime | Equity + Crypto |
| `features/turbulence.py` | Market turbulence index (env-specific algo) | Runtime on-the-fly | Equity + Crypto |
| `features/normalization.py` | Rolling z-score over features (pre-assembly) | Runtime | Equity + Crypto |
| `features/correlation.py` | Correlation pruning (dev analysis tool) | **Offline only** | Pipeline dev |
| `features/health.py` | Consecutive-failure tracker → block trading | Runtime (live only) | Live pipeline |
| `features/assembler.py` | Flat obs vector concatenation + NaN check | Runtime | Both envs |
| `features/pipeline.py` | End-to-end compute → normalize → store flow | Runtime orchestrator | Both envs |

## Equity observation features (164 dims, default)

### Per-asset technical — 15 features × 8 symbols, alpha-sorted

| # | Feature | Source / computation | Raw range | Normalization | File |
|---|---------|---------------------|-----------|---------------|------|
| 1 | `price_sma50_ratio` | `close / close_50_sma` (stockstats) | ~[0.5, 1.5] | rolling z-score | `technical.py::compute_price_action` |
| 2 | `price_sma200_ratio` | `close / close_200_sma` | ~[0.5, 1.5] | rolling z-score | same |
| 3 | `rsi_14` | stockstats `rsi_14` | [0, 100] | rolling z-score | same |
| 4 | `macd_line` | stockstats `macd` | real | rolling z-score | same |
| 5 | `macd_histogram` | stockstats `macdh` | real | rolling z-score | same |
| 6 | `bb_position` | `(close − lb) / (ub − lb)`; 0.5 if bands collapse | mostly [0, 1], unbounded on breakout | rolling z-score | same |
| 7 | `atr_14_pct` | `atr_14 / close` | (0, 1] | rolling z-score | same |
| 8 | `volume_sma20_ratio` | `volume / vol_sma_20`; 1.0 if SMA=0 | (0, ∞) | rolling z-score | same |
| 9 | `adx_14` | stockstats `adx` | [0, 100] | rolling z-score | same |
| 10 | `weekly_trend_dir` | `close > weekly_sma_10` resampled W-FRI, ffill | {0, 1} | rolling z-score | `technical.py::compute_weekly_features` |
| 11 | `weekly_rsi_14` | Weekly-resampled RSI-14 | [0, 100] | rolling z-score | same |
| 12 | `pe_zscore` | Sector-relative z-score of trailing P/E | real | (already z-scored) | `fundamentals.py::sector_relative_zscore` |
| 13 | `earnings_growth` | yfinance `earningsQuarterlyGrowth` | real | rolling z-score | `fundamentals.py::_fetch_from_yfinance` |
| 14 | `debt_to_equity` | yfinance `debtToEquity` (≥0 or NaN) | ≥0 | rolling z-score | same |
| 15 | `dividend_yield` | yfinance `dividendYield` | ≥0 | rolling z-score | same |

**Sentiment (optional, +2 per asset = 17/asset, 180 total):**
`sentiment_score` + `sentiment_confidence` — only when `config.sentiment.enabled=true`. Source: `SentimentConfig.model_name` (default `ProsusAI/finbert`) + Finnhub headlines.

### Shared blocks (equity)

| Block | # | Feature | Source / computation | Range |
|-------|---|---------|---------------------|-------|
| Macro | 1 | `macro_vix_zscore` | VIX (FRED `VIXCLS`), 252-day rolling z-score | real |
| Macro | 2 | `macro_yield_curve_spread` | FRED `T10Y2Y` raw value | real |
| Macro | 3 | `macro_yield_curve_direction` | `1 if spread > 0 else 0` | {0, 1} |
| Macro | 4 | `macro_fed_funds_90d_change` | `DFF − DFF.shift(90)` (business days) | real |
| Macro | 5 | `macro_cpi_yoy` | `CPI / CPI.shift(252) − 1` | real |
| Macro | 6 | `macro_unemployment_3m_direction` | `1 if UNRATE < UNRATE.shift(63) else 0` | {0, 1} |
| HMM | 1 | `hmm_p_bull` | Gaussian HMM P(state 0) — highest mean return | [0, 1] |
| HMM | 2 | `hmm_p_bear` | Gaussian HMM P(state 1) — mid mean return | [0, 1] |
| Turbulence | 1 | `turbulence_index` | EWMA Mahalanobis distance (half-life 126 days) | ≥ 0 |

See **Known issues** for the missing P(crisis) probability.

### Portfolio state (35 dims, live, not stored)

| Index | Name | Range | Source |
|-------|------|-------|--------|
| 0 | `portfolio_cash_ratio` | [0, 1] | `BaseTradingEnv._get_portfolio_state` |
| 1 | `portfolio_exposure` | [0, 1] | same |
| 2 | `portfolio_daily_return` | real | same |
| 3 + 4i | `portfolio_{symbol}_weight` | [0, 1] | same (per asset, alpha-sorted) |
| 4 + 4i | `portfolio_{symbol}_weight_deviation` | real | same |
| 5 + 4i | `portfolio_{symbol}_unrealized_pnl_pct` | real | same |
| 6 + 4i | `portfolio_{symbol}_bars_since_trade` | ≥ 0 | same |

## Crypto observation features (47 dims)

### Per-asset technical — 13 features × 2 symbols

Same 9 price-action features as equity (rows 1–9 above), plus **4 multi-timeframe** (replace equity's weekly + fundamentals):

| # | Feature | Source / computation | Raw range | Normalization | File |
|---|---------|---------------------|-----------|---------------|------|
| 10 | `daily_trend_dir` | 4H→daily resample, `close > daily_sma_10` | {0, 1} | rolling z-score | `technical.py::compute_crypto_multi_timeframe` |
| 11 | `daily_rsi_14` | Daily-aggregated RSI-14, ffill to 4H | [0, 100] | rolling z-score | same |
| 12 | `four_h_rsi_14` | RSI-14 computed on 4H bars directly | [0, 100] | rolling z-score | same |
| 13 | `four_h_price_sma20_ratio` | `close / close_20_sma` on 4H | ~[0.5, 1.5] | rolling z-score | same |

### Shared blocks (crypto)

- Macro: same 6 features as equity; lag periods scaled for 4H bars (6× daily) — see `macro.py::_compute_derived_macro_crypto`.
- HMM: `hmm_p_bull`, `hmm_p_bear`. Proxy = BTCUSDT (`config.crypto.hmm_proxy_symbol`). Covariance `diag`, window 2000 4H-bars, refit daily.
- Turbulence: **different algorithm** — vol z-score × correlation spike composite; 360-bar warmup, 1080-bar rolling window. Robust with limited history.
- **Overnight context** (crypto-only, 1 dim): `overnight_hours_since_equity_close` — hours since last equity market close.

### Portfolio state (11 dims)

Same layout as equity: 3 fixed + 4 × 2 assets (BTCUSDT, ETHUSDT alpha-sorted).

## Configurable values (yaml) — `features.*`

| Key | Default | Notes |
|-----|---------|-------|
| `equity_hmm_window` | 1260 | ≥ 100. Bars used to fit HMM |
| `crypto_hmm_window` | 2000 | ≥ 100 (4H bars) |
| `hmm_n_iter` | 200 | EM iterations |
| `hmm_n_inits` | 5 | Multi-start attempts for initial fit |
| `hmm_ridge` | 1e-6 | Covariance regularization |
| `equity_zscore_window` | 252 | ≥ 50 |
| `crypto_zscore_window` | 360 | ≥ 50 |
| `correlation_threshold` | 0.85 | For offline pruner (0, 1] |
| `zscore_epsilon` | 1e-8 | Std floor to prevent divide-by-zero |
| `equity_turbulence_warmup` | 252 | ≥ 50 |
| `equity_turbulence_half_life` | 126 | ≥ 10 (EWMA decay) |
| `crypto_turbulence_window` | 1080 | ≥ 100 |
| `crypto_turbulence_warmup` | 360 | ≥ 50 |

**HMM proxy symbols:** `equity.hmm_proxy_symbol` (default `SPY`), `crypto.hmm_proxy_symbol` (default `BTCUSDT`) — validated to be in respective `symbols` list.

**Sentiment:** `sentiment.enabled` (default `False`), `sentiment.model_name`, `sentiment.max_headlines_per_asset`, `sentiment.finnhub_api_key`.

## Hardcoded values (not yaml-tunable)

### Technical indicator parameters

| Value | Where | Current |
|-------|-------|---------|
| Price-action indicator set | `technical.py::compute_price_action` | 9 fixed indicators (SMA-50/200, RSI-14, MACD, BB, ATR-14, vol-SMA-20, ADX) |
| Weekly resample rule | `technical.py::compute_weekly_features` | `W-FRI` (Friday close) |
| Weekly SMA window | same | 10 |
| Weekly RSI window | same | 14 |
| Daily-from-4H SMA window | `technical.py::compute_crypto_multi_timeframe` | 10 |
| 4H SMA window | same | 20 |
| BB default on collapse | `technical.py::compute_price_action` | 0.5 |
| Volume ratio default on zero SMA | same | 1.0 |

### Macro feature wiring

| Value | Where | Current |
|-------|-------|---------|
| FRED series IDs | `macro.py` module constants | `VIXCLS, T10Y2Y, DFF, CPIAUCSL, UNRATE` |
| VIX z-score window | `macro.py::_VIX_ZSCORE_WINDOW` | 252 (business days) |
| VIX std floor | `macro.py::compute_derived_macro` | `1e-8` |
| CPI YoY lag (equity) | `macro.py::compute_derived_macro` | 252 rows |
| Unemployment lag (equity) | same | 63 rows |
| Fed funds lag (equity) | same | 90 rows |
| Crypto-scaled lags | `macro.py::_compute_derived_macro_crypto` | 540 / 1512 / 396 / 1512 rows |

### HMM regime

| Value | Where | Current |
|-------|-------|---------|
| Number of states | `hmm_regime.py::HMMRegimeDetector.initial_fit` | 3 (bull / bear / crisis) |
| HMM input features | `hmm_regime.py::compute_hmm_inputs` | `[log_return_1d, realized_vol_20d]` |
| Realized vol window | same | 20 |
| Equity covariance type | `hmm_regime.py::__init__` | `full` |
| Crypto covariance type | same | `diag` |
| Label ordering | `hmm_regime.py::_ensure_label_order` | sort states by mean return desc (bull → bear → crisis) |
| Cold-start priors | `hmm_regime.py::cold_start_fit` | startprob `[0.6, 0.3, 0.1]`, transmat diag 0.90 |
| **Obs vector HMM carries** | `assembler.py::_HMM_FEATURE_NAMES` | only 2 of 3 states — `p_bull`, `p_bear` |

### Turbulence

| Value | Where | Current |
|-------|-------|---------|
| Equity algo | `turbulence.py::EquityTurbulenceCalculator` | EWMA Mahalanobis, pseudo-inverse for stability |
| Crypto algo | `turbulence.py::CryptoTurbulenceCalculator` | vol-z × `(1 + corr_spike)` composite |
| Crypto vol window | `CryptoTurbulenceCalculator.VOL_WINDOW` | 30 |
| Corr-spike scale | `CryptoTurbulenceCalculator._corr_spike` | × 10.0 |
| Not stored in pg16 | `turbulence.py` module docstring | computed on-the-fly per bar |

### Fundamentals

| Value | Where | Current |
|-------|-------|---------|
| Primary source | `fundamentals.py::_fetch_from_yfinance` | yfinance Ticker.info |
| Fallback source | `fundamentals.py::_fetch_from_alpha_vantage` | Alpha Vantage (if `ALPHA_VANTAGE_API_KEY` env var set) |
| yfinance field map | `fundamentals.py::_YF_FIELD_MAP` | trailingPE, earningsQuarterlyGrowth, debtToEquity, dividendYield |
| AV rate limit | `fundamentals.py::_fetch_from_alpha_vantage` | 12s gap between calls (5/min) |
| Validation | `fundamentals.py::validate_fundamentals` | P/E negative → NaN, D/E negative → NaN |
| z-score scope | `fundamentals.py::sector_relative_zscore` | only P/E is z-scored; others stay raw |

### Correlation pruning (offline dev tool)

| Value | Where | Current |
|-------|-------|---------|
| Keep-priority map | `correlation.py::KEEP_PRIORITY` | `rsi_14: 10, macd_histogram: 9, price_sma50/200_ratio: 8, vix_zscore: 7` |
| SMA-pair exception threshold | `correlation.py::SMA_EXCEPTION_THRESHOLD` | 0.90 |
| Tiebreaker | `correlation.py::select_drops` | alphabetical (deterministic) |

### Assembly / integrity

| Value | Where | Current |
|-------|-------|---------|
| NaN check on assemble | `assembler.py::assemble_equity` / `::assemble_crypto` | raises `DataError` on any NaN |
| Symbol ordering | `ObservationAssembler.__init__` | alphabetical |
| Macro/HMM/turbulence prefix in names | `assembler.py` `_*_FEATURE_NAMES` | `macro_`, `hmm_`, `turbulence_`, `overnight_`, `portfolio_` |

## Two normalization layers (important)

| Layer | Where | When applied | Scope |
|-------|-------|--------------|-------|
| `RollingZScoreNormalizer` | `features/normalization.py` | Feature pipeline, **before** obs assembly | Per-feature rolling window (252 equity / 360 crypto) |
| SB3 `VecNormalize` | Training wrapper (agents code) | **After** env step returns raw obs | Running mean/std over all obs dims during training |

If you change one, review the other. Portfolio state block is **live-computed and not z-scored** (values already in natural scales).

## Invariants

- Per-asset features are assembled **alpha-sorted by symbol**. Changing symbol order requires retraining.
- Obs vector NaN triggers `DataError` — pipeline must produce fully-populated rows (or drop warmup).
- Pipeline writes macro + HMM + fundamentals to pg16; turbulence is recomputed per bar; portfolio state is live.
- `hmm_proxy_symbol` must be in the symbols list (pydantic validator).

## Known issues / open questions

- **HMM state reduction:** the model is 3-state but the obs vector only carries `P(bull)` and `P(bear)`. `P(crisis)` is computed, stored in `hmm_state_history`, but not exposed to the agent. Deliberate design or oversight? If 3 states are needed by policy, add to `_HMM_FEATURE_NAMES` + bump `HMM_REGIME` constant.
- **Sentiment 180-dim variant** is end-to-end implemented but not enabled in production yaml.
- **Fundamentals freshness:** `fetch_all` writes one row per symbol per day via `fundamentals` table with `(symbol, date)` upsert. If a refresh is skipped, forward-fill behavior depends on the pipeline reader.
- **Offline-only tool:** `CorrelationPruner` is not run automatically at train time. Any feature additions should go through it manually to avoid reintroducing redundant inputs.

## Source of truth

| Concern | File |
|---------|------|
| Technical indicators | `src/swingrl/features/technical.py` |
| Fundamentals fetch + validation | `src/swingrl/features/fundamentals.py` |
| Macro LATERAL JOIN + derivations | `src/swingrl/features/macro.py` |
| HMM regime model | `src/swingrl/features/hmm_regime.py` |
| Turbulence calculators | `src/swingrl/features/turbulence.py` |
| Rolling z-score normalizer | `src/swingrl/features/normalization.py` |
| Correlation pruner (offline) | `src/swingrl/features/correlation.py` |
| Feature health tracker | `src/swingrl/features/health.py` |
| Obs assembler + names | `src/swingrl/features/assembler.py` |
| Pipeline orchestrator | `src/swingrl/features/pipeline.py` |
| Config schema | `src/swingrl/config/schema.py` (`FeaturesConfig`, `SentimentConfig`) |

## Changelog

- **2026-04-15** — Initial version.
