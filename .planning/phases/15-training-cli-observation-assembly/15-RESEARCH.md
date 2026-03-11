# Phase 15: Training CLI Observation Assembly - Research

**Researched:** 2026-03-10
**Domain:** DuckDB feature querying, ObservationAssembler API, train.py integration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Pre-compute full `(N, obs_dim)` observation matrix before passing to TrainingOrchestrator
- Use `ObservationAssembler.assemble_equity()` / `assemble_crypto()` per timestep, stack into matrix
- Default portfolio state (100% cash via `_default_portfolio_state()`) — env overwrites at step time
- Read per-symbol feature rows from DuckDB `features_equity` / `features_crypto`
- Group by timestep, split into per-asset dicts + macro + HMM + turbulence components
- Pass grouped components to ObservationAssembler for each timestep
- Stack assembled vectors into `(N, obs_dim)` numpy array
- Extract close prices per-asset for the price array the env/trainer needs
- Maintain existing `_load_features_prices` return signature: `tuple[np.ndarray, np.ndarray]`
- `train.py._load_features_prices()` is the only function that needs changes
- `TrainingOrchestrator.train(features, prices)` unchanged, receives pre-assembled matrix
- `BaseTradingEnv.__init__(features=...)` unchanged, receives `(N, obs_dim)` array

### Claude's Discretion
- Exact DuckDB query structure for grouping features by timestep
- Whether to add HMM/macro/turbulence column identification via naming convention or config
- Error handling for missing feature groups (e.g., HMM not computed yet)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | StockTradingEnv — Gymnasium-compatible, 8 ETFs, 156-dim observation | Fix `_load_features_prices` to pass correctly-shaped (N, 156) feature array so env construction succeeds |
| TRAIN-02 | CryptoTradingEnv — Gymnasium-compatible, BTC/ETH, 45-dim observation | Fix `_load_features_prices` to pass correctly-shaped (N, 45) feature array for crypto env |
| TRAIN-03 | PPO agent training with locked hyperparameters | Once observation shape is correct, model.learn() proceeds; no changes to HYPERPARAMS needed |
| VAL-01 | Walk-forward backtesting framework with 3-month test folds | Correct observation matrix enables backtest fold construction; assembler used for fold data too |
</phase_requirements>

---

## Summary

Phase 15 closes INT-GAP-01: `train.py._load_features_prices()` currently reads raw DuckDB feature rows and selects all numeric columns (`df.select_dtypes(include=[np.number]).values`), producing an array with 15/13 per-symbol columns rather than the 156/45 observation space that `BaseTradingEnv` expects. This causes a shape mismatch during SB3 model construction — the `observation_space = Box(shape=(obs_dim,))` in `BaseTradingEnv.__init__()` conflicts with the actual data shape.

The fix is surgically contained to `_load_features_prices()`. All upstream components (`ObservationAssembler`, `FeaturePipeline._get_equity_observation()`, HMM/macro/turbulence query helpers) are fully implemented and tested. The plan rewires the function to: query per-timestep per-symbol feature rows, split into the groups the assembler expects, call `assemble_equity()` / `assemble_crypto()` for each timestep, stack the results into an `(N, obs_dim)` array, and extract close prices as a separate `(N, n_assets)` array.

The existing `FeaturePipeline._get_equity_observation()` and `_get_crypto_observation()` methods in `src/swingrl/features/pipeline.py` are the reference implementation — `_load_features_prices()` should replicate their query + assembly logic but applied across all timesteps in bulk rather than one date at a time.

**Primary recommendation:** Implement a vectorised bulk assembly loop in `_load_features_prices()` that mirrors FeaturePipeline's per-timestep logic, using the same DuckDB tables and assembler API. No new abstractions required.

---

## Standard Stack

### Core (all already installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| duckdb | project-pinned | Query `features_equity` / `features_crypto`, `macro_features`, `hmm_state_history`, `ohlcv_daily`, `ohlcv_4h` | Existing DB, already used by train.py |
| numpy | project-pinned | Build per-asset arrays, stack observation matrix, extract price columns | Core numeric type throughout project |
| ObservationAssembler | Phase 5 in-project | Deterministic 156/45-dim vector assembly with NaN/shape validation | Only correct path per Phase 5 contract |

### No new dependencies required
This phase installs nothing new. All libraries are in `pyproject.toml`.

---

## Architecture Patterns

### Data Flow: DuckDB -> Grouped Components -> Assembler -> Stack

```
DuckDB: features_equity (symbol, date, feat_col...)
  -> GROUP BY date, pivot to {symbol: (15,) array}
  -> macro_features -> (6,) array per date
  -> hmm_state_history -> (2,) [p_bull, p_bear] per date
  -> turbulence (compute from ohlcv_daily) -> float per date
     |
     v
ObservationAssembler.assemble_equity(per_asset, macro, hmm, turb)
     |
     v
Stack N observations -> (N, 156) float32
Extract close prices -> (N, 8) float32
```

### Pattern 1: Bulk timestep assembly (equity)

The key is to pivot the `features_equity` table from long format (one row per symbol per date) to wide format (all symbols' features available per date), then iterate over each unique date calling the assembler.

```python
# Source: src/swingrl/features/pipeline.py _get_equity_observation() + schema.py

def _load_features_prices(
    conn: duckdb.DuckDBPyConnection,
    env_name: str,
    config: SwingRLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    assembler = ObservationAssembler(config)

    if env_name == "equity":
        # 1. Fetch all rows, ordered by date then symbol (alpha)
        df = conn.execute(
            "SELECT symbol, date, "
            + ", ".join(_EQUITY_FEATURE_COLS)
            + " FROM features_equity ORDER BY date, symbol"
        ).fetchdf()

        # 2. Also fetch close prices from ohlcv_daily
        prices_df = conn.execute(
            "SELECT symbol, date, close FROM ohlcv_daily ORDER BY date, symbol"
        ).fetchdf()

        # 3. Collect unique timesteps
        dates = df["date"].unique()  # sorted by ORDER BY

        # 4. Fetch shared macro/HMM per date (query once, index by date)
        macro_map = _bulk_macro_array(conn, dates)
        hmm_map = _bulk_hmm_probs(conn, "equity", dates)

        observations = []
        prices_out = []
        for d in dates:
            day_rows = df[df["date"] == d]
            per_asset = {
                row["symbol"]: np.nan_to_num(
                    row[_EQUITY_FEATURE_COLS].values.astype(float), nan=0.0
                )
                for _, row in day_rows.iterrows()
            }
            macro = macro_map.get(d, np.zeros(6))
            hmm = hmm_map.get(d, np.array([0.5, 0.5]))
            turb = 0.0  # Fallback when turbulence not pre-computed

            obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
            observations.append(obs)

            # Prices: close per symbol in alpha order
            day_prices = prices_df[prices_df["date"] == d]
            price_row = [
                float(day_prices[day_prices["symbol"] == s]["close"].iloc[0])
                if not day_prices[day_prices["symbol"] == s].empty
                else 0.0
                for s in sorted(config.equity.symbols)
            ]
            prices_out.append(price_row)

        features = np.stack(observations).astype(np.float32)
        prices = np.array(prices_out, dtype=np.float32)
        return features, prices
```

**Note:** The loop above is illustrative. Implementation should use vectorised pandas operations (pivot_table + merge) for speed over large datasets rather than row-by-row iteration.

### Pattern 2: Efficient pivot approach (preferred)

```python
# More efficient: pivot wide, then iterate dates
per_asset_wide = df.pivot_table(
    index="date",
    columns="symbol",
    values=_EQUITY_FEATURE_COLS,
    aggfunc="first",
)
# per_asset_wide.loc[date, (feat_col, symbol)] -> scalar
```

### Pattern 3: Prices extraction from ohlcv tables

```python
# Equity: ohlcv_daily table, alpha-sorted symbols -> (N, 8) float32
# Crypto: ohlcv_4h table, alpha-sorted symbols -> (N, 2) float32

# Example equity prices pivot:
close_df = conn.execute(
    "SELECT symbol, date, close FROM ohlcv_daily ORDER BY date"
).fetchdf()
close_pivot = close_df.pivot_table(index="date", columns="symbol", values="close")
# Reorder columns to alpha-sorted symbol order
close_pivot = close_pivot[sorted(config.equity.symbols)]
prices = close_pivot.values.astype(np.float32)
```

### Anti-Patterns to Avoid

- **`df.select_dtypes(include=[np.number]).values`**: This is the current broken approach — it grabs ALL numeric columns (including `date` ordinals or timestamp integers) in arbitrary column order. Produces wrong shape and wrong column assignment.
- **Hardcoding 156 or 45**: Use `equity_obs_dim(config.sentiment.enabled, len(config.equity.symbols))` and `CRYPTO_OBS_DIM`. The assembler module-level docstring is explicit: "Never hardcode 156 or 172 — always derive via equity_obs_dim()."
- **Calling assembler before all symbols are present**: If a date has missing symbols in `features_equity`, `assembler.assemble_equity()` raises `KeyError`. Guard with a `nan_to_num` fill / zero default for missing symbols.
- **Not aligning dates between features and prices**: The features array and prices array must have identical row indices (same N timesteps in same order). Misalignment causes the env to read wrong prices for a given observation.
- **Re-instantiating ObservationAssembler per timestep**: Create one instance before the loop — it reads config.equity.symbols once in `__init__`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Observation vector assembly | Custom concatenation logic in train.py | `ObservationAssembler.assemble_equity/assemble_crypto` | Assembler has shape validation, NaN detection, and deterministic column ordering already tested |
| Dimension computation | Hardcoded 156/45 | `equity_obs_dim(sentiment_enabled, n_symbols)` / `CRYPTO_OBS_DIM` | Dynamic — changes with sentiment flag and symbol count |
| Per-asset feature lists | New column name lists | `_EQUITY_FEATURE_COLS` / `_CRYPTO_FEATURE_COLS` from `pipeline.py` | Already defined and kept in sync with DB schema |
| Macro/HMM query patterns | New SQL queries | Replicate `_get_macro_array()` / `_get_hmm_probs()` from `pipeline.py` | Reference patterns already tested and handle empty-result gracefully |

**Key insight:** `FeaturePipeline._get_equity_observation()` and `_get_crypto_observation()` are the canonical reference for observation construction from DB. `_load_features_prices()` needs to apply the same pattern across all N timesteps rather than one at a time.

---

## Common Pitfalls

### Pitfall 1: Missing macro/HMM data causes assembler DataError
**What goes wrong:** `assembler.assemble_equity()` raises `DataError` if NaN propagates from missing macro or HMM rows. If `macro_features` or `hmm_state_history` tables are empty (not yet populated), zero-fill defaults must be applied.
**Why it happens:** Macro and HMM pipelines run separately from feature engineering. In dry-run scenarios, these tables may be empty.
**How to avoid:** Always provide fallback defaults — `np.zeros(6)` for macro, `np.array([0.5, 0.5])` for HMM — when query returns empty. Mirror the `try/except` pattern from `pipeline.py:_get_macro_array()`.
**Warning signs:** `DataError: Equity observation contains N NaN values` in training log.

### Pitfall 2: Symbol ordering mismatch between features and prices
**What goes wrong:** `BaseTradingEnv._build_observation()` reads `self._features[step]` and expects portfolio state in the last `_portfolio_dim` positions. The price array `self._prices[step]` must be indexed in the same symbol order as what the env's `PortfolioSimulator` expects.
**Why it happens:** DuckDB ORDER BY may not guarantee alpha-sorted symbols unless explicitly specified. If price columns appear in a different order than observation feature columns, the env computes wrong weights.
**How to avoid:** Always use `sorted(config.equity.symbols)` / `sorted(config.crypto.symbols)` for BOTH the feature pivot column order AND the price column extraction order. ObservationAssembler already sorts `self._equity_symbols = sorted(config.equity.symbols)` internally.
**Warning signs:** Episode returns are nonsensical (extremely high or low) despite valid features.

### Pitfall 3: Date/timestamp alignment between features and prices tables
**What goes wrong:** `features_equity` dates and `ohlcv_daily` dates may not be 1:1. Features are computed from OHLCV but require enough bars for moving averages (SMA200 needs 200+ bars). The first N rows of features may be NaN/missing.
**Why it happens:** Feature computation drops the warmup period. OHLCV has more rows than computed features.
**How to avoid:** Filter `ohlcv_daily` prices to only the dates that exist in `features_equity`. Use an INNER JOIN on date when fetching both tables.
**Warning signs:** `IndexError` or shape mismatch when `prices[step]` is accessed — array lengths differ.

### Pitfall 4: `_load_features_prices` signature change breaks existing tests
**What goes wrong:** `train.py:main()` calls `_load_features_prices(conn, env_name)` with 2 args. The function may need `config` to instantiate `ObservationAssembler`. Changing the signature without updating the call site breaks the CLI.
**Why it happens:** Current signature is `(conn, env_name)` — no config access.
**How to avoid:** Either add `config` as a third parameter (update the call site in `main()`), or pass it as a module-level config reference. Updating the call site in `main()` is cleaner — `config` is already in scope at that point.
**Warning signs:** `TypeError: _load_features_prices() takes 2 positional arguments but 3 were given` or the inverse.

### Pitfall 5: `--dry-run` flag not present in current CLI
**What goes wrong:** TRAIN-03 success criteria requires `python scripts/train.py --env equity --dry-run` to complete model construction without error. The `--dry-run` flag does not exist in the current `build_parser()`.
**Why it happens:** The flag was referenced in phase success criteria but not yet implemented.
**How to avoid:** Add `--dry-run` argument to `build_parser()` and wire it to skip `model.learn()` (or use `total_timesteps=1`). The dry-run path should still call `_load_features_prices()` so shape validation is exercised.
**Warning signs:** `error: unrecognized arguments: --dry-run` from argparse.

---

## Code Examples

Verified patterns from existing codebase:

### ObservationAssembler instantiation (from assembler.py)
```python
# Source: src/swingrl/features/assembler.py ObservationAssembler.__init__
from swingrl.features.assembler import ObservationAssembler, equity_obs_dim, CRYPTO_OBS_DIM

assembler = ObservationAssembler(config)
# assembler._equity_symbols == sorted(config.equity.symbols)
# assembler._crypto_symbols == sorted(config.crypto.symbols)
```

### Equity observation assembly (from assembler.py assemble_equity)
```python
# Source: src/swingrl/features/assembler.py assemble_equity()
obs = assembler.assemble_equity(
    per_asset_features={"SPY": np.array([...]),  # (15,) each symbol
                        "QQQ": np.array([...])},
    macro=np.zeros(6),             # fallback if no macro data
    hmm_probs=np.array([0.5, 0.5]),  # fallback if no HMM
    turbulence=0.0,                # fallback if not computed
    portfolio_state=None,          # None -> _default_portfolio_state("equity") = 100% cash
)
# obs.shape == (equity_obs_dim(sentiment_enabled, n_symbols),)
```

### Crypto observation assembly (from assembler.py assemble_crypto)
```python
# Source: src/swingrl/features/assembler.py assemble_crypto()
obs = assembler.assemble_crypto(
    per_asset_features={"BTCUSDT": np.array([...]),  # (13,) each symbol
                        "ETHUSDT": np.array([...])},
    macro=np.zeros(6),
    hmm_probs=np.array([0.5, 0.5]),
    turbulence=0.0,
    overnight_context=0.0,
    portfolio_state=None,          # None -> _default_portfolio_state("crypto") = 100% cash
)
# obs.shape == (45,)
```

### DuckDB feature query pattern (from pipeline.py)
```python
# Source: src/swingrl/features/pipeline.py _get_equity_observation()
_EQUITY_FEATURE_COLS = [
    "price_sma50_ratio", "price_sma200_ratio", "rsi_14", "macd_line",
    "macd_histogram", "bb_position", "atr_14_pct", "volume_sma20_ratio",
    "adx_14", "weekly_trend_dir", "weekly_rsi_14",
    "pe_zscore", "earnings_growth", "debt_to_equity", "dividend_yield",
]

row = conn.execute(
    "SELECT * FROM features_equity WHERE symbol = ? AND date <= CAST(? AS DATE) "
    "ORDER BY date DESC LIMIT 1",
    [symbol, date_str],
).fetchdf()
vals = row[_EQUITY_FEATURE_COLS].values[0]
per_asset[symbol] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)
```

### HMM query pattern (from pipeline.py)
```python
# Source: src/swingrl/features/pipeline.py _get_hmm_probs()
row = conn.execute(
    "SELECT p_bull, p_bear FROM hmm_state_history "
    "WHERE environment = ? AND date <= CAST(? AS DATE) "
    "ORDER BY date DESC LIMIT 1",
    [environment, date_str],
).fetchdf()
hmm = np.array([0.5, 0.5]) if row.empty else np.array([
    float(row["p_bull"].iloc[0]), float(row["p_bear"].iloc[0])
])
```

### Dynamic dimension computation (CRITICAL — never hardcode)
```python
# Source: src/swingrl/features/assembler.py equity_obs_dim()
from swingrl.features.assembler import equity_obs_dim, CRYPTO_OBS_DIM

# Equity: dynamically accounts for sentiment flag and symbol count
expected_dim = equity_obs_dim(
    sentiment_enabled=config.sentiment.enabled,
    n_equity_symbols=len(config.equity.symbols),
)
# Default with 8 symbols, no sentiment -> 156
# With sentiment enabled -> 172

# Crypto: fixed at 45
assert CRYPTO_OBS_DIM == 45
```

### Existing `_default_portfolio_state` behavior
```python
# Source: src/swingrl/features/assembler.py _default_portfolio_state()
# Called automatically when portfolio_state=None is passed to assemble_*()
# Equity: np.zeros(27) with state[0] = 1.0 (cash_ratio = 100%)
# Crypto: np.zeros(9) with state[0] = 1.0 (cash_ratio = 100%)
# BaseTradingEnv overwrites the last portfolio_dim positions at step time
# via _build_observation() -> obs[-self._portfolio_dim:] = portfolio_state
```

---

## State of the Art

| Old Approach (current train.py) | Correct Approach (Phase 15) | Impact |
|---------------------------------|-----------------------------|--------|
| `df.select_dtypes(np.number).values` — grabs all numeric columns in DB order | `ObservationAssembler.assemble_equity/crypto()` per timestep | Fixes shape mismatch; SB3 model construction succeeds |
| `features[:, 0:1]` as price proxy (wrong) | `ohlcv_daily` / `ohlcv_4h` close prices, alpha-sorted by symbol | Correct prices for portfolio simulation |
| No config passed to `_load_features_prices` | Config required for `ObservationAssembler(config)` | Enables sentiment-aware dim computation |
| No `--dry-run` flag | `--dry-run` arg added to argparse | Enables TRAIN-03 success criteria test |

---

## Implementation Scope

### Only file that changes: `scripts/train.py`

| Change | Details |
|--------|---------|
| `_load_features_prices(conn, env_name)` signature | Add `config: SwingRLConfig` parameter |
| `_load_features_prices` body | Replace broken `select_dtypes` logic with assembler-based loop |
| `main()` call site | Pass `config` as third arg to `_load_features_prices(conn, env_name, config)` |
| `build_parser()` | Add `--dry-run` flag (boolean, default False) |
| `main()` dry-run branch | Skip `orchestrator.train()` when `args.dry_run` is True; log + return 0 |

### Files that do NOT change
- `src/swingrl/training/trainer.py` — `TrainingOrchestrator.train()` signature unchanged
- `src/swingrl/envs/base.py` — `BaseTradingEnv.__init__()` unchanged
- `src/swingrl/features/assembler.py` — no changes needed
- `src/swingrl/features/pipeline.py` — no changes needed

---

## Open Questions

1. **Turbulence for bulk assembly: compute on-the-fly vs default to 0.0**
   - What we know: `FeaturePipeline._compute_turbulence_equity()` computes turbulence from `ohlcv_daily` log returns per date. The assembler accepts any float.
   - What's unclear: Whether to replicate the TurbulenceCalculator loop across all N dates (expensive) or use `0.0` as training-time default.
   - Recommendation: Default to `0.0` for bulk training assembly. The env's `_get_turbulence()` returns `0.0` in `BaseTradingEnv` base anyway. Turbulence is used for circuit breakers in paper trading, not for observation quality in training. Document this as acceptable training approximation.

2. **Overnight context for crypto: 0.0 default acceptable?**
   - What we know: `FeaturePipeline._get_crypto_observation()` already uses `overnight = 0.0` as a simplified default.
   - Recommendation: Use `0.0` — the pipeline itself does this, so it is the established pattern.

3. **Macro column name mapping: `_MACRO_COLS` vs `_MACRO_FEATURE_NAMES`**
   - What we know: `pipeline.py` defines `_MACRO_COLS = ["vix_zscore", ...]` (DB column names). `assembler.py` defines `_MACRO_FEATURE_NAMES = ["macro_vix_zscore", ...]` (observation slot names — prefixed with `macro_`). The pipeline's `_get_macro_array()` builds the `(6,)` array by raw FRED series ID lookup, not by column name.
   - Recommendation: Replicate `_get_macro_array()` logic — query `macro_features` for latest values before each date and map FRED series IDs to array positions. The macro prefix in `_MACRO_FEATURE_NAMES` is for feature name labeling only, not DB column lookup.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project-standard) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/test_phase15.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | `_load_features_prices("equity")` returns `(N, 156)` shaped array | unit | `uv run pytest tests/test_phase15.py::TestLoadFeaturesEquity -x` | Wave 0 |
| TRAIN-02 | `_load_features_prices("crypto")` returns `(N, 45)` shaped array | unit | `uv run pytest tests/test_phase15.py::TestLoadFeaturesCrypto -x` | Wave 0 |
| TRAIN-03 | `train.py --env equity --dry-run` completes without shape mismatch error | integration | `uv run pytest tests/test_phase15.py::TestDryRunCLI -x` | Wave 0 |
| VAL-01 | Walk-forward fold features/prices have correct shapes for backtest | unit | `uv run pytest tests/test_phase15.py::TestFoldShapes -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_phase15.py -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phase15.py` — covers TRAIN-01, TRAIN-02, TRAIN-03, VAL-01; uses in-memory DuckDB with seeded feature rows

*(Existing test infrastructure — `conftest.py` fixtures `loaded_config`, `equity_env_config` — covers all config needs; only the test file itself is missing)*

---

## Sources

### Primary (HIGH confidence)
- `src/swingrl/features/assembler.py` (read in full) — definitive ObservationAssembler API: method signatures, dimension constants, `_default_portfolio_state()`, `_EQUITY_FEATURE_NAMES`/`_CRYPTO_FEATURE_NAMES`/`_MACRO_FEATURE_NAMES`
- `src/swingrl/features/pipeline.py` (read in full) — reference implementation for per-timestep DB query patterns, `_EQUITY_FEATURE_COLS`, `_MACRO_COLS`, macro/HMM query helpers
- `src/swingrl/features/schema.py` (read in full) — DuckDB DDL for `features_equity` (symbol, date, 15 feature cols) and `features_crypto` (symbol, datetime, 13 feature cols)
- `scripts/train.py` (read in full) — current broken implementation, exact function signatures, call sites
- `src/swingrl/envs/base.py` (read in full) — `BaseTradingEnv.__init__` confirms `observation_space = Box(shape=(obs_dim,))`, `_build_observation()` confirms last `_portfolio_dim` elements are overwritten

### Secondary (HIGH confidence — in-project tests)
- `tests/training/test_trainer.py` — confirms `TrainingOrchestrator.train(features, prices)` receives pre-shaped arrays; `tiny_equity_features` fixture is `(60, equity_obs_dim(False, 2))`, confirming the shape contract
- `tests/conftest.py` — `equity_features_array` fixture is `(300, 156)`, `crypto_features_array` is `(600, 45)` — confirms expected shapes
- `.planning/STATE.md` decisions — Phase 05 and Phase 14 decisions confirm dimension rules: `equity_obs_dim(False, 8)==156`, `equity_obs_dim(True, 8)==172`, `EQUITY_PER_ASSET=15`, portfolio dim = 27 equity / 9 crypto

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries in-project, no new deps
- Architecture: HIGH — reference implementation (`FeaturePipeline`) exists and is tested; shapes verified against conftest fixtures and assembler tests
- Pitfalls: HIGH — derived from direct code inspection of the broken `_load_features_prices` and the assembler's validation logic

**Research date:** 2026-03-10
**Valid until:** 2026-06-10 (stable — internal codebase only, no external API dependencies)
