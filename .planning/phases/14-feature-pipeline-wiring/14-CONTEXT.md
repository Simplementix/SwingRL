# Phase 14: Feature Pipeline Wiring - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire two existing but disconnected functions to production callers: (1) `compare_features()` gets a CLI consumer for manual A/B feature experiments, and (2) `get_sentiment_features()` output integrates into the equity observation vector when `SentimentConfig.enabled=True`. When disabled (default), observation dimensions remain unchanged (156 equity, 45 crypto).

</domain>

<decisions>
## Implementation Decisions

### compare_features consumer
- CLI script in `scripts/` (not a scheduled job) — Doc 08 describes A/B as a manual protocol: add candidate feature, retrain with same hyperparams, compare validation Sharpe
- Script accepts baseline and candidate metric files (or inline args) and calls `compare_features()` from `swingrl.features.pipeline`
- Output: accepted/rejected verdict with Sharpe improvement and reason (matching existing function return dict)
- Follows existing CLI patterns from `scripts/train.py` and `scripts/backtest.py`

### Sentiment observation integration
- Sentiment features are per-asset: 2 features per symbol (sentiment_score, confidence) — per Doc 07 FinBERT pipeline spec
- Equity only — crypto has no news sources wired (Alpaca News + Finnhub are equity-focused)
- When enabled: EQUITY_PER_ASSET goes from 15 → 17, total equity observation from 156 → 172 (8 assets × 2 extra = +16)
- When disabled (default): dimensions unchanged, `get_sentiment_features()` returns empty dict, assembly skips sentiment
- Placement in observation: appended to each asset's per-asset feature vector (after the existing 15 features, before macro/HMM/turbulence)

### Dynamic dimension handling
- Assembler constants (EQUITY_OBS_DIM, EQUITY_PER_ASSET) become config-driven based on `SentimentConfig.enabled`
- RL environments read observation space size from config at initialization (not hardcoded)
- Model retraining required when toggling sentiment on/off (different observation space shape)

### Claude's Discretion
- Exact CLI argument parsing pattern for compare_features script
- Whether to add a `--format json` flag for programmatic output
- Test fixture design for sentiment-enabled observation assembly

</decisions>

<specifics>
## Specific Ideas

- Doc 07 §5: FinBERT scores 5-20 headlines per asset per day. If no headlines exist for an asset, sentiment feature is set to 0.0 (neutral)
- Doc 08 §16: Feature Addition Protocol — add candidate, retrain with identical hyperparams, compare validation Sharpe ≥ 0.05 to keep
- Doc 12: "With FinBERT (M5+): +20-40 sentiment features on top of above" — our implementation adds exactly 16 (2 × 8 equity assets)
- Sentiment pipeline already has graceful degradation — failures produce (0.0, 0.0) defaults, never crashes the pipeline

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `compare_features()` at `pipeline.py:635` — fully implemented, accepts baseline/candidate metric dicts, returns accepted/rejected verdict
- `get_sentiment_features()` at `pipeline.py:570` — fully implemented, returns `{symbol: (score, confidence)}` when enabled
- `FinBERTScorer` at `sentiment/finbert.py` — scores headlines, aggregates sentiment
- `NewsFetcher` at `sentiment/news_fetcher.py` — fetches from Alpaca News + Finnhub
- `SentimentConfig` in `config/schema.py` — has `enabled` field (default False)
- `ObservationAssembler` at `assembler.py` — deterministic assembly with shape validation

### Established Patterns
- CLI scripts in `scripts/` use `load_config()` and `configure_logging()` at entry
- Assembly order: [per-asset alpha-sorted] + [macro] + [HMM] + [turbulence] + [overnight crypto-only] + [portfolio]
- Feature constants (EQUITY_PER_ASSET, EQUITY_OBS_DIM) defined at module level in `assembler.py`
- `_get_equity_observation()` in `pipeline.py` builds per-asset dict → calls assembler

### Integration Points
- `pipeline.py:_get_equity_observation()` — where sentiment features join the observation
- `assembler.py:assemble_equity()` — needs to accept optional sentiment features
- `assembler.py` constants — need to become dynamic or config-aware
- RL environments (`envs/stock_trading.py`, `envs/crypto_trading.py`) — observation space size initialization

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-feature-pipeline-wiring*
*Context gathered: 2026-03-10*
