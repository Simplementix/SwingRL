---
phase: 10-production-hardening
plan: 05
subsystem: features
tags: [finbert, sentiment, nlp, transformers, alpaca-news, finnhub, ab-testing]

# Dependency graph
requires:
  - phase: 10-01
    provides: SentimentConfig in schema.py, swingrl_retry decorator, tenacity dep
provides:
  - FinBERTScorer with lazy model loading for headline sentiment scoring
  - NewsFetcher with Alpaca News primary and Finnhub fallback
  - get_sentiment_features function for conditional pipeline integration
  - A/B experiment infrastructure using +0.05 Sharpe threshold
affects: [training, environments]

# Tech tracking
tech-stack:
  added: [transformers (optional dep group), requests]
  patterns: [lazy module-level import for heavy ML models, conditional feature toggle via config]

key-files:
  created:
    - src/swingrl/sentiment/__init__.py
    - src/swingrl/sentiment/finbert.py
    - src/swingrl/sentiment/news_fetcher.py
    - tests/sentiment/__init__.py
    - tests/sentiment/test_finbert.py
    - tests/sentiment/test_ab_experiment.py
  modified:
    - src/swingrl/features/pipeline.py

key-decisions:
  - "_transformers module-level global for lazy import to avoid ruff N814 CamelCase alias violations"
  - "nosec B107 on NewsFetcher __init__ for empty-string API key defaults (test-friendly)"

patterns-established:
  - "Lazy ML model loading: module-level _transformers global, checked in _ensure_loaded, avoids 2GB+ memory when disabled"
  - "Dual-source news fetching: primary API with fallback, always returning empty list on failure"

requirements-completed: [HARD-03, HARD-04]

# Metrics
duration: 8min
completed: 2026-03-10
---

# Phase 10 Plan 05: FinBERT Sentiment Pipeline Summary

**Lazy-loaded FinBERT scorer with Alpaca/Finnhub news fetching and +0.05 Sharpe A/B experiment threshold**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-10T03:27:24Z
- **Completed:** 2026-03-10T03:36:23Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- FinBERTScorer with lazy model loading (model stays None until first score_headlines call)
- NewsFetcher with Alpaca News API primary and Finnhub fallback, respects max_headlines_per_asset
- get_sentiment_features pipeline integration function with conditional import and NaN-safe defaults
- A/B experiment infrastructure reuses existing compare_features with +0.05 Sharpe threshold
- All 16 sentiment tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: FinBERT scorer and news fetcher**
   - `d758e39` (test: failing tests for FinBERT scorer and news fetcher)
   - `40e46d9` (feat: implement FinBERT scorer and news fetcher)
2. **Task 2: A/B experiment and feature pipeline integration**
   - `d465116` (test: failing tests for A/B experiment and sentiment integration)
   - `ecbb488` (feat: add sentiment feature integration and A/B experiment infrastructure)

_TDD: Each task has RED test commit followed by GREEN implementation commit._

## Files Created/Modified
- `src/swingrl/sentiment/__init__.py` - Sentiment module init
- `src/swingrl/sentiment/finbert.py` - FinBERTScorer with lazy loading, score_headlines, aggregate_sentiment
- `src/swingrl/sentiment/news_fetcher.py` - NewsFetcher with Alpaca primary, Finnhub fallback
- `src/swingrl/features/pipeline.py` - Added get_sentiment_features function
- `tests/sentiment/__init__.py` - Test package init
- `tests/sentiment/test_finbert.py` - 10 tests for FinBERT scoring and news fetching
- `tests/sentiment/test_ab_experiment.py` - 6 tests for A/B comparison and sentiment integration

## Decisions Made
- Used _transformers module-level global for lazy import to avoid ruff N814 CamelCase alias violations with AutoTokenizer/AutoModelForSequenceClassification
- Added nosec B107 on NewsFetcher __init__ for empty-string API key defaults (these are test-friendly defaults, not hardcoded secrets)
- type: ignore[import-untyped,import-not-found] on transformers import since it is an optional dependency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-commit hooks reformatted files from other uncommitted plans (shadow, backup) causing staging conflicts. Resolved by unstaging non-task files and committing only sentiment-related changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Sentiment module ready for integration with training pipeline
- User can enable sentiment via config toggle: `sentiment.enabled: true`
- A/B experiment: train with and without sentiment, compare using compare_features threshold

---
*Phase: 10-production-hardening*
*Completed: 2026-03-10*
