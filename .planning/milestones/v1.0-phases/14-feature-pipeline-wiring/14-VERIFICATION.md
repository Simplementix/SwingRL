---
phase: 14-feature-pipeline-wiring
verified: 2026-03-10T22:30:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 14: Feature Pipeline Wiring Verification Report

**Phase Goal:** Wire feature pipeline: compare_features CLI + sentiment->obs integration
**Verified:** 2026-03-10T22:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                       | Status     | Evidence                                                                                  |
|----|---------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------------------------|
| 1  | compare_features() is callable from a CLI script that prints accepted/rejected verdict      | VERIFIED   | `scripts/compare_features.py` L131 calls `compare_features(baseline, candidate, ...)` and prints text/JSON verdict |
| 2  | CLI returns exit code 0 on accept, 1 on reject, enabling shell scripting                    | VERIFIED   | L150: `return 0 if result["accepted"] else 1`; 5 tests in TestCompareFeaturesScript pass including test_cli_returns_0_on_accept and test_cli_returns_1_on_reject |
| 3  | When SentimentConfig.enabled=True, equity observation vector is (172,) with sentiment values at correct positions | VERIFIED | `assemble_equity()` with `sentiment_features` kwarg concatenates `np.array([score, confidence])` per asset; obs[15]=0.75, obs[16]=0.90 asserted in test_sentiment_values_in_correct_positions (PASSED) |
| 4  | When SentimentConfig.enabled=False (default), equity observation vector remains (156,) with no code changes needed | VERIFIED | sentinel `None` default on `sentiment_features` kwarg; backward-compat aliases `EQUITY_OBS_DIM=156` and `EQUITY_PER_ASSET=15` retained; test_shape_156_when_disabled PASSED |
| 5  | Crypto observation dimensions (45) are never affected by sentiment toggle                   | VERIFIED   | `base.py` L77: crypto path uses `CRYPTO_OBS_DIM` (unchanged constant); pipeline `_get_crypto_observation` has no sentiment path; 835 total tests pass with no crypto regressions |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact                                  | Expected                                                | Status     | Details                                                                                                              |
|-------------------------------------------|---------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------|
| `scripts/compare_features.py`             | CLI entrypoint for A/B feature comparison               | VERIFIED   | 159 lines; exports `build_parser()` and `main()`; follows train.py KeyboardInterrupt pattern; imports `compare_features` from `swingrl.features.pipeline` |
| `src/swingrl/features/assembler.py`       | Config-aware dimension helpers and sentiment assembly   | VERIFIED   | Exports `EQUITY_PER_ASSET_BASE=15`, `SENTIMENT_FEATURES_PER_ASSET=2`, `equity_per_asset_dim()`, `equity_obs_dim()`; `assemble_equity()` accepts `sentiment_features` kwarg; `get_feature_names_equity(sentiment_enabled=True)` returns 172-element list |
| `tests/test_phase14.py`                   | Tests for FEAT-09 and FEAT-10                           | VERIFIED   | 342 lines, 18 tests across 5 classes (TestCompareFeaturesScript, TestEquityDimHelpers, TestAssembleSentiment, TestFeatureNames, TestEnvObsSpace); all 18 PASSED |

---

### Key Link Verification

| From                              | To                                             | Via                                           | Status   | Details                                                                                                               |
|-----------------------------------|------------------------------------------------|-----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------|
| `scripts/compare_features.py`     | `swingrl.features.pipeline.compare_features`   | direct import and call                        | WIRED    | L22: `from swingrl.features.pipeline import compare_features`; L131: `result = compare_features(baseline, candidate, threshold=args.threshold)` |
| `src/swingrl/features/assembler.py` | `src/swingrl/envs/base.py`                   | equity_obs_dim() call replaces EQUITY_OBS_DIM | WIRED    | `base.py` L26: `from swingrl.features.assembler import (..., equity_obs_dim)`; L66-69: `self._obs_dim = equity_obs_dim(sentiment_enabled=config.sentiment.enabled, n_equity_symbols=len(config.equity.symbols))`; `EQUITY_OBS_DIM` import removed |
| `src/swingrl/features/pipeline.py` | `src/swingrl/features/assembler.py`           | assemble_equity() receives sentiment_features kwarg | WIRED | `pipeline.py` L27: `equity_per_asset_dim` imported; L310: used for fallback array size; L336-347: sentiment guard + `sentiment_features=sentiment` kwarg passed to `assemble_equity()` |

---

### Requirements Coverage

| Requirement | Source Plan | Description in REQUIREMENTS.md                                                                 | Status        | Notes                                                                                                                                  |
|-------------|-------------|------------------------------------------------------------------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------|
| FEAT-09     | 14-01-PLAN  | "Correlation pruning: remove features with pairwise Pearson r > 0.85"                          | SATISFIED (re-labelled) | REQUIREMENTS.md definition is stale — refers to Phase 5 work already done. The Phase 14 PLAN and ROADMAP use FEAT-09 to mean "compare_features CLI consumer". The traceability table in REQUIREMENTS.md maps FEAT-09 to Phase 14 and marks it Complete. CLI implementation verified. |
| FEAT-10     | 14-01-PLAN  | "Weekly-derived features (SMA trend direction, weekly RSI-14) computed from aggregated weekly bars" | SATISFIED (re-labelled) | Same mismatch — REQUIREMENTS.md definition is stale Phase 5 text. Phase 14 PLAN and ROADMAP use FEAT-10 to mean "sentiment->obs integration". Traceability table maps FEAT-10 to Phase 14 and marks Complete. Sentiment integration verified. |

**Requirements note:** There is a terminology mismatch — the requirement *descriptions* in REQUIREMENTS.md for FEAT-09 and FEAT-10 describe Phase 5 feature-engineering work, while the Phase 14 PLAN and ROADMAP repurpose those IDs to mean the CLI consumer and sentiment wiring. The traceability table and "Pending (gap closure)" counter in REQUIREMENTS.md are consistent with the Phase 14 usage, confirming this is a documentation labelling issue (descriptions not updated when IDs were reused), not an implementation gap. Both capabilities (CLI + sentiment) are fully implemented and tested.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | No TODOs, placeholders, stubs, or empty returns detected in any modified file |

---

### Human Verification Required

None. All behavioral truths were verifiable programmatically:

- Exit codes verified by 18 unit tests (all passing)
- Observation dimensions verified by shape assertions
- Backward compatibility verified by full suite regression (835 passed, 0 failures)
- Lint verified by ruff (all checks passed)

---

### Summary

Phase 14 goal is achieved. Both deliverables are fully implemented, substantive, and wired:

**FEAT-09 (compare_features CLI):** `scripts/compare_features.py` is a real 159-line CLI that reads JSON metrics files, calls `compare_features()` from `swingrl.features.pipeline`, prints a human-readable or JSON verdict, and exits 0 (accept) or 1 (reject/error). It follows the canonical `build_parser()/main(argv)/KeyboardInterrupt` pattern from `scripts/train.py`. Five dedicated tests cover all contract behaviors.

**FEAT-10 (sentiment->obs integration):** `assembler.py` adds `equity_per_asset_dim()` and `equity_obs_dim()` helpers that eliminate all hardcoded 156/172 constants. `assemble_equity()` accepts a `sentiment_features` sentinel kwarg — `None` by default for backward compatibility, producing `(156,)`. When populated it concatenates 2 sentiment values per asset to produce `(172,)`. `BaseTradingEnv` derives `observation_space.shape` from `config.sentiment.enabled` at init time. `pipeline._get_equity_observation()` calls `get_sentiment_features()` and passes the result through when enabled. Thirteen dedicated tests cover dimensions, value positions, feature names, and env observation space.

Full suite: **835 passed, 0 failures**. Ruff: **all checks passed**. No regressions.

---

_Verified: 2026-03-10T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
