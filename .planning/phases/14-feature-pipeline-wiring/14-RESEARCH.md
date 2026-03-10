# Phase 14: Feature Pipeline Wiring - Research

**Researched:** 2026-03-10
**Domain:** Feature pipeline consumer wiring — CLI entrypoint for A/B comparison, sentiment observation integration
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- CLI script in `scripts/` (not a scheduled job) — Doc 08 describes A/B as a manual protocol
- Script accepts baseline and candidate metric files (or inline args) and calls `compare_features()` from `swingrl.features.pipeline`
- Output: accepted/rejected verdict with Sharpe improvement and reason (matching existing function return dict)
- Follows existing CLI patterns from `scripts/train.py` and `scripts/backtest.py`
- Sentiment features are per-asset: 2 features per symbol (sentiment_score, confidence)
- Equity only — crypto has no news sources wired
- When enabled: EQUITY_PER_ASSET goes from 15 → 17, total equity observation from 156 → 172 (8 assets × 2 extra = +16)
- When disabled (default): dimensions unchanged, `get_sentiment_features()` returns empty dict, assembly skips sentiment
- Placement: appended to each asset's per-asset feature vector (after the existing 15 features, before macro/HMM/turbulence)
- Assembler constants (EQUITY_OBS_DIM, EQUITY_PER_ASSET) become config-driven based on `SentimentConfig.enabled`
- RL environments read observation space size from config at initialization (not hardcoded)
- Model retraining required when toggling sentiment on/off

### Claude's Discretion
- Exact CLI argument parsing pattern for compare_features script
- Whether to add a `--format json` flag for programmatic output
- Test fixture design for sentiment-enabled observation assembly

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FEAT-09 | `compare_features()` callable from CLI entrypoint or scheduled job producing A/B comparison results | `compare_features()` at `pipeline.py:635` is fully implemented; needs `scripts/compare_features.py` CLI wrapper following `train.py` pattern |
| FEAT-10 | When `SentimentConfig.enabled=True`, `get_sentiment_features()` output included in equity observation vector | `get_sentiment_features()` at `pipeline.py:570` returns `{symbol: (score, confidence)}` when enabled; `assembler.py` needs config-aware `EQUITY_PER_ASSET`/`EQUITY_OBS_DIM` and `assemble_equity()` signature extension; `base.py` reads constants at `__init__` time from `assembler.py` imports |
</phase_requirements>

## Summary

Phase 14 wires two existing but isolated functions to production callers. Both `compare_features()` and `get_sentiment_features()` are fully implemented in `pipeline.py` — this phase is integration and plumbing, not new logic.

The first task creates `scripts/compare_features.py`, a manual CLI tool that accepts baseline and candidate metric dicts (via JSON files or inline `--baseline-*` / `--candidate-*` args), calls `compare_features()`, and prints the verdict. The existing `scripts/train.py` is the canonical pattern to follow: `build_parser()`, `main(argv)`, `if __name__ == "__main__": sys.exit(main())`, `load_config()` + `configure_logging()` at entry.

The second task modifies `assembler.py` so that `EQUITY_PER_ASSET` and `EQUITY_OBS_DIM` are computed from config rather than hardcoded module-level constants, extends `assemble_equity()` to accept an optional `sentiment_features` argument (`dict[str, tuple[float, float]] | None`), and injects sentiment values into each asset's per-asset slice when provided. The `BaseTradingEnv.__init__()` in `base.py` imports `EQUITY_OBS_DIM` from `assembler.py` — that import must also become config-driven. Since the env receives a `config` parameter at `__init__`, it can call a helper that computes the dimension from `config.sentiment.enabled` rather than importing the hardcoded constant.

**Primary recommendation:** Keep all dimension math in `assembler.py` via two new helper functions — `equity_per_asset_dim(sentiment_enabled: bool) -> int` and `equity_obs_dim(sentiment_enabled: bool, n_equity_symbols: int) -> int` — so both pipeline and env use one authoritative source of truth.

## Standard Stack

### Core (all already installed)
| Library | Purpose | How Used |
|---------|---------|----------|
| `argparse` (stdlib) | CLI argument parsing | Follow `train.py` / `backtest.py` pattern exactly |
| `pathlib.Path` (stdlib) | File I/O for JSON metric files | CLI script reads metric JSON files |
| `json` (stdlib) | Parse metric files passed to compare_features CLI | `json.loads()` / `json.load()` |
| `numpy` | Observation vector construction | Concatenation of sentiment features into per-asset slice |
| `swingrl.features.pipeline` | `compare_features()`, `get_sentiment_features()` | Called directly — no new logic in either |
| `swingrl.features.assembler` | `ObservationAssembler`, dimension constants | Refactored to be config-aware |
| `swingrl.config.schema` | `load_config()`, `SentimentConfig` | `config.sentiment.enabled` drives conditional paths |
| `swingrl.utils.logging` | `configure_logging()` | Called at CLI entry point |
| `structlog` | Logging | Module-level `log = structlog.get_logger(__name__)` |

### No New Dependencies
This phase installs nothing new. `transformers` is already in the optional `[sentiment]` dep group (Phase 10 decision). All other libraries are stdlib or already in the main dep group.

## Architecture Patterns

### Recommended File Changes

```
scripts/
└── compare_features.py    # NEW — FEAT-09 CLI entrypoint

src/swingrl/features/
└── assembler.py           # MODIFIED — FEAT-10 config-aware dims + sentiment injection

src/swingrl/envs/
└── base.py                # MODIFIED — FEAT-10 reads obs_dim from config helper, not hardcoded constant

tests/
└── test_phase14.py        # NEW — all FEAT-09 and FEAT-10 tests
```

### Pattern 1: CLI Script Structure (compare_features.py)

Follow `scripts/train.py` exactly. Key structural elements:

```python
# Source: scripts/train.py (existing canonical pattern)
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

from swingrl.config.schema import load_config
from swingrl.features.pipeline import compare_features
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B feature comparison for SwingRL — manual protocol per Doc 08 §16.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""...""",
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline metrics JSON file.")
    parser.add_argument("--candidate", type=str, required=True,
                        help="Path to candidate metrics JSON file.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Min validation Sharpe improvement to accept (default: 0.05).")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text).")
    parser.add_argument("--config", type=str, default="config/swingrl.yaml",
                        help="Path to config YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())

    result = compare_features(baseline, candidate, threshold=args.threshold)

    if args.format == "json":
        import json as _json
        print(_json.dumps(result, indent=2))
    else:
        verdict = "ACCEPTED" if result["accepted"] else "REJECTED"
        print(f"Verdict: {verdict}")
        print(f"Sharpe improvement: {result['sharpe_improvement']:.4f}")
        print(f"Reason: {result['reason']}")

    log.info("compare_features_complete", accepted=result["accepted"],
             sharpe_improvement=result["sharpe_improvement"])

    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("compare_features_interrupted")
        sys.exit(130)
```

**Exit code convention:** Return 0 if accepted, 1 if rejected — enables shell scripting (`if python scripts/compare_features.py ...; then ...`).

**Metrics JSON file format** (what `--baseline` and `--candidate` point to):
```json
{
  "train_sharpe": 0.82,
  "validation_sharpe": 0.71
}
```

Both keys `train_sharpe` and `validation_sharpe` are required — `compare_features()` will raise `KeyError` if either is absent.

### Pattern 2: Config-Aware Dimension Helpers (assembler.py)

The cleanest design keeps dimension math in `assembler.py` as pure functions — no class required, no circular imports.

```python
# In assembler.py — REPLACE hardcoded module-level constants with:

EQUITY_PER_ASSET_BASE = 15   # without sentiment
SENTIMENT_FEATURES_PER_ASSET = 2  # (sentiment_score, confidence)

def equity_per_asset_dim(sentiment_enabled: bool = False) -> int:
    """Return per-asset feature count for equity, sentiment-aware."""
    return EQUITY_PER_ASSET_BASE + (SENTIMENT_FEATURES_PER_ASSET if sentiment_enabled else 0)

def equity_obs_dim(sentiment_enabled: bool = False, n_equity_symbols: int = 8) -> int:
    """Return total equity observation dimension, sentiment-aware."""
    per_asset = equity_per_asset_dim(sentiment_enabled)
    return (per_asset * n_equity_symbols) + SHARED_MACRO + HMM_REGIME + TURBULENCE + EQUITY_PORTFOLIO

# Keep backward-compatible module-level names pointing to base (disabled) values:
EQUITY_PER_ASSET = EQUITY_PER_ASSET_BASE  # for imports that don't need config-awareness
EQUITY_OBS_DIM = equity_obs_dim(sentiment_enabled=False, n_equity_symbols=8)  # = 156
```

**Backward compatibility:** Keeping `EQUITY_PER_ASSET` and `EQUITY_OBS_DIM` as module-level names (set to the disabled values) means existing code that imports them (notably `base.py`) continues to work unchanged for the default sentiment-disabled case. Only the sentiment-enabled path needs to call `equity_obs_dim(True, n)`.

### Pattern 3: Assembler Sentiment Injection (assemble_equity signature)

```python
# In ObservationAssembler.assemble_equity() — extend signature:

def assemble_equity(
    self,
    per_asset_features: dict[str, np.ndarray],
    macro: np.ndarray,
    hmm_probs: np.ndarray,
    turbulence: float,
    portfolio_state: np.ndarray | None = None,
    sentiment_features: dict[str, tuple[float, float]] | None = None,  # NEW
) -> np.ndarray:
    ...
    # Build per-asset slices with optional sentiment appended:
    for symbol in self._equity_symbols:
        asset_vec = per_asset_features[symbol]  # shape (15,)
        if sentiment_features is not None and symbol in sentiment_features:
            score, confidence = sentiment_features[symbol]
            asset_vec = np.concatenate([asset_vec, np.array([score, confidence])])
        parts.append(asset_vec)
    ...
    # Validate shape against computed (not hardcoded) expected dim:
    expected_dim = equity_obs_dim(
        sentiment_enabled=(sentiment_features is not None),
        n_equity_symbols=len(self._equity_symbols),
    )
    if obs.shape != (expected_dim,):
        raise DataError(...)
```

### Pattern 4: BaseTradingEnv obs_dim (base.py)

`BaseTradingEnv.__init__()` currently reads `EQUITY_OBS_DIM` from `assembler.py` imports:

```python
# Current in base.py (line 66):
self._obs_dim = EQUITY_OBS_DIM  # hardcoded 156
```

Change to use the helper:

```python
# Modified in base.py:
from swingrl.features.assembler import equity_obs_dim, CRYPTO_OBS_DIM

if environment == "equity":
    self._obs_dim = equity_obs_dim(
        sentiment_enabled=config.sentiment.enabled,
        n_equity_symbols=len(config.equity.symbols),
    )
```

This is the **only change needed in base.py**. The `ObservationAssembler` is not used inside the env — the env just reads the pre-computed features array from DuckDB. The observation space shape must match what the features array actually contains.

### Pattern 5: Pipeline _get_equity_observation (pipeline.py)

When sentiment is enabled, `_get_equity_observation()` must call `get_sentiment_features()` and pass the result to `assembler.assemble_equity()`:

```python
def _get_equity_observation(self, date_str: str) -> np.ndarray:
    ...  # existing per-asset, macro, hmm, turb logic unchanged

    # Sentiment features (FEAT-10)
    sentiment: dict[str, tuple[float, float]] | None = None
    if self._config.sentiment.enabled:
        sentiment = get_sentiment_features(
            enabled=True,
            symbols=self._equity_symbols,
            alpaca_api_key=self._config.equity.alpaca_api_key if hasattr(...) else "",
            ...
        )

    return self._assembler.assemble_equity(per_asset, macro, hmm, turb,
                                           sentiment_features=sentiment)
```

**Note:** `get_sentiment_features()` already handles failures gracefully — it returns `(0.0, 0.0)` per symbol on any exception and never raises. No additional error handling is needed at the pipeline call site.

### Anti-Patterns to Avoid

- **Hard-coding 172 anywhere**: The dimension 172 is derived, not a constant. Use `equity_obs_dim(True, 8)` everywhere.
- **Calling `configure_logging()` from anywhere but the CLI main()**: Pipeline and assembler modules must not call `configure_logging()`.
- **Relative imports in `scripts/`**: Scripts import from `swingrl.*` with absolute paths (pattern from `train.py`).
- **Importing `SentimentConfig` at module level in pipeline.py**: The current `get_sentiment_features()` already uses a local import inside the function body (line 601-603). Preserve this pattern — `transformers` is an optional dep that must not be imported at module level.
- **Mutating module-level constants**: Never do `assembler.EQUITY_OBS_DIM = 172`. Use the helper functions instead.
- **Forgetting to update `get_feature_names_equity()`**: If sentiment is enabled, the feature names list returned by `ObservationAssembler.get_feature_names_equity()` must also include `{symbol}_sentiment_score` and `{symbol}_sentiment_confidence` for each asset.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Argument validation for metric files | Custom validator | `argparse` type=str + `Path.read_text()` with try/except for clean error messages |
| Sentiment aggregation | Custom scoring | `FinBERTScorer.aggregate_sentiment()` already implemented |
| News fetching | Custom HTTP | `NewsFetcher.fetch_headlines()` already handles Alpaca + Finnhub fallback |
| Feature registry / plugin system | Dynamic dispatch | Simple `if sentiment_features is not None:` in assembler |

## Common Pitfalls

### Pitfall 1: Observation Shape Mismatch at RL Environment Init
**What goes wrong:** `StockTradingEnv` is initialized with a pre-computed features array of shape `(n_steps, 156)` but `observation_space` is set to `(172,)` because `config.sentiment.enabled=True`. Gymnasium will raise a shape mismatch warning or the SB3 `check_env` will fail.
**Why it happens:** The features array is pre-built (from DuckDB) before the env is initialized. If sentiment is enabled but the stored features were computed without sentiment, they will be 156-wide, not 172-wide.
**How to avoid:** The observation space dim in `BaseTradingEnv` must match the actual features array dimension. Since the env reads `config.sentiment.enabled`, and the features array must be computed to match, both must agree. The safest invariant: `features.shape[1] == self._obs_dim` — assert this at `__init__` time.
**Warning signs:** `AssertionError` or Gymnasium space validation error at env creation.

### Pitfall 2: compare_features() KeyError on Missing Metric Keys
**What goes wrong:** `compare_features()` at line 653 does `candidate_metrics["validation_sharpe"]` with no guard. If a metrics JSON file is missing either `train_sharpe` or `validation_sharpe`, it raises `KeyError`.
**Why it happens:** The user runs the CLI with a malformed JSON file.
**How to avoid:** In the CLI script, validate that both keys exist after loading the JSON, and print a clear error message before calling `compare_features()`. Raise `SystemExit(1)` with a human-readable message.

### Pitfall 3: Backward Compatibility of assemble_equity() Callers
**What goes wrong:** Adding `sentiment_features` as a new kwarg to `assemble_equity()` is safe (it's keyword-only with a `None` default). But `get_feature_names_equity()` also needs updating, and callers who check `len(assembler.get_feature_names_equity())` will get 156 even when sentiment is enabled if the method is not updated.
**Why it happens:** The feature names method is a separate code path from assembly.
**How to avoid:** Update `get_feature_names_equity()` to accept `sentiment_enabled: bool = False` and conditionally append sentiment name strings.

### Pitfall 4: SentimentConfig API Key Wiring
**What goes wrong:** `get_sentiment_features()` requires `alpaca_api_key`, `alpaca_api_secret`, and `finnhub_api_key`. These come from environment variables, not `SentimentConfig`. Calling the function without them causes all symbols to return `(0.0, 0.0)` silently.
**Why it happens:** `SentimentConfig` only stores `finnhub_api_key`. Alpaca keys live in the environment or a separate config section.
**How to avoid:** In `_get_equity_observation()`, read keys from `os.environ` (or the broker config section if available). Since failures are graceful, this does not crash — but tests should assert that keys are passed through correctly.
**Warning signs:** All sentiment scores are 0.0 in integration tests.

### Pitfall 5: Module-Level Constant Import in base.py
**What goes wrong:** `base.py` imports `EQUITY_OBS_DIM` at the top of the file. If this constant is still a bare integer (156), it will always be 156 regardless of config. The assignment `self._obs_dim = EQUITY_OBS_DIM` captures the value at import time.
**Why it happens:** Python module-level constants are bound once at import.
**How to avoid:** Replace the import of `EQUITY_OBS_DIM` with a call to `equity_obs_dim(config.sentiment.enabled, len(config.equity.symbols))` inside `__init__`. This is a one-line change in `base.py`.

## Code Examples

### compare_features() — Existing Function Signature
```python
# Source: src/swingrl/features/pipeline.py:635
def compare_features(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    # Returns: {"accepted": bool, "sharpe_improvement": float, "reason": str}
```

### get_sentiment_features() — Existing Function Signature
```python
# Source: src/swingrl/features/pipeline.py:570
def get_sentiment_features(
    *,
    enabled: bool,
    symbols: list[str],
    alpaca_api_key: str = "",
    alpaca_api_secret: str = "",
    finnhub_api_key: str = "",
    max_headlines_per_asset: int = 10,
    model_name: str = "ProsusAI/finbert",
) -> dict[str, tuple[float, float]]:
    # Returns empty dict if enabled=False
    # Returns {symbol: (sentiment_score, confidence)} when enabled=True
    # Never raises — failures return (0.0, 0.0) per symbol
```

### ObservationAssembler.assemble_equity() — Current Shape Validation
```python
# Source: src/swingrl/features/assembler.py:155
if obs.shape != (EQUITY_OBS_DIM,):
    msg = f"Equity observation shape {obs.shape} != expected ({EQUITY_OBS_DIM},)"
    raise DataError(msg)
```
This line must change to use the dynamic `equity_obs_dim()` helper.

### BaseTradingEnv — Current Hardcoded Import
```python
# Source: src/swingrl/envs/base.py:22-27
from swingrl.features.assembler import (
    CRYPTO_OBS_DIM,
    CRYPTO_PORTFOLIO,
    EQUITY_OBS_DIM,  # This import becomes the refactoring target
    EQUITY_PORTFOLIO,
)
# And line 66:
self._obs_dim = EQUITY_OBS_DIM
```

### Existing CLI main() Pattern (from train.py)
```python
# Source: scripts/train.py:187
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    ...
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("training_interrupted")
        sys.exit(130)
```

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Hardcoded `EQUITY_OBS_DIM = 156` | Config-derived helper `equity_obs_dim(enabled, n)` | Enables runtime dimension switching without code changes |
| `compare_features()` as library-only function | Wrapped in `scripts/compare_features.py` CLI | Callable from shell for A/B experiment workflow |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, configured in pyproject.toml) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/test_phase14.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-09 | `compare_features()` callable from CLI, returns 0 on accept | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript -x` | Wave 0 |
| FEAT-09 | CLI returns 1 on reject | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_returns_1_on_reject -x` | Wave 0 |
| FEAT-09 | CLI accepts JSON file path for baseline/candidate | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_reads_json_files -x` | Wave 0 |
| FEAT-09 | CLI prints JSON output with `--format json` | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_json_format -x` | Wave 0 |
| FEAT-09 | CLI exits with error on missing key in metrics JSON | unit | `uv run pytest tests/test_phase14.py::TestCompareFeaturesScript::test_cli_missing_key_exits_1 -x` | Wave 0 |
| FEAT-10 | `equity_obs_dim(False, 8)` returns 156 | unit | `uv run pytest tests/test_phase14.py::TestEquityDimHelpers::test_disabled_returns_156 -x` | Wave 0 |
| FEAT-10 | `equity_obs_dim(True, 8)` returns 172 | unit | `uv run pytest tests/test_phase14.py::TestEquityDimHelpers::test_enabled_returns_172 -x` | Wave 0 |
| FEAT-10 | `assemble_equity()` with sentiment produces (172,) vector | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_shape_172 -x` | Wave 0 |
| FEAT-10 | `assemble_equity()` without sentiment produces (156,) vector | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_shape_156_when_disabled -x` | Wave 0 |
| FEAT-10 | Sentiment values appear at correct positions in obs vector | unit | `uv run pytest tests/test_phase14.py::TestAssembleSentiment::test_sentiment_values_in_correct_positions -x` | Wave 0 |
| FEAT-10 | `BaseTradingEnv` observation_space.shape is (172,) when sentiment enabled | unit | `uv run pytest tests/test_phase14.py::TestEnvObsSpace::test_equity_env_obs_172_when_sentiment_enabled -x` | Wave 0 |
| FEAT-10 | `BaseTradingEnv` observation_space.shape is (156,) when sentiment disabled | unit | `uv run pytest tests/test_phase14.py::TestEnvObsSpace::test_equity_env_obs_156_default -x` | Wave 0 |
| FEAT-10 | `get_feature_names_equity()` returns 172-element list when sentiment enabled | unit | `uv run pytest tests/test_phase14.py::TestFeatureNames::test_feature_names_172_when_enabled -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_phase14.py -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phase14.py` — covers FEAT-09 and FEAT-10 (does not exist yet, must be created in Wave 0 of plan)

*(No new test infrastructure needed — existing pytest + conftest.py fixtures apply)*

## Sources

### Primary (HIGH confidence)
- `src/swingrl/features/pipeline.py` — `compare_features()` at line 635, `get_sentiment_features()` at line 570, `_get_equity_observation()` at line 306
- `src/swingrl/features/assembler.py` — `ObservationAssembler`, dimension constants, `assemble_equity()` signature
- `src/swingrl/envs/base.py` — `BaseTradingEnv.__init__()` showing how `EQUITY_OBS_DIM` is consumed (line 66)
- `src/swingrl/config/schema.py` — `SentimentConfig` at line 213, `SwingRLConfig` at line 229
- `scripts/train.py` — canonical CLI pattern with `build_parser()`, `main(argv)`, structlog, `load_config()`, `configure_logging()`
- `.planning/phases/14-feature-pipeline-wiring/14-CONTEXT.md` — locked decisions and implementation specifics

### Secondary (MEDIUM confidence)
- `scripts/backtest.py` — secondary CLI pattern confirmation (same structure as `train.py`)
- `.planning/STATE.md` — accumulated decisions including Phase 10 transformer optional-dep decision

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are already installed; only stdlib additions (argparse, json)
- Architecture: HIGH — all integration points identified from direct code inspection; no inference
- Pitfalls: HIGH — shape mismatch and backward-compat issues identified from reading exact import chain

**Research date:** 2026-03-10
**Valid until:** 2026-06-10 (stable internal codebase; no external library changes required)
