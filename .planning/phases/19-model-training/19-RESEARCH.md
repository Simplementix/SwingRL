# Phase 19: Model Training - Research

**Researched:** 2026-03-11
**Domain:** Stable Baselines3 RL training pipeline orchestration, walk-forward validation, ensemble blending, hyperparameter tuning automation
**Confidence:** HIGH — all findings based on direct codebase inspection of implemented components

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Gate Failure Strategy**
- 2 tuning rounds if baseline ensemble Sharpe < 0.5 (auto-triggered, no human intervention)
- Round 1: PPO only — adjust learning_rate (try 0.0001 and 0.0005), ent_coef, clip_range. Keep [64,64] net_arch
- Round 2: extend to A2C/SAC learning rates (0.0003 and 0.001 for A2C, 0.0001 and 0.001 for SAC)
- Keep [64,64] net_arch throughout — increasing risks overfitting with current data volume (per spec Doc 03 Section 22)
- If both tuning rounds fail: BLOCK deployment. Print detailed diagnostic report. Operator must investigate
- Ensemble gate only for deployment decision — individual fold failures are logged but don't block
- Per-fold gate results (Sharpe>0.7, MDD<0.15, PF>1.5, overfit gap<0.20) are informational/diagnostic

**Training Time Budget**
- No wall-clock time limit — let training run as long as needed
- Sequential execution: equity first (all 3 algos), then crypto
- Start with spec defaults: 1M timesteps equity, 500K timesteps crypto
- If convergence not reached at spec defaults, escalate to 2M equity / 1M crypto
- Checkpoint per completed algo — if interrupted, resume from last completed algo

**Training Data Windowing**
- Walk-forward validation across FULL historical dataset (10yr equity, 8yr crypto) for statistical confidence
- Growing window folds (existing WalkForwardBacktester pattern)
- FINAL deployed model trains on RECENT data only: last 3 years equity, last 1 year crypto
- Rationale: walk-forward proves the approach works historically; deployed model uses recent data for current market relevance

**SAC Buffer Size**
- Use spec default: buffer_size = 1,000,000 (already in trainer.py)
- Remove the 200K conservative concern from STATE.md — 64GB homelab RAM handles 1M easily (~1-1.5GB)

**Operator Reporting**
- Console summary: pass/fail per algo + ensemble metrics + timing
- JSON report saved to data/training_report.json — fold-by-fold details, ensemble weights, gate results, tuning round info, wall-clock timing per algo/fold
- DuckDB: all fold results stored in backtest_results table (already implemented in WalkForwardBacktester)
- Timing info included: wall-clock per algo, per fold, and total — feeds Phase 22 retraining scheduling

**CLI Invocation Model**
- NEW script: scripts/train_pipeline.py (existing scripts/train.py kept for simple single-algo training)
- Supports --env equity|crypto|all (all = sequential equity then crypto)
- Invocation via Docker exec: `docker exec swingrl python scripts/train_pipeline.py --env all`
- Orchestrates full flow: walk-forward validation -> ensemble weight computation -> tuning if needed -> deployment to models/active/

### Claude's Discretion
- Exact checkpoint format and resume detection logic
- How to structure the tuning hyperparameter grid within the specified ranges
- JSON report schema details beyond the specified content
- Console summary formatting (table style, colors)
- How to detect convergence failure vs successful convergence for timestep escalation

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Baseline training runs with default hyperparameters (no tuning) for all 3 algos on equity (3-year rolling window) | TrainingOrchestrator.train() already works; train_pipeline.py needs RECENT data slicing (last 3yr) for final deployment model |
| TRAIN-02 | Baseline training runs with default hyperparameters (no tuning) for all 3 algos on crypto (1-year rolling window) | Same orchestrator handles crypto; needs 1yr recent slice for final model |
| TRAIN-03 | Walk-forward validation records Sharpe, max drawdown, and profit factor per agent and ensemble | WalkForwardBacktester.run() and _store_results() already capture all metrics; pipeline needs to aggregate across algos |
| TRAIN-04 | If baseline ensemble Sharpe < 0.5, targeted tuning phase runs (PPO first, then A2C/SAC) | No tuning orchestrator exists yet; needs new logic in train_pipeline.py using HYPERPARAMS variant dicts passed as override |
| TRAIN-05 | Sharpe-weighted ensemble blending produces valid ensemble weights from backtest results | EnsembleBlender.sharpe_softmax_weights() exists; train.py currently uses placeholder Sharpe=1.0; pipeline must wire real walk-forward Sharpe into compute_weights() |
| TRAIN-06 | Trained models deploy to models/active/{env}/{algo}/ with VecNormalize files | TrainingOrchestrator._save_model() already targets this path; pipeline needs to confirm both files present after final training run |
| TRAIN-07 | Training success gate: ensemble Sharpe > 1.0 and MDD < 15% before proceeding to paper trading | No ensemble-level gate function exists yet; needs new check_ensemble_gate() comparing mean OOS Sharpe aggregated across all folds and algos |
</phase_requirements>

---

## Summary

Phase 19 is an **integration and orchestration phase**, not a from-scratch implementation phase. The core building blocks are fully implemented and tested: `TrainingOrchestrator` (trains any algo/env combo), `WalkForwardBacktester` (generates growing-window folds, retrains per fold, stores to DuckDB), `EnsembleBlender` (softmax Sharpe weights), `ConvergenceCallback` (early stopping), and all metric/validation functions. The existing `scripts/train.py` is a single-env, single-algo launcher that uses placeholder ensemble weights.

The primary deliverable is `scripts/train_pipeline.py`: a hands-off orchestrator that sequences walk-forward validation across all 6 combinations (3 algos x 2 envs), computes real ensemble weights from walk-forward results, evaluates the ensemble gate (Sharpe > 1.0 and MDD < 15%), triggers up to 2 tuning rounds if Sharpe < 0.5, trains final deployment models on RECENT data only, and writes a structured JSON report. The distinction between walk-forward data (full history) and deployment training data (recent window) is the most important correctness concern.

The tech debt item from STATE.md — "Ensemble Sharpe weights are placeholder in train.py" — is resolved here by wiring walk-forward OOS Sharpe ratios from WalkForwardBacktester fold results directly into EnsembleBlender.compute_weights().

**Primary recommendation:** Build train_pipeline.py as a thin orchestration layer over existing components. The pipeline's complexity is in state management (checkpoint/resume, tuning round tracking), not in algorithm implementation.

---

## Standard Stack

### Core (all already in pyproject.toml)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| stable-baselines3 | >=2.0,<3 | PPO/A2C/SAC training, VecNormalize, EvalCallback | Pinned, CPU index on Linux |
| torch | >=2.2,<2.4 | SB3 backend — CPU-only on homelab Linux | Pinned to pytorch-cpu index |
| duckdb | >=1.0,<2 | Feature loading, fold result storage, model metadata | Pinned |
| numpy | >=1.26,<2 | Feature arrays, metrics computation | Pinned |
| structlog | any | Structured logging per CLAUDE.md | Active |

### No New Dependencies Required
All libraries needed for Phase 19 are already installed. The phase adds no new dependencies.

### Installation
```bash
# No new installs — all deps already in pyproject.toml
uv sync
```

---

## Architecture Patterns

### Recommended Script Structure
```
scripts/
├── train.py           # KEEP — lightweight single-algo entry point for testing
└── train_pipeline.py  # NEW — full orchestration pipeline (this phase's artifact)

src/swingrl/training/
├── trainer.py         # EXISTS — TrainingOrchestrator (no changes needed)
├── ensemble.py        # EXISTS — EnsembleBlender (no changes needed)
└── callbacks.py       # EXISTS — ConvergenceCallback (no changes needed)

tests/training/
├── test_trainer.py    # EXISTS — covers TrainingOrchestrator
├── test_ensemble.py   # EXISTS — covers EnsembleBlender
├── test_callbacks.py  # EXISTS — covers ConvergenceCallback
└── test_pipeline.py   # NEW — covers train_pipeline.py behavior
```

### Pattern 1: Walk-Forward vs Final Training Separation
**What:** Phase performs TWO distinct training passes. Walk-forward uses full historical data (10yr/8yr) for statistical validation. Final deployment training uses RECENT data only (last 3yr equity / last 1yr crypto).
**When to use:** Always — this is the locked decision from CONTEXT.md.
**Example:**
```python
# Walk-forward: full history for statistical validation
wf_results = backtester.run(env_name, algo_name, full_features, full_prices, ...)

# Final deployment: recent window only
recent_features, recent_prices = slice_recent(features, prices, env_name)
final_result = orchestrator.train(env_name, algo_name, recent_features, recent_prices, ...)
```

### Pattern 2: Checkpoint Resume Detection
**What:** Detect completed algos by checking existence of models/active/{env}/{algo}/model.zip before starting training.
**When to use:** Before each (env, algo) training pair in sequential execution.
**Example:**
```python
checkpoint_path = models_dir / "active" / env_name / algo_name / "model.zip"
if checkpoint_path.exists():
    log.info("checkpoint_found_skipping", env=env_name, algo=algo_name)
    continue  # skip to next algo
```

### Pattern 3: Tuning Round State Machine
**What:** Two-round tuning with bounded hyperparameter variants. Round 1 PPO-only, Round 2 all algos. State tracked in training report, not persisted to disk (single-run assumption).
**When to use:** When baseline ensemble Sharpe < 0.5 after walk-forward validation.

TUNING_GRID (for planner reference):
- Round 1 PPO: lr in [0.0001, 0.0005], ent_coef in [0.005, 0.02], clip_range in [0.1, 0.3]
- Round 2 A2C: lr in [0.0003, 0.001]
- Round 2 SAC: lr in [0.0001, 0.001]
- Always keep net_arch=[64, 64] — never increase

### Pattern 4: Ensemble Gate Check (new function needed)
**What:** Compute mean OOS Sharpe and mean MDD across all walk-forward folds and all algos for a given environment, then compare against gate thresholds (Sharpe > 1.0, MDD < 15%).
**When to use:** After all walk-forward runs for an environment complete.
**Example:**
```python
def check_ensemble_gate(
    all_fold_results: dict[str, list[FoldResult]]  # algo -> folds
) -> tuple[bool, float, float]:
    """Returns (passed, ensemble_sharpe, ensemble_mdd)."""
    all_oos_sharpes = []
    all_oos_mdds = []
    for algo_results in all_fold_results.values():
        for fold in algo_results:
            all_oos_sharpes.append(fold.out_of_sample_metrics.get("sharpe", 0.0))
            all_oos_mdds.append(fold.out_of_sample_metrics.get("mdd", 1.0))
    ensemble_sharpe = float(np.mean(all_oos_sharpes)) if all_oos_sharpes else 0.0
    ensemble_mdd = float(np.mean(all_oos_mdds)) if all_oos_mdds else 1.0
    passed = ensemble_sharpe > 1.0 and ensemble_mdd < 0.15
    return passed, ensemble_sharpe, ensemble_mdd
```

### Pattern 5: Real Ensemble Weight Wiring
**What:** Aggregate mean OOS Sharpe per algo across all walk-forward folds, then pass to EnsembleBlender.compute_weights(). Replaces the placeholder Sharpe=1.0 in existing train.py.
**When to use:** After walk-forward validation, before final training and deployment.
**Example:**
```python
algo_sharpes: dict[str, float] = {}
for algo_name, fold_results in wf_results.items():
    oos_sharpes = [f.out_of_sample_metrics.get("sharpe", 0.0) for f in fold_results]
    algo_sharpes[algo_name] = float(np.mean(oos_sharpes))

blender = EnsembleBlender(config)
weights = blender.compute_weights(env_name, algo_sharpes)
```

### Anti-Patterns to Avoid
- **Using placeholder Sharpe=1.0 for ensemble weights:** train.py does this today; train_pipeline.py must NOT. Real Sharpe from walk-forward must flow into EnsembleBlender.
- **Training final model on full history:** CONTEXT locked decision — final deployed model uses RECENT data only (3yr equity, 1yr crypto). Walk-forward is for validation.
- **Running walk-forward AND final training with same WalkForwardBacktester.run() call:** The backtester's final fold model is trained on the last growing window, not the recent-only slice. Always do a separate final training call after walk-forward completes.
- **Sharing VecNormalize stats between walk-forward folds and final model:** Each training run creates its own VecNormalize with independently computed stats. Never reuse fold VecNormalize for the final deployed model.
- **Blocking on per-fold gate failures:** Per CONTEXT, per-fold gates are informational only. Only the ensemble gate blocks deployment.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RL training loop | Custom SB3 wrapper | TrainingOrchestrator.train() | Already implements DummyVecEnv+VecNormalize+EvalCallback+ConvergenceCallback+6 smoke tests |
| Walk-forward fold generation | Custom date-slicing | generate_folds() + WalkForwardBacktester.run() | Growing window, embargo gap, min_folds guard all implemented and tested |
| Softmax ensemble weights | Custom weight calculation | sharpe_softmax_weights() / EnsembleBlender.compute_weights() | Numerical stability, adaptive window, logging all handled |
| Metric computation | Custom Sharpe/MDD formulas | annualized_sharpe(), max_drawdown() from swingrl.agents.metrics | Per-environment period factors already correct |
| DuckDB fold result storage | Custom INSERT logic | WalkForwardBacktester._store_results() | Already writes to backtest_results table with correct schema |
| Model file saving | Custom serialization | TrainingOrchestrator._save_model() | Handles model.zip + VecNormalize file atomically |
| Early stopping | Custom patience logic | ConvergenceCallback (via EvalCallback) | 1% improvement threshold, patience=10, already wired into orchestrator |

**Key insight:** Phase 19 is ~90% wiring and ~10% new logic. The new logic is: (a) data windowing for recent-only final training, (b) ensemble gate check function, (c) tuning round state machine, and (d) JSON report serialization.

---

## Common Pitfalls

### Pitfall 1: Walking Over Wrong Data Window
**What goes wrong:** Walk-forward validation uses same data slice as final training (both recent-only), producing too few folds for statistical confidence. Or vice versa — final model trained on full history, missing the recency design intent.
**Why it happens:** Easy to pass the same `features, prices` arrays to both WalkForwardBacktester.run() and TrainingOrchestrator.train().
**How to avoid:** Compute `full_features, full_prices` for walk-forward. Separately compute `recent_features, recent_prices` sliced by bar count for final training. Name variables explicitly to prevent confusion.
**Warning signs:** Fold count for equity walk-forward should be ~40+ folds over 10yr (63-bar test windows). If you see < 10 folds, wrong data range.

### Pitfall 2: VecNormalize Stats Contamination
**What goes wrong:** Loading a walk-forward fold's VecNormalize for inference on a different data slice produces incorrect normalization, corrupting metrics.
**Why it happens:** Each VecNormalize is fitted on its training window. Stats are dataset-specific.
**How to avoid:** Never reuse fold VecNormalize files outside their fold. The WalkForwardBacktester._evaluate_fold() correctly sets `vec_env.training=False` and `vec_env.norm_reward=False` when loading saved stats.
**Warning signs:** OOS Sharpe ratios wildly different from IS with no explainable cause.

### Pitfall 3: EvalCallback Eval Environment Contamination
**What goes wrong:** Using the same VecNormalize instance for both training and eval environments; eval updates running stats and corrupts normalization.
**Why it happens:** Passing the same vec_env reference to both model and EvalCallback.
**How to avoid:** TrainingOrchestrator already creates TWO separate _create_env() calls (vec_env and eval_vec_env). Never combine them.
**Warning signs:** Training appears unstable; reward variance increases with training steps.

### Pitfall 4: SAC learning_starts on Short Walk-Forward Folds
**What goes wrong:** SAC has learning_starts=10,000 in HYPERPARAMS. Early walk-forward folds with short training windows may not accumulate sufficient transitions before learning starts.
**Why it happens:** SAC requires replay buffer warmup before gradient updates begin.
**How to avoid:** Per-fold timesteps (1M default) far exceed learning_starts=10,000. This is only a concern if total_timesteps is dramatically reduced for testing. Short folds are expected in early walk-forward — SAC simply learns less on those folds.
**Warning signs:** SAC fold Sharpe consistently near zero for early (small training window) folds.

### Pitfall 5: Checkpoint Resume After Partial Walk-Forward
**What goes wrong:** Resuming from checkpoint skips final training but not walk-forward, so walk-forward results are regenerated without matching the checkpoint model state.
**Why it happens:** Checkpoint logic only checks model.zip presence, not walk-forward result presence.
**How to avoid:** Checkpoint detection should be separate for walk-forward phase vs final training phase. Store walk-forward results to data/training_report.json incrementally so partial results can be resumed.
**Warning signs:** Duplicate rows in backtest_results DuckDB table with different model_id timestamps.

### Pitfall 6: Ensemble Sharpe Cross-Environment Contamination
**What goes wrong:** Averaging Sharpe across equity and crypto environments produces an equity-dominated ensemble Sharpe that doesn't reflect crypto performance.
**Why it happens:** Equity generates ~40 folds, crypto generates ~52 folds. Different weighting unless handled explicitly.
**How to avoid:** Ensemble gate and weight computation are per-environment. Run gate check and weight computation independently per env. Never aggregate Sharpe across equity and crypto.
**Warning signs:** Ensemble gate passes on equity metrics but deploys to crypto with unchecked performance.

---

## Code Examples

Verified patterns from codebase inspection:

### Recent Data Slicing (3yr equity, 1yr crypto)
```python
# Implements CONTEXT.md locked decision in train_pipeline.py

RECENT_WINDOW_BARS: dict[str, int] = {
    "equity": 252 * 3,     # 756 trading days = 3 years
    "crypto": 2191 * 1,    # 2191 4H bars = 1 year (365.25 * 6)
}

def slice_recent(
    features: np.ndarray,
    prices: np.ndarray,
    env_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice to recent training window for final deployed model."""
    recent_bars = RECENT_WINDOW_BARS[env_name]
    start = max(0, len(features) - recent_bars)
    return features[start:], prices[start:]
```

### Aggregate Walk-Forward Sharpe for Ensemble Weights
```python
# Wires real walk-forward OOS Sharpe into EnsembleBlender
# Replaces placeholder Sharpe=1.0 from scripts/train.py

from swingrl.agents.backtest import FoldResult
from swingrl.training.ensemble import EnsembleBlender

def compute_ensemble_weights_from_wf(
    config: SwingRLConfig,
    wf_results: dict[str, list[FoldResult]],
    env_name: str,
) -> dict[str, float]:
    """Aggregate mean OOS Sharpe per algo and compute softmax weights."""
    algo_sharpes: dict[str, float] = {}
    for algo_name, folds in wf_results.items():
        oos_sharpes = [f.out_of_sample_metrics.get("sharpe", 0.0) for f in folds]
        algo_sharpes[algo_name] = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    blender = EnsembleBlender(config)
    return blender.compute_weights(env_name, algo_sharpes)
```

### JSON Report Schema (recommended structure)
```python
# Claude's discretion — recommended schema for data/training_report.json
import json
from pathlib import Path

def write_training_report(
    report: dict,
    output_path: Path = Path("data/training_report.json"),
) -> None:
    """Write structured training report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

# Expected report shape:
# {
#     "generated_at": "2026-03-11T...",
#     "equity": {
#         "walk_forward": {
#             "ppo": [{"fold": 0, "oos_sharpe": 1.1, "oos_mdd": 0.08, ...}],
#             "a2c": [...],
#             "sac": [...],
#         },
#         "ensemble_weights": {"ppo": 0.45, "a2c": 0.28, "sac": 0.27},
#         "ensemble_gate": {"passed": True, "sharpe": 1.2, "mdd": 0.09},
#         "tuning_rounds": [],
#         "final_training": {
#             "ppo": {"timesteps": 1000000, "converged_at": 750000, "wall_clock_s": 3600},
#         },
#         "wall_clock_total_s": 14400,
#     },
#     "crypto": { ... }
# }
```

### Tuning Hyperparameter Grid
```python
# Implements CONTEXT.md locked tuning ranges
# Claude's discretion: grid structure within those ranges

from typing import Any

TUNING_GRID: dict[int, dict[str, list[dict[str, Any]]]] = {
    1: {  # Round 1: PPO only
        "ppo": [
            {"learning_rate": 0.0001, "ent_coef": 0.005,  "clip_range": 0.1},
            {"learning_rate": 0.0001, "ent_coef": 0.01,   "clip_range": 0.2},
            {"learning_rate": 0.0005, "ent_coef": 0.01,   "clip_range": 0.2},
            {"learning_rate": 0.0005, "ent_coef": 0.02,   "clip_range": 0.3},
        ],
    },
    2: {  # Round 2: A2C and SAC
        "a2c": [
            {"learning_rate": 0.0003},
            {"learning_rate": 0.001},
        ],
        "sac": [
            {"learning_rate": 0.0001},
            {"learning_rate": 0.001},
        ],
    },
}
```

### Timestep Escalation Logic
```python
# Implements CONTEXT.md — escalate if convergence not reached at spec defaults

DEFAULT_TIMESTEPS: dict[str, int] = {"equity": 1_000_000, "crypto": 500_000}
ESCALATED_TIMESTEPS: dict[str, int] = {"equity": 2_000_000, "crypto": 1_000_000}

def decide_final_timesteps(
    env_name: str,
    result: TrainingResult,
) -> int:
    """Return escalated timesteps if convergence was not detected.

    converged_at_step is None when ConvergenceCallback never triggered
    (i.e., training ran to completion without stagnation detection).
    """
    if result.converged_at_step is None:
        return ESCALATED_TIMESTEPS[env_name]
    return DEFAULT_TIMESTEPS[env_name]
```

### TrainingOrchestrator Hyperparameter Override (recommended design)
```python
# Enables tuning without mutating HYPERPARAMS global constant
# Planner should implement this as a new optional param on train()

def train(
    self,
    env_name: str,
    algo_name: str,
    features: np.ndarray,
    prices: np.ndarray,
    total_timesteps: int = 1_000_000,
    hyperparams_override: dict[str, Any] | None = None,  # NEW
) -> TrainingResult:
    params = HYPERPARAMS[algo_name].copy()
    if hyperparams_override:
        params.update(hyperparams_override)  # merge override
    # ... rest of existing logic unchanged
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Placeholder Sharpe=1.0 for ensemble weights (train.py) | Real walk-forward OOS Sharpe via EnsembleBlender | Phase 19 | TRAIN-05 tech debt resolved |
| Single-algo, single-env train.py entry point | Multi-env, multi-algo train_pipeline.py with gate/tuning | Phase 19 | Hands-off operator workflow |
| SAC buffer_size=200_000 (STATE.md concern) | buffer_size=1_000_000 (CONTEXT locked — homelab 64GB handles it) | Phase 19 CONTEXT | Better SAC policy quality |

**No deprecated patterns:** All active SB3 APIs (DummyVecEnv, VecNormalize, EvalCallback, BaseCallback) are stable SB3 v2.x APIs.

---

## Open Questions

1. **Walk-forward fold count and wall-clock feasibility for crypto**
   - What we know: ENV_PARAMS["crypto"] uses test_bars=540, min_train_bars=2190, embargo_bars=130. Total bars = ~37,225. Expected folds: (37225 - 2190 - 130) / (540 + 130) = ~52 folds.
   - What's unclear: 52 folds x 3 algos x 500K timesteps = ~78 training runs for crypto walk-forward alone. Wall-clock on homelab CPU could be 40+ hours.
   - Recommendation: Consider running walk-forward at reduced timesteps (e.g. 100K per fold) and full timesteps only for the final deployment training. The phase has no wall-clock limit, but the operator should be informed of expected duration.

2. **TrainingOrchestrator hyperparams_override interface**
   - What we know: HYPERPARAMS is a module-level dict. Tuning needs per-run overrides.
   - What's unclear: Whether to add hyperparams_override to TrainingOrchestrator.train() signature or create a subclass for tuning.
   - Recommendation: Add optional hyperparams_override: dict[str, Any] | None = None to TrainingOrchestrator.train(). Merge with base HYPERPARAMS copy. This is minimal-invasive and keeps the tuning concern inside the existing class.

3. **DuckDB connection lifetime during long training runs**
   - What we know: train_pipeline.py will run for potentially 12+ hours sequentially on homelab.
   - What's unclear: Whether holding a single DuckDB connection for the entire pipeline run is safe, or whether per-algo open/close is safer.
   - Recommendation: Open/close DuckDB connection per algo unit (not held open for entire pipeline run) to minimize lock contention risk. Document in phase notes.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, configured in pyproject.toml) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/training/ tests/agents/ -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | Baseline equity training (all 3 algos) at default timesteps | integration | `uv run pytest tests/training/test_pipeline.py::test_equity_baseline_training -x` | Wave 0 |
| TRAIN-02 | Baseline crypto training (all 3 algos) at default timesteps | integration | `uv run pytest tests/training/test_pipeline.py::test_crypto_baseline_training -x` | Wave 0 |
| TRAIN-03 | Walk-forward records Sharpe, MDD, PF per algo and ensemble | unit | `uv run pytest tests/training/test_pipeline.py::test_wf_metrics_recorded -x` | Wave 0 |
| TRAIN-04 | Tuning auto-triggers when ensemble Sharpe < 0.5 | unit | `uv run pytest tests/training/test_pipeline.py::test_tuning_triggers_on_low_sharpe -x` | Wave 0 |
| TRAIN-05 | Ensemble weights derived from real walk-forward Sharpe (not placeholder) | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_weights_from_wf_sharpe -x` | Wave 0 |
| TRAIN-06 | model.zip and vec_normalize.pkl present in models/active/{env}/{algo}/ after pipeline | unit | `uv run pytest tests/training/test_pipeline.py::test_model_files_deployed -x` | Wave 0 |
| TRAIN-07 | Ensemble gate blocks deployment when Sharpe < 1.0 or MDD >= 0.15 | unit | `uv run pytest tests/training/test_pipeline.py::test_ensemble_gate_blocks_deployment -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/training/ tests/agents/ -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/training/test_pipeline.py` — covers TRAIN-01 through TRAIN-07 (all Phase 19 requirements)

*(No new test infrastructure files needed — conftest.py and existing fixtures are sufficient. scripts/train_pipeline.py is production code, not test infrastructure.)*

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `src/swingrl/training/trainer.py` — TrainingOrchestrator, HYPERPARAMS, SEED_MAP, _save_model(), _run_smoke_tests()
- Direct codebase inspection: `src/swingrl/training/ensemble.py` — EnsembleBlender, sharpe_softmax_weights()
- Direct codebase inspection: `src/swingrl/training/callbacks.py` — ConvergenceCallback
- Direct codebase inspection: `src/swingrl/agents/backtest.py` — WalkForwardBacktester, generate_folds(), ENV_PARAMS, FoldResult, _store_results()
- Direct codebase inspection: `src/swingrl/agents/validation.py` — check_validation_gates(), diagnose_overfitting(), GateResult
- Direct codebase inspection: `scripts/train.py` — existing CLI, _load_features_prices(), _write_model_metadata(), placeholder weight pattern identified
- Direct codebase inspection: `tests/training/test_trainer.py`, `tests/agents/test_backtest.py`, `tests/agents/test_validation.py`
- `.planning/phases/19-model-training/19-CONTEXT.md` — locked decisions, discretion areas
- `.planning/REQUIREMENTS.md` — TRAIN-01 through TRAIN-07 definitions

### Secondary (MEDIUM confidence)
- `pyproject.toml` — confirmed SB3 >=2.0,<3, torch CPU-only on Linux, duckdb >=1.0,<2, no new deps needed

### Tertiary (LOW confidence)
- None — all findings are from direct codebase inspection

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps in pyproject.toml, versions pinned, verified by direct inspection
- Architecture: HIGH — all components exist and are tested; pipeline pattern derived from existing train.py structure
- Pitfalls: HIGH — derived from direct code analysis (VecNormalize usage, checkpoint patterns, data slicing intent)
- Tuning grid: MEDIUM — grid structure is Claude's discretion; hyperparameter ranges are locked from CONTEXT.md

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (30 days — stable SB3 API, no fast-moving ecosystem concerns)
