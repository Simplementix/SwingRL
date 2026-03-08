# Phase 7: Agent Training and Validation - Research

**Researched:** 2026-03-08
**Domain:** Stable Baselines3 RL training, walk-forward backtesting, ensemble blending
**Confidence:** HIGH

## Summary

Phase 7 implements the full RL training pipeline: training PPO, A2C, and SAC agents on both equity and crypto environments using Stable Baselines3 (SB3), blending them into Sharpe-weighted softmax ensembles, running walk-forward backtesting with purge gaps, and gating on four validation metrics. The phase produces 6 trained models (3 algos x 2 envs), stores metadata and backtest results in DuckDB, and provides CLI scripts for training and backtesting.

All three algorithms are already dependencies (`stable-baselines3[extra]>=2.0,<3`). SB3 provides `DummyVecEnv`, `VecNormalize`, `EvalCallback`, and `StopTrainingOnNoModelImprovement` out of the box. The environments (Phase 6) are Gymnasium-compatible and validated with SB3's `check_env`. The primary technical challenge is the walk-forward backtesting framework with proper purge gaps and the performance metric calculators, which must be built from scratch. The ensemble blending (Sharpe-weighted softmax) is a straightforward computation over validation Sharpe ratios.

**Primary recommendation:** Build the training pipeline in layers -- performance metrics first (pure math, highly testable), then training orchestrator with SB3 callbacks, then walk-forward backtesting, then ensemble blending, then CLI scripts. Each layer is independently testable.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **net_arch=[64,64]** for all three agents (PPO, A2C, SAC) on both environments -- MlpPolicy, ReLU activation (SB3 defaults)
- **1M total timesteps** per algorithm per environment (6 training runs total)
- **Sequential training:** PPO -> A2C -> SAC for each environment, one agent at a time
- **DummyVecEnv with 1 environment** -- simplest, lowest memory usage on M1 Mac
- **Different random seeds:** PPO=42, A2C=43, SAC=44
- **VecNormalize** on both observations AND rewards
- **Initial training data window:** All available historical data (not rolling window)
- **TensorBoard logging** enabled: `tensorboard_log='logs/tensorboard/'`
- **ConvergenceCallback:** eval_freq=10,000 steps, min_improvement_pct=1%, patience=10 evaluations
- **Save best + final model** via EvalCallback
- **Post-training smoke tests** (6 checks from Doc 15)
- **Walk-forward test fold:** 3 months per environment (~63 equity trading days, ~540 crypto 4H bars)
- **Training fold minimum:** 1 year (252 equity bars, ~2,190 crypto 4H bars)
- **Embargo:** Per-environment values (NOT flat 200 bars)
- **Min trades across folds:** 100+ total per environment
- **Sharpe-weighted softmax** ensemble blending
- **Rolling validation windows:** 63 trading days equity, 126 4H bars crypto
- **Validation gates:** Sharpe > 0.7, MDD < 15%, PF > 1.5, overfit gap < 20%
- **Model versioning:** `{env}-v{major}.{minor}.{patch}-{algo}-{date}` format
- **Directory structure:** `models/active/{equity,crypto}/{ppo,a2c,sac}/model.zip`
- **DuckDB tables:** `model_metadata` and `backtest_results`
- **CLI scripts:** `scripts/train.py` and `scripts/backtest.py`

### Claude's Discretion
- Growing vs sliding training window for walk-forward (recommend growing per Doc 04/FinRL)
- Exact fold count (dynamic based on data availability, minimum 3)
- Per-fold vs final ensemble weight computation
- Overfitting diagnostic implementation details
- Performance metric calculator internals (Sharpe, Sortino, Calmar, Rachev, MDD, avg DD, DD duration)
- Trade-level metric extraction from environment info dicts
- DuckDB table DDL details beyond what Doc 14 specifies
- Training script argument parsing and progress reporting
- Module structure (src/swingrl/training/ and src/swingrl/agents/)

### Deferred Ideas (OUT OF SCOPE)
- Hyperparameter tuning (Doc 03 Phase 2 targeted tuning)
- Net architecture scale-up to [128,128] or [256,256]
- Automated retraining pipeline with rolling windows
- Shadow mode for new models (PROD-03/04)
- FinBERT sentiment as additional feature
- Multi-seed statistical significance runs
- Jupyter analysis notebooks

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-03 | PPO agent training with specified hyperparameters | SB3 PPO API verified -- lr=0.0003, n_steps=2048, batch_size=64, ent_coef=0.01, clip_range=0.2, gamma=0.99. Note: SB3 default ent_coef=0.0, spec requires 0.01 override |
| TRAIN-04 | A2C agent training with specified hyperparameters | SB3 A2C API verified -- lr=0.0007, n_steps=5, vf_coef=0.5, ent_coef=0.01, gamma=0.99. Note: SB3 default ent_coef=0.0, spec requires 0.01 override |
| TRAIN-05 | SAC agent training with specified hyperparameters | SB3 SAC API verified -- lr=0.0003, batch_size=256, tau=0.005, ent_coef="auto", gamma=0.99. All match SB3 defaults except batch_size |
| TRAIN-06 | Sharpe-weighted softmax ensemble blending | Pure computation: softmax(sharpe_ratios) produces weights. Validation windows: 63 bars equity, 126 bars crypto |
| TRAIN-12 | Model metadata and ensemble weights stored in DuckDB | DuckDB DDL for model_metadata and ensemble_weights tables. DatabaseManager already supports DuckDB writes |
| VAL-01 | Walk-forward backtesting framework with 3-month test folds | Custom implementation with growing training windows, per-environment fold sizes |
| VAL-02 | Purge gap and embargo between training and test folds | Per-environment embargo values (5-10 equity trading days, 90-130 crypto 4H bars). NOT flat 200 bars |
| VAL-03 | Performance metric calculators: Sharpe, Sortino, Calmar, Rachev, MDD, avg DD, DD duration | Pure math functions over return series -- highly testable, no external deps |
| VAL-04 | Trade-level metrics: win rate, Profit Factor, trade frequency | Extract from PortfolioSimulator trade_log via environment info dicts |
| VAL-05 | ConvergenceCallback for SB3 early stopping | Custom callback extending BaseCallback, uses EvalCallback eval results. StopTrainingOnNoModelImprovement available as reference |
| VAL-06 | Overfitting detection: in-sample vs out-of-sample Sharpe gap | Compute Sharpe on training fold vs test fold, compare gap percentage |
| VAL-07 | Validation gates: Sharpe > 0.7, MDD < 15%, PF > 1.5, overfit gap < 20% | Gate checker function consuming metrics dict, returning pass/fail with reasons |
| VAL-08 | Backtest results stored in DuckDB per model and fold | DuckDB DDL for backtest_results table with per-fold rows |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| stable-baselines3 | >=2.0,<3 | PPO, A2C, SAC training | Already installed, provides all three algorithms + VecNormalize + callbacks |
| gymnasium | >=0.29.1 | Environment interface | Already installed, Phase 6 envs are Gymnasium-compatible |
| torch | >=2.2,<2.4 | Neural network backend | Already installed, MPS on M1 Mac |
| duckdb | >=1.0,<2 | Model metadata + backtest results storage | Already installed, Phase 4 DatabaseManager pattern |
| numpy | >=1.26,<2 | Performance metric computation | Already installed |
| tensorboard | (bundled with sb3[extra]) | Training curve visualization | Included in sb3[extra] dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog | >=24.0 | Training progress logging | All training/backtest operations |
| pydantic | >=2.5 | Config validation | Config access for hyperparameters |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom walk-forward | timeseriescv / skfolio | Overkill -- we need simple growing-window splits, not CPCV. Custom is ~50 lines |
| Custom metric calcs | empyrical / quantstats | Additional deps with numpy 2.0 conflicts. Our metrics are 7 pure functions |

**Installation:**
```bash
# tensorboard may need explicit install if not bundled
uv add tensorboard
```

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/
  training/
    __init__.py
    trainer.py          # TrainingOrchestrator: wraps SB3 learn() with VecNormalize, callbacks, saving
    callbacks.py        # ConvergenceCallback (custom), callback factory
    ensemble.py         # Sharpe-weighted softmax ensemble blending
  agents/
    __init__.py
    metrics.py          # Performance metrics: sharpe, sortino, calmar, rachev, mdd, avg_dd, dd_duration
    backtest.py         # WalkForwardBacktester: fold generation, model evaluation, result aggregation
    validation.py       # ValidationGates: gate checker, overfitting detector
  data/
    db.py               # Extended with model_metadata + backtest_results DDL

scripts/
  train.py              # CLI: --env equity|crypto --algo ppo|a2c|sac|all
  backtest.py           # CLI: --env equity|crypto

tests/
  training/
    test_trainer.py
    test_callbacks.py
    test_ensemble.py
  agents/
    test_metrics.py
    test_backtest.py
    test_validation.py
```

### Pattern 1: Training Orchestrator
**What:** Single class that wraps the full training flow per agent: env creation, VecNormalize wrapping, model instantiation, callback setup, training, saving model + VecNormalize stats + metadata.
**When to use:** Every training run.
**Example:**
```python
# Source: SB3 docs + project conventions
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

ALGO_MAP = {"ppo": PPO, "a2c": A2C, "sac": SAC}
SEED_MAP = {"ppo": 42, "a2c": 43, "sac": 44}

def train_agent(
    env_name: str,  # "equity" or "crypto"
    algo_name: str,  # "ppo", "a2c", "sac"
    features: np.ndarray,
    prices: np.ndarray,
    config: SwingRLConfig,
) -> tuple[BaseAlgorithm, VecNormalize]:
    """Train a single agent on one environment."""
    algo_cls = ALGO_MAP[algo_name]
    seed = SEED_MAP[algo_name]

    # Wrap env
    vec_env = DummyVecEnv([lambda: make_env(env_name, features, prices, config)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Create model with policy_kwargs for net_arch
    model = algo_cls(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log="logs/tensorboard/",
        seed=seed,
        verbose=1,
        **get_hyperparams(algo_name),
    )

    # Setup callbacks
    eval_env = DummyVecEnv([lambda: make_env(env_name, features, prices, config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    convergence_cb = ConvergenceCallback(
        min_improvement_pct=0.01,
        patience=10,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/active/{env_name}/{algo_name}/",
        eval_freq=10_000,
        callback_after_eval=convergence_cb,
        deterministic=True,
    )

    model.learn(total_timesteps=1_000_000, callback=eval_cb)
    return model, vec_env
```

### Pattern 2: ConvergenceCallback
**What:** Custom SB3 callback that monitors mean reward improvement and stops training when improvement falls below threshold for N consecutive evaluations.
**When to use:** Passed to EvalCallback as callback_after_eval.
**Example:**
```python
# Source: SB3 BaseCallback docs
from stable_baselines3.common.callbacks import BaseCallback

class ConvergenceCallback(BaseCallback):
    """Stop training when mean reward improvement < threshold for patience evals."""

    def __init__(
        self,
        min_improvement_pct: float = 0.01,
        patience: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._min_improvement_pct = min_improvement_pct
        self._patience = patience
        self._best_mean_reward: float = -float("inf")
        self._no_improvement_count: int = 0

    def _on_step(self) -> bool:
        """Check improvement after each eval (called by EvalCallback)."""
        # Access parent EvalCallback's last mean reward
        parent = self.parent  # type: ignore[attr-defined]
        if parent is None or not hasattr(parent, "last_mean_reward"):
            return True

        current = parent.last_mean_reward
        if self._best_mean_reward == -float("inf"):
            self._best_mean_reward = current
            return True

        improvement = (current - self._best_mean_reward) / abs(self._best_mean_reward)
        if improvement > self._min_improvement_pct:
            self._best_mean_reward = current
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self._patience:
            log.info("convergence_reached", patience=self._patience, best=self._best_mean_reward)
            return False  # Stop training

        return True
```

### Pattern 3: Walk-Forward Growing Window
**What:** Generate train/test fold splits with growing training windows, fixed test windows, and per-environment purge gaps.
**When to use:** Backtesting phase.
**Example:**
```python
def generate_folds(
    total_bars: int,
    test_bars: int,       # 63 equity, 540 crypto
    min_train_bars: int,  # 252 equity, 2190 crypto
    embargo_bars: int,    # ~10 equity, ~130 crypto
) -> list[tuple[range, range]]:
    """Generate growing-window walk-forward folds."""
    folds = []
    test_start = min_train_bars + embargo_bars

    while test_start + test_bars <= total_bars:
        train_range = range(0, test_start - embargo_bars)
        test_range = range(test_start, test_start + test_bars)
        folds.append((train_range, test_range))
        test_start += test_bars + embargo_bars

    return folds  # Minimum 3 folds required
```

### Pattern 4: Sharpe-Weighted Softmax Ensemble
**What:** Compute ensemble weights as softmax of per-agent validation Sharpe ratios.
**When to use:** After all 3 agents trained for one environment.
**Example:**
```python
def sharpe_softmax_weights(sharpe_ratios: dict[str, float]) -> dict[str, float]:
    """Compute ensemble weights via softmax of Sharpe ratios."""
    names = list(sharpe_ratios.keys())
    values = np.array([sharpe_ratios[n] for n in names])
    shifted = values - np.max(values)  # Numerical stability
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    return {names[i]: float(weights[i]) for i in range(len(names))}
```

### Anti-Patterns to Avoid
- **Training with VecNormalize training=True during eval:** Always set `vec_env.training = False` and `vec_env.norm_reward = False` before evaluation/inference. Phase 6 TRAIN-08 already specifies this.
- **Sharing VecNormalize between train and eval envs:** Create separate VecNormalize wrappers. The eval env's stats should be synced from training env periodically, or use EvalCallback which handles this.
- **Forgetting to save VecNormalize stats:** The model.zip alone is useless without the corresponding VecNormalize stats file. Always save both.
- **Using the same seed for all agents:** Defeats ensemble diversity. PPO=42, A2C=43, SAC=44 is locked.
- **Evaluating walk-forward folds with training VecNormalize:** Each fold's test evaluation should use the VecNormalize stats from that fold's training run, not global stats.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Vectorized env wrapping | Custom env wrapper | `DummyVecEnv` + `VecNormalize` | SB3 handles obs/reward normalization stats tracking, saving/loading |
| Periodic evaluation | Custom eval loop | `EvalCallback` | Handles deterministic eval, best model saving, logging |
| Early stopping | Manual step counting | `StopTrainingOnNoModelImprovement` or custom `ConvergenceCallback` | SB3 callback protocol integrates cleanly with learn() |
| TensorBoard logging | Custom logging | `tensorboard_log` kwarg | Built into SB3, zero additional code |
| Model save/load | Custom serialization | `model.save()` / `PPO.load()` | SB3 handles PyTorch state dict + hyperparams |
| Action softmax | Custom softmax | `process_actions()` from envs/portfolio.py | Already implemented with numerical stability in Phase 6 |

**Key insight:** SB3 provides 80% of the training infrastructure. The custom work is: (1) ConvergenceCallback with improvement-pct logic, (2) walk-forward fold generator, (3) performance metric calculators, (4) ensemble blending, (5) DuckDB persistence. All are pure functions or simple classes.

## Common Pitfalls

### Pitfall 1: VecNormalize Train/Serve Skew
**What goes wrong:** Model trained with VecNormalize but evaluated without it, or with different running stats.
**Why it happens:** VecNormalize is a wrapper, not part of the model. Easy to forget.
**How to avoid:** Always save VecNormalize alongside model. Load with `training=False`, `norm_reward=False` for inference. Pattern established in Phase 6 (TRAIN-08).
**Warning signs:** Model produces wildly different actions on the same input during eval vs training.

### Pitfall 2: SB3 Default ent_coef=0.0
**What goes wrong:** PPO and A2C default to `ent_coef=0.0` (no entropy regularization), but the spec requires `ent_coef=0.01`.
**Why it happens:** Easy to assume SB3 defaults match the spec.
**How to avoid:** Explicitly pass `ent_coef=0.01` for PPO and A2C. SAC uses `ent_coef="auto"` (its own default, which matches spec).
**Warning signs:** Agents converge to deterministic policies too quickly, losing exploration.

### Pitfall 3: SAC learning_starts Default is 100
**What goes wrong:** SAC default `learning_starts=100` starts training very early, before the replay buffer has meaningful diversity.
**Why it happens:** SB3 default is conservative for quick experimentation.
**How to avoid:** Keep the spec value of `learning_starts=10_000` (1% of 1M timesteps). This gives the replay buffer time to fill with diverse experiences.
**Warning signs:** SAC training loss spikes or instability in early training.

### Pitfall 4: Walk-Forward Data Leakage
**What goes wrong:** Training data overlaps with test data, or purge gap too small.
**Why it happens:** Off-by-one errors in fold boundary calculation, or using wrong embargo size.
**How to avoid:** Assert `train_end + embargo <= test_start` for every fold. Use per-environment embargo values (NOT flat 200 bars). Add explicit tests with known fold boundaries.
**Warning signs:** Suspiciously high out-of-sample performance, overfit gap near zero.

### Pitfall 5: Growing Window Retraining
**What goes wrong:** Walk-forward backtesting requires retraining the model for each fold (growing training window), but this is expensive (6 models x N folds).
**Why it happens:** Proper walk-forward validation requires models that haven't seen test data.
**How to avoid:** Accept the compute cost. Use convergence callback to stop early when possible. On M1 Mac with 1M timesteps and [64,64], each training run takes ~10-20 minutes, so 6 models x 3+ folds = 1-2 hours total. Plan for this.
**Warning signs:** If backtesting finishes in seconds, you're probably reusing the same model across folds (invalid).

### Pitfall 6: Sharpe Ratio Edge Cases
**What goes wrong:** Division by zero when returns have zero variance, or meaningless Sharpe on very few trades.
**Why it happens:** Short test folds with few trades, or agent stays 100% cash.
**How to avoid:** Guard with minimum trade count (100+ across folds). Return NaN or -inf for zero-variance returns. Annualize correctly (sqrt(252) for equity daily, sqrt(6*365.25) for crypto 4H).
**Warning signs:** Sharpe ratios > 5 (suspiciously high) or exactly 0 (no trading).

### Pitfall 7: Memory Usage with SAC Replay Buffer
**What goes wrong:** SAC's default `buffer_size=1_000_000` with 156-dim observations consumes significant memory.
**Why it happens:** SAC stores all transitions in memory.
**How to avoid:** Monitor memory during training. With 156 floats x 4 bytes x 1M transitions x 2 (obs + next_obs) = ~1.2 GB. M1 Mac with 8/16GB should handle this. If tight, consider `optimize_memory_usage=True` (saves ~37.5% buffer memory).
**Warning signs:** Python process killed by OOM, training stalls.

## Code Examples

### Hyperparameter Configuration
```python
# Source: CONTEXT.md locked decisions + SB3 API verification
HYPERPARAMS: dict[str, dict[str, Any]] = {
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,      # SB3 default
        "gamma": 0.99,
        "gae_lambda": 0.95,  # SB3 default
        "clip_range": 0.2,
        "ent_coef": 0.01,    # Override SB3 default (0.0)
        "vf_coef": 0.5,      # SB3 default
    },
    "a2c": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,   # SB3 default
        "ent_coef": 0.01,    # Override SB3 default (0.0)
        "vf_coef": 0.5,
    },
    "sac": {
        "learning_rate": 0.0003,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",          # SB3 default
        "learning_starts": 10_000,   # Override SB3 default (100)
        "buffer_size": 1_000_000,    # SB3 default
    },
}
```

### Model Saving Pattern
```python
# Source: SB3 examples + project conventions
from pathlib import Path

def save_trained_model(
    model: BaseAlgorithm,
    vec_normalize: VecNormalize,
    env_name: str,
    algo_name: str,
    models_dir: Path,
) -> tuple[Path, Path]:
    """Save model and VecNormalize stats."""
    model_dir = models_dir / "active" / env_name / algo_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model"
    vec_path = model_dir / "vec_normalize.pkl"

    model.save(str(model_path))
    vec_normalize.save(str(vec_path))

    log.info(
        "model_saved",
        environment=env_name,
        algorithm=algo_name,
        model_path=str(model_path),
        vec_normalize_path=str(vec_path),
    )
    return model_path, vec_path
```

### Performance Metrics (Annualized Sharpe)
```python
# Source: Standard financial math
def annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: float,  # 252 for equity daily, 2190 for crypto 4H
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from a return series."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year
    mean_r = float(np.mean(excess))
    std_r = float(np.std(excess, ddof=1))
    if std_r < 1e-10:
        return float("nan")
    return (mean_r / std_r) * np.sqrt(periods_per_year)
```

### DuckDB Table DDL
```sql
-- model_metadata: track trained models
CREATE TABLE IF NOT EXISTS model_metadata (
    model_id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,       -- 'equity' or 'crypto'
    algorithm TEXT NOT NULL,         -- 'ppo', 'a2c', 'sac'
    version TEXT NOT NULL,           -- 'v1.0.0'
    training_start_date TEXT,
    training_end_date TEXT,
    total_timesteps INTEGER,
    converged_at_step INTEGER,       -- NULL if ran full 1M
    validation_sharpe DOUBLE,
    ensemble_weight DOUBLE,
    model_path TEXT NOT NULL,
    vec_normalize_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

-- backtest_results: per-model, per-fold rows
CREATE TABLE IF NOT EXISTS backtest_results (
    result_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    fold_number INTEGER NOT NULL,
    fold_type TEXT NOT NULL,         -- 'in_sample' or 'out_of_sample'
    train_start_idx INTEGER,
    train_end_idx INTEGER,
    test_start_idx INTEGER,
    test_end_idx INTEGER,
    sharpe DOUBLE,
    sortino DOUBLE,
    calmar DOUBLE,
    mdd DOUBLE,
    profit_factor DOUBLE,
    win_rate DOUBLE,
    total_trades INTEGER,
    avg_drawdown DOUBLE,
    max_dd_duration INTEGER,
    final_portfolio_value DOUBLE,
    total_return DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp
);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SB3 create_eval_env param | EvalCallback (separate) | SB3 2.x | Must use EvalCallback, cannot pass eval env to model constructor |
| SB3 eval_freq on model | eval_freq on EvalCallback | SB3 2.x | Eval frequency is a callback param, not a model param |
| Simple train/test split | Walk-forward with purge/embargo | Lopez de Prado (2018) | Prevents data leakage, more realistic OOS estimates |
| Equal-weight ensemble | Sharpe-weighted softmax | FinRL standard | Adaptive weights based on recent performance |

**Deprecated/outdated:**
- SB3 `create_eval_env` parameter: Removed in SB3 2.x. Use `EvalCallback` instead.
- SB3 `eval_freq` on model constructor: Removed. Pass to `EvalCallback`.

## Open Questions

1. **Growing vs Sliding Training Window**
   - What we know: Doc 04 says "growing/FinRL approach", CONTEXT.md lists this as Claude's discretion
   - Recommendation: Use **growing window** -- each fold trains on all data from start to fold boundary. More data = better model stability. Standard in FinRL literature.

2. **Exact Fold Count**
   - What we know: Minimum 3, dynamic based on data availability
   - Recommendation: Calculate dynamically: `n_folds = (total_bars - min_train_bars) // (test_bars + embargo_bars)`. With 8+ years of crypto data (~17,500 4H bars), this yields ~6-7 folds. With ~5 years equity (~1,260 daily bars), ~3-4 folds.

3. **Per-Fold vs Final Ensemble Weights**
   - What we know: Doc 03 FinRL approach suggests per-fold
   - Recommendation: Compute ensemble weights **per fold** during backtesting (each fold gets its own Sharpe-based weights from the validation window before that fold). For production, use weights from the final training run's validation window.

4. **Annualization Periods**
   - What we know: Equity = daily bars, Crypto = 4H bars
   - Recommendation: Equity: 252 trading days/year. Crypto: 6 bars/day x 365.25 days/year = 2,191.5 bars/year. Use these for Sharpe annualization.

5. **Rachev Ratio Implementation**
   - What we know: Required by VAL-03, less common metric
   - Recommendation: Rachev ratio = CVaR(alpha) of gains / CVaR(alpha) of losses, typically alpha=0.05. Compute as mean of top 5% returns / mean of bottom 5% returns (absolute value).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/training/ tests/agents/ -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-03 | PPO trains with correct hyperparams | unit | `uv run pytest tests/training/test_trainer.py::test_ppo_hyperparams -x` | No -- Wave 0 |
| TRAIN-04 | A2C trains with correct hyperparams | unit | `uv run pytest tests/training/test_trainer.py::test_a2c_hyperparams -x` | No -- Wave 0 |
| TRAIN-05 | SAC trains with correct hyperparams | unit | `uv run pytest tests/training/test_trainer.py::test_sac_hyperparams -x` | No -- Wave 0 |
| TRAIN-06 | Sharpe-weighted softmax ensemble | unit | `uv run pytest tests/training/test_ensemble.py -x` | No -- Wave 0 |
| TRAIN-12 | Model metadata stored in DuckDB | unit | `uv run pytest tests/training/test_trainer.py::test_metadata_stored -x` | No -- Wave 0 |
| VAL-01 | Walk-forward fold generation | unit | `uv run pytest tests/agents/test_backtest.py::test_fold_generation -x` | No -- Wave 0 |
| VAL-02 | Purge gap verified between folds | unit | `uv run pytest tests/agents/test_backtest.py::test_purge_gap -x` | No -- Wave 0 |
| VAL-03 | Performance metrics correct | unit | `uv run pytest tests/agents/test_metrics.py -x` | No -- Wave 0 |
| VAL-04 | Trade-level metrics correct | unit | `uv run pytest tests/agents/test_metrics.py::test_trade_metrics -x` | No -- Wave 0 |
| VAL-05 | ConvergenceCallback stops training | unit | `uv run pytest tests/training/test_callbacks.py -x` | No -- Wave 0 |
| VAL-06 | Overfitting detection correct | unit | `uv run pytest tests/agents/test_validation.py::test_overfitting -x` | No -- Wave 0 |
| VAL-07 | Validation gates pass/fail correctly | unit | `uv run pytest tests/agents/test_validation.py::test_gates -x` | No -- Wave 0 |
| VAL-08 | Backtest results stored in DuckDB | unit | `uv run pytest tests/agents/test_backtest.py::test_results_stored -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/training/ tests/agents/ -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/training/__init__.py` -- package init
- [ ] `tests/training/test_trainer.py` -- trainer orchestrator tests
- [ ] `tests/training/test_callbacks.py` -- ConvergenceCallback tests
- [ ] `tests/training/test_ensemble.py` -- ensemble blending tests
- [ ] `tests/agents/__init__.py` -- package init
- [ ] `tests/agents/test_metrics.py` -- performance metric tests
- [ ] `tests/agents/test_backtest.py` -- walk-forward backtester tests
- [ ] `tests/agents/test_validation.py` -- validation gate tests
- [ ] `tests/conftest.py` additions -- training-specific fixtures (small feature arrays for fast training, mock models)

## Sources

### Primary (HIGH confidence)
- [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) -- constructor params, defaults verified
- [SB3 A2C docs](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) -- constructor params, defaults verified
- [SB3 SAC docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) -- constructor params, defaults verified
- [SB3 Callbacks docs](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) -- BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement API
- [SB3 Examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) -- VecNormalize save/load, DummyVecEnv usage
- [SB3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) -- VecNormalize best practices

### Secondary (MEDIUM confidence)
- [Walk-Forward Optimization](https://blog.quantinsti.com/walk-forward-optimization-introduction/) -- growing vs sliding window concepts
- [Cross Validation in Finance](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/) -- purge/embargo methodology
- Existing codebase: Phase 6 environments, Phase 4 DatabaseManager, Phase 2 config schema

### Tertiary (LOW confidence)
- Rachev ratio implementation details -- based on standard CVaR formula, verify with quantitative finance references during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and used in prior phases
- Architecture: HIGH -- SB3 API verified against current docs, patterns match project conventions
- Pitfalls: HIGH -- verified against SB3 docs, common issues well-documented
- Walk-forward: MEDIUM -- fold generation is custom, but the math is straightforward
- Rachev ratio: LOW -- less common metric, implementation details should be verified

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable -- SB3 2.x API is mature)
