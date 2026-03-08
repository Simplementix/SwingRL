# Phase 7: Agent Training and Validation - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Train PPO, A2C, and SAC agents on both equity and crypto environments, blend into Sharpe-weighted softmax ensembles per environment, run walk-forward backtesting with growing training windows, and validate against four performance gates (Sharpe > 0.7, MDD < 15%, PF > 1.5, overfit gap < 20%). Covers TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-12, VAL-01 through VAL-08. Produces 6 trained models (3 algos x 2 envs) stored with full versioning in models/active/. Production execution middleware and risk veto layer are Phase 8.

</domain>

<decisions>
## Implementation Decisions

### Network architecture
- **net_arch=[64,64]** for all three agents (PPO, A2C, SAC) on both environments — Doc 03 exploration starting values, prevents overfitting with current data volume
- Same architecture for equity (156-dim) and crypto (45-dim) — input dimension difference handled by input layer automatically
- **MlpPolicy** (SB3 default string alias) for all three algorithms — no custom policy classes needed at [64,64]
- SB3 default activation functions (ReLU) — no deviation from defaults
- Scale-up to [128,128] or [256,256] is a **manual review decision** if validation Sharpe plateaus — not automated

### Training configuration
- **1M total timesteps** per algorithm per environment (6 training runs total: 3 algos x 2 envs)
- **Sequential training:** PPO → A2C → SAC for each environment, one agent at a time
- **DummyVecEnv with 1 environment** — simplest, lowest memory usage on M1 Mac
- **Different random seeds** per agent: PPO=42, A2C=43, SAC=44 — encourages ensemble diversity
- **VecNormalize** on both observations AND rewards — SB3 default, stabilizes training across market regimes
- **Initial training data window:** All available historical data (not rolling window). Rolling 3yr equity / 1yr crypto applies to retraining only
- **TensorBoard logging** enabled: `tensorboard_log='logs/tensorboard/'` per model — essential for training curve diagnostics

### Convergence and checkpointing
- **ConvergenceCallback:** eval_freq=10,000 steps (100 evaluations over 1M timesteps), min_improvement_pct=1%, patience=10 evaluations — matches Doc 04 spec
- **Save best + final model:** EvalCallback saves best model (highest eval reward) during training, plus final model at completion. Both available for comparison
- **Post-training smoke tests** (6 checks from Doc 15): model deserializes, output shape valid, non-degenerate actions, inference < 100ms, VecNormalize loads, no NaN outputs

### Walk-forward backtesting
- **Test fold duration:** 3 months per environment (~63 equity trading days, ~540 crypto 4H bars) — Doc 04 spec
- **Training fold minimum:** 1 year (252 equity bars, ~2,190 crypto 4H bars) — Doc 04 spec
- **Embargo (purge gap):** Per-environment values from Doc 04/12 — ~5-10 trading days equity, ~90-130 4H bars (~2 weeks) crypto. NOT the flat "200 bars" from roadmap success criteria (which appears to be a generalization)
- **Min trades across folds:** 100+ total per environment — Doc 04 spec

### Ensemble blending (TRAIN-06)
- **Sharpe-weighted softmax** of per-agent validation Sharpe — Doc 03 spec
- Rolling validation windows: **63 trading days** equity, **126 4H bars** (~21 days) crypto — distinct from walk-forward test folds
- Adaptive validation windows: shrink by 50% during high turbulence, expand during calm — Phase 6 TRAIN-10

### Performance validation gates (VAL-07)
- Sharpe > 0.7 per environment
- MDD < 15%
- Profit Factor > 1.5
- Overfitting gap < 20% (in-sample vs out-of-sample Sharpe)
- Metrics computed per-fold and aggregated across all folds

### Model artifact management
- **Full Doc 07 versioning from day one:** `{env}-v{major}.{minor}.{patch}-{algo}-{date}` (e.g., `equity-v1.0.0-ppo-2026-04-01`)
- **Directory structure:** `models/active/{equity,crypto}/{ppo,a2c,sac}/model.zip` + VecNormalize stats
- **DuckDB tables:** `model_metadata` (model_id, environment, algorithm, training dates, validation_sharpe, ensemble_weight, file_path, vec_normalize_path) and `backtest_results` (per-model, per-fold rows)
- **Semver convention:** Major = feature set change, Minor = routine retrain, Patch = emergency retrain. Initial training = v1.0.0

### CLI interface
- **`scripts/train.py`** — Training entry point: `python scripts/train.py --env equity --algo ppo` or `--all` for full ensemble. Handles training, VecNormalize saving, model_metadata writes, versioning
- **`scripts/backtest.py`** — Walk-forward backtesting: `python scripts/backtest.py --env equity`. Runs after training. Can re-run without retraining. Writes to DuckDB backtest_results table
- Follows existing `scripts/` pattern (compute_features.py, etc.)

### Claude's Discretion
- Growing vs sliding training window for walk-forward (Doc 04 says growing/FinRL approach — likely growing)
- Exact fold count (dynamic based on data availability, minimum 3)
- Per-fold vs final ensemble weight computation (Doc 03 FinRL approach suggests per-fold)
- Overfitting diagnostic implementation details (Doc 04 §8.4 reference code)
- Performance metric calculator internals (Sharpe, Sortino, Calmar, Rachev, MDD, avg DD, DD duration)
- Trade-level metric extraction from environment info dicts
- DuckDB table DDL details beyond what Doc 14 specifies
- Training script argument parsing and progress reporting
- How to structure the training module (src/swingrl/training/ and src/swingrl/agents/)

</decisions>

<specifics>
## Specific Ideas

- Doc 03 Phase 1 baseline guidance: "Train each agent with exact recommended starting configurations. Do not tune anything. If baseline ensemble has Sharpe above 0.5, you are in good shape." Phase 7 is this baseline
- Doc 04 ConvergenceCallback reference implementation exists as pseudocode — implement from that spec
- Doc 04 overfitting diagnostic function (diagnose_overfitting) exists as reference code — implement from that spec
- Doc 07 retraining pipeline is 7-stage with CLI interface — Phase 7 builds the foundation that retraining builds on
- Doc 04 §8.6 decision framework: "If model meets success criteria, stop optimizing. Move to paper trading. Do not chase perfection."
- Ensemble diversity comes from: different algorithms (PPO/A2C/SAC), different seeds (42/43/44), and Sharpe-weighted blending that adapts to market regime

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `StockTradingEnv` (envs/equity.py): Gymnasium-compatible equity environment, SB3-validated — training wraps this with DummyVecEnv + VecNormalize
- `CryptoTradingEnv` (envs/crypto.py): Gymnasium-compatible crypto environment with random-start episodes
- `BaseTradingEnv` (envs/base.py): Base class with step/reset contract, portfolio simulation, rolling Sharpe reward
- `PortfolioSimulator` (envs/portfolio.py): Tracks portfolio value, positions, trade log — info dict available for metrics
- `RollingSharpeReward` (envs/rewards.py): 20-day rolling Sharpe with expanding warmup
- `ObservationAssembler` (features/assembler.py): 156-dim equity / 45-dim crypto vectors
- `SwingRLConfig` (config/schema.py): EnvironmentConfig with initial_amount, episode_bars, transaction costs, penalty coefficients
- `DatabaseManager` (data/db.py): DuckDB/SQLite context managers for model_metadata and backtest_results tables
- Exception hierarchy (utils/exceptions.py): ModelError ready for training failures
- structlog (utils/logging.py): Established logging pattern

### Established Patterns
- Type hints on all signatures (disallow_untyped_defs)
- Absolute imports only (from swingrl.training.trainer import ...)
- Pydantic config access via load_config()
- TDD: failing test first, then implementation
- structlog with keyword args
- DuckDB replacement scan for DataFrame-to-table sync
- Scripts in scripts/ directory with CLI argument parsing
- Gymnasium registration in envs/__init__.py

### Integration Points
- Phase 6 environments: wrapped with DummyVecEnv + VecNormalize for training
- Phase 5 feature pipeline: environments consume observations via ObservationAssembler
- Phase 5 DuckDB tables: features_equity, features_crypto (training data source)
- Phase 8 execution: loads trained models + frozen VecNormalize for inference
- Phase 10 deployment: deploy_model.sh transfers models from M1 Mac to homelab
- Empty modules ready for code: src/swingrl/training/__init__.py, src/swingrl/agents/__init__.py
- models/active/, models/shadow/, models/archive/ directories exist with .gitkeep (Phase 2)

### New Dependencies (Phase 7)
- tensorboard: For training curve visualization (SB3 built-in support)
- No other new dependencies — stable-baselines3, gymnasium, torch already installed

</code_context>

<deferred>
## Deferred Ideas

- Hyperparameter tuning (Doc 03 Phase 2 targeted tuning) — only if baseline underperforms, manual review
- Net architecture scale-up to [128,128] or [256,256] — manual decision if validation plateaus
- Automated retraining pipeline with rolling windows (Doc 05 §2.9, Doc 07) — Phase 10 or retraining milestone
- Shadow mode for new models (PROD-03/04) — Phase 10
- FinBERT sentiment as additional feature (HARD-03/04) — Phase 10
- Multi-seed statistical significance runs (3+ seeds per experiment) — future hyperparameter tuning
- Jupyter analysis notebooks for training review (HARD-01) — Phase 10

</deferred>

---

*Phase: 07-agent-training-and-validation*
*Context gathered: 2026-03-08*
