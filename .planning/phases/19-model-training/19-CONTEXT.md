# Phase 19: Model Training - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Train PPO/A2C/SAC agents for equity and crypto on homelab CPU, validate via walk-forward across full historical dataset, compute real Sharpe-weighted ensemble weights, pass the training success gate (ensemble Sharpe > 1.0, MDD < 15%), and deploy models to models/active/{env}/{algo}/ with VecNormalize files. Automated retraining, Discord alerts, and production deployment are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Gate Failure Strategy
- 2 tuning rounds if baseline ensemble Sharpe < 0.5 (auto-triggered, no human intervention)
- Round 1: PPO only — adjust learning_rate (try 0.0001 and 0.0005), ent_coef, clip_range. Keep [64,64] net_arch
- Round 2: extend to A2C/SAC learning rates (0.0003 and 0.001 for A2C, 0.0001 and 0.001 for SAC)
- Keep [64,64] net_arch throughout — increasing risks overfitting with current data volume (per spec Doc 03 Section 22)
- If both tuning rounds fail: BLOCK deployment. Print detailed diagnostic report. Operator must investigate
- Ensemble gate only for deployment decision — individual fold failures are logged but don't block
- Per-fold gate results (Sharpe>0.7, MDD<0.15, PF>1.5, overfit gap<0.20) are informational/diagnostic

### Training Time Budget
- No wall-clock time limit — let training run as long as needed
- Sequential execution: equity first (all 3 algos), then crypto
- Start with spec defaults: 1M timesteps equity, 500K timesteps crypto
- If convergence not reached at spec defaults, escalate to 2M equity / 1M crypto
- Checkpoint per completed algo — if interrupted, resume from last completed algo

### Training Data Windowing
- Walk-forward validation across FULL historical dataset (10yr equity, 8yr crypto) for statistical confidence
- Growing window folds (existing WalkForwardBacktester pattern)
- FINAL deployed model trains on RECENT data only: last 3 years equity, last 1 year crypto
- Rationale: walk-forward proves the approach works historically; deployed model uses recent data for current market relevance

### SAC Buffer Size
- Use spec default: buffer_size = 1,000,000 (already in trainer.py)
- Remove the 200K conservative concern from STATE.md — 64GB homelab RAM handles 1M easily (~1-1.5GB)

### Operator Reporting
- Console summary: pass/fail per algo + ensemble metrics + timing
- JSON report saved to data/training_report.json — fold-by-fold details, ensemble weights, gate results, tuning round info, wall-clock timing per algo/fold
- DuckDB: all fold results stored in backtest_results table (already implemented in WalkForwardBacktester)
- Timing info included: wall-clock per algo, per fold, and total — feeds Phase 22 retraining scheduling

### CLI Invocation Model
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

</decisions>

<specifics>
## Specific Ideas

- Fold-level data in DuckDB is diagnostic gold — if a specific market regime fails consistently, that's a signal problem, not a hyperparameter problem
- Keep scripts/train.py as a lightweight entry point for testing individual algos
- The pipeline should be fully hands-off: operator runs one command, walks away, checks results later
- Timestep escalation (1M -> 2M equity, 500K -> 1M crypto) should use the same convergence callback logic already in trainer.py

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/swingrl/training/trainer.py`: TrainingOrchestrator with locked hyperparams, VecNormalize, 6 smoke tests — core training engine
- `src/swingrl/training/ensemble.py`: EnsembleBlender with sharpe_softmax_weights — produces real ensemble weights from Sharpe ratios
- `src/swingrl/training/callbacks.py`: ConvergenceCallback — detects training stagnation (1% improvement threshold, 10 patience)
- `src/swingrl/agents/backtest.py`: WalkForwardBacktester with generate_folds(), per-fold retraining, DuckDB storage
- `src/swingrl/agents/validation.py`: check_validation_gates() (4 gates), diagnose_overfitting()
- `src/swingrl/agents/metrics.py`: annualized_sharpe, max_drawdown, sortino, calmar, profit_factor, etc.
- `scripts/train.py`: Existing CLI with --env, --algo, --timesteps, --dry-run, feature loading from DuckDB

### Established Patterns
- Docker exec for operator commands (Phase 18: `docker exec swingrl python -m swingrl.data.ingest_all --backfill`)
- JSON report files at data/ path (Phase 18: data/verification.json)
- Config loaded via load_config() from SwingRLConfig
- structlog with keyword args for all logging
- Typed exceptions: ModelError for training/model failures

### Integration Points
- scripts/train.py already has _load_features_prices() that assembles observation matrices from DuckDB
- WalkForwardBacktester._store_results() writes to DuckDB backtest_results table
- _write_model_metadata() writes to DuckDB model_metadata table
- Models save to models/active/{env}/{algo}/model.zip + vec_normalize.pkl
- HYPERPARAMS dict in trainer.py holds the locked starting values — tuning creates variants

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-model-training*
*Context gathered: 2026-03-11*
