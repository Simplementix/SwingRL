# Agent Architecture Reference

Living reference for SwingRL's RL agents: per-algo hyperparameters, the training orchestrator, callback wiring, ensemble blending, walk-forward backtester, and validation gates. Source-of-truth for "where does X come from" and "what's tunable from yaml". Update when any referenced module changes.

**Last verified against code:** 2026-04-16

## Algorithms at a glance

| Algo | SB3 class | Type | Rollout shape | Used for |
|------|-----------|------|---------------|----------|
| PPO | `stable_baselines3.PPO` | on-policy | `n_steps × n_envs = 2048 × 6 = 12,288` | Stable baseline; primary equity workhorse |
| A2C | `stable_baselines3.A2C` | on-policy | `n_steps × n_envs = 5 × 6 = 30` | Many small updates; reactive on crypto |
| SAC | `stable_baselines3.SAC` | off-policy | replay buffer (200K–500K) | Continuous control; high sample reuse |

All three are wrapped by `TrainingOrchestrator` and blended at inference by `EnsembleBlender`. Policy network is `MlpPolicy` with `net_arch=[64, 64]` for every algo (`trainer.py:266, 270`).

## TrainingOrchestrator

`TrainingOrchestrator` (`training/trainer.py:98`) is the single entry point for training one algo on one env. `train()` (`trainer.py:129-383`) performs:

1. **Build train env** — base VecEnv (`SubprocVecEnv` or `DummyVecEnv` per `config.training.vecenv_backend` × `n_envs`), optionally wrap with `MemoryVecRewardWrapper`, then with `VecNormalize(norm_obs=True, norm_reward=True)` (`trainer.py:188-222`). Stack details and fail-open behavior are in [`reward-shaping.md`](reward-shaping.md).
2. **Build eval env** — single-env `DummyVecEnv(1)` mirroring the train wrapper stack (`trainer.py:224-251, 445-482`). Independent `VecNormalize` instance — does **not** share stats with the train env.
3. **Instantiate model** — `params = HYPERPARAMS[algo].copy(); params.update(hyperparams_override)`; pass to the SB3 class with `policy="MlpPolicy"`, `policy_kwargs={"net_arch":[64,64]}`, `verbose=0`, env=train env (`trainer.py:253-273`).
4. **Attach callbacks** — `ConvergenceCallback` is attached as `callback_after_eval` to `EvalCallback`, plus `MemoryEpochCallback` if `memory_client` was supplied (`trainer.py:275-323`).
5. **Run** — `model.learn(total_timesteps, callback=callbacks)` (`trainer.py:328-330`).
6. **Save** — model + `VecNormalize` stats to `models/active/{env}/{algo}/` (`trainer.py:484-539`); the `MemoryVecRewardWrapper` is unwrapped before the stats file is written so loaders see a plain `VecNormalize`.
7. **Smoke-test** — 6 deterministic checks (deserialize, output shape, action diversity, inference speed, normalize-load, NaN-free) in `trainer.py:541-644`.
8. **Always close** train and eval envs in `finally` (`trainer.py:384-387`).

`train()` returns a `TrainingResult` (`trainer.py:86`) with `model_path`, `vec_normalize_path`, `converged_at_step`, `total_timesteps`, and `advice_stats` (when memory was active).

### Per-algo deterministic seeds

`SEED_MAP = {"ppo": 42, "a2c": 43, "sac": 44}` (`trainer.py:71`). Passed as `seed=` to the SB3 constructor — every fold for a given algo trains from the same seed unless overridden.

## Hyperparameters

`HYPERPARAMS` (`trainer.py:42-69`) is a hardcoded dict; LLM-suggested values override per call via `hyperparams_override`. Final values reaching SB3 = `HYPERPARAMS[algo] | hyperparams_override` (`trainer.py:259-260`).

### PPO (`trainer.py:42-52`)

| HP | Value | Notes |
|----|------:|-------|
| `learning_rate` | 3e-4 | Within `HYPERPARAM_BOUNDS.learning_rate=(1e-5, 1e-3)` |
| `n_steps` | 2048 | Per-env rollout |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Optimizer passes per rollout |
| `gamma` | 0.99 | Bounded `(0.95, 0.995)` for LLM (`bounds.py:45`) |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `ent_coef` | 0.01 | Entropy bonus |
| `vf_coef` | 0.5 | Value-loss weight |

### A2C (`trainer.py:53-60`)

| HP | Value | Notes |
|----|------:|-------|
| `learning_rate` | 7e-4 | |
| `n_steps` | 5 | Short rollout — see mismatch-factor invariant |
| `gamma` | 0.99 | Bounded `(0.95, 0.985)` for LLM (`bounds.py:46`) |
| `gae_lambda` | 0.92 | Lower than PPO — bias/variance trade for short rollout |
| `ent_coef` | 0.01 | |
| `vf_coef` | 0.5 | |

`bounds.py:237-254` also enforces a constraint `(1/(1−γ)) / n_steps < 8` for A2C to prevent the iter-3 collapse.

### SAC (`trainer.py:61-68`)

| HP | Value | Notes |
|----|------:|-------|
| `learning_rate` | 3e-4 | |
| `batch_size` | 256 | |
| `tau` | 0.005 | Polyak target update |
| `gamma` | 0.99 | Bounded `(0.95, 0.995)` for LLM (`bounds.py:47`) |
| `ent_coef` | `"auto_0.1"` | Auto-entropy with conservative initial — was `"auto"` (caused reward drowning) |
| `learning_starts` | 10,000 | Random rollout warm-up |
| `buffer_size` | `config.training.sac_buffer_size` | yaml-tunable; schema default 500K, current yaml **200K** (`schema.py:451-458`, `swingrl.yaml:181`) |

## Vectorization & callbacks

- **n_envs:** `config.training.n_envs` default 6 (`schema.py:459-467`).
- **Backend:** `config.training.vecenv_backend` ∈ {`"subproc"`, `"dummy"`}, default `"subproc"` (`schema.py:468-474`).
- **Eval env** is always single-env (`DummyVecEnv(1)`) — comment notes ~1,260 steps per 5-episode eval, no parallelism win, saves ~1.8 GB (`trainer.py:452-457`).

| Callback | Class | Fires | Purpose | File:Line |
|----------|-------|-------|---------|-----------|
| `EvalCallback` | SB3 built-in | every `eval_freq = max(total_timesteps//10, 1)` steps | Runs 5-episode deterministic eval, logs `mean_reward` | `trainer.py:285-294` |
| `ConvergenceCallback` | SwingRL | after each `EvalCallback` eval | Stops training after `patience=10` evals without `min_improvement_pct=0.01` improvement | `callbacks.py:16-106`, `trainer.py:276-280` |
| `MemoryEpochCallback` | SwingRL | every rollout end (gated by per-algo cadence) | Snapshots, LLM advice, reward-weight updates — full chain in [`reward-shaping.md`](reward-shaping.md) | `trainer.py:303-315`, `epoch_callback.py:82-150` |

Convergence early-stop sets `TrainingResult.converged_at_step = model.num_timesteps`; callers use this to decide whether to escalate timesteps for the next fold.

## Total timesteps

| Source | Equity | Crypto | File:Line |
|--------|-------:|-------:|-----------|
| `train()` default | 1,000,000 | 1,000,000 | `trainer.py:135` |
| `DEFAULT_TIMESTEPS` (used by `MetaTrainingOrchestrator`) | 1,000,000 | 500,000 | `pipeline_helpers.py:45-48` |
| `ESCALATED_TIMESTEPS` (when fold did not converge) | 2,000,000 | 1,000,000 | `pipeline_helpers.py:52-53` |

`pipeline_helpers.py:259-291::decide_final_timesteps` picks the escalated value when `result.converged_at_step is None` for the prior fold.

## LLM hyperparameter override path

```
LLM run_config (meta_orchestrator.py:159)
  → clamp_run_config(advised, algo)            (meta_orchestrator.py:160 → bounds.py:181-256)
      • clamp learning_rate / entropy_coeff / clip_range / n_epochs / batch_size / gamma to HYPERPARAM_BOUNDS
      • round batch_size to nearest power of 2 (bounds.py:224-233)
      • for A2C: enforce mismatch-factor < 8 (bounds.py:237-254)
  → filter to algo-valid keys (_VALID_HP_KEYS, meta_orchestrator.py:57-80, "entropy_coeff" → "ent_coef")
  → trainer.train(hyperparams_override=merged)  (meta_orchestrator.py:202-213)
  → params.update(hyperparams_override)         (trainer.py:259-260)
```

Bounds ranges are read once at `bounds.py` import (`HYPERPARAM_BOUNDS, REWARD_BOUNDS = _load_bounds()`, `bounds.py:95`); yaml changes require a container restart.

## Ensemble blending

Source: `training/ensemble.py`. Stateless; one orchestrator per env.

### `sharpe_softmax_weights(sharpe_ratios)` (`ensemble.py:28-59`)

```
shifted = ratios − max(ratios)        # numerical-stability shift (ensemble.py:45)
weights = exp(shifted) / Σ exp(shifted)
```

- **No temperature parameter** — pure softmax. Future temperature tuning would require code change.
- All-negative input still produces valid weights (relative ordering preserved).
- All-zero input → uniform weights.
- Empty input raises `ValueError` — caller validates.

### `EnsembleBlender` (`ensemble.py:62-125`)

- **Stateless** — only stores `config`.
- `compute_weights(env_name, agent_sharpes)` (`ensemble.py:75-95`) — delegates to `sharpe_softmax_weights`.
- `blend_actions(actions, weights)` (`ensemble.py:97-125`) — element-wise weighted sum of per-algo action vectors; `ModelError` if `actions` empty.

### Where weights come from

`pipeline_helpers.py:219-256::compute_ensemble_weights_from_wf` aggregates **mean OOS Sharpe across walk-forward folds per algo** then applies `sharpe_softmax_weights`. Missing algos default to Sharpe 0.0 (effectively a low — but nonzero — weight; the active algos dominate via their positive Sharpe).

Weights are **persisted to the `model_metadata` pg16 table** (`algorithm`, `ensemble_weight`, `training_end_date`, `environment`); `ExecutionPipeline._get_ensemble_weights` (`pipeline/py:410-445`) reads them per cycle, defaulting to equal `1/3` if the row is missing.

### Per-algo metrics (`agents/metrics.py`)

| Metric | Formula | Edge case | File:Line |
|--------|---------|-----------|-----------|
| `annualized_sharpe` | `mean(r − rf) / std(r, ddof=1) × √periods_per_year` | NaN if `len<2` or `std<1e-10` | `metrics.py:21-46` |
| `sortino_ratio` | `mean(excess) / √mean(min(excess, 0)²) × √periods_per_year` | clamps 999.0 if downside_var=0 & excess>0; 0.0 if excess≤0 | `metrics.py:49-79` |
| `calmar_ratio` | `annualized_return / max_drawdown` | NaN if `max_dd<1e-10` | `metrics.py:82-109` |
| `rachev_ratio` | `CVaR(top α gains) / CVaR(bottom α losses)` (α=0.05) | NaN if no gains or no losses | `metrics.py:112-146` |
| `max_drawdown` | `max((running_max − cum) / running_max)` | 0.0 if empty; prepends 1.0 to capture initial drawdown | `metrics.py:149-166` |
| `avg_drawdown` | `mean(underwater_curve)` | 0.0 if empty | `metrics.py:169-186` |
| `max_dd_duration` | longest bar count in drawdown | 0 if empty | `metrics.py:189-215` |

Trade-level metrics (`win_rate, profit_factor, total_trades, trade_frequency`) are computed by `compute_trade_metrics` (`metrics.py:218-272`).

## Walk-forward backtester

Source: `agents/backtest.py`.

### `ENV_PARAMS` (`backtest.py:44-58`)

Hardcoded per env — **not yaml-tunable**.

| Param | Equity | Crypto |
|-------|-------:|-------:|
| `test_bars` | 63 | 540 |
| `min_train_bars` | 252 | 2,190 |
| `embargo_bars` | 10 | 130 |
| `periods_per_year` | 252 | 2,191.5 |
| `bars_per_week` | 5 | 42 |

### `generate_folds(total_bars, test_bars, min_train_bars, embargo_bars)` (`backtest.py:181-237`)

Growing-window walk-forward with embargo:

- Train always starts at index 0; train end advances each fold.
- First test starts at `min_train_bars + embargo_bars` (`backtest.py:210`).
- Per fold: `train_end = test_start − embargo_bars`, `test_range = [test_start, test_start + test_bars)` (`backtest.py:213-215`).
- Advance: `test_start += test_bars + embargo_bars` (`backtest.py:218`).
- Stops when next test would exceed `total_bars`. **Raises `DataError` if fewer than 3 folds** (`backtest.py:220-227`).
- Output: list of `(train_range, test_range)` tuples (Python `range` objects).

### `WalkForwardBacktester.run()` (`backtest.py:261-462`)

Per algo, per env. Per fold:

1. Slice `features[train_range]`, `prices[train_range]`, etc.
2. `orchestrator.train(advice_enabled = advice_enabled and not is_control)` — returns `TrainingResult` with model + `VecNormalize` paths.
3. **In-sample eval** on training window via `_evaluate_fold` (`backtest.py:382-390`).
4. **Out-of-sample eval** on test window via `_evaluate_fold` (`backtest.py:393-401`).
5. `diagnose_overfitting(IS_sharpe, OOS_sharpe)` (`backtest.py:404-407`).
6. `check_validation_gates(...)` (`backtest.py:410-415`).
7. Build `FoldResult` (`backtest.py:417-430`).
8. Optionally enqueue to `fold_queue` for streaming consumers (`backtest.py:435-439`).
9. INSERT row to pg16 `backtest_results` immediately (`backtest.py:442-444`).

`_evaluate_fold` (`backtest.py:464-596`) loads the saved model, recreates a single-env eval env, and **freezes `VecNormalize`** (`training=False, norm_reward=False`, `backtest.py:521-528`) — the running stats from training are reused, never recomputed on test data. Trades are reconstructed via `_reconstruct_round_trips` (`backtest.py:93-178`, FIFO buy/sell pairing) before metrics.

### `FoldResult` (`backtest.py:62-90`)

`fold_number, train_range, test_range, in_sample_metrics, out_of_sample_metrics, trades, gate_result, overfitting, converged_at_step, total_timesteps, is_control_fold, advice_stats`.

### Persisted tables

| Table | Writer | When | Notable columns |
|-------|--------|------|-----------------|
| `backtest_results` | `_store_results` (`backtest.py:598-654`) | per fold | `model_id, environment, algorithm, fold_number, fold_type='walk_forward', train/test indices, OOS metrics, is_control_fold` |
| `backtest_results` | `store_fold_results_to_duckdb` (`backtest.py:724-879`) | post-iteration enrichment | adds `iteration_number, IS metrics, overfitting gap/class, regime context (HMM, VIX, yield), trade extremes` |
| `iteration_results` | `store_iteration_results_to_duckdb` (`backtest.py:882-1015`) | end of iteration (all algos × folds) | per-algo aggregate Sharpe/MDD, ensemble Sharpe/MDD, gate_passed, ensemble weights, HP JSON, wall-clock time, `memory_enabled` |

`_compute_regime_context` (`backtest.py:663-704`) and `_compute_trade_extremes` (`backtest.py:707-721`) feed extra columns to the iteration writer.

## Validation gates

Source: `agents/validation.py`.

### Hard gates (`check_validation_gates`, `validation.py:79-149`)

All four must pass — fail-closed. **All thresholds are hardcoded** (not yaml-tunable):

| Gate | Direction | Threshold | File:Line |
|------|-----------|----------:|-----------|
| Sharpe | `>` | 0.7 | `validation.py:106-109` |
| Max drawdown | `<` | 0.15 | `validation.py:112-115` |
| Profit factor | `>` | 1.5 | `validation.py:118-125` |
| Overfit gap | `<` | 0.20 | `validation.py:128-135` |

`GateResult` (`validation.py:18-31`) carries `passed`, `failures: list[str]`, and `details` (per-gate `{value, threshold, passed}`).

`max_drawdown` returns a positive fraction (`metrics.py:149-166`, e.g. `0.0756` = 7.56% drawdown), so the `0.15` MDD threshold is absolute — the gate fails when drawdown ≥ 15%, not when a signed value is more negative.

### `diagnose_overfitting(is_sharpe, oos_sharpe)` (`validation.py:33-76`)

`gap = 1 − (OOS / IS)`. Classification:

| Range | Label |
|-------|-------|
| `gap < 0.20` | `healthy` |
| `0.20 ≤ gap ≤ 0.50` | `marginal` |
| `gap > 0.50` | `reject` |

If `IS sharpe ≤ 0` → `gap = inf, classification = "reject"`.

The 0.20 / 0.50 thresholds are **informational** — only the hard `< 0.20` gate is enforced (the rest is metadata for analysis).

### Promotion (separate from validation gates)

Shadow → active promotion lives in `shadow/promoter.py`, not in the training path. Criteria: `shadow_sharpe > active_sharpe`, `shadow_mdd ≤ shadow.mdd_tolerance_ratio × active_mdd` (default 1.2, yaml-tunable), `shadow_pf > 1.5`, no circuit-breaker triggers during shadow window. Auto-promotion gated by `config.shadow.auto_promote` (default `true`).

## Configurable values (yaml)

### Training (`training.*`)

| Key | Schema default | Validator | Current yaml | File:Line |
|-----|---------------:|-----------|:-------------|-----------|
| `training.sac_buffer_size` | 500,000 | `gt=0` | 200,000 | `schema.py:451-458`, `swingrl.yaml:181` |
| `training.n_envs` | 6 | `ge=1` | 6 | `schema.py:459-467`, `swingrl.yaml:182` |
| `training.vecenv_backend` | `subproc` | Literal | `subproc` | `schema.py:468-474`, `swingrl.yaml:183` |

### LLM HP bounds (`training.bounds.hyperparam_bounds.*`)

Used by `clamp_run_config` to clip LLM-suggested HPs before they reach the trainer.

| Key | Default (min, max) | Current yaml | File:Line |
|-----|-------------------:|:-------------|-----------|
| `learning_rate` | (1e-5, 1e-3) | [1e-5, 1e-3] | `schema.py:419`, `swingrl.yaml:170` |
| `entropy_coeff` | (0.0, 0.05) | [0.0, 0.05] | `schema.py:420`, `swingrl.yaml:171` |
| `clip_range` | (0.1, 0.4) | [0.1, 0.4] | `schema.py:421`, `swingrl.yaml:172` |
| `n_epochs` | (3, 20) | [3, 20] | `schema.py:422`, `swingrl.yaml:173` |
| `batch_size` | (32, 512) | [32, 512] | `schema.py:423`, `swingrl.yaml:174` |
| `gamma` | (0.95, 0.995) | [0.95, 0.995] | `schema.py:424`, `swingrl.yaml:175` |

Reward-weight bounds (`training.bounds.reward_bounds.*`) are documented in [`reward-shaping.md`](reward-shaping.md).

### Memory-agent cadence & control folds (`memory_agent.*`)

| Key | Schema default | Current yaml | File:Line |
|-----|---------------:|:-------------|-----------|
| `memory_agent.epoch_cadence_ppo` | 20 | 20 | `schema.py:390`, `swingrl.yaml:108` |
| `memory_agent.epoch_cadence_a2c` | 2,000 | 8,000 | `schema.py:391`, `swingrl.yaml:109` |
| `memory_agent.epoch_cadence_sac` | 10,000 | 40,000 | `schema.py:392`, `swingrl.yaml:110` |
| `memory_agent.control_folds_equity` | `[]` | `[0, 5, 10, 15, 20]` | `schema.py:412`, `swingrl.yaml:114` |
| `memory_agent.control_folds_crypto` | `[]` | `[0, 4, 9, 13]` | `schema.py:413`, `swingrl.yaml:115` |

Cadence enforcement details and notable-event thresholds are in [`reward-shaping.md`](reward-shaping.md).

### Shadow promotion (`shadow.*`)

| Key | Default | Current yaml | File:Line |
|-----|--------:|:-------------|-----------|
| `shadow.equity_eval_days` | 10 (`ge=5`) | 10 | `schema.py:233`, `swingrl.yaml:73` |
| `shadow.crypto_eval_cycles` | 30 (`ge=10`) | 30 | `swingrl.yaml:74` |
| `shadow.auto_promote` | `true` | `true` | `schema.py:235`, `swingrl.yaml:75` |
| `shadow.mdd_tolerance_ratio` | 1.2 (`gt=1.0`) | 1.2 | `schema.py:236`, `swingrl.yaml:76` |

## Hardcoded values (not yaml-tunable — code edit required)

### `HYPERPARAMS` (`trainer.py:42-69`)

| Algo | LR | n_steps | batch | n_epochs | gamma | gae_λ | clip | ent_coef | extras |
|------|-----:|-------:|-----:|---------:|------:|------:|-----:|---------:|--------|
| PPO | 3e-4 | 2048 | 64 | 10 | 0.99 | 0.95 | 0.2 | 0.01 | `vf_coef=0.5` |
| A2C | 7e-4 | 5 | — | — | 0.99 | 0.92 | — | 0.01 | `vf_coef=0.5` |
| SAC | 3e-4 | — | 256 | — | 0.99 | — | — | `auto_0.1` | `tau=0.005`, `learning_starts=10000`, `buffer_size` from yaml |

### Per-algo seeds & policy

| Constant | Value | Location |
|----------|-------|----------|
| `SEED_MAP["ppo"]` | 42 | `trainer.py:71` |
| `SEED_MAP["a2c"]` | 43 | `trainer.py:71` |
| `SEED_MAP["sac"]` | 44 | `trainer.py:71` |
| `policy` | `MlpPolicy` | `trainer.py:266` |
| `policy_kwargs.net_arch` | `[64, 64]` | `trainer.py:270` |
| `verbose` | 0 | `trainer.py:271` |

### Walk-forward / `ENV_PARAMS`

| Constant | Equity | Crypto | Location |
|----------|-------:|-------:|----------|
| `test_bars` | 63 | 540 | `backtest.py:46, 53` |
| `min_train_bars` | 252 | 2,190 | `backtest.py:47, 54` |
| `embargo_bars` | 10 | 130 | `backtest.py:48, 55` |
| `periods_per_year` | 252 | 2,191.5 | `backtest.py:49, 56` |
| `bars_per_week` | 5 | 42 | `backtest.py:50, 57` |
| Min folds | 3 | 3 | `backtest.py:220-227` |

### Convergence callback

| Constant | Value | Location |
|----------|------:|----------|
| `min_improvement_pct` | 0.01 | `callbacks.py`, default in `trainer.py:276-280` |
| `patience` | 10 | same |
| `n_eval_episodes` | 5 | `trainer.py:285-294` |
| `eval_freq` | `total_timesteps // 10` | `trainer.py:288` |

### Validation gates

| Constant | Value | Location |
|----------|------:|----------|
| Sharpe floor | 0.7 | `validation.py:106` |
| MDD ceiling | 0.15 | `validation.py:112` |
| Profit factor floor | 1.5 | `validation.py:118` |
| Overfit gap ceiling | 0.20 | `validation.py:128` |
| Overfit "marginal" upper | 0.50 | `validation.py:63` |

### Bounds layer (`memory/training/bounds.py`)

`_ALGO_GAMMA_BOUNDS` — PPO/SAC `(0.95, 0.995)`, A2C `(0.95, 0.985)` (`bounds.py:44-48`). A2C mismatch-factor invariant `(1/(1−γ)) / n_steps < 8` (`bounds.py:237-254`). `MIN_TRAINING_PROGRESS=0.20`, `MAX_EPOCHS=200`. Reward-side max-deltas, cooldowns, and the PPO-crypto disable are documented in [`reward-shaping.md`](reward-shaping.md).

### Default / escalated timesteps

| Constant | Equity | Crypto | Location |
|----------|-------:|-------:|----------|
| `DEFAULT_TIMESTEPS` | 1,000,000 | 500,000 | `pipeline_helpers.py:46-47` |
| `ESCALATED_TIMESTEPS` | 2,000,000 | 1,000,000 | `pipeline_helpers.py:52-53` |

### Ensemble

`sharpe_softmax_weights` uses standard softmax with max-subtraction for numerical stability (`ensemble.py:44-49`). **No temperature parameter** — adding one is a code change. Risk-free rate hardcoded to `0.0` in `metrics.py:21-46` (no per-env override).

## Invariants

- All three algos share the same observation and action spaces (defined by `BaseTradingEnv` — see [`rl-environments.md`](rl-environments.md)). Training a fourth algo would need only HYPERPARAMS + SB3 class + seed — no env work.
- `VecNormalize` stats are **frozen** during evaluation (`training=False`) — eval data never mutates `obs_rms` / `ret_rms` (`backtest.py:521-528`). Train and eval normalizers are independent (intentional — eval distribution shift is part of the test signal).
- Walk-forward folds are **strictly sequential**, growing-window, with embargo only between train and test of the same fold — never between adjacent test windows (`backtest.py:210-218`).
- Validation gates are fail-closed: a single threshold failure blocks the entire `GateResult` (`validation.py:137`).
- Ensemble weights always satisfy `Σ = 1.0` and `weights ≥ 0` (softmax invariant). Missing algos collapse to weight 0 via Sharpe=0.0 default (`pipeline_helpers.py:242-247`).
- LLM-suggested HPs **never** reach SB3 without passing through `clamp_run_config` (`bounds.py:181-256`); LLM-suggested reward weights similarly via `clamp_reward_weights`.
- All metrics are computed from period returns arrays — no path-dependent metrics (avoids ambiguity when sliced).
- Risk-free rate = 0 throughout (`metrics.py:21-46`). Update both Sharpe and Sortino if this ever changes.

## Known issues / open questions

- **A2C `gae_lambda=0.92`** is intentionally lower than PPO's 0.95 to balance bias/variance with the very short `n_steps=5` rollout. Worth re-verifying after each major reward change.
- **Train/eval VecNormalize independence** — eval reward scale differs from training reward scale. Useful for fair eval but creates a mismatch when comparing `mean_reward` across the boundary.
- **Hardcoded validation gates** — Sharpe 0.7 / MDD 0.15 / PF 1.5 / overfit 0.20 are not yaml-tunable. Promote to `validation.*` config block if regimes start failing them systematically.
- **SAC buffer 200K vs 500K** — yaml currently runs at 200K (research found smaller buffers better for non-stationary markets). Schema default still 500K.
- **No ensemble temperature knob** — sharp ensemble weights when one algo dominates Sharpe; no way to soften without code change.
- **Risk-free rate fixed at 0** — fine for short-horizon swing trading; revisit if/when treasury yields become non-negligible relative to expected alpha.
- **PPO-crypto reward shaping disabled** (cross-link to [`reward-shaping.md`](reward-shaping.md)) but the underlying agent is still trained — disable only blocks LLM weight updates.
- **`_VALID_HP_KEYS` is per-algo** (`meta_orchestrator.py:57-80`) — when adding a new tunable HP, update both the bounds map and the per-algo whitelist or the suggestion is silently dropped.

## Source of truth

| Concern | File |
|---------|------|
| Training entry point + per-algo HPs + callbacks | `src/swingrl/training/trainer.py` |
| Convergence callback | `src/swingrl/training/callbacks.py` |
| Default / escalated timesteps + ensemble weight aggregation + gate composition | `src/swingrl/training/pipeline_helpers.py` |
| LLM HP / reward bounds + clamp + gamma bounds + A2C mismatch invariant | `src/swingrl/memory/training/bounds.py` |
| Meta-training orchestrator + LLM HP override pipeline | `src/swingrl/memory/training/meta_orchestrator.py` |
| Ensemble blender + softmax weights | `src/swingrl/training/ensemble.py` |
| Per-period metrics (Sharpe / Sortino / Calmar / Rachev / MDD / trade) | `src/swingrl/agents/metrics.py` |
| Walk-forward backtester + `ENV_PARAMS` + storage helpers | `src/swingrl/agents/backtest.py` |
| Validation gates + overfit diagnosis | `src/swingrl/agents/validation.py` |
| Shadow → active promotion | `src/swingrl/shadow/promoter.py` |
| pg16 DDL for `backtest_results` / `iteration_results` / `model_metadata` | `src/swingrl/data/postgres_schema.py` |
| Config schema (`TrainingConfig`, `MemoryAgentConfig`, `ShadowConfig`, `HyperparamBoundsConfig`, `RewardBoundsConfig`) | `src/swingrl/config/schema.py` |

## Changelog

- **2026-04-16** — Initial version.
- **2026-05-08** — Clarified MDD sign convention in validation gates section (positive-stored, absolute threshold).
