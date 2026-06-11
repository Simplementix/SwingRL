# Reward Shaping Reference

Living reference for SwingRL's reward system. Source-of-truth for the per-step reward formula, the component decomposition, the memory-guided shaping wrapper, and the LLM-driven weight-adjustment loop. Update this doc when any referenced module changes.

**Last verified against code:** 2026-04-16

## Reward pathways at a glance

| Pathway | When active | Formula | Wrapper |
|---------|-------------|---------|---------|
| Raw | `memory_client=None` in `trainer.train()` | `rolling_sharpe_20(daily_return) − risk_penalty` | none |
| Shaped | memory client wired (production) | `Σ weight_k × components[k]` for k ∈ {profit, sharpe, drawdown, turnover} | `MemoryVecRewardWrapper` |

**Stack order when shaping is active:** `Base VecEnv → MemoryVecRewardWrapper → VecNormalize` (`training/trainer.py:189-215`). The eval env mirrors this stack (`trainer.py:230-249`). On model save, the wrapper is unwrapped so the `VecNormalize` stats file is a plain `VecNormalize` instance (`trainer.py:510-531`).

**Fail-open:** if `MemoryVecRewardWrapper` construction raises, trainer logs `memory_reward_wrapper_failed` and falls back to a plain `Base → VecNormalize` stack (`trainer.py:217-220`).

## Raw reward (base env)

Computed in `BaseTradingEnv.step` (`envs/base.py:219-223`):

```
sharpe_reward = RollingSharpeReward(window=20).compute(daily_return)
risk_penalty  = _compute_risk_penalty(weights_after, new_value)
reward        = sharpe_reward − risk_penalty
```

**Timing (important):**

- `daily_return = (new_value − prev_value) / prev_value` — computed post-rebalance at new prices (`base.py:212-214`). Sharpe observes the post-trade return.
- `peak_value` is updated to `max(peak_value, new_value)` **before** the penalty is computed (`base.py:217`). On a fresh peak the drawdown penalty drops to zero immediately. Scope is **episode-level** — initialized to `initial_amount` at `base.py:115` and reset on each `env.reset()` at `:152`; it persists across steps within an episode but never crosses episode boundaries.
- `prev_value` is bumped to `new_value` at the end of `step()` (`base.py:255`).

### Rolling Sharpe (`envs/rewards.py`)

| Attribute | Value | Line |
|-----------|-------|------|
| Window | 20 bars (`deque(maxlen=20)`) | `rewards.py:26-28` |
| Std ddof | 1 (sample std) | `rewards.py:46` |
| Near-zero std guard | `1e-8` → returns `0.0` | `rewards.py:48` |
| Insufficient data | `len(returns) < 2` → returns `0.0` | `rewards.py:42-43` |
| Warmup → rolling | Expanding through bar 19, then rolls automatically via deque | `rewards.py:17-19, 28` |
| Reset | `deque.clear()` — next compute returns `0.0` | `rewards.py:53-55` |

Formula (`rewards.py:51`): `mean(returns) / std(returns, ddof=1)`.

### Risk penalty (`envs/base.py:331-354`)

```
penalty = Σ_i position_penalty_coeff × max(0, w_i − max_position_size)²     # quadratic
        + drawdown_penalty_coeff × max(0, drawdown − max_drawdown_pct)       # linear
```

- Position limit is per-env: `equity.max_position_size=0.25`, `crypto.max_position_size=0.50` (schema: `config/schema.py:54, 98`; read at `envs/base.py:74, 82`).
- Drawdown limit is per-env: `equity.max_drawdown_pct=0.10`, `crypto.max_drawdown_pct=0.12` (schema: `config/schema.py:55, 99`; read at `envs/base.py:75, 83`; applied at `base.py:345-352`).
- Drawdown term gated on `peak_value > 0` (`base.py:349`). Penalty is always ≥ 0.

## Reward components dict

Attached to `info["reward_components"]` on every `step` (not on `reset`). Computed in `base.py:226-236`:

| Key | Formula | Sign | Notes |
|-----|---------|------|-------|
| `profit` | `daily_return` | ± | = `(new_value − prev_value) / prev_value` |
| `sharpe` | `sharpe_reward` | ± | Output of `RollingSharpeReward.compute()` |
| `drawdown` | `−(peak_value − new_value) / peak_value` | **negative** or 0 | Negated at source (`base.py:234`); 0 if `peak_value ≤ 0` |
| `turnover` | `−cost / prev_value` | **negative** or 0 | Negated at source (`base.py:235`); 0 if `prev_value ≤ 0` |

**Critical:** these are **observations**, not reward terms in the raw pathway. The raw reward formula is `sharpe − risk_penalty`; it never reads `reward_components`. The dict exists so `MemoryVecRewardWrapper` can synthesize a shaped reward downstream.

## Shaped reward (`MemoryVecRewardWrapper`)

Source: `memory/training/reward_wrapper.py`. Wraps a `VecEnv` (base class `VecEnvWrapper`, `reward_wrapper.py:39`); overrides `step_wait` (89-107) and `reset` (109-118).

### Shaping math (`_shape_rewards`, `reward_wrapper.py:120-157`)

Per-env in the vectorized batch:

1. If `info["reward_components"]` is missing, not a `dict`, or shares no keys with `REWARD_COMPONENT_KEYS = ("profit","sharpe","drawdown","turnover")` → **pass raw reward through unchanged** (`reward_wrapper.py:141-146`).
2. Otherwise compute `weighted = Σ_k weight[k] × components.get(k, 0.0)` (missing keys default to `0.0`) and **replace the raw reward entirely** (`reward_wrapper.py:149-155`).

The raw `sharpe − risk_penalty` signal is **silently discarded** when shaping activates. `risk_penalty` has no representation in `reward_components` and therefore does not survive into the shaped pathway.

### Default weights

| Key | Default | File |
|-----|--------:|------|
| `profit` | 0.50 | `reward_wrapper.py:29` |
| `sharpe` | 0.25 | `reward_wrapper.py:30` |
| `drawdown` | 0.15 | `reward_wrapper.py:31` |
| `turnover` | 0.10 | `reward_wrapper.py:32` |

`_normalize_weights` (`reward_wrapper.py:225-241`) clamps to ≥ 0, renormalizes to sum=1.0, and on all-zero input logs `reward_weights_zero_using_defaults` and falls back to `DEFAULT_WEIGHTS` normalized.

### Initial-weights flow

```
LLM run_config (meta_orchestrator.py:159)
  → clamp_reward_weights(...) if non-empty else None   (meta_orchestrator.py:189-193)
  → trainer.train(initial_reward_weights=..., ...)     (meta_orchestrator.py:211)
  → MemoryVecRewardWrapper(initial_weights=..., periods_per_year=...) (trainer.py:206-210, 240-244)
  → dict(DEFAULT_WEIGHTS) | provided → _normalize_weights → self._weights  (reward_wrapper.py:68-73)
```

`periods_per_year` is sourced from `ENV_PARAMS[env_name]["periods_per_year"]` in `agents/backtest.py:44-58`: **252** (equity) or **2191.5** (crypto 4H).

### Rolling metrics (window 500)

Stored on the wrapper; populated in `step_wait` from **shaped** rewards (`reward_wrapper.py:103-105`), not raw. Cleared on `reset()`.

| Metric | Formula | File |
|--------|---------|------|
| `rolling_mean_reward` | `sum(history) / len(history)` | `reward_wrapper.py:205-213` |
| `rolling_sharpe` | `mean / std(ddof=1) × √periods_per_year` | `reward_wrapper.py:175-189` |
| `rolling_mdd` | `min(cumsum − cumulative_max(cumsum))` (negative) | `reward_wrapper.py:191-203` |
| `rolling_win_rate` | fraction of steps with shaped reward > 0 | `reward_wrapper.py:215-223` |
| `rolling_trade_rate` | mean of `info["trades_this_step"]` per step over rolling window; missing key counts as 0 | `reward_wrapper.py` |
| `baseline_trade_rate` | first-full-window `rolling_trade_rate`, locked permanently at fold start; 0.0 until window fills | `reward_wrapper.py` |

Edge cases: < 2 obs → `0.0` for Sharpe (181), empty history → `0.0` for MDD/mean/win-rate, `std < 1e-10` → `0.0` for Sharpe (186). Note the **wrapper's std floor `1e-10` differs from the env's `RollingSharpeReward` floor `1e-8`**.

`baseline_trade_rate` edge case: a fold whose first full window contains zero trades locks baseline at 0.0, permanently disabling the "trade-shy collapse" detector for that fold (`diagnose_rolling` treats baseline 0.0 as "window not yet full"). This is intentional and conservative — fail-safe over false alarms.

## Memory-driven weight adjustments

`MemoryEpochCallback` (`memory/training/epoch_callback.py`) ingests each rollout-end epoch, queries the LLM on a cadence, and applies clamped weight updates to the live wrapper.

### Trigger cadence & notable events

`_on_rollout_end` fires every SB3 rollout. Snapshot & advice gates:

- **Store snapshot** if `epoch % cadence == 0` **or** a notable event triggered (`epoch_callback.py:360-365`).
- **Notable events:** `approx_kl > 0.10` → `"kl_spike"`; `rolling_mdd < -25.0` → `"mdd_breach"` (`epoch_callback.py:76-77`).
- **Cadence:** loaded from yaml `memory_agent.epoch_cadence_{algo}` with hardcoded fallback in `ALGO_EPOCH_CADENCE` (PPO 60, A2C 8000, SAC 40000; unknown-algo fallback `EPOCH_STORE_CADENCE=500`) (`epoch_callback.py:40-45, 180-210`). Current yaml: **PPO 20, A2C 8000, SAC 40000, default 500** (`config/swingrl.yaml:108-111`).

### LLM advice path & guardrail chain

`_query_epoch_advice` (`epoch_callback.py:592-745`) gates in order:

1. **Cadence** — returns early if `epoch % cadence != 0` (`:599-600`).
2. **`advice_enabled`** — if False (typical for control folds), returns early (`:602-603`).
3. **Response validity** — must be `dict` and non-empty (`:649`).
4. **Cooldown** — reject if `timesteps_since_last_adjustment < get_adjustment_cooldown(algo)` (`:657-667`). Cooldowns: PPO 24,576 / A2C 500 / SAC 20,000 (`bounds.py:121-125`), default 5,000.
5. **Max-delta gate** — reject entirely if `get_max_reward_delta(algo, env) <= 0.0` (`:670-678`). This is how PPO-crypto is disabled (`bounds.py:110`).
6. **Bounds clamp** — `clamp_reward_weights(new_weights)` clips per-component to `REWARD_BOUNDS` and renormalizes to sum=1.0 (`bounds.py:259-297`).
7. **Change-detection floor** — skip if max absolute delta `< 0.01` (`:684-694`).
8. **Delta-cap scaling** — if any component's delta exceeds `get_max_reward_delta(algo, env)`, scale and renormalize (`:697-715`).
9. **Training-progress floor** — LLM-suggested `stop_training=True` is rejected if `progress < MIN_TRAINING_PROGRESS = 0.20` (`:628-637`, `bounds.py:97`).

LLM exceptions are caught; `_advice_timed_out` increments and training continues (fail-open, `:737-745`).

### Two-pass trigger/outcome

Adjustments are written to pg16 in two passes (`epoch_callback.py:3-8, 519-590`):

**Pass 1 — trigger (on apply):** `_ingest_adjustment_trigger` queues an INSERT with `run_id, epoch_trigger, algo, env, trigger_metric, trigger_value, trigger_reason, weight_before, weight_after, sharpe_at_trigger, mdd_at_trigger`. Also ingested to the memory store with source `reward_adjustment:{env}:{algo}` (`:508`).

**Pass 2 — outcome (10 epochs later):** `_resolve_pending_adjustment` UPDATEs the same row with `epoch_outcome, outcome_sharpe, sharpe_delta, mdd_delta, effective` (`:286-291`). `effective = sharpe_delta > 0 OR mdd_delta > 0` (`:549`). `ADJUSTMENT_RESOLVE_EPOCHS = 10` (`epoch_callback.py:79`). If a new adjustment arrives before resolution, the pending one is resolved early (`:718-724`).

All records are buffered in `_epoch_queue` / `_adjustment_trigger_queue` / `_adjustment_outcome_queue` during training and flushed in one transaction by `flush_telemetry()` after the fold completes (`:223-304`).

### Control folds

When `is_control_fold=True`, snapshots are still written (tagged `is_control_fold=True`) but the LLM is never called (typically paired with `advice_enabled=False`), so no adjustments occur. This is the scientific control arm for treatment-effect measurement.

## Configurable values (yaml)

All paths, defaults, and validators live in `src/swingrl/config/schema.py`.

### Raw-reward penalties & costs (`environment.*`)

| Key | Default | Validator | File:Line |
|-----|---------|-----------|-----------|
| `environment.initial_amount` | 100_000.0 | `gt=0` | `schema.py:185` |
| `environment.equity_transaction_cost_pct` | 0.0006 | `ge=0` | `schema.py:188` |
| `environment.crypto_transaction_cost_pct` | 0.0022 | `ge=0` | `schema.py:189` |
| `environment.signal_deadzone` | 0.02 | `ge=0, le=0.1` | `schema.py:190` |
| `environment.position_penalty_coeff` | 10.0 | `ge=0` | `schema.py:191` |
| `environment.drawdown_penalty_coeff` | 5.0 | `ge=0` | `schema.py:192` |

### Per-env risk caps (`equity.*` / `crypto.*`)

| Key | Equity default | Crypto default | File:Line |
|-----|---------------:|---------------:|-----------|
| `max_position_size` | 0.25 | 0.50 | `schema.py:54, 98` |
| `max_drawdown_pct` | 0.10 | 0.12 | `schema.py:55, 99` |

### LLM advice bounds (`training.bounds.reward_bounds.*`)

Used by `clamp_reward_weights` as the outer clip on LLM-suggested weights before renormalization.

| Key | Default (min, max) | Current yaml | File:Line |
|-----|-------------------:|:-------------|-----------|
| `profit` | (0.10, 0.70) | [0.10, 0.70] | `schema.py:434`, `swingrl.yaml:177` |
| `sharpe` | (0.10, 0.60) | [0.10, 0.60] | `schema.py:435`, `swingrl.yaml:178` |
| `drawdown` | (0.05, 0.50) | [0.05, 0.50] | `schema.py:436`, `swingrl.yaml:179` |
| `turnover` | (0.00, 0.20) | [0.00, 0.20] | `schema.py:437`, `swingrl.yaml:180` |

### Epoch-advice cadence (`memory_agent.epoch_cadence_*`)

| Key | Current yaml | File:Line |
|-----|-------------:|-----------|
| `memory_agent.epoch_cadence_ppo` | 20 | `swingrl.yaml:108` |
| `memory_agent.epoch_cadence_a2c` | 8000 | `swingrl.yaml:109` |
| `memory_agent.epoch_cadence_sac` | 40000 | `swingrl.yaml:110` |
| `memory_agent.epoch_cadence_default` | 500 | `swingrl.yaml:111` |

Fallback (hardcoded when yaml absent): PPO 60 / A2C 8000 / SAC 40000 / unknown 500 (`epoch_callback.py:40-45`).

## Hardcoded values (not yaml-tunable — code edit required)

### Raw-reward math

| Value | Location | Current |
|-------|----------|---------|
| `RollingSharpeReward` window | `rewards.py:26` | 20 |
| Sharpe `ddof` | `rewards.py:46` | 1 |
| Near-zero std guard | `rewards.py:48` | `1e-8` |
| Min returns before non-zero output | `rewards.py:42-43` | 2 |
| Position penalty shape | `base.py:344-346` | quadratic in excess weight |
| Drawdown penalty shape | `base.py:349-352` | linear in excess drawdown |
| Peak-value update point | `base.py:217` | before penalty |
| Trade-detection weight delta | `base.py:200` | `1e-8` |

### Shaped-reward wrapper

| Value | Location | Current |
|-------|----------|---------|
| `REWARD_COMPONENT_KEYS` | `reward_wrapper.py:25` | `("profit","sharpe","drawdown","turnover")` |
| `DEFAULT_WEIGHTS` | `reward_wrapper.py:28-33` | profit 0.50 / sharpe 0.25 / drawdown 0.15 / turnover 0.10 |
| `_ROLLING_WINDOW` | `reward_wrapper.py:36` | 500 |
| `rolling_sharpe` std floor | `reward_wrapper.py:186` | `1e-10` |
| Default `periods_per_year` | `reward_wrapper.py:55` | 252 (overridden to 2191.5 for crypto via `ENV_PARAMS`) |

### Adjustment guardrails (`bounds.py`)

| Value | Location | Current |
|-------|----------|---------|
| `_MAX_REWARD_DELTA["ppo"]` | `bounds.py:110` | equity 0.03 / crypto **0.0 (disabled)** |
| `_MAX_REWARD_DELTA["a2c"]` | `bounds.py:111` | equity 0.02 / crypto 0.05 |
| `_MAX_REWARD_DELTA["sac"]` | `bounds.py:112` | equity 0.02 / crypto 0.02 |
| `_DEFAULT_MAX_DELTA` | `bounds.py:114` | 0.03 |
| `_ADJUSTMENT_COOLDOWN_STEPS` | `bounds.py:121-125` | PPO 24,576 / A2C 500 / SAC 20,000 |
| `_DEFAULT_COOLDOWN` | `bounds.py:126` | 5,000 |
| `MIN_TRAINING_PROGRESS` | `bounds.py:97` | 0.20 |
| `MAX_EPOCHS` | `bounds.py:98` | 200 |
| `_FALLBACK_REWARD_BOUNDS` | `bounds.py:50-55` | same values as yaml defaults — used when yaml load raises |

### Callback thresholds

| Value | Location | Current |
|-------|----------|---------|
| `NOTABLE_KL_THRESHOLD` | `epoch_callback.py:76` | 0.10 |
| `NOTABLE_MDD_THRESHOLD` | `epoch_callback.py:77` | -25.0 |
| `ADJUSTMENT_RESOLVE_EPOCHS` | `epoch_callback.py:79` | 10 |
| Change-detection min delta | `epoch_callback.py:688` | 0.01 |

## pg16 telemetry tables

### `training_epochs` (one row per stored snapshot)

`run_id, epoch, algo, env, timestep, mean_reward, policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction, rolling_sharpe, rolling_mdd, rolling_win_rate, reward_weights (TEXT json), notable_event, is_control_fold, stop_training, rationale, created_at` (`postgres_schema.py:300-322`).

### `reward_adjustments` (one row per trigger, UPDATEd at outcome)

`run_id, epoch_trigger, epoch_outcome, algo, env, trigger_metric, trigger_value, trigger_reason, weight_before (TEXT json), weight_after (TEXT json), sharpe_at_trigger, mdd_at_trigger, sharpe_delta, mdd_delta, effective, outcome_sharpe, created_at` (`postgres_schema.py:338-359`).

Both tables are consumed by analysis notebooks / dashboards; `reward_adjustments` is the primary source for treatment-effect studies.

## Invariants

- Raw reward is always `sharpe − risk_penalty`; shaped reward **replaces** it (does not add). `risk_penalty` is therefore absent from the shaped pathway.
- Wrapper weights always satisfy `Σ weights = 1.0` and `weights ≥ 0` (post-normalization). Any zero-total input collapses to `DEFAULT_WEIGHTS`.
- Every LLM weight suggestion passes through `clamp_reward_weights` before reaching the wrapper — no path bypasses it.
- `reward_components` is attached only on `step()`, never on `reset()`. Callers must tolerate its absence on episode start.
- Rolling metrics on the wrapper reflect **shaped** rewards; the raw reward stream is not accessible downstream once shaping activates.
- Eval and train stacks share identical wrapper layering so metric comparisons are apples-to-apples (`trainer.py:230-249`).
- `REWARD_BOUNDS` is read once at `bounds.py` import time — yaml changes to `training.bounds.reward_bounds.*` require a container restart.

## Known issues / open questions

- **PPO-crypto shaping disabled** (`bounds.py:105, 110`) — iter-4 pattern analysis (patterns 157/163/169) found treatment folds underperforming control by ~29.5%; `_MAX_REWARD_DELTA["ppo"]["crypto"] = 0.0` blocks all updates. Research: `.planning/research/algo-reward-shaping.md`.
- **Risk-penalty invisibility under shaping** — once `MemoryVecRewardWrapper` activates, the agent's reward signal loses the position / drawdown penalty entirely unless `drawdown` component weight is raised. Current default weights put only 15% on drawdown and 0% on position limits.
- **Inconsistent std floors** — env `RollingSharpeReward` uses `1e-8` (`rewards.py:48`); wrapper `rolling_sharpe` uses `1e-10` (`reward_wrapper.py:186`). Probably benign but easy to miss.
- **Rolling-metric horizon vs adjustment cooldown** — wrapper metrics are over the last 500 shaped steps, but SAC's cooldown is 20,000 steps; the LLM sees a metric window that is 40× shorter than its minimum action interval for SAC.
- **Cadence drift PPO** — yaml cadence `epoch_cadence_ppo=20` (was 60) produces ~4 calls/fold; reverting to the code-level default of 60 would deliver ~1.4 calls/fold (comment in `epoch_callback.py:37`).
- **Sentiment variant** does not touch reward — noted here only to avoid confusion.

## Source of truth

| Concern | File |
|---------|------|
| Raw reward assembly | `src/swingrl/envs/base.py` |
| Rolling Sharpe | `src/swingrl/envs/rewards.py` |
| Shaping wrapper | `src/swingrl/memory/training/reward_wrapper.py` |
| Epoch callback (advice, guardrails, two-pass ingest) | `src/swingrl/memory/training/epoch_callback.py` |
| Bounds + clamp + cooldowns + max-deltas | `src/swingrl/memory/training/bounds.py` |
| Trainer wiring (stack order, fail-open, unwrap-on-save) | `src/swingrl/training/trainer.py` |
| Meta-orchestrator (initial weights flow) | `src/swingrl/memory/training/meta_orchestrator.py` |
| pg16 DDL | `src/swingrl/data/postgres_schema.py` |
| Config schema (`EnvironmentConfig`, `EquityConfig`, `CryptoConfig`, `RewardBoundsConfig`, `TrainingBoundsConfig`) | `src/swingrl/config/schema.py` |
| `periods_per_year` per env | `src/swingrl/agents/backtest.py` (`ENV_PARAMS`) |

## Changelog

- **2026-04-16** — Initial version.
- **2026-05-15** — Added file:line citations for `max_position_size` / `max_drawdown_pct` defaults (schema + read sites). Clarified `peak_value` scope as episode-level (reset on `env.reset()`).
- **2026-06-11** — Added `rolling_trade_rate` and `baseline_trade_rate` to the rolling-metrics table. These track per-fold trade activity using `info["trades_this_step"]` emitted by the envs (Task 4). `baseline_trade_rate` locks on the first full window so Task 8 mid-fold advice can detect "trade-shy collapse" against the fold's own starting rate.
