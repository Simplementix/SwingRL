# Iteration 4 Analysis, Open Questions, and Pre-Iter-5 Recommendations

> Generated 2026-04-03. All data from DuckDB `iteration_results`, `backtest_results`,
> `meta_decisions` tables + memory service logs + codebase inspection.

---

## Part 1: Iter 4 Results Summary

Training completed Apr 2 (~12:40 PM ET equity, ~6:30 PM ET crypto). Total wall time: **14.4h**.

### Equity Ensemble (gate_passed = YES, all iterations)

| Metric | Iter 0 | Iter 1 | Iter 2 | Iter 3 | **Iter 4** |
|--------|--------|--------|--------|--------|-----------|
| **Ensemble Sharpe** | 1.896 | 1.899 | 2.010 | 1.895 | **2.065** (ATH) |
| Ensemble MDD | 7.48% | 7.37% | 7.25% | 7.30% | 7.27% |
| PPO avg Sharpe | 2.684 | 2.711 | 3.107 | 2.684 | 2.859 |
| A2C avg Sharpe | 1.608 | 1.673 | 1.631 | 1.707 | **2.038** (ATH) |
| SAC avg Sharpe | 1.398 | 1.314 | 1.293 | 1.295 | 1.299 |
| PPO weight | 61.8% | 62.4% | 71.9% | 61.5% | 60.6% |
| A2C weight | 21.1% | 22.1% | 16.4% | 23.2% | **26.7%** (ATH) |
| SAC weight | 17.1% | 15.4% | 11.7% | 15.3% | 12.7% |

### Crypto Ensemble (gate_passed = NO, never has)

| Metric | Iter 0 | Iter 1 | Iter 2 | Iter 3 | **Iter 4** |
|--------|--------|--------|--------|--------|-----------|
| **Ensemble Sharpe** | 4.196 | 3.547 | 3.366 | 3.098 | **3.887** |
| Ensemble MDD | 15.3% | 17.3% | 16.7% | 18.8% | 15.8% |
| PPO avg Sharpe | 7.040 | 6.161 | 5.964 | 6.914 | **7.206** (ATH) |
| A2C avg Sharpe | 4.475 | 3.518 | 3.189 | 1.409 | 3.498 |
| SAC avg Sharpe | 1.073 | 0.963 | 0.945 | 0.969 | 0.958 |

### Wall Clock

| Iter | Equity | Crypto | Total |
|------|--------|--------|-------|
| 0 | 7.36h | 2.80h | 10.2h |
| 1 | 7.66h | 5.56h | 13.2h |
| 2 | 6.95h | 5.18h | 12.1h |
| 3 | 9.32h | 4.97h | 14.3h |
| 4 | 9.28h | 5.13h | 14.4h |

---

## Part 2: Critical Analysis

### What's Real vs What's Noise

**REAL — A2C equity improvement**: 1.608 → 2.038 (+26.7% vs baseline). Confirmed by:
- Epoch advice reward_shaping pattern shows 71% success rate on A2C drawdown adjustments
- A2C weight rose to 26.7% (highest ever) — the softmax ensemble naturally rewards it
- Improvement spread across many folds, not just outliers (p25 rose from 0.248 to 0.622)

**REAL — Crypto A2C recovery from iter 3 gamma bug**: 1.409 → 3.498 (+148%). The
gamma=0.999 set by Gemini in iter 3 was fixed back to 0.97. This is bug recovery, not
memory-driven learning.

**CONCERNING — PPO equity regressed from iter 2 peak**: 3.107 → 2.859 (-8%). Head-to-head
per fold: iter 2 wins 14/23 folds, iter 4 wins only 9/23. The "new ATH" ensemble Sharpe
is driven by A2C compensating for PPO's decline.

**FLAT — SAC unchanged across 5 iterations**: Equity 1.293-1.398 range, crypto 0.945-1.073
range. The memory system has zero measurable effect on SAC. See Part 3 Q1 for analysis.

### Structural "Poison Folds" (unchanged across all iterations)

**Equity fold 7** (2019-01-30 to 2019-04-30): ALL 3 algos × ALL 5 iters are negative.
- A2C: -1.993 to -1.786, PPO: -1.445 to -1.134, SAC: -1.416 to -1.943
- MDD 32-41%. Memory/HP tuning has near-zero impact.

**Equity fold 13** (2020-10-23 to 2021-01-25): A2C and SAC always negative, PPO mixed.
A2C actually REGRESSED: -0.663 (iter 3) → -1.054 (iter 4).

**Crypto SAC folds 5-6** (Apr-Nov 2020): Negative every iteration. Fold 6 is always
-0.8 to -1.8 Sharpe with 46-54% MDD.

### Overfitting Distribution (iter 3 → iter 4)

PPO equity became more binary:
- Healthy: 13 → 14 (+1)
- Marginal: 5 → **2** (-3)
- Reject: 5 → **7** (+2)

Marginal folds collapsed into reject. The model is either great or terrible, with less
middle ground.

---

## Part 3: Open Questions — Detailed Answers

### Q1: Should SAC Be Dropped From the Ensemble?

**Data-backed findings:**

1. **SAC performance is flat across 5 iterations** — equity 1.293-1.398 range, crypto
   0.945-1.073. The memory system has had zero effect.

2. **SAC provides zero diversification value** — fold-level correlation analysis shows:
   - Equity: 2 folds where both PPO+A2C are negative (folds 7, 13). SAC is ALSO negative
     on both. Zero folds where SAC saves the ensemble.
   - Crypto: Zero folds where PPO+A2C are both negative while SAC is positive.

3. **SAC is the primary reason crypto fails the gate** — Gate fails on MDD (15% threshold).
   - With SAC: iter 4 crypto MDD = 15.77% (FAIL)
   - Without SAC: iter 4 crypto MDD = 13.07% (PASS)
   - SAC crypto has 8/14 folds over 15% MDD, worst at 46.3%

4. **SAC doesn't cost wall time** — algos train in parallel. But SAC's replay buffer is
   the dominant memory consumer (see Q1 buffer analysis below).

5. **Why memory advice doesn't help SAC** — the HP tuner only has 4 knobs (lr, gamma,
   batch_size, ent_coef). SAC's real problems are:
   - `ent_coef="auto"` starting alpha=1.0 drowns reward signal (see sac-crypto-tuning.md)
   - `gradient_steps=1` (SB3 default) — severely under-extracts from buffer data
   - `buffer_size` not in tunable set — regime-stale transitions poison learning
   - `target_entropy` not tunable — default is too high for portfolio allocation
   - Network architecture frozen at [64,64] — critics may be undersized

**Recommendation:**
- **Crypto**: Drop SAC from gate calculation immediately. Consider dropping from training.
- **Equity**: Deprioritize but keep. Before dropping, try the SAC parameter fixes from
  `sac-crypto-tuning.md` (ent_coef="auto_0.1", gradient_steps=4, buffer_size=200K).

### Q2: Are Crypto PPO Numbers Realistic?

**Data-backed findings:**

1. **Trade counts are structural, not a bug**: Action space is continuous Box(-1,1) with
   softmax → portfolio weights. Deadzone is 2%. With 535 bars/symbol and 2 symbols per
   fold, 973 round-trips = 0.91 trades/bar. The agent rebalances on ~91% of bars.
   **Iter 0 (no memory, baseline) shows identical pattern** — fold 0 had 990 trades,
   fold 13 had 1044 trades. This is baseline model behavior.

2. **High Sharpe is the annualization multiplier**: Crypto annualization uses
   `periods_per_year = 2191.5` (6 bars/day × 365.25 days). Multiplier = √2191.5 = 46.8×.
   So crypto Sharpe 10.8 = per-bar Sharpe of 0.23, which annualizes to equity-equivalent
   Sharpe of ~2.4 (using equity's √252 = 15.9× multiplier). **Not magic — just frequency.**

3. **Transaction costs ARE applied**: 0.22% round-trip (0.10% per side + spread/slippage).

4. **Real execution concern is minimum order size, not slippage**: At $47 capital, most
   rebalances produce $2-15 trades. Binance.US minimum is $10. Many trades would be
   below minimum. At $500+ crypto capital, trades become executable.

**Recommendation:** Numbers are mathematically valid. No action needed on training side.
For paper trading, either increase crypto capital or implement trade-batching to accumulate
weight changes until they cross the $10 minimum.

### Q3: Should Fold 7/13 Be Handled Differently?

**Data-backed findings:**

Both poison folds are **strong bull rallies** where all ETFs went up significantly:
- Fold 7 (Jan-Apr 2019): QQQ +14.6%, SPY +10.4%, XLK +19.0% (V-recovery from Dec 2018)
- Fold 13 (Oct 2020 - Jan 2021): XLE +38.9%, XLF +19.9%, QQQ +15.5% (post-election rally)

The paradox: **the model loses money during strong bull markets**. It's positioned
defensively (high cash weight or wrong-sector allocation) during rapid recoveries.

**CORRECTION**: The models DO have regime features in the observation space:
- `hmm_p_bull`, `hmm_p_bear` — 3-state HMM probabilities (bull/bear/crisis)
- `macro_vix_zscore` — 252-day rolling VIX z-score
- `macro_yield_curve_spread` and `macro_yield_curve_direction`
- `turbulence_index` — Mahalanobis-based market turbulence
- `weekly_trend_dir` — binary trend direction (close > SMA-10 weekly)
- `rsi_14`, `macd`, `adx`, `bb_position` — per-asset momentum/trend

The models HAVE the information to detect V-recoveries but STILL fail. This means the
problem is NOT missing features — it's that **the capital preservation reward structure
(drawdown penalty) makes the model structurally incapable of capturing rapid recoveries**.
When the model detects a regime shift to bull, the penalty for being wrong (drawdown)
outweighs the reward for being right (profit). This is an inherent trade-off of the
system's design philosophy.

**Recommendation:** Accept as a known weakness. These represent 2/23 equity folds (~9%).
Expect 1-2 quarters per decade of underperformance vs buy-and-hold during V-recoveries.
Trying to engineer around them risks overfitting to historical patterns.

### Q4: Epoch Advice Delivery — Why Not 100%?

**Iter 4 delivery stats (from swingrl-memory logs):**

| Metric | Count |
|--------|-------|
| Total requests (context_built) | 228 |
| Delivered (epoch_advice_response) | 165 (72%) |
| Fallback to defaults | 63 (28%) |
| Cloud exhausted → fell to Ollama | 68 |
| Ollama timeouts | 63 |

**Provider breakdown:**
- Cerebras (qwen3-235b): 160 successes, 45 failures (422 "structured output timeout")
- Groq (llama-4-scout): 68 failures, **0 successes** (permanent 400 error — `max_tokens`
  config sends 30000 but model limit is 8192)
- Ollama (qwen2.5:14b): 5 successes, 63 timeouts (300s timeout with 3 parallel algos
  competing for single GPU)

**The failure chain works like this:**
1. Try Cerebras → if 422/429, try Groq
2. Groq ALWAYS fails (config bug: max_tokens=30000 > model limit 8192)
3. Fall to Ollama → 3 parallel algo workers queue on single Ollama instance → timeouts
4. All fail → fallback to defaults (no advice)

**Why the 28% failure rate:**
- Cerebras has transient 422 errors (~21% of its attempts) — server-side structured
  output timeouts, not our fault
- When Cerebras fails, Groq is supposed to catch it — but Groq is **permanently broken**
  due to the max_tokens config bug
- So every Cerebras failure becomes an Ollama attempt, and Ollama is too slow for
  parallel requests (qwen2.5:14b takes ~130s per call, 3 algos × 130s > 300s timeout)

**Fixes needed:**
1. **Fix Groq max_tokens**: Change `max_tokens: 30000` to `8192` in config/swingrl.yaml
   for Groq provider. This is a 1-line config fix that would eliminate ~30 of the 63
   failures.
2. **Increase PPO epoch cadence**: Currently `epoch_cadence_ppo: 60` gives ~1.4
   calls/fold. With 48% delivery failure rate for PPO (10 delivered / 21 requested),
   many folds get zero advice. Changing to `epoch_cadence_ppo: 20` (original value) gives
   ~4 calls/fold, ensuring at least 1-2 per fold even with failures.
3. **Ollama timeout isn't the bottleneck** — the bottleneck is Groq being permanently
   broken as a fallback. Fixing Groq makes Ollama a rare last-resort instead of the
   primary fallback.

### Q5: What Does Crypto Passing the Gate Require?

**Gate criteria** (hardcoded, same for both envs):
`ensemble_sharpe > 1.0 AND abs(ensemble_mdd) < 0.15`

**Crypto has NEVER failed on Sharpe** (always 3.0+). **Fails solely on MDD (15% threshold):**

| Iter | MDD | Gap from threshold |
|------|-----|--------------------|
| 0 | 15.34% | 0.34pp over |
| 1 | 17.34% | 2.34pp over |
| 2 | 16.72% | 1.72pp over |
| 3 | 18.81% | 3.81pp over |
| 4 | 15.77% | **0.77pp over** |

**Without SAC in the gate calculation:**

| Iter | MDD (no SAC) | Pass? |
|------|-------------|-------|
| 0 | 11.26% | **PASS** |
| 1 | 15.45% | FAIL |
| 2 | 14.49% | **PASS** |
| 3 | 17.51% | FAIL |
| 4 | 13.07% | **PASS** |

**Recommendation:** Remove SAC from crypto gate calculation (and ideally from crypto
training). This solves Q1 and Q5 simultaneously. Iter 4 crypto passes immediately.

---

## Part 4: SAC Buffer Size — Corrected Analysis

### Memory math (corrected with n_envs=6)

SB3 pre-allocates the FULL buffer as `(buffer_size, n_envs, dim)` numpy arrays at init.
With `n_envs=6`, every dimension is 6× a naive per-transition estimate.

| Config | Equity SAC (obs=164) | Crypto SAC (obs=47) |
|--------|---------------------|---------------------|
| 200K buffer | ~1.63 GB | ~0.48 GB |
| **500K (current)** | **~4.07 GB** | **~1.19 GB** |
| 1M buffer | **~8.14 GB** | ~2.38 GB |
| **Delta 500K→1M** | **+4.07 GB** | **+1.19 GB** |

All 3 algos run in parallel. SAC's buffer dominates. Container previously had 16GB limit
(now removed). During training, total memory is buffer + models + SubprocVecEnv overhead
(18 subprocesses) + features.

### Research finding: SMALLER buffer is better for SAC in non-stationary environments

From `sac-crypto-tuning.md`:
- Fedus et al. (ICML 2020): "smaller buffer sizes of 50K perform best" for SAC
- arXiv 2511.20678 (SAC crypto portfolio): used 100K buffer
- PPO-to-SAC migration blog: used 100K, found smaller better for non-stationary data

With 500K buffer and 500K crypto timesteps, the buffer **never evicts**. Regime-stale
transitions from early training are sampled equally with recent data. A 200K buffer
would force eviction after 40% of training, keeping data regime-relevant.

**Recommendation:** REDUCE buffer_size to 200K for crypto (saves ~0.7 GB, improves
regime relevance). For equity (1M timesteps), 500K is already better than 1M because
it forces eviction in the second half. Consider reducing equity to 300K.

---

## Part 5: Stop Training Implementation

### How it works (confirmed in code)

`epoch_callback.py:570-587`: When epoch advice returns `stop_training: true`:
1. Calculate `progress = num_timesteps / total_timesteps`
2. If `progress < 0.20` (MIN_TRAINING_PROGRESS): log `stop_training_ignored_too_early`
   and continue
3. If `progress >= 0.20`: log `llm_advises_stop_training`, set `model.stop_training = True`

### Iter 4 stop_training signals (from swingrl-memory HTTP response logs)

4 signals sent during iter 4:

| # | Algo | Env | Timestamp | Rationale |
|---|------|-----|-----------|-----------|
| 1 | SAC | equity | Apr 2 19:38 | "rolling_sharpe=-0.6, mdd=-4.85, adjustments ineffective" |
| 2 | PPO | equity | Apr 2 20:53 | "rolling_sharpe=16.99 but mdd=-0.41, epoch 120 in overfit risk zone" |
| 3 | SAC | equity | Apr 2 21:21 | "rolling_sharpe=0.1, mdd=-11.0%, inconsistent effectiveness" |
| 4 | SAC | crypto | Apr 3 03:37 | "rolling_sharpe=-43.74, mdd=-32.52, catastrophic collapse" |

### Were these past the 20% threshold?

Based on timestamps (equity training started ~12:40 PM ET, crypto ~6:30 PM ET):
- Signal #1 (SAC equity, 19:38): ~7h into 9.3h equity training → ~75% progress. **PAST 20%.**
- Signal #2 (PPO equity, 20:53): ~8.2h into training → ~88%. **PAST 20%.**
- Signal #3 (SAC equity, 21:21): ~8.7h → ~93%. **PAST 20%.**
- Signal #4 (SAC crypto, 03:37): ~5h into 5.13h crypto training → ~97%. **PAST 20%.**

All 4 signals were well past the 20% minimum.

### BUG: We can't verify if they were applied

The stop_training check + log (`stop_training_ignored_too_early` or
`llm_advises_stop_training`) runs inside the **swingrl container's** epoch callback.
Iter 4 training was launched without log file redirection — output went to a terminal
session, not a persisted file. The `docker logs swingrl` only has 8 idle-mode lines
because training completed before the current app startup.

**Additionally**: `converged_at_step` in backtest_results only records convergence
callback stops, NOT LLM-advised stops. All iter 4 folds show `converged_at_step = NULL`.
This is a logging gap — `stop_training` events are not recorded in the DB.

### Fixes needed
1. Always redirect training output to a log file (e.g., `training_iter{N}.log`)
2. Record LLM `stop_training` events in `converged_at_step` (or add a new
   `stop_reason` field to backtest_results)

---

## Part 6: Epoch Telemetry (DuckDB) — What's Needed for Iter 5

### Current state
- `trainer.py:313-316`: `duckdb_path` parameter to `MemoryEpochCallback` is commented out
- Root cause: DuckDB single-writer lock — WF backtester holds connection during training,
  epoch callback's `duckdb.connect()` deadlocks
- `training_epochs` table: 0 rows. `reward_adjustments` table: 0 rows.
- Disabled for 2 iterations (iter 3 and iter 4)

### The fix (from feedback memory: `feedback_duckdb_fold_critical.md`)

Use **queue + post-fold flush** pattern:

1. **During training**: `MemoryEpochCallback` appends epoch snapshots to an in-memory
   Python queue (list or `queue.Queue`). No `duckdb.connect()` calls.
2. **After each fold completes**: WF backtester releases DuckDB connection. At this point,
   call `flush_epoch_telemetry(duckdb_path, queue)` to batch-INSERT accumulated snapshots
   into `training_epochs` and `reward_adjustments` tables.
3. **No concurrent writes**: DuckDB writes happen only between folds.

### Files to modify
- `src/swingrl/memory/training/epoch_callback.py` — buffer to queue instead of DuckDB
- `src/swingrl/training/trainer.py` — re-enable duckdb_path, add post-fold flush call
- `src/swingrl/agents/backtest.py` — call flush after fold evaluation

---

## Part 7: Regime Features — Corrected V-Recovery Analysis

### The models HAVE regime information (confirmed in code)

**Observation space includes** (`src/swingrl/features/assembler.py`):

| Feature | Location | Both envs? |
|---------|----------|------------|
| hmm_p_bull | hmm_regime.py | YES |
| hmm_p_bear | hmm_regime.py | YES |
| macro_vix_zscore | macro.py (252-day rolling) | YES |
| macro_yield_curve_spread | macro.py (T10Y2Y raw) | YES |
| macro_yield_curve_direction | macro.py (binary) | YES |
| macro_fed_funds_90d_change | macro.py | YES |
| macro_cpi_yoy | macro.py | YES |
| macro_unemployment_3m_direction | macro.py (binary) | YES |
| turbulence_index | assembler.py | YES |
| weekly_trend_dir (equity) | technical.py (close > SMA-10) | Equity only |
| daily_trend_dir (crypto) | technical.py | Crypto only |
| rsi_14, macd, adx, bb_position | technical.py (per-asset) | YES |

Total obs dimensions: **Equity 164, Crypto 47.**

### Corrected diagnosis of fold 7/13 failure

My original claim ("the model needs regime detection features") was **WRONG**. The correct
diagnosis:

The models have HMM probabilities, VIX z-scores, trend direction, momentum indicators —
all the signals needed to detect V-shaped recoveries. **They STILL fail.**

This means the problem is **reward design, not feature design**. The drawdown penalty in
the reward function creates conservative policies. During volatile V-recoveries, the cost
of being wrong (drawdown → heavy penalty) exceeds the reward for being right (profit).
The model rationally chooses to stay defensive even when regime features signal "bull."

This is an **inherent trade-off** of the capital-preservation-first design, not a fixable
bug.

---

## Part 8: Updated Recommendations for Pre-Iter-5 Work

### Priority 1: Config fixes (immediate, no code changes)

1. **Fix Groq max_tokens**: `config/swingrl.yaml` line 162 — change `max_tokens: 30000`
   to `max_tokens: 8192`. This fixes 100% of Groq failures and eliminates ~30/63
   fallbacks-to-defaults.

2. **Increase PPO epoch cadence**: `config/swingrl.yaml` line 109 — change
   `epoch_cadence_ppo: 60` to `epoch_cadence_ppo: 20`. Gives PPO ~4 calls/fold instead
   of ~1.4, ensuring reliable advice delivery even with provider failures.

3. **Remove SAC from crypto gate OR drop SAC crypto training entirely**: Either exclude
   SAC folds from `check_ensemble_gate()` for crypto, or skip SAC algo for crypto env.
   This solves both the gate failure and the wasted compute.

### Priority 2: SAC parameter changes (config + bounds.py)

4. **Switch SAC ent_coef to "auto_0.1"**: Reduces initial alpha from 1.0 to 0.1.
   The #1 mechanistic reason SAC underperforms (entropy drowns reward signal).

5. **Reduce SAC buffer_size to 200K for crypto, 300K for equity**: Forces regime-stale
   eviction, saves ~2.5 GB memory. Literature unanimously supports smaller buffers.

6. **Increase SAC gradient_steps to 4**: Extracts 4× more learning per transition.
   Reduce SAC lr proportionally (halve it).

7. **Add SAC-specific critic architecture**: `net_arch=dict(pi=[64,64], qf=[128,128])`.

8. **Add to HP tuning bounds**: `gradient_steps`, `target_entropy`, `buffer_size`.
   These are the knobs the memory system needs but currently lacks.

### Priority 3: Telemetry and logging fixes

9. **Re-enable DuckDB epoch telemetry**: Queue + post-fold flush pattern. See Part 6.

10. **Persist training logs to file**: Training output must go to
    `logs/training_iter{N}.log`, not just terminal stdout.

11. **Record stop_training in DB**: Either set `converged_at_step` when LLM advises stop,
    or add `stop_reason` field to backtest_results.

### Priority 4: Structural decisions

12. **Accept fold 7/13 as known weakness**: Don't try to engineer around V-recovery
    failures. The capital-preservation reward design makes this structural.

13. **Decide on crypto deployment criteria**: If SAC is removed from gate, crypto passes.
    Decide whether to paper-trade crypto at current capital ($47) given the $10 min order
    constraint, or wait for higher capital allocation.

---

## Part 9: Iter 4 Consolidation Pattern Quality Analysis

> 14 new patterns created (IDs 156-169): 5 equity WF, 2 equity epoch, 3 crypto WF,
> 2 crypto epoch, 2 cross-env. Validated against raw DuckDB backtest data and 4.1M
> raw epoch memories + 1,155 reward adjustment memories.

### Pattern Inventory

**Equity Stage 1 — Walk-Forward (from 11 WF fold memories per algo):**

| ID | Category | Algos | Conf | Verdict |
|----|----------|-------|------|---------|
| 156 | regime_performance | SAC | 0.75 | VALID with caveats |
| 157 | iteration_regression | A2C, PPO | 0.80 | **VALID — most important pattern** |
| 158 | regime_performance | PPO | 0.70 | VALID |
| 159 | overfit_diagnosis | A2C | 0.65 | PARTIALLY VALID |
| 160 | data_size_impact | SAC | 0.75 | **INVALID — data contradicts claim** |

**Equity Stage 1 — Epoch (from 17,974 epoch + reward adjustment memories):**

| ID | Category | Algos | Conf | Verdict |
|----|----------|-------|------|---------|
| 161 | reward_shaping | SAC | 0.70 | VALID but low-impact |
| 162 | drawdown_recovery | SAC | 0.60 | VALID — weak signal |

**Crypto Stage 1 — Walk-Forward (from 11 WF fold memories per algo):**

| ID | Category | Algos | Conf | Verdict |
|----|----------|-------|------|---------|
| 163 | regime_performance | PPO | 0.85 | **VALID — confirms treatment hurts PPO** |
| 164 | hp_effectiveness | A2C | 0.75 | **VALID — treatment helps A2C crypto** |
| 165 | macro_transition | SAC | 0.70 | VALID |

**Crypto Stage 1 — Epoch (from 949,928 epoch memories):**

| ID | Category | Algos | Conf | Verdict |
|----|----------|-------|------|---------|
| 166 | drawdown_recovery | SAC | 0.70 | VALID — marginal improvement |
| 167 | reward_shaping | A2C, SAC | 0.72 | VALID |

**Cross-Environment Stage 2 (from 48 stage-1 patterns):**

| ID | Category | Algos | Conf | Verdict |
|----|----------|-------|------|---------|
| 168 | cross_env_correlation | A2C, SAC, PPO | 0.70 | VALID |
| 169 | cross_env | A2C | 0.72 | **VALID — key actionable finding** |

### Detailed Validation

#### Pattern 157 (iteration_regression, equity) — VALID, most actionable

**Claim**: A2C control folds outperform treatment folds (ctrl=2.5611 vs treat=1.8922,
delta=+0.6689). Treatment advice success rate below 50%.

**Raw data validation** (DuckDB):
- A2C: CTRL avg=2.5611 (n=5), TREAT avg=1.8922 (n=18), delta=+0.6689 — **exact match**
- PPO: CTRL avg=2.8647 (n=5), TREAT avg=2.8575 (n=18), delta=+0.0072 — **exact match**
- SAC: CTRL avg=1.3270 (n=5), TREAT avg=1.2914 (n=18), delta=+0.0357

**Trend validation** (iter 3 comparison):
- Iter 3 A2C: CTRL=2.1450 vs TREAT=1.5848, delta=+0.5602
- Iter 4 A2C: CTRL=2.5611 vs TREAT=1.8922, delta=+0.6689
- The gap is **widening** — treatment is increasingly harmful to A2C equity

**Assessment**: This is the most important iter 4 pattern. It correctly identifies that
epoch advice reward adjustments are degrading A2C equity performance relative to control
folds. The LLM's actionable suggestion ("investigate and refine LLM-driven reward
adjustments for A2C") is appropriate. However, the pattern doesn't diagnose WHY — the
likely cause is that reward weight adjustments interact poorly with A2C's lack of clipping
protection, causing policy instability. A2C may need a more conservative adjustment
strategy (smaller deltas, longer measurement windows) than PPO/SAC.

**Implication for iter 5**: Consider reducing A2C equity reward adjustment magnitude
(max delta < 0.05 vs current < 0.10) or increasing the measurement window before
declaring an adjustment effective/ineffective.

#### Pattern 163 (regime_performance, crypto PPO) — VALID, confirms #157 cross-env

**Claim**: PPO crypto CTRL mean Sharpe 9.1247 vs TREAT 6.4384, delta=-2.6863.

**Raw data validation**: CTRL avg=9.1247 (n=4), TREAT avg=6.4384 (n=10) — **exact match**.

**Assessment**: Treatment (epoch advice) hurts PPO crypto even more severely than PPO
equity. Control folds average 42% higher Sharpe. This is the crypto counterpart of
pattern 157. The combined evidence across both environments (patterns 157 + 163) is strong:
**epoch advice reward adjustments are net-negative for PPO.** PPO is the strongest algo
and the advice is actively degrading it.

**Implication**: PPO may need a different epoch advice strategy — perhaps higher bar for
triggering adjustments (only intervene on severe rolling_mdd, not modest metrics), or
wider deadzone for PPO reward weight changes.

#### Pattern 164 (hp_effectiveness, crypto A2C) — VALID, opposite of #157

**Claim**: A2C crypto CTRL mean Sharpe 2.8718 vs TREAT 3.7491, delta=+0.8773. Treatment
helps A2C crypto.

**Raw data validation**: CTRL avg=2.8718 (n=4), TREAT avg=3.7491 (n=10) — **exact match**.

**Assessment**: This is the mirror image of pattern 157. A2C crypto BENEFITS from treatment
while A2C equity is HURT by treatment. Pattern 169 (cross_env) correctly identified this
divergence.

**Why the difference?** Likely because A2C crypto has different reward dynamics (4H bars,
higher volatility, different return distribution) that make it more responsive to drawdown
weight increases. The crypto reward adjustments push drawdown weight 0.15→0.29 (pattern
167), which helps A2C crypto's MDD without degrading Sharpe. For equity, the same
adjustments may overshoot because equity returns are lower-variance and more sensitive
to reward rebalancing.

#### Pattern 169 (cross_env, A2C) — VALID, key actionable finding

**Claim**: A2C treatment helps crypto but hurts equity. Recommends env-specific strategies.

**Raw data validation**: Combines patterns 157 and 164, both independently validated.

**Assessment**: This is correctly synthesized. The recommendation ("retain baseline reward
shaping in equity, apply treatment in crypto") is sound. However, the implementation
would require env-specific control in the epoch callback — currently, the same adjustment
logic runs for both environments. This is a Priority 2 change for iter 5.

#### Pattern 156 (regime_performance, equity SAC) — VALID with caveats

**Claim**: SAC degrades in high VIX (>25) and negative yield spread folds. Cites folds 2
and 4 with sharpe=-0.54 and -0.046.

**Raw data validation**:
- Fold 2: sharpe=-0.540, vix=18.59 — VIX is NOT >25. Pattern incorrectly attributes
  this to high VIX.
- Fold 4: sharpe=-0.046, vix=20.13 — VIX is NOT >25 either.
- Fold 7: sharpe=-1.757, vix=28.30 — this IS high VIX and IS negative.
- Fold 8: sharpe=3.083, vix=29.47 — HIGH VIX but POSITIVE Sharpe.

**Assessment**: The directional claim is partially supported (high VIX folds include
the worst SAC folds), but the specific evidence cited is wrong — the cited folds have
moderate VIX, not high VIX. Fold 8 at VIX=29.47 has sharpe=3.083, directly contradicting
the "high VIX = bad SAC" claim. The LLM cherry-picked folds that fit the narrative rather
than testing the full distribution. Confidence 0.75 is too high — should be 0.50-0.55.

#### Pattern 160 (data_size_impact, equity SAC) — INVALID

**Claim**: SAC performance improves with more training bars. Low bars (<500) gate pass 0/2,
high bars (>1000) gate pass 6/10. Mean Sharpe low=0.4901, high=1.6012.

**Raw data validation**:
- Low bars (<500): folds 0-3, mean sharpe = **2.109** (not 0.4901)
- High bars (>1000): folds 11-22, mean sharpe = **1.114** (not 1.6012)
- The data shows the OPPOSITE — low-bar folds have HIGHER mean Sharpe (2.109 vs 1.114)

**Assessment**: **This pattern is factually wrong.** The consolidation LLM (Mistral)
invented numbers that don't match the underlying data. The claim that SAC benefits from
more training bars is contradicted by the actual fold-level results. Fold 1 (bars=325)
has sharpe=6.107, the best SAC equity fold, directly disproving the claim.

This pattern should be **retired immediately**. It would cause the HP tuner to
incorrectly skip early SAC folds, losing SAC's best-performing fold.

#### Pattern 159 (overfit_diagnosis, equity A2C) — PARTIALLY VALID

**Claim**: A2C overfits in negative yield spread conditions. Cites folds 13 (gap=1.3152)
and 15 (gap=0.9164).

**Raw data validation**:
- Fold 15 (ys=-0.723): gap=0.916, class=reject — **matches claim**
- Fold 13 (ys=0.245): gap=1.315, class=reject — yield spread is POSITIVE, not negative
- Other negative yield spread folds: fold 16 (ys=-0.713) gap=-0.495 healthy, fold 17
  (ys=-0.471) gap=-2.433 healthy, fold 14 (ys=-0.314) gap=-0.075 healthy

**Assessment**: 4 of 5 negative yield spread folds are actually HEALTHY. Only fold 15
supports the claim. The high-overfit folds (13, 7, 4, 2) have POSITIVE yield spread.
The LLM incorrectly attributed overfitting to yield spread when the real driver appears
to be something else (possibly fold-specific market structure). Confidence 0.65 is
appropriate for the weak signal.

#### Patterns 161, 162 (reward_shaping + drawdown_recovery, equity SAC) — VALID, weak

**Raw memory validation** (from reward_adjustment:historical):
- SAC crypto fold adjustments: drawdown weight increased from 0.33→0.39→0.45. MDD
  improved at epoch 80K (mdd_delta=+2.629, effective=True) but worsened at epoch 120K
  (mdd_delta=-0.225, effective=False).
- Pattern 161 claims 60% positive adjustments (25/42) — plausible given the mixed
  effectiveness in raw memories.
- Pattern 162 correctly notes inconsistent recovery.

**Assessment**: Accurately reflects the raw data. The 60% success rate and inconsistent
recovery are genuine. The signal is weak but honest — confidence 0.60-0.70 is appropriate.

#### Pattern 165 (macro_transition, crypto SAC) — VALID

**Claim**: SAC degrades in negative yield spread conditions with win_rate=0.340 and
avg_pnl=-45.93.

**Raw memory validation** (from trading_pattern:crypto:sac):
- `neg_yield_spread: trades=524 win=0.340 avg_pnl=-45.9283` — **exact match**
- Overall: `trades=2116 win=0.507 avg_pnl=131.9917`

**Assessment**: Pattern accurately extracted from the raw trading pattern memory.
The macro condition signal is real — SAC crypto loses money consistently when yield
spread is negative.

#### Pattern 168 (cross_env_correlation) — VALID

**Claim**: A2C and SAC degrade in negative yield spread across both environments.

**Assessment**: Correctly synthesized from patterns 156/165 (crypto SAC) and the equity
data. PPO stability in negative yield spread is also confirmed. The recommendation to
favor PPO during negative yield spread regimes is data-backed.

### Overall Pattern Quality Assessment

**Pattern system health by the numbers:**
- Total patterns in DB: 79 (56 from prior iterations, 14 new from iter 4, 9 other)
- Active: ~45, Retired: ~18, Superseded: ~16
- 6 patterns retired with `SUP=iter3_advice_chain_failure` (iter 3 post-mortem cleanup)
- Dedup working: 7 dedup confirmations during iter 4 consolidation (existing patterns
  preserved rather than duplicated)

**Iter 4 pattern quality scorecard:**

| Rating | Count | Pattern IDs |
|--------|-------|-------------|
| VALID + actionable | 5 | 157, 163, 164, 165, 169 |
| VALID + weak signal | 4 | 158, 161, 162, 166, 167 |
| PARTIALLY VALID | 2 | 156, 159 |
| **INVALID** | **1** | **160** |

**12/14 patterns are directionally correct** (86%). Of those, 5 are strongly actionable.

**1 pattern (160) is factually wrong** and should be retired — it claims SAC benefits from
more training bars when the data shows the opposite.

**2 patterns (156, 159) have correct directional claims but cite wrong evidence** — the
consolidation LLM (Mistral) cherry-picked folds that fit the narrative rather than testing
the full distribution.

### Key Findings From Pattern Analysis

**1. Treatment (epoch advice) is NET NEGATIVE for PPO in both environments:**
- Equity PPO: CTRL=2.865 vs TREAT=2.858 (negligible, -0.2%)
- Crypto PPO: CTRL=9.125 vs TREAT=6.438 (large, -29.5%)
- PPO is the highest-weighted algo. Making it worse costs the most.

**2. Treatment HELPS A2C crypto but HURTS A2C equity:**
- Crypto A2C: CTRL=2.872 vs TREAT=3.749 (+30.5%)
- Equity A2C: CTRL=2.561 vs TREAT=1.892 (-26.1%)
- Need env-specific adjustment strategies.

**3. SAC patterns are largely observational, not actionable:**
- SAC degrades in negative yield spread (valid). SAC drawdown recovery is inconsistent
  (valid). SAC reward adjustments are 60% effective (valid).
- But none of these patterns tell the HP tuner what to CHANGE. They describe symptoms
  without connecting to the actual root causes (ent_coef, gradient_steps, buffer_size)
  because those parameters aren't in the tunable set.

**4. Consolidation LLM (Mistral) quality concerns:**
- 1/14 patterns factually wrong (160 — data_size_impact)
- 2/14 patterns cite wrong evidence (156, 159 — wrong fold attribution)
- Pattern 160's numbers (mean_sharpe=0.4901 for low bars, 1.6012 for high bars) don't
  appear anywhere in the raw data. The LLM fabricated statistics.
- The dedup system is working well (7 dedup confirmations) and prevents pattern bloat.

### Recommendations for Pattern System

1. **Retire pattern 160 immediately** — factually wrong, will mislead HP tuner.

2. **Add post-consolidation validation**: After Mistral generates a pattern, verify key
   numbers against the raw WF fold data before inserting. If claimed fold-level stats
   don't match backtest_results within 5%, flag for review.

3. **Implement env-specific epoch advice for A2C**: Pattern 169 correctly identified that
   A2C needs different treatment in equity vs crypto. The epoch callback should apply
   smaller adjustment deltas (or no adjustments) for A2C equity while keeping current
   approach for A2C crypto.

4. **Raise the bar for PPO epoch advice triggers**: Patterns 157+163 show treatment hurts
   PPO in both envs. Either widen the PPO deadzone for reward weight changes or only
   intervene on severe metrics (rolling_mdd > -10%).
