# PPO General Tuning Reference for Financial RL

**Context**: SwingRL PPO agent for equity daily (8 ETFs) and crypto 4H (BTC/ETH) swing trading.
Continuous action space: `Box(-1,1)` shape `(n_assets+1,)` mapped to portfolio weights via softmax.
SB3 implementation. This document synthesizes research on PPO hyperparameter tuning with emphasis
on financial RL and continuous portfolio allocation.

**Current Config**:
| Parameter | Value |
|-----------|-------|
| learning_rate | 0.0001 |
| n_epochs | 5 |
| clip_range | 0.2 |
| batch_size | 256 |
| gamma | 0.98 |
| ent_coef | 0.015 |
| n_steps | 2048 |
| n_envs | 6 |
| gae_lambda | 0.95 |
| vf_coef | 0.5 |

**Performance**: Equity Sharpe 2.86, Crypto Sharpe 7.21 (current). Peak equity Sharpe 3.11
(iter 2 — regressed). Control folds outperform treatment by 29.5% in crypto.

---

## 1. HP Sensitivity Ranking for PPO in Financial RL

### General PPO Sensitivity (Engstrom et al., 2020 — "Implementation Matters")

Engstrom et al. (ICLR 2020) showed that PPO's performance is dominated by **code-level
implementation details** (advantage normalization, value function clipping, reward scaling) rather
than the headline algorithm. They found that stripping these details from PPO and adding them to
REINFORCE or A2C closed 70%+ of the performance gap. This means tuning implementation details
matters as much as tuning the "algorithm hyperparameters."

### Sensitivity Ranking (Henderson et al., 2018 + Andrychowicz et al., 2021)

Henderson et al. ("Deep Reinforcement Learning that Matters," AAAI 2018) showed RL results are
highly sensitive to hyperparameters, random seeds, and even codebases. Andrychowicz et al.
("What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," 2021) ran
>250,000 experiments across continuous control tasks and ranked PPO hyperparameters by sensitivity:

**Tier 1 — Highest Sensitivity (tune first)**:
1. **Number of epochs (n_epochs)** — Most sensitive single parameter. Too many epochs causes
   policy collapse via stale importance sampling.
2. **Learning rate** — Classic most-sensitive HP. Log-uniform sampling essential.
3. **Discount factor (gamma)** — Especially in financial RL where horizon matters enormously.
   Equity daily with 252 trading days/year needs gamma > 0.95 to see multi-week effects.

**Tier 2 — High Sensitivity (tune second)**:
4. **Batch size / minibatch size** — Interacts with n_epochs. Smaller minibatches = more gradient
   steps per rollout = amplifies staleness.
5. **Entropy coefficient** — Controls exploration-exploitation. Too low causes premature
   convergence to suboptimal allocations; too high prevents learning any structure.
6. **Clip range** — Moderate sensitivity, but critical for stability.

**Tier 3 — Moderate Sensitivity**:
7. **GAE lambda** — Usually robust in 0.9–0.99 range.
8. **n_steps (rollout length)** — Interacts with gamma and environment horizon.
9. **vf_coef** — Moderate impact when using shared network architecture.

**Tier 4 — Lower Sensitivity (but still matters)**:
10. **Network architecture** — Width/depth of policy and value networks.
11. **Activation function** — Tanh vs ReLU (SB3 default is Tanh for policy).
12. **Max gradient norm** — Usually robust at 0.5.

### Financial-Specific Sensitivity

In financial RL specifically, the following parameters gain outsized importance relative to
robotics/games:

- **gamma** moves to Tier 1 because financial reward horizons are fundamentally different from
  game/robotics environments. A gamma of 0.98 with daily steps gives an effective horizon of
  ~50 days; 0.99 gives ~100 days; 0.999 gives ~1000 days. For swing trading (1-4 week holds),
  gamma=0.98 is reasonable for equity, but may be too low for crypto 4H where a "day" is 6 bars.
- **Reward scaling** becomes Tier 1 because financial returns have small magnitude (daily returns
  ~0.001) and heavy tails. PPO's advantage normalization helps but doesn't fully solve this.
- **n_steps** gains importance because it determines how many market regimes the agent sees per
  update. With n_steps=2048 and daily bars, that's ~8 years of data per rollout — essentially the
  entire training set per environment.

---

## 2. n_epochs — Stale Importance Sampling

### The Mechanism

PPO performs `n_epochs` passes of minibatch SGD over each rollout buffer. The rollout was collected
under the **old policy** pi_old. After the first gradient step, the current policy pi_theta has
diverged from pi_old, making the importance sampling ratio r(theta) = pi_theta(a|s) / pi_old(a|s)
increasingly inaccurate.

The clipping mechanism `clip(r, 1-eps, 1+eps)` bounds the policy ratio but does **not** fix the
underlying problem: the advantages A_hat were computed under pi_old and become stale as theta
moves. After several epochs, the agent is optimizing against stale value estimates, which can
cause:

1. **Overoptimization of stale advantages** — The policy exploits inaccuracies in the old value
   function rather than learning genuine improvements.
2. **Effective learning rate amplification** — With n_epochs=10 and batch_size=256 on a buffer of
   12,288 (2048 * 6), that's 48 minibatches per epoch × 10 epochs = 480 gradient steps on the
   same data. This dramatically amplifies the effective learning rate.
3. **Policy collapse** — In the worst case, the policy moves so far from pi_old that the clipping
   becomes active on nearly all samples ("clip fraction" approaches 1.0), and learning stalls or
   becomes unstable.

### Interaction with clip_range and batch_size

**n_epochs × (buffer_size / batch_size) = total gradient steps per rollout**

Current config: 5 × (12288 / 256) = 5 × 48 = **240 gradient steps** per rollout collection.

This is on the higher end. Andrychowicz et al. (2021) found:
- Optimal n_epochs is typically 3-10 for continuous control.
- Performance often peaks at n_epochs=5 then degrades at 10+.
- Larger batch sizes tolerate more epochs (lower variance gradients are less likely to overshoot).
- Smaller clip_range tolerates fewer epochs (the policy can't move far, so staleness is bounded).

**Key insight**: The iter 2 regression from Sharpe 3.11 to 2.86 could be explained by the LLM HP
tuner having changed n_epochs or learning rate in a way that increased effective gradient steps
past the stability threshold.

### Schulman et al. (2017) Original Guidance

The original PPO paper (Schulman et al., "Proximal Policy Optimization Algorithms," 2017) used:
- n_epochs=10 with clip_range=0.2 for Mujoco (continuous control)
- n_epochs=3 with clip_range=0.2 for Atari (discrete actions)

However, these were for stationary environments with well-shaped rewards. Financial environments
are non-stationary with sparse, noisy rewards — n_epochs should be on the **lower** end.

### Recommendation for SwingRL

n_epochs=5 is reasonable. If the LLM HP tuner pushes this above 7, monitor clip_fraction closely.
If clip_fraction > 0.20, reduce n_epochs. If the tuner reduces to 3, it should simultaneously
allow a higher learning rate to compensate for fewer gradient steps.

**Monitoring**: SB3 logs `clip_fraction` — this is the fraction of samples where the clipping was
active. Healthy range: 0.05-0.15. Above 0.20 indicates the policy is trying to move too far per
update (too many epochs, lr too high, or clip_range too tight).

---

## 3. Clip Range Tuning

### Fixed vs Adaptive

**Fixed clip_range** (current: 0.2):
- Schulman et al. (2017) showed 0.2 is robust across most tasks.
- Simple and predictable. No additional complexity.

**Linear annealing** (common in practice):
- SB3 supports passing a callable: `clip_range = lambda progress: 0.2 * (1 - progress)`.
- Rationale: early training needs larger policy updates (exploration); late training needs
  stability (exploitation).
- Andrychowicz et al. (2021) found annealing clip_range provided marginal improvement in some
  tasks but was not universally beneficial.

**Adaptive KL-based** (PPO-Penalty variant):
- Original PPO paper proposed an alternative: instead of clipping, use a KL penalty with adaptive
  coefficient. If KL > target, increase penalty; if KL < target, decrease.
- Less commonly used because clipping is simpler and works as well or better in practice.
- However, for non-stationary environments like finance, adaptive KL could help because the
  "right" step size varies with market regime.

**DARC / Adaptive Clip** (research):
- Some works (e.g., Hilton et al., 2022) propose adapting clip_range based on the observed
  KL divergence. If approx_kl is consistently low (policy barely moving), widen the clip; if
  high, tighten it.
- This effectively automates what a human tuner would do by watching clip_fraction.

### Clip Fraction Monitoring

`clip_fraction` in SB3 is computed as the mean fraction of samples where:
`|r(theta) - 1| > clip_range`

Interpretation:
- **clip_fraction < 0.02**: Policy barely moving. Learning rate too low, or n_epochs too few, or
  the policy has converged. In financial RL, this could also mean the reward signal is too weak.
- **clip_fraction 0.05-0.15**: Healthy range. Policy is updating meaningfully but not violently.
- **clip_fraction 0.15-0.25**: Aggressive updates. Monitor for instability.
- **clip_fraction > 0.25**: Policy is fighting the clip constraint. Either lr is too high,
  n_epochs too many, or a sudden reward distribution shift (regime change) is forcing large
  policy updates.

### Value Function Clipping

SB3 by default clips value function updates similarly to policy updates (`clip_range_vf=None`
means it mirrors `clip_range`). Engstrom et al. (2020) showed value function clipping often
**hurts** performance. Andrychowicz et al. (2021) confirmed: value clipping provided no benefit
in their large-scale study.

**Recommendation**: Set `clip_range_vf = None` (disabled) in SB3 PPO. This allows the value
function to update freely, which is especially important in non-stationary financial environments
where value estimates need to adapt quickly to regime changes.

### Recommendation for SwingRL

- Keep fixed clip_range=0.2 as the baseline. It's well-tested and our n_epochs=5 is moderate.
- Consider linear annealing (0.2 to 0.05) within each walk-forward fold if fold length is
  sufficient (>200 episodes).
- **Priority**: Disable value function clipping (`clip_range_vf=None`) — this is likely a free
  improvement.
- Let the HP tuner search clip_range in [0.1, 0.3]. Below 0.1 is too conservative for financial
  RL where reward signals are weak; above 0.3 allows too-large policy swings.

---

## 4. GAE Lambda

### Mechanism

Generalized Advantage Estimation (Schulman et al., "High-Dimensional Continuous Control Using
Generalized Advantage Estimation," 2016) computes advantages as an exponentially-weighted average
of n-step TD errors:

A_hat_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

- **lambda=0**: Pure 1-step TD advantage. Low variance, high bias. The agent only looks one step
  ahead for advantage estimation.
- **lambda=1**: Monte Carlo advantage. High variance, low bias. The agent uses the entire
  remaining trajectory.
- **lambda=0.95** (current): Balances bias and variance. Most of the advantage signal comes from
  the first ~20 steps (1/(1-0.95) = 20 step effective horizon).

### Interaction with Gamma

The **effective horizon** for advantage estimation is determined by `gamma * lambda`:
- gamma=0.98, lambda=0.95 → effective discount per step = 0.931 → horizon ~14 steps
- gamma=0.99, lambda=0.95 → effective discount per step = 0.9405 → horizon ~17 steps
- gamma=0.98, lambda=0.99 → effective discount per step = 0.9702 → horizon ~34 steps

For equity daily (1 bar = 1 day), a 14-step advantage horizon means the agent evaluates actions
based on their ~3-week impact. For crypto 4H (1 bar = 4 hours), 14 steps = 56 hours ≈ 2.3 days.

**Key question**: Is a 2.3-day advantage horizon appropriate for crypto swing trading? If the
typical crypto swing trade lasts 3-7 days (18-42 bars), then lambda=0.95 with gamma=0.98 may be
too short for crypto. Consider lambda=0.97-0.99 for the crypto environment.

### Interaction with n_steps

GAE requires a full rollout buffer to compute advantages. If `n_steps` is shorter than the
effective horizon, the advantage estimates are truncated and biased toward shorter horizons
regardless of lambda. With n_steps=2048, this is not an issue (2048 >> 20 effective steps).

However, if episodes terminate early (done=True before n_steps is reached), the advantage
estimation resets. In financial RL where episodes may be fixed-length (one fold), this typically
isn't an issue.

### Sensitivity

Andrychowicz et al. (2021) found GAE lambda to be **moderately sensitive** in [0.9, 1.0]:
- lambda=0.95 was consistently near-optimal across tasks.
- lambda < 0.9 hurt performance significantly (too much bias).
- lambda=1.0 (pure MC) was slightly worse than 0.95 in most tasks (too much variance).

### Recommendation for SwingRL

- lambda=0.95 is solid for equity daily. No strong reason to change.
- For crypto 4H, consider lambda=0.97-0.98 to extend the advantage horizon from 2.3 days to
  ~4-5 days, better matching typical swing trade durations.
- HP tuner should search lambda in [0.92, 0.99]. Narrower range than most HPs because sensitivity
  is moderate and the optimum is well-characterized.

---

## 5. n_steps x n_envs Rollout Buffer and Batch Size

### The Rollout Buffer

PPO collects `n_steps` transitions from each of `n_envs` parallel environments before performing
an update. Total buffer size = `n_steps * n_envs`.

Current: 2048 * 6 = **12,288 transitions** per update.

This buffer is then split into minibatches of size `batch_size` for SGD. Each epoch makes
`buffer_size / batch_size` gradient steps.

### Minibatch SGD Mechanics

With buffer=12,288 and batch_size=256:
- Minibatches per epoch: 12,288 / 256 = 48
- Total gradient steps per rollout: 48 * 5 (epochs) = 240

**Comparison to common setups**:
- OpenAI Five (Dota 2): Massive buffers (>100K), large batch sizes (>4096), n_epochs=4.
- Mujoco baselines: buffer=2048, batch_size=64, n_epochs=10 → 320 gradient steps.
- Our setup: 240 gradient steps is moderate-to-high.

### Batch Size Effects

**Smaller batch_size** (e.g., 64):
- More gradient steps per epoch → more staleness amplification.
- Higher gradient noise → can help escape local optima but also causes instability.
- Each gradient step sees less data → higher variance estimates of the policy gradient.

**Larger batch_size** (e.g., 512 or 1024):
- Fewer gradient steps per epoch → less staleness.
- Lower gradient noise → more stable training.
- Better GPU utilization (up to a point).
- Risk: may smooth over important signals in individual transitions.

McCandlish et al. ("An Empirical Model of Large-Batch Training," 2018) showed a critical batch
size below which increasing batch size improves compute efficiency and above which it wastes
compute. For RL (noisier gradients than supervised learning), this critical batch size is
typically smaller.

### n_steps Considerations

**Shorter n_steps** (e.g., 512):
- More frequent policy updates → faster adaptation to non-stationary environments.
- But: shorter rollouts may not capture full market cycles.
- Advantage estimates truncated if episode is longer than n_steps.

**Longer n_steps** (e.g., 4096 or 8192):
- Each rollout captures more market dynamics.
- Fewer updates per wall-clock time.
- Better advantage estimates (longer trajectories for GAE).
- Risk: by the time the update happens, the policy is very stale relative to the oldest data.

**For financial RL**: With daily bars, n_steps=2048 covers ~8 years. This is essentially the
entire training fold. Each PPO update sees the full walk-forward fold's worth of data. This means
there's only **1 policy update per full pass of the fold**, then the loop repeats with the
updated policy generating a new rollout. This is fundamentally different from game RL where
episodes are short and you get many updates per pass.

### n_envs Parallelism

n_envs=6 means 6 independent copies of the environment run in parallel. Each environment may see
the **same data** (if using the same fold) or different data (if using data augmentation or
different random seeds for noise).

**If all 6 envs see the same fold data** (likely in walk-forward backtesting):
- The 6 environments provide 6x more transitions per rollout, but they are **correlated** — the
  market data is identical, only the actions differ (due to stochastic policy sampling).
- This is equivalent to running 6 Monte Carlo rollouts of the same trajectory with different
  action samples. It reduces variance of the policy gradient estimate but does not provide truly
  independent samples.
- Effective sample diversity is less than 6x.

**Recommendation**: If envs share the same data, consider reducing n_envs to 4 and increasing
n_steps to 3072 (same total buffer) — fewer correlated samples but longer advantage estimation
windows. Alternatively, use different data augmentation (noise injection, random start points)
across environments to increase diversity.

### Recommendation for SwingRL

- Current buffer_size=12,288 is reasonable.
- batch_size=256 with 48 minibatches/epoch is moderate. Consider testing 512 (24 minibatches)
  for more stable gradient estimates.
- HP tuner should search batch_size in {64, 128, 256, 512}. Must be a divisor of buffer_size.
- n_steps in {1024, 2048, 4096} — lower values give more frequent updates but truncated
  advantages.
- n_envs in {4, 6, 8} — more is better only if environments provide diverse experience.

---

## 6. PPO for Continuous Portfolio Allocation

### Action Space Design

SwingRL uses `Box(-1, 1)` shape `(n_assets+1,)` with softmax mapping to portfolio weights. The
+1 dimension represents cash. This is a standard approach but has specific implications for PPO:

**Softmax mapping**:
- Raw actions are in (-1, 1), then softmax converts to probability simplex (weights sum to 1).
- The softmax creates **competition between assets** — increasing one weight necessarily decreases
  others. This means the policy gradient for one asset's weight depends on all other assets.
- The Jacobian of softmax introduces correlations in the action gradient that PPO's diagonal
  Gaussian policy does not model. PPO assumes independent action dimensions, but softmax creates
  dependence.

**Implication**: PPO's diagonal Gaussian policy (SB3 default) models each action dimension
independently. After softmax, the actual portfolio weight for asset i depends on **all** raw
actions. This mismatch can slow learning because the policy doesn't model cross-asset
relationships in its action distribution.

**Alternatives considered in literature**:
- Ye et al. ("Reinforcement-Learning based Portfolio Management with Augmented Asset Movement
  Prediction States," AAAI 2020): Used separate actor heads per asset — avoids softmax coupling.
- Jiang et al. ("Deep Reinforcement Learning for Cryptocurrency Trading," 2017): Direct weight
  output with cash as the residual (1 - sum of asset weights) — simpler but can produce invalid
  weights.
- Wang et al. ("Commission Fee is not Enough: A Hierarchical Reinforced Framework for Portfolio
  Management," AAAI 2021): Hierarchical PPO with a high-level policy selecting asset subsets and
  a low-level policy allocating within the subset.

### Reward Design for Portfolio PPO

Common reward formulations in financial PPO (Liu et al., "FinRL: A Deep Reinforcement Learning
Library for Automated Stock Trading in Quantitative Finance," 2020):

1. **Log returns**: r_t = log(portfolio_value_t / portfolio_value_{t-1}). Most common. Additive
   over time, handles compounding naturally.
2. **Differential Sharpe ratio** (Moody & Saffell, 1998): Incrementally estimates Sharpe ratio
   from recent returns. Directly optimizes risk-adjusted performance.
3. **Reward shaping with drawdown penalty**: r_t = return_t - alpha * max(0, drawdown_t). Penalizes
   drawdown directly.

**Scaling matters**: Daily returns are O(0.001). PPO's advantage normalization helps, but if the
reward range is too narrow, the signal-to-noise ratio of advantages is poor. Consider reward
multipliers (e.g., 100x or 252x to annualize) to improve learning signal strength.

### Key Papers

- **Liu et al. (2021)**: "FinRL: Deep Reinforcement Learning Framework to Automate Trading in
  Quantitative Finance." Comprehensive framework paper. Found PPO consistently outperformed A2C
  and DDPG for portfolio allocation on DJIA stocks. PPO achieved Sharpe 1.98 vs A2C 1.87 vs
  DDPG 1.64 on their benchmark.
- **Yang et al. (2020)**: "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble
  Strategy." The ensemble approach (PPO + A2C + DDPG) with Sharpe-based model selection
  outperformed individual agents. PPO was typically selected during trending markets, A2C during
  volatile periods.
- **Zhang et al. (2020)**: "Deep Reinforcement Learning for Trading." Showed PPO outperformed
  DQN and A2C on S&P 500 stock selection, with higher risk-adjusted returns and lower drawdowns.

### SwingRL-Specific Considerations

With 8 ETFs + cash = 9-dimensional action space:
- 9D is moderate complexity for PPO's diagonal Gaussian. Not too high to cause issues, but
  cross-asset correlations are important (ETFs in the same sector move together).
- The softmax forces zero-sum weight allocation. PPO must learn that going "risk-off" means
  increasing the cash dimension, which simultaneously decreases all asset weights.
- Consider: the current action space allows the agent to go "all cash" (high cash dimension).
  Does it actually learn to use this? If not, entropy coefficient may need to be higher to
  encourage exploring the cash dimension.

For crypto with 2 assets + cash = 3-dimensional:
- Much simpler action space. PPO should learn this more easily.
- The high Sharpe (7.21) suggests it has. But the 29.5% control > treatment regression suggests
  the HP tuner may be overtuning for this simple space.

---

## 7. PPO Failure Modes

### 7.1 Value Function Estimation Errors

**Problem**: If the value function V(s) is inaccurate, the advantage estimates A_hat = Q - V are
biased, and the policy optimizes against wrong signals.

**Symptoms**: Value loss remains high or oscillates. Policy loss decreases but episodic reward
does not improve (the policy is "gaming" inaccurate value estimates).

**Financial RL specifics**: Financial state spaces are high-dimensional and non-stationary. The
value of being in a particular market state changes over time (regime shifts). A value function
trained on a bull market will systematically overestimate values when a bear market begins.

**Mitigation**:
- Increase `vf_coef` (e.g., 0.5 to 1.0) to prioritize value function accuracy.
- Use separate networks for policy and value (SB3: set `net_arch` with separate `pi` and `vf`).
- Increase network capacity (wider hidden layers) for the value function.
- Consider PopArt normalization (Hessel et al., 2019) for the value function output to handle
  varying reward scales across market regimes.

### 7.2 KL Divergence Explosion

**Problem**: The policy changes too rapidly between updates, violating the trust region that PPO
approximates.

**Symptoms**: `approx_kl` (SB3 logged value) spikes above 0.05-0.1. Clip fraction approaches
0.3+. Policy performance suddenly degrades.

**Causes**:
- Learning rate too high.
- n_epochs too high (compounding staleness).
- Sudden reward distribution change (regime shift in market data).
- Batch size too small (noisy gradients cause large policy swings).

**Mitigation**:
- SB3's `target_kl` parameter: if set, PPO early-stops epochs when approx_kl exceeds this
  threshold. E.g., `target_kl=0.015` is a common conservative value. This is the **single most
  important stability guard** and SwingRL should use it if not already.
- Reduce learning rate (or use lr scheduling).
- Increase batch size for lower gradient variance.

### 7.3 Entropy Collapse

**Problem**: The policy becomes deterministic prematurely. The standard deviation of the Gaussian
action distribution shrinks to near-zero, the agent stops exploring, and gets stuck in a
suboptimal allocation.

**Symptoms**: `entropy` (SB3 log) drops steadily and does not recover. Actions become concentrated
on a single allocation pattern. Performance plateaus early.

**Financial RL specifics**: In portfolio allocation, entropy collapse might manifest as the agent
always allocating to the same 2-3 assets regardless of market conditions. The agent learns a
"default portfolio" that works on average but fails to adapt to changing regimes.

**Causes**:
- `ent_coef` too low.
- Reward signal too strong for one particular allocation (e.g., 100% SPY during a bull market
  backtest always wins).
- Too many gradient steps driving the policy to exploit known good actions.

**Mitigation**:
- Increase `ent_coef` (current 0.015 is moderate; try 0.02-0.05 for more exploration).
- Use entropy annealing: start high (0.05) and decay to low (0.005) over training.
- Monitor per-dimension action standard deviations. If any dimension's std drops below 0.1,
  entropy is collapsing for that asset.

### 7.4 Reward Scale Sensitivity

**Problem**: PPO is highly sensitive to the absolute scale of rewards. Financial returns are
naturally small (daily: ~0.001, 4H: ~0.0003) and have fat tails.

**Symptoms**: Learning is extremely slow because advantages are tiny relative to gradient noise.
Or, after a normalization change, previously-good HPs become unstable.

**Quantitative evidence**: Engstrom et al. (2020) showed that reward normalization/clipping
changed PPO's Mujoco score by up to **50%** even with identical algorithmic HPs.

**Mitigation**:
- SB3's `normalize_advantage=True` (default) helps but normalizes advantages, not rewards.
- Consider reward scaling: multiply raw returns by a constant (e.g., 100 or 252).
- Consider reward normalization: running mean/std normalization (SB3's `VecNormalize` wrapper).
  **Caution**: VecNormalize maintains running statistics that become stale across walk-forward
  folds. Must be reset or adapted per fold.
- Clip extreme rewards (e.g., cap at +/- 5 std) to prevent tail events from dominating single
  updates.

### 7.5 The "Noisy TV" Problem / Reward Hacking

**Problem**: In financial RL, the agent can appear to learn when it's actually just exploiting
patterns in the training data (lookahead bias, data snooping, or overfitting to specific market
regimes).

**Symptoms**: Training performance is excellent, but walk-forward OOS performance is poor.
Performance degrades on control folds (exactly what we see: control > treatment by 29.5% in
crypto).

**This is likely the most relevant failure mode for SwingRL's current issues.**

**Mitigation**:
- Stronger regularization: higher entropy coefficient, weight decay, dropout.
- Shorter training (fewer total timesteps) to reduce overfitting.
- Data augmentation: noise injection, random start times, synthetic data.
- Ensemble disagreement: if PPO, A2C, and SAC disagree on an action, reduce position size.

---

## 8. PPO in Non-Stationary Financial Environments

### The Core Challenge

Financial markets are fundamentally **non-stationary**: the data generating process changes over
time (regime shifts, structural breaks, changing correlations). Standard RL assumes a stationary
MDP. This tension is the central challenge of financial RL.

### Evidence of Non-Stationarity Impact

- **Carta et al. (2021)**: "Multi-Horizon Forecasting for Limit Order Books." Showed that RL
  agents trained on historical financial data degrade at a rate of ~2-5% per month on OOS data,
  consistent with non-stationarity.
- **Lim et al. (2019)**: "Reinforcement Learning for Market Making." Found that online adaptation
  (retraining periodically) was essential — agents trained once degraded to random within 3-6
  months.

### Walk-Forward as Partial Solution

SwingRL's walk-forward (WF) backtesting partially addresses non-stationarity by retraining on
expanding/rolling windows. However, PPO still assumes stationarity **within** each training fold.
If a fold contains a regime transition (e.g., from low-vol to high-vol), PPO will try to learn a
single policy that works for both regimes — this is suboptimal.

### PPO-Specific Adaptations for Non-Stationarity

**1. Shorter effective memory (lower gamma)**:
- Lower gamma makes the agent more "present-focused," adapting faster to recent conditions.
- Trade-off: loses ability to plan over longer horizons.
- Current gamma=0.98 (effective horizon ~50 steps) is reasonable for daily equity.

**2. Higher entropy for continual adaptation**:
- Non-stationary environments require ongoing exploration because the optimal policy changes.
- An agent that converges to a deterministic policy cannot adapt when the regime shifts.
- ent_coef should be higher for non-stationary environments than stationary ones.
- 0.015 is moderate; consider 0.02-0.03 for crypto (higher non-stationarity).

**3. Plasticity preservation**:
- Lyle et al. ("Understanding Plasticity in Neural Networks," ICML 2023) showed that neural
  networks lose the ability to learn new tasks over time ("loss of plasticity"). This is
  especially problematic in non-stationary RL.
- Mitigation: periodic weight perturbation, continual learning techniques, or simply limiting
  total training timesteps per fold.

**4. Context-dependent policies**:
- Include regime indicators in the state space (volatility regime, trend regime, correlation
  regime) so the policy can condition on the current market regime.
- SwingRL already includes macro features (VIX z-score, yield spread) which serve this purpose.

**5. Meta-learning approaches** (research frontier):
- Finn et al. (2017, MAML) and variants: train PPO to be **quickly adaptable** to new conditions
  rather than optimal for any single condition.
- Computationally expensive; not recommended for SwingRL's current scale.

### Recommendation for SwingRL

The 29.5% control > treatment gap in crypto strongly suggests **overfitting within walk-forward
folds**. The HP tuner is finding settings that overfit to training folds, and these settings
transfer poorly to OOS folds. Concrete recommendations:

1. Reduce total training timesteps per fold (currently unknown — check config).
2. Increase entropy coefficient for crypto (try 0.02-0.03).
3. Consider a higher `target_kl` threshold so epochs early-stop before overfitting.
4. Use validation-based early stopping within each fold: hold out the last 10% of each training
   fold as a validation set and stop training when validation performance degrades.

---

## 9. SB3 PPO Implementation Specifics

### normalize_advantage (default: True)

SB3 normalizes advantages within each minibatch:
```
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Impact**: This is crucial for financial RL because raw advantages from financial returns are
tiny. Without normalization, gradient magnitudes would be negligible.

**Gotcha**: Normalization is per-minibatch, not per-rollout. If minibatch size is small (e.g., 32),
the normalization can be noisy. With batch_size=256, this is fine. Andrychowicz et al. (2021)
found this to be one of the most important implementation details.

### Value Function Clipping (clip_range_vf)

SB3 default: `clip_range_vf = None` (which means it uses `clip_range`).

As discussed in Section 3, value function clipping often hurts. From Engstrom et al. (2020):
"We find that value function clipping has no statistically significant positive effect on
performance across our experiments, and in some cases hurts."

**Action item**: Explicitly set `clip_range_vf = None` to disable value function clipping if
SB3's default behavior clips the value function when `clip_range_vf` is not explicitly set.
(Note: SB3 implementation detail — check current behavior. In some SB3 versions, `None` means
"use clip_range" while in others it means "disabled." Verify.)

**Update (SB3 >= 1.6.0)**: `clip_range_vf = None` **disables** value function clipping. This is
the recommended setting.

### Shared vs Separate Networks

SB3 default `net_arch` for PPO: `[dict(pi=[64, 64], vf=[64, 64])]` — **separate** networks for
policy and value function.

**Shared networks** (`net_arch = [64, 64]`): Policy and value share feature extraction layers.
- Faster training (fewer parameters).
- Can help when the same features are useful for both policy and value estimation.
- Risk: gradient interference between policy and value objectives (the value function gradient
  can destabilize the policy features).

**Separate networks**: No gradient interference. Value function can learn different features than
policy. Slightly more parameters.

For financial RL, **separate networks are preferred** because:
- The features that predict "what will happen" (value) are different from those that determine
  "what to do about it" (policy). E.g., high VIX predicts high volatility (value) but the
  optimal response depends on current holdings (policy).
- Gradient interference is especially problematic with non-stationary data.

### Orthogonal Initialization

SB3 uses orthogonal initialization (Saxe et al., 2014) for policy and value networks by default,
with gain=sqrt(2) for hidden layers and gain=0.01 for the policy output layer.

The 0.01 gain for the policy output means initial actions are near-zero with very small variance.
After softmax, this produces near-uniform portfolio weights initially — the agent starts with an
equal-weight portfolio. This is actually a reasonable starting point for portfolio allocation.

### Log-std Parameterization

SB3's PPO uses a **state-independent log-std** parameter vector (one learnable parameter per
action dimension). This means the exploration noise is the same regardless of market state.

**Implication**: The agent cannot learn to be more exploratory in unfamiliar states and more
exploitative in familiar ones. This is a known limitation of SB3's PPO for financial RL.

**Alternative**: State-dependent log-std (output a std from the network). This is available in
some SB3 configurations (`log_std_init` parameter controls the initial value; setting
`use_sde=True` enables state-dependent exploration via Generalized State-Dependent Exploration).

**SDE (State-Dependent Exploration)**: SB3's `use_sde=True` replaces the diagonal Gaussian with
a structured noise that depends on the current state features. Raffin et al. (2022,
"Smooth Exploration for Robotic Reinforcement Learning") showed SDE improves performance in
continuous control. For portfolio allocation, SDE could allow the agent to explore more during
uncertain market conditions and less during clear trends.

**Recommendation**: Consider testing `use_sde=True` with `sde_sample_freq=8` (resample noise
every 8 steps). This is a potentially significant improvement for financial RL but is under-tested
in this domain.

---

## 10. Parameters the HP Tuner Should Control

### Currently Tuned (verify these are all exposed)

| Parameter | Current | Needs Tuning? |
|-----------|---------|---------------|
| learning_rate | 0.0001 | YES — Tier 1 |
| n_epochs | 5 | YES — Tier 1 |
| gamma | 0.98 | YES — Tier 1 |
| batch_size | 256 | YES — Tier 2 |
| ent_coef | 0.015 | YES — Tier 2 |
| clip_range | 0.2 | YES — Tier 2 |
| gae_lambda | 0.95 | YES — Tier 3 |
| n_steps | 2048 | Moderate — Tier 3 |
| vf_coef | 0.5 | Moderate — Tier 3 |

### Potentially Missing (should the tuner control these?)

| Parameter | Current (SB3 default) | Impact | Recommendation |
|-----------|-----------------------|--------|----------------|
| `target_kl` | None (disabled) | **HIGH** — prevents KL explosion, acts as adaptive n_epochs | **Add to tuner**, range [0.01, 0.05] |
| `max_grad_norm` | 0.5 | Moderate — gradient clipping threshold | Consider [0.3, 1.0] |
| `clip_range_vf` | None (= clip_range) | Moderate — should likely be disabled | **Set to None (disabled)**, don't tune |
| `normalize_advantage` | True | HIGH — but should always be True for financial RL | **Do not tune — always True** |
| `use_sde` | False | Potentially high for financial RL | **Test manually** before adding to tuner |
| `sde_sample_freq` | -1 | Only relevant if use_sde=True | If SDE enabled, tune in [4, 16] |
| `net_arch` | [dict(pi=[64,64], vf=[64,64])] | Moderate | Consider adding [64,64] vs [128,64] vs [256,128] |
| `activation_fn` | Tanh | Low — but ReLU may work better for financial features | Test manually |
| `log_std_init` | 0.0 | Moderate — controls initial exploration | Consider [-1.0, 0.5] |
| `ortho_init` | True | Low — should stay True | Do not tune |

### Priority Additions

**1. `target_kl` (HIGHEST PRIORITY)**:
- This is the single most impactful parameter that appears to be missing.
- When set, PPO early-stops its epoch loop if the approximate KL divergence exceeds the
  threshold. This provides adaptive n_epochs without explicitly tuning n_epochs.
- Effectively prevents the "too many epochs" failure mode automatically.
- Start with `target_kl=0.02` and let the tuner search [0.01, 0.05].

**2. `max_grad_norm`**:
- Currently at SB3 default of 0.5. With financial reward scales, this may be too aggressive
  (clipping informative gradients) or too loose.
- Search range: [0.3, 1.0].

**3. `log_std_init`**:
- Controls initial exploration level. Default 0.0 means initial std = exp(0) = 1.0.
- For portfolio weights in (-1, 1), initial std=1.0 means very wide exploration.
- A lower value (e.g., -0.5, giving std=0.6) might give more focused initial exploration.
- Search range: [-1.0, 0.5].

**4. Network architecture**:
- Fixed in most PPO implementations, but can have significant impact.
- Test: [64, 64] vs [128, 64] vs [128, 128] for both pi and vf.
- Could be tuned as a categorical HP.

### Parameters the Tuner Should NOT Control

- `normalize_advantage`: Always True.
- `ortho_init`: Always True.
- `clip_range_vf`: Always None (disabled).
- `n_envs`: Infrastructure parameter, not an HP. Keep at 6.
- `policy`: Always "MlpPolicy" for tabular features.

---

## 11. Recommended Bounds for Automated HP Tuning

### Search Space Definition

```yaml
ppo_hp_search_space:
  # Tier 1 — Highest impact, widest search
  learning_rate:
    type: log_uniform
    low: 1e-5
    high: 3e-3
    # Note: Log-uniform sampling is critical. Linear sampling wastes
    # budget on high-lr configurations that diverge.

  n_epochs:
    type: int_uniform
    low: 3
    high: 10
    # Do not go above 10. If the tuner wants >10, it should increase
    # batch_size instead.

  gamma:
    type: uniform
    low: 0.95
    high: 0.999
    # Equity daily: prefer 0.97-0.99. Crypto 4H: prefer 0.95-0.98.
    # Consider separate bounds per environment.

  # Tier 2 — High impact
  batch_size:
    type: categorical
    choices: [64, 128, 256, 512]
    # Must divide n_steps * n_envs evenly.

  ent_coef:
    type: log_uniform
    low: 0.001
    high: 0.05
    # Log-uniform because the effect is roughly logarithmic.
    # Below 0.001: entropy collapse risk.
    # Above 0.05: too much exploration, never converges.

  clip_range:
    type: uniform
    low: 0.1
    high: 0.3
    # Narrow range because 0.2 is well-established.
    # Below 0.1: updates too small, learning is glacial.
    # Above 0.3: trust region too wide, instability.

  target_kl:
    type: uniform
    low: 0.01
    high: 0.05
    # NEW — add this parameter.
    # Conservative: 0.01 (very stable, slow learning).
    # Aggressive: 0.05 (faster but risk of instability).

  # Tier 3 — Moderate impact
  gae_lambda:
    type: uniform
    low: 0.92
    high: 0.99
    # Narrow range, well-characterized optimum around 0.95.

  n_steps:
    type: categorical
    choices: [1024, 2048, 4096]
    # Lower values = more frequent updates.
    # Higher values = better advantage estimates.

  vf_coef:
    type: uniform
    low: 0.25
    high: 1.0
    # Higher values prioritize value function accuracy.

  max_grad_norm:
    type: uniform
    low: 0.3
    high: 1.0

  # Tier 4 — Test manually before adding to automated tuning
  log_std_init:
    type: uniform
    low: -1.0
    high: 0.5

  net_arch_choice:
    type: categorical
    choices:
      - "small"    # pi=[64,64], vf=[64,64]
      - "medium"   # pi=[128,64], vf=[128,64]
      - "large"    # pi=[256,128], vf=[256,128]
```

### Per-Environment Overrides

Equity and crypto have fundamentally different characteristics. Consider separate bounds:

| Parameter | Equity Daily | Crypto 4H | Rationale |
|-----------|-------------|-----------|-----------|
| gamma | [0.97, 0.999] | [0.95, 0.985] | Equity needs longer horizon for weekly swings |
| ent_coef | [0.005, 0.03] | [0.01, 0.05] | Crypto is more non-stationary, needs more exploration |
| n_steps | [1024, 2048] | [2048, 4096] | Crypto has 6x more bars per day |
| gae_lambda | [0.93, 0.98] | [0.95, 0.99] | Crypto needs longer advantage horizon in bar-space |

### Interaction Constraints

The tuner should enforce these constraints to avoid pathological configurations:

1. **n_epochs × (n_steps × n_envs / batch_size) < 500**: Total gradient steps per rollout.
   Above 500 risks severe staleness.
2. **batch_size divides (n_steps × n_envs)**: Mathematical requirement for minibatch SGD.
3. **If n_epochs > 7, require target_kl to be set**: Safety valve for high-epoch configs.
4. **If learning_rate > 1e-3, require clip_range < 0.2**: Large LR + large clip = instability.

### Bayesian Optimization Considerations

If using Bayesian optimization (e.g., Optuna) for the HP search:

- **Pruning**: Use median pruning based on intermediate rewards (e.g., rolling 50-episode average).
  Kill trials that are clearly underperforming after 25% of training.
- **Multi-objective**: Optimize for both Sharpe ratio AND maximum drawdown. A Pareto front of
  Sharpe vs MDD gives better insight than optimizing Sharpe alone.
- **Number of trials**: For 10 HPs, need minimum 50-100 trials for Bayesian optimization to
  converge. With walk-forward folds, each trial is expensive — consider using a single fold for
  initial screening, then validate top-5 configs on full WF.

---

## Appendix A: Key Papers Referenced

| Paper | Year | Key Finding for SwingRL |
|-------|------|----------------------|
| Schulman et al., "Proximal Policy Optimization Algorithms" | 2017 | PPO algorithm definition; clip_range=0.2 baseline |
| Schulman et al., "High-Dimensional Continuous Control Using GAE" | 2016 | GAE; lambda=0.95-0.99 optimal range |
| Engstrom et al., "Implementation Matters in Deep RL" | 2020 | Code-level details (advantage norm, reward scaling) matter more than algorithm choice |
| Andrychowicz et al., "What Matters In On-Policy RL?" | 2021 | Large-scale HP sensitivity study; n_epochs and LR most sensitive |
| Henderson et al., "Deep RL that Matters" | 2018 | RL results highly sensitive to HPs, seeds, and codebases |
| Liu et al., "FinRL" | 2020-2021 | PPO outperforms A2C/DDPG for portfolio allocation (Sharpe 1.98 vs 1.87 vs 1.64) |
| Yang et al., "Deep RL for Automated Stock Trading: Ensemble Strategy" | 2020 | Ensemble approach with regime-based model selection |
| Lyle et al., "Understanding Plasticity in Neural Networks" | 2023 | Loss of plasticity in continual learning; relevant to non-stationary financial RL |
| Raffin et al., "Smooth Exploration for Robotic RL (SDE)" | 2022 | State-dependent exploration; potentially valuable for financial RL |
| McCandlish et al., "Empirical Model of Large-Batch Training" | 2018 | Critical batch size concept |
| Moody & Saffell, "Learning to Trade via Direct RL" | 1998 | Differential Sharpe ratio reward; foundational financial RL |
| Saxe et al., "Exact Solutions to Nonlinear Dynamics of Learning" | 2014 | Orthogonal initialization theory |

## Appendix B: Diagnostic Checklist

When PPO performance degrades, check these SB3-logged metrics in order:

| Metric | Healthy Range | If Outside Range |
|--------|--------------|-----------------|
| `approx_kl` | 0.005 - 0.02 | >0.03: reduce lr or n_epochs. <0.002: increase lr or n_epochs. |
| `clip_fraction` | 0.05 - 0.15 | >0.20: policy fighting constraint. <0.02: not learning. |
| `entropy_loss` | Stable or slowly declining | Sharp drop: entropy collapse. Increase ent_coef. |
| `value_loss` | Decreasing over training | Increasing: value function failing. Increase vf_coef or capacity. |
| `policy_gradient_loss` | Small and stable | Large oscillations: lr too high or reward scale issues. |
| `explained_variance` | 0.5 - 0.95 | <0.3: value function is poor predictor. >0.99: possible overfitting. |
| `std` (action std) | 0.3 - 1.0 | <0.1: collapsed exploration. >2.0: not learning structure. |

## Appendix C: Quick Reference — Current Config Assessment

| Parameter | Current | Assessment | Action |
|-----------|---------|------------|--------|
| learning_rate | 0.0001 | Reasonable but on the low side | Let tuner explore up to 3e-4 |
| n_epochs | 5 | Good default | Add target_kl=0.02 as safety net |
| clip_range | 0.2 | Standard, fine | Disable vf clipping |
| batch_size | 256 | Good balance | Test 512 for stability |
| gamma | 0.98 | Good for equity; may be too high for crypto 4H | Consider 0.96-0.97 for crypto |
| ent_coef | 0.015 | Moderate; may be too low for non-stationary crypto | Try 0.02-0.03 for crypto |
| n_steps | 2048 | Covers full fold — fine | Consider 1024 for more frequent updates |
| n_envs | 6 | Fine if envs have diverse data | Verify env data diversity |
| gae_lambda | 0.95 | Standard, good | Consider 0.97 for crypto |
| vf_coef | 0.5 | Standard | Consider 0.75 for better value fn |
| target_kl | NOT SET | **Missing critical stability guard** | **Add target_kl=0.02** |
| clip_range_vf | NOT SET | Likely clipping (hurts) | **Set to None (disabled)** |
| max_grad_norm | 0.5 (default) | May clip useful gradients | Test 0.7-1.0 |

**Top 3 immediate actions**:
1. Add `target_kl=0.02` — prevents KL explosion and provides adaptive early-stopping of epochs.
2. Set `clip_range_vf=None` — disables value function clipping (shown to hurt by multiple papers).
3. Investigate crypto control > treatment gap as likely overfitting — increase ent_coef to 0.025
   for crypto and consider validation-based early stopping within folds.
