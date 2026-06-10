# Algorithm-Specific Reward Shaping: PPO, A2C, and SAC

Research reference for per-algorithm reward weight adjustment strategies in SwingRL.
Written to inform epoch advice system prompts, bounds enforcement, and consolidation
heuristics. Based on production data from 5 iterations of walk-forward training (Iter 0-4)
plus RL theory and empirical literature.

---

## Context: SwingRL's Reward Architecture

SwingRL uses a composite reward computed per step:

```
R = w_profit * profit + w_sharpe * sharpe + w_drawdown * (-drawdown_frac) + w_turnover * (-turnover_ratio)
```

An LLM epoch advisor queries training health metrics every N rollouts and may adjust
`{w_profit, w_sharpe, w_drawdown, w_turnover}` mid-training. Weights are clamped to bounds
(`bounds.py`) and renormalized to sum=1.0 before application (`reward_wrapper.py`).

**The same adjustment logic currently applies identically to PPO, A2C, and SAC.** This is
the root cause of divergent outcomes:

| Algo | Env | CTRL Sharpe | TREAT Sharpe | Delta | Verdict |
|------|-----|-------------|--------------|-------|---------|
| PPO | Equity | 2.865 | 2.858 | -0.2% | Negligible harm |
| PPO | Crypto | 9.125 | 6.438 | **-29.5%** | Severe harm |
| A2C | Equity | 2.561 | 1.892 | **-26.1%** | Severe harm |
| A2C | Crypto | 2.872 | 3.749 | **+30.5%** | Clear benefit |
| SAC | Both | -- | -- | ~60% effective | Mixed/weak |

The data demands per-algorithm and per-environment reward shaping strategies.

---

## 1. PPO and Mid-Training Reward Changes

### 1.1 Value Function Staleness

PPO trains a shared critic V(s) alongside the policy. The critic estimates expected
*discounted cumulative reward* under current weights. When the epoch advisor changes
weights (e.g., drawdown 0.15 -> 0.30), the critic's predictions are instantly wrong --
it was trained to predict rewards under the old weighting.

The staleness propagates through the advantage function:

```
A(s,a) = R_shaped + gamma * V(s') - V(s)
```

R_shaped now uses new weights, but V(s) and V(s') were trained on old weights. The
advantage becomes a mix of two different objectives. In PPO, the critic is updated via
MSE loss on the *new* rewards over `n_epochs` passes, but the rollout buffer contains
rewards computed under the *old* weights for transitions collected before the change.

**SwingRL specifics**: With n_steps=2048 and n_envs=6, each rollout contains 12,288
transitions. PPO performs 5 epochs of minibatch gradient descent on this buffer. If
weights change at rollout boundary, the entire buffer was collected under old weights but
targets are computed with a V(s) that's partially updated toward new weights. The critic
needs multiple full rollouts (not just epochs within one rollout) to recalibrate.

**Empirical estimate**: At lr=0.0001 and batch_size=256, the critic processes 12,288/256 =
48 batches per epoch, or 240 updates total across 5 epochs. With financial data's low
signal-to-noise ratio, the critic needs 3-5 full rollouts (not epochs) to adapt to a new
reward scale. At 2048 steps per rollout, that is 6,144-10,240 timesteps of degraded
advantage estimates.

### 1.2 Clipping Interaction

PPO's clipped surrogate objective constrains the policy ratio:

```
L_clip = min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)
```

with clip_range eps=0.2. The clip creates a "trust region" calibrated for the current
reward scale. When reward weights shift:

- If the shift increases reward magnitude (e.g., raising profit weight from 0.3 to 0.5
  during a profitable period), advantage magnitudes increase, but clipping prevents the
  policy from responding proportionally. The clip fraction rises (>0.3 indicates clipping
  is too tight), and the policy update is effectively "wasted" -- the optimizer computes
  large gradients that get clipped away.

- If the shift decreases reward magnitude, advantages shrink. The clip region becomes
  relatively loose, offering less regularization. Combined with a stale critic, small
  advantages from the new scale can be dominated by noise.

**Key interaction with normalize_advantage**: SB3 PPO normalizes advantages to zero mean
and unit variance within each minibatch. This partially compensates for reward scale
changes by keeping advantage magnitudes stable. However, normalization operates within a
single minibatch (256 transitions), not across rollouts. The *direction* of advantages
(which actions are labeled good vs bad) still depends on the stale critic, and
normalization cannot fix directional errors.

### 1.3 GAE with Shifting Rewards

Generalized Advantage Estimation (GAE) computes:

```
A_GAE = sum_{l=0}^{T} (gamma * lambda)^l * delta_t+l
where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

With gamma=0.98 and gae_lambda=0.95 (SB3 default), the effective discount is
0.98 * 0.95 = 0.931, giving an effective horizon of ~14 steps. Each advantage estimate
mixes TD residuals across ~14 steps.

When weights change mid-rollout (which does not happen in SwingRL -- changes happen at
rollout boundaries), some delta_t use old weights and some use new. In SwingRL's case,
the issue is subtler: all rewards in the current buffer used old weights, but the critic
has been partially updated toward new weights by prior rollouts. The GAE computation uses
the partially-updated critic with the old-weight rewards, producing biased advantages.

**With PPO's configuration (n_steps=2048)**: The long rollout means each GAE window of
~14 steps is locally consistent (same weights within the buffer). The problem is that the
*V(s) estimates* embedded in the GAE formula are stale. The bias is smaller than in A2C
(where short rollouts mean more frequent stale-V encounters relative to data collected)
but still measurable.

### 1.4 Why Crypto PPO Suffers More Than Equity PPO

Crypto PPO shows -29.5% degradation vs -0.2% for equity. The key difference:

- **Crypto 4H has 2191 periods/year vs 252 for equity daily**. Sharpe annualization
  amplifies per-step noise by sqrt(2191) = 46.8x vs sqrt(252) = 15.9x. A small shift in
  per-step reward statistics from a weight change gets amplified ~3x more in the crypto
  Sharpe metric.

- **Crypto reward components have different distributional properties**. BTC/ETH 4H returns
  are fat-tailed (kurtosis > 10) with occasional 5-10% moves. The profit component is
  already noisy; changing its weight injects additional variance that PPO's clipping cannot
  fully absorb.

- **Fewer training timesteps per fold in crypto** (~100K vs ~1M equity). The critic has
  less data to adapt after each weight change.

---

## 2. A2C and Mid-Training Reward Changes

### 2.1 No Clipping = Faster Adaptation but Fragile

A2C has no surrogate clipping. The policy gradient is computed directly:

```
grad J = E[A(s,a) * grad log pi(a|s)]
```

When reward weights change, A2C's policy gradient responds immediately to the new
advantage magnitudes -- there is no clip to dampen the response. This is a double-edged
sword:

- **Benefit**: A2C can adapt to new reward weights faster than PPO. The policy moves
  toward the new objective without fighting a trust region constraint. This explains why
  A2C crypto benefits from treatment (+30.5%): the LLM adjusts drawdown weight upward
  during volatile crypto periods, and A2C's unconstrained gradient immediately pivots
  toward capital preservation.

- **Fragility**: Without clipping, A2C is vulnerable to the *direction* of gradient updates
  being wrong due to stale V(s). A large, wrong-direction update is catastrophic because
  there is no mechanism to limit its magnitude (max_grad_norm=0.5 clips gradient *norm*
  but not policy *change*). One bad rollout after a weight change can undo many good ones.

### 2.2 Short Rollouts Amplify Weight Change Impact

A2C uses n_steps=5 with n_envs=6, producing 30 transitions per update. This is orders of
magnitude smaller than PPO's 12,288.

**Impact of weight changes**: After a weight change, A2C processes the new reward signal
within 5 steps (one rollout). The critic is updated immediately on the new rewards. In
theory, this means A2C "forgets" the old objective faster. In practice, with only 30
samples per gradient update, the gradient estimate is extremely noisy. The critic update
from 30 new-weight samples is unreliable.

**The math**: Gradient variance scales as O(1/n) where n = minibatch size. A2C with n=30
has gradient variance ~410x higher than PPO with n=12,288. Adding reward weight changes
to already-noisy gradients can push variance beyond the point where learning makes
progress.

### 2.3 Why Crypto Benefits but Equity Does Not

**A2C Crypto (+30.5%)**: Crypto has higher volatility, making the drawdown component
more informative per step. When the LLM increases drawdown weight during a volatile
period, A2C gets a strong, consistent signal (every bar has meaningful drawdown
information for BTC/ETH). A2C's fast adaptation works well because:
- The signal-to-noise ratio of the drawdown component is high in crypto
- n_steps=5 means A2C updates 33,334 times per fold -- 16x more updates than PPO
  (2,048-step rollouts produce ~82 updates per fold)
- More updates means faster convergence to the new objective
- gae_lambda=1.0 (full Monte Carlo) with gamma=0.97 gives effective horizon ~33 steps,
  which is a reasonable lookback for 4H crypto (5.5 days)

**A2C Equity (-26.1%)**: Equity daily bars have lower per-step information content.
Daily returns for ETFs are typically 0.1-1%, with Sharpe-informative signals buried in
noise. When the LLM adjusts weights:
- The new signal is weak relative to gradient noise
- A2C's 30-sample updates cannot distinguish weight-change effects from market noise
- gae_lambda=1.0 with gamma=0.97 and n_steps=5 means the advantage is computed from
  only 5 noisy daily returns, bootstrapped by a critic that was just perturbed by the
  weight change
- The combined noise (market + weight change + tiny batch) exceeds the signal

**The critical asymmetry**: Crypto reward shaping adjustments carry higher signal-to-noise
ratio because crypto per-step volatility is ~3-5x higher than equity. A2C's sensitivity
to noise means it can only benefit from reward adjustments when the adjustment produces a
signal strong enough to overcome its inherent gradient variance.

### 2.4 GAE Lambda = 1.0 Compounds the Problem

A2C defaults to gae_lambda=1.0 (full Monte Carlo advantage), unlike PPO's 0.95. With
lambda=1.0, the advantage estimate uses *all* remaining returns in the rollout without
bias reduction from the critic:

```
A_MC = R_0 + gamma*R_1 + gamma^2*R_2 + ... + gamma^(T-1)*R_{T-1} - V(s_0)
```

When weights change, the V(s_0) term is stale (trained on old weights), and the R_t terms
may span a weight change boundary if the change occurs at a rollout boundary. The full
Monte Carlo estimate amplifies any inconsistency between the returns and the baseline.

Reducing gae_lambda to 0.92-0.95 would add bias but dramatically reduce variance from
weight-change transients. This is the single highest-leverage change for A2C reward
adaptation stability (see hp-tuning-reference.md).

---

## 3. SAC and Mid-Training Reward Changes

### 3.1 Entropy-Reward Interaction

SAC optimizes a maximum entropy objective:

```
J = E[sum_t (r_t + alpha * H(pi(.|s_t)))]
```

The temperature parameter alpha sets the relative importance of entropy (exploration) vs
reward (exploitation). With `ent_coef="auto"`, SB3 learns alpha to maintain target
entropy = -dim(action_space).

When reward weights change, the *ratio* of reward magnitude to entropy bonus shifts:
- If new weights increase overall reward magnitude, the entropy term becomes relatively
  less important. Alpha should decrease, but auto-tuning has a lag (~100-1000 gradient
  steps to converge).
- If new weights decrease magnitude, entropy dominates. The agent explores more randomly
  until alpha catches up.

**SwingRL specifics**: SAC starts with alpha=1.0 (or `ent_coef="auto"`). Daily equity
returns are ~0.001-0.01. Even with reward shaping, per-step rewards rarely exceed 0.1.
Alpha=1.0 means the entropy bonus (typically 1-5 nats for continuous action spaces) can
be 10-100x the reward signal. The auto-tuning eventually reduces alpha, but during the
transition, reward weight changes are effectively invisible to the policy -- entropy
dominates the objective regardless of weights. This explains SAC's 60% treatment
effectiveness: the reward adjustments are partially masked by entropy dominance.

### 3.2 Replay Buffer Contains Old-Weight Transitions

SAC's replay buffer (500K transitions, FIFO) stores `(s, a, r, s', done)` tuples. The
stored rewards `r` were computed under the weights active at collection time. When the
epoch advisor changes weights:

- New transitions use new weights
- Old transitions (up to 500K) use old weights
- SAC samples batches of 256 uniformly from the buffer

The critic trains on a *mixture of reward signals* from different objective functions. A
batch might contain 200 transitions from old weights and 56 from new weights (proportional
to how recently the change happened). The Q-function learns an average of the two
objectives, which may not correspond to any coherent policy.

**Severity**: With 500K buffer and ~100K timesteps per fold, the buffer accumulates
transitions across the entire fold. If weights change at epoch 50K, half the buffer
contains old-weight data. The buffer only fully "rotates" to new weights after 500K more
timesteps -- potentially never within a single fold.

**Contrast with on-policy (PPO/A2C)**: PPO and A2C discard their rollout buffer after
each update. They never train on old-weight data from previous rollouts. SAC's off-policy
nature makes it structurally more vulnerable to weight changes.

### 3.3 Twin Q-Network Staleness

SAC uses twin Q-networks (Q1, Q2) with target networks updated via Polyak averaging
(tau=0.005):

```
Q_target = (1 - tau) * Q_target + tau * Q_current
```

The target network absorbs only 0.5% of the current critic per step. After a weight
change:
1. Current Q-networks start training on mixed old/new rewards (Section 3.2)
2. Target networks lag further -- they absorb the mixed signal at 0.5% per step
3. The Bellman backup uses target Q: `y = r + gamma * (Q_target(s',a') - alpha * log pi)`
4. Since Q_target reflects the *old* reward scale, the backup target is biased

The twin-Q minimum (`min(Q1, Q2)`) provides pessimistic estimation that partially
mitigates overestimation from stale targets, but it cannot correct the systematic bias
from reward scale mismatches.

**Recovery time estimate**: With tau=0.005 and batch_size=256, 200 gradient steps
transfer ~63% of the current critic to the target (1 - 0.995^200). At 1 gradient step
per env step (SB3 default `gradient_steps=1`), that is 200 * 6 = 1,200 timesteps. But
the current critic itself needs ~5,000-10,000 steps to adapt to the new reward scale
(limited by buffer rotation speed). Total recovery: ~10,000-50,000 timesteps, which is
10-50% of a crypto fold.

### 3.4 Why SAC Shows Mixed Results (60% Effective)

SAC's results are neither clearly helped nor clearly harmed:

1. **Entropy auto-tuning provides a partial adaptive buffer**. As reward scale changes,
   alpha adjusts, partially compensating. This prevents catastrophic failure (unlike A2C
   equity) but also prevents clear benefit (unlike A2C crypto).

2. **The replay buffer smooths transitions**. Old-weight data dilutes new-weight signal,
   acting as a low-pass filter. This reduces both the benefit of good adjustments and the
   harm of bad ones, pushing results toward 50/50 -- exactly what we see (60%).

3. **SAC's fundamental problem is not reward weights**. Production data shows SAC's issues
   are structural: alpha=1.0 start drowning reward signal, gradient_steps=1 under-utilizing
   buffer data, buffer_size=500K causing regime staleness. Reward weight adjustments are
   tweaking a secondary parameter while the primary parameters are misconfigured.

---

## 4. Per-Algorithm Reward Adjustment Best Practices

### 4.1 PPO: Conservative, Infrequent Adjustments

| Parameter | Recommended | Current | Rationale |
|-----------|-------------|---------|-----------|
| **Max delta per component** | 0.03 | 0.10 | Critic needs 3-5 rollouts to adapt; smaller changes preserve advantage quality |
| **Measurement window** | 4 rollouts (8,192 steps) | 1 rollout | Must observe critic stabilization before declaring adjustment effective |
| **Frequency** | 1 per 5 rollouts minimum | 1 per rollout | Overlapping adjustments make attribution impossible |
| **Cooldown after adjustment** | 3 rollouts (6,144 steps) | 0 | Let critic fully recalibrate; early measurement captures noise |
| **Crypto-specific** | Disable reward adjustments | Same as equity | -29.5% treatment harm; PPO is best left alone in crypto |

**PPO adjustment formula** (constrained):
```
w_new[i] = w_old[i] + clamp(delta[i], -0.03, +0.03)
renormalize to sum = 1.0
```

**Rationale for disabling crypto PPO adjustments**: PPO crypto CTRL Sharpe=9.125 vs
TREAT=6.438. With n=4 CTRL and n=10 TREAT folds across iter 4, this is a consistent
pattern also observed in iter 3. The -29.5% degradation exceeds any plausible benefit.
PPO crypto should use static weights (the defaults or the iter 0 converged values).

### 4.2 A2C: Environment-Dependent Strategy

| Parameter | Equity | Crypto | Rationale |
|-----------|--------|--------|-----------|
| **Max delta per component** | 0.02 | 0.05 | Equity SNR too low for large shifts; crypto SNR supports them |
| **Measurement window** | 5,000 steps | 3,000 steps | A2C updates every 30 transitions; 5K = 167 updates |
| **Frequency** | 1 per 10,000 steps | 1 per 5,000 steps | Equity needs longer stabilization |
| **Cooldown after adjustment** | 3,000 steps | 1,500 steps | Crypto adapts faster |
| **Drawdown weight max** | 0.25 | 0.40 | Equity drawdown penalty above 0.25 causes inactivity |

**A2C equity adjustments should be nearly disabled** until gae_lambda is reduced from 1.0
to 0.92-0.95. The current configuration (gae_lambda=1.0, n_steps=5) produces gradient
variance so high that reward weight adjustments are indistinguishable from noise. The
CTRL=2.561 vs TREAT=1.892 gap (-26.1%) shows active harm.

**A2C crypto adjustments should be encouraged**, particularly for the drawdown component.
The +30.5% treatment benefit indicates the LLM is correctly detecting volatile periods
and increasing drawdown weight, which A2C can rapidly incorporate.

### 4.3 SAC: Minimal Adjustments, Focus on Structural Fixes

| Parameter | Recommended | Current | Rationale |
|-----------|-------------|---------|-----------|
| **Max delta per component** | 0.02 | 0.10 | Replay buffer cannot flush old-weight data fast enough |
| **Measurement window** | 50,000 steps | 1 rollout | Must wait for partial buffer rotation |
| **Frequency** | 1 per fold maximum | 1 per rollout | Multiple adjustments create unresolvable mixture in buffer |
| **Cooldown after adjustment** | 30,000 steps | 0 | Buffer needs ~30K steps to rotate 30% of transitions |
| **Priority** | Fix ent_coef, gradient_steps first | Reward weights | Structural issues dominate |

**SAC should receive at most 1-2 reward weight adjustments per fold.** The replay buffer
makes frequent adjustments counterproductive -- each change creates a new "layer" of
transitions with different reward semantics, and the critic averages over all layers.
With 500K buffer and ~100K steps per fold, a mid-fold adjustment means the buffer never
fully contains consistent-weight data during that fold.

### 4.4 Summary: Adjustment Frequency by Algorithm

```
PPO:  [----cooldown----][--measure--][adjust][----cooldown----][--measure--]
      |<--- 6144 steps ---|--- 8192 --->|     |<--- 6144 steps ---|--- 8192

A2C:  [cool][meas][adj][cool][meas][adj][cool][meas][adj]...   (crypto)
      |1500|3000|     |1500|3000|

      [---cooldown---][-----measure-----][adj][---cooldown---]   (equity)
      |--- 3000 steps ---|--- 5000 steps --->|

SAC:  [=========================== fold ===========================]
      |                    [adj]                                    |
      |<-------- 50K measurement -------->|<-- 50K+ cooldown ----->|
```

---

## 5. Reward Normalization

### 5.1 VecNormalize in SwingRL

SwingRL wraps all training environments with SB3's `VecNormalize`:

```
Base VecEnv --> MemoryVecRewardWrapper --> VecNormalize
```

VecNormalize maintains running mean and variance for both observations and rewards.
Rewards are normalized via:

```
r_normalized = clip((r - running_mean) / sqrt(running_var + epsilon), -clip_reward, +clip_reward)
```

Default clip_reward = 10.0. Running statistics use an exponential moving average.

### 5.2 VecNormalize and Reward Weight Changes

When reward weights change, VecNormalize faces a problem:

1. Running mean/variance reflect the *distribution of old-weight rewards*
2. New-weight rewards may have a different mean and variance
3. The running statistics update incrementally -- they lag behind the true distribution

**Best case**: If the weight change preserves reward scale (e.g., shifting between
components of similar magnitude), VecNormalize adapts within ~100-500 steps.

**Worst case**: If the weight change alters reward scale significantly (e.g., doubling
drawdown weight when drawdown is the largest component), the running variance is
temporarily wrong. Rewards are either over-normalized (compressed toward zero) or
under-normalized (allowing large values through), depending on direction.

**Interaction with PPO advantage normalization**: PPO normalizes advantages *within each
minibatch*. VecNormalize normalizes rewards *across the rolling window*. Both mechanisms
partially compensate for scale changes, but neither corrects the *direction* of value
function errors. The double normalization can also create a cascade: VecNormalize rescales
the reward -> critic trains on rescaled reward -> advantage is computed from rescaled
reward and stale critic -> advantage is re-normalized within minibatch. Each layer of
normalization adds latency to adaptation.

### 5.3 Per-Algorithm Impact of VecNormalize

**PPO**: VecNormalize + advantage normalization provide a double buffer against reward
scale changes. This is why equity PPO shows only -0.2% degradation (negligible) despite
receiving reward adjustments. The normalizations mask most of the weight change impact.
Crypto PPO suffers more because the higher-frequency environment generates more
transitions during the VecNormalize lag period.

**A2C**: VecNormalize is the *only* normalization (A2C does not normalize advantages by
default in SB3). With n_steps=5, A2C sees 5 VecNormalize-processed rewards per update.
If VecNormalize running stats are lagging (which they are after a weight change), all 5
rewards in the minibatch are scaled incorrectly. There is no minibatch-level
renormalization to compensate.

**SAC**: VecNormalize normalizes rewards at collection time, and the normalized rewards
are stored in the replay buffer. After a weight change, new transitions enter the buffer
with a different normalization baseline than old transitions. The buffer now contains
rewards from *two different normalization regimes*, making Q-value learning even noisier.
This is a secondary but non-trivial contributor to SAC's mixed results.

### 5.4 Recommendations for Normalization

1. **Reset VecNormalize running stats after weight changes** -- Not recommended for
   SwingRL. Resetting discards valuable reward scale information and causes a transient
   period of extreme normalization noise. The cure is worse than the disease.

2. **Use clip_reward conservatively** -- Current default of 10.0 is appropriate. Reducing
   to 5.0 would clip outlier rewards from weight transitions, providing additional
   stability at the cost of signal loss. Worth testing for A2C equity.

3. **Consider disabling reward normalization for SAC** -- SAC's entropy auto-tuning
   provides its own implicit reward scaling. VecNormalize on top of auto-alpha creates
   two competing adaptation mechanisms. Setting `norm_reward=False` for SAC would
   eliminate the dual-normalization problem and let auto-alpha handle scale changes.
   The Q-networks would see raw reward magnitudes, which auto-alpha already accounts for.

---

## 6. Literature on Dynamic Reward Shaping

### 6.1 Potential-Based Reward Shaping (Ng et al. 1999)

**Core theorem**: The only class of reward transformations that preserves the set of
optimal policies (policy invariance) is potential-based reward shaping (PBRS):

```
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

for some potential function Phi: S -> R. Any shaping reward of this form, when added to
the base reward, produces a new MDP with the same optimal policy as the original.

**Critical implication for SwingRL**: The multi-objective reward scalarization
(`w1*profit + w2*sharpe + w3*drawdown + w4*turnover`) is **not PBRS**. Changing weights
changes the optimal policy. This is by design -- the LLM is intentionally steering the
agent toward different trade-offs. But it means the PBRS convergence guarantees do not
apply. Specifically:
- There is no guarantee that training converges after a weight change
- The Q-function from before the change may be arbitrarily far from optimal under new weights
- Multiple weight changes create a non-stationary optimization problem with no fixed point

**Citation**: Ng, A.Y., Harada, D., Russell, S. (1999). "Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping." ICML 1999.

### 6.2 Dynamic Potential-Based Reward Shaping (Devlin & Kudenko 2012)

Extended PBRS to non-stationary potentials that change during training:

```
F(s, a, s', t) = gamma * Phi(s', t+1) - Phi(s, t)
```

The time-indexed potential function allows the shaping signal to evolve while preserving
policy invariance, *provided* the potential depends on the current timestep. This is
more restrictive than SwingRL's approach (which changes the base reward weights, not an
additive shaping function).

**Relevance**: If SwingRL's reward adjustments were restructured as additive PBRS terms
rather than base reward weight changes, the convergence guarantees of dynamic PBRS would
apply. However, this would require:
- Defining a potential function Phi(s) over the portfolio state
- Only adjusting Phi, not the base reward weights
- A fundamental redesign of the reward architecture

**Citation**: Devlin, S. & Kudenko, D. (2012). "Dynamic Potential-Based Reward Shaping."
AAMAS 2012.

### 6.3 Reward Training Wheels (RTW, 2025)

A teacher RL agent dynamically adjusts auxiliary reward weights via a meta-policy.
Key findings:

- **Inverted U-shape pattern**: Stability penalty weight peaks mid-training then decreases.
  Performance reward weight increases monotonically. The optimal strategy is to emphasize
  stability early (let the agent learn safe behavior first) then relax toward performance.

- **5/5 success rate** vs 2/5 for expert-designed static weights. The dynamic adaptation
  outperforms even domain-expert static tuning.

- **Random weight changes performed 1/5** -- worse than static. Principled, structured
  adaptation is essential. Arbitrary LLM suggestions without clear curriculum logic risk
  being equivalent to random.

**Relevance for SwingRL**: The RTW finding suggests a curriculum approach: start training
with high drawdown weight (safety emphasis), gradually reduce it while increasing profit
weight. The current LLM advisor makes reactive adjustments based on rolling metrics, not
a principled curriculum. Implementing a baseline curriculum with LLM adjustments layered
on top (constrained to small deltas) would combine the benefits of both approaches.

**Citation**: "Reward Training Wheels: Adaptive Auxiliary Rewards for RL." arXiv, 2025.

### 6.4 DynaOpt: Dynamic Reward Emphasis (2024)

Uses multi-armed bandits (Exp3 algorithm) to dynamically select reward emphasis:

```
w_{t+1} = w_t * exp(gamma_bandit * r_hat / K)
```

Key findings:

- **Multiplicative updates** inherently limit change rate (the exponential ensures small
  relative changes per step). This is superior to additive updates (what SwingRL currently
  uses) for stability.

- **7.80% improvement over static baselines** across tested domains.

- The Exp3 bandit provides a principled exploration-exploitation trade-off for weight
  selection, avoiding the need for an LLM to make judgment calls.

**Relevance**: SwingRL could replace the LLM's additive weight suggestions with
multiplicative updates. Instead of the LLM saying "set drawdown weight to 0.35," it would
say "increase drawdown emphasis by 10%." The multiplicative form naturally constrains the
rate of change and prevents extreme jumps.

**Citation**: "DynaOpt: Dynamic Reward Adjustment in Multi-Reward RL." arXiv, 2024.

### 6.5 Reward Curriculum and Self-Play (Meta-Learning)

Several lines of work on learned reward schedules:

- **LIRPG** (Zheng et al., 2018): Learn intrinsic reward functions via meta-gradient.
  The reward function itself is parameterized and optimized to maximize task performance
  on a held-out set. This is close to what SwingRL's LLM advisor attempts, but LIRPG
  uses gradient-based optimization rather than LLM judgment.

- **Evolved Policy Gradients** (Houthooft et al., 2018): Evolutionary optimization of
  the loss function (including reward shaping terms). Found that evolved loss functions
  generalize across tasks and provide better exploration.

- **Meta-gradient RL** (Xu et al., 2018): Online cross-validation of the return
  computation (discount, lambda, reward weights) by differentiating through the RL update.
  The meta-parameters are updated to minimize a validation loss. This is the most
  theoretically grounded approach to dynamic reward adaptation.

**Common finding across all approaches**: Learned/optimized reward schedules consistently
outperform static and hand-tuned schedules, but only when the adaptation is *structured*
(curriculum, meta-gradient, or bandit-based). Unstructured adaptation (random, or ad-hoc
LLM suggestions without constraints) often performs worse than static.

**Citations**:
- Zheng, Z. et al. (2018). "Learning Intrinsic Reward for Policy Gradient." NeurIPS.
- Houthooft, R. et al. (2018). "Evolved Policy Gradients." NeurIPS.
- Xu, Z. et al. (2018). "Meta-Gradient Reinforcement Learning." NeurIPS.

### 6.6 Reward Hacking Under Dynamic Shaping

Skalse et al. (2023) demonstrated that optimizing a proxy reward follows a characteristic
curve: initial improvement, peak, then decline on the true objective. This "Goodhart's Law"
effect is amplified when the proxy reward changes during training:

- Each weight change redefines the proxy
- The agent may already be over-optimized for the previous proxy
- The new proxy may point in a conflicting direction, causing oscillation

The recommended mitigation is early stopping, which sacrifices 10-44% of true reward.
In SwingRL's context, the `stop_training` mechanism in epoch_callback.py (when
`progress >= 0.20`) serves this role.

**Citation**: Skalse, J. et al. (2023). "Defining and Characterizing Reward Hacking."
NeurIPS (ICLR 2024 version).

---

## 7. Practical Recommendations for SwingRL

### 7.1 Immediate Changes (No Architecture Modification)

1. **Implement per-algorithm max_delta in bounds.py**:
   ```python
   _ALGO_MAX_REWARD_DELTA: dict[str, float] = {
       "ppo": 0.03,
       "a2c_equity": 0.02,
       "a2c_crypto": 0.05,
       "sac": 0.02,
   }
   ```
   Compare new weights against previous weights. Clamp each component's change to
   max_delta before renormalization. This is the single highest-impact change.

2. **Disable PPO crypto reward adjustments**: Set PPO crypto max_delta to 0.0 or add
   a config flag `disable_reward_adjustment: true` for ppo+crypto. Use static weights
   from iter 0 baseline (which produced Sharpe 9.12 without any treatment).

3. **Add per-algorithm cooldown in epoch_callback.py**:
   ```python
   _ALGO_COOLDOWN_STEPS: dict[str, int] = {
       "ppo": 6144,     # 3 rollouts * 2048 steps
       "a2c_equity": 3000,
       "a2c_crypto": 1500,
       "sac": 30000,
   }
   ```
   Track `_last_adjustment_step` per wrapper. Reject LLM suggestions during cooldown.

4. **Add measurement window validation in epoch_callback.py**: Do not query the LLM for
   advice until `measurement_window` steps have elapsed since the last adjustment. This
   ensures the rolling metrics (Sharpe, MDD) reflect the new weights, not a transient.

5. **Cap SAC adjustments to 1-2 per fold**: Add a counter in the epoch callback.
   After 2 adjustments, stop requesting advice for the remainder of the fold.

### 7.2 Medium-Term Changes (Architecture-Aware)

6. **Switch to multiplicative weight updates** (DynaOpt-inspired):
   ```python
   # Instead of: w_new = llm_suggested_absolute_weight
   # Use: w_new = w_old * (1 + clamp(llm_delta_pct, -max_pct, +max_pct))
   ```
   The LLM suggests percentage changes rather than absolute weights. This naturally
   constrains the rate of change and makes bounds enforcement more intuitive.

7. **Implement a reward weight curriculum** (RTW-inspired):
   - Phase 1 (0-20% of fold): High drawdown weight (0.35), low profit (0.30)
   - Phase 2 (20-60%): LLM adjustments enabled within max_delta constraints
   - Phase 3 (60-100%): Freeze weights, let policy converge on stable objective
   
   This follows the inverted-U pattern from RTW (stability first, then adapt, then
   stabilize).

8. **Disable VecNormalize reward normalization for SAC** (`norm_reward=False`). Let
   SAC's auto-alpha handle reward scaling. Test whether this improves the 60% treatment
   effectiveness.

9. **Reduce A2C gae_lambda from 1.0 to 0.93** before further reward shaping experiments.
   This reduces gradient variance, making the effect of weight changes measurable rather
   than lost in noise.

### 7.3 Long-Term Research Directions

10. **Per-algorithm LLM system prompts**: The epoch advice system prompt should include
    algo-specific constraints (max delta, cooldown, whether adjustments are enabled for
    this algo+env combination). Currently, the same prompt is used for all algo/env pairs.

11. **Meta-gradient reward optimization**: Replace the LLM advisor with a meta-gradient
    approach (Xu et al., 2018) for reward weight selection. Use a held-out validation
    fold (from walk-forward cross-validation) to compute the meta-gradient. This is
    theoretically optimal but requires significant implementation effort.

12. **Bandit-based weight selection**: Implement Exp3 (DynaOpt) as a lightweight
    alternative to the LLM advisor. Bandit maintains exploration-exploitation trade-off
    automatically and does not require LLM API calls. Could run as a fallback when
    all LLM providers are unavailable (currently 28% of the time).

### 7.4 Priority Order

| Priority | Change | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| P0 | Per-algo max_delta in bounds.py | Prevents >50% of treatment harm | 1 hour |
| P0 | Disable PPO crypto adjustments | Recovers +29.5% Sharpe for PPO crypto | 30 min |
| P1 | Per-algo cooldown in epoch_callback | Prevents stale-V-based adjustments | 2 hours |
| P1 | Cap SAC to 2 adjustments/fold | Prevents buffer contamination | 30 min |
| P2 | Multiplicative updates | Better rate-of-change control | 4 hours |
| P2 | Reward curriculum (3-phase) | Structured adaptation per RTW | 6 hours |
| P2 | Reduce A2C gae_lambda to 0.93 | Reduces gradient noise 3x | 30 min (HP change) |
| P3 | Disable SAC norm_reward | Eliminates dual-normalization | 2 hours + testing |
| P3 | Per-algo LLM system prompts | More appropriate advice generation | 4 hours |
| P4 | Bandit-based fallback | Eliminates LLM dependency | 2 days |
| P5 | Meta-gradient optimization | Theoretically optimal | 1 week |

---

## Sources

| Short Name | Full Citation | Year |
|------------|---------------|------|
| Ng et al. | "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." ICML. | 1999 |
| Wiewiora | "Potential-based Shaping and Q-Value Initialization Are Equivalent." JAIR. | 2003 |
| Devlin & Kudenko | "Dynamic Potential-Based Reward Shaping." AAMAS. | 2012 |
| Haarnoja et al. | "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor." ICML. | 2018 |
| Zheng et al. | "Learning Intrinsic Reward for Policy Gradient (LIRPG)." NeurIPS. | 2018 |
| Houthooft et al. | "Evolved Policy Gradients." NeurIPS. | 2018 |
| Xu et al. | "Meta-Gradient Reinforcement Learning." NeurIPS. | 2018 |
| Engstrom et al. | "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO." ICLR. | 2020 |
| Liu et al. | "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance." arXiv. | 2021 |
| Hayes et al. | "A Practical Guide to Multi-Objective Reinforcement Learning and Planning." JAMAS. | 2022 |
| Van Moffaert & Nowe | "Multi-Objective Reinforcement Learning Using Sets of Pareto Dominating Policies." JMLR. | 2014 |
| Betancourt & Chen | "Deep Reinforcement Learning for Financial Trading Using Sharpe-Based Reward." Springer. | 2023 |
| Skalse et al. | "Defining and Characterizing Reward Hacking." NeurIPS / ICLR 2024. | 2023 |
| Zhang et al. | "Generalized Maximum Entropy Reinforcement Learning via Reward Shaping." IEEE. | 2023 |
| Blom et al. | "Combining Automated Optimisation of Hyperparameters and Reward Shape." arXiv. | 2024 |
| DynaOpt | "Dynamic Reward Adjustment in Multi-Reward Reinforcement Learning." arXiv. | 2024 |
| RTW | "Reward Training Wheels: Adaptive Auxiliary Rewards for RL." arXiv. | 2025 |
| Duarte et al. | "Risk-Aware Reinforcement Learning Reward for Financial Trading." arXiv. | 2025 |
| SB3 Docs | Stable Baselines3 Tips and Tricks documentation. | 2024 |
