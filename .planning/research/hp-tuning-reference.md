# Hyperparameter Tuning Reference for SwingRL

Reference for LLM-guided HP tuning prompts. Each section explains what the HP controls
mechanistically, its relationship to overfitting, and recommended ranges for financial RL.

---

## PPO (Proximal Policy Optimization)

**Architecture**: On-policy, clipped surrogate objective. Collects rollouts of `n_steps`,
then performs `n_epochs` of minibatch gradient descent. Clipping prevents large policy shifts.

### learning_rate (default: 3e-4)

**Controls**: Step size for Adam optimizer on shared actor+critic network.

**Overfitting relationship**: The most direct lever. Higher LR causes larger weight updates per
step, making the model chase noise in individual rollouts. In financial data with low SNR, each
gradient step can "memorize" the current rollout's spurious patterns. Lower LR averages over
noise across many updates but risks underfitting if too conservative.

**Financial RL range**: 5e-5 to 3e-4. Use linear decay for long runs.

**Key interactions**:
- With batch_size: smaller batches have noisier gradients, compound with high LR
- With n_epochs: more epochs on same data with high LR compounds overfitting
- With clip_range: clipping limits policy change but NOT value function change

### n_steps (default: 2048)

**Controls**: Rollout buffer size (steps per env before update). Effective batch = n_steps * n_envs.

**Overfitting relationship**: Smaller rollouts = fewer data points per update = higher variance
gradients = more overfitting to specific price sequences. Larger rollouts average over more
market conditions per update. n_steps=2048 with daily equity covers ~8 years per env.

**Financial RL range**: 1024-4096. Current 2048 is well-chosen for equity.

### batch_size (default: 64)

**Controls**: Minibatch size for gradient updates within each epoch. Buffer is shuffled and
split into chunks of this size.

**Overfitting relationship**: Small batches (32-64) provide implicit regularization through
gradient noise but can wash out weak financial signals. Large batches (256-512) give cleaner
gradients but can more reliably overfit. Acts as a noise/regularization tradeoff.

**Financial RL range**: 64-256.

### n_epochs (default: 10)

**Controls**: Number of full passes over the rollout buffer per policy update.

**Overfitting relationship**: THE MOST CRITICAL OVERFITTING PARAMETER FOR PPO. Each additional
epoch reuses the same fixed data with progressively stale importance sampling correction. The
policy drifts from the data-collecting policy, and on noisy financial data this means
over-adapting to specific return sequences. Consider adding target_kl=0.015 as a safety net.

**Financial RL range**: 3-6 (conservative). SB3 default of 10 is aggressive for noisy data.
With target_kl early stopping, 10 is safer.

### gamma (default: 0.99)

**Controls**: Discount factor. Effective horizon = 1/(1-gamma). gamma=0.99 = ~100 steps,
gamma=0.95 = ~20 steps.

**Overfitting relationship**: Higher gamma considers longer reward sequences, requiring the
value function to predict further into the future. Errors compound and the critic overfits to
long-horizon patterns that may not repeat. Lower gamma acts as implicit regularization (the
agent sees a simpler, shorter-horizon problem).

**Financial RL range**: 0.95-0.99. For swing trading (daily, 5-20 day holds): 0.97-0.99.
For capital preservation: slightly lower (0.95-0.97) to weight near-term drawdown penalties
more heavily.

### gae_lambda (default: 0.95)

**Controls**: Bias-variance tradeoff in advantage estimation. lambda=1.0 = full Monte Carlo
(high variance, low bias). lambda=0 = one-step TD (low variance, high bias).

**Overfitting relationship**: Higher lambda means advantage estimates reflect entire noisy
trajectories, attributing market noise to the agent's actions. Lower lambda relies more on
the critic's smoothed value estimate, reducing overfitting to specific price sequences.

**Financial RL range**: 0.90-0.97. For noisy financial data, slightly lower than default.

### clip_range (default: 0.2)

**Controls**: PPO clipping epsilon. Policy ratio clamped to [1-clip, 1+clip]. Prevents large
policy changes from any single transition.

**Overfitting relationship**: Narrower clip (0.1-0.15) = more conservative updates, prevents
overfitting to any single rollout but learning is slow. Wider (0.3+) allows larger shifts,
higher overfitting risk. Monitor clip_fraction: >0.3 = too narrow or too many epochs.

**Financial RL range**: 0.1-0.25. Decay schedules (0.2 to 0.05) can help: explore early,
exploit carefully late.

### ent_coef (default: 0.01)

**Controls**: Entropy bonus weight. Higher = more random actions = more exploration.

**Overfitting relationship**: Acts as direct regularization by preventing deterministic
policies. A stochastic policy cannot memorize specific action sequences. This is one of the
most effective anti-overfitting tools alongside n_epochs reduction.

**Financial RL range**: 0.005-0.02. Consider schedule: 0.02 early, decay to 0.005.

### vf_coef (default: 0.5)

**Controls**: Value function loss weight relative to policy loss. Both share one optimizer.

**Overfitting relationship**: Higher vf_coef trains a better critic (better advantage
estimates) but can overfit the value function. If critic overfits (explained_variance >0.9 on
train), it provides overconfident advantages that mislead the policy.

**Financial RL range**: 0.25-0.5. Reduce if value function overfitting detected.

### PPO Overfitting Diagnostic Summary

| Symptom | Primary HP Lever | Secondary |
|---------|-----------------|-----------|
| High overfit_gap (IS >> OOS Sharpe) | Reduce n_epochs (10 -> 4-6) | Add target_kl=0.015 |
| Exploding approx_kl | Reduce learning_rate | Narrow clip_range |
| clip_fraction > 0.3 | Reduce n_epochs | Narrow clip_range |
| Policy collapse (entropy -> 0) | Increase ent_coef | Check LR not too high |
| Value function overfit (explained_var > 0.9) | Reduce vf_coef | Reduce n_epochs |

---

## A2C (Advantage Actor-Critic)

**Architecture**: On-policy, synchronous updates. NO clipping protection (unlike PPO).
Single gradient step per rollout. Learning rate is the ONLY lever controlling update magnitude.
Higher variance gradients than PPO. More sensitive to all HP choices.

### learning_rate (default: 7e-4)

**Controls**: Step size for RMSProp optimizer on shared actor+critic.

**Overfitting relationship**: A2C's MOST SENSITIVE parameter. Without PPO's clipping, LR is the
only constraint on update magnitude. Higher LR causes the policy to chase noise in individual
short rollouts. With n_steps=5 (default), each gradient is estimated from only 30 transitions
(5 steps * 6 envs). High LR on noisy 30-sample gradients = catastrophic overfitting.

**Financial RL range**: 1e-4 to 5e-4 (lower than SB3 default of 7e-4).

**Key difference from PPO**: PPO's clipping provides a safety net for high LR. A2C has NO
safety net. If you see A2C overfitting, reducing LR should be the first action.

### n_steps (default: 5)

**Controls**: Steps collected per env before gradient update. Effective batch = n_steps * n_envs.

**Overfitting relationship**: Default of 5 is EXTREMELY short. With 6 envs, each gradient is
computed from 30 transitions. This produces very high-variance estimates. For daily equity,
5 steps = 1 week of data per update. The agent overfits to weekly patterns.

**Financial RL range**: 16-64. For daily equity: 20-64 captures 1-3 months per update.
Increasing n_steps is the single highest-leverage stability improvement for A2C.

**Note**: Currently NOT in bounds.py, so the memory agent cannot tune it.

### gamma (default: 0.99)

**Controls**: Same as PPO. Effective horizon ~100 steps at 0.99.

**Overfitting relationship**: gamma=0.99 with n_steps=5 means the agent sees 5 real reward
steps but bootstraps a 100-step horizon from V(s). The critic bears enormous prediction
burden. Either increase n_steps or decrease gamma to reduce this mismatch.

**Financial RL range**: 0.95-0.98. Lower than PPO due to the n_steps constraint.

### gae_lambda (default: 1.0)

**Controls**: Same as PPO's GAE. But A2C default is 1.0 (full Monte Carlo) vs PPO's 0.95.

**Overfitting relationship**: lambda=1.0 with noisy financial data means advantage estimates
reflect the full noisy trajectory. Combined with A2C's short n_steps=5, each advantage is
based on 5 noisy returns with no variance reduction from the critic. This is the most
aggressive variance setting possible.

**SINGLE HIGHEST-LEVERAGE CHANGE**: Reduce from 1.0 to 0.92-0.95. This dramatically reduces
gradient noise. With gamma=0.99 and lambda=1.0, effective discount = 0.99^k. With lambda=0.95,
it becomes 0.9405^k, shortening effective horizon from ~100 to ~17 steps.

**Financial RL range**: 0.90-0.95.

### ent_coef (default: 0.01)

**Controls**: Same as PPO. Entropy bonus for exploration/regularization.

**Overfitting relationship**: Without PPO's clipping, entropy is even more important as
regularization. Prevents the policy from becoming deterministic and overfitting.

**Financial RL range**: 0.005-0.02.

### max_grad_norm (default: 0.5)

**Controls**: Gradient norm clipping. The ONLY hard constraint on A2C update size (since there's
no PPO-style ratio clipping).

**Overfitting relationship**: Indirect. Caps how far any single update can move weights. With
financial price shocks creating extreme gradients, this is critical for stability.

**Financial RL range**: 0.3-0.5. Keep at default.

### A2C Overfitting Diagnostic Summary

| Symptom | Primary HP Lever | Secondary |
|---------|-----------------|-----------|
| High overfit_gap | Reduce learning_rate (first!) | Reduce gae_lambda (1.0 -> 0.95) |
| Erratic policy (whipsawing) | Reduce learning_rate | Increase n_steps |
| High gradient variance | Increase n_steps (5 -> 16+) | Reduce gae_lambda |
| Policy collapse | Increase ent_coef | Reduce learning_rate |
| Myopic trading (too reactive) | Increase gamma | Increase n_steps |

---

## SAC (Soft Actor-Critic)

**Architecture**: Off-policy with replay buffer. Maximum entropy framework with automatic
entropy tuning. Twin Q-networks reduce overestimation. Separate actor, 2 critics, and alpha
optimizer. Higher sample efficiency than PPO/A2C due to replay.

### learning_rate (default: 3e-4)

**Controls**: Step size for Adam on all four components (actor, critic_1, critic_2, log_alpha).
Single shared LR in SB3 (unlike some implementations that use separate actor/critic LRs).

**Overfitting relationship**: Applied to both actor AND critic. High LR causes rapid Q-value
memorization from replay buffer. With noisy financial returns, the critic can learn spurious
state-action-reward correlations. The twin-Q trick provides some protection but cannot fully
compensate for high LR.

**Financial RL range**: 1e-5 to 1e-4 (lower than default). Consider 3e-5 to 1e-4.

**Key interaction with batch_size**: Reducing batch from 256 to 64 increases gradient noise.
Should proportionally reduce LR (e.g., from 3e-4 to ~7.5e-5 using linear scaling rule).

### batch_size (default: 256)

**Controls**: Transitions sampled from replay buffer per gradient update.

**Overfitting relationship**: Smaller batches (64) provide implicit regularization through
gradient noise. Larger batches (256) give cleaner gradients that can more reliably overfit.
With a finite replay buffer of financial data, large batches mean each transition is resampled
more frequently.

**Financial RL range**: 64-256. Smaller batches are a reasonable choice for noisy data.

### tau (default: 0.005)

**Controls**: Polyak averaging for target network updates. target = (1-tau)*target + tau*current.
At 0.005, targets absorb 0.5% of current critic per step.

**Overfitting relationship**: Higher tau propagates critic overfit to targets faster, creating
a feedback loop. Lower tau provides smoother, less overfit training signal.

**Financial RL range**: 0.002-0.01. Default 0.005 is almost always fine. Rarely worth changing.

### gamma (default: 0.99)

**Controls**: Same as PPO/A2C. But SAC uses it in Bellman equation:
Q(s,a) = r + gamma * (Q_target(s',a') - alpha * log pi(a'|s'))

**Overfitting relationship**: Higher gamma means Q-values sum many noisy future rewards.
Bootstrapping error compounds across longer chains. Combined with off-policy data from
potentially different market regimes, this can cause Q-value divergence.

**Financial RL range**: 0.95-0.99. For swing trading: 0.95-0.97 better matches typical holding
periods and acknowledges distant prices are genuinely unpredictable.

### ent_coef (default: "auto")

**Controls**: Entropy coefficient alpha in objective J = E[sum(r + alpha * H(pi))].

**How "auto" works**: Learns log_alpha to keep policy entropy near target = -dim(action_space).
Starts at alpha=1.0 (or custom via "auto_0.1"). Typically decreases as policy improves.

**Overfitting relationship**: Auto-tuning starts with strong exploration (alpha=1.0) that
naturally regularizes. As the policy learns, alpha decreases allowing more exploitation. This
adaptive schedule is generally better than any fixed value.

**CRITICAL for capital preservation**: Default alpha=1.0 start can overwhelm small financial
reward signals (daily returns ~0.001). Consider "auto_0.1" to start lower. Ensure drawdown
penalties in your reward function outweigh the entropy incentive to try risky actions.

**When to use fixed**: If auto-tuning oscillates or if you need tight control. Fixed 0.01 is
reasonable. But auto is generally preferred for adapting to regime-dependent reward scales.

**Financial RL range**: "auto" (preferred) or "auto_0.1" for capital preservation. Fixed: 0.01-0.1.

### learning_starts (default: 10,000)

**Controls**: Environment steps before first gradient update. Random actions during warmup.

**Overfitting relationship**: Larger warmup = more diverse initial replay buffer = less early
overfitting. Small warmup = critic memorizes few transitions, creating "primacy bias" that's
hard to recover from.

**Financial RL range**: 5,000-50,000. Current 10,000 is good for equity. Consider 20,000 for
crypto (longer episodes need more initial coverage).

### buffer_size (default: 500,000)

**Controls**: Replay buffer capacity. FIFO eviction when full.

**Overfitting relationship**: Larger buffer = each transition sampled less often = less
overfitting. BUT for non-stationary financial data, old transitions from different market
regimes can mislead the critic. The Q-function learns a "weighted average" of all regimes in
the buffer, which may not reflect any actual regime.

**Financial RL range**: 100,000-500,000. Smaller (100K-200K) may actually improve learning
quality by keeping data regime-relevant. Tradeoff: diversity vs relevance.

### SAC Overfitting Diagnostic Summary

| Symptom | Primary HP Lever | Secondary |
|---------|-----------------|-----------|
| Q-values exploding | Reduce learning_rate | Reduce tau |
| High overfit_gap | Reduce learning_rate | Reduce buffer_size (regime staleness) |
| Over-exploration (random trades) | Use "auto_0.1" for ent_coef | Strengthen reward penalties |
| Premature exploitation | Increase ent_coef or use "auto" | Increase learning_starts |
| Regime confusion (bull/bear averaging) | Reduce buffer_size (100K-200K) | Reduce gamma |

---

## Cross-Algo Comparison: When to Adjust What

| Problem | PPO Fix | A2C Fix | SAC Fix |
|---------|---------|---------|---------|
| **Overfitting** | Reduce n_epochs (10->5), add target_kl | Reduce LR (first!), reduce gae_lambda | Reduce LR, reduce buffer_size |
| **Underfitting** | Increase LR, increase n_epochs | Increase LR, increase n_steps | Increase LR, increase batch_size |
| **Policy instability** | Narrow clip_range | Reduce LR (only lever!) | Reduce tau, reduce LR |
| **Poor exploration** | Increase ent_coef | Increase ent_coef | Use "auto" ent_coef, increase learning_starts |
| **Myopic trading** | Increase gamma | Increase gamma + n_steps | Increase gamma |
| **Capital preservation** | Lower gamma (0.95-0.97), narrow clip | Lower gamma, lower LR | Lower gamma, "auto_0.1" ent_coef |

## Reward Weight Adjustment Guide

The MemoryVecRewardWrapper shapes rewards using weights: {profit, sharpe, drawdown, turnover}.

| Observed Problem | Adjustment |
|-----------------|------------|
| High MDD, frequent large losses | Increase drawdown weight (0.2 -> 0.35) |
| Too many trades, high turnover | Increase turnover weight (0.05 -> 0.15) |
| Good Sharpe but low total return | Increase profit weight, decrease sharpe weight |
| Volatile returns, inconsistent | Increase sharpe weight (smooths returns) |
| Agent holds through drawdowns | Increase drawdown weight + decrease gamma |

Bounds: profit [0.1, 0.7], sharpe [0.1, 0.6], drawdown [0.05, 0.5], turnover [0.0, 0.2].
Weights should sum to ~1.0.
