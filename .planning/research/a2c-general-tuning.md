# A2C General Tuning Reference for SwingRL

Comprehensive reference for tuning Stable Baselines3 A2C in SwingRL's financial RL context.
Covers HP sensitivity, failure modes, optimizer choices, and automated tuning bounds.

**SwingRL context**: Equity daily (8 ETFs via Alpaca), crypto 4H (BTC/ETH via Binance.US).
Continuous action space: `Box(-1,1)` shape `(n_assets+1)` mapped to portfolio weights via softmax.

**Current config**: `lr=0.00015`, `n_steps=5`, `gamma=0.97`, `gae_lambda=1.0`, `ent_coef=0.015`,
`vf_coef=0.5`, `n_envs=6`, optimizer=RMSprop (SB3 default). `normalize_advantage=False`.

**Critical production evidence**: A2C equity improved +27% from baseline. Iter 3 gamma=0.999
caused -69% crypto Sharpe collapse. Treatment (reward adjustments) helps crypto (+30.5%) but
hurts equity (-26.1%), suggesting HP-reward interaction is env-dependent.

---

## 1. HP Sensitivity Ranking: No Clipping Changes Everything

### Why A2C Is Fundamentally More Sensitive Than PPO

Huang et al. (2022) proved that A2C is a strict special case of PPO where the surrogate clipping
mechanism is deactivated. When PPO runs with `n_epochs=1` and no clipping, it produces bit-for-bit
identical trajectories to A2C. This means every safety mechanism PPO provides --- clip_range,
multiple epochs with early stopping via target_kl, minibatch variance reduction --- is absent
in A2C.

The practical consequence: **every hyperparameter in A2C has a direct, unattenuated effect on
the policy update**. There is no "guardrail" that caps how far a single gradient step can move
the policy. The only hard constraint is `max_grad_norm` (gradient clipping at the parameter level,
default 0.5), which limits the L2 norm of the gradient vector but says nothing about the
direction or magnitude of the resulting policy change in probability space.

A comparative study at KDD 2024 (de la Fuente et al.) confirmed this empirically: "A2C
illustrated significant performance dependency on hyperparameter tuning" while "PPO exhibited
reasonable stability" across the same task configurations.

### Sensitivity Ranking (Highest to Lowest Impact)

| Rank | Parameter | Why This Rank |
|------|-----------|---------------|
| 1 | **learning_rate** | Only lever controlling update magnitude. No clipping safety net. |
| 2 | **gamma** | Defines effective horizon. Mismatch with n_steps = catastrophic (iter 3: -69%). |
| 3 | **n_steps** | Controls gradient quality. n_steps=5 with 6 envs = 30 transitions per update. |
| 4 | **gae_lambda** | With lambda=1.0 (current), using pure MC returns. Interacts multiplicatively with gamma. |
| 5 | **ent_coef** | Main regularizer. Too low = collapse. Too high = washes out reward signal. |
| 6 | **vf_coef** | Shared optimizer means critic gradient competes with policy gradient. |
| 7 | **max_grad_norm** | Hard safety floor. Almost never needs changing from 0.5. |

**Key insight**: In PPO, the sensitivity ranking would be `n_epochs > learning_rate > clip_range`.
In A2C, learning_rate absorbs ALL the sensitivity that PPO distributes across n_epochs, clip_range,
and batch_size. This is why A2C learning_rate tuning demands 2-3x more care than PPO.

**Citations**:
- Huang et al. 2022 --- "A2C is a special case of PPO" (arXiv:2205.09123)
- de la Fuente et al. 2024 --- "A Comparative Study of DRL Models: DQN vs PPO vs A2C" (KDD 2024)
- Engstrom et al. 2020 --- "Implementation Matters in Deep Policy Gradients" (ICLR 2020)

---

## 2. learning_rate: More Sensitive Than PPO

### Mechanism

Learning rate controls the step size for RMSprop (or Adam) on the shared actor+critic network.
In A2C, each gradient update is computed from a single rollout of `n_steps * n_envs` transitions.
With the current config (n_steps=5, n_envs=6), that is **30 transitions per gradient step**.

For comparison: PPO with n_steps=2048, n_envs=6 uses **12,288 transitions** per update cycle
(spread across multiple minibatch epochs). A2C's gradient has ~400x less data backing it.
Higher variance gradients amplified by a high learning rate = catastrophic overfitting.

### Interaction with RMSprop

RMSprop maintains per-parameter running averages of squared gradients:

```
v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
```

The adaptive scaling means the *effective* learning rate varies per parameter. Parameters that
receive consistently large gradients get smaller effective LR (stabilizing). Parameters with
sparse/small gradients get larger effective LR (destabilizing for financial signals).

This creates a specific interaction pattern for financial RL:
- **Frequently activated features** (price, volume): LR is naturally dampened by RMSprop
- **Rarely activated features** (drawdown signals, macro indicators): LR stays near nominal,
  causing outsized updates when they fire

SB3's default RMSprop uses `alpha=0.99` (smoothing) and `eps=1e-5`. The high alpha means
the running average adapts slowly, which is good for stability but bad for regime changes
where gradient statistics shift suddenly.

### Financial RL Evidence

Mnih et al. (2016) showed A3C/A2C was "robust to learning rates" on Atari, but this does NOT
transfer to financial RL. Atari has dense, high-magnitude rewards with stable distributions.
Financial returns have:
- Low SNR (daily returns ~0.001-0.01 magnitude)
- Non-stationary reward distributions (regime changes)
- Sparse meaningful signals (most days are noise)

The KDD 2024 comparative study confirmed "A2C displays greater sensitivity to learning rate
variations compared to DQN and PPO."

### Current Config Assessment

Current `lr=0.00015` (1.5e-4) is well within the safe range. SB3 default of 7e-4 would be
dangerously high for financial data.

### Recommended Range

| Context | Range | Notes |
|---------|-------|-------|
| Financial RL (conservative) | 5e-5 to 3e-4 | Current 1.5e-4 is well-centered |
| Financial RL (aggressive) | 3e-4 to 5e-4 | Only with increased n_steps (16+) |
| SB3 default (Atari) | 7e-4 | Too high for financial data |
| HP tuner bounds | [3e-5, 5e-4] log-uniform | Wider than manual range for exploration |

**Rule of thumb**: If increasing n_steps by factor K, learning_rate can increase by sqrt(K)
(linear scaling rule from minibatch SGD literature).

**Citations**:
- Mnih et al. 2016 --- "Asynchronous Methods for Deep RL" (ICML 2016)
- de la Fuente et al. 2024 --- "DQN vs PPO vs A2C" (KDD 2024)
- SB3 Tips and Tricks documentation

---

## 3. n_steps (Rollout Length): Short Rollouts Are a Feature AND a Bug

### Mechanism

n_steps controls how many environment steps each parallel environment takes before a gradient
update. Effective batch size = `n_steps * n_envs`.

| n_steps | n_envs=6 | Effective Batch | Equity Coverage | Crypto Coverage |
|---------|----------|-----------------|-----------------|-----------------|
| 5 | 6 | 30 | 1 week | 20 hours |
| 16 | 6 | 96 | ~3 weeks | 2.7 days |
| 32 | 6 | 192 | ~6 weeks | 5.3 days |
| 64 | 6 | 384 | ~3 months | 10.7 days |

### The n_steps / gamma Interaction (CRITICAL)

This is the most dangerous interaction in A2C tuning. The effective horizon of the value
function is `1 / (1 - gamma)`:

| gamma | Effective Horizon | With n_steps=5 | Mismatch Factor |
|-------|-------------------|-----------------|-----------------|
| 0.95 | 20 steps | 5 seen, 15 bootstrapped | 4x |
| 0.97 | 33 steps | 5 seen, 28 bootstrapped | 6.6x |
| 0.99 | 100 steps | 5 seen, 95 bootstrapped | 20x |
| 0.999 | 1000 steps | 5 seen, 995 bootstrapped | 200x |

**Mismatch factor** = ratio of bootstrapped-to-observed rewards. Higher = more critic burden.

When the mismatch factor is high, the critic must predict the value of states it has never
seen in the current rollout. For financial data where prices are non-stationary, this prediction
is unreliable. The critic effectively hallucinates future values, and the policy gradient
computed from these hallucinated advantages pushes the agent in arbitrary directions.

**This is exactly what happened in iter 3**: gamma=0.999 gave a mismatch factor of 200x with
n_steps=5. The critic was asked to predict ~1000 steps of crypto 4H returns (~167 days) from
5 observed steps (~20 hours). The result: -69% Sharpe collapse.

### Why Short Rollouts Can Be Beneficial

Short rollouts are not purely bad. For non-stationary financial environments, they provide:

1. **Recency bias**: Each gradient is based on the most recent data, not stale history
2. **Fast adaptation**: More gradient updates per episode = faster response to regime changes
3. **Memory efficiency**: Small rollout buffers use less RAM
4. **Compatibility with on-policy learning**: A2C discards data after each update, so short
   rollouts mean less wasted computation

The original A3C paper (Mnih et al. 2016) used n_steps=5 for Atari and achieved SOTA. But
Atari environments are stationary (same game rules), whereas financial markets are not.

### Recommendation for SwingRL

The current n_steps=5 is viable IF gamma is kept low (0.95-0.97). The danger is increasing
gamma without increasing n_steps. If the HP tuner wants to increase gamma above 0.97, it
MUST also increase n_steps proportionally.

**Safe combinations**:

| gamma | Minimum n_steps | Mismatch Factor |
|-------|-----------------|-----------------|
| 0.95 | 5 | 4x (acceptable) |
| 0.97 | 8-10 | ~3x |
| 0.98 | 16 | ~3x |
| 0.99 | 32 | ~3x |

**Note**: n_steps is currently NOT exposed in bounds.py for the HP tuner. This is a significant
gap --- see Section 13 for recommendation.

**Citations**:
- Mnih et al. 2016 --- "Asynchronous Methods for Deep RL" (ICML 2016)
- Andrychowicz et al. 2021 --- "What Matters in On-Policy RL?" (ICLR 2021)

---

## 4. gamma: HIGH-RISK Parameter

### Why gamma Is Uniquely Dangerous in A2C

gamma (discount factor) controls how much the agent values future rewards relative to immediate
ones. Effective horizon = `1 / (1 - gamma)`.

In PPO, gamma interacts with a large rollout buffer (n_steps=2048) that provides substantial
real data for the critic to train on. In A2C with n_steps=5, the critic sees 5 real transitions
and must bootstrap the rest. gamma determines how much "the rest" matters.

### The Critic Prediction Burden

The value function V(s) estimates the expected discounted return from state s:

```
V(s) = E[r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^T*r_T]
```

With n_steps=5, A2C computes the actual returns for steps 0-4, then bootstraps:

```
G_t = r_t + gamma*r_{t+1} + ... + gamma^4*r_{t+4} + gamma^5 * V(s_{t+5})
```

The gamma^5 * V(s_{t+5}) term carries ALL the information about steps 5 through infinity.
With gamma=0.999, this term contributes `0.999^5 = 0.995` of the total value estimate to a
single critic prediction. The 5 observed rewards contribute only 0.5% of the signal.

**This is why gamma=0.999 collapsed crypto**: The agent's policy was driven almost entirely
by the critic's (unreliable) prediction of ~167 days of future crypto returns, not by the
20 hours of actual observed data.

### Discount Factor as Regularizer

Amit et al. (2020) formalized that "indirect regularization can be applied by running the
learning algorithm with a discount factor lower than specified by the task (discount
regularization)." Lower gamma:
- Reduces variance by downweighting uncertain distant returns
- Focuses learning on short-term patterns where predictions are more reliable
- Acts as implicit horizon truncation

For financial RL where long-horizon predictions are genuinely unreliable (markets are
non-stationary), this regularization is beneficial, not harmful.

### Production Evidence

| Iter | gamma | Env | Sharpe Delta | Notes |
|------|-------|-----|-------------|-------|
| Baseline | 0.97 | Equity | -- | Starting point |
| Iter 1 | 0.96 | Equity | +42% (A2C) | Lower gamma helped equity |
| Iter 1 | 0.96 | Crypto | Regression | Too myopic for 4H bars |
| Iter 3 | 0.999 | Crypto | -69% | Catastrophic critic burden |
| Current | 0.97 | Both | Stable | Good balance |

### Recommended Bounds

| Context | Range | Notes |
|---------|-------|-------|
| Equity daily | 0.95 - 0.98 | 20-50 day effective horizon matches swing trading |
| Crypto 4H | 0.96 - 0.985 | 25-67 bar horizon (4-11 days) |
| Combined (current bounds.py) | 0.95 - 0.985 | Algo-specific A2C bounds |
| HP tuner | [0.95, 0.985] | Current bounds are correct |

**MANDATORY CONSTRAINT**: gamma * n_steps should keep mismatch factor below 8x. Any gamma
suggestion from the HP tuner should be validated against current n_steps:
`mismatch = (1 / (1 - gamma)) / n_steps`. If mismatch > 8, reject or simultaneously increase
n_steps.

**Citations**:
- Amit et al. 2020 --- "Discount Factor as a Regularizer in RL" (arXiv:2007.02040)
- Hu et al. 2022 --- "On the Role of Discount Factor in Offline RL" (arXiv:2206.03383)
- SwingRL Iter 3 Analysis (internal)

---

## 5. gae_lambda: Pure MC at lambda=1.0 Is Suboptimal with n_steps=5

### GAE Recap

Generalized Advantage Estimation (Schulman et al. 2016) computes advantage as an exponentially
weighted average of n-step advantage estimates:

```
A^GAE(gamma, lambda) = sum_{l=0}^{T-1} (gamma * lambda)^l * delta_{t+l}
```

Where `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)` is the TD error.

- **lambda=0**: Pure 1-step TD. `A_t = delta_t`. Low variance, high bias (relies entirely on
  critic accuracy).
- **lambda=1**: Pure Monte Carlo. `A_t = sum(gamma^l * r_{t+l}) - V(s_t)`. Zero bias if
  trajectory is complete, but high variance from noisy returns.
- **lambda=0.95**: Blended. Exponentially decays the weight on longer n-step estimates,
  creating a soft cutoff that balances bias and variance.

### The lambda/n_steps Interaction

With n_steps=5, the GAE sum only has 5 terms regardless of lambda. So lambda=1.0 means:

```
A_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + gamma^3*delta_{t+3} + gamma^4*delta_{t+4}
```

All 5 TD errors are weighted equally (up to gamma discounting). There is no "soft cutoff"
because the trajectory is truncated at 5 steps anyway.

**However**, the effective discount for the advantage changes with lambda:
- lambda=1.0: effective discount = gamma^k (only gamma dampens)
- lambda=0.95: effective discount = (gamma * 0.95)^k (faster dampening)

With gamma=0.97 and lambda=1.0: effective discounts = [1.0, 0.97, 0.94, 0.91, 0.89]
With gamma=0.97 and lambda=0.92: effective discounts = [1.0, 0.89, 0.80, 0.71, 0.63]

The lambda=0.92 version places **29% less weight on step 4** compared to lambda=1.0. For
noisy financial data, this means less attribution of market noise to the agent's early actions.

### Current Assessment

gae_lambda=1.0 with n_steps=5 means A2C is using pure Monte Carlo advantage estimation on
5-step truncated trajectories. This is the **maximum variance** configuration for GAE with
this rollout length.

The SB3 A2C default is lambda=1.0 (inherited from Mnih 2016). PPO defaults to lambda=0.95.
The PPO default is more appropriate for financial RL.

### Recommendation

Reduce gae_lambda from 1.0 to 0.90-0.95. This is one of the **highest-leverage, lowest-risk
changes** available:
- Reduces gradient variance without changing the problem formulation
- No risk of catastrophic failure (unlike gamma changes)
- Monotonic effect: lower lambda = lower variance, slightly higher bias

**Recommended range for HP tuner**: [0.85, 1.0]

The effective horizon with GAE is approximately `1 / (1 - gamma * lambda)`:
- gamma=0.97, lambda=1.0: effective horizon ~33 steps
- gamma=0.97, lambda=0.92: effective horizon ~9 steps

For swing trading with daily bars, an effective horizon of 9-33 steps (roughly 2 weeks to
1.5 months) is appropriate.

**Citations**:
- Schulman et al. 2016 --- "High-Dimensional Continuous Control Using GAE" (ICLR 2016)
- Andrychowicz et al. 2021 --- "What Matters in On-Policy RL?" (ICLR 2021)

---

## 6. ent_coef: The Main Regularizer Without Clipping

### Why Entropy Is Critical for A2C

In PPO, three mechanisms prevent policy collapse:
1. Clipping (limits policy ratio to [1-eps, 1+eps])
2. Multiple epochs with diminishing returns
3. Entropy bonus

In A2C, only mechanism 3 exists. Entropy regularization is the **sole explicit mechanism**
preventing the policy from collapsing to a deterministic (and overfit) mapping.

The entropy bonus is added to the objective:

```
L = L_policy + ent_coef * H(pi) - vf_coef * L_value
```

Where H(pi) is the entropy of the action distribution. For continuous actions (Gaussian),
entropy = `0.5 * log(2*pi*e * sigma^2)` per dimension. Higher sigma = higher entropy =
more exploration.

### Entropy-Reward Scale Interaction

This is a subtle but critical interaction for financial RL. The policy loss has magnitude
proportional to the advantage, which scales with reward magnitude. Daily equity returns
are ~0.001-0.01 in magnitude. If ent_coef is too large relative to the reward signal,
the entropy bonus dominates and the agent prioritizes action diversity over profit.

Conversely, if ent_coef is too small, the policy rapidly converges to whatever pattern
it found first (potentially spurious), with no mechanism to escape.

### Production Evidence

| Iter | ent_coef | Outcome | Notes |
|------|----------|---------|-------|
| Iter 0 | 0.0 | Baseline | SB3 default. No entropy regularization at all. |
| Iter 1 | 0.025 | Regression | Too high --- exceeded [0.005, 0.02] range, washed out signals |
| Current | 0.015 | Stable | Within recommended range |

A2C's default ent_coef in SB3 is 0.0 (no entropy bonus). This is safe for Atari where
reward signals are strong but dangerous for financial RL where signals are weak.

### Diagnostic

Monitor entropy during training via TensorBoard/logging:
- **Entropy decreasing steadily**: Normal learning, policy becoming more confident
- **Entropy near zero**: Policy collapse imminent. Increase ent_coef immediately.
- **Entropy not decreasing**: ent_coef too high or reward signal too weak
- **Entropy oscillating**: Learning rate too high, policy unstable

### Recommended Range

| Context | Range | Notes |
|---------|-------|-------|
| Financial RL | 0.005 - 0.025 | Current 0.015 is good |
| Capital preservation | 0.01 - 0.02 | Slightly higher to avoid concentration risk |
| HP tuner bounds | [0.001, 0.05] | Current bounds.py range |
| Decay schedule | 0.02 -> 0.005 | Explore early, exploit late (not yet implemented) |

**Citations**:
- Mnih et al. 2016 --- "Asynchronous Methods for Deep RL" (ICML 2016)
- Williams & Peng 1991 --- "Function Optimization Using Connectionist RL Algorithms" (original
  entropy regularization in policy gradient)
- arXiv:1912.01557 --- "Policy Optimization RL with Entropy Regularization"

---

## 7. vf_coef: Shared Optimizer Gradient Competition

### Mechanism

vf_coef controls the weight of the value function loss in the combined loss:

```
L_total = L_policy + ent_coef * H(pi) - vf_coef * L_vf
```

Since A2C uses a single optimizer for the shared network, the policy gradient and value
gradient compete for the shared representation layers. The gradient from the critic
(scaled by vf_coef) can overwhelm the policy gradient if:
1. Value loss is large (common early in training when V(s) is poorly initialized)
2. vf_coef is too high
3. Policy gradient is small (common with weak financial signals)

### The Scale Mismatch Problem

The policy loss involves `log_prob(action) * advantage`. For continuous actions, log_prob
can be in the range [-10, 0] and advantage magnitudes for financial data are small (~0.001-0.1).
So policy loss magnitude is ~0.001-1.0.

The value loss is `MSE(V_pred, V_target)`. If V(s) is predicting cumulative discounted returns
and gamma=0.97, V can be in the range [-1, 5] for financial data. MSE of these values gives
loss magnitudes of ~0.1-10.

With vf_coef=0.5, the value gradient is 0.5 * (0.1-10) = 0.05-5.0, which can easily
dominate the policy gradient of 0.001-1.0. This means the shared layers are primarily
trained to be a good value estimator, with policy learning as a side effect.

### When to Adjust

- **High overfit_gap (IS >> OOS)**: Reduce vf_coef to 0.25-0.4. The critic may be overfitting
  and dominating the shared representation.
- **Poor value estimates (explained_variance < 0.3)**: Increase vf_coef to 0.5-0.75 temporarily.
- **Policy not learning (entropy flat, rewards flat)**: Check if value gradient is drowning
  policy gradient. Reduce vf_coef.

### Separate Networks Consideration

Andrychowicz et al. (2021) found that "separate policy and value networks lead to better
performance on 4 out of 5 environments" in their large-scale on-policy RL study. However, SB3's
A2C uses a shared network by default (shared feature extractor with separate heads). Switching
to fully separate networks would eliminate the gradient competition but double memory usage.

### Recommended Range

| Context | Range | Notes |
|---------|-------|-------|
| Default | 0.5 | SB3 default, usually fine |
| Financial RL | 0.25 - 0.75 | Adjust based on gradient diagnostics |
| HP tuner bounds | [0.1, 1.0] | Wide range, let tuner explore |

**Citations**:
- Andrychowicz et al. 2021 --- "What Matters in On-Policy RL?" (ICLR 2021)
- SB3 A2C documentation

---

## 8. RMSprop vs Adam: SB3 Defaults and When to Switch

### SB3 A2C Defaults

SB3 A2C uses RMSprop by default (`use_rms_prop=True`) with:
- `alpha=0.99` (smoothing factor for running average of squared gradients)
- `eps=1e-5` (numerical stability)
- `weight_decay=0` (no L2 regularization)
- No momentum

This matches the original A3C paper (Mnih et al. 2016) which used shared RMSprop across
asynchronous workers.

### RMSprop Characteristics for Financial RL

**Advantages**:
- Per-parameter adaptive learning rate dampens frequently-updated parameters
- No momentum means no overshooting on regime changes
- Simple: fewer hyperparameters than Adam (no beta1, beta2)
- Historically validated for non-stationary RL objectives

**Disadvantages**:
- No first-moment tracking (no momentum correction) --- slower convergence
- alpha=0.99 means very slow adaptation to gradient statistic changes
- eps=1e-5 is aggressive --- large gradients from price shocks may cause instability
- Known numerical differences between PyTorch and TensorFlow implementations

### Adam Characteristics for Financial RL

**Advantages**:
- First-moment tracking (momentum) provides smoother gradient estimates
- Bias correction in early training prevents cold-start artifacts
- Better large-batch behavior (Stooke & Abbeel 2018, "Accelerated Methods for Deep RL")
- Default in most modern deep learning --- more battle-tested

**Disadvantages**:
- More hyperparameters (beta1, beta2, eps) to tune
- Momentum can cause overshooting on regime changes
- Can mask learning rate sensitivity (making debugging harder)

### The RMSpropTFLike Issue

SB3 provides `RMSpropTFLike` in `stable_baselines3.common.sb2_compat.rmsprop_tf_like` for
backward compatibility with Stable Baselines 2. The issue: PyTorch's RMSprop implementation
differs numerically from TensorFlow's. If training seems unstable with default RMSprop,
switching to RMSpropTFLike can help.

### Recommendation for SwingRL

**Keep RMSprop** for now. Reasons:
1. Current training is stable with the existing optimizer
2. The no-momentum property is actually beneficial for non-stationary financial data
3. Switching optimizer requires re-tuning learning_rate and potentially other HPs
4. The Nota (2019) study on RAdam found "the choice between Adam and RMSprop can be empirical"
   and stability was more about LR/gradient clipping than optimizer variant

**Consider switching to Adam if**:
- Training shows slow convergence despite appropriate learning_rate
- Need to increase n_steps significantly (Adam is better with larger effective batches)
- Want to use learning rate warmup (Adam's bias correction makes warmup more natural)

To switch in SB3: `A2C(policy_kwargs=dict(optimizer_class=torch.optim.Adam, optimizer_kwargs=dict(eps=1e-5)))` and set `use_rms_prop=False`.

### Recommended eps Values

The original DQN paper (Mnih et al. 2015) used eps=0.01 for RMSprop, much larger than the
default 1e-5. For financial RL with potentially tiny gradients on some features, a larger
eps (1e-4 to 1e-3) provides additional stability by preventing extremely large effective
learning rates on sparse features.

**Citations**:
- Mnih et al. 2016 --- "Asynchronous Methods for Deep RL" (ICML 2016)
- Stooke & Abbeel 2018 --- "Accelerated Methods for Deep RL" (arXiv:1803.02811)
- Nota 2019 --- "RAdam: A New SOTA Optimizer for RL?" (Medium/Autonomous Learning Library)
- SB3 A2C documentation

---

## 9. A2C Failure Modes

### Mode 1: Policy Collapse (Entropy -> 0)

**Mechanism**: Without clipping, a single strong gradient can push the policy toward
determinism. Once entropy approaches zero, the policy samples nearly identical actions
regardless of state. The gradient then reinforces whatever action was taken (since it's the
only action sampled), creating a self-reinforcing collapse.

**Detection**: Entropy metric drops below 50% of initial value within first 10% of training.

**Causes**:
- ent_coef too low (especially 0.0, the SB3 default)
- learning_rate too high --- single update pushes sigma too low
- Reward scale too large relative to ent_coef

**Fix**:
1. Increase ent_coef (primary)
2. Reduce learning_rate (secondary)
3. Clip action log-std to minimum value (architecture change)

### Mode 2: Gradient Explosion

**Mechanism**: A price shock (e.g., crypto flash crash) creates an extreme reward signal.
With n_steps=5, this one event can dominate the entire advantage estimate. The resulting
gradient can be orders of magnitude larger than normal.

max_grad_norm=0.5 clips the gradient norm, but the *direction* of the clipped gradient still
points toward the extreme event. Repeated extreme events in a volatile period can push the
policy into an unstable region.

**Detection**: Loss spikes > 10x running average. Value function predictions diverge.

**Causes**:
- Reward not normalized or clipped
- Extremely volatile period without VecNormalize
- max_grad_norm too large

**Fix**:
1. Ensure reward wrapper clips extreme returns
2. Use VecNormalize for observation normalization (but NOT reward normalization, which
   conflicts with the memory agent's reward weight adjustments)
3. Keep max_grad_norm at 0.5

### Mode 3: n_steps/gamma Mismatch (Critic Hallucination)

**Mechanism**: As described in Section 4, high gamma with low n_steps forces the critic to
predict values based almost entirely on bootstrapping, not observed data. The critic's
predictions become detached from reality, and the policy follows these hallucinated advantages.

**Detection**:
- explained_variance near 0 or negative
- Value function predictions diverge from actual returns
- Sharpe collapses despite "normal" training metrics

**Causes**:
- gamma increased without corresponding n_steps increase
- SwingRL production evidence: gamma=0.999 with n_steps=5 = -69% Sharpe

**Fix**:
1. Reduce gamma (primary)
2. Increase n_steps (secondary)
3. Enforce mismatch_factor < 8 constraint in bounds.py

### Mode 4: Reward Weight Whipsaw

**Mechanism**: The memory agent changes reward weights between epochs. On-policy A2C
immediately reflects these changes in the next gradient (no replay buffer averaging). Large
weight changes cause the value function to become stale (trained on old reward distribution)
and advantages become misleading.

**Detection**: Performance oscillates between weight adjustment epochs.

**Causes**:
- Reward weight adjustment magnitude too large per step
- Value function not given time to adapt to new reward distribution

**Fix**:
1. Cap per-adjustment weight change (max 0.1 per component per adjustment)
2. Minimum epochs between adjustments for value function adaptation

### Mode 5: Shared Network Representational Collapse

**Mechanism**: The value function gradient dominates the shared layers, pushing the
representation toward pure value prediction. The policy head receives features optimized
for value estimation, not action selection. Policy performance degrades while value metrics
look healthy.

**Detection**: Good explained_variance but poor policy performance. Policy entropy stable
but actions not improving.

**Causes**:
- vf_coef too high relative to policy gradient magnitude
- Value loss scale much larger than policy loss scale

**Fix**:
1. Reduce vf_coef
2. Consider separate networks (SB3 `net_arch=dict(pi=[64,64], vf=[64,64])`)

**Citations**:
- OpenReview 2024 --- "Overcoming Policy Collapse in Deep RL"
- arXiv:2512.04220 --- "On Group Relative Policy Optimization Collapse" (LLD failure mode)
- arXiv:2505.22617 --- "The Entropy Mechanism of RL for Reasoning Language Models"
- Weng 2018 --- "Policy Gradient Algorithms" (lilianweng.github.io)

---

## 10. Why A2C Responds Better to Reward Adjustments Than PPO

### The Direct Gradient Path

A2C processes reward changes through a single, direct path:

```
Reward change -> Advantage estimate -> Single gradient step -> Policy update
```

PPO processes reward changes through a buffered, constrained path:

```
Reward change -> Advantage estimate -> K epochs of clipped minibatch updates -> Policy update
```

The clipping in PPO limits how much a single reward change can move the policy. If the reward
weight shift causes a large advantage change, the clipped ratio `min(r * A, clip(r) * A)`
truncates the update. A2C has no such truncation.

### On-Policy Freshness

A2C discards all data after each gradient update. This means:
1. The NEXT update after a reward change uses ONLY data generated under the new reward
2. There is no "stale data" contamination from the old reward distribution
3. The value function immediately starts learning the new reward distribution

PPO's larger rollout buffer means updates after a reward change mix old-reward and new-reward
transitions within the same rollout. The K epochs of updates on this mixed data create a
muddled gradient signal.

### Production Evidence

SwingRL iter 1 data confirms this asymmetry:
- **A2C equity**: reward adjustments correlated with +42% Sharpe improvement
- **A2C drawdown weight** converged to ~0.33 across all 14 folds (consistent adaptation)
- **PPO**: folds WITH reward adjustments performed WORSE than folds WITHOUT (-1.14 vs -0.54)

The Blom et al. (2024) finding that "reward function and hyperparameters are mutually dependent"
explains why: PPO's clipping interacts with reward scale in complex ways, while A2C's direct
gradient means reward changes have predictable, proportional effects.

### Caveat: Sensitivity Is a Double-Edged Sword

A2C responds MORE to reward adjustments, but it also responds more to BAD adjustments.
Production evidence: treatment helps crypto (+30.5%) but hurts equity (-26.1%). The same
directness that makes A2C responsive to correct guidance also makes it vulnerable to incorrect
guidance.

**Implication for the HP tuner**: Reward weight adjustment bounds should be TIGHTER for A2C
than for PPO, because each adjustment has a larger, more immediate effect.

**Citations**:
- Blom et al. 2024 --- "Combining Automated Optimisation of HPs and Reward Shape" (arXiv)
- SwingRL Iter 1 production data (internal)
- Ng et al. 1999 --- "Policy Invariance Under Reward Transformations" (ICML 1999)

---

## 11. A2C in Non-Stationary Environments

### Why A2C Is Structurally Suited for Financial Markets

Financial markets exhibit non-stationarity at multiple timescales:
- **Intraday**: Volatility clusters, microstructure noise
- **Daily**: Trend regimes, mean-reversion periods
- **Monthly-Quarterly**: Sector rotation, macro regime shifts
- **Yearly**: Business cycles, secular trends

A2C's architecture has several properties that match these challenges:

**1. Short memory, fresh gradients**: With n_steps=5 and no replay buffer, A2C's gradients
are always computed from the most recent data. Unlike SAC (which may have months-old data
in its replay buffer), A2C never trains on stale market conditions.

**2. No regime-averaging**: SAC's replay buffer creates a value function that averages across
all regimes in the buffer. If the buffer contains both bull and bear data, Q-values represent
a "weighted average" regime that may not reflect any actual market condition. A2C's value
function always reflects the current trajectory's regime.

**3. Fast policy adaptation**: A2C makes one gradient step per rollout. With n_steps=5 and
n_envs=6, that's a policy update every 30 transitions. For daily equity, that's ~5 updates
per month per env. This frequency allows the policy to track regime changes within weeks.

### FinRL Ensemble Evidence

Liu et al. (2021) and the FinRL framework's ensemble strategy dynamically selects between PPO,
A2C, and DDPG based on market turbulence:
- **A2C** achieves lowest annual volatility (10.4%) and lowest MDD (-10.2%)
- **PPO** achieves highest returns (15.0% annual)
- **Ensemble** outperforms all individuals (Sharpe 2.81 vs A2C 2.24, PPO 2.23)

The pattern: A2C is the conservative, risk-aware learner. It naturally gravitates toward
lower-volatility strategies. This aligns with SwingRL's capital preservation objective.

Yang et al. (2020) showed this ensemble approach "effectively preserves robustness under
different market conditions" across pre-COVID, COVID crash, recovery, inflation downturn,
and partial recovery periods.

### Limitation: A2C in Trending Markets

A2C's short-horizon nature means it can underperform in strong trends. When markets trend
persistently (2020-2021 bull run), PPO's longer rollouts capture the trend signal better.
A2C's 5-step window may see the trend as noise.

**Mitigation**: Increase gamma slightly (0.97 -> 0.98) during trending periods, or use the
ensemble weighting to increase PPO allocation during trends.

**Citations**:
- Liu et al. 2021 --- "FinRL: Deep RL Framework to Automate Trading" (arXiv)
- Yang et al. 2020 --- "Deep RL for Automated Stock Trading: An Ensemble Strategy" (ICAIF)
- Nature 2026 --- "Behaviorally Informed Deep RL for Portfolio Optimization"

---

## 12. SB3 A2C Implementation Specifics

### Key Parameters and Their SB3 Defaults

```python
A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,       # SwingRL: 1.5e-4 (correctly reduced)
    n_steps=5,                # SwingRL: 5 (matches default)
    gamma=0.99,               # SwingRL: 0.97 (correctly reduced)
    gae_lambda=1.0,           # SwingRL: 1.0 (consider reducing)
    ent_coef=0.0,             # SwingRL: 0.015 (correctly increased)
    vf_coef=0.5,              # SwingRL: 0.5 (default)
    max_grad_norm=0.5,        # SwingRL: 0.5 (default)
    rms_prop_eps=1e-5,        # SwingRL: default
    use_rms_prop=True,        # SwingRL: True (default)
    normalize_advantage=False, # SwingRL: False (see below)
    # stats_window_size=100,  # For logging running averages
    # use_sde=False,          # State-dependent exploration
    # sde_sample_freq=-1,     # SDE re-sample frequency
)
```

### normalize_advantage: Currently False

When `normalize_advantage=True`, SB3 normalizes advantages per rollout:
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Should SwingRL enable this?**

Arguments for:
- Reduces gradient magnitude dependence on reward scale
- Makes training more stable across different reward weight configurations
- Andrychowicz et al. (2021) found advantage normalization improves performance

Arguments against:
- With n_steps=5 and n_envs=6, normalization is over only 30 values --- high variance
  in the normalization statistics themselves
- The memory agent's reward weight adjustments change reward scale deliberately; normalizing
  away the scale change reduces the agent's responsiveness to weight adjustments
- Can mask reward weight effects, making the memory agent's signal less interpretable

**Recommendation**: Keep `normalize_advantage=False` for SwingRL. The small effective batch
size makes normalization statistics unreliable, and it conflicts with the memory agent's
reward shaping mechanism.

### RMSprop Implementation Note

SB3's A2C documentation explicitly warns: "If you find training unstable or want to match
performance of stable-baselines A2C, consider using RMSpropTFLike optimizer." The PyTorch
RMSprop and TensorFlow RMSprop have numerical differences that can affect training stability.

If instability is observed with no clear HP cause, try:
```python
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
A2C(policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
```

### Known SB3 A2C Issues

1. **ent_coef=0.0 default**: Unlike PPO (ent_coef=0.01), A2C defaults to NO entropy bonus.
   This is appropriate for Atari but dangerous for financial RL. SwingRL correctly overrides
   to 0.015.

2. **No learning rate schedule**: SB3 A2C supports `learning_rate` as a callable schedule,
   but this is not commonly used. Linear decay could help: start at 1.5e-4, end at 5e-5.

3. **VecNormalize interaction**: If using VecNormalize for observation normalization, the
   running statistics are updated during rollout collection. With n_steps=5, statistics
   update very frequently, which can cause observation normalization to oscillate during
   volatile periods.

**Citations**:
- SB3 A2C documentation (stable-baselines3.readthedocs.io)
- SB3 source code (github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py)
- Andrychowicz et al. 2021 --- "What Matters in On-Policy RL?" (ICLR 2021)

---

## 13. Parameters the HP Tuner Should Control

### Currently in bounds.py

| Parameter | Current Bounds | Assessment |
|-----------|---------------|------------|
| learning_rate | [1e-5, 1e-3] | Good. Could narrow to [3e-5, 5e-4] for A2C. |
| gamma | [0.95, 0.985] (A2C-specific) | Good. Well-informed by iter 3 failure. |
| entropy_coeff | [0.0, 0.05] | Good. Lower bound of 0.0 is risky for A2C --- consider 0.005. |
| batch_size | [32, 512] | N/A for A2C (batch_size not used; n_steps*n_envs is the batch). |
| clip_range | [0.1, 0.4] | N/A for A2C (no clipping). |
| n_epochs | [3, 20] | N/A for A2C (always 1 epoch). |

### Missing from bounds.py (Should Add)

| Parameter | Suggested Bounds | Priority | Rationale |
|-----------|-----------------|----------|-----------|
| **gae_lambda** | [0.85, 1.0] | HIGH | Currently hardcoded at 1.0. Pure MC is suboptimal. |
| **n_steps** | [5, 64] | HIGH | n_steps/gamma mismatch is the #1 failure mode. |
| **vf_coef** | [0.1, 1.0] | MEDIUM | Gradient competition with policy loss. |
| **max_grad_norm** | [0.3, 1.0] | LOW | Rarely needs changing. Safety floor. |
| **use_rms_prop** | {True, False} | LOW | Optimizer choice. Rare to switch. |
| **rms_prop_eps** | [1e-5, 1e-3] | LOW | Stability parameter. Edge case only. |

### Parameters That Should NOT Be Tuned

| Parameter | Why Fixed |
|-----------|-----------|
| n_envs | Architecture choice (6 parallel envs). Changing requires infra changes. |
| normalize_advantage | Should stay False for memory agent compatibility. |
| use_sde | State-dependent exploration is experimental and poorly validated for finance. |
| Network architecture | MLP [64, 64] is fixed. Architecture search is a separate concern. |

---

## 14. Recommended Bounds for Automated HP Tuning

### A2C-Specific Tuning Bounds

These bounds are informed by all preceding sections and validated against SwingRL production
data. They are designed for Optuna or similar automated tuners using log-uniform sampling for
learning_rate and uniform sampling for others.

```python
A2C_HP_BOUNDS = {
    # Core parameters (high sensitivity)
    "learning_rate": {
        "range": (3e-5, 5e-4),
        "sampling": "log_uniform",
        "default": 1.5e-4,
        "sensitivity": "CRITICAL",
        "notes": "Only lever controlling update magnitude. No clipping safety net.",
    },
    "gamma": {
        "range": (0.95, 0.985),
        "sampling": "uniform",
        "default": 0.97,
        "sensitivity": "CRITICAL",
        "notes": "MUST validate mismatch_factor < 8 against n_steps.",
        "constraint": "mismatch_factor = (1/(1-gamma)) / n_steps < 8",
    },
    "n_steps": {
        "range": (5, 64),
        "sampling": "categorical",  # [5, 8, 10, 16, 20, 32, 48, 64]
        "default": 5,
        "sensitivity": "HIGH",
        "notes": "Short rollouts = fresh data but noisy gradients. Must scale with gamma.",
    },
    "gae_lambda": {
        "range": (0.85, 1.0),
        "sampling": "uniform",
        "default": 1.0,
        "sensitivity": "HIGH",
        "notes": "lambda=1.0 is max variance. 0.90-0.95 recommended for financial RL.",
    },

    # Regularization parameters (medium sensitivity)
    "ent_coef": {
        "range": (0.005, 0.03),
        "sampling": "log_uniform",
        "default": 0.015,
        "sensitivity": "MEDIUM",
        "notes": "Main regularizer. Lower bound 0.005 prevents collapse. Upper bound prevents signal washout.",
    },
    "vf_coef": {
        "range": (0.1, 1.0),
        "sampling": "uniform",
        "default": 0.5,
        "sensitivity": "MEDIUM",
        "notes": "Gradient competition in shared network. Lower if policy stagnates.",
    },

    # Safety parameters (low sensitivity, rarely change)
    "max_grad_norm": {
        "range": (0.3, 1.0),
        "sampling": "uniform",
        "default": 0.5,
        "sensitivity": "LOW",
        "notes": "Hard safety floor. Almost never needs changing.",
    },
}
```

### Reward Weight Bounds (A2C-Specific Tightening)

Because A2C responds more directly to reward changes (Section 10), consider tighter bounds
than what is used for PPO/SAC:

```python
A2C_REWARD_BOUNDS = {
    "profit":   (0.15, 0.55),   # Narrower than universal (0.10, 0.70)
    "sharpe":   (0.15, 0.50),   # Narrower than universal (0.10, 0.60)
    "drawdown": (0.10, 0.40),   # Narrower than universal (0.05, 0.50)
    "turnover": (0.00, 0.15),   # Narrower than universal (0.00, 0.20)
}
```

### Constraint: gamma/n_steps Mismatch

The HP tuner MUST enforce this constraint on every configuration it proposes:

```python
def validate_a2c_config(gamma: float, n_steps: int) -> bool:
    """Reject configs where critic prediction burden is too high."""
    effective_horizon = 1.0 / (1.0 - gamma)
    mismatch_factor = effective_horizon / n_steps
    return mismatch_factor < 8.0  # Must be True
```

Configurations failing this check should be rejected before evaluation, saving training time.

### Per-Environment Considerations

| Parameter | Equity (Daily) | Crypto (4H) | Reasoning |
|-----------|---------------|-------------|-----------|
| gamma | 0.95 - 0.98 | 0.96 - 0.985 | Crypto has more bars per period, can afford longer horizon |
| n_steps | 5 - 32 | 5 - 64 | Crypto has 6x more bars per day |
| ent_coef | 0.01 - 0.025 | 0.005 - 0.02 | Equity needs more exploration (8 assets vs 2) |
| gae_lambda | 0.88 - 0.97 | 0.90 - 1.0 | Equity benefits more from variance reduction |

---

## Appendix A: Complete Reference Table

| Parameter | SB3 Default | SwingRL Current | Recommended Range | Sensitivity |
|-----------|-------------|-----------------|-------------------|-------------|
| learning_rate | 7e-4 | 1.5e-4 | 3e-5 - 5e-4 | CRITICAL |
| n_steps | 5 | 5 | 5 - 64 | HIGH |
| gamma | 0.99 | 0.97 | 0.95 - 0.985 | CRITICAL |
| gae_lambda | 1.0 | 1.0 | 0.85 - 1.0 | HIGH |
| ent_coef | 0.0 | 0.015 | 0.005 - 0.03 | MEDIUM |
| vf_coef | 0.5 | 0.5 | 0.1 - 1.0 | MEDIUM |
| max_grad_norm | 0.5 | 0.5 | 0.3 - 1.0 | LOW |
| rms_prop_eps | 1e-5 | 1e-5 | 1e-5 - 1e-3 | LOW |
| normalize_advantage | False | False | Keep False | N/A |
| use_rms_prop | True | True | Keep True | LOW |

## Appendix B: Diagnostic Decision Tree

```
Training underperforming?
|
+-- Sharpe collapsing rapidly?
|   |
|   +-- Check gamma/n_steps mismatch factor
|   |   +-- Mismatch > 8x? -> Reduce gamma or increase n_steps
|   |
|   +-- Check entropy
|       +-- Entropy near 0? -> Increase ent_coef, reduce learning_rate
|       +-- Entropy stable? -> Check value function (Mode 5, vf_coef too high)
|
+-- High overfit gap (IS >> OOS)?
|   |
|   +-- Reduce learning_rate (FIRST action for A2C)
|   +-- Reduce gae_lambda (1.0 -> 0.92)
|   +-- Check if reward adjustments are too aggressive
|
+-- Policy whipsawing (erratic actions)?
|   |
|   +-- Reduce learning_rate
|   +-- Increase n_steps (more data per gradient)
|   +-- Check for gradient explosion (loss spikes)
|
+-- Agent not learning (flat performance)?
|   |
|   +-- Check if vf_coef is drowning policy gradient
|   +-- Increase learning_rate cautiously
|   +-- Check if ent_coef is too high (drowning reward signal)
|
+-- Good training, bad OOS?
    |
    +-- Classic overfitting. Reduce learning_rate + gae_lambda
    +-- Consider shorter gamma (reduce horizon)
    +-- Increase ent_coef slightly for regularization
```

## Appendix C: Full Citation List

1. Mnih et al. 2016 --- "Asynchronous Methods for Deep Reinforcement Learning" (ICML 2016).
   Original A3C/A2C paper. Established n_steps=5, RMSprop, entropy regularization defaults.

2. Schulman et al. 2016 --- "High-Dimensional Continuous Control Using Generalized Advantage
   Estimation" (ICLR 2016). Introduced GAE with lambda for bias-variance tradeoff.

3. Schulman et al. 2017 --- "Proximal Policy Optimization Algorithms" (arXiv:1707.06347).
   PPO clipping mechanism that A2C lacks.

4. Huang et al. 2022 --- "A2C is a special case of PPO" (arXiv:2205.09123). Proved formal
   equivalence when PPO's clipping and multi-epoch are disabled.

5. Engstrom et al. 2020 --- "Implementation Matters in Deep Policy Gradients: A Case Study
   on PPO and TRPO" (ICLR 2020). Code-level optimizations matter more than algorithm choice.

6. Andrychowicz et al. 2021 --- "What Matters for On-Policy Deep Actor-Critic Methods? A
   Large-Scale Study" (ICLR 2021). 250K agents, 50+ design choices. Separate networks often
   outperform shared.

7. de la Fuente et al. 2024 --- "A Comparative Study of Deep RL Models: DQN vs PPO vs A2C"
   (KDD 2024). A2C most sensitive to hyperparameters.

8. Liu et al. 2021 --- "FinRL: Deep RL Framework to Automate Trading" (arXiv). A2C lowest
   volatility (10.4%) and MDD (-10.2%) in ensemble.

9. Yang et al. 2020 --- "Deep RL for Automated Stock Trading: An Ensemble Strategy" (ICAIF).
   Dynamic algo selection based on market turbulence.

10. Amit et al. 2020 --- "Discount Factor as a Regularizer in RL" (arXiv:2007.02040).
    Lower gamma as implicit regularization.

11. Blom et al. 2024 --- "Combining Automated Optimisation of Hyperparameters and Reward
    Shape" (arXiv). HP-reward mutual dependency.

12. Ng et al. 1999 --- "Policy Invariance Under Reward Transformations" (ICML 1999).
    Potential-based reward shaping theory.

13. Williams & Peng 1991 --- "Function Optimization Using Connectionist RL Algorithms".
    Original entropy regularization in policy gradient.

14. Stooke & Abbeel 2018 --- "Accelerated Methods for Deep RL" (arXiv:1803.02811).
    Adam optimizer advantages for large-batch RL.

15. Raffin et al. 2021 --- "Stable-Baselines3: Reliable RL Implementations" (JMLR).
    SB3 framework paper and implementation details.

16. Nature 2026 --- "Behaviorally Informed Deep RL for Portfolio Optimization with Loss
    Aversion and Overconfidence" (Scientific Reports). A2C stability in non-stationary markets.
