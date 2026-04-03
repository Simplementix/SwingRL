# SAC General Tuning — Literature Reference

> Researched 2026-04-03. General SAC hyperparameter tuning best practices from academic
> literature, not specific to finance/crypto. For crypto-specific findings, see
> `sac-crypto-tuning.md`.

---

## 1. Buffer Size Impact on Learning

### Fedus et al. (ICML 2020) — "Revisiting Fundamentals of Experience Replay"

- **Buffer capacity effects are algorithm-dependent.** Increasing DQN's buffer from 1M to
  10M yielded no improvement. Rainbow with n-step returns improved 25-40% median
  performance with the same increase.
- **For SAC specifically**: SAC with buffer sizes 50K to 1M performed best with **larger
  batches of 512 and smaller buffers of 50K** in terms of learning speed and max return.
  Directly contradicts "bigger is better" intuition.
- **Oldest policy age**: Defined as gradient updates since a transition was generated.
  Fixing capacity, decreasing oldest policy age improves performance. Buffer age (proxy
  for off-policyness) directly hurts learning quality.
- **N-step returns uniquely benefit from larger buffers.** Without n-step returns, larger
  capacity provides negligible benefit.
- **No universal optimal buffer-to-timesteps ratio** was established.

### Zhang & Sutton (2017) — "A Deeper Look at Experience Replay"

- **Large replay buffers can significantly hurt performance.** Performance is sensitive to
  buffer size with huge drops for buffers both too small AND too large.
- **Buffer sizes tested**: 100, 1K, 10K, 1M. Optimal occurred at 1K-10K for simple tasks.
- **Mechanism**: When buffer is large, rare important transitions get sampled much later.
  Delayed learning of rare transitions has a compounding negative effect.
- **Combined Experience Replay (CER)**: Always include the most recent transition in each
  minibatch. CER made performance relatively insensitive to buffer size.
- **The buffer size hyperparameter's importance has been systematically underestimated.**

### Practical Synthesis

For SAC, evidence favors **smaller buffers (50K-200K)** that force FIFO eviction and keep
data fresh. If buffer_size >= total_timesteps, old data never evicts — worst case for
non-stationary problems.

---

## 2. ent_coef Tuning

### "auto" (SAC-v2, Haarnoja et al. 2019)

Learns log_alpha via gradient descent. Target entropy = -dim(action_space). Alpha starts
at 1.0 by default.

**Failure modes:**
- **Reward scale sensitivity (fundamental)**: Haarnoja et al. explicitly acknowledged that
  "unlike in conventional RL where the optimal policy is independent of reward scaling,
  in maximum entropy RL the scaling factor must be compensated by suitable temperature."
  ent_coef is mathematically equivalent to the inverse of reward scale.
- **Slow convergence on small-reward environments**: Starting alpha=1.0 with rewards in
  [-2, +2] means entropy bonus dominates. Agent behaves randomly until alpha decreases.
- **Oscillation around target entropy**: Constrained optimization can oscillate, especially
  during early training or after reward weight changes.
- **Target entropy mismatch**: Default = -dim(action_space) assumes all action dimensions
  should have ~1 nat of entropy. Too high for constrained action spaces (e.g., softmax
  portfolio weights where actions are correlated).

### Meta-SAC (Wang & Ni, ICML AutoML Workshop 2020)

Identified SAC-v2's constrained optimization as limited. Key finding: alpha naturally
converges to near-zero in later training, making SAC approximately equivalent to DDPG.
Ideal trajectory: high early (exploration) → near-zero late (exploitation). Standard
"auto" does not guarantee this.

### Recommended Approaches

| Approach | When to use |
|----------|------------|
| `"auto_0.1"` | Small reward scale (<10). Retains adaptation, avoids swamping. |
| Fixed `0.01` | Need tight control, reward scale well-understood. |
| `"auto"` (default) | Large reward scale (>100), or well-tuned reward normalization. |

**Interaction with reward scale**: ent_coef and reward scale are inversely related.
Doubling rewards ≈ halving ent_coef. When reward weights change mid-training (as in
epoch advice reward adjustments), the effective entropy-reward balance shifts.

---

## 3. gradient_steps / UTD Ratio

### Hussing & Voelcker et al. (RLC 2024) — "Dissecting Deep RL with High Update Ratios"

- **Q-value divergence mechanism**: At high UTD, Q-values diverge exponentially. Root
  cause: OOD action predictions trigger large gradients, compounded by Adam's second-order
  momentum. Using SGD with only first-order momentum nearly eliminated divergence.
- **Priming experiments**: With 1,000 samples, agents updated for 25K-100K steps showed
  progressively worse final returns. Higher priming = longer recovery periods.
- **Output Feature Normalization (OFN)**: Unit-ball normalization of Q-network outputs
  completely mitigated divergence. OFN + UTD=8 matched standard resetting at UTD=32.
- **Standard mitigations insufficient**: L2 regularization and dropout provided only
  partial relief. Weight decay performed poorly.

### REDQ (Chen et al., ICLR 2021)

First successful model-free algorithm for continuous control with UTD >> 1. Three
ingredients: UTD >> 1, ensemble of 10 Q functions, in-target minimization across random
subset. REDQ with UTD=20 significantly outperformed naive SAC at UTD=20.

### SR-SAC (D'Oro et al., ICLR 2023)

SR-SAC with replay ratio 128 compared favorably with REDQ at ratio 20 despite being
simpler. Used periodic resets every 40,000 updates.

### Plasticity Loss (Lyle et al., ICML 2023)

Networks lose plasticity (ability to change predictions) over training. Connected to loss
landscape curvature changes, not just saturated units. High UTD accelerates plasticity
loss.

### Practical Synthesis

Increasing gradient_steps above 1 extracts more learning but introduces three risks:
1. Q-value divergence from OOD actions + optimizer momentum
2. Plasticity loss from excessive updates on limited data
3. Primacy bias amplification

**Mitigations**: Layer normalization (proven), periodic network resets (proven), reduced
learning rate (common practice), output feature normalization (cutting-edge).

**When increasing gradient_steps by factor k, reduce tau by ~1/k** to maintain same
effective target update rate per environment step.

---

## 4. Network Architecture for SAC

### Separate Actor/Critic Architectures

SAC uses fully disentangled actor and critic networks. Each can be sized independently.

**When critic needs more capacity than actor:**
- Critic approximates Q(s,a) over full observation AND action space. Actor only maps
  observations to actions. Critic input dimensionality is strictly larger.
- BRO (Nauman et al., NeurIPS 2024): critic networks of ~5M parameters (7x standard SAC)
  showed dramatically better performance. Standard SAC critics (~0.7M params) are
  significantly undersized for complex tasks.
- **A 5M parameter model with UTD=1 outperformed a 1M model with UTD=15 while being 5x
  faster** — model capacity can substitute for high UTD ratios.

### Layer Normalization

- **CrossQ (Bhatt et al., 2024)**: Batch normalization in critics (without target networks)
  matched REDQ/DroQ sample efficiency at UTD=1 instead of UTD=20. Achieved 4x faster
  wallclock. Eliminated need for target networks.
- **BRO**: Layer normalization after each dense layer was "essential to unlocking scaling."
- **DroQ (Hiraoka et al., 2022)**: Layer normalization + dropout stabilized SAC at UTD=20
  with only 2 critics.

### Practical Recommendation

Use `policy_kwargs={"net_arch": dict(pi=[64,64], qf=[128,128])}` minimum. Adding layer
normalization to critics is the single highest-impact architectural change for training
stability, especially at gradient_steps > 1.

**CrossQ available in sb3-contrib** as drop-in SAC replacement — eliminates need for
high gradient_steps and tau tuning entirely.

---

## 5. tau (Soft Update Coefficient)

Controls Polyak averaging: `target = (1-tau)*target + tau*online`. Default 0.005.

**Key interaction with gradient_steps**: tau's effective update rate scales linearly.
At UTD=4 with tau=0.005, effective tau = ~0.020 per env step. This 4x increase can
destabilize training.

**Rule of thumb**: When increasing gradient_steps by factor k, reduce tau by ~1/k.
gradient_steps=4 → tau ≈ 0.001-0.002.

**Standard range**: 0.001-0.01. Only adjust when changing gradient_steps or observing
Q-value instability. CrossQ eliminates target networks entirely, making tau irrelevant.

---

## 6. learning_starts

Steps before any gradient updates begin. During warmup, actions are random.

**Key findings:**
- Too small: primacy bias — network overfits to tiny, non-diverse dataset. Nikishin et al.
  (ICML 2022) showed resets improved performance by >100% at high replay ratios,
  demonstrating how damaging early overfitting is.
- Too large: wastes environment steps on random exploration.
- **SB3 default is 100 — vastly lower than Spinning Up's 10,000.** This is the most
  common SB3 gotcha. Almost always needs increasing.

**Guidelines**: 1K-50K depending on episode length. Should be at least 1-2 full episodes
for diverse state coverage.

---

## 7. train_freq

Controls data collection frequency. Effective UTD = gradient_steps / train_freq.

**With gradient_steps > 1, keep train_freq=1.** UTD is better controlled via
gradient_steps alone. Only increase train_freq for wall-clock optimization when env.step()
is very fast relative to gradient computation.

With vectorized environments (n_envs > 1), train_freq=1 already collects n_envs
transitions per update. Further increases may not help.

---

## 8. Common SAC Failure Modes

### 8a. Q-Value Overestimation

Despite twin Q-networks (taking minimum), SAC can still overestimate. SAC-B (DAI 2025):
"Q-value overestimation bias in stochastic environments critically impairs policy
optimization." Twin Q mitigates but does not eliminate.

### 8b. Q-Value Divergence

Q-values grow unboundedly. Caused by OOD action predictions + Adam momentum (Hussing
& Voelcker 2024). Risk increases with: higher UTD, smaller buffers, larger lr.
Symptom: Q-values reach 10^4+ while actual returns are O(10^2).

### 8c. Entropy Collapse

Policy becomes near-deterministic, exploration ceases. Creates feedback loop: sharp Q →
low entropy → less exploration → sharper Q on limited data.

### 8d. Primacy Bias

Nikishin et al. (ICML 2022): SAC overfits to early experiences. Network parameters resist
updating when new data arrives. Especially damaging in non-stationary environments.
Fix: periodic resets of final network layers while preserving buffer.

### 8e. Pessimistic Underexploration

Twin Q's min operator can introduce underestimation bias, making policy overly
conservative. Agent avoids actions where Q-estimates disagree.

### 8f. Plasticity Loss

Lyle et al. (ICML 2023): Networks progressively lose ability to adapt. Connected to loss
landscape curvature changes. Accelerated by high UTD.

### 8g. Reward Scale Sensitivity

Unique to maximum entropy RL: ent_coef is equivalent to inverse of reward scale. Changing
reward magnitudes without adjusting ent_coef changes effective exploration-exploitation
tradeoff. Fundamental property, not a bug.

---

## 9. SAC vs PPO/A2C — When Each Wins

| Characteristic | Favors SAC | Favors PPO/A2C |
|----------------|-----------|----------------|
| Sample budget | Limited | Unlimited |
| Action space | Continuous | Discrete |
| Reward density | Dense | Sparse |
| Stationarity | Stationary | **Non-stationary** |
| Training stability | Lower priority | Higher priority |
| Action dimensionality | Low-moderate | Any |

**SAC wins**: High sample efficiency needed, continuous actions, dense rewards, stationary
environments. De facto standard for MuJoCo locomotion.

**PPO/A2C win**: Non-stationary environments (only use fresh data), sparse rewards,
discrete actions, stability priority, large-scale distributed training. PPO has "lowest
standard deviation across runs" in comparison studies.

---

## 10. SB3-Specific Gotchas

1. **Single shared learning rate**: SB3 uses one lr for actor, both critics, AND
   log_alpha. Research implementations use separate rates. SB3's lr is a compromise.

2. **Default learning_starts=100**: Vastly lower than Spinning Up's 10,000. Almost always
   needs increasing.

3. **Buffer pre-allocation**: `buffer_size * n_envs * obs_dim * sizeof(float32)`. With
   n_envs=6, obs_dim=164, buffer_size=1M: ~8.14 GB. Pure waste if total_timesteps <
   buffer_size. **Always set buffer_size <= total_timesteps.**

4. **ReLU activation**: SB3 SAC uses ReLU (matching original paper), unlike PPO/A2C which
   use tanh. ReLU networks more susceptible to dead neurons under high UTD.

5. **gradient_steps=-1 shortcut**: Makes SB3 perform exactly as many gradient steps as
   env steps collected since last train call.

6. **target_entropy parameter**: Exposed but defaults to "auto" (-dim(action_space)). To
   override: `SAC(..., target_entropy=-1.0)`. Not well-documented.

7. **CrossQ in sb3-contrib**: Drop-in SAC replacement with batch-normalized critics,
   no target networks. UTD=1 matches REDQ at UTD=20. Eliminates gradient_steps and tau
   tuning.

---

## Sources

- Fedus et al. 2020 — "Revisiting Fundamentals of Experience Replay" (ICML 2020)
- Zhang & Sutton 2017 — "A Deeper Look at Experience Replay"
- Haarnoja et al. 2018 — "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. 2019 — "Soft Actor-Critic Algorithms and Applications"
- Wang & Ni 2020 — "Meta-SAC: Auto-tune Temperature via Metagradient" (ICML AutoML)
- Nikishin et al. 2022 — "The Primacy Bias in Deep RL" (ICML 2022)
- Lyle et al. 2023 — "Understanding Plasticity in Neural Networks" (ICML 2023)
- Chen et al. 2021 — "REDQ: Randomized Ensembled Double Q-Learning" (ICLR 2021)
- D'Oro et al. 2023 — "Sample-Efficient RL by Breaking the Replay Ratio Barrier" (ICLR)
- Hussing & Voelcker et al. 2024 — "Dissecting Deep RL with High Update Ratios" (RLC)
- Bhatt et al. 2024 — "CrossQ: Batch Normalization in Deep RL" (sb3-contrib)
- Nauman et al. 2024 — "BRO: Bigger, Regularized, Optimistic" (NeurIPS 2024)
- Hiraoka et al. 2022 — "DroQ: Dropout Q-Functions" (ICLR 2022)
- SAC-B 2025 — "SAC with Bias for Suppressing Q-value Overestimation" (DAI 2025)
- SB3 SAC Documentation
- SB3 GitHub Issue #2013 — log_alpha optimization
- OpenAI Spinning Up — SAC
- CrossQ in sb3-contrib
