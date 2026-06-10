# SAC Tuning for Cryptocurrency Trading — Research Reference

> Researched 2026-04-03. Context: SAC shows zero improvement across 5 training iterations
> in SwingRL (equity Sharpe 1.293-1.398, crypto Sharpe 0.945-1.073). This research
> investigates why and what controls would actually move the needle.

## 1. Known Challenges of SAC in Crypto/Non-Stationary Environments

### Entropy-Reward Scale Mismatch (HIGH IMPACT)

SAC's default `ent_coef="auto"` starts alpha at **1.0**. With rolling Sharpe rewards
typically in [-2, +2], the entropy bonus completely dominates the reward signal early in
training. The agent behaves randomly for far longer than necessary.

- arXiv 2511.20678 (SAC crypto portfolio): found "computational overhead from
  entropy-regularized updates" was a key challenge
- DSAC portfolio implementation: uses `target_entropy = -0.3 * num_assets` with alpha
  decay of 0.99 every 100 iterations

**This is likely the #1 reason SAC underperforms in SwingRL.** Fixed `ent_coef=0.01`
helps but `"auto_0.1"` (start alpha at 0.1) is better — retains adaptive tuning without
drowning the reward signal.

### Regime Staleness in Replay Buffer

Off-policy learning means the buffer accumulates transitions from multiple market regimes.
The Q-function learns state-action values that reflect a weighted average of all regimes —
potentially matching none of them.

- Fedus et al. (ICML 2020, "Revisiting Fundamentals of Experience Replay"): confirmed
  that buffer age (proxy for off-policyness) directly hurts performance
- PPO-to-SAC migration blog: found SAC's buffer is both greatest advantage (rare
  conditions revisited many times) and greatest weakness (regime averaging)
- DEER paper (arXiv 2509.15032): proposes detecting regime changes and deprioritizing
  pre-change transitions

### Primacy Bias

Nikishin et al. (ICML 2022, "The Primacy Bias in Deep RL"): SAC overfits to early
experiences, creating bias that damages learning for the rest of training. Their fix:
periodic re-initialization of final network layers while preserving the buffer.

Directly relevant to SwingRL where early crypto training data may represent a single
market regime.

## 2. Buffer Size — Smaller Is Better for Non-Stationary Data

**Consensus across multiple sources: reduce buffer, don't increase it.**

| Source | Buffer Size | Timesteps | Notes |
|--------|------------|-----------|-------|
| arXiv 2511.20678 (SAC crypto portfolio) | 100K | 10M | LSTM-enhanced |
| CryptoBot-FinRL | 1M | configurable | `ent_coef="auto_0.1"` |
| DSAC portfolio optimization | 500K | not stated | Distributional, 30 quantiles |
| PPO-to-SAC migration blog | 100K | not stated | Found smaller better for non-stationary |
| SB3 default | 1M | n/a | General purpose, not finance-specific |
| Fedus et al. (ICML 2020) | 50K best | various | **"smaller buffer sizes of 50K perform best in learning speed and max return"** for SAC |

### SwingRL-specific analysis

- **Current**: 500K buffer, 500K crypto timesteps → buffer never evicts. Every transition
  stays forever. Regime-stale early data sampled equally with recent data.
- **At 1M buffer**: With 500K timesteps, buffer still never fills. Extra memory wasted.
  No functional change.
- **At 200K buffer**: Forces eviction after 40% of training. Keeps buffer regime-relevant.
  Matches literature recommendations.

### Memory implications (SB3 pre-allocates full buffer × n_envs)

With `n_envs=6` and crypto obs_dim=47:

| Buffer Size | Memory (crypto) | Memory (equity, obs_dim=164) |
|------------|-----------------|------------------------------|
| 200K | ~0.48 GB | ~1.63 GB |
| 500K (current) | ~1.19 GB | ~4.07 GB |
| 1M | ~2.38 GB | ~8.14 GB |

## 3. High-Impact Parameters the HP Tuner Should Control

### ent_coef / initial alpha (HIGHEST IMPACT — currently not tuned effectively)

- Default `"auto"` starts alpha=1.0, target_entropy = -dim(action_space)
- For crypto: action_dim=3, so target_entropy=-3.0 (too high for portfolio allocation)
- **Recommended**: `"auto_0.1"` to start alpha at 0.1, or set target_entropy=-0.9
  (= -0.3 × 3 assets, per DSAC paper)
- CryptoBot-FinRL uses `"auto_0.1"` specifically

### gradient_steps (MAJOR OVERLOOKED PARAMETER)

- Current: 1 (SB3 default) — exactly 1 gradient update per env step
- UTD ratio of 1:1 means the model under-extracts from available data
- **Recommended**: 4 for crypto. Each transition sampled ~4× more. Equivalent to
  quadrupling training without more data.
- Combine with halved lr (e.g., 1.5e-4) to compensate for increased update frequency
- arXiv 2403.05996: increasing gradient_steps improves sample efficiency but can cause
  Q-value divergence if lr isn't reduced accordingly

### target_entropy (NOT TUNED — should be)

- Default: -dim(action_space) = -3 for crypto, -9 for equity
- Too high for portfolio allocation where the agent should be decisive
- **Recommended range**: -0.5 to -dim(action_space)
- -1.0 for crypto would allow faster convergence to decisive allocations

### buffer_size (NOT TUNED — should be, but REDUCE not increase)

- **Recommended range**: [100K, 500K] for crypto, [200K, 500K] for equity
- Smaller buffers force regime-stale transitions to be evicted
- Literature unanimously supports smaller buffers for SAC in non-stationary envs

### Network architecture — critic capacity (NOT TUNED)

- Current: shared `net_arch=[64, 64]` for actor and both critics
- SAC's twin critics need more capacity than the actor for Q-value approximation
- **Recommended**: `policy_kwargs={"net_arch": dict(pi=[64, 64], qf=[128, 128])}`
- The critics approximate Q(s,a) over the full observation space; undersized critics
  produce value estimation errors that cascade into poor policy updates

### tau (soft update coefficient) — LOW PRIORITY

- Current: 0.005 (standard)
- Literature range for financial RL: 0.002-0.01
- DSAC paper used 0.05 (aggressive). Not recommended for SwingRL.
- **Leave at 0.005 unless Q-value instability observed.**

### learning_starts — MINOR

- Current: 10,000. With 500K timesteps, that's 2% warmup.
- Could increase to 20K-25K (5%) to build more diverse initial buffer
- Helps combat primacy bias

### train_freq — MINOR

- Current: 1 (SB3 default, collect 1 step then update)
- Could increase to 2-4 to collect small batches before updating
- Interacts with gradient_steps — if gradient_steps is increased, train_freq=1 is fine

## 4. Recommended Bounds for Memory Agent HP Tuning

```python
# Extend existing bounds.py with SAC-specific parameters
SAC_TUNABLE_BOUNDS = {
    "learning_rate": (1e-5, 3e-4),           # narrower than general [1e-5, 1e-3]
    "batch_size": (64, 512),                  # current range is fine
    "gamma": (0.95, 0.99),                    # narrow from [0.95, 0.995]; 0.99→100-step horizon too long for 4H
    "ent_coef_init": (0.01, 0.5),             # for "auto_X" format
    "gradient_steps": (1, 8),                 # NEW — highest impact new knob
    "target_entropy_scale": (0.2, 1.0),       # multiplier on -dim(action_space)
    "buffer_size": (100_000, 500_000),        # NEW — smaller is better for non-stationary
    "tau": (0.002, 0.01),                     # low priority
}
```

## 5. Is SAC Fundamentally Unsuited for This Use Case?

**No — but the current configuration is mechanistically handicapped.**

### SAC + Softmax portfolio weights: good fit

- arXiv 2511.20678 explicitly validated SAC for crypto portfolio weight allocation with
  softmax-constrained weights — "superior risk-adjusted performance vs DDPG"
- SAC's stochastic Gaussian policy naturally expresses allocation uncertainty through
  variance — mechanistically aligned with portfolio weights
- PPO-to-SAC migration blog: SAC produced "smoother position sizing with fewer erratic
  volatile-period adjustments"

### Double nonlinearity concern (minor)

SwingRL applies softmax on top of SB3's tanh squashing:
1. SAC's Gaussian → tanh → [-1, 1]
2. process_actions: softmax(raw_actions) → portfolio weights

The compressed mapping means large raw actions (near ±1, in tanh saturation) produce
nearly identical weights regardless of exact value. Gradient signal is diluted in these
regions. Not fatal, but worth noting.

### Entropy maximization vs capital preservation: real tension

SAC says "try everything, don't commit." The drawdown penalty says "be conservative."
These conflict. Resolution: make entropy bonus small relative to reward scale.
With `ent_coef="auto"` at alpha=1.0 and rewards in [-2, +2], entropy dominates.
With `"auto_0.1"`, it doesn't.

### Data requirements

SAC is theoretically more sample-efficient than PPO/A2C (off-policy). But in practice,
the quality of transitions matters more than quantity. 500K crypto timesteps is sufficient
if other parameters are correct (especially ent_coef and gradient_steps).

## 6. Top 5 Actionable Changes (Ranked by Expected Impact)

1. **Switch `ent_coef` from fixed 0.01 to `"auto_0.1"`** — Starts alpha 10× lower than
   default while retaining adaptive tuning. The #1 reason SAC underperforms.

2. **Reduce `buffer_size` from 500K to 200K for crypto** — Forces regime-stale eviction.
   Literature unanimously supports smaller buffers for non-stationary environments.
   Saves ~0.7 GB memory.

3. **Increase `gradient_steps` from 1 to 4**, reduce lr to compensate — Extracts 4× more
   learning per transition. Equivalent to quadrupling training without more data.

4. **Use separate critic architecture**: `net_arch=dict(pi=[64,64], qf=[128,128])` —
   Twin critics need more capacity for accurate Q-value estimation.

5. **Set `target_entropy = -1.0` for crypto** (instead of default -3.0) — Allows agent
   to converge on decisive allocations faster.

## Sources

- [Cryptocurrency Portfolio Management with RL: SAC and DDPG (arXiv 2511.20678)](https://arxiv.org/html/2511.20678v1)
- [When PPO Stops Working: Migrating to SAC for Non-Stationary Time-Series RL](https://skyliquid.medium.com/when-ppo-stops-working-migrating-to-sac-for-non-stationary-time-series-rl-3ac1be189e9c)
- [Distributional SAC Portfolio Optimization](https://medium.com/@abatrek059/a-distributional-soft-actor-critic-portfolio-optimization-a-pursuit-of-stability-a4057826a0b1)
- [SAC for Forex Trading Implementation](https://medium.com/@abatrek059/soft-actor-critic-sac-for-forex-trading-an-example-implementation-11c679b80f32)
- [Dissecting Deep RL with High Update Ratios (arXiv 2403.05996)](https://arxiv.org/html/2403.05996v1)
- [Sample Efficient Experience Replay in Non-Stationary Environments (DEER, arXiv 2509.15032)](https://arxiv.org/html/2509.15032v1)
- [Revisiting Fundamentals of Experience Replay (Fedus et al., ICML 2020)](https://arxiv.org/abs/2007.06700)
- [The Primacy Bias in Deep RL (Nikishin et al., ICML 2022)](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)
- [Risk Sensitive Distributional SAC for Portfolio Management (IEEE 2024)](https://ieeexplore.ieee.org/document/10607182/)
- [Hidden-layer Configurations in RL for Stock Portfolio Optimization (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S2667305324001418)
- [SAC Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [CryptoBot-FinRL Implementation](https://github.com/inoue0406/CryptoBot-FinRL)
- [The Importance of Experience Replay Buffer Size (ResearchGate 2024)](https://www.researchgate.net/publication/379190323)
- [Soft Actor-Critic: Off-Policy Maximum Entropy (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [SAC Algorithms and Applications (Haarnoja et al., 2019)](https://arxiv.org/abs/1812.05905)
- [FinRL Contest 2023/2024 Ensemble Methods (arXiv 2501.10709)](https://arxiv.org/html/2501.10709v1)
- [DSAC: Distributional SAC for Risk-Sensitive RL (arXiv 2004.14547)](https://arxiv.org/html/2004.14547v3)
