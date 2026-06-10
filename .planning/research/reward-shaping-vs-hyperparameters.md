# Reward Shaping vs Hyperparameter Impacts in RL — Research Reference

Research on how reward weight adjustments and hyperparameter changes independently and jointly
affect RL agent performance. Used to inform LLM consolidation prompts for distinguishing
HP-caused vs reward-caused regressions in SwingRL.

---

## 1. Reward Shaping Theory (Ng et al. 1999)

**Key finding**: Potential-based reward shaping (PBRS) is the *necessary and sufficient* condition
for policy invariance. The shaped reward must take the form `F(s, a, s') = gamma * Phi(s') - Phi(s)`
for some potential function Phi. Any other transformation may change the optimal policy.

**Critical implication for SwingRL**: The reward wrapper computes:
```
weighted_reward = w_profit * profit + w_sharpe * sharpe + w_drawdown * (-drawdown_frac) + w_turnover * (-turnover_ratio)
```
This is **NOT potential-based reward shaping** — it is multi-objective reward scalarization.
Changing the weights **changes the objective function itself**, which **changes the optimal policy**.
The PBRS guarantee does not apply.

The LLM memory agent is not "shaping" rewards in the Ng et al. sense — it is **redefining what
"good" means**. This distinction matters for attribution: a weight change that improves Sharpe but
worsens MDD is not a bug — the agent correctly optimized a different objective.

**Citations**:
- Ng, Harada, Russell 1999 — "Policy Invariance Under Reward Transformations" (ICML 1999)
- Wiewiora 2003 — "Potential-based Shaping and Q-Value Initialization"

---

## 2. Multi-Objective Reward Composition

**Key finding**: Linear weighted sum scalarization has a fundamental limitation: it **cannot find
Pareto-optimal policies on nonconvex regions of the Pareto front**. The relationship between
weights and outcomes is generally **nonlinear** due to policy-weight interactions.

Small weight changes can cause large behavioral shifts near Pareto front "knees," while large
changes may have minimal effect in flat regions.

**Implication**: The four reward components (profit, sharpe, drawdown, turnover) likely have a
nonconvex Pareto front — high Sharpe and low drawdown may conflict in certain market regimes.
Linear scalarization means some desirable policy configurations may be structurally unreachable
regardless of weight settings.

**Citations**:
- Hayes et al. 2022 — "A Practical Guide to MORL and Planning" (JAMAS)
- Van Moffaert & Nowe 2014 — "MORL using Sets of Pareto Dominating Policies" (JMLR)

---

## 3. Reward Shaping in Financial RL

**Key findings**:

a) **Sharpe-based reward shaping** (Betancourt & Chen 2023): Combining PnL and Sharpe ratio in the
reward "balances returns against volatility," with agents achieving 5.03% MDD compared to 45.57%
for competitors. However, **no systematic ablation** was performed to isolate individual component
contributions.

b) **Risk-aware composite rewards** (Duarte et al. 2025): Propose
`R = w1*R_ann - w2*sigma_down + w3*D_ret + w4*T_ry`. Weights are tuned via grid search over the
probability simplex. They found the composite reward "guides the agent toward smoother equity
growth" but acknowledge weight tuning is a hyperparameter task without providing sensitivity
analysis.

c) **FinRL ensemble results**: A2C achieves lowest annual volatility (10.4%) and MDD (-10.2%),
PPO achieves highest returns (15.0% annual), while SAC/DDPG show higher drawdowns.

d) **A critical finding**: Adding MDD directly as a per-step penalty is problematic because
"risk indicators are calculated based on period-level data and cannot be assigned to each time
step, making it difficult for the agent to optimize." Per-step drawdown fraction is a reasonable
approximation but introduces noise.

**Citations**:
- Duarte et al. 2025 — "Risk-Aware RL Reward for Financial Trading" (arXiv)
- Betancourt & Chen 2023 — "Sharpe Ratio Based Reward in DRL for Trading" (Springer)
- Liu et al. 2021 — "FinRL: Deep RL Framework to Automate Trading" (arXiv)

---

## 4. Reward Hacking / Misspecification Failure Modes

**Key findings**:

a) **Goodhart's Law in RL** (Skalse et al. 2023, ICLR 2024): Optimizing a proxy reward follows a
characteristic curve — initial improvement, peak, then decline on the true objective. The
divergence begins when the optimization trajectory hits the occupancy measure polytope boundary.
Occurs in ~19.3% of tested scenarios. **Mitigation is early stopping**, but sacrifices 10-44% of
true reward.

b) **Specific failure modes relevant to SwingRL**:

| Failure Mode | Mechanism | How It Manifests |
|--------------|-----------|------------------|
| **Excessive drawdown penalty → increased actual drawdown** | Agent avoids *small* drawdowns by concentrating into fewer, larger positions. Reduces penalty frequency but increases tail risk. | MDD increases despite drawdown weight increase |
| **Inactivity bias** | When drawdown+turnover penalties dominate profit reward, agent rationally stops trading. | Trade count drops to near-zero |
| **Turnover penalty → holding losers** | Closing losing positions incurs "turnover," so agent holds them. | Increased drawdown duration, not magnitude |
| **Misweighting** (Pan et al. 2022) | Wrong relative importance between objectives | Sharpe improves but total return collapses, or vice versa |

c) **Combining multiple rewards** can reduce individual hacking risk, but the combination itself
must be carefully calibrated.

**Citations**:
- Skalse et al. 2023 — "Goodhart's Law in Reinforcement Learning" (ICLR 2024)
- Lilian Weng — "Reward Hacking in RL" (survey, 2024)
- Denny Britz — "Learning to Trade with RL" (blog)

---

## 5. Hyperparameter vs Reward Interaction

**Key finding** (Blom et al. 2024): Demonstrated empirically that "an RL algorithm's hyperparameter
configurations and reward function are often mutually dependent."

Their landscape analysis showed:
- Some reward components had **strong dependency** on HPs (successful ranges shifted with HP values)
- Others showed **weak or no dependency**
- The interaction creates **non-convex regions** where optimal configurations shift

**The circular dependency**: "Effective hyperparameter tuning requires an effective reward signal,
and effective reward shaping depends on good hyperparameter configurations."

**Their recommendation**: Combined optimization should be **best practice**. When done jointly, it
"matches the performance of individual optimization with the same compute budget."

**Critical implication for attribution**: When the LLM changes both HPs and reward weights in the
same iteration, **attribution is impossible** without controlled experiments. The recommendation is
to either:
1. Change HPs and reward weights in **alternating iterations** (for attribution)
2. Change both simultaneously (for performance, sacrificing attribution)

**Citations**:
- Blom et al. 2024 — "Combining Automated Optimisation of Hyperparameters and Reward Shape" (arXiv)
- Nature 2025 — "Reward Design and HP Tuning for Generalizable DRL Agents"

---

## 6. Reward Weight Sensitivity by Algorithm

### PPO
Most sensitive to reward scale changes because:
- Value function clipping (`clip_range_vf`) depends on reward scaling
- Advantage estimation via GAE amplifies reward magnitude issues
- Clip range creates a "trust region" that interacts with reward magnitude — if rewards suddenly
  change scale, the clip region may be too tight or too loose

### A2C
Moderate sensitivity:
- No clipping mechanism, so reward scale changes propagate directly to gradients
- GAE lambda means advantage estimates span multiple steps of reward history
- The connection between GAE and reward shaping is direct: maximizing the (gamma*lambda)-discounted
  sum of transformed rewards with Phi=V gives precisely GAE
- Compensates through gradient noise from single-step updates (no replay buffer)

### SAC
Unique sensitivity due to entropy-reward coupling:
- SAC's temperature parameter alpha acts as the ratio between reward scale and entropy bonus
- "SAC is sensitive to reward scaling since it is related to the temperature of the optimal policy"
- Automatic entropy tuning partially compensates for reward scale changes, but not instantaneously
- Reward weight changes effectively change the entropy-reward balance

**Threshold effects**: The literature does not identify clean thresholds. Sensitivity is continuous
but nonlinear — small changes in flat Pareto regions have minimal impact, while small changes near
"knees" can cause large behavioral shifts.

**Citations**:
- SB3 Tips & Tricks documentation
- Haarnoja et al. 2018 — "SAC: Off-Policy MaxEnt DRL" (arXiv)
- Engstrom et al. 2020 — "Implementation Matters in Deep Policy Gradients"

---

## 7. Reward Shaping and Entropy/Exploration Interaction

**Key finding**: Generalized Maximum Entropy RL (Zhang et al. 2023, IEEE) showed that incorporating
policy entropy into the reward function via potential-based shaping creates a unified framework.

**Can high entropy negate reward shaping signals?** Yes. If ent_coef is large relative to reward
component magnitudes, the entropy bonus dominates the objective, and the agent prioritizes action
diversity over reward maximization. Particularly relevant when:
- Profit component has small magnitude (daily returns ~0.001-0.01)
- Sharpe component after normalization is in similar range
- Entropy bonus can be much larger if alpha is not properly scaled

**Practical implication**: When the LLM increases profit or sharpe weight, but ent_coef is high,
the signal may be washed out. The LLM should consider ent_coef magnitude when making reward weight
recommendations.

**Citations**:
- Zhang et al. 2023 — "Generalized Maximum Entropy RL via Reward Shaping" (IEEE)

---

## 8. Online Reward Adaptation (Dynamic Weight Changes During Training)

**Key findings**:

a) **Reward Training Wheels (RTW)** (2025): A teacher agent dynamically adjusts auxiliary reward
weights via a separate RL policy. Key finding: effective adaptation follows a "training wheels"
pattern — stability penalty weight follows an inverted U-shape (emphasize stability early, relax
later), while performance weights increase over time. RTW achieved **perfect 5/5 success rate**
vs expert-designed static 2/5.

b) **DynaOpt** (2024): Multi-armed bandits (Exp3) dynamically select reward emphasis. Weight
updates are multiplicative: `w_{t+1} = w_t * exp(gamma * r_hat / K)`. Achieved 7.80% improvement
over static baselines. **Random weight changes performed worse than static** (1/5 vs 2/5 success),
confirming *principled* adaptation is essential.

c) **Stability considerations**: Changing reward weights mid-training causes:
- Value function estimate becomes stale (trained on old rewards)
- Advantage estimates become noisy/biased (PPO especially)
- VecNormalize running statistics become mismatched
- Clip range may not accommodate scale change

d) **Rate of change matters**: Gradual, smooth changes beneficial. Abrupt/random changes harmful.
Exponential multiplicative updates inherently limit change rate.

**Practical implication for SwingRL**: The epoch advice approach (changing weights every N epochs)
is sound in principle, but needs constraints on:
- Maximum per-adjustment weight change magnitude
- Minimum epochs between changes (for value function adaptation)
- Existing bounds clamping limits range but not rate of change

**Citations**:
- RTW 2025 — "Reward Training Wheels: Adaptive Auxiliary Rewards" (arXiv)
- DynaOpt 2024 — "Dynamic Reward Adjustment in Multi-Reward RL" (arXiv)
- Devlin & Kudenko 2012 — "Dynamic Potential-Based Reward Shaping" (AAMAS)

---

## 9. The Attribution Problem

**Key findings**:

a) **Controlled ablation methodology** requires: (1) establish baseline, (2) change ONE component
holding all others constant, (3) retune remaining HPs after ablation to avoid confounding,
(4) compare under identical conditions.

b) **The fundamental confound**: When iteration N changes both learning_rate (3e-4 → 1e-4) and
drawdown weight (0.15 → 0.30), attribution is **mathematically impossible** without factorial
experiments.

c) **Practical attribution heuristics** (when full factorial is infeasible):

| Heuristic | Signal Strength | How It Works |
|-----------|----------------|--------------|
| Zero-adjustment folds | **Strongest** | Folds with no reward adjustments isolate HP effect |
| Directionality check | Strong | Did the outcome match expected direction? (higher drawdown weight → lower MDD?) |
| Magnitude comparison | Moderate | If HP changed 3x and reward weight changed 0.2x, HP is likely dominant |
| Timing analysis | Moderate | HP changes affect from epoch 0; reward adjustments have delayed effect |
| Cross-fold weight uniformity | Strong | If all folds end with ~same weights but different outcomes, HPs dominate |
| Adjustment-outcome correlation | Moderate | If more adjustment correlates with better outcomes, adjustments are compensating |

d) **Recommendation**: Rather than post-hoc attribution, prefer **alternating HP-only and
reward-only changes** when possible. For joint changes, use the heuristic table above.

**Citations**:
- Blom et al. 2024 — Combined Optimization (arXiv)
- RUDDER — RL with Delayed Rewards (reward attribution)

---

## 10. SwingRL-Specific Production Evidence (Iter 0→1)

Cross-referencing the literature with actual production data from iter 1:

### Evidence that HPs dominated the regression:

1. **PPO had 4 folds with zero adjustments that regressed** — pure HP effect (Section 9 heuristic)
2. **SAC trades collapsed 612→45 despite turnover weight barely changing** (0.100→0.077) —
   the lr 0.0001→0.0003 HP change (exceeding documented range of 1e-5 to 1e-4) caused Q-value
   instability, consistent with Section 4 reward hacking via HP misconfiguration
3. **A2C reward weights converged to same values across all 14 folds** (~profit=0.386, drawdown=0.331)
   but outcomes ranged from -3.75 to +1.11 — Section 9 "cross-fold weight uniformity" heuristic
   confirms HPs are the dominant factor
4. **A2C ent_coef=0.025 exceeded documented range** (0.005-0.02) — per Section 7, high entropy
   can wash out reward shaping signals

### Evidence that reward adjustments had secondary effects:

1. **Correlation(adjustment_magnitude, sharpe_delta) = +0.35 to +0.38** across all algos —
   MORE adjustment correlated with BETTER outcomes, consistent with Section 8 finding that
   principled adaptation helps
2. **A2C drawdown weight 0.15→0.33 but MDD worse in 9/14 folds** — classic Section 4 failure
   mode: excessive drawdown penalty under high ent_coef (Section 7 interaction) may have caused
   inactivity or position concentration
3. **PPO folds WITH adjustments performed worse than folds WITHOUT** (-1.14 vs -0.54) — but
   confounded by fold-market conditions, not necessarily causal (Section 9 attribution limitations)

### What the hp-tuning-reference.md predicted correctly:

| HP Change | Reference Prediction | Actual Outcome | Correct? |
|-----------|---------------------|----------------|----------|
| SAC lr 0.0001→0.0003 | "Exceeds range, causes Q-value memorization" | Trades collapsed | **YES** |
| A2C ent_coef 0.0→0.025 | "Exceeds 0.005-0.02 range" | Regression | **YES** |
| PPO batch_size 64→256 | "Can more reliably overfit" | Regression | **YES** |
| PPO n_epochs 10→5 | "Should reduce overfitting" | Still regressed | **PARTIALLY** (offset by batch_size) |
| A2C gamma 0.99→0.96 | "Lower gamma = implicit regularization" | Still regressed | **NO** (myopic for 4H crypto) |
