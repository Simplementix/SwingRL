# Turbulence Method Review — Decision Memo (2026-07-07)

> Produced during G2 (Plan A walkthrough) at the user's request: "research and review the
> best way to calculate turbulence and make sure it's ideal for what we are trying to
> measure based on the data we have." Feeds: Plan A Task 6 (halt trigger) and Plan B's
> era-1 observation-feature definition (A28). Method: literature research (cited) +
> read of `src/swingrl/features/turbulence.py`. Claims labeled VERIFIED-BY-SOURCE /
> VERIFIED-FROM-CODE / JUDGMENT.

## Summary recommendation matrix

| Cell | Verdict | Change | Confidence |
|---|---|---|---|
| Equity halt | Keep EWMA-Mahalanobis; fix hygiene + threshold | Hard halt 90th → **97th–99th pct**; optional graded de-risk 90th→97th; **Ledoit-Wolf shrinkage or eigenvalue floor** instead of bare `pinv`; fix EWMA warm-start bias (zero-init leaves ~25% weight at bar 252 → turbulence inflated ~15% post-warmup) via de-bias factor or ≥378-bar warmup | High (threshold), Med-High (shrinkage) |
| Crypto halt | **Replace composite** | Two verified defects: `abs(vol z)` → dead-calm scores as turbulence; `abs(corr)` → a +0.8→−0.8 decoupling flip registers zero spike. Replace with **2-asset EWMA-Mahalanobis** (K&L's own didactic case is n=2) OR-gated with a **signed realized-vol percentile**; warmup 360 → ≥1080 bars | High (defects), Medium (replacement) |
| Equity obs feature (era 1) | Keep family, **decompose** | Feed **magnitude surprise + correlation surprise** (Kinlaw-Turkington) as percentile-ranks, not the raw distance. NOT redundant with HMM p_crisis (fast single-bar outlier vs smoothed posterior) nor VIX (K&L Table 6: turbulence filter IR 0.92 vs VIX filter 0.50) | Medium |
| Crypto obs feature (era 1) | Replace with components | Signed vol z-score + signed Δcorrelation (+ optional RV−BV jump flag) — never the multiplicative composite | Medium |

## Key findings

1. **Method precedent (VERIFIED-BY-SOURCE):** Kritzman & Li 2010 ran turbulence on US
   sectors, daily, 1973–2009 — same asset count/correlation class as our 8 ETFs.
   Window choice is insensitive (6mo–multi-year all similar, practitioner replication).
   126-day half-life is defensible; no evidence-based reason to change.
2. **`pinv` weakness (JUDGMENT from code+math):** 8 ETFs share one dominant factor →
   tiny-but-real smallest eigenvalues; `pinv` inverts them happily, so noise along
   "spread between near-identical ETFs" directions can dominate the score. Fix =
   shrinkage (K&L themselves shrink) or eigenvalue floor (clip < ~1e-3·λmax). The
   `abs()` in the quadratic form masks the symptom; after shrinkage it can be an assert.
3. **Threshold (VERIFIED):** literature uses 70th–90th pct for *graded exposure
   scaling*, not hard halts; FinRL's current master code effectively halts at ~99th.
   Turbulence is highly persistent (clusters in real stress), which makes a 90th-pct
   halt less costly than iid-thinking suggests, but a hard halt at 90th still keeps a
   swing book flat ~1 day in 10 including every ordinary CPI-day spike. Percentile
   computed on trailing ≥3y (equity) / full history (crypto), recomputed — not frozen.
4. **Crypto composite (VERIFIED-FROM-CODE defects):** see matrix. Also baseline
   contamination: historical chunk-vol distribution overlaps the current window,
   deflating the z-score exactly during spikes. The "not enough history for covariance"
   rationale is weak — a 2×2 covariance is the easiest case.
5. **Redundancy (JUDGMENT + mechanism):** turbulence fires on the FIRST unusual bar;
   HMM p_crisis needs accumulating evidence (Stöckl-Hanke timing point). Complementary,
   not redundant — but run a cheap MI/correlation check between turbulence percentile
   and p_crisis on our history before era-1 feature freeze; if ρ>~0.9 the obs-feature
   case weakens (the halt case does not).

## Honest gaps (literature does not settle)

- No published optimal hard-halt percentile; 97th–99th is triangulated JUDGMENT —
  validate on our own backtests (era-1 gate-derivation replay is a natural harness).
- No peer-reviewed turbulence application to crypto or 4H frequency (math is
  frequency-agnostic; unvalidated extrapolation).
- No published ablation of turbulence as an RL state feature; era-1 vs era-2
  season-over-season comparison is the only real test.
- EWMA-weighted turbulence specifically has no dedicated study; half-life
  unconstrained by evidence beyond window-insensitivity.

## Concrete parameters adopted (pending user sign-off in Plan A walkthrough)

| Parameter | Current | Adopted |
|---|---|---|
| Equity halt percentile | 90th (never live — F1) | 97th hard halt (config-tunable) |
| Equity covariance | bare `pinv` | eigenvalue floor (one-line) or Ledoit-Wolf |
| Equity EWMA warmup | 252 + zero-init bias | de-bias factor 1/(1−(1−α)^t), warmup kept 252 |
| Crypto method | vol-z × corr composite | 2-asset EWMA-Mahalanobis + signed RV pct OR-gate |
| Crypto warmup | 360 bars | ≥1080 bars (180d) |
| Percentile lookback | undefined (was broken) | trailing 3y equity / full history crypto |
| Era-1 obs features | raw score (was 0.0 bug) | decomposed percentile-rank features (Plan B) |

## Sources

Kritzman & Li 2010 (FAJ 66(5)); Kinlaw & Turkington 2014 (J. Asset Mgmt 14(6));
Kritzman, Li, Page & Rigobon 2011 (JPM 37(4), absorption ratio); Stöckl & Hanke 2014
(Applied Econ & Finance 1(2)); Yang et al. 2020 (ICAIF, FinRL ensemble);
FinRL master `models.py` (threshold logic); Portfolio Optimizer replications (2022);
Ledoit & Wolf 2003/04; Barndorff-Nielsen & Shephard 2004 (bipower variation);
Economic Modelling 160 (2026) BTC jump dynamics; Sarrafshirazi 2025 (HAR-RV BTC).
Full memo with links: produced 2026-07-07 by the research agent (session record).
