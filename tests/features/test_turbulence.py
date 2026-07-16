"""Tests for Turbulence Index calculators (equity + crypto).

Equity: EWMA-weighted Mahalanobis distance with 126-day half-life, eigenvalue
floor (near-singular robustness) and warm-start de-bias (6a).

Crypto: 2-asset EWMA-Mahalanobis OR-gated with a signed realized-vol component
(6b) — replaces the old vol-z x abs(corr) composite which mis-scored dead-calm
stretches (abs vol-z) and correlation-sign flips (abs corr).
"""

from __future__ import annotations

import numpy as np


def _make_returns(
    n_periods: int = 500,
    n_assets: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic multi-asset return matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, (n_periods, n_assets))


def _make_crash_returns(
    n_calm: int = 300,
    n_crash: int = 50,
    n_assets: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Create returns with calm period followed by crash period."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0005, 0.005, (n_calm, n_assets))
    crash = rng.normal(-0.03, 0.04, (n_crash, n_assets))
    return np.vstack([calm, crash])


def _corr_series(n: int, rho: float, vol: float, rng: np.random.Generator) -> np.ndarray:
    """Two-asset returns with correlation ``rho`` and identical marginal ``vol``."""
    z = rng.standard_normal((n, 2))
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho * rho)]])
    return (z @ chol.T) * vol


def _flip_series(seed: int, flip_at: int = 1200, total: int = 1450) -> np.ndarray:
    """Two-asset series whose correlation flips +0.8 -> -0.8 at ``flip_at``.

    The underlying shock stream is shared across the boundary so per-asset
    realized volatility is continuous — only the correlation SIGN changes.
    """
    z = np.random.default_rng(seed).standard_normal((total, 2))
    chol_pos = np.array([[1.0, 0.0], [0.8, 0.6]])
    chol_neg = np.array([[1.0, 0.0], [-0.8, 0.6]])
    out = np.empty((total, 2))
    out[:flip_at] = (z[:flip_at] @ chol_pos.T) * 0.02
    out[flip_at:] = (z[flip_at:] @ chol_neg.T) * 0.02
    return out


class TestEquityTurbulence:
    """Test equity turbulence with expanding lookback."""

    def test_uses_expanding_lookback(self) -> None:
        """Equity turbulence uses expanding window after 252-bar warmup."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500)
        # After warmup, turbulence should be computable
        turb = calc.compute(returns, current_idx=300)
        assert isinstance(turb, float)
        assert np.isfinite(turb)

    def test_returns_nan_before_warmup(self) -> None:
        """Returns NaN before minimum 252-bar warmup for equity."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=300)
        turb = calc.compute(returns, current_idx=100)
        assert np.isnan(turb)


class TestCryptoTurbulence:
    """Test crypto turbulence with the new EWMA-Mahalanobis calculator."""

    def test_uses_expanding_lookback(self) -> None:
        """Crypto turbulence is computable after the (raised) warmup."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        returns = _make_returns(n_periods=1500, n_assets=2)
        turb = calc.compute(returns, current_idx=1400)
        assert isinstance(turb, float)
        assert np.isfinite(turb)

    def test_returns_nan_before_warmup(self) -> None:
        """Returns NaN before the crypto warmup (raised to 1080)."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        returns = _make_returns(n_periods=500, n_assets=2)
        turb = calc.compute(returns, current_idx=200)
        assert np.isnan(turb)


class TestEquityHygiene:
    """6a — eigenvalue floor + EWMA warm-start de-bias."""

    def test_stable_under_near_singular_covariance(self) -> None:
        """Near-singular covariance (8 ETFs ~ one factor) stays finite/bounded.

        The eigenvalue floor prevents noise along near-null directions from
        dominating the Mahalanobis distance.
        """
        from swingrl.features.turbulence import TurbulenceCalculator

        rng = np.random.default_rng(3)
        n = 600
        factor = rng.normal(0.0, 0.012, (n, 1))
        # 8 assets share one dominant factor + tiny idiosyncratic noise
        returns = np.tile(factor, (1, 8)) + rng.normal(0.0, 1e-4, (n, 8))
        calc = TurbulenceCalculator(environment="equity")
        series = calc.compute_series(returns)
        valid = series[252:]
        valid = valid[np.isfinite(valid)]
        assert valid.size > 0
        assert np.all(valid >= 0.0)
        assert np.all(np.isfinite(valid))
        assert float(valid.max()) < 1e6  # no blow-up from inverting tiny eigenvalues

    def test_no_systematic_early_inflation(self) -> None:
        """Post-warmup series shows no systematic early inflation vs long-run.

        The zero-init EWMA covariance biases the distance high right after
        warmup (~10% here); the 1/(1-(1-a)^t) de-bias removes it. Averaged
        over seeds so the systematic component dominates the sampling noise.
        """
        from swingrl.features.turbulence import TurbulenceCalculator

        ratios: list[float] = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            n = 2000
            returns = rng.normal(0.0, 0.01, (n, 1)) + rng.normal(0.0, 0.004, (n, 8))
            series = TurbulenceCalculator(environment="equity").compute_series(returns)
            valid = series[252:]
            valid = valid[np.isfinite(valid)]
            ratios.append(float(valid[:60].mean() / valid[-800:].mean()))
        assert float(np.mean(ratios)) < 1.05


class TestCryptoReplacement:
    """6b — dead-calm scores LOW, correlation-sign flip scores HIGH."""

    def test_dead_calm_scores_low(self) -> None:
        """A dead-calm stretch scores below the normal regime (kills abs vol-z)."""
        from swingrl.features.turbulence import TurbulenceCalculator

        rng = np.random.default_rng(1)
        normal = _corr_series(1300, rho=0.5, vol=0.02, rng=rng)
        calm = _corr_series(300, rho=0.5, vol=0.0015, rng=rng)
        series = np.vstack([normal, calm])

        calc = TurbulenceCalculator(environment="crypto")
        scores = calc.compute_series(series)
        normal_score = float(np.nanmean(scores[1200:1300]))
        calm_score = float(np.nanmean(scores[1450:1580]))
        assert calm_score < 0.7 * normal_score

    def test_correlation_flip_scores_high(self) -> None:
        """A +0.8 -> -0.8 correlation flip scores HIGH (kills abs corr).

        Seed-averaged over the window immediately after the flip, where the
        old composite's abs(corr) spike had not yet fired — the structural
        anti-correlation is what the EWMA-Mahalanobis catches.
        """
        from swingrl.features.turbulence import TurbulenceCalculator

        pre_scores: list[float] = []
        post_scores: list[float] = []
        for seed in range(12):
            series = _flip_series(seed, flip_at=1200, total=1450)
            scores = TurbulenceCalculator(environment="crypto").compute_series(series)
            pre_scores.append(float(np.nanmean(scores[1120:1200])))
            post_scores.append(float(np.nanmean(scores[1200:1240])))
        assert float(np.mean(post_scores)) > 1.2 * float(np.mean(pre_scores))

    def test_warmup_default_raised_to_1080(self) -> None:
        """Crypto warmup default raised 360 -> 1080 bars (method review 2026-07-07)."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        assert calc.min_warmup == 1080


class TestTurbulenceProperties:
    """Test general turbulence properties."""

    def test_non_negative(self) -> None:
        """Turbulence is always non-negative."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500)
        turb = calc.compute(returns, current_idx=300)
        assert turb >= 0.0

    def test_higher_during_crash(self) -> None:
        """Turbulence is higher during crash period than calm period."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_crash_returns(n_calm=300, n_crash=50)
        # Calm period turbulence
        calm_turb = calc.compute(returns, current_idx=280)
        # Crash period turbulence (computed using calm history)
        crash_turb = calc.compute(returns, current_idx=320)
        assert crash_turb > calm_turb

    def test_near_singular_handled(self) -> None:
        """Near-singular covariance (BTC/ETH r~0.9) handled via eigenvalue floor."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        rng = np.random.default_rng(42)
        btc = rng.normal(0.001, 0.02, 1500)
        # ETH highly correlated with BTC (r ~0.9)
        eth = btc * 1.2 + rng.normal(0.0, 0.005, 1500)
        returns = np.column_stack([btc, eth])
        turb = calc.compute(returns, current_idx=1400)
        assert np.isfinite(turb)
        assert turb >= 0.0

    def test_single_asset_graceful(self) -> None:
        """Single-asset (1D) turbulence degrades gracefully."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500, n_assets=1)
        turb = calc.compute(returns, current_idx=300)
        assert np.isfinite(turb)
        assert turb >= 0.0


class TestComputeSeries:
    """Test batch turbulence computation."""

    def test_output_shape(self) -> None:
        """compute_series returns array with same length as input."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500)
        series = calc.compute_series(returns)
        assert series.shape == (500,)

    def test_nan_during_warmup(self) -> None:
        """Warmup bars are NaN in series output."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500)
        series = calc.compute_series(returns)
        # First 252 bars should be NaN (warmup)
        assert np.all(np.isnan(series[:252]))
        # After warmup, should have finite values
        assert np.all(np.isfinite(series[252:]))

    def test_compute_matches_series(self) -> None:
        """compute(returns, idx) equals compute_series(returns)[idx] (no divergence)."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="equity")
        returns = _make_returns(n_periods=500)
        series = calc.compute_series(returns)
        for idx in (300, 400, 499):
            assert np.isclose(calc.compute(returns, idx), series[idx], rtol=1e-9, atol=1e-9)
