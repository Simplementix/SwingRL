"""Tests for Turbulence Index calculator.

Turbulence index computes Mahalanobis distance for regime detection.
Equity uses expanding lookback, crypto uses rolling 1080-bar window.
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
    """Test crypto turbulence with rolling 1080-bar lookback."""

    def test_uses_rolling_lookback(self) -> None:
        """Crypto turbulence uses rolling 1080-bar window."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        returns = _make_returns(n_periods=1500, n_assets=2)
        turb = calc.compute(returns, current_idx=1400)
        assert isinstance(turb, float)
        assert np.isfinite(turb)

    def test_returns_nan_before_warmup(self) -> None:
        """Returns NaN before minimum 360-bar warmup for crypto."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        returns = _make_returns(n_periods=500, n_assets=2)
        turb = calc.compute(returns, current_idx=200)
        assert np.isnan(turb)


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
        """Near-singular covariance (BTC/ETH r=0.9) handled via pinv."""
        from swingrl.features.turbulence import TurbulenceCalculator

        calc = TurbulenceCalculator(environment="crypto")
        rng = np.random.default_rng(42)
        btc = rng.normal(0.001, 0.02, 500)
        # ETH highly correlated with BTC (r ~0.9)
        eth = btc * 1.2 + rng.normal(0.0, 0.005, 500)
        returns = np.column_stack([btc, eth])
        turb = calc.compute(returns, current_idx=400)
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
