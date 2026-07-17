"""Turbulence index calculators for equity and crypto environments.

Both calculators are built on an EWMA-weighted Mahalanobis distance (Kritzman &
Li 2010), sharing the same numerically-hardened core so training and live
inference cannot diverge:

- Equity: 8-asset EWMA-Mahalanobis, 126-day half-life, 252-bar warmup.
- Crypto: 2-asset EWMA-Mahalanobis, 750-bar (4H) half-life, 1080-bar warmup,
  OR-gated with a *signed* realized-vol component so a dead-calm stretch scores
  LOW while a correlation-sign flip scores HIGH.

Hardening (method review 2026-07-07, Plan A Task 6a):
- Eigenvalue floor: eigenvalues < 1e-3 * lambda_max are clipped before
  inversion (replaces bare ``pinv``), so noise along near-null directions of a
  near-singular covariance cannot dominate the score. After flooring the
  quadratic form is provably non-negative, so the old ``abs()`` becomes an
  assert.
- EWMA warm-start de-bias: the zero-initialised covariance is divided by
  ``1 - (1 - alpha) ** t`` so the ~10-15% post-warmup inflation is removed.

Computed on-the-fly at each decision step, NOT stored in PostgreSQL.
"""

from __future__ import annotations

import abc
from typing import Any, Literal

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class BaseTurbulenceCalculator(abc.ABC):
    """Abstract base for turbulence calculators.

    Provides the shared EWMA-Mahalanobis machinery (de-biased covariance +
    eigenvalue-floored inverse) so both concrete calculators use identical,
    hardened numerics.
    """

    MIN_WARMUP: int = 0
    # Eigenvalues below this fraction of lambda_max are floored before inversion.
    EIGENVALUE_FLOOR_FRAC: float = 1e-3

    @property
    def min_warmup(self) -> int:
        """Minimum number of bars required before turbulence can be computed."""
        return self.MIN_WARMUP

    @abc.abstractmethod
    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute turbulence for a single time step.

        Args:
            returns: (n_periods, n_assets) full return history.
            current_idx: Row index of the "current" bar.

        Returns:
            Turbulence score (non-negative float), or NaN if insufficient warmup.
        """

    @abc.abstractmethod
    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute turbulence for all bars.

        Args:
            returns: (n_periods, n_assets) full return history.

        Returns:
            (n_periods,) array with NaN for warmup bars and turbulence values after.
        """

    # -- shared numerics ---------------------------------------------------

    @staticmethod
    def _alpha_from_half_life(half_life: int) -> float:
        """EWMA decay from a half-life: ``alpha = 1 - exp(-ln2 / half_life)``."""
        return 1.0 - float(np.exp(-np.log(2.0) / half_life))

    @staticmethod
    def _debias_factor(n_updates: int, alpha: float) -> float:
        """Warm-start de-bias factor ``1 - (1 - alpha) ** n_updates``.

        The zero-initialised EWMA covariance accumulates weight
        ``1 - (1 - alpha) ** t`` after ``t`` updates; dividing by this factor
        removes the systematic low bias (and the resulting turbulence inflation).
        """
        if n_updates <= 0:
            return 1.0
        return 1.0 - (1.0 - alpha) ** n_updates

    def _mahalanobis(self, current: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Mahalanobis distance with an eigenvalue floor on the covariance.

        Args:
            current: Current return vector.
            mean: EWMA mean vector.
            cov: (de-biased) EWMA covariance matrix.

        Returns:
            Non-negative Mahalanobis distance.
        """
        diff = np.atleast_1d(np.asarray(current, dtype=float) - np.asarray(mean, dtype=float))
        cov_2d = np.atleast_2d(np.asarray(cov, dtype=float))
        # Symmetric eigendecomposition (cov is symmetric by construction).
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        max_eig = float(eigvals[-1])  # eigh returns eigenvalues in ascending order
        if max_eig <= 0.0:
            # Degenerate all-zero covariance (e.g. identical returns) — no distance.
            return 0.0
        floor = self.EIGENVALUE_FLOOR_FRAC * max_eig
        eigvals_floored = np.maximum(eigvals, floor)
        inv_cov = (eigvecs / eigvals_floored) @ eigvecs.T
        dist_sq = float(diff @ inv_cov @ diff)
        # After flooring, inv_cov is positive-definite ⇒ the quadratic form is
        # non-negative. Any negative value is floating-point noise near zero; the
        # max(..., 0.0) below keeps this safe even when asserts are stripped (-O).
        assert dist_sq >= -1e-9, f"Mahalanobis dist_sq negative after floor: {dist_sq}"  # nosec B101
        return float(np.sqrt(max(dist_sq, 0.0)))

    def _ewma_mahalanobis_series(
        self, returns: np.ndarray, warmup: int, alpha: float
    ) -> np.ndarray:
        """De-biased expanding EWMA-Mahalanobis for every bar.

        Args:
            returns: (n_periods, n_assets) return history.
            warmup: Bars used to bootstrap EWMA stats (NaN before this).
            alpha: EWMA decay.

        Returns:
            (n_periods,) array — NaN for warmup bars, distances after.
        """
        r = returns if returns.ndim > 1 else returns.reshape(-1, 1)
        n_periods, n_assets = r.shape
        result = np.full(n_periods, np.nan)
        if n_periods <= warmup:
            return result

        mean = r[0].astype(float).copy()
        cov = np.zeros((n_assets, n_assets))
        n_updates = 0
        for i in range(1, warmup):
            diff = r[i] - mean
            mean = (1 - alpha) * mean + alpha * r[i]
            cov = (1 - alpha) * cov + alpha * np.outer(diff, diff)
            n_updates += 1

        for idx in range(warmup, n_periods):
            debias = self._debias_factor(n_updates, alpha)
            result[idx] = self._mahalanobis(r[idx], mean, cov / debias)
            diff = r[idx] - mean
            mean = (1 - alpha) * mean + alpha * r[idx]
            cov = (1 - alpha) * cov + alpha * np.outer(diff, diff)
            n_updates += 1

        return result

    def _ewma_mahalanobis_point(
        self, returns: np.ndarray, current_idx: int, warmup: int, alpha: float
    ) -> float:
        """De-biased expanding EWMA-Mahalanobis for a single bar.

        Equivalent to ``_ewma_mahalanobis_series(returns, ...)[current_idx]`` —
        EWMA stats are accumulated over ``returns[:current_idx]`` and the
        distance evaluated at ``returns[current_idx]``.
        """
        r = returns if returns.ndim > 1 else returns.reshape(-1, 1)
        if current_idx < warmup or current_idx >= len(r):
            return float("nan")

        mean = r[0].astype(float).copy()
        cov = np.zeros((r.shape[1], r.shape[1]))
        n_updates = 0
        for i in range(1, current_idx):
            diff = r[i] - mean
            mean = (1 - alpha) * mean + alpha * r[i]
            cov = (1 - alpha) * cov + alpha * np.outer(diff, diff)
            n_updates += 1

        debias = self._debias_factor(n_updates, alpha)
        return self._mahalanobis(r[current_idx], mean, cov / debias)


class EquityTurbulenceCalculator(BaseTurbulenceCalculator):
    """EWMA-weighted Mahalanobis distance with 126-day half-life.

    Exponential weighting gives more weight to recent regime shifts while
    maintaining an expanding lookback after the 252-bar warmup. Uses an
    eigenvalue-floored inverse and a warm-start de-bias factor for numerical
    stability (method review 2026-07-07).
    """

    # Defaults match FeaturesConfig schema — overridden by config when available
    MIN_WARMUP = 252
    HALF_LIFE = 126

    def __init__(
        self,
        min_warmup: int | None = None,
        half_life: int | None = None,
    ) -> None:
        """Initialize equity turbulence calculator.

        Args:
            min_warmup: Override MIN_WARMUP (from config.features.equity_turbulence_warmup).
            half_life: Override HALF_LIFE (from config.features.equity_turbulence_half_life).
        """
        if min_warmup is not None:
            self.MIN_WARMUP = min_warmup
        if half_life is not None:
            self.HALF_LIFE = half_life
        self._alpha = self._alpha_from_half_life(self.HALF_LIFE)

    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute EWMA Mahalanobis turbulence for a single time step."""
        return self._ewma_mahalanobis_point(returns, current_idx, self.MIN_WARMUP, self._alpha)

    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute EWMA Mahalanobis turbulence for all bars."""
        return self._ewma_mahalanobis_series(returns, self.MIN_WARMUP, self._alpha)


class CryptoTurbulenceCalculator(BaseTurbulenceCalculator):
    """2-asset EWMA-Mahalanobis OR-gated with a signed realized-vol component.

    Replaces the old ``vol_z * (1 + abs(corr))`` composite, which had two
    verified defects (method review 2026-07-07):

    - ``abs(vol z)`` scored dead-calm stretches as turbulence.
    - ``abs(corr)`` gave a +0.8 -> -0.8 correlation flip zero spike.

    The EWMA-Mahalanobis term (shared with equity) captures correlation-structure
    changes, including sign flips, because a flipped bar is unusual against the
    covariance the EWMA has learned. The OR-gate adds a *signed* realized-vol
    z-score (clamped at zero) so genuine volatility spikes still register while
    a dead-calm stretch — vol far below the historical mean — contributes zero.

    ``turbulence = max(mahalanobis, max(0, signed_vol_zscore))``
    """

    # Defaults match FeaturesConfig schema — overridden by config when available
    MIN_WARMUP = 1080
    HALF_LIFE = 750
    VOL_WINDOW = 30  # short-term realized-vol measurement window
    VOL_LOOKBACK = 1080  # rolling history for the realized-vol percentile OR-gate

    def __init__(
        self,
        min_warmup: int | None = None,
        half_life: int | None = None,
        vol_window: int | None = None,
        vol_lookback: int | None = None,
    ) -> None:
        """Initialize crypto turbulence calculator.

        Args:
            min_warmup: Override MIN_WARMUP (config.features.crypto_turbulence_warmup).
            half_life: Override HALF_LIFE (config.features.crypto_turbulence_half_life).
            vol_window: Override VOL_WINDOW (short-term realized-vol window).
            vol_lookback: Override VOL_LOOKBACK (config.features.crypto_turbulence_window).
        """
        if min_warmup is not None:
            self.MIN_WARMUP = min_warmup
        if half_life is not None:
            self.HALF_LIFE = half_life
        if vol_window is not None:
            self.VOL_WINDOW = vol_window
        if vol_lookback is not None:
            self.VOL_LOOKBACK = vol_lookback
        self._alpha = self._alpha_from_half_life(self.HALF_LIFE)

    def _signed_vol_component(self, returns: np.ndarray, current_idx: int) -> float:
        """Signed realized-vol z-score, clamped at zero.

        Positive only when current short-term realized vol is *above* its
        rolling-window history — a dead-calm stretch (vol below the mean)
        returns 0.0. This is the "signed" fix for the old ``abs(vol z)`` defect.

        Args:
            returns: Full return history.
            current_idx: Current bar index.

        Returns:
            ``max(0, (current_vol - hist_mean) / hist_std)``, or 0.0 when
            history is insufficient.
        """
        r = returns if returns.ndim > 1 else returns.reshape(-1, 1)
        start = max(0, current_idx - self.VOL_LOOKBACK)

        recent = r[max(start, current_idx - self.VOL_WINDOW + 1) : current_idx + 1]
        if len(recent) < 2:
            return 0.0
        current_vol = float(np.std(recent, axis=0).mean())

        historical = r[start:current_idx]
        n_chunks = len(historical) // self.VOL_WINDOW
        if n_chunks < 2:
            return 0.0
        chunk_vols = [
            float(
                np.std(historical[i * self.VOL_WINDOW : (i + 1) * self.VOL_WINDOW], axis=0).mean()
            )
            for i in range(n_chunks)
        ]
        hist_mean = float(np.mean(chunk_vols))
        hist_std = float(np.std(chunk_vols))
        if hist_std < 1e-12:
            return 0.0
        signed_z = (current_vol - hist_mean) / hist_std
        return max(0.0, signed_z)

    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute EWMA-Mahalanobis OR signed-vol turbulence for a single bar."""
        maha = self._ewma_mahalanobis_point(returns, current_idx, self.MIN_WARMUP, self._alpha)
        if np.isnan(maha):
            return float("nan")
        vol = self._signed_vol_component(returns, current_idx)
        return max(float(maha), float(vol))

    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute EWMA-Mahalanobis OR signed-vol turbulence for all bars."""
        maha_series = self._ewma_mahalanobis_series(returns, self.MIN_WARMUP, self._alpha)
        n_periods = returns.shape[0]
        result = np.full(n_periods, np.nan)
        for idx in range(self.MIN_WARMUP, n_periods):
            maha = maha_series[idx]
            if np.isnan(maha):
                continue
            vol = self._signed_vol_component(returns, idx)
            result[idx] = max(float(maha), float(vol))
        return result


def TurbulenceCalculator(  # noqa: N802
    environment: Literal["equity", "crypto"],
    config: Any | None = None,
) -> BaseTurbulenceCalculator:
    """Factory function returning the appropriate turbulence calculator.

    Args:
        environment: "equity" or "crypto" — determines which algorithm is used.
        config: Optional SwingRLConfig for reading turbulence params from yaml.

    Returns:
        EquityTurbulenceCalculator for equity, CryptoTurbulenceCalculator for crypto.
    """
    features = getattr(config, "features", None) if config is not None else None
    if environment == "equity":
        return EquityTurbulenceCalculator(
            min_warmup=getattr(features, "equity_turbulence_warmup", None),
            half_life=getattr(features, "equity_turbulence_half_life", None),
        )
    return CryptoTurbulenceCalculator(
        min_warmup=getattr(features, "crypto_turbulence_warmup", None),
        half_life=getattr(features, "crypto_turbulence_half_life", None),
        vol_lookback=getattr(features, "crypto_turbulence_window", None),
    )
