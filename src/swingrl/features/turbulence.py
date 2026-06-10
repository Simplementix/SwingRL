"""Turbulence index calculators for equity and crypto environments.

Two specialized calculators with different algorithms suited to each asset class:
- Equity: EWMA-weighted Mahalanobis distance with 126-day half-life
- Crypto: Vol z-score x correlation spike composite (robust with limited history)

Both calculators expose compute() and compute_series() with identical algorithms
to eliminate train/live divergence.

Computed on-the-fly at each decision step, NOT stored in PostgreSQL.
"""

from __future__ import annotations

import abc
from typing import Any, Literal

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class BaseTurbulenceCalculator(abc.ABC):
    """Abstract base for turbulence calculators."""

    MIN_WARMUP: int = 0

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


class EquityTurbulenceCalculator(BaseTurbulenceCalculator):
    """EWMA-weighted Mahalanobis distance with 126-day half-life.

    Exponential weighting gives more weight to recent regime shifts while
    maintaining expanding lookback after 252-bar warmup. Uses pseudo-inverse
    for numerical stability.
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
        # EWMA decay from half-life: alpha = 1 - exp(-ln2 / half_life)
        self._alpha = 1.0 - np.exp(-np.log(2.0) / self.HALF_LIFE)

    def _ewma_stats(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute EWMA mean and covariance from a returns window.

        Args:
            returns: (n_periods, n_assets) return history.

        Returns:
            Tuple of (ewma_mean, ewma_cov) arrays.
        """
        alpha = self._alpha
        n_assets = returns.shape[1]

        ewma_mean = returns[0].copy()
        ewma_cov = np.zeros((n_assets, n_assets))

        for i in range(1, len(returns)):
            diff = returns[i] - ewma_mean
            ewma_mean = (1 - alpha) * ewma_mean + alpha * returns[i]
            ewma_cov = (1 - alpha) * ewma_cov + alpha * np.outer(diff, diff)

        return ewma_mean, ewma_cov

    def _mahalanobis(
        self,
        current: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        """Compute Mahalanobis distance from mean using pseudo-inverse.

        Args:
            current: Current return vector.
            mean: EWMA mean vector.
            cov: EWMA covariance matrix.

        Returns:
            Non-negative Mahalanobis distance.
        """
        diff = np.atleast_1d(current - mean)
        if cov.ndim == 0:
            cov = np.atleast_2d(cov)
        inv_cov = np.linalg.pinv(cov)
        return float(np.sqrt(np.abs(diff @ inv_cov @ diff)))

    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute EWMA Mahalanobis turbulence for a single time step.

        Args:
            returns: (n_periods, n_assets) full return history.
            current_idx: Row index of the "current" bar.

        Returns:
            Turbulence score (non-negative float), or NaN if insufficient warmup.
        """
        if current_idx < self.MIN_WARMUP:
            return float("nan")

        historical = returns[:current_idx]
        if len(historical) < self.MIN_WARMUP:
            return float("nan")

        ewma_mean, ewma_cov = self._ewma_stats(historical)
        return self._mahalanobis(returns[current_idx], ewma_mean, ewma_cov)

    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute EWMA Mahalanobis turbulence for all bars.

        Uses identical EWMA statistics as compute() — incremental update
        avoids O(n^2) recomputation while producing the same results.

        Args:
            returns: (n_periods, n_assets) full return history.

        Returns:
            (n_periods,) array with NaN for warmup bars and turbulence after.
        """
        n_periods = returns.shape[0]
        n_assets = returns.shape[1] if returns.ndim > 1 else 1
        result = np.full(n_periods, np.nan)

        if n_periods <= self.MIN_WARMUP:
            return result

        alpha = self._alpha

        # Bootstrap EWMA stats from warmup period
        warmup = returns[: self.MIN_WARMUP]
        ewma_mean = warmup[0].copy()
        ewma_cov = np.zeros((n_assets, n_assets))

        for i in range(1, self.MIN_WARMUP):
            diff = warmup[i] - ewma_mean
            ewma_mean = (1 - alpha) * ewma_mean + alpha * warmup[i]
            ewma_cov = (1 - alpha) * ewma_cov + alpha * np.outer(diff, diff)

        # Compute turbulence from warmup onward with incremental updates
        for idx in range(self.MIN_WARMUP, n_periods):
            current = returns[idx]
            result[idx] = self._mahalanobis(current, ewma_mean, ewma_cov)

            # Update EWMA stats for next iteration
            diff = current - ewma_mean
            ewma_mean = (1 - alpha) * ewma_mean + alpha * current
            ewma_cov = (1 - alpha) * ewma_cov + alpha * np.outer(diff, diff)

        return result


class CryptoTurbulenceCalculator(BaseTurbulenceCalculator):
    """Vol z-score x correlation spike composite for crypto.

    Crypto lacks enough history for robust covariance estimation, so we use
    a composite of two signals:
    1. Volatility z-score: how unusual current vol is vs rolling history
    2. Correlation spike: how much cross-asset correlation deviates from norm

    Both are computed on rolling windows after a 360-bar warmup.
    """

    # Defaults match FeaturesConfig schema — overridden by config when available
    MIN_WARMUP = 360
    ROLLING_WINDOW = 1080
    VOL_WINDOW = 30  # Short-term vol measurement window

    def __init__(
        self,
        min_warmup: int | None = None,
        rolling_window: int | None = None,
    ) -> None:
        """Initialize crypto turbulence calculator.

        Args:
            min_warmup: Override MIN_WARMUP (from config.features.crypto_turbulence_warmup).
            rolling_window: Override ROLLING_WINDOW (from config.features.crypto_turbulence_window).
        """
        if min_warmup is not None:
            self.MIN_WARMUP = min_warmup
        if rolling_window is not None:
            self.ROLLING_WINDOW = rolling_window

    def _vol_zscore(self, returns: np.ndarray, current_idx: int, start: int) -> float:
        """Compute volatility z-score for current bar.

        Args:
            returns: Full return history.
            current_idx: Current bar index.
            start: Start of lookback window.

        Returns:
            Absolute z-score of current short-term vol vs historical vol.
        """
        # Current short-term realized vol (last VOL_WINDOW bars)
        vol_start = max(start, current_idx - self.VOL_WINDOW)
        recent = returns[vol_start : current_idx + 1]
        current_vol = np.std(recent, axis=0).mean()

        # Historical vol distribution (rolling VOL_WINDOW chunks)
        historical = returns[start:current_idx]
        n_chunks = max(1, len(historical) // self.VOL_WINDOW)
        chunk_vols = []
        for i in range(n_chunks):
            chunk_start = start + i * self.VOL_WINDOW
            chunk_end = min(chunk_start + self.VOL_WINDOW, current_idx)
            if chunk_end <= chunk_start:
                break
            chunk = returns[chunk_start:chunk_end]
            chunk_vols.append(np.std(chunk, axis=0).mean())

        if len(chunk_vols) < 2:
            return 0.0

        hist_mean = np.mean(chunk_vols)
        hist_std = np.std(chunk_vols)
        if hist_std < 1e-10:
            return 0.0

        return float(np.abs((current_vol - hist_mean) / hist_std))

    def _corr_spike(self, returns: np.ndarray, current_idx: int, start: int) -> float:
        """Compute correlation spike metric.

        Measures how much current short-term cross-asset correlation exceeds
        the historical average correlation. During crises, correlations spike
        toward 1.0.

        Args:
            returns: Full return history.
            current_idx: Current bar index.
            start: Start of lookback window.

        Returns:
            Non-negative correlation spike score.
        """
        n_assets = returns.shape[1] if returns.ndim > 1 else 1
        if n_assets < 2:
            return 0.0

        # Current short-term correlation
        vol_start = max(start, current_idx - self.VOL_WINDOW)
        recent = returns[vol_start : current_idx + 1]
        if len(recent) < 3:
            return 0.0
        current_corr = np.corrcoef(recent.T)
        # Mean off-diagonal correlation
        mask = ~np.eye(n_assets, dtype=bool)
        current_avg_corr = np.mean(np.abs(current_corr[mask]))

        # Historical average correlation
        historical = returns[start:current_idx]
        if len(historical) < self.VOL_WINDOW:
            return 0.0
        hist_corr = np.corrcoef(historical.T)
        hist_avg_corr = np.mean(np.abs(hist_corr[mask]))

        # Spike = how much current exceeds historical (clamped to non-negative)
        spike = max(0.0, float(current_avg_corr - hist_avg_corr))

        # Normalize: correlation ranges [0,1], so spike in [0,1]
        # Scale to similar magnitude as vol z-score
        return spike * 10.0

    def _compute_single(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute composite turbulence for a single bar.

        Args:
            returns: Full return history.
            current_idx: Current bar index.

        Returns:
            Composite turbulence score, or NaN if insufficient warmup.
        """
        if current_idx < self.MIN_WARMUP:
            return float("nan")

        start = max(0, current_idx - self.ROLLING_WINDOW)
        available = current_idx - start
        if available < self.MIN_WARMUP:
            return float("nan")

        vol_z = self._vol_zscore(returns, current_idx, start)
        corr_s = self._corr_spike(returns, current_idx, start)

        # Composite: geometric-ish blend — both must be elevated for high score
        # Add 1 to vol_z to ensure non-zero base, multiply by corr component
        turbulence = vol_z * (1.0 + corr_s)

        return max(0.0, turbulence)

    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute vol-corr composite turbulence for a single time step.

        Args:
            returns: (n_periods, n_assets) full return history.
            current_idx: Row index of the "current" bar.

        Returns:
            Turbulence score (non-negative float), or NaN if insufficient warmup.
        """
        return self._compute_single(returns, current_idx)

    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute vol-corr composite turbulence for all bars.

        Uses identical algorithm as compute() for each bar — no divergence
        between training and live inference.

        Args:
            returns: (n_periods, n_assets) full return history.

        Returns:
            (n_periods,) array with NaN for warmup bars and turbulence after.
        """
        n_periods = returns.shape[0]
        result = np.full(n_periods, np.nan)

        if n_periods <= self.MIN_WARMUP:
            return result

        for idx in range(self.MIN_WARMUP, n_periods):
            result[idx] = self._compute_single(returns, idx)

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
        rolling_window=getattr(features, "crypto_turbulence_window", None),
    )
