"""Turbulence index calculator using Mahalanobis distance.

Measures how unusual current market returns are relative to historical norms.
- Equity: expanding lookback after 252-bar warmup
- Crypto: rolling 1080-bar lookback after 360-bar warmup

Computed on-the-fly at each decision step, NOT stored in DuckDB.
Uses np.linalg.pinv (pseudo-inverse) for numerical stability with
near-singular covariance matrices (e.g., BTC/ETH correlation ~0.9).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class TurbulenceCalculator:
    """Turbulence index via Mahalanobis distance.

    Equity uses expanding lookback (all history after 252-bar warmup).
    Crypto uses rolling 1080-bar lookback (after 360-bar warmup).
    """

    def __init__(self, environment: Literal["equity", "crypto"]) -> None:
        """Initialize turbulence calculator for a specific environment.

        Args:
            environment: "equity" or "crypto" — determines lookback mode and warmup.
        """
        self.environment = environment

        if environment == "equity":
            self.min_warmup = 252
            self.rolling_window: int | None = None  # expanding
        else:
            self.min_warmup = 360
            self.rolling_window = 1080

    def compute(self, returns: np.ndarray, current_idx: int) -> float:
        """Compute turbulence index for a single time step.

        Args:
            returns: (n_periods, n_assets) full return history.
            current_idx: Row index of the "current" bar.

        Returns:
            Turbulence score (non-negative float), or NaN if insufficient warmup.
        """
        if current_idx < self.min_warmup:
            return float("nan")

        # Select historical returns based on mode
        if self.rolling_window is not None:
            start = max(0, current_idx - self.rolling_window)
        else:
            start = 0

        historical = returns[start:current_idx]

        if len(historical) < self.min_warmup:
            return float("nan")

        current = returns[current_idx]
        mean_returns = historical.mean(axis=0)
        cov_matrix = np.cov(historical.T)

        # Handle 1D case: np.cov returns scalar for single asset
        if cov_matrix.ndim == 0:
            cov_matrix = np.atleast_2d(cov_matrix)

        inv_cov = np.linalg.pinv(cov_matrix)
        diff = current - mean_returns

        # Ensure diff is 1D for matrix multiply
        diff = np.atleast_1d(diff)

        turbulence = float(np.sqrt(np.abs(diff @ inv_cov @ diff)))
        return turbulence

    def compute_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute turbulence for all bars in the return matrix.

        Args:
            returns: (n_periods, n_assets) full return history.

        Returns:
            (n_periods,) array with NaN for warmup bars and turbulence values after.
        """
        n_periods = returns.shape[0]
        result = np.full(n_periods, np.nan)

        for idx in range(self.min_warmup, n_periods):
            result[idx] = self.compute(returns, idx)

        return result
