"""Reward functions for RL trading environments.

Provides rolling Sharpe ratio reward with expanding-window warmup for the
initial bars, transitioning to a fixed rolling window.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RollingSharpeReward:
    """Rolling Sharpe ratio reward with expanding-window warmup.

    For the first ``window - 1`` bars, uses an expanding window (all returns
    collected so far). From bar ``window`` onward, uses a fixed rolling window
    of the most recent ``window`` returns. Returns 0.0 when fewer than 2
    returns are available or when standard deviation is near-zero.

    Args:
        window: Rolling window size for Sharpe computation. Default 20.
    """

    def __init__(self, window: int = 20) -> None:
        self._window: int = window
        self._returns: deque[float] = deque(maxlen=window)

    def compute(self, daily_return: float) -> float:
        """Compute rolling Sharpe ratio after observing a new return.

        Args:
            daily_return: Portfolio return for the current period.

        Returns:
            Sharpe ratio (mean / std with ddof=1), or 0.0 if insufficient
            data or near-zero standard deviation.
        """
        self._returns.append(daily_return)

        if len(self._returns) < 2:
            return 0.0

        arr = np.array(self._returns, dtype=np.float64)
        std = float(np.std(arr, ddof=1))

        if std < 1e-8:
            return 0.0

        return float(np.mean(arr)) / std

    def reset(self) -> None:
        """Clear all stored returns."""
        self._returns.clear()
