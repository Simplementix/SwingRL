"""Equity daily trading environment for RL training.

StockTradingEnv specializes BaseTradingEnv for 8-ETF daily bar trading
with 252-step episodes (one trading year). Sequential start positioning.

TRAIN-01: Equity environment with Gymnasium-compatible step/reset contract.
"""

from __future__ import annotations

import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.envs.base import BaseTradingEnv
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)


class StockTradingEnv(BaseTradingEnv):
    """Equity daily bar trading environment for 8 ETFs.

    Extends BaseTradingEnv with equity-specific configuration:
    - 252-step episodes (one trading year)
    - Sequential start positioning (step 0)
    - 8 ETF assets (SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK)

    Args:
        features: Pre-loaded feature array, shape (n_steps, 164), float32.
        prices: Pre-loaded close price array, shape (n_steps, 8), float32.
        config: Validated SwingRLConfig instance.
        render_mode: Gymnasium render mode (optional).
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: SwingRLConfig,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            features=features,
            prices=prices,
            config=config,
            environment="equity",
            render_mode=render_mode,
        )

    def _select_start_step(self) -> int:
        """Return a random start step for equity episodes."""
        max_start = len(self._features) - self._episode_bars
        if max_start <= 0:
            return 0
        return int(self.np_random.integers(0, max_start))

    @classmethod
    def from_arrays(
        cls,
        features: np.ndarray,
        prices: np.ndarray,
        config: SwingRLConfig,
        render_mode: str | None = None,
    ) -> StockTradingEnv:
        """Factory method to create StockTradingEnv from pre-loaded arrays.

        Validates array shapes before construction.

        Args:
            features: Feature array, shape (n_steps, 164), float32.
            prices: Price array, shape (n_steps, n_assets), float32.
            config: Validated SwingRLConfig instance.
            render_mode: Gymnasium render mode (optional).

        Returns:
            Configured StockTradingEnv instance.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        n_assets = len(config.equity.symbols)
        if features.ndim != 2:
            raise DataError(f"features must be 2D, got {features.ndim}D")
        if prices.ndim != 2:
            raise DataError(f"prices must be 2D, got {prices.ndim}D")
        if features.shape[0] != prices.shape[0]:
            raise DataError(
                f"features and prices must have same number of steps: "
                f"{features.shape[0]} != {prices.shape[0]}"
            )
        if prices.shape[1] != n_assets:
            raise DataError(
                f"prices must have {n_assets} columns (equity symbols), got {prices.shape[1]}"
            )

        log.info(
            "stock_env_created",
            n_steps=features.shape[0],
            obs_dim=features.shape[1],
            n_assets=n_assets,
        )

        return cls(
            features=features,
            prices=prices,
            config=config,
            render_mode=render_mode,
        )
