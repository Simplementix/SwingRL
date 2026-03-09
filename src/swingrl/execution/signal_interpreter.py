"""Signal interpreter: ensemble blended actions to discrete trade signals.

Stage 1 of the execution pipeline. Converts continuous portfolio weight targets
from the ensemble blender into buy/sell/hold TradeSignals per symbol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog

from swingrl.execution.types import TradeSignal

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class SignalInterpreter:
    """Convert ensemble blended action arrays into TradeSignal lists.

    Actions within +/- deadzone of the current weight are classified as hold
    and excluded from output. Only actionable buy/sell signals are returned.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize with config for symbol lists and deadzone.

        Args:
            config: SwingRL configuration with equity/crypto symbols and
                    environment.signal_deadzone threshold.
        """
        self._config = config
        self._deadzone = config.environment.signal_deadzone

    def interpret(
        self,
        env_name: str,
        blended_actions: np.ndarray,
        current_weights: np.ndarray,
    ) -> list[TradeSignal]:
        """Interpret blended ensemble actions into trade signals.

        Args:
            env_name: Environment name ("equity" or "crypto").
            blended_actions: Target portfolio weights from ensemble blender.
            current_weights: Current portfolio weights per symbol.

        Returns:
            List of TradeSignal for non-hold actions only.
        """
        symbols = self._get_symbols(env_name)
        signals: list[TradeSignal] = []
        hold_count = 0

        for i, symbol in enumerate(symbols):
            target = float(blended_actions[i])
            current = float(current_weights[i])
            diff = target - current

            if diff > self._deadzone:
                action: Literal["buy", "sell"] = "buy"
            elif diff < -self._deadzone:
                action = "sell"
            else:
                hold_count += 1
                continue  # holds excluded from output

            env_literal: Literal["equity", "crypto"] = (
                "equity" if env_name == "equity" else "crypto"
            )
            signals.append(
                TradeSignal(
                    environment=env_literal,
                    symbol=symbol,
                    action=action,
                    raw_weight=target,
                )
            )

        log.info(
            "signals_interpreted",
            env=env_name,
            symbol_count=len(symbols),
            trade_count=len(signals),
            hold_count=hold_count,
        )
        return signals

    def _get_symbols(self, env_name: str) -> list[str]:
        """Get symbol list for the given environment."""
        if env_name == "equity":
            return self._config.equity.symbols
        return self._config.crypto.symbols
