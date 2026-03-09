"""Position sizer: quarter-Kelly sizing with ATR stops and crypto floor.

Stage 2 of the execution pipeline. Converts TradeSignals into SizedOrders
with risk-adjusted position sizes, stop-losses, and take-profit levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from swingrl.execution.types import SizedOrder, TradeSignal

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)

# Maximum risk per trade as fraction of portfolio value
MAX_RISK_PER_TRADE = 0.02

# Take-profit to stop-loss ratio (reward:risk)
REWARD_RISK_RATIO = 2.0

# ATR multiplier for stop-loss distance
ATR_STOP_MULTIPLIER = 2.0


class PositionSizer:
    """Size trades using quarter-Kelly criterion with ATR-based stops.

    Conservative defaults for paper trading phase. Trading statistics
    (win_rate, avg_win, avg_loss) will be updated from backtest results
    in later phases.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        win_rate: float = 0.50,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
    ) -> None:
        """Initialize position sizer.

        Args:
            config: SwingRL configuration.
            win_rate: Historical win rate (default 0.50 for paper phase).
            avg_win: Average winning trade return (default 3%).
            avg_loss: Average losing trade return (default 2%).
        """
        self._config = config
        self._win_rate = win_rate
        self._avg_win = avg_win
        self._avg_loss = avg_loss

    def size(
        self,
        signal: TradeSignal,
        current_price: float,
        atr_value: float,
        portfolio_value: float,
    ) -> SizedOrder | None:
        """Size a trade signal into a concrete order.

        Args:
            signal: Trade signal from SignalInterpreter.
            current_price: Current market price of the asset.
            atr_value: Current ATR value for stop placement.
            portfolio_value: Total portfolio value for the environment.

        Returns:
            SizedOrder if trade is viable, None if Kelly is negative or
            post-floor risk check fails.
        """
        # Step 1: Compute Kelly criterion
        reward_risk = self._avg_win / self._avg_loss
        kelly = self._win_rate - (1.0 - self._win_rate) / reward_risk

        if kelly <= 0:
            log.info(
                "kelly_negative_skip",
                symbol=signal.symbol,
                kelly=kelly,
                win_rate=self._win_rate,
            )
            return None

        # Step 2: Quarter-Kelly
        quarter_kelly = kelly * 0.25

        # Step 3: Risk cap at 2%
        risk_pct = min(quarter_kelly, MAX_RISK_PER_TRADE)

        # Step 4: Dollar amount
        dollar_amount = risk_pct * portfolio_value

        # Step 5-6: ATR stops and take-profit
        stop_distance = ATR_STOP_MULTIPLIER * atr_value
        tp_distance = stop_distance * REWARD_RISK_RATIO

        if signal.action == "buy":
            stop_loss = current_price - stop_distance
            take_profit = current_price + tp_distance
        else:  # sell
            stop_loss = current_price + stop_distance
            take_profit = current_price - tp_distance

        # Step 7: Crypto floor
        if signal.environment == "crypto":
            min_order = self._config.crypto.min_order_usd
            if dollar_amount < min_order:
                dollar_amount = min_order
                log.info(
                    "crypto_floor_applied",
                    symbol=signal.symbol,
                    floor_usd=min_order,
                )

            # Step 8: Post-floor risk check
            max_pos_value = self._config.crypto.max_position_size * portfolio_value
            if dollar_amount > max_pos_value:
                log.warning(
                    "post_floor_risk_rejection",
                    symbol=signal.symbol,
                    dollar_amount=dollar_amount,
                    max_position_value=max_pos_value,
                )
                return None

        # Step 9: Quantity
        quantity = dollar_amount / current_price

        order = SizedOrder(
            symbol=signal.symbol,
            side=signal.action,
            quantity=quantity,
            dollar_amount=dollar_amount,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            environment=signal.environment,
        )

        log.info(
            "order_sized",
            symbol=order.symbol,
            side=order.side,
            dollar_amount=order.dollar_amount,
            quantity=order.quantity,
            stop_loss=order.stop_loss_price,
            take_profit=order.take_profit_price,
            kelly=kelly,
            quarter_kelly=quarter_kelly,
            risk_pct=risk_pct,
        )
        return order
