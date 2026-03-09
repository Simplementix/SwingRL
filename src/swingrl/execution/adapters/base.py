"""ExchangeAdapter Protocol definition.

Defines the broker-agnostic interface that all exchange adapters must satisfy.
Both AlpacaAdapter and BinanceSimAdapter conform to this Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swingrl.execution.types import FillResult, ValidatedOrder


@runtime_checkable
class ExchangeAdapter(Protocol):
    """Protocol for exchange adapter implementations.

    All broker adapters must implement these four methods so the execution
    pipeline can submit orders without knowing which broker is behind it.
    """

    def submit_order(self, order: ValidatedOrder) -> FillResult:
        """Submit an order to the exchange and return the fill result.

        Args:
            order: Validated order that has passed risk checks.

        Returns:
            FillResult with trade_id, fill_price, commission, etc.

        Raises:
            BrokerError: If order submission fails after all retries.
        """
        ...

    def get_positions(self) -> list[dict[str, object]]:
        """Get all current positions from the exchange.

        Returns:
            List of position dicts with symbol, quantity, market_value, etc.
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its ID.

        Args:
            order_id: Exchange-specific order identifier.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        ...

    def get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "BTCUSDT").

        Returns:
            Current price as a float.
        """
        ...
