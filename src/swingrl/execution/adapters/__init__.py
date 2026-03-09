"""Exchange adapter abstractions and implementations.

Provides the ExchangeAdapter Protocol and concrete broker adapters:
- AlpacaAdapter: Alpaca paper/live trading with bracket orders
- BinanceSimAdapter: Binance.US simulated fills with virtual balance
"""

from __future__ import annotations

from swingrl.execution.adapters.base import ExchangeAdapter

__all__ = ["ExchangeAdapter"]
