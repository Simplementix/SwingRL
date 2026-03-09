"""Alpaca paper trading adapter with bracket orders.

Submits bracket orders via alpaca-py TradingClient using notional (dollar)
amounts for fractional share support. Includes retry logic with exponential
backoff and Discord critical alerts on final failure.

Usage:
    from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter
    adapter = AlpacaAdapter(config=config, alerter=alerter)
    fill = adapter.submit_order(validated_order)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from swingrl.execution.types import FillResult, ValidatedOrder
from swingrl.utils.exceptions import BrokerError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


class AlpacaAdapter:
    """Alpaca paper/live trading adapter with bracket orders.

    Satisfies the ExchangeAdapter Protocol. Uses notional parameter for
    fractional share support and attaches stop-loss/take-profit legs as
    bracket orders.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        alerter: Alerter | None = None,
    ) -> None:
        """Initialize Alpaca adapter.

        Args:
            config: Validated SwingRLConfig for trading mode detection.
            alerter: Optional Discord alerter for critical failure notifications.

        Raises:
            BrokerError: If ALPACA_API_KEY or ALPACA_SECRET_KEY env vars are missing.
        """
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            log.error("alpaca_credentials_missing")
            raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY env vars required")

        paper = config.trading_mode == "paper"
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self._alerter = alerter

        log.info("alpaca_adapter_initialized", paper=paper)

    def submit_order(self, order: ValidatedOrder) -> FillResult:
        """Submit a bracket order to Alpaca with stop-loss and take-profit.

        Args:
            order: Validated order with SizedOrder containing dollar_amount,
                   stop_loss_price, and take_profit_price.

        Returns:
            FillResult with Alpaca order ID, fill price, and zero commission.

        Raises:
            BrokerError: If all retry attempts fail.
        """
        sized = order.order
        side = OrderSide.BUY if sized.side == "buy" else OrderSide.SELL

        if sized.stop_loss_price is None or sized.take_profit_price is None:
            log.error("bracket_order_missing_prices", symbol=sized.symbol)
            raise BrokerError(
                f"Bracket order requires stop_loss and take_profit prices for {sized.symbol}"
            )

        order_req = MarketOrderRequest(
            symbol=sized.symbol,
            notional=sized.dollar_amount,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=sized.stop_loss_price),
            take_profit=TakeProfitRequest(limit_price=sized.take_profit_price),
        )

        response = self._retry(
            lambda: self._client.submit_order(order_data=order_req),
        )

        fill_price = float(response.filled_avg_price)
        quantity = float(response.filled_qty)

        log.info(
            "order_submitted",
            symbol=sized.symbol,
            side=sized.side,
            notional=sized.dollar_amount,
            fill_price=fill_price,
            quantity=quantity,
            order_id=str(response.id),
        )

        return FillResult(
            trade_id=str(response.id),
            symbol=sized.symbol,
            side=sized.side,
            quantity=quantity,
            fill_price=fill_price,
            commission=0.0,
            slippage=0.0,
            environment=sized.environment,
            broker="alpaca",
        )

    def get_positions(self) -> list[dict[str, object]]:
        """Get all current positions from Alpaca.

        Returns:
            List of position dicts with symbol, quantity, market_value, etc.
        """
        positions = self._retry(lambda: self._client.get_all_positions())
        return [
            {
                "symbol": p.symbol,
                "quantity": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its Alpaca order ID.

        Args:
            order_id: Alpaca order UUID string.

        Returns:
            True if cancellation succeeded, False on failure.
        """
        try:
            self._client.cancel_order_by_id(order_id)
            log.info("order_cancelled", order_id=order_id)
            return True
        except Exception:
            log.error("order_cancel_failed", order_id=order_id, exc_info=True)
            return False

    def get_current_price(self, symbol: str) -> float:
        """Get the latest trade price for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "SPY").

        Returns:
            Latest trade price as float.
        """
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self._retry(lambda: self._data_client.get_stock_latest_trade(request))
        return float(trades[symbol].price)

    def _retry(self, fn: Callable[[], Any], max_attempts: int = _MAX_RETRIES) -> Any:  # noqa: ANN401
        """Execute fn with exponential backoff retry.

        Args:
            fn: Callable to execute.
            max_attempts: Maximum number of attempts (default 3).

        Returns:
            Result of fn() on success.

        Raises:
            BrokerError: If all attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning(
                    "alpaca_retry",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                if attempt < max_attempts:
                    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    time.sleep(delay)

        error_msg = str(last_error) if last_error else "Unknown error"
        log.error("alpaca_all_retries_failed", error=error_msg)

        if self._alerter is not None:
            self._alerter.send_alert(
                level="critical",
                title="Alpaca API Failure",
                message=f"All {max_attempts} retry attempts failed: {error_msg}",
                environment="equity",
            )

        raise BrokerError(error_msg)
