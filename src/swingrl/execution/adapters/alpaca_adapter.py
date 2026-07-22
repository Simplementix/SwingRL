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
from datetime import UTC, datetime
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

        self._config = config
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

        # D11 opening-auction path: the 09:15 pre-open cycle runs while the market is closed.
        # Submitting a synchronous market order into a closed market and polling for a fill
        # would time out and cancel a good order (or record a $0 trade). Instead submit a
        # plain DAY market order (buys as notional dollars, sells as qty) that Alpaca fills at
        # the official opening print, and return an honest "pending" — the ~09:35 confirmation
        # job records the real fill. Never a synthetic synchronous fill.
        if not self._market_open():
            return self._submit_preopen(sized, side)

        if sized.stop_loss_price is not None and sized.take_profit_price is not None:
            # Bracket order with SL/TP legs
            order_req = MarketOrderRequest(
                symbol=sized.symbol,
                notional=sized.dollar_amount,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=sized.stop_loss_price),
                take_profit=TakeProfitRequest(limit_price=sized.take_profit_price),
            )
        else:
            # Simple market order (no SL/TP — agents manage risk via weights)
            order_req = MarketOrderRequest(
                symbol=sized.symbol,
                notional=sized.dollar_amount,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.SIMPLE,
            )

        submitted_at = datetime.now(UTC).isoformat()
        response = self._retry(
            lambda: self._client.submit_order(order_data=order_req),
        )

        # Immediate fill — no polling needed.
        if response.filled_avg_price is not None:
            return self._filled_result(response, sized, submitted_at)

        # Not filled at submit time: poll the order status up to the bounded timeout
        # instead of recording a $0 trade (review C2). Filled-during-poll returns the
        # real price; still-unfilled at timeout is cancelled and returned as "pending".
        order_id = str(response.id)
        timeout_s = self._config.equity.order_fill_timeout_s
        interval_s = self._config.equity.order_poll_interval_s
        n_polls = max(1, timeout_s // interval_s)

        for attempt in range(1, n_polls + 1):
            time.sleep(interval_s)
            latest = self._retry(lambda: self._client.get_order_by_id(order_id))
            if latest.filled_avg_price is not None:
                log.info("order_filled_during_poll", order_id=order_id, poll=attempt)
                return self._filled_result(latest, sized, submitted_at)

        # Timeout: cancel the resting order and return an honest pending result.
        self.cancel_order(order_id)
        log.warning(
            "order_unfilled_cancelled",
            order_id=order_id,
            symbol=sized.symbol,
            side=sized.side,
            timeout_s=timeout_s,
        )
        if self._alerter is not None:
            self._alerter.send_alert(
                level="warning",
                title="Order Not Filled",
                message=(
                    f"{sized.symbol} {sized.side} not filled within {timeout_s}s — "
                    "cancelled (no trade recorded)."
                ),
                environment="equity",
            )
        return FillResult(
            trade_id=order_id,
            symbol=sized.symbol,
            side=sized.side,
            quantity=0.0,
            fill_price=0.0,
            commission=0.0,
            slippage=0.0,
            environment=sized.environment,
            broker="alpaca",
            status="pending",
            submitted_at=submitted_at,
            filled_at=None,
        )

    def _filled_result(self, response: Any, sized: Any, submitted_at: str) -> FillResult:  # noqa: ANN401
        """Build a filled FillResult from an Alpaca order response.

        Args:
            response: Alpaca order object with filled_avg_price/filled_qty/id.
            sized: The SizedOrder that produced the submission.
            submitted_at: UTC ISO timestamp captured just before submission.

        Returns:
            FillResult with status="filled" and both lifecycle timestamps set.
        """
        fill_price = float(response.filled_avg_price)
        quantity = float(response.filled_qty)
        filled_at = datetime.now(UTC).isoformat()

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
            status="filled",
            submitted_at=submitted_at,
            filled_at=filled_at,
        )

    def _market_open(self) -> bool:
        """Return True when the Alpaca clock reports the market currently open.

        Drives the submit_order routing (D11): open -> synchronous poll-fill lifecycle;
        closed -> pre-open opening-auction submission returning a pending result. Any error
        reaching the clock is treated as "closed" so a submission never silently falls back
        to a synchronous $0-trade path when the market state is unknown — the safe direction.

        Returns:
            True only when the clock explicitly reports the market open.
        """
        try:
            clock = self.get_clock()
        except Exception:
            log.warning("market_clock_unreachable_in_submit", exc_info=True)
            return False
        return bool(getattr(clock, "is_open", False))

    def _submit_preopen(self, sized: Any, side: OrderSide) -> FillResult:  # noqa: ANN401
        """Submit a pre-open opening-auction DAY market order and return a pending result (D11).

        Buys submit ``notional`` dollar amounts (rounded to cents), sells submit ``qty`` — the
        auction fills each at the official opening print. The order id is returned in
        ``trade_id`` so the ~09:35 confirmation job can poll and record the fill. No poll, no
        cancel, no synthetic fill here.

        Args:
            sized: The SizedOrder being submitted.
            side: The Alpaca OrderSide (BUY/SELL).

        Returns:
            FillResult with status="pending", zero quantity/price, and the broker order id.
        """
        if sized.side == "buy":
            order_req = MarketOrderRequest(
                symbol=sized.symbol,
                notional=round(sized.dollar_amount, 2),
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.SIMPLE,
            )
        else:
            order_req = MarketOrderRequest(
                symbol=sized.symbol,
                qty=sized.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.SIMPLE,
            )

        submitted_at = datetime.now(UTC).isoformat()
        response = self._retry(lambda: self._client.submit_order(order_data=order_req))
        order_id = str(response.id)
        log.info(
            "preopen_order_submitted",
            symbol=sized.symbol,
            side=sized.side,
            notional=round(sized.dollar_amount, 2) if sized.side == "buy" else None,
            quantity=sized.quantity if sized.side == "sell" else None,
            order_id=order_id,
        )
        return FillResult(
            trade_id=order_id,
            symbol=sized.symbol,
            side=sized.side,
            quantity=0.0,
            fill_price=0.0,
            commission=0.0,
            slippage=0.0,
            environment=sized.environment,
            broker="alpaca",
            status="pending",
            submitted_at=submitted_at,
            filled_at=None,
        )

    def get_order_status(self, order_id: str) -> Any:  # noqa: ANN401
        """Fetch an order's current broker state by id (D11 fill confirmation).

        The ~09:35 fill-confirmation job calls this per pending order to read
        ``status``/``filled_avg_price``/``filled_qty``/timestamps and decide whether the
        opening auction filled it.

        Args:
            order_id: Alpaca broker order id.

        Returns:
            The Alpaca order object.
        """
        return self._retry(lambda: self._client.get_order_by_id(order_id))

    def get_clock(self) -> Any:  # noqa: ANN401
        """Return the Alpaca market clock (is_open, next_open, next_close).

        Returns:
            Alpaca Clock object. Timestamps are in Eastern time.
        """
        return self._client.get_clock()

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
