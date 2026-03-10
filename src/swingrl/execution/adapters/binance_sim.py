"""Binance.US simulated fill adapter with virtual balance.

Fetches order book mid-price from Binance.US public API, applies slippage,
and records fills locally. No actual orders are placed -- this is a simulation
adapter for paper trading the crypto environment.

Usage:
    from swingrl.execution.adapters.binance_sim import BinanceSimAdapter
    adapter = BinanceSimAdapter(config=config, db=db_manager)
    fill = adapter.submit_order(validated_order)
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import requests
import structlog

from swingrl.execution.types import FillResult, ValidatedOrder
from swingrl.utils.exceptions import BrokerError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_DEFAULT_SLIPPAGE = 0.0003  # 0.03%
_COMMISSION_RATE = 0.001  # 0.10% per side
_SPREAD_WARNING_THRESHOLD = 0.005  # 0.5%


class BinanceSimAdapter:
    """Binance.US simulated fill adapter with virtual balance tracking.

    Satisfies the ExchangeAdapter Protocol. Fetches real order book data
    from Binance.US but simulates fills locally instead of placing orders.
    """

    BINANCE_US_BASE = "https://api.binance.us"

    def __init__(
        self,
        config: SwingRLConfig,
        db: DatabaseManager,
        alerter: Alerter | None = None,
        slippage: float = _DEFAULT_SLIPPAGE,
    ) -> None:
        """Initialize Binance.US simulated adapter.

        Args:
            config: Validated SwingRLConfig.
            db: DatabaseManager for position reads and virtual balance tracking.
            alerter: Optional Discord alerter for critical failures.
            slippage: Slippage rate to apply (default 0.03%).
        """
        self._config = config
        self._db = db
        self._alerter = alerter
        self._slippage = slippage

        log.info("binance_sim_adapter_initialized", slippage=slippage)

    def submit_order(self, order: ValidatedOrder) -> FillResult:
        """Simulate a fill using order book mid-price with slippage.

        Args:
            order: Validated order with sized_order details.

        Returns:
            FillResult with simulated fill price, commission, and UUID trade_id.

        Raises:
            BrokerError: If mid-price fetch fails after all retries.
        """
        sized = order.order
        mid_price, _bid, _ask = self._get_mid_price(sized.symbol)

        # Apply slippage: buy pays more, sell receives less
        if sized.side == "buy":
            fill_price = mid_price * (1 + self._slippage)
        else:
            fill_price = mid_price * (1 - self._slippage)

        commission = sized.dollar_amount * _COMMISSION_RATE
        slippage_amount = abs(fill_price - mid_price) * sized.quantity
        trade_id = str(uuid.uuid4())

        log.info(
            "simulated_fill",
            symbol=sized.symbol,
            side=sized.side,
            fill_price=fill_price,
            mid_price=mid_price,
            slippage=slippage_amount,
            commission=commission,
            trade_id=trade_id,
        )

        return FillResult(
            trade_id=trade_id,
            symbol=sized.symbol,
            side=sized.side,
            quantity=sized.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
            environment=sized.environment,
            broker="binance_us",
        )

    def get_positions(self) -> list[dict[str, object]]:
        """Get current crypto positions from SQLite.

        Returns:
            List of position dicts for the crypto environment.
        """
        with self._db.sqlite() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, cost_basis, last_price, unrealized_pnl "
                "FROM positions WHERE environment = ?",
                ("crypto",),
            ).fetchall()

        return [
            {
                "symbol": row["symbol"],
                "quantity": row["quantity"],
                "cost_basis": row["cost_basis"],
                "last_price": row["last_price"],
                "unrealized_pnl": row["unrealized_pnl"],
            }
            for row in rows
        ]

    def cancel_order(self, order_id: str) -> bool:
        """No-op for simulated fills.

        Args:
            order_id: Order identifier (unused in simulation).

        Returns:
            Always True since simulated fills are instant.
        """
        log.info("cancel_order_noop", order_id=order_id, reason="simulated_fills")
        return True

    def get_current_price(self, symbol: str) -> float:
        """Get the current mid-price for a symbol from order book.

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT").

        Returns:
            Mid-price as float.
        """
        mid, _bid, _ask = self._get_mid_price(symbol)
        return mid

    def _get_mid_price(self, symbol: str) -> tuple[float, float, float]:
        """Fetch order book and compute mid-price with spread check.

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT").

        Returns:
            Tuple of (mid_price, best_bid, best_ask).

        Raises:
            BrokerError: If all retry attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                url = f"{self.BINANCE_US_BASE}/api/v3/depth"
                response = requests.get(
                    url,
                    params={"symbol": symbol, "limit": "5"},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                best_bid = float(data["bids"][0][0])
                best_ask = float(data["asks"][0][0])
                mid = (best_bid + best_ask) / 2.0

                # Spread sanity check
                spread_pct = (best_ask - best_bid) / mid
                if spread_pct > _SPREAD_WARNING_THRESHOLD:
                    log.warning(
                        "wide_spread_detected",
                        symbol=symbol,
                        spread_pct=spread_pct,
                        bid=best_bid,
                        ask=best_ask,
                    )

                return mid, best_bid, best_ask

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning(
                    "binance_price_fetch_retry",
                    attempt=attempt,
                    max_attempts=_MAX_RETRIES,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    time.sleep(delay)

        error_msg = str(last_error) if last_error else "Unknown error"
        log.error("binance_price_fetch_failed", symbol=symbol, error=error_msg)

        # Record API error for automated trigger detection
        try:
            with self._db.sqlite() as conn:
                conn.execute(
                    "INSERT INTO api_errors (timestamp, broker, status_code, endpoint, "
                    "error_message) VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(UTC).isoformat(),
                        "binance_us",
                        0,
                        f"/api/v3/depth?symbol={symbol}",
                        error_msg,
                    ),
                )
        except Exception:
            log.warning("api_error_tracking_failed", exc_info=True)

        if self._alerter is not None:
            self._alerter.send_alert(
                level="critical",
                title="Binance.US Price Fetch Failed",
                message=f"All {_MAX_RETRIES} attempts failed for {symbol}: {error_msg}",
                environment="crypto",
            )

        raise BrokerError(f"Failed to fetch price for {symbol}: {error_msg}")
