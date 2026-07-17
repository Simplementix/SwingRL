"""Binance.US simulated fill adapter.

Fetches the order book from the Binance.US public API and simulates a fill by crossing
the real spread (buy pays the best ask, sell receives the best bid), then records the fill
locally. No actual orders are placed -- this is a simulation adapter for paper trading the
crypto environment. See ``docs/execution/sim-fidelity.md`` for the divergence audit; the
D1/D4/D8/D9 fixes cross the book, charge commission on the fill notional, reject
fantasy-wide spreads, and expose the modeled cost as the single source of truth for
``fill_quality.expected_cost_frac``.

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

from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.types import FillResult, ValidatedOrder
from swingrl.utils.exceptions import BrokerError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_DEFAULT_SLIPPAGE = 0.0003  # 0.03% — modeled baseline half-spread (expected slippage)
_COMMISSION_RATE = 0.001  # 0.10% per side (see docs/execution/sim-fidelity.md D2 caveat)
_SPREAD_WARNING_THRESHOLD = 0.005  # 0.5% — log a warning, still fill
# D8: hard reject above this spread. Set to 2× the warn band — a book this wide (100 bps on
# BTC/ETH, normally 1–5 bps) is broken/illiquid, and a fill would be fiction. A named constant
# (not config): this is an execution-safety guardrail, not a user-tunable trading parameter,
# so it stays out of the config surface to avoid schema churn.
_SPREAD_REJECT_THRESHOLD = 0.01  # 1.0%


def modeled_crypto_cost_frac() -> float:
    """Modeled per-fill crypto cost fraction — single source of truth (D9).

    Commission constant + baseline slippage constant. ``fill_processor`` reads this for
    ``fill_quality.expected_cost_frac`` so the *expected* cost and the sim's *realized* cost
    share one definition, instead of the config round-trip figure (``crypto_transaction_cost_pct``)
    diverging from the sim by ~0.09% on every fill. This value is inherently per-fill (per-side),
    resolving the round-trip-vs-per-side semantic mismatch.

    After D1 the sim fills across the real spread, so ``_DEFAULT_SLIPPAGE`` is the *expected*
    half-spread baseline here, not a constant applied to the fill price.

    Returns:
        The modeled per-fill cost fraction (commission + baseline slippage).
    """
    return _COMMISSION_RATE + _DEFAULT_SLIPPAGE


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
    ) -> None:
        """Initialize Binance.US simulated adapter.

        Args:
            config: Validated SwingRLConfig.
            db: DatabaseManager for position reads.
            alerter: Optional Discord alerter for critical failures.
        """
        self._config = config
        self._db = db
        self._alerter = alerter
        # Emergency sells are recorded through the same fill processor as normal
        # fills so a trades row is written and the position row is deleted (review M3).
        self._fill_processor = FillProcessor(db=db)

        log.info(
            "binance_sim_adapter_initialized",
            reject_spread_pct=_SPREAD_REJECT_THRESHOLD,
        )

    def submit_order(self, order: ValidatedOrder) -> FillResult:
        """Simulate a fill by crossing the real spread (best bid/ask).

        The fill crosses the book: a buy pays the best ask, a sell receives the best bid
        (D1) — so the recorded slippage is the real half-spread, not a constant. Commission
        is charged on the executed fill notional (D4). A spread wider than the reject
        threshold is refused rather than filled at a fantasy mid (D8).

        Args:
            order: Validated order with sized_order details.

        Returns:
            FillResult with simulated fill price, commission, and UUID trade_id.

        Raises:
            BrokerError: If the price fetch fails after all retries, or the spread is so wide
                a simulated fill would be unrealistic (D8 hard reject).
        """
        sized = order.order
        mid_price, best_bid, best_ask = self._get_mid_price(sized.symbol)

        # D8: refuse a book so wide a fill would be fiction (guardrail, above warn-only).
        spread_pct = (best_ask - best_bid) / mid_price
        if spread_pct > _SPREAD_REJECT_THRESHOLD:
            log.warning(
                "wide_spread_rejected",
                symbol=sized.symbol,
                side=sized.side,
                spread_pct=spread_pct,
                reject_threshold=_SPREAD_REJECT_THRESHOLD,
            )
            raise BrokerError(
                f"Spread {spread_pct:.4f} for {sized.symbol} exceeds reject threshold "
                f"{_SPREAD_REJECT_THRESHOLD} — order refused (no simulated fill)"
            )

        # D1: cross the book — buy pays the ask, sell receives the bid.
        fill_price = best_ask if sized.side == "buy" else best_bid

        # D4: commission on the executed fill notional (consistent with emergency_sell).
        commission = fill_price * sized.quantity * _COMMISSION_RATE
        slippage_amount = abs(fill_price - mid_price) * sized.quantity
        trade_id = str(uuid.uuid4())
        # Simulated fills are synchronous — status and both lifecycle timestamps are set here.
        now = datetime.now(UTC).isoformat()

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
            status="filled",
            submitted_at=now,
            filled_at=now,
        )

    def get_positions(self) -> list[dict[str, object]]:
        """Get current crypto positions from database.

        Returns:
            List of position dicts for the crypto environment.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, cost_basis, last_price, unrealized_pnl "
                "FROM positions WHERE environment = %s",
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

    def emergency_sell(self, symbol: str, quantity: float) -> FillResult:
        """Emergency market-sell a crypto position (simulated).

        Args:
            symbol: Binance symbol (e.g., "BTCUSDT").
            quantity: Number of units to sell.

        Returns:
            FillResult with simulated fill details.

        Raises:
            BrokerError: If mid-price fetch fails.
        """
        # D1: a forced sell receives the best bid. D8's wide-spread reject is deliberately
        # NOT applied here — an emergency exit (circuit breaker / stop) must never be blocked
        # by a wide book; being stuck in a position we must liquidate is the worse outcome.
        mid_price, best_bid, _best_ask = self._get_mid_price(symbol)
        fill_price = best_bid
        commission = quantity * fill_price * _COMMISSION_RATE  # D4: on the fill notional
        slippage_amount = abs(fill_price - mid_price) * quantity
        trade_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        fill = FillResult(
            trade_id=trade_id,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
            environment="crypto",
            broker="binance_us",
            status="filled",
            submitted_at=now,
            filled_at=now,
        )

        # Record through the fill processor: writes a trades row and deletes the
        # position row (sell-to-zero) instead of leaving a zero-qty ghost (review M3).
        try:
            self._fill_processor.process(fill)
        except Exception:
            log.error("emergency_sell_record_failed", symbol=symbol, exc_info=True)
            raise

        log.info(
            "emergency_sell_simulated",
            symbol=symbol,
            quantity=quantity,
            fill_price=fill_price,
            trade_id=trade_id,
        )

        return fill

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

                if not data.get("bids") or not data.get("asks"):
                    raise BrokerError(f"Empty orderbook for {symbol} — no bids or asks")

                best_bid = float(data["bids"][0][0])
                best_ask = float(data["asks"][0][0])
                mid = (best_bid + best_ask) / 2.0

                if mid <= 0:
                    raise BrokerError(f"Zero or negative mid-price for {symbol}: {mid}")

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
            with self._db.connection() as conn:
                conn.execute(
                    "INSERT INTO api_errors (timestamp, broker, status_code, endpoint, "
                    "error_message) VALUES (%s, %s, %s, %s, %s)",
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
