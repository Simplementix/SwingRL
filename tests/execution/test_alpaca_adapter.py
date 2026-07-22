"""Tests for Alpaca paper trading adapter.

PAPER-01, PAPER-02: AlpacaAdapter submits bracket orders via alpaca-py
with notional amounts, retry logic, and Protocol conformance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter
from swingrl.execution.adapters.base import ExchangeAdapter
from swingrl.execution.types import FillResult, SizedOrder, ValidatedOrder
from swingrl.utils.exceptions import BrokerError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig


@pytest.fixture
def mock_alpaca_response() -> MagicMock:
    """Mock Alpaca order response with fill data."""
    resp = MagicMock()
    resp.id = "alpaca-order-123"
    resp.filled_avg_price = "150.25"
    resp.filled_qty = "2.5"
    resp.symbol = "SPY"
    resp.side = "buy"
    resp.status = "filled"
    return resp


@pytest.fixture
def validated_order() -> ValidatedOrder:
    """Sample validated equity order."""
    return ValidatedOrder(
        order=SizedOrder(
            symbol="SPY",
            side="buy",
            quantity=2.5,
            dollar_amount=375.50,
            stop_loss_price=145.00,
            take_profit_price=160.00,
            environment="equity",
        ),
        risk_checks_passed=["position_size", "drawdown", "daily_loss"],
    )


@pytest.fixture
def adapter(exec_config: SwingRLConfig) -> AlpacaAdapter:
    """AlpacaAdapter with mocked TradingClient and StockHistoricalDataClient."""
    with (
        patch.dict(
            "os.environ",
            {
                "ALPACA_API_KEY": "test-key",  # pragma: allowlist secret
                "ALPACA_SECRET_KEY": "test-secret",  # pragma: allowlist secret
            },
        ),
        patch("swingrl.execution.adapters.alpaca_adapter.TradingClient") as mock_client_cls,
        patch(
            "swingrl.execution.adapters.alpaca_adapter.StockHistoricalDataClient"
        ) as mock_data_cls,
    ):
        mock_client_cls.return_value = MagicMock()
        mock_data_cls.return_value = MagicMock()
        return AlpacaAdapter(config=exec_config)


class TestAlpacaAdapterProtocol:
    """Verify AlpacaAdapter satisfies ExchangeAdapter Protocol."""

    def test_satisfies_protocol(self, adapter: AlpacaAdapter) -> None:
        """AlpacaAdapter must be recognized as ExchangeAdapter."""
        assert isinstance(adapter, ExchangeAdapter)


class TestSubmitBracketOrder:
    """Verify bracket order submission with correct parameters."""

    def test_submit_bracket_order(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """submit_order creates bracket order with stop-loss and take-profit."""
        adapter._client.submit_order.return_value = mock_alpaca_response

        adapter.submit_order(validated_order)

        call_args = adapter._client.submit_order.call_args
        order_req = call_args.kwargs.get("order_data") or call_args[1].get("order_data")

        # Verify bracket order class
        from alpaca.trading.enums import OrderClass

        assert order_req.order_class == OrderClass.BRACKET

        # Verify stop-loss and take-profit attached
        assert order_req.stop_loss is not None
        assert order_req.take_profit is not None
        assert order_req.stop_loss.stop_price == 145.00
        assert order_req.take_profit.limit_price == 160.00

    def test_notional_not_qty(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """submit_order uses notional parameter (not qty) for fractional shares."""
        adapter._client.submit_order.return_value = mock_alpaca_response

        adapter.submit_order(validated_order)

        call_args = adapter._client.submit_order.call_args
        order_req = call_args.kwargs.get("order_data") or call_args[1].get("order_data")

        assert order_req.notional == 375.50
        assert order_req.qty is None

    def test_fill_result_mapping(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """FillResult fields map correctly from Alpaca response."""
        adapter._client.submit_order.return_value = mock_alpaca_response

        result = adapter.submit_order(validated_order)

        assert isinstance(result, FillResult)
        assert result.trade_id == "alpaca-order-123"
        assert result.symbol == "SPY"
        assert result.side == "buy"
        assert result.fill_price == 150.25
        assert result.quantity == 2.5
        assert result.commission == 0.0
        assert result.broker == "alpaca"
        assert result.environment == "equity"


class TestFillLifecycle:
    """PAPER fill-lifecycle (review C2): honest fill status, timestamps, poll+cancel."""

    def test_synchronous_fill_sets_status_and_timestamps(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """(b) An immediately filled order → status='filled', submitted_at/filled_at set."""
        adapter._client.submit_order.return_value = mock_alpaca_response

        result = adapter.submit_order(validated_order)

        assert result.status == "filled"
        assert result.fill_price == 150.25
        assert result.quantity == 2.5
        # Both lifecycle timestamps must be populated (Task 10 consumes them).
        assert result.submitted_at is not None
        assert result.filled_at is not None
        # No poll needed for an immediate fill.
        adapter._client.get_order_by_id.assert_not_called()

    def test_unfilled_order_polls_then_cancels_returns_pending(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
    ) -> None:
        """(a) A submit that never fills → poll loop runs, order cancelled, status='pending'.

        Critically: quantity/price stay 0 but status='pending' — the pipeline drops it,
        so it never becomes a $0 trades row (review C2/M11).
        """
        unfilled = MagicMock()
        unfilled.id = "alpaca-order-unfilled"
        unfilled.status = "new"
        unfilled.filled_avg_price = None
        unfilled.filled_qty = "0"

        adapter._client.submit_order.return_value = unfilled
        adapter._client.get_order_by_id.return_value = unfilled

        # Bound the poll to two iterations for a fast, deterministic test.
        adapter._config.equity.order_fill_timeout_s = 4
        adapter._config.equity.order_poll_interval_s = 2

        with patch("swingrl.execution.adapters.alpaca_adapter.time.sleep"):
            result = adapter.submit_order(validated_order)

        # Poll loop was actually queried (bounded: timeout/interval = 2 polls).
        assert adapter._client.get_order_by_id.call_count == 2
        # Unfilled order was cancelled.
        adapter._client.cancel_order_by_id.assert_called_once_with("alpaca-order-unfilled")
        # Honest pending result — never recordable as a trade.
        assert result.status == "pending"
        assert result.quantity == 0.0
        assert result.fill_price == 0.0
        assert result.submitted_at is not None
        assert result.filled_at is None

    def test_poll_catches_delayed_fill(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """(a/b) An order that fills during the poll window → status='filled', real price."""
        pending = MagicMock()
        pending.id = "alpaca-order-delayed"
        pending.status = "new"
        pending.filled_avg_price = None
        pending.filled_qty = "0"

        adapter._client.submit_order.return_value = pending
        # First poll still pending, second poll filled.
        filled = mock_alpaca_response
        adapter._client.get_order_by_id.side_effect = [pending, filled]
        adapter._config.equity.order_fill_timeout_s = 60
        adapter._config.equity.order_poll_interval_s = 2

        with patch("swingrl.execution.adapters.alpaca_adapter.time.sleep"):
            result = adapter.submit_order(validated_order)

        assert result.status == "filled"
        assert result.fill_price == 150.25
        adapter._client.cancel_order_by_id.assert_not_called()


class TestPreOpenAuctionSubmission:
    """EXEC-D11: pre-open (market closed at submission) equity execution.

    Buys submit ``notional`` DAY market orders (no qty), sells submit ``qty``; both return
    ``status='pending'`` immediately — Alpaca fills them at the primary exchange's opening
    print and the ~09:35 confirmation job records the fill. Never a synthetic synchronous
    fill, never the poll/cancel path (that is the regular-hours honest-fill lifecycle).
    """

    @staticmethod
    def _order(side: str, dollar_amount: float, quantity: float) -> ValidatedOrder:
        """A validated equity order with no stops (pre-open orders are plain market)."""
        return ValidatedOrder(
            order=SizedOrder(
                symbol="SPY" if side == "buy" else "QQQ",
                side=side,  # type: ignore[arg-type]
                quantity=quantity,
                dollar_amount=dollar_amount,
                stop_loss_price=None,
                take_profit_price=None,
                environment="equity",
            ),
            risk_checks_passed=["position_size"],
        )

    def test_preopen_market_order_submits_notional_and_pends(self, adapter: AlpacaAdapter) -> None:
        """EXEC-D11: pre-open equity buys submit notional DAY market orders and return
        status='pending' — never a synthetic synchronous fill, never a poll/cancel."""
        adapter._client.get_clock.return_value = MagicMock(is_open=False)
        submitted = MagicMock()
        submitted.id = "auction-buy-1"
        submitted.filled_avg_price = None
        submitted.filled_qty = "0"
        adapter._client.submit_order.return_value = submitted

        result = adapter.submit_order(self._order("buy", 25.0, 0.05))

        from alpaca.trading.enums import OrderType, TimeInForce

        order_req = adapter._client.submit_order.call_args.kwargs["order_data"]
        assert order_req.notional == 25.0
        assert order_req.qty is None
        assert order_req.time_in_force == TimeInForce.DAY
        assert order_req.type == OrderType.MARKET
        assert result.status == "pending"
        assert result.trade_id == "auction-buy-1"
        assert result.quantity == 0.0
        assert result.fill_price == 0.0
        assert result.submitted_at is not None
        assert result.filled_at is None
        # Pre-open orders rest until the opening auction — never polled or cancelled here.
        adapter._client.get_order_by_id.assert_not_called()
        adapter._client.cancel_order_by_id.assert_not_called()

    def test_preopen_notional_rounded_to_cents(self, adapter: AlpacaAdapter) -> None:
        """EXEC-D11: the submitted notional is rounded to two decimals (Alpaca cents)."""
        adapter._client.get_clock.return_value = MagicMock(is_open=False)
        submitted = MagicMock()
        submitted.id = "auction-buy-2"
        submitted.filled_avg_price = None
        adapter._client.submit_order.return_value = submitted

        adapter.submit_order(self._order("buy", 25.017, 0.05))

        order_req = adapter._client.submit_order.call_args.kwargs["order_data"]
        assert order_req.notional == 25.02

    def test_preopen_sell_submits_qty_and_pends(self, adapter: AlpacaAdapter) -> None:
        """EXEC-D11: pre-open equity sells submit qty (not notional) DAY market orders and
        return pending — a sell closes a specific share quantity at the open."""
        adapter._client.get_clock.return_value = MagicMock(is_open=False)
        submitted = MagicMock()
        submitted.id = "auction-sell-1"
        submitted.filled_avg_price = None
        submitted.filled_qty = "0"
        adapter._client.submit_order.return_value = submitted

        result = adapter.submit_order(self._order("sell", 25.0, 0.0416))

        from alpaca.trading.enums import OrderSide, TimeInForce

        order_req = adapter._client.submit_order.call_args.kwargs["order_data"]
        assert order_req.qty == 0.0416
        assert order_req.notional is None
        assert order_req.side == OrderSide.SELL
        assert order_req.time_in_force == TimeInForce.DAY
        assert result.status == "pending"
        assert result.trade_id == "auction-sell-1"
        adapter._client.get_order_by_id.assert_not_called()
        adapter._client.cancel_order_by_id.assert_not_called()


class TestOrderStatusPolling:
    """EXEC-D11: get_order_status exposes an order's broker state for the 09:35 job."""

    def test_get_order_status_returns_broker_order(self, adapter: AlpacaAdapter) -> None:
        """get_order_status fetches the order by id so the confirmation job can read
        status/filled_avg_price/filled_qty."""
        order = MagicMock()
        order.status = "filled"
        order.filled_avg_price = "600.10"
        order.filled_qty = "0.0416"
        adapter._client.get_order_by_id.return_value = order

        result = adapter.get_order_status("auction-buy-1")

        adapter._client.get_order_by_id.assert_called_once_with("auction-buy-1")
        assert result.status == "filled"
        assert result.filled_avg_price == "600.10"


class TestRetryLogic:
    """Verify exponential backoff retry on TradingClient failures."""

    def test_retry_on_failure(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
        mock_alpaca_response: MagicMock,
    ) -> None:
        """First 2 calls fail, 3rd succeeds -- verify 3 attempts made."""
        adapter._client.submit_order.side_effect = [
            Exception("timeout"),
            Exception("connection reset"),
            mock_alpaca_response,
        ]

        with patch("swingrl.execution.adapters.alpaca_adapter.time.sleep"):
            result = adapter.submit_order(validated_order)

        assert adapter._client.submit_order.call_count == 3
        assert result.trade_id == "alpaca-order-123"

    def test_all_retries_fail(
        self,
        adapter: AlpacaAdapter,
        validated_order: ValidatedOrder,
    ) -> None:
        """All 3 attempts fail -- verify BrokerError raised."""
        adapter._client.submit_order.side_effect = Exception("permanent failure")

        with (
            patch("swingrl.execution.adapters.alpaca_adapter.time.sleep"),
            pytest.raises(BrokerError, match="permanent failure"),
        ):
            adapter.submit_order(validated_order)

        assert adapter._client.submit_order.call_count == 3
