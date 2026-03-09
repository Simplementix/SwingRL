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
    """AlpacaAdapter with mocked TradingClient."""
    with (
        patch.dict(
            "os.environ",
            {
                "ALPACA_API_KEY": "test-key",  # pragma: allowlist secret
                "ALPACA_SECRET_KEY": "test-secret",  # pragma: allowlist secret
            },
        ),
        patch("swingrl.execution.adapters.alpaca_adapter.TradingClient") as mock_client_cls,
    ):
        mock_client_cls.return_value = MagicMock()
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
