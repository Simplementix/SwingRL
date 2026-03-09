"""Tests for Binance.US simulated fill adapter.

PAPER-01, PAPER-10: BinanceSimAdapter fetches order book mid-price,
applies slippage, computes commission, and satisfies ExchangeAdapter Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

from swingrl.execution.adapters.base import ExchangeAdapter
from swingrl.execution.types import SizedOrder, ValidatedOrder

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager


@pytest.fixture
def order_book_response() -> dict[str, list[list[str]]]:
    """Mock Binance order book response."""
    return {
        "bids": [["50000.00", "1.5"], ["49999.00", "2.0"]],
        "asks": [["50010.00", "1.0"], ["50020.00", "0.5"]],
    }


@pytest.fixture
def crypto_order() -> ValidatedOrder:
    """Sample validated crypto order."""
    return ValidatedOrder(
        sized_order=SizedOrder(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.001,
            dollar_amount=50.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
            environment="crypto",
        ),
        risk_checks_passed=["position_size", "daily_loss"],
    )


@pytest.fixture
def sim_adapter(
    exec_config: SwingRLConfig,
    mock_db: DatabaseManager,
) -> BinanceSimAdapter:
    """BinanceSimAdapter with mocked HTTP calls."""
    return BinanceSimAdapter(config=exec_config, db=mock_db)


class TestBinanceSimProtocol:
    """Verify BinanceSimAdapter satisfies ExchangeAdapter Protocol."""

    def test_satisfies_protocol(self, sim_adapter: BinanceSimAdapter) -> None:
        """BinanceSimAdapter must be recognized as ExchangeAdapter."""
        assert isinstance(sim_adapter, ExchangeAdapter)


class TestMidPriceCalculation:
    """Verify order book mid-price calculation."""

    def test_mid_price(
        self,
        sim_adapter: BinanceSimAdapter,
        order_book_response: dict[str, list[list[str]]],
    ) -> None:
        """Mid-price is average of best bid and best ask."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = order_book_response
        mock_resp.raise_for_status = MagicMock()

        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=mock_resp):
            mid, bid, ask = sim_adapter._get_mid_price("BTCUSDT")

        assert bid == 50000.00
        assert ask == 50010.00
        assert mid == pytest.approx(50005.00)


class TestSlippageAndCommission:
    """Verify slippage application and commission calculation."""

    def test_buy_slippage_applied(
        self,
        sim_adapter: BinanceSimAdapter,
        crypto_order: ValidatedOrder,
        order_book_response: dict[str, list[list[str]]],
    ) -> None:
        """Buy order fill price includes positive slippage."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = order_book_response
        mock_resp.raise_for_status = MagicMock()

        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=mock_resp):
            result = sim_adapter.submit_order(crypto_order)

        # Mid = 50005.0, slippage 0.03% => fill = 50005 * 1.0003 = 50020.0015
        expected_fill = 50005.0 * 1.0003
        assert result.fill_price == pytest.approx(expected_fill, rel=1e-6)

    def test_sell_slippage_applied(
        self,
        sim_adapter: BinanceSimAdapter,
        order_book_response: dict[str, list[list[str]]],
    ) -> None:
        """Sell order fill price includes negative slippage."""
        sell_order = ValidatedOrder(
            sized_order=SizedOrder(
                symbol="BTCUSDT",
                side="sell",
                quantity=0.001,
                dollar_amount=50.0,
                stop_loss_price=55000.0,
                take_profit_price=48000.0,
                environment="crypto",
            ),
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = order_book_response
        mock_resp.raise_for_status = MagicMock()

        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=mock_resp):
            result = sim_adapter.submit_order(sell_order)

        expected_fill = 50005.0 * (1 - 0.0003)
        assert result.fill_price == pytest.approx(expected_fill, rel=1e-6)

    def test_commission_calculation(
        self,
        sim_adapter: BinanceSimAdapter,
        crypto_order: ValidatedOrder,
        order_book_response: dict[str, list[list[str]]],
    ) -> None:
        """Commission is 0.10% of dollar amount."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = order_book_response
        mock_resp.raise_for_status = MagicMock()

        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=mock_resp):
            result = sim_adapter.submit_order(crypto_order)

        # 0.10% of $50 = $0.05
        assert result.commission == pytest.approx(0.05)
        assert result.broker == "binance_sim"


class TestSpreadWarning:
    """Verify wide spread detection."""

    def test_wide_spread_warning(
        self,
        sim_adapter: BinanceSimAdapter,
    ) -> None:
        """Spread > 0.5% logs a warning."""
        wide_book = {
            "bids": [["49000.00", "1.0"]],
            "asks": [["50000.00", "1.0"]],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = wide_book
        mock_resp.raise_for_status = MagicMock()

        with patch("swingrl.execution.adapters.binance_sim.requests.get", return_value=mock_resp):
            mid, _bid, _ask = sim_adapter._get_mid_price("BTCUSDT")

        # Should still return values (warning only, not error)
        assert mid == pytest.approx(49500.0)


class TestRetryOnTimeout:
    """Verify HTTP request retry on failure."""

    def test_retry_on_request_error(
        self,
        sim_adapter: BinanceSimAdapter,
        order_book_response: dict[str, list[list[str]]],
    ) -> None:
        """HTTP request retries on timeout."""
        import requests

        mock_resp = MagicMock()
        mock_resp.json.return_value = order_book_response
        mock_resp.raise_for_status = MagicMock()

        with (
            patch(
                "swingrl.execution.adapters.binance_sim.requests.get",
                side_effect=[
                    requests.exceptions.Timeout("timeout"),
                    mock_resp,
                ],
            ),
            patch("swingrl.execution.adapters.binance_sim.time.sleep"),
        ):
            mid, _, _ = sim_adapter._get_mid_price("BTCUSDT")

        assert mid == pytest.approx(50005.0)
