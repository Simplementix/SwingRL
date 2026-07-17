"""Sim-fidelity tests for the approved Task 13 fix bundle (D1/D4/D8/D9).

RED-first (TDD). These assert the *target* behavior from docs/execution/sim-fidelity.md:

- D1: crypto fills cross the book — buy pays best ask, sell receives best bid (not mid ±
  a constant); the recorded slippage becomes the real half-spread, not a tautological 0.03%.
- D4: commission is charged on the executed fill notional in both submit_order and
  emergency_sell (consistent basis).
- D8: a spread wider than the hard reject threshold raises BrokerError instead of filling at
  mid; a spread between the 0.5% warn band and the reject threshold still fills.
- D9: fill_quality.expected_cost_frac is derived from the sim's own commission + slippage
  constants (single source of truth), not the config round-trip figure; equity stays 0.0.

No DATABASE_URL: submit_order's happy path never touches the DB, and the FillProcessor D9
tests use the make_mock_db MagicMock helper (Task 10 precedent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from swingrl.execution.adapters.binance_sim import (
    _COMMISSION_RATE,
    _DEFAULT_SLIPPAGE,
    BinanceSimAdapter,
)
from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.types import FillResult, SizedOrder, ValidatedOrder
from swingrl.utils.exceptions import BrokerError
from tests.conftest import make_mock_db

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig

# Modeled per-fill crypto cost = commission + baseline slippage (the sim's own constants).
_SIM_COST_FRAC = _COMMISSION_RATE + _DEFAULT_SLIPPAGE  # 0.001 + 0.0003 = 0.0013

# Tight book: bid 50000 / ask 50010 -> mid 50005, spread 0.02% (< 0.5% warn).
_TIGHT_BOOK = {"bids": [["50000.00", "1.5"]], "asks": [["50010.00", "1.0"]]}
_TIGHT_MID = 50005.0
_TIGHT_BID = 50000.0
_TIGHT_ASK = 50010.0

# Moderate book: spread 0.7% -> above the 0.5% warn band, below the 1.0% reject threshold.
_MODERATE_BOOK = {"bids": [["49825.00", "1.0"]], "asks": [["50175.00", "1.0"]]}
# Wide book: bid 49000 / ask 50000 -> mid 49500, spread ~2.02% (> 1.0% reject threshold).
_WIDE_BOOK = {"bids": [["49000.00", "1.0"]], "asks": [["50000.00", "1.0"]]}

_QTY = 0.001


def _resp(book: dict[str, list[list[str]]]) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = book
    mock.raise_for_status = MagicMock()
    return mock


def _adapter(exec_config: SwingRLConfig) -> BinanceSimAdapter:
    db, _conn = make_mock_db(fetchone_returns=[None])
    return BinanceSimAdapter(config=exec_config, db=db)


def _buy(symbol: str = "BTCUSDT") -> ValidatedOrder:
    return ValidatedOrder(
        order=SizedOrder(
            symbol=symbol,
            side="buy",
            quantity=_QTY,
            dollar_amount=50.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
            environment="crypto",
        ),
    )


def _sell(symbol: str = "BTCUSDT") -> ValidatedOrder:
    return ValidatedOrder(
        order=SizedOrder(
            symbol=symbol,
            side="sell",
            quantity=_QTY,
            dollar_amount=50.0,
            stop_loss_price=55000.0,
            take_profit_price=48000.0,
            environment="crypto",
        ),
    )


class TestD1FillCrossesTheBook:
    """D1: fills cross the real spread instead of mid ± a constant."""

    def test_buy_fills_at_best_ask(self, exec_config: SwingRLConfig) -> None:
        """A buy pays the best ask, not mid × (1 + 0.0003)."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_TIGHT_BOOK)
        ):
            result = _adapter(exec_config).submit_order(_buy())
        assert result.fill_price == pytest.approx(_TIGHT_ASK)

    def test_sell_fills_at_best_bid(self, exec_config: SwingRLConfig) -> None:
        """A sell receives the best bid, not mid × (1 − 0.0003)."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_TIGHT_BOOK)
        ):
            result = _adapter(exec_config).submit_order(_sell())
        assert result.fill_price == pytest.approx(_TIGHT_BID)

    def test_recorded_slippage_is_real_half_spread(self, exec_config: SwingRLConfig) -> None:
        """Recorded slippage = |fill − mid| × qty = the real half-spread, not a constant 0.03%."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_TIGHT_BOOK)
        ):
            result = _adapter(exec_config).submit_order(_buy())
        assert result.slippage == pytest.approx((_TIGHT_ASK - _TIGHT_MID) * _QTY)


class TestD4CommissionOnFillNotional:
    """D4: commission is charged on the executed fill notional in both paths."""

    def test_submit_commission_on_fill_notional(self, exec_config: SwingRLConfig) -> None:
        """Buy commission = best_ask × qty × rate (fill notional), not decision dollar_amount."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_TIGHT_BOOK)
        ):
            result = _adapter(exec_config).submit_order(_buy())
        assert result.commission == pytest.approx(_TIGHT_ASK * _QTY * _COMMISSION_RATE)

    def test_emergency_sell_commission_on_fill_notional(self, exec_config: SwingRLConfig) -> None:
        """Emergency sell fills at best bid and charges commission on that fill notional."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_TIGHT_BOOK)
        ):
            fill = _adapter(exec_config).emergency_sell("BTCUSDT", _QTY)
        assert fill.fill_price == pytest.approx(_TIGHT_BID)
        assert fill.commission == pytest.approx(_TIGHT_BID * _QTY * _COMMISSION_RATE)


class TestD8HardSpreadReject:
    """D8: a spread past the reject threshold refuses the order; below it still fills."""

    def test_wide_spread_rejects_order(self, exec_config: SwingRLConfig) -> None:
        """A ~2% spread raises BrokerError (no fantasy mid-fill, no FillResult)."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_WIDE_BOOK)
        ):
            with pytest.raises(BrokerError):
                _adapter(exec_config).submit_order(_buy())

    def test_moderate_spread_still_fills(self, exec_config: SwingRLConfig) -> None:
        """A 0.7% spread (above warn, below reject) still fills — reject is not over-eager."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get",
            return_value=_resp(_MODERATE_BOOK),
        ):
            result = _adapter(exec_config).submit_order(_buy())
        assert result.status == "filled"

    def test_emergency_sell_not_blocked_by_wide_spread(self, exec_config: SwingRLConfig) -> None:
        """Forced exits must never be blocked: emergency_sell fills even on a wide spread."""
        with patch(
            "swingrl.execution.adapters.binance_sim.requests.get", return_value=_resp(_WIDE_BOOK)
        ):
            fill = _adapter(exec_config).emergency_sell("BTCUSDT", _QTY)
        assert fill.status == "filled"
        assert fill.fill_price == pytest.approx(49000.0)  # best bid of the wide book


def _crypto_buy_fill() -> FillResult:
    return FillResult(
        trade_id="fid-crypto-buy",
        symbol="BTCUSDT",
        side="buy",
        quantity=0.001,
        fill_price=50100.0,
        commission=0.05,
        slippage=0.0,
        environment="crypto",
        broker="binance_us",
        status="filled",
    )


def _equity_buy_fill() -> FillResult:
    return FillResult(
        trade_id="fid-equity-buy",
        symbol="SPY",
        side="buy",
        quantity=10.0,
        fill_price=105.0,
        commission=0.0,
        slippage=0.0,
        environment="equity",
        broker="alpaca",
        status="filled",
    )


class TestD9ExpectedCostSingleSource:
    """D9: expected_cost_frac is sim-constant-derived (not the config round-trip figure)."""

    def test_crypto_expected_cost_frac_is_sim_derived_not_config(
        self, exec_config: SwingRLConfig
    ) -> None:
        """Crypto expected_cost_frac == commission + baseline slippage constants, not 0.0022."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        processor.process(_crypto_buy_fill(), decision_price=50000.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][5] == pytest.approx(_SIM_COST_FRAC)
        assert fq_call.args[1][5] != pytest.approx(
            exec_config.environment.crypto_transaction_cost_pct
        )

    def test_crypto_expected_fill_price_uses_sim_derived_cost(
        self, exec_config: SwingRLConfig
    ) -> None:
        """expected_fill_price_usd applies the sim-derived cost side-aware to decision price."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        processor.process(_crypto_buy_fill(), decision_price=50000.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][2] == str(round(50000.0 * (1 + _SIM_COST_FRAC), 8))

    def test_equity_expected_cost_frac_still_zero(self, exec_config: SwingRLConfig) -> None:
        """Equity is commission-free per the Task 10 contract — unchanged by D9."""
        db, conn = make_mock_db(fetchone_returns=[None])
        processor = FillProcessor(db=db, config=exec_config)

        processor.process(_equity_buy_fill(), decision_price=100.0)

        fq_call = next(
            c for c in conn.execute.call_args_list if "INSERT INTO fill_quality" in c.args[0]
        )
        assert fq_call.args[1][5] == pytest.approx(0.0)
