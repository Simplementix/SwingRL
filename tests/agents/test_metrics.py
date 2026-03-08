"""Tests for swingrl.agents.metrics — performance metric calculators.

TDD RED phase: these tests define expected behavior for all 8 metric functions.
"""

from __future__ import annotations

import math

import numpy as np

from swingrl.agents.metrics import (
    annualized_sharpe,
    avg_drawdown,
    calmar_ratio,
    compute_trade_metrics,
    max_drawdown,
    max_drawdown_duration,
    rachev_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# annualized_sharpe
# ---------------------------------------------------------------------------


class TestAnnualizedSharpe:
    """VAL-03: Sharpe ratio matches hand-calculated values."""

    def test_known_equity_returns(self) -> None:
        """Hand-calculated: mean=0.001, std=0.01, sqrt(252)~15.8745 -> sharpe=1.5875."""
        returns = np.array([0.001] * 10 + [-0.009] * 0)  # constant positive
        # mean=0.001, std=0.0 -> NaN (zero std guard)
        # Use varying returns instead
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
        mean_r = np.mean(returns)  # 0.006
        std_r = np.std(returns, ddof=1)
        expected = (mean_r / std_r) * np.sqrt(252)
        result = annualized_sharpe(returns, periods_per_year=252.0)
        assert abs(result - expected) < 1e-6

    def test_known_crypto_returns(self) -> None:
        """Crypto uses 2191.5 periods per year (6 bars/day * 365.25)."""
        returns = np.array([0.005, -0.003, 0.008, -0.002, 0.004, 0.001])
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        expected = (mean_r / std_r) * np.sqrt(2191.5)
        result = annualized_sharpe(returns, periods_per_year=2191.5)
        assert abs(result - expected) < 1e-6

    def test_with_risk_free_rate(self) -> None:
        """Risk-free rate is subtracted from returns before calculation."""
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
        rf = 0.001
        excess = returns - rf
        mean_r = np.mean(excess)
        std_r = np.std(excess, ddof=1)
        expected = (mean_r / std_r) * np.sqrt(252)
        result = annualized_sharpe(returns, periods_per_year=252.0, risk_free_rate=rf)
        assert abs(result - expected) < 1e-6

    def test_empty_array_returns_nan(self) -> None:
        """Empty returns array -> NaN."""
        result = annualized_sharpe(np.array([]), periods_per_year=252.0)
        assert math.isnan(result)

    def test_single_element_returns_nan(self) -> None:
        """len < 2 -> NaN (can't compute std)."""
        result = annualized_sharpe(np.array([0.01]), periods_per_year=252.0)
        assert math.isnan(result)

    def test_zero_std_returns_nan(self) -> None:
        """All identical returns -> std=0 -> NaN."""
        result = annualized_sharpe(np.array([0.01, 0.01, 0.01]), periods_per_year=252.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    """VAL-03: Sortino ratio uses downside deviation only."""

    def test_known_returns(self) -> None:
        """Only negative returns contribute to downside deviation."""
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
        mean_r = np.mean(returns)
        neg = returns[returns < 0]
        downside_dev = np.sqrt(np.mean(neg**2))
        expected = (mean_r / downside_dev) * np.sqrt(252)
        result = sortino_ratio(returns, periods_per_year=252.0)
        assert abs(result - expected) < 1e-6

    def test_no_negative_returns_nan(self) -> None:
        """All positive returns -> no downside deviation -> NaN."""
        returns = np.array([0.01, 0.02, 0.03])
        result = sortino_ratio(returns, periods_per_year=252.0)
        assert math.isnan(result)

    def test_empty_returns_nan(self) -> None:
        """Empty array -> NaN."""
        result = sortino_ratio(np.array([]), periods_per_year=252.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    """VAL-03: Calmar ratio = annualized return / abs(MDD)."""

    def test_known_returns(self) -> None:
        """Known returns with a clear drawdown."""
        returns = np.array([0.05, -0.10, 0.03, 0.02, 0.01])
        cum = np.cumprod(1 + returns)
        total_return = cum[-1] / cum[0] - 1  # not annualized simple
        n_periods = len(returns)
        ann_return = (1 + total_return) ** (252 / n_periods) - 1
        mdd = max_drawdown(returns)
        expected = ann_return / mdd if mdd > 0 else float("nan")
        result = calmar_ratio(returns, periods_per_year=252.0)
        assert abs(result - expected) < 1e-4

    def test_no_drawdown_returns_nan(self) -> None:
        """Monotonically increasing -> MDD=0 -> NaN."""
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        result = calmar_ratio(returns, periods_per_year=252.0)
        assert math.isnan(result)

    def test_empty_returns_nan(self) -> None:
        """Empty array -> NaN."""
        result = calmar_ratio(np.array([]), periods_per_year=252.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# rachev_ratio
# ---------------------------------------------------------------------------


class TestRachevRatio:
    """VAL-03: Rachev ratio = CVaR(gains) / CVaR(losses)."""

    def test_known_returns(self) -> None:
        """With enough data, top 5% gains vs bottom 5% losses."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 200)
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        n_top = max(1, int(len(gains) * 0.05))
        n_bot = max(1, int(len(losses) * 0.05))
        sorted_gains = np.sort(gains)[::-1][:n_top]
        sorted_losses = np.sort(losses)[:n_bot]
        expected = np.mean(sorted_gains) / abs(np.mean(sorted_losses))
        result = rachev_ratio(returns, alpha=0.05)
        assert abs(result - expected) < 1e-6

    def test_all_positive_returns_nan(self) -> None:
        """No losses -> NaN."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = rachev_ratio(returns, alpha=0.05)
        assert math.isnan(result)

    def test_all_negative_returns_nan(self) -> None:
        """No gains -> NaN."""
        returns = np.array([-0.01, -0.02, -0.03, -0.04, -0.05])
        result = rachev_ratio(returns, alpha=0.05)
        assert math.isnan(result)

    def test_empty_returns_nan(self) -> None:
        """Empty returns -> NaN."""
        result = rachev_ratio(np.array([]), alpha=0.05)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """VAL-04: MDD matches known worst peak-to-trough drop."""

    def test_known_series(self) -> None:
        """Known drawdown: peak at 1.10, trough at 0.95 -> dd = (1.10-0.95)/1.10."""
        returns = np.array([0.10, -0.05, -0.10, 0.03, 0.02])
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = (running_max - cum) / running_max
        expected = np.max(drawdowns)
        result = max_drawdown(returns)
        assert abs(result - expected) < 1e-10

    def test_no_drawdown(self) -> None:
        """Monotonic increase -> MDD = 0."""
        returns = np.array([0.01, 0.02, 0.03])
        result = max_drawdown(returns)
        assert result == 0.0

    def test_empty_returns(self) -> None:
        """Empty array -> 0.0."""
        result = max_drawdown(np.array([]))
        assert result == 0.0


# ---------------------------------------------------------------------------
# avg_drawdown
# ---------------------------------------------------------------------------


class TestAvgDrawdown:
    """VAL-04: Average drawdown from underwater curve."""

    def test_known_series(self) -> None:
        """Average of the underwater (drawdown) curve."""
        returns = np.array([0.10, -0.05, -0.10, 0.03, 0.15])
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = (running_max - cum) / running_max
        expected = np.mean(drawdowns)
        result = avg_drawdown(returns)
        assert abs(result - expected) < 1e-10

    def test_empty_returns(self) -> None:
        """Empty array -> 0.0."""
        result = avg_drawdown(np.array([]))
        assert result == 0.0


# ---------------------------------------------------------------------------
# max_drawdown_duration
# ---------------------------------------------------------------------------


class TestMaxDrawdownDuration:
    """VAL-04: Longest number of bars spent in drawdown."""

    def test_known_series(self) -> None:
        """Returns that create a drawdown lasting 3 bars before recovery."""
        # Start at 1.0, goes up, then drops for 3 bars, then recovers
        returns = np.array([0.10, -0.05, -0.05, -0.02, 0.20])
        # Cumulative: 1.1, 1.045, 0.99275, 0.97290, 1.16748
        # Bars in drawdown: 1.045<1.1, 0.993<1.1, 0.973<1.1 = 3 bars
        # Then recovery at bar 5 (1.167 > 1.1)
        result = max_drawdown_duration(returns)
        assert result == 3

    def test_no_drawdown(self) -> None:
        """Monotonic increase -> duration = 0."""
        returns = np.array([0.01, 0.02, 0.03])
        result = max_drawdown_duration(returns)
        assert result == 0

    def test_empty_returns(self) -> None:
        """Empty array -> 0."""
        result = max_drawdown_duration(np.array([]))
        assert result == 0

    def test_never_recovers(self) -> None:
        """Drawdown that never recovers counts until end."""
        returns = np.array([0.10, -0.05, -0.05, -0.05])
        # Cumulative: 1.1, 1.045, 0.99275, 0.94311
        # Bars in drawdown from peak: 3 bars (never recovers)
        result = max_drawdown_duration(returns)
        assert result == 3


# ---------------------------------------------------------------------------
# compute_trade_metrics
# ---------------------------------------------------------------------------


class TestComputeTradeMetrics:
    """VAL-06: Trade metrics from trade list."""

    def test_known_trades(self) -> None:
        """Hand-calculated: 3 wins, 2 losses."""
        trades = [
            {"pnl": 100.0},
            {"pnl": -50.0},
            {"pnl": 200.0},
            {"pnl": -30.0},
            {"pnl": 150.0},
        ]
        total_bars = 100
        bars_per_week = 5.0
        result = compute_trade_metrics(trades, total_bars, bars_per_week)
        assert result["total_trades"] == 5
        assert abs(result["win_rate"] - 0.6) < 1e-10
        # PF = (100+200+150) / abs(-50-30) = 450/80 = 5.625
        assert abs(result["profit_factor"] - 5.625) < 1e-10
        assert abs(result["avg_win"] - 150.0) < 1e-10
        assert abs(result["avg_loss"] - (-40.0)) < 1e-10
        # trade_frequency = 5 trades / (100 bars / 5 bars_per_week) = 5/20 = 0.25
        assert abs(result["trade_frequency_per_week"] - 0.25) < 1e-10

    def test_no_losses_pf_inf(self) -> None:
        """No losing trades -> profit factor = inf."""
        trades = [{"pnl": 100.0}, {"pnl": 50.0}]
        result = compute_trade_metrics(trades, total_bars=50, bars_per_week=5.0)
        assert result["profit_factor"] == float("inf")

    def test_no_wins(self) -> None:
        """No winning trades -> profit factor = 0, win_rate = 0."""
        trades = [{"pnl": -100.0}, {"pnl": -50.0}]
        result = compute_trade_metrics(trades, total_bars=50, bars_per_week=5.0)
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0

    def test_empty_trades(self) -> None:
        """Empty trade list -> sensible defaults."""
        result = compute_trade_metrics([], total_bars=100, bars_per_week=5.0)
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["trade_frequency_per_week"] == 0.0

    def test_single_winning_trade(self) -> None:
        """Single trade that wins."""
        trades = [{"pnl": 50.0}]
        result = compute_trade_metrics(trades, total_bars=20, bars_per_week=5.0)
        assert result["total_trades"] == 1
        assert result["win_rate"] == 1.0
        assert result["profit_factor"] == float("inf")
        assert abs(result["avg_win"] - 50.0) < 1e-10
        assert result["avg_loss"] == 0.0

    def test_single_losing_trade(self) -> None:
        """Single trade that loses."""
        trades = [{"pnl": -30.0}]
        result = compute_trade_metrics(trades, total_bars=20, bars_per_week=5.0)
        assert result["total_trades"] == 1
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["avg_win"] == 0.0
        assert abs(result["avg_loss"] - (-30.0)) < 1e-10
