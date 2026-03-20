"""Performance metric calculators for RL agent evaluation.

Pure-math functions computing risk-adjusted returns, drawdown statistics,
and trade-level metrics. All functions are stateless and operate on numpy
arrays of period returns.

Equity uses periods_per_year=252 (trading days).
Crypto uses periods_per_year=2191.5 (6 four-hour bars * 365.25 days).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of period returns (e.g. daily or 4H).
        periods_per_year: Annualization factor (252 equity, 2191.5 crypto).
        risk_free_rate: Per-period risk-free rate to subtract.

    Returns:
        Annualized Sharpe ratio, or NaN if len<2 or zero standard deviation.
    """
    if len(returns) < 2:
        return float("nan")

    excess = returns - risk_free_rate
    mean_r = np.mean(excess)
    std_r = np.std(excess, ddof=1)

    if std_r < 1e-10:
        return float("nan")

    return float((mean_r / std_r) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio using downside deviation.

    Args:
        returns: Array of period returns.
        periods_per_year: Annualization factor.
        risk_free_rate: Per-period risk-free rate to subtract.

    Returns:
        Annualized Sortino ratio, or NaN if no downside deviation.
    """
    if len(returns) < 2:
        return float("nan")

    excess = returns - risk_free_rate
    downside = np.minimum(excess, 0.0)
    downside_var = float(np.mean(downside**2))
    mean_excess = float(np.mean(excess))

    if downside_var == 0.0:
        if mean_excess > 0.0:
            return 999.0
        return 0.0

    downside_dev = float(np.sqrt(downside_var))
    raw = float((mean_excess / downside_dev) * np.sqrt(periods_per_year))
    return min(raw, 999.0)


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: float,
) -> float:
    """Compute Calmar ratio: annualized return / max drawdown.

    Args:
        returns: Array of period returns.
        periods_per_year: Annualization factor.

    Returns:
        Calmar ratio, or NaN if max drawdown is zero or returns empty.
    """
    if len(returns) == 0:
        return float("nan")

    cum = np.cumprod(1 + returns)
    total_return = float(cum[-1] - 1.0)
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    mdd = max_drawdown(returns)
    if mdd < 1e-10:
        return float("nan")

    return float(ann_return / mdd)


def rachev_ratio(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute Rachev ratio: CVaR of top-alpha gains / CVaR of bottom-alpha losses.

    Args:
        returns: Array of period returns.
        alpha: Tail percentile (default 5%).

    Returns:
        Rachev ratio, or NaN if no gains or no losses.
    """
    if len(returns) == 0:
        return float("nan")

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    if len(gains) == 0 or len(losses) == 0:
        return float("nan")

    n_top = max(1, int(len(gains) * alpha))
    n_bot = max(1, int(len(losses) * alpha))

    sorted_gains = np.sort(gains)[::-1][:n_top]
    sorted_losses = np.sort(losses)[:n_bot]

    cvar_gains = float(np.mean(sorted_gains))
    cvar_losses = float(abs(np.mean(sorted_losses)))

    if cvar_losses < 1e-10:
        return float("nan")

    return cvar_gains / cvar_losses


def max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown as a positive fraction.

    Args:
        returns: Array of period returns.

    Returns:
        Maximum peak-to-trough decline (0.15 = 15% drawdown). Returns 0.0 if empty.
    """
    if len(returns) == 0:
        return 0.0

    # Prepend 1.0 so drawdown from initial investment is captured
    cum = np.concatenate([[1.0], np.cumprod(1 + returns)])
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max

    return float(np.max(drawdowns))


def avg_drawdown(returns: np.ndarray) -> float:
    """Compute average drawdown from the underwater curve.

    Args:
        returns: Array of period returns.

    Returns:
        Mean of all drawdown values. Returns 0.0 if empty.
    """
    if len(returns) == 0:
        return 0.0

    # Prepend 1.0 so drawdown from initial investment is captured
    cum = np.concatenate([[1.0], np.cumprod(1 + returns)])
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max

    return float(np.mean(drawdowns))


def max_drawdown_duration(returns: np.ndarray) -> int:
    """Compute longest number of bars spent in drawdown before recovery.

    Args:
        returns: Array of period returns.

    Returns:
        Number of bars in longest drawdown period. Returns 0 if empty.
    """
    if len(returns) == 0:
        return 0

    # Prepend 1.0 so drawdown from initial investment is captured
    cum = np.concatenate([[1.0], np.cumprod(1 + returns)])
    running_max = np.maximum.accumulate(cum)

    longest = 0
    current = 0

    for i in range(len(cum)):
        if cum[i] < running_max[i]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def compute_trade_metrics(
    trades: list[dict[str, Any]],
    total_bars: int,
    bars_per_week: float,
) -> dict[str, float]:
    """Compute trade-level performance metrics from a list of trades.

    Args:
        trades: List of trade dicts, each with a 'pnl' float key.
        total_bars: Total number of bars in the evaluation period.
        bars_per_week: Number of bars per week (5 equity, 42 crypto).

    Returns:
        Dict with win_rate, profit_factor, total_trades, avg_win, avg_loss,
        trade_frequency_per_week.
    """
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "trade_frequency_per_week": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / total_trades
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))

    if sum_losses < 1e-10:
        profit_factor = float("inf") if sum_wins > 0 else 0.0
    else:
        profit_factor = sum_wins / sum_losses

    weeks = total_bars / bars_per_week if bars_per_week > 0 else 0
    trade_frequency = total_trades / weeks if weeks > 0 else 0.0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trade_frequency_per_week": trade_frequency,
    }
