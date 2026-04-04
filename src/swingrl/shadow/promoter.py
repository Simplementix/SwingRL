"""Shadow model auto-promotion logic.

Evaluates 4 criteria after minimum evaluation period:
1. Shadow annualized Sharpe > Active annualized Sharpe
2. Shadow MDD <= mdd_tolerance_ratio * Active MDD
3. Shadow profit factor > 1.5
4. No circuit breaker triggers during shadow period

On success: promotes shadow to active via lifecycle.
On failure: archives shadow model with Discord alert.

Usage:
    from swingrl.shadow.promoter import evaluate_shadow_promotion
    promoted = evaluate_shadow_promotion(config, db, "equity", lifecycle, alerter)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import structlog

from swingrl.agents.metrics import annualized_sharpe, max_drawdown

log = structlog.get_logger(__name__)

# Annualization factors matching training gate framework
_PERIODS_PER_YEAR: dict[str, float] = {
    "equity": 252.0,
    "crypto": 2191.5,
}

# Minimum profit factor threshold (matches training gate)
_MIN_PROFIT_FACTOR = 1.5


def evaluate_shadow_promotion(
    config: Any,
    db: Any,
    env_name: str,
    lifecycle: Any,
    alerter: Any,
) -> bool:
    """Evaluate whether a shadow model should be promoted to active.

    Returns False immediately if fewer than the minimum evaluation cycles
    have been completed. Otherwise evaluates 4 criteria and either promotes
    or archives the shadow model.

    Args:
        config: SwingRLConfig with shadow settings.
        db: DatabaseManager instance.
        env_name: Environment name (equity or crypto).
        lifecycle: ModelLifecycle instance for promote/archive.
        alerter: Alerter instance for Discord notifications.

    Returns:
        True if shadow model was promoted, False otherwise.
    """
    # Determine minimum eval threshold
    if env_name == "equity":
        min_trades = config.shadow.equity_eval_days
    else:
        min_trades = config.shadow.crypto_eval_cycles

    # Count shadow trades and find earliest shadow trade timestamp
    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt, MIN(timestamp) as earliest "
            "FROM shadow_trades WHERE environment = %s",
            [env_name],
        ).fetchone()
        shadow_count = row["cnt"] if row else 0
        shadow_start = row["earliest"] if row else None

    if shadow_count < min_trades:
        log.info(
            "shadow_eval_insufficient_data",
            env=env_name,
            shadow_trades=shadow_count,
            min_required=min_trades,
        )
        return False

    # Compute metrics using portfolio snapshots for returns, trades for profit factor
    periods_per_year = _PERIODS_PER_YEAR.get(env_name, 252.0)

    shadow_returns = _get_portfolio_returns(db, env_name, source="shadow")
    active_returns = _get_portfolio_returns(db, env_name, source="active")

    shadow_sharpe = _safe_annualized_sharpe(shadow_returns, periods_per_year)
    active_sharpe = _safe_annualized_sharpe(active_returns, periods_per_year)
    shadow_mdd = _safe_max_drawdown(shadow_returns)
    active_mdd = _safe_max_drawdown(active_returns)
    shadow_pf = _compute_profit_factor(db, "shadow_trades", env_name)
    cb_triggered = _check_cb_during_shadow(db, env_name, shadow_start)

    mdd_tolerance = config.shadow.mdd_tolerance_ratio

    # Log comparison
    log.info(
        "shadow_eval_metrics",
        env=env_name,
        shadow_sharpe=round(shadow_sharpe, 4),
        active_sharpe=round(active_sharpe, 4),
        shadow_mdd=round(shadow_mdd, 4),
        active_mdd=round(active_mdd, 4),
        shadow_pf=round(shadow_pf, 4),
        mdd_tolerance=mdd_tolerance,
        cb_triggered=cb_triggered,
    )

    # Evaluate 4 criteria
    sharpe_ok = shadow_sharpe > active_sharpe
    mdd_ok = shadow_mdd <= mdd_tolerance * active_mdd if active_mdd > 0 else shadow_mdd <= 0
    pf_ok = shadow_pf > _MIN_PROFIT_FACTOR
    no_cb = not cb_triggered

    all_pass = sharpe_ok and mdd_ok and pf_ok and no_cb

    if all_pass and config.shadow.auto_promote:
        # Promote shadow to active
        lifecycle.promote(env_name)
        alerter.send_alert(
            "info",
            f"Shadow Model Promoted: {env_name.title()}",
            f"Shadow model promoted to active.\n"
            f"Shadow Sharpe: {shadow_sharpe:.4f} vs Active: {active_sharpe:.4f}\n"
            f"Shadow MDD: {shadow_mdd:.4f} vs Active: {active_mdd:.4f}\n"
            f"Shadow PF: {shadow_pf:.4f}\n"
            f"CB triggers: {cb_triggered}",
        )
        log.info("shadow_promoted", env=env_name)
        return True

    # Failed evaluation -- archive shadow model
    lifecycle.archive_shadow(env_name)
    reasons = []
    if not sharpe_ok:
        reasons.append(f"Sharpe: shadow {shadow_sharpe:.4f} <= active {active_sharpe:.4f}")
    if not mdd_ok:
        reasons.append(
            f"MDD: shadow {shadow_mdd:.4f} > {mdd_tolerance:.0%} of active {active_mdd:.4f}"
        )
    if not pf_ok:
        reasons.append(f"Profit factor: shadow {shadow_pf:.4f} <= {_MIN_PROFIT_FACTOR}")
    if not no_cb:
        reasons.append("Circuit breaker triggered during shadow period")

    alerter.send_alert(
        "warning",
        f"Shadow Model Failed: {env_name.title()}",
        f"Shadow model archived after failing evaluation.\n"
        f"Reasons: {'; '.join(reasons)}\n"
        f"Shadow Sharpe: {shadow_sharpe:.4f} vs Active: {active_sharpe:.4f}\n"
        f"Shadow MDD: {shadow_mdd:.4f} vs Active: {active_mdd:.4f}\n"
        f"Shadow PF: {shadow_pf:.4f}",
    )
    log.warning("shadow_failed", env=env_name, reasons=reasons)
    return False


def _get_portfolio_returns(db: Any, env_name: str, source: str) -> np.ndarray:
    """Compute period returns from portfolio value series.

    For active models, uses portfolio_snapshots table (total_value column).
    For shadow models, uses shadow_trades to build a synthetic portfolio value
    series from paired buy/sell trades.

    Args:
        db: DatabaseManager instance.
        env_name: Environment name (equity or crypto).
        source: Either "active" or "shadow".

    Returns:
        Numpy array of period returns (may be empty).
    """
    if source == "active":
        return _returns_from_portfolio_snapshots(db, env_name)
    return _returns_from_shadow_trades(db, env_name)


def _returns_from_portfolio_snapshots(db: Any, env_name: str) -> np.ndarray:
    """Compute returns from portfolio_snapshots total_value series.

    Args:
        db: DatabaseManager instance.
        env_name: Environment name.

    Returns:
        Numpy array of period returns.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT total_value FROM portfolio_snapshots WHERE environment = %s ORDER BY timestamp",
            [env_name],
        ).fetchall()

    if len(rows) < 2:
        return np.array([], dtype=np.float64)

    values = np.array([float(row["total_value"]) for row in rows], dtype=np.float64)
    # Filter out zero/negative values to avoid division errors
    valid_mask = values > 0
    if np.sum(valid_mask) < 2:
        return np.array([], dtype=np.float64)

    values = values[valid_mask]
    returns: np.ndarray = np.diff(values) / values[:-1]
    return returns


def _returns_from_shadow_trades(db: Any, env_name: str) -> np.ndarray:
    """Compute returns from shadow trade pairs (buy then sell per symbol).

    Pairs sequential trades per symbol to compute per-trade returns.
    This approximates portfolio returns for the shadow model which has
    no portfolio_snapshots entries.

    Args:
        db: DatabaseManager instance.
        env_name: Environment name.

    Returns:
        Numpy array of per-trade returns sorted by timestamp.
    """
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT symbol, side, price, quantity, timestamp "
            "FROM shadow_trades WHERE environment = %s ORDER BY timestamp",
            [env_name],
        ).fetchall()

    if len(rows) < 2:
        return np.array([], dtype=np.float64)

    # Pair sequential trades per symbol: buy→sell or sell→buy
    open_positions: dict[str, tuple[str, float, float, str]] = {}  # symbol → (side, price, qty, ts)
    trade_returns: list[tuple[str, float]] = []  # (timestamp, return)

    for row in rows:
        symbol = row["symbol"]
        side = row["side"]
        price = float(row["price"])
        qty = float(row["quantity"])
        ts = row["timestamp"]

        if symbol not in open_positions:
            open_positions[symbol] = (side, price, qty, ts)
            continue

        open_side, open_price, _open_qty, _open_ts = open_positions[symbol]

        if open_price > 0:
            if open_side == "buy":
                ret = (price - open_price) / open_price
            else:
                ret = (open_price - price) / open_price
            trade_returns.append((ts, ret))

        del open_positions[symbol]

    if not trade_returns:
        return np.array([], dtype=np.float64)

    # Sort by timestamp and return the returns
    trade_returns.sort(key=lambda x: x[0])
    return np.array([r for _, r in trade_returns], dtype=np.float64)


def _safe_annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Compute annualized Sharpe, returning 0.0 for insufficient data.

    Args:
        returns: Array of period returns.
        periods_per_year: Annualization factor.

    Returns:
        Annualized Sharpe ratio, or 0.0 if insufficient data or NaN result.
    """
    if len(returns) < 2:
        return 0.0
    result = annualized_sharpe(returns, periods_per_year)
    return 0.0 if math.isnan(result) else result


def _safe_max_drawdown(returns: np.ndarray) -> float:
    """Compute max drawdown, returning 0.0 for insufficient data.

    Args:
        returns: Array of period returns.

    Returns:
        Maximum drawdown as positive fraction.
    """
    if len(returns) == 0:
        return 0.0
    return max_drawdown(returns)


def _compute_profit_factor(db: Any, table: str, env_name: str) -> float:
    """Compute profit factor from paired trades in the given table.

    Pairs sequential buy/sell trades per symbol and computes gross wins / gross losses.

    Args:
        db: DatabaseManager instance.
        table: Table name (shadow_trades or trades).
        env_name: Environment name filter.

    Returns:
        Profit factor (gross wins / gross losses). Returns 0.0 if no completed trades.
    """
    with db.connection() as conn:
        rows = conn.execute(
            f"SELECT symbol, side, price, quantity FROM {table} "  # noqa: S608
            f"WHERE environment = %s ORDER BY timestamp",  # nosec B608
            [env_name],
        ).fetchall()

    if len(rows) < 2:
        return 0.0

    # Pair trades per symbol
    open_positions: dict[str, tuple[str, float, float]] = {}
    gross_wins = 0.0
    gross_losses = 0.0

    for row in rows:
        symbol = row["symbol"]
        side = row["side"]
        price = float(row["price"])
        qty = float(row["quantity"])

        if symbol not in open_positions:
            open_positions[symbol] = (side, price, qty)
            continue

        open_side, open_price, open_qty = open_positions[symbol]

        if open_price > 0:
            if open_side == "buy":
                pnl = (price - open_price) * min(qty, open_qty)
            else:
                pnl = (open_price - price) * min(qty, open_qty)

            if pnl > 0:
                gross_wins += pnl
            else:
                gross_losses += abs(pnl)

        del open_positions[symbol]

    if gross_losses < 1e-10:
        return float("inf") if gross_wins > 0 else 0.0

    return gross_wins / gross_losses


def _check_cb_during_shadow(
    db: Any,
    env_name: str,
    shadow_start: str | None,
) -> bool:
    """Check if any circuit breaker events occurred during shadow period.

    Only counts events that occurred on or after the shadow evaluation start date.

    Args:
        db: DatabaseManager instance.
        env_name: Environment name.
        shadow_start: ISO timestamp of earliest shadow trade (None if no trades).

    Returns:
        True if any CB events found during the shadow period.
    """
    if shadow_start is None:
        return False

    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM circuit_breaker_events "
            "WHERE environment = %s AND triggered_at >= %s",
            [env_name, shadow_start],
        ).fetchone()

    return (row["cnt"] if row else 0) > 0
