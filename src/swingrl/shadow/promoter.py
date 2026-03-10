"""Shadow model auto-promotion logic.

Evaluates 3 criteria after minimum evaluation period:
1. Shadow Sharpe >= Active Sharpe
2. Shadow MDD <= mdd_tolerance_ratio * Active MDD
3. No circuit breaker triggers during shadow period

On success: promotes shadow to active via lifecycle.
On failure: archives shadow model with Discord alert.

Usage:
    from swingrl.shadow.promoter import evaluate_shadow_promotion
    promoted = evaluate_shadow_promotion(config, db, "equity", lifecycle, alerter)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


def evaluate_shadow_promotion(
    config: Any,
    db: Any,
    env_name: str,
    lifecycle: Any,
    alerter: Any,
) -> bool:
    """Evaluate whether a shadow model should be promoted to active.

    Returns False immediately if fewer than the minimum evaluation cycles
    have been completed. Otherwise evaluates 3 criteria and either promotes
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

    # Count shadow trades
    with db.sqlite() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM shadow_trades WHERE environment = ?",
            (env_name,),
        ).fetchone()
        shadow_count = row["cnt"] if row else 0

    if shadow_count < min_trades:
        log.info(
            "shadow_eval_insufficient_data",
            env=env_name,
            shadow_trades=shadow_count,
            min_required=min_trades,
        )
        return False

    # Compute metrics for both shadow and active
    shadow_sharpe = _compute_sharpe(db, "shadow_trades", env_name)
    active_sharpe = _compute_sharpe(db, "trades", env_name)
    shadow_mdd = _compute_mdd(db, "shadow_trades", env_name)
    active_mdd = _compute_mdd(db, "trades", env_name)
    cb_triggered = _check_cb_during_shadow(db, env_name)

    mdd_tolerance = config.shadow.mdd_tolerance_ratio

    # Log comparison
    log.info(
        "shadow_eval_metrics",
        env=env_name,
        shadow_sharpe=round(shadow_sharpe, 4),
        active_sharpe=round(active_sharpe, 4),
        shadow_mdd=round(shadow_mdd, 4),
        active_mdd=round(active_mdd, 4),
        mdd_tolerance=mdd_tolerance,
        cb_triggered=cb_triggered,
    )

    # Evaluate 3 criteria
    sharpe_ok = shadow_sharpe >= active_sharpe
    mdd_ok = shadow_mdd <= mdd_tolerance * active_mdd if active_mdd > 0 else shadow_mdd <= 0
    no_cb = not cb_triggered

    all_pass = sharpe_ok and mdd_ok and no_cb

    if all_pass and config.shadow.auto_promote:
        # Promote shadow to active
        lifecycle.promote(env_name)
        alerter.send_alert(
            "info",
            f"Shadow Model Promoted: {env_name.title()}",
            f"Shadow model promoted to active.\n"
            f"Shadow Sharpe: {shadow_sharpe:.4f} vs Active: {active_sharpe:.4f}\n"
            f"Shadow MDD: {shadow_mdd:.4f} vs Active: {active_mdd:.4f}\n"
            f"CB triggers: {cb_triggered}",
        )
        log.info("shadow_promoted", env=env_name)
        return True

    # Failed evaluation -- archive shadow model
    lifecycle.archive_shadow(env_name)
    reasons = []
    if not sharpe_ok:
        reasons.append(f"Sharpe: shadow {shadow_sharpe:.4f} < active {active_sharpe:.4f}")
    if not mdd_ok:
        reasons.append(
            f"MDD: shadow {shadow_mdd:.4f} > {mdd_tolerance:.0%} of active {active_mdd:.4f}"
        )
    if not no_cb:
        reasons.append("Circuit breaker triggered during shadow period")

    alerter.send_alert(
        "warning",
        f"Shadow Model Failed: {env_name.title()}",
        f"Shadow model archived after failing evaluation.\n"
        f"Reasons: {'; '.join(reasons)}\n"
        f"Shadow Sharpe: {shadow_sharpe:.4f} vs Active: {active_sharpe:.4f}\n"
        f"Shadow MDD: {shadow_mdd:.4f} vs Active: {active_mdd:.4f}",
    )
    log.warning("shadow_failed", env=env_name, reasons=reasons)
    return False


def _compute_sharpe(db: Any, table: str, env_name: str) -> float:
    """Compute Sharpe ratio from trade returns in the given table.

    Uses simple price-based returns from sequential trades.

    Args:
        db: DatabaseManager instance.
        table: Table name (shadow_trades or trades).
        env_name: Environment name filter.

    Returns:
        Sharpe ratio (0.0 if insufficient data).
    """
    with db.sqlite() as conn:
        rows = conn.execute(
            f"SELECT price FROM {table} WHERE environment = ? ORDER BY timestamp",  # noqa: S608  # nosec B608
            (env_name,),
        ).fetchall()

    if len(rows) < 2:
        return 0.0

    prices = [row["price"] for row in rows]
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

    if not returns:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    if len(returns) < 2:
        return 0.0

    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0

    if std_ret == 0:
        return 0.0

    return float(mean_ret / std_ret)


def _compute_mdd(db: Any, table: str, env_name: str) -> float:
    """Compute maximum drawdown from trade prices.

    Args:
        db: DatabaseManager instance.
        table: Table name (shadow_trades or trades).
        env_name: Environment name filter.

    Returns:
        Maximum drawdown as positive fraction (e.g., 0.15 = 15%).
    """
    with db.sqlite() as conn:
        rows = conn.execute(
            f"SELECT price FROM {table} WHERE environment = ? ORDER BY timestamp",  # noqa: S608  # nosec B608
            (env_name,),
        ).fetchall()

    if len(rows) < 2:
        return 0.0

    prices = [row["price"] for row in rows]
    peak = prices[0]
    max_dd = 0.0

    for price in prices[1:]:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def _check_cb_during_shadow(db: Any, env_name: str) -> bool:
    """Check if any circuit breaker events occurred during shadow period.

    Args:
        db: DatabaseManager instance.
        env_name: Environment name.

    Returns:
        True if any CB events found for the environment.
    """
    with db.sqlite() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM circuit_breaker_events WHERE environment = ?",
            (env_name,),
        ).fetchone()

    return (row["cnt"] if row else 0) > 0
