"""Walk-forward backtester with growing-window fold generation.

Generates non-overlapping folds with purge gaps (embargo), retrains
per fold via TrainingOrchestrator, evaluates on held-out test data,
and stores results to DuckDB.

Usage:
    from swingrl.agents.backtest import WalkForwardBacktester, generate_folds
    backtester = WalkForwardBacktester(config=config, db=db)
    results = backtester.run("equity", "ppo", features, prices, models_dir)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

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
from swingrl.agents.validation import GateResult, check_validation_gates, diagnose_overfitting

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Per-environment constants
ENV_PARAMS: dict[str, dict[str, int | float]] = {
    "equity": {
        "test_bars": 63,
        "min_train_bars": 252,
        "embargo_bars": 10,
        "periods_per_year": 252,
        "bars_per_week": 5,
    },
    "crypto": {
        "test_bars": 540,
        "min_train_bars": 2190,
        "embargo_bars": 130,
        "periods_per_year": 2191.5,
        "bars_per_week": 42,
    },
}


@dataclass
class FoldResult:
    """Result of a single walk-forward fold evaluation.

    Args:
        fold_number: Sequential fold index (0-based).
        train_range: Tuple of (start_idx, end_idx) for training data.
        test_range: Tuple of (start_idx, end_idx) for test data.
        in_sample_metrics: Metric dict from training evaluation.
        out_of_sample_metrics: Metric dict from test evaluation.
        trades: List of trade dicts from test evaluation.
        gate_result: Validation gate pass/fail result.
        overfitting: Overfitting diagnosis dict.
    """

    fold_number: int
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    in_sample_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]
    trades: list[dict[str, Any]]
    gate_result: GateResult
    overfitting: dict[str, Any]


def _reconstruct_round_trips(
    trade_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct round-trip trades from portfolio trade log using FIFO matching.

    A round-trip starts with a buy and closes (fully or partially) with a sell.
    PnL = sell_value - buy_value - total_costs. Each trade_log entry must have
    'asset_idx', 'side', 'shares', 'value', 'price', 'cost'.

    Args:
        trade_log: List of order dicts from PortfolioSimulator.trade_log.

    Returns:
        List of round-trip trade dicts, each with 'asset_idx', 'pnl',
        'entry_price', 'exit_price', 'shares', 'entry_value', 'exit_value',
        'cost'.
    """
    if not trade_log:
        return []

    # FIFO queue per asset: list of (remaining_shares, price, value_per_share, cost)
    open_buys: dict[int, list[list[float]]] = {}
    round_trips: list[dict[str, Any]] = []

    for order in trade_log:
        asset_idx = int(order["asset_idx"])
        side = str(order["side"])
        shares = float(order["shares"])
        price = float(order["price"])
        value = float(order["value"])
        cost = float(order["cost"])

        if shares < 1e-12:
            continue

        if side == "buy":
            if asset_idx not in open_buys:
                open_buys[asset_idx] = []
            open_buys[asset_idx].append([shares, price, value, cost])

        elif side == "sell":
            if asset_idx not in open_buys or not open_buys[asset_idx]:
                continue

            remaining = shares
            sell_price = price
            sell_cost = cost
            # Proportional cost allocation for partial matches
            cost_per_share = sell_cost / shares if shares > 1e-12 else 0.0

            while remaining > 1e-12 and open_buys.get(asset_idx):
                entry = open_buys[asset_idx][0]
                buy_shares, buy_price, buy_value, buy_cost = entry

                matched = min(remaining, buy_shares)
                buy_fraction = matched / buy_shares if buy_shares > 1e-12 else 0.0

                entry_value = buy_value * buy_fraction
                exit_value = matched * sell_price
                total_cost = (buy_cost * buy_fraction) + (cost_per_share * matched)
                pnl = exit_value - entry_value - total_cost

                round_trips.append(
                    {
                        "asset_idx": asset_idx,
                        "pnl": pnl,
                        "shares": matched,
                        "entry_price": buy_price,
                        "exit_price": sell_price,
                        "entry_value": entry_value,
                        "exit_value": exit_value,
                        "cost": total_cost,
                    }
                )

                remaining -= matched

                if matched >= buy_shares - 1e-12:
                    open_buys[asset_idx].pop(0)
                else:
                    leftover = buy_shares - matched
                    entry[0] = leftover
                    entry[2] = buy_value * (1.0 - buy_fraction)
                    entry[3] = buy_cost * (1.0 - buy_fraction)

    return round_trips


def generate_folds(
    total_bars: int,
    test_bars: int,
    min_train_bars: int,
    embargo_bars: int,
    min_folds: int = 3,
) -> list[tuple[range, range]]:
    """Generate growing-window walk-forward folds with purge gaps.

    Train window always starts at index 0 (growing window). Each fold
    advances by test_bars + embargo_bars. The embargo gap ensures no
    data leakage between training and test periods.

    Args:
        total_bars: Total number of bars in the dataset.
        test_bars: Number of bars in each test window.
        min_train_bars: Minimum training bars required for the first fold.
        embargo_bars: Purge gap between train end and test start.
        min_folds: Minimum number of folds required (default 3).

    Returns:
        List of (train_range, test_range) tuples.

    Raises:
        ValueError: If fewer than min_folds can be generated.
    """
    folds: list[tuple[range, range]] = []

    # First test window starts after min_train_bars + embargo_bars
    test_start = min_train_bars + embargo_bars

    while test_start + test_bars <= total_bars:
        train_end = test_start - embargo_bars
        train_range = range(0, train_end)
        test_range = range(test_start, test_start + test_bars)
        folds.append((train_range, test_range))

        test_start += test_bars + embargo_bars

    if len(folds) < min_folds:
        msg = (
            f"Generated {len(folds)} folds, fewer than {min_folds} required. "
            f"Need more data: total_bars={total_bars}, "
            f"min_train_bars={min_train_bars}, test_bars={test_bars}, "
            f"embargo_bars={embargo_bars}"
        )
        raise ValueError(msg)

    log.info(
        "folds_generated",
        fold_count=len(folds),
        total_bars=total_bars,
        test_bars=test_bars,
        embargo_bars=embargo_bars,
    )

    return folds


class WalkForwardBacktester:
    """Walk-forward backtester with per-fold retraining and evaluation.

    For each fold: trains on growing training window, evaluates on
    held-out test window with VecNormalize in eval mode, computes
    metrics, checks validation gates, and optionally stores results
    to DuckDB.

    Args:
        config: Validated SwingRLConfig instance.
        db: Optional DatabaseManager for result storage.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        db: DatabaseManager | None = None,
    ) -> None:
        self._config = config
        self._db = db

    def run(
        self,
        env_name: str,
        algo_name: str,
        features: np.ndarray,
        prices: np.ndarray,
        models_dir: Path,
        total_timesteps: int = 1_000_000,
    ) -> list[FoldResult]:
        """Run walk-forward backtest for one algorithm on one environment.

        Args:
            env_name: Environment type ("equity" or "crypto").
            algo_name: Algorithm name ("ppo", "a2c", or "sac").
            features: Full feature array (all bars).
            prices: Full price array (all bars).
            models_dir: Root directory for model storage.
            total_timesteps: Training timesteps per fold.

        Returns:
            List of FoldResult for each fold.
        """
        from swingrl.training.trainer import TrainingOrchestrator

        params = ENV_PARAMS[env_name]
        total_bars = len(features)

        folds = generate_folds(
            total_bars=total_bars,
            test_bars=int(params["test_bars"]),
            min_train_bars=int(params["min_train_bars"]),
            embargo_bars=int(params["embargo_bars"]),
        )

        log.info(
            "walk_forward_started",
            env_name=env_name,
            algo_name=algo_name,
            fold_count=len(folds),
            total_bars=total_bars,
        )

        orchestrator = TrainingOrchestrator(
            config=self._config,
            models_dir=models_dir,
            logs_dir=Path(self._config.paths.logs_dir),
        )

        results: list[FoldResult] = []

        for fold_idx, (train_range, test_range) in enumerate(folds):
            log.info(
                "fold_started",
                fold=fold_idx,
                train_bars=len(train_range),
                test_bars=len(test_range),
            )

            # Slice data for this fold
            train_features = features[train_range.start : train_range.stop]
            train_prices = prices[train_range.start : train_range.stop]
            test_features = features[test_range.start : test_range.stop]
            test_prices = prices[test_range.start : test_range.stop]

            # Train on training window
            training_result = orchestrator.train(
                env_name=env_name,
                algo_name=algo_name,
                features=train_features,
                prices=train_prices,
                total_timesteps=total_timesteps,
            )

            # Evaluate on train data (in-sample)
            is_metrics, is_trades = self._evaluate_fold(
                training_result.model_path,
                training_result.vec_normalize_path,
                algo_name,
                train_features,
                train_prices,
                env_name,
                params,
            )

            # Evaluate on test data (out-of-sample)
            oos_metrics, oos_trades = self._evaluate_fold(
                training_result.model_path,
                training_result.vec_normalize_path,
                algo_name,
                test_features,
                test_prices,
                env_name,
                params,
            )

            # Overfitting diagnosis
            overfit = diagnose_overfitting(
                in_sample_sharpe=is_metrics.get("sharpe", 0.0),
                out_of_sample_sharpe=oos_metrics.get("sharpe", 0.0),
            )

            # Validation gates on OOS metrics
            gate = check_validation_gates(
                sharpe=oos_metrics.get("sharpe", 0.0),
                mdd=oos_metrics.get("mdd", 1.0),
                profit_factor=oos_metrics.get("profit_factor", 0.0),
                overfit_gap=overfit.get("gap", float("inf")),
            )

            fold_result = FoldResult(
                fold_number=fold_idx,
                train_range=(train_range.start, train_range.stop),
                test_range=(test_range.start, test_range.stop),
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                trades=oos_trades,
                gate_result=gate,
                overfitting=overfit,
            )

            results.append(fold_result)

            # Store to DuckDB if available
            if self._db is not None:
                model_id = f"{env_name}-{algo_name}-fold{fold_idx}"
                self._store_results(fold_result, model_id)

            log.info(
                "fold_complete",
                env_name=env_name,
                algo_name=algo_name,
                fold=fold_idx,
                gate_passed=gate.passed,
                failures=gate.failures,
                oos_sharpe=round(oos_metrics.get("sharpe", 0.0), 4),
                oos_mdd=round(oos_metrics.get("mdd", 0.0), 4),
                oos_profit_factor=round(oos_metrics.get("profit_factor", 0.0), 4),
                overfit_gap=round(overfit.get("gap", 0.0), 4),
                overfit_class=overfit.get("classification"),
                total_trades=int(oos_metrics.get("total_trades", 0)),
                win_rate=round(oos_metrics.get("win_rate", 0.0), 4),
            )

        return results

    def _evaluate_fold(
        self,
        model_path: Path,
        vec_normalize_path: Path,
        algo_name: str,
        features: np.ndarray,
        prices: np.ndarray,
        env_name: str,
        params: dict[str, int | float],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate a trained model on a data slice.

        Loads the model and VecNormalize in eval mode (training=False,
        norm_reward=False), runs through the data, collects returns
        and trades, and computes all metrics.

        Args:
            model_path: Path to saved SB3 model.
            vec_normalize_path: Path to saved VecNormalize.
            algo_name: Algorithm name for model loading.
            features: Feature array for evaluation.
            prices: Price array for evaluation.
            env_name: Environment type.
            params: Environment parameters dict.

        Returns:
            Tuple of (metrics_dict, trades_list).
        """
        # pickle is required for SB3 VecNormalize deserialization
        import pickle  # nosec B403  # noqa: S403

        from stable_baselines3.common.vec_env import DummyVecEnv

        from swingrl.envs.crypto import CryptoTradingEnv
        from swingrl.envs.equity import StockTradingEnv
        from swingrl.training.trainer import ALGO_MAP

        algo_cls = ALGO_MAP[algo_name]
        model = algo_cls.load(str(model_path))

        # Create eval environment with VecNormalize in eval mode
        env_cls = StockTradingEnv if env_name == "equity" else CryptoTradingEnv

        def _make_env() -> StockTradingEnv | CryptoTradingEnv:
            return env_cls(
                features=features,
                prices=prices,
                config=self._config,
            )

        from stable_baselines3.common.vec_env import VecNormalize

        dummy_env = DummyVecEnv([_make_env])

        # Load saved VecNormalize to extract running stats, then create a
        # fresh wrapper around the single-env DummyVecEnv. This avoids
        # shape mismatches when training used SubprocVecEnv with n_envs > 1.
        with Path(vec_normalize_path).open("rb") as f:
            saved_vec = pickle.load(f)  # noqa: S301  # nosec B301  -- SB3 VecNormalize requires pickle

        vec_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=False)
        vec_env.obs_rms = saved_vec.obs_rms
        vec_env.ret_rms = saved_vec.ret_rms
        vec_env.training = False
        vec_env.norm_reward = False

        # Run through all steps
        obs = vec_env.reset()
        returns: list[float] = []
        raw_trade_log: list[dict[str, Any]] = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = vec_env.step(action)
            obs = step_result[0]  # type: ignore[assignment]
            info = step_result[3]  # info dict (list of dicts for VecEnv)

            if isinstance(info, list) and len(info) > 0:
                step_info = info[0]
            else:
                step_info = info if isinstance(info, dict) else {}

            daily_return = step_info.get("daily_return", 0.0)
            returns.append(float(daily_return))

            # DummyVecEnv auto-resets on done=True, clearing the portfolio.
            # Grab trade_log from terminal info (env embeds it before reset).
            if "trade_log" in step_info:
                raw_trade_log = step_info["trade_log"]

            done_arr = step_result[2]
            done = bool(done_arr[0]) if hasattr(done_arr, "__len__") else bool(done_arr)

        trades = _reconstruct_round_trips(raw_trade_log)

        vec_env.close()

        # Compute metrics
        returns_arr = np.array(returns)
        periods_per_year = float(params["periods_per_year"])
        bars_per_week = float(params["bars_per_week"])

        trade_metrics = compute_trade_metrics(
            trades=trades,
            total_bars=len(returns),
            bars_per_week=bars_per_week,
        )

        metrics: dict[str, float] = {
            "sharpe": annualized_sharpe(returns_arr, periods_per_year),
            "sortino": sortino_ratio(returns_arr, periods_per_year),
            "calmar": calmar_ratio(returns_arr, periods_per_year),
            "rachev": rachev_ratio(returns_arr),
            "mdd": max_drawdown(returns_arr),
            "avg_drawdown": avg_drawdown(returns_arr),
            "max_dd_duration": float(max_drawdown_duration(returns_arr)),
            "profit_factor": trade_metrics["profit_factor"],
            "win_rate": trade_metrics["win_rate"],
            "total_trades": trade_metrics["total_trades"],
        }

        # Portfolio value from returns
        if len(returns_arr) > 0:
            cum_return = float(np.prod(1 + returns_arr))
            initial = self._config.environment.initial_amount
            metrics["final_portfolio_value"] = initial * cum_return
            metrics["total_return"] = cum_return - 1
        else:
            metrics["final_portfolio_value"] = self._config.environment.initial_amount
            metrics["total_return"] = 0.0

        return metrics, trades

    def _store_results(self, fold_result: FoldResult, model_id: str) -> None:
        """Write fold results to DuckDB backtest_results table.

        Args:
            fold_result: Completed fold result.
            model_id: Model identifier for linking results.
        """
        if self._db is None:
            return

        result_id = str(uuid.uuid4())
        oos = fold_result.out_of_sample_metrics

        with self._db.duckdb() as cursor:
            cursor.execute(
                """
                INSERT INTO backtest_results (
                    result_id, model_id, environment, algorithm, fold_number,
                    fold_type, train_start_idx, train_end_idx,
                    test_start_idx, test_end_idx,
                    sharpe, sortino, calmar, mdd, profit_factor,
                    win_rate, total_trades, avg_drawdown, max_dd_duration,
                    final_portfolio_value, total_return
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result_id,
                    model_id,
                    model_id.split("-")[0],  # env name from model_id
                    model_id.split("-")[1] if "-" in model_id else "",
                    fold_result.fold_number,
                    "walk_forward",
                    fold_result.train_range[0],
                    fold_result.train_range[1],
                    fold_result.test_range[0],
                    fold_result.test_range[1],
                    oos.get("sharpe"),
                    oos.get("sortino"),
                    oos.get("calmar"),
                    oos.get("mdd"),
                    oos.get("profit_factor"),
                    oos.get("win_rate"),
                    int(oos.get("total_trades", 0)),
                    oos.get("avg_drawdown"),
                    int(oos.get("max_dd_duration", 0)),
                    oos.get("final_portfolio_value"),
                    oos.get("total_return"),
                ],
            )

        log.info(
            "backtest_result_stored",
            result_id=result_id,
            model_id=model_id,
            fold=fold_result.fold_number,
        )
