"""Walk-forward backtester with growing-window fold generation.

Generates non-overlapping folds with purge gaps (embargo), retrains
per fold via TrainingOrchestrator, evaluates on held-out test data,
and stores results to PostgreSQL.

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
from swingrl.utils.exceptions import DataError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.memory.client import MemoryClient

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
        converged_at_step: Step where training converged (None if not converged).
        total_timesteps: Total timesteps configured for this fold's training.
    """

    fold_number: int
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    in_sample_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]
    trades: list[dict[str, Any]]
    gate_result: GateResult
    overfitting: dict[str, Any]
    converged_at_step: int | None = None
    total_timesteps: int | None = None
    is_control_fold: bool = False
    advice_stats: dict[str, Any] | None = None


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
        raise DataError(msg)

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
    to PostgreSQL.

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
        hyperparams_override: dict[str, Any] | None = None,
        memory_client: MemoryClient | None = None,
        fold_queue: Any | None = None,
        advice_enabled: bool = True,
        control_fold_indices: set[int] | None = None,
        iteration: int | None = None,
    ) -> list[FoldResult]:
        """Run walk-forward backtest for one algorithm on one environment.

        Args:
            env_name: Environment type ("equity" or "crypto").
            algo_name: Algorithm name ("ppo", "a2c", or "sac").
            features: Full feature array (all bars).
            prices: Full price array (all bars).
            models_dir: Root directory for model storage.
            total_timesteps: Training timesteps per fold.
            hyperparams_override: Optional dict of hyperparameters to override
                defaults during training. Passed through to orchestrator.train().
            memory_client: Optional MemoryClient for epoch-level memory ingestion
                and LLM-guided reward weight adjustments during training.
                Fail-open: if None, training proceeds without memory integration.
            fold_queue: Optional multiprocessing.Queue for real-time per-fold
                DB writes. Each completed fold is enqueued as
                (algo_name, FoldResult) for the background writer thread.
            advice_enabled: When True, enable LLM epoch advice and reward weight
                adjustments during training. When False, only capture epoch
                memories without querying for advice (used for iteration 0).
            control_fold_indices: Fold indices that skip reward adjustments
                (epoch snapshots still captured). Used as scientific control
                group for measuring reward shaping impact.

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

        # Validate control fold indices against actual fold count
        valid_control: set[int] = set()
        if control_fold_indices:
            valid_control = {i for i in control_fold_indices if 0 <= i < len(folds)}
            invalid = control_fold_indices - valid_control
            if invalid:
                log.warning(
                    "control_fold_indices_out_of_range",
                    invalid=sorted(invalid),
                    max_fold=len(folds) - 1,
                )

        log.info(
            "walk_forward_started",
            env_name=env_name,
            algo_name=algo_name,
            fold_count=len(folds),
            total_bars=total_bars,
            hp_override=bool(hyperparams_override),
            control_folds=sorted(valid_control) if valid_control else None,
        )

        orchestrator = TrainingOrchestrator(
            config=self._config,
            models_dir=models_dir,
            logs_dir=Path(self._config.paths.logs_dir),
        )

        results: list[FoldResult] = []

        for fold_idx, (train_range, test_range) in enumerate(folds):
            is_control = fold_idx in valid_control

            log.info(
                "fold_started",
                fold=fold_idx,
                train_bars=len(train_range),
                test_bars=len(test_range),
                is_control=is_control,
            )

            # Slice data for this fold
            train_features = features[train_range.start : train_range.stop]
            train_prices = prices[train_range.start : train_range.stop]
            test_features = features[test_range.start : test_range.stop]
            test_prices = prices[test_range.start : test_range.stop]

            # Control folds: train normally but skip reward adjustments
            fold_advice = advice_enabled and not is_control
            ctrl_suffix = "_CTRL" if is_control else ""

            # Train on training window
            training_result = orchestrator.train(
                env_name=env_name,
                algo_name=algo_name,
                features=train_features,
                prices=train_prices,
                total_timesteps=total_timesteps,
                hyperparams_override=hyperparams_override,
                memory_client=memory_client,
                run_id=f"{env_name}_{algo_name}_fold{fold_idx}{ctrl_suffix}",
                advice_enabled=fold_advice,
                is_control_fold=is_control,
                iteration=iteration,
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
                converged_at_step=training_result.converged_at_step,
                total_timesteps=total_timesteps,
                is_control_fold=is_control,
                advice_stats=training_result.advice_stats,
            )

            results.append(fold_result)

            # Enqueue fold result for real-time DB write
            if fold_queue is not None:
                try:
                    fold_queue.put((algo_name, fold_result))
                except Exception:
                    log.warning("fold_queue_put_failed", fold=fold_result.fold_number)

            # Store to PostgreSQL if available
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
        """Write fold results to backtest_results table.

        Args:
            fold_result: Completed fold result.
            model_id: Model identifier for linking results.
        """
        if self._db is None:
            return

        result_id = str(uuid.uuid4())
        oos = fold_result.out_of_sample_metrics

        with self._db.connection() as cursor:
            cursor.execute(
                """
                INSERT INTO backtest_results (
                    result_id, model_id, environment, algorithm, fold_number,
                    fold_type, train_start_idx, train_end_idx,
                    test_start_idx, test_end_idx,
                    sharpe, sortino, calmar, mdd, profit_factor,
                    win_rate, total_trades, avg_drawdown, max_dd_duration,
                    final_portfolio_value, total_return, is_control_fold
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    fold_result.is_control_fold,
                ],
            )

        log.info(
            "backtest_result_stored",
            result_id=result_id,
            model_id=model_id,
            fold=fold_result.fold_number,
        )


# ---------------------------------------------------------------------------
# Deferred DB storage functions (called from main process after
# subprocess workers return FoldResult lists via ProcessPoolExecutor).
# ---------------------------------------------------------------------------


def _compute_regime_context(
    features: np.ndarray,
    test_start: int,
    test_end: int,
    n_symbols: int,
    per_asset: int,
) -> dict[str, float | None]:
    """Extract HMM and macro feature means for a fold's test period.

    Feature layout: [n_symbols * per_asset | 6 macro | 2 HMM | 1 turb]

    Args:
        features: Full feature array (bars x features).
        test_start: Test period start index.
        test_end: Test period end index.
        n_symbols: Number of symbols in the environment.
        per_asset: Features per asset (equity=15, crypto=13).

    Returns:
        Dict with hmm_p_bull, hmm_p_bear, vix_mean, yield_spread_mean.
    """
    if test_end > len(features):
        return {
            "hmm_p_bull": None,
            "hmm_p_bear": None,
            "vix_mean": None,
            "yield_spread_mean": None,
        }

    macro_start = n_symbols * per_asset
    hmm_start = macro_start + 6

    test_features = features[test_start:test_end]
    hmm_cols = test_features[:, hmm_start : hmm_start + 2]
    macro_cols = test_features[:, macro_start : macro_start + 6]

    return {
        "hmm_p_bull": float(hmm_cols[:, 0].mean()),
        "hmm_p_bear": float(hmm_cols[:, 1].mean()),
        "vix_mean": float(macro_cols[:, 0].mean()),
        "yield_spread_mean": float(macro_cols[:, 1].mean()),
    }


def _compute_trade_extremes(
    trades: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Compute max single loss and best single trade from round-trip trades.

    Args:
        trades: List of round-trip trade dicts with 'pnl' key.

    Returns:
        Tuple of (max_single_loss, best_single_trade). None if no trades.
    """
    if not trades:
        return None, None
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    return min(pnls), max(pnls)


def store_fold_results_to_duckdb(
    conn: Any,
    fold_results: list[FoldResult],
    env_name: str,
    algo_name: str,
    iteration_number: int = 0,
    run_type: str = "baseline",
    features: np.ndarray | None = None,
    dates: np.ndarray | None = None,
    n_symbols: int = 0,
    per_asset: int = 0,
) -> int:
    """Write fold results to backtest_results table.

    Designed for deferred writes from the main process after subprocess
    workers return FoldResult lists via ProcessPoolExecutor. Includes full
    OOS + IS metrics, regime context, convergence, trade quality, and dates.

    Args:
        conn: Active PostgreSQL connection.
        fold_results: List of completed FoldResult from walk-forward.
        env_name: Environment name ("equity" or "crypto").
        algo_name: Algorithm name ("ppo", "a2c", or "sac").
        iteration_number: Training iteration (0=baseline, 1+=memory-enhanced).
        run_type: One of "baseline", "tuning_r1", "tuning_r2".
        features: Full feature array for regime context extraction. None to skip.
        dates: Date strings aligned with feature indices. None to skip.
        n_symbols: Number of symbols (needed for regime context index math).
        per_asset: Features per asset (equity=15, crypto=13).

    Returns:
        Number of rows written.
    """
    if not fold_results:
        return 0

    rows_written = 0
    for fold in fold_results:
        result_id = str(uuid.uuid4())
        model_id = f"{env_name}-{algo_name}-iter{iteration_number}-fold{fold.fold_number}"
        oos = fold.out_of_sample_metrics
        ins = fold.in_sample_metrics

        # Regime context
        regime: dict[str, float | None] = {
            "hmm_p_bull": None,
            "hmm_p_bear": None,
            "vix_mean": None,
            "yield_spread_mean": None,
        }
        if features is not None and n_symbols > 0:
            regime = _compute_regime_context(
                features, fold.test_range[0], fold.test_range[1], n_symbols, per_asset
            )

        # Trade quality extremes
        max_loss, best_trade = _compute_trade_extremes(fold.trades)

        # Date context
        train_start_date = None
        train_end_date = None
        test_start_date = None
        test_end_date = None
        if dates is not None:
            t0, t1 = fold.train_range
            ts0, ts1 = fold.test_range
            if t0 < len(dates):
                train_start_date = str(dates[t0])
            if t1 - 1 < len(dates) and t1 > 0:
                train_end_date = str(dates[t1 - 1])
            if ts0 < len(dates):
                test_start_date = str(dates[ts0])
            if ts1 - 1 < len(dates) and ts1 > 0:
                test_end_date = str(dates[ts1 - 1])

        conn.execute(
            """
            INSERT INTO backtest_results (
                result_id, model_id, environment, algorithm, fold_number,
                fold_type, train_start_idx, train_end_idx,
                test_start_idx, test_end_idx,
                sharpe, sortino, calmar, mdd, profit_factor,
                win_rate, total_trades, avg_drawdown, max_dd_duration,
                final_portfolio_value, total_return,
                iteration_number, run_type,
                is_sharpe, is_sortino, is_mdd, is_total_return,
                overfitting_gap, overfitting_class,
                hmm_p_bull, hmm_p_bear, vix_mean, yield_spread_mean,
                converged_at_step, total_timesteps_configured,
                max_single_loss, best_single_trade,
                train_start_date, train_end_date, test_start_date, test_end_date,
                is_control_fold
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s
            )
            """,
            [
                result_id,
                model_id,
                env_name,
                algo_name,
                fold.fold_number,
                "walk_forward",
                fold.train_range[0],
                fold.train_range[1],
                fold.test_range[0],
                fold.test_range[1],
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
                iteration_number,
                run_type,
                ins.get("sharpe"),
                ins.get("sortino"),
                ins.get("mdd"),
                ins.get("total_return"),
                fold.overfitting.get("gap"),
                fold.overfitting.get("classification"),
                regime["hmm_p_bull"],
                regime["hmm_p_bear"],
                regime["vix_mean"],
                regime["yield_spread_mean"],
                fold.converged_at_step,
                fold.total_timesteps,
                max_loss,
                best_trade,
                train_start_date,
                train_end_date,
                test_start_date,
                test_end_date,
                fold.is_control_fold,
            ],
        )
        rows_written += 1

    log.info(
        "fold_results_stored",
        env_name=env_name,
        algo_name=algo_name,
        iteration=iteration_number,
        run_type=run_type,
        rows=rows_written,
    )
    return rows_written


def store_iteration_results_to_duckdb(
    conn: Any,
    iteration_number: int,
    env_name: str,
    ensemble_sharpe: float,
    ensemble_mdd: float,
    gate_passed: bool,
    ensemble_weights: dict[str, float],
    all_wf_results: dict[str, list[FoldResult]],
    run_type: str = "baseline",
    wall_clock_s: float = 0.0,
    memory_enabled: bool = False,
    hp_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write ensemble-level iteration results to iteration_results table.

    Computes per-algo mean Sharpe/MDD from fold data and stores alongside
    ensemble metrics, gate result, weights, and hyperparameters.
    Uses INSERT ... ON CONFLICT to be idempotent on re-runs.

    Args:
        conn: Active PostgreSQL connection.
        iteration_number: Training iteration (0=baseline, 1+=memory-enhanced).
        env_name: Environment name.
        ensemble_sharpe: Mean OOS Sharpe across all algos/folds.
        ensemble_mdd: Mean OOS MDD across all algos/folds.
        gate_passed: Whether ensemble gate passed.
        ensemble_weights: Per-algo softmax weights from OOS Sharpe.
        all_wf_results: Dict mapping algo name to list of FoldResult.
        run_type: One of "baseline", "tuning_r1", "tuning_r2".
        wall_clock_s: Wall clock seconds for this environment's pipeline.
        memory_enabled: Whether memory agent was enabled for this iteration.
        hp_overrides: Per-algo hyperparameter overrides (None for baseline).
    """
    import json as _json

    result_id = str(uuid.uuid4())

    # Per-algo mean metrics
    algo_means: dict[str, dict[str, float | None]] = {}
    total_folds = 0
    for algo in ("ppo", "a2c", "sac"):
        folds = all_wf_results.get(algo, [])
        total_folds += len(folds)
        if folds:
            sharpes = [f.out_of_sample_metrics.get("sharpe", 0.0) for f in folds]
            mdds = [f.out_of_sample_metrics.get("mdd", 0.0) for f in folds]
            algo_means[algo] = {
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_mdd": float(np.mean(mdds)),
            }
        else:
            algo_means[algo] = {"mean_sharpe": None, "mean_mdd": None}

    # Serialize hyperparams as JSON text
    hp_source = "memory_advised" if hp_overrides else "baseline"
    ppo_hp = _json.dumps(hp_overrides["ppo"]) if hp_overrides and "ppo" in hp_overrides else None
    a2c_hp = _json.dumps(hp_overrides["a2c"]) if hp_overrides and "a2c" in hp_overrides else None
    sac_hp = _json.dumps(hp_overrides["sac"]) if hp_overrides and "sac" in hp_overrides else None

    # Use INSERT ... ON CONFLICT for idempotent re-runs
    conn.execute(
        """
        INSERT INTO iteration_results (
            result_id, iteration_number, environment,
            ensemble_sharpe, ensemble_mdd, gate_passed,
            ppo_weight, a2c_weight, sac_weight,
            ppo_mean_sharpe, a2c_mean_sharpe, sac_mean_sharpe,
            ppo_mean_mdd, a2c_mean_mdd, sac_mean_mdd,
            total_folds,
            ppo_hyperparams, a2c_hyperparams, sac_hyperparams, hp_source,
            run_type, wall_clock_s, memory_enabled
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (iteration_number, environment, run_type) DO UPDATE SET
            ensemble_sharpe = EXCLUDED.ensemble_sharpe,
            ensemble_mdd = EXCLUDED.ensemble_mdd,
            gate_passed = EXCLUDED.gate_passed,
            ppo_weight = EXCLUDED.ppo_weight,
            a2c_weight = EXCLUDED.a2c_weight,
            sac_weight = EXCLUDED.sac_weight,
            ppo_mean_sharpe = EXCLUDED.ppo_mean_sharpe,
            a2c_mean_sharpe = EXCLUDED.a2c_mean_sharpe,
            sac_mean_sharpe = EXCLUDED.sac_mean_sharpe,
            ppo_mean_mdd = EXCLUDED.ppo_mean_mdd,
            a2c_mean_mdd = EXCLUDED.a2c_mean_mdd,
            sac_mean_mdd = EXCLUDED.sac_mean_mdd,
            total_folds = EXCLUDED.total_folds,
            ppo_hyperparams = EXCLUDED.ppo_hyperparams,
            a2c_hyperparams = EXCLUDED.a2c_hyperparams,
            sac_hyperparams = EXCLUDED.sac_hyperparams,
            hp_source = EXCLUDED.hp_source,
            wall_clock_s = EXCLUDED.wall_clock_s,
            memory_enabled = EXCLUDED.memory_enabled
        """,
        [
            result_id,
            iteration_number,
            env_name,
            ensemble_sharpe,
            ensemble_mdd,
            gate_passed,
            ensemble_weights.get("ppo"),
            ensemble_weights.get("a2c"),
            ensemble_weights.get("sac"),
            algo_means["ppo"]["mean_sharpe"],
            algo_means["a2c"]["mean_sharpe"],
            algo_means["sac"]["mean_sharpe"],
            algo_means["ppo"]["mean_mdd"],
            algo_means["a2c"]["mean_mdd"],
            algo_means["sac"]["mean_mdd"],
            total_folds,
            ppo_hp,
            a2c_hp,
            sac_hp,
            hp_source,
            run_type,
            wall_clock_s,
            memory_enabled,
        ],
    )

    log.info(
        "iteration_results_stored",
        iteration=iteration_number,
        env_name=env_name,
        ensemble_sharpe=round(ensemble_sharpe, 4),
        gate_passed=gate_passed,
        total_folds=total_folds,
        hp_source=hp_source,
    )
