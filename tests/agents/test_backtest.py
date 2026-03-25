"""Tests for walk-forward backtester fold generation, integration, and DuckDB storage.

VAL-01: Walk-forward generates non-overlapping folds with growing training windows.
VAL-02: Purge gap enforced between train/test boundaries (no data leakage).
STORE-01: Fold results written to DuckDB with full metrics and iteration metadata.
STORE-02: Iteration results written to DuckDB with ensemble data and hyperparameters.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from swingrl.agents.backtest import (
    FoldResult,
    generate_folds,
    store_fold_results_to_duckdb,
    store_iteration_results_to_duckdb,
)
from swingrl.utils.exceptions import DataError


class TestGenerateFolds:
    """Tests for generate_folds function."""

    def test_equity_fold_count(self) -> None:
        """Equity parameters produce at least 13 folds."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        assert len(folds) >= 13

    def test_minimum_three_folds(self) -> None:
        """Must produce at least 3 folds."""
        folds = generate_folds(
            total_bars=600,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        assert len(folds) >= 3

    def test_no_leakage_embargo_gap(self) -> None:
        """Train end + embargo <= test start for every fold."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for train_range, test_range in folds:
            assert train_range.stop + 10 <= test_range.start, (
                f"Leakage: train_end={train_range.stop}, embargo=10, test_start={test_range.start}"
            )

    def test_growing_window(self) -> None:
        """Train window starts at 0 and grows with each fold."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for train_range, _ in folds:
            assert train_range.start == 0, "Train window must always start at index 0"

        # Each successive train range should be longer
        train_sizes = [train_range.stop - train_range.start for train_range, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Train window not growing: fold {i - 1} has {train_sizes[i - 1]} bars, "
                f"fold {i} has {train_sizes[i]}"
            )

    def test_insufficient_data_raises(self) -> None:
        """DataError when not enough data for min_folds."""
        with pytest.raises(DataError, match="fewer than 3"):
            generate_folds(
                total_bars=300,
                test_bars=63,
                min_train_bars=252,
                embargo_bars=10,
            )

    def test_test_range_size(self) -> None:
        """Each test range has exactly test_bars elements."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        for _, test_range in folds:
            assert len(test_range) == 63

    def test_no_test_overlap_across_folds(self) -> None:
        """No test range overlaps with any other test range."""
        folds = generate_folds(
            total_bars=1260,
            test_bars=63,
            min_train_bars=252,
            embargo_bars=10,
        )
        all_test_indices: set[int] = set()
        for _, test_range in folds:
            test_set = set(test_range)
            overlap = all_test_indices & test_set
            assert len(overlap) == 0, f"Test overlap found: {overlap}"
            all_test_indices.update(test_set)

    def test_custom_min_folds(self) -> None:
        """Raising min_folds increases the minimum fold requirement."""
        with pytest.raises(DataError, match="fewer than 5"):
            generate_folds(
                total_bars=600,
                test_bars=63,
                min_train_bars=252,
                embargo_bars=10,
                min_folds=5,
            )

    def test_crypto_parameters(self) -> None:
        """Crypto parameters (larger windows) produce valid folds."""
        folds = generate_folds(
            total_bars=10000,
            test_bars=540,
            min_train_bars=2190,
            embargo_bars=130,
        )
        assert len(folds) >= 3
        for train_range, test_range in folds:
            assert train_range.start == 0
            assert train_range.stop + 130 <= test_range.start
            assert len(test_range) == 540


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self) -> None:
        """FoldResult holds all required fields."""
        from swingrl.agents.validation import GateResult

        result = FoldResult(
            fold_number=1,
            train_range=(0, 252),
            test_range=(262, 325),
            in_sample_metrics={"sharpe": 1.2},
            out_of_sample_metrics={"sharpe": 0.9},
            trades=[{"pnl": 100.0}],
            gate_result=GateResult(passed=True, failures=[], details={}),
            overfitting={"gap": 0.25, "classification": "marginal"},
        )
        assert result.fold_number == 1
        assert result.train_range == (0, 252)

    def test_fold_result_convergence_fields(self) -> None:
        """STORE-01: FoldResult has optional convergence fields."""
        from swingrl.agents.validation import GateResult

        result = FoldResult(
            fold_number=0,
            train_range=(0, 252),
            test_range=(262, 325),
            in_sample_metrics={"sharpe": 1.5},
            out_of_sample_metrics={"sharpe": 1.0},
            trades=[],
            gate_result=GateResult(passed=True, failures=[], details={}),
            overfitting={"gap": 0.10, "classification": "healthy"},
            converged_at_step=50000,
            total_timesteps=100000,
        )
        assert result.converged_at_step == 50000
        assert result.total_timesteps == 100000

    def test_fold_result_convergence_defaults_none(self) -> None:
        """STORE-01: converged_at_step and total_timesteps default to None."""
        from swingrl.agents.validation import GateResult

        result = FoldResult(
            fold_number=0,
            train_range=(0, 252),
            test_range=(262, 325),
            in_sample_metrics={},
            out_of_sample_metrics={},
            trades=[],
            gate_result=GateResult(passed=True, failures=[], details={}),
            overfitting={},
        )
        assert result.converged_at_step is None
        assert result.total_timesteps is None


def _create_backtest_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create backtest_results and iteration_results tables in an in-memory DuckDB."""
    conn.execute("""
        CREATE TABLE backtest_results (
            result_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            environment TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            fold_number INTEGER NOT NULL,
            fold_type TEXT NOT NULL,
            train_start_idx INTEGER,
            train_end_idx INTEGER,
            test_start_idx INTEGER,
            test_end_idx INTEGER,
            sharpe DOUBLE,
            sortino DOUBLE,
            calmar DOUBLE,
            mdd DOUBLE,
            profit_factor DOUBLE,
            win_rate DOUBLE,
            total_trades INTEGER,
            avg_drawdown DOUBLE,
            max_dd_duration INTEGER,
            final_portfolio_value DOUBLE,
            total_return DOUBLE,
            created_at TIMESTAMP DEFAULT current_timestamp,
            iteration_number INTEGER DEFAULT 0,
            run_type TEXT DEFAULT 'baseline',
            is_sharpe DOUBLE,
            is_sortino DOUBLE,
            is_mdd DOUBLE,
            is_total_return DOUBLE,
            overfitting_gap DOUBLE,
            overfitting_class TEXT,
            hmm_p_bull DOUBLE,
            hmm_p_bear DOUBLE,
            vix_mean DOUBLE,
            yield_spread_mean DOUBLE,
            converged_at_step INTEGER,
            total_timesteps_configured INTEGER,
            max_single_loss DOUBLE,
            best_single_trade DOUBLE,
            train_start_date TEXT,
            train_end_date TEXT,
            test_start_date TEXT,
            test_end_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE iteration_results (
            result_id TEXT PRIMARY KEY,
            iteration_number INTEGER NOT NULL,
            environment TEXT NOT NULL,
            ensemble_sharpe DOUBLE,
            ensemble_mdd DOUBLE,
            gate_passed BOOLEAN,
            ppo_weight DOUBLE,
            a2c_weight DOUBLE,
            sac_weight DOUBLE,
            ppo_mean_sharpe DOUBLE,
            a2c_mean_sharpe DOUBLE,
            sac_mean_sharpe DOUBLE,
            ppo_mean_mdd DOUBLE,
            a2c_mean_mdd DOUBLE,
            sac_mean_mdd DOUBLE,
            total_folds INTEGER,
            ppo_hyperparams TEXT,
            a2c_hyperparams TEXT,
            sac_hyperparams TEXT,
            hp_source TEXT DEFAULT 'baseline',
            run_type TEXT DEFAULT 'baseline',
            wall_clock_s DOUBLE,
            memory_enabled BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT current_timestamp,
            UNIQUE(iteration_number, environment, run_type)
        )
    """)


def _make_test_fold(
    fold_number: int = 0,
    sharpe: float = 1.5,
    mdd: float = -0.08,
    is_sharpe: float = 1.8,
    converged_at_step: int | None = 50000,
    total_timesteps: int | None = 100000,
) -> FoldResult:
    """Create a real FoldResult for storage tests."""
    from swingrl.agents.validation import GateResult

    return FoldResult(
        fold_number=fold_number,
        train_range=(0, 252 * (fold_number + 1)),
        test_range=(252 * (fold_number + 1) + 10, 252 * (fold_number + 1) + 73),
        in_sample_metrics={
            "sharpe": is_sharpe,
            "sortino": is_sharpe * 1.1,
            "mdd": mdd * 0.7,
            "total_return": 0.12,
        },
        out_of_sample_metrics={
            "sharpe": sharpe,
            "sortino": sharpe * 1.1,
            "calmar": sharpe * 0.5,
            "mdd": mdd,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "total_trades": 30,
            "avg_drawdown": mdd * 0.5,
            "max_dd_duration": 12,
            "final_portfolio_value": 105000.0,
            "total_return": 0.05,
        },
        trades=[
            {"pnl": 100.0, "asset_idx": 0},
            {"pnl": -30.0, "asset_idx": 1},
            {"pnl": 60.0, "asset_idx": 0},
        ],
        gate_result=GateResult(passed=True, failures=[], details={}),
        overfitting={"gap": 0.17, "classification": "healthy"},
        converged_at_step=converged_at_step,
        total_timesteps=total_timesteps,
    )


class TestStoreFoldResultsToDuckdb:
    """Tests for store_fold_results_to_duckdb()."""

    def test_writes_correct_row_count(self) -> None:
        """STORE-01: Writes one row per fold with correct iteration and run_type."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        folds = [_make_test_fold(fold_number=i) for i in range(3)]

        rows = store_fold_results_to_duckdb(
            conn=conn,
            fold_results=folds,
            env_name="equity",
            algo_name="ppo",
            iteration_number=1,
            run_type="baseline",
        )

        assert rows == 3
        result = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()
        assert result[0] == 3

        # Check iteration_number and run_type
        row = conn.execute(
            "SELECT iteration_number, run_type, environment, algorithm "
            "FROM backtest_results LIMIT 1"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "baseline"
        assert row[2] == "equity"
        assert row[3] == "ppo"
        conn.close()

    def test_is_metrics_populated(self) -> None:
        """STORE-01: In-sample metrics columns are populated."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold(is_sharpe=2.0)

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="equity",
            algo_name="a2c",
        )

        row = conn.execute("SELECT is_sharpe, is_sortino, is_mdd FROM backtest_results").fetchone()
        assert row[0] == pytest.approx(2.0)
        assert row[1] is not None
        assert row[2] is not None
        conn.close()

    def test_overfitting_populated(self) -> None:
        """STORE-01: Overfitting gap and class are stored."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold()

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="equity",
            algo_name="ppo",
        )

        row = conn.execute(
            "SELECT overfitting_gap, overfitting_class FROM backtest_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.17)
        assert row[1] == "healthy"
        conn.close()

    def test_convergence_data_stored(self) -> None:
        """STORE-01: converged_at_step and total_timesteps stored."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold(converged_at_step=75000, total_timesteps=200000)

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="crypto",
            algo_name="sac",
        )

        row = conn.execute(
            "SELECT converged_at_step, total_timesteps_configured FROM backtest_results"
        ).fetchone()
        assert row[0] == 75000
        assert row[1] == 200000
        conn.close()

    def test_trade_quality_extremes(self) -> None:
        """STORE-01: max_single_loss and best_single_trade computed from trades."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold()  # trades: +100, -30, +60

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="equity",
            algo_name="ppo",
        )

        row = conn.execute(
            "SELECT max_single_loss, best_single_trade FROM backtest_results"
        ).fetchone()
        assert row[0] == pytest.approx(-30.0)
        assert row[1] == pytest.approx(100.0)
        conn.close()

    def test_regime_context_with_features(self) -> None:
        """STORE-01: HMM and macro means populated when features provided."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold(fold_number=0)

        # Create synthetic features: 8 symbols * 15 per_asset = 120 tech + 6 macro + 2 HMM + 1 turb
        n_bars = 400
        n_features = 8 * 15 + 6 + 2 + 1  # 129
        features = np.zeros((n_bars, n_features), dtype=np.float32)
        # Set HMM probs in test range
        test_start, test_end = fold.test_range
        features[test_start:test_end, 126] = 0.7  # p_bull
        features[test_start:test_end, 127] = 0.3  # p_bear
        # Set macro: VIX
        features[test_start:test_end, 120] = 1.5  # vix_mean
        features[test_start:test_end, 121] = 0.8  # yield_spread

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="equity",
            algo_name="ppo",
            features=features,
            n_symbols=8,
            per_asset=15,
        )

        row = conn.execute(
            "SELECT hmm_p_bull, hmm_p_bear, vix_mean, yield_spread_mean FROM backtest_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.7, abs=0.01)
        assert row[1] == pytest.approx(0.3, abs=0.01)
        assert row[2] == pytest.approx(1.5, abs=0.01)
        assert row[3] == pytest.approx(0.8, abs=0.01)
        conn.close()

    def test_date_context_with_dates(self) -> None:
        """STORE-01: Date columns populated when dates array provided."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = _make_test_fold(fold_number=0)

        dates = np.array([f"2020-01-{i + 1:02d}" for i in range(400)])

        store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="equity",
            algo_name="ppo",
            dates=dates,
        )

        row = conn.execute(
            "SELECT train_start_date, train_end_date, test_start_date, test_end_date "
            "FROM backtest_results"
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] is not None
        conn.close()

    def test_empty_fold_list_noop(self) -> None:
        """STORE-01: Empty fold list writes 0 rows."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)

        rows = store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[],
            env_name="equity",
            algo_name="ppo",
        )

        assert rows == 0
        result = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()
        assert result[0] == 0
        conn.close()

    def test_sparse_metrics_graceful(self) -> None:
        """STORE-01: FoldResult with missing metrics doesn't raise."""
        from swingrl.agents.validation import GateResult

        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        fold = FoldResult(
            fold_number=0,
            train_range=(0, 100),
            test_range=(110, 150),
            in_sample_metrics={},
            out_of_sample_metrics={"sharpe": 0.5},
            trades=[],
            gate_result=GateResult(passed=False, failures=["sharpe"], details={}),
            overfitting={},
        )

        rows = store_fold_results_to_duckdb(
            conn=conn,
            fold_results=[fold],
            env_name="crypto",
            algo_name="sac",
        )

        assert rows == 1
        row = conn.execute(
            "SELECT sharpe, is_sharpe, overfitting_gap FROM backtest_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.5)
        assert row[1] is None
        assert row[2] is None
        conn.close()


class TestStoreIterationResultsToDuckdb:
    """Tests for store_iteration_results_to_duckdb()."""

    def test_writes_ensemble_metrics(self) -> None:
        """STORE-02: Writes correct ensemble Sharpe/MDD/gate."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        folds = {"ppo": [_make_test_fold(sharpe=2.0, mdd=-0.05)]}

        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=0,
            env_name="equity",
            ensemble_sharpe=1.8,
            ensemble_mdd=-0.07,
            gate_passed=True,
            ensemble_weights={"ppo": 0.6, "a2c": 0.25, "sac": 0.15},
            all_wf_results=folds,
        )

        row = conn.execute(
            "SELECT ensemble_sharpe, ensemble_mdd, gate_passed FROM iteration_results"
        ).fetchone()
        assert row[0] == pytest.approx(1.8)
        assert row[1] == pytest.approx(-0.07)
        assert row[2] is True
        conn.close()

    def test_per_algo_weights_stored(self) -> None:
        """STORE-02: Per-algo weights stored correctly."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)

        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=0,
            env_name="equity",
            ensemble_sharpe=1.5,
            ensemble_mdd=-0.08,
            gate_passed=True,
            ensemble_weights={"ppo": 0.57, "a2c": 0.28, "sac": 0.15},
            all_wf_results={"ppo": [], "a2c": [], "sac": []},
        )

        row = conn.execute(
            "SELECT ppo_weight, a2c_weight, sac_weight FROM iteration_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.57)
        assert row[1] == pytest.approx(0.28)
        assert row[2] == pytest.approx(0.15)
        conn.close()

    def test_per_algo_mean_metrics_computed(self) -> None:
        """STORE-02: Per-algo mean Sharpe/MDD computed from fold data."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        ppo_folds = [_make_test_fold(sharpe=2.0, mdd=-0.05), _make_test_fold(sharpe=1.6, mdd=-0.09)]
        a2c_folds = [_make_test_fold(sharpe=1.0, mdd=-0.12)]

        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=1,
            env_name="equity",
            ensemble_sharpe=1.5,
            ensemble_mdd=-0.08,
            gate_passed=True,
            ensemble_weights={"ppo": 0.6, "a2c": 0.3, "sac": 0.1},
            all_wf_results={"ppo": ppo_folds, "a2c": a2c_folds, "sac": []},
        )

        row = conn.execute(
            "SELECT ppo_mean_sharpe, a2c_mean_sharpe, sac_mean_sharpe, total_folds "
            "FROM iteration_results"
        ).fetchone()
        assert row[0] == pytest.approx(1.8)  # mean(2.0, 1.6)
        assert row[1] == pytest.approx(1.0)  # mean(1.0)
        assert row[2] is None  # no SAC folds
        assert row[3] == 3  # 2 PPO + 1 A2C
        conn.close()

    def test_hyperparams_stored_as_json(self) -> None:
        """STORE-02: HP overrides stored as JSON text."""
        import json

        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)

        hp = {"ppo": {"learning_rate": 1e-4, "n_epochs": 5}, "a2c": {"ent_coef": 0.015}}
        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=1,
            env_name="equity",
            ensemble_sharpe=1.5,
            ensemble_mdd=-0.08,
            gate_passed=True,
            ensemble_weights={"ppo": 0.6, "a2c": 0.3, "sac": 0.1},
            all_wf_results={"ppo": [], "a2c": [], "sac": []},
            hp_overrides=hp,
        )

        row = conn.execute(
            "SELECT ppo_hyperparams, a2c_hyperparams, sac_hyperparams, hp_source "
            "FROM iteration_results"
        ).fetchone()
        assert json.loads(row[0]) == {"learning_rate": 1e-4, "n_epochs": 5}
        assert json.loads(row[1]) == {"ent_coef": 0.015}
        assert row[2] is None  # no SAC overrides
        assert row[3] == "memory_advised"
        conn.close()

    def test_idempotent_on_rerun(self) -> None:
        """STORE-02: Re-running with same iteration/env/run_type updates, not duplicates."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)
        folds = {"ppo": [_make_test_fold(sharpe=1.5)], "a2c": [], "sac": []}

        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=0,
            env_name="equity",
            ensemble_sharpe=1.5,
            ensemble_mdd=-0.08,
            gate_passed=True,
            ensemble_weights={"ppo": 0.6, "a2c": 0.25, "sac": 0.15},
            all_wf_results=folds,
        )

        # Re-run with different sharpe
        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=0,
            env_name="equity",
            ensemble_sharpe=1.9,
            ensemble_mdd=-0.06,
            gate_passed=True,
            ensemble_weights={"ppo": 0.5, "a2c": 0.3, "sac": 0.2},
            all_wf_results=folds,
        )

        count = conn.execute("SELECT COUNT(*) FROM iteration_results").fetchone()[0]
        assert count == 1  # Not 2
        row = conn.execute("SELECT ensemble_sharpe FROM iteration_results").fetchone()
        assert row[0] == pytest.approx(1.9)  # Updated value
        conn.close()

    def test_wall_clock_and_memory_flag(self) -> None:
        """STORE-02: wall_clock_s and memory_enabled stored."""
        conn = duckdb.connect(":memory:")
        _create_backtest_schema(conn)

        store_iteration_results_to_duckdb(
            conn=conn,
            iteration_number=1,
            env_name="crypto",
            ensemble_sharpe=2.0,
            ensemble_mdd=-0.12,
            gate_passed=False,
            ensemble_weights={"ppo": 0.9, "a2c": 0.08, "sac": 0.02},
            all_wf_results={"ppo": [], "a2c": [], "sac": []},
            wall_clock_s=26640.5,
            memory_enabled=True,
        )

        row = conn.execute("SELECT wall_clock_s, memory_enabled FROM iteration_results").fetchone()
        assert row[0] == pytest.approx(26640.5)
        assert row[1] is True
        conn.close()
