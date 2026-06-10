"""Tests for ingest_all orchestrator — single-command data pipeline.

Covers run_equity, run_crypto, run_macro, resolve_crypto_gaps, run_features,
run_pipeline, and CLI invocation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig
from swingrl.utils.exceptions import DataError

# Repo root and src path for subprocess tests
_REPO_ROOT = Path(__file__).parent.parent.parent
_SRC_PATH = str(_REPO_ROOT / "src")


def _subprocess_env() -> dict[str, str]:
    """Return environment dict with PYTHONPATH set to include src/.

    Required because editable install .pth files may not be processed
    by the subprocess's Python when running from a uv-managed venv.
    """
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_PATH}:{existing_pp}" if existing_pp else _SRC_PATH
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_ctx_mock(row_count: int = 0) -> MagicMock:
    """Return a mock DatabaseManager whose connection() yields a cursor with COUNT(*)."""
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value.fetchone.return_value = {"cnt": row_count}

    @contextmanager
    def _connection_ctx():  # type: ignore[return]
        yield mock_cursor

    mock_db.connection.side_effect = _connection_ctx
    return mock_db


# ---------------------------------------------------------------------------
# Task 1 unit tests
# ---------------------------------------------------------------------------


class TestRunEquity:
    """DATA-01, DATA-05: run_equity orchestrates AlpacaIngestor."""

    def test_run_equity_calls_run_all_with_since_none_on_backfill(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-01: Calls AlpacaIngestor.run_all(symbols, since=None) when backfill=True."""
        from swingrl.data.ingest_all import run_equity

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = []

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_ingestor),
        ):
            run_equity(loaded_config, backfill=True)

        mock_ingestor.run_all.assert_called_once_with(loaded_config.equity.symbols, since=None)

    def test_run_equity_raises_data_error_on_failed_symbols(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-01: Raises DataError when AlpacaIngestor.run_all returns failed symbols."""
        from swingrl.data.ingest_all import run_equity

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = ["SPY"]

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_ingestor),
        ):
            with pytest.raises(DataError):
                run_equity(loaded_config, backfill=True)

    def test_run_equity_returns_row_delta(self, loaded_config: SwingRLConfig) -> None:
        """DATA-01: Returns non-negative integer row delta."""
        from swingrl.data.ingest_all import run_equity

        # First call returns 100, second call returns 150 → delta = 50
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        call_count = [0]

        def _fetchone_side_effect() -> dict[str, int]:
            call_count[0] += 1
            return {"cnt": 100} if call_count[0] == 1 else {"cnt": 150}

        mock_cursor.execute.return_value.fetchone.side_effect = _fetchone_side_effect

        @contextmanager
        def _connection_ctx():  # type: ignore[return]
            yield mock_cursor

        mock_db.connection.side_effect = _connection_ctx

        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = []

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_ingestor),
        ):
            delta = run_equity(loaded_config, backfill=True)

        assert delta == 50


class TestRunCrypto:
    """DATA-02, DATA-05: run_crypto orchestrates BinanceIngestor."""

    def test_run_crypto_calls_backfill_for_each_symbol_when_backfill_true(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-02: Calls BinanceIngestor.backfill(symbol) for each symbol when backfill=True."""
        from swingrl.data.ingest_all import run_crypto

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.backfill.return_value = pd.DataFrame()
        mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
        mock_ingestor.__exit__ = MagicMock(return_value=False)

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_ingestor),
        ):
            run_crypto(loaded_config, backfill=True)

        expected_calls = [call(symbol) for symbol in loaded_config.crypto.symbols]
        assert mock_ingestor.backfill.call_args_list == expected_calls

    def test_run_crypto_calls_run_all_when_backfill_false(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-02: Calls BinanceIngestor.run_all(symbols, since=None) when backfill=False."""
        from swingrl.data.ingest_all import run_crypto

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = []
        mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
        mock_ingestor.__exit__ = MagicMock(return_value=False)

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_ingestor),
        ):
            run_crypto(loaded_config, backfill=False)

        mock_ingestor.run_all.assert_called_once_with(loaded_config.crypto.symbols, since=None)

    def test_run_crypto_returns_row_delta(self, loaded_config: SwingRLConfig) -> None:
        """DATA-02: Returns non-negative integer row delta after ingestion."""
        from swingrl.data.ingest_all import run_crypto

        mock_db = MagicMock()
        mock_cursor = MagicMock()
        call_count = [0]

        def _fetchone_side_effect() -> dict[str, int]:
            call_count[0] += 1
            return {"cnt": 200} if call_count[0] == 1 else {"cnt": 300}

        mock_cursor.execute.return_value.fetchone.side_effect = _fetchone_side_effect

        @contextmanager
        def _connection_ctx():  # type: ignore[return]
            yield mock_cursor

        mock_db.connection.side_effect = _connection_ctx

        mock_ingestor = MagicMock()
        mock_ingestor.backfill.return_value = pd.DataFrame()
        mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
        mock_ingestor.__exit__ = MagicMock(return_value=False)

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_ingestor),
        ):
            delta = run_crypto(loaded_config, backfill=True)

        assert delta == 100


class TestRunMacro:
    """DATA-03, DATA-05: run_macro orchestrates FREDIngestor."""

    def test_run_macro_calls_fred_run_all_with_backfill_flag(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-03: Calls FREDIngestor.run_all(backfill=True) when backfill=True."""
        from swingrl.data.ingest_all import run_macro

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = []

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_ingestor),
        ):
            run_macro(loaded_config, backfill=True)

        mock_ingestor.run_all.assert_called_once_with(backfill=True)

    def test_run_macro_raises_data_error_on_failed_series(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-03: Raises DataError when FREDIngestor.run_all returns failed series."""
        from swingrl.data.ingest_all import run_macro

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_ingestor = MagicMock()
        mock_ingestor.run_all.return_value = ["VIXCLS"]

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_ingestor),
        ):
            with pytest.raises(DataError):
                run_macro(loaded_config, backfill=True)


class TestResolveCryptoGaps:
    """DATA-02: resolve_crypto_gaps forward-fills short gaps and rejects long ones."""

    def test_resolve_crypto_gaps_fills_up_to_2_bars(self) -> None:
        """DATA-02: Forward-fills NaN values when gap is 1-2 bars."""
        from swingrl.data.ingest_all import resolve_crypto_gaps

        df = pd.DataFrame(
            {
                "close": [100.0, None, None, 110.0],
                "volume": [1.0, None, None, 1.5],
            }
        )
        result = resolve_crypto_gaps(df, symbol="BTCUSDT")
        assert not result["close"].isna().any()
        assert result["close"].iloc[1] == pytest.approx(100.0)
        assert result["close"].iloc[2] == pytest.approx(100.0)

    def test_resolve_crypto_gaps_raises_data_error_for_gaps_over_2_bars(self) -> None:
        """DATA-02: Raises DataError when gap exceeds 2 consecutive bars."""
        from swingrl.data.ingest_all import resolve_crypto_gaps

        df = pd.DataFrame(
            {
                "close": [100.0, None, None, None, 110.0],
                "volume": [1.0, None, None, None, 1.5],
            }
        )
        with pytest.raises(DataError):
            resolve_crypto_gaps(df, symbol="BTCUSDT")


class TestRunFeatures:
    """DATA-05: run_features orchestrates FeaturePipeline."""

    def test_run_features_calls_compute_equity_and_compute_crypto(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-05: Calls FeaturePipeline.compute_equity() and compute_crypto()."""
        from swingrl.data.ingest_all import run_features

        mock_db = _make_db_ctx_mock()
        mock_pipeline = MagicMock()

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
        ):
            run_features(loaded_config)

        mock_pipeline.compute_equity.assert_called_once()
        mock_pipeline.compute_crypto.assert_called_once()


class TestRunPipeline:
    """DATA-05: run_pipeline orchestrates full pipeline flow."""

    def _build_full_mock_ctx(
        self,
        loaded_config: SwingRLConfig,
        rows_added: int = 50,
        verification_passed: bool = True,
    ) -> dict:
        """Build all mock dependencies for run_pipeline."""
        from swingrl.data.verification import VerificationResult

        mock_db = MagicMock()
        mock_cursor = MagicMock()
        call_count = [0]
        rows_per_call = [0, rows_added // 3, rows_added // 3, rows_added - 2 * (rows_added // 3)]

        def _fetchone():
            val = call_count[0]
            call_count[0] += 1
            if val < len(rows_per_call):
                return {"cnt": rows_per_call[val]}
            return {"cnt": 50}

        mock_cursor.execute.return_value.fetchone.side_effect = _fetchone

        @contextmanager
        def _connection_ctx():  # type: ignore[return]
            yield mock_cursor

        mock_db.connection.side_effect = _connection_ctx

        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(
            passed=verification_passed,
            checks=[],
        )

        return {
            "db": mock_db,
            "alpaca": mock_alpaca,
            "binance": mock_binance,
            "fred": mock_fred,
            "pipeline": mock_pipeline,
            "vr": vr,
        }

    def test_pipeline_skips_features_when_no_rows_added(
        self, loaded_config: SwingRLConfig, tmp_path: Path
    ) -> None:
        """DATA-05: Feature computation skipped when total rows_added == 0."""
        from swingrl.data.ingest_all import run_pipeline
        from swingrl.data.verification import VerificationResult

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(passed=True, checks=[])

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_alpaca),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_binance),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_fred),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
            patch("swingrl.data.ingest_all.run_verification", return_value=vr),
            patch("swingrl.data.ingest_all.write_report"),
            patch("swingrl.data.ingest_all.print_summary"),
            patch("swingrl.data.ingest_all.detect_and_fill_crypto_gaps", return_value=[]),
        ):
            run_pipeline(loaded_config, backfill=True)

        mock_pipeline.compute_equity.assert_not_called()
        mock_pipeline.compute_crypto.assert_not_called()

    def test_pipeline_runs_features_when_rows_added(
        self, loaded_config: SwingRLConfig, tmp_path: Path
    ) -> None:
        """DATA-05: Feature computation runs when rows_added > 0."""
        from swingrl.data.ingest_all import run_pipeline
        from swingrl.data.verification import VerificationResult

        # Rows: before=0, after=50 for equity; same zero for crypto/macro
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        call_results = [0, 50, 0, 0, 0, 0]
        call_idx = [0]

        def _fetchone():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(call_results):
                return {"cnt": call_results[idx]}
            return {"cnt": 0}

        mock_cursor.execute.return_value.fetchone.side_effect = _fetchone

        @contextmanager
        def _connection_ctx():  # type: ignore[return]
            yield mock_cursor

        mock_db.connection.side_effect = _connection_ctx

        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(passed=True, checks=[])

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_alpaca),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_binance),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_fred),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
            patch("swingrl.data.ingest_all.run_verification", return_value=vr),
            patch("swingrl.data.ingest_all.write_report"),
            patch("swingrl.data.ingest_all.print_summary"),
            patch("swingrl.data.ingest_all.detect_and_fill_crypto_gaps", return_value=[]),
        ):
            run_pipeline(loaded_config, backfill=True)

        mock_pipeline.compute_equity.assert_called_once()
        mock_pipeline.compute_crypto.assert_called_once()

    def test_pipeline_exits_nonzero_when_verification_fails(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-05: Returns non-zero exit code when verification fails."""
        from swingrl.data.ingest_all import run_pipeline
        from swingrl.data.verification import VerificationResult

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(passed=False, checks=[])

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_alpaca),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_binance),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_fred),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
            patch("swingrl.data.ingest_all.run_verification", return_value=vr),
            patch("swingrl.data.ingest_all.write_report"),
            patch("swingrl.data.ingest_all.print_summary"),
            patch("swingrl.data.ingest_all.detect_and_fill_crypto_gaps", return_value=[]),
        ):
            exit_code = run_pipeline(loaded_config, backfill=True)

        assert exit_code != 0

    def test_pipeline_exits_zero_when_verification_passes(
        self, loaded_config: SwingRLConfig
    ) -> None:
        """DATA-05: Returns 0 exit code when verification passes."""
        from swingrl.data.ingest_all import run_pipeline
        from swingrl.data.verification import VerificationResult

        mock_db = _make_db_ctx_mock(row_count=0)
        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(passed=True, checks=[])

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_alpaca),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_binance),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_fred),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
            patch("swingrl.data.ingest_all.run_verification", return_value=vr),
            patch("swingrl.data.ingest_all.write_report"),
            patch("swingrl.data.ingest_all.print_summary"),
            patch("swingrl.data.ingest_all.detect_and_fill_crypto_gaps", return_value=[]),
        ):
            exit_code = run_pipeline(loaded_config, backfill=True)

        assert exit_code == 0


class TestCliHelp:
    """DATA-05: Module invokable via python -m swingrl.data.ingest_all."""

    def test_module_help_exits_zero(self) -> None:
        """DATA-05: python -m swingrl.data.ingest_all --help exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "swingrl.data.ingest_all", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=_subprocess_env(),
        )
        assert result.returncode == 0
        assert "--backfill" in result.stdout


# ---------------------------------------------------------------------------
# Task 2: Integration smoke tests
# ---------------------------------------------------------------------------


class TestModuleImportable:
    """DATA-05: All public symbols importable from ingest_all."""

    def test_module_importable(self) -> None:
        """DATA-05: All exported functions importable."""
        from swingrl.data.ingest_all import (  # noqa: F401
            main,
            run_crypto,
            run_equity,
            run_features,
            run_macro,
            run_pipeline,
        )


class TestCliHelp2:
    """DATA-05: CLI integration via subprocess using current interpreter."""

    def test_cli_help(self) -> None:
        """DATA-05: subprocess --help returns exit 0 and contains --backfill.

        Uses sys.executable + explicit PYTHONPATH so the subprocess can
        locate swingrl regardless of editable install .pth processing state.
        """
        result = subprocess.run(
            [sys.executable, "-m", "swingrl.data.ingest_all", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=_subprocess_env(),
        )
        assert result.returncode == 0
        assert "--backfill" in result.stdout


class TestFullPipelineMock:
    """DATA-05: Full pipeline end-to-end with all dependencies mocked."""

    def test_full_pipeline_mock(self, loaded_config: SwingRLConfig) -> None:
        """DATA-05: Full pipeline wires all ingestors, features, and verification."""
        from swingrl.data.ingest_all import run_pipeline
        from swingrl.data.verification import CheckResult, VerificationResult

        # Setup: row counts return 0 before and 10 after equity ingest so features run
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        call_results = [0, 10, 0, 0, 0, 0]
        call_idx = [0]

        def _fetchone():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(call_results):
                return {"cnt": call_results[idx]}
            return {"cnt": 0}

        mock_cursor.execute.return_value.fetchone.side_effect = _fetchone

        @contextmanager
        def _connection_ctx():  # type: ignore[return]
            yield mock_cursor

        mock_db.connection.side_effect = _connection_ctx

        mock_alpaca = MagicMock()
        mock_alpaca.run_all.return_value = []
        mock_binance = MagicMock()
        mock_binance.backfill.return_value = pd.DataFrame()
        mock_binance.__enter__ = MagicMock(return_value=mock_binance)
        mock_binance.__exit__ = MagicMock(return_value=False)
        mock_fred = MagicMock()
        mock_fred.run_all.return_value = []
        mock_pipeline = MagicMock()

        vr = VerificationResult(
            passed=True,
            checks=[CheckResult(name="equity_rows", passed=True, detail="ok")],
        )

        with (
            patch("swingrl.data.ingest_all.DatabaseManager", return_value=mock_db),
            patch("swingrl.data.ingest_all.AlpacaIngestor", return_value=mock_alpaca),
            patch("swingrl.data.ingest_all.BinanceIngestor", return_value=mock_binance),
            patch("swingrl.data.ingest_all.FREDIngestor", return_value=mock_fred),
            patch("swingrl.data.ingest_all.FeaturePipeline", return_value=mock_pipeline),
            patch("swingrl.data.ingest_all.run_verification", return_value=vr) as mock_verify,
            patch("swingrl.data.ingest_all.write_report") as mock_write,
            patch("swingrl.data.ingest_all.print_summary") as mock_print,
            patch("swingrl.data.ingest_all.detect_and_fill_crypto_gaps", return_value=[]),
        ):
            exit_code = run_pipeline(loaded_config, backfill=True)

        # AlpacaIngestor.run_all called with since=None (backfill)
        mock_alpaca.run_all.assert_called_once_with(loaded_config.equity.symbols, since=None)

        # BinanceIngestor.backfill called once per crypto symbol
        assert mock_binance.backfill.call_count == len(loaded_config.crypto.symbols)
        for sym in loaded_config.crypto.symbols:
            mock_binance.backfill.assert_any_call(sym)

        # FREDIngestor.run_all called with backfill=True
        mock_fred.run_all.assert_called_once_with(backfill=True)

        # FeaturePipeline called (rows added > 0)
        mock_pipeline.compute_equity.assert_called_once()
        mock_pipeline.compute_crypto.assert_called_once()

        # run_verification called exactly once
        mock_verify.assert_called_once()

        # Returns 0 (verification passed)
        assert exit_code == 0

        # write_report and print_summary called
        mock_write.assert_called_once()
        mock_print.assert_called_once()
