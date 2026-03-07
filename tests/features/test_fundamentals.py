"""Tests for fundamental data fetching, validation, and storage.

Covers: FEAT-03 — yfinance primary, Alpha Vantage fallback, validation, z-scores, DuckDB storage.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from swingrl.config.schema import load_config
from swingrl.features.fundamentals import FundamentalFetcher


@pytest.fixture
def fetcher(tmp_path: Path) -> FundamentalFetcher:
    """Create a FundamentalFetcher with test config."""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(
        "trading_mode: paper\n"
        "equity:\n"
        "  symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]\n"
        "  max_position_size: 0.25\n"
        "  max_drawdown_pct: 0.10\n"
        "  daily_loss_limit_pct: 0.02\n"
        "crypto:\n"
        "  symbols: [BTCUSDT, ETHUSDT]\n"
        "  max_position_size: 0.50\n"
        "  max_drawdown_pct: 0.12\n"
        "  daily_loss_limit_pct: 0.03\n"
        "  min_order_usd: 10.0\n"
        "system:\n"
        "  duckdb_path: data/db/market_data.ddb\n"
        "  sqlite_path: data/db/trading_ops.db\n"
    )
    config = load_config(config_file)
    return FundamentalFetcher(config)


class TestFetchSymbol:
    """FEAT-03: fetch_symbol returns dict with 4 fundamental metrics."""

    @patch("swingrl.features.fundamentals.yfinance")
    def test_fetch_symbol_returns_four_keys(
        self, mock_yf: MagicMock, fetcher: FundamentalFetcher
    ) -> None:
        """fetch_symbol('SPY') returns dict with pe_ratio, earnings_growth, debt_to_equity, dividend_yield."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 25.5,
            "earningsQuarterlyGrowth": 0.12,
            "debtToEquity": 45.0,
            "dividendYield": 0.015,
            "sector": "Technology",
        }
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher.fetch_symbol("SPY")

        assert "pe_ratio" in result
        assert "earnings_growth" in result
        assert "debt_to_equity" in result
        assert "dividend_yield" in result
        assert result["pe_ratio"] == 25.5
        assert result["earnings_growth"] == 0.12
        assert result["debt_to_equity"] == 45.0
        assert result["dividend_yield"] == 0.015

    @patch("swingrl.features.fundamentals.yfinance")
    def test_missing_field_returns_nan(
        self, mock_yf: MagicMock, fetcher: FundamentalFetcher
    ) -> None:
        """When yfinance returns None for a field, value is NaN (not zero)."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": None,
            "earningsQuarterlyGrowth": 0.05,
            "debtToEquity": None,
            "dividendYield": 0.01,
            "sector": "Financials",
        }
        mock_yf.Ticker.return_value = mock_ticker

        result = fetcher.fetch_symbol("XLF")

        assert math.isnan(result["pe_ratio"])
        assert result["earnings_growth"] == 0.05
        assert math.isnan(result["debt_to_equity"])

    @patch("swingrl.features.fundamentals.yfinance")
    def test_yfinance_failure_triggers_av_fallback(
        self, mock_yf: MagicMock, fetcher: FundamentalFetcher
    ) -> None:
        """When yfinance raises an exception, Alpha Vantage fallback is attempted."""
        mock_yf.Ticker.side_effect = Exception("yfinance rate limit")

        with patch.dict(
            "os.environ",
            {"ALPHA_VANTAGE_API_KEY": "test_key"},  # pragma: allowlist secret
        ):
            with patch("swingrl.features.fundamentals.FundamentalData") as mock_av:
                mock_av_instance = MagicMock()
                mock_av.return_value = mock_av_instance
                mock_av_instance.get_company_overview.return_value = (
                    pd.DataFrame(
                        {
                            "TrailingPE": ["20.0"],
                            "QuarterlyEarningsGrowthYOY": ["0.08"],
                            "DebtToEquityRatio": ["30.0"],
                            "DividendYield": ["0.02"],
                            "Sector": ["Technology"],
                        }
                    ),
                    None,
                )
                # Re-create fetcher to pick up env var
                fetcher_with_av = FundamentalFetcher(fetcher._config)
                fetcher_with_av._av_api_key = "test_key"  # pragma: allowlist secret
                result = fetcher_with_av.fetch_symbol("SPY")

                assert result["pe_ratio"] == pytest.approx(20.0)


class TestValidation:
    """FEAT-03: Validation of fundamental data."""

    def test_negative_pe_becomes_nan(self, fetcher: FundamentalFetcher) -> None:
        """validate_fundamentals() rejects negative P/E (sets to NaN)."""
        data = {
            "pe_ratio": -5.0,
            "earnings_growth": 0.10,
            "debt_to_equity": 30.0,
            "dividend_yield": 0.01,
        }
        result = fetcher.validate_fundamentals(data)
        assert math.isnan(result["pe_ratio"])

    def test_negative_de_becomes_nan(self, fetcher: FundamentalFetcher) -> None:
        """validate_fundamentals() rejects negative D/E (sets to NaN)."""
        data = {
            "pe_ratio": 20.0,
            "earnings_growth": 0.10,
            "debt_to_equity": -10.0,
            "dividend_yield": 0.01,
        }
        result = fetcher.validate_fundamentals(data)
        assert math.isnan(result["debt_to_equity"])

    def test_valid_fundamentals_unchanged(self, fetcher: FundamentalFetcher) -> None:
        """validate_fundamentals() accepts valid fundamentals without modification."""
        data = {
            "pe_ratio": 25.0,
            "earnings_growth": 0.12,
            "debt_to_equity": 40.0,
            "dividend_yield": 0.015,
        }
        result = fetcher.validate_fundamentals(data)
        assert result["pe_ratio"] == 25.0
        assert result["earnings_growth"] == 0.12
        assert result["debt_to_equity"] == 40.0
        assert result["dividend_yield"] == 0.015


class TestFetchAll:
    """FEAT-03: fetch_all returns DataFrame with all symbols."""

    @patch("swingrl.features.fundamentals.yfinance")
    def test_fetch_all_returns_dataframe(
        self, mock_yf: MagicMock, fetcher: FundamentalFetcher
    ) -> None:
        """fetch_all(symbols) returns DataFrame with one row per symbol, 4 metrics + sector."""

        def make_ticker(symbol: str) -> MagicMock:
            t = MagicMock()
            t.info = {
                "trailingPE": 20.0,
                "earningsQuarterlyGrowth": 0.10,
                "debtToEquity": 35.0,
                "dividendYield": 0.02,
                "sector": "Technology",
            }
            return t

        mock_yf.Ticker.side_effect = make_ticker

        result = fetcher.fetch_all(["SPY", "QQQ"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "symbol" in result.columns
        assert "pe_ratio" in result.columns
        assert "earnings_growth" in result.columns
        assert "debt_to_equity" in result.columns
        assert "dividend_yield" in result.columns
        assert "sector" in result.columns


class TestSectorRelativeZscore:
    """FEAT-03: sector_relative_zscore computes cross-sectional z-scores for P/E."""

    def test_pe_zscore_computed(self, fetcher: FundamentalFetcher) -> None:
        """sector_relative_zscore() computes z-score for P/E grouped by sector."""
        df = pd.DataFrame(
            {
                "symbol": ["SPY", "QQQ", "XLK", "XLV"],
                "pe_ratio": [20.0, 30.0, 25.0, 15.0],
                "earnings_growth": [0.10, 0.15, 0.12, 0.08],
                "debt_to_equity": [40.0, 50.0, 45.0, 30.0],
                "dividend_yield": [0.02, 0.01, 0.015, 0.025],
                "sector": ["Tech", "Tech", "Tech", "Health"],
            }
        )

        result = fetcher.sector_relative_zscore(df)

        assert "pe_zscore" in result.columns
        # Other columns remain raw (not z-scored)
        assert "earnings_growth" in result.columns
        assert "debt_to_equity" in result.columns
        assert "dividend_yield" in result.columns

        # Tech group: mean=25, std=5 -> SPY zscore = (20-25)/5 = -1.0
        tech_spy = result[result["symbol"] == "SPY"]["pe_zscore"].iloc[0]
        assert tech_spy == pytest.approx(-1.0)


class TestStoreFundamentals:
    """FEAT-03: store_fundamentals writes to DuckDB."""

    def test_store_writes_to_duckdb(self, tmp_path: Path) -> None:
        """store_fundamentals() writes to DuckDB fundamentals table with fetched_at."""
        import duckdb

        db_path = tmp_path / "test.ddb"
        conn = duckdb.connect(str(db_path))
        conn.execute("""
            CREATE TABLE fundamentals (
                symbol TEXT,
                date DATE,
                pe_ratio DOUBLE,
                earnings_growth DOUBLE,
                debt_to_equity DOUBLE,
                dividend_yield DOUBLE,
                sector TEXT,
                fetched_at TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Create a mock DatabaseManager
        mock_db = MagicMock()
        mock_db.duckdb.return_value.__enter__ = MagicMock(return_value=conn.cursor())
        mock_db.duckdb.return_value.__exit__ = MagicMock(return_value=False)

        df = pd.DataFrame(
            {
                "symbol": ["SPY", "QQQ"],
                "pe_ratio": [20.0, 30.0],
                "earnings_growth": [0.10, 0.15],
                "debt_to_equity": [40.0, 50.0],
                "dividend_yield": [0.02, 0.01],
                "sector": ["Technology", "Technology"],
            }
        )

        config_file = tmp_path / "swingrl.yaml"
        config_file.write_text(
            "trading_mode: paper\n"
            "equity:\n"
            "  symbols: [SPY, QQQ]\n"
            "  max_position_size: 0.25\n"
            "  max_drawdown_pct: 0.10\n"
            "  daily_loss_limit_pct: 0.02\n"
            "crypto:\n"
            "  symbols: [BTCUSDT, ETHUSDT]\n"
            "  max_position_size: 0.50\n"
            "  max_drawdown_pct: 0.12\n"
            "  daily_loss_limit_pct: 0.03\n"
            "  min_order_usd: 10.0\n"
            "system:\n"
            "  duckdb_path: data/db/market_data.ddb\n"
            "  sqlite_path: data/db/trading_ops.db\n"
        )
        config = load_config(config_file)
        fetcher = FundamentalFetcher(config)

        rows = fetcher.store_fundamentals(mock_db, df)
        assert rows == 2

        stored = conn.execute("SELECT * FROM fundamentals").fetchall()
        assert len(stored) == 2
        conn.close()
