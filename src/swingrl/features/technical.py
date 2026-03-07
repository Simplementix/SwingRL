"""Technical indicator computation for SwingRL feature pipeline.

Uses stockstats StockDataFrame.retype() to compute 9 price-action indicators,
weekly-derived features, crypto multi-timeframe features, and derived features
for HMM input.

Usage:
    from swingrl.features.technical import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator()
    price_action = calc.compute_price_action(ohlcv_df)
    weekly = calc.compute_weekly_features(daily_ohlcv)
    derived = calc.compute_derived(close_series)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import stockstats
import structlog

log = structlog.get_logger(__name__)


class TechnicalIndicatorCalculator:
    """Compute technical indicators from OHLCV data using stockstats.

    All methods accept standard pandas DataFrames with columns
    [open, high, low, close, volume] and return new DataFrames
    without mutating the input.
    """

    def compute_price_action(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute 9 price-action indicators from OHLCV bars.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume].
                   Must have a DatetimeIndex.

        Returns:
            DataFrame with 9 feature columns, same index as input.
            NaN values are expected for the warmup period (~200 bars for SMA-200).
        """
        sdf = stockstats.StockDataFrame.retype(ohlcv.copy())

        result = pd.DataFrame(index=ohlcv.index)

        # 1-2. Price/SMA ratios (dimensionless, close to 1.0)
        result["price_sma50_ratio"] = ohlcv["close"] / sdf["close_50_sma"]
        result["price_sma200_ratio"] = ohlcv["close"] / sdf["close_200_sma"]

        # 3. RSI-14 (0-100 range)
        result["rsi_14"] = sdf["rsi_14"]

        # 4-5. MACD line and histogram (raw values, z-scored later)
        result["macd_line"] = sdf["macd"]
        result["macd_histogram"] = sdf["macdh"]

        # 6. Bollinger Band position: (close - lb) / (ub - lb)
        # Do NOT clip -- values outside [0, 1] are valid breakout signals
        band_width = sdf["boll_ub"] - sdf["boll_lb"]
        result["bb_position"] = np.where(
            band_width > 0,
            (ohlcv["close"].values - sdf["boll_lb"].values) / band_width.values,
            0.5,  # default when bands collapse
        )

        # 7. ATR-14 as percentage of price (small positive fraction)
        result["atr_14_pct"] = sdf["atr_14"] / ohlcv["close"]

        # 8. Volume/SMA-20 ratio (default 1.0 when SMA is zero)
        vol_sma = sdf["volume_20_sma"]
        result["volume_sma20_ratio"] = np.where(
            vol_sma > 0,
            ohlcv["volume"].values / vol_sma.values,
            1.0,
        )

        # 9. ADX-14
        result["adx_14"] = sdf["adx"]

        log.info(
            "price_action_computed",
            rows=len(result),
            nan_rows=int(result.isna().any(axis=1).sum()),
        )
        return result

    def compute_weekly_features(self, daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute weekly-derived features from daily OHLCV bars.

        Resamples daily data to weekly (Friday close), computes weekly SMA-10
        trend direction and weekly RSI-14, then forward-fills back to daily index.

        Args:
            daily_ohlcv: Daily OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame with weekly_trend_dir (0/1) and weekly_rsi_14,
            aligned to the daily index via forward-fill.
        """
        # Resample to weekly bars (Friday close)
        weekly = daily_ohlcv.resample("W-FRI").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        weekly = weekly.dropna(subset=["close"])

        # Weekly SMA-10 for trend direction
        weekly_sma10 = weekly["close"].rolling(10).mean()
        weekly_trend = (weekly["close"] > weekly_sma10).astype(float)

        # Weekly RSI-14 via stockstats
        weekly_sdf = stockstats.StockDataFrame.retype(weekly.copy())
        weekly_rsi = weekly_sdf["rsi_14"]

        # Build weekly result
        weekly_result = pd.DataFrame(
            {"weekly_trend_dir": weekly_trend, "weekly_rsi_14": weekly_rsi},
            index=weekly.index,
        )

        # Reindex to daily and forward-fill
        result = weekly_result.reindex(daily_ohlcv.index, method="ffill")

        log.info("weekly_features_computed", weekly_bars=len(weekly), daily_rows=len(result))
        return result

    def compute_crypto_multi_timeframe(self, four_h_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute crypto multi-timeframe features from 4H OHLCV bars.

        Aggregates 4H bars to daily for daily_trend_dir and daily_rsi_14.
        Computes 4H RSI-14 and 4H Price/SMA-20 ratio directly on 4H data.

        Args:
            four_h_ohlcv: 4H OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame with 4 columns aligned to 4H index.
        """
        # Aggregate 4H to daily
        daily = four_h_ohlcv.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        daily = daily.dropna(subset=["close"])

        # Daily SMA-10 trend direction
        daily_sma10 = daily["close"].rolling(10).mean()
        daily_trend = (daily["close"] > daily_sma10).astype(float)

        # Daily RSI-14 via stockstats
        daily_sdf = stockstats.StockDataFrame.retype(daily.copy())
        daily_rsi = daily_sdf["rsi_14"]

        daily_result = pd.DataFrame(
            {"daily_trend_dir": daily_trend, "daily_rsi_14": daily_rsi},
            index=daily.index,
        )

        # 4H indicators directly
        four_h_sdf = stockstats.StockDataFrame.retype(four_h_ohlcv.copy())
        four_h_rsi = four_h_sdf["rsi_14"]
        four_h_sma20 = four_h_sdf["close_20_sma"]
        four_h_price_ratio = four_h_ohlcv["close"] / four_h_sma20

        four_h_result = pd.DataFrame(
            {"four_h_rsi_14": four_h_rsi, "four_h_price_sma20_ratio": four_h_price_ratio},
            index=four_h_ohlcv.index,
        )

        # Reindex daily to 4H and forward-fill, then join with 4H features
        daily_on_4h = daily_result.reindex(four_h_ohlcv.index, method="ffill")
        result = pd.concat([daily_on_4h, four_h_result], axis=1)

        log.info(
            "crypto_multi_timeframe_computed",
            daily_bars=len(daily),
            four_h_rows=len(result),
        )
        return result

    def compute_derived(self, close: pd.Series) -> pd.DataFrame:
        """Compute derived features for HMM input and correlation analysis.

        These are intermediate features, NOT part of the observation vector.
        Includes log returns at multiple horizons and Bollinger Band width.

        Args:
            close: Close price Series with DatetimeIndex.

        Returns:
            DataFrame with log_return_1d, log_return_5d, log_return_20d,
            and bb_width columns.
        """
        result = pd.DataFrame(index=close.index)

        # Log returns at multiple horizons
        result["log_return_1d"] = np.log(close / close.shift(1))
        result["log_return_5d"] = np.log(close / close.shift(5))
        result["log_return_20d"] = np.log(close / close.shift(20))

        # Bollinger Band width from stockstats
        # Need OHLCV-like DataFrame for stockstats
        dummy_df = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(len(close)),
            },
            index=close.index,
        )
        sdf = stockstats.StockDataFrame.retype(dummy_df)
        result["bb_width"] = sdf["boll_ub"] - sdf["boll_lb"]

        log.info("derived_features_computed", rows=len(result))
        return result
