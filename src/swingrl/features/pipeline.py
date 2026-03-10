"""Feature pipeline orchestrator for SwingRL.

Orchestrates the full compute-normalize-store flow for both equity and crypto
environments. Produces observation vectors via the assembler.

Usage:
    from swingrl.features.pipeline import FeaturePipeline, compare_features

    pipeline = FeaturePipeline(config, duckdb_conn)
    features_df = pipeline.compute_equity()
    obs = pipeline.get_observation("equity", "2024-01-15")
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.features.assembler import (
    CRYPTO_PER_ASSET,
    ObservationAssembler,
    equity_per_asset_dim,
)
from swingrl.features.hmm_regime import HMMRegimeDetector
from swingrl.features.macro import MacroFeatureAligner
from swingrl.features.normalization import RollingZScoreNormalizer
from swingrl.features.technical import TechnicalIndicatorCalculator
from swingrl.features.turbulence import TurbulenceCalculator

log = structlog.get_logger(__name__)

# Equity feature columns stored in features_equity table
_EQUITY_FEATURE_COLS = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "weekly_trend_dir",
    "weekly_rsi_14",
    "pe_zscore",
    "earnings_growth",
    "debt_to_equity",
    "dividend_yield",
]

# Crypto feature columns stored in features_crypto table
_CRYPTO_FEATURE_COLS = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "daily_trend_dir",
    "daily_rsi_14",
    "four_h_rsi_14",
    "four_h_price_sma20_ratio",
]

# Macro feature column names as stored by MacroFeatureAligner
_MACRO_COLS = [
    "vix_zscore",
    "yield_curve_spread",
    "yield_curve_direction",
    "fed_funds_90d_change",
    "cpi_yoy",
    "unemployment_3m_direction",
]


class FeaturePipeline:
    """Orchestrates feature computation, normalization, and storage.

    Instantiates all feature modules and coordinates the full pipeline for
    both equity and crypto environments.
    """

    def __init__(self, config: SwingRLConfig, conn: Any) -> None:
        """Initialize pipeline with config and DuckDB connection.

        Args:
            config: Validated SwingRLConfig.
            conn: DuckDB connection for reading OHLCV and writing features.
        """
        self._config = config
        self._conn = conn

        # Feature modules
        self._technical = TechnicalIndicatorCalculator()
        self._macro = MacroFeatureAligner(conn)
        self._normalizer = RollingZScoreNormalizer(config)
        self._assembler = ObservationAssembler(config)

        # HMM detectors (one per environment)
        self._hmm_equity = HMMRegimeDetector("equity", config)
        self._hmm_crypto = HMMRegimeDetector("crypto", config)

        # Turbulence calculators
        self._turb_equity = TurbulenceCalculator("equity")
        self._turb_crypto = TurbulenceCalculator("crypto")

        self._equity_symbols = sorted(config.equity.symbols)
        self._crypto_symbols = sorted(config.crypto.symbols)

    def compute_equity(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Compute equity features: technical -> normalize -> store to DuckDB.

        Args:
            symbols: Override symbol list (default: config equity symbols).
            start: Start date filter (YYYY-MM-DD).
            end: End date filter (YYYY-MM-DD).

        Returns:
            DataFrame with computed features for all symbols.
        """
        symbols = symbols or self._equity_symbols

        all_features: list[pd.DataFrame] = []

        for symbol in symbols:
            ohlcv = self._read_equity_ohlcv(symbol, start, end)
            if ohlcv.empty:
                log.warning("no_ohlcv_data", symbol=symbol, environment="equity")
                continue

            # Compute technical indicators
            price_action = self._technical.compute_price_action(ohlcv)
            weekly = self._technical.compute_weekly_features(ohlcv)

            # Combine per-asset features
            features = pd.concat([price_action, weekly], axis=1)

            # Add stub fundamental columns (NaN — real fetching is Plan 02)
            for col in ["pe_zscore", "earnings_growth", "debt_to_equity", "dividend_yield"]:
                features[col] = 0.0

            features["symbol"] = symbol
            all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features)

        # Normalize numeric features per symbol
        normalized_parts: list[pd.DataFrame] = []
        for symbol in symbols:
            sym_data = combined[combined["symbol"] == symbol].drop(columns=["symbol"])
            numeric_cols = sym_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(sym_data) > 0 and len(numeric_cols) > 0:
                normalized = self._normalizer.normalize(sym_data[numeric_cols], "equity")
                normalized["symbol"] = symbol
                normalized_parts.append(normalized)

        if not normalized_parts:
            return pd.DataFrame()

        normalized_df = pd.concat(normalized_parts)

        # Store to features_equity
        self._store_equity_features(normalized_df)

        # Fit HMM on SPY
        spy_ohlcv = self._read_equity_ohlcv("SPY", start, end)
        if not spy_ohlcv.empty and len(spy_ohlcv) >= 100:
            try:
                self._hmm_equity.initial_fit(spy_ohlcv["close"])
                probs = self._hmm_equity.predict_proba(spy_ohlcv["close"])
                last_probs = probs[-1:]
                score = float(
                    self._hmm_equity._model.score(  # type: ignore[union-attr]
                        self._hmm_equity.compute_hmm_inputs(spy_ohlcv["close"])
                    )
                )
                self._hmm_equity.store_hmm_state(
                    self._conn, spy_ohlcv.index[-1].date(), last_probs, score
                )
            except (ValueError, RuntimeError):
                log.warning("hmm_equity_fit_failed")

        log.info(
            "equity_pipeline_complete",
            symbols=len(symbols),
            rows=len(normalized_df),
        )
        return normalized_df

    def compute_crypto(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Compute crypto features: technical -> normalize -> store to DuckDB.

        Args:
            symbols: Override symbol list (default: config crypto symbols).
            start: Start datetime filter.
            end: End datetime filter.

        Returns:
            DataFrame with computed features for all symbols.
        """
        symbols = symbols or self._crypto_symbols

        all_features: list[pd.DataFrame] = []

        for symbol in symbols:
            ohlcv = self._read_crypto_ohlcv(symbol, start, end)
            if ohlcv.empty:
                log.warning("no_ohlcv_data", symbol=symbol, environment="crypto")
                continue

            # Compute technical indicators
            price_action = self._technical.compute_price_action(ohlcv)
            multi_tf = self._technical.compute_crypto_multi_timeframe(ohlcv)

            # Combine per-asset features
            features = pd.concat([price_action, multi_tf], axis=1)
            features["symbol"] = symbol
            all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features)

        # Normalize numeric features per symbol
        normalized_parts: list[pd.DataFrame] = []
        for symbol in symbols:
            sym_data = combined[combined["symbol"] == symbol].drop(columns=["symbol"])
            numeric_cols = sym_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(sym_data) > 0 and len(numeric_cols) > 0:
                normalized = self._normalizer.normalize(sym_data[numeric_cols], "crypto")
                normalized["symbol"] = symbol
                normalized_parts.append(normalized)

        if not normalized_parts:
            return pd.DataFrame()

        normalized_df = pd.concat(normalized_parts)

        # Store to features_crypto
        self._store_crypto_features(normalized_df)

        # Fit HMM on BTC
        btc_ohlcv = self._read_crypto_ohlcv("BTCUSDT", start, end)
        if not btc_ohlcv.empty and len(btc_ohlcv) >= 100:
            try:
                self._hmm_crypto.initial_fit(btc_ohlcv["close"])
                probs = self._hmm_crypto.predict_proba(btc_ohlcv["close"])
                last_probs = probs[-1:]
                score = float(
                    self._hmm_crypto._model.score(  # type: ignore[union-attr]
                        self._hmm_crypto.compute_hmm_inputs(btc_ohlcv["close"])
                    )
                )
                last_date = btc_ohlcv.index[-1]
                dt = last_date.date() if hasattr(last_date, "date") else last_date
                self._hmm_crypto.store_hmm_state(self._conn, dt, last_probs, score)
            except (ValueError, RuntimeError):
                log.warning("hmm_crypto_fit_failed")

        log.info(
            "crypto_pipeline_complete",
            symbols=len(symbols),
            rows=len(normalized_df),
        )
        return normalized_df

    def get_observation(
        self,
        environment: Literal["equity", "crypto"],
        date_or_datetime: str,
    ) -> np.ndarray:
        """Assemble observation vector from stored features.

        Args:
            environment: "equity" or "crypto".
            date_or_datetime: Date/datetime string for feature lookup.

        Returns:
            (156,) for equity or (45,) for crypto observation vector.
        """
        if environment == "equity":
            return self._get_equity_observation(date_or_datetime)
        return self._get_crypto_observation(date_or_datetime)

    def _get_equity_observation(self, date_str: str) -> np.ndarray:
        """Build equity observation from stored features."""
        per_asset: dict[str, np.ndarray] = {}
        per_asset_size = equity_per_asset_dim(self._config.sentiment.enabled)

        for symbol in self._equity_symbols:
            row = self._conn.execute(
                """SELECT * FROM features_equity
                   WHERE symbol = ? AND date <= CAST(? AS DATE)
                   ORDER BY date DESC LIMIT 1""",
                [symbol, date_str],
            ).fetchdf()

            if row.empty:
                per_asset[symbol] = np.zeros(per_asset_size)
            else:
                vals = row[_EQUITY_FEATURE_COLS].values[0]
                per_asset[symbol] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        # Macro features — use last available
        macro = self._get_macro_array("equity", date_str)

        # HMM
        hmm = self._get_hmm_probs("equity", date_str)

        # Turbulence — compute on-the-fly from recent returns
        turb = self._compute_turbulence_equity(date_str)

        # Sentiment features — fetched when enabled
        sentiment: dict[str, tuple[float, float]] | None = None
        if self._config.sentiment.enabled:
            sentiment = get_sentiment_features(
                enabled=True,
                symbols=self._equity_symbols,
                alpaca_api_key=os.environ.get("APCA_API_KEY_ID", ""),
                alpaca_api_secret=os.environ.get("APCA_API_SECRET_KEY", ""),
                finnhub_api_key=self._config.sentiment.finnhub_api_key,
            )

        return self._assembler.assemble_equity(
            per_asset, macro, hmm, turb, sentiment_features=sentiment
        )

    def _get_crypto_observation(self, datetime_str: str) -> np.ndarray:
        """Build crypto observation from stored features."""
        per_asset: dict[str, np.ndarray] = {}

        for symbol in self._crypto_symbols:
            row = self._conn.execute(
                """SELECT * FROM features_crypto
                   WHERE symbol = ? AND datetime <= CAST(? AS TIMESTAMP)
                   ORDER BY datetime DESC LIMIT 1""",
                [symbol, datetime_str],
            ).fetchdf()

            if row.empty:
                per_asset[symbol] = np.zeros(CRYPTO_PER_ASSET)
            else:
                vals = row[_CRYPTO_FEATURE_COLS].values[0]
                per_asset[symbol] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        # Macro features
        macro = self._get_macro_array("crypto", datetime_str)

        # HMM
        hmm = self._get_hmm_probs("crypto", datetime_str)

        # Turbulence
        turb = self._compute_turbulence_crypto(datetime_str)

        # Overnight context — hours since 16:00 ET (simplified)
        overnight = 0.0

        return self._assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)

    def _get_macro_array(self, environment: str, date_str: str) -> np.ndarray:
        """Get macro features as (6,) array from stored macro data."""
        # Read latest macro values before the given date
        try:
            if environment == "equity":
                query = """
                    SELECT series_id, value FROM macro_features
                    WHERE date <= CAST(? AS DATE)
                    ORDER BY date DESC
                """
            else:
                query = """
                    SELECT series_id, value FROM macro_features
                    WHERE date <= CAST(? AS DATE)
                    ORDER BY date DESC
                """
            rows = self._conn.execute(query, [date_str]).fetchdf()

            if rows.empty:
                return np.zeros(6)

            # Get latest value for each series
            latest: dict[str, float] = {}
            for _, row in rows.iterrows():
                sid = str(row["series_id"])
                if sid not in latest:
                    latest[sid] = float(row["value"])

            # Map to macro array positions (simplified — actual macro alignment is richer)
            vix = latest.get("VIXCLS", 0.0)
            spread = latest.get("T10Y2Y", 0.0)
            direction = 1.0 if spread > 0 else 0.0
            fed = latest.get("DFF", 0.0)
            cpi = latest.get("CPIAUCSL", 0.0)
            unemp = latest.get("UNRATE", 0.0)

            return np.array([vix, spread, direction, fed, cpi, unemp])
        except Exception:
            log.warning("macro_fetch_failed", environment=environment, date=date_str)
            return np.zeros(6)

    def _get_hmm_probs(self, environment: str, date_str: str) -> np.ndarray:
        """Get HMM probabilities from stored state."""
        try:
            row = self._conn.execute(
                """SELECT p_bull, p_bear FROM hmm_state_history
                   WHERE environment = ? AND date <= CAST(? AS DATE)
                   ORDER BY date DESC LIMIT 1""",
                [environment, date_str],
            ).fetchdf()

            if row.empty:
                return np.array([0.5, 0.5])

            return np.array([float(row["p_bull"].iloc[0]), float(row["p_bear"].iloc[0])])
        except Exception:
            return np.array([0.5, 0.5])

    def _compute_turbulence_equity(self, date_str: str) -> float:
        """Compute equity turbulence from recent returns."""
        try:
            returns_df = self._conn.execute(
                """SELECT symbol, date, close FROM ohlcv_daily
                   WHERE date <= CAST(? AS DATE)
                   ORDER BY date""",
                [date_str],
            ).fetchdf()

            if returns_df.empty:
                return 0.0

            pivot = returns_df.pivot_table(index="date", columns="symbol", values="close")
            log_returns = np.log(pivot / pivot.shift(1)).dropna()

            if len(log_returns) < self._turb_equity.min_warmup + 1:
                return 0.0

            return self._turb_equity.compute(log_returns.values, len(log_returns) - 1)
        except Exception:
            log.warning("turbulence_equity_failed")
            return 0.0

    def _compute_turbulence_crypto(self, datetime_str: str) -> float:
        """Compute crypto turbulence from recent returns."""
        try:
            returns_df = self._conn.execute(
                """SELECT symbol, datetime, close FROM ohlcv_4h
                   WHERE datetime <= CAST(? AS TIMESTAMP)
                   ORDER BY datetime""",
                [datetime_str],
            ).fetchdf()

            if returns_df.empty:
                return 0.0

            pivot = returns_df.pivot_table(index="datetime", columns="symbol", values="close")
            log_returns = np.log(pivot / pivot.shift(1)).dropna()

            if len(log_returns) < self._turb_crypto.min_warmup + 1:
                return 0.0

            return self._turb_crypto.compute(log_returns.values, len(log_returns) - 1)
        except Exception:
            log.warning("turbulence_crypto_failed")
            return 0.0

    def _read_equity_ohlcv(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Read equity OHLCV from DuckDB."""
        conditions = ["symbol = ?"]
        params: list[Any] = [symbol]

        if start:
            conditions.append("date >= CAST(? AS DATE)")
            params.append(start)
        if end:
            conditions.append("date <= CAST(? AS DATE)")
            params.append(end)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM ohlcv_daily WHERE {where} ORDER BY date"  # nosec B608

        result: pd.DataFrame = self._conn.execute(query, params).fetchdf()
        if not result.empty and "date" in result.columns:
            result = result.set_index("date")
            result.index = pd.DatetimeIndex(result.index, tz="UTC")
        return result

    def _read_crypto_ohlcv(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        """Read crypto OHLCV from DuckDB."""
        conditions = ["symbol = ?"]
        params: list[Any] = [symbol]

        if start:
            conditions.append("datetime >= CAST(? AS TIMESTAMP)")
            params.append(start)
        if end:
            conditions.append("datetime <= CAST(? AS TIMESTAMP)")
            params.append(end)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM ohlcv_4h WHERE {where} ORDER BY datetime"  # nosec B608

        result: pd.DataFrame = self._conn.execute(query, params).fetchdf()
        if not result.empty and "datetime" in result.columns:
            result = result.set_index("datetime")
            dt_index = pd.DatetimeIndex(result.index)
            if dt_index.tz is None:
                dt_index = dt_index.tz_localize("UTC")
            result.index = dt_index
        return result

    def _store_equity_features(self, features: pd.DataFrame) -> None:
        """Store normalized equity features to DuckDB via replacement scan."""
        if features.empty:
            return

        store_df = features.copy()
        # Ensure date column from index
        if "date" not in store_df.columns:
            store_df["date"] = store_df.index

        # Select only expected columns
        available_cols = [c for c in _EQUITY_FEATURE_COLS if c in store_df.columns]
        select_cols = ["symbol", "date"] + available_cols
        store_df = store_df[select_cols].copy()

        # Replace NaN with None for DuckDB
        sync_df = store_df  # noqa: F841
        try:
            self._conn.execute(
                f"""INSERT OR REPLACE INTO features_equity
                    SELECT symbol, date, {", ".join(available_cols)}
                    FROM sync_df"""  # nosec B608
            )
            log.info("equity_features_stored", rows=len(store_df))
        except Exception:
            log.error("equity_features_store_failed")

    def _store_crypto_features(self, features: pd.DataFrame) -> None:
        """Store normalized crypto features to DuckDB via replacement scan."""
        if features.empty:
            return

        store_df = features.copy()
        if "datetime" not in store_df.columns:
            store_df["datetime"] = store_df.index

        available_cols = [c for c in _CRYPTO_FEATURE_COLS if c in store_df.columns]
        select_cols = ["symbol", "datetime"] + available_cols
        store_df = store_df[select_cols].copy()

        sync_df = store_df  # noqa: F841
        try:
            self._conn.execute(
                f"""INSERT OR REPLACE INTO features_crypto
                    SELECT symbol, datetime, {", ".join(available_cols)}
                    FROM sync_df"""  # nosec B608
            )
            log.info("crypto_features_stored", rows=len(store_df))
        except Exception:
            log.error("crypto_features_store_failed")


def get_sentiment_features(
    *,
    enabled: bool,
    symbols: list[str],
    alpaca_api_key: str = "",
    alpaca_api_secret: str = "",
    finnhub_api_key: str = "",
    max_headlines_per_asset: int = 10,
    model_name: str = "ProsusAI/finbert",
) -> dict[str, tuple[float, float]]:
    """Get sentiment features for symbols when enabled.

    Returns empty dict when disabled. When enabled, fetches headlines and scores
    them with FinBERT, returning (sentiment_score, confidence) per symbol.
    Failures produce (0.0, 0.0) defaults -- never crashes the pipeline.

    Args:
        enabled: Whether sentiment features are active.
        symbols: List of ticker symbols.
        alpaca_api_key: Alpaca API key for news access.
        alpaca_api_secret: Alpaca API secret for news access.
        finnhub_api_key: Finnhub API key (fallback).
        max_headlines_per_asset: Max headlines per symbol.
        model_name: HuggingFace model identifier for FinBERT.

    Returns:
        Dict of {symbol: (sentiment_score, confidence)}. Empty dict if disabled.
    """
    if not enabled:
        return {}

    from swingrl.config.schema import SentimentConfig
    from swingrl.sentiment.finbert import FinBERTScorer
    from swingrl.sentiment.news_fetcher import NewsFetcher

    config = SentimentConfig(
        enabled=True,
        model_name=model_name,
        max_headlines_per_asset=max_headlines_per_asset,
        finnhub_api_key=finnhub_api_key,
    )
    scorer = FinBERTScorer(model_name=model_name)
    fetcher = NewsFetcher(
        config,
        alpaca_api_key=alpaca_api_key,
        alpaca_api_secret=alpaca_api_secret,
    )

    result: dict[str, tuple[float, float]] = {}
    for symbol in symbols:
        try:
            headlines = fetcher.fetch_headlines(symbol)
            if not headlines:
                result[symbol] = (0.0, 0.0)
                continue
            scores = scorer.score_headlines(headlines)
            sentiment_score, confidence = scorer.aggregate_sentiment(scores)
            result[symbol] = (sentiment_score, confidence)
        except Exception:
            log.warning("sentiment_feature_failed", symbol=symbol)
            result[symbol] = (0.0, 0.0)

    return result


def compare_features(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare baseline vs candidate feature sets for A/B testing.

    Accepts when validation Sharpe improves by >= threshold.
    Rejects when train Sharpe improves but validation Sharpe decreases (overfitting).

    Args:
        baseline_metrics: Dict with train_sharpe and validation_sharpe.
        candidate_metrics: Dict with train_sharpe and validation_sharpe.
        threshold: Minimum validation Sharpe improvement to accept.

    Returns:
        Dict with accepted (bool), sharpe_improvement (float), reason (str).
    """
    val_improvement = candidate_metrics["validation_sharpe"] - baseline_metrics["validation_sharpe"]
    train_improvement = candidate_metrics["train_sharpe"] - baseline_metrics["train_sharpe"]

    # Overfitting guard: train improves but validation decreases
    if train_improvement > 0 and val_improvement < 0:
        return {
            "accepted": False,
            "sharpe_improvement": val_improvement,
            "reason": (
                f"Overfitting detected: train Sharpe improved by {train_improvement:.4f} "
                f"but validation Sharpe decreased by {abs(val_improvement):.4f}"
            ),
        }

    if val_improvement >= threshold:
        return {
            "accepted": True,
            "sharpe_improvement": val_improvement,
            "reason": (
                f"Validation Sharpe improved by {val_improvement:.4f} (>= threshold {threshold})"
            ),
        }

    return {
        "accepted": False,
        "sharpe_improvement": val_improvement,
        "reason": (
            f"Validation Sharpe improvement {val_improvement:.4f} below threshold {threshold}"
        ),
    }
