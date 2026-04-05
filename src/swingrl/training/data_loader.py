"""Training data loader — loads features and prices from PostgreSQL.

Assembles observation matrices using ObservationAssembler so the output
shapes match the SB3 environment observation_space dimensions:
- equity: (N, 164) features without sentiment, (N, 180) with sentiment, (N, 8) prices
- crypto: (N, 47) features, (N, 2) prices

Usage:
    from swingrl.training.data_loader import load_features_prices

    with db.connection() as conn:
        features, prices = load_features_prices(conn, "equity", config)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.data.pg_helpers import fetchdf
from swingrl.features.assembler import ObservationAssembler
from swingrl.features.pipeline import _CRYPTO_FEATURE_COLS, _EQUITY_FEATURE_COLS
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

# Macro series IDs to (6,) array position mapping
_MACRO_SERIES_IDS = ["VIXCLS", "T10Y2Y", "DFF", "CPIAUCSL", "UNRATE"]


def _get_macro_array(conn: Any, date_str: str) -> np.ndarray:
    """Fetch macro features as (6,) array from macro_features table.

    Args:
        conn: PostgreSQL connection.
        date_str: ISO date string (YYYY-MM-DD).

    Returns:
        (6,) float64 array.
    """
    try:
        cur = conn.execute(
            """
            SELECT series_id, value FROM macro_features
            WHERE date <= %s::DATE
            ORDER BY date DESC
            """,
            [date_str],
        )
        rows = fetchdf(cur)

        if rows.empty:
            return np.zeros(6)

        latest: dict[str, float] = {}
        for _, row in rows.iterrows():
            sid = str(row["series_id"])
            if sid not in latest:
                latest[sid] = float(row["value"])

        vix = latest.get("VIXCLS", 0.0)
        spread = latest.get("T10Y2Y", 0.0)
        direction = 1.0 if spread > 0 else 0.0
        fed = latest.get("DFF", 0.0)
        cpi = latest.get("CPIAUCSL", 0.0)
        unemp = latest.get("UNRATE", 0.0)
        return np.array([vix, spread, direction, fed, cpi, unemp])
    except Exception:
        log.warning("macro_fetch_failed", date=date_str)
        return np.zeros(6)


def _get_hmm_probs(conn: Any, environment: str, date_str: str) -> np.ndarray:
    """Fetch HMM regime probabilities as (2,) array from hmm_state_history.

    Args:
        conn: PostgreSQL connection.
        environment: "equity" or "crypto".
        date_str: ISO date/datetime string.

    Returns:
        (2,) float64 array [p_bull, p_bear].
    """
    try:
        cur = conn.execute(
            """
            SELECT p_bull, p_bear FROM hmm_state_history
            WHERE environment = %s AND date <= %s::DATE
            ORDER BY date DESC
            LIMIT 1
            """,
            [environment, date_str],
        )
        row = fetchdf(cur)

        if row.empty:
            return np.array([0.5, 0.5])
        return np.array([float(row["p_bull"].iloc[0]), float(row["p_bear"].iloc[0])])
    except Exception:
        return np.array([0.5, 0.5])


def load_features_prices(
    conn: Any,
    env_name: str,
    config: SwingRLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and prices from PostgreSQL for the given environment.

    Assembles observation matrices using ObservationAssembler so the output
    shapes match the SB3 environment observation_space dimensions.

    Args:
        conn: Active PostgreSQL connection.
        env_name: Environment name ("equity" or "crypto").
        config: Validated SwingRLConfig for symbol lists and sentiment flag.

    Returns:
        Tuple of (features_array, prices_array) both float32.

    Raises:
        DataError: If no data found in the feature table.
    """
    assembler = ObservationAssembler(config)

    if env_name == "equity":
        return _load_equity(conn, config, assembler)
    return _load_crypto(conn, config, assembler)


def _load_equity(
    conn: Any,
    config: SwingRLConfig,
    assembler: ObservationAssembler,
) -> tuple[np.ndarray, np.ndarray]:
    """Load equity features and prices from PostgreSQL.

    Args:
        conn: PostgreSQL connection.
        config: SwingRLConfig with equity symbol list.
        assembler: ObservationAssembler initialized from config.

    Returns:
        Tuple of ((N, obs_dim) features, (N, n_symbols) prices) float32.
        obs_dim is 164 without sentiment, 180 with sentiment (8 symbols).

    Raises:
        DataError: If no data found in features_equity table.
    """
    equity_symbols = sorted(config.equity.symbols)
    per_asset_size = 15  # EQUITY_PER_ASSET_BASE

    cur = conn.execute(
        """
        SELECT
            f.date,
            f.symbol,
            {feat_cols}
        FROM features_equity f
        INNER JOIN (
            SELECT DISTINCT date FROM ohlcv_daily
        ) o ON f.date = o.date
        ORDER BY f.date, f.symbol
        """.format(feat_cols=", ".join(f"f.{c}" for c in _EQUITY_FEATURE_COLS))  # nosec B608
    )
    feat_df = fetchdf(cur)

    if feat_df.empty:
        raise DataError("No data found in features_equity table")

    cur = conn.execute(
        """
        SELECT date, symbol, close
        FROM ohlcv_daily
        WHERE symbol IN ({sym_list})
        ORDER BY date, symbol
        """.format(sym_list=", ".join(f"'{s}'" for s in equity_symbols))  # nosec B608
    )
    prices_df = fetchdf(cur)

    feat_dates = sorted(feat_df["date"].unique())

    macro_map: dict[str, np.ndarray] = {}
    hmm_map: dict[str, np.ndarray] = {}
    for d in feat_dates:
        date_str = str(d)[:10]
        macro_map[date_str] = _get_macro_array(conn, date_str)
        hmm_map[date_str] = _get_hmm_probs(conn, "equity", date_str)

    prices_pivot = prices_df.pivot_table(index="date", columns="symbol", values="close")
    for sym in equity_symbols:
        if sym not in prices_pivot.columns:
            prices_pivot[sym] = 0.0
    prices_pivot = prices_pivot[equity_symbols]

    # Fetch live sentiment if enabled (FinBERT + Alpaca/Finnhub headlines).
    # Computed once — same current-day scores applied to all historical dates.
    sentiment_data: dict[str, tuple[float, float]] | None = None
    if config.sentiment.enabled:
        from swingrl.features.pipeline import get_sentiment_features  # noqa: PLC0415

        sentiment_data = get_sentiment_features(
            enabled=True,
            symbols=equity_symbols,
            alpaca_api_key=os.environ.get("APCA_API_KEY_ID", ""),
            alpaca_api_secret=os.environ.get("APCA_API_SECRET_KEY", ""),
            finnhub_api_key=config.sentiment.finnhub_api_key,
        )

    obs_rows: list[np.ndarray] = []
    price_rows: list[np.ndarray] = []

    for date_val in feat_dates:
        date_str = str(date_val)[:10]
        sym_group = feat_df[feat_df["date"] == date_val]

        per_asset: dict[str, np.ndarray] = {}
        for sym in equity_symbols:
            sym_row = sym_group[sym_group["symbol"] == sym]
            if sym_row.empty:
                per_asset[sym] = np.zeros(per_asset_size)
            else:
                vals = sym_row[_EQUITY_FEATURE_COLS].values[0]
                per_asset[sym] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        macro = macro_map[date_str]
        hmm = hmm_map[date_str]

        obs = assembler.assemble_equity(
            per_asset_features=per_asset,
            macro=macro,
            hmm_probs=hmm,
            turbulence=0.0,
            portfolio_state=None,
            sentiment_features=sentiment_data,
        )
        obs_rows.append(obs)

        if date_val in prices_pivot.index:
            price_row = prices_pivot.loc[date_val].values.astype(np.float32)
        else:
            price_row = np.ones(len(equity_symbols), dtype=np.float32)
        price_rows.append(price_row)

    features = np.array(obs_rows, dtype=np.float32)
    prices = np.array(price_rows, dtype=np.float32)

    log.info(
        "data_loaded",
        env_name="equity",
        rows=len(feat_dates),
        obs_dim=features.shape[1],
        n_symbols=len(equity_symbols),
    )

    return features, prices


def _load_crypto(
    conn: Any,
    config: SwingRLConfig,
    assembler: ObservationAssembler,
) -> tuple[np.ndarray, np.ndarray]:
    """Load crypto features and prices from PostgreSQL.

    Args:
        conn: PostgreSQL connection.
        config: SwingRLConfig with crypto symbol list.
        assembler: ObservationAssembler initialized from config.

    Returns:
        Tuple of ((N, 47) features, (N, n_symbols) prices) float32.

    Raises:
        DataError: If no data found in features_crypto table.
    """
    crypto_symbols = sorted(config.crypto.symbols)
    per_asset_size = 13  # CRYPTO_PER_ASSET

    cur = conn.execute(
        """
        SELECT
            f.datetime,
            f.symbol,
            {feat_cols}
        FROM features_crypto f
        INNER JOIN (
            SELECT DISTINCT datetime FROM ohlcv_4h
        ) o ON f.datetime = o.datetime
        ORDER BY f.datetime, f.symbol
        """.format(feat_cols=", ".join(f"f.{c}" for c in _CRYPTO_FEATURE_COLS))  # nosec B608
    )
    feat_df = fetchdf(cur)

    if feat_df.empty:
        raise DataError("No data found in features_crypto table")

    cur = conn.execute(
        """
        SELECT datetime, symbol, close
        FROM ohlcv_4h
        WHERE symbol IN ({sym_list})
        ORDER BY datetime, symbol
        """.format(sym_list=", ".join(f"'{s}'" for s in crypto_symbols))  # nosec B608
    )
    prices_df = fetchdf(cur)

    feat_datetimes = sorted(feat_df["datetime"].unique())

    macro_map: dict[str, np.ndarray] = {}
    hmm_map: dict[str, np.ndarray] = {}
    for dt in feat_datetimes:
        date_str = str(dt)[:10]
        macro_map[str(dt)] = _get_macro_array(conn, date_str)
        hmm_map[str(dt)] = _get_hmm_probs(conn, "crypto", date_str)

    prices_pivot = prices_df.pivot_table(index="datetime", columns="symbol", values="close")
    for sym in crypto_symbols:
        if sym not in prices_pivot.columns:
            prices_pivot[sym] = 0.0
    prices_pivot = prices_pivot[crypto_symbols]

    obs_rows: list[np.ndarray] = []
    price_rows: list[np.ndarray] = []

    for dt_val in feat_datetimes:
        dt_str = str(dt_val)
        sym_group = feat_df[feat_df["datetime"] == dt_val]

        per_asset: dict[str, np.ndarray] = {}
        for sym in crypto_symbols:
            sym_row = sym_group[sym_group["symbol"] == sym]
            if sym_row.empty:
                per_asset[sym] = np.zeros(per_asset_size)
            else:
                vals = sym_row[_CRYPTO_FEATURE_COLS].values[0]
                per_asset[sym] = np.nan_to_num(np.array(vals, dtype=float), nan=0.0)

        macro = macro_map[dt_str]
        hmm = hmm_map[dt_str]

        obs = assembler.assemble_crypto(
            per_asset_features=per_asset,
            macro=macro,
            hmm_probs=hmm,
            turbulence=0.0,
            overnight_context=0.0,
            portfolio_state=None,
        )
        obs_rows.append(obs)

        if dt_val in prices_pivot.index:
            price_row = prices_pivot.loc[dt_val].values.astype(np.float32)
        else:
            price_row = np.ones(len(crypto_symbols), dtype=np.float32)
        price_rows.append(price_row)

    features = np.array(obs_rows, dtype=np.float32)
    prices = np.array(price_rows, dtype=np.float32)

    log.info(
        "data_loaded",
        env_name="crypto",
        rows=len(feat_datetimes),
        obs_dim=features.shape[1],
        n_symbols=len(crypto_symbols),
    )

    return features, prices
