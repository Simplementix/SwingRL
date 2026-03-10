"""News headline fetcher with Alpaca primary and Finnhub fallback.

Retrieves financial news headlines for sentiment scoring. Uses Alpaca News API
as primary source, falling back to Finnhub when Alpaca fails.

Usage:
    from swingrl.sentiment.news_fetcher import NewsFetcher

    fetcher = NewsFetcher(config, alpaca_api_key="...", alpaca_api_secret="...")
    headlines = fetcher.fetch_headlines("SPY")
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import requests
import structlog

from swingrl.config.schema import SentimentConfig

log = structlog.get_logger(__name__)

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"


class NewsFetcher:
    """Fetches news headlines from Alpaca (primary) with Finnhub fallback.

    Respects max_headlines_per_asset config limit. Returns empty list on
    total failure (never crashes).
    """

    def __init__(  # nosec B107
        self,
        config: SentimentConfig,
        alpaca_api_key: str = "",
        alpaca_api_secret: str = "",
    ) -> None:
        """Initialize with sentiment config and API credentials.

        Args:
            config: SentimentConfig with max_headlines_per_asset and finnhub_api_key.
            alpaca_api_key: Alpaca API key for news access.
            alpaca_api_secret: Alpaca API secret for news access.
        """
        self._config = config
        self._alpaca_api_key = alpaca_api_key
        self._alpaca_api_secret = alpaca_api_secret

    def fetch_headlines(self, symbol: str, lookback_hours: int = 24) -> list[str]:
        """Fetch headlines for a symbol, Alpaca first then Finnhub fallback.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "BTCUSDT").
            lookback_hours: How far back to look for news.

        Returns:
            List of headline strings, limited to max_headlines_per_asset.
            Empty list if both sources fail.
        """
        limit = self._config.max_headlines_per_asset

        # Try Alpaca first
        headlines = self._fetch_alpaca(symbol, lookback_hours, limit)
        if headlines:
            return headlines[:limit]

        # Fallback to Finnhub
        headlines = self._fetch_finnhub(symbol, lookback_hours, limit)
        if headlines:
            return headlines[:limit]

        log.warning("all_news_sources_failed", symbol=symbol)
        return []

    def fetch_all_assets(self, symbols: list[str]) -> dict[str, list[str]]:
        """Fetch headlines for all symbols.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping symbol to list of headline strings.
        """
        result: dict[str, list[str]] = {}
        for symbol in symbols:
            result[symbol] = self.fetch_headlines(symbol)
        return result

    def _fetch_alpaca(self, symbol: str, lookback_hours: int, limit: int) -> list[str]:
        """Fetch from Alpaca News API.

        Args:
            symbol: Ticker symbol.
            lookback_hours: Hours to look back.
            limit: Maximum headlines to return.

        Returns:
            List of headline strings, or empty list on failure.
        """
        if not self._alpaca_api_key:
            return []

        try:
            now = datetime.now(tz=UTC)
            start = now - timedelta(hours=lookback_hours)

            resp = requests.get(
                _ALPACA_NEWS_URL,
                headers={
                    "APCA-API-KEY-ID": self._alpaca_api_key,
                    "APCA-API-SECRET-KEY": self._alpaca_api_secret,
                },
                params={
                    "symbols": symbol,
                    "start": start.isoformat(),
                    "end": now.isoformat(),
                    "limit": str(limit),
                    "sort": "desc",
                },
                timeout=10,
            )
            resp.raise_for_status()

            data = resp.json()
            news_items = data.get("news", [])
            return [item["headline"] for item in news_items if "headline" in item]
        except Exception:
            log.warning("alpaca_news_fetch_failed", symbol=symbol)
            return []

    def _fetch_finnhub(self, symbol: str, lookback_hours: int, limit: int) -> list[str]:
        """Fetch from Finnhub News API.

        Args:
            symbol: Ticker symbol.
            lookback_hours: Hours to look back.
            limit: Maximum headlines to return.

        Returns:
            List of headline strings, or empty list on failure.
        """
        if not self._config.finnhub_api_key:
            return []

        try:
            now = datetime.now(tz=UTC)
            start = now - timedelta(hours=lookback_hours)

            resp = requests.get(
                _FINNHUB_NEWS_URL,
                params={
                    "symbol": symbol,
                    "from": start.strftime("%Y-%m-%d"),
                    "to": now.strftime("%Y-%m-%d"),
                    "token": self._config.finnhub_api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()

            news_items = resp.json()
            return [item["headline"] for item in news_items[:limit] if "headline" in item]
        except Exception:
            log.warning("finnhub_news_fetch_failed", symbol=symbol)
            return []
