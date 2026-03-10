"""Tests for FinBERT sentiment scoring and news headline fetching.

HARD-03: FinBERT lazy-loads model and scores headlines.
HARD-04: News fetcher retrieves from Alpaca with Finnhub fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from swingrl.config.schema import SentimentConfig
from swingrl.sentiment.finbert import FinBERTScorer
from swingrl.sentiment.news_fetcher import NewsFetcher

# ---------------------------------------------------------------------------
# FinBERTScorer tests
# ---------------------------------------------------------------------------


def _make_mock_transformers(logits_array: np.ndarray) -> MagicMock:
    """Create a mock transformers module with tokenizer and model."""
    mock_transformers = MagicMock()

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    # Mock model
    mock_model = MagicMock()
    mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
    mock_logits = MagicMock()
    mock_logits.detach.return_value.cpu.return_value.numpy.return_value = logits_array
    mock_model.return_value.logits = mock_logits

    return mock_transformers


class TestFinBERTScorerLazyLoading:
    """HARD-03: FinBERT model is lazy-loaded only when sentiment.enabled=True."""

    def test_model_not_loaded_at_init(self) -> None:
        """Model and tokenizer are None at construction time."""
        scorer = FinBERTScorer()
        assert scorer._model is None
        assert scorer._tokenizer is None

    def test_model_loaded_on_first_score(self) -> None:
        """Model loads lazily on the first score_headlines call."""
        mock_tf = _make_mock_transformers(np.array([[0.5, -0.3, 0.1]]))

        with patch("swingrl.sentiment.finbert._transformers", mock_tf):
            scorer = FinBERTScorer()
            assert scorer._model is None
            scorer.score_headlines(["Test headline"])
            assert scorer._model is not None


class TestFinBERTScorerScoring:
    """HARD-03: score_headlines returns sentiment scores summing to ~1.0."""

    def test_score_headlines_returns_correct_format(self) -> None:
        """Each result has positive, negative, neutral keys summing to ~1.0."""
        # Logits for 2 headlines: first positive, second negative
        mock_tf = _make_mock_transformers(np.array([[2.0, -1.0, 0.5], [-1.0, 2.0, 0.3]]))

        with patch("swingrl.sentiment.finbert._transformers", mock_tf):
            scorer = FinBERTScorer()
            results = scorer.score_headlines(["Good news", "Bad news"])

        assert len(results) == 2
        for result in results:
            assert "positive" in result
            assert "negative" in result
            assert "neutral" in result
            total = result["positive"] + result["negative"] + result["neutral"]
            assert abs(total - 1.0) < 1e-5

    def test_score_headlines_empty_list(self) -> None:
        """Empty headline list returns empty result list (no model load)."""
        scorer = FinBERTScorer()
        results = scorer.score_headlines([])
        assert results == []
        assert scorer._model is None  # Model not loaded for empty input


class TestFinBERTScorerAggregation:
    """HARD-03: aggregate_sentiment computes mean sentiment_score and confidence."""

    def test_aggregate_sentiment_basic(self) -> None:
        """Mean sentiment_score = mean(positive - negative), confidence = mean(max(probs))."""
        scorer = FinBERTScorer()
        scores = [
            {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
            {"positive": 0.3, "negative": 0.5, "neutral": 0.2},
        ]
        sentiment_score, confidence = scorer.aggregate_sentiment(scores)

        # Mean of (0.7-0.1, 0.3-0.5) = mean(0.6, -0.2) = 0.2
        assert abs(sentiment_score - 0.2) < 1e-5
        # Mean of (max(0.7,0.1,0.2), max(0.3,0.5,0.2)) = mean(0.7, 0.5) = 0.6
        assert abs(confidence - 0.6) < 1e-5

    def test_aggregate_sentiment_empty(self) -> None:
        """Empty scores return (0.0, 0.0) defaults."""
        scorer = FinBERTScorer()
        sentiment_score, confidence = scorer.aggregate_sentiment([])
        assert sentiment_score == 0.0
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# NewsFetcher tests
# ---------------------------------------------------------------------------


class TestNewsFetcherBasic:
    """HARD-04: NewsFetcher retrieves headlines from Alpaca with Finnhub fallback."""

    @patch("swingrl.sentiment.news_fetcher.requests")
    def test_fetch_headlines_alpaca_success(self, mock_requests: MagicMock) -> None:
        """Primary Alpaca fetch returns headline strings."""
        config = SentimentConfig(enabled=True, max_headlines_per_asset=5)
        fetcher = NewsFetcher(config, alpaca_api_key="key", alpaca_api_secret="secret")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "news": [
                {"headline": "SPY hits all-time high"},
                {"headline": "Market rallies on data"},
            ]
        }
        mock_requests.get.return_value = mock_resp

        headlines = fetcher.fetch_headlines("SPY")
        assert isinstance(headlines, list)
        assert len(headlines) == 2
        assert headlines[0] == "SPY hits all-time high"

    @patch("swingrl.sentiment.news_fetcher.requests")
    def test_fetch_headlines_finnhub_fallback(self, mock_requests: MagicMock) -> None:
        """Falls back to Finnhub when Alpaca fails."""
        config = SentimentConfig(enabled=True, max_headlines_per_asset=5, finnhub_api_key="fk")
        fetcher = NewsFetcher(config, alpaca_api_key="key", alpaca_api_secret="secret")

        # Alpaca fails
        alpaca_resp = MagicMock()
        alpaca_resp.status_code = 500
        alpaca_resp.raise_for_status.side_effect = ConnectionError("Alpaca down")

        # Finnhub succeeds
        finnhub_resp = MagicMock()
        finnhub_resp.status_code = 200
        finnhub_resp.json.return_value = [
            {"headline": "Finnhub headline 1"},
            {"headline": "Finnhub headline 2"},
        ]

        mock_requests.get.side_effect = [
            alpaca_resp,  # Alpaca call fails
            finnhub_resp,  # Finnhub call succeeds
        ]

        headlines = fetcher.fetch_headlines("SPY")
        assert len(headlines) == 2
        assert "Finnhub headline 1" in headlines

    @patch("swingrl.sentiment.news_fetcher.requests")
    def test_fetch_headlines_respects_max_limit(self, mock_requests: MagicMock) -> None:
        """Respects max_headlines_per_asset limit."""
        config = SentimentConfig(enabled=True, max_headlines_per_asset=2)
        fetcher = NewsFetcher(config, alpaca_api_key="key", alpaca_api_secret="secret")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"news": [{"headline": f"Headline {i}"} for i in range(10)]}
        mock_requests.get.return_value = mock_resp

        headlines = fetcher.fetch_headlines("SPY")
        assert len(headlines) == 2

    @patch("swingrl.sentiment.news_fetcher.requests")
    def test_fetch_headlines_both_fail(self, mock_requests: MagicMock) -> None:
        """Returns empty list when both sources fail (no crash)."""
        config = SentimentConfig(enabled=True, finnhub_api_key="fk")
        fetcher = NewsFetcher(config, alpaca_api_key="key", alpaca_api_secret="secret")

        mock_requests.get.side_effect = ConnectionError("Network down")

        headlines = fetcher.fetch_headlines("SPY")
        assert headlines == []
