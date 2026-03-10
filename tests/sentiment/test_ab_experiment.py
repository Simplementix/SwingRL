"""Tests for A/B experiment Sharpe threshold comparison and sentiment integration.

HARD-03: Sentiment features conditionally added to observation vector.
HARD-04: A/B comparison uses +0.05 Sharpe improvement threshold.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from swingrl.features.pipeline import compare_features


class TestABExperimentSharpeThreshold:
    """HARD-04: A/B comparison rejects sentiment if Sharpe improvement < 0.05."""

    def test_keep_sentiment_when_sharpe_improvement_above_threshold(self) -> None:
        """Sharpe improvement >= 0.05 recommends keeping sentiment."""
        baseline = {"train_sharpe": 0.65, "validation_sharpe": 0.70}
        candidate = {"train_sharpe": 0.75, "validation_sharpe": 0.80}

        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is True
        assert result["sharpe_improvement"] == pytest.approx(0.10)

    def test_discard_sentiment_when_sharpe_improvement_below_threshold(self) -> None:
        """Sharpe improvement < 0.05 recommends disabling sentiment."""
        baseline = {"train_sharpe": 0.65, "validation_sharpe": 0.70}
        candidate = {"train_sharpe": 0.67, "validation_sharpe": 0.72}

        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is False
        assert result["sharpe_improvement"] == pytest.approx(0.02)

    def test_reject_overfitting_sentiment(self) -> None:
        """Train improves but validation decreases -> reject (overfitting)."""
        baseline = {"train_sharpe": 0.65, "validation_sharpe": 0.70}
        candidate = {"train_sharpe": 0.80, "validation_sharpe": 0.68}

        result = compare_features(baseline, candidate, threshold=0.05)
        assert result["accepted"] is False
        assert "Overfitting" in result["reason"]


class TestSentimentFeatureIntegration:
    """HARD-03: Sentiment features conditionally added to observation vector."""

    def test_sentiment_disabled_observation_unchanged(self) -> None:
        """When sentiment.enabled=False, observation dimensions are standard."""
        from swingrl.features.pipeline import get_sentiment_features

        # sentiment disabled returns empty array
        result = get_sentiment_features(enabled=False, symbols=["SPY"])
        assert result == {}

    def test_sentiment_enabled_adds_features(self) -> None:
        """When sentiment.enabled=True, returns 2 features per asset."""
        from swingrl.features.pipeline import get_sentiment_features

        mock_tf = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_model = MagicMock()
        mock_tf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
        mock_logits = MagicMock()
        mock_logits.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.7, 0.1, 0.2]]
        )
        mock_model.return_value.logits = mock_logits

        with (
            patch("swingrl.sentiment.finbert._transformers", mock_tf),
            patch("swingrl.sentiment.news_fetcher.requests") as mock_requests,
        ):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"news": [{"headline": "SPY rallies"}]}
            mock_requests.get.return_value = mock_resp

            result = get_sentiment_features(
                enabled=True,
                symbols=["SPY"],
                alpaca_api_key="key",  # pragma: allowlist secret
                alpaca_api_secret="secret",  # pragma: allowlist secret
            )

        assert "SPY" in result
        assert len(result["SPY"]) == 2  # (sentiment_score, confidence)

    def test_sentiment_nan_safe_defaults(self) -> None:
        """Missing headlines produce (0.0, 0.0) defaults."""
        from swingrl.features.pipeline import get_sentiment_features

        with patch("swingrl.sentiment.news_fetcher.requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("No network")

            result = get_sentiment_features(
                enabled=True,
                symbols=["SPY"],
                alpaca_api_key="key",  # pragma: allowlist secret
                alpaca_api_secret="secret",  # pragma: allowlist secret
            )

        assert "SPY" in result
        assert result["SPY"] == (0.0, 0.0)
