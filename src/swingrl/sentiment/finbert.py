"""FinBERT sentiment scorer with lazy model loading.

Lazy-loads the ProsusAI/finbert model only when score_headlines is first called.
Scores financial headlines into positive/negative/neutral probabilities.

Usage:
    from swingrl.sentiment.finbert import FinBERTScorer

    scorer = FinBERTScorer()
    scores = scorer.score_headlines(["Market rallies on Fed news"])
    sentiment, confidence = scorer.aggregate_sentiment(scores)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Lazy import -- populated by _ensure_loaded
_transformers: Any = None


class FinBERTScorer:
    """Lazy-loaded FinBERT sentiment scorer for financial headlines.

    The model is NOT loaded at __init__ time. It loads on the first call to
    score_headlines, avoiding 2GB+ memory usage when sentiment is disabled.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        """Initialize scorer without loading the model.

        Args:
            model_name: HuggingFace model identifier for FinBERT.
        """
        self._model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use.

        Imports transformers and downloads/loads model only when called.
        Logs finbert_model_loaded on success.
        """
        if self._model is not None:
            return

        global _transformers  # noqa: PLW0603

        if _transformers is None:
            import transformers  # type: ignore[import-untyped,import-not-found]

            _transformers = transformers

        self._tokenizer = _transformers.AutoTokenizer.from_pretrained(self._model_name)
        self._model = _transformers.AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        log.info("finbert_model_loaded", model_name=self._model_name)

    def score_headlines(self, headlines: list[str]) -> list[dict[str, float]]:
        """Score headlines into positive/negative/neutral probabilities.

        Args:
            headlines: List of headline strings to score.

        Returns:
            List of dicts with "positive", "negative", "neutral" float scores
            summing to ~1.0 per headline. Empty list if input is empty.
        """
        if not headlines:
            return []

        self._ensure_loaded()

        inputs = self._tokenizer(
            headlines,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self._model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # FinBERT label order: positive, negative, neutral
        results: list[dict[str, float]] = []
        for row in probs:
            results.append(
                {
                    "positive": float(row[0]),
                    "negative": float(row[1]),
                    "neutral": float(row[2]),
                }
            )

        log.info("headlines_scored", count=len(headlines))
        return results

    def aggregate_sentiment(self, scores: list[dict[str, float]]) -> tuple[float, float]:
        """Compute mean sentiment score and confidence from headline scores.

        Args:
            scores: List of score dicts from score_headlines.

        Returns:
            Tuple of (sentiment_score, confidence) where:
            - sentiment_score = mean(positive - negative) across headlines
            - confidence = mean(max(positive, negative, neutral)) across headlines
            Returns (0.0, 0.0) for empty input.
        """
        if not scores:
            return 0.0, 0.0

        sentiment_scores = [s["positive"] - s["negative"] for s in scores]
        confidences = [max(s["positive"], s["negative"], s["neutral"]) for s in scores]

        return float(np.mean(sentiment_scores)), float(np.mean(confidences))
