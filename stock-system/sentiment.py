"""
sentiment.py — FinBERT-based sentiment analysis for financial news articles.

Class
-----
SentimentAnalyzer
    load()                          — load ProsusAI/finbert (lazy)
    score_articles(articles) → float — mean sentiment score in [-1, 1]
    score_ticker_news(ticker_news_map) → {ticker: float}

The model is loaded once and cached on the instance.  GPU is used
automatically when available (``device="cuda"``).

Labels and their numeric mappings
----------------------------------
    positive →  +1
    neutral  →   0
    negative →  -1

The final score for a batch of articles is the probability-weighted mean::

    score = Σ( p_pos - p_neg ) / N
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"


class SentimentAnalyzer:
    """
    FinBERT-based financial sentiment analyser.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``ProsusAI/finbert``.
    batch_size : int
        Number of texts per forward pass.
    max_length : int
        Maximum token length (FinBERT was trained with 512).
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self._tokenizer = None
        self._model     = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the FinBERT tokenizer and model from HuggingFace Hub.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s …", self.model_name, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self._device)
        self._model.eval()
        logger.info("FinBERT loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_articles(self, articles: list[str]) -> float:
        """
        Score a list of article texts and return the mean sentiment score.

        Each article is scored as::

            article_score = P(positive) - P(negative)

        The method returns the unweighted mean across all articles.

        Parameters
        ----------
        articles : list[str]
            Raw text strings (titles, descriptions, or full text).
            Empty or whitespace-only strings are silently dropped.

        Returns
        -------
        float
            Mean sentiment score in **[-1, 1]**.
            Returns ``0.0`` if ``articles`` is empty after filtering.
        """
        self.load()

        texts = [t.strip() for t in articles if t and t.strip()]
        if not texts:
            return 0.0

        import torch.nn.functional as F

        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**enc).logits   # (B, 3)

            probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, 3)

            # FinBERT label order: positive=0, negative=1, neutral=2
            # (verify with model.config.id2label if re-using a different checkpoint)
            label_map = self._get_label_map()
            pos_idx = label_map["positive"]
            neg_idx = label_map["negative"]

            for p in probs:
                scores.append(float(p[pos_idx]) - float(p[neg_idx]))

        return round(sum(scores) / len(scores), 6) if scores else 0.0

    def score_ticker_news(
        self, ticker_news_map: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """
        Score news for multiple tickers in one call.

        Parameters
        ----------
        ticker_news_map : dict
            ``{ticker: [{title, description, ...}, ...]}``
            as returned by ``DataIngestion.get_news()``.

        Returns
        -------
        dict
            ``{ticker: mean_sentiment_score}``
        """
        results: dict[str, float] = {}
        for ticker, articles in ticker_news_map.items():
            texts = []
            for art in articles:
                if art.get("title"):
                    texts.append(art["title"])
                if art.get("description"):
                    texts.append(art["description"])
            results[ticker] = self.score_articles(texts)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_label_map(self) -> dict[str, int]:
        """
        Return a mapping from label name to logit index.

        FinBERT's ``config.id2label`` is ``{0: 'positive', 1: 'negative', 2: 'neutral'}``.
        We invert it here for O(1) lookup.
        """
        id2label: dict[int, str] = self._model.config.id2label
        return {v.lower(): k for k, v in id2label.items()}


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------


def _main() -> None:
    analyzer = SentimentAnalyzer()
    analyzer.load()

    samples = [
        "Apple reports record quarterly earnings, beats Wall Street expectations.",
        "NVIDIA faces antitrust probe as regulators scrutinise AI chip dominance.",
        "Microsoft Azure growth remains steady amid cloud-computing competition.",
        "Markets closed mixed as investors await Federal Reserve rate decision.",
        "Tesla recalls thousands of vehicles due to software defect.",
    ]

    print("FinBERT sentiment scores\n" + "=" * 40)
    for text in samples:
        score = analyzer.score_articles([text])
        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        print(f"  [{score:+.3f} {label:8s}]  {text[:70]}")

    mean = analyzer.score_articles(samples)
    print(f"\n  Mean score across all samples: {mean:+.4f}")


if __name__ == "__main__":
    _main()
