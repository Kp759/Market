"""
sentiment.py — FinBERT-based sentiment analysis for news articles.

Planned functionality
---------------------
* Load ProsusAI/finbert from HuggingFace Transformers.
* Accept a list of article dicts (as returned by DataIngestion.get_stock_data).
* Return per-article sentiment (positive / negative / neutral) with confidence.
* Aggregate to a per-ticker sentiment score.

Status: stub — implementation coming in the next sprint.
"""

from __future__ import annotations

from typing import Any


class SentimentAnalyser:
    """FinBERT sentiment analyser (stub)."""

    async def analyse(
        self, ticker: str, articles: list[dict[str, Any]]
    ) -> dict[str, Any]:
        raise NotImplementedError("SentimentAnalyser is not yet implemented.")
