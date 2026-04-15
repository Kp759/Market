"""
orchestrator.py — end-to-end signal assembly and LLM-powered ranking.

Class
-----
Orchestrator
    run(tickers) → pd.DataFrame  with columns:
        ticker, score, recommendation, justification, risk_flag

Pipeline
--------
1. Fetch raw data via DataIngestion (OHLCV, fundamentals, news, IV).
2. Compute quant signals: EGARCH vol, FF5 alpha, IV-GARCH spread.
3. Score news sentiment per ticker via FinBERT.
4. Forecast price trend via Chronos-T5.
5. Build XGBoost feature matrix and predict return scores.
6. Compute composite score:
       score = 0.2·sentiment + 0.25·xgb_pred + 0.2·ff_alpha
             - 0.2·egarch_vol + 0.15·chronos_trend
7. Send structured JSON to LLM (Claude or vLLM) → buy/hold/sell +
   justification + risk flags.
8. Persist recommendations to SQLite.

LLM backend
-----------
Controlled by ``settings.llm_backend``:
    "claude"  → Anthropic Messages API  (claude-sonnet-4-20250514)
    "vllm"    → OpenAI-compatible endpoint at ``settings.vllm_base_url``
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiosqlite
import numpy as np
import pandas as pd

from config import settings
from data_ingestion import DataIngestion, _ensure_schema

logger = logging.getLogger(__name__)

# Composite score weights
_WEIGHTS = {
    "sentiment":    0.20,
    "xgb_pred":     0.25,
    "ff_alpha":     0.20,
    "egarch_vol":  -0.20,   # higher vol → lower score
    "chronos_trend":0.15,
}


class Orchestrator:
    """
    Full-pipeline orchestrator that assembles all signals and calls an LLM
    to produce final ranked buy/hold/sell recommendations.

    Parameters
    ----------
    llm_backend : str, optional
        Override ``settings.llm_backend``.  Accepts ``"claude"`` or ``"vllm"``.
    db_path : str, optional
        SQLite database path.  Defaults to ``settings.db_path``.
    """

    def __init__(
        self,
        llm_backend: str = settings.llm_backend,
        db_path: str | None = None,
    ) -> None:
        self.llm_backend = llm_backend
        self._db_path    = db_path or settings.db_path

        # Lazy-loaded heavy components
        self._sentiment_analyzer = None
        self._forecast_engine    = None
        self._return_predictor   = None
        self._quant_models       = None

    # ==================================================================
    # Main entry-point
    # ==================================================================

    async def run(self, tickers: list[str]) -> pd.DataFrame:
        """
        Execute the full analysis pipeline for the given tickers.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to analyse, e.g. ``["AAPL", "NVDA", "MSFT"]``.

        Returns
        -------
        pd.DataFrame
            Ranked table with columns:
            ``ticker, score, recommendation, justification, risk_flag``.
        """
        logger.info("Orchestrator starting for %s", tickers)

        # 1. Fetch raw data (all tickers in parallel)
        async with DataIngestion(self._db_path) as ing:
            raw = await ing.get_stock_data(tickers)

        # 2. Compute quant signals in thread-pool (CPU-bound)
        loop = asyncio.get_event_loop()
        raw  = await loop.run_in_executor(None, self._add_quant_signals, raw)

        # 3. Sentiment (GPU/CPU)
        raw = await loop.run_in_executor(None, self._add_sentiment, raw)

        # 4. Chronos trend
        raw = await loop.run_in_executor(None, self._add_chronos_trend, raw)

        # 5. XGBoost prediction
        raw = await loop.run_in_executor(None, self._add_xgb_scores, raw)

        # 6. Composite score
        for ticker in tickers:
            raw[ticker]["composite_score"] = self._composite_score(raw[ticker])

        # 7. LLM ranking
        llm_output = await self._call_llm(raw, tickers)

        # 8. Assemble DataFrame
        rows = []
        for ticker in tickers:
            d     = raw.get(ticker, {})
            llm_d = llm_output.get(ticker, {})
            rows.append(
                {
                    "ticker":         ticker,
                    "score":          round(d.get("composite_score", 0.0), 6),
                    "recommendation": llm_d.get("recommendation", "hold"),
                    "justification":  llm_d.get("justification", ""),
                    "risk_flag":      llm_d.get("risk_flag", ""),
                    "sentiment":      round(d.get("sentiment_score", 0.0), 4),
                    "egarch_vol":     round(d.get("egarch_vol", 0.0), 4),
                    "ff_alpha":       round(d.get("ff_alpha", 0.0), 4),
                    "iv_garch_spread":round(d.get("iv_garch_spread", 0.0) or 0.0, 4),
                    "chronos_trend":  round(d.get("chronos_trend", 0.0), 4),
                    "xgb_pred":       round(d.get("xgb_pred", 0.0), 4),
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        # 9. Persist
        await self._persist_recommendations(df, raw)

        return df

    # ==================================================================
    # Signal computation (sync, run in executor)
    # ==================================================================

    def _add_quant_signals(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Add egarch_vol, ff_alpha, iv_garch_spread to each ticker dict."""
        from quant_models import QuantModels
        import numpy as np

        qm = self._get_quant_models()

        for ticker, d in raw.items():
            if "error" in d:
                continue

            prices = d.get("prices", [])
            if len(prices) < 30:
                continue

            closes = np.array([p["close"] for p in prices], dtype=float)
            log_ret = pd.Series(np.log(closes)).diff().dropna()

            # EGARCH
            try:
                eg = qm.fit_egarch(log_ret)
                d["egarch_vol"] = eg["vol_forecast"]
            except Exception as exc:
                logger.warning("EGARCH failed for %s: %s", ticker, exc)
                d["egarch_vol"] = None

            # FF alpha
            try:
                dates = pd.to_datetime([p["date"][:10] for p in prices[1:]])
                ret_series = pd.Series(log_ret.values, index=dates)
                ff = qm.fama_french_alpha(ticker, ret_series)
                d["ff_alpha"] = ff.get("alpha_annual") if "error" not in ff else None
            except Exception as exc:
                logger.warning("FF alpha failed for %s: %s", ticker, exc)
                d["ff_alpha"] = None

            # IV–GARCH spread
            iv = d.get("options_iv")
            d["iv_garch_spread"] = qm.iv_garch_spread(iv, d.get("egarch_vol"))

        return raw

    def _add_sentiment(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Add sentiment_score to each ticker dict."""
        from sentiment import SentimentAnalyzer

        analyzer = self._get_sentiment_analyzer()

        ticker_news = {
            ticker: d.get("news", [])
            for ticker, d in raw.items()
            if "error" not in d
        }
        scores = analyzer.score_ticker_news(ticker_news)

        for ticker, score in scores.items():
            raw[ticker]["sentiment_score"] = score

        return raw

    def _add_chronos_trend(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Add chronos_trend (7-day expected return) to each ticker dict."""
        from forecasting import ForecastEngine

        engine = self._get_forecast_engine()

        for ticker, d in raw.items():
            if "error" in d:
                continue
            prices = d.get("prices", [])
            if len(prices) < 30:
                d["chronos_trend"] = 0.0
                continue
            try:
                closes = pd.Series([p["close"] for p in prices], dtype=float)
                d["chronos_trend"] = engine.chronos_trend(closes, horizon=7)
            except Exception as exc:
                logger.warning("Chronos failed for %s: %s", ticker, exc)
                d["chronos_trend"] = 0.0

        return raw

    def _add_xgb_scores(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Add xgb_pred to each ticker dict."""
        from xgb_ranker import ReturnPredictor
        from pathlib import Path

        predictor = self._get_return_predictor()
        feature_df = predictor.build_feature_matrix(raw)

        try:
            scores = predictor.predict(feature_df)
            for ticker in scores.index:
                raw[ticker]["xgb_pred"] = float(scores[ticker])
        except FileNotFoundError:
            logger.warning(
                "No trained XGB model found; xgb_pred will be 0. "
                "Run main.py --mode train to build the model."
            )
            for ticker in raw:
                raw[ticker]["xgb_pred"] = 0.0

        return raw

    # ==================================================================
    # Composite score
    # ==================================================================

    @staticmethod
    def _composite_score(d: dict[str, Any]) -> float:
        """
        Compute a weighted composite signal score.

            score = 0.20·sentiment + 0.25·xgb_pred + 0.20·ff_alpha
                  - 0.20·egarch_vol + 0.15·chronos_trend
        """
        def _get(key: str) -> float:
            return float(d.get(key) or 0.0)

        return (
            _WEIGHTS["sentiment"]     * _get("sentiment_score")
            + _WEIGHTS["xgb_pred"]   * _get("xgb_pred")
            + _WEIGHTS["ff_alpha"]   * _get("ff_alpha")
            + _WEIGHTS["egarch_vol"] * _get("egarch_vol")   # weight is negative
            + _WEIGHTS["chronos_trend"] * _get("chronos_trend")
        )

    # ==================================================================
    # LLM call — Claude or vLLM
    # ==================================================================

    async def _call_llm(
        self, raw: dict[str, Any], tickers: list[str]
    ) -> dict[str, Any]:
        """
        Send structured signal data to the LLM and parse buy/hold/sell output.

        Returns
        -------
        dict
            ``{ticker: {recommendation, justification, risk_flag}}``
        """
        prompt = self._build_prompt(raw, tickers)

        if self.llm_backend == "claude":
            response_text = await self._call_claude(prompt)
        else:
            response_text = await self._call_vllm(prompt)

        return self._parse_llm_response(response_text, tickers)

    def _build_prompt(
        self, raw: dict[str, Any], tickers: list[str]
    ) -> str:
        """Build the structured JSON prompt for the LLM."""
        signals = {}
        for ticker in tickers:
            d = raw.get(ticker, {})
            fund = d.get("fundamentals") or {}
            signals[ticker] = {
                "composite_score":  round(d.get("composite_score", 0), 4),
                "sentiment_score":  round(d.get("sentiment_score", 0), 4),
                "egarch_vol":       round(d.get("egarch_vol") or 0, 4),
                "ff_alpha":         round(d.get("ff_alpha") or 0, 6),
                "iv_garch_spread":  round(d.get("iv_garch_spread") or 0, 4),
                "chronos_trend":    round(d.get("chronos_trend", 0), 4),
                "xgb_pred":         round(d.get("xgb_pred", 0), 4),
                "pe_ratio":         fund.get("pe_ratio"),
                "eps_growth":       fund.get("eps_growth"),
                "revenue_growth":   fund.get("revenue_growth"),
                "market_cap":       fund.get("market_cap"),
                "beta":             fund.get("beta"),
                "sector":           fund.get("sector"),
            }

        system = (
            "You are a quantitative equity analyst. "
            "You receive structured signal data for a list of stocks and must "
            "rank them as buy, hold, or sell with a concise justification and "
            "any notable risk flags. "
            "Respond ONLY with a valid JSON object — no markdown, no extra text."
        )

        user = (
            "Analyse the following stock signals and return a JSON object "
            "with this exact structure for each ticker:\n\n"
            '{"TICKER": {"recommendation": "buy|hold|sell", '
            '"justification": "2-3 sentence rationale", '
            '"risk_flag": "one-line risk or empty string"}}\n\n'
            f"Signal data:\n{json.dumps(signals, indent=2)}"
        )

        return json.dumps({"system": system, "user": user})

    async def _call_claude(self, prompt_json: str) -> str:
        """Call the Anthropic Messages API."""
        import anthropic

        payload = json.loads(prompt_json)
        client  = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        message = await client.messages.create(
            model=settings.llm_model_claude,
            max_tokens=1024,
            system=payload["system"],
            messages=[{"role": "user", "content": payload["user"]}],
        )
        return message.content[0].text

    async def _call_vllm(self, prompt_json: str) -> str:
        """Call a vLLM OpenAI-compatible endpoint."""
        from openai import AsyncOpenAI

        payload = json.loads(prompt_json)
        client  = AsyncOpenAI(
            api_key="EMPTY",
            base_url=settings.vllm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model_vllm,
            messages=[
                {"role": "system", "content": payload["system"]},
                {"role": "user",   "content": payload["user"]},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_llm_response(
        text: str, tickers: list[str]
    ) -> dict[str, Any]:
        """Parse LLM JSON output, falling back to 'hold' on any error."""
        try:
            # Strip potential markdown fences
            clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
        except Exception as exc:
            logger.warning("LLM response parse error: %s\nRaw: %s", exc, text[:200])
            parsed = {}

        result = {}
        for ticker in tickers:
            entry = parsed.get(ticker, {})
            result[ticker] = {
                "recommendation": entry.get("recommendation", "hold"),
                "justification":  entry.get("justification", ""),
                "risk_flag":      entry.get("risk_flag", ""),
            }
        return result

    # ==================================================================
    # Persistence
    # ==================================================================

    async def _persist_recommendations(
        self, df: pd.DataFrame, raw: dict[str, Any]
    ) -> None:
        """Write recommendations to SQLite for backtest replay."""
        now        = time.time()
        expires_at = now + settings.price_ttl

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await _ensure_schema(db)
            for _, row in df.iterrows():
                ticker = row["ticker"]
                signals_blob = json.dumps(raw.get(ticker, {}))
                await db.execute(
                    "INSERT INTO recommendations "
                    "(ticker, score, recommendation, justification, risk_flag, "
                    " signals, fetched_at, ttl_expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ticker,
                        float(row["score"]),
                        str(row["recommendation"]),
                        str(row["justification"]),
                        str(row["risk_flag"]),
                        signals_blob,
                        now,
                        expires_at,
                    ),
                )
            await db.commit()

    # ==================================================================
    # Lazy component getters
    # ==================================================================

    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            from sentiment import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer()
            self._sentiment_analyzer.load()
        return self._sentiment_analyzer

    def _get_forecast_engine(self):
        if self._forecast_engine is None:
            from forecasting import ForecastEngine
            self._forecast_engine = ForecastEngine()
            self._forecast_engine.load()
        return self._forecast_engine

    def _get_return_predictor(self):
        if self._return_predictor is None:
            from xgb_ranker import ReturnPredictor
            self._return_predictor = ReturnPredictor()
        return self._return_predictor

    def _get_quant_models(self):
        if self._quant_models is None:
            from quant_models import QuantModels
            self._quant_models = QuantModels()
        return self._quant_models


# ---------------------------------------------------------------------------
# __main__ — smoke-test (requires valid API keys in .env)
# ---------------------------------------------------------------------------


async def _main() -> None:
    tickers = ["AAPL", "NVDA", "MSFT"]
    print(f"Orchestrator smoke-test  tickers={tickers}\n")

    orch = Orchestrator()
    df   = await orch.run(tickers)

    print(df[["ticker", "score", "recommendation", "risk_flag"]].to_string(index=False))


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
