"""
screener.py — cross-sectional stock screener targeting 1–3 month opportunities.

Class
-----
Screener
    screen(tickers, top_n=20) → pd.DataFrame

Pipeline per ticker
-------------------
1. Fetch data via DataIngestion (OHLCV, fundamentals, earnings, SI, insider, news, IV).
2. Compute EGARCH vol and IV-GARCH spread via QuantModels.
3. Score news sentiment via SentimentAnalyzer (0.0 default when no API key).
4. Compute momentum, SUE, short-squeeze, insider, revenue-accel via SignalEngine.
5. Predict XGBoost return score via ReturnPredictor (0.0 when no model file).
6. Assemble composite score:
       score = 0.30·momentum + 0.25·sue + 0.20·xgb + 0.15·squeeze
             + 0.10·insider - egarch_penalty
   where egarch_penalty = max(0, egarch_vol - cross_sectional_median_vol) × 0.5
7. Cross-sectionally z-score all signals before combining.
8. Return top_n tickers as DataFrame with all raw signals + composite score.

All errors per ticker are caught and logged; that ticker receives NaN
signals and is ranked at the bottom.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# Composite score weights
_W = {
    "momentum":     0.30,
    "sue_score":    0.25,
    "xgb_pred":     0.20,
    "short_squeeze":0.15,
    "insider_signal":0.10,
}
_EGARCH_PENALTY_SCALE = 0.50   # applied to above-median vol excess


class Screener:
    """
    Full screening pipeline for a universe of tickers.

    Parameters
    ----------
    db_path : str, optional
        SQLite database path.
    top_n : int, optional
        Number of top tickers to return.
    """

    def __init__(
        self,
        db_path: str | None = None,
        top_n: int | None = None,
    ) -> None:
        self._db_path = db_path or settings.db_path
        self._top_n   = top_n or settings.top_n

        # Lazy-loaded heavy components
        self._sentiment_analyzer = None
        self._quant_models       = None
        self._return_predictor   = None
        self._signal_engine      = None

    # ==================================================================
    # Main entry-point
    # ==================================================================

    async def screen(
        self, tickers: list[str], top_n: int | None = None
    ) -> pd.DataFrame:
        """
        Score all *tickers* and return the top picks.

        Parameters
        ----------
        tickers : list[str]
            Universe of tickers to screen (~1800 for full run).
        top_n : int, optional
            Override instance top_n.

        Returns
        -------
        pd.DataFrame
            Sorted descending by ``score``, top ``top_n`` rows.
            Columns: ticker, score, momentum, sue_score, short_squeeze,
                     insider_signal, revenue_accel, sentiment_score,
                     egarch_vol, iv_spread, xgb_pred, pe_ratio, eps_growth,
                     revenue_growth, market_cap, sector, egarch_flag
        """
        top_n = top_n or self._top_n
        logger.info("Screener: scoring %d tickers …", len(tickers))

        # 1. Fetch raw data for all tickers (parallel, semaphore-limited)
        from data_ingestion import DataIngestion
        async with DataIngestion(self._db_path) as ing:
            raw = await ing.get_stock_data(tickers)

        # 2. Compute CPU-bound signals in executor
        loop = asyncio.get_event_loop()
        signal_dict = await loop.run_in_executor(
            None, self._compute_all_signals, raw
        )

        # 3. Build feature matrix and predict XGB scores
        try:
            predictor = self._get_return_predictor()
            feat_df   = predictor.build_feature_matrix(signal_dict)
            xgb_scores = predictor.predict(feat_df)
            for ticker in signal_dict:
                signal_dict[ticker]["xgb_pred"] = float(xgb_scores.get(ticker, 0.0))
        except FileNotFoundError:
            logger.warning(
                "No XGB model found — xgb_pred=0 for all tickers. "
                "Run main.py --mode train to build the model."
            )
            for ticker in signal_dict:
                signal_dict[ticker]["xgb_pred"] = 0.0

        # 4. Composite score with cross-sectional z-scoring
        rows = self._composite_score(signal_dict)

        # 5. Sort and return top_n
        df = (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        logger.info("Screener: top %d tickers selected.", len(df))
        return df

    # ==================================================================
    # Signal computation (sync — runs in executor)
    # ==================================================================

    def _compute_all_signals(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """
        For every ticker in *raw*, compute quant signals, sentiment, and
        all price/fundamental signals.  Returns an enriched signal dict.
        """
        qm = self._get_quant_models()
        se = self._get_signal_engine()
        sa = self._get_sentiment_analyzer()

        # Batch sentiment first (GPU-friendly)
        ticker_news = {
            t: d.get("news", [])
            for t, d in raw.items()
            if not isinstance(d.get("news"), Exception)
        }
        sent_scores = sa.score_ticker_news(ticker_news)

        signal_dict: dict[str, Any] = {}

        for ticker, d in raw.items():
            if "error" in d:
                signal_dict[ticker] = {"error": d["error"]}
                continue

            prices = d.get("prices", [])
            fund   = d.get("fundamentals") or {}
            iv     = d.get("options_iv")

            # ---- EGARCH vol ----
            egarch_vol  = None
            egarch_flag = False
            if len(prices) >= 60:
                try:
                    import numpy as np
                    closes  = np.array([p["close"] for p in prices], dtype=float)
                    log_ret = pd.Series(np.log(closes)).diff().dropna()
                    eg = qm.fit_egarch(log_ret)
                    egarch_vol  = eg["vol_forecast"]
                    egarch_flag = eg.get("high_vol_flag", False)
                except Exception as exc:
                    logger.debug("EGARCH %s: %s", ticker, exc)

            # ---- IV-GARCH spread ----
            iv_spread = qm.iv_garch_spread(iv, egarch_vol)

            # ---- Price/earnings/insider/SI signals ----
            sigs = se.compute_all(ticker, d)

            # ---- Sentiment ----
            sentiment = sent_scores.get(ticker, 0.0)

            signal_dict[ticker] = {
                "momentum":       sigs["momentum"],
                "sue_score":      sigs["sue_score"],
                "short_squeeze":  sigs["short_squeeze"],
                "insider_signal": sigs["insider_signal"],
                "revenue_accel":  sigs["revenue_accel"],
                "sentiment_score":sentiment,
                "egarch_vol":     egarch_vol,
                "egarch_flag":    egarch_flag,
                "iv_spread":      iv_spread,
                "fundamentals":   fund,
            }

        return signal_dict

    # ==================================================================
    # Composite scoring
    # ==================================================================

    def _composite_score(
        self, signal_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Cross-sectionally z-score each signal, apply weights, subtract
        EGARCH penalty, and return a list of row dicts.
        """
        valid = {
            t: d for t, d in signal_dict.items() if "error" not in d
        }
        if not valid:
            return []

        # Build a raw signal DataFrame for z-scoring
        signal_keys = ["momentum", "sue_score", "short_squeeze",
                       "insider_signal", "egarch_vol"]
        raw_df = pd.DataFrame(
            {t: {k: d.get(k) for k in signal_keys} for t, d in valid.items()}
        ).T.apply(pd.to_numeric, errors="coerce")

        # Cross-sectional z-score (std=1, mean=0)
        z_df = (raw_df - raw_df.mean()) / raw_df.std().replace(0, 1)
        z_df = z_df.fillna(0.0)

        median_vol = raw_df["egarch_vol"].median() or 0.0

        rows: list[dict[str, Any]] = []
        for ticker, d in valid.items():
            z = z_df.loc[ticker] if ticker in z_df.index else pd.Series(dtype=float)

            xgb_pred  = float(d.get("xgb_pred") or 0.0)
            egarch_vol = float(d.get("egarch_vol") or 0.0)

            # EGARCH penalty: only penalise above-median vol
            penalty = max(0.0, egarch_vol - median_vol) * _EGARCH_PENALTY_SCALE

            score = (
                _W["momentum"]      * float(z.get("momentum",      0.0))
                + _W["sue_score"]   * float(z.get("sue_score",     0.0))
                + _W["xgb_pred"]    * xgb_pred
                + _W["short_squeeze"]* float(z.get("short_squeeze",0.0))
                + _W["insider_signal"]* float(z.get("insider_signal",0.0))
                - penalty
            )

            fund = d.get("fundamentals") or {}
            rows.append({
                "ticker":         ticker,
                "score":          round(score, 6),
                "momentum":       d.get("momentum"),
                "sue_score":      d.get("sue_score"),
                "short_squeeze":  d.get("short_squeeze"),
                "insider_signal": d.get("insider_signal"),
                "revenue_accel":  d.get("revenue_accel"),
                "sentiment_score":d.get("sentiment_score"),
                "egarch_vol":     d.get("egarch_vol"),
                "egarch_flag":    d.get("egarch_flag", False),
                "iv_spread":      d.get("iv_spread"),
                "xgb_pred":       xgb_pred,
                "pe_ratio":       fund.get("pe_ratio"),
                "eps_growth":     fund.get("eps_growth"),
                "revenue_growth": fund.get("revenue_growth"),
                "market_cap":     fund.get("market_cap"),
                "sector":         fund.get("sector"),
            })

        # Append error tickers at the bottom
        for ticker, d in signal_dict.items():
            if "error" in d:
                rows.append({"ticker": ticker, "score": float("nan"),
                             "error": d["error"]})

        return rows

    # ==================================================================
    # Lazy component getters
    # ==================================================================

    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            from sentiment import SentimentAnalyzer
            self._sentiment_analyzer = SentimentAnalyzer()
            self._sentiment_analyzer.load()
        return self._sentiment_analyzer

    def _get_quant_models(self):
        if self._quant_models is None:
            from quant_models import QuantModels
            self._quant_models = QuantModels()
        return self._quant_models

    def _get_return_predictor(self):
        if self._return_predictor is None:
            from xgb_ranker import ReturnPredictor
            self._return_predictor = ReturnPredictor()
        return self._return_predictor

    def _get_signal_engine(self):
        if self._signal_engine is None:
            from signals import SignalEngine
            self._signal_engine = SignalEngine()
        return self._signal_engine


# ---------------------------------------------------------------------------
# __main__ — smoke-test with a small universe
# ---------------------------------------------------------------------------

async def _main() -> None:
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NFLX"]
    print(f"Screener smoke-test — universe: {tickers}\n")

    screener = Screener()
    df = await screener.screen(tickers, top_n=5)

    cols = ["ticker", "score", "momentum", "sue_score", "short_squeeze",
            "sentiment_score", "egarch_vol", "sector"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(_main())
