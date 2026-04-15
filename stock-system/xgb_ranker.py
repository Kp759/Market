"""
xgb_ranker.py — XGBoost return-score predictor for stock ranking.

Classes
-------
ReturnPredictor
    build_feature_matrix(stock_data_dict) → pd.DataFrame
    train(feature_df, forward_returns)    → None  (saves model)
    predict(feature_df)                   → pd.Series  (score per ticker)

Feature set
-----------
    sentiment_score   — FinBERT mean sentiment in [-1, 1]
    egarch_vol        — annualised EGARCH volatility forecast
    ff_alpha          — Fama-French 5-factor annualised alpha
    iv_garch_spread   — IV minus EGARCH vol (risk-premium signal)
    chronos_trend     — Chronos-T5 expected return over 7 days
    pe_ratio          — trailing or forward P/E
    eps_growth        — year-over-year EPS growth rate
    revenue_growth    — year-over-year revenue growth rate

All features are forward-filled then median-imputed for missing values
before model training or inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import settings

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "sentiment_score",
    "egarch_vol",
    "ff_alpha",
    "iv_garch_spread",
    "chronos_trend",
    "pe_ratio",
    "eps_growth",
    "revenue_growth",
]


class ReturnPredictor:
    """
    XGBoost-based cross-sectional return scorer.

    Parameters
    ----------
    model_path : str, optional
        Path to load / save the XGBoost JSON model.
        Defaults to ``settings.xgb_model_path``.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.xgb_model_path
        self._model: xgb.XGBRegressor | None = None

    # ------------------------------------------------------------------
    # Feature matrix builder
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self, stock_data_dict: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert the unified ``get_stock_data`` output into a feature
        matrix ready for :meth:`train` or :meth:`predict`.

        The input dict is expected to carry pre-computed signal keys
        alongside the raw data; callers (e.g. ``Orchestrator``) are
        responsible for injecting them before calling this method.

        Expected signal keys per ticker (all optional; NaN if absent):
            ``sentiment_score``, ``egarch_vol``, ``ff_alpha``,
            ``iv_garch_spread``, ``chronos_trend``

        Fundamental keys are extracted from the nested
        ``stock_data_dict[ticker]["fundamentals"]`` sub-dict.

        Parameters
        ----------
        stock_data_dict : dict
            ``{ticker: {fundamentals: {...}, sentiment_score: float, ...}}``

        Returns
        -------
        pd.DataFrame
            Shape ``(n_tickers, 8)`` with columns ``FEATURE_COLS``,
            index = ticker symbols.
        """
        rows = {}
        for ticker, d in stock_data_dict.items():
            fund = d.get("fundamentals") or {}
            row = {
                "sentiment_score": d.get("sentiment_score"),
                "egarch_vol":      d.get("egarch_vol"),
                "ff_alpha":        d.get("ff_alpha"),
                "iv_garch_spread": d.get("iv_garch_spread"),
                "chronos_trend":   d.get("chronos_trend"),
                "pe_ratio":        fund.get("pe_ratio"),
                "eps_growth":      fund.get("eps_growth"),
                "revenue_growth":  fund.get("revenue_growth"),
            }
            rows[ticker] = row

        df = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLS)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = self._impute(df)
        return df

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        feature_df: pd.DataFrame,
        forward_returns: pd.Series,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        """
        Fit an XGBoostRegressor to predict forward returns.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix as returned by :meth:`build_feature_matrix`.
        forward_returns : pd.Series
            1-period forward return per ticker (same index as ``feature_df``).
        """
        X, y = self._align(feature_df, forward_returns)

        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

        # Persist model
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("XGBoost model saved → %s", path)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Return a predicted return score for each ticker.

        Loads the persisted model if not already in memory.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix (same columns as used during :meth:`train`).

        Returns
        -------
        pd.Series
            Predicted return scores indexed by ticker.
        """
        if self._model is None:
            self._load_model()

        df = feature_df[FEATURE_COLS].copy()
        df = self._impute(df)
        preds = self._model.predict(df.values)
        return pd.Series(preds, index=df.index, name="xgb_pred")

    def feature_importance(self) -> pd.Series:
        """
        Return normalised feature importance scores from the trained model.

        Returns
        -------
        pd.Series
            Importance scores sorted descending, indexed by feature name.
        """
        if self._model is None:
            self._load_model()
        scores = self._model.get_booster().get_fscore()
        s = pd.Series(scores).reindex(FEATURE_COLS).fillna(0)
        return (s / s.sum()).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NaNs with column medians (or 0 if all-NaN)."""
        medians = df.median()
        return df.fillna(medians).fillna(0.0)

    def _align(
        self, feature_df: pd.DataFrame, forward_returns: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        common = feature_df.index.intersection(forward_returns.index)
        X = feature_df.loc[common, FEATURE_COLS].values
        y = forward_returns.loc[common].values
        return X, y

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found at {path}. "
                "Run ReturnPredictor.train() first."
            )
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path))
        logger.info("XGBoost model loaded ← %s", path)


# ---------------------------------------------------------------------------
# __main__ — smoke-test with synthetic data
# ---------------------------------------------------------------------------


def _main() -> None:
    import random

    random.seed(42)
    np.random.seed(42)

    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]

    # Build a synthetic stock_data_dict with signals already injected
    stock_data_dict = {
        t: {
            "fundamentals": {
                "pe_ratio":       random.uniform(15, 50),
                "eps_growth":     random.uniform(-0.2, 0.5),
                "revenue_growth": random.uniform(-0.1, 0.4),
            },
            "sentiment_score":  random.uniform(-1, 1),
            "egarch_vol":       random.uniform(0.15, 0.60),
            "ff_alpha":         random.uniform(-0.05, 0.05),
            "iv_garch_spread":  random.uniform(-0.10, 0.10),
            "chronos_trend":    random.uniform(-0.05, 0.05),
        }
        for t in tickers
    }

    predictor = ReturnPredictor(model_path="/tmp/test_xgb.json")
    feature_df = predictor.build_feature_matrix(stock_data_dict)

    print("Feature matrix:")
    print(feature_df.to_string())
    print()

    # Synthetic forward returns for training
    forward_returns = pd.Series(
        np.random.randn(len(tickers)) * 0.02,
        index=tickers,
    )

    predictor.train(feature_df, forward_returns)
    scores = predictor.predict(feature_df)

    print("Predicted return scores (ranked):")
    print(scores.sort_values(ascending=False).to_string())
    print()
    print("Feature importance:")
    print(predictor.feature_importance().to_string())


if __name__ == "__main__":
    _main()
