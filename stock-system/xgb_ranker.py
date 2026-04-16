"""
xgb_ranker.py — XGBoost 60-day return predictor with walk-forward training.

Feature set (10 features)
--------------------------
    momentum        — 12-1 month price momentum
    sue_score       — standardised unexpected earnings
    short_squeeze   — days-to-cover + float short % composite
    insider_signal  — role-weighted net buy/sell ratio
    revenue_accel   — second derivative of quarterly revenue growth
    sentiment_score — FinBERT mean score (-1 to 1)
    iv_spread       — IV minus EGARCH vol (fear premium)
    egarch_vol      — annualised EGARCH vol forecast
    pe_ratio        — trailing / forward P/E
    eps_growth      — YoY EPS growth rate

Walk-forward training
---------------------
    train_walkforward(panel_df, forward_return_col)
        Trains on 6-month rolling windows, validates on the next month,
        accumulates OOS predictions, then fits a final model on all data.
        Saves per-fold feature importances to models/feature_importance.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import settings

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "momentum",
    "sue_score",
    "short_squeeze",
    "insider_signal",
    "revenue_accel",
    "sentiment_score",
    "iv_spread",
    "egarch_vol",
    "pe_ratio",
    "eps_growth",
]

_XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)


class ReturnPredictor:
    """
    XGBoost cross-sectional 60-day return predictor.

    Parameters
    ----------
    model_path : str, optional
        Path to save / load the JSON model.
    feature_importance_path : str, optional
        Path to save feature importances JSON.
    """

    def __init__(
        self,
        model_path: str | None = None,
        feature_importance_path: str | None = None,
    ) -> None:
        self.model_path             = model_path or settings.xgb_model_path
        self.feature_importance_path = (
            feature_importance_path or settings.feature_importance_path
        )
        self._model: xgb.XGBRegressor | None = None

    # ------------------------------------------------------------------
    # Feature matrix builder
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self, signal_dict: dict[str, dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Build a (n_tickers × 10) feature matrix from pre-computed signals.

        Parameters
        ----------
        signal_dict : dict
            ``{ticker: {momentum, sue_score, …, fundamentals: {…}}}``
            Produced by ``Screener._assemble_signals()``.

        Returns
        -------
        pd.DataFrame
            Columns = ``FEATURE_COLS``, index = ticker symbols.
            NaNs are median-imputed.
        """
        rows: dict[str, dict] = {}
        for ticker, d in signal_dict.items():
            fund = d.get("fundamentals") or {}
            rows[ticker] = {
                "momentum":       d.get("momentum"),
                "sue_score":      d.get("sue_score"),
                "short_squeeze":  d.get("short_squeeze"),
                "insider_signal": d.get("insider_signal"),
                "revenue_accel":  d.get("revenue_accel"),
                "sentiment_score":d.get("sentiment_score"),
                "iv_spread":      d.get("iv_spread"),
                "egarch_vol":     d.get("egarch_vol"),
                "pe_ratio":       fund.get("pe_ratio"),
                "eps_growth":     fund.get("eps_growth"),
            }

        df = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURE_COLS)
        df = df.apply(pd.to_numeric, errors="coerce")
        return self._impute(df)

    # ------------------------------------------------------------------
    # Simple train (single pass)
    # ------------------------------------------------------------------

    def train(
        self,
        feature_df: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> None:
        """
        Fit a single XGBoost model on the full dataset.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix (columns = FEATURE_COLS).
        forward_returns : pd.Series
            60-day forward return per ticker.
        """
        X, y = self._align(feature_df, forward_returns)
        self._model = xgb.XGBRegressor(**_XGB_PARAMS)
        self._model.fit(X, y)
        self._save_model()
        self._save_feature_importance()
        logger.info("XGBoost model trained (single pass) → %s", self.model_path)

    # ------------------------------------------------------------------
    # Walk-forward train
    # ------------------------------------------------------------------

    def train_walkforward(
        self,
        panel_df: pd.DataFrame,
        forward_return_col: str = "fwd_return_60d",
        train_months: int = 6,
        test_months: int = 1,
    ) -> pd.Series:
        """
        Walk-forward training with 6-month rolling windows.

        Splits *panel_df* (must have a ``date`` column) into successive
        (train, test) folds:
            fold 1: train months 0-5, test month 6
            fold 2: train months 1-6, test month 7
            …

        Accumulates out-of-sample predictions.  After all folds, retrains
        a final model on the full dataset.  Saves per-fold feature
        importances to ``settings.feature_importance_path``.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Columns must include all ``FEATURE_COLS`` + ``forward_return_col``
            + ``date`` (``pd.Timestamp`` or parseable string).
        forward_return_col : str
            Name of the target column.
        train_months, test_months : int
            Rolling window sizes in calendar months.

        Returns
        -------
        pd.Series
            Out-of-sample predictions indexed like *panel_df*.
        """
        df = panel_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        min_date = df["date"].min()
        max_date = df["date"].max()

        oos_preds = pd.Series(index=df.index, dtype=float, name="oos_pred")
        importance_folds: list[dict] = []

        train_start = min_date
        fold = 0
        while True:
            train_end = train_start + pd.DateOffset(months=train_months)
            test_end  = train_end   + pd.DateOffset(months=test_months)
            if test_end > max_date:
                break

            train_mask = (df["date"] >= train_start) & (df["date"] < train_end)
            test_mask  = (df["date"] >= train_end)   & (df["date"] < test_end)

            if train_mask.sum() < 20 or test_mask.sum() < 5:
                train_start += pd.DateOffset(months=test_months)
                continue

            X_tr = df.loc[train_mask, FEATURE_COLS].values
            y_tr = df.loc[train_mask, forward_return_col].values
            X_te = df.loc[test_mask,  FEATURE_COLS].values

            m = xgb.XGBRegressor(**_XGB_PARAMS)
            m.fit(X_tr, y_tr)

            oos_preds.loc[test_mask] = m.predict(X_te)

            # Collect feature importances for this fold
            fscore = m.get_booster().get_fscore()
            importance_folds.append({"fold": fold, "importances": fscore})
            fold += 1
            train_start += pd.DateOffset(months=test_months)

        logger.info("Walk-forward: completed %d folds", fold)

        # ---- retrain on full data ----
        X_all = df[FEATURE_COLS].values
        y_all = df[forward_return_col].values
        self._model = xgb.XGBRegressor(**_XGB_PARAMS)
        self._model.fit(X_all, y_all)

        self._save_model()
        self._save_feature_importance(importance_folds)

        return oos_preds

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Return predicted 60-day return scores for each ticker.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature matrix built by :meth:`build_feature_matrix`.

        Returns
        -------
        pd.Series
            Scores indexed by ticker symbol.
        """
        if self._model is None:
            self._load_model()

        df    = self._impute(feature_df[FEATURE_COLS].copy())
        preds = self._model.predict(df.values)
        return pd.Series(preds, index=df.index, name="xgb_pred")

    def feature_importance(self) -> pd.Series:
        """Return normalised feature importance scores (sorted descending)."""
        if self._model is None:
            self._load_model()
        scores = self._model.get_booster().get_fscore()
        s = pd.Series(scores).reindex(FEATURE_COLS).fillna(0)
        total = s.sum()
        return (s / total if total > 0 else s).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NaNs with column medians, then 0 for all-NaN columns."""
        medians = df.median()
        return df.fillna(medians).fillna(0.0)

    def _align(
        self, feature_df: pd.DataFrame, forward_returns: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        common = feature_df.index.intersection(forward_returns.index)
        return (
            feature_df.loc[common, FEATURE_COLS].values,
            forward_returns.loc[common].values,
        )

    def _save_model(self) -> None:
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("XGBoost model saved → %s", path)

    def _load_model(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run main.py --mode train first."
            )
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path))
        logger.info("XGBoost model loaded ← %s", path)

    def _save_feature_importance(
        self, folds: list[dict] | None = None
    ) -> None:
        """Save current-model importance + optional per-fold data to JSON."""
        imp = self.feature_importance().to_dict()
        payload = {"model": imp}
        if folds:
            payload["folds"] = folds
        path = Path(self.feature_importance_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Feature importances saved → %s", path)


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

def _main() -> None:
    import random
    random.seed(0)
    np.random.seed(0)

    tickers = [f"T{i:03d}" for i in range(30)]
    signal_dict = {
        t: {
            "momentum":       random.gauss(0, 0.1),
            "sue_score":      random.gauss(0, 1),
            "short_squeeze":  random.uniform(0, 1),
            "insider_signal": random.uniform(-1, 1),
            "revenue_accel":  random.gauss(0, 0.05),
            "sentiment_score":random.uniform(-1, 1),
            "iv_spread":      random.gauss(0, 0.05),
            "egarch_vol":     random.uniform(0.15, 0.6),
            "fundamentals": {
                "pe_ratio":  random.uniform(10, 50),
                "eps_growth":random.gauss(0.1, 0.3),
            },
        }
        for t in tickers
    }

    predictor = ReturnPredictor(
        model_path="/tmp/test_xgb.json",
        feature_importance_path="/tmp/test_importance.json",
    )
    feat_df = predictor.build_feature_matrix(signal_dict)
    fwd     = pd.Series(np.random.randn(30) * 0.05, index=tickers)

    predictor.train(feat_df, fwd)
    scores = predictor.predict(feat_df)
    print("Top 5 predicted scores:")
    print(scores.sort_values(ascending=False).head())
    print("\nFeature importances:")
    print(predictor.feature_importance())


if __name__ == "__main__":
    _main()
