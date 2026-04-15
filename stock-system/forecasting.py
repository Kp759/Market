"""
forecasting.py — Chronos-T5 probabilistic time-series forecasting.

Class
-----
ForecastEngine
    load()                              — lazy-load amazon/chronos-t5-large
    forecast(price_series, horizon=7)   → median forecast + 80 % CI

The model is loaded once per instance and cached.  GPU is used when
available.  Chronos operates on raw price levels; log-returns are NOT
required.

Reference
---------
    Ansari et al. (2024) "Chronos: Learning the Language of Time Series"
    https://arxiv.org/abs/2403.07815
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "amazon/chronos-t5-large"


class ForecastEngine:
    """
    Chronos-T5 probabilistic price forecaster.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.  Defaults to ``amazon/chronos-t5-large``.
    num_samples : int
        Number of sample paths to draw; higher = smoother CI but slower.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        num_samples: int = 100,
    ) -> None:
        self.model_id    = model_id
        self.num_samples = num_samples
        self._pipeline   = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Download and cache the Chronos pipeline from HuggingFace Hub.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._pipeline is not None:
            return

        from chronos import ChronosPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s …", self.model_id, self._device)

        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Chronos pipeline loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        price_series: pd.Series,
        horizon: int = 7,
    ) -> dict[str, Any]:
        """
        Generate a probabilistic price forecast.

        The model receives the full ``price_series`` as context and
        produces ``horizon`` forward steps.

        Parameters
        ----------
        price_series : pd.Series
            Historical closing prices indexed by date (not returns).
            A minimum of 30 observations is recommended.
        horizon : int
            Number of trading days to forecast.  Default: 7.

        Returns
        -------
        dict
            ``{median, lower_80, upper_80, samples, horizon_days,
               last_price, expected_return}``

            All price values are in the same units as ``price_series``.
            ``expected_return`` is ``(median[-1] - last_price) / last_price``.
        """
        self.load()

        prices = price_series.dropna().values.astype(float)
        if len(prices) < 10:
            raise ValueError(
                f"price_series too short ({len(prices)} obs); need ≥ 10."
            )

        context = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)

        # Chronos returns (num_samples, batch=1, horizon)
        quantile_levels = [0.1, 0.5, 0.9]
        forecast_samples, quantiles, _ = self._pipeline.predict_quantiles(
            context=context,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
            num_samples=self.num_samples,
        )
        # quantiles shape: (batch=1, horizon, n_quantiles)
        q = quantiles[0].numpy()  # (horizon, 3)

        lower_80 = q[:, 0].tolist()
        median   = q[:, 1].tolist()
        upper_80 = q[:, 2].tolist()

        last_price     = float(prices[-1])
        median_final   = float(median[-1])
        expected_return = (median_final - last_price) / last_price

        return {
            "horizon_days":     horizon,
            "last_price":       round(last_price, 4),
            "median":           [round(v, 4) for v in median],
            "lower_80":         [round(v, 4) for v in lower_80],
            "upper_80":         [round(v, 4) for v in upper_80],
            "expected_return":  round(expected_return, 6),
        }

    def chronos_trend(
        self,
        price_series: pd.Series,
        horizon: int = 7,
    ) -> float:
        """
        Convenience wrapper that returns just the ``expected_return``
        scalar — used as the ``chronos_trend`` feature in XGBRanker.

        Parameters
        ----------
        price_series : pd.Series
            Historical closing prices.
        horizon : int
            Forecast horizon in trading days.

        Returns
        -------
        float
            Median expected return over the horizon.
        """
        result = self.forecast(price_series, horizon=horizon)
        return result["expected_return"]


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------


def _main() -> None:
    import yfinance as yf

    ticker = "AAPL"
    hist   = yf.Ticker(ticker).history(period="1y", interval="1d")
    prices = hist["Close"].dropna()

    print(f"Chronos-T5 forecast for {ticker}  (last price: {prices.iloc[-1]:.2f})\n")

    engine = ForecastEngine()
    engine.load()

    result = engine.forecast(prices, horizon=7)

    print(f"  Horizon          : {result['horizon_days']} days")
    print(f"  Last price       : {result['last_price']}")
    print(f"  Expected return  : {result['expected_return']:+.4%}")
    print(f"  Median forecast  : {result['median']}")
    print(f"  80% CI lower     : {result['lower_80']}")
    print(f"  80% CI upper     : {result['upper_80']}")


if __name__ == "__main__":
    _main()
