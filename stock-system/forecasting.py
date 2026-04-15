"""
forecasting.py — Chronos-T5 time-series forecasting for price data.

Planned functionality
---------------------
* Load amazon/chronos-t5-small (or -large) from HuggingFace.
* Accept OHLCV records (as returned by DataIngestion.get_stock_data).
* Produce a probabilistic forecast horizon of N trading days.
* Return median forecast + 10th/90th percentile bands.

Status: stub — implementation coming in the next sprint.
"""

from __future__ import annotations

from typing import Any


class PriceForecaster:
    """Chronos-T5 time-series forecaster (stub)."""

    async def forecast(
        self, ticker: str, prices: list[dict[str, Any]], horizon_days: int = 5
    ) -> dict[str, Any]:
        raise NotImplementedError("PriceForecaster is not yet implemented.")
