"""
signals.py — pure signal-computation layer for the stock screener.

All methods are synchronous, stateless, and never raise — they return
``None`` (or a neutral value) when data is insufficient, logging a
debug message.

Class
-----
SignalEngine
    momentum_signal(prices)                      → float | None
    sue_score(actual, estimate, std)             → float | None
    short_squeeze_score(si, avg_vol, float_sh)   → float | None
    insider_signal(transactions)                 → float
    revenue_acceleration(quarterly_revenue)      → float | None

These are designed to be called on per-ticker data dicts and then
assembled by Screener.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Insider role weights: higher = more informative signal
_ROLE_WEIGHTS: dict[str, float] = {
    "CEO":       3.0,
    "CFO":       2.5,
    "COO":       2.0,
    "President": 2.0,
    "Director":  1.5,
    "Officer":   1.2,
    "Other":     1.0,
}


class SignalEngine:
    """
    Collection of signal-generation functions for the screener pipeline.

    All public methods return ``float | None``.  ``None`` means the signal
    could not be computed (insufficient data) and the Screener will
    substitute the cross-sectional median before scoring.
    """

    # ------------------------------------------------------------------
    # 1. Price momentum — 12-1 month return (skip last month)
    # ------------------------------------------------------------------

    def momentum_signal(self, prices: list[dict[str, Any]]) -> float | None:
        """
        Compute the 12-1 month momentum signal.

        Strategy: return over months [-12, -1] (i.e. skip the most
        recent 21 trading days to avoid short-term reversal).

        Parameters
        ----------
        prices : list[dict]
            OHLCV records ``{date, close, …}``.  Must contain ≥ 250 rows
            (≈ 12 months of daily data) for a valid signal.

        Returns
        -------
        float or None
            Log return from day -252 to day -21 relative to today.
        """
        if len(prices) < 252:
            logger.debug("momentum_signal: insufficient data (%d rows)", len(prices))
            return None
        try:
            closes   = np.array([p["close"] for p in prices], dtype=float)
            p_start  = closes[-252]
            p_end    = closes[-21]       # skip last ~1 month
            if p_start <= 0:
                return None
            return float(np.log(p_end / p_start))
        except Exception as exc:
            logger.debug("momentum_signal: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 2. SUE — standardised unexpected earnings
    # ------------------------------------------------------------------

    def sue_score(
        self,
        actual: float | None,
        estimate: float | None,
        historical_surprises: list[float] | None = None,
    ) -> float | None:
        """
        Compute the Standardised Unexpected Earnings (SUE) score.

            SUE = (actual - estimate) / σ(historical_surprises)

        If no historical surprise std is available, falls back to
        ``|estimate|`` as the denominator (scaled SUE).

        Parameters
        ----------
        actual : float or None
            Most-recent EPS actual.
        estimate : float or None
            Analyst EPS estimate.
        historical_surprises : list[float] or None
            Recent EPS surprises (actual − estimate per quarter).
            Used to compute σ for normalisation.

        Returns
        -------
        float or None
            SUE score, clamped to [-5, 5].  None if inputs are missing.
        """
        if actual is None or estimate is None:
            return None
        try:
            surprise = float(actual) - float(estimate)
            if historical_surprises and len(historical_surprises) >= 3:
                std = float(np.std(historical_surprises, ddof=1))
            else:
                std = abs(float(estimate)) if float(estimate) != 0 else 1.0
            std = max(std, 1e-6)
            return float(np.clip(surprise / std, -5.0, 5.0))
        except Exception as exc:
            logger.debug("sue_score: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 3. Short squeeze score
    # ------------------------------------------------------------------

    def short_squeeze_score(
        self,
        short_interest: float | None,
        avg_volume: float | None,
        float_shares: float | None,
    ) -> float | None:
        """
        Compute a short-squeeze composite score.

        Components
        ----------
        days_to_cover  = short_interest / avg_daily_volume
        float_short_%  = short_interest / float_shares

        Score = 0.6 × z(days_to_cover) + 0.4 × z(float_short_%)
        where z() is a soft normalisation: raw value / 10.

        Clamped to [0, 1].

        Parameters
        ----------
        short_interest : float or None
            Number of shares sold short.
        avg_volume : float or None
            Average daily trading volume.
        float_shares : float or None
            Floating shares.

        Returns
        -------
        float or None
            Score in [0, 1].  None if all inputs are missing.
        """
        if short_interest is None or short_interest <= 0:
            return None
        try:
            si  = float(short_interest)
            dtc = si / float(avg_volume)  if avg_volume and avg_volume > 0 else None
            fsp = si / float(float_shares) if float_shares and float_shares > 0 else None

            if dtc is None and fsp is None:
                return None

            score = 0.0
            weight = 0.0
            if dtc is not None:
                score  += 0.6 * min(dtc / 10.0, 1.0)
                weight += 0.6
            if fsp is not None:
                score  += 0.4 * min(fsp, 1.0)
                weight += 0.4

            return round(float(score / weight) if weight > 0 else 0.0, 6)
        except Exception as exc:
            logger.debug("short_squeeze_score: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 4. Insider signal
    # ------------------------------------------------------------------

    def insider_signal(
        self, transactions: list[dict[str, Any]]
    ) -> float:
        """
        Compute a role-weighted insider buy/sell signal over the last 30 days.

        Weighting
        ---------
        CEO = 3×, CFO = 2.5×, COO / President = 2×,
        Director = 1.5×, Officer = 1.2×, Other = 1×

        Each buy contributes +weight, each sell contributes -weight.
        The final score is normalised by the total weight and clamped to [-1, 1].

        Parameters
        ----------
        transactions : list[dict]
            Form-4 dicts with keys ``role`` and ``transaction_type``.
            ``transaction_type`` values: ``"P"`` (purchase), ``"S"`` (sale),
            or ``"unknown"`` (EDGAR title only; treated as 0).

        Returns
        -------
        float
            Score in [-1, 1].  Returns 0.0 if no transactions.
        """
        if not transactions:
            return 0.0
        net    = 0.0
        total_w = 0.0
        for tx in transactions:
            role   = (tx.get("role") or "Other")
            weight = _ROLE_WEIGHTS.get(role, 1.0)
            ttype  = (tx.get("transaction_type") or "").upper()
            if ttype in ("P", "BUY", "PURCHASE"):
                net     += weight
                total_w += weight
            elif ttype in ("S", "SELL", "SALE"):
                net     -= weight
                total_w += weight
            else:
                total_w += weight  # unknown — counts toward denominator

        if total_w == 0:
            return 0.0
        return float(np.clip(net / total_w, -1.0, 1.0))

    # ------------------------------------------------------------------
    # 5. Revenue acceleration — second derivative of quarterly growth
    # ------------------------------------------------------------------

    def revenue_acceleration(
        self, quarterly_revenue: list[float]
    ) -> float | None:
        """
        Compute the second derivative (acceleration) of revenue growth.

        Steps
        -----
        1. Compute QoQ growth rates from the raw revenue series.
        2. Compute the first difference of those growth rates → acceleration.
        3. Return the most-recent acceleration value.

        A positive value means growth is *speeding up*; negative means
        it is *slowing down* (even if still positive).

        Parameters
        ----------
        quarterly_revenue : list[float]
            Quarterly revenue figures in **chronological order** (oldest
            first).  Minimum 4 quarters required.

        Returns
        -------
        float or None
            Most-recent revenue acceleration.
        """
        if not quarterly_revenue or len(quarterly_revenue) < 4:
            return None
        try:
            rev = np.array(quarterly_revenue, dtype=float)
            # QoQ growth: avoid division by zero
            growth = np.diff(rev) / np.where(rev[:-1] != 0, np.abs(rev[:-1]), 1.0)
            if len(growth) < 2:
                return None
            accel = np.diff(growth)
            return float(accel[-1])
        except Exception as exc:
            logger.debug("revenue_acceleration: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Convenience: compute all signals for one ticker's data dict
    # ------------------------------------------------------------------

    def compute_all(self, ticker: str, d: dict[str, Any]) -> dict[str, Any]:
        """
        Compute all five signals for one ticker given its full data dict.

        Parameters
        ----------
        ticker : str
            Ticker symbol (used only for logging).
        d : dict
            Output of ``DataIngestion.get_stock_data`` for a single ticker.

        Returns
        -------
        dict
            ``{momentum, sue_score, short_squeeze, insider_signal,
               revenue_accel}``
        """
        prices  = d.get("prices", [])
        earn    = d.get("earnings", {}) or {}
        si_data = d.get("short_interest", {}) or {}
        insider = d.get("insider_transactions", []) or []

        # momentum
        momentum = self.momentum_signal(prices)

        # SUE
        actual   = earn.get("actual")
        estimate = earn.get("estimate")
        sue      = self.sue_score(actual, estimate)

        # Short squeeze
        squeeze = self.short_squeeze_score(
            si_data.get("short_interest"),
            si_data.get("avg_volume"),
            si_data.get("float_shares"),
        )

        # Insider
        ins_sig = self.insider_signal(insider)

        # Revenue acceleration
        q_rev = earn.get("quarterly_revenue", [])
        # Reverse if newest-first (yfinance returns newest first in some fields)
        if q_rev and len(q_rev) >= 2 and q_rev[0] < q_rev[-1]:
            q_rev = list(reversed(q_rev))
        rev_accel = self.revenue_acceleration(q_rev)

        return {
            "momentum":      momentum,
            "sue_score":     sue,
            "short_squeeze": squeeze,
            "insider_signal":ins_sig,
            "revenue_accel": rev_accel,
        }


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

def _main() -> None:
    import yfinance as yf
    from data_ingestion import _infer_role

    se = SignalEngine()

    # --- momentum ---
    hist = yf.Ticker("AAPL").history(period="15mo", interval="1d")
    prices = [
        {"date": str(ts), "close": float(row["Close"])}
        for ts, row in hist.iterrows()
    ]
    mom = se.momentum_signal(prices)
    print(f"Momentum (AAPL): {mom:.4f}" if mom else "Momentum: n/a")

    # --- SUE ---
    sue = se.sue_score(actual=1.52, estimate=1.43, historical_surprises=[0.05, 0.08, 0.03, 0.06])
    print(f"SUE: {sue:.4f}")

    # --- short squeeze ---
    sq = se.short_squeeze_score(short_interest=50_000_000, avg_volume=80_000_000, float_shares=1_000_000_000)
    print(f"Short squeeze: {sq:.4f}")

    # --- insider signal ---
    txns = [
        {"role": "CEO", "transaction_type": "P"},
        {"role": "CFO", "transaction_type": "S"},
        {"role": "Director", "transaction_type": "P"},
    ]
    ins = se.insider_signal(txns)
    print(f"Insider signal: {ins:.4f}")

    # --- revenue acceleration ---
    q_rev = [8_000, 8_500, 9_200, 10_100, 11_500, 13_200, 14_800, 16_500]  # millions
    ra = se.revenue_acceleration(q_rev)
    print(f"Revenue acceleration: {ra:.4f}" if ra else "Rev accel: n/a")


if __name__ == "__main__":
    _main()
