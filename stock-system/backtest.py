"""
backtest.py — simple vectorbt-based backtester for stored recommendations.

Class
-----
Backtester
    load_recommendations(start, end) → pd.DataFrame
    run(start, end)                  → BacktestResult

The backtester replays historical recommendations stored in the SQLite
``recommendations`` table.  On each rebalance date it goes long the
tickers with a ``buy`` recommendation (equal-weight) and short the
``sell`` tickers (optional).  Results are compared against an SPY
benchmark.

Performance metrics returned
-----------------------------
    sharpe_ratio, max_drawdown, total_return, annualised_return,
    calmar_ratio, sortino_ratio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""
    total_return:      float
    annualised_return: float
    sharpe_ratio:      float
    sortino_ratio:     float
    max_drawdown:      float
    calmar_ratio:      float
    spy_total_return:  float
    spy_sharpe:        float
    n_rebalance_dates: int
    equity_curve:      pd.Series   # portfolio cumulative returns
    spy_curve:         pd.Series   # SPY cumulative returns

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Backtest Results",
            "=" * 55,
            f"  Total return       : {self.total_return:+.2%}",
            f"  Annualised return  : {self.annualised_return:+.2%}",
            f"  Sharpe ratio       : {self.sharpe_ratio:.3f}",
            f"  Sortino ratio      : {self.sortino_ratio:.3f}",
            f"  Max drawdown       : {self.max_drawdown:.2%}",
            f"  Calmar ratio       : {self.calmar_ratio:.3f}",
            f"  SPY total return   : {self.spy_total_return:+.2%}",
            f"  SPY Sharpe         : {self.spy_sharpe:.3f}",
            f"  Rebalance dates    : {self.n_rebalance_dates}",
            "=" * 55,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Replay historical recommendations and compute portfolio performance.

    Parameters
    ----------
    db_path : str, optional
        SQLite path.  Defaults to ``settings.db_path``.
    risk_free_rate : float
        Annual risk-free rate used for Sharpe/Sortino.  Default: 0.05.
    """

    def __init__(
        self,
        db_path: str | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        self._db_path      = db_path or settings.db_path
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Load recommendations from SQLite
    # ------------------------------------------------------------------

    def load_recommendations(
        self, start: str, end: str
    ) -> pd.DataFrame:
        """
        Query the ``recommendations`` table for records in [start, end].

        Parameters
        ----------
        start, end : str
            Date strings in ``"YYYY-MM-DD"`` format.

        Returns
        -------
        pd.DataFrame
            Columns: ``date, ticker, recommendation, score``.
        """
        import sqlite3

        start_ts = datetime.strptime(start, "%Y-%m-%d").timestamp()
        end_ts   = datetime.strptime(end,   "%Y-%m-%d").timestamp()

        conn = sqlite3.connect(self._db_path)
        try:
            df = pd.read_sql_query(
                "SELECT fetched_at, ticker, recommendation, score "
                "FROM recommendations "
                "WHERE fetched_at BETWEEN ? AND ? "
                "ORDER BY fetched_at",
                conn,
                params=(start_ts, end_ts),
            )
        finally:
            conn.close()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["fetched_at"], unit="s").dt.normalize()
        return df[["date", "ticker", "recommendation", "score"]]

    # ------------------------------------------------------------------
    # Run backtest
    # ------------------------------------------------------------------

    def run(
        self,
        start: str,
        end: str,
        long_only: bool = True,
    ) -> BacktestResult:
        """
        Execute the backtest between *start* and *end*.

        Strategy
        --------
        On each rebalance date, go equal-weight long on all ``buy``
        tickers.  If ``long_only=False``, also go equal-weight short on
        ``sell`` tickers.  Hold until the next rebalance date.

        Parameters
        ----------
        start, end : str
            Backtest date range ``"YYYY-MM-DD"``.
        long_only : bool
            If True, ignore sell signals.

        Returns
        -------
        BacktestResult
        """
        import vectorbt as vbt
        import yfinance as yf

        recs = self.load_recommendations(start, end)
        if recs.empty:
            raise ValueError(
                "No recommendations found in the database for the given date range."
            )

        rebalance_dates = sorted(recs["date"].unique())
        all_tickers = sorted(recs["ticker"].unique())

        # Download price data for all tickers + SPY
        download_tickers = all_tickers + ["SPY"]
        logger.info("Downloading price data for %s", download_tickers)
        prices = yf.download(
            download_tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]
        prices = prices.ffill().dropna(how="all")

        spy_prices = prices["SPY"]
        prices     = prices.drop(columns=["SPY"], errors="ignore")

        # Build daily portfolio weights
        weights = self._build_weights(
            recs, rebalance_dates, prices.index, all_tickers, long_only
        )

        # ---- vectorbt portfolio ------------------------------------------
        pf = vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",
            group_by=True,             # treat all tickers as one portfolio
            cash_sharing=True,
            freq="D",
            init_cash=100_000,
        )

        equity = pf.value() / 100_000 - 1          # cumulative return series
        daily_ret = pf.returns()

        # ---- SPY benchmark -----------------------------------------------
        spy_ret   = spy_prices.pct_change().dropna()
        spy_curve = (1 + spy_ret).cumprod() - 1

        # ---- Metrics -------------------------------------------------------
        rf_daily = self.risk_free_rate / 252

        total_return      = float(equity.iloc[-1])
        n_days            = len(daily_ret)
        annualised_return = float((1 + total_return) ** (252 / n_days) - 1)

        sharpe_ratio  = self._sharpe(daily_ret.values, rf_daily)
        sortino_ratio = self._sortino(daily_ret.values, rf_daily)
        max_drawdown  = float(pf.max_drawdown())
        calmar_ratio  = annualised_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        spy_total  = float(spy_curve.iloc[-1])
        spy_sharpe = self._sharpe(spy_ret.values, rf_daily)

        return BacktestResult(
            total_return=total_return,
            annualised_return=annualised_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            spy_total_return=spy_total,
            spy_sharpe=spy_sharpe,
            n_rebalance_dates=len(rebalance_dates),
            equity_curve=equity,
            spy_curve=spy_curve,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_weights(
        recs: pd.DataFrame,
        rebalance_dates: list,
        price_index: pd.DatetimeIndex,
        tickers: list[str],
        long_only: bool,
    ) -> pd.DataFrame:
        """Build a daily (date × ticker) target-weight DataFrame."""
        weight_df = pd.DataFrame(0.0, index=price_index, columns=tickers)

        current_weights = {t: 0.0 for t in tickers}

        for reb_date in rebalance_dates:
            day_recs = recs[recs["date"] == reb_date]
            buys  = day_recs[day_recs["recommendation"] == "buy"]["ticker"].tolist()
            sells = day_recs[day_recs["recommendation"] == "sell"]["ticker"].tolist()

            current_weights = {t: 0.0 for t in tickers}

            if buys:
                w = 1.0 / len(buys)
                for t in buys:
                    if t in tickers:
                        current_weights[t] = w

            if not long_only and sells:
                w = -1.0 / len(sells)
                for t in sells:
                    if t in tickers:
                        current_weights[t] = w

            # Apply from this rebalance date onwards
            mask = price_index >= reb_date
            for t, wt in current_weights.items():
                weight_df.loc[mask, t] = wt

        return weight_df

    @staticmethod
    def _sharpe(returns: np.ndarray, rf_daily: float) -> float:
        excess = returns - rf_daily
        std    = excess.std()
        return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    @staticmethod
    def _sortino(returns: np.ndarray, rf_daily: float) -> float:
        excess     = returns - rf_daily
        downside   = excess[excess < 0]
        down_std   = downside.std()
        return float(excess.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0.0


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------


def _main() -> None:
    bt = Backtester()
    try:
        result = bt.run(start="2023-01-01", end="2024-12-31")
        print(result.summary())
    except ValueError as exc:
        print(f"[backtest] {exc}")
        print("Tip: run main.py --mode live first to populate recommendations.")


if __name__ == "__main__":
    _main()
