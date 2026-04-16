"""
backtest.py — walk-forward vectorbt backtester for screener recommendations.

Class
-----
Backtester
    run(start, end) → BacktestResult

Strategy
--------
* Monthly rebalancing: on the first trading day of each month, fetch
  the stored recommendations and go long-only, equal-weight on all
  ``buy`` tickers.
* Performance is compared against an SPY buy-and-hold benchmark.

Metrics
-------
    Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio,
    cumulative return, annualised return, hit rate (% of months
    the portfolio beat SPY), SPY comparison stats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""
    total_return:       float
    annualised_return:  float
    sharpe_ratio:       float
    sortino_ratio:      float
    max_drawdown:       float
    calmar_ratio:       float
    hit_rate:           float    # % of monthly periods beating SPY
    spy_total_return:   float
    spy_sharpe:         float
    n_rebalance_dates:  int
    equity_curve:       pd.Series = field(repr=False)
    spy_curve:          pd.Series = field(repr=False)
    monthly_pnl:        pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Walk-Forward Backtest Results",
            "=" * 60,
            f"  Total return         : {self.total_return:+.2%}",
            f"  Annualised return    : {self.annualised_return:+.2%}",
            f"  Sharpe ratio         : {self.sharpe_ratio:.3f}",
            f"  Sortino ratio        : {self.sortino_ratio:.3f}",
            f"  Max drawdown         : {self.max_drawdown:.2%}",
            f"  Calmar ratio         : {self.calmar_ratio:.3f}",
            f"  Hit rate vs SPY      : {self.hit_rate:.1%}",
            f"  SPY total return     : {self.spy_total_return:+.2%}",
            f"  SPY Sharpe           : {self.spy_sharpe:.3f}",
            f"  Rebalance periods    : {self.n_rebalance_dates}",
            "=" * 60,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtest replaying stored recommendations.

    Parameters
    ----------
    db_path : str, optional
        SQLite path.
    risk_free_rate : float
        Annual risk-free rate for Sharpe / Sortino.  Default: 0.05.
    """

    def __init__(
        self,
        db_path: str | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        self._db_path       = db_path or settings.db_path
        self.risk_free_rate = risk_free_rate

    # ==================================================================
    # Load stored recommendations
    # ==================================================================

    def load_recommendations(self, start: str, end: str) -> pd.DataFrame:
        """
        Query the ``recommendations`` table for records in [start, end].

        Returns
        -------
        pd.DataFrame
            Columns: date, ticker, recommendation, score.
        """
        import sqlite3

        start_ts = datetime.strptime(start, "%Y-%m-%d").timestamp()
        end_ts   = datetime.strptime(end,   "%Y-%m-%d").timestamp()

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                "SELECT fetched_at, ticker, recommendation, score "
                "FROM recommendations "
                "WHERE fetched_at BETWEEN ? AND ? "
                "ORDER BY fetched_at",
                conn,
                params=(start_ts, end_ts),
            )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["fetched_at"], unit="s").dt.normalize()
        return df[["date", "ticker", "recommendation", "score"]]

    # ==================================================================
    # Run walk-forward backtest
    # ==================================================================

    def run(self, start: str, end: str) -> BacktestResult:
        """
        Execute the walk-forward backtest.

        On each monthly rebalance date, the portfolio holds equal-weight
        positions in all tickers with ``recommendation == "buy"``.

        Parameters
        ----------
        start, end : str
            Date range ``"YYYY-MM-DD"``.

        Returns
        -------
        BacktestResult
        """
        import vectorbt as vbt
        import yfinance as yf

        recs = self.load_recommendations(start, end)
        if recs.empty:
            raise ValueError(
                "No recommendations found in the database for the given range. "
                "Run main.py --mode screen first."
            )

        all_tickers     = sorted(recs["ticker"].unique().tolist())
        rebalance_dates = sorted(recs["date"].unique())

        # Download price data
        logger.info("Downloading price data for %d tickers …", len(all_tickers) + 1)
        prices_raw = yf.download(
            all_tickers + ["SPY"],
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]
        prices_raw = prices_raw.ffill().dropna(how="all")

        spy_prices = prices_raw["SPY"].dropna()
        prices     = prices_raw.drop(columns=["SPY"], errors="ignore")
        prices     = prices[[t for t in all_tickers if t in prices.columns]]

        # Build daily target-weight matrix
        weights = self._build_weights(
            recs, rebalance_dates, prices.index, prices.columns.tolist()
        )

        # vectorbt Portfolio
        pf = vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            freq="D",
            init_cash=100_000,
        )

        equity     = pf.value() / 100_000 - 1
        daily_ret  = pf.returns()
        rf_daily   = self.risk_free_rate / 252

        # ---- metrics ----
        total_return      = float(equity.iloc[-1])
        n_days            = max(len(daily_ret), 1)
        annualised_return = float((1 + total_return) ** (252 / n_days) - 1)
        sharpe            = self._sharpe(daily_ret.values, rf_daily)
        sortino           = self._sortino(daily_ret.values, rf_daily)
        max_dd            = float(pf.max_drawdown())
        calmar            = annualised_return / abs(max_dd) if max_dd != 0 else 0.0

        # ---- SPY benchmark ----
        spy_ret   = spy_prices.pct_change().dropna()
        spy_curve = (1 + spy_ret).cumprod() - 1
        spy_total = float(spy_curve.iloc[-1])
        spy_sh    = self._sharpe(spy_ret.values, rf_daily)

        # ---- Monthly PnL & hit rate ----
        monthly_pnl = self._monthly_pnl(equity, spy_curve, rebalance_dates)
        beat_months = (monthly_pnl["port_ret"] > monthly_pnl["spy_ret"]).sum()
        hit_rate    = beat_months / max(len(monthly_pnl), 1)

        return BacktestResult(
            total_return=total_return,
            annualised_return=annualised_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            hit_rate=float(hit_rate),
            spy_total_return=spy_total,
            spy_sharpe=spy_sh,
            n_rebalance_dates=len(rebalance_dates),
            equity_curve=equity,
            spy_curve=spy_curve,
            monthly_pnl=monthly_pnl,
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _build_weights(
        recs: pd.DataFrame,
        rebalance_dates: list,
        price_index: pd.DatetimeIndex,
        tickers: list[str],
    ) -> pd.DataFrame:
        """Build a (date × ticker) target-weight DataFrame."""
        weight_df = pd.DataFrame(0.0, index=price_index, columns=tickers)
        current_weights = {t: 0.0 for t in tickers}

        for reb_date in rebalance_dates:
            day_recs = recs[recs["date"] == reb_date]
            buys = day_recs[day_recs["recommendation"] == "buy"]["ticker"].tolist()
            current_weights = {t: 0.0 for t in tickers}
            if buys:
                w = 1.0 / len(buys)
                for t in buys:
                    if t in current_weights:
                        current_weights[t] = w
            mask = price_index >= reb_date
            for t, wt in current_weights.items():
                weight_df.loc[mask, t] = wt

        return weight_df

    @staticmethod
    def _monthly_pnl(
        equity: pd.Series,
        spy_curve: pd.Series,
        rebalance_dates: list,
    ) -> pd.DataFrame:
        """Compute portfolio vs SPY return between consecutive rebalance dates."""
        rows = []
        dates = sorted([pd.Timestamp(d) for d in rebalance_dates])
        for i in range(len(dates) - 1):
            d0, d1 = dates[i], dates[i + 1]
            try:
                p0 = equity.asof(d0) if not equity.empty else 0.0
                p1 = equity.asof(d1) if not equity.empty else 0.0
                s0 = spy_curve.asof(d0) if not spy_curve.empty else 0.0
                s1 = spy_curve.asof(d1) if not spy_curve.empty else 0.0
                # Convert cumulative levels back to period returns
                port_ret = (1 + p1) / (1 + p0) - 1
                spy_ret  = (1 + s1) / (1 + s0) - 1
                rows.append({"period_start": d0, "period_end": d1,
                             "port_ret": port_ret, "spy_ret": spy_ret})
            except Exception:
                pass
        return pd.DataFrame(rows)

    @staticmethod
    def _sharpe(returns: np.ndarray, rf: float) -> float:
        excess = returns - rf
        std    = excess.std()
        return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    @staticmethod
    def _sortino(returns: np.ndarray, rf: float) -> float:
        excess   = returns - rf
        downside = excess[excess < 0]
        dstd     = downside.std()
        return float(excess.mean() / dstd * np.sqrt(252)) if dstd > 0 else 0.0


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

def _main() -> None:
    bt = Backtester()
    try:
        result = bt.run(start="2023-01-01", end="2024-12-31")
        print(result.summary())
        print("\nMonthly PnL sample:")
        print(result.monthly_pnl.head(6).to_string(index=False))
    except ValueError as exc:
        print(f"[backtest] {exc}")
        print("Tip: run  python main.py --mode screen  first.")


if __name__ == "__main__":
    _main()
