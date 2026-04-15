"""
quant_models.py — quantitative signal generation for the stock-analysis system.

Classes
-------
QuantModels
    fit_egarch(returns_series)      → annualised EGARCH volatility forecast
    fama_french_alpha(ticker, ...)  → FF5 alpha + factor loadings
    iv_garch_spread(iv, garch_vol)  → risk-premium signal

All methods are synchronous (CPU-bound); call them inside
``asyncio.get_event_loop().run_in_executor(None, ...)`` from async code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from arch import arch_model

from config import settings


class QuantModels:
    """
    Collection of quantitative-finance signal generators.

    All public methods are pure functions (no instance state required).
    They are defined as regular methods for easy mocking in tests.
    """

    # ==================================================================
    # EGARCH volatility forecast
    # ==================================================================

    def fit_egarch(self, returns_series: pd.Series) -> dict[str, Any]:
        """
        Fit an EGARCH(1,1) model to a returns series and return the
        one-step-ahead annualised volatility forecast.

        Parameters
        ----------
        returns_series : pd.Series
            Daily log-returns (or percentage returns) in decimal form,
            e.g. ``np.log(prices).diff().dropna()``.

        Returns
        -------
        dict
            ``{vol_forecast, aic, bic, params}``
            ``vol_forecast`` is annualised (×√252).
        """
        # Scale to percentage returns for numerical stability
        r = returns_series.dropna() * 100

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(r, vol="EGARCH", p=1, q=1, dist="Normal")
            result = model.fit(disp="off", show_warning=False)

        # One-step-ahead conditional variance forecast
        forecast = result.forecast(horizon=1, reindex=False)
        daily_var = float(forecast.variance.iloc[-1, 0])
        daily_vol = np.sqrt(max(daily_var, 0)) / 100  # back to decimal
        annual_vol = daily_vol * np.sqrt(252)

        return {
            "vol_forecast":  round(annual_vol, 6),
            "daily_vol":     round(daily_vol,  6),
            "aic":           round(float(result.aic), 4),
            "bic":           round(float(result.bic), 4),
            "params":        result.params.to_dict(),
        }

    # ==================================================================
    # Fama-French 5-factor alpha
    # ==================================================================

    def fama_french_alpha(
        self,
        ticker: str,
        returns: pd.Series,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """
        Regress ticker returns against the Fama-French 5 factors (daily) and
        return the annualised alpha plus factor loadings.

        Fetches FF5 factor data from Kenneth French's data library via
        ``pandas_datareader``.

        Parameters
        ----------
        ticker : str
            Used only for labelling output.
        returns : pd.Series
            Daily returns indexed by ``pd.DatetimeIndex`` (decimal form).
        start : str, optional
            Start date ``"YYYY-MM-DD"`` for factor data.  Defaults to the
            start of ``returns``.
        end : str, optional
            End date.  Defaults to the end of ``returns``.

        Returns
        -------
        dict
            ``{alpha_annual, alpha_daily, mkt_rf, smb, hml, rmw, cma,
               r_squared, n_obs}``
        """
        from sklearn.linear_model import LinearRegression

        start = start or str(returns.index[0].date())
        end   = end   or str(returns.index[-1].date())

        # ---- fetch FF5 daily factors --------------------------------
        try:
            ff = web.DataReader(
                settings.ff_dataset,
                "famafrench",
                start=start,
                end=end,
            )[0]
        except Exception as exc:
            return {"error": f"Could not fetch FF5 factors: {exc}"}

        # FF data is in percent; convert to decimal
        ff = ff / 100
        ff.index = pd.to_datetime(ff.index)

        # Align on common dates
        ret = returns.rename("ret")
        merged = ff.join(ret, how="inner").dropna()
        if len(merged) < 30:
            return {"error": f"Insufficient overlapping observations ({len(merged)})"}

        excess_ret = merged["ret"] - merged["RF"]
        X = merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values
        y = excess_ret.values

        reg = LinearRegression(fit_intercept=True).fit(X, y)
        alpha_daily  = float(reg.intercept_)
        alpha_annual = alpha_daily * 252
        r_sq = float(reg.score(X, y))

        loadings = dict(zip(["mkt_rf", "smb", "hml", "rmw", "cma"], reg.coef_.tolist()))

        return {
            "ticker":        ticker,
            "alpha_annual":  round(alpha_annual, 6),
            "alpha_daily":   round(alpha_daily,  8),
            "r_squared":     round(r_sq, 4),
            "n_obs":         len(merged),
            **{k: round(v, 6) for k, v in loadings.items()},
        }

    # ==================================================================
    # IV–GARCH spread signal
    # ==================================================================

    def iv_garch_spread(
        self, iv: float | None, garch_vol: float | None
    ) -> float | None:
        """
        Compute the spread between market-implied volatility and the
        EGARCH realised-volatility forecast.

        A positive spread means the options market is pricing in *more*
        risk than the historical model suggests (potential mean-reversion
        opportunity, or genuine fear premium).

        Parameters
        ----------
        iv : float or None
            ATM implied volatility (annualised, decimal).
        garch_vol : float or None
            Annualised EGARCH volatility forecast (``vol_forecast`` from
            :meth:`fit_egarch`).

        Returns
        -------
        float or None
            ``iv - garch_vol``, or ``None`` if either input is missing.
        """
        if iv is None or garch_vol is None:
            return None
        return round(float(iv) - float(garch_vol), 6)


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------


def _main() -> None:
    import yfinance as yf

    ticker = "AAPL"
    print(f"QuantModels smoke-test for {ticker}\n")
    qm = QuantModels()

    hist = yf.Ticker(ticker).history(period="2y", interval="1d")
    log_returns = np.log(hist["Close"]).diff().dropna()
    log_returns.index = pd.to_datetime(log_returns.index).tz_localize(None)

    print("1) EGARCH(1,1) fit …")
    egarch = qm.fit_egarch(log_returns)
    print(f"   Annualised vol forecast : {egarch['vol_forecast']:.4%}")
    print(f"   AIC / BIC               : {egarch['aic']} / {egarch['bic']}\n")

    print("2) Fama-French 5-factor alpha …")
    ff = qm.fama_french_alpha(ticker, log_returns)
    if "error" in ff:
        print(f"   {ff['error']}\n")
    else:
        print(f"   Alpha (annualised) : {ff['alpha_annual']:.4%}")
        print(f"   R²                 : {ff['r_squared']:.4f}")
        print(f"   Mkt-RF beta        : {ff['mkt_rf']:.4f}\n")

    print("3) IV–GARCH spread …")
    spread = qm.iv_garch_spread(iv=0.28, garch_vol=egarch["vol_forecast"])
    print(f"   Spread (IV=0.28, GARCH={egarch['vol_forecast']:.4f}): {spread:.4f}")


if __name__ == "__main__":
    _main()
