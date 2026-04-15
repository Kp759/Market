"""
main.py — CLI entry-point for the stock-analysis system.

Usage
-----
Live analysis::

    python main.py --tickers AAPL NVDA MSFT TSLA AMZN --mode live

Backtest::

    python main.py --mode backtest --start 2023-01-01 --end 2024-12-31

Train XGBoost model on historical recommendations::

    python main.py --tickers AAPL NVDA MSFT --mode train

Options
-------
--tickers   One or more ticker symbols (default: AAPL NVDA MSFT TSLA AMZN)
--mode      live | backtest | train  (default: live)
--start     Backtest start date YYYY-MM-DD (default: 2023-01-01)
--end       Backtest end date   YYYY-MM-DD (default: today)
--db        SQLite database path (default: settings.db_path)
--backend   LLM backend: claude | vllm (overrides .env)
--no-llm    Skip LLM call; show composite scores only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date

import pandas as pd

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

DEFAULT_TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def run_live(
    tickers: list[str],
    db_path: str,
    llm_backend: str,
    no_llm: bool,
) -> None:
    """Fetch live data, compute all signals, print ranked table."""
    if no_llm:
        await _run_signals_only(tickers, db_path)
        return

    from orchestrator import Orchestrator

    orch = Orchestrator(llm_backend=llm_backend, db_path=db_path)
    logger.info("Running live pipeline for %s …", tickers)
    df = await orch.run(tickers)

    _print_table(df)


async def _run_signals_only(tickers: list[str], db_path: str) -> None:
    """Run data ingestion + quant signals only (no LLM call)."""
    import numpy as np
    from data_ingestion import DataIngestion
    from quant_models import QuantModels
    from sentiment import SentimentAnalyzer

    qm       = QuantModels()
    analyzer = SentimentAnalyzer()
    analyzer.load()

    async with DataIngestion(db_path) as ing:
        raw = await ing.get_stock_data(tickers)

    rows = []
    for ticker, d in raw.items():
        prices = d.get("prices", [])
        fund   = d.get("fundamentals") or {}

        egarch_vol = None
        if len(prices) >= 30:
            closes  = np.array([p["close"] for p in prices], dtype=float)
            log_ret = pd.Series(np.log(closes)).diff().dropna()
            try:
                eg = qm.fit_egarch(log_ret)
                egarch_vol = eg["vol_forecast"]
            except Exception:
                pass

        news   = d.get("news", [])
        texts  = [a.get("title", "") for a in news] + [a.get("description", "") for a in news]
        sent   = analyzer.score_articles([t for t in texts if t])

        rows.append(
            {
                "ticker":     ticker,
                "sentiment":  round(sent, 4),
                "egarch_vol": round(egarch_vol, 4) if egarch_vol else None,
                "pe_ratio":   fund.get("pe_ratio"),
                "eps":        fund.get("eps"),
                "market_cap": fund.get("market_cap"),
                "iv":         d.get("options_iv"),
            }
        )

    df = pd.DataFrame(rows)
    _print_table(df)


def run_backtest(start: str, end: str, db_path: str) -> None:
    """Replay stored recommendations and print performance metrics."""
    from backtest import Backtester

    bt = Backtester(db_path=db_path)
    try:
        result = bt.run(start=start, end=end)
        print(result.summary())
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)


async def run_train(tickers: list[str], db_path: str) -> None:
    """
    Build training data from stored recommendations and fit XGBoost model.

    Computes realised 1-month forward returns for each ticker on each
    recommendation date, then trains the XGBoost regressor.
    """
    import numpy as np
    import yfinance as yf
    from backtest import Backtester
    from xgb_ranker import ReturnPredictor, FEATURE_COLS

    logger.info("Loading stored recommendations …")
    bt   = Backtester(db_path=db_path)
    recs = bt.load_recommendations(start="2020-01-01", end=str(date.today()))

    if recs.empty:
        logger.error("No recommendations found. Run --mode live first.")
        sys.exit(1)

    all_tickers = sorted(recs["ticker"].unique().tolist())
    logger.info("Downloading price data for forward-return computation …")
    prices = yf.download(
        all_tickers,
        start=str(recs["date"].min().date()),
        end=str(date.today()),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"].ffill()

    predictor = ReturnPredictor()
    feature_rows = []
    return_rows  = []

    for _, row in recs.iterrows():
        t    = row["ticker"]
        dt   = row["date"]
        if t not in prices.columns:
            continue
        fwd_idx = prices.index.searchsorted(dt)
        if fwd_idx + 21 >= len(prices):
            continue
        p0 = prices[t].iloc[fwd_idx]
        p1 = prices[t].iloc[fwd_idx + 21]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue
        fwd_return = (p1 - p0) / p0

        feature_rows.append({"ticker": t, "score": row["score"]})
        return_rows.append({"ticker": t, "fwd_return": fwd_return})

    if not feature_rows:
        logger.error("Could not construct any feature rows. Aborting.")
        sys.exit(1)

    feat_df = pd.DataFrame(feature_rows).set_index("ticker")
    # Pad missing features with 0
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[FEATURE_COLS]

    fwd_series = pd.DataFrame(return_rows).set_index("ticker")["fwd_return"]

    predictor.train(feat_df, fwd_series)
    logger.info("XGBoost model trained and saved → %s", predictor.model_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_table(df: pd.DataFrame) -> None:
    """Pretty-print a DataFrame with column alignment."""
    pd.set_option("display.max_colwidth",  60)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df.to_string(index=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock analysis system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        metavar="TICKER", help="Ticker symbols to analyse",
    )
    parser.add_argument(
        "--mode", choices=["live", "backtest", "train"],
        default="live", help="Pipeline mode",
    )
    parser.add_argument(
        "--start", default="2023-01-01", metavar="YYYY-MM-DD",
        help="Backtest start date",
    )
    parser.add_argument(
        "--end", default=str(date.today()), metavar="YYYY-MM-DD",
        help="Backtest end date",
    )
    parser.add_argument(
        "--db", default=settings.db_path, metavar="PATH",
        help="SQLite database path",
    )
    parser.add_argument(
        "--backend", choices=["claude", "vllm"],
        default=settings.llm_backend,
        help="LLM backend (overrides .env LLM_BACKEND)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM call; print signal scores only",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "live":
        asyncio.run(
            run_live(
                tickers=args.tickers,
                db_path=args.db,
                llm_backend=args.backend,
                no_llm=args.no_llm,
            )
        )

    elif args.mode == "backtest":
        run_backtest(start=args.start, end=args.end, db_path=args.db)

    elif args.mode == "train":
        asyncio.run(run_train(tickers=args.tickers, db_path=args.db))


if __name__ == "__main__":
    main()
