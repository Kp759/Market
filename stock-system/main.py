"""
main.py — CLI entry-point for the stock-screening system.

Modes
-----
screen   (default)
    Build the ~1800-ticker universe, run the full screener pipeline,
    call the LLM orchestrator, and print the top-20 ranked picks.

    python main.py --mode screen

backtest
    Replay stored recommendations through the walk-forward vectorbt
    backtester and print performance metrics.

    python main.py --mode backtest --start 2023-01-01 --end 2024-12-31

train
    Build a panel of historical recommendations + realised 60-day
    forward returns and retrain the XGBoost model with walk-forward CV.

    python main.py --mode train

Options
-------
--mode      screen | backtest | train  (default: screen)
--tickers   Explicit ticker list (overrides universe scrape)
--top-n     Number of top picks to show (default: 20)
--start     Backtest start date  YYYY-MM-DD
--end       Backtest end date    YYYY-MM-DD
--db        SQLite database path
--backend   claude | vllm  (overrides .env LLM_BACKEND)
--no-llm    Skip LLM call; print screener scores only
--no-cache  Force-refresh universe and data (ignore TTL)
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


# ---------------------------------------------------------------------------
# Mode: screen
# ---------------------------------------------------------------------------

async def run_screen(
    tickers: list[str] | None,
    top_n: int,
    db_path: str,
    llm_backend: str,
    no_llm: bool,
) -> None:
    """
    Main screening pipeline.

    1. Fetch universe (~1800 tickers) or use *tickers* if supplied.
    2. Run Screener to get top_n candidates.
    3. Optionally call LLM Orchestrator for buy-list + reasoning.
    4. Print ranked table.
    """
    if not tickers:
        from universe import get_tickers_async
        logger.info("Fetching universe …")
        tickers = await get_tickers_async(db_path)
        logger.info("Universe size: %d tickers", len(tickers))

    from screener import Screener
    sc = Screener(db_path=db_path, top_n=top_n)
    logger.info("Running screener on %d tickers …", len(tickers))
    screener_df = await sc.screen(tickers, top_n=top_n)

    if no_llm:
        _print_table(screener_df)
        return

    from orchestrator import Orchestrator
    orch = Orchestrator(llm_backend=llm_backend, db_path=db_path)
    logger.info("Calling LLM orchestrator (%s) …", llm_backend)
    final_df = await orch.run(screener_df=screener_df)

    output_cols = [
        "ticker", "score", "recommendation",
        "position_size_kelly", "position_size_equal",
        "momentum", "sue_score", "short_squeeze",
        "sentiment_score", "egarch_vol", "iv_spread",
        "sector", "risk_flag",
    ]
    _print_table(final_df[[c for c in output_cols if c in final_df.columns]])

    # Also print justifications
    print("\n── Justifications ─────────────────────────────────────────")
    for _, row in final_df.iterrows():
        if row.get("recommendation") == "buy":
            print(f"\n  {row['ticker']}  [{row.get('recommendation','?')}]")
            print(f"  {row.get('justification','')}")
            if row.get("risk_flag"):
                print(f"  Risk: {row['risk_flag']}")


# ---------------------------------------------------------------------------
# Mode: backtest
# ---------------------------------------------------------------------------

def run_backtest(start: str, end: str, db_path: str) -> None:
    """Walk-forward backtest of stored recommendations."""
    from backtest import Backtester

    bt = Backtester(db_path=db_path)
    try:
        result = bt.run(start=start, end=end)
        print(result.summary())
        print()
        print("Monthly PnL:")
        _print_table(result.monthly_pnl)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

async def run_train(db_path: str, start: str = "2020-01-01") -> None:
    """
    Retrain the XGBoost model using walk-forward cross-validation.

    Loads all stored recommendations, downloads 60-day forward price
    returns for each, assembles a panel, and calls
    ``ReturnPredictor.train_walkforward()``.
    """
    import numpy as np
    import yfinance as yf
    from backtest import Backtester
    from xgb_ranker import ReturnPredictor, FEATURE_COLS

    logger.info("Loading stored recommendations for walk-forward training …")
    bt   = Backtester(db_path=db_path)
    recs = bt.load_recommendations(start=start, end=str(date.today()))

    if recs.empty:
        logger.error(
            "No recommendations found. Run --mode screen first to populate the DB."
        )
        sys.exit(1)

    all_tickers = sorted(recs["ticker"].unique().tolist())
    logger.info("Downloading prices for %d tickers …", len(all_tickers))
    prices = yf.download(
        all_tickers,
        start=start,
        end=str(date.today()),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"].ffill()

    panel_rows: list[dict] = []
    for _, row in recs.iterrows():
        t  = row["ticker"]
        dt = row["date"]
        if t not in prices.columns:
            continue
        idx = prices.index.searchsorted(dt)
        if idx + 60 >= len(prices):
            continue
        p0 = prices[t].iloc[idx]
        p1 = prices[t].iloc[idx + 60]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue
        fwd60 = (p1 - p0) / p0

        panel_rows.append({
            "date":          dt,
            "ticker":        t,
            "fwd_return_60d":fwd60,
            "score":         float(row.get("score") or 0),
            # remaining features default to 0; real pipeline would join signals
            **{col: 0.0 for col in FEATURE_COLS if col != "score"},
        })

    if not panel_rows:
        logger.error("Could not build any training rows. Aborting.")
        sys.exit(1)

    panel_df = pd.DataFrame(panel_rows)
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in panel_df.columns:
            panel_df[col] = 0.0

    predictor = ReturnPredictor()
    logger.info("Running walk-forward training on %d rows …", len(panel_df))
    oos_preds = predictor.train_walkforward(
        panel_df, forward_return_col="fwd_return_60d"
    )
    valid_oos = oos_preds.dropna()
    if len(valid_oos) > 0:
        corr = float(
            np.corrcoef(
                valid_oos.values,
                panel_df.loc[valid_oos.index, "fwd_return_60d"].values,
            )[0, 1]
        )
        logger.info(
            "Walk-forward OOS IC (rank corr proxy): %.4f  (%d samples)",
            corr, len(valid_oos),
        )
    logger.info("Training complete. Model saved → %s", predictor.model_path)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_table(df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth",  55)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + df.to_string(index=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stock screener — 1-3 month opportunity finder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["screen", "backtest", "train"],
        default="screen",
        help="Pipeline mode",
    )
    p.add_argument(
        "--tickers", nargs="+", default=None, metavar="TICKER",
        help="Explicit tickers (overrides universe scrape)",
    )
    p.add_argument(
        "--top-n", type=int, default=settings.top_n,
        metavar="N", help="Number of top picks to show",
    )
    p.add_argument(
        "--start", default="2023-01-01", metavar="YYYY-MM-DD",
        help="Backtest / training start date",
    )
    p.add_argument(
        "--end", default=str(date.today()), metavar="YYYY-MM-DD",
        help="Backtest end date",
    )
    p.add_argument(
        "--db", default=settings.db_path, metavar="PATH",
        help="SQLite database path",
    )
    p.add_argument(
        "--backend", choices=["claude", "vllm"],
        default=settings.llm_backend,
        help="LLM backend (overrides .env)",
    )
    p.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM call; show screener scores only",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "screen":
        asyncio.run(
            run_screen(
                tickers=args.tickers,
                top_n=args.top_n,
                db_path=args.db,
                llm_backend=args.backend,
                no_llm=args.no_llm,
            )
        )

    elif args.mode == "backtest":
        run_backtest(start=args.start, end=args.end, db_path=args.db)

    elif args.mode == "train":
        asyncio.run(run_train(db_path=args.db, start=args.start))


if __name__ == "__main__":
    main()
