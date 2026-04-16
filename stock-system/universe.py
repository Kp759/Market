"""
universe.py — stock universe builder for S&P 500, NASDAQ 100, and Russell 2000.

Public API
----------
    get_tickers(db_path=None) → list[str]

Sources
-------
    S&P 500   : Wikipedia (List_of_S%26P_500_companies)
    NASDAQ 100: Wikipedia (NASDAQ-100)
    Russell 2000: iShares IWM holdings CSV (primary) with HTTP fallback URL

Caching
-------
All ~1800 tickers are cached to SQLite (table: ``universe``) with a 24-hour
TTL.  Stale or missing cache triggers a fresh scrape.

Graceful degradation
--------------------
Each index is fetched independently.  If one source fails the others still
contribute to the final deduplicated list.  All errors are logged; the
function never raises.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
_NDX100_URL = "https://en.wikipedia.org/wiki/NASDAQ-100"

# Primary iShares CSV endpoint
_IWM_CSV_PRIMARY = (
    "https://www.ishares.com/us/products/239710/"
    "ishares-russell-2000-etf/1467271812596.ajax"
    "?tab=holdings&fileType=csv"
)
# Identical URL used as explicit fallback (spec requirement)
_IWM_CSV_FALLBACK = _IWM_CSV_PRIMARY

_UNIVERSE_TTL = 24 * 60 * 60  # 24 hours


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _ensure_universe_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS universe (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            data           TEXT    NOT NULL,
            fetched_at     REAL    NOT NULL,
            ttl_expires_at REAL    NOT NULL
        )
        """
    )
    await db.commit()


async def _cache_get(db: aiosqlite.Connection) -> list[str] | None:
    now = time.time()
    async with db.execute(
        "SELECT data FROM universe WHERE ttl_expires_at > ? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (now,),
    ) as cur:
        row = await cur.fetchone()
    return json.loads(row[0]) if row else None


async def _cache_set(db: aiosqlite.Connection, tickers: list[str]) -> None:
    now        = time.time()
    expires_at = now + _UNIVERSE_TTL
    # Prune all stale rows
    await db.execute("DELETE FROM universe WHERE ttl_expires_at <= ?", (now,))
    await db.execute(
        "INSERT INTO universe (data, fetched_at, ttl_expires_at) VALUES (?, ?, ?)",
        (json.dumps(tickers), now, expires_at),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# Index scrapers
# ---------------------------------------------------------------------------

async def _fetch_sp500(client: httpx.AsyncClient) -> list[str]:
    """Scrape S&P 500 tickers from Wikipedia."""
    try:
        r = await client.get(_SP500_URL, timeout=30.0)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df     = tables[0]
        tickers = df["Symbol"].dropna().tolist()
        # Wikipedia uses dots; yfinance uses hyphens for class shares
        tickers = [str(t).replace(".", "-").strip().upper() for t in tickers]
        logger.info("S&P 500: fetched %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        logger.error("S&P 500 scrape failed: %s", exc)
        return []


async def _fetch_nasdaq100(client: httpx.AsyncClient) -> list[str]:
    """Scrape NASDAQ-100 tickers from Wikipedia."""
    try:
        r = await client.get(_NDX100_URL, timeout=30.0)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        # The component table is usually table index 4; search for 'Ticker' column
        tickers: list[str] = []
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if "ticker" in cols:
                col = df.columns[[i for i, c in enumerate(cols) if "ticker" in c][0]]
                tickers = df[col].dropna().tolist()
                break
        tickers = [str(t).replace(".", "-").strip().upper() for t in tickers if t]
        logger.info("NASDAQ-100: fetched %d tickers", len(tickers))
        return tickers
    except Exception as exc:
        logger.error("NASDAQ-100 scrape failed: %s", exc)
        return []


async def _fetch_russell2000(client: httpx.AsyncClient) -> list[str]:
    """
    Download iShares IWM holdings CSV.

    Tries the primary URL; if the response doesn't look like CSV data,
    retries the fallback URL (same endpoint per spec).  Returns empty
    list on both failures.
    """
    for label, url in [("primary", _IWM_CSV_PRIMARY), ("fallback", _IWM_CSV_FALLBACK)]:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; StockScreener/1.0)"
                ),
                "Referer": "https://www.ishares.com/",
            }
            r = await client.get(url, timeout=60.0, headers=headers,
                                 follow_redirects=True)
            r.raise_for_status()
            text = r.text

            # iShares CSVs have a multi-line header before the actual data;
            # find the row that starts with "Ticker" or "Name"
            lines = text.splitlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("ticker") or (
                    "," in line and line.strip().lower().split(",")[0] in ("ticker", "name")
                ):
                    data_start = i
                    break

            csv_text = "\n".join(lines[data_start:])
            df = pd.read_csv(io.StringIO(csv_text))

            # Column might be 'Ticker', 'TICKER', or similar
            ticker_col = None
            for col in df.columns:
                if str(col).lower().strip() == "ticker":
                    ticker_col = col
                    break
            if ticker_col is None:
                raise ValueError(f"No 'Ticker' column found. Columns: {list(df.columns)}")

            tickers = (
                df[ticker_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            # Drop cash/misc rows (no alpha chars = likely '-' or cash entries)
            tickers = [t for t in tickers if t.isalpha() or ("-" in t and t.replace("-", "").isalpha())]
            logger.info("Russell 2000 (%s): fetched %d tickers", label, len(tickers))
            return tickers
        except Exception as exc:
            logger.warning("Russell 2000 %s URL failed: %s", label, exc)

    logger.error("Russell 2000: both primary and fallback URLs failed.")
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_tickers_async(db_path: str | None = None) -> list[str]:
    """
    Async version of :func:`get_tickers`.

    Returns a deduplicated, sorted list of ~1800 ticker symbols.
    Results are cached in SQLite for 24 hours.
    """
    db_path = db_path or settings.db_path

    async with aiosqlite.connect(db_path) as db:
        await _ensure_universe_table(db)
        cached = await _cache_get(db)
        if cached:
            logger.info("Universe: returning %d cached tickers", len(cached))
            return cached

    logger.info("Universe: cache miss — scraping fresh data …")

    async with httpx.AsyncClient(
        headers={"User-Agent": "StockScreener/1.0 (research)"},
        follow_redirects=True,
    ) as client:
        sp500, ndx100, russ2000 = await asyncio.gather(
            _fetch_sp500(client),
            _fetch_nasdaq100(client),
            _fetch_russell2000(client),
            return_exceptions=False,
        )

    combined = sorted(set(sp500) | set(ndx100) | set(russ2000))
    # Remove obviously invalid tokens
    combined = [t for t in combined if 1 <= len(t) <= 10 and t.replace("-", "").isalpha()]

    logger.info(
        "Universe: S&P=%d  NDX=%d  R2K=%d  → deduplicated=%d",
        len(sp500), len(ndx100), len(russ2000), len(combined),
    )

    async with aiosqlite.connect(db_path) as db:
        await _ensure_universe_table(db)
        await _cache_set(db, combined)

    return combined


def get_tickers(db_path: str | None = None) -> list[str]:
    """
    Synchronous wrapper around :func:`get_tickers_async`.

    Returns a deduplicated, sorted list of ~1800 ticker symbols from
    S&P 500, NASDAQ 100, and Russell 2000.  Results are cached in
    SQLite for 24 hours.

    Parameters
    ----------
    db_path : str, optional
        SQLite path.  Defaults to ``settings.db_path``.

    Returns
    -------
    list[str]
        Sorted, deduplicated ticker symbols.
    """
    return asyncio.run(get_tickers_async(db_path))


# ---------------------------------------------------------------------------
# __main__ — smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    tickers = get_tickers()
    print(f"\nTotal unique tickers: {len(tickers)}")
    print(f"First 20: {tickers[:20]}")
    print(f"Last  20: {tickers[-20:]}")
